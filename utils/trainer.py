# coding: utf-8
from typing import List, Tuple, Dict
import torch
import logging
import sys
import time
import os
import copy
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
# My Staff
from utils.iter_helper import PadCollate, FewShotDataset, SimilarLengthSampler
from utils.preprocessor import FewShotFeature, FewShotSupportFeature
from utils.model_helper import make_model, load_model,  \
    make_attention_rnn_model, load_attention_rnn_model

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


class TrainerBase:
    """
    Build a pytorch trainer, it is design to be:
        - reusable for different training data
        - reusable for different training model instance
        - contains 2 model selection strategy:
            - dev and test(optional) during training. (not suitable when the model is very large)
            - store all checkpoint to disk.
    Support features:
        - multi-gpu [accelerating]
        - distributed gpu [accelerating]
        - 16bit-float training [save space]
        - split batch [save space]
        - model selection(dev & test) [better result & unexpected exit]
        - check-point [unexpected exit]
        - early stop [save time]
        - padding when forward [better result & save space]
        - grad clipping [better result]
        - step learning rate decay [better result]
    """
    def __init__(self, opt, optimizer, scheduler, param_to_optimize, device, n_gpu, tester=None):
        """
        :param opt: args
        :param optimizer:
        :param scheduler:
        :param param_to_optimize: model's params to optimize
        :param device: torch class for training device,
        :param n_gpu:  number of gpu used
        :param tester: class for evaluation
        """
        if opt.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                opt.gradient_accumulation_steps))

        self.opt = opt
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.param_to_optimize = param_to_optimize
        self.tester = tester  # for model selection, set 'None' to not select
        self.gradient_accumulation_steps = opt.gradient_accumulation_steps
        # Following is used to split the batch to save space
        self.batch_size = int(opt.train_batch_size / opt.gradient_accumulation_steps)
        self.device = device
        self.n_gpu = n_gpu
        self.finetune = False
        self.train_epoch = None

    def do_train(self, model, train_features, num_train_epochs,
                 dev_features=None, dev_id2label_map=None,
                 test_features=None, test_id2label_map=None,
                 best_dev_score_now=0, do_model_select=True, finetune=False, ft_id=0):
        """
        do training and dev model selection
        :param model:
        :param train_features:
        :param num_train_epochs:
        :param dev_features:
        :param dev_id2label_map:
        :param test_features:
        :param test_id2label_map:
        :param best_dev_score_now:
        :param do_model_select:
        :param finetune:
        :param ft_id:
        :return:
        """
        self.finetune = finetune
        self.train_epoch = num_train_epochs

        num_train_steps = int(
            len(train_features) / self.batch_size / self.gradient_accumulation_steps * num_train_epochs)

        logger.info("***** Running training *****")
        logger.info("  Num features = %d", len(train_features))
        logger.info("  Batch size = %d", self.batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        global_step = 0  # used for args.fp16
        total_step = 0
        best_dev_score_now = best_dev_score_now
        best_model_now = model
        test_score_map = None
        min_loss = 100000000000000
        loss_now = 0
        no_new_best_dev_num = 0
        no_loss_decay_steps = 0
        is_convergence = False
        cross_steps = 0

        # cross
        one_slot_steps = self.opt.cross_steps
        one_intent_steps = int(one_slot_steps * self.opt.cross_ibs_rate)

        model.train()
        dataset = self.get_dataset(train_features)
        sampler = self.get_sampler(dataset)
        data_loader = self.get_data_loader(dataset, sampler)

        # set intent2slot mask
        self.opt.intent2slot_mask = self.opt.train_intent2slot_mask

        ''' set learning task '''
        if self.opt.cross_update and self.opt.task == 'slu':  # in slu mode: cross update model
            model.set_learning_task('intent')  # intent update first

        for epoch_id in trange(int(num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(data_loader, desc="Train-Batch Progress")):
                if self.n_gpu == 1:
                    # multi-gpu does scattering it-self
                    batch = tuple(t.to(self.device) if not isinstance(t, dict)
                                  else {task: item.to(self.device) for task, item in t.items()} for t in batch)
                ''' set learning task '''
                if self.opt.cross_update and self.opt.task == 'slu':  # in slu mode: cross update model
                    if (model.learning_task == 'intent' and cross_steps == one_intent_steps) \
                            or (model.learning_task == 'slot_filling' and cross_steps == one_slot_steps):
                        cross_steps = 0  # reset cross step
                        # switch task
                        model.set_learning_task('slot_filling' if model.learning_task == 'intent' else 'intent')

                ''' loss '''
                loss = self.do_forward(batch, model, epoch_id, step)

                ''' update '''
                if self.opt.loss_optim_mode == 'cat':  # multi task optimize at the same time
                    loss = self.process_special_loss(loss)  # for parallel process, split batch and so on
                    loss.backward()
                    ''' optimizer step '''
                    global_step, model, is_nan, update_model = self.optimizer_step(step, model, global_step)
                elif self.opt.loss_optim_mode == 'sep':  # optimize separately
                    loss_dict = model.get_loss_dict()
                    is_nan, update_model = False, False
                    all_tasks = list(loss_dict.keys())
                    for idx, task in enumerate(all_tasks):
                        # for parallel process, split batch and so on
                        loss_dict[task] = self.process_special_loss(loss_dict[task])
                        if idx == len(all_tasks) - 1:
                            loss_dict[task].backward()  # release graph which is used to compute grad at the last task
                        else:
                            # retain graph which is used to compute grad before last task
                            loss_dict[task].backward(retain_graph=True)
                        ''' optimizer step '''
                        global_step, model, is_nan, update_model = self.optimizer_step(step, model, global_step)
                else:
                    raise TypeError('the loss optimize mode `{}` is not defined'.format(self.opt.loss_optim_mode))

                ''' handle bad condition in fp16 mode'''
                if is_nan:  # FP16 TRAINING: Nan in gradients, reducing loss scaling
                    continue
                total_step += 1
                cross_steps += 1

                ''' model selection '''
                if do_model_select and self.time_to_make_check_point(total_step, data_loader):
                    if self.tester and self.opt.eval_when_train:  # this is not suit for training big model
                        print("Start dev eval.")
                        dev_score_map, test_score_map, copied_best_model = self.model_selection(
                            model, best_dev_score_now, dev_features, dev_id2label_map, test_features, test_id2label_map,
                            finetune=finetune, ft_id=ft_id)

                        if self.is_bigger(dev_score_map, best_dev_score_now):
                            best_dev_score_now = dev_score_map
                            best_model_now = copied_best_model
                            no_new_best_dev_num = 0
                        else:
                            no_new_best_dev_num += 1
                    else:
                        # step_str = total_step if not finetune else '{}-{}'.format(ft_id, total_step)
                        self.make_check_point_(model=model, step=total_step)

                ''' convergence detection & early stop '''
                loss_now = loss.item() if update_model else loss.item() + loss_now
                if self.opt.convergence_window > 0 and update_model:
                    if global_step % 100 == 0 or total_step % len(data_loader) == 0:
                        print('Current loss {}, global step {}, min loss now {}, no loss decay step {}'.format(
                            loss_now, global_step, min_loss, no_loss_decay_steps))
                    if loss_now < min_loss:
                        min_loss = loss_now
                        no_loss_decay_steps = 0
                    else:
                        no_loss_decay_steps += 1
                        if no_loss_decay_steps >= self.opt.convergence_window:
                            logger.info('=== Reach convergence point!!!!!! ====')
                            print('=== Reach convergence point!!!!!! ====')
                            is_convergence = True
                if no_new_best_dev_num >= self.opt.convergence_dev_num > 0:
                    logger.info('=== Reach convergence point!!!!!! ====')
                    print('=== Reach convergence point!!!!!! ====')
                    is_convergence = True
                if is_convergence:
                    break
            if is_convergence:
                break
            print(" --- The {} epoch Finish --- ".format(epoch_id))

        return best_model_now, best_dev_score_now, test_score_map

    def time_to_make_check_point(self, step, data_loader):
        if not self.finetune:
            interval_size = int(len(data_loader) / self.opt.cpt_per_epoch)
            remained_step = len(data_loader) - (step % len(data_loader))  # remained step for current epoch
            return (step % interval_size == 0 < interval_size <= remained_step) or (step % len(data_loader) == 0)
        else:
            global_size = self.train_epoch * len(data_loader)
            interval_size = global_size if global_size < self.opt.check_step else self.opt.check_step
            return (step % interval_size == 0) or (step % global_size == 0)

    def get_dataset(self, features):
        return TensorDataset([self.unpack_feature(f) for f in features])

    def get_sampler(self, dataset):
        if self.opt.local_rank == -1:
            sampler = RandomSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)
        return sampler

    def get_data_loader(self, dataset, sampler):
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)
        return data_loader

    def process_special_loss(self, loss):
        if self.n_gpu > 1:
            # loss = loss.sum()  # sum() to average on multi-gpu.
            loss = loss.mean()  # mean() to average on multi-gpu.
        if self.opt.fp16 and self.opt.loss_scale != 1.0:
            # rescale loss for fp16 training
            # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
            loss = loss * self.opt.loss_scale
        if self.opt.gradient_accumulation_steps > 1:
            loss = loss / self.opt.gradient_accumulation_steps
        return loss

    def set_optimizer_params_grad(self, param_to_optimize, named_params_model, test_nan=False):
        """ Utility function for optimize_on_cpu and 16-bits training.
            Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
        """
        is_nan = False
        for (name_opti, param_opti), (name_model, param_model) in zip(param_to_optimize, named_params_model):
            if name_opti != name_model:
                logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
                raise ValueError
            if param_model.grad is not None:
                if test_nan and torch.isnan(param_model.grad).sum() > 0:
                    is_nan = True
                if param_opti.grad is None:
                    param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
                param_opti.grad.data.copy_(param_model.grad.data)
            else:
                param_opti.grad = None
        return is_nan

    def copy_optimizer_params_to_model(self, named_params_model, named_params_optimizer):
        """ Utility function for optimize_on_cpu and 16-bits training.
            Copy the parameters optimized on CPU/RAM back to the model on GPU
        """
        for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
            if name_opti != name_model:
                logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
                raise ValueError
            param_model.data.copy_(param_opti.data)

    def make_check_point(self, model, step):
        logger.info("Save model check point to file:%s", os.path.join(
            self.opt.output_dir, 'model.step{}.cpt.pl'.format(step)))
        torch.save(
            self.check_point_content(model), os.path.join(self.opt.output_dir, 'model.step{}.cpt.pl'.format(step)))

    def make_check_point_(self, model, step):
        """ deal with IO error version """
        try:
            logger.info("Save model check point to file:%s", os.path.join(
                self.opt.output_dir, 'model.step{}.cpt.pl'.format(step)))
            torch.save(
                self.check_point_content(model), os.path.join(self.opt.output_dir, 'model.step{}.cpt.pl'.format(step)))
        except IOError:
            logger.info("Failed to make cpt, sleeping ...")
            time.sleep(300)
            self.make_check_point_(model, step)

    def is_bigger(self, first_score: Dict[str, float], second_score: Dict[str, float],
                  task_weight: Dict[str, float] = None)->bool:
        task_lst = [task_name for task_name in first_score.keys() if task_name != 'success']
        if isinstance(second_score, dict):
            assert first_score.keys() == second_score.keys(), "the two scores should have same keys"
        elif isinstance(second_score, int) or isinstance(second_score, float):
            second_score = {task: second_score for task in task_lst}
            if 'sentence_acc' in first_score:
                second_score['sentence_acc'] = 0
        else:
            raise TypeError('the second_score should be int or a dict to assign every task a int or float value')

        if task_weight:
            assert sum(task_weight.values()) == 1.0, "the weight for all tasks should sum to 1"
        else:
            task_weight = {task: 1.0 / len(task_lst) for task in task_lst}

        first_avg_score = sum([score * task_weight[task] for task, score in first_score.items() if task in task_lst])
        second_avg_score = sum([score * task_weight[task] for task, score in second_score.items() if task in task_lst])

        if self.opt.judge_joint_success and len(task_lst) > 1:
            bigger = first_score['sentence_acc'] > second_score['sentence_acc']
        else:
            bigger = first_avg_score > second_avg_score

        return bigger

    def model_selection(self, model, best_score, dev_features, dev_id2label_map,
                        test_features=None, test_id2label_map=None, finetune=False, ft_id=0):
        """ do model selection during training"""
        print("Start dev model selection.")
        # do dev eval at every dev_interval point and every end of epoch
        dev_model = self.tester.clone_model(model, dev_id2label_map)  # copy reusable params, for a different domain
        if self.opt.mask_transition and self.opt.task in ['slot_filling', 'slu']:
            dev_model.label_mask = self.opt.dev_label_mask.to(self.device)

        log_mark = 'dev_pred' if not finetune else 'ft_dev_pred'
        dev_score_map = self.tester.do_test(dev_model, dev_features, dev_id2label_map, log_mark=log_mark)
        logger.info("  dev score(F1) = {}".format(dev_score_map))
        print("  dev score(F1) = {}".format(dev_score_map))
        best_model = None
        test_score_map = None
        if self.is_bigger(dev_score_map, best_score):
            logger.info(" === Found new best!! === ")
            ''' store new best model  '''
            best_model = self.clone_model(model)  # copy model to avoid writen by latter training
            ''' save model file '''
            logger.info("Save model to file:%s", os.path.join(self.opt.output_dir, 'model.pl'))
            model_name = 'model.pl' if not finetune else 'finetune_model.{}.pl'.format(ft_id)
            torch.save(self.check_point_content(model), os.path.join(self.opt.output_dir, model_name))

            ''' get current best model's test score '''
            if test_features:
                # copy reusable params for different domain
                test_model = self.tester.clone_model(model, test_id2label_map)
                if self.opt.mask_transition and self.opt.task in ['slot_filling', 'slu']:
                    test_model.label_mask = self.opt.test_label_mask.to(self.device)

                test_score_map = self.tester.do_test(test_model, test_features, test_id2label_map, log_mark='test_pred')
                logger.info("  test score(F1) = {}".format(test_score_map))
                print("  test score(F1) = {}".format(test_score_map))
        # reset the model status
        model.train()
        return dev_score_map, test_score_map, best_model

    def check_point_content(self, model):
        """ necessary staff for rebuild the model """
        model = model
        return model.state_dict()

    def select_model_from_check_point(
            self, train_id2label_map, dev_features, dev_id2label_map,
            test_features=None, test_id2label_map=None, rm_cpt=True, finetune=False, ft_id=0):
        # set intent2slot mask
        self.opt.intent2slot_mask = self.opt.dev_intent2slot_mask
        all_cpt_file = list(filter(lambda x: '.cpt.pl' in x, os.listdir(self.opt.output_dir)))
        best_score = 0
        test_score_then = 0
        best_model = None
        all_cpt_file = sorted(all_cpt_file, key=lambda x: int(x.replace('model.step', '').replace('.cpt.pl', '')))
        for cpt_file in all_cpt_file:
            logger.info('testing check point: {}'.format(cpt_file))
            if not finetune:
                model = load_model(os.path.join(self.opt.output_dir, cpt_file))
            else:
                if self.opt.normal == 'attn_rnn':
                    model = load_attention_rnn_model(os.path.join(self.opt.output_dir, cpt_file))
            dev_score_map, test_score_map, copied_model = self.model_selection(
                model, best_score, dev_features, dev_id2label_map, test_features, test_id2label_map, finetune, ft_id)
            if self.is_bigger(dev_score_map, best_score):
                best_score = dev_score_map
                test_score_then = test_score_map
                best_model = copied_model
        if rm_cpt:  # delete all check point
            for cpt_file in all_cpt_file:
                os.unlink(os.path.join(self.opt.output_dir, cpt_file))
        return best_model, best_score, test_score_then

    def unpack_feature(self, feature) -> List[torch.Tensor]:
        raise NotImplementedError

    def clone_model(self, model):
        # get a new instance
        return copy.deepcopy(model)

    def do_forward(self, batch, model, epoch_id, step):
        loss = model(*batch)
        return loss

    def optimizer_step(self, step, model, global_step):
        is_nan = False
        update_model = False
        if (step + 1) % self.gradient_accumulation_steps == 0:  # for both memory saving setting and normal setting
            if self.opt.clip_grad > 0:
                torch.nn.utils.clip_grad_value_(model.parameters(), self.opt.clip_grad)
            if self.opt.fp16 or self.opt.optimize_on_cpu:
                if self.opt.fp16 and self.opt.loss_scale != 1.0:
                    # scale down gradients for fp16 training
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.data = param.grad.data / self.opt.loss_scale
                is_nan = self.set_optimizer_params_grad(self.param_to_optimize, model.named_parameters(), test_nan=True)
                if is_nan:
                    logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                    self.opt.loss_scale = self.opt.loss_scale / 2
                    model.zero_grad()
                    return global_step, model, is_nan
                self.optimizer.step()
                self.copy_optimizer_params_to_model(model.named_parameters(), self.param_to_optimize)
            else:
                self.optimizer.step()
            if self.scheduler:  # decay learning rate
                self.scheduler.step()
            model.zero_grad()
            global_step += 1
            update_model = True
        return global_step, model, is_nan, update_model


class FewShotTrainer(TrainerBase):
    """
    Support features:
        - multi-gpu [accelerating]
        - distributed gpu [accelerating]
        - 16bit-float training [save space]
        - split batch [save space]
        - model selection(dev & test) [better result & unexpected exit]
        - check-point [unexpected exit]
        - early stop [save time]
        - padding when forward [better result & save space]
    """
    def __init__(self, opt, optimizer, scheduler, param_to_optimize, device, n_gpu, tester=None):
        super(FewShotTrainer, self).__init__(opt, optimizer, scheduler, param_to_optimize, device, n_gpu, tester)

    def get_dataset(self, features):
        return FewShotDataset([self.unpack_feature(f) for f in features])

    def get_sampler(self, dataset):
        if self.opt.local_rank == -1:
            if self.opt.sampler_type == 'similar_len':
                sampler = SimilarLengthSampler(dataset, batch_size=self.batch_size)
            elif self.opt.sampler_type == 'random':
                sampler = RandomSampler(dataset)
            else:
                raise TypeError('the sampler_type is not true')
        else:
            sampler = DistributedSampler(dataset)
        return sampler

    def get_data_loader(self, dataset, sampler):
        pad_collate = PadCollate(dim=-1, sp_dim=-2, sp_item_idx=[3, 9, 14, 16])  # nwp_index, spt_tgt need special padding
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size, collate_fn=pad_collate)
        return data_loader

    def unpack_feature(self, feature: FewShotFeature) -> List[torch.Tensor]:
        ret = [
            torch.LongTensor([feature.gid]),  # 0
            # test
            feature.test_input.token_ids,  # 1
            feature.test_input.segment_ids,  # 2
            feature.test_input.nwp_index,  # 3
            feature.test_input.input_mask,  # 4
            feature.test_input.slot_output_mask,  # 5
            feature.test_input.intent_output_mask,  # 6
            # support
            feature.support_input.token_ids,  # 7
            feature.support_input.segment_ids,  # 8
            feature.support_input.nwp_index,  # 9
            feature.support_input.input_mask,  # 10
            feature.support_input.slot_output_mask,  # 11
            feature.support_input.intent_output_mask,  # 12
            # target
            feature.slot_test_target,  # 13
            feature.slot_support_target,  # 14
            feature.intent_test_target,  # 15
            feature.intent_support_target,  # 16
            # Special
            torch.LongTensor([len(feature.support_feature_items)]),  # support num
        ]
        return ret

    def do_forward(self, batch, model, epoch_id, step):
        (
            gid,  # 0
            test_token_ids,  # 1
            test_segment_ids,  # 2
            test_nwp_index,  # 3
            test_input_mask,  # 4
            slot_test_output_mask,  # 5
            intent_test_output_mask,  # 6
            support_token_ids,  # 7
            support_segment_ids,  # 8
            support_nwp_index,  # 9
            support_input_mask,  # 10
            slot_support_output_mask,  # 11
            intent_support_output_mask,  # 12
            slot_test_target,  # 13
            slot_support_target,  # 14
            intent_test_target,  # 15
            intent_support_target,  # 16
            support_num,  # 17
        ) = batch

        loss = model(
            test_token_ids,  # 1
            test_segment_ids,  # 2
            test_nwp_index,  # 3
            test_input_mask,  # 4
            slot_test_output_mask,  # 5
            intent_test_output_mask,  # 6
            support_token_ids,  # 7
            support_segment_ids,  # 8
            support_nwp_index,  # 9
            support_input_mask,  # 10
            slot_support_output_mask,  # 11
            intent_support_output_mask,  # 12
            slot_test_target,  # 13
            slot_support_target,  # 14
            intent_test_target,  # 15
            intent_support_target,  # 16
            support_num,  # 17
        )
        return loss

    def check_point_content(self, model):
        """ save staff for rebuild a model """
        model = model  # save sub-module may cause issues
        sub_model = model if self.n_gpu <= 1 else model.module
        ret = {
            'state_dict': model.state_dict(),
            'opt': self.opt,
            'config': model.config,
        }
        return ret

    def get_value_from_order_dict(self, order_dict, key):
        """"""
        for k, v in order_dict.items():
            if key in k:
                return v
        return []

    def clone_model(self, model):
        # deal with data parallel model
        if self.opt.local_rank != -1 or self.n_gpu > 1:  # the model is parallel class here
            old_model = model.module
        else:
            old_model = model
        # get a new instance for different domain (cpu version to save resource)
        config = {'num_tags': old_model.config['num_tags'], 'id2label': old_model.config['id2label']}
        if 'num_anchors' in old_model.config:
            config['num_anchors'] = old_model.config['num_anchors']  # Use previous model's random anchors.
        best_model = make_model(opt=old_model.opt, config=config)
        # copy weights and stuff
        best_model.load_state_dict(old_model.state_dict())
        return best_model



class SchemaFewShotTrainer(FewShotTrainer):
    def __init__(self, opt, optimizer, scheduler, param_to_optimize, device, n_gpu, tester=None):
        super(SchemaFewShotTrainer, self).__init__(opt, optimizer, scheduler, param_to_optimize, device, n_gpu, tester)

    def get_data_loader(self, dataset, sampler):
        """ add label index into special padding """
        pad_collate = PadCollate(dim=-1, sp_dim=-2, sp_item_idx=[3, 9, 14, 16, 20, 25])  # nwp_index, spt_tgt need sp-padding
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size, collate_fn=pad_collate)
        return data_loader

    def unpack_feature(self, feature: FewShotFeature) -> List[torch.Tensor]:
        ret = [
            torch.LongTensor([feature.gid]),  # 0
            # test
            feature.test_input.token_ids,  # 1
            feature.test_input.segment_ids,  # 2
            feature.test_input.nwp_index,  # 3
            feature.test_input.input_mask,  # 4
            feature.test_input.slot_output_mask,  # 5
            feature.test_input.intent_output_mask,  # 6
            # support
            feature.support_input.token_ids,  # 7
            feature.support_input.segment_ids,  # 8
            feature.support_input.nwp_index,  # 9
            feature.support_input.input_mask,  # 10
            feature.support_input.slot_output_mask,  # 11
            feature.support_input.intent_output_mask,  # 12
            # target
            feature.slot_test_target,  # 13
            feature.slot_support_target,  # 14
            feature.intent_test_target,  # 15
            feature.intent_support_target,  # 16
            # Special
            torch.LongTensor([len(feature.support_feature_items)]),  # 17, support num
            # label feature
            # slot
            feature.slot_label_input.token_ids,  # 18
            feature.slot_label_input.segment_ids,  # 19
            feature.slot_label_input.nwp_index,  # 20
            feature.slot_label_input.input_mask,  # 21
            feature.slot_label_input.intent_output_mask,  # 22
            # intent
            feature.intent_label_input.token_ids,  # 23
            feature.intent_label_input.segment_ids,  # 24
            feature.intent_label_input.nwp_index,  # 25
            feature.intent_label_input.input_mask,  # 26
            feature.intent_label_input.intent_output_mask,  # 27
        ]
        return ret

    def do_forward(self, batch, model, epoch_id, step):
        (
            gid,  # 0
            test_token_ids,  # 1
            test_segment_ids,  # 2
            test_nwp_index,  # 3
            test_input_mask,  # 4
            slot_test_output_mask,  # 5
            intent_test_output_mask,  # 6
            support_token_ids,  # 7
            support_segment_ids,  # 8
            support_nwp_index,  # 9
            support_input_mask,  # 10
            slot_support_output_mask,  # 11
            intent_support_output_mask,  # 12
            slot_test_target,  # 13
            slot_support_target,  # 14
            intent_test_target,  # 15
            intent_support_target,  # 16
            support_num,  # 17
            # label feature
            # slot
            slot_label_token_ids,  # 18
            slot_label_segment_ids,  # 19
            slot_label_nwp_index,  # 20
            slot_label_input_mask,  # 21
            slot_label_output_mask,  # 22
            # intent
            intent_label_token_ids,  # 23
            intent_label_segment_ids,  # 24
            intent_label_nwp_index,  # 25
            intent_label_input_mask,  # 26
            intent_label_output_mask,  # 27
        ) = batch

        loss = model(
            # loss, prediction = model(
            test_token_ids,  # 1
            test_segment_ids,  # 2
            test_nwp_index,  # 3
            test_input_mask,  # 4
            slot_test_output_mask,  # 5
            intent_test_output_mask,  # 6
            support_token_ids,  # 7
            support_segment_ids,  # 8
            support_nwp_index,  # 9
            support_input_mask,  # 10
            slot_support_output_mask,  # 11
            intent_support_output_mask,  # 12
            slot_test_target,  # 13
            slot_support_target,  # 14
            intent_test_target,  # 15
            intent_support_target,  # 16
            support_num,  # 17
            # label feature
            (slot_label_token_ids,  # 18
             slot_label_segment_ids,  # 19
             slot_label_nwp_index,  # 20
             slot_label_input_mask,  # 21
             slot_label_output_mask),  # 22
            # intent
            (intent_label_token_ids,  # 23
             intent_label_segment_ids,  # 24
             intent_label_nwp_index,  # 25
             intent_label_input_mask,  # 26
             intent_label_output_mask),  # 27
        )
        return loss


def prepare_optimizer(opt, model, num_train_features, upper_structures=None):
    """
    :param opt:
    :param model:
    :param num_train_features:
    :param upper_structures: list of param name that use different learning rate. These names should be unique sub-str.
    :return:
    """
    num_train_steps = int(
        num_train_features / opt.train_batch_size / opt.gradient_accumulation_steps * opt.num_train_epochs)

    ''' special process for space saving '''
    if opt.fp16:
        param_to_optimize = [(n, param.clone().detach().to('cpu').float().requires_grad_())
                             for n, param in model.named_parameters()]
    elif opt.optimize_on_cpu:
        param_to_optimize = [(n, param.clone().detach().to('cpu').requires_grad_())
                             for n, param in model.named_parameters()]
    else:
        param_to_optimize = list(model.named_parameters())  # all parameter name and parameter

    ''' construct optimizer '''
    if upper_structures and opt.upper_lr > 0:  # use different learning rate for upper structure parameter
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_to_optimize if not any(nd in n for nd in upper_structures)],
             'weight_decay': 0.01, 'lr': opt.learning_rate},
            {'params': [p for n, p in param_to_optimize if any(nd in n for nd in upper_structures)],
             'weight_decay': 0.1, 'lr': opt.upper_lr},
        ]
        if opt.do_debug:
            up_params = [n for n, _ in param_to_optimize if any(nd in n for nd in upper_structures)]
            no_up_params = [n for n, p in param_to_optimize if not any(nd in n for nd in upper_structures)]
    else:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_to_optimize if not any(nd in n for nd in no_decay)],
             'weight_decay': opt.weight_decay, 'lr': opt.learning_rate},
            {'params': [p for n, p in param_to_optimize if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': opt.learning_rate},
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.learning_rate, correct_bias=False)

    ''' construct scheduler '''
    num_warmup_steps = int(opt.warmup_proportion * num_train_steps)
    if opt.scheduler == 'linear_warmup':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)  # PyTorch scheduler
    elif opt.scheduler == 'linear_decay':
        if 0 < opt.decay_lr < 1:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.decay_epoch_size, gamma=opt.decay_lr)
        else:
            raise ValueError('illegal lr decay rate.')
    else:
        raise ValueError('Wrong scheduler')
    return param_to_optimize, optimizer, scheduler

