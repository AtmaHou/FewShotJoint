# coding: utf-8
from typing import List, Tuple, Dict
import torch
import logging
import sys
import os
import copy
import json
import collections
import subprocess
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
# My Staff
from utils.iter_helper import PadCollate, FewShotDataset
from utils.preprocessor import FewShotFeature
from utils.device_helper import prepare_model
from utils.model_helper import make_model, load_model,  \
    make_attention_rnn_model, load_attention_rnn_model

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


RawResult = collections.namedtuple("RawResult", ["feature", "prediction"])


class TesterBase:
    """
    Support features:
        - multi-gpu [accelerating]
        - distributed gpu [accelerating]
        - padding when forward [better result & save space]
    """
    def __init__(self, opt, device, n_gpu):
        if opt.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                opt.gradient_accumulation_steps))

        self.opt = opt
        # Following is used to split the batch to save space
        self.batch_size = opt.test_batch_size
        self.device = device
        self.n_gpu = n_gpu

    def do_test(self, model: torch.nn.Module, test_features: List[FewShotFeature], id2label_map: dict,
                log_mark: str = 'test_pred')->Dict[str, float]:
        logger.info("***** Running eval *****")
        logger.info("  Num features = %d", len(test_features))
        logger.info("  Batch size = %d", self.batch_size)
        task_lst = id2label_map.keys()
        all_results = []

        model.eval()
        data_loader = self.get_data_loader(test_features)

        # set intent2slot mask
        if 'dev_pred' in log_mark:
            self.opt.intent2slot_mask = self.opt.dev_intent2slot_mask
        else:
            self.opt.intent2slot_mask = self.opt.test_intent2slot_mask

        for batch in tqdm(data_loader, desc="Eval-Batch Progress"):
            if self.n_gpu == 1:
                # multi-gpu does scattering it-self
                batch = tuple(t.to(self.device) if not isinstance(t, dict)
                              else {task: item.to(self.device) for task, item in t.items()} for t in batch)
            with torch.no_grad():
                predictions = self.do_forward(batch, model)
            for i, feature_gid in enumerate(batch[0]):  # iter over feature global id
                tmp_dict = {}
                for task in task_lst:
                    if predictions[task] is not None:
                        if self.opt.record_proto:
                            prediction = (predictions[task][i], predictions[task + "_proto"][i])
                            feature = test_features[feature_gid.item()]
                            tmp_dict[task] = RawResult(feature=feature, prediction=prediction)
                        else:
                            prediction = predictions[task][i]
                            feature = test_features[feature_gid.item()]
                            tmp_dict[task] = RawResult(feature=feature, prediction=prediction)
                all_results.append(tmp_dict)
                if model.emb_log:
                    model.emb_log.write('text_' + str(feature_gid.item()) + '\t'
                                        + '\t'.join(feature.test_feature_item.data_item.seq_in) + '\n')

        # close file handler
        if model.emb_log:
            model.emb_log.close()

        scores_map = self.eval_predictions(all_results, id2label_map, log_mark)
        return scores_map

    def get_data_loader(self, features):
        dataset = TensorDataset([self.unpack_feature(f) for f in features])
        if self.opt.local_rank == -1:
            sampler = RandomSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)
        return data_loader

    def clone_model(self, model, id2label):
        # get a new instance
        return copy.deepcopy(model)

    def unpack_feature(self, feature) -> List[torch.Tensor]:
        raise NotImplementedError

    def do_forward(self, batch, model):
        prediction = model(*batch)
        return prediction

    def eval_predictions(self, *args, **kwargs) -> float:
        raise NotImplementedError


class FewShotTester(TesterBase):
    """
        Support features:
            - multi-gpu [accelerating]
            - distributed gpu [accelerating]
            - padding when forward [better result & save space]
    """
    def __init__(self, opt, device, n_gpu):
        super(FewShotTester, self).__init__(opt, device, n_gpu)

    def to_list(self, item):
        if isinstance(item, str):
            item = [item]
        return item

    def eval_predictions(self, all_results: List[Dict[str, RawResult]], id2label_map: Dict[str, Dict[int, str]],
                         log_mark: str) -> Dict[str, float]:
        """ Our result score is average score of all few-shot batches. """
        all_batches = self.reform_few_shot_batch(all_results)
        all_scores = []
        for b_id, fs_batch in all_batches:
            f1_map = self.eval_one_few_shot_batch(b_id, fs_batch, id2label_map, log_mark)
            all_scores.append(f1_map)

        res = {}
        if 'slot' in all_scores[0]:
            res['slot'] = sum([item['slot'] for item in all_scores]) * 1.0 / len(all_scores)
        if 'intent' in all_scores[0]:
            res['intent'] = sum([item['intent'] for item in all_scores]) * 1.0 / len(all_scores)

        # TODO: change to more general
        if self.opt.task == 'slu':
            all_intents = [[item['intent'] for item in fs_batch] for b_id, fs_batch in all_batches]
            all_slots = [[item['slot'] for item in fs_batch] for b_id, fs_batch in all_batches]

            if self.opt.record_proto:
                # prediction is directly the predict ids [pad is removed in decoder]
                all_intent_pred_labels = [[id2label_map['intent'][result.prediction[0][0]] for result in batch_intents]
                                          for batch_intents in all_intents]
            else:
                all_intent_pred_labels = [[id2label_map['intent'][result.prediction[0]] for result in batch_intents]
                                          for batch_intents in all_intents]

            all_intent_target_labels = [[self.to_list(result.feature.test_feature_item.data_item.label)[0]
                                        for result in batch_intents] for batch_intents in all_intents]

            all_intent_pred_labels = np.array([label for batch_labels in all_intent_pred_labels
                                               for label in batch_labels])
            all_intent_target_labels = np.array([label for batch_labels in all_intent_target_labels
                                                 for label in batch_labels])

            assert len(all_intent_pred_labels) == len(all_intent_target_labels), \
                "the number of all_intent_pred_labels is not equal to all_intent_target_labels"

            sentence_acc = (all_intent_pred_labels == all_intent_target_labels)

            if self.opt.record_proto:
                all_slot_pred_labels = [[[id2label_map['slot'][pred_id] for pred_id in result.prediction[0]]
                                         for result in batch_slots] for batch_slots in all_slots]
            else:
                all_slot_pred_labels = [[[id2label_map['slot'][pred_id] for pred_id in result.prediction]
                                         for result in batch_slots] for batch_slots in all_slots]
            all_slot_target_labels = [[result.feature.test_feature_item.data_item.seq_out for result in batch_slots]
                                      for batch_slots in all_slots]

            idx = 0
            for b_idx, (b_pred_labels, b_target_labels) in enumerate(zip(all_slot_pred_labels, all_slot_target_labels)):
                for i_idx, (pred_labels, target_labels) in enumerate(zip(b_pred_labels, b_target_labels)):
                    if sentence_acc[idx]:
                        for p_label, t_label in zip(pred_labels, target_labels):
                            if p_label != t_label:
                                sentence_acc[idx] = False
                                break
                    idx += 1

            sentence_acc = np.mean(sentence_acc.astype(float))
            res['sentence_acc'] = sentence_acc

        return res

    def eval_one_few_shot_batch(self, b_id, fs_batch: List[Dict[str, RawResult]], id2label_map: Dict[str, Dict[int, str]],
                                log_mark: str) -> Dict[str, float]:
        output_dir = self.opt.output_dir if self.opt.do_train else self.opt.output_dir + '.test'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        f1_map = {}
        if self.opt.task in ['slot_filling', 'slu']:
            pred_file_name = 'slot.{}.{}.txt'.format(log_mark, b_id)
            output_prediction_file = os.path.join(output_dir, pred_file_name)
            sl_fs_batch = [item['slot'] for item in fs_batch]
            self.writing_sl_prediction(sl_fs_batch, output_prediction_file, id2label_map['slot'])
            precision, recall, f1 = self.eval_with_script(output_prediction_file)
            f1_map['slot'] = f1

        if self.opt.task in ['intent', 'slu']:
            pred_file_name = 'intent.{}.{}.txt'.format(log_mark, b_id)
            output_prediction_file = os.path.join(output_dir, pred_file_name)
            sc_fs_batch = [item['intent'] for item in fs_batch]
            precision, recall, f1 = self.writing_sc_prediction(sc_fs_batch, output_prediction_file,
                                                               id2label_map['intent'])
            f1_map['intent'] = f1

        return f1_map

    def writing_sc_prediction(self, fs_batch: List[RawResult], output_prediction_file: str, id2label: dict):
        tp, fp, fn = 0, 0, 0
        writing_content = []
        if self.opt.record_proto:
            proto_writing_content = []
        for result in fs_batch:
            prediction = result.prediction
            if self.opt.record_proto:
                pred_ids = prediction[0]
                pred_proto = prediction[1]
            else:
                pred_ids = prediction  # prediction is directly the predict ids [pad is removed in decoder]
            feature = result.feature
            pred_label = set([id2label[pred_id] for pred_id in pred_ids])
            label = set(self.to_list(feature.test_feature_item.data_item.label))
            writing_content.append({
                'seq_in': feature.test_feature_item.data_item.seq_in,
                'pred': list(pred_label),
                'label': list(label),
            })
            tp, fp, fn = self.update_f1_frag(pred_label, label, tp, fp, fn)  # update tp, fp, fn

            if self.opt.record_proto:
                proto_writing_content.append('{}'.format(json.dumps(pred_proto.tolist())))

        with open(output_prediction_file, "w") as writer:
            json.dump(writing_content, writer, indent=2, ensure_ascii=False)

        if self.opt.record_proto:
            with open(output_prediction_file + '_proto', "w") as proto_writer:
                proto_writer.write('\n'.join(proto_writing_content))

        return self.compute_f1(tp, fp, fn)

    def update_f1_frag(self, pred_label, label, tp=0, fp=0, fn=0):
        tp += len(pred_label & label)
        fp += len(pred_label - label)
        fn += len(label - pred_label)
        return tp, fp, fn

    def compute_f1(self, tp, fp, fn):
        tp += 0.0000001  # to avoid zero division
        fp += 0.0000001
        fn += 0.0000001
        precision = 1.0 * tp / (tp + fp)
        recall = 1.0 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def writing_sl_prediction(self, fs_batch: List[RawResult], output_prediction_file: str, id2label: dict):
        writing_content = []
        if self.opt.record_proto:
            proto_writing_content = []
        for result in fs_batch:
            prediction = result.prediction
            feature = result.feature
            if self.opt.record_proto:
                pred_ids = prediction[0]
                pred_proto = prediction[1]
            else:
                pred_ids = prediction  # prediction is directly the predict ids

            if len(pred_ids) != len(feature.test_feature_item.data_item.seq_in):
                raise RuntimeError("Failed to align the pred_ids to texts: {},{} \n{},{} \n{},{}".format(
                    len(pred_ids), pred_ids,
                    len(feature.test_feature_item.data_item.seq_in), feature.test_feature_item.data_item.seq_in,
                    len(feature.test_feature_item.data_item.seq_out), feature.test_feature_item.data_item.seq_out
                ))
            for pred_id, word, true_label in zip(pred_ids, feature.test_feature_item.data_item.seq_in, feature.test_feature_item.data_item.seq_out):
                pred_label = id2label[pred_id]
                writing_content.append('{0} {1} {2}'.format(word, true_label, pred_label))
            writing_content.append('')

            if self.opt.record_proto:
                proto_writing_content.append('{}'.format(json.dumps(pred_proto.tolist())))
        with open(output_prediction_file, "w") as writer:
            writer.write('\n'.join(writing_content))

        if self.opt.record_proto:
            with open(output_prediction_file + '_proto', "w") as proto_writer:
                proto_writer.write('\n'.join(proto_writing_content))

    def eval_with_script(self, output_prediction_file):
        script_args = ['perl', self.opt.eval_script]
        with open(output_prediction_file, 'r') as res_file:
            p = subprocess.Popen(script_args, stdout=subprocess.PIPE, stdin=res_file)
            p.wait()

            std_results = p.stdout.readlines()
            if self.opt.verbose:
                for r in std_results:
                    print(r)
            std_results = str(std_results[1]).split()
        precision = float(std_results[3].replace('%;', ''))
        recall = float(std_results[5].replace('%;', ''))
        f1 = float(std_results[7].replace('%;', '').replace("\\n'", ''))
        f1 = f1 / 100  # normalize to [0, 1]
        return precision, recall, f1

    def reform_few_shot_batch(self, all_results: List[Dict[str, RawResult]]
                              ) -> List[List[Tuple[int, Dict[str, RawResult]]]]:
        """
        Our result score is average score of all few-shot batches.
        So here, we classify all result according to few-shot batch id.
        """
        all_batches = {}
        task_lst = all_results[0].keys()
        has_task = list(task_lst)[0]
        for result in all_results:
            b_id = result[has_task].feature.batch_gid
            if b_id not in all_batches:
                all_batches[b_id] = [result]
            else:
                all_batches[b_id].append(result)
        return sorted(all_batches.items(), key=lambda x: x[0])

    def get_data_loader(self, features):
        dataset = FewShotDataset([self.unpack_feature(f) for f in features])
        if self.opt.local_rank == -1:
            sampler = SequentialSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)
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

    def do_forward(self, batch, model):
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

        prediction = model(
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
        if self.opt.record_proto:
            prediction['intent_proto'] = model.record_intent_prototype
            prediction['slot_proto'] = model.record_slot_prototype
        return prediction

    def get_value_from_order_dict(self, order_dict, key):
        """"""
        for k, v in order_dict.items():
            if key in k:
                return v
        return []

    def clone_model(self, model, id2label_map):
        """ clone only part of params """
        # deal with data parallel model
        if self.opt.local_rank != -1 or self.n_gpu > 1 and hasattr(model, 'module'):  # the model is parallel class here
            old_model = model.module
        else:
            old_model = model
        # emission_dict = old_model.emission_scorer.state_dict()
        # old_num_tags = len(self.get_value_from_order_dict(emission_dict, 'label_reps'))

        config = {'num_tags': {'slot': len(id2label_map['slot']), 'intent': len(id2label_map['intent'])},
                  'id2label': id2label_map}
        if 'num_anchors' in old_model.config:
            config['num_anchors'] = old_model.config['num_anchors']  # Use previous model's random anchors.
        # get a new instance for different domain
        new_model = make_model(opt=self.opt, config=config)
        new_model = prepare_model(self.opt, new_model, self.device, self.n_gpu)
        if self.opt.local_rank != -1 or self.n_gpu > 1:
            sub_new_model = new_model.module
        else:
            sub_new_model = new_model
        ''' copy weights and stuff '''
        if old_model.opt.task in ['slot_filling', 'slu'] and old_model.slot_decoder.transition_scorer:
            # copy one-by-one because target transition and decoder will be left un-assigned
            sub_new_model.context_embedder.load_state_dict(old_model.context_embedder.state_dict())
            sub_new_model.slot_decoder.emission_scorer.load_state_dict(
                old_model.slot_decoder.emission_scorer.state_dict())
            for param_name in ['backoff_trans_mat', 'backoff_start_trans_mat', 'backoff_end_trans_mat']:
                sub_new_model.slot_decoder.transition_scorer.state_dict()[param_name].copy_(
                    old_model.slot_decoder.transition_scorer.state_dict()[param_name].data)
        else:
            sub_new_model.load_state_dict(old_model.state_dict())

        return new_model



class SchemaFewShotTester(FewShotTester):
    def __init__(self, opt, device, n_gpu):
        super(SchemaFewShotTester, self).__init__(opt, device, n_gpu)

    def get_data_loader(self, features):
        """ add label index into special padding """
        dataset = FewShotDataset([self.unpack_feature(f) for f in features])
        if self.opt.local_rank == -1:
            sampler = SequentialSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)
        pad_collate = PadCollate(dim=-1, sp_dim=-2, sp_item_idx=[3, 9, 14, 16, 20, 25])  # nwp_index, spt_tgt need sp-padding
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size, collate_fn=pad_collate)
        return data_loader

    def unpack_feature(self, feature: FewShotFeature) -> List[torch.Tensor]:
        ret = [
            torch.LongTensor([feature.gid]),  # 1
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

    def do_forward(self, batch, model):
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

        prediction = model(
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
        return prediction


def eval_check_points(opt, tester, test_features, test_id2label_map, device, finetune=False):
    all_cpt_file = list(filter(lambda x: '.cpt.pl' in x, os.listdir(opt.saved_model_path)))
    all_cpt_file = sorted(all_cpt_file,
                          key=lambda x: int(x.replace('model.step', '').replace('.cpt.pl', '')))
    max_score = 0
    for cpt_file in all_cpt_file:
        if not finetune:
            cpt_model = load_model(os.path.join(opt.saved_model_path, cpt_file))
        else:
            if opt.normal == 'attn_rnn':
                cpt_model = load_attention_rnn_model(os.path.join(opt.saved_model_path, cpt_file))
        testing_model = tester.clone_model(cpt_model, test_id2label_map)
        if opt.mask_transition and opt.task in ['slot_filling', 'slu']:
            testing_model.label_mask = opt.test_label_mask.to(device)
        test_score = tester.do_test(testing_model, test_features, test_id2label_map, log_mark='test_pred')
        if test_score > max_score:
            max_score = test_score
        logger.info('cpt_file:{} - test:{}'.format(cpt_file, test_score))
    return max_score
