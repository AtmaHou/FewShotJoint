from utils.config import *
import logging
import os
import sys


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

"""  Default path setting """
DEFAULT_RAW_DIR = '/your_dir'
DEFAULT_DATA_DIR = '/your_dir'
BERT_BASE_UNCASED = '/your_dir'
BERT_BASE_UNCASED_VOCAB = '/your_dir'

def define_args(parser, *args_builders):
    """ Set program args"""
    for args_builder in args_builders:
        parser = args_builder(parser)
    return parser


def basic_args(parser):
    group = parser.add_argument_group('Path')  # define path
    group.add_argument('--train_path', required=False, help='the path to the training file.')
    group.add_argument('--dev_path', required=False, help='the path to the validation file.')
    group.add_argument('--test_path', required=False, help='the path to the testing file.')
    group.add_argument("--eval_script", default='./scripts/conlleval.pl', help="The path to the evaluation script")
    group.add_argument("--bert_path", type=str, default=BERT_BASE_UNCASED, help="path to pretrained BERT")
    group.add_argument("--bert_vocab", type=str, default=BERT_BASE_UNCASED_VOCAB, help="path to BERT vocab file")
    group.add_argument('--output_dir', help='The dir to the output file, and to save model,eg: ./')
    group.add_argument("--saved_model_path", default='', help="path to the pre-trained model file")
    group.add_argument("--embedding_cache", type=str, default='/users4/ythou/Projects/Homework/ComputationalSemantic/.word_vectors_cache',
                       help="path to embedding cache dir. if use pytorch nlp, use this path to avoid downloading")

    group = parser.add_argument_group('Function')
    parser.add_argument("--task", default='slu', choices=['slu', 'slot_filling', 'intent'],
                        help="Task: sl:sequence labeling, sc:single label sent classify")
    group.add_argument('--allow_override', default=False, action='store_true', help='allow override experiment file')
    group.add_argument('--load_feature', default=False, action='store_true', help='load feature from file')
    group.add_argument('--save_feature', default=False, action='store_true', help='save feature to file')
    group.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    group.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")
    group.add_argument("--do_debug", default=False, action='store_true', help="debug model, only load few data.")
    group.add_argument("-doft", "--do_overfit_test", default=False, action='store_true', help="debug model, test/dev on train")
    group.add_argument("--verbose", default=False, action='store_true', help="Verbose logging")
    group.add_argument('--seed', type=int, default=42, help="the ultimate answer")

    group = parser.add_argument_group('Device')  # default to use all available GPUs
    group.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    group.add_argument("--no_cuda", default=False, action='store_true', help="Whether not to use CUDA when available")

    group = parser.add_argument_group('Device')  # default to use all available GPUs
    group.add_argument("-rp", "--record_proto", default=False, action='store_true',
                       help="Whether to record prototype when in testing")
    return parser


def preprocess_args(parser):
    group = parser.add_argument_group('Preprocess')
    group.add_argument("--sim_annotation", default='match', type=str,
                       choices=['match', 'BI-match', 'attention'], help="Define annotation of token similarity")
    group.add_argument("--label_wp", action='store_true',
                       help="For sequence label, use this to generate label for word piece, which is Abandon Now.")
    return parser


def train_args(parser):
    group = parser.add_argument_group('Train')
    group.add_argument("--restore_cpt", action='store_true', help="Restart training from a checkpoint ")
    group.add_argument("--cpt_per_epoch", default=2, type=int, help="The num of check points of each epoch")
    group.add_argument("-c_step", "--use_check_step", default=False, action="store_true",
                       help="Whether use check step to store checkpoint")
    group.add_argument("--convergence_window", default=5000, type=int,
                       help="A observation window for early stop when model is in convergence, set <0 to disable")
    group.add_argument("--convergence_dev_num", default=5, type=int,
                       help="A observation window for early stop when model is in convergence, set <0 to disable")
    group.add_argument("--train_batch_size", default=2, type=int, help="Total batch size for training.")
    group.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    group.add_argument("--num_train_epochs", default=20, type=float,
                       help="Number of training epochs to perform.")
    group.add_argument("--warmup_proportion", default=0.1, type=float,
                       help="Proportion of training to perform linear learning rate warmup for. E.g.10% of training.")
    group.add_argument("--eval_when_train", default=False, action='store_true',
                       help="Test model found new best model")
    group.add_argument("--slu_model_type", default="simple", choices=["simple", "split_metric", "split_intent_metric",
                                                                      "split_slot_metric", "intent_derived_by_slot",
                                                                      "cat_intent_to_slot", "emission_merge_intent",
                                                                      "emission_merge_slot", "emission_merge_iteration",
                                                                      "baseline", "proto_merge", "mix_proto_merge",
                                                                      "multi_proto_merge", "multi_mix_proto_merge",
                                                                      "con_prom"]
                       )
    group.add_argument("--metric_rate", default=0.5, type=float, help="The rate for metric result")
    group.add_argument("-aol", "--add_origin_loss", default=False, action='store_true',
                       help="Add update progress in origin output")
    group.add_argument("-um", "--use_multi", default=False, action='store_true', help="Whether use multi merge method")
    group.add_argument("--rl_type", default='origin', choices=['origin', 'metric', 'mix', 'none'],
                       help="Where to add regular loss, origin space, metric space or both")
    group.add_argument("--pm_type", default='origin', choices=['origin', 'metric', 'mix', 'none'],
                       help="Where to merge prototype, origin space, metric space or both")
    group.add_argument("--em_type", default='origin', choices=['origin', 'metric', 'mix', 'none'],
                       help="Where to merge emission, origin space, metric space or both")
    group.add_argument("--use_cls", default=False, action='store_true', help="Use [cls] for test reps and [sep]")
    group.add_argument("--just_train_mode", default=False, action='store_true',
                       help="Control some merge operations whether run in only train mode or in eval mode too")

    group = parser.add_argument_group('FineTune')  # Optimize space usage
    group.add_argument("-ws", "--with_source", default=False, action='store_true',
                       help="use source data in fine-tune step")
    group.add_argument("-ft", "--finetune", default=False, action='store_true',
                       help="Fine-Tune the target domain in the last N layers of encoder")
    group.add_argument("--ft_model_type", default='slu', choices=['slu', 'bert_slu'], help="the finetune model type")
    group.add_argument("--ft_add_query", default=False, action='store_true',
                       help="Add query, and merge it to support, and regard it as non-few-shot dataset")
    group.add_argument("--ft_num_train_epochs", default=50, type=float, help="The epoch num in Pre-train")
    group.add_argument("--ft_test_train_epochs", default=50, type=float, help="The epoch num in Fine-tune")
    group.add_argument("--ft_layer_num", default=2, type=int, help="The last N num layers for Fine-tuning")
    group.add_argument("--ft_dataset", default='origin', type=str, help="The method of organizing fine-tune dataset")
    group.add_argument("--ft_td_rate", default=0.8, type=float, help="The rate for splitting train and dev")
    group.add_argument("-ft_nms", "--ft_no_model_select", default=False, action='store_true',
                       help="do not select model in fine-tune")
    group.add_argument("--check_step", default=2000, type=int, help="the steps of checking checkpoint")
    group.add_argument("--ft_id", default=-1, type=int, help="the assigned id for fine-tuning model")
    group.add_argument("--ft_lr", default=5e-5, type=float, help="The initial learning rate for Adam in Pre-Train.")
    group.add_argument("--ft_up_lr", default=0.0, type=float,
                       help="The upper learning rate for some modules in Pre-Train.")
    group.add_argument("--ft_grad_acc", default=1, type=int, help="gradient accumulation steps in Pre-Train.")
    group.add_argument("--ft_test_label2id", default='all', choices=['all', 'small'],
                       help="gradient accumulation steps in Pre-Train.")
    group.add_argument("--saved_pre_train_path", default='./data', type=str, help="The path of pre-trained model.")

    group = parser.add_argument_group('JointModel')  # Optimize space usage
    group.add_argument("-pm", "--proto_merge", default=False, action="store_true", help="use proto merging component")
    group.add_argument("--proto_merge_type", default='dot_attention', choices=[
        'cat_intent_to_slot', 'intent_derived_by_slot', 'merge_both', 'dot_attention', 'scale_dot_attention',
        'self_attention', 'add_attention', '2linear_attention', 'logit_attention',
        'cat_intent_to_slot-scale_dot_attention', 'cat_intent_to_slot-add_attention',
        'cat_intent_to_slot-2linear_attention', 'intent_derived_by_slot-scale_dot_attention',
        'intent_derived_by_slot-add_attention', 'intent_derived_by_slot-2linear_attention', 'none', 'fix'],
                       help="the methods of merging prototype")
    group.add_argument("-pm_rpl", "--proto_replace", default=False, action='store_true',
                       help="whether replace prototype with generated prototype or not")
    group.add_argument("-pm_ua", "--pm_use_attn", default=False, action='store_true',
                       help="whether use attention to construct attention or not")
    group.add_argument("-pm_r", "--pm_attn_r", default=0.5, type=float,
                       help="the rate for the prototype derived by using attention from support set")
    group.add_argument("--proto_update", default='slu', choices=['intent', 'slot', 'slu'],
                       help="the prototype of which task will be updated")
    group.add_argument("--pm_stop_grad", default=False, action='store_true',
                       help="stop the gradient of those prototype who will be merged ")
    group.add_argument("--emission_merge_type", default='none', type=str, choices=['none', 'intent', 'slot', 'iteration'],
                       help="The method type of merging emission")
    group.add_argument("--extr_sent_reps", default='none', type=str, choices=['none', 'slot_attn', 'self_attn',
                                                                              'coarse_slot_self_attn',
                                                                              'fine_slot_self_attn', 'self_attn_w'],
                       help="method of getting sentence level representation")
    group.add_argument("--slot_attn_r", default=0.5, type=float, help="method of getting sentence level representation")
    group.add_argument("--attn_hidden_size", default=100, type=int, help="the hidden size of attention in proto merge")
    group.add_argument("--attn_spt_mask", default=False, action="store_true",
                       help="get intent2slot attention mask from support data")
    group.add_argument("--attn_glb_mask", default=False, action="store_true",
                       help="get intent2slot attention mask from all global data")

    group = parser.add_argument_group('SpaceOptimize')  # Optimize space usage
    group.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help="Number of updates steps to accumulate before performing a backward/update pass."
                            "Every time a variable is back propogated through,"
                            "the gradient will be accumulated instead of being replaced.")
    group.add_argument('--optimize_on_cpu', default=False, action='store_true',
                       help="Whether to perform optimization and keep the optimizer averages on CPU")
    group.add_argument('--fp16', default=False, action='store_true',
                       help="Whether to use 16-bit float precision instead of 32-bit")
    group.add_argument('--loss_scale', type=float, default=128,
                       help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    group.add_argument('-d_cpt', '--delete_checkpoint', default=False, action='store_true',
                       help="Only keep the best model to save disk space")

    group = parser.add_argument_group('PerformanceTricks')  # Training Tricks
    group.add_argument('--clip_grad', type=float, default=-1, help="clip grad, set < 0 to disable this")
    group.add_argument("--scheduler", default='linear_warmup', type=str, help="select pytorch scheduler for training",
                       choices=['linear_warmup', 'linear_decay'])
    group.add_argument('--decay_lr', type=float, default=0.5,
                       help="When choose linear_decay scheduler, rate of lr decay. ")
    group.add_argument('--weight_decay', type=float, default=0.0,  help="The learning rate decay rate. ")
    group.add_argument('--decay_epoch_size', type=int, default=1,
                       help="When choose linear_decay scheduler, decay lr every decay_epoch_size")
    group.add_argument("--sampler_type", default='similar_len', choices=['similar_len', 'random'],
                       help="method to sample batch")

    # for few shot seq labeling model
    group = parser.add_argument_group('FewShotSetting')  # Training Tricks
    group.add_argument("--warmup_epoch", type=int, default=-1,
                       help="set > 0 to active warm up training. "
                            "Train model in two step: "
                            "1: fix bert part  "
                            "2: train entire model"
                            "(As we use new optimizer in 2nd stage, it also has restart effects. )")
    group.add_argument("--fix_embed_epoch", default=-1, type=int, help="Fix embedding for first x epochs.[abandon]")
    group.add_argument("--upper_lr", default=-1, type=float,
                       help="Use different LR for upper structure comparing to embedding LR. -1 to off it")
    group.add_argument("--no_embedder_grad", default=False, action='store_true', help="not perform grad on embedder")
    group.add_argument("--fix_encoder_layer", default=0, type=int, help="fix some layers of embedder encoder")
    group.add_argument("--train_label_mask")

    group = parser.add_argument_group('SplitMetric')
    group.add_argument("--metric_dim", default=128, type=int, help="the dim of metric space")
    group.add_argument("--split_metric", default='none', type=str, choices=['none', 'intent', 'slot', 'both'],
                       help="the method for splitting metric space, "
                            "none: means use embedding space as metric space,, "
                            "intent: means split intent metric space which is built on embedding space, "
                            "slot: means split slot metric space which is built on embedding space, "
                            "both: means split intent and slot metric space which are built on embedding space")
    group.add_argument("--metric_activation", default='none', type=str, choices=['none', 'relu', 'sigmoid', 'tanh',
                                                                                 'sigmoid-none', 'none-relu',
                                                                                 'sigmoid-relu', 'relu-none'],
                       help="the activation function of metric space")
    group.add_argument("--no_up_metric_params", default=False, action='store_true',
                       help="set no upper_lr for metric parameters")
    group.add_argument("-up_saw", "--up_self_attn_w_params", default=False, action='store_true',
                       help="set upper_lr for self-attention parameters")
    group.add_argument("--encoder_dim", default=512, type=int, help="the encoder dim")
    group.add_argument("--encoder_layer_num", default=1, type=int, help="the number of encoder layers")
    group.add_argument("--encoder_direction", default=1, choices=[1, 0], type=int,
                       help="bidirectional or not, 1 is true, 0 is not true")

    group = parser.add_argument_group('EmissionMerge')
    group.add_argument("--emission_merge_iter_num", default=1, type=int, help="the iter number of emission merge")

    group = parser.add_argument_group('LossScale')
    group.add_argument("--loss_scale_type", default='none', type=str, choices=['none', 'batch_mean', 'total_mean'],
                       help="the iter number of emission merge")

    group = parser.add_argument_group('RelationNetwork')
    group.add_argument("--relation_norm", default='none', type=str, choices=['none', 'layer_norm'],
                       help="the norm mechanism in Relation Network")
    group.add_argument("--relation_hidden_size", default=256, type=int, help="the hidden size in Relation Network")

    group = parser.add_argument_group('CrossUpdate')
    group.add_argument("--cross_update", default=False, action='store_true',
                       help="Cross update model by switching task")
    group.add_argument("--cross_steps", default=10, type=int, help="the accumulate steps for one learning task")
    group.add_argument("--cross_ibs_rate", default=1.0, type=float, help="the update step rate for intent by slot")

    return parser


def test_args(parser):
    group = parser.add_argument_group('Test')
    group.add_argument("--test_batch_size", default=2, type=int, help="Must same to few-shot batch size now")
    group.add_argument("--test_on_cpu", default=False, action='store_true', help="eval on cpu")
    group.add_argument("--judge_joint_success", default=False, action='store_true',
                       help="set the eval method to judge joint score with joint success")

    return parser


def model_args(parser):
    group = parser.add_argument_group('Encoder')
    group.add_argument("--context_emb", default='bert', type=str,
                       choices=['bert', 'elmo', 'glove', 'raw', 'sep_bert', 'electra', 'roberta_base', 'roberta_large'],
                       help="select word representation type")
    group.add_argument("--similarity", default='dot', type=str,
                       choices=['cosine', 'dot', 'bi-affine', 'l2', 'relation'], help="Metric for evaluating 2 tokens.")
    group.add_argument("--emb_dim", default=64, type=int, help="Embedding dimension for baseline")
    group.add_argument("--label_reps", default='sep', type=str,
                       choices=['cat', 'sep', 'sep_sum'], help="Method to represent label")
    group.add_argument("--use_schema", default=False, action='store_true',
                       help="(For MNet) Divide emission by each tag's token num in support set")

    group = parser.add_argument_group('Decoder')
    group.add_argument("--decoder", default='crf', type=str, choices=['crf', 'sms', 'smcrf', 'rule'],
                       help="decode method, "
                            "crf: our crf decoder use transition mask (statistic), "
                            "sms: simple sequence decoder, no transition mask, "
                            "smcrf: simple crf decoder, use pytorch-crf function, no transition mask, "
                            "rule: simple sequence decocer, with rule transition mask")

    # ===== emission layer setting =========
    group.add_argument("--emission", type=str, default="mnet",
                       choices=['mnet', 'rank', 'proto', 'proto_with_label', 'tapnet'],
                       help="Method for calculate emission score, match with task list")
    group.add_argument("-e_nm", "--emission_normalizer", type=str, default='', choices=['softmax', 'norm', 'none'],
                       help="normalize emission into 1-0")
    group.add_argument("-e_scl", "--emission_scaler", type=str, default=None,
                       choices=['learn', 'fix', 'relu', 'exp', 'softmax', 'norm', 'none'],
                       help="method to scale emission and transition into 1-0")
    group.add_argument("--ems_scale_r", default=1, type=float, help="Scale transition to x times")
    # proto setting
    group.add_argument("-pr_nm", "--proto_normalizer", type=str, default='', choices=['softmax', 'norm', 'none'],
                       help="normalize scaled label embedding into 1-0")
    group.add_argument("-pr_scl", "--proto_scaler", type=str, default=None,
                       choices=['learn', 'fix', 'relu', 'exp', 'softmax', 'norm', 'none'],
                       help="method to scale label embedding into 1-0")
    group.add_argument("--proto_scale_r", default=0.5, type=float, help="Scale given prototype reps to x times")
    # proto with label setting
    group.add_argument("-ple_nm", "--ple_normalizer", type=str, default='', choices=['softmax', 'norm', 'none'],
                       help="normalize scaled label embedding into 1-0")
    group.add_argument("-ple_scl", "--ple_scaler", type=str, default=None,
                       choices=['learn', 'fix', 'relu', 'exp', 'softmax', 'norm', 'none'],
                       help="method to scale label embedding into 1-0")
    group.add_argument("--ple_scale_r", default=1, type=float, help="Scale label embedding to x times")
    # tap net setting
    group.add_argument("--tap_random_init", default=False, action='store_true',
                       help="Set random init for label reps in tap-net")
    group.add_argument("--tap_random_init_r", default=1, type=float,
                       help="Set random init rate for label reps in tap-net")
    group.add_argument("--tap_mlp", default=False, action='store_true', help="Set MLP in tap-net")
    group.add_argument("--tap_mlp_out_dim", default=768, type=int, help="The dimension of MLP in tap-net")
    group.add_argument("--tap_proto", default=False, action='store_true',
                       help="choose use proto or label in projection space in tap-net method")
    group.add_argument("--tap_proto_r", default=1, type=float,
                       help="the rate of prototype in mixing with label reps")
    # Matching Network setting
    group.add_argument('-dbt', "--div_by_tag_num", default=False, action='store_true',
                       help="(For MNet) Divide emission by each tag's token num in support set")

    group.add_argument("--emb_log", default=False, action='store_true', help="Save embedding log in all emission step")

    # ===== decoding layer setting =======
    # CRF setting
    group.add_argument('--transition', default='learn',
                       choices=['merge', 'target', 'source', 'learn', 'none', 'learn_with_label'],
                       help='transition for target domain')
    group.add_argument("-t_nm", "--trans_normalizer", type=str, default='', choices=['softmax', 'norm', 'none'],
                       help="normalize back-off transition into 1-0")
    group.add_argument("-t_scl", "--trans_scaler", default=None,
                       choices=['learn', 'fix', 'relu', 'exp', 'softmax', 'norm', 'none'],
                       help='transition matrix scaler, such as re-scale the value to non-negative')

    group.add_argument('--backoff_init', default='rand', choices=['rand', 'fix'],
                       help='back-off transition initialization method')
    group.add_argument("--trans_r", default=1, type=float, help="Transition trade-off rate of src(=1) and tgt(=0)")
    group.add_argument("--trans_scale_r", default=1, type=float, help="Scale transition to x times")

    group.add_argument("-lt_nm", "--label_trans_normalizer", type=str, default='', choices=['softmax', 'norm', 'none'],
                       help="normalize transition FROM LABEL into 1-0")
    group.add_argument("-lt_scl", "--label_trans_scaler", default='fix', choices=['none', 'fix', 'learn'],
                       help='transition matrix FROM LABEL scaler, such as re-scale the value to non-negative')
    group.add_argument("--label_trans_scale_r", default=1, type=float, help="Scale transition FROM LABEL to x times")

    group.add_argument('-mk_tr', "--mask_transition", default=False, action='store_true',
                       help="Block out-of domain transitions.")
    group.add_argument("--add_transition_rules", default=False, action='store_true', help="Block invalid transitions.")

    group = parser.add_argument_group('Loss')
    group.add_argument("--loss_func", default='cross_entropy', type=str, choices=['mse', 'cross_entropy'],
                       help="Loss function for label prediction, when use crf, this factor is useless. ")
    group.add_argument("--loss_optim_mode", default="cat", choices=["cat", "sep"], help="the mode of optimizing loss")
    group.add_argument("--slu_regular", default="none", choices=[
        "none", "contrastive", "triplet", "triplet2", "strict_triplet2", "triplet_semi_hard",
        "strict_triplet_semi_hard", "homo_contrastive"],
                       help="add slu regular loss to stride intent domain to more far")
    group.add_argument("--rg_inter", default=False, action="store_true", help="weather to use inter-loss")
    group.add_argument("-rgi", "--rg_intra_intent", default=False, action="store_true",
                       help="weather to use intent intra-loss")
    group.add_argument("-rgs", "--rg_intra_slot", default=False, action="store_true",
                       help="weather to use slot intra-loss")
    group.add_argument("--slu_regular_rate", default=10, type=float, help="the regular loss rate")
    group.add_argument("--margin", default=10, type=float, help="the margin for Contrastive Loss")
    group.add_argument("--intra_intent_margin", default=10, type=float, help="the margin for Contrastive Loss")
    group.add_argument("--intra_slot_margin", default=10, type=float, help="the margin for Contrastive Loss")
    group.add_argument("-wh", "--with_homo", default=False, action="store_true", help="add homo contrastive loss")
    group.add_argument("--sep_o_type", default='max', choices=['max', 'mean'],
                       help="the method to construct binary logits")
    group.add_argument("--sep_o_loss", default=False, action='store_true',
                       help="the separate O-label and non-O-label loss")
    group.add_argument("-lt", "--loss_rate", default=1.0, type=float, help="the loss scale rate for slot`")
    return parser


def maml_args(parser):
    group = parser.add_argument_group('MAML')
    group.add_argument("--maml", default=False, action="store_true", help="Whether run MAML model or not")
    group.add_argument("--fs_maml", default=False, action="store_true", help="Whether run MAML FewShot model or not")
    group.add_argument("--maml_model", default="slu", choices=[
        "reptile_intent", "reptile_slot", "fix_reptile_simple_slu", "learnable_reptile_simple_slu",
        "flexible_reptile_simple_slu", "reptile_rnn_slu", "attn_reptile_rnn_slu"],
                       help="the type of model")
    group.add_argument("--lstm_hidden_size", default=128, type=int, help="")
    group.add_argument("--n_layers", default=1, type=int, help="the number of lstm layer")
    group.add_argument("-bd", "--bi_direction", default=False, action="store_true",
                       help="whether to use double direction or not")
    group.add_argument("--alpha", default=0.5, type=float, help="the scale parameter for loss between intent and slot")
    group.add_argument("--inner_steps", default=5, type=int, help="the running steps for inner loop")
    group.add_argument("--inner_steps_test", default=10, type=int, help="the running steps for inner loop in testing")
    group.add_argument("--multi_step_loss_num_epochs", default=1, type=int, help="")
    group.add_argument("-en_ilo", "--enable_inner_loop_optimizable_bn_params", default=False, action="store_true",
                       help="")
    group.add_argument("-lrn_rate", "--learnable_per_layer_per_step_inner_loop_learning_rate", default=False,
                       action="store_true", help="")
    group.add_argument("--task_learning_rate", default=0.1, type=float, help="Learning rate per task gradient step")
    group.add_argument("--meta_learning_rate", default=0.001, type=float, help="")
    group.add_argument("--min_learning_rate", default=0.00001, type=float, help="")
    group.add_argument("-ums", "--use_multi_step_loss_optimization", default=False, action="store_true",
                       help="Whether use multiple step loss optimization or not")
    group.add_argument("-uso", "--use_second_order", default=False, action="store_true",
                       help="whether to use second order or not")
    group.add_argument("--second_order_start_epoch", default=1, type=int, help="the start epoch of using second order")

    group.add_argument("--normal", default="none", choices=["attn_rnn", "none"], help="the type of normal model")
    group.add_argument("-stdm", "--save_test_domain_model", default=False, action="store_true",
                       help="save test domain model in testing")

    return parser


def option_check(opt):
    if opt.do_debug:
        if not opt.do_overfit_test:
            opt.num_train_epochs = 1
            opt.ft_num_train_epochs = 1
        opt.load_feature = False
        opt.save_feature = False
        opt.cpt_per_epoch = 1
        opt.allow_override = True
        opt.ft_num_train_epochs = 1
        opt.ft_test_train_epochs = 1
        opt.inner_steps = 1
        opt.inner_steps_test = 1

    if not(opt.local_rank == -1 or opt.no_cuda):
        if opt.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            opt.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)

    if not opt.do_train and not opt.do_predict:
        raise ValueError("At least one of 'do_train' or 'do_predict' must be True.")

    output_dir = opt.output_dir if opt.do_train else opt.output_dir + '.test'

    if os.path.exists(output_dir) and os.listdir(opt.output_dir) and not opt.allow_override:
        raise ValueError("Output directory () already exists and is not empty.")

    if opt.do_train and not (opt.train_path and opt.dev_path):
        raise ValueError("If `do_train` is True, then `train_file` and dev file must be specified.")

    if opt.do_predict and not opt.test_path:
        raise ValueError("If `do_predict` is True, then `predict_file` must be specified.")

    if opt.gradient_accumulation_steps > opt.train_batch_size:
        raise ValueError('if split batch "gradient_accumulation_steps" must less than batch size')

    if opt.label_wp:
        raise ValueError('This function is kept in feature process but not support by current models.')
    return opt
