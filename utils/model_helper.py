# coding: utf-8
import os
import logging
import sys
import torch
import time
import copy
# my staff
from models.modules.context_embedder_base import ContextEmbedderBase, BertContextEmbedder, \
    BertSeparateContextEmbedder, NormalContextEmbedder, BertSchemaContextEmbedder, BertSchemaSeparateContextEmbedder, \
    ElectraContextEmbedder, ElectraSchemaContextEmbedder, RobertaContextEmbedder
from models.modules.similarity_scorer_base import SimilarityScorerBase, MatchingSimilarityScorer, \
    PrototypeSimilarityScorer, ProtoWithLabelSimilarityScorer, TapNetSimilarityScorer, \
    reps_dot, reps_l2_sim, reps_cosine_sim, RelationNetSim
from models.modules.emission_scorer_base import EmissionScorerBase, MNetEmissionScorer, \
    PrototypeEmissionScorer, ProtoWithLabelEmissionScorer, TapNetEmissionScorer
from models.modules.transition_scorer import FewShotTransitionScorer, FewShotTransitionScorerFromLabel
from models.modules.seq_labeler import SequenceLabeler, RuleSequenceLabeler
from models.modules.text_classifier import SingleLabelTextClassifier
from models.modules.conditional_random_field import ConditionalRandomField, allowed_transitions

from models.few_shot_slu import FewShotSLU, SchemaFewShotSLU, EmissionMergeIntentFewShotSLU, \
    EmissionMergeSlotFewShotSLU, ProtoMergeFewShotSLU
from models.few_shot_seq_labeler import FewShotSeqLabeler, SchemaFewShotSeqLabeler
from models.few_shot_text_classifier import FewShotTextClassifier, SchemaFewShotTextClassifier
from models.modules.scale_controller import build_scale_controller, ScaleControllerBase
from utils.device_helper import prepare_model
from models.normal_slu import AttentionRNNSLU
from transformers import BertConfig


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def make_scaler_args(name: str, normalizer: ScaleControllerBase, scale_r: float = None):
    ret = None
    if name == 'learn':
        ret = {'normalizer': normalizer}
    elif name == 'fix':
        ret = {'normalizer': normalizer, 'scale_rate': scale_r}
    return ret


def build_context_embedder(opt):
    """ 
    Build context_embedder
    :param opt: 
    :return: 
    """
    if opt.context_emb == 'bert':
        context_embedder = BertSchemaContextEmbedder(opt=opt) if opt.use_schema else BertContextEmbedder(opt=opt)
    elif opt.context_emb == 'sep_bert':
        context_embedder = BertSchemaSeparateContextEmbedder(opt=opt) if opt.use_schema else \
            BertSeparateContextEmbedder(opt=opt)
    elif opt.context_emb == 'electra':
        context_embedder = ElectraSchemaContextEmbedder(opt=opt) if opt.use_schema else ElectraContextEmbedder(opt=opt)
    elif opt.context_emb in ['roberta_large', 'roberta_base']:
        context_embedder = RobertaContextEmbedder(opt=opt)
    elif opt.context_emb == 'elmo':
        raise NotImplementedError
    elif opt.context_emb == 'glove':
        context_embedder = NormalContextEmbedder(opt=opt, num_token=len(opt.word2id))
        context_embedder.load_embedding()
    elif opt.context_emb == 'raw':
        context_embedder = NormalContextEmbedder(opt=opt, num_token=len(opt.word2id))
    else:
        raise TypeError('wrong component type')
    return context_embedder


def build_emission_scorer(opt, config, ems_scaler, sim_func, emb_log, task='intent'):

    trans_normalizer = build_scale_controller(name=opt.proto_normalizer)
    proto_scaler = build_scale_controller(
        name=opt.proto_scaler, kwargs=make_scaler_args(opt.proto_scaler, trans_normalizer, opt.proto_scale_r))
    if opt.emission == 'mnet':
        similarity_scorer = MatchingSimilarityScorer(sim_func=sim_func, emb_log=emb_log)
        emission_scorer = MNetEmissionScorer(similarity_scorer, ems_scaler, opt.div_by_tag_num)
    elif opt.emission == 'proto':
        similarity_scorer = PrototypeSimilarityScorer(sim_func=sim_func, scaler=proto_scaler,
                                                      scaler_r=opt.proto_scale_r, emb_log=emb_log)
        emission_scorer = PrototypeEmissionScorer(similarity_scorer, ems_scaler)
    elif opt.emission == 'proto_with_label':
        similarity_scorer = ProtoWithLabelSimilarityScorer(sim_func=sim_func, scaler=opt.ple_scale_r, emb_log=emb_log,
                                                           proto_scaler=proto_scaler, scaler_r=opt.proto_scale_r)
        emission_scorer = ProtoWithLabelEmissionScorer(similarity_scorer, ems_scaler)
    elif opt.emission == 'tapnet':
        # set num of anchors:
        # (1) if provided in config, use it (usually in load model case.)
        # (2) *3 is used to ensure enough anchors ( > num_tags of unseen domains )
        print('config: {}'.format(config))
        num_anchors = config['num_anchors'][task] if 'num_anchors' in config and task in config['num_anchors'] \
            else config['num_tags'][task] * 3
        if 'num_anchors' not in config:
            config['num_anchors'] = {}
        config['num_anchors'][task] = num_anchors
        anchor_dim = 256 if opt.context_emb == 'electra' else 768
        similarity_scorer = TapNetSimilarityScorer(
            sim_func=sim_func, num_anchors=num_anchors, mlp_out_dim=opt.tap_mlp_out_dim,
            random_init=opt.tap_random_init, random_init_r=opt.tap_random_init_r,
            mlp=opt.tap_mlp, emb_log=emb_log, tap_proto=opt.tap_proto, tap_proto_r=opt.tap_proto_r,
            anchor_dim=anchor_dim, scaler=proto_scaler, scaler_r=opt.proto_scale_r)
        emission_scorer = TapNetEmissionScorer(similarity_scorer, ems_scaler)
    else:
        raise TypeError('wrong component type')
    return emission_scorer


def build_slot_decoder(opt, config, emission_scorer, emb_log):
    transition_scorer = None
    if opt.decoder == 'sms':
        decoder = SequenceLabeler()
    elif opt.decoder == 'rule':
        decoder = RuleSequenceLabeler(config['id2label']['slot'])
    elif opt.decoder == 'crf':
        # logger.info('We only support back-off trans training now!')
        # Notice: only train back-off now
        trans_normalizer = build_scale_controller(name=opt.trans_normalizer)
        trans_scaler = build_scale_controller(
            name=opt.trans_scaler, kwargs=make_scaler_args(opt.trans_scaler, trans_normalizer, opt.trans_scale_r))
        if opt.transition == 'learn':
            transition_scorer = FewShotTransitionScorer(
                num_tags=config['num_tags']['slot'], normalizer=trans_normalizer, scaler=trans_scaler,
                r=opt.trans_r, backoff_init=opt.backoff_init)
        elif opt.transition == 'learn_with_label':
            label_trans_normalizer = build_scale_controller(name=opt.label_trans_normalizer)
            label_trans_scaler = build_scale_controller(name=opt.label_trans_scaler, kwargs=make_scaler_args(
                opt.label_trans_scaler, label_trans_normalizer, opt.label_trans_scale_r))
            transition_scorer = FewShotTransitionScorerFromLabel(
                num_tags=config['num_tags']['slot'], normalizer=trans_normalizer, scaler=trans_scaler,
                r=opt.trans_r, backoff_init=opt.backoff_init, label_scaler=label_trans_scaler)
        else:
            raise ValueError('Wrong choice of transition.')
        if opt.add_transition_rules and 'id2label' in config:  # 0 is [PAD] label id, here remove it.
            non_pad_id2label = copy.deepcopy(config['id2label']['slot']).__delitem__(0)
            print('before - non_pad_id2label: {}'.format(non_pad_id2label))
            for k, v in non_pad_id2label.items():
                # TODO: maybe error? `k - 1` rather than `v - 1`
                non_pad_id2label[k] = v - 1  # we 0 as [PAD] label id, here remove it.
            print('after - non_pad_id2label: {}'.format(non_pad_id2label))
            constraints = allowed_transitions(constraint_type='BIO', labels=non_pad_id2label)
        else:
            constraints = None
        decoder = ConditionalRandomField(
            num_tags=transition_scorer.num_tags, constraints=constraints)  # accurate tags
    else:
        raise TypeError('wrong component type')

    seq_laber = SchemaFewShotSeqLabeler if opt.use_schema else FewShotSeqLabeler
    slot_model = seq_laber(opt=opt,
                           emission_scorer=emission_scorer,
                           decoder=decoder,
                           transition_scorer=transition_scorer,
                           config=config,
                           emb_log=emb_log)
    return slot_model


def build_intent_decoder(opt, config, emission_scorer, emb_log):
    decoder = SingleLabelTextClassifier()
    text_classifier = SchemaFewShotTextClassifier if opt.use_schema else FewShotTextClassifier
    intent_model = text_classifier(opt=opt,
                                   emission_scorer=emission_scorer,
                                   decoder=decoder,
                                   config=config,
                                   emb_log=emb_log)
    return intent_model


def make_model(opt, config):
    """ Customize and build the few-shot learning model from components """

    ''' Create log file to record testing data '''
    emb_log = open(os.path.join(opt.output_dir, 'emb.log'), 'w') if opt.emb_log else None

    context_embedder = build_context_embedder(opt)

    '''Build emission scorer and similarity scorer '''
    ems_normalizer = build_scale_controller(name=opt.emission_normalizer)

    ems_scaler = build_scale_controller(
        name=opt.emission_scaler, kwargs=make_scaler_args(opt.emission_scaler, ems_normalizer, opt.ems_scale_r))

    if opt.similarity == 'dot':
        sim_func = reps_dot
    elif opt.similarity == 'cosine':
        sim_func = reps_cosine_sim
    elif opt.similarity == 'l2':
        sim_func = reps_l2_sim
    elif opt.similarity == 'relation':
        emb_dim = 768 if opt.context_emb in ['bert', 'sep_bert'] else 256
        pair_emb_dim = 2 * emb_dim
        sim_func = RelationNetSim(opt=opt, input_size=pair_emb_dim, hidden_size=opt.relation_hidden_size)
    else:
        raise TypeError('wrong component type')

    intent_emission_scorer = build_emission_scorer(opt, config, ems_scaler, sim_func, emb_log, 'intent')
    slot_emission_scorer = build_emission_scorer(opt, config, ems_scaler, sim_func, emb_log, 'slot')

    ''' Build decoder '''
    intent_decoder = None
    slot_decoder = None
    if opt.task in ['slot_filling', 'slu']:

        slot_decoder = build_slot_decoder(opt, config, slot_emission_scorer, emb_log)

    if opt.task in ['intent', 'slu']:
        intent_decoder = build_intent_decoder(opt, config, intent_emission_scorer, emb_log)

    ''' Build the whole model '''
    opt.id2label = config['id2label']

    if opt.slu_model_type == 'simple':
        few_shot_slu = SchemaFewShotSLU if opt.use_schema else FewShotSLU
    elif opt.slu_model_type == 'proto_merge':
        few_shot_slu = ProtoMergeFewShotSLU
    elif opt.slu_model_type == 'emission_merge_intent':
        few_shot_slu = EmissionMergeIntentFewShotSLU
    elif opt.slu_model_type == 'emission_merge_slot':
        few_shot_slu = EmissionMergeSlotFewShotSLU
    else:
        raise NotImplementedError
    model = few_shot_slu(
        opt=opt,
        context_embedder=context_embedder,
        slot_decoder=slot_decoder,
        intent_decoder=intent_decoder,
        config=config,
        emb_log=emb_log
    )

    return model



def make_attention_rnn_model(opt, config):
    """ Customize and build the few-shot learning model from components """

    ''' Create log file to record testing data '''
    emb_log = open(os.path.join(opt.output_dir, 'emb.log'), 'w') if opt.emb_log else None

    context_embedder = build_context_embedder(opt)

    model = AttentionRNNSLU(
        opt=opt,
        context_embedder=context_embedder,
        intent_size=config['num_tags']['intent'],
        slot_size=config['num_tags']['slot'],
        config=config,
        emb_log=emb_log
    )

    return model


def load_model(path):
    try:
        with open(path, 'rb') as reader:
            cpt = torch.load(reader, map_location='cpu')
            model = make_model(opt=cpt['opt'], config=cpt['config'])
            model = prepare_model(args=cpt['opt'], model=model, device=cpt['opt'].device, n_gpu=cpt['opt'].n_gpu)
            model.load_state_dict(cpt['state_dict'])
            return model
    except IOError as e:
        logger.info("Failed to load model from {} \n {}".format(path, e))
        return None




def load_attention_rnn_model(path):
    try:
        with open(path, 'rb') as reader:
            cpt = torch.load(reader, map_location='cpu')
            model = make_attention_rnn_model(opt=cpt['opt'], config=cpt['config'])
            model = prepare_model(args=cpt['opt'], model=model, device=cpt['opt'].device, n_gpu=cpt['opt'].n_gpu)
            model.load_state_dict(cpt['state_dict'])
            return model
    except IOError as e:
        logger.info("Failed to load model from {} \n {}".format(path, e))
        return None



def get_value_from_order_dict(order_dict, key):
    """"""
    for k, v in order_dict.items():
        if key in k:
            return v
    return []

