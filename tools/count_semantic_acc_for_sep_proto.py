#!/usr/bin/env python3
# -*- coding: utf8 -*-

"""
    Calculate the Semantic Accuracy of Single Intent & Single Slot Filling Result
"""

import os
import json
import sys


def read_intent_result(intent_file):
    with open(intent_file, 'r') as fr:
        intent_data = json.load(fr)
    # split for texts, labels, predictions
    texts = [''.join([token for token in item['seq_in'] if token != '[PAD]']) for item in intent_data]
    labels = [item['label'] for item in intent_data]
    preds = [item['pred'] for item in intent_data]
    return texts, labels, preds


def read_slot_result(slot_file):
    with open(slot_file, 'r') as fr:
        lines = fr.read()
    frames = lines.split('\n\n')
    texts, labels, preds = [], [], []
    for frame in frames:
        items = frame.strip().split('\n')
        items = [item.strip().split() for item in items]

        text = ''.join([item[0] for item in items])
        label = [item[1] for item in items]
        pred = [item[2] for item in items]

        texts.append(text)
        labels.append(label)
        preds.append(pred)
    return texts, labels, preds


def get_test_files(path, spt_str=''):
    all_test_files = [item for item in os.listdir(path) if 'test_' in item and spt_str in item and 'ft_' not in item]
    print('number of test files: {}'.format(len(all_test_files)))
    return all_test_files


def get_dev_files(path, spt_str=''):
    all_dev_files = [item for item in os.listdir(path) if 'dev_' in item and spt_str in item and 'ft_' not in item]
    print('number of dev files: {}'.format(len(all_dev_files)))
    return all_dev_files


def cal_batch_semantic_acc(intent_res, slot_res):
    intent_texts, intent_labels, intent_preds = intent_res
    slot_texts, slot_labels, slot_preds = slot_res
    assert len(intent_texts) == len(slot_texts), "the number of text not equal between intent and slot results"

    true_num, all_num = 0, 0
    for idx in range(len(intent_texts)):
        intent_text, slot_text = intent_texts[idx], slot_texts[idx]
        # transfer result will not satisfy the follow rule
        # assert intent_text == slot_text, "the intent text: {} \n " \
        #                                  "do not match the slot intent: {}".format(intent_text, slot_text)
        intent_label, intent_pred = intent_labels[idx], intent_preds[idx]
        slot_label, slot_pred = slot_labels[idx], slot_preds[idx]
        assert len(intent_label) == len(intent_pred) == 1, "the intent number of one sentence should be 1"

        if intent_label[0] == intent_pred[0]:
            flag = True
            for label, pred in zip(slot_label, slot_pred):
                if label != pred:
                    flag = False
                    break
            if flag:
                true_num += 1
        all_num += 1

    semantic_acc = true_num / all_num if all_num != 0 else 0
    return semantic_acc, true_num, all_num


def get_semantic_acc(intent_files, slot_files):
    # sort for matching file
    intent_files = sorted(intent_files)
    slot_files = sorted(slot_files)
    assert len(intent_files) == len(slot_files), "the test file number should be equal, intent: {} - slot: {}".format(
        len(intent_files), len(slot_files))

    file_num = len(intent_files)
    semantic_acc_lst = []
    true_num = 0
    all_num = 0
    for file_idx in range(file_num):
        intent_res = read_intent_result(intent_files[file_idx])
        slot_res = read_slot_result(slot_files[file_idx])
        s_acc, t_num, a_num = cal_batch_semantic_acc(intent_res, slot_res)
        semantic_acc_lst.append(s_acc)
        true_num += t_num
        all_num += a_num

    macro_final_semantic_acc = sum(semantic_acc_lst) / len(semantic_acc_lst) if len(semantic_acc_lst) != 0 else 0
    micro_final_semantic_acc = true_num / all_num if all_num != 0 else 0
    return macro_final_semantic_acc, micro_final_semantic_acc


if __name__ == '__main__':
    '''
    python count_semantic_acc.py snips slu
    python count_semantic_acc.py smp slu
    '''
    args = sys.argv
    if len(args) == 1:
        args.append('snips')
        args.append('baseline')
    
    if args[1] == 'snips':
        seeds = [6150, 6151, 6152]
        cross_ids = [2]
    else:
        seeds = [6150, 6151, 6152]
        cross_ids = [0]
    
    slu_path = args[2]
    if 'intent' in slu_path:
        intent_dir_tmp = slu_path
        slot_dir_tmp = slu_path.replace('intent', 'slot_filling')
    elif 'slot_filling' in slu_path:
        slot_dir_tmp = slu_path
        intent_dir_tmp = slu_path.replace('slot_filling', 'intent')
    else:
        raise NotImplementedError

    # get files
    for cross_id in cross_ids:
        for seed in seeds:
            intent_path = intent_dir_tmp + str(cross_id) + '.m_seed_' + str(seed)
            slot_path = slot_dir_tmp + str(cross_id) + '.m_seed_' + str(seed)

            if os.path.exists(intent_path):

                intent_test_files = get_test_files(intent_path, 'intent')
                intent_test_paths = [os.path.join(intent_path, filename) for filename in intent_test_files]

                slot_test_files = get_test_files(slot_path, 'slot')
                slot_test_paths = [os.path.join(slot_path, filename) for filename in slot_test_files]

                test_macro_semantic_acc, test_micro_semantic_acc = get_semantic_acc(intent_test_paths, slot_test_paths)

            else:
                test_macro_semantic_acc, test_micro_semantic_acc = 0, 0

            print('cross_id_{}.m_seed_{}: '
                  '\n\ttest: (macro) {} \t (micro) {}'.format(cross_id, seed, test_macro_semantic_acc,
                                                              test_micro_semantic_acc,
                                                              ))
        print('------------')


