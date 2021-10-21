#!/usr/bin/env python3
# -*- coding: utf8 -*-

"""
Author: laiyongkui
Email: yongkuilai@gmail.com
Date: 2020-11-26 19:26
"""

import os
import sys


def read_slot_result(slot_file):
    with open(slot_file, 'r') as fr:
        lines = fr.read()
    frames = lines.split('\n\n')
    texts, labels, preds = [], [], []
    for frame in frames:
        items = frame.strip().split('\n')
        items = [item.strip().split() for item in items if item]

        text = ''.join([item[0] for item in items])
        label = [item[1] for item in items]
        pred = [item[2] for item in items]

        texts.append(text)
        labels.append(label)
        preds.append(pred)
    return texts, labels, preds


def get_test_files(path, spt_str=''):
    all_test_files = [item for item in os.listdir(path) if 'test_' in item and 'slot' in item and spt_str in item]
    return all_test_files


def cal_one_slot_acc(slot_res):
    slot_texts, slot_labels, slot_preds = slot_res

    right_num = 0
    all_num = len(slot_texts)
    for text, label, pred in zip(slot_texts, slot_labels, slot_preds):
        if label == pred:
            right_num += 1
    return right_num, all_num


def get_slot_acc(slot_files):
    # sort for matching file
    slot_files = sorted(slot_files)
    file_num = len(slot_files)

    right_num, all_num = 0, 0

    for file_idx in range(file_num):
        slot_res = read_slot_result(slot_files[file_idx])
        one_right_num, one_all_num = cal_one_slot_acc(slot_res)
        right_num += one_right_num
        all_num += one_all_num

    slot_acc = right_num / all_num if all_num else 0
    return slot_acc


if __name__ == '__main__':
    '''
    python count_semantic_acc.py snips slu
    python count_semantic_acc.py smp slu
    '''
    args = sys.argv
    if len(args) < 2:
        args[1] = 'snips'

    seeds = [6150, 6151, 6152]
    if args[1] == 'snips':
        cross_ids = [2]
    elif args[1] == 'smp':
        cross_ids = [0]
    else:
        raise NotImplementedError(f'The 1 args `{args[1]}` is not defined')

    slot_dir_tmp = args[2]

    # get files
    for cross_id in cross_ids:
        for seed in seeds:
            slot_path = slot_dir_tmp + str(cross_id) + '.m_seed_' + str(seed)

            if os.path.exists(slot_path):
                slot_test_files = get_test_files(slot_path, 'slot')
                slot_test_paths = [os.path.join(slot_path, filename) for filename in slot_test_files]

                slot_acc = get_slot_acc(slot_test_paths)
            else:
                slot_acc = None

            print('cross_id_{}.m_seed_{}: (slot acc) {} '.format(cross_id, seed, slot_acc))



