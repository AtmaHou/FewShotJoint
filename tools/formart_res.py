#!/usr/bin/env python3
# -*- coding: utf8 -*-

"""
Author: laiyongkui
Email: yongkuilai@gmail.com
Date: 2020-09-26 22:11
"""

import json

filename = 'test.txt'
outfile = 'res.txt'

with open(outfile, 'w') as fw:
    with open(filename, 'r') as fr:
        lines = fr.readlines()
        lines = [': '.join(line.replace("best check points scores:", "").split(': ')[1:]).strip().replace("'", '"') for line in lines]
        lines = [json.loads(line) if line.startswith('{') else {'intent': None, 'slot': None, 'sentence_acc': None}
                 for line in lines]

        print('number of test_res: {}'.format(len(lines)))
        iter_num = len(lines) // 2
        iter_num = 2 if iter_num > 2 else iter_num
        print('iter_num: {}'.format(iter_num))

        for i in range(iter_num):
            fw.write('\t'.join([str(item['intent']) for item in lines[i * 3:i * 3 + 3]]) + '\n')
            fw.write('\t'.join([str(item['slot']) for item in lines[i * 3:i * 3 + 3]]) + '\n')
            fw.write('\t'.join([str(item['sentence_acc']) for item in lines[i * 3:i * 3 + 3]]) + '\n\n')
