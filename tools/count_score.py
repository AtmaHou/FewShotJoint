#!/usr/bin/env python3
# -*- coding: utf8 -*-

import os
import sys

args = sys.argv

seed_lst = [6150, 6151, 6152]
if args[1] == 'snips':
    cross_id_lst = [2]
elif args[1] == 'smp':
    cross_id_lst = [0]
else:
    raise ValueError("Args error")

file_head = args[2]
print('file_head: {}'.format(file_head))

for cross_id in cross_id_lst:
    for seed in seed_lst:
        file_name = file_head + str(cross_id) + '.m_seed_' + str(seed) + '.log'
        if os.path.exists(file_name):
            with open(file_name) as fr:
                lines = fr.readlines()
                score_line = lines[-1] if lines else ''
                score = score_line.strip().replace('test:', '')
                print('cross_id_{}.m_seed_${}: {}'.format(cross_id, seed, score))
        else:
            print('cross_id_{}.m_seed_{}: not exists'.format(cross_id, seed))


