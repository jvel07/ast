# -*- coding: utf-8 -*-
# @Time    : 10/19/20 5:15 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : prep_esc50.py

import numpy as np
import json
import os
import zipfile

import pandas as pd
import wget


# label = np.loadtxt('/data/sls/scratch/yuangong/aed-pc/src/utilities/esc50_label.csv', delimiter=',', dtype='str')
# f = open("/data/sls/scratch/yuangong/aed-pc/src/utilities/esc_class_labels_indices.csv", "w")
# f.write("index,mid,display_name\n")
#
# label_set = []
# idx = 0
# for j in range(0, 5):
#     for i in range(0, 10):
#         cur_label = label[i][j]
#         cur_label = cur_label.split(' ')
#         cur_label = "_".join(cur_label)
#         cur_label = cur_label.lower()
#         label_set.append(cur_label)
#         f.write(str(idx)+',/m/07rwj'+str(idx).zfill(2)+',\"'+cur_label+'\"\n')
#         idx += 1
# f.close()
#
from sklearn.model_selection import train_test_split, StratifiedKFold


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

# label_set = np.loadtxt('./data/dementia_class_labels_indices.csv', delimiter=',', dtype='str')
# label_map = {}
# for i in range(1, len(label_set)):
#     label_map[eval(label_set[i][2])] = label_set[i][0]
# print(label_map)

label_map = {'1': 'alzheimer', '2': 'mci', '3': 'hc'}

# fix bug: generate an empty directory to save json files
if os.path.exists('./data/datafiles') == False:
    os.mkdir('./data/datafiles')

base_path_16k = '/media/jvel/data/audio/demencia94B-wav16k'
meta = pd.read_csv('dementia_meta.csv')

X = meta.loc[:, 'folder':'filename']
y = meta['label']

skf = StratifiedKFold(n_splits=5)
for idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    X_train['label'] = y_train
    X_test['label'] = y_test

    train = X_train.values.tolist()
    test = X_test.values.tolist()
    train_wav_list = []
    eval_wav_list = []

    for val in train:
        cur_dict = {"wav": os.path.join(base_path_16k, val[1]), "labels": '/m/21rwj' + str(val[2]).zfill(2)}
        train_wav_list.append(cur_dict)

    for val in test:
        cur_dict = {"wav": os.path.join(base_path_16k, val[1]), "labels": '/m/21rwj' + str(val[2]).zfill(2)}
        eval_wav_list.append(cur_dict)

    with open('./data/datafiles/dementia_train_data_' + str(idx) + '.json', 'w') as f:
        json.dump({'data': train_wav_list}, f, indent=1)

    with open('./data/datafiles/dementia_eval_data_' + str(idx) + '.json', 'w') as f:
        json.dump({'data': eval_wav_list}, f, indent=1)


tot = train + test
total_data = []
for val in tot:
    cur_dict = {"wav": os.path.join(base_path_16k, val[1]), "labels": '/m/21rwj' + str(val[2]).zfill(2)}
    total_data.append(cur_dict)

with open('./data/datafiles/dementia_total_data.json', 'w') as f:
    json.dump({'data': total_data}, f, indent=1)
