# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/05/25 20:35
# @author  : zh-atom
# @function:

from keras_textclassification.data_preprocess.text_preprocess import load_json, save_json, txt_read
from keras_textclassification.conf.path_config import path_model_dir
from keras_textclassification.conf.path_config import path_multi_label_train, path_multi_label_valid, path_multi_label_labels, path_root
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os

def removePunctuation(content):
    """
    文本去标点
    """
    punctuation = r"~!@#$%^&*()_+`{}|\[\]\:\";\-\\\='<>?,./，。、《》？；：‘“{【】}|、！@#￥%……&*（）——+=-"
    content = re.sub(r'[{}]+'.format(punctuation), '', content)

    if content.startswith(' ') or content.endswith(' '):
        re.sub(r"^(\s+)|(\s+)$", "", content)
    return content.strip()

def excel2csv():
    labels = []
    trains = ['label|,|ques']
    data = pd.read_excel(os.path.dirname(path_multi_label_train)+'/01-anhui.xlsx')
    s_list = data.tolist()
    print(data.values)
    for line in s_list:
        print(line)


    # 生成 label 文件
    with open(path_multi_label_valid, 'w', encoding='utf-8') as f_label:
        for line in labels:
            labels.write(line + '\n')
        f_label.close()

    # 生成 train.csv 文件
    with open(path_multi_label_train, 'w', encoding='utf-8') as f_train:
        for line in trains:
            f_train.write(line + '\n')
        f_train.close()

    return None
