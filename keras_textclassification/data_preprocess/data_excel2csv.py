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


data = pd.read_excel(os.path.dirname(path_multi_label_train)+'/01-anhui.xlsx')

print('list', data,columns)
