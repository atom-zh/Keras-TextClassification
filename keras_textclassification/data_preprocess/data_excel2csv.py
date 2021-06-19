# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/05/25 20:35
# @author  : zh-atom
# @function:

from keras_textclassification.data_preprocess.text_preprocess import load_json, save_json, txt_read
from keras_textclassification.conf.path_config import path_model_dir
from keras_textclassification.conf.path_config import path_train, path_valid, path_label, path_root, path_embedding_vector_word2vec_word, path_embedding_random_word, path_embedding_vector_word2vec_word_bin
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import jieba
import word2vec
import os
import re

class preprocess_excel_data:
    def __init__(self):
        self.corpus = []
        pass

    def removePunctuation(self, content):
        """
        文本去标点
        """
        punctuation = r"~!@#$%^&*()_+`{}|\[\]\:\";\-\\\='<>?,.，。、《》？；：‘""“”{【】}|、！@#￥%……&*（）——+=-"
        content = re.sub(r'[{}]+'.format(punctuation), '', content)

        if content.startswith(' ') or content.endswith(' '):
            re.sub(r"^(\s+)|(\s+)$", "", content)
        return content.strip()

    def list_all_files(self, rootdir):
        import os
        _files = []
        # 列出文件夹下所有的目录与文件
        list_file = os.listdir(rootdir)

        for i in range(0, len(list_file)):
            # 构造路径
            path = os.path.join(rootdir, list_file[i])
            # 判断路径是否是一个文件目录或者文件
            # 如果是文件目录，继续递归
            if os.path.isdir(path):
                _files.extend(self.list_all_files(path))
            if os.path.isfile(path):
                _files.append(path)
        return _files

    def excel2csv(self):
        labels = []
        trains = []
        data = []
        files = self.list_all_files(os.path.dirname(path_train))
        for file in files:
            if file.endswith('.xlsx'):
                print('Will read execel file：' + file)
                data += np.array(pd.read_excel(file)).tolist()

        for s_list in data:
            print(s_list)
            label_tmp = self.removePunctuation(str(s_list[5]))
            self.corpus.append(list(label_tmp.split(' ')))
            self.corpus.append(list(jieba.cut(self.removePunctuation(s_list[3]), cut_all=False, HMM=False)))
            if ' ' in label_tmp:
                train_tmp = []
                label_tmp = label_tmp.split('/')
                for i in label_tmp:
                    #label = self.removePunctuation(s_list[4]) + '/' + self.removePunctuation(i)
                    label = self.removePunctuation(i)
                    labels.append(label)
                    train_tmp.append(label)
                train = ','.join(train_tmp) + '|,|' + self.removePunctuation(s_list[3])
                trains.append(train)
            else:
                #label = self.removePunctuation(s_list[4]) + '/' + self.removePunctuation(s_list[5])
                label = self.removePunctuation(str(s_list[5]))
                labels.append(label)
                trains.append(label + '|,|' + self.removePunctuation(s_list[3]))

        # 生成 label 文件
        with open(path_label, 'w', encoding='utf-8') as f_label:
            labels = list(set(labels))
            labels.sort(reverse=False)
            for line in labels:
                f_label.write(line + '\n')
            f_label.close()

        # 生成 train.csv vaild.csv文件
        f_train = open(path_train, 'w', encoding='utf-8')
        f_valid = open(path_valid, 'w', encoding='utf-8')
        random.shuffle(trains)
        f_valid.write('label|,|ques' + '\n')
        f_train.write('label|,|ques' + '\n')
        for i in range(len(trains)):
            print(trains[i])
            if i%5 == 0:
                f_valid.write(trains[i] + '\n')
            else:
                f_train.write(trains[i] + '\n')
        f_valid.close()
        f_train.close()

    def gen_vec(self):
        # 生成 word2vec 预训练 文件
        with open(path_embedding_random_word, 'w', encoding='utf-8') as f_vec_bin:
            for line in self.corpus:
                line = ' '.join(line)
                f_vec_bin.write(line + '\n')
            f_vec_bin.close()
        print(self.corpus)

        word2vec.word2vec(path_embedding_random_word, path_embedding_vector_word2vec_word_bin, size=300, verbose=True)
        print("start to load vec file")
        model = word2vec.load(path_embedding_vector_word2vec_word_bin)
        print(model.vocab)
        with open(path_embedding_vector_word2vec_word, 'w', encoding='utf-8') as f_vec:
            f_vec.write(str(len(model.vocab)) + ' ' + '300' + '\n')
            for i in range(len(model.vectors)):
                word = model.vocab[i]
                vec = model.vectors[i]
                vec = ' '.join(str(i) for i in vec)
                line = str(word) + ' ' + vec  + '\n'
                f_vec.write(line)
            f_vec.close()
