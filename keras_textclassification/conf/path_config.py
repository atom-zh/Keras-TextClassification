# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/5 21:04
# @author   :Mo
# @function :file of path

import os

# 项目的根目录
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
path_root = path_root.replace('\\', '/')

# path of embedding
path_embedding_user_dict = path_root + '/data/embeddings/user_dict.txt'
path_embedding_random_char = path_root + '/data/embeddings/term_char.txt'
path_embedding_random_word = path_root + '/data/embeddings/term_word.txt'
path_embedding_bert = path_root + '/data/embeddings/chinese_L-12_H-768_A-12/'
path_embedding_xlnet = path_root + '/data/embeddings/chinese_xlnet_mid_L-24_H-768_A-12/'
path_embedding_albert = path_root + '/data/embeddings/albert_base_zh'
path_embedding_vector_word2vec_char = path_root + '/data/embeddings/multi_label_char.vec'
path_embedding_vector_word2vec_word = path_root + '/data/embeddings/multi_label_word.vec'
path_embedding_vector_word2vec_char_bin = path_root + '/data/embeddings/multi_label_char.bin'
path_embedding_vector_word2vec_word_bin = path_root + '/data/embeddings/multi_label_word.bin'

# classify data of baidu qa 2019
path_baidu_qa_2019_train = path_root + '/data/baidu_qa_2019/baike_qa_train.csv'
path_baidu_qa_2019_valid = path_root + '/data/baidu_qa_2019/baike_qa_valid.csv'

# 今日头条新闻多标签分类
path_byte_multi_news_train = path_root + '/data/byte_multi_news/train.csv'
path_byte_multi_news_valid = path_root + '/data/byte_multi_news/valid.csv'
path_byte_multi_news_label = path_root + '/data/byte_multi_news/labels.csv'

# classify data of baidu qa 2019
path_sim_webank_train = path_root + '/data/sim_webank/train.csv'
path_sim_webank_valid = path_root + '/data/sim_webank/valid.csv'
path_sim_webank_test = path_root + '/data/sim_webank/test.csv'

# classfiy multi labels 2021
path_multi_label_train = path_root + '/data/multi_label/train.csv'
path_multi_label_valid = path_root + '/data/multi_label/valid.csv'
path_multi_label_labels = path_root + '/data/multi_label/labels.csv'
path_multi_label_tests = path_root + '/data/multi_label/tests.csv'

# 路径抽象层
path_label = path_multi_label_labels
path_train = path_multi_label_train
path_valid = path_multi_label_valid
path_tests = path_multi_label_tests
path_edata = path_root + "/../out/error_data.csv"

# fast_text config
path_out = path_root + "/../out"
# 模型目录
path_model_dir =  path_root + "/data/model/fast_text/"
# 语料地址
path_model = path_root + '/data/model/fast_text/model_fast_text.h5'
# 超参数保存地址
path_hyper_parameters =  path_root + '/data/model/fast_text/hyper_parameters.json'
# embedding微调保存地址
path_fineture = path_root + "/data/model/fast_text/embedding_trainable.h5"
# 保持 分类-标签 索引
path_category = path_root + '/data/multi_label/category2labels.json'
# l2i_i2l
path_l2i_i2l = path_root + '/data/multi_label/l2i_i2l.json'
