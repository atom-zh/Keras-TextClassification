# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/12 14:11
# @author   :Mo
# @function :

# 适配linux
import pathlib
import sys
import os
project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(project_path)
# 地址
from keras_textclassification.conf.path_config import path_model_dir, path_hyper_parameters
# 训练验证数据地址
from keras_textclassification.conf.path_config import path_valid
# 数据预处理, 删除文件目录下文件
from keras_textclassification.data_preprocess.text_preprocess import PreprocessTextMulti, load_json, transform_multilabel_to_multihot
# 模型图
from keras_textclassification.m05_TextRCNN.graph import RCNNGraph as Graph
# 模型评估
from sklearn.metrics import classification_report, hamming_loss
# 计算时间
import time
import numpy as np


def pred_tet(path_hyper_parameter=path_hyper_parameters, path_test=None, rate=1.0):
    # 测试集的准确率
    hyper_parameters = load_json(path_hyper_parameter)
    time_start = time.time()

    # graph初始化
    graph = Graph(hyper_parameters)
    print("graph init ok!")
    graph.load_model()
    print("graph load ok!")
    ra_ed = graph.word_embedding

    # 数据预处理
    pt = PreprocessTextMulti(path_model_dir)
    x, y = pt.preprocess_label_ques_to_idx(hyper_parameters['embedding_type'], path_test,
                                                                ra_ed, rate, shuffle=True)
    y_pred = []
    index_y = []
    pred = graph.predict(x)

    print(pred)
    for i in range(len(pred)):
        pre = pt.prereocess_idx(pred[i])
        label_pred = pre[0][0][0]
        label_pred = pt.l2i_i2l['l2i'][label_pred]
        label_multi_idex = transform_multilabel_to_multihot(label_pred, label=51)
        y_pred.append(label_multi_idex)
        index_y.append(y[i].tolist())
        print(pre)
        print(label_multi_idex)
        print(y[i].tolist())
        print('=========================')

    print("data pred ok!")
    # 预测结果转为int类型
    #index_y = [pt.l2i_i2l['l2i'][i] for i in y]
    #index_pred = [pt.l2i_i2l['l2i'][i] for i in y_pred]
    target_names = [pt.l2i_i2l['i2l'][str(i)] for i in range(51)]
    print(target_names)
    # 评估
    report_predict = classification_report(index_y, y_pred, digits=9, target_names=target_names)
    print(report_predict)

    h_loss = hamming_loss(index_y, y_pred)
    print("Hamming Loss = {:.6f}".format(h_loss))
    print("耗时:" + str(time.time() - time_start))

if __name__=="__main__":
    # 测试集预测
    pred_tet(path_test=path_valid, rate=1) # sample条件下设为1,否则训练语料可能会很少