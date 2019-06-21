# coding=utf-8

import sys
from utils import load_model, extend_maps, prepocess_data_for_lstmcrf
from data import build_corpus
from evaluating import Metrics
from evaluate import ensemble_evaluate

HMM_MODEL_PATH = './result/hmm.pkl'
CRF_MODEL_PATH = './result/crf.pkl'
BiLSTM_MODEL_PATH = './result/bilstm.pkl'
BiLSTMCRF_MODEL_PATH = './result/bilstm_crf.pkl'

REMOVE_O = False  # 在评估的时候是否去除O标记

def main(corpus_dir="./data/eng_ner_coll2003/"):
    # 读取数据
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train", data_dir=corpus_dir)
    dev_word_lists, dev_tag_lists = build_corpus("val", make_vocab=False, data_dir=corpus_dir)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False, data_dir=corpus_dir)

    '''
    print("加载并评估hmm模型...")
    hmm_model = load_model(HMM_MODEL_PATH)
    hmm_pred = hmm_model.test(test_word_lists,
                              word2id,
                              tag2id)
    metrics = Metrics(test_tag_lists, hmm_pred, remove_O=REMOVE_O)
    metrics.report_scores()  # 打印每个标记的精确度、召回率、f1分数
    metrics.report_confusion_matrix()  # 打印混淆矩阵

    # 加载并评估CRF模型
    print("加载并评估crf模型...")
    crf_model = load_model(CRF_MODEL_PATH)
    crf_pred = crf_model.test(test_word_lists)
    metrics = Metrics(test_tag_lists, crf_pred, remove_O=REMOVE_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    # bilstm模型
    print("加载并评估bilstm模型...")
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
    bilstm_model = load_model(BiLSTM_MODEL_PATH)
    bilstm_model.model.bilstm.flatten_parameters()  # remove warning
    lstm_pred, target_tag_list = bilstm_model.test(test_word_lists, test_tag_lists,
                                                   bilstm_word2id, bilstm_tag2id)
    metrics = Metrics(target_tag_list, lstm_pred, remove_O=REMOVE_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()
    '''

    print("加载并评估bilstm+crf模型...")
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    bilstm_model = load_model(BiLSTMCRF_MODEL_PATH)
    bilstm_model.model.bilstm.bilstm.flatten_parameters()  # remove warning
    test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
        test_word_lists, test_tag_lists, test=True
    )
    lstmcrf_pred, target_tag_list = bilstm_model.test(test_word_lists, test_tag_lists,
                                                      crf_word2id, crf_tag2id)
    metrics = Metrics(target_tag_list, lstmcrf_pred, remove_O=REMOVE_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    '''
    ensemble_evaluate(
        [hmm_pred, crf_pred, lstm_pred, lstmcrf_pred],
        test_tag_lists
    )
    '''


if __name__ == "__main__":
    '''
    训练模型脚本，可以指定训练语料的路径
    '''
    if len(sys.argv) >= 2:
        corpus_dir = sys.argv[1]
        main(corpus_dir)
    else:
        main()
