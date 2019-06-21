# coding=utf-8

import sys, os
from utils import load_model, extend_maps, prepocess_data_for_lstmcrf
from data import build_corpus, build_corpus_test
from evaluating import Metrics
from evaluate import ensemble_evaluate

HMM_MODEL_PATH = './result/hmm.pkl'
CRF_MODEL_PATH = './result/crf.pkl'
BiLSTM_MODEL_PATH = './result/bilstm.pkl'
BiLSTMCRF_MODEL_PATH = './result/bilstm_crf.pkl'

REMOVE_O = False  # 在评估的时候是否去除O标记

def main(corpus_dir, test_file):
    # 读取数据
    # print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train", data_dir=corpus_dir)
    test_word_lists, test_tag_lists = build_corpus_test(test_file)
    # print(test_word_lists, test_tag_lists)

    # print("加载并评估bilstm+crf模型...")
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    bilstm_model = load_model(BiLSTMCRF_MODEL_PATH)
    bilstm_model.model.bilstm.bilstm.flatten_parameters()  # remove warning
    test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
        test_word_lists, test_tag_lists, test=True
    )
    lstmcrf_pred, target_tag_list = bilstm_model.test(test_word_lists, test_tag_lists,
                                                      crf_word2id, crf_tag2id)
    for i in range(len(lstmcrf_pred)):
        for j in range(len(lstmcrf_pred[i])):
            print(test_word_lists[i][j] + "\t" + lstmcrf_pred[i][j])
        print()



if __name__ == "__main__":
    '''
    训练模型脚本，可以指定训练语料的路径
    '''
    if len(sys.argv) >= 3:
        corpus_dir = sys.argv[1]
        test_file = sys.argv[2]
    else:
        print("Usage: %s corpus_dir test_file" % sys.argv[0])
        sys.exit()

    ## 将test file的格式修改
    test_dir = os.path.dirname(os.path.abspath(test_file))
    test_name = os.path.basename(os.path.abspath(test_file))
    new_test_file = os.path.join(test_dir, "format." + test_name)

    fw = open(new_test_file, 'w')
    with open(test_file, 'r', encoding="utf-8") as fin:
        for line in fin:
            line = line.strip("\n").strip()
            if line == "":
                fw.write("\n")
                continue
            line_lst = [x for x in line]
            sentence = "\n".join(line_lst)
            fw.write(sentence + 2*"\n")
    fw.close()

    # test
    main(corpus_dir, new_test_file)
