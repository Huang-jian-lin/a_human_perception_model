#coding=utf-8
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os
import joblib
from argparse import ArgumentParser


def getName(img_names_txt):
    img_name = []
    names_file = open(img_names_txt, 'r')
    for line in names_file.readlines():
        img_name.append(line.strip())

    names_file.close()
    return img_name


def getFeatures(features_file):
    features = np.loadtxt(features_file)
    return features


def prediction(features_txt, img_names_txt, model_path, result_file_path):

    datas = getFeatures(features_txt)
    names = getName(img_names_txt)
    rf_model = joblib.load(model_path)
    rf_result = rf_model.predict(datas)
    fp = open(result_file_path, 'w')
    for i in range(len(rf_result)):
        fp.write(names[i])
        fp.write('\t')
        fp.write('%.4f' % rf_result[i])
        fp.write('\n')


def main():
    parser = ArgumentParser()
    parser.add_argument('--probability_feature_file_path', default='probability_feature_file/beautiful_category_probability_info.txt',
                        help='Path for category probability feature file')
    parser.add_argument('--semantic_feature_file_path', default='semantic_feature_file/semantic_info.txt',
                        help='Path for semantic feature file')
    parser.add_argument('--model_path', default='model_weights/beautiful/beautiful-model-1.49-3.46-1.86.m',
                        help='Path for model')
    parser.add_argument('--result_file_path', default='result_file/score_result.txt',
                        help='Path for result file')
    args = parser.parse_args()

    probability_feature_file = open(args.probability_feature_file_path, 'r')
    semantic_feature_file = open(args.semantic_feature_file_path, 'r')
    features_txt = 'fusion_feature_file/fusion_feature.txt'
    img_names_txt = 'img_name.txt'
    fusion_feature_file = open(features_txt, 'w')
    img_name_file = open(img_names_txt, 'w')

    for line1, line2 in zip(probability_feature_file.readlines(), semantic_feature_file.readlines()):
        fusion_feature_file.write(line1.strip().replace(line1.strip().split('\t')[0] + '\t', '') + '\t' +
                                  line2.strip().replace(line2.strip().split('\t')[0] + '\t', '') + '\n')
        img_name_file.write(line1.split('\t')[0] + '\n')

    probability_feature_file.close()
    semantic_feature_file.close()
    fusion_feature_file.close()
    img_name_file.close()

    prediction(features_txt, img_names_txt, args.model_path, args.result_file_path)


if __name__ == '__main__':

    main()
