#coding=utf-8
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from argparse import ArgumentParser


def feature_extraction(file):

    features = np.loadtxt(file)
    return features


def getLable(file):

    label = []
    with open(file, 'r',encoding='UTF-8') as f:
        line = f.readlines()

    for i in range(0, len(line)):
        targets = line[i].strip('\n')
        target = targets.split('\t')[-1]
        label.append(float(target))
    f.close()

    new_label = np.array(label)

    return new_label


def train(file,feature_Path):

    features = feature_extraction(feature_Path)
    lables = getLable(file)

    rf = RandomForestRegressor(n_estimators=256,max_depth=6,max_features=3,
                             min_samples_leaf=2,min_samples_split=3,bootstrap=False)
    rf_model = rf.fit(features,lables)

    return rf_model


def evaluate(model, test_file, feature_Path, save_model_path):

    features = feature_extraction(feature_Path)
    test_labels = getLable(test_file)

    rf_predict = model.predict(features)
    MAE = mean_absolute_error(test_labels, rf_predict)
    MSE = mean_squared_error(test_labels, rf_predict)
    RMSE = MSE**0.5
    r2 = r2_score(test_labels, rf_predict)

    print('Model Performance')
    print('MAE: {:0.2f}'.format(MAE))
    print('MSE: {:0.2f}'.format(MSE))
    print('RMSE: {:0.2f}'.format(RMSE))
    print('r2: {:0.2f}'.format(r2))

    joblib.dump(model, save_model_path + '-' + format(MAE,'.2f')+ '-' +
                format(MSE,'.2f') + '-' + format(RMSE,'.2f') + '-' + format(r2,'.2f') + '.m')


def main():

    parser = ArgumentParser()
    parser.add_argument('--train_label_path', default='dataset/beautiful/beautiful_trainsets_lable.txt',
                        help='Path for train label file')
    parser.add_argument('--train_features_path', default='dataset/beautiful/beautiful.txt',
                        help='Path for train features file')
    parser.add_argument('--test_label_path', default='dataset/beautiful/beautiful_testsets_lable.txt',
                        help='Path for test label file')
    parser.add_argument('--test_features_path', default='dataset/beautiful/beautiful.txt',
                        help='Path for test features file')
    parser.add_argument('--save_model_path', default='model_weights/beautiful/beautiful-model',
                        help='Path to save model')
    args = parser.parse_args()

    train_label = args.train_label_path
    test_label = args.test_label_path
    train_features = args.train_features_path
    test_features = args.test_features_path

    model = train(train_label, train_features)
    evaluate(model, test_label, test_features, args.save_model_path)


if __name__ == '__main__':

    main()
