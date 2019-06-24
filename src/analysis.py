import os
import time
import argparse
import datetime
import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt

import sys
sys.path.append('../')
from track_identifier.utils import features, vis, misc

from sklearn import ensemble, preprocessing, metrics
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib 


# feature name
FEATURE_NAMES = [
        'pitch_mean',
        'pitch_lowest',
        'num_pitches',
        'poly_ratio',
        'duratoin_mean',
        'duratoin_std']

# instruments
INSTR_CLASS = ['melody', 'drum', 'bass', 'other']
INSTR_COLOR = ['blue', 'tomato', 'green', 'gold']

# hyper parameter
N_ESTIMATORS = 100

def proc(features_dir, result_dir, model_dir, model_name):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # list files
    file_list = []
    for root, _, files in os.walk(features_dir):
        for file in files:
            if file.endswith(".npz"):
                file_list.append(os.path.join(root, file))
                    
    # print to check
    # for idx, file in enumerate(file_list):
    #   print(idx, file) 

    # random samples
    num_all = len(file_list)
    split_ratio = 0.7
    num_train = int(num_all * split_ratio)
    random_idx = np.random.permutation(num_all)
    train_idx = random_idx[:num_train]
    test_idx = random_idx[num_train:]

    # training data
    train_x = []
    train_y = []
    train_fn = []
    for idx in train_idx:
        filename = file_list[idx]
        entry = np.load(filename)
        train_x.append(entry['x'])
        train_y.append(entry['y'])
        train_fn.append(filename)
        
    # testing data
    test_x = []
    test_y = []
    test_fn = []
    for idx in test_idx:
        filename = file_list[idx]
        entry = np.load(filename)
        test_x.append(entry['x'])
        test_y.append(entry['y'])
        test_fn.append(filename)
        
    # statitics
    cnt_class_test = []
    for idx in range(len(INSTR_CLASS)):
        cnt_class_test.append(test_y.count(idx))
        
    cnt_class_train = []
    for idx in range(len(INSTR_CLASS)):
        cnt_class_train.append(train_y.count(idx))


    print('\n[*] Data Amount')
    print('> total: {:5d} - {:s}'.format(
        num_all, 
        str([item1 + item2 for item1, item2 in zip(cnt_class_train, cnt_class_test)])))
    print('> train: {:5d} - {:s}'.format(sum(cnt_class_train), str(cnt_class_train)))
    print('>  test: {:5d} - {:s}'.format(sum(cnt_class_test), str(cnt_class_test)))

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    # build random forest
    forest = ensemble.RandomForestClassifier(n_estimators=N_ESTIMATORS)
    forest.fit(train_x, train_y)

    # prediction
    predict_test_y = forest.predict(test_x)
    predict_train_y = forest.predict(train_x)

    # save
    model_name = os.path.join(model_dir, model_name)
    joblib.dump(forest, model_name)

    # accuracy
    accuracy_test = metrics.accuracy_score(test_y, predict_test_y)
    accuracy_train = metrics.accuracy_score(train_y, predict_train_y)
    print('\n[*] Result')
    print('> train accuracy:', accuracy_train)  
    print('> test accuracy:', accuracy_test)  

    # confusion table
    # train

    plt.figure(dpi=300)
    vis.plot_confusion_table(train_y, predict_train_y, INSTR_CLASS)
    fig_name = os.path.join(result_dir, 'confusion_train.png')
    plt.savefig(fig_name)

    # test
    plt.figure(dpi=300)
    vis.plot_confusion_table(test_y, predict_test_y, INSTR_CLASS)
    fig_name = os.path.join(result_dir, 'confusion_test.png')
    plt.savefig(fig_name)
    
    # error diagnosis
    print('\n--- train ---')
    misc.diagnose_error(train_y, predict_train_y, train_fn, INSTR_CLASS)
    print('\n--- test---')
    misc.diagnose_error(test_y, predict_test_y, test_fn, INSTR_CLASS)

    # plot training distribution
    n_std = 4
    for feature_idx in range(len(FEATURE_NAMES)):
        feature = train_x[:, feature_idx]
        mean = np.mean(feature)
        std = np.std(feature)
        x_range = (mean-std * n_std, mean+std * n_std)
        
        # plot
        plt.figure(dpi=300) 
        for instr_idx in range(len(INSTR_CLASS)):
            track_feature = feature[train_y == instr_idx]
            x_est, y_est = vis.estimate_pdf(track_feature, x_range=x_range)
            vis.plot_distribution(
                x_est, 
                y_est, 
                color=INSTR_COLOR[instr_idx], 
                alpha='0.3', 
                label=INSTR_CLASS[instr_idx])
            
        plt.title(FEATURE_NAMES[feature_idx])
        plt.legend(loc='upper right')
        fig_name = os.path.join(result_dir, FEATURE_NAMES[feature_idx] + '.png')
        plt.savefig(fig_name)
        

if __name__ == '__main__':
    # path
    features_dir = '../data/features'
    result_dir = '../doc'
    model_dir = '../track_identifier/model'
    model_name =  '2019-6-24.pkl'

    # processing
    start_time = time.time()
    proc(features_dir, result_dir, model_dir, model_name)
    end_time = time.time()

    # finish
    runtime = end_time - start_time
    print('\n[*] Finished!')
    print('> Elapsed Time:', str(datetime.timedelta(seconds=runtime))+'\n')
