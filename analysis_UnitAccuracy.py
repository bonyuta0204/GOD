# coding: utf-8

import os

import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt
import gd_parameters as gd
import pickle
import bdpy
import sys

from memory_profiler import profile

from gd_features import Features
from itertools import product
from bdpy.stats import corrcoef
from time import time


# get parameters from gd_features
data_dir = gd.data_dir
subject_list = gd.subject_list
roi_list = gd.roi_list
feature_file = gd.feature_file
feature_type = gd.feature_type
result_dir = gd.result_dir
feature_file = gd.feature_file
feature_type = gd.feature_type
result_dir = gd.result_dir
features = Features(os.path.join(data_dir, feature_file), feature_type)


def unit_accuracy(acc_result, sbj, roi, feat):
    start_time = time()

    analysis_id = '%s-%s-%s' % (sbj, roi, feat)
    print 'Analysis %s' % analysis_id

    ## Load feature prediction results for each unit
    result_unit_file = os.path.join(result_dir, sbj, roi, feat + '.pkl')
    
    if not os.path.exists(result_unit_file):
        acc_result[sbj][roi][feat]["predacc_image_percept"] = None
        acc_result[sbj][roi][feat]["predacc_category_percept"] = None
        acc_result[sbj][roi][feat]["predacc_category_imagery"] = None
        return acc_result
    with open(result_unit_file, 'rb') as f:
        
        results_unit = pickle.load(f)

    ## Preparing data (image features)
    feature = features.get('value', feat)
    feature_type = features.get('type')
    feature_catlabel = features.get('category_label')

    ## Calculate image and category feature prediction accuracy

    ## Aggregate all units prediction (num_sample * num_unit)
    pred_percept = np.vstack(results_unit['predict_percept_catave']).T
    pred_imagery = np.vstack(results_unit['predict_imagery_catave']).T
    
    cat_percept = results_unit['category_test_percept'][0]
    cat_imagery = results_unit['category_test_imagery'][0]

    ind_imgtest = feature_type == 2
    ind_cattest = feature_type == 3

    test_feat_img = bdpy.get_refdata(feature[ind_imgtest, :],
                                     feature_catlabel[ind_imgtest],
                                     cat_percept);
    test_feat_cat_percept = bdpy.get_refdata(feature[ind_cattest, :],
                                             feature_catlabel[ind_cattest],
                                             cat_percept)
    test_feat_cat_imagery = bdpy.get_refdata(feature[ind_cattest, ],
                                             feature_catlabel[ind_cattest],
                                                     cat_imagery)

    ## Get image and category feature prediction accuracy
    predacc_image_percept = corrcoef(pred_percept, test_feat_img, var='col')
    predacc_category_percept = corrcoef(pred_percept, test_feat_cat_percept, var='col')
    predacc_category_imagery = corrcoef(pred_imagery, test_feat_cat_imagery, var='col')
    
    acc_result[sbj][roi][feat]["predacc_image_percept"] = predacc_image_percept
    #acc_result[sbj][roi][feat]["predacc_category_percept"] = predacc_category_percept
    #acc_result[sbj][roi][feat]["predacc_category_imagery"] = predacc_category_imagery
    return acc_result


def main():
    if not os.path.exists("temp.pkl"):
        # set data dict
        acc_result = {}
        for sbj in subject_list:
            acc_result[sbj] = {}
            for roi in roi_list:
                acc_result[sbj][roi] = {}
                for feat in features.layers:
                    acc_result[sbj][roi][feat] = {}

        # compute accuracy
        for sbj, roi, feat in product(subject_list, roi_list, features.layers):
            acc_result=unit_accuracy(acc_result, sbj, roi, feat)
            print(sys.getsizeof(acc_result))

        with open("temp.pkl", "wb") as f:
            pickle.dump(acc_result, f)
    else:
        with open("temp.pkl", "rb") as f:
            acc_result = pickle.load(f)

    df_list = []
    for sbj in subject_list:
        for roi in roi_list:
            for feat in features.layers:
                if acc_result[sbj][roi][feat]["predacc_image_percept"] is not None:
                    for unit in range(len(acc_result[sbj][roi][feat]["predacc_image_percept"])):
                        df_list.append({
                                "subject":sbj,
                                "ROI":roi,
                                "feature":feat,
                                "unit":unit + 1,
                                "accuracy":acc_result[sbj][roi][feat]["predacc_image_percept"][unit]
                            })

    df = pd.DataFrame(df_list)
    with open(os.path.join(result_dir, "UnitAccuracy.pkl"), "wb") as f:
        pickle.dump(df, f)
if __name__  ==  "__main__":
    main()
