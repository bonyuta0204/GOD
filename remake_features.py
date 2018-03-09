
# coding: utf-8

import pandas as pd
import os
import pickle
import argparse
from makedata_features_alexnet import add_features_df

"""
This script integrate 4 types of features already computed
"""

# set data to work on 
parser = argparse.ArgumentParser(description="Format feature data")
parser.add_argument("directory", action="store", help="Directory which contains feature data")

args = parser.parse_args()
print(args.directory)
## Manually load DataFrame and integrate the result


featuredir = "./data_alex/ImageFeatures_caffe_test/"
outputfile = "./data_alex/ImageFeatures_caffe_processed.pkl"

## Init Pandas dataframe
df = pd.DataFrame(
    columns=['ImageID', 'CatID', 'FeatureType', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6-conv', 'fc7-conv',
             'fc8-conv'])

## Image features --------------------------------------------------
with open(os.path.join(featuredir, 'feature_training.pkl')) as f:
    features = pickle.load(f)
df = add_features_df(df, features, featuretype=1)

with open(os.path.join(featuredir, 'feature_test.pkl')) as f:
    features = pickle.load(f)
df = add_features_df(df, features, featuretype=2)

with open(os.path.join(featuredir, 'feature_category_ave_test.pkl')) as f:
    features = pickle.load(f)
df = add_features_df(df, features, featuretype=3)

with open(os.path.join(featuredir, 'feature_category_ave_candidate.pkl')) as f:
    features = pickle.load(f)
df = add_features_df(df, features, featuretype=4)


## Save merged features --------------------------------------------
df['FeatureType'] = df['FeatureType'].astype('int')
df['CatID'] = df['CatID'].astype('int')

## Drop invalid features
conv1 = df.conv1
df = df[conv1.notnull()]

with open(outputfile, 'wb') as f:
    pickle.dump(df, f)


