import json
import pandas as pd
from global_variables import TRAIN_DATA_FN, TEST_DATA_FN, VALID_DATA_FN
from keras.metrics import top_k_categorical_accuracy

#load data
def save_traing_df():
    data = json.load(open(TRAIN_DATA_FN))
    #Saved On Data Frame
    data_url = pd.DataFrame.from_dict(data['images'])
    labels = pd.DataFrame.from_dict(data['annotations'])
    train_data = data_url.merge(labels, how='inner', on=['image_id'])
    train_data['url'] = train_data['url'].str.get(0)
    del data, data_url, labels
    train_data.to_csv('assets/train_data.csv', index=False)

def save_valid_df():
    data = json.load(open(VALID_DATA_FN))
    #Saved On Data Frame
    data_url = pd.DataFrame.from_dict(data['images'])
    labels = pd.DataFrame.from_dict(data['annotations'])
    valid_data = data_url.merge(labels, how='inner', on=['image_id'])
    valid_data['url'] = valid_data['url'].str.get(0)
    del data, data_url, labels
    valid_data.to_csv('assets/valid_data.csv')

def save_test_df():
    data = json.load(open(TEST_DATA_FN))
    #Saved On Data Frame
    data_url = pd.DataFrame.from_dict(data['images'])
    data_url.to_csv('assets/test_data.csv', index=False)


def top1_loss(y_true,y_pred):
    return 1- top_k_categorical_accuracy(y_true,y_pred,k=1)