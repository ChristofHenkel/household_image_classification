import json
import pandas as pd
from global_variables import TRAIN_DATA_FN, TEST_DATA_FN, VALID_DATA_FN,SAMPLE_SUBMISSION, TEST_FOLDER, VALID_FOLDER
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import load_model
import os
import numpy as np
from tqdm import tqdm



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
    def name(row):
        fp = VALID_FOLDER + str(row['label_id']) + '/' + str(row['image_id']) + '.jpeg'
        return fp
    valid_data['fp'] = valid_data.apply(name, axis=1)
    del data, data_url, labels
    valid_data.to_csv('assets/valid_data.csv', index=False)

def save_test_df():
    data = json.load(open(TEST_DATA_FN))
    #Saved On Data Frame
    data_url = pd.DataFrame.from_dict(data['images'])
    data_url.to_csv('assets/test_data.csv', index=False)


def top1_loss(y_true,y_pred):
    return 1- top_k_categorical_accuracy(y_true,y_pred,k=1)

def score_model(model_fn, image_size = (224,224)):
    #model_fn = 'models/InceptionResNetV2/LSTM/inception_resnet_lstm.hdf5'
    valid_data_gen = ImageDataGenerator(rescale=1. / 255)
    valid_generator = valid_data_gen.flow_from_directory(directory='assets/valid/',
                                                         target_size=image_size,
                                                         batch_size=32,
                                                         class_mode='categorical',
                                                         shuffle=False)
    model = load_model(model_fn,custom_objects={'top1_loss': top1_loss})
    score = model.evaluate_generator(valid_generator,197)
    return score


#prediction = pd.read_csv('models/Inceptionv3/Inception_LSTM/prediction_valid.csv',index_col=0)

def acc_prediction_valid(df):
    df_copy = df.copy()
    df_copy['prediction'] = df_copy.idxmax(axis=1)
    df_copy.reset_index(level=0, inplace=True)
    df_copy['label'] = df_copy['fns'].apply(lambda row: row.split('/')[0])

    correct = df_copy[df_copy['prediction'] == df_copy['label']]
    acc = 1-correct.shape[0]/df_copy.shape[0]
    return acc

def bag_by_average(predicts_list):
    bagged_predicts = np.zeros(predicts_list[0].shape)
    for predict in predicts_list:
        bagged_predicts += predict

    bagged_predicts/= len(predicts_list)
    return  bagged_predicts

def bag_by_geomean(predicts_list):
    bagged_predicts = np.ones(predicts_list[0].shape)
    for predict in predicts_list:
        bagged_predicts *= predict

    bagged_predicts **= (1. / len(predicts_list))
    return  bagged_predicts

