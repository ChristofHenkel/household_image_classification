from global_variables import VALID_FOLDER
from keras.preprocessing.image import load_img, ImageDataGenerator
from keras.models import Model, load_model
from keras.applications.mobilenet import relu6, DepthwiseConv2D
import numpy as np
import pandas as pd
from utilities import top1_loss
from tqdm import tqdm
import os
from utilities import bag_by_geomean, get_label2class_id
import pickle

def create_l2_train_data(model_path, model_fn, tta=0, out_path = None, out_name = None):
    if tta==0:
        valid_data_gen = ImageDataGenerator(rescale=1. / 255)
    else:
        with open(model_path + 'train_data_gen.p','rb') as f:
            valid_data_gen = pickle.load(f)
    #valid_data_gen = ImageDataGenerator(rescale=1. / 255, vertical_flip=True,
    #                                    rotation_range=20,
    #                                    width_shift_range=0.2,
    #                                    height_shift_range=0.2,
    #                                    zoom_range=0.2,
    #                                    shear_range=0.2)

    valid_generator = valid_data_gen.flow_from_directory(directory=VALID_FOLDER,
                                                         target_size=(224, 224),
                                                         batch_size=16,
                                                        class_mode='categorical', shuffle=False)
    print('loading model')
    custom_model_objects = {'top1_loss':top1_loss}
    if 'MobileNet' in model_path:
        custom_model_objects['relu6'] = relu6
        custom_model_objects['DepthwiseConv2D']: DepthwiseConv2D
    model = load_model(model_path + model_fn, custom_objects=custom_model_objects)

    xs = []


    for i in range(tta+1):
        x = model.predict_generator(valid_generator,verbose=1)
        xs.append(x)
    x_bagged = bag_by_geomean(xs)
    fns = valid_generator.filenames

    label2labelid = (valid_generator.class_indices)
    #labelid2label = {label2labelid[label]:label for label in label2labelid}

    prediction = pd.DataFrame()
    prediction['fns'] = fns
    labels = [l for l in label2labelid]
    for l in labels:
        prediction[l] = x_bagged[:,label2labelid[l]]

    if out_path is None:
        out_path = model_path
    if out_name is None:
        out_name ='prediction_valid_tta%s.csv'%tta
    prediction.to_csv(out_path + out_name,index=False)

def create_l2_test_data(model_path, model_fn, tta=0, out_path = None, out_name = None):

    with open(model_path + 'train_data_gen.p','rb') as f:
        test_data_gen = pickle.load(f)
    #valid_data_gen = ImageDataGenerator(rescale=1. / 255, vertical_flip=True,
    #                                    rotation_range=20,
    #                                    width_shift_range=0.2,
    #                                    height_shift_range=0.2,
    #                                    zoom_range=0.2,
    #                                    shear_range=0.2)

    test_generator = test_data_gen.flow_from_directory(directory='assets/',
                                                         target_size=(224, 224),
                                                         batch_size=16,
                                                       classes = ['test'],
                                                        class_mode='categorical', shuffle=False)

    label2class_id = get_label2class_id('assets/train/')
    #labelid2label = {label2class_id[label]: label for label in label2class_id}

    print('loading model')
    custom_model_objects = {'top1_loss':top1_loss}
    if 'MobileNet' in model_path:
        custom_model_objects['relu6'] = relu6
        custom_model_objects['DepthwiseConv2D']: DepthwiseConv2D
    model = load_model(model_path + model_fn, custom_objects=custom_model_objects)

    xs = []
    for i in range(tta+1):
        x = model.predict_generator(test_generator,verbose=1)
        xs.append(x)
    x_bagged = bag_by_geomean(xs)
    fns = test_generator.filenames

    prediction = pd.DataFrame()
    prediction['fns'] = fns
    labels = [l for l in label2class_id]
    for l in labels:
        prediction[l] = x_bagged[:,label2class_id[l]]

    if out_path is None:
        out_path = model_path
    if out_name is None:
        out_name ='prediction_test_tta%s.csv'%tta
    prediction.to_csv(out_path + out_name,index=False)


OLD_MODELS = ['DenseNet121/LSTM_old/',
              'InceptionResNetV2/LSTM_old/',
              'Inceptionv3/Dense_old/',
              'Inceptionv3/LSTM_old/',
              'MobileNet/LSTM_old/',
              'MobileNet/SeperableConv_old/',
              'Xception/GlobalPooling_old/',
              'Xception/Gru_256_old/',
              'Xception/Gru_512_old/',
              'ResNet50/Dense_old/'
              'Xception/LSTM_old/',
              ]
OLD_MODELS = ['models/'+m for m in OLD_MODELS]

#for m in OLD_MODELS:
#    create_l2_train_data(m,'model.hdf5',tta=12)
#    create_l2_test_data(m, 'model.hdf5', tta=12)

#create_l2_train_data('models/DenseNet201/LSTM512/','model.hdf5',tta=0)
create_l2_train_data('models/DenseNet201/LSTM512/','model.hdf5',tta=12)
