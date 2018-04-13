from global_variables import VALID_FOLDER
from keras.preprocessing.image import load_img, ImageDataGenerator
from keras.models import Model, load_model
from keras.applications.mobilenet import relu6, DepthwiseConv2D
import numpy as np
import pandas as pd
from utilities import top1_loss
from tqdm import tqdm
import os

def create_l2_data(model_path, model_fn):
    valid_data_gen = ImageDataGenerator(rescale=1. / 255)
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
    x = model.predict_generator(valid_generator,verbose=1)
    fns = valid_generator.filenames

    label2labelid = (valid_generator.class_indices)
    #labelid2label = {label2labelid[label]:label for label in label2labelid}

    prediction = pd.DataFrame()
    prediction['fns'] = fns
    labels = [l for l in label2labelid]
    for l in labels:
        prediction[l] = x[:,label2labelid[l]]

    prediction.to_csv(model_path + 'prediction_valid.csv',index=False)


#create_l2_data('models/Inceptionv3/Inception_LSTM/','inception_lstm.hdf5')
#create_l2_data('models/InceptionResNetV2/LSTM/','inception_resnet_lstm.hdf5')
#create_l2_data('models/DenseNet121/LSTM/','model.hdf5')
#create_l2_data('models/Inceptionv3/Inception_Dense/','inception_dense.hdf5')
#create_l2_data('models/Xception/GlobalPooling/','model.hdf5')
#create_l2_data('models/Xception/Gru_256/','model_256.hdf5')
#create_l2_data('models/Xception/Gru_512/','model_512.hdf5')
#create_l2_data('models/Xception/LSTM2/','model.hdf5')
#create_l2_data('models/MobileNet/BiLSTM_256/','mobilenet_bilstm.hdf5')
#create_l2_data('models/MobileNet/SeperableConv/','mobilenet_sepconv.hdf5')
#create_l2_data('models/DenseNet201/Last_layer/','model.hdf5')
create_l2_data('models/DenseNet201/Last_layer_with_dropout/','model.hdf5')
