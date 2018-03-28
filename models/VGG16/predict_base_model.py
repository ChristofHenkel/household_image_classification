from tqdm import tqdm
import shutil
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from keras.applications.vgg16 import VGG16
import pickle
import gzip
from global_variables import IMAGE_NET_MEAN


maxlen = 192171
img_nums = {i:len(os.listdir('assets/train_224/'+ str(i) + '/')) for i in range(1,129)}
train_data_gen = ImageDataGenerator(rescale=1. / 255,featurewise_center=True)
train_data_gen.mean = 1. / 255 * np.array(IMAGE_NET_MEAN,dtype=np.float32).reshape(1,1,3)
train_generator = train_data_gen.flow_from_directory(directory='assets/train_224/',
                                                     target_size=(224, 224),
                                                     batch_size=32,
                                                    class_mode='categorical', shuffle=False)

base_model = VGG16(weights='imagenet', include_top=False,input_shape=(224,224,3))

for b in tqdm(range(maxlen//32)):
    x = train_generator.next()
    z = base_model.predict_on_batch(x[0])
    with gzip.open('assets/bn_vgg16_224/' + str(b) + '.pz','wb',compresslevel=6) as f:
        pickle.dump((z,x[1]),f)
