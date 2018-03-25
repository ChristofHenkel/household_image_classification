from tqdm import tqdm
import shutil
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
maxlen = 192171
img_nums = {i:len(os.listdir('assets/train_224/'+ str(i) + '/')) for i in range(1,129)}
train_data_gen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_data_gen.flow_from_directory(directory='assets/train_224/',
                                                     target_size=(224, 224),
                                                     batch_size=1,
                                                    class_mode='categorical', shuffle=False)
for i in range(1,maxlen+1):
    x = train_generator.next()

