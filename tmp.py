import os
import shutil
from tqdm import tqdm
train = os.listdir('assets/train/')

for item in tqdm(train):
    label = item.strip('.jpeg').split('_')[-1]
    fn = ''.join(item.strip('.jpeg').split('_')[:-1]) + '.jpeg'
    if not os.path.exists('assets/train/' + label + '/'):
        os.mkdir('assets/train/' + label + '/')
    shutil.move('assets/train/' + item,'assets/train/' + label + '/' + fn)

valid = os.listdir('assets/valid/')

for item in tqdm(valid):
    label = item.strip('.jpeg').split('_')[-1]
    fn = ''.join(item.strip('.jpeg').split('_')[:-1]) + '.jpeg'
    if not os.path.exists('assets/valid/' + label + '/'):
        os.mkdir('assets/valid/' + label + '/')
    shutil.move('assets/valid/' + item,'assets/valid/' + label + '/' + fn)

from tqdm import tqdm
import shutil
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, ConvLSTM2D, TimeDistributed, CuDNNLSTM
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import top_k_categorical_accuracy
import os

for i in tqdm(range(1,129)):
    num_img = len(os.listdir('assets/train/'+ str(i) + '/'))
    #shutil.copytree('assets/train/'+ str(i) + '/', 'assets/tmp2/'+ str(i) + '/'+ str(i) + '/')
    train_data_gen = ImageDataGenerator()
    if not os.path.exists('assets/tmp3/'+ str(i) + '/'):
        os.mkdir('assets/tmp3/'+ str(i) + '/')
    train_generator = train_data_gen.flow_from_directory(directory='assets/tmp2/'+ str(i) + '/',
                                                         target_size=(224, 224),
                                                         batch_size=num_img,
                                                         class_mode='categorical', save_to_dir='assets/tmp3/'+ str(i) + '/',shuffle=False)
    _ = train_generator.next()
