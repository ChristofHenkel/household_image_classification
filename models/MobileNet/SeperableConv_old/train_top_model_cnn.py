import numpy as np
import os
import pickle
from tqdm import tqdm
from keras.layers import Input, Dense,SeparableConv2D, Bidirectional, Conv2D, Flatten, MaxPool2D, TimeDistributed, CuDNNLSTM, GlobalMaxPool2D, GlobalAveragePooling2D
from keras import layers
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from keras.metrics import top_k_categorical_accuracy
import random
from utilities import top1_loss

bn_folder = 'assets/bn_mobilenet_224/'
fns = os.listdir(bn_folder)
random.shuffle(fns,random=random.seed(43))
split_at = len(fns)//10
fns_train = fns[split_at:]
fns_valid = fns[:split_at]
batch_size = 32

def data_gen_train():

    while True:
        for fn in fns_train:
            with open(bn_folder + fn,'rb') as f:
                content = pickle.load(f)

            yield content[0], content[1]

def data_gen_valid():
    while True:
        for fn in fns_valid:
            with open(bn_folder + fn,'rb') as f:
                content = pickle.load(f)

            yield content[0], content[1]


inp = Input(shape=(7,7,1024))
main = layers.SeparableConv2D(2048,3,padding='same')(inp)
main = layers.AveragePooling2D((3,3))(main)
main = Flatten()(main)
main = layers.Dropout(0.4)(main)
out = Dense(128, activation = 'sigmoid')(main)

model = Model(inputs=inp, outputs = out)
model.compile(optimizer=Adam(lr = 0.0001), loss='categorical_crossentropy',metrics=[top1_loss])
model.summary()


check_point = ModelCheckpoint('models/MobileNet/CNN/top_model.hdf5', monitor="val_loss", mode="min", save_best_only=True, verbose=1)
early_stop = EarlyStopping(patience=3)
history = model.fit_generator(data_gen_train(),
                    validation_data = data_gen_valid(),
                    callbacks=[early_stop, check_point],
                    validation_steps= 540,
                    steps_per_epoch=5400,
                    epochs = 100)