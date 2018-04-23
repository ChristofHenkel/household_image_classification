import numpy as np
import os
import pickle
from tqdm import tqdm
from keras.layers import Input, Dense, Conv1D, Flatten, MaxPool1D, TimeDistributed, CuDNNLSTM, GlobalMaxPool2D, GlobalAveragePooling2D
from keras import layers
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from keras.metrics import top_k_categorical_accuracy
import random
from utilities import top1_loss


fns = os.listdir('assets/bn_inception_resnet/')
random.shuffle(fns)
split_at = len(fns)//10
fns_train = fns[split_at:]
fns_valid = fns[:split_at]
batch_size = 32

def data_gen_train():

    while True:
        for fn in fns_train:
            with open('assets/bn_inception_resnet/' + fn,'rb') as f:
                content = pickle.load(f)

            yield content[0], content[1]

def data_gen_valid():
    while True:
        for fn in fns_valid:
            with open('assets/bn_inception_resnet/' + fn,'rb') as f:
                content = pickle.load(f)

            yield content[0], content[1]


inp = Input(shape=(5,5,1536))
main = TimeDistributed(CuDNNLSTM(256))(inp)
main = CuDNNLSTM(256)(main)
main = layers.Dropout(0.4)(main)
out = Dense(128, activation = 'sigmoid')(main)

model = Model(inputs=inp, outputs = out)
model.compile(optimizer=Adam(lr = 0.0001), loss='categorical_crossentropy',metrics=[top1_loss])
model.summary()


check_point = ModelCheckpoint('models/InceptionResNetV2/LSTM/top_model.hdf5', monitor="val_loss", mode="min", save_best_only=True, verbose=1)
early_stop = EarlyStopping(patience=3)
model.fit_generator(data_gen_train(),
                    validation_data = data_gen_valid(),
                    callbacks=[early_stop, check_point],
                    validation_steps= 540,
                    steps_per_epoch=5400,
                    epochs = 100)