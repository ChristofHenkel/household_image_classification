import numpy as np
import os
import pickle
from tqdm import tqdm
from keras.layers import Input, Dense, Bidirectional,LSTM, Conv1D, Flatten, MaxPool1D, TimeDistributed, CuDNNLSTM, GlobalMaxPool2D, GlobalAveragePooling2D
from keras import layers
from keras.callbacks import EarlyStopping,ModelCheckpoint, Callback
from keras.models import Sequential, Model
from keras.optimizers import Nadam, SGD, Adam
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.metrics import top_k_categorical_accuracy
import random
import gzip
from utilities import top1_loss

bn_folder = 'assets/bn_xception_train_224/'
fns = os.listdir(bn_folder)
random.shuffle(fns,random=random.seed(43))
split_at = len(fns)//10
fns_train = fns[split_at:]
fns_valid = fns[:split_at]
batch_size = 32

def data_gen_train():

    while True:
        for fn in fns_train:
            with gzip.open(bn_folder + fn,'rb') as f:
                content = pickle.load(f)

            yield content[0], content[1]

def data_gen_valid():
    while True:
        for fn in fns_valid:
            with gzip.open(bn_folder + fn,'rb') as f:
                content = pickle.load(f)

            yield content[0], content[1]



def build_model(lr, decay):
    inp = Input(shape=(7,7,2048))
    main = GlobalAveragePooling2D()(inp)
    out = Dense(128, activation = 'sigmoid')(main)

    model = Model(inputs=inp, outputs = out)
    model.compile(optimizer=Adam(lr = lr, decay=decay), loss='categorical_crossentropy',metrics=[top1_loss])
    model.summary()
    return model

class LearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print('\nLR: {:.6f}\n'.format(lr))

def grid_search_generator(building_model_func, lrs, decays, train_gen, valid_gen, path):


    for lr in lrs:
        for d in decays:
            model = building_model_func(lr = lr, decay = d)
            check_point = ModelCheckpoint(path + 'top_model_' + '_'.join([str(lr),str(d)]) + '.hdf5',
                                          monitor="val_loss", mode="min",
                                          save_best_only=True, verbose=1)
            early_stop = EarlyStopping(patience=3)
            history = model.fit_generator(train_gen(),
                                          validation_data = valid_gen(),
                                          callbacks=[early_stop, check_point, LearningRateTracker()],
                                          validation_steps= 540,
                                          steps_per_epoch=5400,
                                          epochs = 100)
            best_val = min(history.history['val_loss'])
            best_val_top1 = min(history.history['val_top1_loss'])
            with open(path + 'summary.txt','a') as f:
                f.write(', '.join([str(lr),str(d),str(best_val),str(best_val_top1)]) + '\n')

#class LearningRateTracker(Callback):
#    def on_epoch_end(self, epoch, logs=None):
#        lr = self.model.optimizer.lr
#        decay = self.model.optimizer.decay
#        iterations = self.model.optimizer.iterations
#        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
#        print(K.eval(lr_with_decay))

"""


# define your model

sgd = SGD(lr=0.01, decay=0.9)
model.compile(loss='mse', optimizer=sgd)
model.fit(x, y, callbacks=[SGDLearningRateTracker()])
"""
grid_search_generator(build_model,
                      lrs=[0.0001],
                      decays=[0.0],
                      train_gen=data_gen_train,
                      valid_gen=data_gen_valid,
                      path='models/Xception/Dense/')