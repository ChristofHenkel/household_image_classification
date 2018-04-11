import numpy as np
import os
import pickle
from tqdm import tqdm
from keras.layers import Input, Dense, Bidirectional,CuDNNGRU, Conv1D, Flatten, MaxPool1D, TimeDistributed, CuDNNLSTM, GlobalMaxPool2D, GlobalAveragePooling2D
from keras import layers
from keras.callbacks import EarlyStopping,ModelCheckpoint, Callback, ReduceLROnPlateau
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.metrics import top_k_categorical_accuracy
import random
import gzip
from utilities import top1_loss

bn_folder = 'assets/bn_DenseNet121_train_224/'
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

def build_model(units, dropout):
    inp = Input(shape=(7,7,1024))
    main = TimeDistributed(Bidirectional(CuDNNLSTM(units)))(inp)
    main = layers.SpatialDropout1D(dropout)(main)
    main = Bidirectional(CuDNNLSTM(units))(main)
    main = layers.Dropout(dropout)(main)
    out = Dense(128, activation = 'sigmoid')(main)

    model = Model(inputs=inp, outputs = out)
    model.compile(optimizer=Adam(lr = 0.0001), loss='categorical_crossentropy',metrics=[top1_loss])
    model.summary()
    return model


class LearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print('\nLR: {:.6f}\n'.format(lr))

def grid_search_generator(building_model_func, units, dropouts, train_gen, valid_gen, path):


    for u in units:
        for d in dropouts:
            model = building_model_func(units=u, dropout = d)
            check_point = ModelCheckpoint(path + 'top_model_' + '_'.join([str(u),str(d)]) + '.hdf5',
                                          monitor="val_loss", mode="min",
                                          save_best_only=True, verbose=1)
            early_stop = EarlyStopping(patience=4)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=2, min_lr=0)
            history = model.fit_generator(train_gen(),
                                          validation_data = valid_gen(),
                                          callbacks=[early_stop, check_point,reduce_lr],
                                          validation_steps= 540,
                                          steps_per_epoch=5400,
                                          epochs = 100)
            best_val = min(history.history['val_loss'])
            best_val_top1 = min(history.history['val_top1_loss'])
            with open(path + 'summary.txt','a') as f:
                f.write(', '.join([str(units),str(d),str(best_val),str(best_val_top1)]) + '\n')

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
                      units=[256],
                      dropouts=[0.5],
                      train_gen=data_gen_train,
                      valid_gen=data_gen_valid,
                      path='models/DenseNet121/LSTM/')