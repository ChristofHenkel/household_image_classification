from keras.applications.densenet import DenseNet201
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout,Bidirectional,CuDNNGRU, SpatialDropout1D, Flatten, Conv2D, MaxPooling2D,GlobalMaxPooling2D, ConvLSTM2D, TimeDistributed , CuDNNLSTM,GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras.metrics import top_k_categorical_accuracy
from keras import regularizers
from utilities import top1_loss
import pickle

BATCH_SIZE = 16
MODEL_PATH = 'models/DenseNet201/Gru512/'

train_data_gen = ImageDataGenerator(rescale=1./255,
                                    vertical_flip=True,
                                    rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    zoom_range=0.2,shear_range=0.2
                                    )
train_generator = train_data_gen.flow_from_directory(directory='assets/train/',
                             target_size=(224,224),
                            batch_size=BATCH_SIZE,
                             class_mode='categorical')

valid_data_gen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_data_gen.flow_from_directory(directory='assets/valid/',
                             target_size=(224,224),
                            batch_size=BATCH_SIZE,
                             class_mode='categorical')

base_model = DenseNet201(weights='imagenet', include_top=False,input_shape=(224,224,3))
main = TimeDistributed(Bidirectional(CuDNNGRU(512)))(base_model.output)
main = SpatialDropout1D(0.5)(main)
main = Bidirectional(CuDNNGRU(512))(main)
main = Dropout(0.5)(main)
predictions = Dense(128,activation='softmax',kernel_regularizer=regularizers.l2(0.0001))(main)


model = Model(inputs=base_model.input, outputs=predictions)


model.summary()
model.compile(optimizer=Adam(lr = 0.00003), loss='categorical_crossentropy',metrics=[top1_loss])

file_path = MODEL_PATH + 'model.hdf5'
# train the model on the new data for a few epochs
check_point = ModelCheckpoint(file_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
early_stop = EarlyStopping(patience=4)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=2, min_lr=0)
history = model.fit_generator(train_generator,
                              steps_per_epoch=train_generator.classes.size//BATCH_SIZE,
                              epochs=20,
                              validation_data=valid_generator,
                              validation_steps=valid_generator.classes.size//BATCH_SIZE,
                              verbose=1,
                              callbacks=[early_stop, check_point, reduce_lr])

with open(MODEL_PATH + 'history.p','wb') as f:
    pickle.dump(history.history,f)

with open(MODEL_PATH + 'train_data_gen.p','wb') as f:
    pickle.dump(train_data_gen,f)
#TB call back