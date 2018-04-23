from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Concatenate,Bidirectional, Flatten, Conv2D, MaxPooling2D,GlobalMaxPooling2D, ConvLSTM2D, TimeDistributed , CuDNNLSTM,GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.metrics import top_k_categorical_accuracy
from utilities import top1_loss

train_data_gen = ImageDataGenerator(rescale=1./255,
                                    #vertical_flip=True,
                                    #rotation_range=20,
                                    #width_shift_range=0.2,
                                    #height_shift_range=0.2,
                                    #zoom_range=0.2
                                    )
train_generator = train_data_gen.flow_from_directory(directory='assets/train/',
                             target_size=(224,224),
                            batch_size=32,
                             class_mode='categorical')

valid_data_gen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_data_gen.flow_from_directory(directory='assets/valid/',
                             target_size=(224,224),
                            batch_size=32,
                             class_mode='categorical')

base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=(224,224,3))
top_model = load_model('models/Inceptionv3/LSTM_old/top_model.hdf5', custom_objects={'top1_loss':top1_loss})
x = base_model.output
predictions = top_model(x)

model = Model(inputs=base_model.input, outputs=predictions)
#for layer in base_model.layers:
#   layer.trainable = False


model.summary()
model.compile(optimizer=Adam(lr = 0.00005), loss='categorical_crossentropy',metrics=[top1_loss])
file_path = 'models/Inceptionv3/LSTM_old/model.hdf5'
# train the model on the new data for a few epochs
check_point = ModelCheckpoint(file_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
early_stop = EarlyStopping(patience=3)
history = model.fit_generator(train_generator,
                              steps_per_epoch=5000,
                              epochs=10,
                              validation_data=valid_generator,
                              validation_steps=500,
                              verbose=1,
                              callbacks=[early_stop, check_point])