"""
data needs to be seperated by classes in folders in 'assests/train/
"""
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, ConvLSTM2D, TimeDistributed , CuDNNLSTM, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.metrics import top_k_categorical_accuracy
from utils import top1_loss

train_data_gen = ImageDataGenerator(rescale=1./255,
                                    horizontal_flip=True,
                                    rotation_range=30,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2)
train_generator = train_data_gen.flow_from_directory(directory='assets/train/',
                             target_size=(224,224),
                            batch_size=32,
                             class_mode='categorical')

valid_data_gen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_data_gen.flow_from_directory(directory='assets/valid/',
                             target_size=(224,224),
                            batch_size=32,
                             class_mode='categorical',)

base_model = ResNet50(weights='imagenet', include_top=False,input_shape=(224,224,3))

x = base_model.output
# let's add a fully-connected layer
x = Flatten()(x)
x = Dropout(0.2)(x)
#x = MaxPooling2D(2)(x)


#x = Dense(256, activation='relu')(encoded_columns)

predictions = Dense(128, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

#for layer in base_model.layers[:-6]:
#    layer.trainable = False


model.summary()
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(lr = 0.00001), loss='categorical_crossentropy',metrics=[top1_loss])
file_path = 'models/ResNet50/Dense/model.hdf5'
# train the model on the new data for a few epochs
check_point = ModelCheckpoint(file_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
early_stop = EarlyStopping(patience=5)
history = model.fit_generator(train_generator,
                              steps_per_epoch=200,
                              epochs=50,
                              validation_data=valid_generator,
                              validation_steps=20,
                              verbose=1,
                              callbacks=[early_stop, check_point])


