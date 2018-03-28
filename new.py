"""
data needs to be seperated by classes in folders in 'assests/train/
"""
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Concatenate,Bidirectional, Flatten, Conv2D, MaxPooling2D,GlobalMaxPooling2D, ConvLSTM2D, TimeDistributed , CuDNNLSTM,GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.metrics import top_k_categorical_accuracy

train_data_gen = ImageDataGenerator(rescale=1./255,
                                    horizontal_flip=True,
                                    rotation_range=30,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2)
train_generator = train_data_gen.flow_from_directory(directory='assets/train/',
                             target_size=(224,224),
                            batch_size=128,
                             class_mode='categorical')

valid_data_gen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_data_gen.flow_from_directory(directory='assets/valid/',
                             target_size=(224,224),
                            batch_size=128,
                             class_mode='categorical',)

base_model = InceptionResNetV2(weights='imagenet', include_top=False,input_shape=(224,224,3),)

x = base_model.output
avg = GlobalAveragePooling2D()(x)
max_ = GlobalMaxPooling2D()(x)
x = Concatenate()([avg,max_])
# let's add a fully-connected layer
#encoded_rows = TimeDistributed(Bidirectional(CuDNNLSTM(64)))(x)

# Encodes columns of encoded rows.
#encoded_columns = Bidirectional(CuDNNLSTM(64))(encoded_rows)
x = Dense(512, activation='relu')(x)
predictions = Dense(128, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers[:-2]:
   layer.trainable = False

def top1_loss(y_true,y_pred):
    return 1- top_k_categorical_accuracy(y_true,y_pred,k=1)

model.summary()
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(), loss='categorical_crossentropy',metrics=[top1_loss])
file_path = 'vgg19.hdf5'
# train the model on the new data for a few epochs
check_point = ModelCheckpoint(file_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
early_stop = EarlyStopping(patience=3)
history = model.fit_generator(train_generator,
                              steps_per_epoch=200,
                              epochs=50,
                              validation_data=valid_generator,
                              validation_steps=50,
                              verbose=1,
                              callbacks=[early_stop, check_point])
