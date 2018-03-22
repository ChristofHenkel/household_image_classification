from keras.preprocessing.image import ImageDataGenerator

data_gen = ImageDataGenerator()
data_gen.flow_from_directory(directory='assets/train/')