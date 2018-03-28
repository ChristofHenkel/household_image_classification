from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from utils import top1_loss



valid_data_gen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_data_gen.flow_from_directory(directory='assets/valid/',
                             target_size=(224,224),
                            batch_size=64,
                             class_mode='categorical',)

model = load_model('models/VGG16/VGG16_LSTM3/vgg16_lstm.hdf5',custom_objects={'top1_loss':top1_loss})
score = model.evaluate_generator(valid_generator, 20, workers=12)

print('loss %s ------- t1loss %s' %(score[0],score[1]))


