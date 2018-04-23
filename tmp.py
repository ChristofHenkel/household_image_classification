import os
import shutil
from tqdm import tqdm
train = os.listdir('assets/train/')

for item in tqdm(train):
    label = item.strip('.jpeg').split('_')[-1]
    fn = ''.join(item.strip('.jpeg').split('_')[:-1]) + '.jpeg'
    if not os.path.exists('assets/train/' + label + '/'):
        os.mkdir('assets/train/' + label + '/')
    shutil.move('assets/train/' + item,'assets/train/' + label + '/' + fn)

valid = os.listdir('assets/valid/')

for item in tqdm(valid):
    label = item.strip('.jpeg').split('_')[-1]
    fn = ''.join(item.strip('.jpeg').split('_')[:-1]) + '.jpeg'
    if not os.path.exists('assets/valid/' + label + '/'):
        os.mkdir('assets/valid/' + label + '/')
    shutil.move('assets/valid/' + item,'assets/valid/' + label + '/' + fn)

from tqdm import tqdm
import shutil
from keras.preprocessing.image import ImageDataGenerator
import os

for i in tqdm(range(1,129)):
    num_img = len(os.listdir('assets/train/'+ str(i) + '/'))
    shutil.copytree('assets/train/'+ str(i) + '/', 'assets/tmp/'+ str(i) + '/'+ str(i) + '/')
    train_data_gen = ImageDataGenerator()
    if not os.path.exists('assets/train_224/'+ str(i) + '/'):
        os.mkdir('assets/train_224/'+ str(i) + '/')
    train_generator = train_data_gen.flow_from_directory(directory='assets/tmp/'+ str(i) + '/',
                                                         target_size=(224, 224),
                                                         batch_size=num_img,
                                                         class_mode='categorical', save_to_dir='assets/train_224/'+ str(i) + '/',shuffle=False)
    _ = train_generator.next()


from tqdm import tqdm
import shutil
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
z = []
for i in tqdm(range(81,129)):
    num_img = len(os.listdir('assets/train_224/'+ str(i) + '/'))
    shutil.copytree('assets/train_224/'+ str(i) + '/', 'assets/tmp/'+ str(i) + '/'+ str(i) + '/')
    train_data_gen = ImageDataGenerator(rescale=1./255)
    train_generator = train_data_gen.flow_from_directory(directory='assets/tmp/'+ str(i) + '/',
                                                         target_size=(224, 224),
                                                         batch_size=num_img,
                                                         class_mode='categorical',shuffle=False)
    x = train_generator.next()
    np.savez_compressed('assets/train_np/' + str(i),np.array(x[0]))
    shutil.rmtree('assets/tmp/'+ str(i) + '/'+ str(i) + '/')





ids = [fn[:-4] for fn in os.listdir('assets/test/')]
ids2 = [fn[:-4] for fn in os.listdir('/home/christof/Downloads/test/')]
new = [i for i in ids2 if i not in ids]

from PIL import Image
for n in new:


    im = Image.open('/home/christof/Downloads/test/' + n + '.png')
    rgb_im = im.convert('RGB')
    rgb_im.save('assets/test/' + n + '.jpg')

from keras.applications.inception_resnet_v2 import InceptionResNetV2

model = InceptionResNetV2(include_top=False,input_shape=(224,224,3))
model.summary()

from preprocessing import CustomImageDataGenerator
ci = CustomImageDataGenerator(contrast_stretching=True,
                              brightness_range=(0.5,1.5),
                              contrast_range=(0.5,1.5),
                              saturation_range=(0.5,1.5))

for i in range(10):
    gen = ci.flow_from_directory(directory='assets/valid/',
                           target_size=(224, 224),
                           batch_size=1, save_to_dir='debug/',
                           class_mode='categorical', shuffle=False)
    x,y = gen.__next__()