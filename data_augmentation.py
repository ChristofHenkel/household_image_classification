"""
additional data augmentation would go here

"""


from preprocessing2 import CustomImageDataGenerator
import os
from tqdm import tqdm
BATCH_SIZE = 1
epochs = 1

train_data_gen = CustomImageDataGenerator(rescale=1./255,
                                    vertical_flip=True,
                                    horizontal_flip=True,
                                    rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    zoom_range=0.2,
                                    shear_range=0.2,
                                    normalize=True,
                                    brightness_range=(0.7,1.3)
                                    )
#folders = os.listdir('assets/train/')
#folder = folders[0]

for folder in os.listdir('assets/train/'):
    if not os.path.exists('debug/'+ folder):
        os.makedirs('debug/'+ folder)
    train_generator = train_data_gen.flow_from_directory(directory='assets/train/',
                                 target_size=(224,224),
                                batch_size=BATCH_SIZE,
                                                         classes=[folder],
                                 class_mode='categorical', save_to_dir='debug/'+ folder)
    for epoch in range(epochs):
        steps = train_generator.classes.size//BATCH_SIZE
        for s in tqdm(range(steps)):
            x, y = train_generator.__next__()