import pandas as pd
from keras.models import load_model
from utilities import top1_loss
import os
import numpy as np
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator,load_img
from global_variables import SAMPLE_SUBMISSION, TEST_FOLDER

submission = pd.read_csv(SAMPLE_SUBMISSION, index_col=0)
model = load_model('models/Inceptionv3/Inception_LSTM/inception_lstm.hdf5',
                   custom_objects={'top1_loss':top1_loss})

fns = os.listdir(TEST_FOLDER)
fps = [TEST_FOLDER + fn for fn in fns]
test_data = [load_img(fn,target_size=(224,224)) for fn in tqdm(fps)]
test_data = [1. / 255 * np.array(im) for im in tqdm(test_data)]
#test_data = np.concatenate(test_data)

train_data_gen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_data_gen.flow_from_directory(directory='assets/train_224/',
                                                     target_size=(224, 224),
                                                     batch_size=1,
                                                    class_mode='categorical', shuffle=False)
label2labelid = (train_generator.class_indices)
labelid2label = {label2labelid[label]:label for label in label2labelid}

res = {}
for i,d in tqdm(enumerate(test_data)):
    y = model.predict(np.expand_dims(d,axis = 0))
    y = np.argmax(y[0])
    l = labelid2label[y]
    res[fns[i][:-4]] = l

s = submission.to_dict()
sub = s['predicted']
for item in res:
    s['predicted'][int(item[:-4])] = int(res[item])

df = pd.DataFrame.from_dict(s)
df.to_csv('test.csv',index_label='id')