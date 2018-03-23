import os
import shutil
from tqdm import tqdm
train = os.listdir('assets/valid/')

for item in tqdm(train):
    label = item.strip('.jpeg').split('_')[-1]
    fn = ''.join(item.strip('.jpeg').split('_')[:-1]) + '.jpeg'
    if not os.path.exists('assets/valid/' + label + '/'):
        os.mkdir('assets/valid/' + label + '/')
    shutil.move('assets/valid/' + item,'assets/valid/' + label + '/' + fn)

