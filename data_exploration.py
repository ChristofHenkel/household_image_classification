from PIL import Image
import pandas as pd
from keras.preprocessing.image import load_img
from tqdm import tqdm

train_data = pd.read_csv('assets/train_data.csv',index_col=0)

data_dict = {}

for id in tqdm(train_data.index.values):
    try:
        img = load_img('assets/train/%s/%s.jpeg'%(train_data.loc[id]['label_id'],id))
        h, w = img.size
        data_dict[id] = {'h':h,
                         'w':w,
                         'square':int(h==w),
                         'ratio':round(h/w,1),
                         'mode':img.mode,
                         'layers':img.layers}
    except:
        pass

size_df = pd.DataFrame.from_dict(data_dict).transpose()
data = train_data.join(size_df)



data['square'].value_counts().plot(kind='bar')