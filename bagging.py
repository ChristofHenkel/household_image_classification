import pandas as pd
import global_variables
from utilities import acc_prediction_valid, bag_by_average, bag_by_geomean
import numpy as np

models = ['models/Inceptionv3/Inception_LSTM/',
          'models/InceptionResNetV2/LSTM/',
          'models/DenseNet121/LSTM/',
          'models/Inceptionv3/Inception_Dense/',
          'models/Xception/GlobalPooling/',
          'models/Xception/Gru_256/',
          'models/Xception/Gru_512/',
          'models/Xception/LSTM2/',
          #'models/MobileNet/BiLSTM_256/',
          #'models/MobileNet/SeperableConv/',
          'models/DenseNet201/Last_layer/',
          'models/DenseNet201/Last_layer_with_dropout/'
          ]

csv_files = [m + 'prediction_valid.csv' for m in models]
dfs = [pd.read_csv(csv_file, index_col=0) for csv_file in csv_files]

for k, df in enumerate(dfs):
    print(acc_prediction_valid(df))

xs1 = [df.values for df in dfs]
print(np.round(np.corrcoef(np.array([x.flatten() for x in xs1])),2))



x1 = bag_by_geomean(xs1)

df_new = dfs[0].copy()
df_new[df_new.columns] = x1
print(acc_prediction_valid(df_new))



