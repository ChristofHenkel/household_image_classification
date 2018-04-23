import pandas as pd

csv_models = CSVPredictions(MODELS, 'prediction_valid_tta12.csv')
csv_models_old = CSVPredictions(OLD_MODELS, tta=0)

xs = [df.values for df in csv_models.dfs]
x = bag_by_geomean(xs)
a = csv_models.dfs[0].copy()
a[list(a.columns.values)] = x

labels = [str(l) for l in range(1,129)]

df_copy = a.copy()




df_copy['prediction'] = df_copy.idxmax(axis=1)
df_copy.reset_index(level=0, inplace=True)
df_copy['label'] = df_copy['fns'].apply(lambda row: row.split('/')[0])
df_copy['max'] = df_copy[labels].max(axis= 1)

df_copy2 = df_copy[df_copy['max'] > 0.98]
correct = df_copy2[df_copy2['prediction'] == df_copy2['label']]
acc = 1- correct.shape[0] / df_copy2.shape[0]

print(acc)
print(df_copy2.shape[0])


df = pd.read_csv('bagging/b0/prediction_test_tta12.csv',index_col = 0)

df_copy = df.copy()
df_copy['prediction'] = df_copy.idxmax(axis=1)
df_copy.reset_index(level=0, inplace=True)
df_copy['label'] = df_copy['fns'].apply(lambda row: row.split('/')[0])
df_copy['max'] = df_copy[labels].max(axis= 1)

df_copy2 = df_copy[df_copy['max'] > 0.98]
print(df_copy2.shape[0])