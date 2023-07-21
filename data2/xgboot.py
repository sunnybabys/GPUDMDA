import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb


f_d = pd.read_csv('./trans_data/data2/nond_feature.csv',header=None,index_col=None).values
f_m = pd.read_csv('./trans_data/data2/nonm_feature.csv',header=None,index_col=None).values

all_associations = pd.read_csv('./trans_data/data2' + '/S_pair.txt', sep=' ', names=['d', 'm', 'label'])

known_associations = all_associations.loc[all_associations['label'] == 1,:]   
unknown_associations = all_associations.loc[all_associations['label'] == 0,:]
random_positive = all_associations.loc[all_associations['label'] == 2,:]  
random_positive['label'] = 0
# p_sample_df = known_associations.drop(random_positive.index.to_list(), axis=0)  
n_sample_df = unknown_associations.append(random_positive)  
all_samples = known_associations.append(n_sample_df)



dataset = []
for i in range(all_samples.shape[0]):
    r = all_samples.iloc[i, 0]
    c = all_samples.iloc[i, 1]
    label = all_samples.iloc[i, 2]
    dataset.append(np.hstack((f_d[r], f_m[c], label)))

all_dataset = pd.DataFrame(dataset).values
# all_dataset = all_dataset.sample(frac=1).values
all_feature = all_dataset[:, :-1]
all_label = all_dataset[:, -1]


model = xgb.XGBClassifier()
model.fit(all_feature, all_label)
y_score0 = model.predict(np.array(all_feature))
y_score1= model.predict_proba(np.array(all_feature))
y_red = y_score1[:, 1]
min_probability = y_red[-(int(random_positive.shape[0])):].min()
prob =y_red.reshape(-1,1)
all_samples['probability']=y_red

a = known_associations.shape[0]
b = unknown_associations.shape[0]
c = random_positive.shape[0]

unknown_samples = all_samples.iloc[a:a + b, :]

balance_samples = unknown_samples[unknown_samples['probability']<min_probability]
print(balance_samples)

association_matrix = pd.read_csv('./trans_data/data2/index_association.csv', header=0, index_col=0)
index_disease = association_matrix.index.to_list()
index_microbe = association_matrix.columns.to_list()

reflection = np.zeros((balance_samples.shape[0],2))
reflection = pd.DataFrame(reflection,columns=['d','m'])

for i in range(balance_samples.shape[0]):
    reflection.iloc[i,0] = index_disease[int(balance_samples.iloc[i,0])]
    reflection.iloc[i,1] = index_microbe[int(balance_samples.iloc[i,1])]

for i in range(reflection.shape[0]):
    disease_name = reflection.iloc[i,0]
    microbe_name = reflection.iloc[i,1]
    association_matrix.loc[disease_name, microbe_name] = -1

association_matrix.to_csv('./trans_data/data2/select_sample_disease_microbe_association_matrix.csv')

print("Finished")




