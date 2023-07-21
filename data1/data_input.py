# encoding=utf-8
import numpy as np
import pandas as pd
class Neg_DataLoader:
    def __init__(self,filename):

        self.interaction = pd.read_csv(filename + '/select_sample_disease_microbe_association_matrix.csv',header=0, index_col=0)
        self.disease_feature = pd.read_csv(filename +'/nond_feature.csv',header=None,index_col=None).values
        self.microbe_feature = pd.read_csv(filename +'/nonm_feature.csv',header=None,index_col=None).values
        dataset = []
        for i in range(self.interaction.shape[0]):
            for j in range(self.interaction.shape[1]):
                dataset.append(np.hstack((self.disease_feature[i], self.microbe_feature[j],self.interaction.iloc[i, j])))
        self.dataset = pd.DataFrame(dataset).values
        self.pre_dataset = pd.DataFrame(dataset)


class Non_Neg_DataLoader:
    def __init__(self,filename):
        self.interaction = pd.read_csv(filename + '/index_association.csv', header=0,index_col=0)
        self.disease_feature = pd.read_csv(filename +'/nond_feature.csv',header=None,index_col=None).values
        self.microbe_feature = pd.read_csv(filename +'/nonm_feature.csv',header=None,index_col=None).values

        dataset = []
        for i in range(self.interaction.shape[0]):
            for j in range(self.interaction.shape[1]):
                    dataset.append(np.hstack((self.disease_feature[i], self.microbe_feature[j],self.interaction.iloc[i, j])))
        self.dataset= pd.DataFrame(dataset).values
        self.pre_dataset = pd.DataFrame(dataset)