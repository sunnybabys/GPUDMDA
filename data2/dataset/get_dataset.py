from scipy.io import loadmat
import pandas as pd
import numpy as np
# data = loadmat( './HMDAD/interaction.mat' )
# association_matrix = pd.DataFrame(data['interaction'])
#
#
# disease = pd.read_excel('./HMDAD/diseases.xlsx',header=None, index_col=0)
# microbe = pd.read_excel('./HMDAD/microbes.xlsx',header=None, index_col=0)
# association_matrix.index = disease.iloc[:,0].tolist()
# association_matrix.columns = microbe.iloc[:,0].tolist()
# association_matrix.to_csv('../trans_data/data1/index_association.csv')

data = loadmat( './Disbiome/interaction.mat' )
association_matrix = pd.DataFrame(data['interaction1'])


disease = pd.read_excel('./Disbiome/diseases.xlsx',header=0, index_col=0)
microbe = pd.read_excel('./Disbiome/microbes.xlsx',header=0, index_col=0)
association_matrix.index = disease.iloc[:,0].tolist()
association_matrix.columns = microbe.iloc[:,0].tolist()
association_matrix.to_csv('../trans_data/data2/index_association.csv')

