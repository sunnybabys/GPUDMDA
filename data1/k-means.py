from numpy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score  # 聚类评判指标包（轮廓系数）


def standard_Data(start_Mat):
    start_Mat = np.asarray(start_Mat).astype(float)
    std = StandardScaler()
    std_Mat = std.fit_transform(start_Mat)
    return std_Mat


def pca(start_Mat, dimension):
    model = PCA(n_components=dimension)
    pca_Data = model.fit_transform(start_Mat)
    return pca_Data




def kmeans(pca_Data, k):
    data = np.asarray(pca_Data).astype(float)
    model = KMeans(n_clusters=k)
    data = np.asarray(data).astype(float)
    model.fit(data)
    centers = model.cluster_centers_
    result = model.predict(data)
    return centers, result


def plotBestFit(pca_Data, centers, result, k):
    mark = ['or', 'ob', 'og', 'ok', 'oy', '^r', '^b', '^g', '^k', '^y']
    if k > len(mark):
        print("分类数大于可以画出图的种类数")
        return 1
    for i, d in enumerate(pca_Data):
        plt.plot(d[0], d[1], mark[result[i]])
    mark = ['*r', '*b', '*g', '*k', '*y', 'dr', 'db', 'dg', 'dk', 'dy']
    if k > len(mark):
        print("分类的中心点数大于可以画出图的种类数")
        return 1
    for i, center in enumerate(centers):
        plt.plot(center[0], center[1], mark[3], markersize=15)
    plt.savefig("kmeans.pdf")
    plt.show()





def calEuclideanDistance(vec1,vec2):
    dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return dist



f_d = pd.read_csv('./trans_data/data1/nond_feature.csv',header=None,index_col=None).values
f_m = pd.read_csv('./trans_data/data1/nonm_feature.csv',header=None,index_col=None).values
all_associations = pd.read_csv('./trans_data/data1' + '/pair.txt', sep=' ', names=['d', 'm', 'label'])

interaction_marix = pd.read_csv("./trans_data/data1/index_association.csv", header=0, index_col=0)

known_associations = all_associations.loc[all_associations['label'] == 1]
dataset = []
for i in range(known_associations.shape[0]):
    r = known_associations.iloc[i, 0]
    c = known_associations.iloc[i, 1]
    label = known_associations.iloc[i, 2]
    dataset.append(np.hstack((f_d[r], f_m[c], label)))

all_dataset = pd.DataFrame(dataset).values
all_feature = all_dataset[:, :-1]
std_Mat = standard_Data(all_feature)
dimension = all_feature.shape[1]
pca_Data = pca(std_Mat, dimension)
cluster_k = [1]
for k in cluster_k:
    centers, result = kmeans(pca_Data, k)
    plotBestFit(pca_Data, centers, result, k)

distance = []

for j in range(known_associations.shape[0]):
    r = known_associations.iloc[j, 0]
    c = known_associations.iloc[j, 1]
    dist = calEuclideanDistance(centers.reshape(-1),np.hstack((f_d[r], f_m[c])))
    distance.append([r,c,dist])

all_distance = pd.DataFrame(distance).values

sort_dis = all_distance[np.argsort(all_distance[:,2])]
print(sort_dis)
number = int(0.15*known_associations.shape[0])

for n in range(number):
    row = int(sort_dis[n,0])
    col = int(sort_dis[n,1])
    interaction_marix.iloc[row,col] = 2

interaction_marix.to_csv('./trans_data/data1/S_interaction.csv',header=None,index=False)



print("Finished")