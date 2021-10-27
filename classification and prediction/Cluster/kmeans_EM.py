###### K_means EM_Cluster
import matplotlib.pyplot as plt
import seaborn as sns; sns.set() # for plot styling
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

## make_blobs() 产生聚类数据
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50)

## k_mean
kmeans = KMeans(n_clusters=4,init='random')
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

## plot result
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_ #获得中心点的坐标
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


#### K-means的聚类边界总是线性的
from sklearn.datasets import make_moons
X, y = make_moons(200, noise=.05, random_state=0)
labels = KMeans(2, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels,s=50, cmap='viridis')

# 使用谱聚类处理，频谱聚类灵活，允许聚类非图数据，不会聚类形式做假设；
from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=2,
                affinity='nearest_neighbors',
                assign_labels='kmeans')
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels,s=50, cmap='viridis')
















