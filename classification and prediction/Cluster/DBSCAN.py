########## 基于密度的聚类算法
###### 代表算法：DBSCAN算法、OPTICS算法、DENCLUE算法
"""
step1. 定义Esp邻域：以对象p为核心，以Eps为半径的d维超球体区域；
step2. 定义核心点与边界点：对于p∈D，给定MinPts，> MinPts，为核心点；< MinPts但落
在了某个核心点的Eps邻域中，为边界点；其他为噪声点；
step3. 直接密度可达，给定（Eps，MinPts），p在q的Eps邻域中，q -> q为直接密度可达；
step4. 密度可达/密度相连

DBSCAN算法描述：
    输入：Eps，MinPts和包含n个对象的数据库；
    输出：基于密度的聚类结果；
    方法：
        step1. 任意选取一个没有加簇标签的点p；
        step2. 得到所有从p关于Eps半径内和MinPts密度可达点；
        step3. 如果p == 核心点，形成一个新的簇，给簇内所有对象点加簇标签；
        step4. 如果p == 边界点，没有从p密度可达的点，DBSCAN将访问数据库中的下一个点；
        step5. 重复step1-4，直到数据库中所有点被处理；
        
DBSCAN算法改进：
    1. 不访问全部点 - 邻域寻找聚类，对足够高密度的区域划分为簇，并可以在带有"噪声"的
    空间数据库中发现任意形状的簇；
    2. 但是，DBSCAN对于用户设置参数敏感，Eps和MinPts的设置会影响聚类的效果。针对这一问题，
    OPTICS算法提出，通过引入核心距离和可达距离，使得聚类算法对于输入的参数不敏感；
    
"""
#### DBSCAN
from sklearn import datasets
import numpy as np
import random
import matplotlib.pyplot as plt

### construct model
def findNeighbor(j,x,eps):
    n = []
    for p in range(x.shape[0]): # 找到所有邻域内对象，按行搜索
        temp = np.sqrt(np.sum(np.square(x[j]-x[p]))) # 欧式距离
        if(temp <= eps): # 邻居点
            n.append(p)
    return n

def dbscan(x, eps, min_pts):
    k = -1
    neighborpts = [] # array, 某点邻域内对象
    ner_neighborpts = []
    fil = [] # 初始时访问列表为空
    gama = [x for x in range(len(x))]  # 初始所有节点为未访问
    cluster = [-1 for y in range(len(x))]
    while len(gama)>0:
        j = random.choice(gama)
        gama.remove(j)
        fil.append(j)
        neighborpts = findNeighbor(j, x, eps)
        if len(neighborpts) < min_pts:
            cluster[j] = -1 # 标记为噪声
        else:
            k = k+1
            cluster[j] = k
            for i in neighborpts:
                if i not in fil:
                    gama.remove(i) # 邻居节点中还没有被访问的节点从数据集中踢出
                    fil.append(i)
                    ner_neighborpts = findNeighbor(i, x, eps)
                    if len(ner_neighborpts) >= min_pts:
                        for a in ner_neighborpts:
                            if a not in neighborpts:
                                neighborpts.append(a)
                    if (cluster[i] == -1):
                        cluster[i] = k
    return cluster
        
### load data
X1, y1 = datasets.make_circles(n_samples=1000, factor=.6,noise=.05) # 生成环形数据
X2, y2 = datasets.make_blobs(n_samples = 300, n_features = 2, centers = [[1.2,1.2]], cluster_std = [[.1]],random_state = 9) # 生成聚类数据
X = np.concatenate((X1, X2))
eps = 0.08
min_Pts = 10
C = dbscan(X,eps,min_Pts)
plt.figure(figsize = (12, 9), dpi = 80)
plt.scatter(X[:,0],X[:,1],c = C)
plt.show()


























