from matplotlib import rcParams
config = {
    "font.family":'Arial',  # 设置字体类型
    "axes.unicode_minus": False #解决负号无法显示的问题
}
rcParams.update(config)


import pandas as pd
df_time1 = pd.read_csv("../cytoscape/aged_opc/ag_opc_prediction1_3000.csv")
df_time2 = pd.read_csv("../cytoscape/aged_opc/ag_opc_prediction2_3000.csv")
df_time3 = pd.read_csv("../cytoscape/aged_opc/ag_opc_prediction3_3000.csv")
df_time4 = pd.read_csv("../cytoscape/aged_opc/ag_opc_prediction4_3000.csv")

# df_time1 = pd.read_csv("../cytoscape/5XFAD_opc/5X_opc_prediction1_3000.csv")
# df_time2 = pd.read_csv("../cytoscape/5XFAD_opc/5X_opc_prediction2_3000.csv")
# df_time3 = pd.read_csv("../cytoscape/5XFAD_opc/5X_opc_prediction3_3000.csv")
# df_time4 = pd.read_csv("../cytoscape/5XFAD_opc/5X_opc_prediction4_3000.csv")


networks = [df_time1,df_time2,df_time3,df_time4]


df_TF = pd.read_csv("../TF.csv")
TF_list = list(df_TF['TF'].values)
print(TF_list)


##--------tf活性向量--------------
# 假设你有一个名为networks的列表，包含四个时间点的网络DataFrame
# 每个DataFrame都有'TF'和'Target'两列，表示转录因子和目标基因

#TF_list = ['NFIB', 'ZFP148', 'DLX2', 'BCL11B']
degree_vectors = {}

# 初始化度向量为0
for TF in TF_list:
    degree_vectors[TF] = [0] * len(networks)

# 遍历每个时间点的网络DataFrame
for i, df in enumerate(networks):
    # 统计每个TF在该时间点的Target数量
    for TF in TF_list:
        degree = len(df[df['TF'] == TF])
        degree_vectors[TF][i] = degree

# 打印每个TF的度向量
for TF in TF_list:
    print(f"{TF}: {degree_vectors[TF]}")


## TF聚类分组------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

from sklearn.metrics import silhouette_score

# 假设你已经有了TF_list和degree_vectors

# 将向量转换为矩阵
tf_matrix = np.array(list(degree_vectors.values()))

# 执行K-means聚类
num_clusters = 6  # 聚类数目
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(tf_matrix)

# 执行PCA降维
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(tf_matrix)

# 绘制聚类结果
#plt.figure(figsize=(12, 12))

plt.figure(figsize=(10, 10))

colors = ['r', 'g', 'b', 'c', 'm', 'y','#f08400','#82cdd0']  # 颜色列表
for i in range(num_clusters):
    plt.scatter(reduced_features[clusters == i, 0], reduced_features[clusters == i, 1], c=colors[i], label=f'Group {i+1}',s=30)

# 添加TF标签
for tf, (x, y) in zip(degree_vectors.keys(), reduced_features):
    plt.annotate(tf, (x, y), textcoords="offset points", xytext=(0, 4), ha='center',fontsize=10,alpha=0.5)


# 添加标题和标签
#plt.title("")
plt.xlabel("Component 1",fontsize=25)
plt.ylabel("Component 2",fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
print('1')

# 显示图形
plt.legend(fontsize=20,markerscale=2)
ax=plt.gca()
# 去除上方和右方的边框线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

#plt.savefig('../fig_new/TF_RA_group.pdf',format='pdf')
plt.show()

# 打印每个类别的TF
for i in range(num_clusters):
    tf_cluster = [tf for tf, cluster in zip(degree_vectors.keys(), clusters) if cluster == i]
    print(f"Cluster {i+1}: {tf_cluster}")


# 计算轮廓系数
silhouette_avg = silhouette_score(tf_matrix, clusters)

# 打印轮廓系数
print(f"聚类后的轮廓系数: {silhouette_avg}")