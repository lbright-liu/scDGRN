from matplotlib import rcParams
config = {
    "font.family":'Arial',  # 
    "axes.unicode_minus": False #
}
rcParams.update(config)


import pandas as pd
df_time1 = pd.read_csv("./demo_data/aged_opc/ag_opc_prediction1_3000.csv")
df_time2 = pd.read_csv("./demo_data/aged_opc/ag_opc_prediction2_3000.csv")
df_time3 = pd.read_csv("./demo_data/aged_opc/ag_opc_prediction3_3000.csv")
df_time4 = pd.read_csv("./demo_data/aged_opc/ag_opc_prediction4_3000.csv")



networks = [df_time1,df_time2,df_time3,df_time4]


df_TF = pd.read_csv("./demo_data/aged_opc/TF.csv")
TF_list = list(df_TF['TF'].values)
print(TF_list)


##---------tf activity vector--------------
# Suppose you have a list named "networks", which contains network Dataframes at four time points
# Each DataFrame has two columns, 'TF' and 'Target', representing the transcription factor and the target gene

#TF_list = ['NFIB', 'ZFP148', 'DLX2', 'BCL11B']
degree_vectors = {}

# 
for TF in TF_list:
    degree_vectors[TF] = [0] * len(networks)

# 
for i, df in enumerate(networks):
    # 
    for TF in TF_list:
        degree = len(df[df['TF'] == TF])
        degree_vectors[TF][i] = degree

# 
for TF in TF_list:
    print(f"{TF}: {degree_vectors[TF]}")


## TF clustering grouping------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

from sklearn.metrics import silhouette_score

# 

# 
tf_matrix = np.array(list(degree_vectors.values()))

# K-means clustering
num_clusters = 6  # 
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(tf_matrix)

# PCA
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(tf_matrix)

# 
#plt.figure(figsize=(12, 12))

plt.figure(figsize=(10, 10))

colors = ['r', 'g', 'b', 'c', 'm', 'y','#f08400','#82cdd0']  # 
for i in range(num_clusters):
    plt.scatter(reduced_features[clusters == i, 0], reduced_features[clusters == i, 1], c=colors[i], label=f'Group {i+1}',s=30)

# 
for tf, (x, y) in zip(degree_vectors.keys(), reduced_features):
    plt.annotate(tf, (x, y), textcoords="offset points", xytext=(0, 4), ha='center',fontsize=10,alpha=0.5)


# 
#plt.title("")
plt.xlabel("Component 1",fontsize=25)
plt.ylabel("Component 2",fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
print('1')

# 
plt.legend(fontsize=20,markerscale=2)
ax=plt.gca()
# 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

#plt.savefig('../fig_new/TF_RA_group.pdf',format='pdf')
plt.show()

# 
for i in range(num_clusters):
    tf_cluster = [tf for tf, cluster in zip(degree_vectors.keys(), clusters) if cluster == i]
    print(f"Cluster {i+1}: {tf_cluster}")


# 
silhouette_avg = silhouette_score(tf_matrix, clusters)

# 

print(f"Silhouette Coefficient: {silhouette_avg}")
