##
import pandas as pd
df_time1 = pd.read_csv("./demo_data/aged_opc/ag_opc_prediction1_3000.csv")
df_time2 = pd.read_csv("./demo_data/aged_opc/ag_opc_prediction2_3000.csv")
df_time3 = pd.read_csv("./demo_data/aged_opc/ag_opc_prediction3_3000.csv")
df_time4 = pd.read_csv("./demo_data/aged_opc/ag_opc_prediction4_3000.csv")

df_total = pd.concat([df_time1,df_time2,df_time3,df_time4],ignore_index=True)
new_df = df_total.drop_duplicates(subset=['TF', 'Target'], keep='first')
print(new_df)
df_edg_total = new_df.reset_index(drop=True)
df_edg_total = df_edg_total.iloc[:,:2]
print(df_edg_total)
##
df_target = pd.read_csv("../Target.csv")
dic_gene_index = df_target.set_index("Gene")["index"].to_dict()

df_score1 = pd.read_csv("./demo_data/aged_opc/ag_prediction1.csv")
df_score2 = pd.read_csv("./demo_data/aged_opc/ag_prediction2.csv")
df_score3 = pd.read_csv("./demo_data/aged_opc/ag_prediction3.csv")
df_score4 = pd.read_csv("./demo_data/aged_opc/ag_prediction4.csv")

dic1_edg_score = df_score1.set_index(['TF','Target'])['score']
dic2_edg_score = df_score2.set_index(['TF','Target'])['score']
dic3_edg_score = df_score3.set_index(['TF','Target'])['score']
dic4_edg_score = df_score4.set_index(['TF','Target'])['score']

edg_total = []
for index,row in df_edg_total.iterrows():
    edg = []
    tf = row['TF']
    target = row['Target']
    tf_idx = dic_gene_index[tf]
    target_idx = dic_gene_index[target]
    #dic1_edg_score[(tf_idx,target_idx)]
    edg.append(dic1_edg_score[(tf_idx,target_idx)])
    edg.append(dic2_edg_score[(tf_idx, target_idx)])
    edg.append(dic3_edg_score[(tf_idx, target_idx)])
    edg.append(dic4_edg_score[(tf_idx, target_idx)])
    edg_total.append(edg)


#print(edg_total)
print(len(edg_total))

##

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

#
vectors = edg_total

#
normalized_vectors = normalize(vectors)

#
min_num_clusters = 2
max_num_clusters = 50
best_num_clusters = None
best_silhouette_avg = float("inf")
best_cluster_labels = None

for num_clusters in range(min_num_clusters, max_num_clusters + 1):
    #
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    #
    kmeans.fit(normalized_vectors)

    #
    cluster_labels = kmeans.labels_

    #
    silhouette_avg = silhouette_score(normalized_vectors, cluster_labels)

    #
    if silhouette_avg < best_silhouette_avg:
        best_silhouette_avg = silhouette_avg
        best_num_clusters = num_clusters
        best_cluster_labels = cluster_labels

#
print(f"最佳聚类数量: {best_num_clusters}")
print(f"最小轮廓系数: {best_silhouette_avg}")

# 打印最佳聚类结果和每个类别的向量个数
print("最佳聚类结果:")
for i in range(best_num_clusters):
    cluster_samples = [vectors[j] for j in range(len(vectors)) if best_cluster_labels[j] == i]
    num_samples = len(cluster_samples)
    sample_indices = [j for j in range(len(vectors)) if best_cluster_labels[j] == i]
    print(f"聚类 {i+1}: {cluster_samples}")
    print("向量数量：{}".format(num_samples))
    print("样本索引：{}".format(sample_indices))


print(best_num_clusters)
##-----------------Save all regulation and all genes for each edge cluster-------------------------------------------
from collections import Counter


for i in range(best_num_clusters):
    cluster_samples = [vectors[j] for j in range(len(vectors)) if best_cluster_labels[j] == i]
    num_samples = len(cluster_samples)
    sample_indices = [j for j in range(len(vectors)) if best_cluster_labels[j] == i]
    df_tmp = df_edg_total.loc[sample_indices]
    df_tmp = df_tmp.reset_index(drop=True)
    path1 = "./demo_data/aged_opc/edg_cluster/" + "edg_rg" + str(i+1) + ".csv"
    path2 = "./demo_data/aged_opc/edg_cluster/" + "edg_gene_c" + str(i+1) + ".csv"
    df_tmp.to_csv(path1)
    l1 = list(df_tmp['TF'].values)
    l2 = list(df_tmp['Target'].values)
    l_all = l1+l2
    l3 = list(set(l_all))
    list_count = [Counter(l_all)[val] for val in l3]
    df = pd.DataFrame()
    df['Gene'] = l3
    df['Count'] = list_count
    df.to_csv(path2,index=False)



## ------------------------------Calculate the importance of all top3000 edges of each edge category at each point in time (1-average position ratio), not in top3000, directly calculate 0, draw a heat map--------------------------------------------------

#

# edg_time_total = []
# for i in range(best_num_clusters):
#     edg_time = []
#     cluster_samples = [vectors[j] for j in range(len(vectors)) if best_cluster_labels[j] == i]
#     num_samples = len(cluster_samples)
#     sample_indices = [j for j in range(len(vectors)) if best_cluster_labels[j] == i]
#
#     t1_avg = []
#     for num in range(num_samples):
#         #cluster_samples[num][0]
#         idx = sample_indices[num]
#         #
#
#         #
#         values_to_search = df_edg_total.iloc[idx, :2].values
#         #
#         indices = df_time1[(df_time1['TF'] == values_to_search[0]) & (df_time1['Target'] == values_to_search[1])].index
#         if len(indices)>0:
#             result = indices[0]
#             result = 1-(result+1)/3000
#         else:
#             result = 0
#         t1_avg.append(result)
#     edg_time.append(sum(t1_avg)/len(t1_avg))
#
#     t2_avg = []
#     for num in range(num_samples):
#         #cluster_samples[num][0]
#         idx = sample_indices[num]
#         #
#
#         #
#         values_to_search = df_edg_total.iloc[idx, :2].values
#         #
#         indices = df_time2[(df_time2['TF'] == values_to_search[0]) & (df_time2['Target'] == values_to_search[1])].index
#         if len(indices) > 0:
#             result = indices[0]
#             result = 1 - (result + 1) / 3000
#         else:
#             result = 0
#         t2_avg.append(result)
#     edg_time.append(sum(t2_avg) / len(t2_avg))
#
#     t3_avg = []
#     for num in range(num_samples):
#         #cluster_samples[num][0]
#         idx = sample_indices[num]
#         #
#
#         #
#         values_to_search = df_edg_total.iloc[idx, :2].values
#         #
#         indices = df_time3[(df_time3['TF'] == values_to_search[0]) & (df_time3['Target'] == values_to_search[1])].index
#         if len(indices) > 0:
#             result = indices[0]
#             result = 1 - (result + 1) / 3000
#         else:
#             result = 0
#         t3_avg.append(result)
#     edg_time.append(sum(t3_avg) / len(t3_avg))
#
#     t4_avg = []
#     for num in range(num_samples):
#         #cluster_samples[num][0]
#         idx = sample_indices[num]
#         #
#
#         #
#         values_to_search = df_edg_total.iloc[idx, :2].values
#         #
#         indices = df_time4[(df_time4['TF'] == values_to_search[0]) & (df_time4['Target'] == values_to_search[1])].index
#         if len(indices) > 0:
#             result = indices[0]
#             result = 1 - (result + 1) / 3000
#         else:
#             result = 0
#         t4_avg.append(result)
#     edg_time.append(sum(t4_avg) / len(t4_avg))
#
#     edg_time_total.append(edg_time)
#
# print(edg_time_total)
# print(len(edg_time_total))


##------------------------ Draw a heat map with rows as edge classes and columns as four timepoints----------------------------------
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# #
# data = edg_time_total
#
# #
# row_labels = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 'E20', 'E21', 'E22', 'E23', 'E24']
# col_labels = ['t1', 't2', 't3', 't4']
#
# #
# sns.heatmap(data, cmap='YlOrRd', annot=True, fmt='.2f', cbar=True,cbar_kws={'label':'Importance score'})
#
# #
# plt.yticks(np.arange(len(row_labels)) + 0.5, row_labels, va='center')
# plt.xticks(np.arange(len(col_labels)) + 0.5, col_labels, ha='center')
#
# #
# #plt.colorbar(label='Importance Score')
# plt.title("Edge clustering in aged OPCs")
# plt.tight_layout()
# #
# plt.savefig("./demo_data/aged_opc/edg_cluser_aged_opc.png",dpi=600)
# plt.show()
##



