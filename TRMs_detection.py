import pandas as pd

df2 = pd.read_csv("./demo_data/hesc2/regulatory_t2.csv")
df3 = pd.read_csv("./demo_data/hesc2/regulatory_t3.csv")
df4 = pd.read_csv("./demo_data/hesc2/regulatory_t4.csv")
df5 = pd.read_csv("./demo_data/hesc2/regulatory_t5.csv")
df6 = pd.read_csv("./demo_data/hesc2/regulatory_t6.csv")


merged_df = df2
dfs = [df3, df4, df5,df6]

for df in dfs:
    merged_df = pd.merge(merged_df, df, on=['TF', 'Target'], how='inner')

#
print(merged_df)

df_gene = pd.read_csv("Target.csv")
dic_index_gene = df_gene.set_index("index")["Gene"].to_dict()
dic_gene_index = df_gene.set_index("Gene")["index"].to_dict()
#
tf_counts = merged_df['TF'].value_counts()

#
top_tf = tf_counts.nlargest(10000)  #

#
print(top_tf)


#
tf_list = []
#filtered_df = merged_df[merged_df['TF'].isin(top_tf.index)]
df_tf = pd.read_csv("TF_large.csv")
for i in df_tf["TF"].values:
    j = i.lower()
    print(j)
    tf_list.append(dic_gene_index[j])

filtered_df = merged_df[merged_df["TF"].isin(tf_list)]
print(filtered_df)


tf = []
target = []
for i in filtered_df["TF"].values:
    tf.append(dic_index_gene[i].upper())

for i in filtered_df["Target"].values:
    target.append(dic_index_gene[i].upper())

co_result = pd.DataFrame({"TF":tf,"Target":target})
print(co_result)

#co_result.to_csv("co_network_tf.csv",index=False)

#
#grouped_targets = filtered_df.groupby('TF')['Target']
#print(grouped_targets)
# #
# for tf, targets in grouped_targets.items():
#     print("TF:", tf)
#     print("Targets:", targets)
#     print()