import pandas as pd


df_stage1 = pd.read_csv('./demo_data/aged_poc/ag_opc_prediction1_3000.csv')
df_stage2 = pd.read_csv('./demo_data/aged_poc/ag_opc_prediction2_3000.csv')
df_stage3 = pd.read_csv('./demo_data/aged_poc/ag_opc_prediction3_3000.csv')
df_stage4 = pd.read_csv('./demo_data/aged_poc/ag_opc_prediction4_3000.csv')


print('df_stage1:{}'.format(df_stage1))
print('df_stage2:{}'.format(df_stage2))
print('df_stage3:{}'.format(df_stage3))
print('df_stage4:{}'.format(df_stage4))


## Find regulatory relationships that are constant
#
# merged_df = pd.merge(df_stage1, df_stage2, on=['Tf', 'Target'], how='inner')
#
# #
# merged_df = pd.merge(merged_df, df_stage3, on=['Tf', 'Target'], how='inner')
# merged_df = pd.merge(merged_df, df_stage4, on=['Tf', 'Target'], how='inner')
#
# print('merged_df:{}'.format(merged_df))
# merged_df.to_csv('./processed_data/Zma_NU_data/invariant_relationship.csv',index=False)


# To find out the specific regulatory relationship of stage 1
#specific_stage1 = df_stage1[~df_stage1.isin(pd.concat([df_stage2, df_stage3, df_stage4]).drop_duplicates()).all(1)]
df_concat = pd.concat([df_stage2, df_stage3, df_stage4],ignore_index=True)
merged = df_stage1.merge(df_concat, on=['TF', 'Target'], how='left', indicator=True)
filtered = merged[merged['_merge'] != 'both']
specific_stage1 = filtered[df_stage1.columns]  #
print("specific_stage1:{}".format(specific_stage1))
specific_stage1.to_csv('./demo_data/aged_opc/specific_stage1.csv',index=False)
#
# To find out the specific regulatory relationship of stage 2
df_concat = pd.concat([df_stage1, df_stage3, df_stage4],ignore_index=True)
merged = df_stage2.merge(df_concat, on=['TF', 'Target'], how='left', indicator=True)
filtered = merged[merged['_merge'] != 'both']
specific_stage2 = filtered[df_stage2.columns]  #
print("specific_stage2:{}".format(specific_stage2))
specific_stage2.to_csv('./demo_data/aged_opc/specific_stage2.csv',index=False)
# To find out the specific regulatory relationship of stage 3
df_concat = pd.concat([df_stage1, df_stage2, df_stage4],ignore_index=True)
merged = df_stage3.merge(df_concat, on=['TF', 'Target'], how='left', indicator=True)
filtered = merged[merged['_merge'] != 'both']
specific_stage3 = filtered[df_stage3.columns]
print("specific_stage3:{}".format(specific_stage3))
specific_stage3.to_csv('./demo_data/aged_opc/specific_stage3.csv',index=False)
# To find out the specific regulatory relationship of stage 4
df_concat = pd.concat([df_stage1, df_stage2, df_stage3],ignore_index=True)
merged = df_stage4.merge(df_concat, on=['TF', 'Target'], how='left', indicator=True)
filtered = merged[merged['_merge'] != 'both']
specific_stage4 = filtered[df_stage4.columns]
print("specific_stage4:{}".format(specific_stage4))
specific_stage4.to_csv('./demo_data/aged_opc/specific_stage4.csv',index=False)




