import numpy as np
import matplotlib.pyplot as plt



##=------------------------------------single and multi--------------------------------------

# #----h2-----
# # AUROC
# DynGRN_S = [[0.711, 0.711, 0.715, 0.704, 0.706],[0.697, 0.708, 0.703, 0.688, 0.695],[0.708, 0.684, 0.711, 0.665, 0.658],[0.688, 0.697, 0.685, 0.654, 0.669],[0.685, 0.69, 0.701, 0.671, 0.706],[0.672, 0.692, 0.689, 0.699, 0.713]]
# DynGRN = [[0.711, 0.711, 0.715, 0.704, 0.706],[0.689, 0.696, 0.691, 0.698, 0.707],[0.728, 0.731, 0.725, 0.716, 0.729],[0.721, 0.718, 0.707, 0.712, 0.7],[0.685, 0.693, 0.686, 0.691, 0.699],[0.731, 0.737, 0.737, 0.725, 0.738]]
# # AUPRC
# #DynGRN_S = [[0.699, 0.7, 0.702, 0.696, 0.701],[0.689, 0.698, 0.693, 0.673, 0.684],[0.698, 0.677, 0.705, 0.643, 0.629],[0.675, 0.688, 0.682, 0.649, 0.661],[0.675, 0.679, 0.69, 0.663, 0.702],[0.663, 0.691, 0.681, 0.687, 0.706]]
# #DynGRN = [[0.699, 0.7, 0.702, 0.696, 0.701],[0.686, 0.692, 0.691, 0.695, 0.703],[0.722, 0.721, 0.714, 0.711, 0.721],[0.717, 0.716, 0.705, 0.709, 0.701],[0.683, 0.69, 0.684, 0.693, 0.7],[0.725, 0.729, 0.729, 0.723, 0.733]]
#
# x1 = np.arange(1,13,2)
# x2 = x1+0.5
# x3 = x2+0.5
# x4 = x3+0.5
# x5 = x4+0.5
# x6 = x5+0.5
#
# box1 = plt.boxplot(DynGRN_S,positions=x1,patch_artist=True,showmeans=True,
#             boxprops={"facecolor": "#B8DDBC",
#                       "edgecolor": "black",
#                       "linewidth": 0.5,
#                       },
#             medianprops={"color": "k", "linewidth": 0.5},
#             meanprops={'marker':'+',
#                        'markerfacecolor':'k',
#                        'markeredgecolor':'k',
#                        'markersize':5})
# box2 = plt.boxplot(DynGRN,positions=x2,patch_artist=True,showmeans=True,
#             boxprops={"facecolor": "#F0A780",
#                       "edgecolor": "black",
#                       "linewidth": 0.5,
#                       },
#             medianprops={"color": "k", "linewidth": 0.5},
#             meanprops={'marker':'+',
#                        'markerfacecolor':'k',
#                        'markeredgecolor':'k',
#                        'markersize':5})
#
# ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#
# data_label = ['t1','t2','t3','t4','t5','t6']
# plt.xticks([1.325,3.325,5.325,7.325,9.325,11.325],data_label,fontsize=15)
# plt.yticks(fontsize=13)
# plt.ylim(0.6,0.8)
# plt.title("hesc2",fontsize = 20)
# plt.ylabel('AUROC',fontsize=15)
# #plt.grid(axis='y',ls='--',alpha=0.8)
#
# # 给箱体添加图例，每类箱线图中取第一个颜色块用于代表图例
# plt.legend(handles=[box1['boxes'][0],box2['boxes'][0]],labels=['Single time point','Incremental time point'],fontsize=15)
#
# #plt.legend(box_plot['boxes'], df.columns, loc='center left', bbox_to_anchor=(1, 0.5))
#
# plt.tight_layout()
# plt.grid(alpha=0.3)
# plt.axhline(y = 0.715,color = 'darkred',linestyle='--')
# plt.savefig('./picture/single-multi_h2_AUROC.pdf',format = 'pdf')
# plt.show()


##----------------------------m1-------------------------
# #AUROC
# DynGRN_S = [[0.732, 0.703, 0.753, 0.742, 0.735],[0.686, 0.75, 0.685, 0.691, 0.702],[0.716, 0.758, 0.707, 0.692, 0.75],[0.676, 0.69, 0.69, 0.691, 0.688],
#             [0.753, 0.724, 0.754, 0.761, 0.713],[0.722, 0.73, 0.732, 0.738, 0.742],[0.767, 0.757, 0.767, 0.747, 0.753],[0.597, 0.657, 0.574, 0.639, 0.68],[0.708, 0.713, 0.72, 0.735, 0.727]]
# DynGRN = [[0.732, 0.703, 0.753, 0.742, 0.735],[0.737, 0.775, 0.741, 0.782, 0.779],[0.729, 0.765, 0.737, 0.736, 0.746],[0.692, 0.719, 0.719, 0.715, 0.725],
#           [0.753, 0.768, 0.735, 0.744, 0.752],[0.767, 0.772, 0.771, 0.753, 0.749],[0.71, 0.726, 0.701, 0.729, 0.715],[0.748, 0.744, 0.776, 0.763, 0.776],[0.776, 0.77, 0.78, 0.773, 0.771]]
#
# #AUPRC
# #DynGRN_S = [[0.721, 0.709, 0.741, 0.736, 0.743],[0.685, 0.737, 0.697, 0.703, 0.707],[0.716, 0.739, 0.683, 0.694, 0.735],[0.694, 0.689, 0.698, 0.695, 0.697],
# #            [0.722, 0.706, 0.733, 0.736, 0.688],[0.717, 0.728, 0.73, 0.738, 0.751],[0.759, 0.744, 0.751, 0.733, 0.744],[0.632, 0.684, 0.623, 0.662, 0.697],[0.722, 0.728, 0.729, 0.742, 0.737]]
# #DynGRN = [[0.721, 0.709, 0.741, 0.736, 0.743],[0.747, 0.763, 0.744, 0.776, 0.77],[0.723, 0.755, 0.736, 0.733, 0.748],[0.711, 0.737, 0.743, 0.733, 0.741],
# #          [0.744, 0.762, 0.738, 0.742, 0.755],[0.762, 0.766, 0.762, 0.765, 0.753],[0.682, 0.707, 0.688, 0.704, 0.693],[0.739, 0.744, 0.769, 0.761, 0.767],[0.764, 0.761, 0.766, 0.766, 0.763]]
#
#
# x1 = np.arange(1,19,2)
# x2 = x1+0.5
# x3 = x2+0.5
# x4 = x3+0.5
# x5 = x4+0.5
# x6 = x5+0.5
# x7 = x6+0.5
# x8 = x7+0.5
# x9 = x8+0.5
#
# box1 = plt.boxplot(DynGRN_S,positions=x1,patch_artist=True,showmeans=True,
#             boxprops={"facecolor": "#8FC9E2",
#                       "edgecolor": "black",
#                       "linewidth": 0.5,
#                       },
#             medianprops={"color": "k", "linewidth": 0.5},
#             meanprops={'marker':'+',
#                        'markerfacecolor':'k',
#                        'markeredgecolor':'k',
#                        'markersize':5})
# box2 = plt.boxplot(DynGRN,positions=x2,patch_artist=True,showmeans=True,
#             boxprops={"facecolor": "#ECC97F",
#                       "edgecolor": "black",
#                       "linewidth": 0.5,
#                       },
#             medianprops={"color": "k", "linewidth": 0.5},
#             meanprops={'marker':'+',
#                        'markerfacecolor':'k',
#                        'markeredgecolor':'k',
#                        'markersize':5})
#
# ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#
# data_label = ['t1','t2','t3','t4','t5','t6','t7','t8','t9']
# plt.xticks([1.325,3.325,5.325,7.325,9.325,11.325,13.325,15.325,17.325],data_label,fontsize=15)
# plt.yticks(fontsize=13)
# plt.ylim(0.5,0.9)
# plt.title("mesc1",fontsize = 20)
# plt.ylabel('AUROC',fontsize=15)
# #plt.grid(axis='y',ls='--',alpha=0.8)
#
# # 给箱体添加图例，每类箱线图中取第一个颜色块用于代表图例
# plt.legend(handles=[box1['boxes'][0],box2['boxes'][0]],labels=['Single time point','Incremental time point'],fontsize=15)
#
# #plt.legend(box_plot['boxes'], df.columns, loc='center left', bbox_to_anchor=(1, 0.5))
#
# plt.tight_layout()
# plt.grid(alpha=0.3)
# plt.axhline(y = 0.723,color = 'darkred',linestyle='--')
# plt.savefig('./picture/single-multi_m1_AUROC.pdf',format = 'pdf')
# plt.show()





##=--------------------------------Equidistant and unequal------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

# 示例数据
# AUROC
# data_es = [[0.703, 0.702, 0.696, 0.708, 0.707], [0.709, 0.699, 0.712, 0.703, 0.705], [0.731, 0.737, 0.737, 0.725, 0.738]]
# data_non_es = [[0.72, 0.723, 0.715, 0.715, 0.72], [0.696, 0.693, 0.706, 0.708, 0.72], [0.731, 0.737, 0.737, 0.725, 0.738]]

# AUPRC
# data_es = [[0.698, 0.696, 0.694, 0.706, 0.704],[0.707, 0.696, 0.709, 0.699, 0.704],[0.725, 0.729, 0.729, 0.723, 0.733]]
# data_non_es = [[0.715, 0.716, 0.703, 0.705, 0.717],[0.7, 0.695, 0.707, 0.708, 0.716],[0.725, 0.729, 0.729, 0.723, 0.733]]


##--------m1--------------------
# AUROC
# data_es = [[0.781, 0.795, 0.778, 0.778, 0.792],[0.721, 0.726, 0.71, 0.708, 0.721],[0.778, 0.769, 0.779, 0.775, 0.769]]
# data_non_es = [[0.732, 0.737, 0.69, 0.685, 0.748],[0.751, 0.752, 0.746, 0.764, 0.742],[0.694, 0.723, 0.696, 0.706, 0.706],[0.778, 0.769, 0.779, 0.775, 0.769]]

# AUPRC
data_es = [[0.768, 0.78, 0.769, 0.774, 0.784],[0.721, 0.731, 0.71, 0.712, 0.727],[0.765, 0.76, 0.766, 0.767, 0.762]]
data_non_es = [[0.744, 0.75, 0.712, 0.712, 0.752],[0.747, 0.753, 0.752, 0.764, 0.752],[0.707, 0.738, 0.718, 0.723, 0.722],[0.765, 0.76, 0.766, 0.767, 0.762]]



plt.figure(figsize=(6, 4))  # 设置宽度为8英寸，高度为6英寸

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置灰度背景
plt.axvspan(0.75, 3.25, facecolor='lightgray', alpha=0.2)

# 绘制第一组箱型图

plt.boxplot(data_es, positions=[1, 2, 3], widths=0.6, patch_artist=True,showmeans=True, boxprops=dict(facecolor='#51B1B7'),medianprops={"color": "k", "linewidth": 0.5},meanprops={'marker':'+','markerfacecolor':'k','markeredgecolor':'k','markersize':10})


# 提取第一组箱子的均值
mean_es = [np.mean(data) for data in data_es]

# 绘制第一组箱子的均值连线
plt.plot([1, 2, 3], mean_es, marker='o', linestyle='-', color='#4DB748',linewidth=2)

# 设置灰度背景
plt.axvspan(3.75, 7.25, facecolor='lightgray', alpha=0.2)

# 绘制第二组箱型图
plt.boxplot(data_non_es, positions=[4, 5, 6,7], widths=0.6, patch_artist=True,showmeans=True, boxprops=dict(facecolor='#E07B54'),medianprops={"color": "k", "linewidth": 0.5},meanprops={'marker':'+','markerfacecolor':'k','markeredgecolor':'k','markersize':10})

# 提取第二组箱子的均值
mean_non_es = [np.mean(data) for data in data_non_es]

# 绘制第二组箱子的均值连线
plt.plot([4, 5, 6,7], mean_non_es, marker='o', linestyle='-', color='#4DB748',linewidth=2)

#plt.ylim(0.63,0.8)
plt.ylim(0.5,0.9)
# 设置横轴刻度标签
plt.xticks([2, 5.5], ['E-S', 'Non-E-S'],fontsize=15)
plt.yticks(fontsize=12)
# 添加标题和标签
plt.title('mesc1',fontsize=20)
#plt.xlabel('Groups')
plt.ylabel('AUPRC',fontsize=15)
plt.tight_layout()

#plt.tight_layout()
# 显示图形
plt.savefig('./picture/point_sample_m1_AUPRC.pdf',format = 'pdf')
plt.show()



##=--------------------------------时序扰动------------------------------------------
#
# #---h2-----
# # AUROC
# #DynGRN = [[0.731, 0.737, 0.737, 0.725, 0.738],[0.723, 0.726, 0.732, 0.715, 0.733],[0.73, 0.734, 0.725, 0.723, 0.734],[0.725, 0.738, 0.727, 0.721, 0.736],[0.727, 0.729, 0.731, 0.724, 0.741],[0.724, 0.731, 0.735, 0.715, 0.738],[0.72, 0.722, 0.727, 0.723, 0.731],[0.73, 0.729, 0.731, 0.714, 0.732]]
# #DynGRN_n = [[0.679, 0.7, 0.699, 0.663, 0.685],[0.692, 0.683, 0.731, 0.703, 0.692],[0.696, 0.678, 0.71, 0.696, 0.712],[0.716, 0.686, 0.722, 0.707, 0.714],[0.693, 0.691, 0.674, 0.658, 0.698],[0.723, 0.688, 0.687, 0.68, 0.688]]
# # AUPRC
# DynGRN = [[0.725, 0.729, 0.729, 0.723, 0.733],[0.72, 0.723, 0.725, 0.713, 0.73],[0.723, 0.728, 0.721, 0.721, 0.729],[0.719, 0.73, 0.723, 0.719, 0.732],[0.722, 0.724, 0.725, 0.722, 0.737],[0.72, 0.727, 0.728, 0.715, 0.733],[0.715, 0.718, 0.721, 0.721, 0.729],[0.721, 0.723, 0.726, 0.712, 0.728]]
#
# #-----m1-----
# #DynGRN = [[0.778, 0.769, 0.779, 0.775, 0.769],[0.774, 0.769, 0.78, 0.773, 0.771],[0.779, 0.766, 0.781, 0.78, 0.761],[0.773, 0.765, 0.771, 0.78, 0.767],[0.78, 0.768, 0.782, 0.778, 0.763],[0.771, 0.768, 0.77, 0.779, 0.773],[0.78, 0.765, 0.776, 0.781, 0.771],[0.77, 0.765, 0.778, 0.774, 0.774]]
# #DynGRN = [[0.765, 0.76, 0.766, 0.767, 0.762],[0.762, 0.761, 0.767, 0.766, 0.762],[0.768, 0.758, 0.771, 0.77, 0.75],[0.763, 0.758, 0.762, 0.769, 0.761],[0.767, 0.761, 0.769, 0.77, 0.756],[0.76, 0.76, 0.76, 0.77, 0.764],[0.766, 0.754, 0.766, 0.772, 0.765],[0.761, 0.757, 0.766, 0.766, 0.758]]
#
#
# x1 = np.arange(1,9,1)
# # x2 = x1
# # x3 = x2
# # x4 = x3
# # x5 = x4
# # x6 = x5
# # x7=x6
# # x8=x7
# plt.figure(figsize=(6, 4))
# box1 = plt.boxplot(DynGRN,positions=x1,patch_artist=True,showmeans=True,
#             boxprops={"facecolor": "#E79397",#"#E07B54", #"#E79397",
#                       "edgecolor": "black",
#                       "linewidth": 0.5,
#                       },
#             medianprops={"color": "k", "linewidth": 0.5},
#             meanprops={'marker':'+',
#                        'markerfacecolor':'k',
#                        'markeredgecolor':'k',
#                        'markersize':5})
# # box2 = plt.boxplot(DynGRN,positions=x2,patch_artist=True,showmeans=True,
# #             boxprops={"facecolor": "#F0A780",
# #                       "edgecolor": "black",
# #                       "linewidth": 0.5,
# #                       },
# #             medianprops={"color": "k", "linewidth": 0.5},
# #             meanprops={'marker':'+',
# #                        'markerfacecolor':'k',
# #                        'markeredgecolor':'k',
# #                        'markersize':5})
#
# #plt.figure(figsize=(8,6))
#
# ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#
# data_label = ['O','P1','P2','P3','P4','P5','P6','P7']
# plt.xticks([1,2,3,4,5,6,7,8],data_label,fontsize=15)
# plt.yticks(fontsize=15)
# plt.ylim(0.65,0.8)
# #plt.ylim(0.7,0.85)
# #plt.xlabel('mesc1',fontsize=15)
# plt.title('hesc2',fontsize=20)
# plt.ylabel('AUPRC',fontsize=15)
# #plt.grid(axis='y',ls='--',alpha=0.8)
#
# # 给箱体添加图例，每类箱线图中取第一个颜色块用于代表图例
# #plt.legend(fontsize=13)
#
# #plt.legend(box_plot['boxes'], df.columns, loc='center left', bbox_to_anchor=(1, 0.5))
#
# plt.tight_layout()
#
# # 添加折线图
# means = [box1['means'][i].get_ydata()[0] for i in range(len(box1['means']))]
# plt.plot(x1, means, color='#4DB748', linestyle='-', marker='o', markersize=5,linewidth=2)
#
# plt.grid(alpha=0.3)
# plt.savefig('./picture/perturbation_h2_AUPRC.pdf',format='pdf')
# plt.show()
