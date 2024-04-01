import matplotlib.pyplot as plt
import numpy as np

# 柱状图数据
x = np.arange(1, 9)
heights = np.arange(8, 0, -1)  # 从5到1递减的高度值

# # 颜色渐变
# colors = plt.cm.Reds(np.linspace(0.2, 1, len(x)))  # 使用红色调色板生成渐变颜色
# # 绘制柱状图
# plt.bar(x, heights, color=colors)


# 创建颜色映射对象
cmap = plt.cm.get_cmap('Reds')
# 绘制柱状图，并使用颜色映射函数设置颜色
plt.bar(x, heights, color=cmap(heights/8))



# 设置标题和坐标轴标签
#plt.title('Decreasing Bar Heights and Colors')
#plt.xlabel('X-axis')
plt.ylabel('Perturbation Score',fontsize=15)
plt.axhline(y=5, color='black', linestyle='--',linewidth = 2)
# 禁用横纵坐标刻度
gene_labels = ['Gene 1', 'Gene 2', 'Gene 3', 'Gene 4', 'Gene 5','Gene 6','Gene 7','Gene 8']
plt.xticks(x, gene_labels,fontsize=12)
plt.yticks([])
# 显示图形
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("./picture/p-score.png",dpi=600)
plt.show()