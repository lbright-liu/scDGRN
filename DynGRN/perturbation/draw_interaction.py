## 时序扰动实验（相互作用推断）
import pandas as pd
import matplotlib.pyplot as plt

##-----------------单个时间点的性能-----------------------------

# 模拟数据
x1 = [1, 2, 3, 4, 5,6]
y1 = [0.699, 0.710, 0.705, 0.646, 0.685,0.698]
x2 = [1, 2, 3, 4, 5,6,7,8,9]
y2 = [0.730,0.712,0.718,0.672,0.714,0.739,0.764,0.635,0.692]
# 创建图像对象并设置大小
fig = plt.figure(figsize=(4, 3))

# 绘制折线图
plt.plot(x1, y1, marker='o', markersize=6, markerfacecolor='lightblue', linestyle='-',label='hesc2')
plt.plot(x2, y2, marker='s', markersize=6, markerfacecolor='orange', linestyle='-',label='mesc1')
# 绘制 y=0.5 的虚线
#plt.axhline(y=0.5, color='black', linestyle='--',linewidth = 2)

# 设置图表标题和坐标轴标签
#plt.title('Line Chart with Highlighted Data Points')
plt.xlabel('time point')
plt.ylabel('AUROC')

# 设置纵轴刻度范围
plt.ylim(0.4, 0.9)
# 显示图例
plt.legend(loc='best')

# 显示图形
plt.grid(True,alpha=0.5)
plt.tight_layout()
ax = plt.gca()
# 隐藏上方和右方的边框线
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#plt.savefig(pdf)
plt.show()








##-----------------时间点依次增加的性能（中间画一条虚线代表一个时间点，数据全部整合的性能）-----------------------------





##----------------等间距采样和不等间距采样的性能比较--------------------------------------



##-----------------时序扰动的性能-----------------------------



