import os
import numpy as np

import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

x = np.linspace(0.1,1,10)




fig, ax = plt.subplots() # 创建图实例
# x = np.linspace(0,2,100) # 创建x的取值范围
y1 = x
y2 = x ** 2
ax.fill_between(x, y1, y2, alpha=0.2)



ax.set_xlabel('x label') #设置x轴名称 x label
ax.set_ylabel('y label') #设置y轴名称 y label
ax.set_title('Simple Plot') #设置图名为Simple Plot
ax.legend() #自动检测要在图例中显示的元素，并且显示

plt.show() #图形可视化


