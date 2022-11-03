
import os
import json
import numpy as np

import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#
# x = np.linspace(0.1,1,10)
# y1 = x
# y2 = x ** 2
# y3 = np.log(x)
#
#
# fig, ax = plt.subplots() # 创建图实例
# x = np.linspace(0,2,100) # 创建x的取值范围
# y1 = x
# ax.plot(x, y1, label='linear') # 作y1 = x 图，并标记此线名为linear
# y2 = x ** 2
# ax.plot(x, y2, label='quadratic') #作y2 = x^2 图，并标记此线名为quadratic
# y3 = x ** 3
# ax.plot(x, y3, label='cubic') # 作y3 = x^3 图，并标记此线名为cubic
# ax.set_xlabel('x label') #设置x轴名称 x label
# ax.set_ylabel('y label') #设置y轴名称 y label
# ax.set_title('Simple Plot') #设置图名为Simple Plot
# ax.legend() #自动检测要在图例中显示的元素，并且显示
#
# plt.show() #图形可视化
#
#

files = os.listdir(os.getcwd())
real_file = [x for x in files if x.split(".")[1] == "txt"]


return_dict = {}

for file in real_file:
    with open(file,"r") as data:
        line = data.readline()
        return_dict[data.name] = data.readline()



fig, ax = plt.subplots() # 创建图实例

for k in return_dict:
    l = return_dict[k].split(",")
    s = []
    for i in l:
        s.append(float(i))
    return_dict[k] = s

episodes_list = list(range(len(return_dict['AC_CartPole-v1_2022-11-02-20-27.txt'])))

for k in return_dict:
    ax.plot(episodes_list,return_dict[k],label = k)

ax.set_xlabel('x label') #设置x轴名称 x label
ax.set_ylabel('y label') #设置y轴名称 y label
ax.set_title('Simple Plot') #设置图名为Simple Plot
ax.legend() #自动检测要在图例中显示的元素，并且显示

plt.show() #图形可视化

