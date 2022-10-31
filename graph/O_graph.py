import os
import numpy as np

import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

x = np.linspace(0.1,1,10)
y1 = x
y2 = x ** 2
y3 = np.log(x)
y4 = x**x

plt.figure()  # 类似于先声明一张图片，这个figure后面所有的设置都是在这张图片上操作的
plt.plot(x, y1,label = "e")  # 制图
plt.plot(x, y2,label = "d")  # 制图
plt.plot(x, y3,label = "c")  # 制图
plt.plot(x, y4,label = "c")  # 制图

plt.show()  # 显示图片

