import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# # 产生多个随机种子
# seeds = [0, 1, 2, 3, 4]
#
# # 产生多个实验结果
# results = []
# for seed in seeds:
#     # 使用不同的随机种子运行实验
#     np.random.seed(seed)
#     # 进行实验，得到实验结果
#     result = np.random.rand(10)  # 示例，10个随机数
#     results.append(result)
#
# # 计算指标的平均值和标准差
# means = np.mean(results, axis=0)
# stds = np.std(results, axis=0)
#
# # 绘制平均曲线和误差线
# plt.errorbar(np.arange(10), means, yerr=stds, fmt='-o')
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.title('Average Reward over Multiple Seeds')
# plt.show()
import matplotlib.pyplot as plt
import numpy as np

# 生成样本数据
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(2*x)
y4 = np.cos(2*x)

# 计算均值和标准差
y_mean = np.mean([y1, y2, y3, y4], axis=0)
y_std = np.std([y1, y2, y3, y4], axis=0)

# 绘制曲线和阴影区间
plt.plot(x, y_mean, color='blue')
plt.fill_between(x, y_mean - y_std, y_mean + y_std, color='lightblue')
plt.show()
