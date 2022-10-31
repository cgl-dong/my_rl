"""
pytorch 实现 多维度Linear
"""
import os

import matplotlib.pyplot as plot
import numpy as np
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

loss = torch.nn.BCELoss(reduction='sum')  # 损失函数
optimizer = torch.optim.SGD(model.parameters(), 0.01)  # 优化器

x_axis = []
y_axis = []
# 训练

for epoch in range(10000):
    y_pred = model(x_data)

    l = loss(y_pred, y_data)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

    x_axis.append(epoch)
    y_axis.append(l.item())

plot.plot(x_axis, y_axis)
plot.xlabel("epoch")
plot.ylabel("loss")
plot.show()


test_data = torch.from_numpy(xy[0, :-1])
res = model(test_data)

print(res)
