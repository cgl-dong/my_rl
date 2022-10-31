"""
pytorch 随机梯度 y=wx
"""
import matplotlib.pyplot as plot
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True
lr = 0.01


def forward(x):
    return x * w


# 损失函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


x_axis = []
y_axis = []
print("before training ", 4, forward(4).item())
for epoch in range(100):
    for k, v in zip(x_data, y_data):
        l = loss(k, v)
        l.backward()
        # 更新权重
        w.data = w.data - lr * w.grad.data
        w.grad.data.zero_()
        x_axis.append(epoch)
        y_axis.append(l.item())
print("after training ", 4, forward(4).item())

plot.plot(x_axis, y_axis)
plot.xlabel("epoch")
plot.ylabel("loss")
plot.show()
