"""
pytorch 随机梯度2 y=w1x**2+w2x+b
"""
import os

import matplotlib.pyplot as plot
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = torch.Tensor([1.0])
w2 = torch.Tensor([1.0])
b = torch.Tensor([1.0])
w1.requires_grad = True
w2.requires_grad = True
b.requires_grad = True
lr = 0.01

# 三个参数的函数
def forward(x):
    return w1 * (x ** 2) + w2 * x + b


# 损失函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


x_axis = []
y_axis = []
print("before training ", 3, forward(3).item())
for epoch in range(100):
    for k, v in zip(x_data, y_data):
        l = loss(k, v)
        l.backward()
        # 更新权重
        w1.data = w1.data - lr * w1.grad.data
        w2.data = w2.data - lr * w2.grad.data
        b.data = b.data - lr * b.grad.data
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
        x_axis.append(epoch)
        y_axis.append(l.item())
print("after training ", 3, forward(3).item())

plot.plot(x_axis, y_axis)
plot.xlabel("epoch")
plot.ylabel("loss")
plot.show()
