"""
随机梯度
"""
import matplotlib.pyplot as plot

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0
lr = 0.01


def forward(x):
    return x * w


# 损失函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# 梯度
def gradient(x, y):
    return 2 * x * (x * w - y)


x_axis = []
y_axis = []
print("before training ", 4, forward(4))
for epoch in range(100):
    for k, v in zip(x_data, y_data):
        grad_val = gradient(k, v)
        # 更新权重
        w = w - lr * grad_val
        l = loss(k, v)
        x_axis.append(epoch)
        y_axis.append(l)
print("after training ", 4, forward(4))

plot.plot(x_axis, y_axis)
plot.xlabel("epoch")
plot.ylabel("loss")
plot.show()
