"""
梯度下降
"""

import matplotlib.pyplot as plot

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0
lr = 0.01


def forward(x):
    return x * w


# 损失函数
def cost(x_d, y_d):
    cost_value = 0
    for x, y in zip(x_d, y_d):
        y_pred = forward(x)
        cost_value += (y_pred - y) ** 2
    return cost_value / len(x_data)


# 梯度
def gradient(x_d, y_d):
    grad = 0
    for x, y in zip(x_d, y_d):
        grad += 2 * x * (x * w - y)
    return grad / len(x_data)

x_axis = []
y_axis = []
print("before training ", 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    # 更新权重
    w -= lr * grad_val
    print("Epoch : ", epoch, " w: ", w, " loss: ", cost_val)
    x_axis.append(epoch)
    y_axis.append(cost_val)
print("after training ", 4, forward(4))

plot.plot(x_axis,y_axis)
plot.xlabel("epoch")
plot.ylabel("cost")
plot.show()