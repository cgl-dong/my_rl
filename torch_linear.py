"""
pytorch 实现 Linear
"""
import os

import matplotlib.pyplot as plot
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


class LinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModule()

loss = torch.nn.MSELoss(reduction='sum')  # 损失函数
optimizer = torch.optim.SGD(model.parameters(), 0.01)  # 优化器

x_axis = []
y_axis = []
# 训练

for epoch in range(100):
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

print('w = ', model.linear.weight)
print('b = ', model.linear.bias)
x_test = torch.Tensor([[3.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)
