import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


# 这里模型主要用到的还是线性回归函数和激活函数两种， 并未用到卷积函数。
# 采用的是将图片拉伸成1*n的形式来进行线性回归运算的

# 鉴别器模型
class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),                                 # 线性回归函数
            nn.LeakyReLU(0.1),                                           # 激活函数-relu
            nn.Linear(128, 1),                                           # 线性回归函数
            nn.Sigmoid(),                                                # 激活函数—sigmoid
        )

    def forward(self, x):
        return self.disc(x)


# 生成器模型
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):                                 # z_dim: 将噪声图片拉伸成一维时候的大小
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),    # 28*28*1---784                 # img_dim： 将生成的二维图像拉伸成一维时候的大小
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


# 超参数的设置
device = "cuda:0" if torch.cuda.is_available() else "cpu"
lr = 3e-4                                                               # 学习率
z_dim = 64                                                              # 128 , 256
image_dim = 28*28*1                                                     # 784
batch_size = 32                                                         # 批量数
num_epochs = 250                                                         # 训练次数

disc = Discriminator(image_dim).to(device)                              # 初始化鉴别器
gen = Generator(z_dim, image_dim).to(device)                            # 初始化生成器

fix_noise = torch.randn((batch_size, z_dim)).to(device)                 # 随机生成的噪声图像，大小为批量数*将噪声图片拉伸成一维时候的大小
transforms = transforms.Compose(
    [transforms.ToTensor(),
     #transforms.Normalize((0.1307, ), (0.3081, )),
     transforms.Normalize((0.5, ), (0.5, )),
     ]
)

# 下载数据和定义数据流
dataset = datasets.MNIST(root="../data/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义优化器类型和需要优化的参数
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)

# 损失函数
criterion = nn.BCELoss()

# 结果文件存储路径
writer_fake = SummaryWriter(f"../sun/GAN_MINIST/fake")
writer_real = SummaryWriter(f"../sun/GAN_MINIST/real")
step = 0

# 进行训练
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):                      # 遍历读取影像数据
        real = real.view(-1, 784).to(device)                            # 将影像拉升成一维的数组形式
        batch_size = real.shape[0]

        # 训练鉴别器  max(log(D(real)))+log(1-D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)               # 噪声影像
        fake = gen(noise)                                               # 将噪声影像放入生成器里面
        disc_real = disc(real).view(-1)                                 # 将真实影像放入鉴别器中进行处理
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))   # a. 计算鉴别器处理的真实影像和 1 的损失值———得log(D(real))

        disc_fake = disc(fake).view(-1)                                 # 将噪声影像放入鉴别器中进行处理
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))  # b. 计算鉴别器处理的生成影像和 0 的损失值———得log(1-D(G(z)))
        lossD = (lossD_real + lossD_fake)/2                             # 得到鉴别器的损失值（上面讲解会提到）
        disc.zero_grad()                                                # 梯度清零
        lossD.backward(retain_graph=True)                               # 后向传播
        opt_disc.step()                                                 # 更新参数

        # 训练生成器  min(log(1-D(G(z)))  ----max (log(D(G(z)))
        output = disc(fake).view(-1)                                    # 将生成的fake图像放入鉴别器中
        lossG = criterion(output, torch.ones_like(output))              # c. 得到生成器的损失值 ——— 得log(D(G(z))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch[{epoch}/{num_epochs}]  Batch {batch_idx}/{len(loader)} \ "
                f"Loss D:{lossD:.4f}, Loss G：{lossG:.4f}"
            )
            with torch.no_grad():
                fake = gen(fix_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist  Real Images", img_grid_real, global_step=step
                )
                step += 1

#　Things to try:-- 怎样去提高模型的性能
# 1. What happens if you use larger network?
# 2．Better normalization with BatchNorm
# 3．Different learning rate (is there a better one)?
# 4. Change architecture to a CNN