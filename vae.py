import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 100
BATCH_SIZE = 50
LR = 0.001         # learning rate
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ]),
    download=DOWNLOAD_MNIST
)
test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ]),
    download=DOWNLOAD_MNIST
)
print(train_data.train_data.numpy().shape)
print(test_data.test_data.numpy().shape)
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

plt.imshow(train_data.train_data[2].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[2])
plt.show()

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class Net(nn.Module):
    def __init__(self, gpu_status):
        super(Net, self).__init__()
        self.gpu_status = gpu_status
        self.hidden = 10

        self.en_conv_1 = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh()
        )
        self.en_conv_2 = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh()
        )
        self.en_fc_1 = nn.Linear(16 * 7 * 7, self.hidden)
        self.en_fc_2 = nn.Linear(16 * 7 * 7, self.hidden)

        self.de_fc = nn.Linear(self.hidden, 16 * 7 * 7)
        self.de_conv = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid()
        )
    def encoder(self, x):
        conv_out_1 = self.en_conv_1(x)
        conv_out_1 = conv_out_1.view(x.size(0), -1)

        conv_out_2 = self.en_conv_2(x)
        conv_out_2 = conv_out_2.view(x.size(0), -1)

        encoded_fc1 = self.en_fc_1(conv_out_1)
        encoded_fc2 = self.en_fc_2(conv_out_2)

        return encoded_fc1, encoded_fc2  # 这里分别表示均值和标准差的采样

    def sampler(self, mean, std):
        var = std.mul(0.5).exp_()
        eps = torch.FloatTensor(var.size()).normal_()  # 生成一个与输入大小一致的标准正态分布随机数
        eps = Variable(eps)
        if self.gpu_status:
            eps = eps.cuda()
        return eps.mul(var).add_(mean)

    def decoder(self, x):
        out = self.de_fc(x)
        out = out.view(-1, 16, 7, 7)
        out = self.de_conv(out)
        return out

    def forward(self, x):
        mean, std = self.encoder(x)
        code = self.sampler(mean, std)
        out = self.decoder(code)
        # return encoded_fc, decoded
        return out, code, mean, std

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

autoencoder = Net(gpu_status=use_cuda).to(device)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
bce = nn.BCELoss()
mse = nn.MSELoss()
bce.size_average = False
mse.size_average = False

if torch.cuda.is_available():
    autoencoder = autoencoder.cuda()
    bce = bce.cuda()
    mse = mse.cuda()


def loss_f(out, target, mean, std):
    # 公式上理解，std实际上是方差的对数
    # bceloss = bce(out, target)
    mseloss = mse(out, target)

    # latent_loss = torch.sum(mean.pow(2).add_(std.exp()).mul_(-1).add_(1).add_(std)).mul_(-0.5)
    KLD = -0.5 * torch.sum(1 + std - mean.pow(2) - std.exp())

    return mseloss + 0.0002 * KLD

# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()   # continuously plot

# original data (first row) for viewing
view_data = train_data.train_data[:N_TEST_IMG].view(-1,1,28,28).type(torch.cuda.FloatTensor)


for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape((view_data.cpu()).data.numpy()[i], (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())

for epoch in range(EPOCH):
    for step, (x, b_label) in enumerate(train_loader):

        b_x = Variable(x).to(device)   # batch x, shape (batch, 28*28)
        b_y = Variable(x).to(device)   # batch y, shape (batch, 28*28)

        output, _, mean, std = autoencoder(b_x)
        loss = loss_f(output, b_y, mean, std)
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.item())

            # plotting decoded image (second row)
            decoded_data, _, _, _ = autoencoder(view_data)
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape((decoded_data.cpu()).data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(()); a[1][i].set_yticks(())
            plt.draw(); plt.pause(0.05)

plt.ioff()
plt.show()

# visualize in 3D plot
view_data = train_data.train_data[:200].view(-1, 28*28).type(torch.FloatTensor)/255.
encoded_data, _ = autoencoder(view_data)
fig = plt.figure(2); ax = Axes3D(fig)
X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
values = train_data.train_labels[:200].numpy()
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
plt.show()
