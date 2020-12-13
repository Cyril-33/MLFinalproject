# -*- coding: utf-8 -*-
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import transforms
from model import discriminator, generator

import numpy as np
import matplotlib.pyplot as plt

start_time = time.time()
plt.rcParams['image.cmap'] = 'gray'



# Discriminator Loss => BCELoss
def d_loss_fun(input, target):
    return nn.BCELoss()(input, target)


def g_loss_fun(input):
    target = torch.ones([input.shape[0], 1])
    target = target.to(device)
    return nn.BCELoss()(input, target)

def show_photo(image):
    sqrtn = int(np.ceil(np.sqrt(image.shape[0])))

    for index, image in enumerate(image):
        plt.subplot(sqrtn, sqrtn, index+1)
        plt.imshow(image.reshape(28, 28))


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)
G = generator().to(device)
D = discriminator().to(device)
print(G)
print(D)
epochs = 200
lr = 0.0002
batch_size = 64
g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimize = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
train_set = datasets.MNIST('mnist/', train=True, download=False, transform=transform)
test_set = datasets.MNIST('mnist/', train=False, download=False, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


# Train
for epoch in range(epochs):
    epoch += 1

    for time, data in enumerate(train_loader):
        time += 1
        true_input = data[0].to(device)
        test = 255 * (0.5 * true_input[0] + 0.5)

        true_input = true_input.view(-1, 784)
        true_output = D(true_input)
        true_label = torch.ones(true_input.shape[0], 1).to(device)

        noise = (torch.rand(true_input.shape[0], 128) - 0.5) / 0.5
        noise = noise.to(device)
        fake_input = G(noise)
        fake_output = D(fake_input)
        fake_label = torch.zeros(fake_input.shape[0], 1).to(device)
        target = torch.cat((true_label, fake_label), 0)
        output = torch.cat((true_output, fake_output), 0)
        

        d_optimize.zero_grad()

        d_loss = d_loss_fun(output, target)
        d_loss.backward()
        d_optimize.step()

        noise = (torch.rand(true_input.shape[0], 128)-0.5)/0.5
        noise = noise.to(device)

        fake_input = G(noise)
        fake_output = D(fake_input)

        g_loss = g_loss_fun(fake_output)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if time % 100 == 0 or time == len(train_loader):
            print('[{}/{}, {}/{}] Discriminator_loss: {:.3f}  Generator_loss: {:.3f}'.format(epoch, epochs, time, len(train_loader), d_loss.item(), g_loss.item()))

    photo_numpy = (fake_input.data.cpu().numpy()+1.0)/2.0
    show_photo(photo_numpy[:16])
    plt.show()

    if epoch % 50 == 0:
        torch.save(G, 'Generator_epoch_{}.pth'.format(epoch))
        print('saved.')


print('Training Finished.')
print('Cost Time: {}s'.format(time.time()-start_time))
