import time
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data_util import DoubleXDataset, Subset

def cuda(v):
  if torch.cuda.is_available():
    return v.cuda()
  return v

class VAE(nn.Module):
  def __init__(self):
    super(VAE, self).__init__()
    self.fc1 = nn.Linear(784, 400)
    self.relu1 = nn.ReLU(True)
    self.fc21 = nn.Linear(400, 20)
    self.fc22 = nn.Linear(400, 20)
    self.decoder = nn.Sequential(
      nn.Linear(20, 400),
      nn.ReLU(True),
      nn.Linear(400, 784),
      nn.Sigmoid()
    )

  def encode(self, x):
    x = x.view(x.shape[0], -1)
    x = self.fc1(x)
    x = self.relu1(x)
    mu = self.fc21(x)
    logvar = self.fc22(x)
    return mu, logvar

  def decode(self, z):
    return self.decoder(z)

  def forward(self, x):
    mu, logvar = self.encode(x)
    z = self.reparametrize(mu, logvar)
    x = self.decode(z)
    return x, mu, logvar

  def reparametrize(self, mu, logvar):
    std = (logvar / 2).exp()
    eps = torch.zeros(std.size()).normal_()
    eps = cuda(Variable(eps))
    return eps * std + mu

def loss_func(rec_x, x, mu, logvar):
  BCE = nn.MSELoss(size_average=False)(rec_x, x)
  KLD = (-0.5) * torch.sum(1 + logvar - mu**2 - logvar.exp())
  return BCE + KLD

def take(dataset, m):
  size = [m] + list(dataset[0][0].size())
  out = torch.FloatTensor(*size)
  for i in range(m):
    out[i] = dataset[i][0]
  return out


def main():
  MNIST_DATA_HOME = 'D:\MLCode\datasets\MNIST'
  mnist_train = datasets.MNIST(MNIST_DATA_HOME, train=True, transform=transforms.ToTensor(), download=True)

  m = 60000
  vae_mnist = DoubleXDataset(Subset(mnist_train, m))
  batch_size = 64
  data_loader = DataLoader(vae_mnist, batch_size=batch_size, shuffle=True)
  num_batches = len(data_loader)

  net = cuda(VAE())
  criterion = loss_func
  optimizer = Adam(net.parameters(), lr=0.003, weight_decay=0)
  verbose = True

  loss_avgs = []
  epochs = 10
  for epoch in range(epochs):
    start = time.time()
    losses = []
    i = 0
    loss_avg = 0
    for batch in data_loader:
      i += 1
      x, y = batch
      x = cuda(Variable(x))
      net.zero_grad()
      output, mu, logvar = net(x)
      loss = criterion(output, x, mu, logvar)
      losses.append(float(loss) / batch_size)
      loss.backward()
      optimizer.step()
      loss_avg = (loss_avg * (i - 1) + losses[-1]) / i
      if verbose:
        print('\rEpoch %3d/%3d %5d/%5d\tloss: %f' % (epoch + 1, epochs, min(i*batch_size, m), m, loss_avg), end='')
    end = time.time()
    cost = end - start
    loss_avgs.append(loss_avg)
    if verbose:
      print("Cost: %.1f" % cost)
    else:
      print('Epoch %3d/%3d \tloss: %f' % (epoch + 1, epochs, loss_avgs[-1]))

  plt.plot(loss_avgs)


  m1 = 15
  src = take(mnist_train, m1).view(m1, -1)
  mu, logvar = net.encode(Variable(src))
  z = net.reparametrize(mu, logvar)
  out = net.decode(z).data.view(-1, 28, 28).numpy()
  imshow(src.view(m1, 28, 28), out)


if __name__ == '__main__':
  main()