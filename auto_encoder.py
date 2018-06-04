from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from trainer import ModuleTrainer



class AutoEncoder(nn.Module):
  def __init__(self):
    super(AutoEncoder, self).__init__()
    self.encoder = nn.Sequential(
      nn.Linear(784, 128),
      nn.ReLU(True),
      nn.Linear(128, 64),
      nn.ReLU(True),
      nn.Linear(64, 12),
      nn.ReLU(True),
      nn.Linear(12, 3),
      nn.ReLU(True)
    )
    self.decoder = nn.Sequential(
      nn.Linear(3, 12),
      nn.ReLU(True),
      nn.Linear(12, 64),
      nn.ReLU(True),
      nn.Linear(64, 128),
      nn.ReLU(True),
      nn.Linear(128, 784),
      nn.Tanh()
    )

  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = self.encoder(x)
    x = self.decoder(x)
    return x

  def encode(self, img):
    img = img.view(img.size(0), -1)
    code = self.encoder(v(img))
    return code

  def decode(self, code):
      out = self.decoder(code)
      out = (out + 1) * 0.5
      out = out.clamp(0, 1)
      img = out.data.view(-1, 28, 28).numpy()
      return img



MNIST_DATA_HOME = 'D:\MLCode\datasets\MNIST'
mnist_train = datasets.MNIST(MNIST_DATA_HOME, train=True, transform=transforms.ToTensor(), download=True)
ae_mnist = AEDataset(Subset(mnist_train, 5000))
data = DataLoader(ae_mnist, batch_size=32, shuffle=True)

net = AutoEncoder()
criterion = nn.MSELoss()
optimizer = Adam(net.parameters(), lr=0.001, weight_decay=0)
model = ModuleTrainer(net)
model.compile(criterion, optimizer)
model.fit(data, epochs=20)

def batch(dataset, m):
  size = [m] + list(dataset[0][0].size())
  out = torch.FloatTensor(*size)
  for i in range(m):
    out[i] = dataset[i][0]
  return out

m = 15
src = batch(mnist_train, m).view(m, 28, 28)
code = net.encode(src)
out = net.decode(code)
imshow(src, out)