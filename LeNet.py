from torch import nn

class LeNet(nn.Module):

  def __init__(self):
    super(LeNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
    self.bn1 = nn.BatchNorm2d(6)
    self.relu1 = nn.ReLU(inplace=True)
    self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
    self.bn2 = nn.BatchNorm2d(16)
    self.relu2 = nn.ReLU(inplace=True)
    self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
    self.bn3 = nn.BatchNorm2d(120)
    self.relu3 = nn.ReLU(inplace=True)
    self.fc1 = nn.Linear(120, 84)
    self.fc1_relu = nn.ReLU(inplace=True)
    self.dropout1 = nn.Dropout()
    self.fc2 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)
    x = self.maxpool1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu2(x)
    x = self.maxpool2(x)    

    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu3(x)

    x = x.view(-1, 120)
    x = self.fc1(x)
    x = self.fc1_relu(x)
    x = self.dropout1(x)
    x = self.fc2(x)
    return x

