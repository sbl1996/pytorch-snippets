import torch
from torch import nn

def conv3x3(in_channels, out_channels):
  return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

def make_layers(cfg, in_channels=3, batch_norm=True):
  layers = []
  for v in cfg:
    if v == 'M':
      layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    else:
      layers.append(conv3x3(in_channels, v))
      if batch_norm:
          layers.append(nn.BatchNorm2d(v))
      layers.append(nn.ReLU(inplace=True))
      in_channels = v
  return nn.Sequential(*layers)

class VGG(nn.Module):
  
  def __init__(self, in_channels, feature_layers, fc_layer, num_classes=10, batch_norm=True):
    super(VGG, self).__init__()
    self.zeropad = nn.ZeroPad2d(2)
    self.features = make_layers(feature_layers, in_channels)
    self.classifier = nn.Sequential(
      nn.Linear(1 * 1 * feature_layers[-2], fc_layer),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(fc_layer, num_classes)
    )

  def forward(self, x):
    x = self.zeropad(x)
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x    