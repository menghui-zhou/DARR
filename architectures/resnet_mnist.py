import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetMNIST(nn.Module):
    def __init__(self, block, num_blocks, feature_dim=128):
        super(ResNetMNIST, self).__init__()
        self.in_planes = 64
        self.feature_dim = feature_dim
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, feature_dim, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return F.normalize(out)


# def ResNet10MNIST(feature_dim=512):
#     return ResNetMNIST(BasicBlock, [1, 1, 1, 1], feature_dim)
#
#
# def MNISTNet(feature_dim=128):
#     return ResNetMNIST(BasicBlock, [1, 1, 1, 1], feature_dim)


class ResNetMNIST_CE(nn.Module):
    def __init__(self, block, num_blocks, feature_dim=128, num_classes=10):
        super(ResNetMNIST_CE, self).__init__()
        self.in_planes = 64
        self.feature_dim = feature_dim
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, feature_dim, num_blocks[3], stride=2)

        # Classification layer
        self.classifier = nn.Linear(feature_dim, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def extract_feature(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return F.normalize(out)

    def forward(self, x):
        out = self.extract_feature(x)
        out = self.classifier(out)
        return out


# def MNISTNet_CE(feature_dim=128, num_classes=10):
#     return ResNetMNIST_CE(BasicBlock, [1, 1, 1, 1], feature_dim, num_classes)

#
# #
# #
class MNISTNet(nn.Module):
    def __init__(self, fd):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=None)
        self.fc1 = nn.Linear(9216, 1024)
        self.fc2 = nn.Linear(1024, fd)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten-> [batch-size, 9214]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = F.normalize(x)

        return x


class MNISTNet_CE(nn.Module):
    def __init__(self, fd, n_class=10):
        super(MNISTNet_CE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=None)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(9216, 1024)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, fd)
        self.fc3 = nn.Linear(fd, n_class)

    def extract_feature(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)  # flatten-> [batch-size, 9216]
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        x = F.normalize(x)
        return x

    def forward(self, x):
        x = self.extract_feature(x)
        x = self.fc3(x)
        return x




class MNISTTinyNet(nn.Module):
    def __init__(self, fd):
        super(MNISTTinyNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, fd)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = F.normalize(x)


        return x


class MNISTTinyNet_CE(nn.Module):
    def __init__(self, fd, n_class=10):
        super(MNISTTinyNet_CE, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, fd)
        self.fc4 = nn.Linear(fd, n_class)

    def extract_feature(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.normalize(x)
        return x

    def forward(self, x):
        x = self.extract_feature(x)
        x = self.fc4(x)
        return x


