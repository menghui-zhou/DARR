import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    # Expansion factor used to adjust the number of output channels. For BasicBlock, it's always 1.
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # First convolutional layer with kernel size 3x3, stride as per argument, padding=1 to maintain spatial size, and no bias.
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                               bias=False)  # watch out, no bias
        # Batch normalization for the output of the first convolutional layer.
        self.bn1 = nn.BatchNorm2d(planes)
        # Second convolutional layer, similar to the first but always with stride=1 to maintain size.
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut connection to allow identity mapping.
        self.shortcut = nn.Sequential()
        # If stride is not 1 or in_planes does not equal planes * expansion, adjust dimensions via 1x1 convolutions.
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # Forward pass through the first conv layer, then batch normalization, then ReLU activation.
        out = F.relu(self.bn1(self.conv1(x)))
        # Forward pass through the second conv layer and then batch normalization.
        out = self.bn2(self.conv2(out))
        # Addition of shortcut connection output and output from the second BN layer before applying ReLU.
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    # Expansion factor for bottleneck blocks, typically set to 4.
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # 1x1 convolution that reduces the dimensionality.
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 3x3 convolution, the core of the bottleneck, with stride and padding to maintain or reduce spatial dimensions.
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 1x1 convolution that increases the dimensionality back to planes * expansion.
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        # Shortcut connection similar to BasicBlock but adjusted for the bottleneck's output size.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # Sequentially apply conv1->bn1->ReLU, conv2->bn2->ReLU, and conv3->bn3 to the input.
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # Add the shortcut connection output before applying the final ReLU.
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Define the ResNet architecture.
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, feature_dim=512):
        super(ResNet, self).__init__()
        self.in_planes = 64  # Initial number of planes.
        self.feature_dim = feature_dim  # Dimensionality of the output features.

        # Initial convolution layer with 3 input channels, 64 output channels, kernel size 3x3, stride 1, padding 1, and no bias.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Creating layers of blocks with varying numbers of blocks and planes.
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # Reshaping network to output feature_dim features.
        self.reshape = torch.nn.Sequential(
            nn.Linear(512 * block.expansion, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True),
        )
        # self.reshape = torch.nn.Sequential(
        #     nn.Linear(512 * block.expansion, feature_dim)
        # )

    # Helper function to create a layer composed of blocks.
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # First block could have a different stride.
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion  # Update in_planes for the next layer.
        return nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through initial conv layer, then through each ResNet layer.
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # Average pooling and reshaping.
        # out = F.avg_pool2d(out, 4,(1, 1))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.reshape(out)
        return F.normalize(out)


class ResNet_CE_Nor(nn.Module):
    def __init__(self, block, num_blocks, feature_dim=512, num_classes=10):
        super(ResNet_CE_Nor, self).__init__()
        self.in_planes = 64  # Initial number of planes.
        self.feature_dim = feature_dim  # Dimensionality of the output features.

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Creating layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Fully connected layers for feature extraction
        self.reshape = nn.Sequential(
            nn.Linear(512 * block.expansion, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True)
        )
        # self.reshape = nn.Sequential(
        #     nn.Linear(512 * block.expansion, feature_dim)
        # )

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

        # Adaptive average pooling
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.reshape(out)
        out = F.normalize(out)
        return out

    def forward(self, x):
        out = self.extract_feature(x)
        # Classification output
        out = self.classifier(out)
        return out


class ResNet_CE(nn.Module):
    def __init__(self, block, num_blocks, feature_dim=512, num_classes=10):
        super(ResNet_CE, self).__init__()
        self.in_planes = 64  # Initial number of planes.
        self.feature_dim = feature_dim  # Dimensionality of the output features.

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Creating layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Fully connected layers for feature extraction
        self.reshape = nn.Sequential(
            nn.Linear(512 * block.expansion, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True)
        )

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

        # Adaptive average pooling
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.reshape(out)
        out = F.normalize(out)
        return out

    def forward(self, x):
        out = self.extract_feature(x)
        # Classification output
        out = self.classifier(out)
        return out


# Simplified ResNet class for control experiments without the final reshaping.
class ResNetControl(nn.Module):
    def __init__(self, block, num_blocks, feature_dim=512):
        super(ResNetControl, self).__init__()
        self.in_planes = 64
        self.feature_dim = feature_dim
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
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


def ResNet18(feature_dim=512):
    # Function to create a ResNet with 18 layers (2 blocks per layer) using BasicBlock.
    return ResNet(BasicBlock, [2, 2, 2, 2], feature_dim)


def ResNet18_CE_Nor(feature_dim=512, num_classes=100):
    # Function to create a ResNet with 18 layers (2 blocks per layer) using BasicBlock.
    return ResNet_CE_Nor(BasicBlock, [2, 2, 2, 2], feature_dim, num_classes)


def ResNet18_CE(feature_dim=512, num_classes=100):
    # Function to create a ResNet with 18 layers (2 blocks per layer) using BasicBlock.
    return ResNet_CE(BasicBlock, [2, 2, 2, 2], feature_dim, num_classes)


def ResNet18Control(feature_dim=512):
    # Function to create a control version of ResNet18 without the final reshaping.
    return ResNetControl(BasicBlock, [2, 2, 2, 2], feature_dim)
