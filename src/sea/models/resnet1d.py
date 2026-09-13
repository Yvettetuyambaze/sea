"""resnet1d_wang as used by Strodthoff et al., IEEE JBHI 2021.

Matches helme/ecg_ptbxl_benchmarking: stem conv k=7, three BasicBlocks with
kernels [5, 3], inplanes=128, no stem pooling, concat-pool classification head.
fastai is not required.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1d(in_planes: int, out_planes: int, stride: int = 1, kernel_size: int = 3) -> nn.Conv1d:
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size - 1) // 2,
        bias=False,
    )


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, kernel_size=(5, 3), downsample=None):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size // 2 + 1)
        self.conv1 = conv1d(inplanes, planes, stride=stride, kernel_size=kernel_size[0])
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1d(planes, planes, kernel_size=kernel_size[1])
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class AdaptiveConcatPool1d(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([F.adaptive_avg_pool1d(x, 1), F.adaptive_max_pool1d(x, 1)], dim=1)


class ClassifierHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.pool = AdaptiveConcatPool1d()
        self.flatten = nn.Flatten()
        self.bn = nn.BatchNorm1d(in_features * 2)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(self.pool(x))
        return self.fc(self.drop(self.bn(x)))


class ResNet1dWang(nn.Module):
    def __init__(self, num_classes: int, input_channels: int = 12, inplanes: int = 128, dropout: float = 0.5):
        super().__init__()
        self.inplanes = inplanes
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, inplanes, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(inplanes, 1, stride=1)
        self.layer2 = self._make_layer(inplanes, 1, stride=2)
        self.layer3 = self._make_layer(inplanes, 1, stride=2)
        self.head = ClassifierHead(self.inplanes, num_classes, dropout=dropout)
        self.num_classes = num_classes

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )
        layers = [BasicBlock1d(self.inplanes, planes, stride=stride, kernel_size=(5, 3), downsample=downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock1d(self.inplanes, planes, kernel_size=(5, 3)))
        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def freeze_backbone(self) -> None:
        for module in (self.stem, self.layer1, self.layer2, self.layer3):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def resnet1d_wang(num_classes: int, **kwargs) -> ResNet1dWang:
    return ResNet1dWang(num_classes=num_classes, **kwargs)
