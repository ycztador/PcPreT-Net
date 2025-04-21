import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info

class PyramidConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv9 = nn.Conv2d(in_channels, out_channels, kernel_size=9, padding=4)

    def forward(self, x):
        return self.conv3(x) + self.conv5(x) + self.conv9(x)


class ResidualSpatial(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1))

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return x + out


class SENetBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class FeatureBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pyramid = PyramidConv(channels, channels)
        self.residual = ResidualSpatial(channels)
        self.se = SENetBlock(channels)

    def forward(self, x):
        p = self.pyramid(x)
        r = self.residual(x)
        merged = p + r
        return self.se(merged)


class GS_CNN(nn.Module):
    def __init__(self, in_channels=1, channels=64, num_blocks=6, out_dim=2048):
        super().__init__()
        self.initial = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[FeatureBlock(channels) for _ in range(num_blocks)])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, out_dim)

    def forward(self, x):
        x = self.initial(x)
        x = self.blocks(x)
        x = self.global_pool(x).flatten(1)
        return self.fc(x)  # Output shape: [N, 2048]

if __name__ == "__main__":

    model = GS_CNN(in_channels=1, channels=64, num_blocks=6, out_dim=2048)

    input_res = (1, 224, 224)

    # FLOPs
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, input_res, as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)

    print(f'FLOPs: {macs}')
    print(f'Params: {params}')