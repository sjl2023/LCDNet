import torch
import torch.nn as nn


class Focus(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1, )

        return x


class Attention(nn.Module):
    def __init__(self, chs, k_size1=3, k_size2=7):
        super(Attention, self).__init__()

        self.conv_ca = nn.Conv1d(1, 1, kernel_size=k_size1, padding=(k_size1 - 1) // 2, bias=False)
        self.conv_sa = nn.Conv2d(1, 1, kernel_size=k_size2, padding=(k_size2 - 1) // 2, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_ca = self.avg_pool(x)
        avg_sa = torch.mean(x, dim=1, keepdim=True)
        weight_ca = self.conv_ca(avg_ca.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        weight_sa = self.conv_sa(avg_sa)
        weight = self.sigmoid(weight_ca + weight_sa)
        return weight * x


class Encoder(nn.Module):

    def __init__(self, in_chs, out_chs, focus=True):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.ReLU6(inplace=True), )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.Conv2d(out_chs, out_chs, kernel_size=3, padding=1, groups=out_chs, bias=True),
            nn.ReLU6(inplace=True), )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_chs, out_chs, kernel_size=5, padding=2, groups=out_chs, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.Conv2d(out_chs, out_chs, 1, 1, 0, bias=True),
            nn.ReLU6(inplace=True), )
        self.focus = Focus()
        self.use_focus = focus
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.att = Attention(out_chs)

    def forward(self, x):
        if self.use_focus:
            x = self.focus(x)
        else:
            x = self.maxpool(x)
        x1 = self.conv1(x)
        x2 = self.conv1x1(x)
        x = self.conv2(x1) + x2
        x = self.att(x)
        return x


class conv(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(conv, self).__init__()

        self.conv1x1 = nn.Conv2d(in_chs, out_chs, 1, 1, 0, bias=False)
        self.conv3x3 = nn.Conv2d(out_chs, out_chs, 3, 1, 1, groups=out_chs, bias=True)
        self.bn = nn.BatchNorm2d(out_chs)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.bn(self.conv1x1(x))
        x = self.silu(self.conv3x3(x) + x)
        return x


class CDNet(nn.Module):

    def __init__(self):
        super(CDNet, self).__init__()
        n = 64
        chs = [n, 4 * n, 4 * n, 4 * n]

        self.e1 = Encoder(6 * 4, chs[0])
        self.e2 = Encoder(chs[0] * 4, chs[1])
        self.e3 = Encoder(chs[1], chs[2], False)
        self.e4 = Encoder(chs[2], chs[3], False)

        self.conv1 = conv(chs[2] + chs[0] // 2, chs[2])
        self.conv2 = conv(chs[1] + chs[0] // 2 + chs[0] // 2, chs[1])
        self.conv3 = conv(chs[0] + chs[0] // 2 + chs[0] // 2 + chs[0] // 2, chs[0])

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv1 = conv(chs[3], chs[0] // 2)
        self.up_conv2 = conv(chs[2], chs[0] // 2)
        self.up_conv3 = conv(chs[1], chs[0] // 2)
        self.up_conv4 = conv(chs[0], chs[0] // 2)

        self.final_conv = nn.Conv2d(chs[0] // 2, 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)

        y1 = self.up_conv1(x4)
        y1 = self.up(y1)
        y2 = self.conv1(torch.cat([y1, x3], 1))
        y2 = self.up_conv2(y2)
        y2 = self.up(y2)

        y1 = self.up(y1)
        y3 = self.conv2(torch.cat([y1, y2, x2], 1))
        y3 = self.up_conv3(y3)
        y3 = self.up(y3)
        y2 = self.up(y2)
        y1 = self.up(y1)
        y4 = self.conv3(torch.cat([y1, y2, y3, x1], 1))
        y4 = self.up_conv4(y4)

        y = self.up(y1 + y2 + y3 + y4)
        y = self.final_conv(y)

        return self.sigmoid(y)


if __name__ == '__main__':
    a = torch.rand(2, 3, 256, 256)
    b = torch.rand(2, 3, 256, 256)
    model = CDNet()
    model(a, b)
