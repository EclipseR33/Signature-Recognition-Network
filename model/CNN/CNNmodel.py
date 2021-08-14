import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import math
import numpy as np
from tqdm import tqdm


def conv_batch(in_ch, out_ch, ksize=3, padding=1, stride=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, ksize, stride=stride, padding=padding, groups=groups),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1)
    )


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, groups):
        super(ResidualBlock, self).__init__()

        res_ch = int(in_ch / 2)

        self.layer1 = conv_batch(in_ch, res_ch, ksize=1, padding=0, groups=groups)
        self.layer2 = conv_batch(res_ch, in_ch)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)

        out += residual

        return out


class ArcMarginProduct(nn.Module):
    r"""
    https://github.com/deepinsight/insightface

    Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0, device='cpu'):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.device = device

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device=device)
        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class ArcSRNet3(nn.Module):
    def __init__(self, num_block, num_classes=1000, groups=1, s=30, margin=0.5, fc_dim=100, device='cpu'):
        super(ArcSRNet3, self).__init__()

        self.conv1 = conv_batch(1, 8)
        self.conv2 = conv_batch(8, 16, stride=2)
        self.res_block1 = self.make_layer(ResidualBlock, in_ch=16, num_blocks=num_block[0], groups=groups)
        self.conv3 = conv_batch(16, 32, stride=2)
        self.res_block2 = self.make_layer(ResidualBlock, in_ch=32, num_blocks=num_block[1], groups=groups)
        self.conv4 = conv_batch(32, 64, stride=2)
        self.res_block3 = self.make_layer(ResidualBlock, in_ch=64, num_blocks=num_block[2], groups=groups)

        self.fc = nn.Linear(64, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.final = ArcMarginProduct(64, num_classes, s=s, m=margin, device=device)

        self.final_dim = 64

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def make_layer(self, block, in_ch, num_blocks, groups):
        layers = []

        for i in range(0, num_blocks):
            layers.append(block(in_ch, groups))
        return nn.Sequential(*layers)

    def forward(self, x, label):
        batch_size = x.shape[0]

        out = self.conv1(x)

        out = self.conv2(out)
        out = self.res_block1(out)

        out = self.conv3(out)
        out = self.res_block2(out)

        out = self.conv4(out)
        out = self.res_block3(out)

        out = self.pooling(out)
        out = out.view(batch_size, -1)

        logits = None
        if label is not None:
            # logits = self.fc(out)
            # logits = self.bn(logits)

            logits = self.final(out, label)

        return out, logits


class ArcSRNet4(nn.Module):
    def __init__(self, num_block, num_classes=1000, groups=1, s=30, margin=0.5, fc_dim=100, device='cpu'):
        super(ArcSRNet4, self).__init__()

        self.conv1 = conv_batch(1, 8)
        self.conv2 = conv_batch(8, 16, stride=2)
        self.res_block1 = self.make_layer(ResidualBlock, in_ch=16, num_blocks=num_block[0], groups=groups)
        self.conv3 = conv_batch(16, 32, stride=2)
        self.res_block2 = self.make_layer(ResidualBlock, in_ch=32, num_blocks=num_block[1], groups=groups)
        self.conv4 = conv_batch(32, 64, stride=2)
        self.res_block3 = self.make_layer(ResidualBlock, in_ch=64, num_blocks=num_block[2], groups=groups)
        self.conv5 = conv_batch(64, 128, stride=2)
        self.res_block4 = self.make_layer(ResidualBlock, in_ch=128, num_blocks=num_block[3], groups=groups)

        self.fc = nn.Linear(128, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.final = ArcMarginProduct(128, num_classes, s=s, m=margin, device=device)

        self.final_dim = 128

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def make_layer(self, block, in_ch, num_blocks, groups):
        layers = []

        for i in range(0, num_blocks):
            layers.append(block(in_ch, groups))
        return nn.Sequential(*layers)

    def forward(self, x, label):
        batch_size = x.shape[0]

        out = self.conv1(x)

        out = self.conv2(out)
        out = self.res_block1(out)

        out = self.conv3(out)
        out = self.res_block2(out)

        out = self.conv4(out)
        out = self.res_block3(out)

        out = self.conv5(out)
        out = self.res_block4(out)

        out = self.pooling(out)
        out = out.view(batch_size, -1)

        logits = None
        if label is not None:
            # logits = self.fc(out)
            # logits = self.bn(logits)

            logits = self.final(out, label)

        return out, logits


class ArcSRNet5(nn.Module):
    def __init__(self, num_block, num_classes=1000, groups=1, s=30, margin=0.5, fc_dim=100, is_arc=True, device='cpu'):
        super(ArcSRNet5, self).__init__()
        self.num_classes = num_classes
        self.is_arc = is_arc

        self.conv1 = conv_batch(1, 8)
        self.conv2 = conv_batch(8, 16, stride=2)
        self.res_block1 = self.make_layer(ResidualBlock, in_ch=16, num_blocks=num_block[0], groups=groups)
        self.conv3 = conv_batch(16, 32, stride=2)
        self.res_block2 = self.make_layer(ResidualBlock, in_ch=32, num_blocks=num_block[1], groups=groups)
        self.conv4 = conv_batch(32, 64, stride=2)
        self.res_block3 = self.make_layer(ResidualBlock, in_ch=64, num_blocks=num_block[2], groups=groups)
        self.conv5 = conv_batch(64, 128, stride=2)
        self.res_block4 = self.make_layer(ResidualBlock, in_ch=128, num_blocks=num_block[3], groups=groups)
        self.conv6 = conv_batch(128, 256, stride=2)
        self.res_block5 = self.make_layer(ResidualBlock, in_ch=256, num_blocks=num_block[4], groups=groups)

        self.fc = nn.Linear(256, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()

        self.pooling = nn.AdaptiveAvgPool2d(1)

        # self.final = ArcMarginProduct(fc_dim, num_classes, s=s, m=margin, device=device)
        if is_arc:
            self.final = ArcMarginProduct(256, num_classes, s=s, m=margin, device=device)
        else:
            self.final = nn.Linear(256, num_classes)

        self.final_dim = 256

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def make_layer(self, block, in_ch, num_blocks, groups):
        layers = []

        for i in range(0, num_blocks):
            layers.append(block(in_ch, groups))
        return nn.Sequential(*layers)

    def forward(self, x, label):
        batch_size = x.shape[0]

        out = self.conv1(x)

        out = self.conv2(out)
        out = self.res_block1(out)

        out = self.conv3(out)
        out = self.res_block2(out)

        out = self.conv4(out)
        out = self.res_block3(out)

        out = self.conv5(out)
        out = self.res_block4(out)

        out = self.conv6(out)
        out = self.res_block5(out)

        out = self.pooling(out)
        out = out.view(batch_size, -1)

        logits = None
        if label is not None:
            # logits = self.fc(out)
            # logits = self.bn(logits)
            if self.is_arc:
                logits = self.final(out, label)
            else:
                logits = self.final(out)
        return out, logits


class SiameseSRNet(nn.Module):
    def __init__(self, net, final_dim):
        super(SiameseSRNet, self).__init__()

        self.net = net

        self.final = nn.Sequential(
            nn.Linear(final_dim * 2, final_dim // 2),
            nn.Linear(final_dim // 2, 2)
        )

    def forward(self, img1, img2):
        out1, _ = self.net(img1, None)
        out2, _ = self.net(img2, None)
        out = torch.cat([out1, out2], dim=1)
        out = self.final(out)
        return out


def embedding(dl, model, device):
    embeds = []

    dl = tqdm(enumerate(dl), total=len(dl))
    with torch.no_grad():
        for i, (img, label) in dl:
            img = img.to(device)
            label = label.to(device)

            output, logits = model(img, label)

            embed = output.detach().cpu().numpy()
            # embed = logits.detach().cpu().numpy()
            embeds.append(embed)

    del model
    np_embeds = np.concatenate(embeds)
    print(f'Embedding shape: {np_embeds.shape}')
    del embeds

    return np_embeds


if __name__ == '__main__':
    device = 'cpu'

    net = Net5([1, 2, 3, 3, 2], groups=4)
    print(net.final_dim)
    test_net = TestNet(net, net.final_dim)

    batch_size = 4
    x1 = torch.randn((batch_size, 1, 128, 128))
    x2 = torch.randn((batch_size, 1, 128, 128))
    label = torch.ones(batch_size)
    print(x1.size())
    test_net(x1, x2)
