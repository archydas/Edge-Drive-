import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // r, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        return x + out if self.use_res_connect else out


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        def block(d):
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=d, dilation=d, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.blocks = nn.ModuleList([block(d) for d in [1, 6, 12, 18]])

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)  # ✅ added
        )

    def forward(self, x):
        res = [b(x) for b in self.blocks]

        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=x.shape[2:], mode='bilinear', align_corners=False)

        res.append(gp)
        return self.project(torch.cat(res, dim=1))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Dropout(0.2)  # ✅ added
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class EdgeDriveModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.stage1 = nn.Sequential(DownsampleBlock(32, 64), InvertedResidual(64, 64))
        self.stage2 = nn.Sequential(DownsampleBlock(64, 128), InvertedResidual(128, 128))
        self.stage3 = nn.Sequential(DownsampleBlock(128, 256), InvertedResidual(256, 256))
        self.stage4 = nn.Sequential(DownsampleBlock(256, 512), InvertedResidual(512, 512))

        self.aspp = ASPP(512, 512)

        self.decoder3 = DecoderBlock(512, 256, 256)
        self.decoder2 = DecoderBlock(256, 128, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)

        self.seg_head = nn.Conv2d(64, 2, 1)

        # ✅ improved aux head
        self.aux_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

        self.condition_embed = nn.Embedding(5, 512)

        self.condition_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 5)
        )

    def forward(self, x, condition=None):
        input_size = x.shape[2:]

        x = self.init_conv(x)

        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        x_neck = self.aspp(x4)

        cond_out = self.condition_head(x2)

        if condition is None:
            condition = cond_out.argmax(dim=1)

        cond_vec = self.condition_embed(condition).unsqueeze(-1).unsqueeze(-1)

        # ✅ stronger conditioning + normalization
        x_neck = x_neck + 0.3 * cond_vec
        x_neck = F.layer_norm(x_neck, x_neck.shape[1:])

        d3 = self.decoder3(x_neck, x3)
        d2 = self.decoder2(d3, x2)
        d1 = self.decoder1(d2, x1)

        seg_out = self.seg_head(d1)
        seg_out = F.interpolate(seg_out, size=input_size, mode='bilinear', align_corners=False)

        if self.training:
            aux_out = self.aux_head(d2)
            aux_out = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=False)
            return seg_out, aux_out, cond_out

        return seg_out, cond_out