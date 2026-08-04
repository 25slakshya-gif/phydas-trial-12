"""
model.py

All architecture classes for RepPhyDAS Trial 12, moved verbatim from the
original single-file script. No mathematical operation, tensor shape,
layer, or variable name has been changed. This file contains ONLY
nn.Module definitions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OmniDetailAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )
        scharr_x = torch.tensor([[ -3, 0,  3], [-10, 0, 10], [ -3, 0,  3]], dtype=torch.float32)
        scharr_y = torch.tensor([[ -3,-10,-3], [  0, 0,  0], [  3, 10,  3]], dtype=torch.float32)
        scharr_d1 = torch.tensor([[  0, 3, 10], [ -3, 0,  3], [-10,-3,  0]], dtype=torch.float32)
        scharr_d2 = torch.tensor([[ 10, 3,  0], [  3, 0, -3], [  0,-3,-10]], dtype=torch.float32)
        weight = torch.stack([scharr_x, scharr_y, scharr_d1, scharr_d2], dim=0).unsqueeze(1)
        self.register_buffer('scharr_weight', weight)

    def forward(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.view(B * C, 1, H, W)
        details = F.conv2d(x_reshaped, self.scharr_weight, padding=1)
        detail_mag = torch.sqrt(torch.sum(details**2, dim=1))
        detail_mag = detail_mag.view(B, C, H, W)
        detail_mag = detail_mag / (detail_mag.max() + 1e-6)
        return x + (x * self.attention(detail_mag))


class AsymFastRepBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv3x3 = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=True)
        self.conv1x1 = nn.Conv2d(c, c, 1, groups=c, bias=True)
        self.conv1x3 = nn.Conv2d(c, c, kernel_size=(1, 3), padding=(0, 1), groups=c, bias=True)
        self.conv3x1 = nn.Conv2d(c, c, kernel_size=(3, 1), padding=(1, 0), groups=c, bias=True)
        self.pw1 = nn.Conv2d(c, c * 2, 1)
        self.pw2 = nn.Conv2d(c, c, 1)

    def forward(self, x):
        h = self.conv3x3(x) + self.conv1x1(x) + self.conv1x3(x) + self.conv3x1(x) + x
        h = self.pw1(h)
        h1, h2 = h.chunk(2, dim=1)
        return x + self.pw2(h1 * torch.sigmoid(h2))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.ca_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sa_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        ca_avg = self.ca_mlp(F.adaptive_avg_pool2d(x, 1))
        ca_max = self.ca_mlp(F.adaptive_max_pool2d(x, 1))
        x = x * torch.sigmoid(ca_avg + ca_max)
        sa_avg = torch.mean(x, dim=1, keepdim=True)
        sa_max, _ = torch.max(x, dim=1, keepdim=True)
        sa_map = self.sa_conv(torch.cat([sa_avg, sa_max], dim=1))
        return x * torch.sigmoid(sa_map)


class OptimizedSSPD(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        gx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
        gy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
        q1 = torch.mean(torch.sqrt(gx**2 + gy**2 + 1e-6), dim=[2,3])
        lap = F.conv2d(x, torch.tensor([[[[0,1,0],[1,-4,1],[0,1,0]]]], dtype=x.dtype, device=x.device), padding=1)
        q2 = torch.mean(torch.abs(lap), dim=[2,3]) / (q1 + 1e-5)
        q3 = torch.std(x, dim=[2,3]) / (torch.mean(x, dim=[2,3]) + 1e-5)
        q4 = torch.mean(((x < 0.0) | (x > 1.0)).float(), dim=[2,3])
        q5 = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        q6 = F.adaptive_max_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1) - -F.adaptive_max_pool2d(-x, (1, 1)).squeeze(-1).squeeze(-1)
        return torch.cat([q1, q2, q3, q4, q5, q6], dim=1)


class LaplacianEdgeExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32))
    def forward(self, x): return F.conv2d(x, self.weight, padding=1)


class NullSpaceDetailEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.extract = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.PReLU(), nn.Conv2d(8, 1, 3, padding=1))
    def forward(self, delta_n): return delta_n + self.extract(delta_n)


class RepPhyDAS_Trial12(nn.Module):
    def __init__(self, base_width=32):
        super().__init__()
        self.base_sharpness = LaplacianEdgeExtractor()
        self.shallow = nn.Conv2d(2, base_width, 3, padding=1)
        self.sspd = OptimizedSSPD()
        self.cond_fusion = nn.Linear(6, base_width)

        self.enc1 = nn.Sequential(AsymFastRepBlock(base_width), AsymFastRepBlock(base_width))
        self.omni_detail_attn = OmniDetailAttention(base_width)

        self.down = nn.Conv2d(base_width, base_width * 2, 2, stride=2)
        self.bottleneck = nn.Sequential(AsymFastRepBlock(base_width * 2), AsymFastRepBlock(base_width * 2), AsymFastRepBlock(base_width * 2))

        self.up = nn.ConvTranspose2d(base_width * 2, base_width, 2, stride=2)
        self.dec1 = nn.Sequential(AsymFastRepBlock(base_width), AsymFastRepBlock(base_width))

        self.pixel_shuffle_up = nn.Sequential(
            nn.Conv2d(base_width, base_width * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(base_width, base_width, 3, padding=1)
        )

        self.clarity_attention = CBAM(base_width)
        self.out_residual = nn.Conv2d(base_width, 1, 3, padding=1)

        self.D_omega = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1, bias=False), nn.AvgPool2d(2, stride=2), nn.Conv2d(8, 1, 1, bias=False))
        for p in self.D_omega.parameters(): p.requires_grad = False

        self.C_psi = nn.Sequential(nn.Conv2d(3, base_width, 3, padding=1), AsymFastRepBlock(base_width), nn.Conv2d(base_width, 1, 3, padding=1))
        self.null_enhancer = NullSpaceDetailEnhancer()
        self.g_min = 0.40; self.tau = 0.5; self.T = 0.1

    def forward(self, y):
        B_y = F.interpolate(y, scale_factor=2, mode='bicubic', align_corners=False)
        sharp_map = self.base_sharpness(y)
        x = self.shallow(torch.cat([y, sharp_map], dim=1))

        p_y = self.sspd(y)
        z = self.cond_fusion(p_y).unsqueeze(-1).unsqueeze(-1)
        x = x + z

        e1 = self.enc1(x)
        e1_fine = self.omni_detail_attn(e1)

        b = self.bottleneck(self.down(e1))
        d1 = self.dec1(self.up(b) + e1_fine)

        d_up = self.pixel_shuffle_up(d1)
        d_sharp = self.clarity_attention(d_up)

        Delta_x = self.out_residual(d_sharp)
        x0 = B_y + Delta_x

        D_x0 = self.D_omega(x0)
        r_lf = F.interpolate(y - D_x0, scale_factor=2, mode='bilinear', align_corners=False)
        r_edge = F.interpolate(torch.abs(y[:,:,1:,:] - y[:,:,:-1,:]) - torch.abs(D_x0[:,:,1:,:] - D_x0[:,:,:-1,:]), size=(x0.shape[2], x0.shape[3]), mode='bilinear', align_corners=False)

        c = self.C_psi(torch.cat([x0, r_lf, r_edge], dim=1))
        Delta = Delta_x + c

        Delta_v = F.interpolate(self.D_omega(Delta), scale_factor=2, mode='bicubic', align_corners=False)
        Delta_n_refined = self.null_enhancer(Delta - Delta_v)

        RCE = torch.mean(torch.abs(self.D_omega(x0) - y), dim=[1,2,3], keepdim=True)
        e_n = torch.mean(torch.abs(Delta_n_refined), dim=[1,2,3], keepdim=True) / (torch.mean(torch.abs(Delta_v), dim=[1,2,3], keepdim=True) + 1e-5)
        g_n = self.g_min + (1.0 - self.g_min) * torch.sigmoid((self.tau - (RCE + e_n)) / self.T)

        return B_y + Delta_v + g_n * Delta_n_refined
