import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from config.hyperparameters import experiment_parameters
import cv2
from torchvision import transforms as T
from pathlib import Path
from mmengine.model import BaseModule
from einops import rearrange
import typing as t


class ShapeFieldBoundaryHead(nn.Module):
    def __init__(self, in_channel, hidden_ratio=2, eps=1e-6):
        super().__init__()
        hidden = max(8, in_channel // hidden_ratio)
        gray_hidden = max(8, hidden // 2)
        self.eps = eps

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channel, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, hidden), num_channels=hidden),
            nn.ReLU(inplace=True),
        )

        self.gray_proj = nn.Sequential(
            nn.Conv2d(hidden, gray_hidden, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, gray_hidden), num_channels=gray_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(gray_hidden, 1, kernel_size=1, bias=True),
        )


        self.head = nn.Sequential(
            nn.Conv2d(hidden + 3, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, hidden), num_channels=hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, hidden), num_channels=hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )

        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        sobel_x = torch.tensor(
            [[1, 0, -1],
             [2, 0, -2],
             [1, 0, -1]], dtype=torch.float32
        ) / 8.0
        sobel_y = sobel_x.t().contiguous()
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def _shape_field(self, x):
        orig_dtype = x.dtype
        x32 = x.float()

        x_gray = self.gray_proj(x32)

        gx = F.conv2d(x_gray, self.sobel_x.float(), padding=1)
        gy = F.conv2d(x_gray, self.sobel_y.float(), padding=1)

        j11 = self.pool(gx * gx)
        j22 = self.pool(gy * gy)
        j12 = self.pool(gx * gy)

        theta = 0.5 * torch.atan2(2.0 * j12, j11 - j22 + self.eps)

        numer = torch.sqrt(torch.clamp((j11 - j22) ** 2 + 4.0 * (j12 ** 2), min=0.0))
        denom = torch.clamp(j11 + j22, min=self.eps)
        kappa = numer / denom

        theta = torch.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)
        kappa = torch.clamp(kappa, 0.0, 1.0)
        kappa = torch.nan_to_num(kappa, nan=0.0, posinf=1.0, neginf=0.0)

        return theta.to(orig_dtype), kappa.to(orig_dtype)

    def forward(self, diff_feat):
        x = self.reduce(diff_feat)
        theta, kappa = self._shape_field(x)

        dir_code = torch.cat([torch.cos(theta), torch.sin(theta), kappa], dim=1)
        boundary_logits = self.head(torch.cat([x, dir_code], dim=1))

        return boundary_logits, theta, kappa



class BidirectionalSimilarityFusion(nn.Module):
    def __init__(self, in_channel):
        super(BidirectionalSimilarityFusion, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels=in_channel * 3, out_channels=in_channel,
                      kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )

    def cosine_similarity(self, x1, x2):
        dot_product = (x1 * x2).sum(dim=-1, keepdim=True)
        norm1 = torch.norm(x1, dim=-1, keepdim=True)
        norm2 = torch.norm(x2, dim=-1, keepdim=True)
        norm1 = torch.max(norm1, torch.tensor(1e-8, device=x1.device))
        norm2 = torch.max(norm2, torch.tensor(1e-8, device=x2.device))
        return dot_product / (norm1 * norm2)

    def l2_normalize(self, tensor, dim=1):
        norm = tensor.norm(p=2, dim=dim, keepdim=True)
        norm = torch.max(norm, torch.tensor(1e-8, device=tensor.device))
        normalized_tensor = tensor.div(norm.expand_as(tensor))
        return normalized_tensor

    def forward(self, x1, x2, log=None, module_name=None, img_name=None):
        x1_normalized = self.l2_normalize(x1, dim=1)
        x2_normalized = self.l2_normalize(x2, dim=1)
        x_sub = torch.abs(x1_normalized - x2_normalized)

        x_sub = self.l2_normalize(x_sub, dim=1)
        x1_permuted = x1_normalized.permute(0, 2, 3, 1)
        x2_permuted = x2_normalized.permute(0, 2, 3, 1)
        x_sub_permuted = x_sub.permute(0, 2, 3, 1)

        cosine_similarity1 = self.cosine_similarity(x1_permuted, x_sub_permuted)
        cosine_similarity2 = self.cosine_similarity(x2_permuted, x_sub_permuted)
        cosine_similarity1 = cosine_similarity1.permute(0, 3, 1, 2)
        cosine_similarity2 = cosine_similarity2.permute(0, 3, 1, 2)

        cosine_similarity1 = cosine_similarity1 + 2
        cosine_similarity2 = cosine_similarity2 + 2
        s = torch.abs(cosine_similarity1 - cosine_similarity2) + 1
        w1 = s * cosine_similarity1 / (cosine_similarity1 + cosine_similarity2)
        w2 = s * cosine_similarity2 / (cosine_similarity1 + cosine_similarity2)
        w = (experiment_parameters.attention_residual_factor - s / (cosine_similarity1 + cosine_similarity2)) * s

        input = torch.cat([w * x_sub, w1 * x1, w2 * x2], dim=1)
        output = self.fuse(input)

        if log:
            log_list = [x1, x2, cosine_similarity1, cosine_similarity2, output]
            feature_name_list = ['x1', 'x2', 'cosine_similarity1', 'cosine_similarity2', 'output']
            export_feature_maps(
                log_list=log_list,
                module_name=module_name,
                feature_name_list=feature_name_list,
                img_name=img_name,
                module_output=True
            )

        return output


class ConvNormActivation(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride):
        super().__init__()
        self.ConvNormActivation = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel,
                      padding=kernel // 2, bias=False, stride=stride),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.ConvNormActivation(x)


class ChannelGroupedShuffleUnit(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        mid_channel = in_channel // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=experiment_parameters.dropout_p)
        )

    def forward(self, x):
        x1, x2 = split_channel_groups(x)
        x1 = self.conv1(x1)
        output = torch.cat([x1, x2], dim=1)
        return output


class DownsamplingChannelGroupedShuffleUnit(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1,
                      stride=2, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=experiment_parameters.dropout_p)
        )
        self.conv_res = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        output1 = self.conv1(x)
        output2 = self.conv_res(x)
        output = torch.cat([output1, output2], dim=1)
        return output


class CrossTemporalChannelExchange(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, x1, x2):
        N, C, H, W = x1.shape
        exchange_mask = (torch.arange(C, device=x1.device) % self.p == 0)
        exchange_mask1 = exchange_mask.int().expand((N, C)).unsqueeze(-1).unsqueeze(-1)
        exchange_mask2 = 1 - exchange_mask1
        out_x1 = exchange_mask1 * x1 + exchange_mask2 * x2
        out_x2 = exchange_mask1 * x2 + exchange_mask2 * x1
        return out_x1, out_x2


class SpatialChannelSelfAttention(BaseModule):
    def __init__(
            self,
            dim: int,
            head_num: int,
            window_size: int = 7,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            norm_cfg: t.Dict = dict(type='BN'),
            act_cfg: t.Dict = dict(type='ReLU'),
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
    ):
        super(SpatialChannelSelfAttention, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.window_size = window_size

        assert self.dim // 4, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = group_chans = self.dim // 4

        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)

        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()

        self.fuse_conv = nn.Sequential(
            nn.Conv1d(2 * dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)
        )

        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2,
                                        out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)

    def space_to_chans(self, x):
        b, c, h, w = x.size()
        ws = self.window_size
        assert h % ws == 0 and w % ws == 0, "H and W should be divisible by window size."
        x = x.reshape(b, c, h // ws, ws, w // ws, ws)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.reshape(b, c * ws * ws, h // ws, w // ws)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h_, w_ = x.size()

        x_avg_h = x.mean(dim=3)
        x_max_h, _ = x.max(dim=3)
        x_h = torch.cat([x_avg_h, x_max_h], dim=1)
        x_h = self.fuse_conv(x_h)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)

        x_avg_w = x.mean(dim=2)
        x_max_w, _ = x.max(dim=2)
        x_w = torch.cat([x_avg_w, x_max_w], dim=1)
        x_w = self.fuse_conv(x_w)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l),
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)

        x = x * x_h_attn * x_w_attn

        y = self.down_func(x)
        y = self.conv_d(y)

        _, _, h_, w_ = y.size()
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)

        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)',
                      head_num=int(self.head_num), head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)',
                      head_num=int(self.head_num), head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)',
                      head_num=int(self.head_num), head_dim=int(self.head_dim))

        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        attn = attn @ v

        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w',
                         h=int(h_), w=int(w_))
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)
        return attn * x


class HybridAttentionBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        dim = in_channel
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv3_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv3_2 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=dim * 4, out_channels=dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.sigmoid = nn.Sigmoid()
        self.k = resolve_adaptive_kernel_size(dim)
        self.channel_conv = nn.Conv1d(2, 1, kernel_size=self.k, padding=self.k // 2)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.SpatialChannelSelfAttention = SpatialChannelSelfAttention(dim=in_channel, head_num=2)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn_3 = self.conv3_1(attn)
        attn_3 = self.conv3_2(attn_3)

        attn = self.conv3(torch.cat([attn_0, attn_1, attn_2, attn_3], dim=1))
        attn = self.sigmoid(attn)
        output1 = attn * u
        output = self.SpatialChannelSelfAttention(output1)
        return output


class EncoderStage(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        assert out_channel == in_channel * 2, 'the out_channel is not in_channel*2 in encoder block'
        self.conv1 = nn.Sequential(
            DownsamplingChannelGroupedShuffleUnit(in_channel=in_channel),
            ChannelGroupedShuffleUnit(in_channel=out_channel),
            ChannelGroupedShuffleUnit(in_channel=out_channel)
        )
        self.conv3 = ConvNormActivation(in_channel=out_channel, out_channel=out_channel, kernel=3, stride=1)
        self.HybridAttentionBlock = HybridAttentionBlock(in_channel=out_channel)

    def forward(self, x, log=False, module_name=None, img_name=None):
        x = self.conv1(x)
        x = self.conv3(x)
        x_res = x.clone()
        output = self.HybridAttentionBlock(x)
        output = x_res + output
        return output


class DecoderStage(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        assert out_channel == in_channel // 2, 'the out_channel is not in_channel//2 in decoder block'
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels=in_channel + out_channel, out_channels=out_channel,
                      kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, de, en):
        de = self.up(de)
        output = torch.cat([de, en], dim=1)
        output = self.fuse(output)
        return output


def resolve_adaptive_kernel_size(in_channel):
    k = int((math.log2(in_channel) + 1) // 2)
    if k % 2 == 0:
        return k + 1
    else:
        return k


def split_channel_groups(x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]


def export_feature_maps(log_list, module_name, feature_name_list, img_name, module_output=True):
    for k, log in enumerate(log_list):
        log = log.clone().detach()
        b, c, h, w = log.size()
        if module_output:
            log = torch.mean(log, dim=1, keepdim=True)
            log = F.interpolate(
                log * 255,
                scale_factor=experiment_parameters.patch_size // h,
                mode='nearest'
            ).reshape(b, experiment_parameters.patch_size, experiment_parameters.patch_size, 1).cpu().numpy().astype(np.uint8)

            log_dir = experiment_parameters.log_path + module_name + '/' + feature_name_list[k] + '/'
            Path(log_dir).mkdir(parents=True, exist_ok=True)

            log_equalize_dir = experiment_parameters.log_path + module_name + '/' + feature_name_list[k] + '_equalize/'
            Path(log_equalize_dir).mkdir(parents=True, exist_ok=True)

            for i in range(b):
                log_i = cv2.applyColorMap(log[i], cv2.COLORMAP_JET)
                cv2.imwrite(log_dir + img_name[i] + '.png', log_i)

                log_i_equalize = cv2.equalizeHist(log[i])
                log_i_equalize = cv2.applyColorMap(log_i_equalize, cv2.COLORMAP_JET)
                cv2.imwrite(log_equalize_dir + img_name[i] + '.png', log_i_equalize)
        else:
            log_dir = experiment_parameters.log_path + module_name + '/' + feature_name_list[k] + '/'
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log = torch.round(log)
            log = F.interpolate(log, scale_factor=experiment_parameters.patch_size // h, mode='nearest').cpu()
            to_pil_img = T.ToPILImage(mode=None)
            for i in range(b):
                log_i = to_pil_img(log[i])
                log_i.save(log_dir + img_name[i] + '.png')


