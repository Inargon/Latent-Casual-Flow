"""
Environment Encoder V2 - CaTSG-Inspired Design
===============================================
重新设计的环境编码器，借鉴 CaTSG 的 EnvInfer 架构。

核心改进：
1. TCN Backbone: 膨胀卷积捕捉多尺度时序模式
2. 三路径特征提取: 统计 + 注意力 + 频谱 (严格按照 CaTSG)
3. Concat 融合: 简单有效的特征组合
4. 支持流模型先验 (可选): 替代高斯先验

与 CaTSG 的区别：
- CaTSG: 输出离散概率 w ∈ R^K (K 个原型的权重)
- LCF:   输出连续高斯 (μ, σ²) ∈ R^{2d} (连续环境空间)

Author: LCF Team+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


# ============================================================================
#                          TCN Backbone (CaTSG Style)
# ============================================================================

class SamePadConv(nn.Module):
    """1D Conv with same padding (保持序列长度不变)."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, :-self.remove]
        return out


class ConvBlock(nn.Module):
    """残差膨胀卷积块."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, dilation: int, final: bool = False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation)
        self.projector = (
            nn.Conv1d(in_channels, out_channels, 1) 
            if in_channels != out_channels or final else None
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network Encoder.
    
    使用膨胀卷积捕捉多尺度时序模式，感受野指数增长。
    """
    
    def __init__(self, in_channels: int, hidden_dim: int, 
                 depth: int = 4, kernel_size: int = 3):
        super().__init__()
        
        # 膨胀因子: 1, 2, 4, 8, ... -> 感受野指数增长
        layers = []
        for i in range(depth):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else hidden_dim
            out_ch = hidden_dim
            layers.append(ConvBlock(in_ch, out_ch, kernel_size, dilation, final=(i == depth - 1)))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) - channel first format
        Returns:
            (B, hidden_dim, T)
        """
        return self.net(x)


# ============================================================================
#                    三路径特征提取 (CaTSG EnvInfer Style)
# ============================================================================

class TemporalStatistics(nn.Module):
    """
    统计路径：提取时序统计特征.
    
    特征: mean, std, max (沿时间维度)
    输出: (B, 3H)
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, T, H) - TCN 输出
        Returns:
            h_stat: (B, 3H)
        """
        h_mean = h.mean(dim=1)                    # (B, H)
        h_std = h.std(dim=1) + 1e-6               # (B, H)
        h_max = h.max(dim=1).values               # (B, H)
        
        return torch.cat([h_mean, h_std, h_max], dim=-1)  # (B, 3H)


class AttentionPooling(nn.Module):
    """
    注意力路径：学习时序权重进行加权池化.
    
    通过线性层 + Softmax 学习每个时间步的重要性.
    输出: (B, H)
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score_fn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, T, H)
        Returns:
            h_attn: (B, H)
            weights: (B, T, 1) - 注意力权重 (用于可视化)
        """
        # 计算注意力分数
        scores = self.score_fn(h)                 # (B, T, 1)
        weights = F.softmax(scores, dim=1)        # (B, T, 1)
        
        # 加权求和
        h_attn = (h * weights).sum(dim=1)         # (B, H)
        
        return h_attn, weights


class SpectralAnalysis(nn.Module):
    """
    频谱路径：提取频域特征.
    
    特征:
    1. 频谱质心 (spectral centroid): 每个通道的加权频率中心
    2. Top-K 峰值: 全局功率谱的主要频率成分
    
    输出: (B, H + K_p)
    """
    
    def __init__(self, hidden_dim: int, topk_peaks: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.topk_peaks = topk_peaks
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, T, H)
        Returns:
            h_spec: (B, H + K_p)
        """
        B, T, H = h.shape
        device = h.device
        
        # 1. rFFT 沿时间维度
        fft = torch.fft.rfft(h, dim=1)            # (B, T//2+1, H) complex
        psd = fft.real ** 2 + fft.imag ** 2       # 功率谱密度
        
        n_freqs = psd.shape[1]
        
        # 2. 频谱质心 (每个通道)
        # centroid = Σ(psd * freq) / Σ(psd)
        freq = torch.linspace(0, 1, n_freqs, device=device).view(1, -1, 1)
        psd_sum = psd.sum(dim=1, keepdim=True) + 1e-8
        centroid = (psd * freq).sum(dim=1) / psd_sum.squeeze(1)  # (B, H)
        
        # 3. Top-K 峰值 (全局平均后)
        psd_mean = psd.mean(dim=2)                # (B, n_freqs)
        
        # 确保 k 不超过 n_freqs
        k = min(self.topk_peaks, n_freqs)
        topk_vals, _ = torch.topk(psd_mean, k=k, dim=-1)  # (B, K)
        
        # 归一化
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 如果 k < topk_peaks，填充零
        if k < self.topk_peaks:
            padding = torch.zeros(B, self.topk_peaks - k, device=device)
            topk_vals = torch.cat([topk_vals, padding], dim=-1)
        
        return torch.cat([centroid, topk_vals], dim=-1)  # (B, H + K_p)


# ============================================================================
#                      CaTSG 风格位置编码
# ============================================================================

def add_positional_encoding_catsg(c: torch.Tensor) -> torch.Tensor:
    """
    添加 CaTSG 风格的序列位置编码.
    
    与 CaTSG prepare_c 完全一致:
    - 使用序列位置 t ∈ [0, T-1] 而非真实小时时间
    - phi = 2π * t / T
    - pos_enc = [sin(phi), cos(phi)]
    
    Args:
        c: (B, T, D_c) 条件序列
        
    Returns:
        c_with_pos: (B, T, D_c + 2) 添加位置编码后的条件
    """
    B, T, D = c.shape
    device = c.device
    dtype = c.dtype
    
    # CaTSG 风格位置编码
    t = torch.arange(T, device=device, dtype=dtype)
    phi = (2.0 * math.pi) * t / float(T)  # [0, 2π)
    pos_enc = torch.stack([torch.sin(phi), torch.cos(phi)], dim=-1)  # (T, 2)
    pos_enc = pos_enc.unsqueeze(0).expand(B, T, 2)  # (B, T, 2)
    
    return torch.cat([c, pos_enc], dim=-1)  # (B, T, D_c + 2)


# ============================================================================
#                         主编码器模块
# ============================================================================

class EnvironmentEncoderV2(nn.Module):
    """
    Environment Encoder V2: CaTSG-Inspired Design
    
    架构流程:
    1. [x, c] -> TCN -> h' (B, T, H)
    2. h' -> 三路径特征提取:
       - 统计路径 -> h_stat (B, 3H)
       - 注意力路径 -> h_attn (B, H)
       - 频谱路径 -> h_spec (B, H + K_p)
    3. Concat([h_stat, h_attn, h_spec]) -> h'' (B, 5H + K_p)
    4. LayerNorm + MLP -> h (B, H)
    5. 高斯头 -> (μ, log σ²)
    
    Args:
        seq_len: 序列长度
        input_dim: 输入 x 的通道数
        cond_dim: 条件 c 的通道数 (不含位置编码)
        hidden_dim: 隐藏层维度
        env_dim: 环境潜在空间维度
        tcn_depth: TCN 深度
        topk_peaks: 频谱 Top-K 峰值数量
        add_positional_encoding: 是否添加 CaTSG 风格位置编码
    """
    
    def __init__(
        self,
        seq_len: int = 96,
        input_dim: int = 1,
        cond_dim: int = 2,
        hidden_dim: int = 64,
        env_dim: int = 4,
        tcn_depth: int = 4,
        topk_peaks: int = 8,
        dropout: float = 0.1,
        add_positional_encoding: bool = False,  # CaTSG 风格位置编码（默认关闭，使用真实小时特征）
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.env_dim = env_dim
        self.topk_peaks = topk_peaks
        self.add_positional_encoding = add_positional_encoding  # 新增
        
        # 实际的条件维度（包含位置编码）
        actual_cond_dim = cond_dim + 2 if add_positional_encoding else cond_dim
        self.actual_cond_dim = actual_cond_dim
        
        # ============ TCN Backbone ============
        fused_dim = input_dim + actual_cond_dim
        self.tcn = TCNEncoder(
            in_channels=fused_dim,
            hidden_dim=hidden_dim,
            depth=tcn_depth,
            kernel_size=3,
        )
        
        # ============ 三路径特征提取 ============
        self.stat_path = TemporalStatistics(hidden_dim)
        self.attn_path = AttentionPooling(hidden_dim)
        self.spec_path = SpectralAnalysis(hidden_dim, topk_peaks)
        
        # ============ 融合层 ============
        # h'' 维度: 3H (stat) + H (attn) + H + K_p (spec) = 5H + K_p
        fused_feat_dim = 5 * hidden_dim + topk_peaks
        
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_feat_dim),
            nn.Linear(fused_feat_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # ============ 高斯输出头 ============
        # 注意: 不使用 Tanh，因为 VICReg 需要 std >= 1
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, env_dim),
            # nn.Tanh(),  # 移除: 与 VICReg 的 std >= 1 目标冲突
        )
        
        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, env_dim),
        )
        
        # 初始化 logvar 偏置为负值 (鼓励探索)
        nn.init.constant_(self.logvar_head[-1].bias, -1.0)
        
        # 是否对 μ 进行 L2 归一化（与 CaTSG 一致）
        self.normalize_mu = True
        
        # ============ 可选: 流模型先验 ============
        self.flow_prior = None  # 将在外部设置
    
    def _to_btd(self, x: torch.Tensor, expected_d: int) -> torch.Tensor:
        """确保张量格式为 (B, T, D)."""
        original_shape = x.shape
        
        # 先去除所有大小为1的额外维度
        while x.dim() > 3:
            squeezed = False
            for dim in range(x.dim()):
                if x.shape[dim] == 1 and dim not in [0]:
                    x = x.squeeze(dim)
                    squeezed = True
                    break
            if not squeezed:
                break
        
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() == 3:
            # 判断哪个维度是 D
            if x.shape[-1] == expected_d:
                pass  # 已经是 (B, T, D)
            elif x.shape[1] == expected_d:
                x = x.transpose(1, 2)  # (B, D, T) -> (B, T, D)
            # 如果都不匹配，假设当前格式正确
        else:
            raise ValueError(f"Cannot convert to (B,T,D): dim={x.dim()}, shape={original_shape}")
        
        return x
    
    def encode(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        编码到高斯参数.
        
        Args:
            x: 输入序列 (B, T, D_x) 或 (B, D_x, T)
            c: 条件序列 (B, T, D_c) 或 (B, D_c, T) 或 (B, D_c)
            return_intermediates: 是否返回中间特征
            
        Returns:
            Dict with 'mu', 'logvar', and optionally intermediate features
        """
        # ============ 格式转换 ============
        x_orig_shape = x.shape
        x = self._to_btd(x, self.input_dim)  # (B, T, D_x)
        B, T, D_x = x.shape
        
        c_orig_shape = c.shape
        if c.dim() == 2:
            c = c.unsqueeze(1).expand(B, T, -1)  # (B, D_c) -> (B, T, D_c)
        else:
            c = self._to_btd(c, self.cond_dim)  # (B, T, D_c)
        
        # 确保 x 和 c 的 T 维度匹配
        if x.shape[1] != c.shape[1]:
            raise ValueError(
                f"T dimension mismatch: x.shape={x.shape} (orig={x_orig_shape}), "
                f"c.shape={c.shape} (orig={c_orig_shape})"
            )
        
        # ============ 添加 CaTSG 风格位置编码 ============
        if self.add_positional_encoding:
            c = add_positional_encoding_catsg(c)  # (B, T, cond_dim + 2)
        
        # ============ Early Fusion + TCN ============
        xc = torch.cat([x, c], dim=-1)        # (B, T, D_x + actual_cond_dim)
        xc = xc.transpose(1, 2)               # (B, D, T) for conv
        
        h_prime = self.tcn(xc)                # (B, H, T)
        h_prime = h_prime.transpose(1, 2)     # (B, T, H)
        
        # ============ 三路径特征提取 ============
        h_stat = self.stat_path(h_prime)              # (B, 3H)
        h_attn, attn_weights = self.attn_path(h_prime)  # (B, H), (B, T, 1)
        h_spec = self.spec_path(h_prime)              # (B, H + K_p)
        
        # ============ Concat 融合 ============
        h_concat = torch.cat([h_stat, h_attn, h_spec], dim=-1)  # (B, 5H + K_p)
        h = self.fusion(h_concat)             # (B, H)
        
        # ============ 高斯参数 ============
        mu = self.mu_head(h)  # 已经过 tanh，范围 [-1, 1]
        
        # 🔑 L2 归一化（与 CaTSG 一致），让 μ 在单位超球面上
        if self.normalize_mu:
            mu = F.normalize(mu, p=2, dim=-1)
        
        logvar = self.logvar_head(h)
        logvar = torch.clamp(logvar, min=-10, max=2)  # 数值稳定
        
        result = {'mu': mu, 'logvar': logvar}
        
        if return_intermediates:
            result.update({
                'h_stat': h_stat,
                'h_attn': h_attn,
                'h_spec': h_spec,
                'h_fused': h,
                'attn_weights': attn_weights,
            })
        
        return result
    
    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        重参数化采样.
        
        Args:
            mu: 均值 (B, D_e)
            logvar: 对数方差 (B, D_e)
            num_samples: 采样数量
            
        Returns:
            e: 采样结果 (B, D_e) 或 (B, N, D_e)
        """
        std = torch.exp(0.5 * logvar)
        
        if num_samples == 1:
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            B, D = mu.shape
            eps = torch.randn(B, num_samples, D, device=mu.device, dtype=mu.dtype)
            return mu.unsqueeze(1) + std.unsqueeze(1) * eps
    
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: Optional[torch.Tensor] = None,  # 保持接口兼容，但 V2 暂不使用 t
        num_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        完整前向传播：编码 + 采样.
        
        Args:
            x: 输入序列 (B, T, D)
            c: 条件 (B, T, D_c) 或 (B, D_c)
            t: 时间步 (暂时忽略，保持接口兼容)
            num_samples: MC 采样数量
            
        Returns:
            Dict with 'e', 'mu', 'logvar'
        """
        encoded = self.encode(x, c)
        mu, logvar = encoded['mu'], encoded['logvar']
        e = self.reparameterize(mu, logvar, num_samples)
        
        return {
            'e': e,
            'mu': mu,
            'logvar': logvar,
        }
    
    def compute_kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        e_samples: Optional[torch.Tensor] = None,
        free_bits: float = 0.0,
    ) -> torch.Tensor:
        """
        计算 KL 散度.
        
        如果设置了 flow_prior，使用蒙特卡洛估计；
        否则使用标准高斯先验的解析解。
        
        Args:
            mu: 后验均值 (B, D)
            logvar: 后验对数方差 (B, D)
            e_samples: 采样的 e (用于流先验)
            free_bits: 每维度最小 KL
            
        Returns:
            kl: 标量
        """
        if self.flow_prior is not None and e_samples is not None:
            # 流模型先验: 蒙特卡洛估计
            # log q(e|x) - log p_φ(e)
            log_q = self._gaussian_log_prob(e_samples, mu, logvar)
            log_p = self.flow_prior.log_prob(e_samples)
            kl_per_sample = log_q - log_p
            return kl_per_sample.mean()
        else:
            # 标准高斯先验: 解析解
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            if free_bits > 0:
                kl_per_dim = F.relu(kl_per_dim - free_bits) + free_bits
            return kl_per_dim.sum(dim=-1).mean()
    
    def _gaussian_log_prob(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """计算高斯分布的对数概率."""
        var = torch.exp(logvar)
        log_prob = -0.5 * (
            torch.log(2 * torch.pi * var) + 
            (x - mu).pow(2) / var
        ).sum(dim=-1)
        return log_prob


# ============================================================================
#                         RealNVP 流模型先验 (可选)
# ============================================================================

class AffineCoupling(nn.Module):
    """
    仿射耦合层 (RealNVP 的基本单元).
    
    将输入分成两半，一半不变，另一半做仿射变换。
    变换参数由不变的一半通过神经网络计算。
    """
    
    def __init__(self, dim: int, hidden_dim: int = 64, mask_type: str = 'even'):
        super().__init__()
        self.dim = dim
        self.mask_type = mask_type
        
        # 创建掩码
        if mask_type == 'even':
            self.register_buffer('mask', torch.arange(dim) % 2 == 0)
        else:
            self.register_buffer('mask', torch.arange(dim) % 2 == 1)
        
        # 缩放和平移网络
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim * 2),  # 输出 scale 和 shift
        )
        
        # 初始化为恒等变换
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        正向变换: z -> e
        
        Returns:
            e: 变换后的值
            log_det: 对数行列式
        """
        x_masked = x * self.mask.float()
        
        params = self.net(x_masked)
        scale, shift = params.chunk(2, dim=-1)
        scale = torch.tanh(scale) * 2  # 限制 scale 范围
        
        # 仿射变换 (只对 ~mask 的维度)
        e = x_masked + (~self.mask).float() * (x * torch.exp(scale) + shift)
        
        # 对数行列式
        log_det = (scale * (~self.mask).float()).sum(dim=-1)
        
        return e, log_det
    
    def inverse(self, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        逆向变换: e -> z
        """
        e_masked = e * self.mask.float()
        
        params = self.net(e_masked)
        scale, shift = params.chunk(2, dim=-1)
        scale = torch.tanh(scale) * 2
        
        # 逆仿射变换
        x = e_masked + (~self.mask).float() * ((e - shift) * torch.exp(-scale))
        
        # 逆变换的对数行列式是负的
        log_det = -(scale * (~self.mask).float()).sum(dim=-1)
        
        return x, log_det


class RealNVPPrior(nn.Module):
    """
    RealNVP 流模型先验.
    
    将标准高斯通过可逆变换，得到可学习的复杂先验分布。
    
    优势：
    1. 先验可以是多峰的
    2. 先验形状自适应后验分布
    3. 支持精确的密度计算
    """
    
    def __init__(self, dim: int, n_blocks: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        
        # 交替使用 even/odd 掩码
        self.blocks = nn.ModuleList([
            AffineCoupling(dim, hidden_dim, mask_type='even' if i % 2 == 0 else 'odd')
            for i in range(n_blocks)
        ])
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z ~ N(0, I) -> e ~ p_φ(e)
        
        Returns:
            e: 变换后的样本
            log_det: 对数行列式 (用于密度计算)
        """
        e = z
        log_det = 0
        
        for block in self.blocks:
            e, ld = block.forward(e)
            log_det = log_det + ld
        
        return e, log_det
    
    def inverse(self, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        e ~ p_φ(e) -> z ~ N(0, I)
        """
        z = e
        log_det = 0
        
        for block in reversed(self.blocks):
            z, ld = block.inverse(z)
            log_det = log_det + ld
        
        return z, log_det
    
    def log_prob(self, e: torch.Tensor) -> torch.Tensor:
        """
        计算 log p_φ(e).
        
        使用变量替换公式:
        log p_φ(e) = log p(z) + log |det ∂z/∂e|
                   = log p(z) + log_det_jacobian
        """
        z, log_det = self.inverse(e)
        
        # 标准高斯的对数概率
        log_pz = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(dim=-1)
        
        return log_pz + log_det
    
    def sample(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """
        从先验采样.
        """
        if device is None:
            device = next(self.parameters()).device
        
        z = torch.randn(batch_size, self.dim, device=device)
        e, _ = self.forward(z)
        return e


# ============================================================================
#                              工具函数
# ============================================================================

def create_encoder_v2(
    seq_len: int = 96,
    input_dim: int = 1,
    cond_dim: int = 2,
    hidden_dim: int = 64,
    env_dim: int = 4,
    use_flow_prior: bool = False,
    flow_blocks: int = 4,
) -> EnvironmentEncoderV2:
    """
    工厂函数：创建 V2 编码器.
    
    Args:
        use_flow_prior: 是否使用流模型先验
        flow_blocks: 流模型的耦合层数量
    """
    encoder = EnvironmentEncoderV2(
        seq_len=seq_len,
        input_dim=input_dim,
        cond_dim=cond_dim,
        hidden_dim=hidden_dim,
        env_dim=env_dim,
    )
    
    if use_flow_prior:
        encoder.flow_prior = RealNVPPrior(
            dim=env_dim,
            n_blocks=flow_blocks,
            hidden_dim=hidden_dim,
        )
    
    return encoder


# ============================================================================
#                              测试代码
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  Environment Encoder V2 Test")
    print("=" * 60)
    
    # 创建编码器
    encoder = create_encoder_v2(
        seq_len=96,
        input_dim=1,
        cond_dim=2,
        hidden_dim=64,
        env_dim=4,
        use_flow_prior=True,
    )
    
    # 打印参数量
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters: {n_params:,}")
    
    # 测试前向传播
    B, T = 32, 96
    x = torch.randn(B, T, 1)
    c = torch.randn(B, T, 2)
    
    output = encoder(x, c, num_samples=1)
    print(f"\nInput x: {x.shape}")
    print(f"Input c: {c.shape}")
    print(f"Output e: {output['e'].shape}")
    print(f"Output μ: {output['mu'].shape}")
    print(f"Output logvar: {output['logvar'].shape}")
    
    # 测试中间特征
    encoded = encoder.encode(x, c, return_intermediates=True)
    print(f"\nIntermediate features:")
    print(f"  h_stat: {encoded['h_stat'].shape}")
    print(f"  h_attn: {encoded['h_attn'].shape}")
    print(f"  h_spec: {encoded['h_spec'].shape}")
    print(f"  h_fused: {encoded['h_fused'].shape}")
    
    # 测试 KL 散度
    kl_gaussian = encoder.compute_kl_divergence(
        output['mu'], output['logvar']
    )
    print(f"\nKL (Gaussian prior): {kl_gaussian.item():.4f}")
    
    kl_flow = encoder.compute_kl_divergence(
        output['mu'], output['logvar'], output['e']
    )
    print(f"KL (Flow prior): {kl_flow.item():.4f}")
    
    # 测试流先验采样
    if encoder.flow_prior is not None:
        samples = encoder.flow_prior.sample(100)
        print(f"\nFlow prior samples: {samples.shape}")
        print(f"  mean: {samples.mean().item():.4f}")
        print(f"  std: {samples.std().item():.4f}")
    
    print("\n✅ All tests passed!")

