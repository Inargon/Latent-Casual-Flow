"""
Velocity Network - Refactored
=============================
Conditional Velocity Network v_θ(x, t, c, e) for Flow Matching.

Key Features:
- Predicts velocity v = dx/dt for flow matching ODE
- Uses AdaLN to inject time t and environment e at every layer
- Transformer-based architecture for temporal dependencies
- Clean interface with consistent (B, T, D) format

Refactored Changes:
- Unified data format handling
- Cleaner AdaLN implementation
- Simplified architecture
- Removed unused code
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        t: (B,) timestep values in [0, 1]
        dim: Embedding dimension
        
    Returns:
        (B, dim) embeddings
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
    emb = t.unsqueeze(-1).float() * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization.
    
    Modulates normalized features: h' = γ(cond) * LayerNorm(h) + β(cond)
    """
    
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 2),
        )
        # Initialize to identity
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) or (B, D) features
            cond: (B, cond_dim) conditioning
        """
        scale, shift = self.modulation(cond).chunk(2, dim=-1)
        
        if x.dim() == 3:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        
        return self.norm(x) * (1 + scale) + shift


class TransformerBlock(nn.Module):
    """Transformer block with AdaLN conditioning."""
    
    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.adaln1 = AdaLN(hidden_dim, cond_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, 
            dropout=dropout, 
            batch_first=True,
        )
        
        self.adaln2 = AdaLN(hidden_dim, cond_dim)
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) features
            cond: (B, cond_dim) conditioning
        """
        # Self-attention with AdaLN
        h = self.adaln1(x, cond)
        h, _ = self.attn(h, h, h)
        x = x + h
        
        # MLP with AdaLN
        h = self.adaln2(x, cond)
        x = x + self.mlp(h)
        
        return x


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


class VelocityNetwork(nn.Module):
    """
    Conditional Velocity Network v_θ(x_t, t, c, e)
    
    Predicts the velocity field for flow matching ODE.
    Uses Transformer architecture with AdaLN for conditioning.
    
    核心三机制（强化 e 的使用）：
    1. 直接注入：e 直接拼接到输入 [x, c, e]，让 e 从第一层就参与特征学习
    2. AdaLN 调制：e 通过 AdaLN 调制每层特征分布，控制每层的"风格"
    3. 强条件 Dropout：训练时随机 mask c，强迫模型必须使用 e
    
    Args:
        seq_len: Sequence length
        input_dim: Input channel dimension
        cond_dim: Condition dimension (不包含位置编码，位置编码会自动添加)
        env_dim: Environment dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        cond_dropout: Dropout rate for condition c (to force using e)
        full_cond_mask_prob: Probability to fully mask c (stronger than dropout)
        direct_env_inject: Whether to directly concatenate e to input
        add_positional_encoding: Whether to add CaTSG-style positional encoding
    """
    
    def __init__(
        self,
        seq_len: int = 96,
        input_dim: int = 1,
        cond_dim: int = 4,
        env_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_dropout: float = 0.2,  # 条件 dropout
        full_cond_mask_prob: float = 0.15,  # 完全 mask c 的概率
        direct_env_inject: bool = True,  # 直接注入 e
        add_positional_encoding: bool = False,  # CaTSG 风格位置编码（默认关闭）
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.env_dim = env_dim
        self.hidden_dim = hidden_dim
        self.cond_dropout = cond_dropout
        self.full_cond_mask_prob = full_cond_mask_prob
        self.direct_env_inject = direct_env_inject
        self.add_positional_encoding = add_positional_encoding
        
        # 实际的条件维度（包含位置编码）
        actual_cond_dim = cond_dim + 2 if add_positional_encoding else cond_dim
        self.actual_cond_dim = actual_cond_dim
        
        # Time embedding
        time_dim = hidden_dim
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Environment projection (for AdaLN)
        self.env_proj = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Condition dimension for AdaLN: time + environment
        adaln_cond_dim = time_dim + hidden_dim
        
        # Input projection: x + c (with pos_enc) + e (optional) -> hidden
        input_proj_dim = input_dim + actual_cond_dim
        if direct_env_inject:
            input_proj_dim += env_dim  # e 也参与输入
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        # Positional embedding
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                cond_dim=adaln_cond_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_norm = AdaLN(hidden_dim, adaln_cond_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        # Initialize output to zero for stable training
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def _to_btd(self, x: torch.Tensor, expected_d: int) -> torch.Tensor:
        """Convert to (B, T, D) format."""
        # Handle 4D tensor (B, T, D, 1) or (B, T, 1, D) -> squeeze to 3D
        while x.dim() > 3:
            if x.shape[-1] == 1:
                x = x.squeeze(-1)
            else:
                x = x.squeeze(-2)
        
        if x.dim() == 3:
            if x.shape[-1] == expected_d:
                return x
            elif x.shape[1] == expected_d:
                return x.transpose(1, 2)
        elif x.dim() == 2:
            # (B, T) -> (B, T, 1)
            x = x.unsqueeze(-1)
        return x
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        e: torch.Tensor,
        return_hidden: bool = False,
    ):
        """
        Predict velocity v = dx/dt.
        
        核心三机制：
        1. 直接注入：e 扩展到序列并与 [x, c] 拼接
        2. AdaLN 调制：e 通过 AdaLN 调制每层
        3. 强 c mask：训练时有概率完全 mask c
        
        Args:
            x_t: Current state - (B, T, D) or (B, D, T)
            t: Time in [0, 1] - (B,)
            c: Condition - (B, T, D_c) or (B, D_c, T)
            e: Environment - (B, env_dim)
            return_hidden: If True, also return hidden states (B, T, hidden_dim)
            
        Returns:
            v: Predicted velocity - same shape as x_t
            (optional) h: Hidden states before output projection - (B, T, hidden_dim)
        """
        # Convert to (B, T, D) format
        input_transposed = False
        if x_t.dim() == 3 and x_t.shape[1] == self.input_dim and x_t.shape[2] == self.seq_len:
            x_t = x_t.transpose(1, 2)
            input_transposed = True
        
        x_t = self._to_btd(x_t, self.input_dim)
        c = self._to_btd(c, self.cond_dim)
        
        B, T, _ = x_t.shape
        
        # 添加 CaTSG 风格位置编码
        if self.add_positional_encoding:
            c = add_positional_encoding_catsg(c)  # (B, T, cond_dim + 2)
        
        # ============ 强条件 Dropout (训练时，向量化实现) ============
        if self.training:
            # 完全 mask c（强迫必须用 e）
            full_mask = (torch.rand(B, 1, 1, device=c.device) < self.full_cond_mask_prob).float()
            # 部分 dropout
            partial_mask = (torch.rand(B, 1, 1, device=c.device) < self.cond_dropout).float()
            # 应用 mask: full_mask -> c=0, partial_mask -> c*0.5, else -> c
            c = c * (1 - full_mask) * (1 - 0.5 * partial_mask * (1 - full_mask))
        
        # ============ Build Conditioning for AdaLN ============
        # Time embedding
        t_emb = sinusoidal_embedding(t, self.hidden_dim)
        t_emb = self.time_embed(t_emb)
        
        # Environment embedding (for AdaLN)
        e_emb = self.env_proj(e)
        
        # Combined conditioning for AdaLN
        cond = torch.cat([t_emb, e_emb], dim=-1)
        
        # ============ Input Processing (多路径注入) ============
        if self.direct_env_inject:
            # 路径 1: e 直接参与输入
            e_seq = e.unsqueeze(1).expand(B, T, -1)  # (B, T, env_dim)
            xce = torch.cat([x_t, c, e_seq], dim=-1)  # [x, c, e] 全部拼接
            h = self.input_proj(xce)
        else:
            # 原始方式：只有 [x, c]
            xc = torch.cat([x_t, c], dim=-1)
            h = self.input_proj(xc)
        
        # Add positional embedding
        h = h + self.pos_emb[:, :T, :]
        
        # ============ Transformer Blocks (机制2: AdaLN 调制) ============
        for block in self.blocks:
            h = block(h, cond)
        
        # ============ Output ============
        h = self.output_norm(h, cond)
        v = self.output_proj(h)
        
        # Convert back to original format if needed
        if input_transposed:
            v = v.transpose(1, 2)
        
        if return_hidden:
            return v, h  # h: (B, T, hidden_dim) — Transformer 隐状态
        return v


# ============ Backward Compatibility Aliases ============

class ConditionalVelocityNet(VelocityNetwork):
    """Alias for backward compatibility."""
    
    def __init__(
        self,
        seq_len: int = 96,
        in_channels: int = 1,
        out_channels: int = 1,
        model_channels: int = 64,
        env_dim: int = 32,
        cond_dim: int = 64,
        num_transformer_blocks: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(
            seq_len=seq_len,
            input_dim=in_channels,
            cond_dim=cond_dim,
            env_dim=env_dim,
            hidden_dim=model_channels * 4,
            num_layers=num_transformer_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )
    
    def forward_mc(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        e_samples: torch.Tensor,
    ) -> torch.Tensor:
        """
        Monte Carlo forward: average velocity over multiple environment samples.
        
        Args:
            x_t: (B, T, D) or (B, D, T)
            t: (B,)
            c: (B, T, D_c) or (B, D_c, T)
            e_samples: (B, N, env_dim)
            
        Returns:
            v_avg: (B, T, D) or (B, D, T)
        """
        B, N, D_e = e_samples.shape
        
        # Expand for MC samples
        x_t_exp = x_t.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, *x_t.shape[1:])
        t_exp = t.unsqueeze(1).expand(-1, N).reshape(B * N)
        c_exp = c.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, *c.shape[1:])
        e_flat = e_samples.reshape(B * N, D_e)
        
        # Forward
        v_flat = self.forward(x_t_exp, t_exp, c_exp, e_flat)
        
        # Average
        v_samples = v_flat.reshape(B, N, *v_flat.shape[1:])
        return v_samples.mean(dim=1)


class VectorVelocityNet(nn.Module):
    """
    Velocity network for vector (non-sequence) data.
    Used for toy experiments.
    """
    
    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        env_dim: int = 4,
        hidden_dim: int = 128,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.env_dim = env_dim
        self.hidden_dim = hidden_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Condition + environment embedding
        self.cond_embed = nn.Sequential(
            nn.Linear(c_dim + env_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Input projection
        self.input_proj = nn.Linear(x_dim, hidden_dim)
        
        # MLP blocks with AdaLN
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(VectorAdaLNBlock(hidden_dim, hidden_dim, dropout))
        
        # Output
        self.output_proj = nn.Linear(hidden_dim, x_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        e: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t: (B, x_dim)
            t: (B,)
            c: (B, c_dim)
            e: (B, env_dim)
        """
        # Time embedding
        t_emb = sinusoidal_embedding(t, self.hidden_dim)
        t_emb = self.time_embed(t_emb)
        
        # Condition embedding
        ce = torch.cat([c, e], dim=-1)
        ce_emb = self.cond_embed(ce)
        
        # Combined conditioning
        cond = t_emb + ce_emb
        
        # Input
        h = self.input_proj(x_t)
        
        # Blocks
        for block in self.blocks:
            h = block(h, cond)
        
        return self.output_proj(h)
    
    def forward_mc(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        e_samples: torch.Tensor,
    ) -> torch.Tensor:
        """MC averaged velocity."""
        B, N, D_e = e_samples.shape
        
        x_t_exp = x_t.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        t_exp = t.unsqueeze(1).expand(-1, N).reshape(B * N)
        c_exp = c.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        e_flat = e_samples.reshape(B * N, -1)
        
        v_flat = self.forward(x_t_exp, t_exp, c_exp, e_flat)
        v_samples = v_flat.reshape(B, N, -1)
        
        return v_samples.mean(dim=1)


class VectorAdaLNBlock(nn.Module):
    """AdaLN block for vector data."""
    
    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 3),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        params = self.modulation(cond)
        scale, shift, gate = params.chunk(3, dim=-1)
        
        h = self.norm(x) * (1 + scale) + shift
        h = self.mlp(h)
        
        return x + gate * h


# ============ Factory Functions ============

def create_velocity_net_with_cpd(
    seq_len: int = 96,
    input_dim: int = 1,
    cond_dim: int = 4,
    env_dim: int = 16,
    hidden_dim: int = 128,
    num_layers: int = 6,
    num_heads: int = 4,
    dropout: float = 0.1,
    cond_dropout: float = 0.2,
    full_cond_mask_prob: float = 0.15,
    direct_env_inject: bool = True,
    add_positional_encoding: bool = False,
):
    """
    创建带 CPD (Causal Pathway Decomposition) 包装的 VelocityNetwork
    
    CPD 在基础速度网络之上添加因果路径分解：
    - v = v_base + α·Δv_c + β·Δv_e + γ·Δv_int
    - 其中 Δv_c (C→X), Δv_e (E→X), Δv_int (C×E→X，双线性交互)
    
    Args:
        seq_len: 序列长度
        input_dim: 输入维度
        cond_dim: 条件维度
        env_dim: 环境维度
        hidden_dim: 隐藏维度
        num_layers: Transformer 层数
        num_heads: 注意力头数
        dropout: Dropout 率
        cond_dropout: 条件 dropout 率
        full_cond_mask_prob: 完全 mask c 的概率
        direct_env_inject: 是否直接注入 e 到输入
        add_positional_encoding: 是否添加位置编码
        
    Returns:
        CausalPathwayDecomposition: 包装后的网络，接口与 VelocityNetwork 相同
    """
    from lcf.modules.causal_attention_plugin import wrap_with_cpd
    
    # 创建基础网络
    base_net = VelocityNetwork(
        seq_len=seq_len,
        input_dim=input_dim,
        cond_dim=cond_dim,
        env_dim=env_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        cond_dropout=cond_dropout,
        full_cond_mask_prob=full_cond_mask_prob,
        direct_env_inject=direct_env_inject,
        add_positional_encoding=add_positional_encoding,
    )
    
    # 用 CPD 包装
    return wrap_with_cpd(
        base_net,
        hidden_dim=hidden_dim,
        env_dim=env_dim,
        cond_dim=cond_dim,
        output_dim=input_dim,
    )
