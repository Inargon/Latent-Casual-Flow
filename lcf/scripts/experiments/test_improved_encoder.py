"""
改进的编码器训练 - 解决 μ 多样性不足问题

核心改进：
1. 条件感知对比学习 - 相似 c 的样本 μ 应该接近，不同 c 的样本 μ 应该远离
2. 增强的方差损失 - 强制 μ 每个维度有足够方差
3. 分量感知训练 - 在 GMM 分配上施加更强的均衡约束
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
import argparse
import os
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, '/root/autodl-tmp/lcf')

from lcf.modules.env_encoder_v2 import EnvironmentEncoderV2
from lcf.modules.velocity_net import VelocityNetwork
from lcf.data.traffic import get_traffic_dataloaders


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============== 改进的 GMM 先验 ==============

class ImprovedGMMPrior(nn.Module):
    """
    改进的 GMM 先验
    - 更强的分离约束
    - 自适应温度
    """
    
    def __init__(self, dim: int, n_components: int = 4, init_temp: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_components = n_components
        
        # 原型初始化更分散
        init_prototypes = torch.randn(n_components, dim)
        # 正交化 - 使用 QR 分解确保形状正确
        if dim >= n_components:
            # 转置后 QR 分解，再转回来
            q, r = torch.linalg.qr(init_prototypes.T)  # (dim, n_components)
            init_prototypes = q.T[:n_components] * 2.0  # (n_components, dim)
        
        self.prototypes = nn.Parameter(init_prototypes)
        self.log_stds = nn.Parameter(torch.zeros(n_components))
        self.log_temperature = nn.Parameter(torch.tensor(np.log(init_temp)))
    
    @property
    def temperature(self):
        return torch.exp(self.log_temperature).clamp(min=0.01, max=1.0)
    
    @property
    def stds(self):
        return torch.exp(self.log_stds).clamp(min=0.1, max=2.0)
    
    def compute_assignment(self, e: torch.Tensor) -> torch.Tensor:
        """计算软分配"""
        dist = ((e.unsqueeze(1) - self.prototypes.unsqueeze(0)) ** 2).sum(dim=-1)
        logits = -dist / (2 * self.temperature ** 2 + 1e-6)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        return F.softmax(logits, dim=-1)
    
    def sample(self, n_samples: int, device=None) -> torch.Tensor:
        if device is None:
            device = self.prototypes.device
        k = torch.randint(0, self.n_components, (n_samples,), device=device)
        means = self.prototypes[k]
        stds = self.stds[k].unsqueeze(-1).expand(-1, self.dim)
        return means + stds * torch.randn_like(means)
    
    def balanced_loss(self, assignment: torch.Tensor) -> torch.Tensor:
        """均衡损失 - 鼓励所有分量被使用"""
        usage = assignment.mean(dim=0).clamp(min=1e-6)
        # 最大化熵 = 均匀使用
        entropy = -(usage * torch.log(usage)).sum()
        max_entropy = np.log(self.n_components)
        return (max_entropy - entropy).clamp(min=0)
    
    def separation_loss(self) -> torch.Tensor:
        """分离损失 - 鼓励原型远离"""
        K = self.n_components
        diff = self.prototypes.unsqueeze(0) - self.prototypes.unsqueeze(1)
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
        mask = ~torch.eye(K, dtype=torch.bool, device=self.prototypes.device)
        # 希望最小距离至少为 2.0
        return F.relu(2.0 - dist[mask]).mean()
    
    def update_prototypes_ema(self, mu: torch.Tensor, momentum: float = 0.1):
        """EMA 更新原型位置，让原型跟随 μ 分布"""
        with torch.no_grad():
            assignment = self.compute_assignment(mu.detach())  # (B, K)
            
            for k in range(self.n_components):
                weights = assignment[:, k]  # (B,)
                if weights.sum() > 1e-6:
                    # 加权平均
                    new_center = (weights.unsqueeze(-1) * mu.detach()).sum(dim=0) / weights.sum()
                    # EMA 更新
                    self.prototypes.data[k] = (1 - momentum) * self.prototypes.data[k] + momentum * new_center


# ============== 条件 GMM 先验 ==============

class ConditionalGMMPrior(nn.Module):
    """
    条件 GMM 先验 p(e|c)
    
    原型位置依赖于条件 c，使得先验更好地匹配后验分布
    """
    
    def __init__(self, dim: int, cond_dim: int, n_components: int = 4, init_temp: float = 0.1):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.n_components = n_components
        
        # 基础原型（可学习）
        init_prototypes = torch.randn(n_components, dim) * 0.5
        self.base_prototypes = nn.Parameter(init_prototypes)
        
        # 条件投影网络：c -> 原型偏移
        self.cond_net = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_components * dim),
        )
        
        # 初始化最后一层为小值，让初始偏移接近 0
        nn.init.zeros_(self.cond_net[-1].weight)
        nn.init.zeros_(self.cond_net[-1].bias)
        
        self.log_stds = nn.Parameter(torch.zeros(n_components))
        self.log_temperature = nn.Parameter(torch.tensor(np.log(init_temp)))
    
    @property
    def temperature(self):
        return torch.exp(self.log_temperature).clamp(min=0.01, max=1.0)
    
    @property
    def stds(self):
        return torch.exp(self.log_stds).clamp(min=0.1, max=2.0)
    
    def get_prototypes(self, c: torch.Tensor) -> torch.Tensor:
        """
        根据条件 c 计算原型位置
        
        Args:
            c: (B, T, C) 或 (B, C) 条件
        Returns:
            prototypes: (B, K, dim) 每个样本的原型位置
        """
        if c.dim() == 3:
            c_pooled = c.mean(dim=1)  # (B, C)
        else:
            c_pooled = c
        
        # 计算条件依赖的偏移
        offset = self.cond_net(c_pooled)  # (B, K*dim)
        offset = offset.reshape(-1, self.n_components, self.dim)  # (B, K, dim)
        
        # 原型 = 基础原型 + 条件偏移
        prototypes = self.base_prototypes.unsqueeze(0) + offset  # (B, K, dim)
        return prototypes
    
    def compute_assignment(self, e: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """计算软分配（条件依赖）"""
        prototypes = self.get_prototypes(c)  # (B, K, dim)
        # e: (B, dim) -> (B, 1, dim)
        dist = ((e.unsqueeze(1) - prototypes) ** 2).sum(dim=-1)  # (B, K)
        logits = -dist / (2 * self.temperature ** 2 + 1e-6)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        return F.softmax(logits, dim=-1)
    
    def sample(self, c: torch.Tensor) -> torch.Tensor:
        """根据条件 c 从 GMM 采样"""
        B = c.shape[0]
        device = c.device
        
        prototypes = self.get_prototypes(c)  # (B, K, dim)
        
        # 随机选择分量
        k = torch.randint(0, self.n_components, (B,), device=device)
        
        # 获取选中分量的均值
        means = prototypes[torch.arange(B, device=device), k]  # (B, dim)
        
        # 添加噪声
        stds = self.stds[k].unsqueeze(-1).expand(-1, self.dim)  # (B, dim)
        return means + stds * torch.randn_like(means)
    
    def balanced_loss(self, assignment: torch.Tensor) -> torch.Tensor:
        """均衡损失"""
        usage = assignment.mean(dim=0).clamp(min=1e-6)
        entropy = -(usage * torch.log(usage)).sum()
        max_entropy = np.log(self.n_components)
        return (max_entropy - entropy).clamp(min=0)
    
    def separation_loss(self) -> torch.Tensor:
        """基础原型分离损失"""
        K = self.n_components
        diff = self.base_prototypes.unsqueeze(0) - self.base_prototypes.unsqueeze(1)
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
        mask = ~torch.eye(K, dtype=torch.bool, device=self.base_prototypes.device)
        return F.relu(2.0 - dist[mask]).mean()
    
    def update_prototypes_ema(self, mu: torch.Tensor, c: torch.Tensor, momentum: float = 0.1):
        """EMA 更新基础原型"""
        with torch.no_grad():
            assignment = self.compute_assignment(mu.detach(), c)
            for k in range(self.n_components):
                weights = assignment[:, k]
                if weights.sum() > 1e-6:
                    new_center = (weights.unsqueeze(-1) * mu.detach()).sum(dim=0) / weights.sum()
                    self.base_prototypes.data[k] = (1 - momentum) * self.base_prototypes.data[k] + momentum * new_center


# ============== 条件感知对比学习 ==============

class ConditionAwareContrastiveLoss(nn.Module):
    """
    条件感知对比学习
    
    核心思想：
    - 相似条件 c 的样本应该有相似的 μ
    - 不同条件 c 的样本应该有不同的 μ
    """
    
    def __init__(self, temperature: float = 0.1, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, mu: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mu: (B, D) 编码器输出的均值
            c: (B, T, C) 或 (B, C) 条件
        """
        B = mu.shape[0]
        if B < 4:
            return torch.tensor(0.0, device=mu.device)
        
        # 计算条件相似度
        if c.dim() == 3:
            c_flat = c.mean(dim=1)  # (B, C)
        else:
            c_flat = c
        
        c_norm = F.normalize(c_flat, dim=-1)
        c_sim = torch.mm(c_norm, c_norm.t())  # (B, B)
        
        # 计算 μ 相似度
        mu_norm = F.normalize(mu, dim=-1)
        mu_sim = torch.mm(mu_norm, mu_norm.t()) / self.temperature  # (B, B)
        
        # 软标签：条件相似的应该 μ 也相似
        # 使用条件相似度作为软正样本权重
        pos_weights = F.softmax(c_sim / self.temperature, dim=-1)
        
        # InfoNCE 风格损失
        log_prob = F.log_softmax(mu_sim, dim=-1)
        loss = -(pos_weights * log_prob).sum(dim=-1).mean()
        
        return loss


# ============== 增强的方差损失 ==============

def enhanced_variance_loss(mu: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    """
    增强的方差损失
    
    强制 μ 的每个维度都有足够的方差
    """
    # 每个维度的标准差
    stds = mu.std(dim=0)
    
    # 惩罚标准差太小的维度
    loss = F.relu(target_std - stds).mean()
    
    # 额外：惩罚整体标准差太小
    overall_std = mu.std()
    loss = loss + F.relu(target_std - overall_std)
    
    return loss


# ============== 协方差正则化 ==============

def covariance_regularization(mu: torch.Tensor) -> torch.Tensor:
    """
    协方差正则化（VICReg 风格）
    
    鼓励不同维度之间独立
    """
    B, D = mu.shape
    mu_centered = mu - mu.mean(dim=0, keepdim=True)
    cov = torch.mm(mu_centered.t(), mu_centered) / (B - 1)
    
    # 非对角元素应该接近 0
    off_diag = cov - torch.diag(torch.diag(cov))
    loss = (off_diag ** 2).sum() / D
    
    return loss


# ============== 完整模型 ==============

class ImprovedLCF(nn.Module):
    """改进的 LCF 模型"""
    
    def __init__(self, encoder, velocity_net, gmm_prior, use_conditional_gmm=False):
        super().__init__()
        self.encoder = encoder
        self.velocity_net = velocity_net
        self.gmm_prior = gmm_prior
        self.use_conditional_gmm = use_conditional_gmm
        
        # 条件对比学习
        self.contrastive_loss = ConditionAwareContrastiveLoss()
    
    def training_step(self, x, c, kl_weight=0.001, var_weight=1.0, cov_weight=0.5,
                      contrastive_weight=0.5, balance_weight=1.0, sep_weight=0.5):
        """
        训练步骤
        """
        B, T = x.shape[:2]
        device = x.device
        
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # 编码
        enc_out = self.encoder(x, c)
        mu = enc_out['mu']
        logvar = enc_out['logvar']
        
        # 重参数化
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        e = mu + std * eps
        
        # Flow Matching 损失
        t = torch.rand(B, device=device)
        noise = torch.randn_like(x)
        x_t = t.view(-1, 1, 1) * x + (1 - t.view(-1, 1, 1)) * noise
        v_target = x - noise
        v_pred = self.velocity_net(x_t, t, c, e)
        fm_loss = F.mse_loss(v_pred, v_target)
        
        # KL 损失
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        
        # 增强的方差损失
        var_loss = enhanced_variance_loss(mu, target_std=1.0)
        
        # 协方差正则化
        cov_loss = covariance_regularization(mu)
        
        # 条件感知对比损失
        contr_loss = self.contrastive_loss(mu, c)
        
        # GMM 均衡损失
        if self.use_conditional_gmm:
            assignment = self.gmm_prior.compute_assignment(e.detach(), c)
        else:
            assignment = self.gmm_prior.compute_assignment(e.detach())
        balance_loss = self.gmm_prior.balanced_loss(assignment)
        sep_loss = self.gmm_prior.separation_loss()
        
        # 总损失
        total_loss = (fm_loss + 
                      kl_weight * kl_loss +
                      var_weight * var_loss +
                      cov_weight * cov_loss +
                      contrastive_weight * contr_loss +
                      balance_weight * balance_loss +
                      sep_weight * sep_loss)
        
        return {
            'total': total_loss,
            'fm': fm_loss,
            'kl': kl_loss,
            'var': var_loss,
            'cov': cov_loss,
            'contrastive': contr_loss,
            'balance': balance_loss,
            'separation': sep_loss,
            'mu_std': mu.std().item(),
            'mu_std_per_dim': mu.std(dim=0).mean().item(),
            'mu': mu.detach(),  # 返回 mu 用于原型更新
        }
    
    def generate_from_prior(self, c, n_steps=50, dynamic_e=False, update_interval=5):
        """
        从 GMM 先验采样生成
        
        Args:
            c: 条件
            n_steps: ODE 步数
            dynamic_e: 是否动态更新 e
            update_interval: 更新 e 的间隔步数
        """
        B, T = c.shape[0], c.shape[1]
        D = 1
        device = c.device
        
        # 初始 e 从先验采样
        if self.use_conditional_gmm:
            e = self.gmm_prior.sample(c)  # 条件 GMM：根据 c 采样
        else:
            e = self.gmm_prior.sample(B, device=device)  # 普通 GMM
        
        x = torch.randn(B, T, D, device=device)
        dt = 1.0 / n_steps
        
        for i in range(n_steps):
            t_val = i * dt
            t = torch.full((B,), t_val, device=device)
            
            # 动态更新 e：根据当前 x_t 重新推断
            if dynamic_e and i > 0 and i % update_interval == 0:
                with torch.no_grad():
                    enc_out = self.encoder(x, c)
                    # 使用 Monte Carlo 采样
                    e = enc_out['mu'] + torch.exp(0.5 * enc_out['logvar']) * torch.randn_like(enc_out['mu'])
            
            v = self.velocity_net(x, t, c, e)
            x = x + v * dt
        
        return x, e
    
    def generate_from_posterior(self, x_real, c, n_steps=50):
        """从后验采样生成"""
        B, T = c.shape[0], c.shape[1]
        D = 1
        device = c.device
        
        enc_out = self.encoder(x_real, c)
        e = enc_out['mu'] + torch.exp(0.5 * enc_out['logvar']) * torch.randn_like(enc_out['mu'])
        
        x = torch.randn(B, T, D, device=device)
        dt = 1.0 / n_steps
        
        for i in range(n_steps):
            t_val = i * dt
            t = torch.full((B,), t_val, device=device)
            v = self.velocity_net(x, t, c, e)
            x = x + v * dt
        
        return x, e


# ============== 评估指标 ==============

def compute_mmd(x_real, x_gen, sigma=1.0):
    x_real = x_real.reshape(x_real.shape[0], -1)
    x_gen = x_gen.reshape(x_gen.shape[0], -1)
    
    def rbf_kernel(x, y, sigma):
        dist = torch.cdist(x, y, p=2)
        return torch.exp(-dist ** 2 / (2 * sigma ** 2))
    
    x_real_t = torch.FloatTensor(x_real)
    x_gen_t = torch.FloatTensor(x_gen)
    
    k_xx = rbf_kernel(x_real_t, x_real_t, sigma).mean()
    k_yy = rbf_kernel(x_gen_t, x_gen_t, sigma).mean()
    k_xy = rbf_kernel(x_real_t, x_gen_t, sigma).mean()
    
    return (k_xx + k_yy - 2 * k_xy).item()


def compute_ks_statistic(x_real, x_gen):
    x_real_flat = x_real.flatten()
    x_gen_flat = x_gen.flatten()
    ks_stat, _ = sp_stats.ks_2samp(x_real_flat, x_gen_flat)
    return ks_stat


def compute_variance_ratio(x_real, x_gen):
    return x_gen.var() / (x_real.var() + 1e-8)


def compute_flat_kl(x_real, x_gen, n_bins=50):
    """Flattened KL Divergence"""
    x_real_flat = x_real.flatten()
    x_gen_flat = x_gen.flatten()
    
    min_val = min(x_real_flat.min(), x_gen_flat.min())
    max_val = max(x_real_flat.max(), x_gen_flat.max())
    
    bins = np.linspace(min_val, max_val, n_bins + 1)
    
    p_real, _ = np.histogram(x_real_flat, bins=bins, density=True)
    p_gen, _ = np.histogram(x_gen_flat, bins=bins, density=True)
    
    p_real = p_real + 1e-10
    p_gen = p_gen + 1e-10
    
    p_real = p_real / p_real.sum()
    p_gen = p_gen / p_gen.sum()
    
    kl = np.sum(p_real * np.log(p_real / p_gen))
    return kl


def compute_mdd(x_real, x_gen):
    """Marginal Distribution Distance (per timestep)"""
    T = x_real.shape[1]
    distances = []
    
    for t in range(T):
        real_t = x_real[:, t].flatten()
        gen_t = x_gen[:, t].flatten()
        
        ks_stat, _ = sp_stats.ks_2samp(real_t, gen_t)
        distances.append(ks_stat)
    
    return np.mean(distances)


# J-FTSD 相关的编码器
class XEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        B = x.shape[0]
        return self.net(x.reshape(B, -1))


class CEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, c):
        if c.dim() == 3:
            c = c.mean(dim=1)
        return self.net(c)


def compute_jftsd(x_real, c_real, x_gen, emb_dim=64, train_steps=200, device="cpu"):
    """J-FTSD (Joint Fréchet Time Series Distance)"""
    if not isinstance(x_real, torch.Tensor):
        x_real = torch.tensor(x_real, dtype=torch.float32).to(device)
    else:
        x_real = x_real.clone().detach().to(device).float()
        
    if not isinstance(x_gen, torch.Tensor):
        x_gen = torch.tensor(x_gen, dtype=torch.float32).to(device)
    else:
        x_gen = x_gen.clone().detach().to(device).float()
        
    if not isinstance(c_real, torch.Tensor):
        c_real = torch.tensor(c_real, dtype=torch.float32).to(device)
    else:
        c_real = c_real.clone().detach().to(device).float()

    B, L = x_real.shape[:2]
    D_x = 1 if x_real.dim() == 2 else x_real.shape[-1]
    D_c = c_real.shape[-1]
    
    if x_real.dim() == 2:
        x_real = x_real.unsqueeze(-1)
    if x_gen.dim() == 2:
        x_gen = x_gen.unsqueeze(-1)
    
    x_evl_encoder = XEncoder(in_dim=L * D_x, out_dim=emb_dim).to(device)
    c_evl_encoder = CEncoder(in_dim=D_c, out_dim=emb_dim).to(device)
    optimizer = torch.optim.Adam(
        list(x_evl_encoder.parameters()) + list(c_evl_encoder.parameters()), 
        lr=1e-3
    )

    # Train encoders
    for _ in range(train_steps):
        idx = torch.randperm(B)
        x = x_real[idx]
        c = c_real[idx]

        z_t = x_evl_encoder(x)
        z_m = c_evl_encoder(c)

        z_t = F.normalize(z_t, dim=-1)
        z_m = F.normalize(z_m, dim=-1)
        logits = (z_t @ z_m.T) / np.sqrt(emb_dim) 

        labels = torch.arange(B).to(device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Compute J-FTSD
    with torch.no_grad():
        x_real_rep = x_evl_encoder(x_real)
        c_real_rep = c_evl_encoder(c_real)
        z_real = torch.cat([x_real_rep, c_real_rep], dim=-1)
        mu_real = z_real.mean(0)
        sigma_real = torch.cov(z_real.T)

        x_gen_rep = x_evl_encoder(x_gen)
        z_gen_xc = torch.cat([x_gen_rep, c_real_rep], dim=-1)
        mu_gen = z_gen_xc.mean(0)
        sigma_gen = torch.cov(z_gen_xc.T)

        diff = mu_real - mu_gen
        
        # Fréchet distance
        sigma_prod = sigma_real @ sigma_gen
        eigenvalues = torch.linalg.eigvals(sigma_prod).real
        eigenvalues = torch.clamp(eigenvalues, min=0)
        trace_sqrt = torch.sqrt(eigenvalues).sum()
        
        jftsd = (diff @ diff + sigma_real.trace() + sigma_gen.trace() - 2 * trace_sqrt).item()
        
    return max(0, jftsd)


# ============== 主函数 ==============

def main(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(f'logs/lcf/traffic_improved/{timestamp}')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("\n=== 加载 Traffic 数据 ===")
    train_loader, val_loader, test_loader, stats = get_traffic_dataloaders(
        batch_size=256,
        seq_len=96
    )
    
    c_dim = stats['c_dim']
    x_dim = 1
    env_dim = args.env_dim
    
    print(f"条件维度: {c_dim}")
    print(f"环境维度: {env_dim}")
    print(f"GMM 分量数: {args.n_components}")
    
    # 创建模型
    encoder = EnvironmentEncoderV2(
        input_dim=x_dim,
        cond_dim=c_dim,
        env_dim=env_dim,
        hidden_dim=64,
    ).to(device)
    
    velocity_net = VelocityNetwork(
        input_dim=x_dim,
        cond_dim=c_dim,
        env_dim=env_dim,
        hidden_dim=128,
        num_layers=4,
    ).to(device)
    
    # 选择 GMM 先验类型
    if args.use_conditional_gmm:
        print(f"  [使用条件 GMM 先验] p(e|c)")
        gmm_prior = ConditionalGMMPrior(
            dim=env_dim,
            cond_dim=c_dim,
            n_components=args.n_components,
        ).to(device)
    else:
        gmm_prior = ImprovedGMMPrior(
            dim=env_dim,
            n_components=args.n_components,
        ).to(device)
    
    model = ImprovedLCF(encoder, velocity_net, gmm_prior, 
                        use_conditional_gmm=args.use_conditional_gmm).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练
    print(f"\n=== 训练 {args.epochs} epochs ===")
    print(f"改进点：")
    print(f"  - 条件感知对比学习 (weight={args.contrastive_weight})")
    print(f"  - 增强的方差损失 (weight={args.var_weight})")
    print(f"  - 协方差正则化 (weight={args.cov_weight})")
    print(f"  - GMM 均衡约束 (weight={args.balance_weight})")
    
    history = {
        'total': [], 'fm': [], 'kl': [], 'var': [], 'cov': [],
        'contrastive': [], 'balance': [], 'separation': [],
        'mu_std': [], 'mu_std_per_dim': [],
    }
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = {k: [] for k in history.keys()}
        
        for batch_idx, batch in enumerate(train_loader):
            x = batch['x'].to(device)
            c = batch['c'].to(device)
            
            optimizer.zero_grad()
            losses = model.training_step(
                x, c,
                kl_weight=args.kl_weight,
                var_weight=args.var_weight,
                cov_weight=args.cov_weight,
                contrastive_weight=args.contrastive_weight,
                balance_weight=args.balance_weight,
                sep_weight=args.sep_weight,
            )
            
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # EMA 更新原型位置（让原型跟随 μ 分布）
            if args.use_ema_prototype:
                if args.use_conditional_gmm:
                    model.gmm_prior.update_prototypes_ema(losses['mu'], c, momentum=0.1)
                else:
                    model.gmm_prior.update_prototypes_ema(losses['mu'], momentum=0.1)
            
            for k in epoch_losses.keys():
                if k in losses:
                    val = losses[k].item() if isinstance(losses[k], torch.Tensor) else losses[k]
                    epoch_losses[k].append(val)
        
        scheduler.step()
        
        # 记录
        for k in history.keys():
            if epoch_losses[k]:
                history[k].append(np.mean(epoch_losses[k]))
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | "
                  f"FM: {history['fm'][-1]:.4f} | "
                  f"KL: {history['kl'][-1]:.4f} | "
                  f"Var: {history['var'][-1]:.4f} | "
                  f"Contr: {history['contrastive'][-1]:.4f} | "
                  f"Balance: {history['balance'][-1]:.4f} | "
                  f"μ_std: {history['mu_std'][-1]:.4f}")
    
    # 评估
    print("\n=== 评估 ===")
    model.eval()
    
    # 收集测试数据
    x_real_list, c_list, mu_list = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            c = batch['c'].to(device)
            x_real_list.append(x.cpu().numpy())
            c_list.append(c.cpu().numpy())
            
            enc_out = model.encoder(x.unsqueeze(-1) if x.dim() == 2 else x, c)
            mu_list.append(enc_out['mu'].cpu().numpy())
    
    x_real = np.concatenate(x_real_list, axis=0)
    c_all = np.concatenate(c_list, axis=0)
    mu_all = np.concatenate(mu_list, axis=0)
    
    print(f"\n测试集样本数: {len(x_real)}")
    print(f"\nμ 统计:")
    print(f"  均值: {mu_all.mean(axis=0)[:4]}")
    print(f"  标准差: {mu_all.std(axis=0)[:4]}")
    print(f"  整体标准差: {mu_all.std():.4f}")
    
    # GMM 分配统计
    mu_tensor = torch.FloatTensor(mu_all).to(device)
    c_tensor = torch.FloatTensor(c_all).to(device)
    with torch.no_grad():
        if args.use_conditional_gmm:
            assignment = model.gmm_prior.compute_assignment(mu_tensor, c_tensor).cpu().numpy()
        else:
            assignment = model.gmm_prior.compute_assignment(mu_tensor).cpu().numpy()
    hard_assignment = assignment.argmax(axis=1)
    
    print(f"\nGMM 分量使用分布:")
    for k in range(args.n_components):
        count = (hard_assignment == k).sum()
        pct = 100 * count / len(hard_assignment)
        print(f"  分量 {k}: {count:4d} ({pct:.1f}%)")
    
    # 生成评估
    print("\n=== 生成质量评估 ===")
    
    # 从先验采样生成
    n_gen = min(500, len(x_real))
    c_sample = torch.FloatTensor(c_all[:n_gen]).to(device)
    x_real_sample = x_real[:n_gen]
    
    with torch.no_grad():
        # 静态 e（原始方式）
        x_gen_prior, e_prior = model.generate_from_prior(c_sample, dynamic_e=False)
        x_gen_prior = x_gen_prior.cpu().numpy().squeeze(-1)
        
        # 动态 e（新方式）
        x_gen_dynamic, e_dynamic = model.generate_from_prior(c_sample, dynamic_e=True, update_interval=5)
        x_gen_dynamic = x_gen_dynamic.cpu().numpy().squeeze(-1)
        
        x_real_tensor = torch.FloatTensor(x_real_sample).to(device)
        if x_real_tensor.dim() == 2:
            x_real_tensor = x_real_tensor.unsqueeze(-1)
        x_gen_posterior, e_posterior = model.generate_from_posterior(x_real_tensor, c_sample)
        x_gen_posterior = x_gen_posterior.cpu().numpy().squeeze(-1)
    
    # 计算完整的四个指标
    c_sample_np = c_all[:n_gen]
    
    print("\n[从先验采样 - 静态 e]")
    mmd_prior = compute_mmd(x_real_sample, x_gen_prior)
    flat_kl_prior = compute_flat_kl(x_real_sample, x_gen_prior)
    mdd_prior = compute_mdd(x_real_sample, x_gen_prior)
    jftsd_prior = compute_jftsd(x_real_sample, c_sample_np, x_gen_prior, device=device)
    ks_prior = compute_ks_statistic(x_real_sample, x_gen_prior)
    var_ratio_prior = compute_variance_ratio(x_real_sample, x_gen_prior)
    
    print(f"  MMD:     {mmd_prior:.4f}")
    print(f"  Flat KL: {flat_kl_prior:.4f}")
    print(f"  MDD:     {mdd_prior:.4f}")
    print(f"  J-FTSD:  {jftsd_prior:.4f}")
    print(f"  KS:      {ks_prior:.4f}")
    print(f"  VarRatio: {var_ratio_prior:.4f}")
    
    print("\n[从先验采样 - 动态 e（每5步更新）]")
    mmd_dynamic = compute_mmd(x_real_sample, x_gen_dynamic)
    flat_kl_dynamic = compute_flat_kl(x_real_sample, x_gen_dynamic)
    mdd_dynamic = compute_mdd(x_real_sample, x_gen_dynamic)
    jftsd_dynamic = compute_jftsd(x_real_sample, c_sample_np, x_gen_dynamic, device=device)
    ks_dynamic = compute_ks_statistic(x_real_sample, x_gen_dynamic)
    var_ratio_dynamic = compute_variance_ratio(x_real_sample, x_gen_dynamic)
    
    print(f"  MMD:     {mmd_dynamic:.4f}")
    print(f"  Flat KL: {flat_kl_dynamic:.4f}")
    print(f"  MDD:     {mdd_dynamic:.4f}")
    print(f"  J-FTSD:  {jftsd_dynamic:.4f}")
    print(f"  KS:      {ks_dynamic:.4f}")
    print(f"  VarRatio: {var_ratio_dynamic:.4f}")
    
    print("\n[从后验采样]")
    mmd_posterior = compute_mmd(x_real_sample, x_gen_posterior)
    flat_kl_posterior = compute_flat_kl(x_real_sample, x_gen_posterior)
    mdd_posterior = compute_mdd(x_real_sample, x_gen_posterior)
    jftsd_posterior = compute_jftsd(x_real_sample, c_sample_np, x_gen_posterior, device=device)
    ks_posterior = compute_ks_statistic(x_real_sample, x_gen_posterior)
    var_ratio_posterior = compute_variance_ratio(x_real_sample, x_gen_posterior)
    
    print(f"  MMD:     {mmd_posterior:.4f}")
    print(f"  Flat KL: {flat_kl_posterior:.4f}")
    print(f"  MDD:     {mdd_posterior:.4f}")
    print(f"  J-FTSD:  {jftsd_posterior:.4f}")
    print(f"  KS:      {ks_posterior:.4f}")
    print(f"  VarRatio: {var_ratio_posterior:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # 1. 训练损失曲线
    ax = axes[0, 0]
    ax.plot(history['fm'], label='FM Loss', alpha=0.8)
    ax.plot(history['var'], label='Var Loss', alpha=0.8)
    ax.plot(history['contrastive'], label='Contrastive', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. μ 标准差变化
    ax = axes[0, 1]
    ax.plot(history['mu_std'], label='Overall', linewidth=2)
    ax.plot(history['mu_std_per_dim'], label='Per-dim mean', linewidth=2)
    ax.axhline(y=1.0, color='r', linestyle='--', label='Target')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Std')
    ax.set_title('μ Standard Deviation (↑ is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. GMM 均衡
    ax = axes[0, 2]
    ax.plot(history['balance'], label='Balance Loss', color='purple')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('GMM Balance Loss (↓ is better)')
    ax.grid(True, alpha=0.3)
    
    # 4. μ 散点图
    ax = axes[1, 0]
    scatter = ax.scatter(mu_all[:, 0], mu_all[:, 1], c=hard_assignment, 
                         cmap='tab10', alpha=0.5, s=10)
    # 画原型
    if args.use_conditional_gmm:
        prototypes = model.gmm_prior.base_prototypes.detach().cpu().numpy()
    else:
        prototypes = model.gmm_prior.prototypes.detach().cpu().numpy()
    ax.scatter(prototypes[:, 0], prototypes[:, 1], c='red', marker='X', s=200, 
               edgecolors='black', linewidths=2, label='Prototypes')
    ax.set_xlabel('μ[0]')
    ax.set_ylabel('μ[1]')
    ax.set_title('μ Distribution with GMM Components')
    ax.legend()
    
    # 5. 分量使用分布
    ax = axes[1, 1]
    counts = [(hard_assignment == k).sum() for k in range(args.n_components)]
    colors = plt.cm.tab10(np.arange(args.n_components))
    ax.bar(range(args.n_components), counts, color=colors)
    ax.axhline(y=len(hard_assignment)/args.n_components, color='r', linestyle='--', 
               label=f'Uniform ({len(hard_assignment)//args.n_components})')
    ax.set_xlabel('Component')
    ax.set_ylabel('Count')
    ax.set_title('GMM Component Usage')
    ax.legend()
    
    # 6. 生成分布对比
    ax = axes[1, 2]
    x_real_flat = x_real_sample.flatten()
    x_gen_flat = np.nan_to_num(x_gen_prior.flatten(), nan=0.0)
    ax.hist(x_real_flat, bins=50, alpha=0.5, density=True, label='Real')
    ax.hist(x_gen_flat, bins=50, alpha=0.5, density=True, label='Generated (Prior)')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution Comparison\nKS={ks_prior:.4f}, VarRatio={var_ratio_prior:.3f}')
    ax.legend()
    
    # 7-9. 样本对比
    for i in range(3):
        ax = axes[2, i]
        idx = np.random.randint(0, n_gen)
        ax.plot(x_real_sample[idx], 'b-', alpha=0.7, label='Real', linewidth=1.5)
        ax.plot(x_gen_prior[idx], 'r--', alpha=0.7, label='Gen (Prior)', linewidth=1.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(f'Sample {idx}')
        if i == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n结果已保存到: {save_dir}")
    
    # 总结
    print("\n" + "="*60)
    print("评估总结")
    print("="*60)
    print(f"\n1. μ 多样性:")
    print(f"   整体标准差: {mu_all.std():.4f} (目标 ≈ 1.0)")
    print(f"   维度标准差: {mu_all.std(axis=0).mean():.4f}")
    
    print(f"\n2. GMM 分量均衡:")
    usage_pcts = [100 * (hard_assignment == k).sum() / len(hard_assignment) 
                  for k in range(args.n_components)]
    print(f"   使用分布: {[f'{p:.1f}%' for p in usage_pcts]}")
    print(f"   最大使用: {max(usage_pcts):.1f}%")
    print(f"   均匀目标: {100/args.n_components:.1f}%")
    
    print(f"\n3. 生成质量 (从先验采样):")
    print(f"   MMD:     {mmd_prior:.4f}")
    print(f"   Flat KL: {flat_kl_prior:.4f}")
    print(f"   MDD:     {mdd_prior:.4f}")
    print(f"   J-FTSD:  {jftsd_prior:.4f}")
    print(f"   KS:      {ks_prior:.4f}")
    print(f"   VarRatio: {var_ratio_prior:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--env_dim', type=int, default=8)
    parser.add_argument('--n_components', type=int, default=4)
    
    # 损失权重
    parser.add_argument('--kl_weight', type=float, default=0.001)
    parser.add_argument('--var_weight', type=float, default=1.0)
    parser.add_argument('--cov_weight', type=float, default=0.5)
    parser.add_argument('--contrastive_weight', type=float, default=0.5)
    parser.add_argument('--balance_weight', type=float, default=1.0)
    parser.add_argument('--sep_weight', type=float, default=0.5)
    parser.add_argument('--use_ema_prototype', action='store_true', help='Enable EMA prototype alignment')
    parser.add_argument('--use_conditional_gmm', action='store_true', help='Use conditional GMM prior p(e|c)')
    
    args = parser.parse_args()
    main(args)

