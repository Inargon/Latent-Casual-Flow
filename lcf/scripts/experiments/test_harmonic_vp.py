"""
Harmonic-VP 实验脚本
====================
测试 EnvironmentEncoderV2 在更具挑战性的 3 维混淆因子任务上的表现

与 Harmonic-VM 的区别：
- VM: 1 个混淆因子 (α - 质量变化率)
- VP: 3 个混淆因子 (α, β, η - 质量/阻尼/弹簧常数变化率)

这要求 encoder 能够捕获更复杂的环境信息。
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr, spearmanr, entropy
from scipy.linalg import sqrtm
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from lcf.data.harmonic_vp import (
    get_harmonic_vp_dataloaders,
    get_harmonic_vp_dataloaders_catsg_style,
)
from lcf.modules.env_encoder_v2 import EnvironmentEncoderV2
from lcf.modules.velocity_net import VelocityNetwork


def set_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
#                    CaTSG 生成质量指标
# ============================================================================

def compute_mmd(real_data, gen_data):
    """Maximum Mean Discrepancy with RBF kernel."""
    real_flat = real_data.reshape(real_data.shape[0], -1)
    gen_flat = gen_data.reshape(gen_data.shape[0], -1)
    
    x_np = np.asarray(real_flat, dtype=np.float64)
    y_np = np.asarray(gen_flat, dtype=np.float64)
    
    xx = rbf_kernel(x_np, x_np)
    yy = rbf_kernel(y_np, y_np)
    xy = rbf_kernel(x_np, y_np)
    mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    
    return float(max(mmd, 0.0))


def compute_flat_kl(real_data, gen_data, n_bins=50):
    """Flattened KL divergence."""
    flat_real = real_data.flatten()
    flat_gen = gen_data.flatten()
    
    flat_real = flat_real[~np.isnan(flat_real)]
    flat_gen = flat_gen[~np.isnan(flat_gen)]
    
    hist_real, edges = np.histogram(flat_real, density=True, bins=n_bins)
    hist_gen, _ = np.histogram(flat_gen, density=True, bins=edges)
    
    kl = entropy(hist_real + 1e-9, hist_gen + 1e-9)
    return float(kl)


def compute_mdd(real_data, gen_data, n_bins=20):
    """
    Marginal Distribution Distance - 使用 CaTSG 原版 HistoLoss 实现.
    """
    # 导入 CaTSG 原始实现
    import sys
    sys.path.insert(0, '/root/autodl-tmp/lcf/_reference/CaTSG-main')
    from utils.metrics.feature_distance_eval import get_mdd_eval
    
    real_data = np.asarray(real_data)
    gen_data = np.asarray(gen_data)
    
    # 确保是 (B, T, D)
    if real_data.ndim == 2:
        real_data = real_data[:, :, np.newaxis]
    if gen_data.ndim == 2:
        gen_data = gen_data[:, :, np.newaxis]
    
    mdd = get_mdd_eval(real_data, gen_data, n_bins=n_bins)
    return float(mdd)


class SimpleXEncoder(nn.Module):
    """X 编码器用于 J-FTSD."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x):
        return self.encoder(x)


class SimpleCEncoder(nn.Module):
    """C 编码器用于 J-FTSD."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    
    def forward(self, c_data):
        B, L, D_c = c_data.shape
        c_flat = c_data.reshape(-1, D_c)
        c_encoded = self.encoder(c_flat)
        c_encoded = c_encoded.reshape(B, L, -1)
        return c_encoded.mean(dim=1)


def compute_jftsd(x_real, c_real, x_gen, emb_dim=64, train_steps=200, device="cpu"):
    """J-FTSD: Joint Feature Temporal Sequence Distance."""
    x_real = torch.tensor(x_real, dtype=torch.float32, device=device)
    x_gen = torch.tensor(x_gen, dtype=torch.float32, device=device)
    c_real = torch.tensor(c_real, dtype=torch.float32, device=device)
    
    # 确保是 3 维 (B, L, D)
    if x_real.dim() == 2:
        x_real = x_real.unsqueeze(-1)
    if x_gen.dim() == 2:
        x_gen = x_gen.unsqueeze(-1)
    if c_real.dim() == 2:
        c_real = c_real.unsqueeze(-1)
    
    B, L, D_x = x_real.shape
    D_c = c_real.shape[-1]
    
    x_encoder = SimpleXEncoder(in_dim=L * D_x, out_dim=emb_dim).to(device)
    c_encoder = SimpleCEncoder(in_dim=D_c, out_dim=emb_dim).to(device)
    optimizer = torch.optim.Adam(
        list(x_encoder.parameters()) + list(c_encoder.parameters()), lr=1e-3
    )
    
    x_encoder.train()
    c_encoder.train()
    
    # 与 CaTSG 一致：每次用全部数据训练
    with torch.enable_grad():
        for step in range(train_steps):
            idx = torch.randperm(B, device=device)  # 打乱顺序
            x = x_real[idx]  # 用全部 B 个样本
            c = c_real[idx]
            
            z_t = F.normalize(x_encoder(x), dim=-1)
            z_m = F.normalize(c_encoder(c), dim=-1)
            
            logits = (z_t @ z_m.T) / np.sqrt(emb_dim)
            labels = torch.arange(B, device=device)  # B 个标签
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    with torch.no_grad():
        x_real_rep = x_encoder(x_real)
        c_real_rep = c_encoder(c_real)
        z_real = torch.cat([x_real_rep, c_real_rep], dim=-1)
        mu_real = z_real.mean(0)
        sigma_real = torch.cov(z_real.T)
        
        x_gen_rep = x_encoder(x_gen)
        z_gen = torch.cat([x_gen_rep, c_real_rep], dim=-1)
        mu_gen = z_gen.mean(0)
        sigma_gen = torch.cov(z_gen.T)
        
        mu_real_np = mu_real.cpu().numpy()
        mu_gen_np = mu_gen.cpu().numpy()
        sigma_real_np = sigma_real.cpu().numpy() + np.eye(sigma_real.shape[0]) * 1e-6
        sigma_gen_np = sigma_gen.cpu().numpy() + np.eye(sigma_gen.shape[0]) * 1e-6
        
        diff = mu_real_np - mu_gen_np
        covmean = sqrtm(sigma_real_np @ sigma_gen_np)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        ftsd = diff.dot(diff) + np.trace(sigma_real_np) + np.trace(sigma_gen_np) - 2 * np.trace(covmean)
        
    return float(max(ftsd, 0.0))


def augment_time_series(x: torch.Tensor, noise_std: float = 0.05, 
                         scale_range: tuple = (0.95, 1.05)) -> torch.Tensor:
    """时间序列数据增强（与 VM 一致）"""
    if x.dim() == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    
    # 1. 添加噪声
    noise = torch.randn_like(x) * noise_std
    x_aug = x + noise
    
    # 2. 随机缩放
    scale = torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(*scale_range)
    x_aug = x_aug * scale
    
    return x_aug


class SimpleLCF(nn.Module):
    """用于 Harmonic-VP 的简化 LCF 模型"""
    
    def __init__(
        self,
        encoder: nn.Module,
        velocity_net: nn.Module,
        use_contrastive: bool = False,
        contrastive_weight: float = 1.0,
        supervised_debug: bool = False,
        env_dim: int = 8,
    ):
        super().__init__()
        self.encoder = encoder
        self.velocity_net = velocity_net
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.supervised_debug = supervised_debug
        
        # 监督调试：从 μ 预测真实参数
        if supervised_debug:
            self.param_predictor = nn.Sequential(
                nn.Linear(env_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 3),  # 预测 α, β, η
            )
    
    def training_step(
        self,
        x_1: torch.Tensor,
        c: torch.Tensor,
        warmup: bool = False,  # 保留参数但不再使用两阶段
        e_true: torch.Tensor = None,  # 真实参数 [α, β, η]
    ):
        """训练步骤 - 简化版：去掉两阶段，始终用 x_t 编码
        
        🔑 核心改动：
        - Encoder 始终看 x_t（噪声数据），与推理时一致
        - FM loss 为主 + 小量 VICReg 防止 collapse
        - 不再区分 warmup/normal
        """
        # 确保 x_1 是 3D
        if x_1.dim() == 4 and x_1.shape[-1] == 1:
            x_1 = x_1.squeeze(-1)
        
        B = x_1.shape[0]
        device = x_1.device
        
        # Flow Matching: 构造 x_t
        t = torch.rand(B, device=device)
        x_0 = torch.randn_like(x_1)
        x_t = (1 - t.view(-1, 1, 1)) * x_0 + t.view(-1, 1, 1) * x_1
        
        # 🔑 核心：Encoder 始终用 x_t（噪声数据）
        enc_out = self.encoder(x_t, c)
        mu, logvar = enc_out['mu'], enc_out['logvar']
        e = enc_out['e']
        
        # Velocity prediction
        v_target = x_1 - x_0
        v_pred = self.velocity_net(x_t, t, c, e)
        fm_loss = F.mse_loss(v_pred, v_target)
        
        # KL 散度（轻微约束）
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        
        # VICReg 防止 collapse（小权重）
        vicreg_loss = self._vicreg_loss(mu)
        
        # 🔑 简化的 loss：FM 为主 + 小量正则化
        total_loss = fm_loss + 0.001 * kl_loss + 0.01 * vicreg_loss
        
        # 对比学习（如果启用）
        contrastive_loss = torch.tensor(0.0, device=device)
        if self.use_contrastive:
            x_view1 = augment_time_series(x_1, noise_std=0.05, scale_range=(0.95, 1.05))
            x_view2 = augment_time_series(x_1, noise_std=0.05, scale_range=(0.95, 1.05))
            
            enc_out1 = self.encoder(x_view1, c)
            enc_out2 = self.encoder(x_view2, c)
            mu1 = enc_out1['mu']
            mu2 = enc_out2['mu']
            
            cos_sim = F.cosine_similarity(mu1, mu2, dim=1)
            loss_consistency = (1 - cos_sim).mean()
            
            mu1_norm = F.normalize(mu1, dim=1)
            mu2_norm = F.normalize(mu2, dim=1)
            sim_matrix = torch.mm(mu1_norm, mu2_norm.t())
            
            temperature = 0.1
            labels = torch.arange(B, device=device)
            loss_infonce = F.cross_entropy(sim_matrix / temperature, labels)
            
            contrastive_loss = loss_consistency + 0.5 * loss_infonce
            total_loss = total_loss + self.contrastive_weight * contrastive_loss
        
        # 监督损失（调试用）
        supervised_loss = torch.tensor(0.0, device=device)
        if self.supervised_debug and e_true is not None:
            pred_params = self.param_predictor(mu)
            supervised_loss = F.mse_loss(pred_params, e_true)
            total_loss = total_loss + supervised_loss
        
        return total_loss, {
            'fm': fm_loss.item(),
            'vicreg': vicreg_loss.item(),
            'kl': kl_loss.item(),
            'contrastive': contrastive_loss.item() if self.use_contrastive else 0.0,
            'supervised': supervised_loss.item(),
        }
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def _vicreg_loss(self, z: torch.Tensor) -> torch.Tensor:
        """完整的 VICReg 损失（与 VM 一致）"""
        # Variance: 每个维度的标准差应该 >= 1
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        var_loss = F.relu(1 - std).mean()
        
        # Covariance: 不同维度应该不相关
        z_centered = z - z.mean(dim=0)
        B = z.shape[0]
        cov = (z_centered.T @ z_centered) / (B - 1)
        
        # 只惩罚非对角元素
        off_diag = cov.pow(2).sum() - cov.diag().pow(2).sum()
        cov_loss = off_diag / z.shape[1]
        
        return var_loss + 0.01 * cov_loss
    
    @torch.no_grad()
    def generate(
        self,
        c: torch.Tensor,
        e: torch.Tensor,
        n_steps: int = 100,
    ) -> torch.Tensor:
        """生成样本（使用固定的 e）"""
        B, T, D = c.shape[0], c.shape[1], 1
        device = c.device
        
        x = torch.randn(B, T, D, device=device)
        dt = 1.0 / n_steps
        
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.velocity_net(x, t, c, e)
            x = x + v * dt
        
        return x
    
    def generate_dynamic_e(
        self,
        c: torch.Tensor,
        n_steps: int = 100,
        update_interval: int = 5,
    ) -> torch.Tensor:
        """
        Dynamic E 生成（CaTSG 风格）
        
        每 update_interval 步从当前生成的 x_t 推断环境 e
        这样环境编码器在推理时也参与了！
        
        Args:
            c: 条件 (B, T, D_c)
            n_steps: 生成步数
            update_interval: 每隔多少步更新 e
        """
        B, T, D = c.shape[0], c.shape[1], 1
        device = c.device
        
        # 从噪声开始
        x = torch.randn(B, T, D, device=device)
        dt = 1.0 / n_steps
        
        # 初始 e：从噪声推断（或随机初始化）
        enc_out = self.encoder(x, c)
        e = enc_out['mu']
        
        for i in range(n_steps):
            # 每隔 update_interval 步，从当前 x_t 推断 e
            if i > 0 and i % update_interval == 0:
                enc_out = self.encoder(x, c)
                e = enc_out['mu']
            
            t = torch.full((B,), i * dt, device=device)
            v = self.velocity_net(x, t, c, e)
            x = x + v * dt
        
        return x
    
    @torch.no_grad()
    def sample_causal_mc(self, c, n_steps=100, n_mc_samples=10):
        """
        Causal MC 生成 - LCF 核心方法！
        
        实现 Theorem 1: v_do(x,t,c) = E_{p(e|x,c)}[v(x,t,c,e)]
        
        通过蒙特卡洛采样近似期望，实现连续后门调整。
        """
        B, T, D = c.shape[0], c.shape[1], 1
        device = c.device
        env_dim = self.encoder.env_dim
        
        x_t = torch.randn(B, T, D, device=device)
        dt = 1.0 / n_steps
        
        for step in range(n_steps):
            t = torch.full((B,), step * dt, device=device)
            
            # 1. 从当前 x_t 推断后验 p(e|x_t, c)
            env_out = self.encoder(x_t, c)
            mu = env_out['mu']
            logvar = env_out['logvar']
            std = torch.exp(0.5 * logvar)
            
            # 2. MC 采样: 从后验采样 n_mc_samples 个 e
            eps = torch.randn(B, n_mc_samples, env_dim, device=device)
            e_samples = mu.unsqueeze(1) + std.unsqueeze(1) * eps
            
            # 3. 对每个 e 计算速度，然后平均
            v_samples = []
            for k in range(n_mc_samples):
                e_k = e_samples[:, k, :]
                v_k = self.velocity_net(x_t, t, c, e_k)
                v_samples.append(v_k)
            
            # v_do = E[v] ≈ (1/K) * Σ_k v_k
            v_do = torch.stack(v_samples, dim=0).mean(dim=0)
            
            x_t = x_t + v_do * dt
        
        return x_t


def compute_correlations(mu: np.ndarray, e_true: np.ndarray) -> dict:
    """
    计算 μ 与真实环境参数的相关性（线性 + 非线性指标）
    
    对于 VP，e_true 有 3 列：[α, β, η]
    """
    results = {}
    param_names = ['alpha', 'beta', 'eta']
    
    for p_idx, p_name in enumerate(param_names):
        # 1. Pearson 相关（线性）
        best_pearson = 0.0
        best_dim = 0
        for d in range(mu.shape[1]):
            try:
                corr, _ = pearsonr(mu[:, d], e_true[:, p_idx])
                if abs(corr) > abs(best_pearson):
                    best_pearson = corr
                    best_dim = d
            except:
                pass
        
        # 2. Spearman 相关（单调关系）
        best_spearman = 0.0
        for d in range(mu.shape[1]):
            try:
                corr, _ = spearmanr(mu[:, d], e_true[:, p_idx])
                if abs(corr) > abs(best_spearman):
                    best_spearman = corr
            except:
                pass
        
        # 3. 互信息（任意非线性关系）- 用整个 μ 向量
        try:
            mi = mutual_info_regression(mu, e_true[:, p_idx], random_state=42)
            best_mi = mi.max()
        except:
            best_mi = 0.0
        
        results[f'{p_name}_pearson'] = best_pearson
        results[f'{p_name}_spearman'] = best_spearman
        results[f'{p_name}_mi'] = best_mi
        results[f'{p_name}_dim'] = best_dim
    
    # 4. 分类准确率：能否区分 train/val/test 区间
    # 根据 α 划分：train [0,0.2], val [0.3,0.5], test [0.6,1.0]
    try:
        alpha = e_true[:, 0]
        labels = np.zeros(len(alpha), dtype=int)
        labels[(alpha >= 0.3) & (alpha <= 0.5)] = 1  # val
        labels[alpha >= 0.6] = 2  # test
        
        scaler = StandardScaler()
        mu_scaled = scaler.fit_transform(mu)
        
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(mu_scaled, labels)
        acc = clf.score(mu_scaled, labels)
        results['classification_acc'] = acc
    except:
        results['classification_acc'] = 0.33  # 随机猜测
    
    # 保留旧接口兼容
    results['alpha_corr'] = results['alpha_pearson']
    results['beta_corr'] = results['beta_pearson']
    results['eta_corr'] = results['eta_pearson']
    
    return results


def evaluate(
    model: SimpleLCF,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    """评估模型
    
    生成方式:
    - Dynamic E: 从生成的 x_t 推断 e (CaTSG 风格，公平比较！)
    - Prior E: 从标准正态分布采样 e (环境编码器没用)
    - Encoded E: 从真实数据编码 e (参考，有信息泄露)
    """
    model.eval()
    
    all_mu = []
    all_e_true = []
    all_x_real = []
    all_x_gen_dynamic = []   # Dynamic E 生成 (CaTSG 风格)
    all_x_gen_prior = []     # Prior E 生成
    all_x_gen_encoded = []   # Encoded E 生成
    all_x_gen_causal_mc = [] # Causal MC 生成 (LCF 核心)
    all_c = []
    
    env_dim = None
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            c = batch['c'].to(device)
            e_true = batch['e'].numpy()
            B = x.shape[0]
            
            if x.dim() == 4:
                x = x.squeeze(-1)
            
            # 编码
            enc_out = model.encoder(x, c)
            mu = enc_out['mu']
            e = enc_out['e']
            
            if env_dim is None:
                env_dim = mu.shape[-1]
            
            # 1. Dynamic E 生成 (CaTSG 风格：从生成的 x_t 推断 e)
            x_gen_dynamic = model.generate_dynamic_e(c, n_steps=100, update_interval=5)
            
            # 2. Prior E 生成 (从标准正态分布采样，环境编码器没用)
            e_prior = torch.randn(B, env_dim, device=device)
            x_gen_prior = model.generate(c, e_prior)
            
            # 3. Encoded E 生成 (使用编码的 e，有信息泄露)
            x_gen_encoded = model.generate(c, e)
            
            # 4. Causal MC 生成 (LCF 核心方法！无信息泄露)
            x_gen_causal_mc = model.sample_causal_mc(c, n_steps=100, n_mc_samples=10)
            
            all_mu.append(mu.cpu().numpy())
            all_e_true.append(e_true)
            all_x_real.append(x.cpu().numpy())
            all_x_gen_dynamic.append(x_gen_dynamic.cpu().numpy())
            all_x_gen_prior.append(x_gen_prior.cpu().numpy())
            all_x_gen_encoded.append(x_gen_encoded.cpu().numpy())
            all_x_gen_causal_mc.append(x_gen_causal_mc.cpu().numpy())
            all_c.append(c.cpu().numpy())
    
    mu = np.concatenate(all_mu)
    e_true = np.concatenate(all_e_true)
    x_real = np.concatenate(all_x_real)
    x_gen_dynamic = np.concatenate(all_x_gen_dynamic)
    x_gen_prior = np.concatenate(all_x_gen_prior)
    x_gen_encoded = np.concatenate(all_x_gen_encoded)
    x_gen_causal_mc = np.concatenate(all_x_gen_causal_mc)
    
    # 计算相关性
    corr_results = compute_correlations(mu, e_true)
    
    # 计算其他指标
    var_ratio_dynamic = x_gen_dynamic.var() / (x_real.var() + 1e-8)
    var_ratio_prior = x_gen_prior.var() / (x_real.var() + 1e-8)
    var_ratio_encoded = x_gen_encoded.var() / (x_real.var() + 1e-8)
    var_ratio_causal_mc = x_gen_causal_mc.var() / (x_real.var() + 1e-8)
    mu_diversity = mu.std(axis=0).mean()
    
    # 计算生成质量指标
    c_data = np.concatenate(all_c)
    
    print("  计算生成质量指标...")
    print("    [Dynamic E] (CaTSG 风格，公平比较)...")
    mmd_dynamic = compute_mmd(x_real, x_gen_dynamic)
    flat_kl_dynamic = compute_flat_kl(x_real, x_gen_dynamic)
    mdd_dynamic = compute_mdd(x_real, x_gen_dynamic)
    jftsd_dynamic = compute_jftsd(x_real, c_data, x_gen_dynamic, device=device)
    
    print("    [Prior E] (随机 e，环境编码器没用)...")
    mmd_prior = compute_mmd(x_real, x_gen_prior)
    flat_kl_prior = compute_flat_kl(x_real, x_gen_prior)
    mdd_prior = compute_mdd(x_real, x_gen_prior)
    jftsd_prior = compute_jftsd(x_real, c_data, x_gen_prior, device=device)
    
    print("    [Encoded E] (参考，有信息泄露)...")
    mmd_encoded = compute_mmd(x_real, x_gen_encoded)
    flat_kl_encoded = compute_flat_kl(x_real, x_gen_encoded)
    mdd_encoded = compute_mdd(x_real, x_gen_encoded)
    jftsd_encoded = compute_jftsd(x_real, c_data, x_gen_encoded, device=device)
    
    print("    [Causal MC] (LCF 核心方法，无信息泄露)...")
    mmd_causal_mc = compute_mmd(x_real, x_gen_causal_mc)
    flat_kl_causal_mc = compute_flat_kl(x_real, x_gen_causal_mc)
    mdd_causal_mc = compute_mdd(x_real, x_gen_causal_mc)
    jftsd_causal_mc = compute_jftsd(x_real, c_data, x_gen_causal_mc, device=device)
    
    results = {
        **corr_results,
        'var_ratio': var_ratio_dynamic,  # 默认使用 Dynamic E
        'var_ratio_dynamic': var_ratio_dynamic,
        'var_ratio_prior': var_ratio_prior,
        'var_ratio_encoded': var_ratio_encoded,
        'var_ratio_causal_mc': var_ratio_causal_mc,
        'mu_diversity': mu_diversity,
        # Dynamic E 指标 (CaTSG 风格，公平比较)
        'mmd': mmd_causal_mc,  # 默认使用 Causal MC (LCF 核心)
        'flat_kl': flat_kl_causal_mc,
        'mdd': mdd_causal_mc,
        'jftsd': jftsd_causal_mc,
        'mmd_dynamic': mmd_dynamic,
        'flat_kl_dynamic': flat_kl_dynamic,
        'mdd_dynamic': mdd_dynamic,
        'jftsd_dynamic': jftsd_dynamic,
        # Prior E 指标 (环境编码器没用)
        'mmd_prior': mmd_prior,
        'flat_kl_prior': flat_kl_prior,
        'mdd_prior': mdd_prior,
        'jftsd_prior': jftsd_prior,
        # Encoded E 指标 (有信息泄露)
        'mmd_encoded': mmd_encoded,
        'flat_kl_encoded': flat_kl_encoded,
        'mdd_encoded': mdd_encoded,
        'jftsd_encoded': jftsd_encoded,
        # Causal MC 指标 (LCF 核心方法！无信息泄露)
        'mmd_causal_mc': mmd_causal_mc,
        'flat_kl_causal_mc': flat_kl_causal_mc,
        'mdd_causal_mc': mdd_causal_mc,
        'jftsd_causal_mc': jftsd_causal_mc,
    }
    
    return results, x_real, x_gen_causal_mc, e_true, mu, c_data


def train_and_evaluate(
    args,
    device: torch.device,
) -> dict:
    """训练和评估"""
    
    # 加载数据
    print("\n" + "=" * 60)
    print("  🔧 加载 Harmonic-VP 数据集")
    print("=" * 60)
    
    if args.catsg_split:
        train_loader, val_loader, test_loader, info = get_harmonic_vp_dataloaders_catsg_style(
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    else:
        train_loader, val_loader, test_loader, info = get_harmonic_vp_dataloaders(
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    
    # 创建模型
    print("\n" + "=" * 60)
    print("  🏗️ 创建模型")
    print("=" * 60)
    
    encoder = EnvironmentEncoderV2(
        input_dim=info['x_dim'],
        cond_dim=info['c_dim'],
        env_dim=args.env_dim,
        hidden_dim=args.hidden_dim,
        seq_len=info['seq_len'],
    ).to(device)
    
    velocity_net = VelocityNetwork(
        seq_len=info['seq_len'],
        input_dim=info['x_dim'],
        cond_dim=info['c_dim'],
        env_dim=args.env_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    
    model = SimpleLCF(
        encoder=encoder,
        velocity_net=velocity_net,
        use_contrastive=args.use_contrastive,
        contrastive_weight=args.contrastive_weight,
        supervised_debug=args.supervised_debug,
        env_dim=args.env_dim,
    ).to(device)
    
    print(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  VelocityNet params: {sum(p.numel() for p in velocity_net.parameters()):,}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 训练
    print("\n" + "=" * 60)
    if args.two_stage:
        print(f"  🎯 两阶段训练: warmup={args.warmup_steps} steps, epochs={args.epochs}")
    else:
        print(f"  🎯 训练: epochs={args.epochs}")
    if args.use_contrastive:
        print(f"  [对比学习] weight={args.contrastive_weight}")
    if args.supervised_debug:
        print(f"  [⚠️ 监督调试模式] 添加参数预测损失")
    print("=" * 60)
    
    global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        
        for batch in train_loader:
            x = batch['x'].to(device)
            c = batch['c'].to(device)
            e_true = batch['e'].to(device) if args.supervised_debug else None
            
            # 判断是否在 warmup 阶段
            warmup = args.two_stage and (global_step < args.warmup_steps)
            
            loss, loss_dict = model.training_step(x, c, warmup=warmup, e_true=e_true)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            global_step += 1
        
        # 定期评估（不计算生成质量指标，太慢）
        if epoch % 5 == 0:
            results, _, _, e_true, mu, _ = evaluate(model, test_loader, device)
            
            print(f"  Epoch {epoch:3d}: loss={np.mean(epoch_losses):.4f}, "
                  f"α_corr={results['alpha_corr']:.4f}, "
                  f"β_corr={results['beta_corr']:.4f}, "
                  f"η_corr={results['eta_corr']:.4f}, "
                  f"var_ratio={results['var_ratio']:.4f}")
    
    # 最终评估
    print("\n  最终评估...")
    final_results, x_real, x_gen, e_true, mu, c_data = evaluate(model, test_loader, device)
    
    return final_results, x_real, x_gen, e_true, mu, model


def visualize_results(
    results: dict,
    x_real: np.ndarray,
    x_gen: np.ndarray,
    e_true: np.ndarray,
    mu: np.ndarray,
    save_path: str,
):
    """可视化结果"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # 1. 生成样本对比
    ax = axes[0, 0]
    n_show = min(5, len(x_real))
    for i in range(n_show):
        ax.plot(x_real[i, :, 0], 'b-', alpha=0.5, label='Real' if i == 0 else '')
        ax.plot(x_gen[i, :, 0], 'r--', alpha=0.5, label='Gen' if i == 0 else '')
    ax.set_title('Generated vs Real')
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('x')
    
    # 2. α 相关性散点图
    ax = axes[0, 1]
    alpha_dim = results['alpha_dim']
    scatter = ax.scatter(e_true[:, 0], mu[:, alpha_dim], c=e_true[:, 0], cmap='viridis', alpha=0.5)
    ax.set_xlabel('True α')
    ax.set_ylabel(f'μ[{alpha_dim}]')
    ax.set_title(f'α Correlation: r={results["alpha_corr"]:.3f}')
    plt.colorbar(scatter, ax=ax)
    
    # 3. β 相关性散点图
    ax = axes[0, 2]
    beta_dim = results['beta_dim']
    scatter = ax.scatter(e_true[:, 1], mu[:, beta_dim], c=e_true[:, 1], cmap='plasma', alpha=0.5)
    ax.set_xlabel('True β')
    ax.set_ylabel(f'μ[{beta_dim}]')
    ax.set_title(f'β Correlation: r={results["beta_corr"]:.3f}')
    plt.colorbar(scatter, ax=ax)
    
    # 4. η 相关性散点图
    ax = axes[0, 3]
    eta_dim = results['eta_dim']
    scatter = ax.scatter(e_true[:, 2], mu[:, eta_dim], c=e_true[:, 2], cmap='coolwarm', alpha=0.5)
    ax.set_xlabel('True η')
    ax.set_ylabel(f'μ[{eta_dim}]')
    ax.set_title(f'η Correlation: r={results["eta_corr"]:.3f}')
    plt.colorbar(scatter, ax=ax)
    
    # 5. 3D 散点图：μ 的前 3 维 vs 真实参数
    ax = axes[1, 0]
    # 用颜色表示 α
    scatter = ax.scatter(mu[:, 0], mu[:, 1], c=e_true[:, 0], cmap='viridis', alpha=0.5)
    ax.set_xlabel('μ[0]')
    ax.set_ylabel('μ[1]')
    ax.set_title('μ space (color=α)')
    plt.colorbar(scatter, ax=ax)
    
    # 6. μ 分布直方图
    ax = axes[1, 1]
    for d in range(min(4, mu.shape[1])):
        ax.hist(mu[:, d], bins=30, alpha=0.5, label=f'μ[{d}]')
    ax.set_title('μ Distributions')
    ax.legend()
    
    # 7. 指标汇总
    ax = axes[1, 2]
    ax.axis('off')
    text = f"""
    ══════════════════════════════
         Harmonic-VP Results
    ══════════════════════════════
    
    Spearman (单调):
    α: {results['alpha_spearman']:.3f}  β: {results['beta_spearman']:.3f}  η: {results['eta_spearman']:.3f}
    
    分类准确率: {results['classification_acc']:.3f}
    
    生成质量 (越低越好):
    ────────────────────────────
    MMD:     {results['mmd']:.6f}
    Flat KL: {results['flat_kl']:.4f}
    MDD:     {results['mdd']:.4f}
    J-FTSD:  {results['jftsd']:.4f}
    
    Variance Ratio: {results['var_ratio']:.4f}
    """
    ax.text(0.02, 0.5, text, fontsize=9, fontfamily='monospace',
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 8. 参数之间的相关性
    ax = axes[1, 3]
    # α vs β
    scatter = ax.scatter(e_true[:, 0], e_true[:, 1], c=e_true[:, 2], 
                         cmap='coolwarm', alpha=0.5, s=10)
    ax.set_xlabel('α')
    ax.set_ylabel('β')
    ax.set_title('True params distribution (color=η)')
    plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n📊 Results saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Harmonic-VP 实验')
    
    # 数据参数
    parser.add_argument('--n_train', type=int, default=3000)
    parser.add_argument('--n_val', type=int, default=1000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--catsg_split', action='store_true', help='使用 CaTSG 80/20 混合采样')
    
    # 模型参数
    parser.add_argument('--env_dim', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=64)
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    
    # 对比学习
    parser.add_argument('--use_contrastive', action='store_true')
    parser.add_argument('--contrastive_weight', type=float, default=0.01, help='大幅降低！对比学习会干扰 FM')
    
    # 两阶段训练
    parser.add_argument('--two_stage', action='store_true')
    parser.add_argument('--warmup_steps', type=int, default=50)
    
    # 监督调试模式
    parser.add_argument('--supervised_debug', action='store_true', 
                        help='添加监督损失调试encoder能力')
    
    args = parser.parse_args()
    
    # 设置
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("  Harmonic-VP Experiment")
    print("  3 维混淆因子: α (质量), β (阻尼), η (弹簧)")
    print("=" * 60)
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path('logs/lcf/harmonic_vp') / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练和评估
    results, x_real, x_gen, e_true, mu, model = train_and_evaluate(args, device)
    
    # 可视化
    visualize_results(
        results, x_real, x_gen, e_true, mu,
        str(save_dir / 'results.png')
    )
    
    # 打印最终结果
    print("\n" + "=" * 60)
    print("  Final Results")
    print("=" * 60)
    
    print(f"\n  📊 线性相关性 (Pearson):")
    print(f"  ────────────────────────────────")
    print(f"  α Pearson:  {results['alpha_pearson']:.4f}")
    print(f"  β Pearson:  {results['beta_pearson']:.4f}")
    print(f"  η Pearson:  {results['eta_pearson']:.4f}")
    
    print(f"\n  📈 单调相关性 (Spearman):")
    print(f"  ────────────────────────────────")
    print(f"  α Spearman: {results['alpha_spearman']:.4f}")
    print(f"  β Spearman: {results['beta_spearman']:.4f}")
    print(f"  η Spearman: {results['eta_spearman']:.4f}")
    
    print(f"\n  🔗 互信息 (非线性依赖):")
    print(f"  ────────────────────────────────")
    print(f"  α MI: {results['alpha_mi']:.4f}")
    print(f"  β MI: {results['beta_mi']:.4f}")
    print(f"  η MI: {results['eta_mi']:.4f}")
    
    print(f"\n  🎯 分类准确率 (区分 train/val/test):")
    print(f"  ────────────────────────────────")
    print(f"  Accuracy: {results['classification_acc']:.4f} (随机=0.33)")
    
    print(f"\n  📦 生成质量对比 (越低越好):")
    print(f"  ┌────────────────────────────────────────────────────────────────────────────────────────┐")
    print(f"  │ {'指标':<10} │ {'Dynamic E':<12} │ {'Causal MC':<12} │ {'Prior E':<12} │ {'Encoded E':<12} │ {'CaTSG':<8} │")
    print(f"  ├────────────────────────────────────────────────────────────────────────────────────────┤")
    print(f"  │ {'MMD':<10} │ {results['mmd_dynamic']:<12.6f} │ {results['mmd_causal_mc']:<12.6f} │ {results['mmd_prior']:<12.6f} │ {results['mmd_encoded']:<12.6f} │ {'0.034':<8} │")
    print(f"  │ {'KL':<10} │ {results['flat_kl_dynamic']:<12.4f} │ {results['flat_kl_causal_mc']:<12.4f} │ {results['flat_kl_prior']:<12.4f} │ {results['flat_kl_encoded']:<12.4f} │ {'0.066':<8} │")
    print(f"  │ {'MDD':<10} │ {results['mdd_dynamic']:<12.4f} │ {results['mdd_causal_mc']:<12.4f} │ {results['mdd_prior']:<12.4f} │ {results['mdd_encoded']:<12.4f} │ {'0.053':<8} │")
    print(f"  │ {'J-FTSD':<10} │ {results['jftsd_dynamic']:<12.4f} │ {results['jftsd_causal_mc']:<12.4f} │ {results['jftsd_prior']:<12.4f} │ {results['jftsd_encoded']:<12.4f} │ {'12.677':<8} │")
    print(f"  └────────────────────────────────────────────────────────────────────────────────────────┘")
    print(f"  🔑 Dynamic E:  从生成的 x_t 推断 e (CaTSG 风格，公平比较！)")
    print(f"     Causal MC:  ⭐ LCF 核心方法！MC 后门调整 (无信息泄露)")
    print(f"     Prior E:    随机 e，环境编码器没用")
    print(f"     Encoded E:  从真实数据编码 e，有信息泄露")
    
    print(f"\n  📊 其他指标:")
    print(f"  ────────────────────────────────")
    print(f"  Variance Ratio (Dynamic):   {results['var_ratio_dynamic']:.4f}")
    print(f"  Variance Ratio (Causal MC): {results['var_ratio_causal_mc']:.4f}")
    print(f"  Variance Ratio (Prior):     {results['var_ratio_prior']:.4f}")
    print(f"  Variance Ratio (Encoded):   {results['var_ratio_encoded']:.4f}")
    print(f"  μ Diversity:                {results['mu_diversity']:.4f}")
    
    # 综合评估
    avg_spearman = (abs(results['alpha_spearman']) + abs(results['beta_spearman']) + abs(results['eta_spearman'])) / 3
    avg_mi = (results['alpha_mi'] + results['beta_mi'] + results['eta_mi']) / 3
    
    print(f"\n  📊 综合指标:")
    print(f"  ────────────────────────────────")
    print(f"  平均 |Spearman|: {avg_spearman:.4f}")
    print(f"  平均 MI:         {avg_mi:.4f}")
    
    if results['classification_acc'] > 0.5 or avg_spearman > 0.3:
        print(f"\n  ✅ 模型学到了有意义的环境表示！")
    else:
        print(f"\n  ⚠️ 需要进一步调优")
    
    print(f"\n  ✅ Results saved to: {save_dir}")


if __name__ == '__main__':
    main()

