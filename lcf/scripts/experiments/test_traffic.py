"""
Traffic Dataset 实验脚本
====================
测试 EnvironmentEncoderV2 在真实世界 Traffic 数据集上的表现

Traffic 数据集特点：
- 目标变量 (x): traffic_volume (交通流量)
- 条件变量 (c): rain_1h, snow_1h, clouds_all, weather_main, holiday
- 环境分割: 按温度划分 - Train (<12°C), Val (12-22°C), Test (>22°C)
- 样本量: Train ~26k, Val ~16k, Test ~5.5k

与合成数据集的区别：
- 没有显式的环境参数（温度是隐式的环境变量）
- 真实世界的噪声和复杂性
- 更大的数据规模
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
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from lcf.data.traffic import get_traffic_dataloaders
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
    import sys
    sys.path.insert(0, '/data/avatar/lcf/_reference/CaTSG-main')
    from utils.metrics.feature_distance_eval import get_mdd_eval
    
    real_data = np.asarray(real_data)
    gen_data = np.asarray(gen_data)
    
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
    
    with torch.enable_grad():
        for step in range(train_steps):
            idx = torch.randperm(B, device=device)
            x = x_real[idx]
            c = c_real[idx]
            
            z_t = F.normalize(x_encoder(x), dim=-1)
            z_m = F.normalize(c_encoder(c), dim=-1)
            
            logits = (z_t @ z_m.T) / np.sqrt(emb_dim)
            labels = torch.arange(B, device=device)
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
    """时间序列数据增强"""
    if x.dim() == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    
    noise = torch.randn_like(x) * noise_std
    x_aug = x + noise
    
    scale = torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(*scale_range)
    x_aug = x_aug * scale
    
    return x_aug


class SimpleLCF(nn.Module):
    """用于 Traffic 的简化 LCF 模型"""
    
    def __init__(
        self,
        encoder: nn.Module,
        velocity_net: nn.Module,
        use_contrastive: bool = False,
        contrastive_weight: float = 1.0,
        env_dim: int = 8,
    ):
        super().__init__()
        self.encoder = encoder
        self.velocity_net = velocity_net
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
    
    def training_step(
        self,
        x_1: torch.Tensor,
        c: torch.Tensor,
        warmup: bool = False,
    ):
        """训练步骤"""
        if x_1.dim() == 4 and x_1.shape[-1] == 1:
            x_1 = x_1.squeeze(-1)
        
        B = x_1.shape[0]
        device = x_1.device
        
        # 编码环境
        enc_out = self.encoder(x_1, c)
        mu, logvar = enc_out['mu'], enc_out['logvar']
        e = enc_out['e']
        
        # 对比学习损失
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
        
        # Flow Matching loss (对 warmup 和正常阶段都需要)
        t = torch.rand(B, device=device)
        x_0 = torch.randn_like(x_1)
        x_t = (1 - t.view(-1, 1, 1)) * x_0 + t.view(-1, 1, 1) * x_1
        
        v_target = x_1 - x_0
        v_pred = self.velocity_net(x_t, t, c, e)
        
        fm_loss = F.mse_loss(v_pred, v_target)
        
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        
        vicreg_loss = self._vicreg_loss(mu)
        
        # Warmup 阶段 (CaTSG 设计: 只用 FM + 正交损失，不用对比学习)
        if warmup:
            # FM loss + 正交损失 (VICReg)，不加对比学习
            total_loss = fm_loss + 0.001 * kl_loss + 0.1 * vicreg_loss
            
            return total_loss, {
                'fm': fm_loss.item(),
                'kl': kl_loss.item(),
                'vicreg': vicreg_loss.item(),
            }
        
        # 正常训练阶段
        total_loss = fm_loss + 0.001 * kl_loss + 0.1 * vicreg_loss
        
        if self.use_contrastive:
            total_loss = total_loss + self.contrastive_weight * contrastive_loss
        
        return total_loss, {
            'fm': fm_loss.item(),
            'kl': kl_loss.item(),
            'vicreg': vicreg_loss.item(),
            'contrastive': contrastive_loss.item() if self.use_contrastive else 0.0,
        }
    
    def _vicreg_loss(self, z: torch.Tensor) -> torch.Tensor:
        """VICReg 损失"""
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        var_loss = F.relu(1 - std).mean()
        
        z_centered = z - z.mean(dim=0)
        B = z.shape[0]
        cov = (z_centered.T @ z_centered) / (B - 1)
        
        off_diag = cov.pow(2).sum() - cov.diag().pow(2).sum()
        cov_loss = off_diag / z.shape[1]
        
        return var_loss + 0.01 * cov_loss
    
    @torch.no_grad()
    def generate(
        self,
        c: torch.Tensor,
        e: torch.Tensor = None,  # 可选的固定环境
        n_steps: int = 50,
        dynamic_env: bool = False,  # 是否动态推断环境
    ) -> torch.Tensor:
        """
        生成样本
        
        Args:
            c: 条件变量 (B, T, c_dim)
            e: 固定环境 (B, env_dim)，仅当 dynamic_env=False 时使用
            n_steps: 积分步数
            dynamic_env: 是否在每一步动态推断环境（类似 CaTSG）
        """
        B, T, D = c.shape[0], c.shape[1], 1
        device = c.device
        
        x = torch.randn(B, T, D, device=device)
        dt = 1.0 / n_steps
        
        for i in range(n_steps):
            t_val = i * dt
            t = torch.full((B,), t_val, device=device)
            
            if dynamic_env:
                # 动态环境推断：用当前的 x_t 推断环境
                # 这是 LCF 论文中 Monte Carlo Flow Matching 的核心思想
                with torch.no_grad():
                    enc_out = self.encoder(x, c)
                    e_current = enc_out['e']
                v = self.velocity_net(x, t, c, e_current)
            else:
                # 固定环境：使用预先推断的 e
                v = self.velocity_net(x, t, c, e)
            
            x = x + v * dt
        
        return x


def evaluate(
    model: SimpleLCF,
    test_loader: DataLoader,
    device: torch.device,
    max_samples: int = 2000,  # 限制样本数以加速
    dynamic_env: bool = False,  # 是否使用动态环境推断
) -> dict:
    """评估模型"""
    model.eval()
    
    all_mu = []
    all_x_real = []
    all_x_gen = []
    all_c = []
    n_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            c = batch['c'].to(device)
            
            if x.dim() == 4:
                x = x.squeeze(-1)
            
            # 编码 (用真实 x 推断 μ 用于可视化)
            enc_out = model.encoder(x, c)
            mu = enc_out['mu']
            e = enc_out['e']
            
            # 生成 (可选择动态环境推断)
            x_gen = model.generate(c, e, dynamic_env=dynamic_env)
            
            all_mu.append(mu.cpu().numpy())
            all_x_real.append(x.cpu().numpy())
            all_x_gen.append(x_gen.cpu().numpy())
            all_c.append(c.cpu().numpy())
            
            n_samples += x.shape[0]
            if n_samples >= max_samples:
                break
    
    mu = np.concatenate(all_mu)
    x_real = np.concatenate(all_x_real)
    x_gen = np.concatenate(all_x_gen)
    c_data = np.concatenate(all_c)
    
    # 计算指标
    var_ratio = x_gen.var() / (x_real.var() + 1e-8)
    mu_diversity = mu.std(axis=0).mean()
    
    print("  计算生成质量指标...")
    mmd = compute_mmd(x_real, x_gen)
    flat_kl = compute_flat_kl(x_real, x_gen)
    mdd = compute_mdd(x_real, x_gen)
    jftsd = compute_jftsd(x_real, c_data, x_gen, device=device)
    
    results = {
        'var_ratio': var_ratio,
        'mu_diversity': mu_diversity,
        'mmd': mmd,
        'flat_kl': flat_kl,
        'mdd': mdd,
        'jftsd': jftsd,
    }
    
    return results, x_real, x_gen, mu, c_data


def train_and_evaluate(
    args,
    device: torch.device,
) -> dict:
    """训练和评估"""
    
    # 加载数据
    print("\n" + "=" * 60)
    print("  🔧 加载 Traffic 数据集")
    print("=" * 60)
    
    train_loader, val_loader, test_loader, stats = get_traffic_dataloaders(
        data_path=args.data_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
    
    # 获取数据维度
    sample_batch = next(iter(train_loader))
    x_dim = sample_batch['x'].shape[-1]  # 1
    c_dim = sample_batch['c'].shape[-1]  # 5
    seq_len = sample_batch['x'].shape[1]  # 96
    
    print(f"  x_dim: {x_dim}, c_dim: {c_dim}, seq_len: {seq_len}")
    
    # 创建模型
    print("\n" + "=" * 60)
    print("  🏗️ 创建模型")
    print("=" * 60)
    
    encoder = EnvironmentEncoderV2(
        input_dim=x_dim,
        cond_dim=c_dim,
        env_dim=args.env_dim,
        hidden_dim=args.hidden_dim,
        seq_len=seq_len,
    ).to(device)
    
    velocity_net = VelocityNetwork(
        seq_len=seq_len,
        input_dim=x_dim,
        cond_dim=c_dim,
        env_dim=args.env_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    
    model = SimpleLCF(
        encoder=encoder,
        velocity_net=velocity_net,
        use_contrastive=args.use_contrastive,
        contrastive_weight=args.contrastive_weight,
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
    if args.dynamic_env:
        print(f"  [动态环境推断] 生成时每步重新推断 e (Monte Carlo FM)")
    print("=" * 60)
    
    global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        
        for batch in train_loader:
            x = batch['x'].to(device)
            c = batch['c'].to(device)
            
            warmup = args.two_stage and (global_step < args.warmup_steps)
            
            loss, loss_dict = model.training_step(x, c, warmup=warmup)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            global_step += 1
            
            # Traffic 数据集大，每500步打印一次
            if global_step % 500 == 0:
                print(f"    Step {global_step}: loss={np.mean(epoch_losses[-100:]):.4f}")
        
        # 每 epoch 打印
        if epoch % 2 == 0:
            print(f"  Epoch {epoch:3d}: loss={np.mean(epoch_losses):.4f}")
    
    # 最终评估
    print("\n  最终评估...")
    final_results, x_real, x_gen, mu, c_data = evaluate(
        model, test_loader, device, 
        dynamic_env=args.dynamic_env
    )
    
    return final_results, x_real, x_gen, mu, c_data, model


def visualize_results(
    results: dict,
    x_real: np.ndarray,
    x_gen: np.ndarray,
    mu: np.ndarray,
    save_path: str,
):
    """可视化结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 生成样本对比
    ax = axes[0, 0]
    n_show = min(5, len(x_real))
    for i in range(n_show):
        ax.plot(x_real[i, :, 0], 'b-', alpha=0.5, label='Real' if i == 0 else '')
        ax.plot(x_gen[i, :, 0], 'r--', alpha=0.5, label='Gen' if i == 0 else '')
    ax.set_title('Generated vs Real Traffic Volume')
    ax.legend()
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Normalized Volume')
    
    # 2. μ 空间分布
    ax = axes[0, 1]
    scatter = ax.scatter(mu[:, 0], mu[:, 1], c=np.arange(len(mu)), cmap='viridis', alpha=0.3, s=1)
    ax.set_xlabel('μ[0]')
    ax.set_ylabel('μ[1]')
    ax.set_title('μ Space Distribution')
    plt.colorbar(scatter, ax=ax, label='Sample Index')
    
    # 3. μ 分布直方图
    ax = axes[0, 2]
    for d in range(min(4, mu.shape[1])):
        ax.hist(mu[:, d], bins=50, alpha=0.5, label=f'μ[{d}]')
    ax.set_title('μ Distributions')
    ax.legend()
    
    # 4. 生成质量分布对比
    ax = axes[1, 0]
    ax.hist(x_real.flatten(), bins=50, alpha=0.5, label='Real', density=True)
    ax.hist(x_gen.flatten(), bins=50, alpha=0.5, label='Generated', density=True)
    ax.set_title('Value Distribution Comparison')
    ax.legend()
    ax.set_xlabel('Normalized Traffic Volume')
    
    # 5. 时间维度统计对比
    ax = axes[1, 1]
    mean_real = x_real.mean(axis=0)[:, 0]
    std_real = x_real.std(axis=0)[:, 0]
    mean_gen = x_gen.mean(axis=0)[:, 0]
    std_gen = x_gen.std(axis=0)[:, 0]
    
    t = np.arange(len(mean_real))
    ax.fill_between(t, mean_real - std_real, mean_real + std_real, alpha=0.3, color='blue')
    ax.plot(t, mean_real, 'b-', label='Real', linewidth=2)
    ax.fill_between(t, mean_gen - std_gen, mean_gen + std_gen, alpha=0.3, color='red')
    ax.plot(t, mean_gen, 'r--', label='Generated', linewidth=2)
    ax.set_title('Temporal Pattern Comparison')
    ax.legend()
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Mean ± Std')
    
    # 6. 指标汇总
    ax = axes[1, 2]
    ax.axis('off')
    text = f"""
    ══════════════════════════════════
           Traffic Dataset Results
    ══════════════════════════════════
    
    📦 生成质量指标 (越低越好):
    ────────────────────────────────
    MMD:       {results['mmd']:.6f}
    Flat KL:   {results['flat_kl']:.4f}
    MDD:       {results['mdd']:.4f}
    J-FTSD:    {results['jftsd']:.4f}
    
    📊 其他指标:
    ────────────────────────────────
    Variance Ratio: {results['var_ratio']:.4f}
    μ Diversity:    {results['mu_diversity']:.4f}
    
    ══════════════════════════════════
    
    Note: Traffic 数据集没有显式环境参数，
    因此不计算相关性指标。
    主要关注生成质量指标。
    """
    ax.text(0.02, 0.5, text, fontsize=10, fontfamily='monospace',
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n📊 Results saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Traffic Dataset 实验')
    
    # 数据参数 (CaTSG: batch_size=256)
    parser.add_argument('--data_path', type=str, 
                        default='/data/avatar/lcf/data_raw/Metro_Interstate_Traffic_Volume.csv')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=256)  # CaTSG: 256
    
    # 模型参数 (测试更大的 env_dim)
    parser.add_argument('--env_dim', type=int, default=8)  # 增加到 8，给更多表达空间
    parser.add_argument('--hidden_dim', type=int, default=32)  # CaTSG: 32
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    
    # 对比学习
    parser.add_argument('--use_contrastive', action='store_true')
    parser.add_argument('--contrastive_weight', type=float, default=1.0)
    
    # 两阶段训练 (CaTSG: warmup=50)
    parser.add_argument('--two_stage', action='store_true')
    parser.add_argument('--warmup_steps', type=int, default=50)  # CaTSG: 50
    
    # 动态环境推断 (CaTSG 的核心机制)
    parser.add_argument('--dynamic_env', action='store_true',
                        help='在生成时每步动态推断环境 (Monte Carlo Flow Matching)')
    
    args = parser.parse_args()
    
    # 设置
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("  Traffic Dataset Experiment")
    print("  Metro Interstate Traffic Volume")
    print("  环境分割: 按温度 - Train (<12°C), Val (12-22°C), Test (>22°C)")
    print("=" * 60)
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path('logs/lcf/traffic') / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练和评估
    results, x_real, x_gen, mu, c_data, model = train_and_evaluate(args, device)
    
    # 可视化
    visualize_results(
        results, x_real, x_gen, mu,
        str(save_dir / 'results.png')
    )
    
    # 打印最终结果
    print("\n" + "=" * 60)
    print("  Final Results")
    print("=" * 60)
    
    print(f"\n  📦 生成质量指标 (越低越好):")
    print(f"  ────────────────────────────────")
    print(f"  MMD:       {results['mmd']:.6f}")
    print(f"  Flat KL:   {results['flat_kl']:.4f}")
    print(f"  MDD:       {results['mdd']:.4f}")
    print(f"  J-FTSD:    {results['jftsd']:.4f}")
    
    print(f"\n  📊 其他指标:")
    print(f"  ────────────────────────────────")
    print(f"  Variance Ratio: {results['var_ratio']:.4f}")
    print(f"  μ Diversity:    {results['mu_diversity']:.4f}")
    
    # 与 CaTSG 论文对比
    print(f"\n  📋 与 CaTSG 论文对比:")
    print(f"  ────────────────────────────────")
    print(f"  指标        我们        CaTSG")
    print(f"  MMD        {results['mmd']:.6f}    -")
    print(f"  Flat KL    {results['flat_kl']:.4f}       -")
    print(f"  MDD        {results['mdd']:.4f}       -")
    print(f"  J-FTSD     {results['jftsd']:.4f}       -")
    print(f"  (CaTSG 论文未报告 Traffic 数据集的具体指标)")
    
    print(f"\n  ✅ Results saved to: {save_dir}")


if __name__ == '__main__':
    main()

