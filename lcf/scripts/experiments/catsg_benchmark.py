"""
CaTSG Benchmark - LCF V2 vs CaTSG
=================================

在 CaTSG 使用的数据集上全面评测 LCF V2

支持的数据集:
1. Harmonic-VM: 可变质量阻尼振荡器
2. Harmonic-VP: 可变参数阻尼振荡器
3. Air Quality: 北京空气质量 (需要下载)
4. Traffic: 交通流量 (需要下载)

评测指标:
- MDD: Marginal Distribution Distance
- KL: KL Divergence
- MMD: Maximum Mean Discrepancy
- J-FTSD: Joint Feature-Temporal Signature Distance
- Environment Recovery: 环境解耦能力

运行:
    python lcf/scripts/experiments/catsg_benchmark.py --dataset harmonic_vm
    python lcf/scripts/experiments/catsg_benchmark.py --dataset harmonic_vp
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from scipy.spatial.distance import cdist
import sys
from datetime import datetime
import json

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from lcf.models.latent_causal_flow_v2 import LatentCausalFlowV2
from lcf.data.catsg_datasets import get_catsg_dataloaders, generate_harmonic_data


# ==================== 评测指标 ====================

def compute_mmd(x_real, x_gen, kernel='rbf', gamma=None):
    """
    Maximum Mean Discrepancy
    """
    x_real = x_real.reshape(len(x_real), -1)
    x_gen = x_gen.reshape(len(x_gen), -1)
    
    if gamma is None:
        gamma = 1.0 / x_real.shape[1]
    
    K_xx = np.exp(-gamma * cdist(x_real, x_real, 'sqeuclidean'))
    K_yy = np.exp(-gamma * cdist(x_gen, x_gen, 'sqeuclidean'))
    K_xy = np.exp(-gamma * cdist(x_real, x_gen, 'sqeuclidean'))
    
    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return float(mmd)


def compute_mdd(x_real, x_gen, n_bins=50):
    """
    Marginal Distribution Distance (KL divergence of histograms)
    """
    x_real_flat = x_real.flatten()
    x_gen_flat = x_gen.flatten()
    
    min_val = min(x_real_flat.min(), x_gen_flat.min())
    max_val = max(x_real_flat.max(), x_gen_flat.max())
    
    hist_real, _ = np.histogram(x_real_flat, bins=n_bins, range=(min_val, max_val), density=True)
    hist_gen, _ = np.histogram(x_gen_flat, bins=n_bins, range=(min_val, max_val), density=True)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    hist_real = hist_real + eps
    hist_gen = hist_gen + eps
    
    # Normalize
    hist_real = hist_real / hist_real.sum()
    hist_gen = hist_gen / hist_gen.sum()
    
    # KL divergence
    kl = np.sum(hist_real * np.log(hist_real / hist_gen))
    return float(kl)


def compute_temporal_correlation(x_real, x_gen):
    """
    时序相关性指标
    """
    # 自相关函数
    def autocorr(x, max_lag=10):
        result = []
        x_flat = x.mean(axis=0).flatten()  # 平均后计算
        for lag in range(max_lag):
            if lag == 0:
                result.append(1.0)
            else:
                corr = np.corrcoef(x_flat[:-lag], x_flat[lag:])[0, 1]
                result.append(corr if not np.isnan(corr) else 0.0)
        return np.array(result)
    
    acf_real = autocorr(x_real)
    acf_gen = autocorr(x_gen)
    
    # MSE between autocorrelation functions
    acf_mse = np.mean((acf_real - acf_gen) ** 2)
    return float(acf_mse)


def compute_all_metrics(x_real, x_gen):
    """计算所有评测指标"""
    metrics = {
        'mmd': compute_mmd(x_real, x_gen),
        'mdd': compute_mdd(x_real, x_gen),
        'temporal_corr': compute_temporal_correlation(x_real, x_gen),
        'variance_ratio': float(x_gen.var() / (x_real.var() + 1e-8)),
        'mean_mse': float(np.mean((x_gen.mean(axis=(1, 2)) - x_real.mean(axis=(1, 2))) ** 2)),
    }
    return metrics


# ==================== 训练和验证 ====================

def train_epoch(model, loader, optimizer, device, epoch, warmup_steps, encoder_lr_mult=0.1, base_lr=1e-3):
    """训练一个 epoch"""
    model.train()
    
    phase, _, _ = model.get_training_phase()
    
    # 动态调整学习率
    if phase == "warmup":
        optimizer.param_groups[0]['lr'] = base_lr
        optimizer.param_groups[1]['lr'] = base_lr
    else:
        optimizer.param_groups[0]['lr'] = base_lr * encoder_lr_mult
        optimizer.param_groups[1]['lr'] = base_lr
    
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase}]")
    
    for batch in pbar:
        x = batch['x'].to(device)
        c = batch['c'].to(device)
        
        batch_dict = {'x': x, 'c': c}
        
        optimizer.zero_grad()
        loss = model.training_step(batch_dict, 0)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        model.increment_step()
        
        new_phase, _, _ = model.get_training_phase()
        if new_phase != phase:
            phase = new_phase
            if phase == "normal":
                optimizer.param_groups[0]['lr'] = base_lr * encoder_lr_mult
                print(f"\n  🔄 Phase -> normal! Encoder LR: {base_lr * encoder_lr_mult:.2e}")
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'phase': phase, 'step': model.get_current_step()})
    
    return {'loss': total_loss / len(loader), 'phase': phase}


@torch.no_grad()
def validate(model, loader, device, n_gen_samples=500):
    """验证"""
    model.eval()
    
    all_e = []
    all_e_true = []
    all_x_real = []
    all_c = []
    all_c_mean = []
    
    for batch in loader:
        x = batch['x'].to(device)
        c = batch['c'].to(device)
        c_mean = batch['c_mean'].to(device)
        e_true = batch['e_true'].to(device)
        
        mu, std = model.encode_environment(x, c)
        
        all_e.append(mu.cpu())
        all_e_true.append(e_true.cpu())
        all_x_real.append(x.cpu())
        all_c.append(c.cpu())
        all_c_mean.append(c_mean.cpu())
    
    all_e = torch.cat(all_e, dim=0).numpy()
    all_e_true = torch.cat(all_e_true, dim=0).numpy()
    all_x_real = torch.cat(all_x_real, dim=0).numpy()
    all_c = torch.cat(all_c, dim=0)
    all_c_mean = torch.cat(all_c_mean, dim=0).numpy()
    
    # 环境恢复
    best_corr = 0
    for i in range(all_e.shape[1]):
        for j in range(all_e_true.shape[1]):
            try:
                r, _ = stats.pearsonr(all_e[:, i], all_e_true[:, j])
                if abs(r) > abs(best_corr):
                    best_corr = r
            except:
                pass
    
    # 生成样本
    n_samples = min(n_gen_samples, len(all_c))
    idx = np.random.choice(len(all_c), n_samples, replace=False)
    c_sample = all_c[idx].to(device)
    
    # 使用先验采样 (无数据泄露)
    x_gen, _ = model.sample(
        c=c_sample,
        num_steps=50,
        num_mc_samples=8,
        use_prior=True,
    )
    x_gen = x_gen.cpu().numpy()
    x_real = all_x_real[idx]
    
    # 使用编码环境 (参考)
    e_sample = torch.from_numpy(all_e[idx]).to(device)
    x_gen_with_e, _ = model.sample(
        c=c_sample,
        num_steps=50,
        num_mc_samples=4,
        e_fixed=e_sample,
    )
    x_gen_with_e = x_gen_with_e.cpu().numpy()
    
    # 计算指标
    metrics_prior = compute_all_metrics(x_real, x_gen)
    metrics_enc = compute_all_metrics(x_real, x_gen_with_e)
    
    return {
        'env_recovery': best_corr,
        **{f'{k}_prior': v for k, v in metrics_prior.items()},
        **{f'{k}_enc': v for k, v in metrics_enc.items()},
        'e': all_e,
        'e_true': all_e_true,
        'x_real': x_real,
        'x_gen': x_gen,
        'x_gen_with_e': x_gen_with_e,
    }


def visualize_results(metrics, save_path, config=None):
    """可视化结果"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    
    # 1. 环境恢复
    ax = axes[0, 0]
    e = metrics['e']
    e_true = metrics['e_true']
    
    best_i, best_j, best_r = 0, 0, 0
    for i in range(e.shape[1]):
        for j in range(e_true.shape[1]):
            try:
                r, _ = stats.pearsonr(e[:, i], e_true[:, j])
                if abs(r) > abs(best_r):
                    best_i, best_j, best_r = i, j, r
            except:
                pass
    
    scatter = ax.scatter(e_true[:, best_j], e[:, best_i], 
                        c=e_true[:, best_j], cmap='coolwarm', alpha=0.6, s=10)
    ax.set_xlabel(f'True E (dim {best_j})')
    ax.set_ylabel(f'Learned E (dim {best_i})')
    ax.set_title(f'Environment Recovery: r = {best_r:.4f}')
    plt.colorbar(scatter, ax=ax)
    
    # 2. 生成样本对比 (Prior)
    ax = axes[0, 1]
    x_real = metrics['x_real']
    x_gen = metrics['x_gen']
    
    for i in range(min(5, len(x_real))):
        ax.plot(x_real[i].squeeze(), 'b-', alpha=0.4, label='Real' if i == 0 else '')
        ax.plot(x_gen[i].squeeze(), 'r--', alpha=0.4, label='Gen (Prior)' if i == 0 else '')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title('Real vs Generated (Prior E)')
    ax.legend()
    
    # 3. 生成样本对比 (Encoded)
    ax = axes[0, 2]
    x_gen_e = metrics['x_gen_with_e']
    
    for i in range(min(5, len(x_real))):
        ax.plot(x_real[i].squeeze(), 'b-', alpha=0.4, label='Real' if i == 0 else '')
        ax.plot(x_gen_e[i].squeeze(), 'g--', alpha=0.4, label='Gen (Enc)' if i == 0 else '')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title('Real vs Generated (Encoded E)')
    ax.legend()
    
    # 4. 分布对比
    ax = axes[1, 0]
    ax.hist(x_real.flatten(), bins=50, alpha=0.5, density=True, label='Real')
    ax.hist(x_gen.flatten(), bins=50, alpha=0.5, density=True, label='Gen (Prior)')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Comparison')
    ax.legend()
    
    # 5. 环境分布
    ax = axes[1, 1]
    for i in range(min(4, e.shape[1])):
        ax.hist(e[:, i], bins=30, alpha=0.5, label=f'E dim {i}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.set_title('Learned Environment Distribution')
    ax.legend(fontsize=8)
    
    # 6. 指标汇总
    ax = axes[1, 2]
    ax.axis('off')
    
    config_str = ""
    if config:
        config_str = f"""
Config:
  dataset: {config.get('dataset_name', 'N/A')}
  warmup_steps: {config.get('warmup_steps', 'N/A')}
  env_dim: {config.get('env_dim', 'N/A')}
"""
    
    text = f"""
═══════════════════════════════════════════════════
         LCF V2 - CaTSG Benchmark Results
═══════════════════════════════════════════════════
{config_str}
Environment Recovery:    {metrics['env_recovery']:.4f}

--- Generation with Prior E (No Leak) ---
MMD:                     {metrics['mmd_prior']:.6f}
MDD (KL):                {metrics['mdd_prior']:.6f}
Temporal Corr MSE:       {metrics['temporal_corr_prior']:.6f}
Variance Ratio:          {metrics['variance_ratio_prior']:.4f}
Mean MSE:                {metrics['mean_mse_prior']:.6f}

--- Generation with Encoded E (Reference) ---
MMD:                     {metrics['mmd_enc']:.6f}
MDD (KL):                {metrics['mdd_enc']:.6f}
Variance Ratio:          {metrics['variance_ratio_enc']:.4f}
Mean MSE:                {metrics['mean_mse_enc']:.6f}
═══════════════════════════════════════════════════
    """
    ax.text(0.05, 0.5, text, fontsize=9, family='monospace',
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Results saved: {save_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LCF V2 CaTSG Benchmark")
    
    # 数据集选择
    parser.add_argument('--dataset', type=str, default='harmonic_vm',
                       choices=['harmonic_vm', 'harmonic_vp', 'aq', 'traffic'],
                       help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./dataset',
                       help='Data directory')
    
    # 训练配置 (对齐 CaTSG)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    # 模型配置
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--env_dim', type=int, default=32)
    
    # V2 配置
    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--vicreg_sim', type=float, default=25.0)
    parser.add_argument('--vicreg_var', type=float, default=25.0)
    parser.add_argument('--vicreg_cov', type=float, default=1.0)
    parser.add_argument('--orth_weight', type=float, default=0.5)
    parser.add_argument('--consistency_weight', type=float, default=0.1)
    parser.add_argument('--encoder_lr_mult', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=10)
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"  LCF V2 - CaTSG Benchmark: {args.dataset}")
    print("=" * 70)
    print(f"Device: {args.device}")
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}_{args.dataset}_env{args.env_dim}"
    save_dir = Path("logs/lcf/catsg_benchmark") / args.dataset / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(save_dir / "config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 加载数据
    train_loader, val_loader, test_loader, config = get_catsg_dataloaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    
    print(f"\n配置:")
    print(f"  数据集: {args.dataset}")
    print(f"  序列长度: {config['seq_len']}")
    print(f"  条件维度: {config['c_dim']}")
    print(f"  训练样本: {config['n_train']}")
    
    # 创建模型
    model = LatentCausalFlowV2(
        seq_len=config['seq_len'],
        channels=config['x_dim'],
        cond_channels=config['c_dim'],
        env_dim=args.env_dim,
        hid_dim=args.hidden_dim,
        
        warmup_steps=args.warmup_steps,
        warmup_losses=['vicreg', 'orth'],
        normal_losses=['fm', 'kl', 'consist', 'orth'],
        
        vicreg_sim_weight=args.vicreg_sim,
        vicreg_var_weight=args.vicreg_var,
        vicreg_cov_weight=args.vicreg_cov,
        
        kl_weight=0.01,
        consistency_weight=args.consistency_weight,
        orth_weight=args.orth_weight,
        cfg_scale=1.5,
        
        c_dropout_schedule="two_stage",
        stage1_c_dropout=0.5,
        stage2_c_dropout=0.2,
        
        learning_rate=args.lr,
    ).to(args.device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数: {n_params:,}")
    
    # 优化器
    param_groups = [
        {'params': model.env_encoder.parameters(), 'lr': args.lr * args.encoder_lr_mult, 'name': 'encoder'},
        {'params': model.velocity_net.parameters(), 'lr': args.lr, 'name': 'velocity_net'},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    steps_per_epoch = len(train_loader)
    warmup_epochs = (args.warmup_steps // steps_per_epoch) + 1
    print(f"  预计 Warmup 结束: ~epoch {warmup_epochs}")
    
    # 训练
    print(f"\n开始训练 ({args.epochs} epochs)...")
    best_score = -float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_metrics': []}
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, args.device, epoch, 
            args.warmup_steps, args.encoder_lr_mult, args.lr
        )
        scheduler.step()
        history['train_loss'].append(train_metrics['loss'])
        
        # 验证
        val_metrics = validate(model, val_loader, args.device)
        history['val_metrics'].append({
            'env_recovery': val_metrics['env_recovery'],
            'variance_ratio': val_metrics['variance_ratio_prior'],
            'mmd': val_metrics['mmd_prior'],
        })
        
        # 综合评分
        score = (
            0.4 * abs(val_metrics['env_recovery']) +
            0.3 * max(0, 1 - abs(val_metrics['variance_ratio_prior'] - 1)) +
            0.3 * max(0, 1 - val_metrics['mmd_prior'] * 10)
        )
        
        print(f"\n[Epoch {epoch}] Phase: {train_metrics['phase']}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Env Recovery: {val_metrics['env_recovery']:.4f}")
        print(f"  Var Ratio: {val_metrics['variance_ratio_prior']:.4f}")
        print(f"  MMD: {val_metrics['mmd_prior']:.6f}")
        print(f"  📊 Score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            print(f"  ✓ New best model!")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{args.patience})")
        
        if patience_counter >= args.patience and epoch > warmup_epochs + 2:
            print(f"\n🛑 Early Stopping at epoch {epoch}!")
            break
    
    # 最终测试
    print("\n" + "=" * 70)
    print("  Final Evaluation on Test Set")
    print("=" * 70)
    
    model.load_state_dict(torch.load(save_dir / "best_model.pt"))
    test_metrics = validate(model, test_loader, args.device)
    
    print(f"\n📊 Final Test Results:")
    print(f"  Environment Recovery: {test_metrics['env_recovery']:.4f}")
    print(f"  MMD (Prior): {test_metrics['mmd_prior']:.6f}")
    print(f"  MDD (Prior): {test_metrics['mdd_prior']:.6f}")
    print(f"  Variance Ratio (Prior): {test_metrics['variance_ratio_prior']:.4f}")
    
    # 保存结果
    results = {
        'dataset': args.dataset,
        'env_recovery': float(test_metrics['env_recovery']),
        'mmd_prior': float(test_metrics['mmd_prior']),
        'mdd_prior': float(test_metrics['mdd_prior']),
        'variance_ratio_prior': float(test_metrics['variance_ratio_prior']),
        'mmd_enc': float(test_metrics['mmd_enc']),
        'variance_ratio_enc': float(test_metrics['variance_ratio_enc']),
    }
    
    with open(save_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # 可视化
    vis_config = {
        'dataset_name': args.dataset,
        'warmup_steps': args.warmup_steps,
        'env_dim': args.env_dim,
    }
    visualize_results(test_metrics, save_dir / "benchmark_results.png", config=vis_config)
    
    print(f"\n✅ 实验完成! 结果保存在: {save_dir}")


if __name__ == "__main__":
    main()





