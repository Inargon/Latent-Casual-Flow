"""
Harmonic-VP Dataset (Variable Parameters)
=========================================
变参数阻尼振荡器数据集

与 Harmonic-VM 不同，VP 版本同时变化三个参数：
- α (alpha): 质量变化率 m(t) = m0 + α*t
- β (beta): 阻尼变化率 γ(t) = γ0*(1 + β*t)
- η (eta): 弹簧常数变化率 k(t) = k0*(1 + η*t)

这是一个更具挑战性的任务，因为环境有 3 维混淆因子。
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.integrate import solve_ivp
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class HarmonicVPConfig:
    """Harmonic-VP 物理参数配置（与 CaTSG 一致）"""
    # 基础物理参数
    m0: float = 0.5       # 初始质量
    gamma0: float = 0.1   # 初始阻尼系数
    k0: float = 1.0       # 初始弹簧常数
    
    # 时间参数
    T: float = 30.0       # 总仿真时间
    seq_len: int = 96     # 序列长度
    
    # 初始条件范围
    x0_range: Tuple[float, float] = (-2.0, 2.0)
    v0_range: Tuple[float, float] = (-1.5, 1.5)
    
    # CaTSG 的参数范围
    # Train: 低变化 - 稳定动态
    train_alpha: Tuple[float, float] = (0.0, 0.2)
    train_beta: Tuple[float, float] = (0.0, 0.01)
    train_eta: Tuple[float, float] = (0.02, 0.08)
    
    # Val: 中等变化 - 过渡动态
    val_alpha: Tuple[float, float] = (0.3, 0.5)
    val_beta: Tuple[float, float] = (0.018, 0.022)
    val_eta: Tuple[float, float] = (0.18, 0.22)
    
    # Test: 高变化 - 挑战性动态
    test_alpha: Tuple[float, float] = (0.6, 1.0)
    test_beta: Tuple[float, float] = (0.035, 0.04)
    test_eta: Tuple[float, float] = (0.42, 0.5)


class HarmonicVPDataset(Dataset):
    """
    Harmonic-VP 数据集
    
    物理方程：
        m(t) * a = -γ(t) * v - k(t) * x
        
    其中：
        m(t) = m0 + α*t      (质量随时间线性增长)
        γ(t) = γ0*(1 + β*t)  (阻尼随时间增长)
        k(t) = k0*(1 + η*t)  (弹簧常数随时间增长)
    
    数据格式：
        x: (N, T, 1) 加速度时间序列
        c: (N, T, 2) [速度, 位置] 作为条件
        e: (N, 3) [α, β, η] 作为环境参数
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        split: str = 'train',
        config: Optional[HarmonicVPConfig] = None,
        seed: int = 42,
        normalize: bool = True,
        stats: Optional[Dict] = None,
    ):
        """
        Args:
            n_samples: 样本数量
            split: 'train', 'val', 或 'test'
            config: 物理参数配置
            seed: 随机种子
            normalize: 是否标准化
            stats: 预计算的统计量（用于 val/test）
        """
        self.n_samples = n_samples
        self.split = split
        self.config = config or HarmonicVPConfig()
        self.seed = seed
        self.normalize = normalize
        
        # 设置随机种子
        np.random.seed(seed)
        
        # 获取参数范围
        self.alpha_range = getattr(self.config, f'{split}_alpha')
        self.beta_range = getattr(self.config, f'{split}_beta')
        self.eta_range = getattr(self.config, f'{split}_eta')
        
        # 生成数据
        self.x, self.c, self.e = self._generate_data()
        
        # 标准化
        if normalize:
            if stats is not None:
                self.stats = stats
            else:
                self.stats = {
                    'x_mean': self.x.mean(),
                    'x_std': self.x.std() + 1e-8,
                    'c_mean': self.c.mean(axis=(0, 1)),
                    'c_std': self.c.std(axis=(0, 1)) + 1e-8,
                }
            
            self.x = (self.x - self.stats['x_mean']) / self.stats['x_std']
            self.c = (self.c - self.stats['c_mean']) / self.stats['c_std']
        else:
            self.stats = None
        
        print(f"[{split}] Generated {n_samples} samples")
        print(f"  α ∈ [{self.alpha_range[0]:.2f}, {self.alpha_range[1]:.2f}]")
        print(f"  β ∈ [{self.beta_range[0]:.3f}, {self.beta_range[1]:.3f}]")
        print(f"  η ∈ [{self.eta_range[0]:.2f}, {self.eta_range[1]:.2f}]")
    
    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成数据"""
        cfg = self.config
        t_eval = np.linspace(0, cfg.T, cfg.seq_len)
        
        x_list = []
        c_list = []
        e_list = []
        
        for _ in range(self.n_samples):
            # 采样环境参数
            alpha = np.random.uniform(*self.alpha_range)
            beta = np.random.uniform(*self.beta_range)
            eta = np.random.uniform(*self.eta_range)
            
            # 采样初始条件
            x0 = np.random.uniform(*cfg.x0_range)
            v0 = np.random.uniform(*cfg.v0_range)
            
            # 定义动力学方程
            def dynamics(t, y):
                pos, vel = y
                m = cfg.m0 + alpha * t
                gamma_t = cfg.gamma0 * (1 + beta * t)
                k_t = cfg.k0 * (1 + eta * t)
                acc = (-gamma_t * vel - k_t * pos) / m
                return [vel, acc]
            
            # 求解 ODE
            sol = solve_ivp(
                dynamics, 
                [0, cfg.T], 
                [x0, v0], 
                t_eval=t_eval, 
                method='RK45'
            )
            
            position = sol.y[0]
            velocity = sol.y[1]
            
            # 计算加速度
            acceleration = np.zeros(cfg.seq_len)
            for i, t in enumerate(t_eval):
                m = cfg.m0 + alpha * t
                gamma_t = cfg.gamma0 * (1 + beta * t)
                k_t = cfg.k0 * (1 + eta * t)
                acceleration[i] = (-gamma_t * velocity[i] - k_t * position[i]) / m
            
            x_list.append(acceleration[:, np.newaxis])  # (T, 1)
            c_list.append(np.stack([velocity, position], axis=-1))  # (T, 2)
            e_list.append([alpha, beta, eta])  # 3个参数
        
        return (
            np.array(x_list, dtype=np.float32),
            np.array(c_list, dtype=np.float32),
            np.array(e_list, dtype=np.float32)
        )
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'x': torch.from_numpy(self.x[idx]),
            'c': torch.from_numpy(self.c[idx]),
            'e': torch.from_numpy(self.e[idx]),  # [α, β, η]
            'alpha': torch.tensor(self.e[idx, 0]),
            'beta': torch.tensor(self.e[idx, 1]),
            'eta': torch.tensor(self.e[idx, 2]),
        }


class HarmonicVPDatasetMixed(HarmonicVPDataset):
    """
    支持 CaTSG 80/20 混合采样的 Harmonic-VP 数据集
    
    80% 样本从主区间采样，20% 从其他区间采样
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        split: str = 'train',
        config: Optional[HarmonicVPConfig] = None,
        seed: int = 42,
        normalize: bool = True,
        stats: Optional[Dict] = None,
        main_ratio: float = 0.8,
    ):
        self.main_ratio = main_ratio
        # 先保存 config 和 split 以便 _generate_data 使用
        self.config = config or HarmonicVPConfig()
        self.split = split
        self.n_samples = n_samples
        self.seed = seed
        self.normalize = normalize
        
        np.random.seed(seed)
        
        # 设置参数范围
        self.alpha_range = getattr(self.config, f'{split}_alpha')
        self.beta_range = getattr(self.config, f'{split}_beta')
        self.eta_range = getattr(self.config, f'{split}_eta')
        
        # 生成数据
        self.x, self.c, self.e = self._generate_data()
        
        # 标准化
        if normalize:
            if stats is not None:
                self.stats = stats
            else:
                self.stats = {
                    'x_mean': self.x.mean(),
                    'x_std': self.x.std() + 1e-8,
                    'c_mean': self.c.mean(axis=(0, 1)),
                    'c_std': self.c.std(axis=(0, 1)) + 1e-8,
                }
            
            self.x = (self.x - self.stats['x_mean']) / self.stats['x_std']
            self.c = (self.c - self.stats['c_mean']) / self.stats['c_std']
        else:
            self.stats = None
    
    def _sample_params(self, split: str) -> Tuple[float, float, float]:
        """从指定 split 的范围采样参数"""
        alpha_range = getattr(self.config, f'{split}_alpha')
        beta_range = getattr(self.config, f'{split}_beta')
        eta_range = getattr(self.config, f'{split}_eta')
        
        alpha = np.random.uniform(*alpha_range)
        beta = np.random.uniform(*beta_range)
        eta = np.random.uniform(*eta_range)
        
        return alpha, beta, eta
    
    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成带 80/20 混合采样的数据"""
        cfg = self.config
        t_eval = np.linspace(0, cfg.T, cfg.seq_len)
        
        # 计算主/交叉采样数量
        n_main = int(self.n_samples * self.main_ratio)
        n_cross = self.n_samples - n_main
        
        # 确定其他 split
        other_splits = [s for s in ['train', 'val', 'test'] if s != self.split]
        
        x_list = []
        c_list = []
        e_list = []
        
        print(f"\n[{self.split}] CaTSG 80/20 混合采样:")
        print(f"  主采样 ({self.main_ratio*100:.0f}%): {n_main} 从 {self.split}")
        print(f"  交叉采样 ({(1-self.main_ratio)*100:.0f}%): {n_cross} 从 {other_splits}")
        
        for i in range(self.n_samples):
            # 决定从哪个 split 采样
            if i < n_main:
                # 主采样
                alpha, beta, eta = self._sample_params(self.split)
            else:
                # 交叉采样：从其他 split 随机选一个
                cross_split = np.random.choice(other_splits)
                alpha, beta, eta = self._sample_params(cross_split)
            
            # 采样初始条件
            x0 = np.random.uniform(*cfg.x0_range)
            v0 = np.random.uniform(*cfg.v0_range)
            
            # 定义动力学方程
            def dynamics(t, y):
                pos, vel = y
                m = cfg.m0 + alpha * t
                gamma_t = cfg.gamma0 * (1 + beta * t)
                k_t = cfg.k0 * (1 + eta * t)
                acc = (-gamma_t * vel - k_t * pos) / m
                return [vel, acc]
            
            # 求解 ODE
            sol = solve_ivp(
                dynamics, 
                [0, cfg.T], 
                [x0, v0], 
                t_eval=t_eval, 
                method='RK45'
            )
            
            position = sol.y[0]
            velocity = sol.y[1]
            
            # 计算加速度
            acceleration = np.zeros(cfg.seq_len)
            for j, t in enumerate(t_eval):
                m = cfg.m0 + alpha * t
                gamma_t = cfg.gamma0 * (1 + beta * t)
                k_t = cfg.k0 * (1 + eta * t)
                acceleration[j] = (-gamma_t * velocity[j] - k_t * position[j]) / m
            
            x_list.append(acceleration[:, np.newaxis])
            c_list.append(np.stack([velocity, position], axis=-1))
            e_list.append([alpha, beta, eta])
        
        self.stats = {
            'x_mean': np.array(x_list).mean(),
            'x_std': np.array(x_list).std() + 1e-8,
            'c_mean': np.array(c_list).mean(axis=(0, 1)),
            'c_std': np.array(c_list).std(axis=(0, 1)) + 1e-8,
        }
        
        return (
            np.array(x_list, dtype=np.float32),
            np.array(c_list, dtype=np.float32),
            np.array(e_list, dtype=np.float32)
        )


def get_harmonic_vp_dataloaders(
    n_train: int = 3000,
    n_val: int = 1000,
    n_test: int = 1000,
    batch_size: int = 64,
    seed: int = 42,
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    获取 Harmonic-VP DataLoader（简单版本，无混合采样）
    """
    config = HarmonicVPConfig()
    
    train_set = HarmonicVPDataset(n_train, 'train', config, seed, normalize)
    val_set = HarmonicVPDataset(n_val, 'val', config, seed+1, normalize, train_set.stats)
    test_set = HarmonicVPDataset(n_test, 'test', config, seed+2, normalize, train_set.stats)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    info = {
        'seq_len': config.seq_len,
        'x_dim': 1,
        'c_dim': 2,
        'e_dim': 3,  # VP 有 3 个混淆因子
        'stats': train_set.stats,
    }
    
    return train_loader, val_loader, test_loader, info


def get_harmonic_vp_dataloaders_catsg_style(
    n_train: int = 3000,
    n_val: int = 1000,
    n_test: int = 1000,
    batch_size: int = 64,
    seed: int = 42,
    normalize: bool = True,
    main_ratio: float = 0.8,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    获取 CaTSG 风格的 Harmonic-VP DataLoader（80/20 混合采样）
    """
    config = HarmonicVPConfig()
    
    print("=" * 60)
    print("  Harmonic VP Dataset - CaTSG 风格 80/20 混合采样")
    print("=" * 60)
    print(f"  主采样比例: {main_ratio*100:.0f}% / 交叉采样: {(1-main_ratio)*100:.0f}%")
    
    train_set = HarmonicVPDatasetMixed(
        n_train, 'train', config, seed, normalize, 
        stats=None, main_ratio=main_ratio
    )
    val_set = HarmonicVPDatasetMixed(
        n_val, 'val', config, seed+1, normalize,
        stats=train_set.stats, main_ratio=main_ratio
    )
    test_set = HarmonicVPDatasetMixed(
        n_test, 'test', config, seed+2, normalize,
        stats=train_set.stats, main_ratio=main_ratio
    )
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    print(f"\n  ✓ 数据集创建完成")
    print(f"    X 形状: (N, {config.seq_len}, 1)")
    print(f"    C 形状: (N, {config.seq_len}, 2)")
    print(f"    E 形状: (N, 3) [α, β, η]")
    
    info = {
        'seq_len': config.seq_len,
        'x_dim': 1,
        'c_dim': 2,
        'e_dim': 3,
        'stats': train_set.stats,
    }
    
    return train_loader, val_loader, test_loader, info


if __name__ == '__main__':
    # 测试数据生成
    print("Testing Harmonic-VP dataset...")
    
    train_loader, val_loader, test_loader, info = get_harmonic_vp_dataloaders_catsg_style(
        n_train=100,
        n_val=50,
        n_test=50,
        batch_size=16,
    )
    
    print("\n" + "=" * 40)
    print("Sample batch:")
    for batch in train_loader:
        print(f"  x: {batch['x'].shape}")
        print(f"  c: {batch['c'].shape}")
        print(f"  e: {batch['e'].shape}")
        print(f"  α range: [{batch['alpha'].min():.3f}, {batch['alpha'].max():.3f}]")
        print(f"  β range: [{batch['beta'].min():.4f}, {batch['beta'].max():.4f}]")
        print(f"  η range: [{batch['eta'].min():.3f}, {batch['eta'].max():.3f}]")
        break

