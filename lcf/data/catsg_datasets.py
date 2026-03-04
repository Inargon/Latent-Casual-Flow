"""
CaTSG Datasets Loader
=====================
加载和处理 CaTSG 格式的数据集

支持的数据集:
1. Harmonic-VM: 可变质量阻尼振荡器 (合成)
2. Harmonic-VP: 可变参数阻尼振荡器 (合成)
3. Air Quality: 北京空气质量数据 (真实)
4. Traffic: 交通流量数据 (真实)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')


class CaTSGDataset(Dataset):
    """
    CaTSG 格式数据集加载器
    
    数据格式:
        x: (N, T, 1) 目标变量
        c: (N, T, C) 条件变量
    """
    
    def __init__(
        self, 
        x_path: str, 
        c_path: str, 
        normalize: bool = True,
        e_true_path: Optional[str] = None,
    ):
        """
        Args:
            x_path: x 数据路径
            c_path: c 数据路径
            normalize: 是否标准化
            e_true_path: 真实环境标签路径（可选，用于评估）
        """
        self.x = np.load(x_path).astype(np.float32)  # (N, T, 1)
        self.c = np.load(c_path).astype(np.float32)  # (N, T, C)
        
        # 确保形状正确
        if self.x.ndim == 2:
            self.x = self.x[:, :, np.newaxis]
        if self.c.ndim == 2:
            self.c = self.c[:, :, np.newaxis]
        
        self.seq_len = self.x.shape[1]
        self.x_dim = self.x.shape[2]
        self.c_dim = self.c.shape[2]
        
        # 标准化
        self.normalize = normalize
        if normalize:
            self.x_mean = self.x.mean()
            self.x_std = self.x.std() + 1e-8
            self.x = (self.x - self.x_mean) / self.x_std
            
            self.c_mean = self.c.mean(axis=(0, 1), keepdims=True)
            self.c_std = self.c.std(axis=(0, 1), keepdims=True) + 1e-8
            self.c = (self.c - self.c_mean) / self.c_std
        
        # 加载真实环境标签（如果有）
        self.e_true = None
        if e_true_path and Path(e_true_path).exists():
            self.e_true = np.load(e_true_path).astype(np.float32)
        
        print(f"Loaded dataset: x={self.x.shape}, c={self.c.shape}")
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        result = {
            'x': torch.from_numpy(self.x[idx]),
            'c': torch.from_numpy(self.c[idx]),
            'c_mean': torch.from_numpy(self.c[idx].mean(axis=0)),
        }
        
        if self.e_true is not None:
            result['e_true'] = torch.from_numpy(self.e_true[idx])
        else:
            # 使用条件的统计量作为伪环境标签
            result['e_true'] = torch.from_numpy(self.c[idx].mean(axis=0)[:2])
        
        return result


def generate_harmonic_data(
    dataset_type: str = 'vm',
    n_train: int = 3000,
    n_val: int = 1000,
    n_test: int = 1000,
    seq_len: int = 96,
    save_dir: str = './dataset',
    seed: int = 42,
) -> Path:
    """
    生成阻尼振荡器数据集
    
    Args:
        dataset_type: 'vm' (Variable Mass) 或 'vp' (Variable Parameters)
        n_train, n_val, n_test: 各 split 的样本数
        seq_len: 序列长度
        save_dir: 保存目录
        seed: 随机种子
        
    Returns:
        数据保存路径
    """
    np.random.seed(seed)
    
    dataset_name = f'harmonic_{dataset_type}'
    output_dir = Path(save_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 物理参数
    m0 = 0.5      # 初始质量
    gamma = 0.1   # 阻尼系数
    k = 1.0       # 弹簧常数
    T_max = 30.0  # 总时间
    
    def generate_samples(n_samples, alpha_range, param_ranges=None):
        """生成一组样本"""
        x_list = []
        c_list = []
        e_list = []
        
        for _ in range(n_samples):
            # 采样参数
            alpha = np.random.uniform(*alpha_range)
            
            if dataset_type == 'vm':
                # Variable Mass: m = m0 + alpha * t
                beta = 0.01
                eta = 0.1
            else:  # vp
                # Variable Parameters: 随机 beta, eta
                if param_ranges:
                    beta = np.random.uniform(*param_ranges['beta'])
                    eta = np.random.uniform(*param_ranges['eta'])
                else:
                    beta = 0.01
                    eta = 0.1
            
            # 初始条件
            x0 = np.random.uniform(-2.0, 2.0)
            v0 = np.random.uniform(-1.5, 1.5)
            
            # 求解 ODE
            def dynamics(t, y):
                x, v = y
                if dataset_type == 'vm':
                    m = m0 + alpha * t
                else:
                    m = m0 + alpha * t
                
                gamma_t = gamma * (1 + beta * t)
                k_t = k * (1 + eta * t)
                
                a = (-gamma_t * v - k_t * x) / m
                return [v, a]
            
            t_span = [0, T_max]
            t_eval = np.linspace(0, T_max, seq_len)
            
            sol = solve_ivp(dynamics, t_span, [x0, v0], t_eval=t_eval, method='RK45')
            
            position = sol.y[0]
            velocity = sol.y[1]
            
            # 计算加速度
            acceleration = np.zeros(seq_len)
            for i, t in enumerate(t_eval):
                if dataset_type == 'vm':
                    m = m0 + alpha * t
                else:
                    m = m0 + alpha * t
                gamma_t = gamma * (1 + beta * t)
                k_t = k * (1 + eta * t)
                acceleration[i] = (-gamma_t * velocity[i] - k_t * position[i]) / m
            
            # x: 加速度, c: [速度, 位置]
            x_list.append(acceleration[:, np.newaxis])
            c_list.append(np.stack([velocity, position], axis=-1))
            e_list.append([alpha, beta if dataset_type == 'vp' else 0.0])
        
        return np.array(x_list), np.array(c_list), np.array(e_list)
    
    # 生成各 split
    if dataset_type == 'vm':
        # Alpha-based split
        x_train, c_train, e_train = generate_samples(n_train, [0.0, 0.2])
        x_val, c_val, e_val = generate_samples(n_val, [0.3, 0.5])
        x_test, c_test, e_test = generate_samples(n_test, [0.6, 1.0])
    else:  # vp
        # Combination-based split
        x_train, c_train, e_train = generate_samples(
            n_train, [0.0, 0.2], 
            {'beta': [0.0, 0.01], 'eta': [0.002, 0.08]}
        )
        x_val, c_val, e_val = generate_samples(
            n_val, [0.3, 0.5],
            {'beta': [0.018, 0.022], 'eta': [0.18, 0.22]}
        )
        x_test, c_test, e_test = generate_samples(
            n_test, [0.6, 1.0],
            {'beta': [0.035, 0.04], 'eta': [0.42, 0.5]}
        )
    
    # 保存
    np.save(output_dir / 'x_train.npy', x_train.astype(np.float32))
    np.save(output_dir / 'c_train.npy', c_train.astype(np.float32))
    np.save(output_dir / 'e_train.npy', e_train.astype(np.float32))
    
    np.save(output_dir / 'x_val.npy', x_val.astype(np.float32))
    np.save(output_dir / 'c_val.npy', c_val.astype(np.float32))
    np.save(output_dir / 'e_val.npy', e_val.astype(np.float32))
    
    np.save(output_dir / 'x_test.npy', x_test.astype(np.float32))
    np.save(output_dir / 'c_test.npy', c_test.astype(np.float32))
    np.save(output_dir / 'e_test.npy', e_test.astype(np.float32))
    
    print(f"Generated {dataset_name} dataset:")
    print(f"  Train: x={x_train.shape}, c={c_train.shape}")
    print(f"  Val: x={x_val.shape}, c={c_val.shape}")
    print(f"  Test: x={x_test.shape}, c={c_test.shape}")
    print(f"  Saved to: {output_dir}")
    
    return output_dir


def get_catsg_dataloaders(
    dataset_name: str,
    data_dir: str = './dataset',
    batch_size: int = 64,
    num_workers: int = 4,
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    获取 CaTSG 数据集的 DataLoader
    
    Args:
        dataset_name: 数据集名称 ('harmonic_vm', 'harmonic_vp', 'aq', 'traffic')
        data_dir: 数据目录
        batch_size: batch size
        num_workers: 数据加载线程数
        normalize: 是否标准化
        
    Returns:
        train_loader, val_loader, test_loader, config
    """
    data_path = Path(data_dir) / dataset_name
    
    # 检查数据是否存在，如果不存在则生成
    if not data_path.exists():
        if dataset_name.startswith('harmonic'):
            dataset_type = dataset_name.split('_')[1]
            generate_harmonic_data(
                dataset_type=dataset_type,
                save_dir=data_dir,
            )
        else:
            raise FileNotFoundError(
                f"Dataset {dataset_name} not found at {data_path}. "
                f"Please download and preprocess the data first."
            )
    
    # 加载数据集
    train_set = CaTSGDataset(
        x_path=str(data_path / 'x_train.npy'),
        c_path=str(data_path / 'c_train.npy'),
        normalize=normalize,
        e_true_path=str(data_path / 'e_train.npy'),
    )
    
    val_set = CaTSGDataset(
        x_path=str(data_path / 'x_val.npy'),
        c_path=str(data_path / 'c_val.npy'),
        normalize=normalize,
        e_true_path=str(data_path / 'e_val.npy'),
    )
    
    test_set = CaTSGDataset(
        x_path=str(data_path / 'x_test.npy'),
        c_path=str(data_path / 'c_test.npy'),
        normalize=normalize,
        e_true_path=str(data_path / 'e_test.npy'),
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    
    config = {
        'seq_len': train_set.seq_len,
        'x_dim': train_set.x_dim,
        'c_dim': train_set.c_dim,
        'dataset_name': dataset_name,
        'n_train': len(train_set),
        'n_val': len(val_set),
        'n_test': len(test_set),
    }
    
    return train_loader, val_loader, test_loader, config


if __name__ == '__main__':
    # 测试数据生成
    print("Generating Harmonic-VM dataset...")
    generate_harmonic_data('vm', save_dir='./dataset')
    
    print("\nGenerating Harmonic-VP dataset...")
    generate_harmonic_data('vp', save_dir='./dataset')
    
    print("\nTesting dataloaders...")
    train_loader, val_loader, test_loader, config = get_catsg_dataloaders('harmonic_vm')
    
    for batch in train_loader:
        print(f"Batch shapes: x={batch['x'].shape}, c={batch['c'].shape}, e_true={batch['e_true'].shape}")
        break





