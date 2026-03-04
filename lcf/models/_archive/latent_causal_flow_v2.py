"""
Latent Causal Flow V2 - Enhanced Version
=========================================
超越 CaTSG 的连续环境因果流模型

核心改进:
1. 两阶段训练: Warmup (只训练Encoder) + Normal (训练全部)
2. VICReg 对比学习: 适配连续环境的自监督学习
3. 多视图数据增强: 让 Encoder 学会从不同视角识别相同环境
4. 干净数据预训练: Warmup 阶段用干净的 x_1 训练 Encoder

理论基础:
    v_do(x, t, c) = E_{e ~ q(e|x,c)} [v_θ(x, t, c, e)]
    
    通过连续环境的后验期望实现精确的因果干预。

Author: Enhanced from CaTSG insights while keeping continuous E advantage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Dict, Optional, Tuple, List
from torch.utils.data import DataLoader
from contextlib import contextmanager
from tqdm import tqdm


class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._initialized = False
    
    def initialize(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self._initialized = True
    
    def update(self):
        if not self._initialized:
            self.initialize()
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                device = param.device
                if self.shadow[name].device != device:
                    self.shadow[name] = self.shadow[name].to(device)
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_shadow(self):
        if not self._initialized:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                device = param.device
                if self.shadow[name].device != device:
                    self.shadow[name] = self.shadow[name].to(device)
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class TimeSeriesAugmentation(nn.Module):
    """
    时间序列多视图增强模块
    
    核心假设: 同一序列的不同视图应该有相同的环境 E
    """
    
    def __init__(
        self,
        noise_scale: float = 0.1,
        max_shift: int = 5,
        scale_range: Tuple[float, float] = (0.9, 1.1),
    ):
        super().__init__()
        self.noise_scale = noise_scale
        self.max_shift = max_shift
        self.scale_range = scale_range
    
    def forward(
        self, 
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        创建两个增强视图
        
        Args:
            x: (B, T, D) 原始时间序列
            c: (B, T, D_c) 或 (B, D_c) 条件
            
        Returns:
            x_view1, x_view2: 两个增强视图
            c_view1, c_view2: 对应的条件（如果适用）
        """
        B, T, D = x.shape
        device = x.device
        
        # View 1: 加噪声 + 轻微缩放
        noise = self.noise_scale * x.std() * torch.randn_like(x)
        scale = torch.empty(B, 1, 1, device=device).uniform_(*self.scale_range)
        x_view1 = x * scale + noise
        
        # View 2: 时间偏移 + 不同噪声
        shifts = torch.randint(-self.max_shift, self.max_shift + 1, (B,), device=device)
        x_view2 = torch.stack([
            torch.roll(x[i], shifts=shifts[i].item(), dims=0) 
            for i in range(B)
        ])
        noise2 = self.noise_scale * x.std() * torch.randn_like(x_view2)
        x_view2 = x_view2 + noise2
        
        # 条件也做相应变换
        c_view1, c_view2 = None, None
        if c is not None:
            if c.dim() == 3 and c.shape[1] == T:
                # (B, T, D_c) 格式
                c_view1 = c * scale
                c_view2 = torch.stack([
                    torch.roll(c[i], shifts=shifts[i].item(), dims=0)
                    for i in range(B)
                ])
            else:
                # (B, D_c) 格式或其他，不变换
                c_view1 = c
                c_view2 = c
        
        return x_view1, x_view2, c_view1, c_view2


class LatentCausalFlowV2(pl.LightningModule):
    """
    Latent Causal Flow V2: 超越 CaTSG 的连续环境模型
    
    核心创新:
    1. 两阶段训练借鉴自 CaTSG
    2. VICReg 对比学习适配连续环境
    3. 多视图学习确保环境表示的一致性
    4. 保持连续环境的理论精确性
    """
    
    def __init__(
        self,
        # Data dimensions
        seq_len: int = 96,
        channels: int = 1,
        cond_channels: int = 4,
        env_dim: int = 16,
        hid_dim: int = 128,
        
        # Monte Carlo configuration
        num_mc_samples_train: int = 1,
        num_mc_samples_eval: int = 10,
        
        # Flow Matching parameters
        sigma_min: float = 1e-4,
        
        # ============ 两阶段训练配置 (借鉴 CaTSG) ============
        warmup_steps: int = 2000,  # Warmup 阶段步数
        warmup_losses: List[str] = None,  # ['vicreg', 'orth']
        normal_losses: List[str] = None,  # ['fm', 'kl', 'consist', 'orth']
        
        # ============ VICReg 配置 (适配连续环境) ============
        vicreg_sim_weight: float = 25.0,
        vicreg_var_weight: float = 25.0,
        vicreg_cov_weight: float = 1.0,
        
        # ============ InfoNCE 对比学习 (可选) ============
        use_infonce: bool = False,
        infonce_temperature: float = 0.1,
        
        # KL regularization
        kl_weight: float = 0.01,
        kl_annealing: bool = True,
        kl_warmup_steps: int = 2000,
        free_bits: float = 0.25,
        
        # Posterior consistency (Teacher-Student)
        consistency_weight: float = 0.1,
        
        # Orthogonal regularization
        orth_weight: float = 0.1,
        
        # Variance regularization (防止 Variance Ratio 下降)
        var_reg_weight: float = 0.1,
        
        # Environment supervised loss (如果有真实环境标签)
        env_supervised_weight: float = 0.0,
        
        # C dropout for forcing E usage
        c_dropout_rate: float = 0.3,
        c_dropout_schedule: str = "two_stage",
        stage1_c_dropout: float = 0.5,
        stage2_c_dropout: float = 0.1,
        
        # Classifier-Free Guidance
        cfg_scale: float = 1.5,
        
        # Multi-view augmentation
        augmentation_noise: float = 0.1,
        augmentation_shift: int = 5,
        
        # Network configs
        env_encoder_config: Optional[Dict] = None,
        velocity_net_config: Optional[Dict] = None,
        
        # EMA
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        
        # Training
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        encoder_lr_mult: float = 0.1,  # Encoder 学习率倍率 (保护 warmup 学到的表示)
        
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Store dimensions
        self.seq_len = seq_len
        self.channels = channels
        self.cond_channels = cond_channels
        self.env_dim = env_dim
        self.hid_dim = hid_dim
        
        # MC config
        self.num_mc_samples_train = num_mc_samples_train
        self.num_mc_samples_eval = num_mc_samples_eval
        
        # Flow matching
        self.sigma_min = sigma_min
        
        # ============ 两阶段训练 ============
        self.warmup_steps = warmup_steps
        self.warmup_losses = warmup_losses or ['vicreg', 'orth']
        self.normal_losses = normal_losses or ['fm', 'kl', 'consist', 'orth']
        
        # ============ VICReg ============
        self.vicreg_sim_weight = vicreg_sim_weight
        self.vicreg_var_weight = vicreg_var_weight
        self.vicreg_cov_weight = vicreg_cov_weight
        
        # ============ InfoNCE ============
        self.use_infonce = use_infonce
        self.infonce_temperature = infonce_temperature
        
        # KL config
        self.kl_weight = kl_weight
        self.kl_annealing = kl_annealing
        self.kl_warmup_steps = kl_warmup_steps
        self.free_bits = free_bits
        
        # Consistency
        self.consistency_weight = consistency_weight
        
        # Orthogonal
        self.orth_weight = orth_weight
        
        # Variance regularization
        self.var_reg_weight = var_reg_weight
        
        # Environment supervised loss
        self.env_supervised_weight = env_supervised_weight
        
        # C dropout config
        self.c_dropout_rate = c_dropout_rate
        self.c_dropout_schedule = c_dropout_schedule
        self.stage1_c_dropout = stage1_c_dropout
        self.stage2_c_dropout = stage2_c_dropout
        
        # CFG
        self.cfg_scale = cfg_scale
        
        # Training config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.encoder_lr_mult = encoder_lr_mult
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        
        # ============ Build Networks ============
        self._build_networks(env_encoder_config, velocity_net_config)
        
        # ============ Multi-view Augmentation ============
        self.augmentation = TimeSeriesAugmentation(
            noise_scale=augmentation_noise,
            max_shift=augmentation_shift,
        )
        
        # Null condition for CFG
        self.register_buffer('null_cond', torch.zeros(1, seq_len, cond_channels))
        self.register_buffer('null_env', torch.zeros(1, env_dim))
        
        # EMA
        self.ema = EMA(self, decay=ema_decay) if use_ema else None
        
        # Manual step tracking for non-Trainer mode
        self._manual_step = 0
    
    def _build_networks(self, env_encoder_config, velocity_net_config):
        """Build encoder and velocity networks."""
        from lcf.modules.env_encoder import EnvironmentEncoder
        from lcf.modules.velocity_net import VelocityNetwork
        
        if env_encoder_config is None:
            self.env_encoder = EnvironmentEncoder(
                seq_len=self.seq_len,
                input_dim=self.channels,
                cond_dim=self.cond_channels,
                hidden_dim=self.hid_dim,
                env_dim=self.env_dim,
            )
        else:
            from lcf.utils.util import instantiate_from_config
            self.env_encoder = instantiate_from_config(env_encoder_config)
        
        if velocity_net_config is None:
            self.velocity_net = VelocityNetwork(
                seq_len=self.seq_len,
                input_dim=self.channels,
                cond_dim=self.cond_channels,
                env_dim=self.env_dim,
                hidden_dim=self.hid_dim,
            )
        else:
            from lcf.utils.util import instantiate_from_config
            self.velocity_net = instantiate_from_config(velocity_net_config)
        
        enc_params = sum(p.numel() for p in self.env_encoder.parameters())
        vel_params = sum(p.numel() for p in self.velocity_net.parameters())
        print(f"[LCF-V2] Environment Encoder: {enc_params:,} params")
        print(f"[LCF-V2] Velocity Network: {vel_params:,} params")
    
    # ==================== 数据格式转换 ====================
    
    def _to_btd(self, x: torch.Tensor, expected_last_dim: int) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(-1)
        if x.dim() == 3:
            if x.shape[-1] == expected_last_dim:
                return x
            elif x.shape[1] == expected_last_dim:
                return x.transpose(1, 2)
        return x
    
    def _to_bdt(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x.transpose(1, 2)
        return x
    
    # ==================== 训练阶段判断 (借鉴 CaTSG) ====================
    
    def get_current_step(self) -> int:
        """Get current training step, works with or without Trainer."""
        # Always use manual step counter for non-Trainer mode
        return self._manual_step
    
    def increment_step(self):
        """Manually increment step counter (for non-Trainer mode)."""
        self._manual_step += 1
        return self._manual_step
    
    def get_training_phase(self) -> Tuple[str, int, Dict[str, bool]]:
        """
        确定当前训练阶段
        
        Returns:
            (phase_name, phase_code, loss_config)
            loss_config: {'fm': bool, 'kl': bool, 'consist': bool, 'vicreg': bool, 'orth': bool}
        """
        step = self.get_current_step()
        
        if step < self.warmup_steps:
            # Warmup 阶段: 只训练 Encoder (用干净数据!)
            loss_config = {
                'fm': False,  # ❌ 不训练 Velocity Net
                'kl': False,
                'consist': False,
                'vicreg': 'vicreg' in self.warmup_losses,  # ✅ VICReg
                'infonce': 'infonce' in self.warmup_losses,
                'orth': 'orth' in self.warmup_losses,
            }
            return "warmup", 0, loss_config
        else:
            # Normal 阶段: 训练 Velocity Net + Encoder
            loss_config = {
                'fm': 'fm' in self.normal_losses,
                'kl': 'kl' in self.normal_losses,
                'consist': 'consist' in self.normal_losses,
                'vicreg': 'vicreg' in self.normal_losses,
                'infonce': 'infonce' in self.normal_losses,
                'orth': 'orth' in self.normal_losses,
            }
            return "normal", 1, loss_config
    
    # ==================== Flow Matching 核心 ====================
    
    def ot_conditional_flow(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """OT 条件流 (线性插值)"""
        t_exp = t.view(-1, 1, 1)
        x_t = (1 - t_exp) * x_0 + t_exp * x_1
        
        if self.sigma_min > 0 and self.training:
            x_t = x_t + self.sigma_min * torch.randn_like(x_t)
        
        u_t = x_1 - x_0
        return x_t, u_t
    
    # ==================== VICReg 损失 (连续环境关键!) ====================
    
    def compute_vicreg_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        VICReg: Variance-Invariance-Covariance Regularization
        
        专为连续环境表示设计，不需要离散原型!
        
        Args:
            z1, z2: (B, env_dim) 两个视图的环境表示
            
        Returns:
            loss, detail_dict
        """
        B, D = z1.shape
        
        # ============ Invariance: 同一样本的不同视图应该相似 ============
        sim_loss = F.mse_loss(z1, z2)
        
        # ============ Variance: 每个维度应该有方差 (防止坍塌) ============
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        # 要求标准差至少为1
        var_loss = (
            torch.mean(F.relu(1 - std_z1)) + 
            torch.mean(F.relu(1 - std_z2))
        )
        
        # ============ Covariance: 不同维度应该不相关 (多样性) ============
        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)
        
        cov_z1 = (z1_centered.T @ z1_centered) / (B - 1 + 1e-8)
        cov_z2 = (z2_centered.T @ z2_centered) / (B - 1 + 1e-8)
        
        # 只惩罚非对角元素
        off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=z1.device)
        cov_loss = (
            cov_z1[off_diag_mask].pow(2).sum() / D +
            cov_z2[off_diag_mask].pow(2).sum() / D
        )
        
        # 总损失
        total_loss = (
            self.vicreg_sim_weight * sim_loss +
            self.vicreg_var_weight * var_loss +
            self.vicreg_cov_weight * cov_loss
        )
        
        details = {
            'vicreg_sim': sim_loss,
            'vicreg_var': var_loss,
            'vicreg_cov': cov_loss,
        }
        
        return total_loss, details
    
    # ==================== InfoNCE 损失 (可选) ====================
    
    def compute_infonce_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        InfoNCE / SimCLR 风格的对比学习
        
        Args:
            z1, z2: (B, env_dim) 两个视图的环境表示
        """
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        # 相似度矩阵
        logits = z1 @ z2.T / self.infonce_temperature
        
        # 对角线是正样本对
        labels = torch.arange(z1.shape[0], device=z1.device)
        
        loss = (
            F.cross_entropy(logits, labels) + 
            F.cross_entropy(logits.T, labels)
        ) / 2
        
        return loss
    
    # ==================== 其他损失函数 ====================
    
    def get_kl_weight(self) -> float:
        if not self.kl_annealing:
            return self.kl_weight
        step = self.get_current_step()
        progress = min(1.0, step / max(1, self.kl_warmup_steps))
        return self.kl_weight * progress
    
    def get_c_dropout_rate(self) -> float:
        phase, _, _ = self.get_training_phase()
        
        if phase == "warmup":
            return 0.0  # Warmup 阶段不使用 C dropout
        
        if self.c_dropout_schedule == "two_stage":
            # Normal 阶段使用 stage2 dropout
            return self.stage2_c_dropout
        
        return self.c_dropout_rate
    
    def apply_c_dropout(self, c: torch.Tensor, cfg_style: bool = True) -> torch.Tensor:
        """
        Apply condition dropout for training.
        
        Args:
            c: Condition tensor (B, T, D)
            cfg_style: If True, drop entire samples to zero (for CFG compatibility).
                      If False, drop random dimensions (legacy behavior).
        """
        if not self.training:
            return c
        
        p = self.get_c_dropout_rate()
        if p <= 0:
            return c
        
        B, T, D = c.shape
        
        if cfg_style:
            # CFG-compatible: 以概率 p 将整个样本的条件设置为零
            # 这样模型学会处理 null_cond，CFG 才能正常工作
            drop_mask = torch.bernoulli(torch.full((B, 1, 1), p, device=c.device))
            return c * (1 - drop_mask)  # drop_mask=1 时整个样本变成零
        else:
            # Legacy: 按维度随机 dropout
            mask = torch.bernoulli(torch.full((B, 1, D), 1 - p, device=c.device))
            return c * mask / (1 - p + 1e-8)
    
    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if self.free_bits > 0:
            kl_per_dim = F.relu(kl_per_dim - self.free_bits) + self.free_bits
        return kl_per_dim.sum(dim=-1).mean()
    
    def compute_orthogonal_loss(self, e_samples: torch.Tensor) -> torch.Tensor:
        """正交损失促进环境多样性"""
        if e_samples.dim() == 3:
            e = e_samples.mean(dim=1)
        else:
            e = e_samples
        
        B = e.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=e.device)
        
        e_norm = F.normalize(e, dim=-1)
        sim = e_norm @ e_norm.T
        off_diag_mask = ~torch.eye(B, dtype=torch.bool, device=e.device)
        orth_loss = (sim[off_diag_mask] ** 2).mean()
        
        return orth_loss
    
    # ==================== 训练步骤 ====================
    
    def _has_trainer(self) -> bool:
        """Check if model is attached to a Trainer."""
        try:
            _ = self.trainer
            return True
        except RuntimeError:
            return False
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, loss_dict = self._shared_step(batch, stage="train")
        
        # Only log if attached to Trainer
        if self._has_trainer():
            self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            phase, phase_code, _ = self.get_training_phase()
            self.log("train/phase", float(phase_code), prog_bar=True, logger=True)
            self.log("train/c_dropout", self.get_c_dropout_rate(), prog_bar=False, logger=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        with self.ema_scope():
            loss, loss_dict = self._shared_step(batch, stage="val")
        if self._has_trainer():
            self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss
    
    def _shared_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        stage: str = "train"
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        核心训练逻辑 - 两阶段训练
        
        Warmup 阶段 (step < warmup_steps):
            - 使用干净数据 x_1 训练 Encoder
            - VICReg 对比学习 + 正交损失
            - 不训练 Velocity Net
            
        Normal 阶段 (step >= warmup_steps):
            - 标准 Flow Matching 训练
            - Encoder 从噪声数据 x_t 提取环境
            - Teacher-Student Consistency
        """
        # ============ 获取数据 ============
        x_1 = batch['x']  # 干净数据
        c = batch['c']
        
        x_1 = self._to_btd(x_1, self.channels)
        c = self._to_btd(c, self.cond_channels)
        
        B, T, _ = x_1.shape
        device = x_1.device
        
        # ============ 获取训练阶段 ============
        phase, phase_code, loss_config = self.get_training_phase()
        
        loss_dict = {}
        loss_total = torch.tensor(0.0, device=device)
        
        # ============ Warmup 阶段: 只训练 Encoder (用干净数据!) ============
        if phase == "warmup" and stage == "train":
            # 创建多视图 (关键: 用干净的 x_1!)
            x_view1, x_view2, _, _ = self.augmentation(x_1, c)
            
            # 从干净数据提取环境 (不是噪声数据!)
            # 注意: 传递完整的 c (B, T, D_c) 用于早期融合
            env_out_0 = self.env_encoder(x_1, c, t=None)
            env_out_1 = self.env_encoder(x_view1, c, t=None)
            env_out_2 = self.env_encoder(x_view2, c, t=None)
            
            mu_0, mu_1, mu_2 = env_out_0['mu'], env_out_1['mu'], env_out_2['mu']
            
            # VICReg 损失
            if loss_config['vicreg']:
                vicreg_01, vicreg_details_01 = self.compute_vicreg_loss(mu_0, mu_1)
                vicreg_02, vicreg_details_02 = self.compute_vicreg_loss(mu_0, mu_2)
                vicreg_12, _ = self.compute_vicreg_loss(mu_1, mu_2)
                
                loss_vicreg = (vicreg_01 + vicreg_02 + vicreg_12) / 3
                loss_total = loss_total + loss_vicreg
                
                loss_dict[f'{stage}/loss_vicreg'] = loss_vicreg
                loss_dict[f'{stage}/vicreg_sim'] = vicreg_details_01['vicreg_sim']
                loss_dict[f'{stage}/vicreg_var'] = vicreg_details_01['vicreg_var']
                loss_dict[f'{stage}/vicreg_cov'] = vicreg_details_01['vicreg_cov']
            
            # InfoNCE 损失 (可选)
            if loss_config.get('infonce', False):
                loss_infonce = (
                    self.compute_infonce_loss(mu_0, mu_1) +
                    self.compute_infonce_loss(mu_0, mu_2)
                ) / 2
                loss_total = loss_total + loss_infonce
                loss_dict[f'{stage}/loss_infonce'] = loss_infonce
            
            # 正交损失
            if loss_config['orth']:
                loss_orth = self.compute_orthogonal_loss(mu_0)
                loss_total = loss_total + self.orth_weight * loss_orth
                loss_dict[f'{stage}/loss_orth'] = loss_orth
            
            # ============ 监督损失 (如果有真实环境标签) ============
            if 'e_true' in batch and self.env_supervised_weight > 0:
                e_true = batch['e_true'].to(device)
                # 让 E 的第一维和真实环境强相关
                if e_true.dim() == 1:
                    e_true = e_true.unsqueeze(-1)
                # 取 E 的前几维（和 e_true 维度匹配）
                e_pred = mu_0[:, :e_true.shape[-1]]
                loss_supervised = F.mse_loss(e_pred, e_true)
                loss_total = loss_total + self.env_supervised_weight * loss_supervised
                loss_dict[f'{stage}/loss_supervised'] = loss_supervised
            
            # 环境统计
            with torch.no_grad():
                loss_dict[f'{stage}/env_mu_norm'] = mu_0.norm(dim=-1).mean()
                std_0 = torch.exp(0.5 * env_out_0['logvar'])
                loss_dict[f'{stage}/env_std_mean'] = std_0.mean()
        
        # ============ Normal 阶段: 标准 Flow Matching ============
        else:
            # 采样噪声和时间
            x_0 = torch.randn_like(x_1)
            t = torch.rand(B, device=device)
            
            # OT 流
            x_t, u_t = self.ot_conditional_flow(x_0, x_1, t)
            
            # ============ 关键改动: Encoder 始终从干净数据编码环境! ============
            # 这是 CaTSG 的核心设计：环境编码与 Flow/Diffusion 解耦
            # Encoder 从干净数据 x_1 编码，Velocity Net 从噪声数据 x_t 预测
            # 注意: 传递完整的 c (B, T, D_c) 用于早期融合
            env_out_1 = self.env_encoder(x_1, c, t=None)
            mu_1 = env_out_1['mu']
            logvar_1 = env_out_1['logvar']
            e = env_out_1['e']  # 使用从干净数据采样的 e
            
            # 为了 KL loss，仍然保留这些引用
            mu_t = mu_1
            logvar_t = logvar_1
            
            # ============ 应用 C dropout ============
            c_dropped = self.apply_c_dropout(c) if stage == "train" else c
            
            # ============ 预测速度 ============
            if loss_config['fm']:
                v_pred = self.velocity_net(x_t, t, c_dropped, e)
                loss_fm = F.mse_loss(v_pred, u_t)
                loss_total = loss_total + loss_fm
                loss_dict[f'{stage}/loss_fm'] = loss_fm
                
                # ============ 自适应方差正则项 (防止 Variance Ratio 下降) ============
                # 只有当方差比偏离 1.0 超过 margin 时才惩罚
                if stage == "train" and self.var_reg_weight > 0:
                    v_pred_var = v_pred.var(dim=0).mean()  # 每个维度的方差的均值
                    v_target_var = u_t.var(dim=0).mean()
                    
                    # 计算方差比
                    var_ratio = v_pred_var / (v_target_var + 1e-8)
                    
                    # 自适应惩罚：只惩罚偏离 [1-margin, 1+margin] 范围的情况
                    margin = 0.1  # 允许 0.9-1.1 范围
                    loss_var_reg = F.relu(torch.abs(var_ratio - 1.0) - margin)
                    
                    loss_total = loss_total + self.var_reg_weight * loss_var_reg
                    loss_dict[f'{stage}/loss_var_reg'] = loss_var_reg
                    loss_dict[f'{stage}/v_pred_var'] = v_pred_var
                    loss_dict[f'{stage}/v_target_var'] = v_target_var
                    loss_dict[f'{stage}/var_ratio'] = var_ratio
            
            # KL 损失
            if loss_config['kl']:
                loss_kl = self.compute_kl_loss(mu_t, logvar_t)
                kl_w = self.get_kl_weight() if stage == "train" else self.kl_weight
                loss_total = loss_total + kl_w * loss_kl
                loss_dict[f'{stage}/loss_kl'] = loss_kl
            
            # Consistency 损失 (Teacher-Student)
            if loss_config['consist']:
                loss_consist = F.mse_loss(mu_t, mu_1.detach())
                loss_total = loss_total + self.consistency_weight * loss_consist
                loss_dict[f'{stage}/loss_consist'] = loss_consist
            
            # 正交损失
            if loss_config['orth']:
                loss_orth = self.compute_orthogonal_loss(e)
                loss_total = loss_total + self.orth_weight * loss_orth
                loss_dict[f'{stage}/loss_orth'] = loss_orth
            
            # ============ Normal 阶段也加入监督损失 (如果有真实环境标签) ============
            if 'e_true' in batch and self.env_supervised_weight > 0:
                e_true = batch['e_true'].to(device)
                if e_true.dim() == 1:
                    e_true = e_true.unsqueeze(-1)
                # 使用干净数据的 mu_1 来监督
                e_pred = mu_1[:, :e_true.shape[-1]]
                loss_supervised = F.mse_loss(e_pred, e_true)
                # Normal 阶段监督损失权重减半，避免过度依赖
                loss_total = loss_total + 0.5 * self.env_supervised_weight * loss_supervised
                loss_dict[f'{stage}/loss_supervised'] = loss_supervised
            
            # VICReg (如果 normal 阶段也使用)
            if loss_config['vicreg'] and stage == "train":
                x_view1, x_view2, _, _ = self.augmentation(x_1, c)
                env_v1 = self.env_encoder(x_view1, c, t=None)
                env_v2 = self.env_encoder(x_view2, c, t=None)
                loss_vicreg, _ = self.compute_vicreg_loss(env_v1['mu'], env_v2['mu'])
                loss_total = loss_total + 0.1 * loss_vicreg  # 降低权重
                loss_dict[f'{stage}/loss_vicreg'] = loss_vicreg
            
            # 环境统计
            with torch.no_grad():
                std = torch.exp(0.5 * logvar_t)
                loss_dict[f'{stage}/env_mu_norm'] = mu_t.norm(dim=-1).mean()
                loss_dict[f'{stage}/env_std_mean'] = std.mean()
                loss_dict[f'{stage}/env_std_min'] = std.min()
        
        loss_dict[f'{stage}/loss'] = loss_total
        
        return loss_total, loss_dict
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.ema is not None:
            self.ema.update()
    
    @contextmanager
    def ema_scope(self, context=None):
        if self.ema is not None:
            self.ema.apply_shadow()
        try:
            yield
        finally:
            if self.ema is not None:
                self.ema.restore()
    
    # ==================== 推理 ====================
    
    @torch.no_grad()
    def sample(
        self,
        c: torch.Tensor,
        batch_size: Optional[int] = None,
        num_steps: int = 100,
        num_mc_samples: Optional[int] = None,
        method: str = 'euler',
        temperature: float = 1.0,
        use_prior: bool = False,
        e_fixed: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        use_ema: bool = True,
        cfg_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        从 P(X | do(C=c)) 采样
        
        使用 Monte Carlo 流积分实现因果干预
        """
        c = self._to_btd(c, self.cond_channels)
        
        if batch_size is None:
            batch_size = c.shape[0]
        if num_mc_samples is None:
            num_mc_samples = self.num_mc_samples_eval
        if cfg_scale is None:
            cfg_scale = self.cfg_scale
        
        device = c.device
        
        ctx = self.ema_scope() if use_ema and self.ema is not None else contextmanager(lambda: iter([None]))()
        
        with ctx:
            x_t = torch.randn(batch_size, self.seq_len, self.channels, device=device)
            intermediates = [x_t.clone()] if return_intermediates else None
            dt = 1.0 / num_steps
            
            for step in tqdm(range(num_steps), desc="Sampling", leave=False):
                t_val = step * dt
                t = torch.full((batch_size,), t_val, device=device)
                
                v = self._get_velocity_cfg(
                    x_t, t, c,
                    num_mc_samples=num_mc_samples,
                    use_prior=use_prior,
                    e_fixed=e_fixed,
                    cfg_scale=cfg_scale,
                )
                
                if method == 'euler':
                    x_t = x_t + v * temperature * dt
                elif method == 'midpoint':
                    t_mid = torch.full((batch_size,), t_val + 0.5 * dt, device=device)
                    x_mid = x_t + v * temperature * (dt / 2)
                    v_mid = self._get_velocity_cfg(
                        x_mid, t_mid, c,
                        num_mc_samples=num_mc_samples,
                        use_prior=use_prior,
                        e_fixed=e_fixed,
                        cfg_scale=cfg_scale,
                    )
                    x_t = x_t + v_mid * temperature * dt
                elif method == 'rk4':
                    k1 = v * temperature
                    t2 = torch.full((batch_size,), t_val + 0.5 * dt, device=device)
                    k2 = self._get_velocity_cfg(x_t + k1 * dt / 2, t2, c, num_mc_samples, use_prior, e_fixed, cfg_scale) * temperature
                    k3 = self._get_velocity_cfg(x_t + k2 * dt / 2, t2, c, num_mc_samples, use_prior, e_fixed, cfg_scale) * temperature
                    t3 = torch.full((batch_size,), t_val + dt, device=device)
                    k4 = self._get_velocity_cfg(x_t + k3 * dt, t3, c, num_mc_samples, use_prior, e_fixed, cfg_scale) * temperature
                    x_t = x_t + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
                
                if return_intermediates:
                    intermediates.append(x_t.clone())
            
            return x_t, intermediates
    
    def _get_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        num_mc_samples: int,
        use_prior: bool = False,
        e_fixed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get velocity from current state.
        
        Args:
            x_t: Current state (B, T, D)
            t: Time (B,)
            c: Full condition sequence (B, T, cond_dim) for early fusion
            num_mc_samples: Number of MC samples
            use_prior: Use prior N(0,1) instead of encoder
            e_fixed: Fixed environment (optional)
        """
        B = x_t.shape[0]
        device = x_t.device
        
        if e_fixed is not None:
            return self.velocity_net(x_t, t, c, e_fixed)
        
        if use_prior:
            if num_mc_samples == 1:
                e = torch.randn(B, self.env_dim, device=device)
                return self.velocity_net(x_t, t, c, e)
            else:
                e_samples = torch.randn(B, num_mc_samples, self.env_dim, device=device)
                return self._mc_velocity(x_t, t, c, e_samples)
        
        # 使用完整的 c 进行早期融合
        if num_mc_samples == 1:
            env_out = self.env_encoder(x_t, c, t=t)
            e = env_out['e']
            return self.velocity_net(x_t, t, c, e)
        else:
            env_out = self.env_encoder(x_t, c, t=t, num_samples=num_mc_samples)
            e_samples = env_out['e']
            return self._mc_velocity(x_t, t, c, e_samples)
    
    def _get_velocity_cfg(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        num_mc_samples: int,
        use_prior: bool = False,
        e_fixed: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        v_cond = self._get_velocity(
            x_t, t, c,
            num_mc_samples=num_mc_samples,
            use_prior=use_prior,
            e_fixed=e_fixed,
        )
        
        if cfg_scale <= 0:
            return v_cond
        
        B = x_t.shape[0]
        device = x_t.device
        
        c_null = self.null_cond.expand(B, -1, -1).to(device)
        e_null = self.null_env.expand(B, -1).to(device)
        v_uncond = self.velocity_net(x_t, t, c_null, e_null)
        
        omega = cfg_scale - 1.0
        v_cfg = (1 + omega) * v_cond - omega * v_uncond
        
        return v_cfg
    
    def _mc_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        e_samples: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D_e = e_samples.shape
        
        x_t_exp = x_t.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, *x_t.shape[1:])
        t_exp = t.unsqueeze(1).expand(-1, N).reshape(B * N)
        c_exp = c.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, *c.shape[1:])
        e_flat = e_samples.reshape(B * N, D_e)
        
        v_flat = self.velocity_net(x_t_exp, t_exp, c_exp, e_flat)
        v_samples = v_flat.reshape(B, N, *v_flat.shape[1:])
        
        return v_samples.mean(dim=1)
    
    @torch.no_grad()
    def encode_environment(
        self, 
        x: torch.Tensor, 
        c: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode environment from data.
        
        Args:
            x: Input sequence (B, T, D)
            c: Condition sequence (B, T, cond_dim) - 完整序列用于早期融合
            t: Optional time (B,)
        """
        x = self._to_btd(x, self.channels)
        
        if c.dim() == 3:
            c = self._to_btd(c, self.cond_channels)
        # 如果 c 是 2D (B, cond_dim)，env_encoder 会自动广播
        
        with self.ema_scope():
            env_out = self.env_encoder(x, c, t=t)
        
        mu = env_out['mu']
        std = torch.exp(0.5 * env_out['logvar'])
        
        return mu, std
    
    # ==================== 优化器配置 ====================
    
    def configure_optimizers(self):
        # 分离参数组：Encoder 使用更低的学习率 (保护 warmup 学到的表示)
        encoder_lr_mult = getattr(self, 'encoder_lr_mult', 0.1)  # Encoder 学习率倍率
        
        param_groups = [
            {
                'params': list(self.env_encoder.parameters()),
                'lr': self.learning_rate * encoder_lr_mult,
                'name': 'encoder'
            },
            {
                'params': list(self.velocity_net.parameters()),
                'lr': self.learning_rate,
                'name': 'velocity_net'
            },
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 100,
            eta_min=self.learning_rate * 0.01,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
    
    # ==================== 数据加载 ====================
    
    def set_datasets(self, train_set, val_set, batch_size: int = 32, num_workers: int = 4):
        self._train_set = train_set
        self._val_set = val_set
        self._batch_size = batch_size
        self._num_workers = num_workers
    
    def train_dataloader(self):
        return DataLoader(
            self._train_set,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self._val_set,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=True,
        )

