"""
Configuration dataclass for training neural operator models.

Provides structured configuration for training hyperparameters,
scheduler settings, and checkpoint management.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json
import os


@dataclass
class TrainingConfig:
    """
    Configuration for training neural operator models.

    Attributes:
        # Optimization
        learning_rate: Initial learning rate (default 1e-3)
        num_epochs: Number of training epochs (default 100)
        batch_size: Batch size for training (default 16)
        weight_decay: L2 regularization weight (default 1e-4)
        optimizer_type: Type of optimizer ('adam', 'adamw', 'soap', default 'adam')
                       Use 'adam' for DeepONet & FNO, 'soap' for UNet

        # SOAP Optimizer Parameters (only used when optimizer_type='soap')
        soap_betas: Beta coefficients for SOAP (default (0.95, 0.95))
        soap_shampoo_beta: Beta for preconditioner moving average (default -1, uses betas[1])
        soap_eps: Epsilon for numerical stability (default 1e-8)
        soap_precondition_frequency: How often to update preconditioner (default 10)
        soap_max_precond_dim: Maximum preconditioner dimension (default 10000)
        soap_merge_dims: Whether to merge dimensions (default False)
        soap_precondition_1d: Whether to precondition 1D gradients (default False)
        soap_normalize_grads: Whether to normalize gradients per layer (default False)

        # Scheduler
        scheduler_type: Type of LR scheduler ('cosine', 'plateau', or 'none')
        cosine_t_max: T_max for CosineAnnealingLR (default: num_epochs * steps_per_epoch)
        cosine_eta_min: Minimum LR for cosine annealing (default 0.0, decays to 0)
        plateau_factor: Factor to reduce LR on plateau (default 0.5)
        plateau_patience: Epochs to wait before reducing LR (default 20)
        plateau_min_lr: Minimum LR for plateau scheduler (default 1e-7)

        # Evaluation
        eval_metrics: List of metrics to compute during validation
        eval_frequency: Evaluate every N epochs (default 1)

        # Checkpointing
        checkpoint_dir: Directory to save checkpoints (default 'checkpoints/')
        save_best: Whether to save best model based on val_loss (default True)
        save_latest: Whether to save latest model after each epoch (default True)
        save_frequency: Save checkpoint every N epochs (default 10, if save_latest=False)

        # Device
        device: Device to use for training ('cuda' or 'cpu', default 'cuda')
        num_workers: Number of dataloader workers (default 4)

        # Mixed Precision Training
        use_amp: Whether to use automatic mixed precision (FP16/BF16) for memory reduction (default True)

        # Logging
        log_frequency: Log training metrics every N batches (default 10)
        verbose: Whether to print detailed logs (default True)
    """

    # Optimization
    learning_rate: float = 1e-3  # Reference CausalityDeepONet uses 1e-3 with Adam
    num_epochs: int = 100
    batch_size: int = 16
    weight_decay: float = 1e-4
    optimizer_type: str = 'adam'  # 'adam' for DeepONet & FNO, 'soap' for UNet
    max_grad_norm: float = 1.0  # Gradient clipping for stability (prevents NaN with large models)

    # SOAP Optimizer Parameters (only used when optimizer_type='soap')
    soap_betas: tuple = (0.95, 0.95)
    soap_shampoo_beta: float = -1.0  # -1 means use betas[1]
    soap_eps: float = 1e-8
    soap_precondition_frequency: int = 10
    soap_max_precond_dim: int = 10000
    soap_merge_dims: bool = False
    soap_precondition_1d: bool = False
    soap_normalize_grads: bool = False

    # Scheduler
    scheduler_type: str = 'cosine'  # 'cosine', 'plateau', or 'none'
    cosine_t_max: Optional[int] = None  # Will be set to num_epochs * steps_per_epoch
    cosine_eta_min: float = 0.0  # Decay learning rate to 0
    plateau_factor: float = 0.5
    plateau_patience: int = 20
    plateau_min_lr: float = 1e-7

    # Evaluation
    eval_metrics: List[str] = field(
        default_factory=lambda: ['mse', 'spectrum_error']
    )
    eval_frequency: int = 1

    # Checkpointing
    checkpoint_dir: str = 'checkpoints/'
    save_best: bool = True
    save_latest: bool = True
    save_frequency: int = 10

    # Device
    device: str = 'cuda'
    num_workers: int = 4

    # Mixed Precision Training
    use_amp: bool = False  # Disabled by default due to numerical instability issues with FP16
                            # Can cause NaN with large models (DeepONet 567K params)
                            # Set to True if you have GPU memory constraints and stable training

    # Logging
    log_frequency: int = 10
    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """
        Create config from dictionary.

        Args:
            config_dict: Dictionary with config parameters

        Returns:
            TrainingConfig instance
        """
        return cls(**config_dict)

    def save(self, path: str) -> None:
        """
        Save configuration to JSON file.

        Args:
            path: Output JSON file path
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """
        Load configuration from JSON file.

        Args:
            path: Input JSON file path

        Returns:
            TrainingConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        # Check positive values
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        # Check optimizer type
        valid_optimizers = ['adam', 'adamw', 'soap']
        if self.optimizer_type not in valid_optimizers:
            raise ValueError(
                f"optimizer_type must be one of {valid_optimizers}, "
                f"got '{self.optimizer_type}'"
            )

        # Check scheduler type
        valid_schedulers = ['cosine', 'plateau', 'none']
        if self.scheduler_type not in valid_schedulers:
            raise ValueError(
                f"scheduler_type must be one of {valid_schedulers}, "
                f"got '{self.scheduler_type}'"
            )

        # Check device
        valid_devices = ['cuda', 'cpu']
        if self.device not in valid_devices:
            raise ValueError(
                f"device must be one of {valid_devices}, "
                f"got '{self.device}'"
            )

        # Check metrics
        valid_metrics = ['mse', 'spectrum_error']
        for metric in self.eval_metrics:
            if metric not in valid_metrics:
                raise ValueError(
                    f"Unknown metric '{metric}'. "
                    f"Valid options: {valid_metrics}"
                )

    def __repr__(self) -> str:
        """String representation showing key parameters."""
        return (
            f"TrainingConfig(\n"
            f"  Optimization:\n"
            f"    optimizer_type='{self.optimizer_type}',\n"
            f"    learning_rate={self.learning_rate},\n"
            f"    num_epochs={self.num_epochs},\n"
            f"    batch_size={self.batch_size},\n"
            f"    weight_decay={self.weight_decay}\n"
            f"  Scheduler:\n"
            f"    scheduler_type='{self.scheduler_type}',\n"
            f"    cosine_eta_min={self.cosine_eta_min}\n"
            f"  Evaluation:\n"
            f"    eval_metrics={self.eval_metrics}\n"
            f"  Checkpointing:\n"
            f"    checkpoint_dir='{self.checkpoint_dir}',\n"
            f"    save_best={self.save_best},\n"
            f"    save_latest={self.save_latest}\n"
            f"  Device: {self.device}\n"
            f")"
        )


def create_default_config(**kwargs) -> TrainingConfig:
    """
    Create training configuration with optional parameter overrides.

    Args:
        **kwargs: Configuration parameters to override

    Returns:
        TrainingConfig instance with defaults + overrides

    Example:
        >>> config = create_default_config(num_epochs=200, learning_rate=1e-4)
        >>> config = create_default_config(device='cpu', batch_size=32)
    """
    config = TrainingConfig(**kwargs)
    config.validate()
    return config
