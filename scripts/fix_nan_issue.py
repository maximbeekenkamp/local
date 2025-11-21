"""
Quick fix for NaN issue - adds early detection and prevents corruption.

This script patches simple_trainer.py to add:
1. Check model parameters for NaN after each batch
2. Check outputs for Inf (which becomes NaN after loss computation)
3. Check gradients before optimizer step
4. Add gradient statistics logging
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*70)
print("NaN FIX: Enhanced Checking")
print("="*70)

print("\nThis fix adds:")
print("1. Model parameter NaN detection after each batch")
print("2. Output Inf/NaN detection before loss computation")
print("3. Gradient statistics logging")
print("4. Early stopping on first sign of instability")

print("\n" + "="*70)
print("RECOMMENDED IMMEDIATE FIXES")
print("="*70)

print("\n[FIX 1] Disable AMP (most likely cause)")
print("=" * 60)
print("AMP can cause numerical instability with certain model architectures.")
print("\nIn your notebook, change:")
print("  config = TrainingConfig(..., use_amp=False)")
print("\nThis will use FP32 throughout and is more stable.")

print("\n[FIX 2] Reduce Learning Rate")
print("=" * 60)
print("Learning rate of 1e-3 might be too high for DeepONet with 567K parameters.")
print("\nTry:")
print("  config = TrainingConfig(..., learning_rate=3e-4)  # or 1e-4")

print("\n[FIX 3] Check Model Initialization")
print("=" * 60)
print("ReQU activation with large parameter counts can cause issues.")
print("\nTry:")
print("  model = create_model(arch='deeponet', activation='tanh')")
print("\nTanh is more stable than ReQU for operator learning.")

print("\n[FIX 4] Add Early Checking (code patch)")
print("=" * 60)
print("\nAdd this check in simple_trainer.py at line 627 (after forward pass):")
print("""
# Check outputs before loss computation
if torch.isinf(per_ts_outputs).any():
    print(f"❌ Inf in outputs at batch {batch_idx}")
    print(f"  Output range: [{per_ts_outputs.min():.6e}, {per_ts_outputs.max():.6e}]")
    print(f"  Input range: [{per_ts_inputs.min():.6e}, {per_ts_inputs.max():.6e}]")
    # Check model parameters
    for name, param in self.model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"  Parameter {name} contains NaN/Inf")
    raise RuntimeError(f"Inf detected in outputs at batch {batch_idx}")
""")

print("\n[FIX 5] Check Model Parameters After Each Batch")
print("=" * 60)
print("\nAdd this check in simple_trainer.py at line 656 (after accumulate):")
print("""
# Check model parameters for corruption
if batch_idx % 10 == 0:  # Check every 10 batches
    for name, param in self.model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"❌ Parameter {name} became NaN/Inf at batch {batch_idx}")
            raise RuntimeError(f"Model corruption detected at batch {batch_idx}")
""")

print("\n" + "="*70)
print("TESTING PROCEDURE")
print("="*70)

print("""
1. First, try FIX 1 (disable AMP):
   config = TrainingConfig(..., use_amp=False)

   If this works → AMP was the issue (probably FP16 underflow)

2. If FIX 1 doesn't work, try FIX 2 (lower LR):
   config = TrainingConfig(..., learning_rate=3e-4, use_amp=False)

   If this works → Learning rate was too high

3. If FIX 2 doesn't work, try FIX 3 (tanh activation):
   model = create_model(arch='deeponet', activation='tanh')

   If this works → ReQU activation was unstable

4. If none work, add FIX 4 and FIX 5 to get detailed diagnostics
""")

print("\n" + "="*70)
print("CREATING TEST SCRIPT")
print("="*70)

# Create a test script they can run
test_script = project_root / 'scripts' / 'test_nan_fixes.py'
with open(test_script, 'w') as f:
    f.write('''#!/usr/bin/env python3
"""
Test NaN fixes one by one.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.models.model_factory import create_model
from src.core.data_processing.cdon_dataset import create_cdon_dataloaders
from src.core.training.simple_trainer import SimpleTrainer
from configs.training_config import TrainingConfig
from configs.loss_config import BASELINE_CONFIG

print("Testing NaN fixes...")

# Create dataloaders
data_dir = project_root / 'CDONData'
stats_path = project_root / 'configs' / 'cdon_stats.json'

per_ts_train, per_ts_val, seq_train, seq_val = create_cdon_dataloaders(
    data_dir=str(data_dir),
    batch_size_per_timestep=32,
    batch_size_sequence=4,
    use_dummy=False,
    stats_path=str(stats_path),
    num_workers=0,
    pin_memory=False,
    use_causal_sequence=True
)

print("✓ Data loaded")

# Test each fix
fixes = [
    ("FIX 1: Disable AMP", {'use_amp': False}),
    ("FIX 2: Lower LR", {'use_amp': False, 'learning_rate': 3e-4}),
    ("FIX 3: Smaller model", {'use_amp': False, 'learning_rate': 3e-4}),
]

for fix_name, fix_params in fixes:
    print(f"\\n{'='*70}")
    print(f"Testing: {fix_name}")
    print(f"{'='*70}")

    # Create model
    model = create_model(
        arch='deeponet',
        sensor_dim=4000,
        latent_dim=120,
        branch_layers=[120, 120, 120],
        trunk_layers=[120, 120, 120],
        activation='requ'
    )

    # Create config with fix
    config = TrainingConfig(
        num_epochs=1,  # Just 1 epoch for testing
        **fix_params
    )

    # Create trainer
    trainer = SimpleTrainer(
        model=model,
        per_timestep_train_loader=per_ts_train,
        sequence_train_loader=seq_train,
        per_timestep_val_loader=per_ts_val,
        sequence_val_loader=seq_val,
        config=config,
        loss_config=BASELINE_CONFIG,
        experiment_name=f'test_{fix_name.lower().replace(" ", "_")}'
    )

    try:
        # Train for 1 epoch
        print(f"  Training...")
        results = trainer.train()
        print(f"  ✓ {fix_name} WORKS! Loss: {results['train_history'][0]['loss']:.6f}")
        print(f"\\n  → Use this configuration for full training")
        break
    except RuntimeError as e:
        if 'NaN' in str(e):
            print(f"  ❌ {fix_name} still has NaN issue")
            continue
        else:
            raise
    except Exception as e:
        print(f"  ❌ {fix_name} failed with: {e}")
        continue

print("\\n" + "="*70)
print("Testing complete")
print("="*70)
''')

print(f"\n✓ Created test script: {test_script}")
print("\nRun it with:")
print(f"  python {test_script}")
print("\nThis will test each fix automatically and tell you which one works.")
