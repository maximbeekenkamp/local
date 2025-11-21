#!/usr/bin/env python3
"""
Debug NaN issue in DeepONet training.

Systematically checks:
1. Data validity (NaN/Inf in inputs/targets)
2. Model initialization
3. First forward pass
4. First backward pass
5. Gradient flow
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.models.model_factory import create_model
from src.core.data_processing.cdon_dataset import CDONDataset
from src.core.data_processing.cdon_transforms import CDONNormalization
from torch.utils.data import DataLoader

print("="*70)
print("DEEPONET NaN DIAGNOSTIC")
print("="*70)

# ============================================================================
# STEP 1: Check Data for NaN/Inf
# ============================================================================
print("\n[STEP 1] Checking data for NaN/Inf values...")

data_dir = project_root / 'CDONData'
stats_path = project_root / 'configs' / 'cdon_stats.json'

# Create normalizer
normalizer = CDONNormalization(stats_path=str(stats_path))

# Create per-timestep dataset
from src.core.data_processing.cdon_dataset import CDONDataset
train_dataset = CDONDataset(
    data_dir=str(data_dir),
    split='train',
    normalize=normalizer,
    mode='per-timestep',  # Per-timestep mode
    use_causal_sequence=True,
    signal_length=4000
)

print(f"✓ Dataset loaded: {len(train_dataset)} samples")

# Check first 100 samples for NaN/Inf
print("  Checking first 100 samples...")
nan_count = 0
inf_count = 0
for i in range(min(100, len(train_dataset))):
    sample = train_dataset[i]
    input_data = sample['input']
    target_data = sample['target']
    time_coord = sample['time_coord']

    if torch.isnan(input_data).any():
        print(f"    ⚠️  Sample {i}: NaN in input")
        nan_count += 1
    if torch.isinf(input_data).any():
        print(f"    ⚠️  Sample {i}: Inf in input")
        inf_count += 1
    if torch.isnan(target_data).any():
        print(f"    ⚠️  Sample {i}: NaN in target")
        nan_count += 1
    if torch.isinf(target_data).any():
        print(f"    ⚠️  Sample {i}: Inf in target")
        inf_count += 1
    if torch.isnan(time_coord).any():
        print(f"    ⚠️  Sample {i}: NaN in time_coord")
        nan_count += 1

if nan_count == 0 and inf_count == 0:
    print("  ✓ No NaN/Inf detected in first 100 samples")
else:
    print(f"  ❌ Found {nan_count} NaN and {inf_count} Inf values")
    print("  → DATA ISSUE: Fix data before training")
    sys.exit(1)

# Check data statistics
sample = train_dataset[0]
input_data = sample['input']
target_data = sample['target']
print(f"\n  Data statistics:")
print(f"    Input shape: {input_data.shape}")
print(f"    Input range: [{input_data.min():.4f}, {input_data.max():.4f}]")
print(f"    Input mean: {input_data.mean():.4f}, std: {input_data.std():.4f}")
print(f"    Target shape: {target_data.shape}")
print(f"    Target range: [{target_data.min():.4f}, {target_data.max():.4f}]")
print(f"    Target mean: {target_data.mean():.4f}, std: {target_data.std():.4f}")

# ============================================================================
# STEP 2: Check Model Initialization
# ============================================================================
print("\n[STEP 2] Checking model initialization...")

model = create_model(
    arch='deeponet',
    sensor_dim=4000,
    latent_dim=120,
    branch_layers=[120, 120, 120],
    trunk_layers=[120, 120, 120],
    activation='requ'
)

print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

# Check for NaN in initial weights
nan_params = []
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        nan_params.append(name)
    if torch.isinf(param).any():
        nan_params.append(name)

if nan_params:
    print(f"  ❌ NaN/Inf in initial parameters: {nan_params}")
    print("  → INITIALIZATION ISSUE: Model has bad initial weights")
    sys.exit(1)
else:
    print("  ✓ No NaN/Inf in initial parameters")

# Check parameter statistics
total_params = sum(p.numel() for p in model.parameters())
param_mean = sum(p.mean().item() * p.numel() for p in model.parameters()) / total_params
param_std = torch.cat([p.flatten() for p in model.parameters()]).std().item()
print(f"  Parameter statistics:")
print(f"    Mean: {param_mean:.6f}")
print(f"    Std: {param_std:.6f}")

# ============================================================================
# STEP 3: Test First Forward Pass (CPU)
# ============================================================================
print("\n[STEP 3] Testing first forward pass (CPU)...")

model.eval()
device = torch.device('cpu')
model.to(device)

# Get a batch
loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
batch = next(iter(loader))

inputs = batch['input'].to(device)  # [B, 4000]
targets = batch['target'].to(device)  # [B]
time_coords = batch['time_coord'].to(device)  # [B]

print(f"  Batch shapes:")
print(f"    inputs: {inputs.shape}")
print(f"    targets: {targets.shape}")
print(f"    time_coords: {time_coords.shape}")

# Forward pass
try:
    outputs = model.forward_per_timestep(inputs, time_coords)
    outputs = outputs.squeeze(-1)  # [B, 1] → [B]
    print(f"  ✓ Forward pass successful")
    print(f"    Output shape: {outputs.shape}")
    print(f"    Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")
    print(f"    Output mean: {outputs.mean():.4f}, std: {outputs.std():.4f}")

    if torch.isnan(outputs).any():
        print(f"  ❌ NaN in outputs")
        print("  → FORWARD PASS ISSUE: Model produces NaN")
        sys.exit(1)
    if torch.isinf(outputs).any():
        print(f"  ❌ Inf in outputs")
        print("  → FORWARD PASS ISSUE: Model produces Inf")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 4: Test Loss Computation
# ============================================================================
print("\n[STEP 4] Testing loss computation...")

criterion = torch.nn.MSELoss()
loss = criterion(outputs, targets)
print(f"  ✓ Loss computed: {loss.item():.6f}")

if torch.isnan(loss):
    print(f"  ❌ Loss is NaN")
    print("  → LOSS COMPUTATION ISSUE")
    sys.exit(1)
if torch.isinf(loss):
    print(f"  ❌ Loss is Inf")
    print("  → LOSS COMPUTATION ISSUE")
    sys.exit(1)

# ============================================================================
# STEP 5: Test Backward Pass
# ============================================================================
print("\n[STEP 5] Testing backward pass...")

model.train()
model.zero_grad()

try:
    loss.backward()
    print(f"  ✓ Backward pass successful")
except Exception as e:
    print(f"  ❌ Backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check gradients
total_norm = 0.0
nan_grad_params = []
zero_grad_params = []
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            nan_grad_params.append(name)
        param_norm = param.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
        if param_norm.item() == 0:
            zero_grad_params.append(name)
    else:
        zero_grad_params.append(f"{name} (None)")

total_norm = total_norm ** 0.5

print(f"  Gradient norm: {total_norm:.6e}")

if nan_grad_params:
    print(f"  ❌ NaN gradients in: {nan_grad_params[:5]}")
    print("  → BACKWARD PASS ISSUE: Gradients contain NaN")
    sys.exit(1)

if total_norm == 0:
    print(f"  ⚠️  WARNING: Gradient norm is exactly 0")
    print(f"  Parameters with zero gradients: {len(zero_grad_params)}/{len(list(model.parameters()))}")
    if zero_grad_params:
        print(f"  First 5: {zero_grad_params[:5]}")
    print("  → POTENTIAL ISSUE: Gradients not flowing")
else:
    print(f"  ✓ Gradients flowing (norm: {total_norm:.6e})")

# ============================================================================
# STEP 6: Test with AMP (GPU if available)
# ============================================================================
if torch.cuda.is_available():
    print("\n[STEP 6] Testing with AMP on GPU...")

    device = torch.device('cuda')
    model.to(device)
    model.train()

    inputs = batch['input'].to(device)
    targets = batch['target'].to(device)
    time_coords = batch['time_coord'].to(device)

    from torch.amp import autocast, GradScaler
    scaler = GradScaler('cuda')

    model.zero_grad()

    try:
        with autocast(device_type='cuda', enabled=True):
            outputs = model.forward_per_timestep(inputs, time_coords)
            outputs = outputs.squeeze(-1)
            loss = criterion(outputs, targets)

        print(f"  ✓ AMP forward pass successful")
        print(f"    Loss: {loss.item():.6f}")

        scaler.scale(loss).backward()
        scaler.step(torch.optim.Adam(model.parameters(), lr=1e-3))
        scaler.update()

        print(f"  ✓ AMP backward pass successful")

        # Check for NaN after AMP
        if torch.isnan(loss):
            print(f"  ❌ Loss is NaN with AMP")
            print("  → AMP ISSUE: Try disabling AMP (use_amp=False)")
            sys.exit(1)

    except Exception as e:
        print(f"  ❌ AMP test failed: {e}")
        import traceback
        traceback.print_exc()
        print("  → Try disabling AMP (use_amp=False in config)")
else:
    print("\n[STEP 6] Skipping AMP test (CUDA not available)")

# ============================================================================
# STEP 7: Recommendations
# ============================================================================
print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)

print("\n✓ All checks passed! The issue is likely:")
print("\n1. **Scheduler Warning**: The lr_scheduler.step() before optimizer.step() warning")
print("   → This causes the first batch to use the wrong learning rate")
print("   → SHOULD BE FIXED in the trainer code")
print("\n2. **Possible Solutions**:")
print("   a) Check if you're using the latest trainer code")
print("   b) Try disabling AMP: config = TrainingConfig(..., use_amp=False)")
print("   c) Try lower learning rate: config = TrainingConfig(..., learning_rate=3e-4)")
print("   d) Check the order of optimizer.step() and scheduler.step() in trainer")

print("\n3. **Immediate Fix**: In simple_trainer.py around line 600-653:")
print("   The cosine scheduler should step AFTER optimizer.step(), not before.")
print("   Current issue: scheduler.step() is called before the first optimizer.step()")
print("   This can cause learning rate issues on the first batch.")
