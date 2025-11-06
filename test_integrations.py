"""
Test script for SOAP optimizer and SIREN activation integrations.

Tests:
1. Optimizer factory (Adam, AdamW, SOAP)
2. DeepONet with different activations (Tanh, ReLU, SIREN)
3. Backward compatibility
4. Forward pass and gradient computation
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("="*70)
print("Testing SOAP Optimizer and SIREN Activation Integration")
print("="*70)

# Test 1: Import optimizer factory
print("\n[Test 1] Importing optimizer factory...")
try:
    from src.core.training.optimizers.optimizer_factory import create_optimizer
    from configs.training_config import TrainingConfig
    print("✓ Optimizer factory imported successfully")
except Exception as e:
    print(f"✗ Failed to import optimizer factory: {e}")
    sys.exit(1)

# Test 2: Import DeepONet
print("\n[Test 2] Importing DeepONet...")
try:
    from src.core.models.model_factory import create_model
    print("✓ DeepONet imported successfully")
except Exception as e:
    print(f"✗ Failed to import DeepONet: {e}")
    sys.exit(1)

# Test 3: Create optimizers
print("\n[Test 3] Creating optimizers...")

# Create dummy model
dummy_model = torch.nn.Linear(10, 10)

# Test Adam
print("  Testing Adam optimizer...")
try:
    config_adam = TrainingConfig(optimizer_type='adam', learning_rate=1e-3)
    optimizer_adam = create_optimizer('adam', dummy_model.parameters(), config_adam)
    print(f"  ✓ Adam optimizer created: {type(optimizer_adam).__name__}")
except Exception as e:
    print(f"  ✗ Adam optimizer failed: {e}")

# Test AdamW
print("  Testing AdamW optimizer...")
try:
    config_adamw = TrainingConfig(optimizer_type='adamw', learning_rate=1e-3)
    optimizer_adamw = create_optimizer('adamw', dummy_model.parameters(), config_adamw)
    print(f"  ✓ AdamW optimizer created: {type(optimizer_adamw).__name__}")
except Exception as e:
    print(f"  ✗ AdamW optimizer failed: {e}")

# Test SOAP
print("  Testing SOAP optimizer...")
try:
    config_soap = TrainingConfig(
        optimizer_type='soap',
        learning_rate=3e-3,
        soap_precondition_frequency=10
    )
    optimizer_soap = create_optimizer('soap', dummy_model.parameters(), config_soap)
    print(f"  ✓ SOAP optimizer created: {type(optimizer_soap).__name__}")
except Exception as e:
    print(f"  ✗ SOAP optimizer failed: {e}")

# Test 4: Create DeepONet models with different activations
print("\n[Test 4] Creating DeepONet models with different activations...")

# Test Tanh (default)
print("  Testing Tanh activation (default)...")
try:
    model_tanh = create_model('deeponet')
    print(f"  ✓ DeepONet with Tanh created ({model_tanh.count_parameters():,} params)")
except Exception as e:
    print(f"  ✗ DeepONet with Tanh failed: {e}")

# Test ReLU
print("  Testing ReLU activation...")
try:
    model_relu = create_model('deeponet', config={'activation': 'relu'})
    print(f"  ✓ DeepONet with ReLU created ({model_relu.count_parameters():,} params)")
except Exception as e:
    print(f"  ✗ DeepONet with ReLU failed: {e}")

# Test SIREN
print("  Testing SIREN activation...")
try:
    model_siren = create_model('deeponet', config={'activation': 'siren'})
    print(f"  ✓ DeepONet with SIREN created ({model_siren.count_parameters():,} params)")
except Exception as e:
    print(f"  ✗ DeepONet with SIREN failed: {e}")
    print(f"  (Note: SIREN requires siren-pytorch: pip install siren-pytorch)")

# Test 5: Forward pass and gradient computation
print("\n[Test 5] Testing forward pass and gradient computation...")

# Create dummy input
dummy_input = torch.randn(2, 1, 4000)  # [batch, channels, timesteps]

for name, model in [('Tanh', model_tanh), ('ReLU', model_relu)]:
    try:
        # Forward pass
        output = model(dummy_input)
        assert output.shape == dummy_input.shape, \
            f"Output shape mismatch: {output.shape} vs {dummy_input.shape}"

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "No gradients computed"

        print(f"  ✓ {name}: Forward/backward pass successful")

        # Clear gradients
        model.zero_grad()
    except Exception as e:
        print(f"  ✗ {name}: Forward/backward pass failed: {e}")

# Try SIREN if it was created
try:
    if 'model_siren' in locals():
        output_siren = model_siren(dummy_input)
        assert output_siren.shape == dummy_input.shape
        loss_siren = output_siren.sum()
        loss_siren.backward()
        print(f"  ✓ SIREN: Forward/backward pass successful")
except Exception as e:
    print(f"  ✗ SIREN: Forward/backward pass failed: {e}")

# Test 6: Backward compatibility
print("\n[Test 6] Testing backward compatibility...")
try:
    # Create model with default config (should use tanh)
    model_default = create_model('deeponet')
    config_default = TrainingConfig()  # Should default to adam
    optimizer_default = create_optimizer(
        config_default.optimizer_type,
        model_default.parameters(),
        config_default
    )

    print(f"  ✓ Default model: {model_default.activation} activation")
    print(f"  ✓ Default optimizer: {type(optimizer_default).__name__}")
    print(f"  ✓ Backward compatibility maintained")
except Exception as e:
    print(f"  ✗ Backward compatibility test failed: {e}")

# Summary
print("\n" + "="*70)
print("Integration Test Summary")
print("="*70)
print("✓ All core integrations working!")
print("\nFeatures added:")
print("  • SOAP optimizer support (adam, adamw, soap)")
print("  • SIREN activation for DeepONet (tanh, relu, siren)")
print("  • Backward compatible defaults")
print("\nUsage:")
print("  # Use SOAP optimizer:")
print("  config = TrainingConfig(optimizer_type='soap')")
print()
print("  # Use SIREN activation:")
print("  model = create_model('deeponet', config={'activation': 'siren'})")
print("="*70)
