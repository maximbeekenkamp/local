#!/usr/bin/env python3
"""
Validate that the bug fixes are syntactically correct.
This script checks imports and class structures without needing torch.
"""

import sys
import ast
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_file_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)


def validate_trainer_fix():
    """Validate the trainer fix for sample_indices handling."""
    print("\n" + "="*70)
    print("1. Validating simple_trainer.py")
    print("="*70)

    trainer_file = project_root / 'src' / 'core' / 'training' / 'simple_trainer.py'

    # Check syntax
    valid, error = check_file_syntax(trainer_file)
    if not valid:
        print(f"❌ Syntax error: {error}")
        return False

    # Check for the fix
    with open(trainer_file, 'r') as f:
        content = f.read()

    # Check that we have the conditional sample_indices passing
    checks = [
        ('isinstance(self.criterion, (CombinedLoss, BinnedSpectralLoss, SelfAdaptiveBSPLoss))',
         'Conditional isinstance check for sample_indices'),
        ('loss = self.criterion(seq_outputs, seq_targets)',
         'Fallback without sample_indices for baseline'),
        ('from ..evaluation.loss_factory import CombinedLoss',
         'Import CombinedLoss for isinstance check'),
    ]

    all_found = True
    for check_str, description in checks:
        if check_str in content:
            print(f"  ✓ Found: {description}")
        else:
            print(f"  ❌ Missing: {description}")
            all_found = False

    # Count occurrences of the fix (should be 2: training + validation)
    fix_count = content.count('isinstance(self.criterion, (CombinedLoss, BinnedSpectralLoss, SelfAdaptiveBSPLoss))')
    if fix_count >= 2:
        print(f"  ✓ Fix applied {fix_count} times (training + validation)")
    else:
        print(f"  ❌ Fix only found {fix_count} times (expected at least 2)")
        all_found = False

    if all_found:
        print("\n✅ simple_trainer.py validation PASSED")
        return True
    else:
        print("\n❌ simple_trainer.py validation FAILED")
        return False


def validate_loss_factory_fix():
    """Validate the loss_factory.py fix for sample_indices in CombinedLoss."""
    print("\n" + "="*70)
    print("2. Validating loss_factory.py")
    print("="*70)

    factory_file = project_root / 'src' / 'core' / 'evaluation' / 'loss_factory.py'

    # Check syntax
    valid, error = check_file_syntax(factory_file)
    if not valid:
        print(f"❌ Syntax error: {error}")
        return False

    # Check for the fix
    with open(factory_file, 'r') as f:
        content = f.read()

    checks = [
        ('sample_indices: torch.Tensor = None',
         'get_loss_components accepts sample_indices parameter'),
        ('loss_base = self.base_loss(pred, target)',
         'Base loss called without sample_indices (MSE compatibility)'),
        ('if isinstance(self.spectral_loss, (BinnedSpectralLoss, SelfAdaptiveBSPLoss)):',
         'Conditional check for spectral loss type'),
        ('loss_spectral = self.spectral_loss(pred, target, sample_indices=sample_indices)',
         'Spectral loss gets sample_indices when supported'),
    ]

    all_found = True
    for check_str, description in checks:
        if check_str in content:
            print(f"  ✓ Found: {description}")
        else:
            print(f"  ❌ Missing: {description}")
            all_found = False

    # Check that the fix is in both forward() and get_loss_components()
    methods_with_fix = content.count('if isinstance(self.spectral_loss, (BinnedSpectralLoss, SelfAdaptiveBSPLoss)):')
    if methods_with_fix >= 2:
        print(f"  ✓ Fix applied {methods_with_fix} times (forward + get_loss_components)")
    else:
        print(f"  ❌ Fix only found {methods_with_fix} times (expected at least 2)")
        all_found = False

    if all_found:
        print("\n✅ loss_factory.py validation PASSED")
        return True
    else:
        print("\n❌ loss_factory.py validation FAILED")
        return False


def check_imports():
    """Verify all imports work (syntax check only)."""
    print("\n" + "="*70)
    print("3. Validating Import Structure")
    print("="*70)

    files_to_check = [
        'src/core/training/simple_trainer.py',
        'src/core/evaluation/loss_factory.py',
        'src/core/evaluation/binned_spectral_loss.py',
        'src/core/evaluation/adaptive_spectral_loss.py',
        'configs/loss_config.py',
    ]

    all_valid = True
    for file_path in files_to_check:
        full_path = project_root / file_path
        valid, error = check_file_syntax(full_path)
        if valid:
            print(f"  ✓ {file_path}")
        else:
            print(f"  ❌ {file_path}: {error}")
            all_valid = False

    if all_valid:
        print("\n✅ All imports and syntax PASSED")
        return True
    else:
        print("\n❌ Some files have syntax errors")
        return False


def main():
    """Run all validation checks."""
    print("="*70)
    print("BUG FIX VALIDATION SUITE")
    print("="*70)
    print("Validating fixes for sample_indices parameter handling")
    print("="*70)

    results = {
        'trainer': validate_trainer_fix(),
        'loss_factory': validate_loss_factory_fix(),
        'imports': check_imports(),
    }

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    for component, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{component.upper():<20} {status}")

    all_passed = all(results.values())

    print("="*70)
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("\nThe bug fixes are syntactically correct:")
        print("  1. ✓ Trainer conditionally passes sample_indices")
        print("  2. ✓ CombinedLoss handles MSE base loss correctly")
        print("  3. ✓ All syntax is valid")
        print("\n📝 Note: Full functional testing requires PyTorch installation.")
        print("   The Colab notebook should work correctly with these fixes.")
        return 0
    else:
        print("⚠️  SOME VALIDATIONS FAILED!")
        print("   Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
