# BSP Loss Debugging Analysis

## Problem Statement

After multiple attempts to fix BSP loss integration with DeepONet and UNet architectures, catastrophic loss values persist:

### Current Results (After All Fixes)

**DeepONet:**
- BASELINE: Val Loss = 19.98, Field Error = 2944.54, Spectrum Error = 1.08
- BSP: Val Loss = 11,770.88 (**still catastrophic**), Field Error = 2947.33, Spectrum Error = 0.33
- SA-BSP: Val Loss = 1,342,918.04 (**catastrophic**), Field Error = 2946.35, Spectrum Error = 0.36

**UNet:**
- BASELINE: Val Loss = 8.54, Field Error = 280.34, Spectrum Error = 2.51
- BSP: Val Loss = 65,936,243.33 (**catastrophic**), Field Error = 2477.22, Spectrum Error = 0.87
- SA-BSP: Val Loss = 42,107,646.00 (**catastrophic**), Field Error = 2469.11, Spectrum Error = 0.74

**FNO:**
- BASELINE: Val Loss = 0.997, Field Error = 0.99, Spectrum Error = 0.19 ✓
- BSP: Val Loss = 1.08, Field Error = 1.00, Spectrum Error = 0.72 (**WORSE** spectrum!)
- SA-BSP: Val Loss = 1.07, Field Error = 1.00, Spectrum Error = 0.81 (**WORSE** spectrum!)

### Critical Observations

1. **DeepONet and UNet have VERY HIGH field errors** (~280-2950) across ALL loss types, even baseline
   - This indicates the models aren't learning the task properly at all
   - BSP is making already-bad models even worse

2. **FNO works but BSP makes spectrum WORSE** (0.19 → 0.72)
   - FNO baseline already has good spectrum error (0.19)
   - Adding BSP increases spectrum error by 3.8x
   - Contradicts the paper's claim that BSP improves spectral accuracy

3. **BSP reduces spectrum error for DeepONet/UNet** (1.08 → 0.33, 2.51 → 0.87)
   - But at catastrophic cost to total loss
   - Models are "cheating" - optimizing spectrum at expense of actual predictions

---

## History of Attempted Fixes

### Attempt 1: LayerNorm + Reduced SIREN w0 (Initial Fix)

**What we did:**
- Added `nn.LayerNorm(sensor_dim)` to DeepONet output normalization
- Reduced SIREN `w0_initial` from 30 → 10
- Added `nn.GroupNorm(1, out_channels)` to UNet output

**Why we thought it would work:**
- LayerNorm normalizes outputs to mean=0, std=1
- Lower w0 produces more stable SIREN activations (less extreme sin outputs)
- GroupNorm constrains UNet output scale
- Paper says BSP is "architecture agnostic" - should handle normalized outputs

**Why it didn't work:**
- **LayerNorm operates in TIME DOMAIN** but BSP operates in FREQUENCY DOMAIN
- LayerNorm shifts mean to 0 (removes DC component) - bad for physical signals
- For a 4000-point signal with std=1:
  - Time domain variance: σ² = 1
  - Frequency domain energy (Parseval's theorem): ∑|FFT|² ≈ 4000
- **The normalization doesn't translate to frequency space properly**
- LayerNorm makes all outputs unit variance regardless of target scale

**Result:**
- DeepONet BSP: 14,359,428 (catastrophic)
- UNet BSP: 6,880-338,265 (highly variable)
- FNO BSP: ~1.7 (reasonable, but has no normalization layer)

---

### Attempt 2: Reduced λ_spectral + RMS Scaling (Current Fix)

**What we did:**
- Changed `lambda_spectral` from 1.0 → 0.1 (paper's Airfoil value)
- Replaced LayerNorm with RMS scaling:
  ```python
  output_rms = torch.sqrt(torch.mean(output ** 2, dim=-1, keepdim=True) + 1e-8)
  output = output / (output_rms + 1e-8)
  ```

**Why we thought it would work:**
- RMS scaling preserves DC component (doesn't shift mean to 0)
- λ=0.1 reduces BSP influence by 10x, giving MSE more weight
- Paper's Airfoil case uses λ=0.1 successfully
- RMS normalization is simpler than LayerNorm, preserves frequency content better

**Why it didn't work:**
- **RMS scaling normalizes to unit RMS (=1) regardless of target scale**
- If target has RMS=0.01 and prediction has RMS=1 (after normalization):
  - Energy ratio: (1.0 / 0.01)² = 10,000x difference
  - BSP sees: pred_binned = 10,000 × target_binned
  - Relative error: 1 - (10,000×target + ε)/(target + ε) ≈ -9,999
  - Squared: (-9,999)² ≈ 100,000,000 (catastrophic!)
- **We're normalizing to the WRONG scale** (unit RMS) instead of target's scale
- Even with λ=0.1, the scale mismatch is 10^7x, so contribution is still 10^6x

**Result:**
- DeepONet BSP: 11,770 (better than 14M, but still catastrophic)
- UNet BSP: 65,936,243 (even worse than before!)
- Reducing λ helped DeepONet but not UNet

---

## Root Cause Analysis

### The Fundamental Problem: Time vs Frequency Domain Mismatch

Both LayerNorm and RMS scaling normalize in **time domain**, but BSP loss computes errors in **frequency domain**. The relationship between the two is governed by **Parseval's theorem**:

```
∑ |x(t)|² = (1/N) ∑ |X(f)|²
```

For a 4000-point signal:
- Time domain energy: E_time = ∑ x²
- Frequency domain energy: E_freq = (1/4000) ∑ |FFT|²
- Relationship: E_freq = E_time / 4000

**Our normalization (RMS = 1):**
- E_time = 4000 × 1² = 4000
- E_freq = 4000 / 4000 = 1

**If target has RMS = 0.01:**
- Target E_time = 4000 × 0.01² = 0.4
- Target E_freq = 0.4 / 4000 = 0.0001

**Energy ratio in frequency domain:**
- pred_energy / target_energy = 1.0 / 0.0001 = 10,000x

**BSP relative error:**
```
relative_error = 1 - (10,000 + 1e-6) / (1 + 1e-6) = -9,999
squared_error = 99,980,001
```

This is exactly what we're seeing!

### Why FNO Works (and Why BSP Makes It Worse)

FNO doesn't have output normalization layers. It:
1. Operates in Fourier space natively
2. Learns spectral coefficients directly
3. Output scale naturally matches target scale through training

**But BSP makes FNO's spectrum error WORSE (0.19 → 0.72)**

Why? FNO's Fourier-based architecture already learns good spectral representations. BSP's hard binning and ratio-based loss may be:
- Interfering with FNO's natural spectral learning
- Over-emphasizing specific frequency bins
- Creating adversarial gradients that hurt Fourier layer training

### Why DeepONet and UNet Have High Field Errors Everywhere

**DeepONet field error ≈ 2950 for ALL losses (baseline, BSP, SA-BSP)**
**UNet field error ≈ 280-2500 for ALL losses**

This indicates:
1. **The models aren't learning the task properly in the first place**
2. Our normalization is making it worse
3. The dataset or architecture may be mismatched

Possible reasons:
- SIREN w0=10 may still be too high (over-parameterized, hard to optimize)
- RMS normalization to unit scale is inappropriate for this dataset
- Dataset targets may have mean ≈ 0, but predictions are being forced to RMS=1
- Training may need more epochs, better learning rate, or different optimizer settings

---

## Why All Normalization Attempts Failed

### The Core Issue: We Don't Know Target Scale During Inference

All our normalization approaches share a fatal flaw:

**During training:**
- We have both prediction and target
- We could normalize to target's scale
- But we didn't - we normalized to fixed scale (mean=0, std=1 or RMS=1)

**During inference:**
- We don't have target
- Model must produce outputs at correct absolute scale
- Our normalization forces wrong scale (unit variance/RMS)

**The fix:** Model must learn to output at target's scale naturally, without forced normalization.

### Why Paper's Approach Works

The BSP paper tests on:
- UNet (2D/3D turbulence, airfoil)
- DCNN (dilated CNN)
- LSTM

**Key difference:** These architectures don't use SIREN or aggressive output normalization. They:
1. Use ReLU, tanh, or no activation on output
2. Learn output scale through supervised training
3. Produce outputs naturally at target's scale

**Our mistake:** We added normalization layers to "fix" scale issues, but normalization DESTROYS the model's ability to learn correct output scale.

---

## The Real Solution: Scale-Invariant BSP Loss

Instead of forcing model outputs to arbitrary scales, **make BSP loss scale-invariant**.

### Current BSP Formula (Scale-Dependent):

```python
# Compute binned energies
pred_binned = bin_average(|FFT(pred)|²)  # Shape: [B, C, n_bins]
target_binned = bin_average(|FFT(target)|²)

# Relative error per bin
relative_error = 1.0 - (pred_binned + ε) / (target_binned + ε)
squared_error = relative_error²

# Loss
bsp_loss = squared_error.sum(dim=1).mean()  # Sum over channels, mean over bins/batch
```

**Problem:** If pred and target differ in absolute scale by 10,000x, this explodes.

### Proposed Fix: Normalize Binned Spectra

```python
# Compute binned energies (same as before)
pred_binned = bin_average(|FFT(pred)|²)  # [B, C, n_bins]
target_binned = bin_average(|FFT(target)|²)

# NORMALIZE: Make each spectrum sum to 1 (distribution over frequencies)
# This makes comparison about FREQUENCY DISTRIBUTION, not absolute magnitude
pred_binned_norm = pred_binned / (pred_binned.sum(dim=-1, keepdim=True) + ε)
target_binned_norm = target_binned / (target_binned.sum(dim=-1, keepdim=True) + ε)

# Relative error per bin (now scale-invariant!)
relative_error = 1.0 - (pred_binned_norm + ε) / (target_binned_norm + ε)
squared_error = relative_error²

# Loss
bsp_loss = squared_error.sum(dim=1).mean()
```

**Why this works:**
- Both spectra normalized to sum to 1.0
- Comparison is about DISTRIBUTION of energy across frequency bins
- Scale-invariant: doubling both pred and target doesn't change loss
- Aligns with paper's intent: "mitigate spectral bias" = learn correct frequency distribution

**Trade-off:**
- Loses information about absolute energy magnitude
- But MSE loss in combined approach handles magnitude
- BSP should focus on DISTRIBUTION (spectral bias), MSE handles MAGNITUDE

### Alternative: Normalize by Target's Total Energy

```python
# Normalize both spectra by TARGET'S total energy
target_total_energy = target_binned.sum(dim=-1, keepdim=True) + ε
pred_binned_norm = pred_binned / target_total_energy
target_binned_norm = target_binned / target_total_energy

# Relative error (target_binned_norm sums to 1.0)
relative_error = 1.0 - (pred_binned_norm + ε) / (target_binned_norm + ε)
```

**Why this works:**
- Normalizes by target's scale, making pred comparable
- Preserves magnitude information relative to target
- If pred has correct distribution but wrong magnitude, loss ≈ (magnitude_ratio - 1)²

---

## Recommended Solution

### Step 1: Remove All Output Normalization from Models

**DeepONet (`src/core/models/deeponet_1d.py`):**
- Remove RMS scaling from forward() method
- Let model learn natural output scale

**UNet (`src/core/models/unet_1d.py`):**
- Remove GroupNorm from output
- Let model learn natural output scale

**Rationale:** Forced normalization prevents model from learning correct scale. Trust the supervised learning process.

### Step 2: Make BSP Loss Scale-Invariant

**File:** `src/core/evaluation/binned_spectral_loss.py`

**In `forward()` method, after binning (line 132):**

```python
# Step 3: Bin-average energies
pred_binned = self._bin_energy_1d(pred_energy, T)  # [B, C, n_bins]
target_binned = self._bin_energy_1d(target_energy, T)

# NEW: Normalize binned spectra to make loss scale-invariant
# Each spectrum sums to 1.0 (probability distribution over frequencies)
pred_total = pred_binned.sum(dim=-1, keepdim=True) + self.epsilon
target_total = target_binned.sum(dim=-1, keepdim=True) + self.epsilon

pred_binned_norm = pred_binned / pred_total
target_binned_norm = target_binned / target_total

# Step 4: Mean Squared Percentage Error per bin (now scale-invariant!)
relative_error = 1.0 - (pred_binned_norm + self.epsilon) / (target_binned_norm + self.epsilon)
```

**Expected behavior:**
- BSP loss magnitude: ~0.01-10.0 (comparable to MSE)
- Scale-invariant: pred at any scale produces reasonable loss
- Focus on distribution: BSP measures spectral SHAPE, MSE measures MAGNITUDE

### Step 3: Keep λ = 0.1 (Airfoil Value)

The paper's Airfoil case uses λ=0.1, which gives MSE 10x more weight than BSP. This is appropriate because:
- MSE handles overall prediction accuracy
- BSP handles spectral distribution
- Combined: Total Loss = MSE + 0.1×BSP ≈ 1.0 + 0.1 = 1.1 (balanced)

### Step 4: SA-BSP Adjustments

**File:** `src/core/evaluation/adaptive_spectral_loss.py`

Same normalization fix needed in `SelfAdaptiveSpectralLoss.compute_base_loss()`.

---

## Expected Results After Fix

**DeepONet:**
- BASELINE: ~20 (unchanged)
- BSP: ~20 + 0.1×(0.5-5.0) ≈ 20-21 (reasonable!)
- Spectrum error: Should improve from 1.08 → ~0.5-0.8

**UNet:**
- BASELINE: ~8 (unchanged)
- BSP: ~8 + 0.1×(0.5-5.0) ≈ 8-9 (reasonable!)
- Spectrum error: Should improve from 2.51 → ~1.0-1.5

**FNO:**
- BASELINE: ~1.0 (unchanged)
- BSP: ~1.0 + 0.1×(0.5-2.0) ≈ 1.05-1.2 (reasonable!)
- Spectrum error: Should NOT get worse (currently 0.19 → 0.72 is bad)

**Field errors:**
- Should remain similar to baseline (models learning actual task, not gaming metrics)
- If still high (~2950 for DeepONet), indicates architecture/hyperparameter issues separate from BSP

---

## Summary

**What failed:**
1. ❌ LayerNorm + reduced w0: Normalized to wrong scale, removed DC component
2. ❌ RMS scaling + λ=0.1: Still normalized to unit scale, not target's scale

**Why everything failed:**
- Normalized in TIME DOMAIN, but BSP operates in FREQUENCY DOMAIN
- Forced models to unit scale (RMS=1), but targets may have RMS=0.01
- Energy mismatch: 10,000x in time → 10,000x in frequency → 100,000,000x in squared error
- Can't normalize to target scale at inference time (don't have target!)

**The real fix:**
- **Don't normalize model outputs** - let them learn correct scale naturally
- **Make BSP loss scale-invariant** - normalize spectra to sum=1 before comparison
- BSP measures spectral DISTRIBUTION, MSE measures MAGNITUDE
- Combined loss with λ=0.1 balances both objectives

**Philosophy shift:**
- Stop trying to fix the model's output scale
- Fix the loss function to handle any scale
- Trust supervised learning to find correct output magnitude
