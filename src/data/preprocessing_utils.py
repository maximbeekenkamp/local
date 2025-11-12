"""
Data preprocessing utilities for causal operator learning.

Implements zero-padding preprocessing from the CausalityDeepONet paper to
enforce physical causality through data preparation rather than architecture.

Reference: Penwarden et al. "A metalearning approach for physics-informed neural networks" (2023)
"""

import torch
import numpy as np
from typing import Tuple, Union


def prepare_causal_deeponet_data(
    inputs: Union[torch.Tensor, np.ndarray],
    outputs: Union[torch.Tensor, np.ndarray],
    signal_length: int = 4000
) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
    """
    Prepare data for causal DeepONet using zero-padding approach from reference.

    Creates sliding windows into left-zero-padded input signal, ensuring that
    prediction at timestep t only uses information from timesteps [0, ..., t].

    Algorithm (from reference Custom_dataset.py):
    1. Left-pad input signal with (signal_length - 1) zeros
    2. For each output timestep t, create input window:
       - Input window = padded_signal[t : t + signal_length]
       - This gives (signal_length - 1 - t) zeros, then inputs[0:t+1]
    3. Each (input_window, output[t]) becomes one training sample

    Args:
        inputs: Input signals [N, T] or [N, C, T] where T=signal_length
        outputs: Output signals [N, T] or [N, C, T] where T=signal_length
        signal_length: Length of signals (default 4000 for CDON)

    Returns:
        causal_inputs: Windowed inputs [N*T, signal_length]
        causal_outputs: Corresponding outputs [N*T]

    Example:
        >>> inputs = torch.randn(100, 4000)   # 100 samples, 4000 timesteps
        >>> outputs = torch.randn(100, 4000)  # 100 samples, 4000 timesteps
        >>> causal_in, causal_out = prepare_causal_deeponet_data(inputs, outputs)
        >>> print(causal_in.shape)   # [400000, 4000] - 100 * 4000 samples
        >>> print(causal_out.shape)  # [400000] - scalar output per sample
    """
    is_torch = isinstance(inputs, torch.Tensor)

    # Convert to numpy for processing
    if is_torch:
        inputs_np = inputs.cpu().numpy()
        outputs_np = outputs.cpu().numpy()
    else:
        inputs_np = inputs
        outputs_np = outputs

    # Handle channel dimension: [N, C, T] → [N, T] by taking first channel
    if inputs_np.ndim == 3:
        inputs_np = inputs_np[:, 0, :]  # Take first channel
    if outputs_np.ndim == 3:
        outputs_np = outputs_np[:, 0, :]  # Take first channel

    n_samples = inputs_np.shape[0]

    # Initialize output arrays
    # Total samples: n_samples * signal_length (one sample per timestep)
    total_samples = n_samples * signal_length
    causal_inputs_list = []
    causal_outputs_list = []

    # Process each sample
    for idx in range(n_samples):
        # Step 1: Left-pad input with (signal_length - 1) zeros
        # Result: [0, 0, ..., 0, input[0], input[1], ..., input[T-1]]
        #         |<- 3999 zeros ->|<-    4000 actual values     ->|
        padded_length = signal_length - 1 + signal_length  # 7999
        loads_add_zero = np.zeros(padded_length, dtype=np.float32)
        loads_add_zero[signal_length - 1:] = inputs_np[idx]  # Place actual signal after zeros

        # Step 2: Create sliding windows for each output timestep
        loads_in_train = np.zeros((signal_length, signal_length), dtype=np.float32)

        for t in range(signal_length):
            # Window from position t to t + signal_length
            # At t=0: [0, 0, ..., 0, input[0]]  (3999 zeros, then input[0])
            # At t=1: [0, 0, ..., 0, input[0], input[1]]  (3998 zeros, then input[0:2])
            # At t=3999: [input[0], input[1], ..., input[3999]]  (full signal, no zeros)
            loads_in_train[t, :] = loads_add_zero[t : t + signal_length]

        # Corresponding outputs (scalar per timestep)
        responses_out_train = outputs_np[idx]  # [signal_length]

        causal_inputs_list.append(loads_in_train)
        causal_outputs_list.append(responses_out_train)

    # Stack all samples
    causal_inputs_np = np.vstack(causal_inputs_list)  # [N*T, signal_length]
    causal_outputs_np = np.concatenate(causal_outputs_list)  # [N*T]

    # Convert back to torch if needed
    if is_torch:
        causal_inputs = torch.from_numpy(causal_inputs_np).to(inputs.device)
        causal_outputs = torch.from_numpy(causal_outputs_np).to(outputs.device)
    else:
        causal_inputs = causal_inputs_np
        causal_outputs = causal_outputs_np

    return causal_inputs, causal_outputs


def create_penalty_weights(
    targets: Union[torch.Tensor, np.ndarray],
    epsilon: float = 1e-8
) -> Union[torch.Tensor, np.ndarray]:
    """
    Create penalty weights inversely proportional to maximum response magnitude.

    From reference implementation (Custom_dataset.py line 121):
        penalty = 1.0 / max(abs(responses))**2

    This weights loss by inverse variance, emphasizing samples with larger
    responses (which are typically harder to predict accurately).

    Args:
        targets: Target values [N] or [N, T]
        epsilon: Small constant for numerical stability (default 1e-8)

    Returns:
        penalty_weights: Penalty weights [N] with same type as input

    Example:
        >>> targets = torch.randn(100, 4000)  # 100 samples, 4000 timesteps
        >>> penalties = create_penalty_weights(targets)
        >>> print(penalties.shape)  # [100] - one penalty per sample
    """
    is_torch = isinstance(targets, torch.Tensor)

    if is_torch:
        # Compute max absolute value per sample (dim=-1 for timesteps)
        if targets.ndim > 1:
            max_abs = torch.max(torch.abs(targets), dim=-1)[0]  # [N]
        else:
            max_abs = torch.abs(targets)  # [N]

        # Penalty = 1 / max²
        penalty = 1.0 / (max_abs ** 2 + epsilon)
    else:
        # Numpy version
        if targets.ndim > 1:
            max_abs = np.max(np.abs(targets), axis=-1)  # [N]
        else:
            max_abs = np.abs(targets)  # [N]

        penalty = 1.0 / (max_abs ** 2 + epsilon)

    return penalty


def prepare_causal_sequence_data(
    inputs: Union[torch.Tensor, np.ndarray],
    signal_length: int = 4000
) -> Union[torch.Tensor, np.ndarray]:
    """
    Zero-pad input sequences for causal prediction (simpler full-sequence approach).

    This is an alternative to the per-timestep windowing approach, suitable for
    models that process full sequences (like UNet, FNO).

    Instead of creating per-timestep samples, this simply left-pads the entire
    input sequence with zeros.

    Args:
        inputs: Input signals [N, T] or [N, C, T] where T=signal_length
        signal_length: Length of signals (default 4000 for CDON)

    Returns:
        padded_inputs: Zero-padded inputs [N, signal_length + (signal_length-1)]
                      = [N, 7999] for signal_length=4000

    Example:
        >>> inputs = torch.randn(100, 1, 4000)  # 100 samples
        >>> padded = prepare_causal_sequence_data(inputs)
        >>> print(padded.shape)  # [100, 1, 7999]
    """
    is_torch = isinstance(inputs, torch.Tensor)

    # Pad amount: signal_length - 1 zeros on the left
    pad_amount = signal_length - 1

    if is_torch:
        # Handle different input shapes
        if inputs.ndim == 2:  # [N, T]
            # Pad: (left, right) for last dimension
            padded = torch.nn.functional.pad(inputs, (pad_amount, 0), value=0.0)
        elif inputs.ndim == 3:  # [N, C, T]
            # Pad: (left, right) for last dimension
            padded = torch.nn.functional.pad(inputs, (pad_amount, 0), value=0.0)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {inputs.ndim}D")
    else:
        # Numpy version
        if inputs.ndim == 2:  # [N, T]
            padded = np.pad(inputs, ((0, 0), (pad_amount, 0)), mode='constant', constant_values=0.0)
        elif inputs.ndim == 3:  # [N, C, T]
            padded = np.pad(inputs, ((0, 0), (0, 0), (pad_amount, 0)), mode='constant', constant_values=0.0)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {inputs.ndim}D")

    return padded
