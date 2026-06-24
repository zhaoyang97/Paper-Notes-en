---
title: >-
  [Paper Note] Memory-Efficient Training with In-Place FFT Implementation
description: >-
  [NeurIPS 2025][Model Compression][FFT] This paper proposes rdFFT—the first truly in-place real-domain Fast Fourier Transform framework—which eliminates intermediate buffers via an implicit complex encoding scheme, achieving zero extra memory overhead for FFT/IFFT computation during training, with memory efficiency improvements exceeding 1500× in extreme cases.
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "FFT"
  - "in-place computation"
  - "memory optimization"
  - "circulant matrix"
  - "parameter-efficient fine-tuning"
date: 2026-05-08
content_hash: b51ec3841036a82a
---

# Memory-Efficient Training with In-Place FFT Implementation

**Conference**: NeurIPS 2025
**arXiv**: [2511.01385](https://arxiv.org/abs/2511.01385)  
**Code**: [PyTorch Issue #171022](https://github.com/pytorch/pytorch/issues/171022) (upstream merge under discussion)  
**Area**: Model Compression / Efficient Training
**Keywords**: FFT, in-place computation, memory optimization, circulant matrix, parameter-efficient fine-tuning

## TL;DR

This paper proposes rdFFT—the first truly in-place real-domain Fast Fourier Transform framework—which eliminates intermediate buffers via an implicit complex encoding scheme, achieving zero extra memory overhead for FFT/IFFT computation during training, with memory efficiency improvements exceeding 1500× in extreme cases.

## Background & Motivation

**Background**: FFT is widely employed in deep learning to reduce computational and memory overhead. Frequency-domain fine-tuning methods such as FourierFT and Block Circulant Adapter (BCA) rely on FFT for frequency-domain parameter transformation.

**Limitations of Prior Work**: Standard FFT maps $N$ real inputs to $N$ complex outputs (occupying $2N$ real-valued storage); rFFT reduces this to $N+2$ real values, which still does not match the input size of $N$ reals, precluding truly in-place computation.

**Key Challenge**: The dimensional and memory mismatch between input and output necessitates the allocation of additional memory buffers, introducing significant memory overhead in large-scale model training. Moreover, existing libraries (FFTW, cuFFT) do not support the bfloat16 data type.

**Goal**: Design a real-domain FFT whose input and output occupy identical memory, enabling both forward and backward passes to be executed within the original real-valued buffer, thereby eliminating intermediate tensor allocations.

**Key Insight**: Exploiting the symmetry of butterfly operations and the conjugate symmetry of the spectrum of real-valued signals to compress $N+2$ storage into $N$ real-valued slots.

**Core Idea**: Through a carefully designed memory layout—storing the real part of $y_k$ at index $k$ and its imaginary part at the conjugate-symmetric index $r-k$—the entire FFT computation is performed fully in-place within $N$ real-valued slots.

## Method

### Overall Architecture

rdFFT is based on the Cooley–Tukey algorithm and maintains conjugate symmetry at each stage of the recursive decomposition. Through a novel memory layout and butterfly operation scheme, it achieves fully in-place FFT and IFFT computation.

### Key Designs

1. **Squeezing $N+2$ into $N$ (Squeeze)**: For a real-valued input to an $r$-point FFT, the output satisfies $y_0, y_{r/2} \in \mathbb{R}$ (always real), and among the remaining $r-2$ complex values, only the real and imaginary parts of $y_1, \dots, y_{r/2-1}$ need to be stored; the rest can be reconstructed via conjugate symmetry $y_{r-k} = \overline{y_k}$. The storage scheme is: $y_0$ at index 0, $y_{r/2}$ at index $r/2$, the real part of $y_k$ at index $k$, and the imaginary part at index $r-k$. Unlike standard rFFT, no additional 2 real-valued slots are required.

2. **Four-Element Symmetry Group (Proposition 1)**: In the Cooley–Tukey recursion, the conjugate-symmetric pairs and their butterfly counterparts at each level form a four-element symmetry group $\{a_1, b_1, a_2, b_2\}$, symmetric about a center index $c$. This guarantees that butterfly operations at every level preserve conjugate symmetry and can be executed fully in-place. For example, in a 16-point FFT, the conjugate pair at indices (2, 6) from an 8-point FFT is paired with (10, 14), and the resulting pairs (2, 14) and (6, 10) remain symmetric.

3. **In-Place IFFT Design**: The IFFT receives conjugate-symmetric complex inputs, but sub-IFFT outputs are not guaranteed to be real-valued or conjugate-symmetric. Leveraging the linearity of the FFT, the computation graph of the forward FFT is inverted, and the real-valued signal is recovered by traversing the butterfly diagram in reverse. The key formula is $x_2 = \frac{y_2 + y_{10}}{2}$, $x_{10} = \frac{y_2 - y_{10}}{2W_N^2}$, decomposing each conjugate pair into symmetric and antisymmetric components.

4. **Circulant Matrix Training Adaptation**: In the circulant matrix product $\mathbf{y} = \text{IFFT}(\text{FFT}(\mathbf{c}) \odot \text{FFT}(\mathbf{x}))$, element-wise multiplication preserves symmetry (since $\overline{A \cdot B} = \overline{A} \cdot \overline{B}$), so the IFFT input always satisfies the symmetric complex condition. Gradient computation likewise satisfies this symmetry. The framework supports bfloat16 and is compatible with modern neural network training.

### Loss & Training

- Linear layer weight matrices are replaced by circulant matrix structures.
- Both forward and backward passes are executed fully in-place in the real domain.
- Integration with automatic differentiation frameworks is supported.

## Key Experimental Results

### Main Results: Single-Layer Memory Efficiency (D=4096, B=256)

| Method | Peak Memory (MB) | Compression vs. Full Fine-tuning |
|--------|:-:|:-:|
| Full Fine-tuning | 164.25 | 1× |
| LoRA (r=64) | 39.38 | 4.17× |
| FFT (p=512) | 164.50 | 1.00× |
| rFFT (p=512) | 156.66 | 1.05× |
| **rdFFT (p=512)** | **20.13** | **8.16×** |
| FFT (p=4096) | 52.05 | 3.16× |
| rFFT (p=4096) | 44.04 | 3.73× |
| **rdFFT (p=4096)** | **20.02** | **8.21×** |

### Full-Model Training Memory (LLaMA2-7B)

| Method | Peak Memory (GB) |
|--------|:-:|
| Full Fine-tuning | 26.90 |
| LoRA (r=32) | 18.96 |
| FFT (p=1024) | 19.20 |
| rFFT (p=1024) | 19.17 |
| **rdFFT (p=1024)** | **17.92** |
| **rdFFT (p=4096)** | **17.91** |

### Ablation Study: Numerical Precision

| Method | Absolute Error | Relative Error |
|--------|:-:|:-:|
| rFFT (p=1024) | 1.92e-07 | 0.0001 |
| rdFFT (p=1024) | 5.75e-07 | 0.0005 |

### Key Findings

- rdFFT achieves **1,014×** memory compression in the extreme single-layer case (D=1024, B=1, p=1024).
- At large batch sizes, intermediate tensor overhead grows sharply for standard FFT/rFFT, whereas rdFFT maintains constant memory usage.
- Numerical errors remain at the floating-point noise level, with mathematical equivalence fully preserved.
- Approximately 1 GB of memory is saved over LoRA in full-model training on LLaMA2-7B.
- Native bfloat16 support is provided, which standard FFT/rFFT implementations lack.

## Highlights & Insights

- The paper addresses memory inefficiency at the arithmetic operator level, offering a distinctive and elegant solution.
- The mathematical proof of the four-element symmetry group is elegant, rigorously formalizing the recursive preservation of conjugate symmetry.
- An upstream merge into PyTorch is under discussion; if successful, it would benefit all deep learning applications that use FFT.
- rdFFT is orthogonal to parameter-efficient methods such as LoRA and can be combined with them for further memory reduction.

## Limitations & Future Work

- The CUDA implementation incurs increased runtime overhead for large sizes (4096) due to thread-block synchronization constraints.
- Only 1D FFT has been validated; extension to 2D FFT (e.g., image-domain applications) has not yet been implemented.
- Downstream task evaluation is limited to NLU tasks (MRPC), lacking broader task coverage.
- Circulant matrix fine-tuning still underperforms LoRA in terms of model quality; the applicable scenarios of the method require further clarification.
- Although bfloat16 support is an advantage, performance under float16 is not sufficiently discussed.

## Related Work & Insights

- LoRA reduces trainable parameters via low-rank matrix decomposition but still requires storing intermediate activations; rdFFT eliminates intermediate tensors at the operator level.
- BCA (Block Circulant Adapter) exploits circulant matrix structure for efficient fine-tuning; rdFFT provides a more memory-efficient FFT backend for it.
- FourierFT performs fine-tuning in the frequency domain and can directly benefit from rdFFT's memory optimization.
- In-place operations have precedents in scenarios such as BN+ReLU fusion and anomaly detection; this paper extends the paradigm to FFT.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First truly in-place real-domain FFT with solid theoretical contributions
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive memory efficiency validation, but limited downstream task coverage
- Writing Quality: ⭐⭐⭐⭐ Rigorous mathematical derivations with clear and intuitive illustrations
- Value: ⭐⭐⭐⭐ Broad impact anticipated if merged into PyTorch, though current applicability is relatively narrow

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] EMLoC: Emulator-based Memory-efficient Fine-tuning with LoRA Correction](emloc_emulator-based_memory-efficient_fine-tuning_with_lora_correction.md)
- [\[ICLR 2026\] Towards Lossless Memory-efficient Training of Spiking Neural Networks via Gradient Checkpointing and Spike Compression](../../ICLR2026/model_compression/towards_lossless_memory-efficient_training_of_spiking_neural_networks_via_gradie.md)
- [\[ICCV 2025\] MSQ: Memory-Efficient Bit Sparsification Quantization](../../ICCV2025/model_compression/msq_memory-efficient_bit_sparsification_quantization.md)
- [\[AAAI 2026\] Towards Test-time Efficient Visual Place Recognition via Asymmetric Query Processing](../../AAAI2026/model_compression/towards_test-time_efficient_visual_place_recognition_via_asymmetric_query_proces.md)
- [\[ICCV 2025\] Task Vector Quantization for Memory-Efficient Model Merging](../../ICCV2025/model_compression/task_vector_quantization_for_memory-efficient_model_merging.md)

</div>

<!-- RELATED:END -->
