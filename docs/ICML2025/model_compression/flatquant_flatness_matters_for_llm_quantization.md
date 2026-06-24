---
title: >-
  [Paper Note] FlatQuant: Flatness Matters for LLM Quantization
description: >-
  [ICML2025][Model Compression][post-training quantization] This paper proposes FlatQuant, which uses learnable affine transformations (via Kronecker decomposition) to flatten weight and activation distributions. This achieves a $\le 1\%$ accuracy loss on LLaMA-3-70B under W4A4 quantization for the first time, while yielding a $2.3\times$ speedup in the prefill stage and a $1.7\times$ speedup during decoding.
tags:
  - "ICML2025"
  - "Model Compression"
  - "post-training quantization"
  - "affine transformation"
  - "Kronecker product"
  - "W4A4"
  - "flatness"
date: 2026-05-08
content_hash: 07d9a348842f0fb7
---

# FlatQuant: Flatness Matters for LLM Quantization

**Conference**: ICML2025  
**arXiv**: [2410.09426](https://arxiv.org/abs/2410.09426)  
**Code**: [ruikangliu/FlatQuant](https://github.com/ruikangliu/FlatQuant)  
**Area**: LLM Quantization / Model Compression  
**Keywords**: post-training quantization, affine transformation, Kronecker product, W4A4, flatness

## TL;DR

This paper proposes FlatQuant, which uses learnable affine transformations (via Kronecker decomposition) to flatten weight and activation distributions. This achieves a $\le 1\%$ accuracy loss on LLaMA-3-70B under W4A4 quantization for the first time, while yielding a $2.3\times$ speedup in the prefill stage and a $1.7\times$ speedup during decoding.

## Background & Motivation

### Key Challenge
The primary bottleneck in LLM inference is memory footprint and computational overhead. Quantization (e.g., converting FP16 parameters to INT4) represents one of the most effective compression methods. The magnitude of quantization error depends on the **flatness** of the weight/activation distributions; sharper distributions with more outliers incur larger uniform quantization errors.

### Limitations of Prior Work
- **Per-channel scaling** (SmoothQuant): Migrates activation outliers to weights, but this steepens the weight distribution and fails to distribute outliers across non-outlier channels.
- **Hadamard transform** (QuaRot, SpinQuant): Orthogonal rotation redistributes outliers, but all layers share the same transformation matrix, making it unable to adapt to layer-specific characteristics. Additionally, its effectiveness is limited for pivot tokens (which contain massive outliers in the initial tokens of a sequence).
- The distributions transformed by the above methods still remain steep and scattered.

### Key Findings
By visualizing the weight/activation distributions across layers of LLaMA-3-8B/70B, the authors find that even after existing transformations, the channel magnitude envelopes remain uneven. Furthermore, quantization error propagates and accumulates along layers, and is particularly severe at pivot tokens. FlatQuant significantly outperforms baselines in both dimensions.

## Method

### Core Idea
A learnable optimal invertible affine transformation matrix $\mathbf{P}^*$ is trained for each linear layer to flatten the transformed weights and activations, making them quantization-friendly:

$$\mathbf{P}^* = \arg\min_{\mathbf{P}} \|\mathbf{Y} - \mathcal{Q}(\mathbf{X}\mathbf{P})\mathcal{Q}(\mathbf{P}^{-1}\mathbf{W}^\top)\|_F^2$$

Directly maintaining the full $n \times n$ matrix $\mathbf{P}$ doubles computation and storage costs; thus, efficient parameterization is critical.

### Kronecker Decomposition (Core Innovation)
The matrix $\mathbf{P}$ is decomposed into the Kronecker product of two smaller matrices:

$$\mathbf{P} = \mathbf{P}_1 \otimes \mathbf{P}_2, \quad \mathbf{P}_1 \in \mathbb{R}^{n_1 \times n_1},\ \mathbf{P}_2 \in \mathbb{R}^{n_2 \times n_2},\ n = n_1 n_2$$

Utilizing vectorization techniques, the matrix multiplication is transformed into two smaller matrix multiplications:

$$\mathcal{Q}(\mathbf{P}_1^\top \times_1 \tilde{\mathbf{X}} \times_2 \mathbf{P}_2) \times \mathcal{Q}(\mathbf{P}_1^{-1} \times_1 \tilde{\mathbf{W}} \times_2 (\mathbf{P}_2^{-1})^\top)^\top$$

- **Storage Savings**: Up to $n/2\times$ (when $n_1 = n_2 = \sqrt{n}$)
- **Computation Savings**: Up to $\sqrt{n}/2\times$
- In practice, $n_1, n_2$ are chosen to minimize $n_1 + n_2$. For instance, $(64, 128)$ is used for $n=8192$.

### Additional Learnable Components
1. **Per-Channel Scaling** $\text{diag}(\mathbf{c})$: Introduced before the affine transformation, which can be fused into the preceding LayerNorm or linear layer to achieve zero additional inference overhead.
2. **Learnable Clipping Thresholds** $\alpha_w, \alpha_a \in (0,1)$: Clipping boundaries are learned independently for weights and activations in each layer, outperforming grid search.

### Loss & Training
For the $l$-th Transformer block:

$$\min_{\Theta} \|\mathcal{F}_l(\mathbf{X}) - \hat{\mathcal{F}}_l(\mathbf{X}; \Theta)\|_F^2, \quad \Theta = \{\mathbf{P}, \mathbf{c}, \alpha_a, \alpha_w\}$$

Calibration is conducted using 128 sentences from WikiText-2, optimized via AdamW with an initial learning rate of 5e-3 over 15 epochs. This takes approximately 0.9 hours for LLaMA-3-8B on a single GPU.

### Transformer Integration
- **Self-Attention**: Four transformation matrices are used: $\mathbf{P}_a$ (Q/K/V input), $\mathbf{P}_o$ (output projection input), $\mathbf{P}_h$ (per-head Key cache), and $\mathbf{P}_v$ (per-head Value cache). Among these, only $\mathbf{P}_a$ and $\mathbf{P}_o$ undergo Kronecker decomposition.
- **FFN**: Two transformation matrices are used: $\mathbf{P}_{ug}$ (FFN input) and $\mathbf{P}_d$ (down-projection input), both of which are decomposed.
- **LayerNorm**: The original LayerNorm is preserved (not converted to RMSNorm), allowing for distinct affine transformations per layer to enhance expressiveness.

### Efficient Kernel Design
Affine transformation and quantization are fused into a single Triton kernel: $\mathbf{P}_1$ and $\mathbf{P}_2$ are loaded into SRAM, and each thread block processes a tiling block $\bar{\mathbf{X}} \in \mathbb{R}^{n_1 \times n_2}$ to compute $\mathbf{P}_1 \bar{\mathbf{X}} \mathbf{P}_2$ with on-the-fly quantization. All intermediate results reside in SRAM, eliminating redundant GPU memory accesses.

## Key Experimental Results

### W4A4 Language Modeling (PPL↓)

| Model | Method | WikiText-2 | C4 |
|------|------|-----------|-----|
| LLaMA-3-70B FP16 | - | 2.86 | 7.17 |
| LLaMA-3-70B | QuaRot+RTN | 55.44 | 79.48 |
| LLaMA-3-70B | SpinQuant+RTN | 7.58 | 15.39 |
| **LLaMA-3-70B** | **FlatQuant+RTN** | **3.78** | **7.86** |
| LLaMA-3-8B FP16 | - | 6.14 | 9.45 |
| LLaMA-3-8B | SpinQuant+RTN | 7.96 | 13.45 |
| **LLaMA-3-8B** | **FlatQuant+RTN** | **6.98** | **11.13** |

### W4A4 Zero-Shot Inference Accuracy (Avg↑)

| Model | Method | Avg (6 tasks) |
|------|------|--------------|
| LLaMA-3-70B FP16 | - | 79.95 |
| LLaMA-3-70B | SpinQuant+RTN | 65.66 |
| **LLaMA-3-70B** | **FlatQuant+RTN** | **79.01** |
| LLaMA-3-8B FP16 | - | 73.23 |
| LLaMA-3-8B | SpinQuant+RTN | 66.98 |
| **LLaMA-3-8B** | **FlatQuant+RTN** | **71.23** |

**Core Conclusions**:

- The accuracy loss of LLaMA-3-70B under W4A4 quantization is only 0.94% (79.01 vs 79.95), which significantly outperforms SpinQuant (14.29% loss).
- Even with the simplest RTN quantizer, FlatQuant outperforms the SpinQuant+GPTQ combination.
- Inference Speedup: Compared to FP16, W4A4 achieves a $2.3\times$ speedup during the prefill stage and a $1.7\times$ speedup in decoding.
- Extremely Low Overhead: The affine transformation accounts for only 2.61% of FLOPs, and the extra storage is only 3.41MB (for LLaMA-2-7B).

## Highlights & Insights

1. **Unified Flatness Perspective**: Attributing the quantization challenge to distribution flatness optimization provides an intuitive and quantifiable optimization objective (kurtosis/MSE landscape).
2. **Exquisite Balance of Kronecker Decomposition**: The $(n_1, n_2)$ decomposition avoids the overhead of a full $n \times n$ matrix, offering an exceptional trade-off between storage and computation.
3. **RTN is Sufficient**: Under a sufficiently high-quality transformation, the simple round-to-nearest (RTN) approach can approximate the effectiveness of GPTQ, demonstrating that the distribution transformation is more critical than the quantization strategy itself.
4. **Choice of Preserving LayerNorm**: Unlike QuaRot/SpinQuant which convert LayerNorm to RMSNorm to share transformations, FlatQuant preserves LayerNorm to allow independent learning per layer, leading to stronger expressiveness.
5. **Fused Triton Kernel Design**: Consolidates memory-bound affine transformations and quantization to prevent intermediate results from writing back to GPU memory.

## Limitations & Future Work

1. **Calibration Cost**: Although described as "lightweight", it still requires block-wise training for 15 epochs (approx. 0.9 hours for LLaMA-3-8B). For larger models (e.g., 405B), the calibration time could be noticeably longer.
2. **W4A4 as the Sweet Spot**: The paper primarily focuses on the W4A4 configuration, and the effectiveness on more aggressive W2/W3 or weight-only quantization has not been deeply explored.
3. **Limited Model Coverage**: The method is mainly validated on the LLaMA series; its applicability to MoE architectures (like Mixtral) or multimodal models remains unexplored.
4. **Limitations of Layer-independent Transformations**: Although Kronecker decomposition is efficient, it remains a linear transformation, limiting its capacity to handle non-linear distribution patterns.
5. **Insufficient Evaluation on Generation Quality**: The paper only evaluates PPL and zero-shot classification, lacking assessment on downstream tasks like long-text generation, conversation, and instruction-following.

## Related Work & Insights

- **SmoothQuant** (Xiao et al., 2023): A pioneer in per-channel scaling. FlatQuant incorporates it as one of its components and extends it to full affine transformation.
- **QuaRot** (Ashkboos et al., 2024): Leverages Hadamard rotation + fused kernels, acting as a direct competitor to FlatQuant.
- **SpinQuant** (Liu et al., 2024): Learns orthogonal matrix rotations but is constrained by sharing the transformations. FlatQuant relaxes the orthogonal constraint to general invertible matrices.
- **AffineQuant** (Ma et al., 2024): Also utilizes affine transformations but lacks Kronecker decomposition, leading to high inference overhead. FlatQuant can be viewed as its highly efficient counterpart.

## Rating
- Novelty: ⭐⭐⭐⭐ — The approach of utilizing Kronecker decomposition for affine transformations is simple and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive comparisons across multiple models and tasks, including inference speeds and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation and rich visualizations.
- Value: ⭐⭐⭐⭐⭐ — First to achieve practical accuracy for W4A4, delivering significant inference speedups with open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SERQ: Saliency-Aware Low-Rank Error Reconstruction for LLM Quantization](../../ICLR2026/model_compression/serq_saliency-aware_low-rank_error_reconstruction_for_llm_quantization.md)
- [\[ICLR 2026\] Rethinking Residual Errors in Compensation-based LLM Quantization](../../ICLR2026/model_compression/rethinking_residual_errors_in_compensation-based_llm_quantization.md)
- [\[CVPR 2026\] What Matters in Practical Learned Image Compression](../../CVPR2026/model_compression/what_matters_in_practical_learned_image_compression.md)
- [\[AAAI 2026\] Beyond Sharpness: A Flatness Decomposition Framework for Efficient Continual Learning](../../AAAI2026/model_compression/beyond_sharpness_a_flatness_decomposition_framework_for_efficient_continual_lear.md)
- [\[ACL 2025\] MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts](../../ACL2025/model_compression/moqae_mixed_precision_kv_cache.md)

</div>

<!-- RELATED:END -->
