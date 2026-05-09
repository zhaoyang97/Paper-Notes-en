---
title: >-
  [Paper Note] SeeDNorm: Self-Rescaled Dynamic Normalization
description: >-
  [ICLR 2026][Model Compression][Normalization layer] This paper proposes SeeDNorm, an adaptive dynamic normalization layer that conditions the scaling coefficients on the input itself, thereby preserving input norm information in the forward pass while retaining RMSNorm-like adaptive gradient adjustment in the backward pass. With negligible additional parameters, SeeDNorm consistently outperforms RMSNorm, LayerNorm, and DyT on both language modeling and vision tasks.
tags:
  - ICLR 2026
  - Model Compression
  - Normalization layer
  - dynamic scaling
  - RMSNorm
  - DyT
  - large language models
date: 2026-05-08
content_hash: 534ded3ccf27b6a7
---

# SeeDNorm: Self-Rescaled Dynamic Normalization

**Conference**: ICLR 2026
**arXiv**: [2510.22777](https://arxiv.org/abs/2510.22777)
**Code**: None
**Area**: Model Compression / Normalization Layers
**Keywords**: Normalization layer, dynamic scaling, RMSNorm, DyT, large language models

## TL;DR

This paper proposes SeeDNorm, an adaptive dynamic normalization layer that conditions the scaling coefficients on the input itself, thereby preserving input norm information in the forward pass while retaining RMSNorm-like adaptive gradient adjustment in the backward pass. With negligible additional parameters, SeeDNorm consistently outperforms RMSNorm, LayerNorm, and DyT on both language modeling and vision tasks.

## Background & Motivation

Normalization layers are fundamental building blocks of modern deep neural networks, playing a critical role in stabilizing training and accelerating convergence. In Transformer architectures, RMSNorm is currently the dominant normalization method: it projects vectors onto the unit hypersphere and then applies learnable per-dimension scaling parameters $\gamma$ to restore representational capacity.

However, RMSNorm has two core limitations:

**Loss of input norm information in the forward pass**: The normalization operation itself discards the scale information of the input, constraining the expressive power of the network—particularly in zero-shot generalization settings.

**Static scaling factor $\gamma$ lacks flexibility**: $\gamma$ is a fixed parameter independent of the input and cannot adapt to wide variations in input data distribution.

Recent alternatives such as DyT (Dynamic Tanh) preserve input norm information in the forward pass, but suffer from gradient vanishing due to the saturation property of tanh. Through theoretical analysis (Proposition 6.1), the authors show that under the assumption of constant input norm, DyT is equivalent in gradient behavior to the element-wise operation of RMSNorm, meaning DyT loses RMSNorm's ability to adaptively adjust gradients based on input norm.

This leads to a fundamental question: **Can a method be designed that simultaneously achieves training stability, optimization efficiency, and preservation of input norm information?**

## Method

### Overall Architecture

The core design of SeeDNorm replaces the static scaling factor in RMSNorm with an input-dependent dynamic scaling factor. Given input $\mathbf{x} \in \mathbb{R}^{N \times D}$, SeeDNorm is formulated as:

$$\text{SeeDNorm}(\mathbf{x}) = [\sigma(\mathbf{x} \cdot \boldsymbol{\beta}^T) \cdot \boldsymbol{\alpha} + \boldsymbol{\gamma}] \odot \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})}$$

where $\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{D}\sum_{i=1}^D x_i^2 + \epsilon}$, $\boldsymbol{\alpha}, \boldsymbol{\beta}, \boldsymbol{\gamma} \in \mathbb{R}^{1 \times D}$ are learnable parameters, and $\sigma$ is a nonlinear activation function (tanh by default).

### Key Designs

1. **Self-Rescaling Matrix**: The term $\sigma(\mathbf{x} \cdot \boldsymbol{\beta}^T) \cdot \boldsymbol{\alpha}$ generates an input-dependent dynamic scaling component. The input $\mathbf{x}$ is projected onto $\boldsymbol{\beta}$ via matrix multiplication to yield a scalar, which is then bounded to $[-1, 1]$ by tanh and multiplied by $\boldsymbol{\alpha}$ to produce a per-dimension scaling matrix. This allows the scaling factor to dynamically adjust to the current input, thereby preserving input norm information.

2. **Scale Invariance Analysis**: When the input is scaled by a factor $k$, the only part of SeeDNorm that changes—due to the scale invariance of RMS normalization—is $\sigma(k\mathbf{x} \cdot \boldsymbol{\beta}^T)$. By initializing $\boldsymbol{\beta}$ to zero so that $\nabla_\mathbf{x} f$ starts at zero, SeeDNorm is insensitive to input scale variations in the early stages of training.

3. **Adaptive Gradient Adjustment**: In the backward pass, when the input $k\mathbf{x}$ is abnormally large, the gradient is dominated by $\frac{1}{\text{RMS}(k\mathbf{x})} = \frac{1}{k \cdot \text{RMS}(\mathbf{x})}$, shrinking the gradient by a factor of $k$. Conversely, when the input is abnormally small, the gradient is amplified accordingly. This adaptive gradient adjustment mechanism ensures training stability.

4. **Multi-Head SeeDNorm**: In high-dimensional spaces, the variance of the dot product $\mathbf{x} \cdot \boldsymbol{\beta}^T$ grows proportionally with dimension $D$ (Theorem 3.2), leading to excessive gradient variance. A multi-head variant is proposed that partitions $\mathbf{x}$ and $\boldsymbol{\beta}$ into $n$ sub-vectors, computes dot products independently, and then concatenates the results, effectively reducing gradient variance. This variant is used for vision tasks.

5. **AdaSeeDNorm**: A variant compatible with class-conditional information injection in DiT's AdaLN structure is designed as:
$$\text{AdaSeeDNorm}(\mathbf{x}, c) = [(\sigma(\mathbf{x} \cdot \boldsymbol{\beta}^T) \cdot \boldsymbol{\alpha} + 1) \odot \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})}](1 + \boldsymbol{\gamma}(c)) + \boldsymbol{\eta}(c)$$

### Loss & Training

- **Parameter initialization**: $\boldsymbol{\gamma}$ is initialized to 1 (consistent with RMSNorm); $\boldsymbol{\beta}$ is initialized to zero (ensuring gradients with respect to $\boldsymbol{\alpha}$ start from small values); $\boldsymbol{\alpha}$ is initialized to 1 (for language modeling tasks).
- **Regularization**: Weight decay is applied to $\boldsymbol{\alpha}$ and $\boldsymbol{\beta}$ for numerical stability; $\boldsymbol{\gamma}$ follows the same regularization as the baseline model.
- **Additional tricks for vision tasks**: Dropout (with the same rate as drop path) is applied to the dynamic scaling term $\sigma(\mathbf{x} \cdot \boldsymbol{\beta}^T) \cdot \boldsymbol{\alpha}$ in ViT classification, and $\boldsymbol{\alpha} \cdot \boldsymbol{\beta}^T$ is divided by the dimension to reduce variance.

## Key Experimental Results

### Main Results

**Large Language Models (MoE Architecture)**

| Model | Training Tokens | c4_en Loss | PPL | ARC-C | ARC-E | HellaSwag | PIQA |
|------|----------|-----------|------|-------|-------|-----------|------|
| OLMoE-1.3B (RMSNorm) | 500B | 2.922 | 18.63 | 32.3 | 62.2 | 55.2 | 72.6 |
| OLMoE-1.3B-DyT | 500B | 2.968 | 19.45 | 30.4 | 61.9 | 53.2 | 70.6 |
| **OLMoE-1.3B-SeeDNorm** | 500B | **2.900** | **18.12** | **34.5** | **65.4** | **56.8** | **73.1** |
| OLMoE-7B (RMSNorm) | 1000B | 2.644 | 14.07 | 40.8 | 73.7 | 71.2 | 76.6 |
| **OLMoE-7B-SeeDNorm** | 1000B | **2.631** | **13.88** | **44.5** | **76.1** | **71.8** | **79.1** |

**Large Language Models (Dense Architecture)**

| Model | Training Tokens | c4_en Loss | PPL | ARC-C | ARC-E |
|------|----------|-----------|------|-------|-------|
| OLMo2-1B (RMSNorm) | 500B | 2.884 | 17.88 | 35.6 | 68.7 |
| **OLMo2-1B-SeeDNorm** | 500B | **2.879** | **17.79** | **37.8** | **70.0** |

**Computer Vision Tasks** (ImageNet-1K Classification Acc@1)

| Model | LayerNorm | DyT | SeeDNorm |
|------|-----------|-----|----------|
| ViT-B | 82.3 | 82.5 | **82.7** |
| ViT-L | 83.1 | 83.6 | **83.6** |
| ConvNeXT-B | 83.7 | 83.7 | **83.7** |
| ConvNeXT-L | 84.3 | 84.4 | **84.6** |
| ViT-B (MAE) | 83.2 | 83.2 | **83.5** |
| ViT-L (MAE) | 85.5 | 85.4 | **85.5** |

### Ablation Study

| Configuration | c4 Loss | PPL | ARC-C | Note |
|------|---------|------|-------|------|
| SeeDNorm (default, α←1) | 2.900 | 18.12 | 34.5 | Best configuration |
| α←0.1 | 2.912 | 18.39 | 31.2 | Too-small init limits convergence |
| α←10 | 3.154 | 23.42 | 27.8 | Too-large init causes instability |
| Scalar α (replacing vector) | 2.909 | 18.33 | 32.6 | Per-dim adjustment outperforms uniform scaling |
| Element-wise x⊙β | 2.909 | 18.33 | 36.5 | Dot product has stronger expressiveness |
| Remove α | 2.907 | 18.29 | 32.1 | Loses per-dimension dynamic adjustment |
| Remove β | 2.911 | 18.37 | 31.9 | Loses nonlinear shape control |
| Remove γ | 2.913 | 18.41 | 33.7 | Equivalent to directly replacing RMS scaling |

**Multi-Head Ablation** (ViT-B ImageNet Classification)

| # Heads | Acc@1 | Note |
|------|-------|------|
| 1 Head | Diverges | Gradient variance too large |
| 8 Heads | 82.5 | Feasible but suboptimal |
| 16 Heads | **82.7** | Best |
| 32 Heads | 82.5 | Too many heads reduce gradient diversity |

### Key Findings

1. **MoE architecture amplifies SeeDNorm's advantage**: The dynamic activated parameters in MoE models make SeeDNorm's convergence acceleration more pronounced; Dense models show smaller training loss improvements but significant zero-shot evaluation gains.
2. **Bounded activation functions are necessary**: Using unbounded functions such as GeLU or Swish leads to divergence; tanh, sigmoid, and hardtanh all converge, with tanh achieving the best performance.
3. **DyT fails in LLM pretraining**: Replacing the normalization layers in OLMoE-1.3B with DyT results in slow convergence and degraded performance.
4. **Advantage grows with more training tokens**: As the number of training tokens increases, the loss advantage of SeeDNorm over the baseline continues to widen.

## Highlights & Insights

1. **Strong theoretical depth**: Proposition 6.1 proves that DyT, under the assumption of constant input norm, is equivalent to the element-wise differential equation solution of RMSNorm, revealing DyT's fundamental limitation—the inability to adaptively adjust gradients based on input norm.
2. **Minimalist yet effective design**: Only two $D$-dimensional parameter vectors ($\boldsymbol{\alpha}$, $\boldsymbol{\beta}$) are added, increasing computational complexity by $O(D)$—far less than the $O(D^2)$ of a linear layer—making this a truly plug-and-play improvement.
3. **Variance analysis motivating the multi-head mechanism**: The training instability is analyzed through the lens of dot-product variance growing proportionally with dimension, and is effectively resolved by splitting into heads, exemplifying a theory-guided design philosophy.
4. **Comprehensive gradient analysis**: Detailed analysis of gradient behavior under extreme conditions is provided for all parameters ($\boldsymbol{\alpha}$, $\boldsymbol{\beta}$, $\boldsymbol{\gamma}$, $\mathbf{x}$), going well beyond empirical observations.

## Limitations & Future Work

1. **Limited efficiency of native PyTorch implementation**: The fragmented operations in SeeDNorm increase memory access overhead, affecting latency and throughput; kernel fusion is required to achieve efficiency comparable to RMSNorm.
2. **AdaLN compatibility requires special handling**: SeeDNorm cannot directly replace AdaLN in DiT; a dedicated AdaSeeDNorm variant must be designed.
3. **Vision tasks require additional hyperparameter tuning**: The number of heads, dropout, and dimension scaling factors in vision tasks require careful adjustment, raising the barrier to adoption.
4. **Not validated at larger model scales**: The largest experiment is OLMoE-7B (1B activated parameters); performance at the 70B+ parameter scale remains unknown.
5. **KV cache compatibility not discussed**: The impact of SeeDNorm on KV cache and its associated overhead during inference warrants further investigation.

## Related Work & Insights

- The evolution from **BatchNorm → LayerNorm → RMSNorm** progressively simplifies normalization operations, yet consistently faces the problem of discarding input scale information.
- **DyT (Zhu et al., 2025b)**: Replaces normalization layers with dynamic tanh, preserving input norm but sacrificing adaptive gradient adjustment.
- **Frac-Connection (Zhu et al., 2025a)**: SeeDNorm can be combined with this method for further performance gains.
- The gradient equivalence between RMSNorm and DyT revealed in this paper may inspire future designs of better normalization or activation replacements.

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea of dynamic scaling factors is intuitive yet effective; the theoretical analysis elevates the design beyond a mere heuristic.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers LLMs (MoE + Dense), classification, generation, and self-supervised learning, with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are complete, experimental presentation is clear, and the appendix is comprehensive.
- Value: ⭐⭐⭐⭐ — A highly practical plug-and-play component, though kernel fusion is needed for broad deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](ld-mole_learnable_dynamic_routing_for_mixture_of_lora_experts.md)
- [\[NeurIPS 2025\] Dependency Parsing is More Parameter-Efficient with Normalization](../../NeurIPS2025/model_compression/dependency_parsing_is_more_parameter-efficient_with_normalization.md)
- [\[CVPR 2026\] Batch Loss Score for Dynamic Data Pruning](../../CVPR2026/model_compression/batch_loss_score_for_dynamic_data_pruning.md)
- [\[AAAI 2026\] DOS: Distilling Observable Softmaps of Zipfian Prototypes for Self-Supervised Point Representation](../../AAAI2026/model_compression/dos_distilling_observable_softmaps_of_zipfian_prototypes_for_self-supervised_poi.md)
- [\[CVPR 2026\] SODA: Sensitivity-Oriented Dynamic Acceleration for Diffusion Transformer](../../CVPR2026/model_compression/soda_sensitivity-oriented_dynamic_acceleration_for_diffusion_transformer.md)

</div>

<!-- RELATED:END -->
