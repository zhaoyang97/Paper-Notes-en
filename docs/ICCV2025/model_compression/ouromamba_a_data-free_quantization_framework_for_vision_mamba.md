---
title: >-
  [Paper Note] OuroMamba: A Data-Free Quantization Framework for Vision Mamba
description: >-
  [ICCV 2025][Model Compression][Data-Free Quantization] The first data-free post-training quantization (PTQ) framework for Vision Mamba Models (VMMs)…
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "Data-Free Quantization"
  - "Vision Mamba"
  - "SSM"
  - "Mixed-Precision"
  - "Post-Training Quantization"
date: 2026-05-08
content_hash: 7c669534eaefc284
---

# OuroMamba: A Data-Free Quantization Framework for Vision Mamba

**Conference**: ICCV 2025
**arXiv**: [2503.10959](https://arxiv.org/abs/2503.10959)  
**Code**: [GitHub](https://github.com/georgia-tech-synergy-lab/ICCV-OuroMamba)  
**Area**: Model Compression / Quantization
**Keywords**: Data-Free Quantization, Vision Mamba, SSM, Mixed-Precision, Post-Training Quantization

## TL;DR

The first data-free post-training quantization (PTQ) framework for Vision Mamba Models (VMMs), which generates high-quality synthetic data via enhanced implicit attention and employs a mixed-precision quantization scheme with dynamic outlier detection. Under W4A4 settings, it significantly outperforms existing data-driven PTQ methods.

## Background & Motivation

Vision Mamba Models (VMMs) have emerged as a competitive alternative to ViTs due to their sub-quadratic computational complexity, yet deploying large-scale VMMs remains constrained by memory and latency bottlenecks. Post-training quantization (PTQ) is an effective compression strategy, but typically requires access to real calibration data, which is restricted in privacy-sensitive scenarios. Data-free quantization (DFQ) addresses this by generating synthetic data from Gaussian noise, yet existing DFQ methods are designed for ViTs and cannot be directly transferred to VMMs due to two fundamental challenges:

**Poor synthetic data quality**: VMMs lack explicit self-attention mechanisms due to their recurrent state transitions. Their implicit attention fails to distinguish foreground from background, rendering ViT-attention-based synthetic data generation methods ineffective. Furthermore, implicit attention across different scan directions exhibits directional bias and inconsistency.

**Dynamic activation outliers**: Unlike the static outlier patterns observed in ViTs, activations in VMM S6 layers (e.g., $\bar{A}$, $\bar{B}$) exhibit dynamically varying outlier channel positions across time steps, causing static PTQ techniques to fail.

## Method

### Overall Architecture

OuroMamba consists of two stages: OuroMamba-Gen (synthetic data generation) and OuroMamba-Quant (mixed-precision quantization).

### Key Designs

1. **Patched Hidden State $h_p(t)$**: To address the inability of VMM implicit attention to capture long-range interactions, a $p \times p$ spatial neighborhood $\mathcal{N}(t)$ is defined for the hidden state $h(t)$ at each time step. Neighboring states are aggregated via weighted summation: $h_p(\tau) = \sum_{k \in \mathcal{N}(\tau)} w_k h(k)$. The weighting factor leverages the channel-wise mean of the discretization tensor $\Delta(t)$, which produces larger responses in information-rich regions (e.g., foreground). The resulting enhanced implicit attention $\alpha_p$ effectively separates foreground from background.

2. **Data Generation Loss $\mathcal{L}^{gen}$**: A patch-level contrastive learning loss $\mathcal{L}^C$ (based on InfoNCE) is applied over the enhanced implicit attention $\alpha_p$, using cosine similarity to identify positive and negative patches. This is combined with a task-specific output loss $\mathcal{L}^O$ (MAE), yielding the final loss: $L^{gen} = \mathcal{L}^C + \mathcal{L}^O$.

3. **Dynamic Outlier Detection and Mixed-Precision Quantization**:

    - An offline calibration phase determines per-time-step inline scaling factors $S^I(t)$ and threshold $\theta$.
    - At inference time, $S^D(t)$ is computed dynamically per time step; if it exceeds $S^I(t)$, outlier channels are identified per-channel and added to $O_{\text{list}}$.
    - Outlier channels are quantized at $b_a^O$-bit (8-bit) per-channel, while inline channels are quantized at $b_a^I$-bit (4-bit) with group quantization.
    - $O_{\text{list}}$ is refreshed every $n_{\text{refresh}}=10$ steps to prevent accumulation of stale outlier entries.

### Loss & Training

- Data generation: $L^{gen} = \mathcal{L}^C + \mathcal{L}^O$, iterating 1000 steps to optimize Gaussian noise into synthetic data.
- Weight quantization: per-channel symmetric group quantization (4-bit).
- Activation quantization: per-time-step dynamic mixed-precision (4-bit inline / 8-bit outlier).

## Key Experimental Results

### Main Results (ImageNet Classification)

| Model | Method | Data | W/A | Top-1 |
|------|------|------|-----|-------|
| Vim-S | FP Baseline | - | 32/32 | 81.60 |
| Vim-S | PTQ4VM | Real 256 | 4/8 | 74.37 |
| Vim-S | QMamba | Real 1024 | 4/8 | 74.12 |
| **Vim-S** | **OuroMamba** | **Syn 128** | **4/8** | **79.81** |
| Vim-S | PTQ4VM | Real 256 | 4/4 | 69.60 |
| Vim-S | QMamba | Real 1024 | 4/4 | 33.64 |
| **Vim-S** | **OuroMamba** | **Syn 128** | **4/4** | **75.93** |
| Vim-B | FP Baseline | - | 32/32 | 81.90 |
| **Vim-B** | **OuroMamba** | **Syn 128** | **4/4** | **77.34** |
| VMamba-B | FP Baseline | - | 32/32 | 83.90 |
| **VMamba-B** | **OuroMamba** | **Syn 128** | **4/4** | **78.91** |
| MambaVision-B | FP Baseline | - | 32/32 | 84.20 |
| **MambaVision-B** | **OuroMamba** | **Syn 128** | **4/4** | **79.24** |

### Ablation Study

| $\mathcal{L}^{PSE}$ | $\mathcal{L}^C$ | $\mathcal{L}^O$ | W/A | Top-1 (%) |
|:---:|:---:|:---:|:---:|:---:|
| ✓ | ✗ | ✗ | 4/8 | 71.68 |
| ✗ | ✗ | ✓ | 4/8 | 21.65 |
| ✓ | ✗ | ✓ | 4/8 | 73.45 |
| ✗ | ✓ | ✗ | 4/8 | 75.52 |
| ✗ | ✓ | ✓ | 4/8 | **79.81** |

**$n_{\text{refresh}}$ Ablation**: A refresh interval of 10 yields optimal latency; smaller intervals cause frequent resets, while larger intervals lead to outlier accumulation and slower inference. Disabling refresh entirely results in latency even worse than FP16.

### Key Findings

- OuroMamba achieves an average improvement of **7.84%** over PTQ4VM and **19.40%** over QMamba under W4A4.
- Using only 128 synthetic images (without any real data) surpasses methods using 256–1024 real images.
- On object detection, box AP improves by up to **21.1**; on segmentation, mIoU improves by up to **6.6**.
- For the diffusion model Zigma under W4A4, FID increases only marginally from 37.8 to 39.2 (FacesHQ), far outperforming PTQ4VM's 89.6.
- A custom CUDA kernel achieves up to **2.36×** end-to-end speedup (Vim-B).

## Highlights & Insights

- **First to identify** the dynamic outlier problem in VMM activations, which stands in sharp contrast to the static outlier patterns in ViTs.
- The $\Delta(t)$-weighted patched hidden state for enhancing implicit attention is an elegant design—leveraging Mamba's own information selection mechanism to remedy its inherent limitations.
- Outperforming data-driven methods in a data-free setting demonstrates that architecture-aware synthetic data can be more effective than generic real data.
- The INT4+INT8 fused pipeline in the mixed-precision GEMM kernel achieves practical end-to-end acceleration.

## Limitations & Future Work

- The dynamic outlier detection assumes that the inline distribution remains stable after calibration; future architectures with high inline value variance may require additional investigation.
- The current kernel implementation supports only specific precision combinations (W4A4O8); broader kernel support for additional precision configurations remains to be developed.
- Integration with other compression techniques such as knowledge distillation has not been explored.
- The neighborhood size $p$ and refresh interval $n_{\text{refresh}}$ are hyperparameters that may require tuning across different architectures.

## Related Work & Insights

- **CLAMP-ViT**: The source of inspiration for the patch-level contrastive learning data generation paradigm adopted in this work, though it is limited to ViTs with explicit self-attention.
- **QMamba**: A concurrent work that identifies dynamic activation variation in VMMs but relies on static group quantization without resolving the issue.
- **PTQ4VM**: Applies a SmoothQuant-style strategy to migrate activation outliers to weights, but does not quantize SSM activations.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The first DFQ framework for VMMs; both stages feature thorough problem analysis and original solutions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers four tasks (classification, detection, segmentation, generation) across multiple models and precision settings, including kernel-level speedup evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated, analysis is thorough, and figures are informative.
- **Value**: ⭐⭐⭐⭐⭐ — Practically significant for VMM deployment; the synthetic data generation and dynamic quantization paradigms are broadly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Selective Coupling of Decoupled Informative Regions: Masked Attention Alignment for Data-Free Quantization of Vision Transformers](../../ICML2026/model_compression/selective_coupling_of_decoupled_informative_regions_masked_attention_alignment_f.md)
- [\[ICCV 2025\] MixA-Q: Revisiting Activation Sparsity for Vision Transformers from a Mixed-Precision Quantization Perspective](mixa-q_revisiting_activation_sparsity_for_vision_transformers_from_a_mixed-preci.md)
- [\[NeurIPS 2025\] Weight Weaving: Parameter Pooling for Data-Free Model Merging](../../NeurIPS2025/model_compression/weight_weaving_parameter_pooling_for_data-free_model_merging.md)
- [\[ICCV 2025\] EA-ViT: Efficient Adaptation for Elastic Vision Transformer](ea-vit_efficient_adaptation_for_elastic_vision_transformer.md)
- [\[ICCV 2025\] FREE-Merging: Fourier Transform for Efficient Model Merging](free-merging_fourier_transform_for_efficient_model_merging.md)

</div>

<!-- RELATED:END -->
