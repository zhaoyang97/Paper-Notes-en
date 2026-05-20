---
title: >-
  [Paper Note] Test-Time Instance-Specific Parameter Composition: A New Paradigm for Adaptive Generative Modeling
description: >-
  [CVPR 2026][Image Generation][Adaptive Generative Modeling] This paper proposes Composer, a plug-and-play meta-generator framework that dynamically generates low-rank parameter updates from each input condition at infere…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Adaptive Generative Modeling"
  - "Test-Time Parameter Composition"
  - "Low-Rank Updates"
  - "Dynamic Weights"
  - "Meta-Generator"
date: 2026-05-08
content_hash: 4d98e49138559ee9
---

# Test-Time Instance-Specific Parameter Composition: A New Paradigm for Adaptive Generative Modeling

**Conference**: CVPR 2026
**arXiv**: [2603.27665](https://arxiv.org/abs/2603.27665)  
**Code**: [https://github.com/tmtuan1307/Composer](https://github.com/tmtuan1307/Composer)  
**Area**: Image Generation / Diffusion Models
**Keywords**: Adaptive Generative Modeling, Test-Time Parameter Composition, Low-Rank Updates, Dynamic Weights, Meta-Generator

## TL;DR

This paper proposes Composer, a plug-and-play meta-generator framework that dynamically generates low-rank parameter updates from each input condition at inference time and injects them into pretrained model weights, achieving instance-specific adaptive generation with negligible computational overhead (+0.2% time, +3.6% memory). The framework consistently improves performance across class-conditional generation, text-to-image synthesis, post-training quantization, and test-time scaling scenarios.

## Background & Motivation

**Background**: Diffusion models and visual autoregressive models have achieved remarkable success in image generation, yet they are fundamentally static — a fixed set of pretrained parameters must handle all input prompts, scenes, and modalities.

**Limitations of Prior Work**: This rigidity limits model adaptability. When faced with complex or ambiguous generation conditions, static weights cannot specialize to the semantic characteristics of each input, frequently resulting in over-smoothed or inconsistent samples. Existing test-time training (TTT) methods can adapt parameters at inference time but require per-instance gradient optimization, incurring prohibitive computational costs (540%+ increase in time, 180%+ increase in memory). Mixture-of-Experts (MoE) architectures offer conditional computation but with coarse routing granularity, fixed expert pools, and requirements for architectural modifications and full retraining.

**Key Challenge**: How can pretrained generative models acquire per-instance adaptability without significant computational overhead?

**Goal**: (1) Enable per-input parameter specialization at inference time without fine-tuning or retraining; (2) remain compatible with arbitrary pretrained generative model backbones in a plug-and-play manner; (3) maintain minimal computational and memory overhead.

**Key Insight**: Inspired by how humans flexibly adjust internal generative representations to accommodate different perceptual and imaginative contexts, the paper forgoes iterative optimization and instead synthesizes parameter updates directly from conditioning signals via a lightweight auxiliary network.

**Core Idea**: A Transformer-based meta-generator maps input conditions to low-rank parameter updates $W' = W + AB$, completing parameter composition in a single pass before inference, thereby achieving per-instance adaptive generation at near-zero overhead.

## Method

### Overall Architecture

Composer operates in two phases. **Training**: the meta-generator Transformer learns to map (pretrained weights, input condition) pairs to low-rank matrices $(A, B)$, producing instance-specific weight updates $W' = W + AB$; the updated weights are then used for standard generative training. **Inference**: the linear projection layers are removed; stored tokens are used to rapidly generate parameter updates, which are composed with the pretrained weights once and applied throughout the entire multi-step generation process.

### Key Designs

1. **Instance-Specific Low-Rank Parameter Composition**:

    - **Function**: Generates customized weight corrections for each input, enabling per-instance specialization of the pretrained model at inference time.
    - **Mechanism**: For the query matrix $W_Q$ and value matrix $W_V$ in the backbone network, low-rank updates are generated as $W' = W + AB$, where $A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times d}$, and $r \ll d$ (default $r=8$). The key distinction from LoRA is that LoRA's $A$ and $B$ are shared across all inputs, whereas Composer's $A$ and $B$ are dynamically generated conditioned on each individual input.
    - **Design Motivation**: The low-rank constraint ensures minimal parameter count and computational efficiency; dynamic generation guarantees per-instance adaptability; modifying only Q and V (leaving K and O unchanged) is guided by prior fine-tuning experience.

2. **Transformer Meta-Parameter Generator**:

    - **Function**: Transforms input conditions (e.g., class labels or text prompts) into high-quality low-rank parameter updates.
    - **Mechanism**: During training, initial tokens $A^0, B^0$ are derived from pretrained weights via linear projection ($\mathbb{R}^{d \times d} \to \mathbb{R}^{2r \times d_{model}}$) and concatenated with input prompt tokens before being processed by the Transformer. The attention scheme is carefully designed: all component tokens attend to prompt tokens (for context acquisition), local intra-block attention reduces computation, and the first token of each block attends across blocks to capture inter-block dependencies. At inference, the linear projection layers are removed and the stored $A^0, B^0$ tokens are used directly.
    - **Design Motivation**: The Transformer effectively models relationships across different weight matrices; the hierarchical attention mechanism maintains context awareness while controlling computational cost.

3. **Context-Aware Training Strategy**:

    - **Function**: Balances consistency among similar inputs and diversity across dissimilar inputs.
    - **Mechanism**: A parameter $\alpha \in [0,1]$ partitions each batch — $\alpha \times b$ samples drawn from the same class ensure adaptation consistency, while $(1-\alpha) \times b$ samples from different classes ensure output discriminability. For text-to-image tasks, semantic similarity in CLIP embedding space is additionally used for batch selection. Default $\alpha = 0.75$.
    - **Design Motivation**: Purely random sampling lacks semantic relationship modeling, leading to unstable adaptation; purely same-class sampling reduces diversity and causes mode collapse.

### Loss & Training

For class-conditional generation, the standard diffusion loss is used: $\mathcal{L} = \mathbb{E}_{x,\epsilon,t}[\|\epsilon - \epsilon_\theta(x_t, t; W', P)\|_2^2]$. For post-training quantization, a knowledge distillation loss is employed: $\mathcal{L}_{KD} = \|h - h_q\|_2^2$. All experiments use AdamW (weight decay 0.05, lr 1e-4), trained for 50 epochs with low-rank dimension $r=8$.

## Key Experimental Results

### Main Results

ImageNet 256×256 class-conditional generation:

| Backbone | Method | FID ↓ | IS ↑ | Inference Time | Memory |
|----------|--------|-------|------|----------------|--------|
| VAR d-16 | Standard | 3.55 | 274.4 | 0.4s | 2.37G |
| VAR d-16 | TTT | 3.22 | 277.2 | 40.52s (+10030%) | 4.58G |
| VAR d-16 | **Composer** | **3.15** | **280.4** | 0.42s (+5%) | 2.57G |
| VAR d-30 | Standard | 1.97 | 323.1 | 1.0s | 16.57G |
| VAR d-30 | TTT | 1.85 | 327.7 | 112.37s (+11137%) | 28.41G |
| VAR d-30 | **Composer** | **1.79** | **330.4** | 1.07s (+7%) | 16.97G |
| DiT-XL/2 | Standard | 2.27 | 278.2 | 45s | 6.1G |
| DiT-XL/2 | **Composer** | **2.06** | **285.6** | 45.03s (+0.07%) | 6.4G |

### Ablation Study

| Dimension | Setting | VAR d-16 FID |
|-----------|---------|--------------|
| Low-rank dimension $r$ | 4/8/16/32 | 3.32/3.15/3.10/3.08 |
| Sampling ratio $\alpha$ | 0.0/0.5/0.75/1.0 | 3.42/3.22/3.15/3.25 |
| Attention mechanism | Standard/Global-Local | 3.55/3.15 |
| Generator architecture | CNN/MLP/Transformer | 3.35/3.32/3.15 |

### Key Findings

- **High efficiency**: Compared to TTT's 10000%+ time overhead, Composer incurs only 0.2%–7% additional inference time and 3.6%–5% additional memory.
- **Consistent cross-backbone improvement**: FID is stably reduced across VAR (d-16 to d-36), DiT (L/2 to XL/2), and SD2.1.
- **Quantization recovery**: Under extreme 2/8-bit quantization, IS improves from 49.08 to 78.21 and FID drops from 43.36 to 35.26 over Q-Diffusion.
- **Composability**: Composer combined with ORM/PARM test-time scaling yields further gains (FID from 13.45 to 12.82).
- **$\alpha = 0.75$ is optimal**: Both excessively low (no consistency constraint) and excessively high (insufficient diversity) values yield suboptimal results.

## Highlights & Insights

- **Paradigm innovation**: The shift from static to dynamic parameter composition is a natural evolution of the LoRA concept — LoRA provides globally shared low-rank adaptation, while Composer delivers per-instance low-rank adaptation.
- **Strong practicality**: Plug-and-play, model-agnostic, and near-zero overhead — a rare "strictly better" method across all practical dimensions.
- **Unique value in quantization**: Using low-rank updates to compensate for quantization errors is a particularly elegant application.
- **Analogy to human cognition**: The model learns to "adjust its mindset" for each input prior to generation, analogous to the mental preparation humans undergo when approaching different creative tasks.

## Limitations & Future Work

- Adaptation is limited to Q and V matrices; the potential of other layers (e.g., FFN) remains unexplored.
- Training still requires 50 epochs; the training cost of the meta-generator is not thoroughly discussed.
- Improvements on text-to-image tasks (FID 13.45→13.07) are relatively modest compared to the class-conditional setting.
- Application to more complex domains such as video generation or 3D generation has not been explored.

## Related Work & Insights

- LoRA is the conceptual predecessor of Composer — the contrast between shared low-rank and dynamic low-rank adaptation is highly instructive.
- The coarse-grained routing of MoE versus the fine-grained per-instance adaptation of Composer illustrates different levels of dynamic computation.
- HyperNetworks also perform parameter generation, but Composer's Transformer architecture with hierarchical attention is more efficient.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Test-time per-instance parameter composition is a genuinely new paradigm with far-reaching implications for generative model design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 5 backbone models, 4 application scenarios, and extensive ablations with highly complete data.
- **Writing Quality**: ⭐⭐⭐⭐ — Concepts are clearly articulated, derivations are complete, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐⭐ — A plug-and-play general framework validated across multiple scenarios with minimal overhead; extremely high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GenCP: Towards Generative Modeling Paradigm of Coupled Physics](../../ICLR2026/image_generation/gencp_towards_generative_modeling_paradigm_of_coupled_physics.md)
- [\[CVPR 2026\] Depth Adaptive Efficient Visual Autoregressive Modeling](depthvar_depth_adaptive_var.md)
- [\[ICLR 2026\] Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](../../ICLR2026/image_generation/compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)
- [\[CVPR 2026\] SPDMark: Selective Parameter Displacement for Robust Video Watermarking](spdmark_selective_parameter_displacement_for_robust_video_watermarking.md)
- [\[ICLR 2026\] Test-Time Iterative Error Correction for Efficient Diffusion Models](../../ICLR2026/image_generation/test-time_iterative_error_correction_for_efficient_diffusion_models.md)

</div>

<!-- RELATED:END -->
