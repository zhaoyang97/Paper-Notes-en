---
title: >-
  [Paper Note] Alias-Free ViT: Fractional Shift Invariance via Linear Attention
description: >-
  [NeurIPS 2025][shift invariance] This paper proposes the Alias-Free Vision Transformer (AFT), which combines anti-aliasing signal processing techniques with shift-equivariant linear cross-covariance attention…
tags:
  - "NeurIPS 2025"
  - "shift invariance"
  - "anti-aliasing"
  - "linear attention"
  - "sub-pixel shift"
  - "shift equivariance"
date: 2026-05-08
content_hash: 17e87b6e2585fba3
---

# Alias-Free ViT: Fractional Shift Invariance via Linear Attention

**Conference**: NeurIPS 2025
**arXiv**: [2510.22673](https://arxiv.org/abs/2510.22673)  
**Code**: [https://github.com/hmichaeli/alias_free_vit](https://github.com/hmichaeli/alias_free_vit)  
**Area**: Vision Transformer / Robustness
**Keywords**: shift invariance, anti-aliasing, linear attention, sub-pixel shift, shift equivariance

## TL;DR

This paper proposes the Alias-Free Vision Transformer (AFT), which combines anti-aliasing signal processing techniques with shift-equivariant linear cross-covariance attention, achieving near-perfect consistency (~99%) under fractional (sub-pixel) shifts for the first time, with negligible degradation in ImageNet classification accuracy.

## Background & Motivation

ViTs have become the dominant architecture for visual tasks, yet they lack the translation-invariance inductive bias of CNNs and are highly sensitive to small image shifts. For CNNs, prior work has shown that aliasing is the root cause of broken shift invariance — aliasing in downsampling layers and nonlinearities distorts signals. Existing anti-aliasing methods such as APS (Adaptive Polyphase Sampling) guarantee consistency under integer-pixel cyclic shifts, but fail under sub-pixel (fractional) shifts and realistic image translations (e.g., crop-shifts due to camera motion).

The core problem is that standard softmax self-attention is not shift-equivariant. After shifting the input signal, the nonlinear row-wise normalization in softmax alters the relative attention weights across tokens, causing the output to no longer be equivariant. The authors' key insight is that removing softmax in favor of linear attention or cross-covariance attention (XCA) renders the attention operation shift-equivariant. Combined with anti-aliased downsampling and nonlinear layer processing, this enables a ViT that is robust even to sub-pixel shifts.

## Method

### Overall Architecture

Building on the XCiT architecture, the authors systematically replace every non-shift-equivariant component: anti-aliased patch embedding → Alias-Free Transformer Block (anti-aliased LayerNorm + XCA + anti-aliased LPI + anti-aliased MLP) → Alias-Free Class Attention. The overall architecture maintains the same parameter count as XCiT.

### Key Designs

1. **Shift-Equivariant Attention (SEA) Theory**:
    - Function: Formally proves that a class of linear attention operations is inherently shift-equivariant.
    - Mechanism: Defines $\text{SEA}(X) = Q f(K^\top V)$ and proves equivariance in three steps: (i) Q/K/V are shift-equivariant (each column is a linear combination of input signal channels); (ii) $K^\top V$ is shift-invariant (by Parseval's theorem, phase shifts cancel in the frequency domain); (iii) the product $Q \cdot f(K^\top V)$ is therefore shift-equivariant.
    - Design Motivation: Standard softmax attention breaks equivariance via row-wise normalization; in XCA, $K^\top Q$ is also shift-invariant, so applying softmax to it does not affect equivariance.

2. **Anti-Aliased Patch Embedding**:
    - Function: Eliminates aliasing introduced during tokenization.
    - Mechanism: Replaces the original stride-$p$ convolution with a stride-1 convolution followed by anti-aliased downsampling (frequency truncation in the FFT domain). A progressive multi-layer convolution scheme is adopted (rather than a single large stride), where each layer performs small-stride downsampling with a higher frequency cutoff, preserving high-frequency features.
    - Design Motivation: A single stride-$p$ step would require a low-pass filter with cutoff $1/p$, severely attenuating high-frequency information.

3. **Anti-Aliased Nonlinear Layers and LayerNorm**:
    - Function: Eliminates aliasing introduced by nonlinear functions such as GELU and by LayerNorm.
    - Mechanism: Inputs are upsampled before GELU; after GELU, a low-pass filter is applied before downsampling back to the original resolution. LayerNorm is replaced by a global variant — mean computed per-token but standard deviation computed per-layer — to prevent per-token scaling from breaking equivariance.
    - Design Motivation: Pointwise nonlinearities generate new frequency components (harmonics) in the frequency domain, leading to aliasing.

### Loss & Training

- The training strategy is identical to XCiT (400 epochs on ImageNet), with no additional shift augmentation required.
- Positional encodings are removed (experiments show they are unnecessary in architectures with convolutional layers, and their removal yields a slight accuracy improvement).
- Class Attention is used in place of global average pooling to obtain the classification representation, yielding better performance.

## Key Experimental Results

### Main Results

| Model | Top-1 Acc | Integer Shift Consistency | Half-Pixel Shift Consistency | Adversarial Integer Grid | Adversarial Half-Pixel Grid |
|------|----------|---------------|-----------------|------------|-------------|
| XCiT-Nano (Baseline) | 70.4 | 83.7 | 82.0 | 50.9 | 52.9 |
| XCiT-Nano-APS | 68.4 | **100.0** | 87.5 | 68.4 | 62.9 |
| XCiT-Nano-AF (ours) | **70.5** | 99.2 | **98.7** | **69.9** | **69.5** |
| XCiT-Small (Baseline) | 82.0 | 91.4 | 89.8 | 70.9 | 71.3 |
| XCiT-Small-APS | 81.3 | **100.0** | 94.0 | 81.3 | 78.2 |
| XCiT-Small-AF (ours) | **81.8** | 99.5 | **99.4** | **81.3** | **81.1** |

### Ablation Study

| Modification | Accuracy | Change |
|------|----------|------|
| Baseline XCiT-Nano | 70.4 | - |
| + Circular convolution | 70.4 | +0.0% |
| + Global average pooling instead of Class Attn | 69.1 | -1.8% |
| + AF-LayerNorm | 69.6 | -1.1% |
| − Positional encoding | 70.7 | **+0.4%** |
| Full AF (AF Class Attention) | 70.6 | +0.3% |

### Key Findings

- Classification accuracy is maintained or marginally improved, while shift consistency increases from ~83% to ~99%.
- APS achieves perfect integer-shift consistency but falls noticeably short of AFT under sub-pixel and realistic shifts.
- Under adversarial shift attacks, AFT degrades minimally (only ~2% relative drop for the Nano variant vs. ~25% for the baseline).
- Under realistic shift scenarios including crop-shift and bilinear fractional shift, AFT consistently outperforms all baselines (including CvT, Swin, and vanilla ViT).
- Removing positional encodings not only avoids accuracy loss but yields a slight improvement, suggesting that explicit positional encodings may be redundant in hybrid architectures that include convolutional layers.

## Highlights & Insights

- The observation that **linear attention is inherently shift-equivariant** is particularly elegant — Parseval's theorem proves that $K^\top V$ is shift-invariant, so the full linear attention $Q f(K^\top V)$ is automatically shift-equivariant. This provides a principled theoretical basis for the choice of attention variant.
- The systematic integration of anti-aliasing signal processing theory into the Transformer architecture is methodologically comprehensive.
- The finding that positional encodings may be redundant in hybrid architectures warrants further attention.
- The proposed approach has direct applicability to domains requiring shift consistency, such as video generation and neural operators.

## Limitations & Future Work

- Runtime overhead is substantial: training time increases from 69 hours to 487 hours (7×), primarily due to insufficient GPU optimization of FFT-domain upsampling and downsampling operations.
- Polynomial activation functions as a replacement for GELU (which would yield theoretically perfect shift invariance) were not adopted, as experiments showed severe accuracy degradation.
- Validation is limited to image classification; downstream tasks such as detection, segmentation, and generation remain untested.
- Whether the expressive capacity of linear attention is sufficient at scale and on complex tasks remains an open question.

## Related Work & Insights

- **vs. APS**: APS guarantees perfect integer cyclic shift consistency (100%) but provides limited improvement under sub-pixel and realistic shifts. AFT trades a marginal reduction in integer consistency (99.5% vs. 100%) for substantially greater sub-pixel robustness.
- **vs. Qian et al.**: Applying a low-pass filter only after the attention layer is an incomplete solution and does not address the non-equivariance of the attention operation itself.
- **vs. AFC (Alias-Free ConvNet)**: AFT extends the ideas of AFC to Transformers; the core novel contribution is the SEA theoretical framework.
- **Insight**: Shift invariance should be treated as an explicit architectural constraint from the design stage, rather than addressed as a post-hoc fix.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematic integration of signal processing and Transformers; the SEA theoretical proof is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple shift types (cyclic / crop / bilinear) evaluated, with multi-model comparisons and ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical proofs are concise and clear; architectural diagrams are intuitive.
- Value: ⭐⭐⭐⭐ Directly valuable for applications requiring shift robustness, though runtime overhead limits general applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Normalization in Attention Dynamics](normalization_in_attention_dynamics.md)
- [\[ICML 2026\] Test-Time Training with KV Binding Is Secretly Linear Attention](../../ICML2026/others/test-time_training_with_kv_binding_is_secretly_linear_attention.md)
- [\[NeurIPS 2025\] Generalized Linear Mode Connectivity for Transformers](generalized_linear_mode_connectivity_for_transformers.md)
- [\[NeurIPS 2025\] Modeling Neural Activity with Conditionally Linear Dynamical Systems](modeling_neural_activity_with_conditionally_linear_dynamical_systems.md)
- [\[NeurIPS 2025\] Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians](recurrent_self-attention_dynamics_an_energy-agnostic_perspective_from_jacobians.md)

</div>

<!-- RELATED:END -->
