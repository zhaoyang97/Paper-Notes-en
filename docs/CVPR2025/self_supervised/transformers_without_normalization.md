---
title: >-
  [Paper Note] Transformers without Normalization
description: >-
  [CVPR 2025][Self-Supervised Learning][Dynamic Tanh] Discovered that the input-output mapping of LayerNorm exhibits a tanh-like shape, and proposed Dynamic Tanh (DyT) as a plug-and-play alternative to normalization layers: $\text{DyT}(x) = \gamma \odot \tanh(\alpha x) + \beta$. DyT achieves comparable or superior performance to LN across multiple tasks such as vision, language, diffusion, and speech.
tags:
  - "CVPR 2025"
  - "Self-Supervised Learning"
  - "Dynamic Tanh"
  - "Normalization Alternative"
  - "LayerNorm"
  - "Transformer Architecture"
  - "Activation Compression"
date: 2026-05-08
content_hash: 2e2ed677492f3f92
---

# Transformers without Normalization

**Conference**: CVPR 2025  
**arXiv**: [2503.10622](https://arxiv.org/abs/2503.10622)  
**Code**: [https://jiachenzhu.github.io/DyT](https://jiachenzhu.github.io/DyT)  
**Area**: Self-Supervised Learning / Network Architecture  
**Keywords**: Dynamic Tanh, Normalization Alternative, LayerNorm, Transformer Architecture, Activation Compression

## TL;DR

Discovered that the input-output mapping of LayerNorm exhibits a tanh-like shape, and proposed Dynamic Tanh (DyT) as a plug-and-play alternative to normalization layers: $\text{DyT}(x) = \gamma \odot \tanh(\alpha x) + \beta$. DyT achieves comparable or superior performance to LN across multiple tasks such as vision, language, diffusion, and speech.

## Background & Motivation

**Background**: Normalization layers (LayerNorm/RMSNorm) are standard components of Transformers, normalizing activations to a standard distribution by computing token-level statistics. Almost all Transformer variants rely on normalization.

**Limitations of Prior Work**: Normalization layers require online computation of the mean and variance for each token, which increases computational overhead and implementation complexity. More importantly, the "necessity" of normalization has never been fully understood—what exactly is it doing?

**Key Challenge**: Normalization is widely considered essential for stabilizing training, but no study has systematically investigated whether it can be replaced by a simpler operation.

**Key Insight**: By visualizing the input-output mapping of LayerNorm in trained models, an surprisingly regular S-shaped curve is observed—closely resembling the tanh function. This suggests that the core utility of normalization might simply be to "compress" the activation range.

**Core Idea**: LayerNorm $\approx$ learnable element-wise tanh compression—replacing normalization without requiring token-level statistics.

## Method

### Key Designs

1. **Dynamic Tanh (DyT)**:

    - **Function**: A normalization alternative that does not require statistics computation.
    - **Mechanism**: $\text{DyT}(x) = \gamma \odot \tanh(\alpha x) + \beta$, where $\alpha$ is a learnable scalar (controlling compression intensity), and $\gamma, \beta$ are channel-wise scaling and shifting parameters, respectively. The tanh function compresses the input into $[-1, 1]$, and $\alpha$ learns proper scaling to match the compression effect of LN.
    - **Design Motivation**: The learned value of $\alpha$ is highly correlated with the reciprocal of the activation standard deviation, indicating that DyT automatically learns a form of "global normalization" without requiring token-by-token statistical calculation.

2. **Cross-Task Universality**:

    - **Function**: Demonstrating that DyT is not merely a task-specific trick.
    - **Mechanism**: Directly replacing LN/RMSNorm across entirely different architectures and tasks, including ViT (classification), MAE/DINO (self-supervised learning), DiT (diffusion), LLaMA (language), and wav2vec (speech).
    - **Design Motivation**: If DyT were only effective on a single task, it might be a coincidence; its consistent effectiveness across multiple domains suggests it uncovers the fundamental function of normalization.

### Loss & Training

Do not alter any training configurations; simply replace LN/RMSNorm with DyT. By default, $\alpha$ is initialized to 0.5. LLaMA requires a specific initialization of $\alpha$ (refer to Section 7).

## Key Experimental Results

### Main Results

| Model/Task | DyT | LayerNorm/RMSNorm |
|----------|-----|-------------------|
| ViT-B (ImageNet) | **82.5%** | 82.3% |
| ViT-L (ImageNet) | **83.6%** | 83.1% |
| DiT-B (FID↓) | **63.9** | 64.9 |
| LLaMA-7B (loss) | ≈ Same | ≈ Same |
| wav2vec 2.0 | ≈ Same | ≈ Same |

### Ablation Study

| Configuration | ViT-B Top-1 | Description |
|------|------------|------|
| DyT (Full) | 82.5% | — |
| W/o $\alpha$ (fixed to 1) | 81.1% | $\alpha$ is critical |
| Using hardtanh | 82.0% | tanh is superior |
| Using sigmoid | 81.8% | tanh is optimal |
| Using identity | Diverged | Compression is indispensable |

### Key Findings
- **tanh compression is the core function of normalization**: the identity (no compression) variant directly leads to training divergence, validating the necessity of compression.
- **$\alpha$ dynamically learns normalization behavior**: its value is highly correlated with $1/\text{std}$ (reciprocal of standard deviation), effectively functioning as a global normalization.
- **Diffusion models benefit the most**: DyT outperforms LN on DiT (FID 63.9 vs 64.9), potentially because the time-step conditioning of diffusion interferes with token-level normalization.

## Highlights & Insights
- **Fundamental questioning of basic Transformer components**—normalization layers have been deemed indispensable for 7 years, yet this paper demonstrates that a simple tanh suffices.
- **Minimalist yet profound**—the entire method is formulated in a single-line equation, but the underlying insight (LN $\approx$ tanh) updates the understanding of what normalization does.
- **Practical implications**: DyT operates without requiring token-level statistics, making it hardware-friendly (no reduce operations) and potentially more efficient for long sequences or large batch sizes.

## Limitations & Future Work
- Since DyT does not compute per-token statistics, it may not be suitable for scenarios with large magnitude variations across tokens.
- LLaMA requires a specific initialization for $\alpha$, slightly compromising its generalizability.
- Lack of theoretical explanation—why is a tanh approximation sufficient?
- Robustness under extreme training settings (e.g., ultra-large batch sizes, mixed precision) has not been fully explored.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Disruptive insight into fundamental components
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Spans five major domains: vision, language, diffusion, speech, and self-supervised learning
- **Writing Quality**: ⭐⭐⭐⭐⭐ Concise and elegant
- **Value**: ⭐⭐⭐⭐⭐ Highly promising to reshape the fundamental paradigms of Transformer architecture design

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Representation Alignment for Diffusion Transformers without External Components](../../ICLR2026/self_supervised/representation_alignment_for_diffusion_transformers_without_external_components.md)
- [\[CVPR 2025\] SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers](sata_spatial_autocorrelation_token_analysis_for_enhancing_the_robustness_of_visi.md)
- [\[CVPR 2026\] Can You Learn to See Without Images? Procedural Warm-Up for Vision Transformers](../../CVPR2026/self_supervised/can_you_learn_to_see_without_images_procedural_warm-up_for_vision_transformers.md)
- [\[ICML 2025\] PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations](../../ICML2025/self_supervised/pde-transformer_efficient_and_versatile_transformers_for_physics_simulations.md)
- [\[ICCV 2025\] CObL: Toward Zero-Shot Ordinal Layering without User Prompting](../../ICCV2025/self_supervised/cobl_toward_zero-shot_ordinal_layering_without_user_prompting.md)

</div>

<!-- RELATED:END -->
