---
title: >-
  [Paper Note] Advantages of Non-Smooth Components in Vision Transformer Fine-Tuning
description: >-
  [ICML 2026][Plasticity] By defining a "plasticity" metric, this paper demonstrates that non-smooth components in ViT (Attention and Feed-Forward layers) exhibit higher plasticity—enabling larger gradient norms during fin…
tags:
  - "ICML 2026"
  - "Plasticity"
  - "Vision Transformer"
  - "Fine-Tuning"
  - "Parameter Efficiency"
  - "Smoothness"
date: 2026-05-08
content_hash: 232303973c348242
---

# Advantages of Non-Smooth Components in Vision Transformer Fine-Tuning

**Conference**: ICML 2026  
**arXiv**: [2602.06883](https://arxiv.org/abs/2602.06883)  
**Code**: https://github.com/ambroiseodt/vit-plasticity  
**Area**: Model Compression / Transfer Learning / Parameter-Efficient Fine-Tuning  
**Keywords**: Plasticity, Vision Transformer, Fine-Tuning, Parameter Efficiency, Smoothness

## TL;DR
By defining a "plasticity" metric, this paper demonstrates that non-smooth components in ViT (Attention and Feed-Forward layers) exhibit higher plasticity—enabling larger gradient norms during fine-tuning to achieve superior and stable transfer learning performance.

## Background & Motivation

**Background**: ViT has become the standard backbone in vision and NLP, with a universal paradigm of pre-training followed by fine-tuning on downstream tasks. PEFT methods have become the industry standard, yet there is a lack of theoretical understanding regarding the adaptability of individual components.

**Limitations of Prior Work**: Current research focuses on which parameters need updating (Attention, FFN, Normalization layers) but lacks principled guidance. Smoothness is generally considered beneficial (improving generalization, stability, and adversarial robustness), but its role in the context of transfer learning is rarely explored.

**Key Challenge**: Excessive smoothness (low Lipschitz constant), while beneficial for generalization, limits the model's ability to respond to input changes, which in turn hinders its adaptation to downstream data.

**Goal**: Replace simple smoothness constraints with "plasticity" (the average response rate to input changes) as a guiding principle for selecting highly adaptable components during fine-tuning.

**Key Insight**: Based on ViT architecture analysis, a combined theoretical-empirical approach is proposed.

**Core Idea**: High plasticity (low smoothness) allows for larger gradient norms, accelerating optimization convergence—which is exactly opposite to the pursuit of smoothness.

## Method

### Overall Architecture
(1) Define the plasticity metric; (2) Derive plasticity upper bounds for each component; (3) Validate theoretical rankings on large-scale pre-trained models; (4) Verify the correspondence between plasticity and performance through >1000 fine-tuning experiments.

### Key Designs

1. **Plasticity Metric**:

    - **Function**: Quantifies the degree to which Transformer components respond to input changes.
    - **Mechanism**: Define $P(f) = \mathbb{E}_{(x,y) \sim \nu}\left[\frac{\|f(x)-f(y)\|_F}{\|x-y\|_F}\right]$. Plasticity is related to the Lipschitz constant $P(f) \leq \text{Lip}(f)$ but captures average behavior rather than the worst-case scenario.
    - **Design Motivation**: It links to input gradient bounds; based on the theory of Béthune et al. (2024), input-parameter smoothness is related, where high plasticity allows for larger gradient norms.

2. **Theoretical Plasticity Ranking**:

    - **Function**: Derives plasticity upper bounds for each component.
    - **Mechanism**: For LayerNorm, $P(f) \leq \frac{\|\gamma\|_\infty}{\sigma}$; for FFN layers, $P(f) \leq \|W\|_2$; for Multi-Head Attention, $P(f) \leq \sum_h \|O^h\|_2\|V^h\|_2\sqrt{3n + (12n+3)r^4\|A^h\|_2^2}$. Relative ranking: MHA > FC1 ≈ FC2 > LN2 ≈ LN1.
    - **Design Motivation**: Precisely characterize why Attention and FFN possess higher plasticity through spectral norm comparisons and sequence length dependency.

3. **Component-wise Isolated Fine-Tuning**:

    - **Function**: Fine-tune individual components on large-scale ViT (86M/307M/632M parameters).
    - **Mechanism**: Each configuration fixes other weights and only updates one class of components; grid search across 11 classification benchmarks × 3 random seeds × 4 learning rates, totaling ~1000 experiments.
    - **Design Motivation**: Avoid confounding effects of component interactions and directly verify the correspondence between plasticity and fine-tuning performance.

## Key Experimental Results

### Main Results

| Component | Cifar10 | Cifar100 | Clipart | Sketch | Avg Accuracy | Key Features |
|-----------|---------|----------|---------|--------|--------------|--------------|
| MHA (Attention) | 93.2 | 84.1 | 78.5 | 62.1 | 90.8 | Highest plasticity |
| FC1 (FFN 1) | 93.0 | 83.8 | 78.1 | 61.9 | 90.7 | Second highest plasticity |
| FC2 (FFN 2) | 92.6 | 83.2 | 77.6 | 61.5 | 90.3 | Medium plasticity |
| LN2 (Norm 2) | 92.1 | 82.4 | 76.8 | 60.2 | 89.9 | Low plasticity |
| LN1 (Norm 1) | 92.0 | 82.1 | 76.5 | 59.8 | 89.8 | Lowest plasticity |

### Key Findings
- Attention modules and FFN layers are significantly superior on most benchmarks, particularly on difficult datasets (Cifar100, Clipart, Sketch).
- Gradient norms align with plasticity rankings—high plasticity → large gradients → fast optimization.
- Sensitivity of fine-tuning performance to learning rates decreases as plasticity increases.

## Highlights & Insights
- **Anti-Conventional Wisdom**: Overturns the classic assumption that "smoothness is always beneficial," proving that non-smoothness (high plasticity) is more advantageous in transfer learning scenarios.
- **Unification of Theory and Empiricism**: A complete chain from gradient bounds and plasticity definitions to large-scale 1000+ experiment validations.
- **Cross-Architecture Consistency**: The same patterns hold across ViT-Base/Large/Huge, DINOv3, and GPT-2.

## Limitations & Future Work
- Experiments are limited to classification tasks; generalization to dense prediction tasks like detection and segmentation requires verification.
- The component isolation setup avoids analyzing interaction effects; actual synergistic updates of multiple components might generate new dynamics.
- The plasticity definition is based on a uniform distribution assumption; generalization to other distributions or specific domains remains to be explored.

## Related Work & Insights
- **vs. PEFT like LoRA**: This paper provides a theoretical basis showing that directly fine-tuning high-plasticity components can be more efficient (28K parameters vs. LoRA 400K, matching performance on Cifar100).
- **vs. Full Fine-Tuning**: Single-component fine-tuning often outperforms full fine-tuning, indicating differences in "teachability" across various parts.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Breaks the common sense that smoothness is always beneficial.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Over 1000 fine-tuning runs + 11 benchmarks + across multiple model architectures.
- **Writing Quality**: ⭐⭐⭐⭐ Clear theoretical derivation, though some space is occupied by complex formulas.
- **Value**: ⭐⭐⭐⭐⭐ Directly guides the selection of components for efficient fine-tuning, providing practical reference value for PEFT method design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Frequency-Aware Token Reduction for Efficient Vision Transformer](../../NeurIPS2025/others/frequency-aware_token_reduction_for_efficient_vision_transformer.md)
- [\[ICML 2026\] FOVI: Bio-inspired Foveated Interface for Deep Vision Models](fovi_a_biologically-inspired_foveated_interface_for_deep_vision_models.md)
- [\[ICML 2026\] Decoupled Conformal Optimisation: Efficient Prediction Sets via Independent Tuning and Calibration](decoupled_conformal_optimisation_efficient_prediction_sets_via_independent_tunin.md)
- [\[CVPR 2026\] ViT3: Unlocking Test-Time Training in Vision](../../CVPR2026/others/vit3_unlocking_test_time_training_in_vision.md)
- [\[ICML 2026\] Comprehensive AI Governance Requires Addressing Non-Model Gains](comprehensive_ai_governance_requires_addressing_non-model_gains.md)

</div>

<!-- RELATED:END -->
