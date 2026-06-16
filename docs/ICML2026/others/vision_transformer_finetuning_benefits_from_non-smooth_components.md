---
title: >-
  [Paper Note] Vision Transformer 微调中的非光滑分量优势
description: >-
  [ICML 2026][Others][Vision Transformer] By defining a "plasticity" metric, this paper demonstrates that non-smooth components in ViTs (Attention and Feed-Forward layers) possess higher plasticity—providing larger gradient norms during fine-tuning to achieve better and more stable transfer learning performance.
tags:
  - ICML 2026
  - Others
  - Vision Transformer
  - Fine-tuning
date: 2026-05-08
content_hash: be63d83366e70241
---
# Advantages of Non-Smooth Components in Vision Transformer Fine-Tuning

**Conference**: ICML 2026  
**arXiv**: [2602.06883](https://arxiv.org/abs/2602.06883)  
**Code**: https://github.com/ambroiseodt/vit-plasticity  
**Area**: Model Compression / Transfer Learning / Parameter-Efficient Fine-Tuning  
**Keywords**: Plasticity, Vision Transformer, Fine-Tuning, Parameter-Efficient, Smoothness

## TL;DR
By defining a "plasticity" metric, this paper demonstrates that non-smooth components in ViTs (Attention and Feed-Forward layers) possess higher plasticity—providing larger gradient norms during fine-tuning to achieve better and more stable transfer learning performance.

## Background & Motivation

**Background**: ViT has become the standard backbone in vision and NLP, with a universal paradigm of pre-training followed by fine-tuning on downstream tasks. PEFT methods have become the industry standard, but a theoretical understanding of the adaptability of individual components is lacking.

**Limitations of Prior Work**: Current research focuses on which parameters to update (Attention, Feed-Forward, Normalization layers) but lacks principled guidance. Smoothness is generally considered beneficial (improving generalization, stability, and adversarial robustness), but its role in the context of transfer learning is rarely explored.

**Key Challenge**: Excessive smoothness (low Lipschitz constant), while beneficial for generalization, limits the model's ability to respond to input changes and hinders its adaptation to downstream data.

**Goal**: To replace pure smoothness constraints with "plasticity" (the average response rate to input changes) as a guiding principle for selecting highly adaptable components during fine-tuning.

**Key Insight**: Proposes a theoretical-empirical combined approach based on the analysis of ViT architecture.

**Core Idea**: High plasticity (low smoothness) allows for larger gradient norms and accelerates optimization convergence—contrary to the pursuit of smoothness.

## Method

### Overall Architecture
(1) Define the plasticity metric; (2) Derive plasticity upper bounds for each component; (3) Verify theoretical rankings on large-scale pre-trained models; (4) Validate the correlation between plasticity and performance through >1000 fine-tuning experiments.

### Key Designs

**1. Plasticity Metric: Replacing Pure Smoothness with "Average Response Rate to Input Changes"**

Classic intuition suggests that smoothness (low Lipschitz constant) is always beneficial for generalization, stability, and adversarial robustness. This paper argues the opposite: excessive smoothness limits the model's response to input changes, hindering its adaptation to downstream data. To this end, "responsiveness" is quantified as a plasticity metric:

$$P(f) = \mathbb{E}_{(x,y) \sim \nu}\left[\frac{\|f(x)-f(y)\|_F}{\|x-y\|_F}\right],$$

which relates to the Lipschitz constant as $P(f)\le\text{Lip}(f)$, but captures average behavior rather than the worst case. The key bridge lies in the input-parameter smoothness correlation (Béthune et al., 2024): high plasticity implies larger gradient norms during fine-tuning, accelerating optimization convergence. Thus, the question of "which component to fine-tune" is translated into the computable problem of "which component has higher plasticity."

**2. Theoretical Plasticity Ranking: Deriving and Ordering Upper Bounds**

The next step is to derive upper bounds for each ViT component to obtain an experimentally testable ranking. For LayerNorm, $P(f)\le\frac{\|\gamma\|_\infty}{\sigma}$; for Feed-Forward layers, $P(f)\le\|W\|_2$; for Multi-Head Attention, $P(f)\le\sum_h \|O^h\|_2\|V^h\|_2\sqrt{3n+(12n+3)r^4\|A^h\|_2^2}$. The explicit dependence on sequence length $n$ in the Attention bound is the primary source of its high plasticity. Combining spectral norm comparisons and sequence length dependence, the relative ranking is MHA > FC1 ≈ FC2 > LN2 ≈ LN1. This ranking transforms the empirical intuition of "why Attention/FFN are better for fine-tuning" into a predictive theory.

**3. Component-Isolated Fine-Tuning: Aligning Theoretical Ranking with Performance**

To verify that the plasticity ranking corresponds to fine-tuning performance without interference from component interactions, the authors performed component-isolated fine-tuning. In this setup, only one type of component is updated while all other weights remain frozen. Experiments were conducted on large-scale ViTs (86M/307M/632M) across 11 classification benchmarks × 3 seeds × 4 learning rates (approx. 1000 runs). The isolation design eliminates synergistic effects, allowing the causal chain "High Plasticity → Large Gradient Norms → Faster Convergence, LR Robustness, Higher Accuracy" to be clearly observed. Results matched the theoretical ranking across all benchmarks, with Attention and FFN showing significant advantages on difficult datasets (Cifar100, Clipart, Sketch).

## Key Experimental Results

### Main Results

| Component | Cifar10 | Cifar100 | Clipart | Sketch | Average Accuracy | Key Feature |
|-----------|---------|----------|---------|--------|------------------|-------------|
| MHA (Attention) | 93.2 | 84.1 | 78.5 | 62.1 | 90.8 | Highest Plasticity |
| FC1 (FFN 1) | 93.0 | 83.8 | 78.1 | 61.9 | 90.7 | Second Highest |
| FC2 (FFN 2) | 92.6 | 83.2 | 77.6 | 61.5 | 90.3 | Medium Plasticity |
| LN2 (Norm 2) | 92.1 | 82.4 | 76.8 | 60.2 | 89.9 | Low Plasticity |
| LN1 (Norm 1) | 92.0 | 82.1 | 76.5 | 59.8 | 89.8 | Lowest Plasticity |

### Plasticity Correlation

| Metric | MHA | FC1 | FC2 | LN2 | LN1 | Note |
|--------|-----|-----|-----|-----|-----|------|
| Plasticity Ranking | 1 | 2 | 3 | 4 | 5 | Consistent with theory |
| Peak Gradient Norm | 2.0 | 1.8 | 1.5 | 1.1 | 0.2 | Higher plasticity -> Larger gradients |
| Val Loss Decay Rate | Fast | Fast | Fast | Slow | Slow | Faster optimization convergence |
| LR Robustness | High | High | Med | Low | Low | Stability across hyperparameters |

### Key Findings
- Attention modules and Feed-Forward layers are significantly superior across most benchmarks, particularly on challenging datasets (Cifar100, Clipart, Sketch).
- Gradient norms align with plasticity rankings: high plasticity → large gradients → rapid optimization.
- Sensitivity of fine-tuning performance to learning rates decreases as plasticity increases.

## Highlights & Insights
- **Reversing Conventional Wisdom**: Overturns the classic assumption that "smoothness is always beneficial" by proving that non-smoothness (high plasticity) is advantageous in transfer learning scenarios.
- **Theoretical and Empirical Unity**: Provides a complete chain from gradient bounds and plasticity definitions to large-scale experiments (>1000 runs).
- **Cross-Architecture Consistency**: The same patterns hold across ViT-Base/Large/Huge, DINOv3, and GPT-2.

## Limitations & Future Work
- Experiments are limited to classification tasks; generalization to dense prediction tasks like detection and segmentation requires validation.
- The isolated component setup avoids analyzing interaction effects; actual synergistic updates of multiple components might yield different dynamics.
- The plasticity definition assumes a uniform distribution; generalization to other distributions or domain-specific data remains to be explored.

## Related Work & Insights
- **vs PEFT (e.g., LoRA)**: Provides a theoretical basis for showing that direct fine-tuning of high-plasticity components can be more efficient (28K parameters vs 400K in LoRA, with comparable performance on Cifar100).
- **vs Full Fine-Tuning**: Single-component fine-tuning often outperforms full fine-tuning, indicating inherent differences in the "teachability" of different parts.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Challenges common knowledge that smoothness is always beneficial.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Over 1000 fine-tuning runs across 11 benchmarks and multiple model architectures.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivation, though some sections are formula-heavy.
- Value: ⭐⭐⭐⭐⭐ Provides direct guidance for component selection in efficient fine-tuning, offering practical reference for PEFT design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Frequency-Aware Token Reduction for Efficient Vision Transformer](../../NeurIPS2025/others/frequency-aware_token_reduction_for_efficient_vision_transformer.md)
- [\[CVPR 2026\] ViT3: Unlocking Test-Time Training in Vision](../../CVPR2026/others/vit3_unlocking_test_time_training_in_vision.md)
- [\[CVPR 2025\] CARE Transformer: Mobile-Friendly Linear Visual Transformer via Decoupled Dual Interaction](../../CVPR2025/others/care_transformer_linear_attention.md)
- [\[AAAI 2026\] Enhancing Noise Resilience in Face Clustering via Sparse Differential Transformer](../../AAAI2026/others/enhancing_noise_resilience_in_face_clustering_via_sparse_differential_transforme.md)
- [\[CVPR 2026\] Computer Vision with a Superpixelation Camera](../../CVPR2026/others/computer_vision_with_a_superpixelation_camera.md)

</div>

<!-- RELATED:END -->
