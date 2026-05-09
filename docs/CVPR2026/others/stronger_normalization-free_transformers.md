---
title: >-
  [Paper Note] Stronger Normalization-Free Transformers
description: >-
  [CVPR 2026][Normalization-free Transformer] By systematically analyzing four key properties required for pointwise functions to replace normalization layers (zero-centeredness, boundedness, center-sensitivity, and monotonicity), this work identifies $\text{Derf}(x) = \text{erf}(\alpha x + s)$ as the optimal normalization-layer substitute through large-scale search. Derf consistently outperforms LayerNorm and DyT across vision recognition, image generation, speech representation, and DNA sequence modeling, with performance gains primarily attributable to stronger generalization rather than fitting capacity.
tags:
  - CVPR 2026
  - Normalization-free Transformer
  - pointwise function
  - Derf
  - normalization layer replacement
  - generalization
date: 2026-05-08
content_hash: c22e3bcef30c094a
---

# Stronger Normalization-Free Transformers

**Conference**: CVPR 2026
**arXiv**: [2512.10938](https://arxiv.org/abs/2512.10938)
**Code**: Available (link provided in paper)
**Area**: Others
**Keywords**: Normalization-free Transformer, pointwise function, Derf, normalization layer replacement, generalization

## TL;DR
By systematically analyzing four key properties required for pointwise functions to replace normalization layers (zero-centeredness, boundedness, center-sensitivity, and monotonicity), this work identifies $\text{Derf}(x) = \text{erf}(\alpha x + s)$ as the optimal normalization-layer substitute through large-scale search. Derf consistently outperforms LayerNorm and DyT across vision recognition, image generation, speech representation, and DNA sequence modeling, with performance gains primarily attributable to stronger generalization rather than fitting capacity.

## Background & Motivation

1. **State of the Field**: Normalization layers (BatchNorm, LayerNorm, RMSNorm) are core components of modern deep networks, stabilizing training and accelerating convergence by regulating the distribution of intermediate activations. Recently, Dynamic Tanh (DyT) demonstrated that the pointwise function $\tanh(\alpha x)$ can serve as a drop-in replacement for normalization layers with comparable performance.

2. **Limitations of Prior Work**:
   - Normalization layers rely on activation statistics (mean, variance), incurring additional memory access and synchronization overhead.
   - Certain normalization schemes are sensitive to batch size, leading to training instability at small batch sizes.
   - While DyT successfully matches normalization layer performance, it does not surpass it—the field has accepted "normalization-free ≈ normalization-based" but no work has demonstrated "normalization-free > normalization-based."

3. **Root Cause**: DyT established that pointwise functions can replace normalization layers, but it remains unclear which other functions in the design space might perform better, what functional properties are critical, and whether any pointwise function can be found that surpasses normalization layers.

4. **Paper Goals**:
   - Systematically understand which properties of pointwise functions affect training dynamics and final performance.
   - Search for the optimal design within a candidate function set.
   - Demonstrate that pointwise functions can not only replace but surpass normalization layers.

5. **Starting Point**: The analysis begins from intrinsic functional properties (zero-centeredness, boundedness, center-sensitivity, monotonicity), isolating the effect of each property through controlled experiments, and then using these principles to guide function search.

6. **Core Idea**: The S-shaped pointwise function $\text{erf}(\alpha x + s)$, which satisfies all four key properties, not only replaces normalization layers but consistently surpasses them through superior generalization.

## Method

### Overall Architecture
The work comprises two parts: (1) functional property analysis—systematically studying the impact of four properties on training; and (2) function search—identifying the optimal function within a candidate set satisfying the property constraints. The final proposal, Derf, serves as a drop-in normalization layer replacement integrated as $y = \gamma * \text{erf}(\alpha x + s) + \beta$.

### Key Designs

1. **Analysis of Four Functional Properties**:

   - **Function**: Establish design principles for pointwise functions as normalization layer substitutes.
   - **Mechanism**: Controlled experiments on ViT-Base analyze each of the four properties in isolation:
     - **Zero-centeredness**: Horizontal/vertical shift experiments show that performance is minimally affected when $|\lambda| \leq 0.5$, while training collapses when $|\lambda| \geq 2$. Outputs must remain balanced around zero.
     - **Boundedness**: Adding clipping to unbounded functions (e.g., arcsinh) consistently improves performance; introducing a linear term to bounded functions to render them unbounded degrades performance. Boundedness is important for stable optimization. There exists an upper limit on the growth rate—logquad(x) is the fastest-growing function that still converges.
     - **Center-sensitivity**: Introducing a flat region near the origin degrades performance as $\lambda$ increases, with training collapsing at $\lambda \geq 3$. Since most activations concentrate near zero, responsiveness at this region directly affects signal propagation.
     - **Monotonicity**: Monotonically increasing or decreasing functions both train normally, while non-monotonic functions (e.g., hump-shaped, oscillatory) show significantly degraded performance. Monotonicity preserves the relative ordering of activations.
   - **Design Motivation**: DyT selected tanh purely by intuition without systematic analysis. These four properties provide explicit necessary conditions for function design.

2. **Large-Scale Function Search**:

   - **Function**: Identify the optimal function within a candidate set satisfying the four-property constraints.
   - **Mechanism**: Starting from common scalar functions and CDFs (polynomial, rational, exponential, logarithmic, trigonometric, etc.), candidate subsets satisfying all four properties are generated via transformations including shifts, scaling, mirroring, rotation, and clipping. The unified form is $y = \gamma * f(\alpha x + s) + \beta$, evaluated on ViT-Base (Top-1 Acc) and DiT-B/4 and DiT-L/4 (FID). Results show that erf(x) achieves the best performance among all candidates: ViT-B 82.8% (vs. LayerNorm 82.3%), DiT-L/4 FID 43.94 (vs. 45.91).
   - **Design Motivation**: Although many S-shaped functions appear similar in form, their performance differences are substantial. Systematic search is more reliable than intuition-based selection.

3. **Dynamic erf (Derf)**:

   - **Function**: The final proposed normalization layer replacement.
   - **Mechanism**: $\text{Derf}(x) = \gamma * \text{erf}(\alpha x + s) + \beta$, where erf(x) is a scaled version of the standard Gaussian CDF, $\frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt$. $\alpha$ is initialized to 0.5, $s$ to 0, $\gamma$ to all-ones, and $\beta$ to all-zeros. As a drop-in replacement, one Derf layer replaces each pre-attention, pre-FFN, and final normalization layer. The learnable parameter $s$ is a scalar rather than a vector (experiments confirm no additional benefit from a vector form).
   - **Design Motivation**: erf(x) naturally satisfies all four properties (zero-centered, bounded in $[-1, 1]$, sensitive at the origin, strictly monotonically increasing), and its smoothness as a Gaussian CDF may favor gradient propagation compared to the exponential saturation of tanh.

### Key Findings: Generalization Rather Than Fitting
By computing training loss in evaluation mode, it is found that across all models and scales, the training loss ranking is Norm < Derf < DyT. That is, Derf has weaker fitting capacity than normalization layers yet achieves better final performance—indicating that Derf's advantage stems from stronger generalization. Since pointwise functions possess only a small number of scalar parameters ($\alpha, s$) rather than adapting based on activation statistics, they limit overfitting and act as an implicit regularizer.

## Key Experimental Results

### Main Results

| Model / Task | LayerNorm | DyT | Derf | ΔLN |
|---|---|---|---|---|
| ViT-B (ImageNet Acc↑) | 82.3% | 82.5% | **82.8%** | +0.5% |
| ViT-L (ImageNet Acc↑) | 83.1% | 83.6% | **83.8%** | +0.7% |
| DiT-B/4 (FID↓) | 64.93 | 63.94 | **63.23** | −1.70 |
| DiT-L/4 (FID↓) | 45.91 | 45.66 | **43.94** | −1.97 |
| DiT-XL/2 (FID↓) | 19.94 | 20.83 | **18.92** | −1.02 |
| wav2vec 2.0 Base (Loss↓) | 1.95 | 1.95 | **1.93** | −0.02 |
| wav2vec 2.0 Large (Loss↓) | 1.92 | 1.91 | **1.90** | −0.02 |
| HyenaDNA (Acc↑) | 85.2% | 85.2% | **85.7%** | +0.5% |
| Caduceus (Acc↑) | 86.9% | 86.9% | **87.3%** | +0.4% |
| GPT-2 (Loss↓) | 2.94 | 2.97 | **2.94** | 0.00 |

### Ablation Study — Function Search Results

| Function | ViT-B Acc↑ | DiT-L/4 FID↓ |
|---|---|---|
| erf(x) **[Derf]** | **82.8%** | **43.94** |
| tanh(x) [DyT] | 82.6% | 45.48 |
| satursin(x) | 82.6% | 44.83 |
| arctan(x) | 82.4% | 46.62 |
| isru(x) | 82.3% | 45.93 |
| linearclip(x) | 82.3% | 45.49 |
| LayerNorm | 82.3% | 45.91 |

### Ablation Study — Effect of Learnable Shift $s$

| Function | w/o $s$ | w/ $s$ | Note |
|---|---|---|---|
| erf(x) | 82.6% | 82.8% | $s$ contributes +0.2% |
| tanh(x) | 82.5% | 82.6% | $s$ contributes +0.1% |
| isru(x) | 82.2% | 82.3% | $s$ contributes +0.1% |

### Key Findings
- **Derf consistently surpasses LayerNorm and DyT across all domains**: ViT, DiT, wav2vec, and DNA models all achieve state-of-the-art results; GPT-2 is the sole exception where Derf matches LN (while still outperforming DyT).
- **The advantage of erf over tanh is not solely attributable to the shift $s$**: Without $s$, erf (82.6%) still outperforms tanh with $s$ (82.6%), and the gap is more pronounced on DiT (FID 63.39 vs. 63.94).
- **Gains originate from generalization, not fitting**: Derf incurs higher training loss than LN yet achieves better test performance, demonstrating that the simplicity of pointwise functions acts as an implicit regularizer.
- **Boundedness and center-sensitivity have the largest impact among the four properties**: Violating boundedness can cause training collapse; violating center-sensitivity leads to a sharp drop in performance.

## Highlights & Insights
- **From "can replace" to "can surpass"**: DyT demonstrated that pointwise functions ≈ normalization layers; Derf demonstrates that pointwise functions > normalization layers, marking a critical step forward in normalization-free Transformer research. This result suggests that normalization layers may not represent the optimal mechanism for activation regulation.
- **The four-property analysis constitutes reusable design principles**: These four properties provide an explicit necessary-condition checklist for designing any future pointwise function replacement. The systematic analytical methodology itself constitutes a contribution.
- **The implicit regularization interpretation is insightful**: Pointwise functions employ a fixed mapping (independent of activation statistics) → limit adaptive capacity → reduce overfitting → improve generalization. This causal chain explains why "weaker fitting = better performance," echoing classical regularization techniques such as dropout.

## Limitations & Future Work
- On GPT-2, Derf merely matches LN; whether it retains advantages at larger LLM scales (e.g., GPT-3-level) remains to be verified.
- All experiments train from scratch; the paper does not discuss how to migrate pretrained models with existing normalization layers to Derf (fine-tuning vs. retraining).
- Function search still relies on manually constructed candidate sets and grid search; it remains open whether differentiable search or meta-learning could automatically discover superior functions.
- Numerical stability of Derf under mixed-precision training (FP16/BF16) is not discussed—specifically, the precision of erf at low precision.
- Although performance gains are consistent, their absolute magnitude is modest (e.g., +0.5% on ViT-B), and whether the engineering switching cost is justified warrants consideration.

## Related Work & Insights
- **vs. DyT (Dynamic Tanh)**: Derf outperforms DyT on all tasks, primarily because the mathematical properties of erf(x) (Gaussian CDF) are better suited to activation regulation than the exponential saturation of tanh. Gains are +0.3% on ViT-B and −1.72 FID on DiT-L/4.
- **vs. LayerNorm**: Derf achieves better generalization with weaker fitting capacity, suggesting that statistics-based adaptation in normalization layers may induce mild overfitting.
- **vs. RMSNorm**: Derf also surpasses RMSNorm by +0.4% on Caduceus (which uses RMSNorm by default), indicating that Derf's advantage is not limited to replacing LN.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The four-property analysis is systematic and rigorous; the selection of erf is well-supported by extensive experiments.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Spans four domains (vision, speech, DNA, language) with highly detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The logical progression from property analysis to function search to the final proposal is exceptionally clear.
- **Value**: ⭐⭐⭐⭐ Demonstrating that pointwise functions can surpass normalization layers is an important research signal; Derf itself is a practical drop-in replacement.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Training Deep Normalization-Free Spiking Neural Networks with Lateral Inhibition](../../ICLR2026/others/training_deep_normalization-free_spiking_neural_networks_with_lateral_inhibition.md)
- [\[CVPR 2026\] ELogitNorm: Enhancing OOD Detection with Extended Logit Normalization](enhancing_outofdistribution_detection_with_extende.md)
- [\[CVPR 2026\] U-F²-CBM: CLIP-Free, Label Free, Unsupervised Concept Bottleneck Models](clipfree_label_free_unsupervised_concept_bottlenec.md)
- [\[CVPR 2026\] AdaSFormer: Adaptive Serialized Transformers for Monocular Semantic Scene Completion from Indoor Environments](adasformer_adaptive_serialized_transformers_for_monocular_semantic_scene_complet.md)
- [\[ICLR 2026\] The Counting Power of Transformers](../../ICLR2026/others/the_counting_power_of_transformers.md)

<!-- RELATED:END -->
