---
title: >-
  [Paper Note] Polynomial, trigonometric, and tropical activations
description: >-
  [ICLR 2026][LLM Pretraining][Activation functions] This paper systematically explores a family of learnable activation functions based on orthogonal bases (Hermite polynomials, Fourier trigonometric bases) and tropicalization. By addressing the gradient explosion/vanishing issues of polynomial activations through variance-preserving initialization, it successfully replaces GELU to achieve effective training on GPT-2 and ConvNeXt.
tags:
  - "ICLR 2026"
  - "LLM Pretraining"
  - "Activation functions"
  - "Hermite polynomials"
  - "Fourier trigonometric bases"
  - "Tropical polynomials"
  - "Variance-preserving initialization"
date: 2026-05-08
content_hash: 4fae9b1073e2e747
---

# Polynomial, trigonometric, and tropical activations

**Conference**: ICLR 2026  
**arXiv**: [2502.01247](https://arxiv.org/abs/2502.01247)  
**Code**: [K-H-Ismail/torchortho](https://github.com/K-H-Ismail/torchortho)  
**Area**: LLM Pre-training  
**Keywords**: Activation functions, Hermite polynomials, Fourier trigonometric bases, Tropical polynomials, Variance-preserving initialization

## TL;DR
This paper systematically explores a family of learnable activation functions based on orthogonal bases (Hermite polynomials, Fourier trigonometric bases) and tropicalization. By addressing the gradient explosion/vanishing issues of polynomial activations through variance-preserving initialization, it successfully replaces GELU to achieve effective training on GPT-2 and ConvNeXt.

## Background & Motivation

Activation functions are core components of deep neural networks, introducing non-linearity to approximate complex functions. Since the introduction of ReLU, GELU, and SwiGLU, most modern deep learning models have adopted fixed activation functions. However, a natural question arises: **which functions can serve as activation functions for deep neural networks?** Can more expressive function families (e.g., polynomials, trigonometric functions) be used as learnable activations?

**Limitations of Prior Work**: Although polynomial activation functions theoretically possess strong approximation capabilities, they face severe gradient and activation explosion in practice. In deep networks, even low-degree polynomials lead to numerical instability due to layer-by-layer composition. This issue has long restricted the practical application of polynomial-based activations.

**Key Challenge**: Polynomial and trigonometric function families have rich mathematical structures and strong expressive power, but their direct use in deep networks causes training instability. Conversely, standard activations (ReLU, GELU) are stable but fixed in form and lack learnable adaptability.

**Key Insight**: Based on the mathematical properties of orthogonal bases, this paper designs a variance-preserving initialization scheme. This allows learnable activation functions based on orthogonal bases to be trained stably in deep networks without additional clamping mechanisms.

**Core Idea**: By selecting appropriate orthogonal bases (Hermite polynomial bases, Fourier trigonometric bases) and designing variance-preserving initialization, learnable polynomials and trigonometric functions can be used as activation functions while maintaining training stability.

## Method

### Overall Architecture
The paper addresses the long-standing question of "what functions can act as activations in deep networks" by transforming the activation function itself into a learnable object. Each scalar activation $F(x)$ is represented as a learnable linear combination of a set of **orthogonal basis** functions, with coefficients trained end-to-end. The primary difficulty is that polynomial and trigonometric bases cause variance explosion when stacked: activation values and gradients grow or decay exponentially through layers, leading to numerical overflow. The solution is a **variance-preserving initialization** that leverages orthogonal bases to obtain closed-form solutions for second moments. This allows the analytical pinning of "forward gain" and "backward gain" to equality, preventing variance drift with depth. Supported by this initialization, the authors implement three families of activations (Hermite polynomials, Fourier trigonometric, and Tropical piecewise linear) and use Hermite interpolation to seamlessly replace GELU in pre-trained models. This enables training GPT-2 and ConvNeXt from scratch without clamping, gradient clipping, or extra normalization.

### Key Designs

**1. Variance-Preserving Initialization: Stabilizing Deep Variance via Closed-Form Second Moments**

Following the logic of He initialization, a stable MLP layer requires variance conservation, meaning both the "forward gain" $\Gamma=\mathrm{Var}[x]\cdot \mathbb{E}[F(x)^2]^{-1}$ and "backward gain" $\Gamma'=\mathrm{Var}[x]\cdot \mathbb{E}[F'(x)^2]^{-1}$ should be 1. Previous work using rational functions struggled here because the second moments of rational fractions lack closed-form expressions. The key observation of this paper is that **switching to orthogonal bases provides simple closed-form solutions for second-moment integrals**. Under standard normal input assumptions, Hermite polynomials are mutually orthogonal ($\int \mathrm{He}_m \mathrm{He}_n\, e^{-x^2/2}\mathrm{d}x = \sqrt{2\pi}\,n!\,\delta_{nm}$), meaning contributions from different orders do not interfere. This allows for an analytical derivation of coefficient initializations ($a_k=1\ (k\ge1)$, $a_0=\sqrt{1-1/n!}$) that maintain unit gain.

**2. Hermite and Fourier: Complementary Orthogonal Bases**

Hermite activations expand the function into a weighted sum of probabilist's Hermite polynomials:
$$F(x) = \sum_{k=0}^{n} \frac{a_k}{k!}\,\mathrm{He}_k(x)$$
where $\mathrm{He}_k$ is the $k$-th order Hermite polynomial and $a_k$ are learnable coefficients. While polynomials excel at local approximation, they struggle with periodicity. Consequently, the authors introduce a Fourier trigonometric basis $F(x) = a_0 + \sum_{k=1}^{n}\frac{a_k\cos(kx)+b_k\sin(kx)}{k!}$, which is orthogonal under uniform distribution and suitable for capturing oscillatory structures.

**3. Tropical Activations: Generalizing ReLU to Learnable Piecewise Linear Functions**

The paper also introduces a piecewise linear family by "tropicalizing" polynomials. In tropical algebra, addition is replaced by $\max$ and multiplication by addition. A tropical polynomial thus reduces to the pointwise maximum of affine functions, forming a piecewise linear curve. This generalizes ReLU (the simplest tropical polynomial $\max(0,x)$) into learnable multi-segment lines.

**4. Hermite Interpolation Migration: Seamless Integration into Pre-trained Models**

To apply learnable activations to existing pre-trained weights without disturbing learned representations, the authors use Hermite interpolation to **simultaneously match the function value and first derivative of GELU**. This ensures the new activation is nearly identical to GELU at initialization, allowing fine-tuning to start from an equivalent point.

### Loss & Training
The training objective follows standard paradigms. Activation coefficients are optimized end-to-end: GPT-2 is trained on OpenWebText using next-token prediction with cross-entropy loss, while ConvNeXt is trained on ImageNet-1K for standard classification. No additional regularization or special scheduling is required for the learnable coefficients.

## Key Experimental Results

### Main Results

| Model/Task | Metric | GELU Baseline | Hermite | Fourier | Tropical | Description |
|-----------|------|----------|---------|---------|----------|------|
| GPT-2 / OpenWebText | Perplexity | Baseline | Lower | Lower | Comparable | Learnable activations improve language modeling |
| ConvNeXt-T / ImageNet | Top-1 Acc | Baseline | Gain | Gain | Comparable | Effective for vision tasks |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| No VP Init | Training Collapse | Initialization is a necessary condition |
| Polynomial Degree | Performance vs Stability | Higher degrees are more effective but require careful init |
| Hermite Interp. vs Scratch | Fine-tuning Efficiency | Interpolation significantly accelerates convergence |

### Key Findings
- Through variance-preserving initialization, polynomial and trigonometric activations successfully train deep models like GPT-2 and ConvNeXt.
- Learnable activations match or exceed fixed GELU activations in both language modeling and image classification.
- Tropical polynomials provide a natural generalization from ReLU to complex piecewise linear functions.
- Hermite interpolation makes it feasible to introduce learnable activations into pre-trained models.
- Variance-preserving initialization is the key to success—without it, polynomial activations overflow after a few layers.

## Highlights & Insights
- **Theoretical Elegance**: Combines orthogonal basis theory with neural network activation design through rigorous mathematical derivation.
- **High Utility**: Provides the `torchortho` library for direct replacement of standard PyTorch activations.
- **Comprehensive Coverage**: Explores polynomial (Hermite), trigonometric (Fourier), and tropical families under a unified framework.
- **Transfer-friendly**: Hermite interpolation allows seamless integration into existing pre-trained models.
- **Deep Insights**: Reveals the algebraic structure of polynomial networks as multivariate polynomial mappings.
- **Efficiency Potential**: Learnable activations may improve training efficiency by adaptively adjusting their form.

## Limitations & Future Work
- Experimental scale is relatively limited (GPT-2 124M and ConvNeXt-T); not yet scaled to LLaMA-sized models.
- Learnable activations introduce extra parameters, potentially increasing memory overhead in massive models.
- While theoretically interesting, the Tropical activation performance gains are less significant than Hermite and Fourier.
- Evolution dynamics during training are not fully explored (e.g., how shapes vary across layers).
- Initialization depends on the assumption of Gaussian inputs, which may not hold for middle-layer activations.
- Direct comparison with KAN (Kolmogorov-Arnold Network) is missing.

## Related Work & Insights
- **KAN (Kolmogorov-Arnold Networks)**: Another design using learnable activations based on B-splines.
- **SwiGLU, GeGLU**: Gated linear unit variants widely used in modern LLMs.
- **Maxout Networks**: Early exploration of piecewise linear activations; Tropical activations serve as a generalization.
- **Mish, Swish**: Automatically searched activations, contrasting with the learnable approach here.
- Insight: Learnable activations may be particularly valuable for scientific computing or PINNs where specific approximation properties are required.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Accessible, Realistic, and Fair Evaluation of Positive-Unlabeled Learning Algorithms](accessible_realistic_and_fair_evaluation_of_positive-unlabeled_learning_algorith.md)
- [\[ICLR 2026\] How Text Quality Interventions Reshape Neural Scaling Laws for LLMs: Empirical Study](how_text_quality_interventions_reshape_neural_scaling_laws_for_llms_empirical_st.md)
- [\[ICLR 2026\] Lossless Vocabulary Reduction for Auto-Regressive Language Models](lossless_vocabulary_reduction_for_auto-regressive_language_models.md)
- [\[ICLR 2026\] ADEPT: Continual Pretraining via Adaptive Expansion and Dynamic Decoupled Tuning](adept_continual_pretraining_via_adaptive_expansion_and_dynamic_decoupled_tuning.md)
- [\[ICLR 2026\] How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining](how_learning_rate_decay_wastes_your_best_data_in_curriculum-based_llm_pretrainin.md)

</div>

<!-- RELATED:END -->
