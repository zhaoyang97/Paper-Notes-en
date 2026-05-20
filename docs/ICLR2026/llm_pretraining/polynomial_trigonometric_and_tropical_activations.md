---
title: >-
  [Paper Note] Polynomial, trigonometric, and tropical activations
description: >-
  [ICLR 2026][LLM Pretraining][Activation functions] This paper systematically explores learnable activation function families based on orthogonal bases (Hermite polynomials…
tags:
  - "ICLR 2026"
  - "LLM Pretraining"
  - "Activation functions"
  - "Hermite polynomials"
  - "Fourier trigonometric basis"
  - "Tropical polynomials"
  - "Variance-preserving initialization"
date: 2026-05-08
content_hash: 9f24df0eaaec1a53
---

# Polynomial, trigonometric, and tropical activations

**Conference**: ICLR 2026
**arXiv**: [2502.01247](https://arxiv.org/abs/2502.01247)  
**Code**: [K-H-Ismail/torchortho](https://github.com/K-H-Ismail/torchortho)  
**Area**: LLM Pre-training
**Keywords**: Activation functions, Hermite polynomials, Fourier trigonometric basis, Tropical polynomials, Variance-preserving initialization

## TL;DR
This paper systematically explores learnable activation function families based on orthogonal bases (Hermite polynomials, Fourier trigonometric basis) and tropicalization, addressing the gradient explosion/vanishing problem of polynomial activations via variance-preserving initialization, and successfully replacing GELU in GPT-2 and ConvNeXt to enable stable training.

## Background & Motivation

Activation functions are a core component of deep neural networks, introducing the nonlinearity that enables networks to approximate complex functions. Since the introduction of ReLU, GELU, SwiGLU, and related activations, the vast majority of modern deep learning models have relied on fixed-form activation functions. A natural question arises: **what functions can serve as activation functions in deep neural networks?** Is it possible to employ more expressive function families—such as polynomials or trigonometric functions—as learnable activations?

- **Limitations of Prior Work**: Although polynomial activations possess strong approximation power in theory, they suffer from severe gradient explosion and activation explosion in practice. Even low-degree polynomials can cause numerical instability in deep networks due to repeated composition across layers, a problem that has long hindered the practical adoption of polynomial-type activations.

- **Key Challenge**: Polynomial and trigonometric function families offer rich mathematical structure and strong expressive capacity, yet their direct use in deep networks leads to training instability. Standard activations (ReLU, GELU), while stable, have fixed functional forms and lack learnable adaptability.

- **Key Insight**: Leveraging the mathematical properties of orthogonal bases, this paper designs variance-preserving initialization schemes that allow learnable activations built on orthogonal bases to train stably in deep networks without requiring additional clamping mechanisms.

- **Core Idea**: By selecting appropriate orthogonal bases (Hermite polynomial basis, Fourier trigonometric basis) and designing variance-preserving initialization, learnable polynomial and trigonometric activations can be used in deep networks while maintaining training stability.

## Method

### Overall Architecture
The paper proposes three families of learnable activation functions:
1. Both input and output are scalar activation values $x$;
2. Each activation function is parameterized by a set of learnable coefficients;
3. The activation functions can directly replace standard activations such as GELU/ReLU in existing networks;
4. Variance-preserving initialization ensures forward-pass variance stability.

### Key Designs

1. **Hermite Polynomial Activations**: Learnable activations are constructed from the probabilistic Hermite orthogonal polynomial basis. Hermite polynomials are orthogonal under the Gaussian measure, so when the input follows a normal distribution, contributions from each basis function are mutually independent. The activation takes the form $\sigma(x) = \sum_{k=0}^{d} c_k H_k(x)$, where $H_k$ is the $k$-th order Hermite polynomial and $c_k$ are learnable coefficients. Variance-preserving initialization requires $\sum_k c_k^2 = 1$ (exploiting orthogonality of the Hermite basis), ensuring that the output variance equals the input variance.

2. **Fourier Trigonometric Activations**: Learnable activations are constructed from the trigonometric basis of Fourier series, taking the form $\sigma(x) = a_0 + \sum_{k=1}^{d} (a_k \cos(kx) + b_k \sin(kx))$. The Fourier basis constitutes a complete orthogonal basis in the space of periodic functions and is particularly well-suited for capturing periodic structure in data. A similar variance-preserving initialization is applied to ensure training stability.

3. **Tropical Polynomial Activations**: Standard polynomials are converted to tropical polynomials via the tropicalization operation. In tropical algebra, addition is replaced by the max operation and multiplication by ordinary addition. Consequently, a tropical polynomial is essentially a pointwise maximum of a set of affine functions, forming a piecewise linear function. This can be viewed as a natural generalization of ReLU (which is the simplest tropical polynomial, $\max(0, x)$). Tropical rational functions are also introduced to further extend expressive capacity to non-convex functions.

4. **Variance-Preserving Initialization**: This is the core technical contribution. For polynomial activations, repeated composition across layers causes the variance of activations and gradients to grow or decay exponentially. The authors exploit the mathematical properties of orthogonal bases to derive initialization conditions that preserve variance under the assumption of standard normal inputs. This enables stable training of deep networks such as GPT-2 (12-layer Transformer) without requiring gradient clipping or clamping.

5. **Hermite Interpolation Transfer**: A practical contribution is demonstrating how standard activations (e.g., GELU) in pretrained models can be converted to learnable activations via Hermite interpolation, which matches both function values and derivative values simultaneously. This ensures that the new learnable activation closely approximates the original at initialization, making fine-tuning more stable and facilitating the application of learnable activations to pretrained model fine-tuning scenarios.

6. **Polynomial Interpretation of Networks**: A theoretical insight is that networks with polynomial activations can be interpreted as multivariate polynomial maps, offering a new perspective on the function approximation behavior of networks and providing tools from algebraic geometry for network analysis.

### Loss & Training
The training strategy follows standard model training:
- GPT-2 language modeling: next-token prediction on OpenWebText with cross-entropy loss;
- ConvNeXt image classification: classification training on ImageNet-1K with standard classification loss;
- The coefficients of the activation functions serve as additional learnable parameters and are optimized end-to-end via gradient descent.

## Key Experimental Results

### Main Results

| Model / Task | Metric | GELU Baseline | Hermite | Fourier | Tropical | Notes |
|---|---|---|---|---|---|---|
| GPT-2 / OpenWebText | Perplexity | Baseline | Reduced | Reduced | Comparable | Learnable activations improve language modeling |
| ConvNeXt-T / ImageNet | Top-1 Acc | Baseline | Improved | Improved | Comparable | Effective on vision tasks as well |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| Without variance-preserving initialization | Training diverges | Demonstrates that initialization is a necessary condition |
| Varying polynomial degree | Performance vs. stability | Higher degree is more effective but requires more careful initialization |
| Hermite interpolation fine-tuning vs. training from scratch | Fine-tuning efficiency | Interpolation initialization significantly accelerates convergence |

### Key Findings
- With variance-preserving initialization, polynomial and trigonometric activations can successfully train deep models such as GPT-2 (12-layer Transformer) and ConvNeXt.
- Learnable activations match or surpass fixed GELU activations on both language modeling (perplexity) and image classification (accuracy).
- Tropical polynomials provide a natural generalization from ReLU to more complex piecewise linear functions.
- Hermite interpolation makes it feasible to introduce learnable activations into pretrained models.
- Variance-preserving initialization is critical to success—without it, polynomial activations overflow numerically within a few layers.

## Highlights & Insights
- **Theoretical elegance**: The work integrates orthogonal basis theory with neural network activation design in a rigorous mathematical framework.
- **Practical utility**: The torchortho library is provided, enabling direct drop-in replacement of standard activations in PyTorch.
- **Comprehensive coverage**: Polynomial (Hermite), trigonometric (Fourier), and tropical function families are all explored under a unified analytical framework.
- **Transfer learning friendly**: The Hermite interpolation method allows learnable activations to be seamlessly integrated into existing pretrained models.
- **Deep theoretical insight**: The algebraic structure of polynomial-activation networks as multivariate polynomial maps is revealed, opening the door to analyzing neural networks with tools from algebraic geometry.
- **Efficiency potential**: Learnable activations may improve training efficiency of large-scale models by adaptively adjusting the activation shape.

## Limitations & Future Work
- Experimental scale is relatively limited: validation is only performed on GPT-2 (124M) and ConvNeXt-T, without extension to larger models (e.g., GPT-3, LLaMA).
- Learnable activations introduce additional parameters (coefficients), which may increase memory overhead in very large-scale models.
- Tropical activations, while theoretically interesting, do not yield performance improvements as pronounced as Hermite and Fourier activations.
- The dynamics of learnable activations during training remain underexplored—how does the activation shape evolve, and do different layers learn different forms?
- The variance-preserving initialization relies on the assumption that inputs follow a Gaussian distribution, which may not hold for intermediate layer activations in practice.
- Direct comparisons with concurrent learnable activation methods such as KAN (Kolmogorov-Arnold Networks) are absent.

## Related Work & Insights
- **KAN (Kolmogorov-Arnold Networks)**: An alternative network design based on learnable activation functions, using B-spline basis functions.
- **SwiGLU, GeGLU**: Gated linear unit activations widely used in LLMs.
- **Maxout Networks**: Early work exploring piecewise linear activations; tropical activations can be seen as their generalization.
- **Mish, Swish**: Activations discovered via automated search, providing contrast to the learnable approach of this paper.
- Insight: Learnable activations may be particularly valuable in settings requiring specific function approximation properties, such as scientific computing and physics-informed networks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Systematic exploration of entirely new activation function families)
- Experimental Thoroughness: ⭐⭐⭐ (Limited scale but sufficiently validated)
- Writing Quality: ⭐⭐⭐⭐ (Mathematical derivations are clearly presented)
- Value: ⭐⭐⭐⭐ (Opens new directions for activation function design, supported by open-source code)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Lossless Vocabulary Reduction for Auto-Regressive Language Models](lossless_vocabulary_reduction_for_auto-regressive_language_models.md)
- [\[ICLR 2026\] A Law of Data Reconstruction for Random Features (and Beyond)](a_law_of_data_reconstruction_for_random_features_and_beyond.md)
- [\[ICLR 2026\] Identifying and Evaluating Inactive Heads in Pretrained LLMs](identifying_and_evaluating_inactive_heads_in_pretrained_llms.md)
- [\[ICLR 2026\] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)
- [\[ICLR 2026\] Understanding the Emergence of Seemingly Useless Features in Next-Token Predictors](understanding_the_emergence_of_seemingly_useless_features_in_next-token_predicto.md)

</div>

<!-- RELATED:END -->
