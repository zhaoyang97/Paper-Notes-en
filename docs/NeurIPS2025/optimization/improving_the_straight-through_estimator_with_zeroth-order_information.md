---
title: >-
  [Paper Note] Improving the Straight-Through Estimator with Zeroth-Order Information
description: >-
  [NeurIPS 2025][Optimization][quantization-aware training] This paper proposes FOGZO (First-Order-Guided Zeroth-Order Gradient Descent), which injects STE gradients as a bias source into zeroth-order gradient estimation. By retaining the computational efficiency of STE while leveraging zeroth-order information to correct occasional erroneous gradient directions, FOGZO achieves 1–22 point improvements in accuracy/perplexity on DeiT, ResNet, and LLaMA with only 2 additional forward passes.
tags:
  - NeurIPS 2025
  - Optimization
  - quantization-aware training
  - straight-through estimator
  - zeroth-order optimization
  - FOGZO
  - gradient estimation
date: 2026-05-08
content_hash: 37a3ce0ad8390809
---

# Improving the Straight-Through Estimator with Zeroth-Order Information

**Conference**: NeurIPS 2025
**arXiv**: [2510.23926](https://arxiv.org/abs/2510.23926)
**Code**: [GitHub](https://github.com/1733116199/fogzo)
**Area**: Optimization
**Keywords**: quantization-aware training, straight-through estimator, zeroth-order optimization, FOGZO, gradient estimation

## TL;DR
This paper proposes FOGZO (First-Order-Guided Zeroth-Order Gradient Descent), which injects STE gradients as a bias source into zeroth-order gradient estimation. By retaining the computational efficiency of STE while leveraging zeroth-order information to correct occasional erroneous gradient directions, FOGZO achieves 1–22 point improvements in accuracy/perplexity on DeiT, ResNet, and LLaMA with only 2 additional forward passes.

## Background & Motivation
**Background**: Quantization-aware training (QAT) is an effective approach to obtaining low-bit models, with the core challenge being that round/sign functions have near-zero gradients almost everywhere. The Straight-Through Estimator (STE), which substitutes the Jacobian of a smooth function for that of the non-differentiable operator, is the de facto standard method in QAT.

**Limitations of Prior Work**: STE performs well at high precision but introduces parameter oscillations at low precision (1–2 bits), occasionally producing gradients in erroneous directions. Despite its weak theoretical foundations, STE has remained the dominant approach for 13 years.

**Key Challenge**: Zeroth-order methods such as n-SPSA are theoretically sounder (based on stochastic smoothing), but require $2n$ forward passes, making them highly impractical for deep networks. With small $n$, gradient variance explodes, leading to slow convergence.

**Core Idea**: STE is a "good-enough but occasionally erroneous" gradient estimator. If zeroth-order information can be used to correct these errors, it becomes possible to surpass STE's accuracy at near-STE computational cost.

## Method

### FOGZO Algorithm

**Core Formula**: A mixed perturbation vector is constructed as:

$$v_i = \sqrt{\beta} \cdot s_i \hat{g} + \sqrt{1-\beta} \cdot u_i$$

where $\hat{g} = g/\|g\|$ is the normalized STE gradient, $s_i \sim 2 \cdot \text{Ber}(0.5) - 1$ ensures zero-mean symmetry, $u_i \sim p(u)$ is an unbiased random perturbation, and $\beta$ controls the degree of trust placed in the STE.

**Gradient Estimation**:

$$G = \frac{1}{n} \sum_{i=1}^n \frac{\hat{L}(\theta + \epsilon v_i) - \hat{L}(\theta - \epsilon v_i)}{2\epsilon} v_i$$

### Heuristic Derivation
Via first-order Taylor expansion and the zero-mean property, $\mathbb{E}[G]$ is approximated as:

$$\mathbb{E}[G] \approx \beta \underbrace{\hat{g}\hat{g}^\top \nabla \hat{L}_{\text{smooth}}}_{\text{biased}} + (1-\beta) \underbrace{\nabla \hat{L}_{\text{smooth}}}_{\text{unbiased}}$$

- When STE is correct ($\hat{g}$ aligned with $\nabla \hat{L}_{\text{smooth}}$), $\hat{g}^\top \nabla \hat{L}_{\text{smooth}}$ is large and the biased term contributes significantly.
- When STE is incorrect ($\hat{g}$ orthogonal to $\nabla \hat{L}_{\text{smooth}}$), the biased term is naturally suppressed (scalar $\hat{g}^\top \nabla \hat{L}_{\text{smooth}} \approx 0$).

### Hyperparameter Selection: From STE to Implicit Smoothing

**Key Insight**: Each STE implicitly defines a form of smoothing. STE replaces the Jacobian of a non-differentiable operator $h(x)$ with that of a smooth surrogate $h_{\text{smooth}}(x)$, which can be interpreted as the expectation of the original operator under a perturbation distribution:

$$h_{\text{smooth}}(x) = \mathbb{E}_{u \sim \bar{p}(u)}[h(x + \bar{\epsilon} u)]$$

Solving this equation yields the corresponding $(\bar{\epsilon}, \bar{p}(u))$:

| STE Type | $\bar{\epsilon}$ | $\bar{p}(u)$ |
|----------|------------------|---------------|
| Identity (round) | $1/(2\sqrt{3})$ | $U(-\sqrt{3}, \sqrt{3})$ |
| Hardtanh (sign) | $1/\sqrt{3}$ | $U(-\sqrt{3}, \sqrt{3})$ |
| Tanh (sign) | $\pi/\sqrt{12}$ | $\bar{\epsilon}(1-\tanh^2(\bar{\epsilon}u))/2$ |
| ApproxSign (sign) | $1/\sqrt{6}$ | $\text{tri}(u/\sqrt{6})/\sqrt{6}$ |

In practice, $\epsilon = \alpha \bar{\epsilon}$ where $\alpha$ is the quantization scale.

### $\beta$ Scheduling Strategy
- At the start of training, $\beta = 1$ (full trust in STE).
- Linear decay to $\beta_{\min}$ (gradually introducing zeroth-order correction).
- In later training stages, smaller learning rates allow tolerance for larger gradient variance.

## Key Experimental Results

### Shallow Network Experiments (2-layer MLP, MNIST, 2-bit)

| Method | $n$ | Relative Compute | Training Loss |
|------|-----|-----------|---------|
| Identity STE | - | 1× | baseline |
| n-SPSA ($n=1$) | 1 | 3× | significantly worse than STE |
| n-SPSA ($n=7960$) | 7960 | 15920× | marginally better than STE |
| FOGZO ($\beta=0.999, n=1$) | 1 | 3× | better than STE |

**Key Finding**: FOGZO surpasses STE with $n=1$, achieving equivalent performance to n-SPSA while saving **796×** in computation.

### Deep Network Experiments (Fixed $\alpha$, Various STEs)

| Model | Dataset | Identity-STE | Identity-FOGZO | tanh-STE | tanh-FOGZO |
|------|--------|-------------|---------------|---------|-----------|
| DeiT-Tiny | ImageNet-100 | 62.72% | **70.06%** (+7.3%) | 41.98% | **46.8%** |
| LLaMA-9m | C4 (ppl) | 109.95 | **105.64** | 123.97 | **121.51** |
| ResNet-18 | ImageNet-100 | 79.92% | **80.42%** | 74.68% | **75.02%** |

### Integration with SOTA Methods (LSQ + FOGZO, 2-bit Weights)

| Model | Dataset | LSQ+STE (loss/acc) | LSQ+FOGZO (loss/acc) |
|------|--------|-------------------|---------------------|
| DeiT-Small | ImageNet-100 | 2.62 / 79.55% | **2.57 / 80.06%** |
| LLaMA-20m | 13B C4 tokens (ppl) | 50.85 | **50.61** |
| ResNet-50 | ImageNet-100 | 0.43 / 82.81% | **0.39 / 83.67%** |

### Weight–Activation Quantization (QuEST/LSQ + FOGZO, 2-bit W+A)

| Model Size | QuEST (ppl) | QuEST-FOGZO (ppl) | LSQ (ppl) | LSQ-FOGZO (ppl) |
|----------|------------|-------------------|----------|----------------|
| 95M | 37.75 | **37.37** | 39.06 | **37.38** |
| 200M | 26.63 | **26.45** | - | - |
| 300M | 22.90 | **22.72** | - | - |

### Training Time Comparison (LLaMA-30M, RTX 5090)

| Method | C4 Tokens | Perplexity | Training Time |
|------|-----------|-------|---------|
| STE | 3.522B | 38.25 | 3.7h |
| 70% STE + 30% FOGZO | 3.0B | **37.93** | 3.7h |

**Note**: Under equal training time, FOGZO achieves lower perplexity with fewer tokens, indicating higher data efficiency.

## Highlights & Insights
- **Minimal Overhead, Strong Effect**: Only 2 additional forward passes ($n=1$); implementation amounts to adding a finite-difference step after standard backward, with no optimizer modifications required.
- **Elegant Connection via Implicit Smoothing**: Inverting the STE surrogate function to recover a stochastic smoothing formulation provides principled guidance for selecting $\epsilon$ and $p(u)$ in the zeroth-order component.
- **Adaptive Suppression Mechanism**: When STE produces an erroneous direction, finite differences naturally reduce its contribution to zero; when correct, the contribution is preserved—without any additional detection mechanism.
- **Broad Compatibility**: Compatible with various STE variants (Identity, tanh, ApproxSign) and SOTA quantization methods (LSQ, QuEST).

## Limitations & Future Work
- **Weak Theoretical Guarantees**: The derivation relies primarily on heuristic arguments (first-order Taylor expansion and zero-mean assumptions), without rigorous convergence proofs.
- **$\beta_{\min}$ Requires Tuning**: Although effective values fall within a narrow range close to 1, some search is still necessary.
- **Tested Only up to 300M Parameters**: Whether gains persist at billion-scale models remains to be validated.
- **Additional Memory and Compute Overhead**: While manageable at $n=1$ (approximately 60–70% additional training time), this may still be non-negligible at very large scale.
- **The "r% STE + (100-r)% FOGZO" Mitigation Strategy**: Introduced by the authors to reduce overhead, but the optimal value of $r$ is itself an additional hyperparameter.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The idea of combining STE with zeroth-order methods is original; the analytic derivation of STE implicit smoothing is a further strength.
- **Theoretical Depth**: ⭐⭐⭐ Primarily heuristic derivations; rigorous convergence analysis is absent.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers MLP/CNN/ViT/LLM, multiple quantization methods, multiple STE variants, and training time comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear, derivations are easy to follow, and experiments are well-organized.
- **Value**: ⭐⭐⭐⭐⭐ Plug-and-play design with direct practical relevance for low-bit quantization-aware training.

## Related Work & Insights

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Zeroth-Order Fine-Tuning of LLMs in Random Subspaces](../../ICCV2025/optimization/zeroth-order_fine-tuning_of_llms_in_random_subspaces.md)
- [\[ICLR 2026\] Converge Faster, Talk Less: Hessian-Informed Federated Zeroth-Order Optimization](../../ICLR2026/optimization/converge_faster_talk_less_hessian-informed_federated_zeroth-order_optimization.md)
- [\[NeurIPS 2025\] Natural Gradient Descent for Improving Variational Inference Based Classification of Radio Galaxies](natural_gradient_descent_for_improving_variational_inference_based_classificatio.md)
- [\[NeurIPS 2025\] Unveiling m-Sharpness Through the Structure of Stochastic Gradient Noise](unveiling_m-sharpness_through_the_structure_of_stochastic_gradient_noise.md)
- [\[NeurIPS 2025\] Learning Orthogonal Multi-Index Models: A Fine-Grained Information Exponent Analysis](learning_orthogonal_multi-index_models_a_fine-grained_information_exponent_analy.md)

</div>

<!-- RELATED:END -->
