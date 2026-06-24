---
title: >-
  [Paper Note] On the Importance of Gaussianizing Representations
description: >-
  [ICML2025][Gaussianization] Based on information-theoretic motivations (the normal distribution is simultaneously the optimal signal and the worst-case noise distribution), this paper proposes the Normality Normalization layer. After conventional normalization, activation values are Gaussianized using a Power Transform, and scaled Gaussian noise is injected for regularization. This universally improves generalization and robustness across ViTs and ResNets without introducing…
tags:
  - "ICML2025"
  - "Gaussianization"
  - "Power Transform"
  - "Normalization Layer"
  - "Mutual Information"
  - "Noise Robustness"
  - "information theory"
date: 2026-05-08
content_hash: 07df55f2441b560d
---

# On the Importance of Gaussianizing Representations

**Conference**: ICML2025  
**arXiv**: [2505.00685](https://arxiv.org/abs/2505.00685)  
**Code**: [GitHub](https://github.com/DanielEftekhari/normality-normalization)  
**Area**: Representation Learning / Normalization  
**Keywords**: Gaussianization, Power Transform, Normalization Layer, Mutual Information, Noise Robustness, information theory

## TL;DR

Based on information-theoretic motivations (the normal distribution is simultaneously the optimal signal and the worst-case noise distribution), this paper proposes the Normality Normalization layer. After conventional normalization, activation values are Gaussianized using a Power Transform, and scaled Gaussian noise is injected for regularization. This universally improves generalization and robustness across ViTs and ResNets without introducing additional learnable parameters.

## Background & Motivation

### Core Problem

Traditional normalization layers (BN/LN/IN/GN) only constrain the mean and variance of activations, but never explicitly specify what distribution the activations should follow. This paper argues from an information-theoretic perspective that **the normal distribution is the optimal encoding distribution for feature representations in deep networks**.

### Information-Theoretic Motivation: Mutual Information Game

In an additive noise channel $Y = X + Z$, the signal $X$ aims to maximize $I(X;Y)$, while the noise $Z$ aims to minimize $I(X;Y)$. Information theory shows that under first- and second-moment constraints:

$$\min_Z \max_X I(X; X+Z) = \max_X \min_Z I(X; X+Z)$$

**The Nash equilibrium strategy for both is the normal distribution** (Theorem 2.1, Cover & Thomas 2006). This implies:

- **Maximum Information Capacity**: The normal distribution is the maximum entropy distribution given mean and variance. Encoding units with a normal distribution maximizes representation capacity.
- **Optimal Noise Robustness**: Normal signals are most robust to random perturbations; Gaussian noise is the worst-case noise, meaning robustness to it implies robustness to any random perturbation.
- **Maximum Independence**: When jointly normal, uncorrelatedness $\implies$ independence; given any correlation, variables are maximally independent under a joint normal distribution.

### Connection to Learning

- Adding noise to activations is an effective regularization method (e.g., Dropout, noise injection), and Gaussianizing activations allows the model to tolerate more regularization noise.
- Mutual information under additive Gaussian noise has a closed-form correspondence with the Minimum Mean Squared Error (MMSE) (Guo et al., 2005), providing a measurable proxy for inter-layer information transfer.

## Method

### Normality Normalization Overall Architecture

In standard normalization (BN/LN/IN/GN) and before the affine transformation, two steps are inserted:

```text
Input u → [Normalization: μ̂, σ̂²] → h → [Power Transform: ψ(h; λ̂)] → x → [Add Scaled Gaussian Noise] → y → [Affine: γ·y+β] → Output v
```

### Step 1: Power Transform Gaussianization

The Yeo-Johnson Power Transform is adopted to map the normalized activations $h$ to $x$, which is closer to a Gaussian distribution:

$$\psi(h; \lambda) = \begin{cases} \frac{1}{\lambda}\left((1+h)^{\lambda}-1\right), & h \geq 0, \lambda \neq 0 \\ \log(1+h), & h \geq 0, \lambda = 0 \\ \frac{-1}{2-\lambda}\left((1-h)^{2-\lambda}-1\right), & h < 0, \lambda \neq 2 \\ -\log(1-h), & h < 0, \lambda = 2 \end{cases}$$

The parameter $\lambda$ is obtained via maximum likelihood estimation (MLE). Utilizing the convexity of the NLL with respect to $\lambda$, a second-order Taylor expansion is performed around $\lambda_0=1$ (identity transform), and a single-step Newton-Raphson method is used to solve directly:

$$\hat{\lambda} = 1 - \frac{\mathcal{L}'(\mathbf{h}; \lambda=1)}{\mathcal{L}''(\mathbf{h}; \lambda=1)}$$

Key Design: Normalizing first and then performing the Power Transform ensures $h$ has zero mean and unit variance, which simplifies the computation of $\hat\lambda$ and improves numerical stability, introducing no additional learnable parameters.

### Step 2: Scaled Additive Gaussian Noise

During training, noise is injected into the Power Transform output:

$$y_i = x_i + z_i \cdot \xi \cdot s, \quad z_i \sim \mathcal{N}(0,1)$$

where $\xi \geq 0$ is the noise factor hyperparameter, and $s = \frac{1}{N}\sum_{i=1}^{N}|x_i - \bar{x}|$ is a channel-wise scaling factor (the $\ell_1$ norm is more robust). The gradient of $s$ is detached, meaning it is only used for noise scaling and does not directly participate in learning.

Noise is not injected during testing, similar to the training/inference discrepancy in Dropout.

## Key Experimental Results

### Table 1: ViT + LayerNormalNorm (with Data Augmentation)

| Dataset | LayerNorm | LayerNormalNorm | Gain |
|---|---|---|---|
| SVHN | 94.61±0.31 | **95.78±0.21** | +1.17 |
| CIFAR-10 | 89.97±0.16 | **91.18±0.13** | +1.21 |
| CIFAR-100 | 66.40±0.42 | **70.12±0.22** | +3.72 |
| Food101 | 73.25±0.19 | **79.11±0.09** | +5.86 |
| ImageNet Top-1 | 71.54±0.16 | **75.25±0.07** | +3.71 |
| ImageNet Top-5 | 89.40±0.11 | **92.23±0.04** | +2.83 |

### Table 2: ResNet + BatchNormalNorm (without Data Augmentation)

| Dataset | Model | BatchNorm | BatchNormalNorm | Gain |
|---|---|---|---|---|
| CIFAR-10 | RN18 | 88.89±0.07 | **90.41±0.09** | +1.52 |
| CIFAR-100 | RN18 | 62.02±0.17 | **65.82±0.11** | +3.80 |
| STL-10 | RN34 | 58.82±0.52 | **63.86±0.45** | +5.04 |
| TinyImageNet Top-1 | RN34 | 58.22±0.12 | **60.57±0.14** | +2.35 |
| Caltech101 | RN50 | 72.60±0.35 | **74.71±0.51** | +2.11 |
| Food101 | RN50 | 61.15±0.44 | **63.51±0.33** | +2.36 |

### Key Findings

- **Effective across different normalization layers**: Instance/Group/Decorrelated BN can all be augmented (on STL-10, GNN > GN, INN > IN, DBNN > DBN).
- **Robustness to width**: Outperforms BN under different width factors in WideResNet, with particularly significant improvements in small-width networks.
- **Robustness to depth**: Effective across different depths of WideResNet, with more pronounced gains as depth increases (correcting deep non-normal distribution shift).
- **Robustness to batch size**: Stable performance across different training batch sizes.
- **Noise robustness ablation**: Scaled additive noise > Gaussian Dropout > Unscaled additive noise.
- **Gaussianization degree ablation**: Performance monotonically improves as $\alpha$ goes from 0 to 1, with $\alpha=1$ (the Newton-Raphson solution) being optimal.

## Highlights & Insights

1. **Solid information-theoretic motivation**: Derived from the mutual information game, demonstrating "normal distribution as the optimal feature encoding" with a complete theoretical chain.
2. **Zero extra parameters**: Compared to existing normalization layers, it does not add any learnable parameters, benefiting purely from altering the distribution shape.
3. **Plug-and-play**: Can augment any BN/LN/IN/GN without changing the model architecture.
4. **Closed-form $\hat\lambda$ estimation**: Single-step Newton-Raphson without iterative optimization or extra hyperparameters (such as learning rate).
5. **Q-Q plot visualization**: Clearly demonstrates that the Gaussianity of activation values across layers after NormalNorm training is significantly superior to BN.
6. **Decoupled ablation of Power Transform + Noise**: Both modules independently contribute to performance improvements.

## Limitations & Future Work

1. **Running speed**: The Power Transform increases computational overhead, and the variance during training is relatively large (Appendix A.2), which may not be suitable for extremely compute-constrained scenarios.
2. **Validated only on vision tasks**: Experiments are focused on image classification (ViT/ResNet), lacking validation in other modalities such as NLP, speech, and recommendation systems.
3. **Lack of large-scale pre-training validation**: ImageNet experiments train ViT from scratch, leaving untested its performance on large models like ViT-Large/Huge or LLMs.
4. **Adversarial robustness only discussed**: The paper notes that robustness to Gaussian noise may transfer to adversarial robustness, but this is only discussed at the distribution level without actual adversarial attack experiments.
5. **Noise factor $\xi$ remains a hyperparameter**: While $\hat\lambda$ requires no hyperparameters, the noise scaling factor still needs manual tuning.
6. **Second-order approximation accuracy of NLL**: Relies on the quality of the quadratic approximation of NLL around $\lambda_0=1$, which may be less accurate for highly non-normal distributions.

## Related Work & Insights

- **Traditional Power Transform**: Box-Cox (1964) only handles positive values, whereas Yeo-Johnson (2000) generalizes it to the entire real line.
- **Evolution of normalization layers**: BN → LN → IN → GN → Decorrelated BN; this work provides orthogonal enhancements.
- **Noise regularization**: Dropout (Srivastava et al., 2014), Gaussian Dropout; the proposed scaled additive noise serves as a superior alternative.
- **Infinite-width limit**: Neal (1996), Lee (2018), etc., proved that infinite-width networks converge to Gaussian processes; NormalNorm might allow finite-width networks to approximate Gaussian processes as well.
- **Feature decoupling**: Decorrelated BN (Huang et al., 2018) de-correlates features; NormalNorm further promotes joint normality $\implies$ independence.

## Rating

- Novelty: ⭐⭐⭐⭐ — The perspective of prescribing distributions driven by information theory is novel, and the method of introducing the Power Transform to deep networks is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Thorough ablation across multiple models/datasets/configurations, but lacks validation on NLP or large-scale models.
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous theoretical derivation, clear logical flow, and rich details.
- Value: ⭐⭐⭐⭐ — A plug-and-play general normalization enhancement with high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MEXMA: Token-level Objectives Improve Sentence Representations](../../ACL2025/others/mexma_token-level_objectives_improve_sentence_representations.md)
- [\[ICML 2026\] Continual Learning of Domain-Invariant Representations](../../ICML2026/others/continual_learning_of_domain-invariant_representations.md)
- [\[NeurIPS 2025\] RNNs Perform Task Computations by Dynamically Warping Neural Representations](../../NeurIPS2025/others/rnns_perform_task_computations_by_dynamically_warping_neural_representations.md)
- [\[AAAI 2026\] Guided Perturbation Sensitivity (GPS): Detecting Adversarial Text via Embedding Stability and Word Importance](../../AAAI2026/others/guided_perturbation_sensitivity_gps_detecting_adversarial_text_via_embedding_sta.md)
- [\[ICLR 2026\] Learning Distributions over Permutations and Rankings with Factorized Representations](../../ICLR2026/others/learning_distributions_over_permutations_and_rankings_with_factorized_representa.md)

</div>

<!-- RELATED:END -->
