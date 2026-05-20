---
title: >-
  [Paper Note] LipNeXt: Scaling up Lipschitz-based Certified Robustness to Billion-parameter Models
description: >-
  [ICLR 2026][Lipschitz constraints] This paper proposes LipNeXt—the first unconstrained, convolution-free 1-Lipschitz architecture—which learns orthogonal matrices via manifold optimization and achieves spatial mixing thr…
tags:
  - "ICLR 2026"
  - "Lipschitz constraints"
  - "certified robustness"
  - "orthogonal matrices"
  - "manifold optimization"
  - "spatial shift module"
date: 2026-05-08
content_hash: a92c87ff3be052f9
---

# LipNeXt: Scaling up Lipschitz-based Certified Robustness to Billion-parameter Models

**Conference**: ICLR 2026
**arXiv**: [2601.18513](https://arxiv.org/abs/2601.18513)  
**Code**: None  
**Area**: Others / Adversarial Robustness
**Keywords**: Lipschitz constraints, certified robustness, orthogonal matrices, manifold optimization, spatial shift module

## TL;DR

This paper proposes LipNeXt—the first unconstrained, convolution-free 1-Lipschitz architecture—which learns orthogonal matrices via manifold optimization and achieves spatial mixing through a theoretically motivated Spatial Shift Module derived from Theorem 1. LipNeXt scales to billion-parameter models and establishes new state-of-the-art certified robust accuracy (CRA) on CIFAR-10/100, Tiny-ImageNet, and ImageNet, with a +8% CRA gain on ImageNet at $\varepsilon=1$.

## Background & Motivation

**The challenge of adversarial robustness**: Adversarial examples pose a fundamental threat to safety-critical applications such as autonomous driving, medical imaging, and malware detection. Empirical defenses cannot provide formal guarantees, leaving models vulnerable to stronger attacks.

**Two paradigms for certified robustness**: (a) Randomized smoothing (RS) provides probabilistic guarantees through noise averaging; (b) Lipschitz-based methods exploit the network's Lipschitz constant to provide deterministic (worst-case) guarantees. This paper focuses on the latter.

**Scaling bottleneck of Lipschitz methods**: Existing methods predominantly use VGG-style architectures with $\leq 32$M parameters, which already underfit on CIFAR-100 and suffer significant performance degradation on ImageNet. Performance gains from larger models saturate quickly.

**Orthogonal matrices: performance-critical yet computationally expensive**: Tight Lipschitz bounds require all weight matrices to be orthogonal. Existing explicit methods (matrix exponential SOC, Cayley transform, LOT-Orth, Cholesky-Orth) and implicit methods (AOL, CPL, SLL layers) introduce substantial computational overhead—including FFT, matrix inversions, and power iterations—that limits scalability and low-precision training.

**Attention mechanisms are incompatible with Lipschitz control**: Transformer attention lacks direct Lipschitz constraint mechanisms. Nevertheless, ConvNeXt and MetaFormer demonstrate that macro-level design principles from the Transformer era can be integrated with Lipschitz architectures.

**Core motivation**: Can one design a 1-Lipschitz architecture that requires neither constrained reparameterization nor convolutions, enabling certified robustness to benefit from scaling laws in the same way as standard training?

## Method

### Overall Architecture

A LipNeXt Block stacks four 1-Lipschitz components: orthogonal projection $R \in \mathcal{M}_C$ (channel mixing) → Spatial Shift $\mathcal{S}$ (spatial mixing) → orthogonal projection $R^\top$ (back-projection) → orthogonal linear $M$ + $\beta$-Abs activation. The complete block is:

$$Z = \sigma(M R^\top \mathcal{S}(R(X + p)) + b)$$

where $p \in \mathbb{R}^{H \times W \times 1}$ is a learnable positional encoding and $\sigma$ is the $\beta$-Abs activation. Spatial dimensions are aggregated via L2 Spatial Pooling: $[\text{L2Pool}(X)]_c = \sqrt{\sum_{h,w} X_{h,w,c}^2}$. The entire network strictly maintains the 1-Lipschitz property.

### Key Design 1: FastExp Manifold Optimization

**Core observation**: During large-model training, the learning rate $\eta \sim 10^{-3}$ is small, implying that the Frobenius norm of the skew-symmetric parameter matrix $A$ in the exponential map (Eq. 3) is also small. This motivates an adaptive truncated Taylor expansion:

$$\text{FastExp}(A) = \begin{cases} I + A + \frac{1}{2}A^2, & \|A\|_F < 0.05 \\ I + A + \frac{1}{2}A^2 + \frac{1}{6}A^3, & 0.05 \leq \|A\|_F < 0.25 \\ I + A + \frac{1}{2}A^2 + \frac{1}{6}A^3 + \frac{1}{24}A^4, & 0.25 \leq \|A\|_F < 1 \\ \exp(A), & \|A\|_F \geq 1 \end{cases}$$

**Two stabilization techniques**:

- **(a) Periodic Polar Retraction**: At the end of each epoch, SVD $X = U\Sigma V^\top$ is performed and the matrix is reset as $X \leftarrow UV^\top$, correcting accumulated truncation errors.
- **(b) Manifold Lookahead**: The standard Lookahead weight interpolation $0.5X_t + 0.5X_{t-K}$ destroys orthogonality. Instead, interpolation is performed in the tangent space over skew-symmetric updates: $X_{\text{slow}} \leftarrow X_{\text{slow}} \cdot \text{FastExp}(\frac{1}{2}\sum_{j=t-K+1}^{t} \Delta_j)$, preserving orthogonality on the manifold.

The additional per-step overhead amounts to at most **5 matrix multiplications**, far less than FFT convolutions or power iterations.

### Key Design 2: Spatial Shift Module (Theorem 1)

**Theorem 1**: Let $f_K$ be a spatial convolution with kernel $K \in \mathbb{R}^{k \times k}$, unit stride, and circular padding. $f_K$ is norm-preserving (tight 1-Lipschitz isometric), i.e., $\|f_K(X) - f_K(Y)\|_F = \|X - Y\|_F, \forall X,Y$, if and only if $K$ contains exactly one nonzero element with value $\pm 1$.

**Implication**: A norm-preserving depthwise convolution **necessarily degenerates into a spatial shift**—this theoretical result directly induces the architectural design.

**2D implementation**: Each token's features are partitioned into 5 groups (shift up / shift down / shift left / shift right / no shift), corresponding to circular shifts. An orthogonal projection $R$ mixes channels before and after the shift, ensuring that the shift does not always act on the same fixed channel subsets. The empirically optimal shift ratio is $\alpha \in \{1/8, 1/16\}$.

**Circular padding vs. zero padding**: Zero padding implicitly introduces positional information, whereas circular padding does not but guarantees norm-preservation. This paper adopts circular padding with an explicit positional encoding $p$, which experiments confirm is superior to zero-padding schemes.

### Key Design 3: β-Abs Activation

$$[\beta\text{-Abs}(\boldsymbol{x})]_i = \begin{cases} |x_i|, & i \leq \beta d \\ x_i, & \text{otherwise} \end{cases}$$

The parameter $\beta \in [0,1]$ controls the degree of nonlinearity. At $\beta = 0.5$, the commonly used MinMax activation can be expressed as: $\exists R \in \mathcal{M}_{2d}, \text{MinMax}(x) = R^\top \beta\text{-Abs}(Rx)$. The activation is 1-Lipschitz and GPU-friendly, requiring no sorting or pairing operations.

### Loss & Training

EMMA loss is used for certified robust training, following the training recipe of LiResNet++. bfloat16 precision training is supported (LiResNet requires float32 due to numerical overflow in power iteration, and BRONet requires the equivalent of float64 due to complex FFT arithmetic). Multi-class classification is handled via one-vs-rest decomposition.

## Key Experimental Results

### Table 1: CIFAR-10/100 + Tiny-ImageNet Main Results

| Dataset | Model | Params | Clean Acc | CRA@36/255 | CRA@72/255 | CRA@108/255 |
|--------|------|--------|-----------|------------|------------|-------------|
| CIFAR-10 | LiResNet | 83M | 81.0 | 69.8 | 56.3 | 42.9 |
| CIFAR-10 | BRONet | 68M | 81.6 | 70.6 | 57.2 | 42.5 |
| CIFAR-10 | **LipNeXt L32W1024** | **64M** | **81.5** | **71.2** | **59.2** | **45.9** |
| CIFAR-10 | **LipNeXt L32W2048** | **256M** | **85.0** | **73.2** | **58.8** | 43.3 |
| CIFAR-100 | LiResNet | 83M | 53.0 | 40.2 | 28.3 | 19.2 |
| CIFAR-100 | BRONet | 68M | 54.3 | 40.2 | 29.1 | 20.3 |
| CIFAR-100 | **LipNeXt L32W2048** | **256M** | **57.4** | **44.1** | **31.9** | **22.2** |
| Tiny-IN | BRONet | 75M | 41.2 | 29.0 | 19.0 | 12.1 |
| Tiny-IN | **LipNeXt L32W2048** | **256M** | **45.5** | **35.0** | **25.9** | **18.0** |

### Table 3: ImageNet Results

| Model | Params | Training Speed (min/epoch) | CRA@ε=1 | Clean@ε=36/255 | CRA@ε=36/255 |
|------|--------|-------------------|---------|----------------|---------------|
| LiResNet | 51M | 5.3 | 14.2 | 45.6 | 35.0 |
| BRONet | 86M | 10.5 | - | 49.3 | 37.6 |
| **LipNeXt 1B** | **1B** | **8.9** | **21.1** | **55.9** | **40.3** |
| **LipNeXt 2B** | **2B** | **17.8** | **22.4** | **57.0** | **41.2** |

CRA at $\varepsilon=1$ on ImageNet improves by +8% over BRONet; CRA at $\varepsilon=36/255$ improves by +3%.

### Table 4: Scaling Experiments (ImageNet 400 classes, ε=1)

| Config | Depth | Width | Clean Acc | CRA |
|------|------|------|-----------|-----|
| Fixed depth=32 | 32 | 1024→4096 | 40.5→51.7 | 22.9→30.0 |
| Fixed width=2048 | 8→128 | 2048 | 30.7→47.5 | 22.4→26.9 |
| Fixed params=1B | 32 | 4096 | **51.7** | **30.0** |
| Fixed params=1B | 64 | 2896 | 51.2 | 29.6 |

A depth of 32 layers is optimal under a fixed parameter budget. Both width and depth yield non-saturating gains.

## Key Findings

1. **Lipschitz certification can benefit from scaling**: CRA continues to improve from 1B to 2B parameters, challenging the conventional wisdom that certified robustness is confined to small models.
2. **Stability under low-precision training**: LipNeXt supports bfloat16 training, whereas LiResNet requires float32 due to numerical overflow in power iteration under bf16, and BRONet's complex FFT arithmetic is equivalent to float64. This enables LipNeXt to continuously benefit from hardware acceleration.
3. **FastExp approximation is sufficiently accurate**: The combination of adaptive Taylor truncation, periodic SVD retraction, and manifold Lookahead ensures numerical stability, with performance on par with exact matrix exponentiation.
4. **Theoretical limits of norm-preserving convolutions**: Theorem 1 proves that norm-preserving depthwise convolutions under circular padding can only be spatial shifts—a tight necessary and sufficient condition.
5. **Necessity of positional encoding**: Circular padding introduces no positional information; explicit positional encoding is required to achieve performance competitive with zero-padding. Experiments confirm that circular padding with explicit PE outperforms zero-padding.

## Highlights & Insights

- **Theory-driven architecture design**: Theorem 1 naturally derives the Spatial Shift Module from the norm-preservation condition, rather than as an empirical design choice.
- **Paradigm shift from constraints to manifolds**: Replacing orthogonal constraints from "reparameterization followed by projection" with "direct optimization on the manifold" yields a conceptually clean and computationally efficient approach (only 5 matrix multiplications per step).
- **First billion-scale certified robust model**: This work demonstrates that deterministic certification need not be confined to small models, opening new directions for subsequent research.
- **Training efficiency**: Despite being 10–20× larger, LipNeXt achieves comparable training throughput to prior work (1B model at 8.9 min/epoch vs. BRONet at 86M parameters and 10.5 min/epoch).

## Limitations & Future Work

- Only $\ell_2$ norm certification is considered; $\ell_\infty$ norm certification is more practically demanded but remains more challenging.
- Training the 2B parameter model requires 16× H100 GPUs; deployment at this scale necessitates distillation or other compression techniques.
- The maximum CRA@ε=108/255 on CIFAR-10 is lower than AOL (45.9 vs. 49.0), as AOL trades clean accuracy for robustness at large perturbation radii.
- The method has not been trained on large-scale image-text datasets, making direct comparisons with randomized smoothing approaches (which can leverage pretrained models such as CLIP) potentially incomplete.

## Related Work & Insights

| Method | Orthogonal Matrix Implementation | Spatial Mixing | Scalable | Low-Precision Training |
|------|-------------|---------|------------|-----------|
| **LipNeXt (Ours)** | Manifold optimization + FastExp | Spatial Shift (parameter-free) | ✅ 1–2B | ✅ bf16 |
| LiResNet (Hu et al., 2024) | Cholesky-Orth | Convolution + power iteration | ❌ Saturates at 83M | ❌ Requires float32 |
| BRONet (Lai et al., 2025) | Block Reflector | FFT-based frequency-domain convolution | ❌ 86M | ❌ Equivalent to float64 |

**vs. LiResNet**: LipNeXt retains the macro-level structure of LiResNet but replaces all core components—manifold optimization replaces Cholesky-Orth, and the Spatial Shift replaces convolution with power iteration—eliminating the scaling bottleneck.

**vs. BRONet**: BRONet's FFT convolutions require complex32 arithmetic, whereas LipNeXt's Spatial Shift is a parameter-free integer index operation. LipNeXt already surpasses BRONet at equal parameter counts, with the advantage growing further at larger scales.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First convolution-free, manifold-optimized billion-scale certified robust architecture; Theorem 1 provides theory-driven design
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Four datasets, scaling experiments, and extensive ablations; $\ell_\infty$ experiments are absent
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are clear, Algorithm 1 is complete, and motivation is developed progressively
- **Value**: ⭐⭐⭐⭐⭐ A significant milestone in certified robustness, demonstrating that deterministic guarantees can track modern scaling trends

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Lipschitz Bandits with Stochastic Delayed Feedback](lipschitz_bandits_with_stochastic_delayed_feedback.md)
- [\[ICLR 2026\] On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets](on_the_lipschitz_continuity_of_set_aggregation_functions_and_neural_networks_for.md)
- [\[NeurIPS 2025\] The Cost of Robustness: Tighter Bounds on Parameter Complexity for Robust Memorization in ReLU Nets](../../NeurIPS2025/others/the_cost_of_robustness_tighter_bounds_on_parameter_complexity_for_robust_memoriz.md)
- [\[ICLR 2026\] The Price of Robustness: Stable Classifiers Need Overparameterization](the_price_of_robustness_stable_classifiers_need_overparameterization.md)
- [\[AAAI 2026\] Certified but Fooled! Breaking Certified Defences with Ghost Certificates](../../AAAI2026/others/certified_but_fooled_breaking_certified_defences_with_ghost_certificates.md)

</div>

<!-- RELATED:END -->
