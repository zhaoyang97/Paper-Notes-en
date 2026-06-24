---
title: >-
  [Paper Note] An Improved Privacy and Utility Analysis of Differentially Private SGD with Bounded Domain and Smooth Losses
description: >-
  [AAAI 2026][AI Safety][Differential Privacy] Under the assumption that the loss function is only $L$-smooth (convexity is not required), this paper derives tighter closed-form RDP privacy bounds for DPSGD and, for the first time, provides a complete convergence/utility analysis in bounded domain scenarios, revealing that a smaller parameter domain diameter can simultaneously improve both privacy and utility.
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Differential Privacy"
  - "DPSGD"
  - "Rényi Differential Privacy"
  - "Bounded Domain"
  - "Privacy-Utility Trade-off"
date: 2026-05-08
content_hash: 6c46bb17ba55792e
---

# An Improved Privacy and Utility Analysis of Differentially Private SGD with Bounded Domain and Smooth Losses

**Conference**: AAAI 2026  
**arXiv**: [2502.17772v4](https://arxiv.org/abs/2502.17772v4)  
**Code**: [https://github.com/HauLiang/DPSGD-DC](https://github.com/HauLiang/DPSGD-DC)  
**Area**: Differential Privacy / Privacy-Preserving Machine Learning  
**Keywords**: Differential Privacy, DPSGD, Rényi Differential Privacy, Bounded Domain, Privacy-Utility Trade-off  

## TL;DR
Under the assumption that the loss function is only $L$-smooth (convexity is not required), this paper derives tighter closed-form RDP privacy bounds for DPSGD and, for the first time, provides a complete convergence/utility analysis in bounded domain scenarios, revealing that a smaller parameter domain diameter can simultaneously improve both privacy and utility.

## Background & Motivation
Differentially Private Stochastic Gradient Descent (DPSGD) is the dominant method for protecting training data privacy, with its core operations being gradient clipping and Gaussian noise injection. However, existing privacy analyses suffer from several limitations of prior work:

1. **Overestimated privacy loss**: Traditional composition theorems assume that all intermediate models are leaked, causing the privacy loss to grow linearly with the number of iterations, which severely overestimates the actual privacy leakage.
2. **Overly strong assumptions**: Existing improved methods (e.g., Altschuler & Talwar 2022) rely on convexity assumptions and impose strict restrictions on the Rényi parameter $\alpha$, making them inapplicable to non-convex losses commonly found in deep learning.
3. **Lack of utility analysis**: Most prior works focus solely on analyzing privacy bounds without training convergence/utility guarantees, failing to guide practical privacy-utility trade-offs.

## Core Problem
Under relaxed convexity assumptions where only $L$-smoothness is retained, can tighter RDP privacy bounds be established for DPSGD (particularly the double-clipped version, DPSGD-DC, which includes both gradient clipping and parameter projection)? Furthermore, can utility/convergence guarantees matching these privacy bounds be established to quantitatively characterize the privacy-utility trade-off?

## Method

### Overall Architecture
This paper analyzes two DPSGD variants:
- **DPSGD-GC** (Gradient Clipping version): Only performs gradient clipping and Gaussian noise injection.
- **DPSGD-DC** (Double Clipping version): Adds parameter projection to a bounded domain $\mathcal{K} = \{\theta \in \mathbb{R}^d : \|\theta\| \leq D\}$ on top of DPSGD-GC.

The analysis follows two main tracks: **privacy analysis** (deriving RDP bounds) and **utility analysis** (deriving convergence bounds), which are ultimately combined to yield the privacy-utility trade-off.

### Key Designs

1. **Noisy Smooth-Reduction Lemma (Lemma 3.2)**: This is the core technical innovation of the paper. The Gaussian noise added in DPSGD is split into two components: $\varrho_t \sim \mathcal{N}(0, \beta\sigma_{DP}^2 I_d)$ and $\varsigma_t \sim \mathcal{N}(0, (1-\beta)\sigma_{DP}^2 I_d)$. The former, combined with the clipped SGD update, forms a "noisy update function," while the latter is used to reduce the displacement of the shifted Rényi divergence via Lemma 2.6. The key is that even though the update function is not a contraction mapping (which cannot be guaranteed under non-convexity), $L$-smoothness ensures that the displacement expands at a rate of only $(1+\eta L)$, thereby controlling the growth of privacy loss after adding noise. This generalizes the contraction-reduction lemma of Feldman et al. (which requires convexity to guarantee contraction).

2. **Privacy Bound of DPSGD-GC (Theorem 3.3)**: Under the unbounded domain and $L$-smooth assumptions, DPSGD-GC satisfies $(\alpha, \varepsilon)$-RDP, where $\varepsilon = \mathcal{O}\left(\frac{\alpha C^2}{nb\sigma_{DP}^2} T\right)$. The privacy loss grows linearly with the number of iterations $T$—matching the complexity of existing methods, but requiring only the smoothness assumption.

3. **Privacy Bound of DPSGD-DC (Theorem 3.4)**: When the parameter domain is bounded (with diameter $D$), the privacy loss can converge to a constant even in non-convex scenarios. Specifically, $\varepsilon = \mathcal{O}\left(\frac{\alpha C^2}{nb\sigma_{DP}^2} \min\{T, \frac{(1+\eta L)^2 nb D^2}{\eta^2 C^2}\}\right)$. The key proof trick is early termination in the recursion: when backtracking to an iteration $\tau$, setting $z_\tau = D$ (guaranteed by the bounded domain) nullifies the base-case shifted Rényi divergence. Consequently, the privacy loss no longer grows infinitely with $T$ but saturates to a constant dependent on $D$. Under non-convexity, the bound is proportional to $D^2$ (whereas it is linear in the convex case), which aligns with intuition.

4. **Utility Bound of DPSGD-DC (Theorem 3.10)**: For population risk functions that are $L$-smooth and $\mu$-strongly convex, the paper derives an upper bound on the minimum expected optimality gap. The proof must address the triple challenges of gradient clipping, SGD sampling, and parameter projection. The analysis is divided into multiple sub-cases (the relative size of clipping threshold $C$ to SGD noise, and the relationship between gradient norm and $C$), utilizing the non-expansiveness of the projection operator and Markov's inequality. This ultimately yields a six-term upper bound, where the first two terms are optimization-related convergence terms (decaying at $1/T$), the third and fourth terms originate from clipping bias and SGD variance, and the last two terms quantify the impact of DP noise.

### Privacy-Utility Trade-off
Substitute Theorem 3.4's RDP bound into Theorem 3.10's utility bound to obtain the privacy-utility trade-off of DPSGD-DC (Proposition 3.12):
$$\mathcal{O}\left(\max\left\{\frac{D^2 dL\log(1/\delta)}{\epsilon^2 n^2}, \frac{\sigma_{SGD}^{3/2}D^{1/2}}{\mu^{1/2}}\left[\frac{dL\log(1/\delta)}{\epsilon^2 n^2}\right]^{1/4}, \frac{\sigma_{SGD}D\sqrt{d\log(1/\delta)}}{\sqrt{b}\epsilon}\right\}\right)$$

Core insight: A smaller domain diameter $D$ can simultaneously reduce both the privacy loss upper bound and the utility upper bound, thereby improving the privacy-utility trade-off.

## Key Experimental Results
The experiments validate the theoretical results by estimating the privacy level via membership inference attacks (MIA). ResNet-18 is trained on CIFAR10, and DPSGD is implemented using the Opacus library.

| Experimental Setup | Key Findings |
|---------|---------|
| DPSGD-GC, different batch sizes | Larger batch size $\rightarrow$ stronger privacy protection, but slower convergence |
| DPSGD-DC, $D \in \{20,60,100\}$ | Smaller $D$ $\rightarrow$ lower privacy leakage and stabilizes (converges) |
| Theory vs. Empirical Comparison | The privacy trend estimated by MIA is consistent with the theoretical bounds |

### Ablation Study
- **Batch size effect**: Increasing the batch size from 100 to 1000 significantly reduces the MIA-estimated privacy parameter $\hat{\epsilon}$, consistent with Theorem 3.3.
- **Bounded domain diameter effect**: When $D=20$, the privacy loss is visibly lower than when $D=100$ and converges rapidly, validating the effectiveness of the $\min$ operation in Theorem 3.4.
- **Numerical comparison** (Figure 1): Under a unified setup ($L=1, C=2, \sigma_{DP}=4, D=1, n=8, b=2$), the RDP bound of this work is strictly tighter than those of Feldman et al., Mironov's composition analysis, and Kong & Ribero et al. Only Altschuler & Talwar can be tighter, but that is because their method additionally requires convexity and stricter parameter constraints.

## Highlights & Insights
- **Requires only the smoothness assumption**: Eliminates the convexity assumption, which is almost never satisfied in deep learning, significantly broadening the applicability.
- **Convergent privacy bounds in non-convex settings**: For the first time, it is proven that even if the loss is non-convex, the privacy loss of DPSGD-DC can still converge to a constant if the parameter domain is bounded—a highly positive result.
- **Noise splitting trick**: Splitting the DP noise into two parts (with proportions $\beta$ and $1-\beta$), which are used for smooth-reduction and shift-reduction respectively, and finally optimizing over $\beta$ to obtain the tightest bound. This trick is highly generalizable to other privacy analyses.
- **Complete characterization of both privacy and utility**: Unlike prior works that only study privacy bounds or convergence analyses, this paper presents matching results on both sides simultaneously.
- **"Small $D$ simultaneously improves privacy and utility"**: This conclusion is highly practical—in practice, appropriately constraining the parameter range is a simple yet effective strategy.

## Limitations & Future Work
- **Practicality of the bounded domain assumption**: Although the paper argues that the bounded domain assumption is reasonable, choosing an appropriate value for $D$ remains non-intuitive in practical deep learning. The authors mention that this can be approximated via sequential constrained subproblems, but provide no concrete implementation.
- **Strong convexity assumption**: The utility analysis (Theorem 3.10) requires the population risk function to be strongly convex, which typically does not hold in deep learning. The non-convex utility bound is only provided for the DPSGD-GC version (adapted from Koloskova's result), while the non-convex utility analysis for DPSGD-DC is missing.
- **Non-convex RDP bound proportional to $D^2$**: Compared to the linear dependence on $D$ in the convex case, the non-convex bounds are looser by an order of magnitude, leaving room for further tightening.
- **Small experimental scale**: Evaluated only on CIFAR10 with ResNet-18, lagging empirical validation in large-scale model scenarios (e.g., LLM fine-tuning).
- **Not extended to other optimizers**: Only analyzes SGD, without covering optimizers like Adam or RMSProp that are more commonly used in deep learning.

## Related Work & Insights
- **vs. Altschuler & Talwar (NeurIPS 2022)**: They achieved a tighter constant bound (linear in $D$) through convexity + Lipschitzness + bounded domain, but under strict restrictions on the Rényi parameter $\alpha$. This paper relaxes the convexity assumption (at the cost of $D^2$ dependence) and achieves broader applicability.
- **vs. Kong & Ribero (2024)**: They handled the weakly convex case and only applied to cyclic data passes, without offering utility analysis. This paper handles general non-convex cases and provides a complete privacy-utility trade-off.
- **vs. Chien & Li (2024)**: Their privacy bounds are solutions to complex optimization problems without closed-form expressions, which are hard to manipulate. This paper offers clean closed-form expressions.

### Insights & Connections
- The noise-splitting trick of this paper (splitting DP noise into $\beta:(1-\beta)$ proportions to serve different purposes) is a valuable general methodology.
- The analysis framework of DPSGD-DC can be directly transferred to privacy-preserving federated learning scenarios.
- The conclusion that "bounded domain improves the privacy-utility trade-off" may provide insights for quantization methods in model compression (which essentially constrain parameters to a bounded domain).

## Rating
- Novelty: ⭐⭐⭐⭐ Removing the convexity assumption is a major advancement, and the noise-splitting trick is innovative, though the overall framework is still an improvement within "privacy amplification by iteration".
- Experimental Thoroughness: ⭐⭐⭐ The theoretical validation is solid, but the scale of experiments is small (only CIFAR10/ResNet-18), lacking verification in large-scale real-world applications.
- Writing Quality: ⭐⭐⭐⭐ Clear structured arguments, rigorous theorem statements, and an easy-to-read comparison with prior works (Table 1).
- Value: ⭐⭐⭐⭐ Provides the most complete privacy and utility theoretical guarantees for non-convex DPSGD to date, with practical significance for the privacy-preserving ML community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Missing Mass for Differentially Private Domain Discovery](../../ICLR2026/ai_safety/differentially_private_domain_discovery.md)
- [\[ICLR 2026\] PE-SGD: Differentially Private Deep Learning via Evolution of Gradient Subspace for Text](../../ICLR2026/ai_safety/pe-sgd_differentially_private_deep_learning_via_evolution_of_gradient_subspace_f.md)
- [\[NeurIPS 2025\] Mitigating Disparate Impact of Differentially Private Learning through Bounded Adaptive Clipping](../../NeurIPS2025/ai_safety/mitigating_disparate_impact_of_differentially_private_learning_through_bounded_a.md)
- [\[ICLR 2026\] INO-SGD: Addressing Utility Imbalance under Individualized Differential Privacy](../../ICLR2026/ai_safety/ino-sgd_addressing_utility_imbalance_under_individualized_differential_privacy.md)
- [\[ICML 2025\] Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off](../../ICML2025/ai_safety/clients_collaborate_flexible_differentially_private_federated_learning_with_guar.md)

</div>

<!-- RELATED:END -->
