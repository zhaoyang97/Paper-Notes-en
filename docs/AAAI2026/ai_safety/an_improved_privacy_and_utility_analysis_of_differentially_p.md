---
title: >-
  [Paper Note] An Improved Privacy and Utility Analysis of Differentially Private SGD with Bounded Domain and Smooth Losses
description: >-
  [AAAI 2026][AI Safety][Differential Privacy] Under the sole assumption of $L$-smoothness (without convexity), this paper derives tighter closed-form RDP privacy bounds for DPSGD and, for the first time…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Differential Privacy"
  - "DPSGD"
  - "Rényi Differential Privacy"
  - "Bounded Domain"
  - "Privacy-Utility Trade-off"
date: 2026-05-08
content_hash: 353b81ffdcdd3bd3
---

# An Improved Privacy and Utility Analysis of Differentially Private SGD with Bounded Domain and Smooth Losses

**Conference**: AAAI 2026
**arXiv**: [2502.17772v4](https://arxiv.org/abs/2502.17772v4)  
**Code**: [https://github.com/HauLiang/DPSGD-DC](https://github.com/HauLiang/DPSGD-DC)  
**Area**: Differential Privacy / Privacy-Preserving Machine Learning
**Keywords**: Differential Privacy, DPSGD, Rényi Differential Privacy, Bounded Domain, Privacy-Utility Trade-off

## TL;DR
Under the sole assumption of $L$-smoothness (without convexity), this paper derives tighter closed-form RDP privacy bounds for DPSGD and, for the first time, provides a complete convergence/utility analysis in the bounded-domain setting, revealing that a smaller parameter domain diameter simultaneously improves both privacy and utility.

## Background & Motivation
Differentially private stochastic gradient descent (DPSGD) is the predominant method for protecting training data privacy, with gradient clipping and Gaussian noise injection as its core operations. However, existing privacy analyses suffer from several limitations:

1. **Overestimation of privacy loss**: Classical composition theorems assume all intermediate models are disclosed, causing privacy loss to grow linearly with the number of iterations and severely overestimating actual privacy leakage.
2. **Overly strong assumptions**: Existing improvements (e.g., Altschuler & Talwar 2022) rely on convexity and impose strict constraints on the Rényi parameter $\alpha$, making them inapplicable to the non-convex losses common in deep learning.
3. **Absence of utility analysis**: Most prior work focuses exclusively on privacy bounds without providing corresponding convergence/utility guarantees, offering little guidance for practical privacy-utility trade-offs.

## Core Problem
Without convexity—retaining only $L$-smoothness—can one derive tighter RDP privacy bounds for DPSGD (specifically the dual-clipping variant DPSGD-DC, which combines gradient clipping with parameter projection)? Furthermore, can utility/convergence guarantees matching these privacy bounds be established, thereby quantitatively characterizing the privacy-utility trade-off?

## Method

### Overall Architecture
The paper analyzes two DPSGD variants:
- **DPSGD-GC** (gradient clipping): gradient clipping plus Gaussian noise only.
- **DPSGD-DC** (dual clipping): DPSGD-GC augmented with parameter projection onto the bounded domain $\mathcal{K} = \{\theta \in \mathbb{R}^d : \|\theta\| \leq D\}$.

The analysis proceeds along two parallel threads—**privacy analysis** (deriving RDP bounds) and **utility analysis** (deriving convergence bounds)—which are ultimately combined to characterize the privacy-utility trade-off.

### Key Designs

1. **Noisy Smooth-Reduction Lemma (Lemma 3.2)**: This is the central technical contribution. The Gaussian noise injected by DPSGD is split into two components: $\varrho_t \sim \mathcal{N}(0, \beta\sigma_{DP}^2 I_d)$ and $\varsigma_t \sim \mathcal{N}(0, (1-\beta)\sigma_{DP}^2 I_d)$. The former, combined with the clipped SGD update, forms a "noisy update function"; the latter is used via Lemma 2.6 to reduce the shift of the shifted Rényi divergence. The key insight is that even when the update function is not a contraction (which cannot be guaranteed without convexity), $L$-smoothness implies the shift expands at most at rate $(1 + \eta L)$, so privacy loss growth remains controlled after noise injection. This generalizes the compression-reduction lemma of Feldman et al. (which requires convexity to ensure contractivity).

2. **Privacy Bound for DPSGD-GC (Theorem 3.3)**: Under the unbounded-domain, $L$-smoothness assumption, DPSGD-GC satisfies $(\alpha, \varepsilon)$-RDP with $\varepsilon = \mathcal{O}\!\left(\frac{\alpha C^2}{nb\sigma_{DP}^2} T\right)$. Privacy loss grows linearly in $T$—consistent with prior methods in terms of complexity, but requiring only smoothness.

3. **Privacy Bound for DPSGD-DC (Theorem 3.4)**: With a bounded parameter domain of diameter $D$, privacy loss converges to a constant even in the non-convex setting. Specifically, $\varepsilon = \mathcal{O}\!\left(\frac{\alpha C^2}{nb\sigma_{DP}^2} \min\!\left\{T,\, \frac{(1+\eta L)^2 nb D^2}{\eta^2 C^2}\right\}\right)$. The key proof technique is early termination in the recursion: when backtracking to iteration $\tau$, setting $z_\tau = D$ (guaranteed by the bounded domain) makes the shifted Rényi divergence at the base case zero. Consequently, privacy loss no longer grows without bound but saturates at a constant depending on $D$. In the non-convex case the bound scales as $D^2$ (versus linear in $D$ under convexity), consistent with intuition.

4. **Utility Bound for DPSGD-DC (Theorem 3.10)**: For a population risk function that is $L$-smooth and $\mu$-strongly convex, the paper derives an upper bound on the minimum expected optimality gap. The proof must handle the triple challenge of gradient clipping, SGD sampling, and parameter projection. The analysis is divided into sub-cases (relative magnitudes of clipping threshold $C$ and SGD noise; relationship between gradient norm and $C$), leveraging the non-expansiveness of the projection operator and Markov's inequality. The resulting bound comprises six terms: the first two are optimization-related convergence terms (decaying as $1/T$); the third and fourth quantify clipping bias and SGD variance, respectively; the last two capture the effect of DP noise.

### Privacy-Utility Trade-off
Substituting the RDP bound from Theorem 3.4 into the utility bound from Theorem 3.10 yields the privacy-utility trade-off for DPSGD-DC (Proposition 3.12):
$$\mathcal{O}\!\left(\max\!\left\{\frac{D^2 dL\log(1/\delta)}{\epsilon^2 n^2},\; \frac{\sigma_{SGD}^{3/2}D^{1/2}}{\mu^{1/2}}\!\left[\frac{dL\log(1/\delta)}{\epsilon^2 n^2}\right]^{1/4},\; \frac{\sigma_{SGD}D\sqrt{d\log(1/\delta)}}{\sqrt{b}\,\epsilon}\right\}\right)$$

The core insight is that a smaller domain diameter $D$ simultaneously tightens both the privacy loss upper bound and the utility upper bound, thereby improving the privacy-utility trade-off.

## Key Experimental Results
Experiments estimate empirical privacy via membership inference attacks (MIA) to validate the theoretical results. ResNet-18 is trained on CIFAR-10 using DPSGD implemented through the Opacus library.

| Experimental Setting | Key Finding |
|---|---|
| DPSGD-GC, varying batch size | Larger batch size → stronger privacy protection, but slower convergence |
| DPSGD-DC, $D \in \{20, 60, 100\}$ | Smaller $D$ → lower privacy leakage that stabilizes (converges) |
| Theory vs. experiment | MIA-estimated privacy trends are consistent with theoretical bounds |

### Ablation Study
- **Batch size effect**: Increasing batch size from 100 to 1000 substantially reduces the MIA-estimated privacy parameter $\hat{\epsilon}$, consistent with the predictions of Theorem 3.3.
- **Bounded domain diameter effect**: At $D=20$, privacy loss is markedly lower than at $D=100$ and converges quickly, validating the effectiveness of the $\min$ operation in Theorem 3.4.
- **Numerical comparison (Figure 1)**: Under a unified setting ($L=1, C=2, \sigma_{DP}=4, D=1, n=8, b=2$), the proposed RDP bound strictly outperforms those of Feldman et al., Mironov composition analysis, and Kong & Ribero; only Altschuler & Talwar may be tighter, but at the cost of additionally requiring convexity and stronger parameter restrictions.

## Highlights & Insights
- **Smoothness-only assumption**: Dropping convexity—a condition rarely satisfied in deep learning—substantially broadens applicability.
- **Convergent privacy bound without convexity**: This paper is the first to prove that, even with non-convex losses, DPSGD-DC's privacy loss converges to a constant as long as the parameter domain is bounded—a surprisingly strong result.
- **Noise-splitting technique**: Splitting the DP noise into two parts (proportions $\beta$ and $1-\beta$) serving smooth-reduction and shift-reduction respectively, then optimizing $\beta$ for the tightest bound, is a general technique transferable to other privacy analyses.
- **Complete characterization of privacy and utility**: Unlike prior work that addresses only privacy bounds or only convergence analysis, this paper simultaneously provides matching results on both sides.
- **"Smaller $D$ improves both privacy and utility"**: This finding is practically valuable—constraining the parameter space is a simple yet effective strategy in practice.

## Limitations & Future Work
- **Practicality of the bounded-domain assumption**: Although the paper argues that this assumption is reasonable, choosing an appropriate value of $D$ in practice remains non-intuitive. The authors mention approximation via sequential constrained subproblems but provide no concrete procedure.
- **Strong convexity in utility analysis**: Theorem 3.10 requires the population risk to be strongly convex, a condition that typically does not hold in deep learning. A non-convex utility bound is provided only for DPSGD-GC (borrowing from Koloskova et al.); a non-convex utility analysis for DPSGD-DC is absent.
- **$D^2$ dependence in the non-convex RDP bound**: Compared to the linear-in-$D$ dependence under convexity, the non-convex bound is one order looser and admits potential tightening.
- **Limited experimental scale**: Validation is confined to CIFAR-10 with ResNet-18; large-scale settings such as LLM fine-tuning are not evaluated.
- **Restricted to SGD**: The analysis covers only SGD, leaving out Adam, RMSProp, and other optimizers prevalent in deep learning.

## Related Work & Insights
- **vs. Altschuler & Talwar (NeurIPS 2022)**: They obtain tighter constant bounds (linear in $D$) via convexity, Lipschitz continuity, and bounded domain, but require strict constraints on the Rényi parameter $\alpha$. The proposed work relaxes convexity (at the cost of $D^2$ dependence) and broadens applicability.
- **vs. Kong & Ribero (2024)**: Their work handles weak convexity and is restricted to cyclic data traversal, without utility analysis. This paper handles general non-convex settings and provides a complete privacy-utility trade-off.
- **vs. Chien & Li (2024)**: Their privacy bound is the solution to a complex optimization problem (no closed form), making it difficult to use in practice. This paper provides a concise closed-form expression.

**Connections and inspiration**:
- The noise-splitting technique (decomposing DP noise in a $\beta:(1-\beta)$ ratio for separate purposes) is a general method worth adopting elsewhere.
- The analytical framework of DPSGD-DC is directly transferable to privacy-preserving federated learning.
- The finding that bounded domains improve the privacy-utility trade-off may have implications for quantization methods in model compression, which inherently constrain parameters to a bounded domain.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Removing the convexity assumption is a meaningful advance, and the noise-splitting technique is original; however, the work remains an improvement within the established privacy amplification by iteration framework.
- **Experimental Thoroughness**: ⭐⭐⭐ — Theoretical validation is adequate, but the experimental scale is limited (CIFAR-10/ResNet-18 only), lacking large-scale empirical evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Structure is clear, theorems are stated rigorously, and comparisons with prior work (Table 1) are immediately interpretable.
- **Value**: ⭐⭐⭐⭐ — Provides the most complete privacy and utility theoretical guarantees to date for non-convex DPSGD, offering practical guidance for the privacy-preserving ML community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mitigating Disparate Impact of Differentially Private Learning through Bounded Adaptive Clipping](../../NeurIPS2025/ai_safety/mitigating_disparate_impact_of_differentially_private_learning_through_bounded_a.md)
- [\[ICLR 2026\] Back to Square Roots: An Optimal Bound on the Matrix Factorization Error for Multi-Epoch Differentially Private SGD](../../ICLR2026/ai_safety/back_to_square_roots_an_optimal_bound_on_the_matrix_factorization_error_for_mult.md)
- [\[ICML 2026\] Position: Embodied AI Requires a Privacy-Utility Trade-off](../../ICML2026/ai_safety/position_embodied_ai_requires_a_privacy-utility_trade-off.md)
- [\[ICML 2026\] PRISM: Gauge-Invariant Tangent-Space Differentially Private LoRA](../../ICML2026/ai_safety/prism_gauge-invariant_tangent-space_differentially_private_lora.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)

</div>

<!-- RELATED:END -->
