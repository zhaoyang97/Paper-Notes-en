---
title: >-
  [Paper Note] Estimation of Stochastic Optimal Transport Maps
description: >-
  [NeurIPS 2025][Optimization][Stochastic OT maps] This paper proposes a transport error metric $\mathcal{E}_p$ for stochastic OT maps—decomposed into an optimality gap and a feasibility gap—and constructs a computationally efficient rounding estimator achieving a near-optimal convergence rate of $\tilde{O}(n^{-1/(d+2p)})$ under minimal assumptions that require neither the existence nor uniqueness of Brenier maps. The framework is further extended to Hölder-continuous kernels and adversarially corrupted settings, establishing the first general theory for OT map estimation.
tags:
  - NeurIPS 2025
  - Optimization
  - Stochastic OT maps
  - transport error
  - finite-sample estimation
  - robust statistics
  - Markov kernels
date: 2026-05-08
content_hash: d2a7cc91c788e62f
---

# Estimation of Stochastic Optimal Transport Maps

**Conference**: NeurIPS 2025
**arXiv**: [2512.09499](https://arxiv.org/abs/2512.09499)
**Code**: None
**Area**: Optimal Transport / Statistical Learning Theory
**Keywords**: Stochastic OT maps, transport error, finite-sample estimation, robust statistics, Markov kernels

## TL;DR
This paper proposes a transport error metric $\mathcal{E}_p$ for stochastic OT maps—decomposed into an optimality gap and a feasibility gap—and constructs a computationally efficient rounding estimator achieving a near-optimal convergence rate of $\tilde{O}(n^{-1/(d+2p)})$ under minimal assumptions that require neither the existence nor uniqueness of Brenier maps. The framework is further extended to Hölder-continuous kernels and adversarially corrupted settings, establishing the first general theory for OT map estimation.

## Background & Motivation

Optimal transport (OT) provides a geometry-based principled framework for comparing and transforming probability distributions. Transport maps—its central object—find broad applications in domain adaptation, single-cell genomics, style transfer, and generative modeling. Existing estimation theory for OT maps relies almost exclusively on Brenier's theorem ($p=2$, absolutely continuous source distribution) to guarantee the existence of a uniquely determined map, further augmented by hard-to-verify regularity conditions such as density upper/lower bounds, Lipschitz continuity, and Hölder smoothness in order to obtain quantitative error bounds.

Many practical scenarios, however, fundamentally violate these prerequisites. In domain adaptation, for instance, the source distribution may lie on a low-dimensional manifold (e.g., text-to-image translation), so that no deterministic map exists. In single-cell developmental trajectory reconstruction, developmental pathways bifurcate over time, making the measure-preserving map from early to late snapshots inherently stochastic. Furthermore, the $L^p(\mu)$ error used in existing theory requires map uniqueness; without it, the metric is meaningless.

The paper's core starting point is to abandon the requirement of map uniqueness or determinism, and instead define a new quality metric $\mathcal{E}_p$ that applies uniformly to both deterministic and stochastic transport maps (i.e., Markov kernels), thereby establishing a general map estimation theory under minimal assumptions.

## Method

### Overall Architecture
The paper develops a four-layer progressive analysis centered on the newly defined transport error $\mathcal{E}_p$: (1) establishing fundamental properties and stability lemmas; (2) deriving finite-sample estimators and convergence rates without regularity assumptions; (3) obtaining improved rates under Hölder-continuous kernel assumptions via a WDRO estimator; and (4) providing robust estimation guarantees under a mixed TV+$W_p$ adversarial corruption model.

### Key Designs

1. **Transport Error $\mathcal{E}_p$ (Core Contribution)**:
    - Function: Evaluates the transport quality of a Markov kernel $\kappa$ for the $W_p(\mu,\nu)$ problem.
    - Mechanism: $\mathcal{E}_p(\kappa;\mu,\nu) = [\text{transport cost} - W_p(\mu,\nu)]_+ + W_p(\kappa_\sharp\mu, \nu)$, where the first term is the "optimality gap" (excess cost above the optimum) and the second is the "feasibility gap" (deviation of the push-forward measure from the target).
    - Design Motivation: $\mathcal{E}_p = 0$ if and only if $\kappa$ is an optimal kernel (without requiring uniqueness), while $\mathcal{E}_p \leq 2\|T - T^\star\|_{L^p(\mu)}$ preserves compatibility with existing $L^p$ benchmarks (Proposition 1). Figure 2 illustrates that $L^p$ can diverge arbitrarily from $\mathcal{E}_p$: when the deterministic map is highly oscillatory, pointwise deviations are large yet transport quality differs negligibly.

2. **Stability Lemma System (Technical Foundation)**:
    - Function: Characterizes the sensitivity of $\mathcal{E}_p$ to perturbations in the source and target distributions.
    - Mechanism: Lemma 3 gives $W_p$-stability with respect to $\nu$ (bound $2W_p(\nu,\nu')$); Lemma 4 gives $W_p$-stability with respect to $\mu$ when the kernel is Hölder continuous; Lemma 5 gives TV-stability with respect to $\mu$; Lemma 6 gives stability under kernel composition.
    - Design Motivation: These lemmas form the skeleton of all subsequent rate analyses, decomposing population-level error into empirical-level error plus the approximation error from empirical to population distributions.

3. **Rounding Estimator (Primary Estimator)**:
    - Function: Achieves near-optimal convergence rates under minimal assumptions requiring only sub-Gaussianity (or bounded $2p$-th moments).
    - Mechanism: A three-step procedure — (a) project the empirical measure $\hat\mu_n$ onto a regular grid via a rounding map to obtain $\mu_n'$; (b) solve for a near-optimal kernel $\bar\kappa_n$ on the grid; (c) return the composed kernel $\hat\kappa_n = \bar\kappa_n \circ r_{\mathcal{P}}$.
    - Design Motivation: Rounding introduces a TV perturbation (rather than a $W_p$ perturbation), allowing direct application of the TV-stability Lemma 5 for sharper rates. The key derivation chain is: $\mathcal{E}_p(\hat\kappa_n;\mu,\nu) \leq \mathcal{E}_p(\bar\kappa_n;\mu',\nu) + \sqrt{d}r \lesssim W_p(\nu,\hat\nu_n) + (nr^d)^{-1/(2p)} + \sqrt{d}r$.
    - Result: $\mathbb{E}[\mathcal{E}_p(\hat\kappa_n;\mu,\nu)] = \tilde{O}_{p,d}(n^{-1/(d+2p)})$, with computational cost $O(n^{2+o_d(1)})$ (a single low-accuracy entropic OT solve).

### Loss & Training

This work operates within a statistical estimation framework rather than machine learning training. Estimator performance is measured by the expected transport error $\mathbb{E}[\mathcal{E}_p]$. The hyperparameters of the rounding estimator (grid side length $r$, truncation radius $R$, solver precision $\delta$) can all be tuned independently of $\mu$ and $\nu$.

## Key Experimental Results

### Main Results
The paper validates the theory on two synthetic settings (Setting A: 1D→2D stochastic splitting; Setting B: orthogonal-quadrant push-apart).

| Setting | Dim $d$ | Estimator | Metric | Trend |
|---------|---------|-----------|--------|-------|
| A | 3,5,10 | NN | L1 | Consistently >1, fails to converge |
| A | 3,5,10 | NN | $\mathcal{E}_1$ | Decreases steadily with $n$ |
| A | 3,5,10 | Rounding | $\mathcal{E}_1$ | Steadily decreasing; gap with NN narrows at high $d$ |
| B | 3,5,10 | NN vs Rounding | $\mathcal{E}_1$ | NN slightly better, gap narrows with $d$ |

### Ablation Study

| Configuration | Key Metric | Remark |
|---------------|-----------|--------|
| $\mathcal{E}_1$ vs $L^1$ (Setting A) | NN's $L^1>1$ vs $\mathcal{E}_1\to 0$ | $L^p$ completely fails for irregular OT maps |
| $d=1$ special case | Convergence rate $n^{-1/2}$ | KS-distance stability allows improvement to parametric rate |

### Key Findings
- $\mathcal{E}_p$ effectively evaluates transport quality even when deterministic maps do not exist or are highly irregular, whereas the $L^p$ metric fails entirely.
- The nearest-neighbor (NN) estimator empirically outperforms the rounding estimator under $\mathcal{E}_1$, though the gap narrows in high dimensions.
- A gap remains between the upper bound $n^{-1/(d+2p)}$ and lower bound $n^{-1/(d\vee 2p)}$, which closes at $d=1$.

## Highlights & Insights
- **First General OT Map Estimation Theory**: Covers both deterministic and stochastic maps, relaxing assumptions from "Brenier map existence + smoothness" to moment conditions alone.
- **$\mathcal{E}_p$ as an Evaluation Metric, Not a Training Objective**: Remark 5 finds that directly training neural maps with $\mathcal{E}_p$ performs poorly (gradient signal weaker than Monge gap); its value lies in providing provable evaluation guarantees.
- **Clean Decoupling of TV and $W_p$ Corruption**: The dual stability of $\mathcal{E}_p$ renders adversarial estimation analysis highly streamlined, with the two corruption types contributing independently to the error.
- **Fundamental Separation in Robust Estimation**: The $d^{1/4}\rho^{1/2}$ term in the minimax lower bound demonstrates that robust map estimation is intrinsically harder than robust distribution estimation—$W_p$ guarantees from distribution estimation cannot be losslessly translated to map estimation guarantees.

## Limitations & Future Work
- A rate gap persists for $d \geq 3$ (upper bound $n^{-1/(d+2p)}$ vs. lower bound $n^{-1/(d\vee 2p)}$); multiscale analysis may help close it.
- The WDRO estimator achieves information-theoretically optimal rates but is computationally intractable; computationally efficient Lipschitz kernel estimators remain to be developed.
- Convergence rate analysis of neural network map estimators under $\mathcal{E}_p$ is an important future direction.
- The framework can in principle be extended to entropic OT, weak OT, and conditional OT variants, though stability lemmas would require tailored adaptation.

## Related Work & Insights
- **vs. Brenier Map Estimation** (Hütter & Rigollet 2021; Balakrishnan & Manole 2025): The latter requires density conditions and smoothness; $\mathcal{E}_p$ needs neither and applies to general $p \geq 1$.
- **vs. Monge Gap** (Uscidda & Cuturi 2023): Lemma 2 establishes $\mathcal{E}_p \leq 4\mathcal{E}_p'$, with equivalence at $p=1$; $\mathcal{E}_p$ is better suited for statistical analysis while the Monge gap is better suited for neural network training.
- **vs. Adversarially Robust Distribution Estimation**: The minimax lower bound term $d^{1/4}\rho^{1/2}$ versus $\rho$ for distribution estimation reveals a fundamental hardness gap between map and distribution estimation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Pioneering definition of a transport error metric applicable to stochastic OT maps, with a complete theoretical framework.
- Experimental Thoroughness: ⭐⭐⭐ Primarily theoretical; synthetic experiments adequately validate key properties but real-world applications are not addressed.
- Writing Quality: ⭐⭐⭐⭐⭐ Theorem–lemma–corollary hierarchy is clear, figures are intuitive, and technical rigor is well balanced with intuitive exposition.
- Value: ⭐⭐⭐⭐⭐ Addresses a fundamental coverage gap in OT map estimation theory, with far-reaching impact for the statistical learning and applied OT communities.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] MDNS: Masked Diffusion Neural Sampler via Stochastic Optimal Control](mdns_masked_diffusion_neural_sampler_via_stochastic_optimal_control.md)
- [\[CVPR 2026\] OTPrune: Distribution-Aligned Visual Token Pruning via Optimal Transport](../../CVPR2026/optimization/otprune_distribution-aligned_visual_token_pruning_via_optimal_transport.md)
- [\[NeurIPS 2025\] Robust Estimation Under Heterogeneous Corruption Rates](robust_estimation_under_heterogeneous_corruption_rates.md)
- [\[NeurIPS 2025\] On Minimax Estimation of Parameters in Softmax-Contaminated Mixture of Experts](on_minimax_estimation_of_parameters_in_softmax-contaminated_mixture_of_experts.md)
- [\[NeurIPS 2025\] Near-Exponential Savings for Mean Estimation with Active Learning](near-exponential_savings_for_mean_estimation_with_active_learning.md)

<!-- RELATED:END -->
