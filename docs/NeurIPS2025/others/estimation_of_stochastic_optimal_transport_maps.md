---
title: >-
  [Paper Note] Estimation of Stochastic Optimal Transport Maps
description: >-
  [NeurIPS 2025][Stochastic OT maps] This paper introduces a transport error metric $\mathcal{E}_p$ for stochastic OT maps, decomposed into an optimality gap and a feasibility gap. Under minimal assumptions that require neither the existence nor uniqueness of a Brenier map, a computationally efficient rounding estimator is constructed that achieves a near-optimal convergence rate of $\tilde{O}(n^{-1/(d+2p)})$. The framework is further extended to Hölder-continuous kernels and a…
tags:
  - "NeurIPS 2025"
  - "Stochastic OT maps"
  - "transport error"
  - "finite-sample estimation"
  - "robust statistics"
  - "Markov kernels"
date: 2026-05-08
content_hash: 216c147c48be98e5
---

# Estimation of Stochastic Optimal Transport Maps

**Conference**: NeurIPS 2025
**arXiv**: [2512.09499](https://arxiv.org/abs/2512.09499)  
**Code**: None  
**Area**: Optimal Transport / Statistical Learning Theory
**Keywords**: Stochastic OT maps, transport error, finite-sample estimation, robust statistics, Markov kernels

## TL;DR
This paper introduces a transport error metric $\mathcal{E}_p$ for stochastic OT maps, decomposed into an optimality gap and a feasibility gap. Under minimal assumptions that require neither the existence nor uniqueness of a Brenier map, a computationally efficient rounding estimator is constructed that achieves a near-optimal convergence rate of $\tilde{O}(n^{-1/(d+2p)})$. The framework is further extended to Hölder-continuous kernels and adversarially corrupted data, establishing the first general theory for OT map estimation.

## Background & Motivation

Optimal transport (OT) provides a geometry-based principled framework for comparing and transforming probability distributions. The transport map — its central object — finds broad applications in domain adaptation, single-cell genomics, style transfer, and generative modeling. Existing theories for OT map estimation almost entirely rely on Brenier's theorem ($p=2$, absolutely continuous source distribution) to guarantee a uniquely determined map, supplemented by hard-to-verify regularity assumptions such as density bounds, Lipschitz continuity, and Hölder smoothness.

However, many practical settings do not satisfy these prerequisites. In domain adaptation, the source distribution may lie on a low-dimensional manifold (e.g., text-to-image translation), where deterministic maps do not exist. In single-cell developmental trajectory reconstruction, developmental paths bifurcate over time, making measure-preserving maps from early to late snapshots inherently stochastic. Furthermore, the $L^p(\mu)$ error used in prior work requires map uniqueness and is otherwise ill-defined.

The paper's core starting point is to abandon the requirement of map uniqueness or determinism, and instead define a new quality metric $\mathcal{E}_p$ applicable to both deterministic and stochastic transport maps (i.e., Markov kernels), thereby establishing a general map estimation theory under minimal assumptions.

## Method

### Overall Architecture
The paper builds a four-layer progressive analysis around the newly defined transport error $\mathcal{E}_p$: (1) establishing fundamental properties and stability lemmas; (2) deriving finite-sample estimators and convergence rates without regularity assumptions; (3) obtaining improved rates via a WDRO estimator under Hölder-continuous kernel assumptions; (4) providing robust estimation guarantees under a mixed TV+$W_p$ adversarial corruption model.

### Key Designs

1. **Transport Error $\mathcal{E}_p$ (Core Contribution)**:
    - Function: Evaluates the transport quality of a Markov kernel $\kappa$ for the $W_p(\mu,\nu)$ problem.
    - Mechanism: $\mathcal{E}_p(\kappa;\mu,\nu) = [\text{transport cost} - W_p(\mu,\nu)]_+ + W_p(\kappa_\sharp\mu, \nu)$, where the first term is the "optimality gap" (how much the cost exceeds the optimum) and the second is the "feasibility gap" (how far the pushforward measure deviates from the target).
    - Design Motivation: $\mathcal{E}_p = 0$ if and only if $\kappa$ is an optimal kernel (without requiring uniqueness), while $\mathcal{E}_p \leq 2\|T - T^\star\|_{L^p(\mu)}$ maintains compatibility with existing $L^p$ benchmarks (Proposition 1). Figure 2 demonstrates that $L^p$ can deviate arbitrarily from $\mathcal{E}_p$: when the deterministic map is highly oscillatory, pointwise deviations are large while transport quality differs only minimally.

2. **Stability Lemma System (Technical Foundation)**:
    - Function: Characterizes the response of $\mathcal{E}_p$ to perturbations in the source and target distributions.
    - Mechanism: Lemma 3 gives $W_p$ stability with respect to $\nu$ (bound $2W_p(\nu,\nu')$); Lemma 4 gives $W_p$ stability with respect to $\mu$ when the kernel is Hölder continuous; Lemma 5 gives TV stability with respect to $\mu$; Lemma 6 addresses compositional stability of kernels.
    - Design Motivation: These lemmas form the skeleton of all subsequent rate analyses, decomposing "population-level error" into "empirical-level error plus approximation error from empirical to population distribution."

3. **Rounding Estimator (Primary Estimator)**:
    - Function: Achieves near-optimal convergence rates under minimal assumptions requiring only sub-Gaussianity (or bounded $2p$-th moments).
    - Mechanism: A three-step procedure — (a) project the empirical measure $\hat\mu_n$ onto a regular grid via a rounding function to obtain $\mu_n'$; (b) solve for an approximately optimal kernel $\bar\kappa_n$ on the grid; (c) return the composed kernel $\hat\kappa_n = \bar\kappa_n \circ r_{\mathcal{P}}$.
    - Design Motivation: Rounding induces a TV perturbation (rather than a $W_p$ perturbation), allowing direct application of the TV stability Lemma 5 to obtain sharper rates. The key derivation chain: $\mathcal{E}_p(\hat\kappa_n;\mu,\nu) \leq \mathcal{E}_p(\bar\kappa_n;\mu',\nu) + \sqrt{d}r \lesssim W_p(\nu,\hat\nu_n) + (nr^d)^{-1/(2p)} + \sqrt{d}r$.
    - Result: $\mathbb{E}[\mathcal{E}_p(\hat\kappa_n;\mu,\nu)] = \tilde{O}_{p,d}(n^{-1/(d+2p)})$, with computational cost $O(n^{2+o_d(1)})$ (a single low-accuracy entropic OT call).

### Loss & Training

This is a statistical estimation framework rather than a machine learning training procedure. Estimator performance is measured by the expected transport error $\mathbb{E}[\mathcal{E}_p]$. The hyperparameters of the rounding estimator (grid cell width $r$, truncation radius $R$, solver accuracy $\delta$) can all be tuned independently of $\mu$ and $\nu$.

## Key Experimental Results

### Main Results
The paper validates the theory on two synthetic settings (Setting A: 1D→2D stochastic splitting; Setting B: orthogonal quadrant separation).

| Setting | Dim $d$ | Estimator | Metric | Trend |
|---------|---------|-----------|--------|-------|
| A | 3,5,10 | NN | L1 | Consistently >1, fails to converge |
| A | 3,5,10 | NN | $\mathcal{E}_1$ | Decreases steadily with $n$ |
| A | 3,5,10 | Rounding | $\mathcal{E}_1$ | Decreases steadily; gap with NN narrows at high $d$ |
| B | 3,5,10 | NN vs Rounding | $\mathcal{E}_1$ | NN slightly better, gap narrows with $d$ |

### Ablation Study

| Configuration | Key Metric | Remarks |
|--------------|-----------|---------|
| $\mathcal{E}_1$ vs $L^1$ (Setting A) | NN's $L^1>1$ vs $\mathcal{E}_1\to 0$ | $L^p$ completely fails for irregular OT maps |
| $d=1$ special case | Convergence rate $n^{-1/2}$ | KS distance stability enables improvement to parametric rate |

### Key Findings
- $\mathcal{E}_p$ effectively evaluates transport quality even when deterministic maps do not exist or are highly irregular, while the $L^p$ metric fails entirely.
- The nearest-neighbor (NN) estimator empirically outperforms the rounding estimator under $\mathcal{E}_1$, though the gap narrows in high dimensions.
- A gap remains between the upper bound $n^{-1/(d+2p)}$ and lower bound $n^{-1/(d\vee 2p)}$, which closes at $d=1$.

## Highlights & Insights
- **First General OT Map Estimation Theory**: Simultaneously covers deterministic and stochastic maps, relaxing assumptions from "existence of Brenier map + smoothness" to moment conditions.
- **$\mathcal{E}_p$ as an Evaluation Metric, Not a Training Objective**: Remark 5 notes that directly training neural maps with $\mathcal{E}_p$ is ineffective (gradient signal is weaker than with Monge gap); its value lies in providing provable evaluation guarantees.
- **Clean Decoupling of TV and $W_p$ Corruption**: The dual stability of $\mathcal{E}_p$ makes adversarial estimation analysis remarkably clean, with the two corruption types contributing independently to the error.
- **Fundamental Separation in Robust Estimation**: The minimax lower bound term $d^{1/4}\rho^{1/2}$ demonstrates that robust map estimation is fundamentally harder than robust distribution estimation — $W_p$ guarantees from distribution estimation cannot be losslessly transferred to map estimation guarantees.

## Limitations & Future Work
- A rate gap persists for $d \geq 3$ (upper bound $n^{-1/(d+2p)}$ vs. lower bound $n^{-1/(d\vee 2p)}$); multiscale analysis may help close it.
- The WDRO estimator achieves information-theoretically optimal rates but is computationally infeasible; computationally efficient Lipschitz kernel estimators remain to be developed.
- Convergence rate analysis of neural network map estimators under $\mathcal{E}_p$ is an important future direction.
- The framework can in principle extend to entropic OT, weak OT, and conditional OT variants, but the stability lemmas would require targeted adaptation.

## Related Work & Insights
- **vs. Brenier Map Estimation** (Hütter & Rigollet 2021, Balakrishnan & Manole 2025): The latter requires density conditions and smoothness; $\mathcal{E}_p$ requires neither and applies to general $p \geq 1$.
- **vs. Monge Gap** (Uscidda & Cuturi 2023): Lemma 2 establishes $\mathcal{E}_p \leq 4\mathcal{E}_p'$, with equivalence at $p=1$; $\mathcal{E}_p$ is better suited for statistical analysis, while Monge gap is more suitable for neural network training.
- **vs. Adversarially Robust Distribution Estimation**: The minimax lower bound term $d^{1/4}\rho^{1/2}$ versus $\rho$ for distribution estimation reveals a fundamental hardness gap between map and distribution estimation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Pioneering definition of a transport error metric applicable to stochastic OT maps, establishing a complete theoretical framework.
- Experimental Thoroughness: ⭐⭐⭐ Theory-focused; synthetic experiments adequately validate key properties but do not address real-world applications.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theorem–lemma–corollary hierarchy, intuitive figures, balancing technical rigor and intuition.
- Value: ⭐⭐⭐⭐⭐ Addresses a fundamental coverage gap in OT map estimation theory, with far-reaching implications for the statistical learning and applied OT communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport](bispectral_ot_dataset_comparison_using_symmetry-aware_optimal_transport.md)
- [\[ICML 2025\] Hierarchical Refinement: Optimal Transport to Infinity and Beyond](../../ICML2025/others/hierarchical_refinement_optimal_transport_to_infinity_and_beyond.md)
- [\[ICCV 2025\] LaCoOT: Layer Collapse through Optimal Transport](../../ICCV2025/others/lacoot_layer_collapse_through_optimal_transport.md)
- [\[ACL 2025\] Quantifying Lexical Semantic Shift via Unbalanced Optimal Transport](../../ACL2025/others/quantifying_lexical_semantic_shift_via_unbalanced_optimal_transport.md)
- [\[ICML 2025\] Lightspeed Geometric Dataset Distance via Sliced Optimal Transport](../../ICML2025/others/lightspeed_geometric_dataset_distance_via_sliced_optimal_transport.md)

</div>

<!-- RELATED:END -->
