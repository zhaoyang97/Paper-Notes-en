---
title: >-
  [Paper Note] Online Minimization of Polarization and Disagreement via Low-Rank Matrix Bandits
description: >-
  [ICLR 2026][Reinforcement Learning][opinion polarization] This paper is the first to formalize the problem of minimizing polarization and disagreement under the Friedkin-Johnsen opinion dynamics model as an online low-ra…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "opinion polarization"
  - "Friedkin-Johnsen model"
  - "low-rank matrix bandit"
  - "regret minimization"
  - "social network intervention"
date: 2026-05-08
content_hash: 015716dd6a527f2b
---

# Online Minimization of Polarization and Disagreement via Low-Rank Matrix Bandits

**Conference**: ICLR 2026
**arXiv**: [2510.00803](https://arxiv.org/abs/2510.00803)  
**Code**: [GitHub](https://github.com/FedericoCinus/online-min-pol)  
**Area**: online learning, social network
**Keywords**: opinion polarization, Friedkin-Johnsen model, low-rank matrix bandit, regret minimization, social network intervention

## TL;DR

This paper is the first to formalize the problem of minimizing polarization and disagreement under the Friedkin-Johnsen opinion dynamics model as an online low-rank matrix bandit problem (OPD-Min). A two-phase algorithm, OPD-Min-ESTR, is proposed that reduces the dimensionality from $|V|^2$ to $O(|V|)$ via subspace estimation, achieving substantial improvements over full-dimensional linear bandit baselines on both synthetic and real-world networks.

## Background & Motivation

**Background**: Social media can exacerbate opinion polarization and societal fragmentation. The Friedkin-Johnsen (FJ) model is a classical framework for studying opinion dynamics: each agent's expressed opinion is a weighted average of its internal opinion and its neighbors' expressed opinions, converging to a unique equilibrium $\bm{z}^* = (\mathbf{I} + \mathbf{L})^{-1}\bm{s}$, where $\mathbf{L}$ is the graph Laplacian and $\bm{s}$ is the vector of internal opinions.

**Limitations of Prior Work**: Musco et al. (2018) pioneered the study of minimizing polarization and disagreement at the FJ equilibrium, but assumed complete knowledge of all agents' internal opinions. In practice, obtaining internal opinions is costly—potentially requiring extensive user surveys or behavioral analysis—and may be infeasible under privacy constraints.

**Key Challenge**: Several works have relaxed this assumption—covering binary internal opinions (Chen et al., 2018), SDP upper-bound optimization (Chaitanya et al., 2024), and inference with limited queries (Cinus et al., 2025)—yet none addresses the truly online setting where internal opinions are entirely unknown and unqueryable, and learning must proceed solely from scalar feedback (polarization + disagreement values) observed after each intervention.

**Goal**: The paper formalizes the problem as Online Polarization and Disagreement Minimization (OPD-Min), establishing a key connection between social media algorithmic intervention and multi-armed bandit theory. By exploiting the rank-one structure of the objective $f(\mathbf{X}) = \langle \bm{s}\bm{s}^\top, \mathbf{X} \rangle$, it designs an efficient low-rank matrix bandit algorithm.

## Method

### Overall Architecture

At each time step $t = 1, \ldots, T$:
1. The learner selects a graph Laplacian $\mathbf{L}_t$ from a finite intervention set $\mathcal{L}$, equivalently selecting a forest matrix $\mathbf{X}_t = (\mathbf{I} + \mathbf{L}_t)^{-1}$.
2. The FJ dynamics converge to equilibrium, and the learner observes a noisy loss $Y_t = \langle \mathbf{\Theta}^*, \mathbf{X}_t \rangle + \eta_t$, where $\mathbf{\Theta}^* = \bm{s}\bm{s}^\top$ is the unknown rank-one parameter.
3. The goal is to minimize cumulative regret $R_T = \sum_{t=1}^T [f(\mathbf{X}_t) - f(\mathbf{X}^*)]$.

The key challenge is that naively treating this as a $|V|^2$-dimensional linear bandit yields a regret bound of $\tilde{O}(|V|^2\sqrt{T})$, while existing low-rank matrix bandit methods assume continuous sampling (e.g., from Gaussian distributions), which is incompatible with the discrete, structured action set of forest matrices.

### Key Design 1: Subspace Estimation (Stage 1)

**Function**: Recover the low-dimensional subspace of the unknown parameter matrix $\mathbf{\Theta}^*$ from $T_1$ rounds of exploration.

**Mechanism**: A nuclear norm-regularized least-squares estimator is used:

$$\widehat{\mathbf{\Theta}} = \arg\min_{\mathbf{\Theta}} \frac{1}{2T_1} \sum_{t=1}^{T_1} (Y_t - \langle \mathbf{X}_t, \mathbf{\Theta} \rangle)^2 + \lambda_{T_1} \|\mathbf{\Theta}\|_{\text{nuc}}$$

with regularization parameter $\lambda_{T_1} = 2\sqrt{2\log(2|V|/\delta)/T_1}$. Under the Restricted Strong Convexity (RSC) condition, the estimation error satisfies:

$$\|\widehat{\mathbf{\Theta}} - \mathbf{\Theta}^*\|_F^2 \leq \frac{36\log(2|V|/\delta)}{\kappa^2 T_1}$$

**Design Motivation**: Unlike prior work that assumes a "nice exploration distribution," this paper directly proves that the RSC condition holds for the forest matrix action set. The curvature parameter $\kappa = \kappa_{\min}(\mathcal{X})$ measures the diversity of the action set; statistical deviation is controlled via Talagrand's concentration inequality and Rademacher processes.

### Key Design 2: Dimensionality-Reduced Linear Bandit (Stage 2)

**Function**: Exploit the estimated subspace to reduce the problem to $O(|V|)$ dimensions.

**Mechanism**: The top eigenvector $\hat{\bm{s}}$ of $\widehat{\mathbf{\Theta}}$ is extracted, and an orthonormal basis $[\hat{\bm{s}}, \hat{\mathbf{S}}_\perp]$ is constructed. Each arm $\mathbf{X}$ is rotated as $\mathbf{X}' = [\hat{\bm{s}}, \hat{\mathbf{S}}_\perp]^\top \mathbf{X} [\hat{\bm{s}}, \hat{\mathbf{S}}_\perp]$, retaining only the first row and column to form a $k = 2|V|-1$ dimensional feature vector $\bm{x}_{\text{sub}}$. A standard linear bandit algorithm (e.g., OFUL) is then run in this low-dimensional space.

**Design Motivation**: Since $\mathbf{\Theta}^*$ is rank-one, the signal is concentrated in the direction of $\bm{s}$; projection error onto the orthogonal complement can be controlled via the Davis-Kahan $\sin\theta$ theorem and decays as $T_1$ grows.

### Regret Bound

**Theorem 4.1**: Under the RSC condition, with the optimal choice $T_1 \asymp \frac{1}{\|\bm{s}\|^2 \kappa} \sqrt{T \log(2|V|/\delta)}$, the total regret satisfies:

$$R_T = \widetilde{\mathcal{O}}\left(\max\left\{\frac{1}{\kappa}, \sqrt{|V|}\right\}\sqrt{|V| \cdot T}\right)$$

The $\sqrt{T}$ dependence is optimal in time, and the replacement of $|V|^2$ by $|V|$ reflects the benefit of dimensionality reduction.

## Key Experimental Results

### Main Results: Cumulative Regret on Synthetic Networks

| Method | $|V|=8$ ER (regret/runtime) | $|V|=16$ ER (regret/runtime) | $|V|=8$ SBM | $|V|=16$ SBM |
|---|---|---|---|---|
| OPD-Min (Ours) | Low regret / **Fast** | Low regret / **Fast** | Low regret / **Fast** | Low regret / **Fast** |
| Full-dim OFUL | High regret / Slow | **Significantly** higher regret / **Extremely slow** | High regret / Slow | High regret / Slow |
| Oracle Subspace | Lowest regret / Fast | Lowest regret / Fast | Lowest regret / Fast | Lowest regret / Fast |

The gap is especially pronounced at $|V|=16$: OFUL incurs substantially higher regret and runtime, while OPD-Min closely tracks the oracle baseline.

### Ablation Study: RSC Parameter Validation

| Graph Type | Arm Regime | $|V|=32$ | $|V|=128$ | $|V|=1024$ |
|---|---|---|---|---|
| ER | Diverse | 0.393 | 0.410 | 0.499 |
| ER | Local | $1.68 \times 10^{-5}$ | $1.49 \times 10^{-7}$ | $2.21 \times 10^{-7}$ |
| SBM | Diverse | 0.386 | 0.476 | 0.462 |
| SBM | Local | $2.97 \times 10^{-6}$ | $8.05 \times 10^{-7}$ | $1.05 \times 10^{-7}$ |

Under the diverse arm regime, $\kappa$ is reasonable (≈0.4–0.5), while the local regime yields extremely small $\kappa$ due to near-collinearity among arms.

### Key Findings

- OPD-Min consistently outperforms full-dimensional OFUL across all network topologies and parameter settings, with runtime advantages growing with $|V|$.
- The algorithm scales to networks with $|V|=1024$ nodes.
- Results are consistent on real-world networks (Florentine families, Karate club, Les Misérables).
- The online algorithm quickly surpasses the offline SDP baseline (Chaitanya et al., 2024), identifying superior strategies after approximately 250 interventions.
- Bimodal opinion distributions make interventions easier to identify and exploit.

## Highlights & Insights

- This work is the first to connect opinion polarization minimization with online learning and bandit theory, opening a new research direction.
- A self-contained RSC analysis is provided for the structured action set of forest matrices, without relying on the continuous sampling assumptions of existing low-rank bandit methods.
- Dimensionality reduction from $|V|^2$ to $O(|V|)$ simultaneously improves statistical efficiency and computational efficiency.
- Only scalar feedback (global polarization + disagreement) is required; individual opinions need not be observed, making the approach privacy-friendly.

## Limitations & Future Work

- The RSC parameter $\kappa$ becomes extremely small in the local arm regime, potentially making theoretical bounds overly pessimistic.
- Only scalar equilibrium feedback is considered; richer signals (e.g., community-level polarization) could improve performance.
- The method assumes FJ dynamics converge to equilibrium between interventions, ignoring the effects of non-equilibrium states.
- The assumption that internal opinions are time-invariant may be too strong in practice.
- The action set is restricted to undirected graph Laplacians, excluding directed graphs or other forms of intervention.

## Related Work & Insights

- **Musco et al. (2018)**: Pioneered offline polarization minimization; this paper extends their framework to the online setting.
- **LowESTR** (Lu et al., 2021) and **G-ESTT** (Kang et al., 2022): This paper builds on the ESTR framework and tailors the analysis to the specific structure of OPD-Min.
- **LowPopArt** (Jang et al., 2024): A low-rank bandit method based on optimal design, but its computational complexity of $O(|V|^6)$ is infeasible for social networks.
- This work inspires analogous frameworks for polarization mitigation in content recommendation system design.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to connect opinion dynamics with online bandit theory; problem formulation is highly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Rich synthetic and real-world experiments with RSC validation and scalability analysis, though real platform data is absent.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous and complete; main text and appendix are well organized.
- Value: ⭐⭐⭐⭐ Pioneering work, though practical deployment on social platforms remains a distant goal.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] GeoRA: Geometry-Aware Low-Rank Adaptation for RLVR](../../ACL2026/reinforcement_learning/geora_geometry-aware_low-rank_adaptation_for_rlvr.md)
- [\[ICLR 2026\] Revisiting Matrix Sketching in Linear Bandits: Achieving Sublinear Regret via Dyadic Block Sketching](revisiting_matrix_sketching_in_linear_bandits_achieving_sublinear_regret_via_dya.md)
- [\[AAAI 2026\] Beyond the Lower Bound: Bridging Regret Minimization and Best Arm Identification in Lexicographic Bandits](../../AAAI2026/reinforcement_learning/beyond_the_lower_bound_bridging_regret_minimization_and_best_arm_identification_.md)
- [\[ICLR 2026\] AWM: Accurate Weight-Matrix Fingerprint for Large Language Models](awm_accurate_weight-matrix_fingerprint_for_large_language_models.md)
- [\[ICLR 2026\] Single Index Bandits: Generalized Linear Contextual Bandits with Unknown Reward Functions](single_index_bandits_generalized_linear_contextual_bandits_with_unknown_reward_f.md)

</div>

<!-- RELATED:END -->
