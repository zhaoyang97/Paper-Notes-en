---
title: >-
  [Paper Note] Unified Privacy Guarantees for Decentralized Learning via Matrix Factorization
description: >-
  [ICLR 2026][AI Safety][decentralized learning] This paper unifies diverse algorithms and trust models in decentralized learning (DL) under a matrix factorization (MF) mechanism framework, extends privacy guarantees to more general matrix types, and proposes the MAFALDA-SGD algorithm that significantly outperforms existing methods on both synthetic and real-world graph topologies by optimizing noise correlation.
tags:
  - ICLR 2026
  - AI Safety
  - decentralized learning
  - matrix factorization
  - differential privacy
  - correlated noise
  - gossip protocol
date: 2026-05-08
content_hash: 9010ca2d4fb87477
---

# Unified Privacy Guarantees for Decentralized Learning via Matrix Factorization

**Conference**: ICLR 2026
**arXiv**: [2510.17480](https://arxiv.org/abs/2510.17480)
**Code**: None
**Area**: AI Security / Differential Privacy / Decentralized Learning
**Keywords**: decentralized learning, matrix factorization, differential privacy, correlated noise, gossip protocol

## TL;DR
This paper unifies diverse algorithms and trust models in decentralized learning (DL) under a matrix factorization (MF) mechanism framework, extends privacy guarantees to more general matrix types, and proposes the MAFALDA-SGD algorithm that significantly outperforms existing methods on both synthetic and real-world graph topologies by optimizing noise correlation.

## Background & Motivation

**State of the Field**: Decentralized learning (DL) enables users to collaboratively train models over peer-to-peer communication graphs without sharing raw data. Strong privacy guarantees are typically achieved via differential privacy (DP). Centralized DP-SGD has a mature MF-based noise correlation analysis framework that leverages temporal noise correlations to improve privacy-utility trade-offs. However, MF methods had previously only been applied in centralized settings.

**Limitations of Prior Work**: (1) DP analysis in DL relies on ad hoc proofs tailored to specific algorithms and trust models, lacking a unified framework; (2) existing analyses fail to fully exploit the noise correlations arising from redundant exchanges in peer-to-peer communication, leading to overly pessimistic privacy guarantees; (3) existing MF theory requires the workload matrix to be square, full-rank, and lower-triangular — conditions not satisfied in DL settings.

**Root Cause**: The matrices arising from the distributed communication structure of DL violate the assumptions of existing MF theory; moreover, adversaries under different trust models (LDP, PNDP, SecLDP) possess different levels of knowledge, necessitating a unified treatment.

**Paper Goals**: (1) Encode DL algorithms as a single matrix multiplication; (2) unify adversary knowledge representations across different trust models; (3) extend MF-based DP guarantees to rectangular and possibly rank-deficient matrices; (4) design new algorithms by optimizing noise correlations.

**Starting Point**: All $T$ iterations of decentralized SGD can be unrolled into a single matrix equation $\theta = (I_T \otimes W)(M\theta_0 - \eta \mathbf{W}_T(G + C^\dagger Z))$, thereby bringing DL into the MF analysis framework.

**Core Idea**: Represent the gradient-noise interaction in decentralized learning uniformly as $\mathcal{O}_\mathcal{A} = AG + BZ$ (where $A = BC$), generalize the GDP privacy guarantee of MF, and optimize the noise correlation matrix $C$ to derive the new algorithm MAFALDA-SGD.

## Method

### Overall Architecture
The approach proceeds in two steps: (1) **Unified modeling** — define linear DL algorithms (Definition 4) and adversary knowledge (Definition 5), and prove that all existing DL algorithms and trust models can be expressed as $\mathcal{O}_\mathcal{A} = AG + BZ$ (Theorem 6); (2) **Algorithm design** — optimize the local noise correlation matrix $C_{local}$ under LDP constraints to obtain MAFALDA-SGD.

### Key Designs

1. **Matrix Factorization Encoding of Decentralized Learning**:

    - *Function*: Unroll multi-round DL algorithm iterations into a unified matrix form.
    - *Mechanism*: $n$ nodes execute $T$ rounds on communication graph $\mathcal{G}$: each round performs a local gradient step $\theta_{t+1/2} = \theta_t - \eta(G_t + C_t^\dagger Z)$ followed by gossip averaging $\theta_{t+1} = W\theta_{t+1/2}$. Stacking $T$ rounds yields $\theta = (I_T \otimes W)(M\theta_0 - \eta \mathbf{W}_T(G + C^\dagger Z))$, where $\mathbf{W}_T \in \mathbb{R}^{nT \times nT}$ is a lower-triangular Toeplitz block matrix.
    - *Design Motivation*: Encoding the temporal unrolling of DL as a single matrix enables direct application of MF theory.

2. **Generalized MF Privacy Guarantee (Theorem 8)**:

    - *Function*: Extend MF-based DP guarantees from square/full-rank/lower-triangular matrices to rectangular/rank-deficient/column-echelon matrices.
    - *Mechanism*: Define the generalized sensitivity $\text{sens}_\Pi(C; B) = \max_{G \simeq_\Pi G'}\|C(G-G')\|_{B^\dagger B}$, and prove that when $A$ is in column-echelon form and $A = BC$, the mechanism $\mathcal{M}$ is $1/\sigma$-GDP. The key correction is that $B^\dagger B$ projects onto the row space of $B$, discarding gradient combinations that are unobservable to the adversary.
    - *Design Motivation*: In DL, adversaries observe only a subset of messages, so the resulting matrix $A$ is typically rectangular and rank-deficient.

3. **MAFALDA-SGD Algorithm (Optimized Noise Correlation)**:

    - *Function*: Maximize the privacy-utility trade-off under LDP by optimizing the local noise correlation matrix.
    - *Mechanism*: Constrain $C = C_{local} \otimes I_n$ (noise correlated only across time steps within each node), define the optimization objective $\mathcal{L}_{opti}(\mathbf{W}_T, B, C) = \text{sens}_\Pi(C;B)^2 \|(I_T \otimes W)\mathbf{W}_T C^\dagger\|_F^2$, and solve for the optimal $C_{local}$ via convex optimization.
    - *Design Motivation*: The correlation patterns of existing methods (e.g., AntiPGD) are not optimized for decentralized settings and perform poorly when directly applied to DL.

### Loss & Training
Standard DP-SGD training: per-sample gradients are clipped to norm $\Delta_g$, and Gaussian noise $Z \sim \mathcal{N}(0, \Delta_g^2 \sigma^2)^{nT \times d}$ is added. The gossip matrix $W$ satisfies stochastic matrix conditions. Time-varying gossip matrices $W_t$ are supported. Privacy accounting is performed using the Gaussian DP (GDP) framework.

## Key Experimental Results

### Main Results (Privacy Guarantee Comparison)

| Algorithm | Trust Model | Privacy Guarantee | vs. Prior Analysis |
|-----------|-------------|-------------------|--------------------|
| DP-D-SGD | LDP | MF analysis (Ours) | Tighter |
| DP-D-SGD | PNDP | MF analysis (Ours) | Tighter |
| Zip-DL | SecLDP | MF analysis (Ours) | Tighter |
| MAFALDA-SGD | LDP | Optimized correlated noise | Significantly outperforms all existing methods |

### Ablation Study (Noise Correlation Strategy Comparison)

| Strategy | Privacy-Utility Trade-off | Notes |
|----------|--------------------------|-------|
| No correlation (DP-D-SGD) | Baseline | Independent noise at each step |
| AntiPGD (centralized strategy) | Worse than baseline | Centralized correlation pattern unsuitable for DL |
| MAFALDA-SGD | Optimal | Correlation pattern optimized for decentralized settings |

### Key Findings
- The unified framework yields tighter privacy guarantees for all existing DL algorithms under all trust models.
- Noise correlation strategies effective in centralized settings (e.g., AntiPGD) perform worse when directly applied to DL, demonstrating the need for decentralization-specific optimization.
- MAFALDA-SGD significantly outperforms existing methods on both synthetic graphs (ring, grid) and real-world social network graphs.
- The column-echelon condition (a generalization of lower-triangularity) ensures that privacy guarantees remain valid under adaptive gradient queries.

## Highlights & Insights
- **Strong theoretical contribution**: The paper seamlessly extends MF theory from centralized to decentralized settings, unifying the fragmented landscape of DP-DL analysis. The generalization in Theorem 8 (rectangular/rank-deficient matrices + generalized sensitivity) constitutes the core contribution.
- **Practical value**: The framework not only unifies the analysis of existing algorithms but also directly guides the design of new ones. MAFALDA-SGD demonstrates the substantial potential of optimizing noise correlations for privacy in DL.

## Limitations & Future Work
- MAFALDA-SGD currently supports only LDP; extending noise optimization to PNDP and SecLDP is a natural direction.
- The computational cost of optimizing $C_{local}$ grows with $T$, potentially rendering it infeasible for large-scale, long-horizon training.
- Experiments are primarily conducted in convex optimization settings; effectiveness in non-convex deep learning training remains to be examined.

## Related Work & Insights
- **vs. Cyffers et al. (2022/2024)**: Originators of PNDP, but their analysis targets specific algorithms; this paper provides a unified framework.
- **vs. Denisov et al. (2022)**: Founders of the centralized MF mechanism, requiring square/full-rank/lower-triangular matrices; this paper generalizes to more general cases.
- **vs. Biswas et al. (2025)**: Zip-DL employs ad hoc correlated noise without optimization; the proposed framework enables systematic optimization.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Both the unified modeling framework and the generalized MF privacy guarantees represent significant theoretical contributions.
- **Experimental Thoroughness**: ⭐⭐⭐ — Experiments are primarily theory-driven; large-scale deep learning scenarios are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — The notation system is complex but definitions are clear, with a well-structured progression.
- **Value**: ⭐⭐⭐⭐ — Provides theoretical infrastructure for the DP-DL field with potential long-term impact.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Back to Square Roots: An Optimal Bound on the Matrix Factorization Error for Multi-Epoch Differentially Private SGD](back_to_square_roots_an_optimal_bound_on_the_matrix_factorization_error_for_mult.md)
- [\[NeurIPS 2025\] Mitigating Privacy-Utility Trade-off in Decentralized Federated Learning via f-Differential Privacy](../../NeurIPS2025/ai_safety/mitigating_privacy-utility_trade-off_in_decentralized_federated_learning_via_f-d.md)
- [\[AAAI 2026\] Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer Collaborative Learning](../../AAAI2026/ai_safety/learning_to_collaborate_an_orchestrated-decentralized_framework_for_peer-to-peer.md)
- [\[ICLR 2026\] Adaptive Methods Are Preferable in High Privacy Settings: An SDE Perspective](adaptive_methods_are_preferable_in_high_privacy_settings_an_sde_perspective.md)
- [\[ICLR 2026\] Membership Privacy Risks of Sharpness Aware Minimization](sam_membership_privacy_risks.md)

<!-- RELATED:END -->
