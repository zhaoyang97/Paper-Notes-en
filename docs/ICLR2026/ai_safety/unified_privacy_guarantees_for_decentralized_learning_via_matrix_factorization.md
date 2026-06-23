---
title: >-
  [Paper Note] Unified Privacy Guarantees for Decentralized Learning via Matrix Factorization
description: >-
  [ICLR 2026][AI Safety][Paper Note] This paper unifies multiple algorithms and trust models in decentralized learning (DL) into a Matrix Factorization (MF) framework. It generalizes privacy guarantees to broader matrix types and proposes the MAFALDA-SGD algorithm, which significantly outperforms existing methods on synthetic and real graph topologies by
tags:
  - ICLR 2026
  - AI Safety
date: 2026-05-08
content_hash: 66f018386936d9ac
---
# Unified Privacy Guarantees for Decentralized Learning via Matrix Factorization

**Conference**: ICLR 2026  
**arXiv**: [2510.17480](https://arxiv.org/abs/2510.17480)  
**Code**: None  
**Area**: AI Safety / Differential Privacy / Decentralized Learning  
**Keywords**: Decentralized learning, matrix factorization, differential privacy, correlated noise, gossip protocol

## TL;DR
This paper unifies multiple algorithms and trust models in decentralized learning (DL) into a Matrix Factorization (MF) framework. It generalizes privacy guarantees to broader matrix types and proposes the MAFALDA-SGD algorithm, which significantly outperforms existing methods on synthetic and real graph topologies by optimizing noise correlation.

## Background & Motivation

**Background**: Decentralized learning (DL) allows users to collaboratively train models via peer-to-peer communication graphs without sharing raw data. Strong privacy is typically achieved through Differential Privacy (DP). Centralized DP-SGD has a mature noise correlation analysis framework based on Matrix Factorization (MF), which leverages temporal noise correlation to improve the privacy-utility tradeoff. However, MF methods were previously limited to centralized scenarios.

**Limitations of Prior Work**: (1) DP analysis in DL relies on ad hoc proofs for specific algorithms and trust models, lacking a unified framework; (2) existing analyses fail to fully exploit noise correlations arising from redundant exchanges in peer-to-peer communication, leading to pessimistic privacy guarantees; (3) existing MF theory requires the workload matrix to be square, full-rank, and lower-triangular, conditions not satisfied in DL.

**Key Challenge**: The distributed communication structure of DL generates matrices that do not satisfy the assumptions of existing MF theory. Furthermore, varying levels of adversary knowledge across different trust models (LDP, PNDP, SecLDP) require a unified modeling approach.

**Goal**: (1) Encode DL algorithms into a single matrix multiplication form; (2) unify adversary knowledge representations across different trust models; (3) generalize MF privacy guarantees to rectangular and potentially rank-deficient matrices; (4) design new algorithms by optimizing noise correlations.

**Key Insight**: The complete $T$ iterations of decentralized SGD can be expanded into a large matrix equation $\theta = (I_T \otimes W)(M\theta_0 - \eta \mathbf{W}_T(G + C^\dagger Z))$, thereby incorporating DL into the MF analysis framework.

**Core Idea**: The gradient-noise interaction in decentralized learning is uniformly represented as $\mathcal{O}_\mathcal{A} = AG + BZ$ (where $A=BC$). The paper generalizes MF's GDP privacy guarantees and proposes the MAFALDA-SGD algorithm by optimizing the noise correlation matrix $C$.

## Method

### Overall Architecture
The paper first expresses the entire process of any decentralized learning algorithm as a matrix equation $\mathcal{O}_\mathcal{A} = AG + BZ$. It then uses the matrix factorization $A=BC$ to transform the privacy analysis into an investigation of the correlation matrix $C$. Following this unified modeling, privacy guarantees shift from algorithm-specific proofs to a standardized theorem. Furthermore, this unified representation serves as an optimization objective—optimizing $C$ yields the new MAFALDA-SGD algorithm.

### Key Designs

**1. Expanding Decentralized Learning into a Single Matrix: Enabling MF Theory**

In decentralized settings, $n$ nodes execute $T$ rounds on a communication graph $\mathcal{G}$. Each round involves a local noisy gradient step $\theta_{t+1/2} = \theta_t - \eta(G_t + C_t^\dagger Z)$, followed by gossip averaging $\theta_{t+1} = W\theta_{t+1/2}$. The coupling between iterations prevents direct application of centralized MF tools. This work stacks all $T$ rounds chronologically to derive a closed-form $\theta = (I_T \otimes W)(M\theta_0 - \eta \mathbf{W}_T(G + C^\dagger Z))$, where $\mathbf{W}_T \in \mathbb{R}^{nT \times nT}$ is a lower-triangular Toeplitz block matrix encoding temporal gossip propagation. By formulating this as "workload matrix times gradients plus noise," all adversary observations are abstracted as $\mathcal{O}_\mathcal{A} = AG + BZ$ (Theorem 6). Different trust models—LDP, PNDP, SecLDP—simply correspond to different $A, B$, consolidating fragmented ad hoc proofs into one framework.

**2. Generalized MF Privacy Guarantees: Accounting for Rectangular and Rank-Deficient Matrices**

Current MF theory (Denisov et al.) requires workload matrices to be square, full-rank, and lower-triangular. However, in DL, adversaries often observe only a subset of messages, making $A$ rectangular and rank-deficient. This paper generalizes the guarantee to cases where $A$ is in column echelon form (a generalization of lower-triangularity). It defines generalized sensitivity $\text{sens}_\Pi(C; B) = \max_{G \simeq_\Pi G'}\|C(G-G')\|_{B^\dagger B}$ and proves that if $A = BC$ and $A$ is in column echelon form, the mechanism $\mathcal{M}$ is $1/\sigma$-GDP (Theorem 8). The key correction is the $B^\dagger B$ term in the norm—it projects sensitivity onto the row space of $B$, automatically discarding gradient combinations invisible to the adversary. This avoids paying a privacy cost for unobservable information, providing tighter bounds than previous per-algorithm analyses. This relaxation also ensures privacy guarantees hold under adaptive gradients.

**3. MAFALDA-SGD: Optimizing Correlation for Decentralized Settings**

The unified framework allows $C$ to be optimized directly. Under LDP, the authors constrain $C = C_{local} \otimes I_n$, meaning noise is correlated across time within a node but independent across nodes. This reduces the high-dimensional optimization to solving for a small matrix $C_{local}$. The objective $\mathcal{L}_{opti}(\mathbf{W}_T, B, C) = \text{sens}_\Pi(C;B)^2 \|(I_T \otimes W)\mathbf{W}_T C^\dagger\|_F^2$ balances privacy sensitivity (numerator) against the disturbance of injected noise on the model after gossip propagation (denominator). Since the objective is convex, the optimal $C_{local}$ can be determined. Specialized optimization is necessary because centralized correlation patterns (like AntiPGD) ignore the spatio-temporal structure of gossip, often performing worse than independent noise in DL—a gap MAFALDA-SGD addresses.

### Loss & Training
Training follows standard DP-SGD: per-sample gradients are clipped to norm $\Delta_g$, and Gaussian noise $Z \sim \mathcal{N}(0, \Delta_g^2 \sigma^2)^{nT \times d}$ is added. The gossip matrix $W$ satisfies stochastic matrix conditions and supports time-varying $W_t$. Privacy accounting is performed within the Gaussian DP (GDP) framework.

## Key Experimental Results

### Main Results (Privacy Guarantee Comparison)

| Algorithm | Trust Model | Privacy Guarantee | vs. Prev. Analysis |
| :--- | :--- | :--- | :--- |
| DP-D-SGD | LDP | This MF Analysis | Tighter |
| DP-D-SGD | PNDP | This MF Analysis | Tighter |
| Zip-DL | SecLDP | This MF Analysis | Tighter |
| MAFALDA-SGD | LDP | Optimized Correlated Noise | Significantly outperforms all |

### Ablation Study (Noise Correlation Strategy Comparison)

| Strategy | Privacy-Utility Tradeoff | Description |
| :--- | :--- | :--- |
| Independent (DP-D-SGD) | Baseline | Independent noise per step |
| AntiPGD (Centralized) | Worse than baseline | Centralized patterns do not suit DL |
| MAFALDA-SGD | Optimal | Correlation optimized for decentralized structures |

### Key Findings
- The unified framework provides tighter privacy guarantees for all existing DL algorithms across all trust models.
- Noise correlation strategies effective in centralized settings (e.g., AntiPGD) perform poorly in DL, indicating a need for decentralized optimization.
- MAFALDA-SGD significantly outperforms existing methods on both synthetic graphs (Ring, Grid) and real social network graphs.
- The column echelon form condition (generalized lower-triangularity) ensures privacy guarantees remain valid for adaptive gradients.

## Highlights & Insights
- Significant theoretical contribution: Seamlessly generalizes MF theory from centralized to decentralized settings, unifying fragmented DP-DL analysis methods. The generalization in Theorem 8 (rectangular/rank-deficient matrices + generalized sensitivity) is a core contribution.
- Practical value: The framework not only unifies the analysis of existing algorithms but also directly guides the design of new ones. MAFALDA-SGD demonstrates the potential of optimized noise correlation for DL privacy.

## Limitations & Future Work
- MAFALDA-SGD currently only supports LDP; extending noise optimization to PNDP and SecLDP is a natural direction.
- The computational cost of optimizing $C_{local}$ scales with $T$, which may be impractical for large-scale, long-duration training.
- Experiments primarily validate the theory in convex optimization scenarios; the effectiveness in non-convex deep learning training remains to be tested.

## Related Work & Insights
- **vs. Cyffers et al. (2022/2024)**: Proponents of PNDP, but their analyses are algorithm-specific, whereas this work provides a unified framework.
- **vs. Denisov et al. (2022)**: Founders of the centralized MF mechanism; this work generalizes their square/full-rank/triangular requirements to more general cases.
- **vs. Biswas et al. (2025)**: Zip-DL uses ad hoc correlated noise without optimization; this paper's framework allows for systematic optimization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The unified framework and generalized MF privacy guarantees are major theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐ Experiments are focused on theoretical validation and lack large-scale deep learning scenarios.
- Writing Quality: ⭐⭐⭐⭐ The notation is complex but definitions are clear and logically structured.
- Value: ⭐⭐⭐⭐ Provides the theoretical infrastructure for the DP-DL field with potential for long-term impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Back to Square Roots: An Optimal Bound on the Matrix Factorization Error for Multi-Epoch Differentially Private SGD](back_to_square_roots_an_optimal_bound_on_the_matrix_factorization_error_for_mult.md)
- [\[AAAI 2026\] Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer Collaborative Learning](../../AAAI2026/ai_safety/learning_to_collaborate_an_orchestrated-decentralized_framework_for_peer-to-peer.md)
- [\[NeurIPS 2025\] Mitigating Privacy-Utility Trade-off in Decentralized Federated Learning via f-Differential Privacy](../../NeurIPS2025/ai_safety/mitigating_privacy-utility_trade-off_in_decentralized_federated_learning_via_f-d.md)
- [\[ICLR 2026\] Convergent Differential Privacy Analysis for General Federated Learning](convergent_differential_privacy_analysis_for_general_federated_learning.md)
- [\[ICLR 2026\] Optimizing Canaries for Privacy Auditing with Metagradient Descent](optimizing_canaries_for_privacy_auditing_with_metagradient_descent.md)

</div>

<!-- RELATED:END -->
