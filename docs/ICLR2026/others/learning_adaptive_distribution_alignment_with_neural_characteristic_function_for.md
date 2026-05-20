---
title: >-
  [Paper Note] Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation
description: >-
  [ICLR 2026][graph domain adaptation] ADAlign is proposed as a framework that leverages neural characteristic functions to adaptively align source/target graph distributions in the spectral domain — eliminating the need f…
tags:
  - "ICLR 2026"
  - "graph domain adaptation"
  - "characteristic function"
  - "spectral-domain alignment"
  - "adaptive frequency sampling"
  - "minimax optimization"
date: 2026-05-08
content_hash: 22850046628003ea
---

# Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation

**Conference**: ICLR 2026
**arXiv**: [2602.10489](https://arxiv.org/abs/2602.10489)  
**Code**: [https://github.com/gxingyu/ADAlign](https://github.com/gxingyu/ADAlign)  
**Area**: Other / Graph Neural Networks
**Keywords**: graph domain adaptation, characteristic function, spectral-domain alignment, adaptive frequency sampling, minimax optimization

## TL;DR
ADAlign is proposed as a framework that leverages neural characteristic functions to adaptively align source/target graph distributions in the spectral domain — eliminating the need for manual selection of alignment criteria by automatically identifying the most prominent distributional discrepancies in each transfer scenario. It achieves state-of-the-art performance across 16 transfer tasks on 10 datasets while reducing memory consumption and training time.

## Background & Motivation
Graph Domain Adaptation (GDA) aims to transfer knowledge from labeled source graphs to unlabeled target graphs. The sources of distribution shift are complex and intertwined — differences in node attributes, degree distributions, and homophily often co-occur. Existing methods rely on manually designed graph filters to extract specific features (e.g., attribute or structural statistics) for alignment, but the dominant discrepancies vary across transfer scenarios, making fixed strategies difficult to generalize.

As visualized in Figure 1, the feature dimensions with the largest KL divergence differ entirely across three Airport transfer tasks — features 2 and 3 dominate in B-E, while features 1, 2, and 4 dominate in U-E. Aligning a fixed subset of features cannot capture the full distributional shift across all scenarios.

The core innovation lies in using characteristic functions (CFs) to represent distributional discrepancies uniformly in the spectral domain — CFs uniquely determine probability distributions (Thm 2) and can adaptively identify the most informative frequency components for alignment via NSD and a learnable frequency sampler.

## Method

### Overall Architecture
GNN encoder → characteristic function transformation → Neural Spectral Discrepancy (NSD) → Adaptive Frequency Sampler → minimax optimization.

### Key Designs

1. **Characteristic Function Transformation**: The empirical distributions of source/target graph embeddings $Z^S, Z^T$ are transformed into the frequency domain: $\Psi(t) = \mathbb{E}[\exp(it^T z)]$. CFs uniquely determine distributions (Thm 2) with convergence guarantees (Thm 1).

2. **Neural Spectral Discrepancy (NSD)**: $\text{NSD} = \int |\Psi^S(t) - \Psi^T(t)|^2 \, dF_T(t)$. This is decomposed into amplitude discrepancy (global structural changes) and phase discrepancy (relational alignment shifts), with coefficient $\kappa$ controlling the balance.

3. **Adaptive Frequency Sampler**: The sampling distribution $p_T(t; \phi)$ is parameterized via a normal scale mixture. Minimax training proceeds as follows: $\phi$ maximizes NSD (finding frequencies of greatest discrepancy), while $\delta$ minimizes NSD (aligning distributions).

4. **Minimax Optimization (Eq 14)**: $\min_\delta \max_\phi \, [L_{\text{source}} + \lambda \cdot L_{\text{align}}]$. GNN parameters $\delta$ optimize classification and alignment jointly, while sampling parameters $\phi$ adversarially search for frequencies of maximum discrepancy.

### Loss & Training
$L = L_{\text{source}}(\text{CE}) + \lambda \cdot L_{\text{align}}(\text{NSD})$. $L_{\text{align}}$ is approximated via Monte Carlo sampling of $M$ frequency points. The reparameterization trick ensures differentiability through sampling.

## Key Experimental Results

### Main Results (partial)

| Task | GAT | GCN | UDAGCN | DEAL | **ADAlign** | Note |
|------|-----|-----|--------|------|-------------|------|
| A→C (Citation) | 62.8 | 69.2 | 72.1 | 74.3 | **76.8** | +2.5 |
| C→D (Citation) | 67.1 | 68.1 | 71.5 | 73.2 | **75.4** | +2.2 |
| B1→B2 (Blog) | 21.2 | 20.5 | 23.1 | 24.8 | **28.3** | +3.5 |

### Ablation Study

| Component | Effect | Note |
|-----------|--------|------|
| Remove adaptive sampler (fixed frequency) | Significant drop | Adaptivity is critical |
| Remove phase alignment | Drop | Both components matter |
| Remove amplitude alignment | Drop | Complementary information |
| $\kappa=0$ (phase only) vs. $\kappa=1$ (amplitude only) | Both worse than $\kappa=0.5$ | Balance is necessary |

### Efficiency Comparison

| Method | Memory (MB) | Training Time (s) | Note |
|--------|-------------|-------------------|------|
| DEAL | 1,245 | 892 | Heavy GNN alignment |
| FLAN | 987 | 756 | Filter-based design |
| **ADAlign** | **423** | **312** | Lightweight spectral operations |

### Key Findings
- ADAlign achieves optimal or near-optimal performance on 16/16 transfer tasks.
- Memory and training time are reduced by 2–3× — CF operations are more lightweight than GNN-based alignment.
- The adaptive frequency sampler automatically focuses on different spectral components across scenarios, validating the design motivation.
- A PAC-Bayesian analysis (Thm 3 + Prop 1) provides theoretical generalization guarantees for NSD.

## Highlights & Insights
- Characteristic functions provide a unified and complete theoretical tool for graph distribution alignment — eliminating the need to manually specify what to align.
- The amplitude/phase decomposition is intuitively meaningful: amplitude $\approx$ global statistical differences; phase $\approx$ relational structural differences.
- The frequency sampler in the minimax framework is a natural formulation for adversarially searching for maximal discrepancy.
- The efficiency advantages make the framework more practically viable.

## Limitations & Future Work
- Monte Carlo frequency sampling introduces variance; the choice of $M$ requires careful trade-off.
- Validation is limited to node classification tasks; graph-level tasks remain unexplored.
- $\kappa$ is currently a hyperparameter; an adaptive $\kappa$ may yield better performance.
- The framework's capacity to handle extreme domain gaps requires further investigation.

## Related Work & Insights
- Introducing characteristic functions — previously used in generative models and knowledge distillation — into GDA opens a new methodological space.
- The idea of adaptive spectral-domain alignment is generalizable to other domain adaptation tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ Characteristic functions + spectral alignment + adaptive sampling
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 datasets, 16 tasks, ablations, and efficiency analysis
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear
- Value: ⭐⭐⭐⭐ A meaningful methodological contribution to GDA

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning Structure-Semantic Evolution Trajectories for Graph Domain Adaptation](learning_structure-semantic_evolution_trajectories_for_graph_domain_adaptation.md)
- [\[ICLR 2026\] Distributionally Robust Classification for Multi-Source Unsupervised Domain Adaptation](distributionally_robust_classification_for_multi-source_unsupervised_domain_adap.md)
- [\[ICLR 2026\] OwlEye: Zero-Shot Learner for Cross-Domain Graph Data Anomaly Detection](owleye_zero-shot_learner_for_cross-domain_graph_data_anomaly_detection.md)
- [\[ICLR 2026\] Probabilistic Kernel Function for Fast Angle Testing](probabilistic_kernel_function_for_fast_angle_testing.md)
- [\[ICLR 2026\] Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning](noisy-pair_robust_representation_alignment_for_positive-unlabeled_learning.md)

</div>

<!-- RELATED:END -->
