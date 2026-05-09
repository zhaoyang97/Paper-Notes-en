---
title: >-
  [Paper Note] LoReTTA: A Low Resource Framework To Poison Continuous Time Dynamic Graphs
description: >-
  [AAAI 2026][Time Series][Temporal Graph Neural Networks] This paper proposes LoReTTA, a two-stage adversarial poisoning attack framework that requires no surrogate model. It first sparsifies high-influence edges via 16 temporal importance metrics, then replaces them with adversarial edges using a degree-preserving negative sampling algorithm. Across 4 datasets × 4 TGNN models, LoReTTA achieves an average performance degradation of 29.47%, while evading 4 anomaly detection systems and resisting 4 defense methods.
tags:
  - AAAI 2026
  - Time Series
  - Temporal Graph Neural Networks
  - Adversarial Attack
  - Data Poisoning
  - Continuous-Time Dynamic Graphs
  - Graph Sparsification
  - Temporal PageRank
date: 2026-05-08
content_hash: 4b20b698a2d44755
---

# LoReTTA: A Low Resource Framework To Poison Continuous Time Dynamic Graphs

**Conference**: AAAI 2026
**arXiv**: [2511.07379](https://arxiv.org/abs/2511.07379)
**Code**: [https://github.com/ansh997/LoReTTA](https://github.com/ansh997/LoReTTA)
**Area**: Time Series / Graph Neural Network Security
**Keywords**: Temporal Graph Neural Networks, Adversarial Attack, Data Poisoning, Continuous-Time Dynamic Graphs, Graph Sparsification, Temporal PageRank

## TL;DR
This paper proposes LoReTTA, a two-stage adversarial poisoning attack framework that requires no surrogate model. It first sparsifies high-influence edges via 16 temporal importance metrics, then replaces them with adversarial edges using a degree-preserving negative sampling algorithm. Across 4 datasets × 4 TGNN models, LoReTTA achieves an average performance degradation of 29.47%, while evading 4 anomaly detection systems and resisting 4 defense methods.

## Background & Motivation

**State of the Field**: Temporal Graph Neural Networks (TGNNs) are widely deployed in high-stakes scenarios such as financial fraud detection, recommender systems, and social network analysis. Continuous-Time Dynamic Graphs (CTDGs) model fine-grained temporal relationships through timestamped edge streams.

**Limitations of Prior Work**:
- Static graph adversarial attack methods do not transfer directly to CTDGs: past perturbations are diluted by new edges, and future edges are unobservable.
- Attacks on Discrete-Time Dynamic Graphs (DTDGs) affect only a single snapshot and lack temporal propagation.
- The existing SotA method T-SPEAR requires training a surrogate model (computationally expensive) and assumes the attacker has access to the complete dataset (training + validation + test), which is unrealistic.

**Root Cause**: CTDG poisoning attacks require precise temporal coordination (when and where to perturb), yet surrogate-model-based approaches are prohibitively costly, and the assumption of full dataset access does not hold in practice.

**Paper Goals**: Design an efficient CTDG poisoning attack framework that requires no surrogate model, accesses only the training set, and satisfies imperceptibility constraints.

**Starting Point**: Directly assess temporal edge importance using graph-theoretic heuristics, then carefully *remove and replace* edges rather than only adding them.

**Core Idea**: Sparsify high-influence edges and fill the gap with constraint-aware negative sampling — requiring no gradients — leveraging Temporal PageRank-driven quantification of temporal edge influence.

## Method

### Overall Architecture
LoReTTA operates in two steps: **Step 1 — Sparsification** (removing high-influence edges) → **Step 2 — Adversarial Negative Sampling** (replacing them with adversarial edges satisfying four imperceptibility constraints). The final poisoned graph is $\tilde{G} = (V, (E \setminus E') \cup \tilde{E})$, with total perturbation budget $\Delta = \lfloor p \cdot |E| \rfloor$.

The attacker is assumed to be **strictly black-box**: no knowledge of model architecture, loss function, or gradients; only partial observation of the training set is permitted.

The four imperceptibility constraints are:
- **C1**: Perturbation budget $\Delta$
- **C2**: Temporal feasibility (new edge timestamps sampled from the original edge time distribution)
- **C3**: Node activity window (new edges connect only nodes active within time window $W$)
- **C4**: Degree preservation (in-degree/out-degree statistics per node remain unchanged)

### Key Designs 1: Edge Sparsification Strategies (16 Heuristics)
**Edge-level sparsification (5 methods)**:
- For each timestamp $t_i$, construct a cumulative static graph $G^{(i)}$ and compute a heuristic score per edge (Degree, PageRank, Jaccard, Preference, Random).
- Sort edges in descending score order and remove the top-$\Delta$ edges.

**Timestamp-level sparsification (11 methods, based on Temporal PageRank drift)**:
- Compute the TPR vector $r^{(t_i)}$ at each timestamp and measure drift between adjacent timestamps as $\delta^{(t_i)} = d(r^{(t_i)}, r^{(t_{i-1})})$.
- Apply 11 distance metrics (MSS, Cosine, Euclidean, Jaccard, JSD, KL, Chebyshev, Wasserstein, etc.).
- High drift → temporal instability period, where small perturbations cause maximum disruption.
- Also includes TER (Temporal EdgeRank) and Combined-TER as edge-level temporal scoring methods.

**Experimental finding**: Similarity metrics (Cosine, Jaccard) consistently outperform distance metrics (KL, JSD), as they better align with the sensitive directions of the TGNN latent space.

### Key Designs 2: Constraint-Aware Negative Sampling Algorithm
For each removed edge, a new edge satisfying constraints C2–C4 is inserted:
1. **C2**: Candidate timestamps are sampled from the original edge time distribution using Kernel Density Estimation (KDE).
2. **C3**: An active node pool is constructed within time window $W$, ensuring endpoints are active within that window.
3. **C4**: Per-node deletion/insertion counts are tracked to maintain in-degree and out-degree statistics.
4. For bipartite graph datasets, endpoints are sampled from distinct node sets; new edges must correspond to node pairs not previously present in the graph.
5. When constraint conflicts yield no valid candidates, a recovery step reinitializes the KDE.

### Key Designs 3: Temporal PageRank and EdgeRank
- **TPR**: Extends classical PageRank to temporal graphs, penalizing intermediate interactions with temporal decay to favor short, temporally consistent paths.
- **TER**: Combines TPR node scores with out-degree normalization to quantify the importance of each edge in propagating temporal signals.
- Time complexity of TPR: $O(|E| + |V|)$; space complexity: $O(|V| \cdot |T|)$.

### Loss & Training
No loss function is involved — LoReTTA is a purely heuristic method with no gradient optimization. Victim models are trained with standard link prediction losses.

## Key Experimental Results

### Main Results (p=0.3, MRR %)

| Method | Wikipedia | UCI | MOOC | Enron | Avg. Drop |
|--------|-----------|-----|------|-------|-----------|
| Clean (TGN) | 80.5 | 44.2 | 61.8 | 27.9 | — |
| T-SPEAR | 69.8 | 30.9 | 32.1 | 27.2 | — |
| ADD-Random | 69.8 | 30.9 | 31.9 | 27.6 | — |
| REM-Degree | 50.5 | 29.7 | 32.1 | 47.6* | — |
| **LoReTTA-Degree** | **66.2** | **17.5** | **20.1** | **12.3** | — |
| **LoReTTA-Cosine** | **54.1** | **15.4** | **23.3** | **20.3** | — |
| **LoReTTA-Jaccard** | **55.2** | **15.7** | **22.9** | **18.3** | — |

- Average performance degradation of **29.47%** across 4 datasets × 4 models.
- Maximum degradation: 42.0% on MOOC, 31.5% on Wikipedia, 28.8% on UCI, 15.6% on Enron.
- **3.91× faster** than T-SPEAR (up to 10×), with no surrogate model required.

### Ablation Study: Defense Robustness (TGN, p=0.3)

| Defense | Wiki Clean→After Attack | UCI Clean→After Attack |
|---------|------------------------|------------------------|
| No defense (LoReTTA-Degree) | 80.5 → 66.2 | 44.2 → 17.5 |
| SVD | 80.5 → 41.0 | 44.2 → 34.0 |
| Cosine Filtering | 80.5 → 27.6 | 44.2 → 8.1 |
| T-shield | 80.5 → 10.3 | 44.2 → 6.0 |

- **Defenses exacerbate performance degradation**: filtering methods mistakenly remove approximately **70%** of legitimate edges (only 30% are actual adversarial edges).
- This demonstrates the strong imperceptibility of LoReTTA.

### Key Findings
1. Similarity metrics (Cosine, Jaccard) consistently outperform distance metrics in the sparsification stage, as they better align with latent space drift in TGNNs.
2. Increasing attacker knowledge (training data ratio 0.2→0.8) does not always improve attack effectiveness, suggesting that vulnerability concentrates in a small number of critical nodes.
3. Attack effectiveness grows approximately linearly with perturbation rate before saturating — LoReTTA's deterministic algorithm prioritizes the highest-impact edges.
4. On Wikipedia, the REM baseline approaches LoReTTA's performance due to the dataset's highly modular semantic clustering structure, which makes sparsification itself highly effective.
5. All four anomaly detectors (MIDAS, F-FADE, AnoEdge-L/G) fail to simultaneously achieve precision and recall above 0.7.

## Highlights & Insights
- **Zero surrogate model, zero gradient black-box attack**: purely heuristic, computationally efficient.
- **Comprehensive constraint system (C1–C4)**: stricter imperceptibility constraints than T-SPEAR, all empirically validated.
- **Systematic evaluation of 16 sparsification strategies**: provides a thorough understanding of temporal graph vulnerability.
- **Dual attack-defense evaluation**: simultaneously validates attack effectiveness, defense robustness, and anomaly detection evasion — among the most complete evaluations in this area.

## Limitations & Future Work
1. On the Wikipedia dataset, the pure removal (REM) baseline approaches LoReTTA's performance when only C1 is enforced, suggesting that negative sampling may introduce implicit regularization on highly modular graphs.
2. The attack targets only the link prediction task; its effectiveness on other downstream tasks such as node classification and graph classification has not been validated.
3. The attack relies strictly on the visibility of training data and is inapplicable in fully black-box settings with no training data access.
4. Ethical risk: the proposed method could be exploited maliciously to compromise real-world systems dependent on TGNNs.

## Related Work & Insights
- **CTDG Poisoning**: T-SPEAR (the first CTDG poisoning attack, requiring a surrogate model); T-Shield (the corresponding defense).
- **TGNN Models**: TGN (memory networks), JODIE (embedding update), TGAT (temporal graph attention), DySAT (dynamic self-attention).
- **Static Graph Attacks**: Structack (degree-based), Nettack (gradient-guided), Metattack (meta-learning).
- **Anomaly Detection**: MIDAS (micro-clustering), F-FADE (frequency domain), AnoEdge (anomalous edge streams).

## Rating
⭐⭐⭐⭐ — This paper makes systematic contributions to the emerging area of CTDG adversarial attacks. The design philosophy of surrogate-free, constraint-aware poisoning is highly pragmatic, and the comprehensive comparison of 16 heuristics provides valuable insight into graph vulnerability. The completeness of the attack-defense evaluation is notable even by security venue standards. Limitations include the absence of a proposed defense mechanism and the unexplored counter-effects of negative sampling on highly modular graphs.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Task-Aware Retrieval Augmentation for Dynamic Recommendation](task-aware_retrieval_augmentation_for_dynamic_recommendation.md)
- [\[AAAI 2026\] GAICo: A Deployed and Extensible Framework for Evaluating Diverse and Multimodal Generative AI Outputs](gaico_a_deployed_and_extensible_framework_for_evaluating_diverse_and_multimodal_.md)
- [\[NeurIPS 2025\] MAESTRO: Adaptive Sparse Attention and Robust Learning for Multimodal Dynamic Time Series](../../NeurIPS2025/time_series/maestro_adaptive_sparse_attention_and_robust_learning_for_multimodal_dynamic_tim.md)
- [\[ICLR 2026\] Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework](../../ICLR2026/time_series/towards_robust_real-world_multivariate_time_series_forecasting_a_unified_framewo.md)
- [\[ICLR 2026\] Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring](../../ICLR2026/time_series/delta-xai_a_unified_framework_for_explaining_prediction_changes_in_online_time_s.md)

<!-- RELATED:END -->
