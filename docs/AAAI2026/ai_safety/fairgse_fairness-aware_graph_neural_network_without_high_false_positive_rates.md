---
title: >-
  [Paper Note] FairGSE: Fairness-Aware Graph Neural Network without High False Positive Rates
description: >-
  [AAAI 2026][AI Safety][Graph Neural Networks] This paper is the first to identify the "FPR shortcut" problem in fairness-aware GNNs — existing methods achieve favorable fairness metrics by massively misclassifying negati…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Graph Neural Networks"
  - "Fairness"
  - "False Positive Rate"
  - "Structural Entropy"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: 843eb9d3aee9c469
---

# FairGSE: Fairness-Aware Graph Neural Network without High False Positive Rates

**Conference**: AAAI 2026
**arXiv**: [2511.12132](https://arxiv.org/abs/2511.12132)  
**Code**: None  
**Area**: AI Safety / Fairness
**Keywords**: Graph Neural Networks, Fairness, False Positive Rate, Structural Entropy, Contrastive Learning

## TL;DR
This paper is the first to identify the "FPR shortcut" problem in fairness-aware GNNs — existing methods achieve favorable fairness metrics by massively misclassifying negative samples as positive — and proposes the FairGSE framework, which reweights graph edges by maximizing 2D structural entropy to simultaneously improve fairness and reduce the false positive rate, achieving a 39% reduction in FPR.

## Background & Motivation

**Background**: Fairness-aware GNNs have achieved satisfactory results on fairness metrics such as $\Delta_{SP}$ (statistical parity) and $\Delta_{EO}$ (equal opportunity) through methods including adversarial learning (FairGNN), feature/structure debiasing (EDITS), and invariant learning (FairINV).

**Limitations of Prior Work**: Existing methods focus solely on fairness metrics while neglecting GNNs' ability to correctly predict negative labels. The authors find that FairSIN achieves an FPR close to 100% on the Credit dataset — effectively classifying almost all customers as credit card defaulters. This shortcut of "achieving fairness by predicting everyone as positive" is extremely dangerous in high-stakes scenarios such as credit assessment and bail decisions.

**Key Challenge**: $\Delta_{SP}$ and $\Delta_{EO}$ only measure the disparity in positive prediction rates across sensitive groups — a model that predicts all nodes as positive achieves a perfect $\Delta_{SP} = \Delta_{EO} = 0$, yet has FPR = 100%.

**Goal**: How can fairness be improved while avoiding the FPR shortcut?

**Key Insight**: The paper leverages 2D structural entropy (2D-SE) — an information-theoretic measure based on graph structural partitioning — and theoretically proves that maximizing 2D-SE simultaneously provides upper-bound constraints on $\Delta_{SP}$, $\Delta_{EO}$, and FPR.

**Core Idea**: By maximizing the 2D structural entropy induced by sensitive attribute partitions to reweight graph edges, the framework achieves balanced message aggregation across sensitive groups while avoiding the FPR shortcut.

## Method

### Overall Architecture
Three components: (1) a **graph structure learner** that learns a trainable adjacency matrix $\mathbf{A}^l$ to maximize 2D-SE; (2) a **contrastive learning module** that uses the original graph as the anchor view and the learned graph as the learner view, maintaining structural consistency via contrastive loss; and (3) a **structure bootstrapping mechanism** that progressively incorporates the learned fair structure into the anchor view.

### Key Designs

1. **Graph Structure Learner**:

    - **Function**: Optimizes edge weights to maximize 2D structural entropy.
    - **Mechanism**: Each edge is associated with a trainable parameter $a_{(i,j)}$, converted to a weight via sigmoid as $\mathbf{A}^l_{(i,j)} = \sigma(a_{(i,j)})$, and the model minimizes $\mathcal{L}_{SE} = -H^{\mathcal{P}_S}(G_l)$.
    - **Theoretical Guarantee**: Theorem 1 proves $\Delta_{SP} \leq \sqrt{2(H^{max} - H^{\mathcal{P}_S}(G))}$; Theorem 2 establishes an analogous upper bound for FPR.
    - **Design Motivation**: Higher 2D-SE encourages cross-group message aggregation, reducing node representations' dependence on sensitive attributes.

2. **Contrastive Learning Module**:

    - **Function**: Prevents the learned graph from deviating excessively from the original structure, preserving classification performance.
    - **Mechanism**: A shared GNN encoder encodes both the anchor view and the learner view; NT-Xent loss aligns each node's representations across the two views.
    - **Design Motivation**: Optimizing 2D-SE alone leads to training instability and structural distortion; contrastive learning constrains the objective by maximizing $I(G_a, G_l)$ to preserve structural information.

3. **Structure Bootstrapping Mechanism**:

    - **Function**: Progressively updates the anchor view to eliminate bias present in the original graph.
    - **Mechanism**: $\mathbf{A}^a = \tau \mathbf{A}^a + (1-\tau) \mathbf{A}^l$, with $\tau = 0.9999$.
    - **Design Motivation**: A fixed anchor view inherits the original graph's bias and causes overfitting; slowly incorporating the high-2D-SE learned structure gradually removes bias.

### Loss & Training
$\mathcal{L} = \mathcal{L}_{task} + \lambda_1 \mathcal{L}_{cont} - \lambda_2 \mathcal{L}_{SE}$, where $\mathcal{L}_{task}$ is the cross-entropy loss for node classification.

## Key Experimental Results

### Main Results

| Method | Credit FPR↓ | Credit $\Delta_{SP}$↓ | Pokec_n FPR↓ | Pokec_z FPR↓ |
|---|---|---|---|---|
| Vanilla GCN | 45.96 | 14.63 | 27.06 | 27.56 |
| FairGNN | 46.48 | 8.61 | 19.10 | 27.22 |
| FairSIN | **99.76** | 0.52 | 23.92 | - |
| FairINV | 70.86 | 4.85 | 23.78 | - |
| **FairGSE** | **41.45** | 5.09 | **18.46** | - |

FairGSE achieves an FPR of only 41.45% on Credit — 58 percentage points lower than FairSIN's 99.76% — while remaining competitive on fairness metrics.

### Ablation Study
- **Removing the 2D-SE objective**: FPR increases significantly, confirming the critical role of structural entropy in avoiding the FPR shortcut.
- **Removing contrastive learning**: Classification performance degrades and structural distortion occurs.
- **Removing structure bootstrapping**: Fairness improvement is limited, as the original bias continues to propagate through the fixed anchor view.

### Key Findings
- The FPR shortcut is a pervasive phenomenon in fairness-aware GNNs — FairSIN, FairVGNN, FairINV, and DAB-GNN all exhibit FPR > 70% on the Credit dataset.
- The theoretical guarantees for maximizing 2D-SE hold empirically: upper bounds on $\Delta_{SP}$ and FPR decrease as 2D-SE increases.
- FairGSE reduces FPR by approximately 39% while maintaining competitive ACC/AUC.

## Highlights & Insights
- **The discovery of the FPR shortcut is itself highly valuable**: It exposes a blind spot in fairness metrics $\Delta_{SP}/\Delta_{EO}$, alerting researchers and practitioners not to rely solely on these indicators.
- **Bridging information theory and fairness**: Unifying fairness and FPR control through 2D structural entropy yields an theoretically elegant framework.
- **Transferable perspective**: The principle of attending to overall prediction quality — rather than fairness metrics alone — is generalizable to other fairness research.

## Limitations & Future Work
- Only binary sensitive attributes and binary labels are considered.
- The computation and gradient of 2D-SE require partitioning by sensitive groups; the computational overhead of extending to multiple groups remains unanalyzed.
- Contrastive learning introduces additional training cost.
- Applicability to dynamic or heterogeneous graphs is not discussed.

## Related Work & Insights
- **vs. FairGNN**: FairGNN applies adversarial debiasing but still exhibits high FPR (46.48%); FairGSE addresses the problem at the source through structural reweighting.
- **vs. FairSIN**: FairSIN's FPR approaches 100%, representing an extreme instance of the FPR shortcut.
- **vs. DAB-GNN**: DAB-GNN constrains representational disparity via Wasserstein distance but similarly fails to control FPR.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First systematic identification of the FPR shortcut problem; the 2D-SE theoretical framework is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple datasets, 8 baselines, and ablation studies; however, the datasets are mostly small-scale graphs.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem motivation is clear, theoretical proofs are rigorous, and the overall narrative is logically coherent.
- **Value**: ⭐⭐⭐⭐⭐ A far-reaching insight for fairness research with significant practical implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Hashed Watermark as a Filter: A Unified Defense Against Forging and Overwriting Attacks in Neural Network Watermarking](hashed_watermark_as_a_filter_defeating_forging_and_overwriting_attacks_in_weight.md)
- [\[ICLR 2026\] ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks](../../ICLR2026/ai_safety/atex-cf_attack-informed_counterfactual_explanations_for_graph_neural_networks.md)
- [\[AAAI 2026\] CoRe-Fed: Bridging Collaborative and Representation Fairness via Federated Embedding Distillation](core-fed_bridging_collaborative_and_representation_fairness_via_federated_embedd.md)
- [\[NeurIPS 2025\] Incentivizing Time-Aware Fairness in Data Sharing](../../NeurIPS2025/ai_safety/incentivizing_time-aware_fairness_in_data_sharing.md)
- [\[AAAI 2026\] Breaking the Dyadic Barrier: Rethinking Fairness in Link Prediction Beyond Demographic Parity](breaking_the_dyadic_barrier_rethinking_fairness_in_link_prediction_beyond_demogr.md)

</div>

<!-- RELATED:END -->
