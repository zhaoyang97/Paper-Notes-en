---
title: >-
  [Paper Note] FairGSE: Fairness-Aware Graph Neural Network without High False Positive Rates
description: >-
  [AAAI 2026][AI Safety][Graph Neural Networks] This work is the first to reveal the "FPR shortcut" issue in fairness-aware GNNs, where existing methods achieve fairness metrics by misclassifying a large number of negative samples as positive. It proposes the FairGSE framework, which reweights graph edges by maximizing 2D structural entropy to simultaneously improve fairness and reduce false positive rates, achieving a 39% reduction in FPR.
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Graph Neural Networks"
  - "Fairness"
  - "False Positive Rate"
  - "Structural Entropy"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: afb4e6f9c8bf3870
---

# FairGSE: Fairness-Aware Graph Neural Network without High False Positive Rates

**Conference**: AAAI 2026  
**arXiv**: [2511.12132](https://arxiv.org/abs/2511.12132)  
**Code**: None  
**Area**: AI Safety/Fairness  
**Keywords**: Graph Neural Networks, Fairness, False Positive Rate, Structural Entropy, Contrastive Learning

## TL;DR
This work is the first to reveal the "FPR shortcut" issue in fairness-aware GNNs, where existing methods achieve fairness metrics by misclassifying a large number of negative samples as positive. It proposes the FairGSE framework, which reweights graph edges by maximizing 2D structural entropy to simultaneously improve fairness and reduce false positive rates, achieving a 39% reduction in FPR.

## Background & Motivation

**Background**: Fairness-aware GNNs have achieved satisfactory performance on fairness metrics such as $\Delta_{SP}$ (Statistical Parity) and $\Delta_{EO}$ (Equal Opportunity) through methods like adversarial learning (FairGNN), feature/structural debiasing (EDITS), and invariant learning (FairINV).

**Limitations of Prior Work**: Existing methods solely focus on fairness metrics while ignoring the capability of GNNs to predict negative labels. The authors found that FairSIN achieves an FPR of nearly 100% on the Credit dataset—essentially classifying almost all customers as defaulters. This shortcut of "achieving fairness by predicting everyone as positive" is extremely dangerous in high-risk scenarios (e.g., credit evaluation, bail decisions).

**Key Challenge**: $\Delta_{SP}$ and $\Delta_{EO}$ only measure the difference in the probability of positive predictions across sensitive groups. A model that predicts all nodes as positive will achieve a perfect score of $\Delta_{SP} = \Delta_{EO} = 0$, but at the cost of an FPR of 100%.

**Goal**: How to improve fairness while avoiding the FPR shortcut?

**Key Insight**: Utilizing two-dimensional structural entropy (2D-SE)—an information-theoretic metric based on graph structure partitioning. It is theoretically proven that maximizing 2D-SE can simultaneously provide upper bounds on $\Delta_{SP}$, $\Delta_{EO}$, and FPR.

**Core Idea**: Reweighting graph edges by maximizing 2D structural entropy based on sensitive attribute partitioning to achieve balanced message aggregation across sensitive groups while avoiding the FPR shortcut.

## Method

### Overall Architecture
Three components: (1) Graph Structure Learner: learns a trainable adjacency matrix $\mathbf{A}^l$ to maximize 2D-SE; (2) Contrastive Learning Component: utilizes the original graph as the anchor view and the learned graph as the learner view, maintaining structural consistency via contrastive loss; (3) Structural Bootstrapping Mechanism: progressively integrates the learned fair structure into the anchor view.

### Key Designs

1. **Graph Structure Learner**:

    - **Function**: Maximizes two-dimensional structural entropy by optimizing edge weights.
    - **Mechanism**: Each edge is associated with a trainable parameter $a_{(i,j)}$, which is converted into a weight $\mathbf{A}^l_{(i,j)} = \sigma(a_{(i,j)})$, and then $\mathcal{L}_{SE} = -H^{\mathcal{P}_S}(G_l)$ is minimized.
    - **Design Motivation**: Theorem 1 proves that $\Delta_{SP} \leq \sqrt{2(H^{max} - H^{\mathcal{P}_S}(G))}$; Theorem 2 proves that FPR has a similar upper bound.
    - **Design Motivation**: High 2D-SE encourages message aggregation across sensitive groups, reducing the dependence of node representations on sensitive attributes.

2. **Contrastive Learning Component**:

    - **Function**: Prevents the learned graph from deviating too much from the original structure, preserving classification performance.
    - **Mechanism**: A shared GNN encoder encodes both the anchor view and the learner view, aligning the representations of the same node in both views using NT-Xent loss.
    - **Design Motivation**: Optimizing 2D-SE alone can lead to training instability and structural distortion; contrastive learning constrains this by maximizing $I(G_a, G_l)$ to preserve structural information.

3. **Structural Bootstrapping Mechanism**:

    - **Function**: Progressively updates the anchor view to eliminate bias in the original graph.
    - **Mechanism**: $\mathbf{A}^a = \tau \mathbf{A}^a + (1-\tau) \mathbf{A}^l$, where $\tau = 0.9999$.
    - **Design Motivation**: A fixed anchor view inherits biases from the original graph and leads to overfitting; slowly incorporating the high 2D-SE learned structure helps debias progressively.

### Loss & Training
$\mathcal{L} = \mathcal{L}_{task} + \lambda_1 \mathcal{L}_{cont} - \lambda_2 \mathcal{L}_{SE}$, where $\mathcal{L}_{task}$ is the cross-entropy loss for node classification.

## Key Experimental Results

### Main Results

| Method | Credit FPR↓ | Credit $\Delta_{SP}$↓ | Pokec_n FPR↓ | Pokec_z FPR↓ |
|------|------------|---------------------|-------------|-------------|
| Vanilla GCN | 45.96 | 14.63 | 27.06 | 27.56 |
| FairGNN | 46.48 | 8.61 | 19.10 | 27.22 |
| FairSIN | **99.76** | 0.52 | 23.92 | - |
| FairINV | 70.86 | 4.85 | 23.78 | - |
| **FairGSE** | **41.45** | 5.09 | **18.46** | - |

FairGSE achieves an FPR of only 41.45% on Credit (58 percentage points lower than FairSIN's 99.76%) while maintaining competitive fairness metrics.

### Ablation Study
- **Removing 2D-SE objective**: FPR significantly rises, validating the critical role of structural entropy in avoiding the FPR shortcut.
- **Removing contrastive learning**: Classification performance drops and structure becomes distorted.
- **Removing structural bootstrapping**: Fairness improvement is limited, as original bias continues to propagate through the anchor view.

### Key Findings
- The FPR shortcut is a prevalent phenomenon in fairness-aware GNNs—FairSIN, FairVGNN, FairINV, and DAB-GNN all exhibit FPR >70% on Credit.
- Theoretical guarantees of maximizing 2D-SE hold: upper bounds of $\Delta_{SP}$ and FPR decrease as 2D-SE increases.
- FairGSE reduces FPR by approximately 39% while maintaining competitive ACC/AUC.

## Highlights & Insights
- **The discovery of the FPR shortcut is valuable in its own right**: It exposes the blind spots of traditional fairness metrics ($\Delta_{SP}/\Delta_{EO}$), reminding researchers and practitioners not to rely solely on fairness scores.
- **Bridging information theory and fairness**: Using 2D structural entropy to unify the optimization of fairness and FPR represents a theoretically elegant approach.
- **Transferable insight**: The concept of focusing on overall model prediction quality (instead of just fairness metrics) can be generalized to other fairness research domains.

## Limitations & Future Work
- Only binary sensitive attributes and binary labels are considered.
- The computation and gradients of 2D-SE require partitioning by sensitive groups; the computational overhead of extending this to multiple groups remains to be analyzed.
- Contrastive learning introduces additional training overhead.
- The applicability to dynamic graphs or heterogeneous graphs has not been discussed.

## Related Work & Insights
- **vs FairGNN**: FairGNN utilizes adversarial learning for debiasing, but its FPR remains high (46.48%), whereas FairGSE fundamentally addresses this via structural reweighting.
- **vs FairSIN**: FairSIN's FPR is close to 100%, serving as an extreme case of the FPR shortcut.
- **vs DAB-GNN**: DAB-GNN constrains representation differences using Wasserstein distance but similarly fails to control the FPR.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to systematically reveal the FPR shortcut issue; the 2D-SE theoretical framework is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple datasets + 8 baselines + ablation study; however, most datasets are small-scale graphs.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear motivation, rigorous theoretical proofs, and fluent logical reasoning.
- **Value**: ⭐⭐⭐⭐⭐ Provides insights with a profound impact on fairness research, holding significant practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Verifying Neural Network Robustness with Dual Perturbations](../../CVPR2026/ai_safety/verifying_neural_network_robustness_with_dual_perturbations.md)
- [\[AAAI 2026\] Hashed Watermark as a Filter: A Unified Defense Against Forging and Overwriting Attacks in Neural Network Watermarking](hashed_watermark_as_a_filter_defeating_forging_and_overwriting_attacks_in_weight.md)
- [\[AAAI 2026\] CoRe-Fed: Bridging Collaborative and Representation Fairness via Federated Embedding Distillation](core-fed_bridging_collaborative_and_representation_fairness_via_federated_embedd.md)
- [\[CVPR 2025\] Lyapunov Stable Graph Neural Flow](../../CVPR2025/ai_safety/lyapunov_stable_graph_neural_flow.md)
- [\[ICLR 2026\] ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks](../../ICLR2026/ai_safety/atex-cf_attack-informed_counterfactual_explanations_for_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
