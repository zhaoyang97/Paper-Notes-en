---
title: >-
  [Paper Note] Robust Watermarking on Gradient Boosting Decision Trees
description: >-
  [AI Safety] This paper proposes the first robust watermarking framework for GBDT models. It embeds watermarks via in-place fine-tuning and introduces four embedding strategies—Wrong Prediction Flip, Outlier Flip, Cluster Center Flip, and Confidence Flip—achieving high embedding success rates, minimal accuracy degradation, and strong robustness against fine-tuning attacks.
tags:
  - AI Safety
date: 2026-05-08
content_hash: fc858e9dfe30addb
---

# Robust Watermarking on Gradient Boosting Decision Trees

- **Conference**: AAAI 2026
- **arXiv**: [2511.09822](https://arxiv.org/abs/2511.09822)
- **Code**: [jc4303/gbdt_watermarking](https://github.com/jc4303/gbdt_watermarking)
- **Area**: AI Security
- **Keywords**: Watermarking, Gradient Boosting Decision Trees, Intellectual Property Protection, Model Security, In-place Fine-tuning

## TL;DR

This paper proposes the first robust watermarking framework for GBDT models. It embeds watermarks via in-place fine-tuning and introduces four embedding strategies—Wrong Prediction Flip, Outlier Flip, Cluster Center Flip, and Confidence Flip—achieving high embedding success rates, minimal accuracy degradation, and strong robustness against fine-tuning attacks.

## Background & Motivation

- **Widespread Use of GBDTs**: Gradient boosting decision trees achieve superior performance on structured data and are extensively adopted in both industry and academia, including privacy-sensitive and healthcare domains.
- **Lack of Watermarking Research**: While watermarking techniques for neural networks have been extensively studied, methods for protecting GBDT models remain severely underdeveloped.
- **Challenges in GBDT Watermarking**:
    - Trees are constructed sequentially; each tree depends on gradients from prior predictions, so modifying existing trees can cause cascading damage.
    - Tree models are non-differentiable, preventing direct transfer of neural network watermarking approaches.
    - Direct tree modification methods designed for random forests are inapplicable to gradient boosting models due to inter-tree dependencies.
- **Limitations of Prior Work**: The watermarking method for boosted trees by Zhao et al. (KDD 2022) focuses solely on fragile integrity verification (weak watermarking) rather than robust embedding.

## Method

### 1. In-place Update Mechanism

Conventional GBDT fine-tuning appends new trees (e.g., in XGBoost), which can be trivially removed by pruning low-contribution trees. This paper proposes **in-place updates** that directly modify the internal parameters of existing trees rather than adding new ones, enabling deeper watermark integration.

Core procedure (Algorithm 1):
- For each boosting iteration $m$ and each class $k$, compute pseudo-residuals to construct the fine-tuning dataset:

$$\mathcal{D}_{\text{fine}}' = \{(\mathbf{x}_i, r_{i,k} - p_{i,k})\}$$

- Compute updated gradients $g_{i,k}'$ and Hessians $h_{i,k}'$.
- For each non-terminal node in the tree (depth-first traversal), recompute the gain and optimal split $S'$.
- If the new split $S' \neq S$, retrain the corresponding subtree; otherwise, only update the affected leaf node predictions.

### 2. Watermark Embedding Framework

Given a candidate dataset $\mathcal{D}_{\text{cand}}$, a candidate sample set $\mathcal{C}$ is identified, from which a subset $\mathcal{W} \subset \mathcal{C}$ of size $k$ is selected for watermark embedding. Each sample encodes one bit of information: a modified label encodes 1, while the original label encodes 0.

The watermark label is set to the most confident incorrect prediction, excluding both the ground-truth and the model's original prediction:

$$y_i^{\text{wm}} = \underset{c \neq y_i,\; c \neq \hat{y}_i}{\text{argmax}}\; F_c(\mathbf{x}_i)$$

### 3. Four Watermark Embedding Strategies

**Wrong Prediction Flip**:
- Selects samples from $\mathcal{D}_{\text{cand}}$ that the model initially misclassifies, retaining the $n$ samples with the lowest confidence as candidates.
- The watermark label is assigned as the second-highest-probability incorrect class (rather than the original erroneous prediction) to avoid confusion with "hard samples" that any unrelated model may also misclassify.
- Advantage: Embedding occurs in regions already prone to error, minimizing impact on overall accuracy.
- Limitation: Relies on the availability of mispredicted samples; GBDTs are typically highly accurate on training data, leaving insufficient candidates.

**Outlier Flip**:
- Selects the $n$ correctly predicted samples that are farthest from all cluster centroids in the feature space:

$$\mathcal{C} = \left\{\underset{\mathbf{x}_i \in \mathcal{D}}{\text{argmax}_n}\; \min_{j \in \{1,\dots,m\}} \|\mathbf{x}_i - \boldsymbol{\mu}_j\| \right\}$$

- Employs k-Means clustering, selecting the number of clusters $m$ that maximizes the silhouette coefficient.
- Embedding watermarks in sparse regions limits accuracy degradation and enhances robustness against fine-tuning.

**Cluster Center Flip**:
- Clusters the data and selects the sample closest to each cluster centroid as a watermark candidate.
- Additionally selects the $l$ nearest neighbors of each centroid sample to retain their original correct labels, forming a local "hole" in the decision boundary.
- The correct-label neighbors serve as anchors to preserve the global decision boundary and minimize accuracy loss.
- To counteract the opposing gradient pressure from neighbors, the centroid sample is duplicated once in the fine-tuning data.

**Confidence Flip**:
- Selects the $n$ correctly predicted samples with the lowest model confidence:

$$\mathcal{C} = \underset{\mathbf{x}_i \in \mathcal{D}}{\text{argmin}_n}\; F_{y_i}(\mathbf{x}_i)$$

- These samples reside near decision boundaries and are thus more amenable to label flipping.
- Embedding has minimal impact on high-confidence regions, yielding strong robustness.

### 4. Candidate Selection Strategies

Two strategies are proposed for selecting the final $k$ watermark samples from the candidate set $\mathcal{C}$:

- **Lowest Confidence Selection**: Selects the $k$ samples with the lowest prediction confidence, which lie near decision boundaries and are easier to embed.
- **Maximum Distance Selection**: Maximizes the pairwise spatial distances among watermark samples, analogous to the maximum diversity problem (NP-hard); a greedy approximation is employed.

## Key Experimental Results

### Experimental Setup

- **Datasets**: Avila, Image Segmentation, Letter Recognition, optdigits, pendigits, Wine Quality
- **Scenarios**: $\mathcal{D}_{\text{cand}} = \mathcal{D}_{\text{train}}$ (insider watermarking) and $\mathcal{D}_{\text{cand}} \neq \mathcal{D}_{\text{train}}$ (post-hoc watermarking)
- **Watermark Ratio**: $|\mathcal{W}|/|\mathcal{D}_{\text{train}}| \in \{0.001, 0.01, 0.1\}$
- **Evaluation Metrics**: Embedding success rate $\mathcal{A}_{\text{wm}}$, adjusted model accuracy $\mathcal{A}_{\text{model}}' = \mathcal{A}_{\text{model}} \cdot \mathcal{A}_{\text{wm}}$, and fine-tuning robustness

### Watermark Embedding Success Rate (Table 1, $\mathcal{D}_{\text{cand}} = \mathcal{D}_{\text{train}}$)

| Method | ratio=0.001 | ratio=0.01 | ratio=0.1 |
|--------|-------------|------------|-----------|
| Cluster (Conf) | 0.792 | 0.980 | 0.999 |
| Outlier (Conf) | 0.896 | 0.953 | 0.999 |
| Conf. (Conf) | 0.771 | 0.951 | 0.999 |
| Random (Conf) | 0.694 | 0.819 | 0.982 |

All proposed methods achieve significantly higher average success rates than the random baseline, approaching 100% at larger watermark ratios.

### Adjusted Model Accuracy (Table 3, $\mathcal{D}_{\text{cand}} = \mathcal{D}_{\text{train}}$)

| Method | ratio=0.001 | ratio=0.01 | ratio=0.1 |
|--------|-------------|------------|-----------|
| Cluster (Conf) | 0.699 | 0.880 | 0.872 |
| Outlier (Conf) | 0.802 | 0.854 | 0.869 |
| Conf. (Conf) | 0.681 | 0.854 | 0.880 |
| Random (Conf) | 0.603 | 0.729 | 0.877 |

Cluster Flip and Confidence Flip demonstrate competitive accuracy preservation, both outperforming the random baseline.

### Fine-tuning Robustness (Table 5, $\mathcal{D}_{\text{cand}} = \mathcal{D}_{\text{train}}$)

| Method | ratio=0.001 | ratio=0.01 | ratio=0.1 |
|--------|-------------|------------|-----------|
| Cluster (Conf) | 0.875 | 0.958 | 0.962 |
| Conf. (Conf) | 0.833 | 0.968 | 0.986 |
| Conf. (Dist) | 0.833 | 0.976 | 0.989 |
| Random (Conf) | 0.778 | 0.865 | 0.923 |

Confidence Flip generally achieves the best robustness, maintaining high watermark detection rates even after subsequent fine-tuning.

## Key Findings

1. **In-place Fine-tuning is Critical**: Directly modifying existing tree structures rather than appending new trees prevents the watermark from being trivially removed via pruning.
2. **Each Strategy Suits Different Scenarios**: Wrong Prediction Flip achieves the highest success rate but is constrained by candidate availability; Cluster Center Flip best preserves model accuracy; Confidence Flip offers the strongest robustness; Outlier Flip performs stably when data distributions are similar.
3. **Larger Watermark Ratios Yield Greater Stability**: At ratio=0.1, nearly all methods achieve near-perfect success rates and robustness scores.
4. **Candidate Data Source Affects Performance**: Using an independent dataset ($\mathcal{D}_{\text{cand}} \neq \mathcal{D}_{\text{train}}$) avoids gradient conflicts, while the insider setting can achieve comparable results via a duplication factor.

## Highlights & Insights

- **Pioneering Contribution**: The first robust watermarking framework specifically designed for GBDTs, filling a critical gap in intellectual property protection for tree-based models.
- **Systematic Design**: The combination of four embedding strategies and two candidate selection strategies forms a comprehensive methodological matrix with targeted solutions for different scenarios.
- **Strong Practicality**: Supports both insider and post-hoc watermarking scenarios, making it applicable to third-party IP protection after model deployment.
- **Solid Theoretical Analysis**: Gradient direction analysis establishes the theoretical constraints on watermark embedding, enhancing the interpretability of the proposed methods.

## Limitations & Future Work

- **Classification Tasks Only**: Regression tasks and other GBDT application scenarios are not explored.
- **Sensitivity to Clustering Parameters**: The effectiveness of Outlier Flip and Cluster Center Flip depends on clustering quality and parameter selection.
- **Distribution Assumption**: Outlier Flip assumes similar distributions between the fine-tuning data and candidate data, which may not hold in practice.
- **Wrong Prediction Flip is Limited**: High-accuracy models have almost no mispredictions, severely restricting the applicability of this strategy.
- **Computational Overhead Not Discussed**: No analysis of the time or space complexity of in-place updates compared to standard fine-tuning is provided.
- **Adversarial Attacks Not Considered**: Robustness is evaluated only against standard fine-tuning; targeted watermark removal attacks are not addressed.

## Related Work & Insights

- **Neural Network Watermarking**: Adi et al. (USENIX 2018) propose backdoor-based watermarking; Uchida et al. (2017) embed watermarks via weight regularization.
- **Tree Model Watermarking**: Calzavara et al. (EDBT 2025) directly modify tree structures for random forests; Zhao et al. (KDD 2022) propose fragile watermarking for boosted trees.
- **GBDT Frameworks**: XGBoost (Chen & Guestrin, KDD 2016), LightGBM (Ke et al., NeurIPS 2017).
- **Robust Watermarking**: Pagnotta et al. (ACSAC 2024) and Yan et al. (USENIX 2023) focus on modification-resistant watermarks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to introduce robust watermarking to GBDTs; the problem formulation is pioneering.
- **Technical Depth**: ⭐⭐⭐ — The four strategies are well-motivated but not technically demanding; in-place update is the primary innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Systematic evaluation across multiple datasets, ratios, and scenarios, though comparisons with additional baseline methods are absent.
- **Value**: ⭐⭐⭐⭐ — Directly addresses the IP protection needs of GBDT models with practical relevance for industry and legal contexts.
- **Overall Recommendation**: ⭐⭐⭐⭐ — A solid work that fills an important gap; the methodology, while not highly complex, is systematic and well-rounded.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer Collaborative Learning](learning_to_collaborate_an_orchestrated-decentralized_framework_for_peer-to-peer.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[AAAI 2026\] ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias](problog4fairness_a_neurosymbolic_approach_to_modeling_and_mitigating_bias.md)
- [\[ICLR 2026\] Skirting Additive Error Barriers for Private Turnstile Streams](../../ICLR2026/ai_safety/skirting_additive_error_barriers_for_private_turnstile_streams.md)

</div>

<!-- RELATED:END -->
