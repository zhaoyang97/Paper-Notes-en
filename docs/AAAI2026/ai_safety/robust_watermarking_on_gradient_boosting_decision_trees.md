---
title: >-
  [Paper Note] Robust Watermarking on Gradient Boosting Decision Trees
description: >-
  [AI Safety] Proposes the first robust watermarking framework for GBDT models. By embedding watermarks through in-place fine-tuning and designing four embedding strategies (Wrong Prediction Flip, Outlier Flip, Cluster Center Flip, Confidence Flip), it achieves high embedding success rates, low accuracy loss, and strong robustness against fine-tuning.
tags:
  - "AI Safety"
date: 2026-05-08
content_hash: e24e7f64f5707219
---

# Robust Watermarking on Gradient Boosting Decision Trees

- **Conference**: AAAI 2026
- **arXiv**: [2511.09822](https://arxiv.org/abs/2511.09822)
- **Code**: [jc4303/gbdt_watermarking](https://github.com/jc4303/gbdt_watermarking)
- **Area**: AI Security
- **Keywords**: Watermarking, Gradient Boosting Decision Trees, Intellectual Property Protection, Model Security, In-place Fine-tuning

## TL;DR

Proposes the first robust watermarking framework for GBDT models. By embedding watermarks through in-place fine-tuning and designing four embedding strategies (Wrong Prediction Flip, Outlier Flip, Cluster Center Flip, Confidence Flip), it achieves high embedding success rates, low accuracy loss, and strong robustness against fine-tuning.

## Background & Motivation

- **Widespread Use of GBDTs**: Gradient Boosting Decision Trees perform exceptionally well on structured data and are widely used in both industry and academia, including privacy-sensitive and healthcare domains.
- **Lack of Watermarking Research**: While watermarking technologies for neural networks have been heavily studied, watermarking methods for GBDT models remain highly deficient.
- **Challenges in GBDT Watermarking**:
    - Trees are constructed sequentially, where each tree depends on the gradients of prior predictions; modifying existing trees causes cascading damage.
    - Tree models are non-differentiable, preventing direct transfer of neural network watermarking methods.
    - Direct tree modification methods for random forests are inapplicable to gradient boosting models since the trees are not independent.
- **Limitations of Prior Work**:
    - The watermarking method for boosting trees by Zhao et al. (KDD 2022) focuses only on fragile integrity verification (fragile watermarking) rather than robust embedding.

## Method

### 1. In-place Update Mechanism

Traditional GBDT fine-tuning is implemented by appending new trees (e.g., in XGBoost). However, newly added trees can be easily removed by pruning low-contribution trees. This paper proposes **in-place updates** to directly modify the internal parameters of existing trees rather than adding new ones, embedding the watermark more deeply.

Core workflow (Algorithm 1):
- For each boosting iteration $m$ and each class $k$, pseudo-residuals are calculated to construct the fine-tuning dataset:

$$\mathcal{D}_{\text{fine}}' = \{(\mathbf{x}_i, r_{i,k} - p_{i,k})\}$$

- Compute the new gradients $g_{i,k}'$ and Hessians $h_{i,k}'$
- For each non-terminal node in the tree (depth-first traversal), recompute the gain and optimal split $S'$
- If the new split $S' \neq S$, retrain the subtree; otherwise, only update the prediction values of the affected leaf nodes.

### 2. Watermark Embedding Framework

Given a candidate dataset $\mathcal{D}_{\text{cand}}$, a candidate sample set $\mathcal{C}$ is identified, from which a subset $\mathcal{W} \subset \mathcal{C}$ (of size $k$) is selected for watermark embedding. Each sample encodes one bit of information: modifying the label to 1, and keeping the original label to 0.

The watermark label is set to the most confident incorrect prediction (excluding the ground-truth label and the model's original prediction):

$$y_i^{\text{wm}} = \underset{c \neq y_i,\; c \neq \hat{y}_i}{\text{argmax}}\; F_c(\mathbf{x}_i)$$

### 3. Four Watermark Embedding Strategies

**Wrong Prediction Flip**:
- Select samples initially mispredicted by the model from $\mathcal{D}_{\text{cand}}$, and take the $n$ samples with the lowest confidence as candidates.
- The watermark label is set to the second most probable incorrect class (instead of the original mispredicted one) to avoid confusion with "hard samples" of unrelated models.
- Advantages: Embedding occurs in already error-prone areas, minimizing the impact on overall accuracy.
- Limitations: Dependent on the number of incorrect predictions; GBDTs are typically highly accurate on training sets, leading to insufficient candidates.

**Outlier Flip**:
- Select the $n$ correctly predicted samples that are farthest from all cluster centers in the feature space:

$$\mathcal{C} = \left\{\underset{\mathbf{x}_i \in \mathcal{D}}{\text{argmax}_n}\; \min_{j \in \{1,\dots,m\}} \|\mathbf{x}_i - \boldsymbol{\mu}_j\| \right\}$$

- Perform k-Means clustering and select the number of clusters $m$ that maximizes the silhouette coefficient.
- Embedding watermarks in sparse regions limits the impact on accuracy while enhancing robustness against fine-tuning.

**Cluster Center Flip**:
- Cluster the dataset and select the sample closest to the centroid in each cluster as a watermark candidate.
- Simultaneously select its $l$-nearest neighbors to maintain their original correct labels, forming a local "hollow."
- Anchoring the decision boundary through neighbor samples with correct labels minimizes the impact on global accuracy.
- To counter the opposing pressure from neighbors, replicate the centroid sample once in the fine-tuning data.

**Confidence Flip**:
- Select the $n$ samples that are correctly predicted by the model but have the lowest confidence:

$$\mathcal{C} = \underset{\mathbf{x}_i \in \mathcal{D}}{\text{argmin}_n}\; F_{y_i}(\mathbf{x}_i)$$

- These samples lie near the decision boundary, making their labels easier to flip.
- Embedding has minimal impact on high-confidence regions, offering relatively good robustness.

### 4. Candidate Selection Strategies

To select the final $k$ watermark samples from the candidate set $\mathcal{C}$, two strategies are proposed:

- **Lowest Confidence Selection**: Selects the $k$ samples with the lowest prediction confidence, which are easier to embed as they lie at the decision boundary.
- **Maximum Distance Selection**: Maximizes the spatial distance between watermark samples, similar to the maximum diversity problem (NP-hard), and is addressed using a greedy approximation algorithm.

## Experiments

### Experimental Setup

- **Datasets**: Avila, Image Segmentation, Letter Recognition, optdigits, pendigits, Wine Quality
- **Scenarios**: $\mathcal{D}_{\text{cand}} = \mathcal{D}_{\text{train}}$ (internal watermarking) and $\mathcal{D}_{\text{cand}} \neq \mathcal{D}_{\text{train}}$ (post-hoc watermarking)
- **Watermark Ratio**: $|\mathcal{W}|/|\mathcal{D}_{\text{train}}| \in \{0.001, 0.01, 0.1\}$
- **Evaluation Metrics**: Watermark embedding success rate $\mathcal{A}_{\text{wm}}$, adjusted model accuracy $\mathcal{A}_{\text{model}}' = \mathcal{A}_{\text{model}} \cdot \mathcal{A}_{\text{wm}}$, and robustness against fine-tuning

### Watermark Embedding Success Rate (Table 1, $\mathcal{D}_{\text{cand}} = \mathcal{D}_{\text{train}}$)

| Method | ratio=0.001 | ratio=0.01 | ratio=0.1 |
|------|-------------|------------|-----------|
| Cluster (Conf) | 0.792 | 0.980 | 0.999 |
| Outlier (Conf) | 0.896 | 0.953 | 0.999 |
| Conf. (Conf) | 0.771 | 0.951 | 0.999 |
| Random (Conf) | 0.694 | 0.819 | 0.982 |

The average success rate of all proposed methods is significantly higher than that of the random baseline, reaching close to 100% particularly at larger watermark ratios.

### Adjusted Model Accuracy (Table 3, $\mathcal{D}_{\text{cand}} = \mathcal{D}_{\text{train}}$)

| Method | ratio=0.001 | ratio=0.01 | ratio=0.1 |
|------|-------------|------------|-----------|
| Cluster (Conf) | 0.699 | 0.880 | 0.872 |
| Outlier (Conf) | 0.802 | 0.854 | 0.869 |
| Conf. (Conf) | 0.681 | 0.854 | 0.880 |
| Random (Conf) | 0.603 | 0.729 | 0.877 |

Cluster Flip and Confidence Flip demonstrate competitive performance in maintaining model accuracy, both outperforming the random baseline.

### Robustness Against Fine-Tuning (Table 5, $\mathcal{D}_{\text{cand}} = \mathcal{D}_{\text{train}}$)

| Method | ratio=0.001 | ratio=0.01 | ratio=0.1 |
|------|-------------|------------|-----------|
| Cluster (Conf) | 0.875 | 0.958 | 0.962 |
| Conf. (Conf) | 0.833 | 0.968 | 0.986 |
| Conf. (Dist) | 0.833 | 0.976 | 0.989 |
| Random (Conf) | 0.778 | 0.865 | 0.923 |

Confidence Flip overall performs slightly better in terms of robustness, with watermarks maintaining high detection rates even after further fine-tuning.

## Key Findings

1. **In-place Fine-Tuning is Crucial**: Modifying existing tree structures directly instead of adding new trees prevents the watermarks from being easily removed by pruning.
2. **Four Strategies Suit Different Scenarios**: Wrong Prediction Flip achieves the highest success rate but has limited candidates; Cluster Center Flip maintains the best accuracy; Confidence Flip possesses the strongest robustness; Outlier Flip performs stably when distributions are similar.
3. **Larger Watermark Ratios Bring Better Stability**: At ratio=0.1, the success rate and robustness of all methods are close to perfect.
4. **Source of Candidate Data Impacts Performance**: Using an independent dataset ($\mathcal{D}_{\text{cand}} \neq \mathcal{D}_{\text{train}}$) avoids gradient conflict, though the internal dataset can also achieve good results using replication factors.

## Highlights & Insights

- **Pioneering Work**: Represents the first robust watermarking framework designed for GBDT, filling an important gap in the intellectual property protection of tree models.
- **Systematic Design**: Four embedding strategies coupled with two candidate selection strategies form a comprehensive method matrix, providing targeted solutions for various scenarios.
- **High Practicality**: Supports both internal and post-hoc watermarking scenarios, which is suitable for third-party protection after model release.
- **Solid Theoretical Analysis**: Analysis of gradient directions reveals the theoretical constraints of watermark embedding, enhancing the interpretability of the method.

## Limitations & Future Work

- **Validation Limited to Classification**: Did not explore regression tasks or other GBDT application scenarios.
- **Sensitivity to Clustering Parameters**: The performance of Outlier Flip and Cluster Center Flip relies heavily on clustering quality and parameter selection.
- **Distribution Assumption**: Outlier Flip assumes similarity between the distributions of fine-tuning data and candidate data, which might not hold true in practice.
- **Limited Wrong Prediction Flip**: Powerful models yield almost no incorrect predictions, which heavily limits the applicability of this strategy.
- **Computational Overhead Omitted**: Lacks time/space complexity analysis of the in-place updates compared to standard fine-tuning.
- **Adversarial Attacks Unconsidered**: Evaluated only standard fine-tuning robustness without considering targeted watermark-removal attacks.

## Related Work & Insights

- **Neural Network Watermarking**: Adi et al. (USENIX 2018) proposed backdoor watermarking; Uchida et al. (2017) embedded weight regularization watermarks.
- **Tree Model Watermarking**: Calzavara et al. (EDBT 2025) directly modified tree structures for random forests; Zhao et al. (KDD 2022) proposed fragile watermarking for boosting trees.
- **GBDT Frameworks**: XGBoost (Chen & Guestrin, KDD 2016), LightGBM (Ke et al., NeurIPS 2017).
- **Robust Watermarking**: Pagnotta et al. (ACSAC 2024), Yan et al. (USENIX 2023) focused on modification-resistant watermarks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introduces the concept of robust watermarking to GBDT for the first time, establishing a pioneering problem formulation.
- **Technical Depth**: ⭐⭐⭐ — The four strategies are well-designed though they present a relatively low technical barrier, with the in-place update being the primary innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Systematic comparison across multiple datasets, ratios, and scenarios, though comparison with other potential methods is lacking.
- **Value**: ⭐⭐⭐⭐ — Directly addresses IP protection needs for GBDT models, providing practical significance for industrial and legal scenarios.
- **Overall Recommendation**: ⭐⭐⭐⭐ — A solid work filling an important gap; although the methodology is not overly complex, it is highly systematic and complete.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias](problog4fairness_a_neurosymbolic_approach_to_modeling_and_mitigating_bias.md)
- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[AAAI 2026\] Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer Collaborative Learning](learning_to_collaborate_an_orchestrated-decentralized_framework_for_peer-to-peer.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [\[ICML 2025\] Retraining with Predicted Hard Labels Provably Increases Model Accuracy](../../ICML2025/ai_safety/retraining_with_predicted_hard_labels_provably_increases_model_accuracy.md)

</div>

<!-- RELATED:END -->
