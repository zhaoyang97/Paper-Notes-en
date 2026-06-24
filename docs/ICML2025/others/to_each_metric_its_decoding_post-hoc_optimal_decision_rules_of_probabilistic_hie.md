---
title: >-
  [Paper Note] To Each Metric Its Decoding: Post-Hoc Optimal Decision Rules of Probabilistic Hierarchical Classifiers
description: >-
  [ICML 2025][hierarchical classification] This paper proposes a post-hoc optimal decoding framework for probabilistic hierarchical classifiers. It derives optimal decision rules for various evaluation metrics (such as hierarchical $F_\beta$) and provides general algorithms when the prediction candidate set is restricted to the node set. Furthermore, it derives a dedicated optimal strategy for hierarchical $hF_\beta$ in subset prediction.
tags:
  - "ICML 2025"
  - "hierarchical classification"
  - "decision rules"
  - "optimal decoding"
  - "F-beta score"
  - "probabilistic classifiers"
date: 2026-05-08
content_hash: d95270a2cb9b587c
---

# To Each Metric Its Decoding: Post-Hoc Optimal Decision Rules of Probabilistic Hierarchical Classifiers

**Conference**: ICML 2025  
**arXiv**: [2506.01552](https://arxiv.org/abs/2506.01552)  
**Code**: [https://github.com/RomanPlaud/hierarchical_decision_rules](https://github.com/RomanPlaud/hierarchical_decision_rules)  
**Area**: Classification / Hierarchical Classification  
**Keywords**: hierarchical classification, decision rules, optimal decoding, F-beta score, probabilistic classifiers

## TL;DR
This paper proposes a post-hoc optimal decoding framework for probabilistic hierarchical classifiers. It derives optimal decision rules for various evaluation metrics (such as hierarchical $F_\beta$) and provides general algorithms when the prediction candidate set is restricted to the node set. Furthermore, it derives a dedicated optimal strategy for hierarchical $hF_\beta$ in subset prediction.

## Background & Motivation
**Background**: Hierarchical classification exploits the tree structure of labels to differentiate the severity of errors (e.g., misclassifying a "dog" as a "cat" is more acceptable than misclassifying it as a "car"). Existing methods typically train a probabilistic model and then use heuristic rules for prediction decoding.

**Limitations of Prior Work**: Heuristic decoding (such as selecting the leaf node with the highest probability or performing a greedy search along the tree) is not necessarily aligned with the evaluation metrics. Different metrics require different optimal strategies, yet a systematic theory of optimal decoding remains absent.

**Key Challenge**: Probabilistic models provide rich uncertainty information, but they fail to fully utilize this information during decoding to optimize the target metrics.

**Goal**: Given a trained probabilistic hierarchical classifier and a target evaluation metric, how can post-processing (decoding) be performed to optimize this target metric.

**Key Insight**: Bayesian optimal decision theory—minimizing the conditional expected loss.

**Core Idea**: Different metrics correspond to different optimal decoding rules. This work derives analytical or algorithmic optimal rules for a series of metrics.

## Method

### Overall Architecture
Input: A trained probabilistic hierarchical classifier $P(y|x)$ ($y$ is a node in the hierarchy), target metric $\Psi$  
Output: The optimal prediction $\hat{y}^*(x)$ for a new sample $x$

### Key Designs

1. **Optimal Rules for Node Prediction**:

    - **Function**: Deriving the optimal prediction when the candidate predictions are restricted to a single node in the hierarchy.
    - **Mechanism**: For a given probability distribution $P(y|x)$ and metric $\Psi$, the optimal prediction is $\hat{y}^* = \arg\max_{v \in \mathcal{V}} \mathbb{E}[\Psi(v, y) | x]$. This is instantiated for different metrics:
        - 0-1 hierarchical loss: Select the leaf node with the highest probability.
        - Tree distance loss: Select the node that optimizes the weighted depth.
        - Hierarchical $F_1$: Select the node that maximizes the expected $F_1$ (requires searching all nodes).
    - **Design Motivation**: To provide a universal algorithm ensuring that the optimal node prediction can be found for any metric.

2. **Optimal Rules for Subset Prediction (Dedicated to $hF_\beta$)**:

    - **Function**: Deriving the optimal strategy for hierarchical $F_\beta$ when the prediction can be a set of nodes (e.g., predicting a path from the root to a node).
    - **Mechanism**: The hierarchical $F_\beta$ metric considers the overlap between the predicted set of nodes and the ancestor set of the true label:
    $hF_\beta = \frac{(1+\beta^2) |A(\hat{y}) \cap A(y)|}{(1+\beta^2)|A(y)| + |A(\hat{y})|}$
      where $A(v)$ is the set of ancestors of node $v$. The optimal subset prediction is efficiently computed via dynamic programming on the tree.
    - **Design Motivation**: Subset prediction is more useful under uncertainty (e.g., predicting only "animal" rather than a specific species), which requires a task-metric-oriented strategy.

3. **Efficient Computation Algorithms**:

    - **Function**: Providing polynomial-time algorithms for all derived optimal rules.
    - **Mechanism**: Utilizing the tree structure to perform dynamic programming. For each node $v$, the optimal contribution within the subtree rooted at $v$ is calculated and merged bottom-up.
    - **Design Motivation**: Brute-force search over all possible subsets is exponential; the tree structure must be leveraged to accelerate computation.

### Loss & Training
This is a post-processing method and does not alter model training. The classifier is trained using standard cross-entropy, and the derived optimal rules are applied during decoding.

## Key Experimental Results

### Main Results

| Dataset | Metric | Optimal Decoding (Ours) | Greedy Decoding | Top-1 Leaf Node | Gain |
|---|---|---|---|---|---|
| iNaturalist (8K classes) | $hF_1$ | **0.685** | 0.641 | 0.612 | +7.3% |
| CIFAR100-H | $hF_1$ | **0.762** | 0.731 | 0.718 | +4.3% |
| ImageNet-H | $hF_\beta$ ($\beta=2$) | **0.801** | 0.768 | 0.745 | +4.3% |
| Text Classification (DBpedia) | tree dist | **0.34** | 0.41 | 0.52 | +17% |

### Ablation Study

| Decoding Strategy | $hF_1$ (iNat) | Computation Time | Description |
|---|---|---|---|
| Optimal Decoding (Ours) | **0.685** | 2.3ms/sample | Dynamic Programming |
| Greedy Search along Tree | 0.641 | 0.5ms/sample | Heuristic |
| Max Probability Leaf Node | 0.612 | 0.1ms/sample | Simplest |
| Random Sampling by Probability | 0.595 | 0.1ms/sample | Baseline |
| Different $\beta$ ($F_0$) | 0.712 | — | Precision-oriented |
| Different $\beta$ ($F_\infty$) | 0.638 | — | Recall-oriented |

### Key Findings
- Optimal decoding consistently and significantly outperforms heuristic methods across all metrics and datasets.
- The improvements are greatest in "underdetermined scenarios" (where the model is highly uncertain and the hierarchy is deep)—this is where the suboptimality of greedy methods is most pronounced.
- Different values of $\beta$ lead to completely different optimal strategies: higher $\beta$ values tend to predict nodes higher up in the hierarchy (more conservative), whereas lower $\beta$ values tend to predict more specific leaf nodes.
- The computational overhead is acceptable (dynamic programming requires only milliseconds).

## Highlights & Insights
- The philosophy of "to each metric its own decoder" is simple yet powerful: the same model can provide optimal predictions for different user needs.
- A theory-driven, practical approach: performance is enhanced solely through better decoding, without requiring model retraining.
- The dynamic programming solution for hierarchical $F_\beta$ is elegant and fully exploits the tree structure.

## Limitations & Future Work
- It relies heavily on the calibration quality of the probabilistic model—if the probability estimates are inaccurate, the optimal decoding will not be truly optimal.
- Generalizing the approach to directed acyclic graph (DAG) hierarchies requires additional work.
- A comparative study with end-to-end training methods (that directly optimize hierarchical metrics) has not been conducted yet; subsequent research can verify if post-hoc methods can complement end-to-end training.

## Related Work & Insights
- It is related to the hierarchical classification error metrics of Deng et al. (2014).
- Post-processing methods share a similar spirit with calibration (Platt scaling, temperature scaling).
- It is directly applicable to deep hierarchical structure scenarios such as biological taxonomy and e-commerce categorization.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematic derivation of optimal hierarchical decoding rules.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple metrics and datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent combination of theory and practice.
- Value: ⭐⭐⭐⭐ A practical and theoretically grounded post-processing method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Regression for the Mean: Auto-Evaluation and Inference with Few Labels through Post-hoc Regression](regression_for_the_mean_auto-evaluation_and_inference_with_few_labels_through_po.md)
- [\[ICML 2025\] Hierarchical Refinement: Optimal Transport to Infinity and Beyond](hierarchical_refinement_optimal_transport_to_infinity_and_beyond.md)
- [\[AAAI 2026\] TaylorPODA: A Taylor Expansion-Based Method to Improve Post-Hoc Attributions for Opaque Models](../../AAAI2026/others/taylorpoda_a_taylor_expansion-based_method_to_improve_post-hoc_attributions_for_.md)
- [\[ICML 2025\] Revisiting Instance-Optimal Cluster Recovery in the Labeled Stochastic Block Model](revisiting_instance-optimal_cluster_recovery_in_the_labeled_stochastic_block_mod.md)
- [\[CVPR 2025\] Potential Field Based Deep Metric Learning](../../CVPR2025/others/potential_field_based_deep_metric_learning.md)

</div>

<!-- RELATED:END -->
