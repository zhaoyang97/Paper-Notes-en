---
title: >-
  [Paper Note] Machine Learning from Explanations
description: >-
  [ICML2025][LLM Pretraining][Explanation-guided learning] Proposes a method to guide machine learning using simple explanation signals (important input features). By employing a two-stage training loop that alternately optimizes prediction accuracy and attention alignment, this method significantly improves performance and stability in scenarios with small data, class imbalance, and spurious features.
tags:
  - "ICML2025"
  - "LLM Pretraining"
  - "Explanation-guided learning"
  - "Small-data learning"
  - "Spurious correlation"
  - "Attention alignment"
  - "Sample efficiency"
date: 2026-05-08
content_hash: 902ee05f8fdb2a56
---

# Machine Learning from Explanations

**Conference**: ICML2025  
**arXiv**: [2507.04788](https://arxiv.org/abs/2507.04788)  
**Code**: TBD  
**Area**: Robotics  
**Keywords**: Explanation-guided learning, Small-data learning, Spurious correlation, Attention alignment, Sample efficiency

## TL;DR
Proposes a method to guide machine learning using simple explanation signals (important input features). By employing a two-stage training loop that alternately optimizes prediction accuracy and attention alignment, this method significantly improves performance and stability in scenarios with small data, class imbalance, and spurious features.

## Background & Motivation

### Fundamental Dilemma of Small Data
1. Identical data combined with the same algorithm can yield models with similar accuracies but entirely different decision functions.
2. Models tend to learn spurious correlations (e.g., background instead of the target object).
3. Samples from minority classes are insufficient to learn the correct reasons for classification.

### Why More Data Alone is Insufficient
- Acquiring large amounts of high-quality annotated data is expensive.
- Certain domains (e.g., rare disease diagnosis, defect detection in new products) are inherently data-scarce.
- More data does not guarantee the elimination of ambiguity in decision functions.

### Difference from Existing Methods
Existing methods (e.g., Ross et al. 2017) penalize the model for learning "wrong reasons" but do not ensure the model learns "correct reasons." This work directly provides correct explanations and guides the learning process.

## Method

### Two-Stage Alternating Training Loop
**Stage 1: Optimize Prediction Accuracy** (standard label loss)  
**Stage 2: Align Attention with Explanations** (explanation alignment loss)  

The optimization alternates until convergence, rather than relying on a simple weighted combination.

### Form of Explanations
Simplest form: A binary mask of input features, marking which features are the key reasons for the label.

### Essential Difference from Joint Loss Methods
- Joint loss only penalizes "wrong reasons" without guiding toward "correct reasons".
- This work ensures the model satisfies both objectives simultaneously through alternating optimization.
- Experiments demonstrate that joint loss fails to learn correct reasons even on simple tasks.

## Key Experimental Results

### Spurious Feature Scenarios

| Method | Accuracy with Spurious Features | Accuracy without Spurious Features | Learned Correct Reason |
|------|---------------|---------------|-----------|
| Standard Training | High (using spurious features) | Low | No |
| Ross et al. (Penalty) | Medium | Medium | Partial |
| **Ours (Explanation Guided)** | **High** | **High** | **Yes** |

### Small-Data Scenarios

| Number of Training Samples | Standard Training | Penalty Method | Ours |
|-----------|---------|---------|---------|
| 50 | 62% | 65% | **78%** |
| 100 | 71% | 73% | **85%** |
| 500 | 82% | 83% | **90%** |

### Key Findings
1. Explanation guidance significantly accelerates convergence (reaching higher accuracy in fewer epochs).
2. Model stability is vastly improved (reduced variance across different runs).
3. The improvement on minority classes is most pronounced in class-imbalanced scenarios.
4. Joint loss methods fail even on minimalist geometric shape detection, whereas our method succeeds.

## Highlights & Insights

1. "Providing the correct reason" is more effective than "penalizing the wrong reason"—an intuitively plausible finding systematically proven by experiments for the first time.
2. Alternating optimization is more stable than joint loss, avoiding the need for hyperparameter tuning to balance the two objectives.
3. The form of explanation is extremely simple (binary mask), resulting in low acquisition costs.
4. The counterexample where joint loss fails even on minimalist tasks is highly convincing.
5. This has important practical guidance implications for trustworthy AI deployment.

## Limitations & Future Work

1. Requires human-provided explanations, which increases annotation costs.
2. Only validated on image classification; other modalities such as text/tabular data remain to be explored.
3. Explanation quality directly affects effectiveness, but how to evaluate explanation quality is not discussed in depth.
4. The diminishing returns of the proposed method in large-data scenarios are not quantified.
5. Comparisons with other knowledge injection methods (e.g., curriculum learning, knowledge distillation) are insufficient.

## Related Work & Insights

- The core difference from Ross et al. is "guidance" vs. "penalty".
- Difference from Rieger et al. (RRR): RRR uses bounding boxes, whereas ours uses feature masks.
- Insight: Explanation signals can be automatically generated using LLMs/GPT-4V, reducing manual annotation costs.

## Rating
- Novelty: 4.0/5 — Conceptually simple but clearly distinguished from existing methods
- Experimental Thoroughness: 4.5/5 — Multiple scenarios + counterexamples + ablations
- Writing Quality: 4.5/5 — Clear problem definition
- Value: 4.5/5 — Direct significance for trustworthy AI deployment

## Supplementary Analysis

### Insights from Minimalist Counterexamples
On the task of detecting simple geometric shapes, joint loss methods fail to learn the correct regions, while alternating training converges with only 50 samples.

### Cost of Obtaining Explanations
In expert domains such as medicine, the marginal cost of annotating explanations is far lower than acquiring new samples. This can be automatically generated using LLMs/VLMs in the future.

### Relationship with Curriculum Learning
Explanation guidance can be viewed as a more precise form of curriculum learning: it not only tells the model what to learn but also why.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Compact Example-Based Explanations for Language Models](../../ACL2026/llm_pretraining/compact_example-based_explanations_for_language_models.md)
- [\[ICML 2025\] Density Ratio Estimation-based Bayesian Optimization with Semi-Supervised Learning](density_ratio_estimation-based_bayesian_optimization_with_semi-supervised_learni.md)
- [\[ICML 2025\] When Can In-Context Learning Generalize Out of Task Distribution?](when_can_in-context_learning_generalize_out_of_task_distribution.md)
- [\[ICML 2025\] On the Role of Label Noise in the Feature Learning Process](on_the_role_of_label_noise_in_the_feature_learning_process.md)
- [\[ICML 2025\] Algebra Unveils Deep Learning -- An Invitation to Neuroalgebraic Geometry](algebra_unveils_deep_learning_--_an_invitation_to_neuroalgebraic_geometry.md)

</div>

<!-- RELATED:END -->
