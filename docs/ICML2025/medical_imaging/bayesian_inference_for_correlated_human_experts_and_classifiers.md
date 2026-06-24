---
title: >-
  [Paper Note] Bayesian Inference for Correlated Human Experts and Classifiers
description: >-
  [ICML 2025][Medical Imaging][Human-AI Collaboration] A general Bayesian framework is proposed to model the joint labeling behavior between correlated human experts and classifiers. It captures correlations among experts using latent representations and evaluates the utility of additional queries via simulation-based inference, significantly reducing the number of expert queries in medical classification and image annotation while maintaining predictive accuracy.
tags:
  - "ICML 2025"
  - "Medical Imaging"
  - "Human-AI Collaboration"
  - "Bayesian Inference"
  - "Expert Querying"
  - "Correlation Modeling"
  - "Active Querying"
date: 2026-05-08
content_hash: cb80db65f3228da2
---

# Bayesian Inference for Correlated Human Experts and Classifiers

**Conference**: ICML 2025  
**arXiv**: [2506.05636](https://arxiv.org/abs/2506.05636)  
**Code**: [https://github.com/markellekelly/consensus](https://github.com/markellekelly/consensus)  
**Area**: Medical Images  
**Keywords**: Human-AI Collaboration, Bayesian Inference, Expert Querying, Correlation Modeling, Active Querying

## TL;DR
A general Bayesian framework is proposed to model the joint labeling behavior between correlated human experts and classifiers. It captures correlations among experts using latent representations and evaluates the utility of additional queries via simulation-based inference, significantly reducing the number of expert queries in medical classification and image annotation while maintaining predictive accuracy.

## Background & Motivation

**Background**: In practical ML applications, predictions often need to combine model outputs and human expert opinions. In domains such as medical imaging, ideally, all experts should be queried and their results aggregated, but this is prohibitively expensive.

**Limitations of Prior Work**:
   - "Learn to defer" methods choose either a human or an AI system to make a decision—but they do not handle scenarios involving querying multiple partial experts.
   - Existing prediction aggregation methods assume independence among experts—but in practice, experts are highly correlated (e.g., radiologists trained at the same medical school).
   - Prediction aggregation methods typically require comparison against a "ground truth label"—but in the setting of this paper, the expert consensus itself is the ground truth.
   - Showalter et al. is the closest work but assumes experts are exchangeable (not accounting for individual differences).

**Key Challenge**: Querying all experts is too expensive, whereas querying too few leads to inaccuracy—how to intelligently select "which experts to query" to predict the expert consensus at minimal cost?

**Goal**: Minimize the number of human queries while accurately predicting the consensus label of the expert group, with the assistance of a pre-trained classifier.

**Key Insight**: Bayesian generative models + joint latent representations—modeling the joint distribution of classifier predictions and expert votes, allowing the inference of unobserved expert votes from partial observations.

**Core Idea**: Each sample has a latent "intrinsic difficulty/class-specific feature" $\rightarrow$ the classifier output and each expert's vote are noisy observations of this latent feature $\rightarrow$ after observing a subset of experts, the remaining experts' votes can be inferred using Bayesian reasoning.

## Method

### Overall Architecture
Online query decision pipeline:
1. A new sample $x$ arrives $\rightarrow$ obtain the classifier's probability output $p(y|x)$ (free/low cost).
2. Based on the classifier output, the Bayesian model estimates: is it necessary to query a human expert? Which expert's query yields the maximum information gain?
3. Query the selected expert $\rightarrow$ update the posterior $\rightarrow$ decide whether to continue querying.
4. Output the predicted consensus label once the confidence threshold is met.

### Key Designs

1. **Joint Latent Representation Model**:

    - **Function**: Models the joint labeling behavior of the classifier and $H$ experts.
    - **Mechanism**: The latent variable $z$ encodes the "intrinsic difficulty and class-specific features" of the sample. The classifier probability $p_c$ and expert votes $y_i$ are conditionally independent given $z$: $p(p_c, y_1, ..., y_H | z)$.
    - **Correlation modeling**: Through the shared $z$—even though conditionally independent, the experts are marginally correlated.
    - **Parameterization**: Parameterize $p(y_i | z)$ using mixture distributions or Dirichlet-Multinomial models.
    - **Design Motivation**: More realistic than assuming expert independence—doctors with the same training background exhibit similar error patterns on similar cases.

2. **Simulation-Based Query Utility Inference**:

    - **Function**: Evaluates the information gain of querying an additional expert $i$ for predicting the consensus.
    - **Mechanism**: Monte Carlo simulation—sample potential votes of expert $i$ from the current posterior $\rightarrow$ compute the expected accuracy gain after querying.
    - **Greedy Strategy**: Query the expert with the maximum expected information gain at each step.
    - **Stopping Condition**: Querying stops when the confidence in predicting the consensus exceeds a threshold.
    - **Design Motivation**: Avoids treating all experts equally—different experts provide different amounts of information for different types of samples.

3. **Online Bayesian Update**:

    - **Function**: Updates model parameters online as new samples accumulate.
    - **Mechanism**: For each new sample, the observed expert votes are used to update the parameters of the joint model $\rightarrow$ the model's estimation of expert correlations becomes increasingly accurate.
    - **Design Motivation**: Eliminates the need to collect a large amount of annotated data beforehand—the model learns progressively from the sequence.

### Loss & Training
- Bayesian inference—no traditional loss functions.
- Prior: Dirichlet prior on the expert confusion matrix.
- Posterior Update: Given a new (classifier output, expert vote) pair.
- Online learning—parameters are updated once for each new sample.

## Key Experimental Results

### Main Results
Medical Image Classification (Skin Lesion + Chest X-ray):

| Method | Avg. Queries/Sample ↓ | Consensus Prediction Accuracy ↑ |
|------|-------------|-------------|
| Query All (All 5 experts) | 5.0 | 100% (By definition) |
| Random Query 3 | 3.0 | 89.2% |
| Independent Expert Model | 2.8 | 90.5% |
| Showalter (Exchangeable) | 2.4 | 92.1% |
| **Ours (Correlation Modeling)** | **1.8** | **95.3%** |

### CIFAR-10H / ImageNet-16H

| Dataset | Ours Queries ↓ | Independent Model Queries | Accuracy Gain |
|--------|-----------|-----------|---------|
| CIFAR-10H | 2.1 | 3.2 | +3.5% |
| ImageNet-16H | 2.5 | 3.8 | +4.2% |

### Ablation Study

| Configuration | Queries | Accuracy | Description |
|------|--------|--------|------|
| No Classifier (Experts only) | 2.8 | 93.1% | Classifier provides initial information |
| No Correlation Modeling | 2.4 | 92.1% | Degenerates to independence assumption |
| **Full Model** | **1.8** | **95.3%** | Correlation + Classifier + Selective Querying |
| Random Query Strategy | 2.5 | 91.8% | Underperforms compared to information-gain selection |
| **Info-Gain Query** | **1.8** | **95.3%** | Selects the most informative expert |
| 1 Expert + Classifier | 1.0 | 87.5% | Insufficient accuracy |
| 2 Selected Experts + Classifier | 2.0 | 95.1% | Close to the full method |

### Key Findings
- Correlation modeling reduces the number of queries from 2.4 (independence assumption) to 1.8—leveraging redundant information provided by correlated experts.
- The classifier provides "free initial information"—reducing the number of expert queries by approximately 0.6.
- The information-gain selection strategy significantly outperforms random selection—the informativeness of different experts varies greatly across different samples.
- Online learning converges rapidly—the model's estimation of expert relationships becomes sufficiently accurate after roughly 50-100 samples.
- In skin lesion classification, some "hard samples" require 4-5 experts, but most samples only require 1-2.

## Highlights & Insights
- **"Predicting the expert consensus rather than the ground truth label"**—the problem formulation itself is highly novel. Instead of attempting to outperform the experts, it focuses on efficiently simulating the expert group.
- The latent variable model elegantly captures expert correlation—achieving conditional independence but marginal correlation through a shared latent variable.
- Bayesian uncertainty naturally drives query decisions—the model queries experts only when it "knows what it does not know."
- The method is agnostic to the classifier—any black-box pre-trained model can be utilized.
- Direct practical value for medical AI deployment under resource constraints.

## Limitations & Future Work
- Latent model assumptions (e.g., Dirichlet-Multinomial) might not suit all patterns of expert behaviors.
- Assumes that experts are always available and identifiable—scenarios with anonymous or changing experts are not addressed.
- The greedy information gain strategy might not lead to a globally optimal query sequence.
- Currently handles only classification tasks—generalization to regression or structured prediction remains to be explored.
- Sample efficiency might degrade when the number of experts is very large (>10).

## Related Work & Insights
- **vs Learn to Defer**: Selects either a human or an AI, without handling partial queries; this work flexibly queries subsets.
- **vs Kim & Ghahramani (2012)**: Models the relationship between the classifier and the true label; this work models relationships among experts.
- **vs Showalter et al.**: Assumes experts are exchangeable; this work models individual differences and correlation.
- **Insight**: The question of "whom to ask and when" in human-AI collaboration has potential application value in broader domains (e.g., law, auditing, content moderation).

## Rating
- Novelty: ⭐⭐⭐⭐ Novel problem formulation, elegant Bayesian framework
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparisons across medicine, CIFAR, and ImageNet against multiple baselines
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition
- Value: ⭐⭐⭐⭐⭐ High practical value for resource-constrained scenarios like medical AI

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts](i2moe_interpretable_multimodal_interaction-aware_mixture-of-experts.md)
- [\[CVPR 2025\] SapiensID: Foundation for Human Recognition](../../CVPR2025/medical_imaging/sapiensid_foundation_for_human_recognition.md)
- [\[NeurIPS 2025\] The Human Brain as a Combinatorial Complex](../../NeurIPS2025/medical_imaging/the_human_brain_as_a_combinatorial_complex.md)
- [\[ICML 2025\] Enhancing Statistical Validity and Power in Hybrid Controlled Trials: A Randomization Inference Approach with Conformal Selective Borrowing](enhancing_statistical_validity_and_power_in_hybrid_controlled_trials_a_randomiza.md)
- [\[NeurIPS 2025\] Multimodal Bayesian Network for Robust Assessment of Casualties in Autonomous Triage](../../NeurIPS2025/medical_imaging/multimodal_bayesian_network_for_robust_assessment_of_casualties_in_autonomous_tr.md)

</div>

<!-- RELATED:END -->
