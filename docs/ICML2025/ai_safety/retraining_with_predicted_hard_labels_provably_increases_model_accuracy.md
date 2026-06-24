---
title: >-
  [Paper Note] Retraining with Predicted Hard Labels Provably Increases Model Accuracy
description: >-
  [AI Safety] In the context of noisy labels, relabeling the training set with hard labels (0/1) predicted by the model itself and retraining can **provably** improve classification accuracy. Furthermore, a consensus filtering strategy is proposed (retraining only on samples where the predicted label matches the given label), which significantly boosts performance in label-differentially private training with no extra privacy cost.
tags:
  - "AI Safety"
date: 2026-05-08
content_hash: cfdb85225c24abab
---

# Retraining with Predicted Hard Labels Provably Increases Model Accuracy

## TL;DR

In the context of noisy labels, relabeling the training set with hard labels (0/1) predicted by the model itself and retraining can **provably** improve classification accuracy. Furthermore, a consensus filtering strategy is proposed (retraining only on samples where the predicted label matches the given label), which significantly boosts performance in label-differentially private training with no extra privacy cost.

## Background & Motivation

Label Differential Privacy (Label DP) protects privacy by injecting noise into labels, but the resulting label noise significantly degrades model accuracy. An intuitively simple idea is to first train a model with noisy labels, and then retrain it by replacing the original labels with the model's own predicted hard labels. When the classes are well-separated, the model can correctly predict many mislabeled samples that lie far from the decision boundary, thereby "self-correcting" the label noise.

However, prior work lacked rigorous theoretical analysis to answer: **Does retraining indeed improve accuracy? Under what conditions is it effective?** This paper fills this theoretical gap and proposes a more practical variant called consensus-based retraining.

## Method

### Overall Architecture

The entire process consists of three steps:

1. **Initial Training**: Train the model on the noisy dataset $\{(\mathbf{x}_j, \hat{y}_j)\}_{j=1}^n$ to obtain the parameter $\hat{\boldsymbol{\theta}}_0$.
2. **Predicted Labeling**: Predict hard labels $\tilde{y}_j = \text{sign}(\langle \mathbf{x}_j, \hat{\boldsymbol{\theta}}_0 \rangle)$ for the training set using $\hat{\boldsymbol{\theta}}_0$.
3. **Retraining**: Retrain the model by replacing the noisy labels with the predicted labels to obtain the improved model $\hat{\boldsymbol{\theta}}_1$.

In step 3, the authors distinguish between two strategies:
- **Full Retraining (Full RT)**: Retrain using the predicted labels of all samples.
- **Consensus-based Retraining (Consensus-based RT)**: Retrain only on the subset of samples where the predicted label matches the given noisy label, $\mathcal{S}_{\text{cons}} = \{j : \tilde{y}_j = \hat{y}_j\}$.

The core idea of the consensus set is that when the model's prediction aligns with the noisy label, the probability of the label being correct is significantly higher than that of the overall dataset. This is equivalent to performing cross-validation with two independent "observations" to filter out a high-quality subset.

### Key Design 1: Gaussian Mixture Model with a Positive Margin

The theoretical analysis is conducted under a linearly separable binary classification setting. The data generating model is:

$$\mathbf{x} = y(1+u)\boldsymbol{\mu} + \boldsymbol{\Sigma}^{1/2}\mathbf{z}$$

where $y \in \{+1, -1\}$ is the true label, $u > 0$ is a sub-Gaussian random variable (ensuring a positive margin), and $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$. Label noise independently flips each label with probability $p < 1/2$. The separation is defined as $\gamma = \|\boldsymbol{\mu}\|_{\ell_2}$.

The classifier from the initial training takes a simplified form $\hat{\boldsymbol{\theta}}_0 = \frac{1}{n}\sum_{i=1}^n \hat{y}_i \mathbf{x}_i$, which is the feature mean weighted by the noisy labels.

### Key Design 2: Theoretical Characterization of Retraining Error

**Theorem 4.9** provides the upper bound on the population error of the retrained classifier $\hat{\boldsymbol{\theta}}_1$. The key comparison lies in the noise factor in the exponential term:

- The exponential term of the initial training error lower bound contains $(1-2p)^2$.
- The dominant term in the retraining error upper bound contains $(1-2p)$ (instead of the square).

When $p \to 1/2$ (high noise), $(1-2p)^2 \ll (1-2p)$, so the retraining error upper bound is significantly smaller than the initial training error lower bound. Specifically, when $p$ is sufficiently close to $1/2$ and the sample size satisfies:

$$\frac{d}{(1-2p)^2} \log \frac{d}{(1-2p)^2} \lesssim n \lesssim \frac{d^2}{(1-2p)^2}$$

retraining is provably superior to initial training.

The technical challenge is that the predicted labels $\tilde{y}_i$ depend on the entire training set (via $\hat{\boldsymbol{\theta}}_0$), which breaks the independence among labels. The authors decouple this dependency by constructing "dummy labels" that only depend on the projection component along the direction of $\boldsymbol{\mu}$, thereby restoring analytical tractability.

### Key Design 3: Filtering Effect of the Consensus Set

Although the consensus set $\mathcal{S}_{\text{cons}}$ is smaller, its label quality is much higher than that of the full dataset. Experiments show that (Table 3) on CIFAR-100 with $\epsilon=3$:
- Accuracy of predicted labels on the full dataset: 24.9%
- Accuracy of given noisy labels on the full dataset: 22.4%
- Accuracy of predicted labels on the consensus set: **76.1%**

The consensus set accounts for only about 11% of the full dataset, but its label accuracy is more than three times higher, achieving a "fewer but better" selection of samples.

## Key Experimental Results

### CIFAR-10/100 Label DP Experiments (ResNet-18)

| Dataset | ε | Baseline | Full RT | Consensus RT |
|--------|---|----------|---------|--------------|
| CIFAR-10 | 1 | 57.78±1.13 | 60.07±0.63 | **63.84±0.56** |
| CIFAR-10 | 2 | 79.06±0.59 | 81.34±0.40 | **83.31±0.28** |
| CIFAR-10 | 3 | 85.18±0.50 | 86.67±0.28 | **87.67±0.28** |
| CIFAR-100 | 3 | 23.53±1.01 | 24.42±1.22 | **29.98±1.11** |
| CIFAR-100 | 4 | 44.53±0.81 | 46.99±0.66 | **51.30±0.98** |
| CIFAR-100 | 5 | 55.75±0.36 | 56.98±0.43 | **59.47±0.26** |

### AG News / DomainNet Experiments

| Dataset | Model | ε | Baseline | Consensus RT | Gain |
|--------|------|---|----------|--------------|------|
| AG News | Small BERT | 0.3 | 54.54 | **65.91** | +11.4% |
| AG News | Small BERT | 0.5 | 69.21 | **80.95** | +11.7% |
| AG News | Small BERT | 0.8 | 79.10 | **84.26** | +5.2% |
| DomainNet | ResNet-50 LP | 3 | 23.60 | **36.30** | +12.7% |
| DomainNet | ResNet-50 LP | 4 | 48.25 | **57.40** | +9.2% |

## Key Findings

1. **Retraining yields higher gains under high noise**: The higher the noise ($p$ closer to $1/2$, smaller $\epsilon$), the more significant the accuracy gain from retraining, which aligns with theoretical predictions.
2. **Consensus filtering is far superior to full retraining**: In all experiments, consensus-based RT consistently outperforms full RT by a large margin, despite using only a small fraction of the training set (even less than 1/3 at low $\epsilon$).
3. **Compatibility with noise-robust methods**: Performing consensus RT on top of other noise-robust techniques such as forward correction and symmetric CE loss remains effective.
4. **Superior to confidence-based selection**: The consensus selection strategy outperforms high-confidence selection commonly used in self-training (Appendix J).
5. **Optimal sample complexity**: The sample complexity of the initial training achieves the information-theoretic lower bound in terms of $d$ and $p$.

## Highlights & Insights

- **Simple post-processing approach**: As a plug-and-play post-processing step, consensus retraining can be stacked on top of any label DP algorithm without modifying the underlying mechanisms and without expending additional privacy budget.
- **Clear theoretical intuition**: Retraining essentially transforms a uniform noise source $p$ into a non-uniform, sample-dependent noise source, which has lower noise for samples far from the decision boundary, thereby reducing the overall error.
- **Dummy label decoupling technique**: By constructing dummy labels that do not depend on the noise components of other samples, the authors cleverly resolve the complex dependency issue among predicted labels, enabling high-probability event analysis.
- **A win-win for privacy and utility**: In label DP scenarios, retraining utilizes the model's own predictions (as post-processing) and does not touch the original sensitive labels, making it entirely "free" without losing privacy guarantees.

## Limitations & Future Work

1. **Theory only covers full retraining**: Although consensus-based RT yields better experimental results, there is currently a lack of theoretical analysis for it.
2. **Theory limited to uniform label noise**: In practical scenarios, label noise is often non-uniform (instance-dependent), limiting the applicability of the theoretical results.
3. **Linear model assumption**: The theoretical analysis is based on linear classifiers and Gaussian mixture data, posing a gap with deep learning practices.
4. **Sample size upper bound may be an artifact of the analysis**: The upper bound of $n \lesssim d^2/(1-2p)^2$ restricts the scope of the theory and may not be an inherent limitation.
5. **Limited experimental scale**: The method has not been validated on larger-scale models and datasets (e.g., ImageNet, large language models).

## Related Work & Insights

- **Self-training**: Retraining is similar to the idea of self-training in semi-supervised learning. However, in the fully supervised with noisy labels scenario, it filters based on consensus rather than confidence, offering a simpler and more effective approach.
- **Self-distillation**: Self-distillation uses soft labels and temperature scaling parameters, whereas retraining uses hard labels without additional hyperparameters.
- **Label DP methods**: The prior method by Ghazi et al. (2021) is the primary baseline for the experiments in this paper, and consensus RT can be seamlessly stacked on top of it.
- **Insights**: The consensus mechanism (multi-source signal consistency check) can be generalized to other noisy learning scenarios, such as consensus filtering among multiple annotators.

## Rating

| Dimension | Score | Description |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | The first theoretical result proving the effectiveness of retraining with hard labels; the consensus selection strategy is simple yet creative. |
| Technical Depth | ⭐⭐⭐⭐⭐ | The techniques to handle the dependency of predicted labels in theoretical analysis are delicate, with sample complexity achieving the information-theoretic optimum. |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Covers both vision and language tasks, comparing multiple DP parameters and noise-robust methods, though the scale is relatively small. |
| Value | ⭐⭐⭐⭐⭐ | Plug-and-play, zero additional privacy cost, extremely simple to implement, making it highly practical in Label DP applications. |
| Writing Quality | ⭐⭐⭐⭐ | Clear motivation, superb intuitive explanation (Figure 1), with a natural transition between theory and experiments. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](../../AAAI2026/ai_safety/privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](../../AAAI2026/ai_safety/rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[AAAI 2026\] Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer Collaborative Learning](../../AAAI2026/ai_safety/learning_to_collaborate_an_orchestrated-decentralized_framework_for_peer-to-peer.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](../../AAAI2026/ai_safety/robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[AAAI 2026\] ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias](../../AAAI2026/ai_safety/problog4fairness_a_neurosymbolic_approach_to_modeling_and_mitigating_bias.md)

</div>

<!-- RELATED:END -->
