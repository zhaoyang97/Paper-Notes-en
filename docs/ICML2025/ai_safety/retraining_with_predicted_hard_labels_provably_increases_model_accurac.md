---
title: >-
  [Paper Note] Retraining with Predicted Hard Labels Provably Increases Model Accuracy
description: >-
  [ICML 2025][AI Safety][label noise] Under noisy labels, retraining a model on a training set relabeled with its own predicted hard labels ($0/1$ labels) can **provably** increase model accuracy. Furthermore, this study proposes consensus-based retraining (retraining only on samples where the predicted labels match the given labels), which significantly improves performance with zero additional privacy cost under label DP scenarios.
tags:
  - "ICML 2025"
  - "AI Safety"
  - "label noise"
  - "label differential privacy"
  - "retraining"
  - "noisy labels"
  - "self-training"
  - "theoretical analysis"
date: 2026-05-08
content_hash: da9fe7936715be39
---

# Retraining with Predicted Hard Labels Provably Increases Model Accuracy

**Conference**: ICML 2025  
**arXiv**: [2406.11206](https://arxiv.org/abs/2406.11206)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: label noise, label differential privacy, retraining, noisy labels, self-training, theoretical analysis  

## TL;DR

Under noisy labels, retraining a model on a training set relabeled with its own predicted hard labels ($0/1$ labels) can **provably** increase model accuracy. Furthermore, this study proposes consensus-based retraining (retraining only on samples where the predicted labels match the given labels), which significantly improves performance with zero additional privacy cost under label DP scenarios.

## Background & Motivation

Label Differential Privacy (Label DP) requires protecting label privacy in supervised learning, which is typically achieved by injecting noise into the labels. However, the label quality degrades significantly after noise injection, leading to severe deterioration in model performance. While prior works have extensively leveraged the model's own predictions via self-training and self-distillation, there is a lack of theoretical analysis proving that "retraining with the model's own predicted **hard labels** can improve accuracy under label noise."

The core observation of this work is that when there is sufficient separation between classes, the model can still make correct predictions on samples far from the decision boundary, despite being trained on noisy labels. Consequently, the model's prediction accuracy on the training set can be significantly higher than the accuracy of the given noisy labels. Retraining with these more accurate predicted labels is capable of improving performance. This is intuitively demonstrated in Figure 1 of the paper—retraining is effective when the class separation is large (test accuracy increases from 89% to 97.67%), whereas no improvement is observed when the separation is small.

## Method

### Overall Architecture

The proposed method consists of two levels:

1. **Full Retraining**: Generate predicted hard labels $\tilde{y}_j = \text{sign}(\langle \boldsymbol{x}_j, \hat{\boldsymbol{\theta}}_0 \rangle)$ for all $n$ training samples using the initial model $\hat{\boldsymbol{\theta}}_0$, and then retrain the model on $\{(\boldsymbol{x}_j, \tilde{y}_j)\}_{j=1}^n$.

2. **Consensus-based Retraining**: Define the consensus set $\mathcal{S}_{\text{cons}} = \{j \mid \tilde{y}_j = \hat{y}_j\}$, which is the subset of samples where the predicted labels match the given noisy labels. Retraining is performed solely on the consensus set $\{(\boldsymbol{x}_j, \tilde{y}_j)\}_{j \in \mathcal{S}_{\text{cons}}}$.

Core intuition: Although the consensus set is smaller, its label accuracy is much higher than that of the full dataset. For instance, on CIFAR-100 with $\epsilon=3$, the consensus set constitutes only 11% of the full dataset, yet its label accuracy reaches up to 76.09%, compared to the predicted label accuracy of the full dataset at only 24.90%.

### Key Design 1: Theoretical Analysis Framework—Gaussian Mixture Model with Positive Margin

The theoretical analysis is conducted under a linearly separable binary classification setting. The data generating model is a Gaussian Mixture Model with a positive margin:

$$\boldsymbol{x} = y(1+u)\boldsymbol{\mu} + \boldsymbol{\Sigma}^{1/2}\boldsymbol{z}$$

where $y \in \{+1, -1\}$ is the true label, $u > 0$ is a sub-gaussian random variable (ensuring a positive margin), $\boldsymbol{z} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}_d)$, and $\gamma = \|\boldsymbol{\mu}\|_{\ell_2}$ represents the inter-class separation. The given noisy label is independently flipped with probability $p < 1/2$: $\hat{y}_i = y_i$ (with probability $1-p$) or $\hat{y}_i = -y_i$ (with probability $p$).

The initial classifier is defined as $\hat{\boldsymbol{\theta}}_0 = \frac{1}{n}\sum_i \hat{y}_i \boldsymbol{x}_i$, and the retrained classifier is $\hat{\boldsymbol{\theta}}_1 = \frac{1}{n}\sum_i \tilde{y}_i \boldsymbol{x}_i$.

### Key Design 2: Theoretical Guarantees for Accuracy Improvement via Retraining

**Core Theorem (Remark 4.10)**: When $p$ is close to $1/2$ (high noise level), and the number of samples satisfies:

$$\frac{\lambda_{\min}^2 d}{\gamma^4(1-2p)^2} \log\frac{\lambda_{\min}^2 d}{\gamma^4(1-2p)^2} \lesssim n \lesssim \frac{\lambda_{\min}^2 d^2}{\gamma^4(1-2p)^2}$$

then the accuracy of the retrained classifier is strictly superior to that of the initial classifier, i.e., $\text{acc}(\hat{\boldsymbol{\theta}}_1) > \text{acc}(\hat{\boldsymbol{\theta}}_0)$.

Key mechanism: The lower bound of the initial classifier's error contains $(1-2p)^2$ in the exponent, whereas the upper bound of the retrained classifier's error contains an effective noise rate $q' = \exp(-\frac{n(1-2p)\gamma^2}{40\lambda_{\max}})$ in the exponent. When $n$ is sufficiently large, $q' \ll p$, which means the effective noise of the retrained model is far smaller than the original noise. Intuitively, the model's predicted labels are more accurate than the given noisy labels (especially for samples far from the boundary), making retraining equivalent to learning under reduced noise.

The technical challenge in the proof lies in the fact that the predicted label $\tilde{y}_i$ depends on the entire training set (non-independent), and the noise is non-uniform and sample-dependent. The authors decouple this dependency by constructing "dummy labels."

### Key Design 3: Zero Additional Privacy Cost under Label DP

Since retraining only employs the model's own predicted labels and the already published noisy labels without accessing any raw true labels, it constitutes a **post-processing** step of the label DP mechanism. Under the post-processing property of differential privacy, this incurs no additional privacy cost. This implies that consensus-based retraining can be combined with any label DP algorithm.

## Key Experimental Results

### Table 1: CIFAR-10 Test Accuracy (ResNet-18)

| $\epsilon$ | Baseline | Full RT | Consensus RT |
|:---:|:---:|:---:|:---:|
| 1 | 57.78 ± 1.13 | 60.07 ± 0.63 | **63.84 ± 0.56** |
| 2 | 79.06 ± 0.59 | 81.34 ± 0.40 | **83.31 ± 0.28** |
| 3 | 85.18 ± 0.50 | 86.67 ± 0.28 | **87.67 ± 0.28** |

### Table 2: CIFAR-100 Test Accuracy (ResNet-18)

| $\epsilon$ | Baseline | Full RT | Consensus RT |
|:---:|:---:|:---:|:---:|
| 3 | 23.53 ± 1.01 | 24.42 ± 1.22 | **29.98 ± 1.11** |
| 4 | 44.53 ± 0.81 | 46.99 ± 0.66 | **51.30 ± 0.98** |
| 5 | 55.75 ± 0.36 | 56.98 ± 0.43 | **59.47 ± 0.26** |

### Table 3: Label Filtering Effect of the Consensus Set (CIFAR-100)

| $\epsilon$ | Predicted Label (Full Set) Accuracy | Given Label (Full Set) Accuracy | Predicted Label (Consensus Set) Accuracy |
|:---:|:---:|:---:|:---:|
| 3 | 24.90% | 22.35% | **76.09%** |
| 4 | 50.85% | 46.32% | **91.59%** |
| 5 | 66.51% | 68.09% | **94.83%** |

### Table 4: AG News Subset (Small BERT)

| $\epsilon$ | Baseline | Full RT | Consensus RT |
|:---:|:---:|:---:|:---:|
| 0.3 | 54.54 ± 0.97 | 60.03 ± 2.90 | **65.91 ± 1.93** |
| 0.5 | 69.21 ± 0.31 | 75.63 ± 1.08 | **80.95 ± 1.47** |
| 0.8 | 79.10 ± 1.43 | 82.19 ± 1.54 | **84.26 ± 1.03** |

On AG News with $\epsilon=0.5$, consensus RT yields an 11.7% improvement over the baseline, with the consensus set constituting only 32% of the training set.

## Key Findings

1. **Retraining Provably Improves Accuracy**: Under a linearly separable setting, this work provides the first proof that retraining with predicted hard labels reduces population error. The performance gain is more pronounced when the noise is larger ($p$ is closer to $1/2$) or the separation is greater ($\gamma$ is larger).
2. **Consensus Set is an Extremely Efficient Filtering Mechanism**: Although the consensus set makes up only a small fraction of the training set (e.g., only 11% for CIFAR-100 when $\epsilon=3$), its label accuracy jumps from approximately 25% to 76%. This is the fundamental reason why consensus RT significantly outperforms full RT.
3. **Consistent Gains Over Noise-Robust Methods**: Even when the initial training already employs noise-robust techniques such as forward correction or symmetric CE, consensus RT can still yield further performance improvements.
4. **Generalization Across Modalities and Architectures**: The method is effective across both computer vision (CIFAR-10/100, DomainNet / ResNet-18/34/50) and NLP (AG News / BERT).

## Highlights & Insights

- **Simple Yet Effective**: Consensus-based retraining does not require any additional unsupervised or semi-supervised learning methods; it achieves significant improvements solely by filtering based on the consistency between the model's own predictions and the given labels. This "less is more" philosophy is highly noteworthy.
- **Bridge Between Theory and Practice**: Although the linear analysis is simplified, it accurately captures the core mechanism—the effective noise rate is reduced from $p$ to $q' \approx \exp(-\Theta(n))$. This exponential noise reduction is key to the effectiveness of retraining.
- **"Double Filtering" Effect of the Consensus Set**: The model's predictions are more accurate on samples far from the boundary, while consensus filtering further removes uncertain samples near the boundary. The combination of these two filtering levels yields a small yet clean subset.
- **Key Difference from Self-Training**: Self-training selects samples based on model confidence in a semi-supervised setting, whereas this work selects samples based on predicted-to-given label consistency in a fully supervised noisy label setting, with the latter yielding superior performance (as verified in Appendix J).

## Limitations & Future Work

1. **Theory Only Covers Full Retraining**: Although consensus retraining delivers the best empirical results, a theoretical analysis is currently lacking.
2. **Theory Plagued by Uniform Label Noise Assumption**: In practice, label noise is often instance-dependent, which is not yet covered by the existing theory.
3. **Theory Limited to Linearly Separable Settings**: There are currently no theoretical guarantees for generalization to deep networks and non-linear classification.
4. **Upper Bound Constraint on Sample Size**: The upper bound $n \lesssim d^2/(1-2p)^2$ might be an artifact of the analysis, but it cannot be eliminated within the current proof framework.
5. **Lack of Validation on Large-Scale Models and Datasets**: The experimental scale is limited (ResNet-18/34, CIFAR-10/100) and does not cover large-scale model scenarios.

## Related Work & Insights

- **Self-training** (Scudder 1965; Lee et al. 2013): Iteratively trains models using predicted labels in a semi-supervised setting, but existing theories do not cover noisy label scenarios. This work represents the first theoretical result in a fully supervised noisy label setting.
- **Self-distillation** (Furlanello et al. 2018; Das & Sanghavi 2023): Trains student models using the **soft labels** of a teacher model. This work focuses on **hard labels** and does not use a temperature parameter, allowing for a more straightforward analysis.
- **Label DP** (Ghazi et al. 2021): A core method for label privacy protection. The consensus RT proposed in this work can serve as a plug-and-play post-processing module for any label DP algorithm.
- **Inspiration**: The filtering philosophy based on "prediction-to-given consistency" can be extended to other noisy learning scenarios (such as crowdsourcing labels and weak supervision) and is not restricted to DP scenarios.

## Rating

| Dimension | Score | Description |
|:---|:---:|:---|
| Novelty | 7 | First theoretical proof that hard-label retraining is effective under noise; consensus filtering is intuitive but has not been previously validated theoretically or through systematic experiments. |
| Technical Depth | 8 | The theoretical analysis tackling non-independent, non-uniform noise in predicted labels is challenging; the dummy labels decoupling technique is elegant. |
| Experimental Thoroughness | 7 | Covers computer vision and NLP, various architectures, and DP parameters, but is limited in scale and lacks large-scale model experiments. |
| Writing Quality | 8 | Theory and experiments are organized clearly, intuitive explanations are well-presented (Figure 1 is great), and core observations are straightforward. |
| Value | 8 | The method is extremely simple, has zero additional privacy cost, and is plug-and-play, holding direct significance for promoting label DP practices. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Improving the Variance of Differentially Private Randomized Experiments through Clustering](improving_the_variance_of_differentially_private_randomized_experiments_through_.md)
- [\[ICML 2025\] Private Model Personalization Revisited](private_model_personalization_revisited.md)
- [\[ICML 2025\] Understanding Model Ensemble in Transferable Adversarial Attack](understanding_model_ensemble_in_transferable_adversarial_attack.md)
- [\[ICML 2025\] Rethinking the Bias of Foundation Model under Long-tailed Distribution](rethinking_the_bias_of_foundation_model_under_long-tailed_distribution.md)
- [\[ICML 2025\] Relative Error Fair Clustering in the Weak-Strong Oracle Model](relative_error_fair_clustering_in_the_weak-strong_oracle_model.md)

</div>

<!-- RELATED:END -->
