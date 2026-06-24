---
title: >-
  [Paper Note] Instance-dependent Noisy-label Learning with Graphical Model Based Noise-rate Estimation
description: >-
  [ECCV 2024][LLM Evaluation][Noisy-label learning] This paper proposes a noise-rate estimation method based on a probabilistic graphical model, which automatically estimates the label noise rate of the training set. The estimated values are used to guide the curriculum design of sample selection strategies. It can be seamlessly integrated into state-of-the-art (SOTA) noisy-label learning methods such as DivideMix and InstanceGM, improving their classification accuracy on both…
tags:
  - "ECCV 2024"
  - "LLM Evaluation"
  - "Noisy-label learning"
  - "Instance-dependent noise"
  - "Noise-rate estimation"
  - "Probabilistic graphical models"
  - "Sample selection"
date: 2026-05-08
content_hash: e5b42cd6e68bc912
---

# Instance-dependent Noisy-label Learning with Graphical Model Based Noise-rate Estimation

**Conference**: ECCV 2024  
**arXiv**: [2305.19486](https://arxiv.org/abs/2305.19486)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Noisy-label learning, Instance-dependent noise, Noise-rate estimation, Probabilistic graphical models, Sample selection

## TL;DR

This paper proposes a noise-rate estimation method based on a probabilistic graphical model, which automatically estimates the label noise rate of the training set. The estimated values are used to guide the curriculum design of sample selection strategies. It can be seamlessly integrated into state-of-the-art (SOTA) noisy-label learning methods such as DivideMix and InstanceGM, improving their classification accuracy on both synthetic and real-world benchmarks.

## Background & Motivation

**Background**: Due to their high-capacity nature, deep learning models are highly prone to overfitting under noisy labels, especially instance-dependent noise (IDN)—which is caused by the ambiguity of the samples themselves (e.g., morphologically similar cats and dogs being mislabeled). IDN is the most realistic and challenging type of noise. Currently, the most successful IDN learning methods typically include a sample selection stage that divides training samples into "clean" and "noisy" groups.

**Limitations of Prior Work**: (1) Sample selection relies on a "curriculum" function $R(t)$, which defines the proportion of samples classified as clean in each training epoch. Existing curricula are either predefined fixed functions (such as the linear decay in Co-teaching) or based on arbitrary clustering thresholds (such as GMM fitting in DivideMix), neither of which considers the actual noise rate of the training set. (2) If the curriculum selects too many samples as clean (which actually contain noise), it leads to overfitting; if it selects too few (discarding clean samples), it leads to underfitting. (3) Label transition matrix estimation methods attempt to recover pairwise label transition probabilities, but they are unstable under high noise rates and a large number of classes, and they follow a different technical route compared to sample selection methods.

**Key Challenge**: The quality of the sample selection curriculum directly determines the performance of noisy-label learning, yet existing curriculum designs do not utilize the natural signal available from the training set—the noise rate. As shown in Figure 1, when replacing the sample selection in DivideMix with a fixed-ratio selection based on the true noise rate $\epsilon=50\%$, the accuracy improves by approximately 6%. This demonstrates that noise-rate information is crucial for sample selection.

**Key Insight**: Designing a probabilistic graphical model to simultaneously estimate the noise rate $\epsilon$ and training model parameters during the training process, and using the estimated noise rate to construct a more effective sample selection curriculum. This method can be integrated as a plugin into any SOTA method based on sample selection.

**Core Idea**: Automatically estimating the label noise rate from training data through a probabilistic graphical model, and replacing the predefined curriculum with the estimated value to guide sample selection, thereby enhancing the effectiveness of existing noisy-label learning methods.

## Method

### Overall Architecture

The overall architecture is divided into two mutually coupled components: (1) Probabilistic Graphical Model—modeling the generation process of noisy labels, and iteratively estimating the noise rate $\epsilon$, the clean label classifier $\theta_y$, and the noisy label classifier $\theta_{\hat{y}}$ via the EM algorithm. (2) Sample Selection and Downstream Model Training—utilizing the estimated noise rate $\epsilon^{(t)}$ to construct the curriculum $R(t) = 1 - \epsilon^{(t)}$, guiding the sample selection process of SOTA noisy-label learning methods (e.g., DivideMix). The two components are jointly optimized during training.

### Key Designs

1. **Probabilistic Graphical Model for Noisy Label Generation**:

    - **Function**: Modeling the generation process from data to noisy labels to estimate the global noise rate.
    - **Mechanism**: Modeling the generation of noisy labels as a three-step process—(1) sample data $x \sim p(X)$; (2) sample clean label $y \sim \text{Cat}(Y; f_{\theta_y}(x))$; (3) sample noisy label $\hat{y} \sim \text{Cat}(\hat{Y}; \epsilon \cdot f_{\theta_{\hat{y}}}(x) + (1-\epsilon) \cdot y)$, where $\epsilon$ represents the global noise rate. All parameters are iteratively estimated via the EM algorithm by maximizing the log-likelihood $\max_{\theta_y, \theta_{\hat{y}}, \epsilon} \mathbb{E}_{(x_i, \hat{y}_i) \sim \mathcal{D}} [\ln p(\hat{y}_i | x_i; \theta_y, \theta_{\hat{y}}, \epsilon)]$.
    - **Design Motivation**: Treating the noise rate as a learnable parameter of the graphical model allows it to be automatically inferred from data without manual setting. The E-step of EM estimates the posterior of clean labels, and the M-step updates the model parameters and the noise rate.

2. **Noise-rate-based Sample Selection Curriculum**:

    - **Function**: Constructing a more precise division standard for clean/noisy samples.
    - **Mechanism**: The curriculum function is defined as $R(t) = 1 - \epsilon^{(t)}$. Specifically, in the $t$-th training epoch, the $\lfloor R(t) \times N \rfloor$ samples with the smallest sorted loss are treated as clean samples, while the rest are treated as noisy. The sorting criterion can be the loss value (DivideMix), the distance to the dominant eigenvector in the feature space (FINE), or the KNN score (SSR). Unlike predefined curricula, $R(t)$ dynamically changes with the estimated noise rate.
    - **Design Motivation**: Ablation experiments (Figure 1a) show that a fixed selection using the correct noise rate $\epsilon=0.5$ improves performance by ~6% compared to the original DivideMix scheme. However, since the true noise rate is unknown, it needs to be estimated. The estimation process also addresses the identifiability problem by constraining the clean label classifier.

3. **Seamless Integration with SOTA Methods**:

    - **Function**: Serving as a general plugin to enhance any noisy-label learning method based on sample selection.
    - **Mechanism**: The clean label classifier of the SOTA method is used as $f_{\theta_y}$ in the graphical model, keeping its original architecture and hyperparameters unchanged. In the M-step, an additional sample selection constraint term $L(\theta_y, \epsilon^{(t)})$ (i.e., cross-entropy loss based on the estimated noise rate) is introduced to be jointly optimized with the original graphical model objective: $\theta_y^{(t+1)}, \theta_{\hat{y}}^{(t+1)}, \epsilon^{(t+1)} = \arg\max Q(\cdot) - \lambda L(\theta_y, \epsilon^{(t)})$. The hyperparameter is set as $\lambda = 1$.
    - **Design Motivation**: Designed in a plug-and-play manner without altering the core architecture of the baseline methods, minimizing integration costs. It has been successfully integrated into six methods, including DivideMix, C2D, InstanceGM, FINE, SSR, and CC.

### Loss & Training

The total loss consists of two parts: (1) the ELBO maximization objective of the probabilistic graphical model (including the posterior estimation of clean labels and the likelihood of noisy labels); (2) the intrinsic loss of the SOTA method (such as the semi-supervised learning loss of DivideMix), with sample selection guided by the estimated noise rate. During training, the clean label classifier is first warmed up, followed by joint training of the graphical model and the downstream classifier. $\epsilon$ is implemented via a learnable parameter with a sigmoid activation function, optimized using SGD.

## Key Experimental Results

### Main Results

IDN noise experiments on CIFAR-100:

| Method | Noise Rate 0.2 | Noise Rate 0.3 | Noise Rate 0.4 | Noise Rate 0.5 |
|------|----------|----------|----------|----------|
| DivideMix | 77.03 | 76.33 | 70.80 | 58.61 |
| DivideMix + Ours | **77.42** | **77.21** | **72.41** | **64.02** |
| InstanceGM | 79.69 | 79.21 | 78.47 | 77.19 |
| InstanceGM + Ours | 79.61 | **79.40** | **79.52** | **77.76** |

Real-world noise experiments on red mini-ImageNet:

| Method | Noise Rate 0.4 | Noise Rate 0.6 | Noise Rate 0.8 |
|------|----------|----------|----------|
| DivideMix | 46.72 | 43.14 | 34.50 |
| DivideMix + Ours | **50.70** | **45.11** | **37.44** |
| InstanceGM | 52.24 | 47.96 | 39.62 |
| InstanceGM + Ours | **56.61** | **51.40** | **43.83** |

### Ablation Study

| Configuration | CIFAR-100 IDN 0.5 (Acc%) | Description |
|------|-------------------------|------|
| Original DivideMix | 58.61 | Baseline, using GMM sample selection |
| DivideMix + Oracle Noise Rate ($\epsilon$=0.5) | 64.44 | Upper bound, assuming true noise rate is known |
| Graphical Model + Pre-trained DivideMix | 52.31 | Poor performance without joint training |
| Joint training but without using estimated $\epsilon$ for selection | 56.30 | Noise-rate estimation is crucial for selection |
| DivideMix + Ours (Full Method) | 64.02 | Close to the oracle upper bound |

### Key Findings

- The full method (64.02%) is very close to the oracle case (64.44%), indicating highly accurate noise-rate estimation.
- The estimated noise rates are reasonably consistent with the true values (e.g., under IDN 0.5, DivideMix estimates 0.53).
- Integrating the proposed method improves baseline performance in over 90% of the experimental configurations.
- The proposed method shows more pronounced improvements under high noise rates (e.g., +5.41% improvement for DivideMix on 0.5 IDN).
- The computational overhead for training is minimal (approx. 18.7h for baseline DivideMix vs. 20.3h when incorporating the proposed method).

## Highlights & Insights

- **Filling the Research Gap**: The first method to directly apply noise-rate estimation to the sample selection curriculum.
- **Plug-and-Play**: Can be integrated with 6 different SOTA methods, demonstrating strong versatility.
- **Clear Motivation**: Persuasively argues the importance of noise rate for sample selection through a simple oracle experiment (assuming known noise rate).
- **Validation of Estimation Quality**: Validates not only classification accuracy but also demonstrates how closely the estimated noise rate approaches the true value.

## Limitations & Future Work

- Global noise-rate estimation might lack granularity in class-imbalanced scenarios (different classes may exhibit different noise rates).
- The noise rate parameter is modeled by a single scalar via a sigmoid function, without considering the class-dependent variations in noise rates.
- Future work could explore instance-level noise-rate estimation instead of a global noise rate.
- Validation on larger-scale datasets (such as the full ImageNet) remains insufficient.

## Related Work & Insights

- **DivideMix**: GMM-based sample selection + semi-supervised learning, serving as the primary baseline of this work.
- **InstanceGM**: Graphical model-based noisy-label learning, but does not model the noise rate.
- **Co-teaching**: Pioneering work in sample selection curricula, using a predefined linear decay function.
- **FINE**: A sample selection criterion based on eigenvector distances in the feature space.
- **Insights**: Noise rate is an underutilized signal in noisy-label learning; probabilistic graphical models are a natural choice for modeling the label noise generation process.

## Rating

- Novelty: ⭐⭐⭐ (The idea of using noise-rate estimation for curriculum design is valuable, but the method itself is somewhat incremental)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Synthetic + 4 real-world datasets, integration with 6 SOTA methods, detailed ablations)
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐ (Provides practical contributions to the field of noisy-label learning)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DiCaP: Distribution-Calibrated Pseudo-labeling for Semi-Supervised Multi-Label Learning](../../AAAI2026/llm_evaluation/dicap_distribution-calibrated_pseudo-labeling_for_semi-supervised_multi-label_le.md)
- [\[ECCV 2024\] Versatile Incremental Learning: Towards Class and Domain-Agnostic Incremental Learning](versatile_incremental_learning_towards_class_and_domain-agnostic_incremental_lea.md)
- [\[ECCV 2024\] MERLiN: Single-Shot Material Estimation and Relighting for Photometric Stereo](merlin_single-shot_material_estimation_and_relighting_for_photometric_stereo.md)
- [\[ICLR 2026\] NAIPv2: Debiased Pairwise Learning for Efficient Paper Quality Estimation](../../ICLR2026/llm_evaluation/naipv2_debiased_pairwise_learning_for_efficient_paper_quality_estimation.md)
- [\[ICLR 2026\] TokUR: Token-Level Uncertainty Estimation for Large Language Model Reasoning](../../ICLR2026/llm_evaluation/tokur_token-level_uncertainty_estimation_for_large_language_model_reasoning.md)

</div>

<!-- RELATED:END -->
