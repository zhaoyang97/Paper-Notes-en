---
title: >-
  [Paper Note] Semi-supervised Deep Transfer for Regression without Domain Alignment
description: >-
  [ICCV 2025][Medical Imaging][Source-free domain adaptation] This paper proposes CRAFT (Contradistinguisher-based Regularization Approach for Flexible Training), a semi-supervised transfer learning framework that requires neither source data nor domain alignment, specifically designed for regression tasks. CRAFT jointly optimizes a supervised loss and an unsupervised Contradistinguisher-based regularization term to substantially improve prediction performance under label-scarc…
tags:
  - "ICCV 2025"
  - "Medical Imaging"
  - "Source-free domain adaptation"
  - "semi-supervised transfer learning"
  - "regression"
  - "EEG decoding"
  - "brain age prediction"
date: 2026-05-08
content_hash: 813a96212e409732
---

# Semi-supervised Deep Transfer for Regression without Domain Alignment

**Conference**: ICCV 2025
**arXiv**: [2509.05092](https://arxiv.org/abs/2509.05092)  
**Code**: Available (see Appendix E.2)  
**Area**: Medical Imaging
**Keywords**: Source-free domain adaptation, semi-supervised transfer learning, regression, EEG decoding, brain age prediction

## TL;DR

This paper proposes CRAFT (Contradistinguisher-based Regularization Approach for Flexible Training), a semi-supervised transfer learning framework that requires neither source data nor domain alignment, specifically designed for regression tasks. CRAFT jointly optimizes a supervised loss and an unsupervised Contradistinguisher-based regularization term to substantially improve prediction performance under label-scarce conditions.

## Background & Motivation

Deep learning models deployed in practice suffer from **domain shift**: models trained on a source domain perform poorly on shifted target data, a problem particularly pronounced in medical and neuroscientific applications. Conventional domain adaptation approaches face several practical challenges:

**Source data unavailability**: Medical data cannot be shared due to privacy regulations or prohibitive storage and computational costs.

**Scarcity of target-domain labels**: Annotation is expensive, leaving only a small number of labeled samples available.

**Neglect of regression tasks**: Most source-free domain adaptation (SF-DA) methods are designed for classification and rely on concepts such as class prototypes, making them ill-suited for continuous-valued outputs.

Among existing methods:
- CUDA (Contradistinguisher) is effective but requires source data and supports classification only.
- TASFAR addresses regression in the SF-UDA setting but is fully unsupervised and relies on uncertainty estimation.
- BBCN depends on class prototypes and does not generalize naturally to regression.

## Method

### Overall Architecture

Given a model $\theta^s$ pre-trained on source data, the target dataset consists of a small labeled set $\{(\mathbf{x}_i^t, y_i^t)\}_{i=1}^{N_l}$ and a large unlabeled set $\{\mathbf{x}_i^t\}_{i=1}^{N_{ul}}$, where $y \in \mathbb{R}$ is a continuous value. CRAFT initializes model parameters with $\theta^s$ and adapts to the target domain via alternating two-step optimization: parameters are fixed to select pseudo-labels, and pseudo-labels are then fixed to update parameters.

### Key Designs

1. **Semi-supervised objective**: CRAFT combines a supervised loss and a CUDA-based unsupervised regularization term with weight $\alpha$:
$$\mathcal{L}(\mathcal{D}^t, \theta) = \sum_{i=1}^{N_l} \log p(y_i^t | \mathbf{x}_i^t, \theta) + \alpha \sum_{i=1}^{N} \log q(\mathbf{x}_i^t, y_i^t | \theta).$$
The supervised term is a standard Gaussian log-likelihood (MSE), and the unsupervised term is the CUDA joint distribution:
$$\log q(\mathbf{x}^t, y^t | \theta) = \log \frac{p(y^t|\mathbf{x}^t, \theta)}{\sum_{i=1}^N p(y^t|\mathbf{x}_i^t, \theta)} p(y^t).$$
The denominator normalizes the model's total predicted probability for a given label, thereby removing prediction bias inherited from the source domain; $p(y^t)$ introduces a target-domain label prior. This can be theoretically derived as a prior term in MAP estimation (Appendix A.1).

2. **Pseudo-label selection for regression**: In classification, labels can be selected from a finite set by maximizing the joint distribution. Since the label space is continuous in regression, CRAFT discretizes the label range into small intervals and uses interval midpoints as candidate pseudo-labels:
$$\tilde{y}_i^t = \arg\max_{y^t \in \mathcal{Y}} \frac{\mathcal{N}(y^t; f(\mathbf{x}_i^t; \theta), c) p(y^t)}{\sum_{l=1}^N \mathcal{N}(y^t; f(\mathbf{x}_l^t; \theta), c)}.$$
Importantly, discretization is used solely for efficient pseudo-label selection and does not constrain model outputs to discrete values. The label prior $p(y)$ is estimated from data via a mixture model. This design avoids nested gradient descent, making optimization efficient while admitting informative priors.

3. **Parameter update (maximization step)**: With pseudo-labels fixed, three terms are jointly optimized:
$$\theta^* = \arg\max_\theta -\sum_{i=1}^{N_l}(y_i^t - f(\mathbf{x}_i^t;\theta))^2 - \alpha\!\left(\sum_{i=1}^{N}(\tilde{y}_i^t - f(\mathbf{x}_i^t;\theta))^2 - \sum_{i=1}^{N}\log\sum_{l=1}^N \exp\!\left(-(\tilde{y}_i^t - f(\mathbf{x}_l^t;\theta))^2\right)\right).$$
Intuitively, the first term encourages alignment with ground-truth labels; the second term forces samples with different pseudo-labels to produce distinct predictions (learning a better regression function); since parameters are initialized from the source model, the adapted model is implicitly regularized to remain close to the source model.

### Loss & Training

- Adam optimizer (lr=1e-4), batch size 128 (EEG) / 4 (MRI).
- $\alpha$ is selected via grid search over {0.01, 0.1, 1.0}; $\alpha=0.1$ is consistently preferred across experiments.
- Each iteration: pseudo-labels for the current batch of unlabeled data are computed first, then supervised and unsupervised terms are jointly optimized with pseudo-labels fixed.
- The checkpoint with the best validation performance is retained (EEG), or the final model is used (MRI, where the dataset is too small for hyperparameter search).
- The log-sum-exp trick is applied to avoid numerical instability.

## Key Experimental Results

### Main Results — Saccade Amplitude Prediction (EEG, 1% labels)

| Method | R ↑ | RMSE (pixels) ↓ |
|--------|-----|-----------------|
| Naive Baseline | — | 149.12 ± 0.02 |
| TL (100% labels, upper bound) | 0.93 | 51.47 ± 0.63 |
| TL (1% labels) | 0.77 | 92.26 ± 1.66 |
| Progressive Mixup | 0.48 | 135.70 ± 1.25 |
| BBCN | 0.76 | 99.80 ± 3.35 |
| TASFAR | 0.76 | 86.41 ± 1.05 |
| DataFree | 0.80 | 87.64 ± 3.08 |
| **CRAFT** | **0.81** | **84.17 ± 3.95** |

CRAFT achieves a 9% RMSE improvement over supervised transfer learning (TL) and more than 4% over the best SF-SSDA baseline.

### Ablation / Extension — Brain Age Prediction (MRI, 20% labels)

| Method | R ↑ | RMSE (years) ↓ |
|--------|-----|----------------|
| Naive Baseline | — | 7.91 ± 0.05 |
| TL (100% labels, upper bound) | 0.66 | 6.14 ± 0.03 |
| TL (20% labels) | 0.41 | 7.41 ± 0.21 |
| Progressive Mixup | 0.34 | 7.71 ± 0.14 |
| BBCN | 0.28 | 8.00 ± 0.15 |
| TASFAR | 0.42 | 7.47 ± 0.15 |
| DataFree | 0.50 | 7.36 ± 0.14 |
| **CRAFT** | **0.51** | **7.14 ± 0.11** |

CRAFT improves over TL by approximately 4% and over the state-of-the-art SF-SSDA method by more than 3%. On crowd counting and tumor size prediction, gains exceed 5% and 2%, respectively.

### Key Findings

- CRAFT maintains the lowest RMSE across all proportions of unlabeled data; its advantage grows as the proportion of unlabeled data increases.
- The closest competing methods are TASFAR (pseudo-label approach) and DataFree (feature alignment approach).
- $\alpha=0.1$ is consistently optimal across all experiments, suggesting that the unsupervised term provides moderate regularization rather than a dominant learning signal.
- Sampling bias experiment: when the label distribution of the training set is deliberately skewed (removing 80% of elderly samples), CRAFT effectively mitigates bias by incorporating an unbiased prior $p(y)$, yielding an RMSE improvement of approximately 5%.
- Computational cost: training time is comparable to DataFree (EEG: 0.45 min/epoch vs. 0.55 min) and far below BBCN (2.71 min).

## Highlights & Insights

- The problem is precisely formulated: the intersection of no source data, sparse labels, and regression represents a genuinely challenging real-world scenario in medicine and neuroscience that prior methods have largely overlooked.
- The extension of CUDA to regression is methodologically elegant: by discretizing the pseudo-label search space rather than the model outputs, the approach efficiently preserves continuous prediction capability.
- The theoretical grounding is solid: the unsupervised objective is derived as a maximum-entropy prior over model parameters (MAP estimation), providing motivation beyond heuristic regularization.
- The incorporation of the label prior $p(y)$ enables the model to actively mitigate sampling bias in the training set, an important property for real-world data.

## Limitations & Future Work

- The bin size for discretization remains a hyperparameter requiring manual tuning, albeit with guidance provided in the paper.
- Validation is limited to relatively small datasets (EEG ~12K samples; MRI ~188 samples); performance on larger datasets remains unknown.
- Applicability to high-dimensional output spaces (e.g., dense prediction or segmentation) is not explored.
- The Gaussian conditional distribution assumption may be inappropriate for multimodal or heavy-tailed regression problems.
- No comparison is made against recent parameter-efficient transfer methods such as LoRA.
- While $\alpha=0.1$ is empirically stable, its optimal value may depend on the degree of domain shift.

## Related Work & Insights

- CRAFT builds on the theoretical foundation of CUDA (Contradistinguisher), with two key extensions: (a) from UDA to SF-SSDA, and (b) from classification to regression.
- Unlike SHOT++, CRAFT requires neither feature alignment nor prototype-based clustering of pseudo-labels.
- The feature alignment approach of DataFree/BUFR is complementary to CRAFT — CRAFT bypasses intermediate representation alignment and directly learns the joint distribution.
- The EEGNet-LSTM architecture introduced for saccade prediction substantially outperforms baselines (>16%), representing a contribution independent of CRAFT.
- The work provides important practical guidance for transfer learning in medical imaging and neuroscience.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The extension of CUDA to SF-SSDA regression carries genuine theoretical novelty.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Four datasets, computational complexity analysis, and sampling bias evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are clear and problem motivation is well-articulated.
- **Value**: ⭐⭐⭐⭐ Fills a gap in SF-SSDA for regression tasks with clearly defined practical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Is User Feedback Always Informative? Retrieval Latent Defending for Semi-Supervised Domain Adaptation without Source Data](../../ECCV2024/medical_imaging/is_user_feedback_always_informative_retrieval_latent_defending_for_semi-supervis.md)
- [\[ICCV 2025\] ViCTr: Vital Consistency Transfer for Pathology Aware Image Synthesis](victr_vital_consistency_transfer_for_pathology_aware_image_synthesis.md)
- [\[CVPR 2026\] SemiGDA: Generative Dual-distribution Alignment for Semi-Supervised Medical Image Segmentation](../../CVPR2026/medical_imaging/semigda_generative_dual-distribution_alignment_for_semi-supervised_medical_image.md)
- [\[ICCV 2025\] SciVid: Cross-Domain Evaluation of Video Models in Scientific Applications](scivid_cross-domain_evaluation_of_video_models_in_scientific_applications.md)
- [\[ICCV 2025\] An OpenMind for 3D Medical Vision Self-supervised Learning](an_openmind_for_3d_medical_vision_selfsupervised_learning.md)

</div>

<!-- RELATED:END -->
