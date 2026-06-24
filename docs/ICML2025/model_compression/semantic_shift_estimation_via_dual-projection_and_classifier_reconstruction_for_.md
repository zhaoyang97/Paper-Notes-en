---
title: >-
  [Paper Note] Semantic Shift Estimation via Dual-Projection and Classifier Reconstruction for Exemplar-Free Class-Incremental Learning
description: >-
  [ICML2025][Model Compression][Class-Incremental Learning] This work proposes the DPCR method, which estimates semantic shift through dual projection (task-level TSSP + category-level CIP) and reconstructs the classifier without backpropagation (BP) using ridge regression. It simultaneously addresses semantic shift and decision bias in exemplar-free class-incremental learning, outperforming SOTA on multiple benchmarks.
tags:
  - "ICML2025"
  - "Model Compression"
  - "Class-Incremental Learning"
  - "Exemplar-Free"
  - "Semantic Shift Estimation"
  - "Dual-Projection"
  - "Ridge Regression Classifier Reconstruction"
date: 2026-05-08
content_hash: 7ae5f5263d022250
---

# Semantic Shift Estimation via Dual-Projection and Classifier Reconstruction for Exemplar-Free Class-Incremental Learning

**Conference**: ICML2025  
**arXiv**: [2503.05423](https://arxiv.org/abs/2503.05423)  
**Code**: [RHe502/ICML25-DPCR](https://github.com/RHe502/ICML25-DPCR)  
**Area**: Domain Adaptation / Continual Learning  
**Keywords**: Class-Incremental Learning, Exemplar-Free, Semantic Shift Estimation, Dual-Projection, Ridge Regression Classifier Reconstruction  

## TL;DR

This work proposes the DPCR method, which estimates semantic shift through dual projection (task-level TSSP + category-level CIP) and reconstructs the classifier without backpropagation (BP) using ridge regression. It simultaneously addresses semantic shift and decision bias in exemplar-free class-incremental learning, outperforming SOTA on multiple benchmarks.

## Background & Motivation

Exemplar-free class-incremental learning (EFCIL) requires models to sequentially learn new classes without storing old data, but faces two core challenges:

**Semantic Shift**: Updating the backbone after learning new tasks causes the embeddings of old classes to shift in the feature space, making the learned feature representations no longer compatible with the old classes.

**Decision Bias**: The classifier is trained only on new task data via BP, leading to a preference for new categories (task-recency bias) and disrupting the balance between old and new knowledge.

Limitations of prior work:
- **Freezing the backbone** (ACIL, FeCAM): Eliminates semantic shift but severely limits plasticity.
- **NCM Classifier** (SDC, ADC): Relies on representation quality, lacks trainable parameters, and has poor adaptability.
- **LDC**: Only captures task-level shift, ignoring differences between categories; furthermore, it requires BP to train the projector, resulting in high computational overhead.

## Method

DPCR consists of three phases: incremental representation learning, dual-projection shift estimation, and ridge regression classifier reconstruction.

### 1. Incremental Representation Learning

Following the knowledge distillation framework of LwF, the training loss is:

$$\mathcal{L}_{\text{rep}} = \mathcal{L}_{\text{ce}}(h_{\tau_t}^{\text{au}}(f_{\theta_t}(\mathcal{X}_t)), y_t) + \alpha \mathcal{L}_{\text{kd}}(\mathcal{X}_t)$$

where $\mathcal{L}_{\text{kd}}$ constrains the logit consistency between the outputs of the old and new backbones, and $\alpha$ is the distillation weight.

### 2. Dual-Projection Shift Estimation (DP)

**Task-level Semantic Shift Projection (TSSP)**: Learns a linear projection $\boldsymbol{P}^{t-1 \to t} \in \mathbb{R}^{d \times d}$ to map the embeddings of the old backbone to the space of the new backbone. It is obtained through the closed-form solution of minimizing MSE:

$$\boldsymbol{P}^{t-1 \to t} = (\boldsymbol{X}_t^{\theta_{t-1}\top} \boldsymbol{X}_t^{\theta_{t-1}} + \epsilon \boldsymbol{I})^{-1} \boldsymbol{X}_t^{\theta_{t-1}\top} \boldsymbol{X}_t^{\theta_t}$$

where $\epsilon = 10^{-9}$ prevents matrix ill-conditioning. Key advantage: **Does not require BP training**, directly solved in closed form.

**Category Information Projection (CIP)**: TSSP shares the same projection across all classes within the same task, ignoring class differences. CIP injects class information via row-space projection:

- Performs SVD decomposition on the uncentered covariance $\Phi_{t-1,c}^{\theta_{t-1}}$ of each class.
- Takes the singular vectors $\boldsymbol{U}_{t-1,c}^r$ corresponding to non-zero singular values to construct the row space projection operator.
- Final category-aware projection: $\boldsymbol{P}_{t-1,c}^{t-1 \to t} = \boldsymbol{P}^{t-1 \to t} \boldsymbol{U}_{t-1,c}^r \boldsymbol{U}_{t-1,c}^{r\top}$

CIP is **training-free** and does not increase training cost.

### 3. Ridge Regression Classifier Reconstruction (RRCR)

Represents classifier training as a ridge regression problem to avoid the decision bias introduced by BP:

$$\hat{\boldsymbol{W}}_t = \left(\sum_{i=1}^{t} \boldsymbol{\Phi}_i^{\theta_t} + \gamma \boldsymbol{I}\right)^{-1} \sum_{i=1}^{t} \boldsymbol{H}_i^{\theta_t}$$

Since new embeddings of old data are inaccessible under the EFCIL constraint, the shift estimated by DP is utilized to calibrate the old information:

- Covariance calibration: $\hat{\boldsymbol{\Phi}}_{i,c}^{\theta_t} = \boldsymbol{P}_{i,c}^{t-1 \to t \top} \boldsymbol{\Phi}_{i,c}^{\theta_{t-1}} \boldsymbol{P}_{i,c}^{t-1 \to t}$
- Prototype calibration: $\hat{\boldsymbol{\mu}}_{i,c}^{\theta_t} = \boldsymbol{\mu}_{i,c}^{\theta_{t-1}} \boldsymbol{P}_{i,c}^{t-1 \to t}$

**Category Normalization (CN)**: Non-unitary transforms from the dual projection introduce numerical imbalance; thus, L2 normalization is performed on the classifier weights column-by-column:

$$\hat{\boldsymbol{W}}_t' = \left[\frac{\boldsymbol{w}_1}{\|\boldsymbol{w}_1\|_2}, \frac{\boldsymbol{w}_2}{\|\boldsymbol{w}_2\|_2}, \ldots, \frac{\boldsymbol{w}_{tC}}{\|\boldsymbol{w}_{tC}\|_2}\right]$$

For each class, only the covariance and prototype of size $d^2 + d$ need to be stored, resulting in a low memory cost.

## Key Experimental Results

Backbone: ResNet-18 | Regularization factor $\gamma$: CIFAR-100=200, Tiny-ImageNet=2000, ImageNet-100=2000

### Main Results (Cold-start setting, average of 3 runs)

| Method | CIFAR-100 T=10 $\mathcal{A}_f$/$\mathcal{A}_{avg}$ | CIFAR-100 T=20 | Tiny-IN T=10 | ImageNet-100 T=10 |
|------|------|------|------|------|
| LwF | 42.60/58.51 | 36.34/51.52 | 26.99/42.92 | 42.25/61.23 |
| ACIL | 35.53/50.53 | 27.22/39.58 | 26.10/41.86 | 44.61/59.77 |
| LDC | 46.60/61.67 | 36.76/53.06 | 33.74/47.37 | 49.98/67.47 |
| **DPCR** | **50.24/63.21** | **38.98/54.42** | **35.20/47.55** | **52.16/67.51** |

DPCR surpasses the second-best LDC in $\mathcal{A}_f$ by **+3.64%** on CIFAR-100 T=10, and by **+3.48%** on ImageNet-100 T=20.

### Ablation Study (CIFAR-100 T=10)

| Component | $\mathcal{A}_f$ (%) | $\mathcal{A}_{avg}$ (%) |
|------|------|------|
| RRCR only | 32.17 | 44.89 |
| +TSSP | 40.86 | 55.76 |
| +TSSP+CIP | 45.56 | 62.15 |
| +TSSP+CIP+CN | **51.04** | **64.44** |

TSSP contributes the most (+8.69%), CIP further improves it by +4.70%, and CN corrects numerical imbalance for another +5.48% gain.

### Large-scale Dataset (ImageNet-1k T=10)

| Method | $\mathcal{A}_f$ (%) | $\mathcal{A}_{avg}$ (%) |
|------|------|------|
| LDC | 35.15 | 53.88 |
| **DPCR** | **35.49** | **54.22** |

## Highlights & Insights

1. **Dual Projection = Task-level + Category-level**: TSSP captures global shift while CIP injects class-specific local information, which is more comprehensive than LDC's task-level only projection.
2. **Complete Closed-form Solutions**: Both TSSP and RRCR do not require BP training, ensuring high computational efficiency and eliminating the instability of iterative optimization.
3. **Stability-Plasticity Balance**: Unlike NCM, RRCR does not solely rely on representation quality, nor does it overwrite old decision boundaries like BP training does.
4. **CIP Simultaneously Enhances Stability and Plasticity**: Ablation visualization shows that CIP improves the accuracy on both old and new classes.
5. **Clever DP-NCM Experimental Design**: By freezing the backbone and varying only the shift estimation methods, it fairly demonstrates that DP outperforms ADC/LDC estimations.

## Limitations & Future Work

1. **Linear Projection Assumption**: Assumes a linear mapping between old and new features, limiting the capacity to capture non-linear shifts.
2. **Covariance Storage Overhead Scales Quadratically with Feature Dimension**: $d^2 + d$ per class, which might become a bottleneck when $d$ is very large.
3. **Accumulated Error of Shifts Across Tasks**: Chained calibration across multiple tasks may accumulate estimation errors.
4. **Only Validated on Classification Tasks**: Has not been extended to more complex incremental learning scenarios such as detection or segmentation.
5. **Cold-start Setting Limitation**: All tasks have equal numbers of classes, without considering more realistic imbalanced-class scenarios.
6. **Backbone Fixed to ResNet-18**: Has not validated performance on modern architectures like ViTs (Vision Transformers).

## Related Work & Insights

- **LDC** (Gomez-Villa et al., 2024): The direct baseline for improvement of DPCR. LDC only performs task-level linear projection and requires BP training.
- **ACIL** (Zhuang et al., 2022): The first work to introduce analytic learning to CIL. DPCR inherits the ridge regression formulation but does not freeze the backbone.
- **SDC/ADC**: Only estimates prototype translation, whereas DPCR's dual projection captures richer transformation details.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combination of dual-projection + RRCR is novel; the CIP row-space projection concept is simple and elegant)
- Experimental Thoroughness: ⭐⭐⭐⭐ (5 datasets + thorough ablations + fair NCM comparison + visualizations, though lacking ViT validation)
- Writing Quality: ⭐⭐⭐⭐ (Clear mathematical derivations, with self-consistent logic from problem statement to solutions)
- Value: ⭐⭐⭐⭐ (Provides a unified framework to simultaneously address shift and bias in EFCIL, with strong practicality)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Adapter Merging with Centroid Prototype Mapping for Scalable Class-Incremental Learning](../../CVPR2025/model_compression/adapter_merging_with_centroid_prototype_mapping_for_scalable_class-incremental_l.md)
- [\[CVPR 2025\] CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning](../../CVPR2025/model_compression/cl-lora_continual_low-rank_adaptation_for_rehearsal-free_class-incremental_learn.md)
- [\[ICCV 2025\] Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning](../../ICCV2025/model_compression/achieving_more_with_less_additive_prompt_tuning_for_rehearsal-free_class-increme.md)
- [\[NeurIPS 2025\] Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning](../../NeurIPS2025/model_compression/mixture_of_noise_for_pre-trained_model-based_class-incremental_learning.md)
- [\[ICCV 2025\] Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning](../../ICCV2025/model_compression/integrating_task-specific_and_universal_adapters_for_pre-trained_model-based_cla.md)

</div>

<!-- RELATED:END -->
