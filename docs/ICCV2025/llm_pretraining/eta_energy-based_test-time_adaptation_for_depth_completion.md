---
title: >-
  [Paper Note] ETA: Energy-based Test-time Adaptation for Depth Completion
description: >-
  [ICCV 2025][LLM Pretraining][test-time adaptation] This paper proposes ETA, a method that employs an energy-based model to quantify the likelihood of depth predictions belonging to the source domain distribution, and guides a pre-trained depth completion model to adapt to new environments at test time by minimizing the energy of target-domain predictions. ETA achieves average improvements of 6.94% and 10.23% over the previous state of the art on outdoor and indoor scenes, respectively.
tags:
  - ICCV 2025
  - LLM Pretraining
  - test-time adaptation
  - energy-based model
  - depth completion
  - domain adaptation
  - adversarial perturbation
date: 2026-05-08
content_hash: 0a86fce1e6d829a4
---

# ETA: Energy-based Test-time Adaptation for Depth Completion

**Conference**: ICCV 2025
**arXiv**: [2508.05989](https://arxiv.org/abs/2508.05989)
**Code**: [https://fuzzythecat.github.io/eta](https://fuzzythecat.github.io/eta)
**Area**: LLM Pre-training
**Keywords**: test-time adaptation, energy-based model, depth completion, domain adaptation, adversarial perturbation

## TL;DR

This paper proposes ETA, a method that employs an energy-based model to quantify the likelihood of depth predictions belonging to the source domain distribution, and guides a pre-trained depth completion model to adapt to new environments at test time by minimizing the energy of target-domain predictions. ETA achieves average improvements of 6.94% and 10.23% over the previous state of the art on outdoor and indoor scenes, respectively.

## Background & Motivation

Depth completion is a multimodal 3D reconstruction task that estimates dense depth maps from RGB images and sparse point clouds, with broad applications in autonomous driving, embodied AI agents, and extended reality. However, when models trained on a source domain are deployed in new environments, performance degrades significantly due to covariate shift caused by changes in lighting, occlusion, and other factors.

Limitations of existing domain adaptation methods: (1) traditional domain adaptation requires labeled data from both source and target domains; (2) source-free domain adaptation does not require source data but demands multiple passes over the target dataset; (3) neither approach is suitable for online scenarios in real-time spatial applications. Test-time adaptation (TTA) allows models to update parameters batch-by-batch on unlabeled streaming data, making it a more practical alternative.

The core challenge is: how to quantify prediction deviation without knowledge of the target domain distribution? The key insight of this work is that the covariate shift in sparse point clouds/depth estimates is far smaller than in RGB images. This motivates defining an energy function over the depth prediction space, conditioning on sparse depth to evaluate prediction plausibility.

## Method

### Overall Architecture

ETA consists of two stages: (1) a preparation stage, in which an energy-based model is trained on source data using adversarially perturbed samples as out-of-distribution training signals; and (2) a test-time adaptation stage, in which both the energy model and the main parameters of the depth completion model are frozen, and only lightweight adapter layers inserted into the RGB encoder are updated by minimizing the energy value to align predictions with the source domain distribution.

### Key Designs

1. **Region-based Depth Energy Model**: An energy function $E_\phi: (\hat{d}; z) \rightarrow e$ is defined, taking the dense depth prediction $\hat{d}$ and sparse depth $z$ as input and producing a low-resolution energy map $e: \Omega \rightarrow [0, 1]$. Low energy indicates that a prediction belongs to the source domain distribution (small error), while high energy indicates an out-of-distribution prediction (large error). Compared to a single image-level scalar, the region-level energy map localizes erroneous regions for targeted updates. Energy targets are mapped via the Gibbs distribution:
    $$y = 1 - \exp(-\Delta / \tau)$$
   where $\Delta$ is the MSE error per patch and $\tau$ is a temperature parameter.

2. **Adversarial Perturbation for Out-of-Distribution Sample Generation**: The key challenge in training the energy model is the absence of out-of-distribution data. Rather than making assumptions about the target distribution, this work cleverly uses FGSM adversarial perturbations as a mechanism to explore the data space:
    $$\tilde{I_s} = I_s + \epsilon_I \cdot \text{sign}(\nabla_{I_s} \mathcal{L}_{\text{sup}}(\hat{d_s}, d_s))$$
    $$\tilde{z_s} = z_s + \epsilon_z \cdot \text{sign}(\nabla_{z_s} \mathcal{L}_{\text{sup}}(\hat{d_s}, d_s))$$
   Adversarial perturbations shift the input toward high-density blind regions of the data space, inducing out-of-distribution errors in the model. This approach achieves broader coverage than any single target domain, allowing the same energy model to be reused across different test datasets.

3. **Energy-guided Test-time Adaptation**: Lightweight adapter modules $m_\psi$ are inserted into the image encoder of the pre-trained depth completion model. During test time, only $\psi$ and BatchNorm statistics are updated. The adaptation loss integrates three components:

    - Energy minimization loss: $\ell_{\text{energy}} = -\frac{1}{|\Omega_p|}\sum_{x} \log(1 - E_\phi(\hat{d}_t, z_t)(x))$
    - Sparse depth consistency: $\ell_{\text{sparse}} = \frac{1}{|\Omega_z|}\sum_{x \in \Omega_z}|\hat{d}_t(x) - z_t(x)|$
    - Local smoothness: an edge-aware L1 penalty on depth gradients, weighted by image gradients to preserve object boundaries

### Loss & Training

The energy model is trained with a cross-entropy loss: $\mathcal{L}_{\text{energy}} = -\frac{1}{|\Omega_p|}\sum_x y(x)\log(y(x)/e(x))$. The test-time adaptation loss is $\mathcal{L}_{\text{adapt}} = w_e \ell_{\text{energy}} + w_z \ell_{\text{sparse}} + w_s \ell_{\text{smooth}}$. Note that a separate energy model is trained for each depth completion model, as the energy distribution is dependent on the parameters of the corresponding depth completion model.

## Key Experimental Results

### Main Results

| Depth Completion Model | Adaptation Method | VKITTI-FOG MAE | nuScenes MAE | NYUv2 MAE | ScanNet MAE |
|------------|---------|---------------|-------------|----------|-----------|
| NLSPN | Pre-trained | 1.309 | 2.656 | 0.388 | 0.233 |
| NLSPN | ProxyTTA | 0.686 | 2.589 | 0.124 | 0.074 |
| NLSPN | **ETA** | **0.545** | **2.359** | **0.105** | **0.067** |
| BP-Net | Pre-trained | 0.893 | 2.787 | 0.234 | 0.123 |
| BP-Net | ProxyTTA | 0.571 | 2.373 | 0.174 | 0.102 |
| BP-Net | **ETA** | **0.544** | **2.281** | **0.161** | **0.093** |
| CostDCNet | Pre-trained | 1.042 | 3.064 | 0.189 | 0.144 |
| CostDCNet | ProxyTTA | 0.512 | 2.062 | 0.095 | 0.068 |
| CostDCNet | **ETA** | **0.508** | **2.048** | **0.089** | **0.059** |

### Ablation Study

| Energy Update Strategy | MAE | RMSE | Notes |
|-----------|------|------|------|
| Global (image-level) | 1.406 | 4.226 | A single energy value cannot localize erroneous regions |
| **Local (region-level)** | **0.703** | **2.996** | Fine-grained energy map enables targeted updates |
| Baseline (no energy) | 2.842 | 6.557 | — |

| Cross-domain Adaptation | Method | NYUv2 MAE | SceneNet MAE | ScanNet MAE |
|-----------|------|----------|-------------|-----------|
| KITTI→Indoor | Pre-trained | 1.987 | 1.432 | 2.657 |
| KITTI→Indoor | ProxyTTA | 1.380 | 0.401 | 0.311 |
| KITTI→Indoor | **ETA** | **1.322** | **0.340** | **0.272** |

### Key Findings

- ETA improves MAE by an average of 5.36% and 10.13% over the previous state-of-the-art ProxyTTA on outdoor and indoor scenes, respectively.
- Compared to directly transferring the classification TTA method TEA to depth estimation, ETA achieves a 24.9% MAE improvement, demonstrating the necessity of task-specific design for regression.
- Models with larger initial errors benefit more from ETA (e.g., MSG-CHN on VKITTI-FOG), indicating that the energy model effectively pulls back severely deviated predictions.
- ETA consistently achieves the best performance even in the extreme outdoor-to-indoor cross-domain setting.

## Highlights & Insights

- Using adversarial perturbations as a substitute for assumptions about the target distribution to generate out-of-distribution data is a particularly elegant design — it enables a single energy model to be reused across diverse target domains.
- The region-level energy map not only improves adaptation performance but also provides a degree of interpretability: high-energy regions correspond to areas where the model is uncertain.
- The stability of sparse depth under domain shift is the key assumption and insight of this work; conditioning the energy function on sparse depth rather than RGB features is central to its effectiveness.

## Limitations & Future Work

- A separate energy model must be trained for each depth completion model, resulting in deployment costs that scale linearly with the number of models.
- The adversarial perturbation strength parameters $\epsilon_I, \epsilon_z$ require tuning and may need different settings across scenes.
- The patch size of the energy model governs a trade-off between localization precision and computational overhead.
- In scenarios where LiDAR point clouds are extremely sparse (e.g., mobile SfM), conditioning on sparse depth may become unreliable.

## Related Work & Insights

- Energy-based models have well-established applications in generative modeling and density estimation; this work effectively transplants them into the novel application of TTA.
- The use of adversarial perturbations as a data augmentation and exploration tool is generalizable to other tasks that lack OOD samples.
- The concept of region-level energy-guided adaptation may be applicable to domain adaptation in other dense prediction tasks, such as semantic segmentation and optical flow.

## Rating

- Novelty: ⭐⭐⭐⭐ Combined innovation of energy-guided TTA and adversarial perturbation for data space exploration
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 depth completion models × 6 datasets × 5 baseline methods, covering indoor, outdoor, and cross-domain settings
- Writing Quality: ⭐⭐⭐⭐ Derivations are clear and ablations are thorough, though the paper is somewhat lengthy overall
- Value: ⭐⭐⭐⭐ Provides a practical and effective TTA solution for domain adaptation in depth completion

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Implicit Bias and Loss of Plasticity in Matrix Completion: Depth Promotes Low-Rank](../../ICLR2026/llm_pretraining/implicit_bias_and_loss_of_plasticity_in_matrix_completion_depth_promotes_low-ran.md)
- [\[NeurIPS 2025\] The Curse of Depth in Large Language Models](../../NeurIPS2025/llm_pretraining/the_curse_of_depth_in_large_language_models.md)
- [\[NeurIPS 2025\] One Prompt Fits All: Universal Graph Adaptation for Pretrained Models](../../NeurIPS2025/llm_pretraining/one_prompt_fits_all_universal_graph_adaptation_for_pretrained_models.md)
- [\[AAAI 2026\] Learning Time in Static Classifiers](../../AAAI2026/llm_pretraining/learning_time_in_static_classifiers.md)
- [\[NeurIPS 2025\] Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks](../../NeurIPS2025/llm_pretraining/gradient-weight_alignment_as_a_train-time_proxy_for_generalization_in_classifica.md)

<!-- RELATED:END -->
