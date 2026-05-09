---
title: >-
  [Paper Note] Implicit Modeling for Transferability Estimation of Vision Foundation Models
description: >-
  [NeurIPS 2025][Self-Supervised Learning][Transferability Estimation] This paper proposes Implicit Transferability Modeling (ITM), a framework that encodes the transferability of model–task pairs via a latent variable $z$, and employs Divide-and-conquer Variational Approximation (DVA) to efficiently simulate embedding space evolution. On 10 downstream tasks with 10 diverse pre-trained models, the weighted Kendall $\tau_w$ improves from the previous state-of-the-art of 0.45 to 0.61.
tags:
  - NeurIPS 2025
  - Self-Supervised Learning
  - Transferability Estimation
  - Pre-trained Model Selection
  - Embedding Space Evolution
  - Variational Approximation
  - Vision Foundation Models
date: 2026-05-08
content_hash: 3d131390e2107367
---

# Implicit Modeling for Transferability Estimation of Vision Foundation Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.23145](https://arxiv.org/abs/2510.23145)
**Authors**: Yaoyan Zheng, Huiqun Wang, Nan Zhou, Di Huang (Beihang University)
**Code**: [BUAAHugeGun/ITM](https://github.com/BUAAHugeGun/ITM)
**Area**: Self-Supervised Learning
**Keywords**: Transferability Estimation, Pre-trained Model Selection, Embedding Space Evolution, Variational Approximation, Vision Foundation Models

## TL;DR

This paper proposes Implicit Transferability Modeling (ITM), a framework that encodes the transferability of model–task pairs via a latent variable $z$, and employs Divide-and-conquer Variational Approximation (DVA) to efficiently simulate embedding space evolution. On 10 downstream tasks with 10 diverse pre-trained models, the weighted Kendall $\tau_w$ improves from the previous state-of-the-art of 0.45 to 0.61.

## Background & Motivation

### State of the Field
Under the pre-train–fine-tune paradigm, a large number of pre-trained models are publicly available, yet models with different architectures and pre-training strategies exhibit substantial performance variation across downstream tasks. Transferability Estimation (TE) aims to predict the performance ranking of models on target tasks with minimal computational overhead, thereby avoiding brute-force exhaustive fine-tuning.

### Limitations of Prior Work
- **Static statistical methods** (NCE, LEEP, LogME, etc.) analyze only the statistical properties of pre-trained embedding spaces, ignoring the dynamic evolution of embeddings during fine-tuning.
- **Dynamic evolution methods** (PED, LEAD, SA, etc.) attempt to simulate embedding space evolution but rely on hand-crafted rules or idealized assumptions, failing to comprehensively capture model properties and task adaptability.
- Existing methods are predominantly evaluated on **architecturally similar, supervised CNN** models; when applied to ViT architectures and self-supervised strategies (instance discrimination, masked image modeling), divergent convergence behaviors across models lead to significant degradation in estimation accuracy.
- The vast majority of methods support only image classification, lacking adaptability to dense prediction tasks such as semantic segmentation.

### Root Cause
A model's performance on a downstream task is governed by two factors: (1) the intrinsic properties of the model (architecture, pre-training data, and strategy); and (2) the characteristics of the downstream task. Their interaction determines adaptation dynamics. Existing methods estimate performance solely by explicitly simulating embedding evolution, without adequately modeling this interaction. The core idea of this paper is to use a small number of learnable parameters to **implicitly model** the transferability of each model–task pair, thereby endowing the estimation framework with better generalization across diverse models.

## Method

### Overall Architecture: Implicit Transferability Modeling (ITM)

ITM decouples the transferability of a pre-trained model into a latent variable $z$, representing the fine-tuned embedding space as a posterior distribution $q(\hat{E} \mid E, z)$, and reformulates the complex mapping problem as a probabilistic estimation framework.

### Divide-and-Conquer Variational Approximation (DVA)

**Step 1: Batch-wise Division**

The global embedding space $E$ is partitioned into $K$ subspaces (each corresponding to the embeddings of one mini-batch in practice). Under the batch independence assumption, the posterior is factorized as a product of subspace posteriors:

$$q_\psi(\hat{E} \mid E, z) = \prod_{j=1}^{K} q_\psi(\hat{E}_j \mid E_j, z)$$

A conditional mapping $f(\cdot;\, W_z)$ embeds the latent variable $z$ into each batch's pre-trained embeddings to generate synthetic posterior conditions $\Theta_j = f(E_j;\, W_z)$. Here, $W_z$ constitutes the learnable parameters that implicitly encode transferability.

**Step 2: Pseudo-cluster Center Generation**

Leveraging the strong convergence property of modern pre-trained models on training data—whereby representations of different classes tend to form well-separated clusters—pseudo-cluster centers are generated as target final states of the embedding space. These can be realized via one-hot vectors, high-dimensional random vectors, or PCA eigenvectors, with mean and standard deviation of the initial features used for offset initialization to accelerate convergence.

**Step 3: Deparametric Approximation**

The key innovation lies in eliminating dependence on learnable parameters $W_g$. Taking MSE loss as an example, the gradient descent update rule for mapping layer $g$ is derived analytically, yielding a closed-form iterative formula for subspace evolution:

$$E_j^{(n+1)} = (I - \eta C)\, E_j^{(n)} + \eta C\, \hat{E}_j$$

where $C = \frac{1}{B}\,\Theta_j \Theta_j^\top$ is a constant matrix determined solely by the initial conditions. This formulation eliminates the need for explicit iterative optimization over each subspace, substantially reducing computational overhead.

### Training and Evaluation Pipeline
- **Training**: The latent variable $z$ is embedded into embedding subspaces via $W_z$; DVA updates are performed using the deparametric approximation, and $W_z$ is optimized with the downstream task objective (e.g., cross-entropy loss).
- **Evaluation**: The accuracy of the evolved embedding space on the evaluation set is computed as the estimation score.
- Training runs for 500 iterations, with evaluation every 100 steps; the highest score is taken as the final estimate.

## Key Experimental Results

### Main Results: Transferability Estimation on 10 Datasets

Ten pre-trained models (4 supervised CNNs + 3 contrastive ViTs + 3 MIM ViTs) are evaluated against 7 competing methods on 10 classification datasets.

| Method | Cal101 | Cars | CIFAR100 | CIFAR10 | DTD | Aircraft | Flowers | Food | Pets | SUN | Avg. $\tau_w$ |
|--------|--------|------|----------|---------|-----|----------|---------|------|------|-----|---------------|
| NLEEP | 0.47 | 0.04 | 0.32 | 0.48 | 0.57 | 0.13 | 0.62 | 0.24 | 0.30 | 0.01 | 0.32 |
| LogME | 0.71 | 0.36 | 0.56 | 0.61 | 0.61 | 0.22 | 0.77 | 0.15 | 0.14 | 0.38 | 0.45 |
| PARC | 0.08 | 0.00 | −0.07 | 0.25 | 0.42 | 0.12 | 0.62 | 0.19 | 0.10 | 0.01 | 0.17 |
| SFDA | 0.59 | 0.07 | 0.48 | 0.79 | 0.13 | 0.18 | −0.39 | 0.33 | 0.28 | 0.09 | 0.25 |
| ETran | 0.13 | −0.06 | −0.14 | 0.21 | 0.36 | 0.27 | 0.08 | 0.23 | 0.38 | −0.06 | 0.14 |
| PED | 0.32 | −0.01 | 0.51 | 0.77 | 0.06 | −0.20 | 0.16 | 0.60 | −0.20 | 0.07 | 0.21 |
| SA (LDA) | 0.31 | −0.11 | −0.06 | 0.34 | 0.33 | 0.22 | 0.14 | 0.18 | 0.33 | −0.12 | 0.16 |
| **ITM (Ours)** | **0.56** | **0.61** | **0.59** | **0.69** | **0.77** | **0.43** | **0.65** | **0.44** | **0.73** | **0.62** | **0.61** |

ITM achieves an average $\tau_w$ of 0.61, representing a 35.6% improvement over the previous best LogME (0.45). Results across five random seeds yield $0.60 \pm 0.01$, demonstrating high stability. Runtime is only 8.42 seconds (CPU), negligible compared to the 738 seconds (GPU) required for feature extraction.

### Generalization to Semantic Segmentation

ITM is validated on two dense prediction datasets, CamVid and Cityscapes, using 5 pre-trained ViT models.

| Dataset | Metric | MoCov3-B16 | DINO-B16 | MAE-B16 | SimMIM-B16 | MAE-L16 | $\tau_w$ |
|---------|--------|------------|----------|---------|------------|---------|----------|
| CamVid | mIoU | 58.11 | 60.05 | 63.99 | 64.52 | 68.25 | **0.61** |
| | ITM Score | 85.87 | 86.41 | 88.58 | 83.85 | 89.03 | |
| Cityscapes | mIoU | 40.06 | 41.45 | 44.21 | 43.72 | 47.33 | **0.72** |
| | ITM Score | 79.77 | 79.14 | 83.11 | 78.03 | 83.86 | |

ITM achieves $\tau_w = 0.61$ and $\tau_w = 0.72$ on the two segmentation datasets respectively, correctly identifying the best model MAE-L16, thereby validating its cross-task generalization.

### Ablation Study

| Loss Function | Avg. $\tau_w$ |
|---------------|---------------|
| Cross-Entropy | 0.554 |
| MAE | 0.566 |
| **MSE** | **0.608** |

MSE loss performs best within the deparametric approximation framework; its smoothness facilitates stable pseudo-cluster updates. A batch size of 256 and 500 training iterations constitute the optimal configuration.

## Highlights & Insights

- **Novel implicit modeling paradigm**: Transferability is encoded as a latent variable $z$; learnable parameters $W_z$ adaptively capture the interaction between intrinsic model and task properties, rather than relying on hand-crafted evolution rules.
- **Elegant derivation of deparametric approximation**: By analytically expanding gradient descent in closed form, the need for explicit optimization of mapping layer parameters is eliminated, substantially reducing computational cost while maintaining estimation accuracy.
- **Cross-architecture and cross-task generalization**: ITM performs consistently on a mixed pool of CNNs and ViTs with supervised/contrastive/MIM pre-training strategies, and successfully generalizes to semantic segmentation tasks.
- **Significant performance gain**: Average $\tau_w$ improves from 0.45 to 0.61, with low variance ($0.60 \pm 0.01$) demonstrating strong robustness.

## Limitations & Future Work

- **Reliance on embedding discriminability**: The method cannot directly handle complex supervision scenarios such as object detection or vision–language tasks.
- **Dependence on final-layer features only**: Intermediate layer representations and rich output embeddings are not exploited for modeling intrinsic model properties.
- **Evaluation limited to full fine-tuning**: Transferability estimation under PEFT paradigms (LoRA, VPT, etc.) is not considered, whereas adaptation characteristics under PEFT may differ substantially from full fine-tuning.
- **Coarse approximation of pseudo-cluster centers**: Replacing the true post-fine-tuning embedding distribution with preset cluster centers may introduce bias for tasks with a large number of categories or high inter-class similarity.
- **Limited model pool size**: Experiments use only 10–14 models; scalability to large-scale model libraries (e.g., 100+ models) has not been validated.

## Related Work & Insights

- **LogME**: A static method based on maximum log evidence; the fastest to compute (1.93s) but limited in accuracy ($\tau_w = 0.45$), ignoring fine-tuning dynamics.
- **LEAD**: Simulates logit space evolution via ordinary differential equations; representative of dynamic methods, but relies on LogME's evaluation pipeline and generalizes poorly to diverse models.
- **PED**: Predicts evolved states using an energy-based model; exhibits negative correlation on some datasets (Pets: −0.20, Aircraft: −0.20), indicating poor generalization.
- **SA (LDA)**: Perturbs the feature space via diffusion and attraction operations; average $\tau_w$ of only 0.16, yielding incorrect rankings on multiple datasets.
- **ETran**: An energy-based metric with average $\tau_w = 0.14$, with negative correlation on CIFAR100 and SUN.
- **ITM (Ours)**: The implicit modeling paradigm removes the dependence on specific evolution assumptions; $\tau_w > 0.4$ on all 10 datasets, making it the only method that achieves positive rank correlation across all tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of implicit modeling and deparametric approximation is a novel framework design that advances TE from explicit rule-based to probabilistic modeling.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across 10 datasets and 10 models, including stability experiments, ablation studies, and generalization to segmentation tasks.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with coherent mathematical derivations; the logical flow from problem formulation to framework design is well-organized.
- **Value**: ⭐⭐⭐⭐ — Proposes a more generalizable TE paradigm with practical significance for large-scale model selection, though the PEFT scenario remains unaddressed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models](../../ICCV2025/self_supervised/loftup_learning_a_coordinatebased_feature_upsampler_for_visi.md)
- [\[CVPR 2026\] Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models](../../CVPR2026/self_supervised/com_pt_chain_of_models_pretraining.md)
- [\[CVPR 2026\] Robustness of Vision Foundation Models to Common Perturbations](../../CVPR2026/self_supervised/robustness_of_vision_foundation_models_to_common_perturbations.md)
- [\[NeurIPS 2025\] One Filters All: A Generalist Filter for State Estimation](one_filters_all_a_generalist_filter_for_state_estimation.md)
- [\[NeurIPS 2025\] Foundation Models for Scientific Discovery: From Paradigm Enhancement to Paradigm Transition](foundation_models_for_scientific_discovery_from_paradigm_enhancement_to_paradigm.md)

</div>

<!-- RELATED:END -->
