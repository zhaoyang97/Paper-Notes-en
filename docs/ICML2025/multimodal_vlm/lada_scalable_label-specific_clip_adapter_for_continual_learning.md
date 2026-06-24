---
title: >-
  [Paper Note] LADA: Scalable Label-Specific CLIP Adapter for Continual Learning
description: >-
  [ICML2025][Multimodal VLM][CLIP] This paper proposes LADA (Label-specific ADApter), which appends lightweight **class-specific memory vectors** after the frozen CLIP image encoder to concentrate the discriminative information of all learned tasks into a unified feature space. This **completely eliminates the parameter selection step during inference** and achieves SOTA performance under the X-TAIL continual learning setting.
tags:
  - "ICML2025"
  - "Multimodal VLM"
  - "CLIP"
  - "Continual Learning"
  - "Label-Specific Adapter"
  - "Feature Distillation"
  - "Class-Incremental Learning"
  - "Cross-Domain Incremental Learning"
date: 2026-05-08
content_hash: 158a7e6e443761a6
---

# LADA: Scalable Label-Specific CLIP Adapter for Continual Learning

**Conference**: ICML2025  
**arXiv**: [2505.23271](https://arxiv.org/abs/2505.23271)  
**Code**: [MaolinLuo/LADA](https://github.com/MaolinLuo/LADA)  
**Area**: CLIP Continual Learning / Parameter-Efficient Fine-Tuning  
**Keywords**: CLIP, Continual Learning, Label-Specific Adapter, Feature Distillation, Class-Incremental Learning, Cross-Domain Incremental Learning

## TL;DR

This paper proposes LADA (Label-specific ADApter), which appends lightweight **class-specific memory vectors** after the frozen CLIP image encoder to concentrate the discriminative information of all learned tasks into a unified feature space. This **completely eliminates the parameter selection step during inference** and achieves SOTA performance under the X-TAIL continual learning setting.

## Background & Motivation

### Problem Definition

Cross-Domain Task-Agnostic Incremental Learning (X-TAIL): The model sequentially learns from $K$ tasks from different domains $\{\mathcal{T}^1, \dots, \mathcal{T}^K\}$. During inference, **no task ID is provided**, requiring the model to simultaneously recognize both learned and unseen classes.

### Limitations of Prior Work

**Prompt-based methods** (L2P, DualPrompt, S-Prompts): Require **selecting** the corresponding prompt from a prompt pool during inference $\rightarrow$ incorrect selection directly degrades performance.

**MoE-Adapter** (Yu et al. 2024): Predefines the number of adapters $\rightarrow$ requires knowing the total number of tasks; also requires **selecting** which adapters to activate during inference.

**Full-parameter fine-tuning** (ZSCL): Updates pre-trained parameters $\rightarrow$ suffers from severe forward forgetting and degradation of zero-shot generalization.

**Classifier expansion** (RAIL): Relies on the original CLIP to judge whether a task has been learned $\rightarrow$ propagates errors.

Key Challenge: **Parameter selection**—existing methods require an extra step during inference to decide which set of parameters to use for feature extraction, which is inherently error-prone.

### Design Motivation

Designing a **parameter-selection-free** adapter: condensing all task-specific discriminative information into unified label-specific features, and directly using all memory vectors during inference to eliminate the selection step.

## Method

### Overall Architecture

LADA consists of two core modules:

1. **Text encoder fine-tuning framework**: Freezes the text features of old tasks and optimizes only the current task's text features as the classifier.
2. **Scalable Label-Specific Adapter**: Appends class-specific memory vectors after the frozen CLIP image encoder.

### Module 1: Text Encoder Fine-Tuning

For the current task $\mathcal{T}^k$, the text features $\boldsymbol{t}$ of old tasks $\mathcal{T}^1, \dots, \mathcal{T}^{k-1}$ are frozen, and only the text features of $\mathcal{T}^k$ are updated. The classification loss is defined as:

$$\mathcal{L}(\boldsymbol{t}; k) = \sum_{i=1}^{k} \sum_{j=1}^{M^i} \hat{\mathcal{L}}(\boldsymbol{t}; k, i, j)$$

For the current task: The softmax cross-entropy is calculated using real images (Eq.3).

For old tasks: Real images are inaccessible, so they are substituted by **$\lambda$ clustering centers** (distillation prototypes $\boldsymbol{p}_j^i$) (Eq.4). The classification loss is calculated using the dot product between the prototypes and the text features of all classes.

### Module 2: Label-Specific Adapter (Core Innovation)

**Step 1: Constructing label-specific features**

For the $j$-th class of task $\mathcal{T}^k$, k-means clustering is applied to the image features of that class to obtain $\lambda_1$ clustering centers $\boldsymbol{W}_j^k \in \mathbb{R}^{\lambda_1 \times d}$.

Define the label-specific feature mapping:

$$\varphi^k(\boldsymbol{i}) = [\boldsymbol{W}_1^k \boldsymbol{i}, \dots, \boldsymbol{W}_{M^k}^k \boldsymbol{i}]$$

Features of all tasks are concatenated into the final representation $\varphi(\boldsymbol{i}) = [\varphi^1(\boldsymbol{i}), \cdots, \varphi^k(\boldsymbol{i})]$, allowing the **features to naturally scale as tasks increase**.

**Step 2: Fixed classifier**

Uses a neighborhood-based fixed classifier $h$ to map label-specific features to classification logits:

$$(h \circ \varphi)(\boldsymbol{i})_j^i = \phi(\boldsymbol{W}_j^i \boldsymbol{i}) \cdot \boldsymbol{1}$$

where $\phi(x) = \exp(-\beta(1-x))$ converts the dot product into a non-negative value, and $\beta$ controls the sharpness.

**Step 3: Training Strategy**

- Freezes old task parameters $\boldsymbol{W}^1, \dots, \boldsymbol{W}^{k-1}$, and updates only $\boldsymbol{W}^k$.
- Current task: Directly calculates softmax cross-entropy using real samples (Eq.7).
- Old tasks: Calculates classification loss using distillation prototypes (Eq.8) to prevent parameters of new classes from interfering with the classification of old classes.

**Step 4: Distribution-Preserved Training**

Using only clustering centers is insufficient, so **GMM (Gaussian Mixture Model)** is employed to fit the feature distribution of each class in old tasks:

$$\{\pi_j^i(l), \boldsymbol{p}_j^i(l), \boldsymbol{\Sigma}_j^i(l)\}_{l=1}^{\lambda_2} = \text{GMM}(\mathcal{D}_j^i)$$

Augmenting prototypes:

$$\tilde{\boldsymbol{p}}_j^i(l) = \boldsymbol{p}_j^i(l) + \boldsymbol{e} \cdot \sqrt{\frac{\text{Tr}(\boldsymbol{\Sigma}_j^i(l))}{d}}$$

where $\boldsymbol{e}$ represents Gaussian noise, with its variance controlled by the trace of the covariance matrix. Distribution-preserved training is performed on old tasks using a weighted loss (Eq.10).

### Key Designs

| Design | Function |
|------|------|
| Memory vectors placed **after** the image encoder | No backpropagation to the CLIP encoder, leading to highly efficient training |
| Freezing old task $\boldsymbol{W}$ | Inherently prevents backward forgetting |
| All memory vectors involved in inference | Eliminates parameter selection and avoids misallocation |
| GMM + noise-augmented prototypes | Preserves old task distribution information with minimal storage |

## Key Experimental Results

### X-TAIL 16-shot Setting

| Method | Transfer | Average | Last |
|------|----------|---------|------|
| Zero-shot CLIP | – | 57.7 | – |
| ZSCL | 59.0 | 60.0 | 63.4 |
| MoE-Adapters | 56.0 | 63.0 | 70.5 |
| Dual-RAIL | – | 71.3 | 82.3 |
| **LADA (Ours)** | **61.5** | **72.7** | **83.1** |

### X-TAIL Full-shot Setting

LADA also surpasses all baselines in the full-shot setting, achieving SOTA across Transfer, Average, and Last metrics.

### Key Findings

- Transfer metric (generalization to unseen tasks): LADA maintains a generalization capability close to zero-shot CLIP, significantly outperforming MoE-Adapters (61.5 vs. 56.0).
- Last metric (final performance): LADA reaches 83.1% after learning 10 datasets, which is 0.8% higher than Dual-RAIL.
- Average metric (mean performance during learning): 72.7%, which is 1.4% higher than Dual-RAIL.

## Highlights & Insights

1. **Eliminating parameter selection in the inference stage**: This is the core breakthrough compared to prompt-based and MoE methods—there is no need to "guess which set of parameters to use" beforehand; all memory vectors are jointly involved.
2. **Excellent scalability**: Each new task only adds $M^k \times \lambda_1$ vectors (a few clustering centers per class), resulting in lightweight, linear parameter growth.
3. **Efficient training**: The adapter is located after the image encoder, so gradients do not propagate back to CLIP $\rightarrow$ low GPU memory footprint.
4. **GMM distribution preservation**: Compressing the feature distribution of old tasks into GMM parameters (means + covariances + weights) is far more efficient than storing raw features or images.
5. **Simple and clean design**: Contains no complex routing mechanisms, gating networks, or attention modules, operating purely based on dot products and k-means/GMM.

## Limitations & Future Work

1. **Storage grows linearly with the number of classes**: Each class requires $\lambda_1$ memory vectors + $\lambda_2$ GMM components; when the number of classes is extremely large (e.g., 10,000 classes), storage might become a bottleneck.
2. **GMM assumption**: Fitting the feature distribution with GMM assumes local Gaussianity, which may not be accurate enough for highly non-convex distributions.
3. **Dependence on frozen-CLIP feature quality**: If CLIP's pre-trained representation itself is poor for certain domains (e.g., medical images), the potential improvement of LADA is limited.
4. **Lack of validation on ultra-large-scale tasks**: Experiments cover sequential learning of up to 10 tasks, whereas practical scenarios may involve hundreds of tasks.
5. **Fixed classifier**: The classifier $h$ based on the design of $\phi = \exp(-\beta(1-x))$ is relatively simple, which may limit expressiveness.

## Related Work & Insights

- **RAIL / Dual-RAIL** (Xu et al. 2024): Expands classifier dimensions but freezes features $\rightarrow$ complementary to LADA, as LADA improves the feature side.
- **MoE-Adapters** (Yu et al. 2024): Blends adapters but requires parameter selection $\rightarrow$ LADA eliminates this step.
- **ZSCL** (Zheng et al. 2023): Knowledge distillation to prevent forgetting but updates pre-trained parameters $\rightarrow$ LADA completely freezes CLIP.
- **Prototype augmentation** (Zhang et al. 2023): Uses prototypes instead of storing samples $\rightarrow$ LADA's GMM distribution preservation is an upgraded version of this.
- Insight: The paradigm of **encoding class-discriminative information into learnable vectors (memory units)** is worth generalizing to other continual learning scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea of using label-specific memory units to eliminate parameter selection is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive comparison on X-TAIL 16-shot/full-shot, with thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, complete mathematical derivations, and intuitive illustrations.
- Value: ⭐⭐⭐⭐ — Practically advances the field of CLIP continual learning; the method is plug-and-play.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning](../../AAAI2026/multimodal_vlm/harnessing_textual_semantic_priors_for_knowledge_transfer_and_refinement_in_clip.md)
- [\[ICLR 2026\] CLIP-FMoE: Scalable CLIP via Fused Mixture-of-Experts with Enforced Specialization](../../ICLR2026/multimodal_vlm/clip-fmoe_scalable_clip_via_fused_mixture-of-experts_with_enforced_specializatio.md)
- [\[CVPR 2026\] PACT: Phase-Like Transition Constraints in Adapter-Based Continual Learning of Vision-Language Models](../../CVPR2026/multimodal_vlm/pact_phase-like_transition_constraints_in_adapter-based_continual_learning_of_vi.md)
- [\[NeurIPS 2025\] Continual Multimodal Contrastive Learning](../../NeurIPS2025/multimodal_vlm/continual_multimodal_contrastive_learning.md)
- [\[CVPR 2025\] NLPrompt: Noise-Label Prompt Learning for Vision-Language Models](../../CVPR2025/multimodal_vlm/nlprompt_noise-label_prompt_learning_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
