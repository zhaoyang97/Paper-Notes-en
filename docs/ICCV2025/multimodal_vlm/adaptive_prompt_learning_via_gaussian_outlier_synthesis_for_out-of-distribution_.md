---
title: >-
  [Paper Note] Adaptive Prompt Learning via Gaussian Outlier Synthesis for Out-of-Distribution Detection
description: >-
  [ICCV 2025][Multimodal VLM][OOD detection] This paper proposes the APLGOS framework, which initializes learnable in-distribution (ID) prompts using ChatGPT-standardized Q&A pairs…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "OOD detection"
  - "prompt learning"
  - "Gaussian outlier synthesis"
  - "vision-language model"
  - "contrastive learning"
date: 2026-05-08
content_hash: 6415daaf9a0feae5
---

# Adaptive Prompt Learning via Gaussian Outlier Synthesis for Out-of-Distribution Detection

**Conference**: ICCV 2025
**arXiv**: N/A (CVF OpenAccess)  
**Code**: Unavailable  
**Area**: Multimodal VLM
**Keywords**: OOD detection, prompt learning, Gaussian outlier synthesis, vision-language model, contrastive learning

## TL;DR

This paper proposes the APLGOS framework, which initializes learnable in-distribution (ID) prompts using ChatGPT-standardized Q&A pairs, synthesizes virtual OOD prompts and images by sampling from the low-likelihood regions of class-conditional Gaussian distributions, and aligns text-image embeddings via contrastive learning to achieve more compact ID/OOD decision boundaries.

## Background & Motivation

### Core Problem
Deep learning models trained on a limited set of known classes (ID) tend to produce high-confidence predictions for unknown classes (OOD) encountered during deployment, posing serious risks in safety-critical applications such as autonomous driving. The goal of OOD detection is to enable models not only to accurately recognize known classes but also to flag unknown inputs as "uncertain."

### Limitations of Prior Work

**Extraction-based methods**: These methods extract pseudo-OOD samples directly from ID data to regularize the model, but the quality of extraction is difficult to control and large amounts of ID data are required.

**Synthesis-based methods**: These methods synthesize OOD RGB images directly or generate virtual outliers in low-dimensional latent spaces, partially alleviating the above issues, yet the quality of synthesis remains a concern.

**Key Gap**: No prior work has incorporated **prompt learning** into OOD detection tasks, leaving the rich pre-trained knowledge and representational power of VLMs underutilized.

### Design Motivation

**Why prompt learning?** Vision-language models (VLMs) possess strong pre-trained knowledge and cross-modal alignment capabilities. By designing appropriate text prompts, one can more effectively establish discrimination boundaries between ID and OOD distributions in the latent space. **Why sample from the low-likelihood regions of Gaussian distributions?** The true distribution of OOD data is unknown, but OOD samples are likely to appear in low-density regions of the ID distribution. Synthesizing virtual OOD prompts in the low-likelihood regions of class-conditional Gaussian distributions therefore better approximates the true OOD distribution.

## Method

### Overall Architecture

APLGOS consists of two core modules:
- **PLM (Prompt Learning Module)**: Responsible for generating ID prompts and synthesizing virtual OOD prompts.
- **TAM (Text-Image Alignment Module)**: Computes text-image similarity scores and aligns multimodal data via contrastive learning.

The training process proceeds in three stages:
1. **Stage 1**: ChatGPT-standardized Q&A pairs are used to generate a sentence set.
2. **Stage 2**: ID prompts and ID images are used for alignment training.
3. **Stage 3**: Synthesized OOD prompts and OOD images are jointly incorporated into training.

**Core Highlight**: Only ID images are drawn from real datasets; ID prompts, OOD prompts, and OOD images are all virtually synthesized.

### Key Designs

#### 1. ID Prompt Generation

Conventional methods rely on a single fixed prompt (e.g., "a photo of a \<CLS\>"), which has limited expressive capacity. APLGOS introduces a richer prompting strategy:

**Predefined Q&A pairs**: `Q: What is in the region with coordinates <loc1>,<loc2>,<loc3>,<loc4>? A: That's a <CLS>.`

**Multi-round ChatGPT standardization**:
$$\Omega_0 = g(QA + M + G_0), \quad \Omega_i = g(\Omega_{i-1} + G_i)$$

where $g$ denotes the ChatGPT standardization operation, $M$ is a predefined template, and $G_i$ is the guiding instruction at round $i$. After $t$ rounds, a sentence set $\Omega_t$ is obtained.

**Learnable prompt structure**: `<loc1><loc2><loc3><loc4><V1><V2>...<Vm><CLS>`
- `<loc>` tokens are learnable position tokens that implicitly encode spatial information.
- `<V>` tokens are learnable descriptive tokens.
- `<CLS>` is replaced by the class label of the current region.

**Why this design?** Introducing coordinate tokens enables finer-grained region-level observation; sampling from the sentence set for initialization provides prompt diversity and avoids the representational bottleneck caused by a single fixed prompt.

#### 2. OOD Prompt Synthesis (Core Contribution)

**Class-conditional Gaussian assumption**: ID prompt embeddings are assumed to follow a class-conditional multivariate Gaussian distribution:
$$p_\theta(\hat{T}|y=i) = \mathcal{N}(\hat{\mu}_i, \hat{\sigma})$$

**Empirical class-conditional mean**:
$$\hat{\mu}_i = \frac{1}{|Q_T|} \sum_{j=1}^{|Q_T|} \hat{T}_{i,j}$$

**Tied covariance matrix** (key formula):
$$\hat{\sigma} = \frac{1}{K|Q_T|} \sum_{i=1}^{K} \sum_{j=1}^{|Q_T|} (\hat{T}_{i,j} + \alpha\varepsilon - \hat{\mu}_i)(\hat{T}_{i,j} + \alpha\varepsilon - \hat{\mu}_i)^T + \beta E$$

where $\varepsilon$ is a learnable matrix initialized with random Gaussian noise, $\alpha$ controls noise intensity, and $\beta E$ provides regularization.

**Sampling virtual OOD prompts from low-likelihood regions**:
$$V_i = \Psi(\hat{T}, \hat{\mu}_i, \hat{\sigma})$$

where $\Psi$ denotes the class-conditional Gaussian probability density function. The top-$k$ prompts with the lowest probability are selected as virtual OOD pseudo-prompts $\hat{T}^\dagger$.

**Why the low-likelihood region?** The low-likelihood region lies at the periphery of the ID class distribution and is precisely where OOD data is most likely to appear. Synthesizing virtual prompts in this region effectively regularizes the decision boundary, encouraging the model to learn more compact classification boundaries.

#### 3. Virtual OOD Image Synthesis

The procedure mirrors OOD prompt synthesis, with ID image embeddings substituted for ID prompt embeddings. An image embedding queue $Q_I$ is used to compute the empirical Gaussian mean and covariance, and virtual OOD images $\hat{X}^\dagger$ are sampled from the low-likelihood region.

#### 4. Text-Image Alignment Module (TAM)

Similarity scores between normalized prompt embeddings and image embeddings are computed as:
$$S = \frac{\|\hat{X}\|_p (\|\hat{T}\|_p)^T}{e^\omega}$$

where $\omega$ is a scaling hyperparameter. ID data is used in Stage 2, while synthesized OOD data is incorporated in Stage 3.

### Loss & Training

The total loss comprises multiple components:

$$\mathcal{L} = \xi_1[\gamma_1 \tau \mathcal{L}_{align}^{id} + \gamma_2(1-\tau)\mathcal{L}_{align}^{ood}] + \gamma_3 \xi_2[\kappa \mathcal{L}_{loc}^{id} + (1-\kappa)\mathcal{L}_{loc}^{ood}] + \gamma_4 \xi_3 \mathcal{L}_{cls} + \gamma_5 \xi_4 \mathcal{L}_{reg} + W$$

- **Alignment loss $\mathcal{L}_{align}$**: Contrastive learning loss based on similarity scores, treating all OOD classes as a single "background" class.
- **Localization loss $\mathcal{L}_{loc}$**: Implicitly encodes spatial information, enabling region-level granularity in the prompts.
- **Classification loss $\mathcal{L}_{cls}$** and **regression loss $\mathcal{L}_{reg}$**: Standard detection losses.
- **Regularization term $W$**: Further regularizes the model.

Training stage control is achieved via $\xi$, $\tau$, and $\kappa$, which govern which loss components are active at each stage.

## Key Experimental Results

### Main Results

| ID Dataset | Method | FPR95↓ (COCO/OI) | AUROC↑ (COCO/OI) | mAP↑ |
|-----------|------|-------------------|-------------------|------|
| PASCAL VOC | VOS-RegX4.0 | 50.53 / 50.27 | 88.10 / 87.08 | 49.1 |
| PASCAL VOC | **APLGOS (RegX4.0)** | **45.96 / 47.10** | **89.19 / 88.49** | **49.4** |
| BDD-100k | VOS-ResNet50 | 46.97 / 31.25 | 84.97 / 89.82 | 35.7 |
| BDD-100k | **APLGOS (ResNet50)** | **41.10 / 23.30** | **87.36 / 92.87** | **35.8** |
| BDD-100k | VOS-RegX4.0 | 42.82 / 27.55 | 86.36 / 92.11 | 37.0 |
| BDD-100k | **APLGOS (RegX4.0)** | **39.48 / 19.79** | **87.47 / 93.59** | **37.6** |

On the BDD-100k + OpenImages setting, APLGOS reduces FPR95 from 27.55% to **19.79%**, a reduction of 7.76%.

### Ablation Study

**Prompt strategy ablation** (PASCAL VOC + COCO/OI):

| Strategy | FPR95↓ | AUROC↑ | mAP↑ |
|------|--------|--------|------|
| (a) VOS-RegX4.0 baseline | 50.53 / 50.27 | 88.10 / 87.08 | 49.1 |
| (b) \<CLS\> only | 50.12 / 49.50 | 88.56 / 86.83 | 48.2 |
| (c) "a region of a" + \<CLS\> | 51.31 / 50.96 | 88.20 / 86.73 | 48.7 |
| (d) ChatGPT-sampled prompt + \<CLS\> | 49.50 / 49.40 | 88.49 / 86.73 | 48.9 |
| (e) \<LOC\> + "a region of a" + \<CLS\> | 49.56 / 47.60 | 88.23 / 87.07 | 49.1 |
| (f) **Full APLGOS** | **45.96 / 47.10** | **89.19 / 88.49** | **49.4** |

**Gaussian noise intensity $\alpha$ ablation**:

| $\alpha$ | FPR95↓ (COCO/OI) | AUROC↑ (COCO/OI) | mAP↑ |
|----------|-------------------|-------------------|------|
| 0 | 51.63 / 50.88 | 87.86 / 87.24 | 49.2 |
| 0.5 | 51.90 / 51.48 | 87.55 / 87.02 | 48.9 |
| **1.0** | **45.96 / 47.10** | **89.19 / 88.49** | **49.4** |
| 1.5 | 55.88 / 53.33 | 86.29 / 86.75 | 48.9 |
| 2.0 | 55.92 / 49.54 | 86.75 / 88.00 | 48.9 |

### Key Findings

1. **Sentence-set sampling > fixed prompts**: Comparison (c) vs. (d) shows that sampling from a diverse sentence set for prompt initialization is more effective than using a fixed template.
2. **Location tokens are critical**: Adding \<LOC\> in (c) vs. (e) reduces FPR95 from 50.96% to 47.60%.
3. **An optimal noise intensity exists**: $\alpha=1.0$ yields the best performance; too small a value narrows the OOD sampling space excessively, while too large a value expands it beyond effective regularization.
4. **ID:OOD = 1:1 is optimal**: Unlike the baseline VOS, which uses a 2:1 ratio, APLGOS achieves best performance at 1:1, suggesting higher quality of synthesized OOD samples.
5. **Optimal number of OOD prompts is K=10,000**: Too few fails to adequately cover the latent space; too many introduces excessive randomness.

## Highlights & Insights

1. **Fully virtualized design**: ID prompts, OOD prompts, and OOD images are all synthetically generated; only ID images are drawn from real data, reducing dependence on large amounts of real ID data.
2. **Multi-round ChatGPT standardization**: LLMs are leveraged to generate diverse region-level descriptions, ensuring semantic consistency while enhancing expressive diversity.
3. **Low-likelihood region sampling strategy**: Theoretically well-motivated — OOD data is expected to appear in the low-density regions of the ID distribution.
4. **Implicit spatial encoding in prompts**: Learnable \<LOC\> tokens encode region coordinates, endowing prompts with spatial awareness.
5. **Real-world generalization**: The method demonstrates strong detection performance on real photographs captured with an iPhone 14 Pro Max.

## Limitations & Future Work

1. **Gaussian distribution assumption**: The framework assumes that ID prompt embeddings follow a Gaussian distribution; the true distribution may be more complex. Mixture-of-Gaussians or normalizing flow models could be explored.
2. **ChatGPT dependency**: Standardization relies on ChatGPT-3.5; the quality of standardization may vary across different LLMs.
3. **Scalability**: For scenarios with a large number of classes (e.g., hundreds), the accuracy of per-class Gaussian distribution estimation may degrade.
4. **Task specificity**: The framework is specifically designed for OOD detection in object detection; its applicability to classification, segmentation, and other tasks warrants further investigation.
5. **Covariance estimation**: The use of a tied covariance matrix simplifies computation but may sacrifice inter-class distributional differences.

## Related Work & Insights

- **VOS (ICLR 2022)**: The direct baseline of this work, which synthesizes virtual outliers in the feature space.
- **CoOp/CoCoOp**: Classical prompt learning methods that inspired the prompt design of APLGOS, though the latter is specifically tailored for OOD detection.
- **CLIP**: Provides pre-trained vision-language alignment capabilities; APLGOS leverages its text encoder.
- **Insight**: Combining LLMs (ChatGPT) with prompt learning for OOD detection represents a novel direction; future work could explore broader LLM-assisted detection paradigms.

## Rating
- Novelty: ⭐⭐⭐⭐ (First to introduce prompt learning into OOD detection; ChatGPT-assisted prompt standardization is creative)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Four datasets, comprehensive ablation studies, real-world scene evaluation)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, detailed mathematical derivations)
- Value: ⭐⭐⭐⭐ (The combination of OOD detection and VLMs has practical value; a 7.76% reduction in FPR95 is significant)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FA: Forced Prompt Learning of Vision-Language Models for Out-of-Distribution Detection](fa_forced_prompt_learning_of_vision-language_models_for_out-of-distribution_dete.md)
- [\[ICCV 2025\] Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection](exploiting_vision_language_model_for_training-free_3d_point_cloud_ood_detection_.md)
- [\[ICCV 2025\] PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation](pro-vpt_distribution-adaptive_visual_prompt_tuning_via_prompt_relocation.md)
- [\[ICCV 2025\] Advancing Textual Prompt Learning with Anchored Attributes](advancing_textual_prompt_learning_with_anchored_attributes.md)
- [\[ICCV 2025\] NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection](negrefine_refining_negative_label-based_zero-shot_ood_detection.md)

</div>

<!-- RELATED:END -->
