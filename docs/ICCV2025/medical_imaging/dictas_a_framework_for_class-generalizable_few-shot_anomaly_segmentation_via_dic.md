---
title: >-
  [Paper Note] DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation via Dictionary Lookup
description: >-
  [ICCV 2025][Medical Imaging][Anomaly Detection] Inspired by the intuition of human inspectors "consulting a dictionary," DictAS reformulates few-shot anomaly segmentation (FSAS) as a dictionary lookup task—a query feature is deemed anomalous if it cannot be retrieved from a dictionary of normal samples. Through self-supervised training, the framework acquires class-agnostic lookup capability and achieves state-of-the-art FSAS performance and inference speed across 7 industrial and medical datasets.
tags:
  - ICCV 2025
  - Medical Imaging
  - Anomaly Detection
  - Few-Shot Anomaly Segmentation
  - Dictionary Lookup
  - CLIP
  - Self-Supervised Learning
date: 2026-05-08
content_hash: ac3f06b905516f2b
---

# DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation via Dictionary Lookup

**Conference**: ICCV 2025
**arXiv**: [2508.13560](https://arxiv.org/abs/2508.13560)
**Code**: [github.com/xiaozhen228/DictAS](https://github.com/xiaozhen228/DictAS)
**Area**: Medical Imaging
**Keywords**: Anomaly Detection, Few-Shot Anomaly Segmentation, Dictionary Lookup, CLIP, Self-Supervised Learning

## TL;DR

Inspired by the intuition of human inspectors "consulting a dictionary," DictAS reformulates few-shot anomaly segmentation (FSAS) as a dictionary lookup task—a query feature is deemed anomalous if it cannot be retrieved from a dictionary of normal samples. Through self-supervised training, the framework acquires class-agnostic lookup capability and achieves state-of-the-art FSAS performance and inference speed across 7 industrial and medical datasets.

## Background & Motivation

### Problem Definition

Few-Shot Anomaly Segmentation (FSAS) aims to identify anomalous regions in query images given only a small number of normal reference samples. This is particularly important in industrial defect detection and medical image analysis, where training data is scarce and pixel-level annotations are limited.

FSAS encompasses two settings:
- **Class-dependent**: A separate model is fine-tuned for each unseen class using its normal samples.
- **Class-generalizable**: A unified model is trained once and applied to unseen classes without retraining, using only a few normal samples as visual prompts.

This paper focuses on the more challenging **class-generalizable** setting.

### Limitations of Prior Work

**RegAD**: Introduces feature registration for alignment but suffers from low inference efficiency due to the need for extensive reference image augmentation.

**FastRecon**: Employs linear regression for feature reconstruction but is prone to over-reconstruction, where both normal and anomalous features are reconstructed equally well.

**CLIP-based methods** (WinCLIP, APRIL-GAN): Leverage CLIP's vision-language alignment and memory banks for visual priors. However, these methods rely on empirical knowledge from real anomaly samples seen during auxiliary training, limiting generalization to entirely new categories.

**PromptAD**: Achieves strong performance but is a class-dependent method requiring fine-tuning for each unseen class, making it non-scalable.

### Core Motivation

**Key Insight**: Even a novice inspector can detect anomalies in unseen categories after examining only a few normal samples—without requiring extensive prior experience. This process resembles consulting a dictionary: if a query region can be matched to a normal pattern in the dictionary, it is normal; otherwise, it is anomalous.

Motivated by this intuition, DictAS reformulates FSAS as a dictionary lookup task. Rather than memorizing normal/anomalous patterns from the training set, the model acquires class-agnostic lookup capability through self-supervised learning.

## Method

### Overall Architecture

DictAS is built upon CLIP (ViT-L-14-336) and consists of three core components:
1. **Dictionary Construction**: Builds a structured dictionary from normal reference images.
2. **Dictionary Lookup**: Retrieves query region features from the dictionary via a sparse lookup strategy.
3. **Query Discrimination Regularization**: Enhances the discriminability between anomalous and normal retrieval results.

### Key Designs

#### 1. Dictionary Construction

- **Function**: Organizes features from normal reference images into a structured dictionary with Keys and Values, and generates corresponding Query vectors for query images.
- **Mechanism**:

  Three independent AttnBlocks (self-attention Transformer blocks) are used to generate the Dictionary Query, Key, and Value, respectively:

  $$\mathbf{F}_Q^l = g_Q(\mathbf{F}_q^l) = AttnBlock\_Q(\mathbf{F}_q^l)$$
  $$\mathbf{F}_K^l = g_K(\mathbf{F}_n^l) = AttnBlock\_K(\mathbf{F}_n^l)$$
  $$\mathbf{F}_V^l = g_V(\mathbf{F}_n^l) = \mathbf{F}_n^l + AttnBlock\_V(\mathbf{F}_n^l)$$

  where $\mathbf{F}_q^l \in \mathbb{R}^{HW \times C}$ denotes the $l$-th layer features of the query image, and $\mathbf{F}_n^l \in \mathbb{R}^{kHW \times C}$ denotes the concatenated features from $k$ reference images. A residual connection is added in the Value Generator to preserve fine-grained normal feature details.

  Each AttnBlock applies multi-head self-attention followed by a two-layer MLP:

  $$\mathbf{F}_{out} = TwoLayerMLP(softmax(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{C}})\mathbf{V})$$

- **Design Motivation**: Self-attention enables each patch to capture global context, improving the robustness of dictionary construction. Separate transformations for Keys and Values decouple the indexing and content functions, analogous to the distinction between "entries" and "definitions" in a real dictionary.

#### 2. Dictionary Lookup

- **Function**: For each patch feature in the query image, the most relevant normal pattern is retrieved from the dictionary. A large retrieval distance indicates an anomaly.
- **Mechanism**:

  The lookup proceeds in two steps: Query-Key matching computes similarity $\mathbf{z} = \mathbf{x}_Q^l \mathbf{F}_K^{lT}$, followed by weighted aggregation of Dictionary Values to obtain the retrieval result $\mathbf{x}_r^l = \hat{\mathbf{w}} \mathbf{F}_V^l$.

  Three lookup strategies are proposed:
  - **Maximum Lookup**: Selects the Value with maximum similarity (one-hot weighting).
  - **Dense Lookup**: Aggregates all Values via softmax weighting.
  - **Sparse Lookup** (default): Sparsifies the weights through a Sparse Probability Module (SPM).

  SPM solves the following constrained optimization problem:

  $$\arg\min_{\triangle} \frac{1}{2} \|\mathbf{w} - \mathbf{z}\|^2, \quad \triangle = \{\mathbf{w} | \mathbf{w}_u \geq 0, \sum_{u=1}^{kHW} \mathbf{w}_u = 1\}$$

  The solution is $\hat{\mathbf{w}}_u = \max(\mathbf{z}_u - \tau, 0)$, where $\tau$ is a dynamic threshold adaptively determined via sorting and cumulative summation, automatically selecting the most relevant Values while suppressing redundancy.

  **Query Loss** (core self-supervised training objective):

  $$\mathcal{L}_q = \sum_l \frac{1}{|\mathcal{N}|} \sum_{j \in \mathcal{N}} d(\mathbf{F}_{q,j}^l, \mathbf{F}_{r,j}^l)$$

  where $d$ denotes cosine distance and $\mathcal{N}$ is the index set of normal regions. The loss minimizes the query-retrieval distance only for normal regions, naturally preserving large distances for anomalous regions.

- **Design Motivation**: Sparse Lookup avoids the pitfall of Dense Lookup, where uniform participation of all Values can produce plausible retrieval results even for anomalous regions, while remaining more flexible than Maximum Lookup. As the number of reference images grows, sparsity suppresses redundancy, making DictAS increasingly advantageous at higher shot counts.

#### 3. Query Discrimination Regularization

- **Function**: Amplifies the distance difference between anomalous and normal retrieval results, preventing over-retrieval where anomalies are also successfully reconstructed.
- **Mechanism**:

  **Contrastive Query Constraint (CQC)**: Enforces that the query-retrieval distance for anomalous regions exceeds that of normal regions:

  $$\mathcal{L}_{CQC} = \sum_l \max(0, \mathbb{E}_\mathcal{N}[d] - \mathbb{E}_\mathcal{A}[d])$$

  **Text Alignment Constraint (TAC)**: Leverages CLIP's vision-language alignment to constrain the global retrieval result toward the "normal" text embedding space:

  $$\mathcal{L}_{TAC} = CE(\tilde{\mathbf{x}}_r \tilde{\mathbf{F}}_{text}^T, 0) + CE(\tilde{\mathbf{x}}_q \tilde{\mathbf{F}}_{text}^T, y_q)$$

  Text embeddings are constructed using prompt templates similar to WinCLIP (e.g., "a photo of a [state] [class]"), and $y_q \in \{0,1\}$ denotes the image-level label.

- **Design Motivation**: While the Query Loss endows the model with strong retrieval capability, this is a double-edged sword—an overly powerful model may compose normal feature combinations from the dictionary to match any query, including anomalies. CQC widens the margin in feature space, while TAC constrains retrieval results semantically to remain "normal."

### Loss & Training

- **Total Loss**: $\mathcal{L} = \mathcal{L}_q + \lambda_1 \mathcal{L}_{CQC} + \lambda_2 \mathcal{L}_{TAC}$, with $\lambda_1 = \lambda_2 = 0.1$
- **Self-Supervised Training**: No pixel-level annotations are required; query images are generated using DRÆM's anomaly synthesis algorithm, and reference images are generated via geometric transformations.
- **Auxiliary Data**: All normal images from VisA are used as the training set (replaced by MVTecAD when testing on VisA).
- **Training Configuration**: Adam optimizer, lr=0.0001, batch size=24, 30 epochs, single RTX 3090 GPU.
- **Reference Images During Training**: $k=1$ (for efficiency); $k \geq 1$ at inference.
- **CLIP Backbone**: Frozen; features are extracted from layers 6/12/18/24.

## Key Experimental Results

### Main Results

**Pixel-level metrics (AUROC, PRO, AP) on 7 datasets under the 4-shot setting**:

| Dataset | WinCLIP | APRIL-GAN | PromptAD† | **DictAS** |
|--------|---------|-----------|----------|----------|
| MVTecAD | (92.4, 83.8, 39.2) | (92.2, 86.6, 46.6) | (96.0, 92.4, 57.5) | **(98.6, 95.1, 66.8)** |
| VisA | (96.0, 86.5, 25.7) | (96.2, 86.6, 30.6) | (97.9, 89.5, 37.5) | **(98.8, 91.9, 41.8)** |
| RESC (Medical) | (93.1, 75.7, 38.4) | (93.7, 77.6, 57.3) | (96.8, 86.8, 71.3) | **(97.5, 89.7, 74.9)** |
| BraTS (Medical) | (93.3, 64.0, 33.4) | (91.3, 63.0, 40.0) | (96.6, 77.0, 54.4) | **(97.3, 77.2, 59.3)** |
| Industrial Avg. | (94.5, 82.7, 29.3) | (94.7, 84.8, 38.5) | (97.1, 89.6, 47.0) | **(98.4, 92.2, 52.5)** |
| Medical Avg. | (93.2, 69.8, 35.9) | (92.5, 70.3, 48.7) | (96.7, 82.2, 62.9) | **(97.4, 83.4, 67.1)** |

†Note: PromptAD is a class-dependent method requiring per-class fine-tuning; DictAS is a class-generalizable method using a unified model.

### Ablation Study

**Component ablation (MVTecAD, 4-shot, %)**:

| Configuration | AUROC | PRO | AP | Note |
|------|-------|-----|-----|------|
| w/o Query Generator | 97.5 | 94.2 | 63.5 | Without query transformation |
| w/o Key Generator | 97.9 | 94.5 | 63.8 | Without index transformation |
| w/o Value Generator | 98.0 | 94.6 | 64.2 | Without content transformation |
| w/o $\mathcal{L}_{CQC}$ | 97.4 | 94.1 | 64.6 | Without contrastive constraint |
| w/o $\mathcal{L}_{TAC}$ | 98.0 | 94.6 | 65.0 | Without text alignment constraint |
| w/o Both Regularizations | 97.1 | 93.5 | 63.7 | Without all regularization |
| **Full DictAS** | **98.6** | **95.1** | **66.8** | Complete model |

**Lookup strategy ablation (MVTecAD, AP%)**:

| Strategy | 1-shot | 4-shot | 8-shot | 16-shot |
|------|--------|--------|--------|---------|
| Maximum Lookup | 52.2 | 59.1 | 59.7 | 60.6 |
| Dense Lookup | 60.2 | 63.7 | 63.6 | 63.8 |
| **Sparse Lookup** | **61.1** | **66.8** | **67.0** | **68.5** |

### Key Findings

1. **Sparse Lookup advantage grows with more shots**: Dense Lookup shows almost no improvement from 4-shot to 16-shot (63.7→63.8), whereas Sparse Lookup improves consistently (66.8→68.5), confirming that the sparse strategy effectively suppresses redundancy introduced by additional reference images.
2. **DictAS surpasses class-dependent methods**: The unified DictAS model outperforms PromptAD—which requires per-class fine-tuning—on all metrics.
3. **Fastest inference speed**: 73.5ms per image, faster than all competing methods (WinCLIP: 8227.5ms, AnomalyGPT: 1555.2ms).
4. **High stability**: AP standard deviation is only 0.4%, lower than all competing methods.
5. **TAC contributes more to fine-grained discrimination**: Ablating $\mathcal{L}_{TAC}$ results in a larger AP drop (+2.2%) compared to ablating $\mathcal{L}_{CQC}$ (+1.8%).

## Highlights & Insights

1. **Intuition-driven problem formulation**: The "dictionary lookup" analogy is both intuitive and principled—recasting anomaly detection as a retrieval problem is more fundamental than reconstruction- or classification-based perspectives.
2. **Self-supervised training paradigm**: No real anomaly samples or pixel-level annotations are required; training relies solely on normal images combined with anomaly synthesis, substantially reducing data requirements.
3. **Elegant sparse lookup design**: SPM adaptively determines the threshold via convex optimization, naturally suppressing redundancy while maintaining retrieval accuracy.
4. **Necessity of dual regularization**: t-SNE visualizations clearly demonstrate the improved separability of residual features with and without regularization.

## Limitations & Future Work

1. **Dependence on CLIP backbone**: The method implicitly relies on CLIP pretraining quality, though ablations suggest limited sensitivity to backbone choice.
2. **Impact of anomaly synthesis**: The use of DRÆM's anomaly synthesis strategy may limit the effectiveness of regularization due to the diversity and realism constraints of synthetic anomalies.
3. **Evaluation restricted to FSAS settings**: It remains unexplored whether the approach retains advantages when sufficient data is available (e.g., full-shot settings).
4. **Simple text prompt design**: The text templates used in TAC rely on class names, which may require more sophisticated prompt engineering in industrial scenarios.

## Related Work & Insights

- **vs. WinCLIP**: WinCLIP leverages CLIP's vision-language alignment and a memory bank for anomaly detection, but relies on predefined text prompts and static visual priors. DictAS instead learns dynamically adaptive retrieval weights for flexible dictionary lookup.
- **vs. AnomalyGPT**: AnomalyGPT incorporates a large language model for multi-turn interactive dialogue but incurs substantial inference overhead (1555ms vs. 73.5ms). DictAS focuses on efficient end-to-end detection.
- **Broader implication**: The dictionary lookup paradigm can be generalized to other detection tasks—any scenario characterized by a limited set of normal patterns and unbounded anomaly patterns is amenable to this framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dictionary lookup analogy is intuitive yet carefully designed; Sparse Lookup combined with dual regularization is effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluated on 7 datasets (industrial + medical) across 5 shot settings, with comprehensive ablations (components, strategies, backbone, resolution) and t-SNE visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ — The dictionary analogy is consistently maintained throughout, with clear and logical presentation.
- **Value**: ⭐⭐⭐⭐ — A practical solution for class-generalizable FSAS, striking an excellent balance between inference speed and detection performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](../../CVPR2026/medical_imaging/bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)
- [\[ICCV 2025\] Toward Long-Tailed Online Anomaly Detection through Class-Agnostic Concepts](toward_long-tailed_online_anomaly_detection_through_class-agnostic_concepts.md)
- [\[ICLR 2026\] Dual Distillation for Few-Shot Anomaly Detection](../../ICLR2026/medical_imaging/dual_distillation_for_few-shot_anomaly_detection.md)
- [\[ICCV 2025\] UKBOB: One Billion MRI Labeled Masks for Generalizable 3D Medical Image Segmentation](ukbob_one_billion_mri_labeled_masks_for_generalizable_3d_medical_image_segmentat.md)
- [\[ICCV 2025\] IDF: Iterative Dynamic Filtering Networks for Generalizable Image Denoising](idf_iterative_dynamic_filtering_networks_for_generalizable_image_denoising.md)

</div>

<!-- RELATED:END -->
