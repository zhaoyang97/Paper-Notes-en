---
title: >-
  [Paper Note] EPSegFZ: Efficient Point Cloud Semantic Segmentation for Few- and Zero-Shot Scenarios
description: >-
  [AAAI 2026][3D Vision][Point Cloud Semantic Segmentation] This paper proposes EPSegFZ, a pre-training-free 3D point cloud few-shot/zero-shot semantic segmention framework. By utilizing ProERA to extract high-frequency features, LGPE to fuse textual information for prototype updates, and DRPE to establish precise query-prototype correspondences, EPSegFZ outperforms SOTA methods on S3DIS and ScanNet by 5.68% and 3.82% respectively.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Point Cloud Semantic Segmentation"
  - "Few-Shot Learning"
  - "Zero-Shot Learning"
  - "Language Guidance"
  - "Attention Mechanism"
date: 2026-05-08
content_hash: cda5c6ddd12c8209
---

# EPSegFZ: Efficient Point Cloud Semantic Segmentation for Few- and Zero-Shot Scenarios

**Conference**: AAAI 2026  
**arXiv**: [2511.11700](https://arxiv.org/abs/2511.11700)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Point Cloud Semantic Segmentation, Few-Shot Learning, Zero-Shot Learning, Language Guidance, Attention Mechanism

## TL;DR

This paper proposes EPSegFZ, a pre-training-free 3D point cloud few-shot/zero-shot semantic segmention framework. By utilizing ProERA to extract high-frequency features, LGPE to fuse textual information for prototype updates, and DRPE to establish precise query-prototype correspondences, EPSegFZ outperforms SOTA methods on S3DIS and ScanNet by 5.68% and 3.82% respectively.

## Background & Motivation

Few-shot semantic segmentation (FS-SemSeg) on 3D point clouds faces three core challenges:

**Over-reliance on pre-training**: Existing methods (such as AttMPTI, COSeg, etc.) heavily rely on fully supervised pre-trained backbones, introducing domain discrepancy bias. Moreover, 3D datasets are small and prone to overfitting, and pre-training itself is computationally expensive.

**Loss of high-frequency information**: Pre-training-free approaches like Seg-PN discard high-frequency information to maintain robustness. However, high-frequency features contain critical boundary details essential for precise segmentation.

**Underutilization of support set information**: Existing methods solely rely on point cloud labels and ignore complementary information such as text annotations, limiting performance and zero-shot capabilities.

The core idea of this work is to leverage high-frequency visual features and low-frequency text features simultaneously, achieving efficient few-shot/zero-shot segmentation without pre-training.

## Method

### Overall Architecture

EPSegFZ comprises three core modules, forming an end-to-end framework that does not require pre-training:

1. **ProERA (Prototype-Enhanced Registers Attention)**: Enhances feature extraction and captures high-frequency information.
2. **LGPE (Language-Guided Prototype Embedding)**: Utilizes textual information to update prototypes, enabling zero-shot inference.
3. **DRPE (Dual Relative Positional Encoding)-based cross-attention**: Establishes precise query-prototype correspondences.

Workflow: DGCNN (trained from scratch) extracts point cloud features $\rightarrow$ MPS samples multi-prototypes $\rightarrow$ ProERA refines features (incorporating register and prototype tokens) $\rightarrow$ LGPE updates prototypes (incorporating text embeddings) $\rightarrow$ DRPE cross-attention establishes correspondence $\rightarrow$ Dot product generates predictions.

### Key Designs

#### ProERA Module

The core innovation lies in progressively focusing on high-frequency information by subtracting low-frequency components:

- In the input token sequence, learnable register tokens $\mathbf{r}_t \in \mathbb{R}^{n_r \times D}$ and prototype tokens are appended.
- Since self-attention naturally acts as a low-pass filter, the mean of the input features (representing the low-frequency component) is subtracted from the self-attention output to obtain high-frequency dominant features:

$$\tilde{\mathbf{X}}_j^i = \text{Res}(\text{SA}([\hat{\mathbf{X}}_j^{i-1}; \mathbf{r}_j; \hat{\mathbf{p}}^{i-1}])) - \frac{1}{n_j}\sum_{n_j}\hat{\mathbf{X}}_j^{i-1}$$

- Register tokens learn to focus on different regions: one focuses on the background/empty regions, while the other focuses on regions containing multiple objects, implicitly mitigating the foreground-background imbalance.

#### LGPE Module

This module addresses the prototype quality issues, particularly when the backbone is randomly initialized in early training stages and generates uninformative prototypes:

- A pre-trained CLIP text encoder is used to obtain class textual embeddings $\mathbf{T}^c$, which are projected into a unified space via a projection network.
- Prototype updates fuse four sources: the prototype token from the previous layer $\tilde{\mathbf{p}}^i$, the raw prototype $\mathbf{p}_{raw}$, the dynamic prototype $\mathbf{p}_{dyn}^i$, and the textual prototype $\mathbf{p}_{text}$.

$$\mathbf{p}^i = \lambda_1 \tilde{\mathbf{p}}^i + \lambda_2 \mathbf{p}_{raw} + \lambda_3 \mathbf{p}_{dyn}^i + \lambda_4 \mathbf{p}_{text}$$

- **Dynamic weights scheduling**: The textual weight $\lambda_4(t) = \lambda_4^* e^{-0.5t}$ is set with exponential decay, while the visual weights $\lambda_i(t) = \lambda_i^*(1 - e^{-0.5t})$ gradually grow. This facilitates a smooth transition from text-driven to a balanced visual-textual space.
- Zero-shot capability: Prototypes can be constructed directly from textual embeddings without requiring support set point clouds.

#### DRPE Module

This module is the first to introduce query-prototype relations in the latent space as positional encoding signals into cross-attention:

- The **Euclidean distance** $d_E^{i,j,c}$ between query points and prototypes is calculated and encoded via a sinusoidal positional encoding function to obtain $\mathbf{R}_E^i$.
- The **cosine similarity** $d_C^{i,j,c}$ between query and prototype vectors is calculated and similarly encoded to obtain $\mathbf{R}_C^i$.
- The dual encodings are summed: $\mathbf{R}^i = \mathbf{R}_C^i + \mathbf{R}_E^i$.
- Advantages: It incurs zero additional trainable parameters while efficiently capturing query-prototype correlations as prior knowledge.

### Loss & Training

Three loss functions are optimized jointly:

1. **Segmentation loss** $\mathcal{L}_{seg} = \text{CE}(\mathbf{Y}_q, \hat{\mathbf{Y}}_q)$: The primary supervision signal.
2. **Foreground consistency loss** $\mathcal{L}_{con} = \text{InfoNCE}(\mathbf{x}_q, \mathbf{x}_s)$: Encourages foreground features of the same class to group in the embedding space, compensating for the lack of a pre-trained backbone.
3. **Foreground-aware alignment loss** $\mathcal{L}_{align}$: Formulated as minimizing the cross-entropy between text-visual similarities and textual labels to enhance the unified text-visual space.

$$\mathcal{L} = \mathcal{L}_{seg} + \lambda_{con}\mathcal{L}_{con} + \lambda_{align}\mathcal{L}_{align}$$

Training strategy: episodic learning with 30,000 iterations, employing a larger learning rate for the backbone with rapid decay.

## Key Experimental Results

### Main Results

**S3DIS Dataset (2-way 1-shot)**:

| Method | S⁰ | S¹ | Mean | Δ |
|------|-----|-----|------|-----|
| AttMPTI | 53.77 | 55.94 | 54.86 | -18.56 |
| PAPFZS3D | 59.45 | 66.08 | 62.76 | -10.66 |
| Seg-PN | 64.84 | 67.98 | 66.41 | -7.01 |
| SDSimPoint | 68.73 | 70.61 | 69.67 | -3.75 |
| **EPSegFZ** | **73.08** | **73.75** | **73.42** | **-** |

**ScanNet Dataset (2-way 1-shot)**:

| Method | Mean | Δ |
|------|------|-----|
| Seg-PN | 63.74 | -5.10 |
| SDSimPoint | 65.19 | -3.65 |
| **EPSegFZ** | **68.84** | **-** |

Efficiency Analysis: EPSegFZ requires only 2.02M parameters, 2.11 GFLOPs, and 0.36s inference time, which is comparable to the lightweight Seg-PN (241K/1.95/0.32) and significantly superior to COSeg (7.69M/9.71/1.35).

### Ablation Study

| Configuration | mIoU | Δ |
|------|------|-----|
| Without any modules | 31.55 | -41.53 |
| ProERA only | 64.84 | -8.24 |
| ProERA + LGPE | 70.48 | -2.60 |
| ProERA + DRPE | 70.17 | -2.91 |
| Full model | **73.08** | - |

In prototype ablation studies, the dynamic prototype $\mathbf{p}_{dyn}$ contributes the most. Removing $\mathcal{L}_{con}$ or $\mathcal{L}_{align}$ leads to a drop of approximately 4-5% each. DRPE outperforms learnable positional encodings and traditional sinusoidal encodings.

### Key Findings

- Zero-shot evaluation (S3DIS, CLIP, 2-way 1-shot): EPSegFZ reaches 63.84%, outperforming PAPFZS3D (61.09%).
- t-SNE visualization demonstrates that intra-class feature distribution is more compact and inter-class separation is clearer with EPSegFZ.
- Using $N+1$ register tokens (where $N$ is the number of classes) yields the best results; 3 decoder blocks provide the optimal performance-efficiency trade-off.

## Highlights & Insights

1. **Elegant High-Frequency Extraction**: Since self-attention naturally acts as a low-pass filter, subtracting the mean from the output effectively extracts high-frequency information in a simple and efficient manner.
2. **Dynamic Weight Scheduling**: Visual-textual weights adapt dynamically across training stages, mitigating the cold-start problem of training without a pre-trained backbone.
3. **Zero Parameter Overhead of DRPE**: Injects query-prototype relations into query-prototype attention layers using sinusoidal encodings, avoiding any increase in trainable parameters.
4. **Unified Few-Shot/Zero-Shot Framework**: LGPE enables constructing prototypes solely from text descriptions, naturally enabling zero-shot learning.

## Limitations & Future Work

- A gap in zero-shot performance still remains compared to large-scale pre-trained models (e.g., SegPoint), though training resources and data scales are not directly comparable.
- Text embeddings are derived from a frozen CLIP encoder without exploring finer-grained textual descriptions.
- Evaluations are restricted to indoor scene datasets (S3DIS, ScanNet), and experiments on large-scale outdoor scenarios are lacking.
- The dynamic weight scheduling uses fixed exponential decay/growth functions; adaptive learning schemes could be explored.

## Related Work & Insights

- Seg-PN (a pre-training-free method) inspired the training-from-scratch architecture, though its discard of high-frequency information served as a starting point for improvement in this work.
- Research on register tokens in ViT (Darcet et al.) inspired the use of registers in ProERA.
- The text-visual alignment of CLIP inspired the LGPE module.
- Insights for future research: The proposed framework can be extended to other 3D tasks (e.g., target detection, instance segmentation) or combined with stronger language models instead of CLIP.

## Rating

- Novelty: ⭐⭐⭐⭐ — The three module designs possess distinct innovations, and the high-frequency extraction approach in ProERA is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablation studies were conducted on multiple benchmarks under varying settings.
- Writing Quality: ⭐⭐⭐⭐ — The structure is clear, and the frequency analysis visualizations are highly convincing.
- Value: ⭐⭐⭐⭐ — Highly practical as it requires no pre-training, features few parameters, and enables fast inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CLIPoint3D: Language-Grounded Few-Shot Unsupervised 3D Point Cloud Domain Adaptation](../../CVPR2026/3d_vision/clipoint3d_language-grounded_few-shot_unsupervised_3d_point_cloud_domain_adaptat.md)
- [\[ICLR 2026\] Point-UQ: An Uncertainty Quantification Paradigm for Point Cloud Few-Shot Class-Incremental Learning](../../ICLR2026/3d_vision/point-uq_an_uncertainty-quantification_paradigm_for_point_cloud_few-shot_class_i.md)
- [\[ICCV 2025\] BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes](../../ICCV2025/3d_vision/bufferx_towards_zeroshot_point_cloud_registration_in_diverse.md)
- [\[CVPR 2026\] Image-to-Point Cloud Feature Back-Projection for Multimodal Training of 3D Semantic Segmentation](../../CVPR2026/3d_vision/image-to-point_cloud_feature_back-projection_for_multimodal_training_of_3d_seman.md)
- [\[CVPR 2026\] PointGS: Semantic-Consistent Unsupervised 3D Point Cloud Segmentation with 3D Gaussian Splatting](../../CVPR2026/3d_vision/pointgs_semantic-consistent_unsupervised_3d_point_cloud_segmentation_with_3d_gau.md)

</div>

<!-- RELATED:END -->
