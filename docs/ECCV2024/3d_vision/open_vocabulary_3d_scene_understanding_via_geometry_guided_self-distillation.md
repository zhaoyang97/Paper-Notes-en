---
title: >-
  [Paper Note] Open Vocabulary 3D Scene Understanding via Geometry Guided Self-Distillation
description: >-
  [ECCV 2024][3D Vision][Open-vocabulary 3D scene understanding] The GGSD framework is proposed, which leverages 3D geometric priors (semantic consistency of superpoints) to guide knowledge distillation from 2D to 3D models. It further uncovers the representational advantages of 3D data through a self-distillation mechanism, significantly outperforming existing methods on both indoor and outdoor open-vocabulary 3D scene understanding tasks.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Open-vocabulary 3D scene understanding"
  - "self-distillation"
  - "geometric priors"
  - "knowledge distillation"
  - "superpoints"
date: 2026-05-08
content_hash: 10900d37e2e1a2c0
---

# Open Vocabulary 3D Scene Understanding via Geometry Guided Self-Distillation

**Conference**: ECCV 2024  
**arXiv**: [2407.13362](https://arxiv.org/abs/2407.13362)  
**Code**: [GitHub](https://github.com/Wang-pengfei/GGSD)  
**Area**: 3D Vision  
**Keywords**: Open-vocabulary 3D scene understanding, self-distillation, geometric priors, knowledge distillation, superpoints  

## TL;DR

The GGSD framework is proposed, which leverages 3D geometric priors (semantic consistency of superpoints) to guide knowledge distillation from 2D to 3D models. It further uncovers the representational advantages of 3D data through a self-distillation mechanism, significantly outperforming existing methods on both indoor and outdoor open-vocabulary 3D scene understanding tasks.

## Background & Motivation

- **Background**: Open-vocabulary 3D scene understanding is a critical technology for applications such as robotic manipulation and autonomous driving. Current prevailing solutions rely on distilling internet-scale 2D vision-language knowledge into 3D models.
- **Limitations of Prior Work**: Existing distillation methods (e.g., OpenScene) essentially mimic 2D models directly. Consequently, inherent visual issues of 2D models such as occlusion, lighting variations, and viewpoint differences are propagated to the 3D models during distillation, limiting their upper bound of representation.
- **Key Challenge**: 3D data naturally possesses representational advantages that are invariant to illumination and viewpoint changes. However, existing methods do not fully exploit this geometric prior, causing the distilled 3D models to be constrained by the noise of the 2D teacher models.
- **Goal**: To exploit 3D geometric priors for denoising during 2D-to-3D distillation, and to further unlock the representational capabilities of 3D data via self-distillation.
- **Key Insight**: Observing that the distilled 3D student model can significantly surpass the 2D teacher model (due to the inherent advantages of 3D representations), a two-stage strategy of "learning from 2D first, then learning from self" is designed.
- **Core Idea**: Leverage the semantic consistency of superpoints to constrain noise during the distillation process, and combine this with a voting mechanism based on an EMA model to achieve reliable self-distillation.

## Method

### Overall Architecture

GGSD consists of two core modules: **Geometry Guided Distillation** and **Self-Distillation**. The first stage learns open-vocabulary capabilities from pre-trained 2D models (e.g., LSeg/OpenSeg) while mitigating 2D noise using 3D geometric priors. The second stage leverages the learned 3D representation advantages to perform self-distillation using an EMA model and a superpoint voting mechanism, further enhancing performance.

### Key Designs

**Module 1: Pixel-to-Point Feature Pair Creation**

Following the pipeline of OpenScene, a pre-trained 2D vision-language segmentation model (e.g., LSeg/OpenSeg) is utilized to extract pixel-wise dense embeddings. The correspondence between 3D points and 2D pixels is established via camera intrinsic and extrinsic matrices, and multi-view features are fused using average pooling:

$$\mathbf{f}^{\text{2D}} = \phi(\mathbf{f}_1, \cdots, \mathbf{f}_K)$$

This yields the fused 2D feature $\mathbf{F}^{\text{2D}} \in \mathbb{R}^{M \times C}$ for each 3D point.

**Module 2: Geometry Guided Distillation**

The VCCS algorithm is used to decompose the point cloud into geometrically homogeneous superpoints $\{\tilde{\mathbf{p}}_1, \cdots, \tilde{\mathbf{p}}_N\}$, where points within each superpoint typically belong to the same semantic category. The mean of both 2D and 3D features within each superpoint is computed as:

$$\tilde{\mathbf{f}}_n^{\text{2D}} = \frac{1}{Q} \sum_{q=1}^{Q} \mathbf{f}_q^{\text{2D}}$$

Subsequently, semantic consistency is constrained via a superpoint-level cosine similarity loss:

$$\mathcal{L}_{sp} = 1 - \cos(\mathbf{F}_{sp}^{\text{2D}}, \mathbf{F}_{sp}^{\text{3D}})$$

The total distillation loss is a combination of point-level and superpoint-level losses: $\mathcal{L}_d = \mathcal{L}_p + \mathcal{L}_{sp}$.

**Module 3: Geometry Guided Self-Distillation**

An EMA model is utilized to predict pseudo-labels for each 3D point, assigning semantic categories by calculating similarities with CLIP text embeddings:

$$\mathbf{f}^{\hat{t}} = \arg\max_l \psi(\mathbf{f}_n^{\text{3D}}, \mathbf{f}_l^t)$$

A voting mechanism is conducted within each superpoint, aligning the labels of all points inside the superpoint to the category with the highest vote to reduce noise. Finally, the network is trained using a contrastive learning loss:

$$\mathcal{L}_{sd} = -\log \frac{\exp(\mathbf{f}^{\text{3D}} \cdot \mathbf{f}^{\hat{t}} / \tau)}{\sum_{i=1}^{n_t} \exp(\mathbf{f}^{\text{3D}} \cdot \mathbf{f}_i^t / \tau)}$$

where the temperature factor is $\tau = 0.01$.

### Loss & Training

- **Two-stage training**: The first stage trains the model with geometry-guided distillation for 70 epochs. The second stage introduces the self-distillation module for an additional 30 epochs, keeping the total number of epochs consistent with OpenScene.
- **EMA Model**: An exponential moving average (EMA) model is used to provide stable pseudo-label supervision, avoiding model collapse that may occur when directly using online network predictions.
- **3D Backbone**: MinkowskiNet18A, with an indoor voxel size of 2cm and outdoor of 5cm.
- **Optimizer**: Adam, learning rate $1 \times 10^{-4}$, single A100 80G GPU, batch size 8.

## Key Experimental Results

### Main Results

| Method | ScanNet mIoU | ScanNet mAcc | nuScenes mIoU | nuScenes mAcc |
|------|:---:|:---:|:---:|:---:|
| OpenScene (2D-3D ensemble) | 54.2 | 66.6 | 42.1 | 61.8 |
| OpenScene (pure 3D) | 52.9 | 63.2 | 42.9 | 57.1 |
| CLIP-FO3D | 30.2 | 49.1 | - | - |
| CNS | 26.8 | - | 33.5 | - |
| **GGSD (Ours)** | **56.5** | **68.6** | **46.1** | **59.2** |

Using only pure 3D point clouds, the model outperforms the 2D-3D ensemble strategy of OpenScene, achieving a gain of +3.6% mIoU on ScanNet and +3.2% mIoU on nuScenes.

### Ablation Study

| Component | ScanNet mIoU | ScanNet mAcc | Matterport mIoU | Matterport mAcc |
|------|:---:|:---:|:---:|:---:|
| 2D Fusion Projection | 50.0 | 62.7 | 32.3 | 40.0 |
| Pixel-Point Distillation | 52.9 | 63.2 | 36.1 | 48.0 |
| + Geometry Guided Distillation | 53.5 | 65.0 | 36.7 | 49.3 |
| + Self-Distillation | 56.1 | 68.2 | 39.0 | 53.3 |
| + Geometry Guided Self-Distillation | **56.5** | **68.6** | **40.1** | **54.4** |

### Key Findings

- Geometry-guided distillation brings a +0.6% mIoU and +1.8% mAcc improvement on ScanNet, effectively mitigating 2D noise through superpoint semantic consistency constraints.
- Self-distillation contributes the most: yielding +2.6% mIoU on ScanNet and +2.3% mIoU on Matterport, validating that 3D representational advantages can be unlocked via self-distillation.
- The EMA model outclasses fixed 2D models and fixed 3D models as the source of supervision signals.
- In cross-domain experiments (ScanNet $\rightarrow$ Matterport), GGSD outperforms OpenScene across 21/40/80/160 category settings, demonstrating strong generalization capability.
- Using SAM to refine 2D features is less effective than leveraging 3D geometric priors (0.1% vs 0.6% mIoU improvement).

## Highlights & Insights

- **Precise Core Insight**: The observation that "a 3D student can surpass a 2D teacher" acts as a strong motivation for the self-distillation design, upgrading traditional distillation from unidirectional mimicry to bidirectional enhancement.
- **Superpoint Voting Mechanism** is a simple yet effective denoising method that leverages the semantic consistency assumption of geometric structures, being both practical and computationally low-cost.
- **No Extra Labeled Data Required**: The entire training pipeline does not rely on any 2D or 3D ground truth annotations.
- **Strong Generalization**: The model trained on ScanNet can be transferred to Matterport3D in a zero-shot manner.

## Limitations & Future Work

- Performance on tail classes (small objects, few-shot samples) is still suboptimal, with a Tail class mIoU of only 16.0%.
- Superpoint generation relies on the VCCS algorithm, which may not be robust enough for extremely sparse or unordered point clouds.
- Language ambiguity issue: an armchair may be contextually identified as either a "sofa" or a "chair" separately.
- The self-distillation stage still employs pre-defined class text templates and does not explore more flexible open-text queries.

## Related Work & Insights

- **OpenScene**: The main baseline of this work, which proposes a pixel-to-point feature distillation framework but is limited by the noise in 2D models.
- **CLIP2Scene / CNS**: Utilizes CLIP for 3D scene understanding, with performance far below this work.
- **Mean Teacher / EMA**: The EMA strategy in self-distillation draws inspiration from classical paradigms in semi-supervised learning.
- **Insights**: The concept of superpoint semantic consistency constraints can be extended to other 3D tasks such as instance segmentation and object detection.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Combining self-distillation with geometric priors is novel; the paradigm of "student surpassing the teacher and then conducting self-learning" is inspiring.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Full coverage of indoor and outdoor datasets, thorough ablation studies, and compelling cross-domain experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, smooth logic, and intuitive diagrams.
- **Value**: ⭐⭐⭐⭐ — Open-source code, significant performance gain, and good potential for application in practical scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Open-Vocabulary Octree-Graph for 3D Scene Understanding](../../ICCV2025/3d_vision/open-vocabulary_octree-graph_for_3d_scene_understanding.md)
- [\[ECCV 2024\] Open-Vocabulary 3D Semantic Segmentation with Text-to-Image Diffusion Models](open-vocabulary_3d_semantic_segmentation_with_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] LightSplat: Fast and Memory-Efficient Open-Vocabulary 3D Scene Understanding in Five Seconds](../../CVPR2026/3d_vision/lightsplat_fast_and_memory-efficient_open-vocabulary_3d_scene_understanding_in_f.md)
- [\[AAAI 2026\] OpenScan: A Benchmark for Generalized Open-Vocabulary 3D Scene Understanding](../../AAAI2026/3d_vision/openscan_a_benchmark_for_generalized_open-vocabulary_3d_scene_understanding.md)
- [\[ICLR 2026\] GeoPurify: A Data-Efficient Geometric Distillation Framework for Open-Vocabulary 3D Segmentation](../../ICLR2026/3d_vision/geopurify_a_data-efficient_geometric_distillation_framework_for_open-vocabulary_.md)

</div>

<!-- RELATED:END -->
