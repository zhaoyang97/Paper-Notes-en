---
title: >-
  [Paper Note] HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] HybridGS proposes the first hybrid 2D+3D Gaussian representation, modeling static scenes with multi-view consistent 3D Gaussians and transient objects with single-view independent 2D Gaussians. Combined with multi-view regulation and multi-stage training, it achieves state-of-the-art (SOTA) novel view synthesis quality in scenes containing distractor elements.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Transient Object Separation"
  - "Hybrid Representation"
  - "Novel View Synthesis"
  - "2D Gaussian"
date: 2026-05-08
content_hash: e9113c6653f4553d
---

# HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting

**Conference**: CVPR 2025  
**arXiv**: [2412.03844](https://arxiv.org/abs/2412.03844)  
**Code**: Yes ([https://gujiaqivadin.github.io/hybridgs/](https://gujiaqivadin.github.io/hybridgs/))  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Transient Object Separation, Hybrid Representation, Novel View Synthesis, 2D Gaussian

## TL;DR
HybridGS proposes the first hybrid 2D+3D Gaussian representation, modeling static scenes with multi-view consistent 3D Gaussians and transient objects with single-view independent 2D Gaussians. Combined with multi-view regulation and multi-stage training, it achieves state-of-the-art (SOTA) novel view synthesis quality in scenes containing distractor elements.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) has achieved excellent results in novel view synthesis, but it typically assumes that the input images contain only static content. Casual real-world photos often contain transient objects such as moving pedestrians and vehicles, and directly training 3DGS on these images leads to artifacts.

**Limitations of Prior Work**: (1) Semantic ambiguity—RobustNeRF uses robust loss to reduce the weight of inconsistent observations, NeRF On-the-go uses DINOv2 features to predict uncertainty, and SpotLessSplats uses semantic feature clustering to detect anomalies. However, using semantic features from vision models to distinguish transient objects is inherently ambiguous (semantically similar objects are not necessarily transient). (2) Lack of explicit transient modeling—these methods only reduce the impact of transient objects to reconstruct a better static scene, without actually modeling the transient objects themselves.

**Key Challenge**: 3DGS is inherently designed for multi-view consistent scenes, whereas transient objects only appear in individual views and do not satisfy the multi-view consistency assumption. Forcing 3D Gaussians to fit transient objects produces artifacts when viewed from other perspectives.

**Goal**: (1) How to reasonably decouple transient and static elements based on the nature of view consistency? (2) How to explicitly model transient objects rather than simply ignoring them?

**Key Insight**: The authors' key observation is that transient objects lack multi-view consistency and typically appear in only a single view, allowing them to be treated as planar objects for that view. This implies that 3D Gaussians are suitable for static scenes, while 2D Gaussians are suitable for transient objects—the choice of dimensionality for the two representations precisely corresponds to their view consistency properties.

**Core Idea**: Modeling single-view transient objects with 2D Gaussians and multi-view consistent static scenes with 3D Gaussians naturally decouples the two based on the geometric essence of view consistency.

## Method

### Overall Architecture
Input: A set of casually captured images with camera parameters (which may contain transient objects). For each view, the scene is decomposed as: $I = M_t \odot I_t + (1-M_t) \odot I_s$, where $I_s$ is rendered by shared multi-view 3D Gaussians, and $I_t$ and $M_t$ (transient mask) are rendered by independent 2D Gaussians for each view. Training is divided into three stages: warm-up $\rightarrow$ alternating training $\rightarrow$ joint training. Output: The decoupled static scene 3DGS (used for novel view synthesis) and the transient decomposition for each view.

### Key Designs

1. **Multi-view Regulated Supervision**:

    - **Function**: Enhance the ability of 3DGS to distinguish between transient and static elements, preventing overfitting to transient objects.
    - **Mechanism**: Unlike standard 3DGS which processes only a single image per iteration, this method randomly samples $K$ images per iteration and computes the intersection of their $K$ camera frustums, restricting the optimization to 3D Gaussian points falling within this intersection. This has two effects: (a) backpropagation of gradients considers mutual information across multiple views simultaneously, encouraging 3D Gaussians to learn cross-view consistent content; (b) focusing optimization in co-visible regions reduces computation, since transient objects typically do not appear in co-visible regions. This scheme achieves sparse training at the algorithmic level by filtering Gaussian points through frustum intersection.
    - **Design Motivation**: The geometric constraints of multi-view joint optimization naturally reject transient objects—as transient objects only appear in a single view and lack corresponding supervision signals in other views, they are not reinforced by multi-view gradients in co-visible regions.

2. **Modeling Transients with 2D Gaussians**:

    - **Function**: Explicitly model the transient objects of each view while generating the transient mask.
    - **Mechanism**: An independent set of 2D Gaussians is maintained for each training image. The 2D Gaussian parameters include the 2D center point $\mathbf{x_{2d}} \in \mathbb{R}^2$, the 2D covariance matrix $\Sigma_{2d}$, color $\mathbf{c_{2d}}$, and opacity $\alpha_{2d}$. The 2D Gaussians are rasterized to produce a transient image $\hat{I}_t$ and a transient mask $\hat{M}_t$—the mask is a simple accumulation of opacities, $\hat{M}_t(\mathbf{y}) = \sum_i \alpha_{2d_i}'$. A mask value close to 1 at a certain pixel indicates a high probability of a transient object. This design leverages existing work (GaussianImage) that uses 2D Gaussians as image representations, but repurposes it to learn residual maps and uncertainty.
    - **Design Motivation**: After the 3DGS models the static scene, 2D Gaussians naturally learn to fit the residual part of the image—namely, the content that 3DGS cannot explain in a multi-view consistent manner, which corresponds to transient objects. The opacity of the 2D Gaussians implicitly provides a probabilistic mask of the transient regions.

3. **Multi-stage Training Scheme**:

    - **Function**: Ensure stable convergence and high-quality decomposition during 2D and 3D Gaussian training.
    - **Mechanism**: Training is divided into three stages: (a) **Warm-up stage**—train only 3DGS to capture the basic structure of the static scene using multi-view regulated supervision with DSSIM+L1 loss; (b) **Alternating training stage**—alternately optimize 2D and 3D Gaussians: first, freeze 3DGS to train 2D Gaussians to learn the residual and mask, and then use the mask to guide 3DGS optimization in non-transient areas, repeating iteratively; (c) **Joint training stage**—optimize both 2D and 3D Gaussians simultaneously to further refine the decomposition. Overall synthesis is achieved via $\hat{I} = \hat{M}_t \odot \hat{I}_t + (1-\hat{M}_t) \odot \hat{I}_s$.
    - **Design Motivation**: Direct joint training of 2D and 3D Gaussians can lead to competition, where both representations attempt to reconstruct the same pixels. The multi-stage training enables 3DGS to first build the foundation of the static scene, and then allows 2D Gaussians to operate in the residual space, avoiding decomposition collapse.

### Loss & Training
The warm-up stage uses $\mathcal{L}_{warmup} = \text{DSSIM} + \text{L1}$. In the alternating and joint training stages, the loss for the final rendered image is $\mathcal{L} = (1-\lambda)\text{L1}(\hat{I}, I) + \lambda\text{DSSIM}(\hat{I}, I)$. It does not use any additional semantic features (such as DINOv2), relying entirely on the geometric principle of multi-view consistency for decomposition.

## Key Experimental Results

### Main Results (NeRF On-the-go Dataset)

| Scene | Occlusion Level | Metric | 3DGS | SLS-mlp | HybridGS |
|------|---------|------|------|---------|----------|
| Mountain | Low | PSNR↑ | 19.40 | 19.84 | **21.73** |
| Fountain | Low | PSNR↑ | 19.96 | 20.19 | **21.11** |
| Corner | Medium | PSNR↑ | 20.90 | 24.03 | **25.03** |
| Patio | Medium | PSNR↑ | 17.48 | 21.55 | **21.98** |
| Spot | High | PSNR↑ | 20.77 | 23.52 | **24.33** |
| Patio-High | High | PSNR↑ | 17.29 | 20.31 | **21.77** |

### RobustNeRF Dataset

| Scene | Metric | RobustNeRF | SLS-mlp | HybridGS |
|------|------|-----------|---------|----------|
| Statue | PSNR/SSIM/LPIPS | 20.60/0.76/0.15 | 22.54/0.84/0.13 | **22.93/0.87/0.10** |
| Android | PSNR/SSIM/LPIPS | 23.28/0.75/0.13 | 25.05/0.85/0.09 | **25.15/0.85/0.07** |
| Yoda | PSNR/SSIM/LPIPS | 29.78/0.82/0.15 | 33.66/0.96/0.10 | **35.32/0.96/0.07** |
| Crab(2) | PSNR/SSIM/LPIPS | - | 34.43/- /- | **35.17/0.96/0.08** |

### Key Findings
- HybridGS achieves SOTA across all scenes and occlusion levels. Specifically, the LPIPS metric is significantly better than all other methods, indicating perceptually superior rendering quality.
- It outperforms SLS-mlp (which uses semantic features) without relying on any semantic features (like DINOv2), demonstrating that the geometric consistency-based decoupling principle is more fundamental than semantic-based methods.
- The improvement is even more significant in high occlusion scenes (Patio-High) (+4.48 PSNR compared to 3DGS), indicating the method's superior advantage in challenging scenarios.
- The largest improvement is observed in the Yoda scene of RobustNeRF (+1.66 PSNR over SLS-mlp), where transient and static objects share similar semantics, validating the advantage of not relying on semantic differentiation.
- The transient masks generated by 2D Gaussians are of high quality, reasonably identifying transient objects like pedestrians and vehicles.

## Highlights & Insights
- Choosing the representation dimensionality (2D vs 3D) based on the essence of multi-view geometric consistency is an extremely elegant design: it requires no extra semantic features or pre-trained models, as the decoupling capability directly stems from the geometric properties of the representations themselves. This is more fundamental than the paradigm of "detecting transients first and then ignoring them".
- The multi-view regulated frustum intersection strategy is a clever and computationally efficient design: it both reinforces the multi-view constraints on static elements and reduces the computational load by limiting the optimization region.
- The dual role of 2D Gaussians, which both model the transient object images and naturally generate transient masks, fully utilizes the structural properties of the representation method itself.
- This method can be generalized to any scenario requiring the separation of "consistent" and "inconsistent" signals, such as illumination changes, weather fluctuations, etc.

## Limitations & Future Work
- Maintaining an independent set of 2D Gaussians for each training image results in high storage overhead when the number of training images is large.
- The modeling of transient objects by 2D Gaussians is planar—for large transient objects with 3D structures (e.g., a large vehicle occupying half of the frame), the planar assumption might be insufficient.
- The multi-stage training increases the complexity of hyperparameter tuning (e.g., the ratio of iterations per stage, the choice of $K$, etc.).
- The method assumes that transient objects appear in only a few views—if an object repeatedly appears in many views (e.g., a permanently parked vehicle), it might be incorrectly modeled as a static element by 3DGS.
- Future work could consider leveraging temporal information in video sequences to further differentiate moving objects from static ones.

## Related Work & Insights
- **vs RobustNeRF**: RobustNeRF uses robust loss to reduce the weight of inconsistent observations but does not model transient objects. HybridGS explicitly models transients and does not rely on manually set robust thresholds.
- **vs NeRF On-the-go**: It utilizes DINOv2 features to predict uncertainty, but semantic features are ambiguous for identifying transients. HybridGS is purely geometry-driven without using any semantic features.
- **vs SpotLessSplats (SLS-mlp)**: SLS-mlp combines semantic clustering and robust optimization, and was previously the strongest 3DGS method. HybridGS comprehensively outperforms it without using external features, demonstrating that "using the right representation" is more important than "using the right features".

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Proposes the first hybrid 2D+3D Gaussian representation to decompose scenes based on the essence of multi-view consistency. The concept is elegant and novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Fully evaluated on two standard datasets and compared with multiple baselines, although the ablation study is not detailed enough in the main paper.
- Writing Quality: ⭐⭐⭐⭐ The motivation and design of the method are clearly explained, and the overall structure is well-organized.
- Value: ⭐⭐⭐⭐⭐ Addresses a core challenge of 3DGS in real-world applications. The method is elegant and practical, with the potential to become a standard solution for 3DGS reconstruction in scenes with distractor elements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] HyperGS: Hyperspectral 3D Gaussian Splatting](hypergs_hyperspectral_3d_gaussian_splatting.md)
- [\[CVPR 2025\] SfM-Free 3D Gaussian Splatting via Hierarchical Training](sfm-free_3d_gaussian_splatting_via_hierarchical_training.md)
- [\[CVPR 2025\] Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting](rethinking_end-to-end_2d_to_3d_scene_segmentation_in_gaussian_splatting.md)
- [\[CVPR 2025\] Ref-GS: Directional Factorization for 2D Gaussian Splatting](ref-gs_directional_factorization_for_2d_gaussian_splatting.md)
- [\[CVPR 2025\] S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting](s2gaussian_sparse-view_super-resolution_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
