---
title: >-
  [Paper Note] High-Precision Self-Supervised Monocular Depth Estimation with Rich-Resource Prior
description: >-
  [ECCV 2024][3D Vision][Self-Supervised Depth Estimation] This paper proposes RPrDepth, which utilizes features and predictions of "rich-resource" models (such as multi-frame and high-resolution models) as priors during training. Through a prior depth fusion module and a rich-resource guided loss, the model achieves or even exceeds the depth estimation accuracy of multi-frame high-resolution models while performing inference using only a **low-resolution single image**.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Self-Supervised Depth Estimation"
  - "Monocular Depth Estimation"
  - "Knowledge Distillation"
  - "Prior Fusion"
  - "Attention-guided Feature Selection"
date: 2026-05-08
content_hash: 6008643b8c2c6ec6
---

# High-Precision Self-Supervised Monocular Depth Estimation with Rich-Resource Prior

**Conference**: ECCV 2024  
**arXiv**: [2408.00361](https://arxiv.org/abs/2408.00361)  
**Code**: [https://github.com/wencheng256/RPrDepth](https://github.com/wencheng256/RPrDepth)  
**Area**: 3D Vision  
**Keywords**: Self-Supervised Depth Estimation, Monocular Depth Estimation, Knowledge Distillation, Prior Fusion, Attention-guided Feature Selection

## TL;DR

This paper proposes RPrDepth, which utilizes features and predictions of "rich-resource" models (such as multi-frame and high-resolution models) as priors during training. Through a prior depth fusion module and a rich-resource guided loss, the model achieves or even exceeds the depth estimation accuracy of multi-frame high-resolution models while performing inference using only a **low-resolution single image**.

## Background & Motivation

**Background**: Self-supervised monocular depth estimation avoids dependency on LiDAR annotations using view-synthesis reconstruction loss. Currently, most top-performing methods rely on "rich-resource" inputs, such as multi-frame images, high-resolution images, and even future frames.

**Limitations of Prior Work**: Rich-resource inputs are often unavailable in practical applications. For example, multi-frame data cannot be obtained when a vehicle is stationary, and future frames are nonexistent in real-time systems. This severely limits the practicality of high-performance methods.

**Key Challenge**: A single low-resolution image lacks key information encoded in rich-resource inputs (such as inter-frame disparity). This fundamental gap in information volume imposes a performance ceiling.

**Goal**: Achieve the depth estimation accuracy of rich-resource models during the inference phase using only a single low-resolution image.

**Key Insight**: Although rich-resource data is unavailable during inference, **it is available during training**. By utilizing the features and predictions of rich-resource models as offline prior information, the information gap of low-resource inputs can be bridged through feature retrieval and fusion.

**Core Idea**: Construct an offline rich-resource reference feature database. During inference, retrieve similar prior features for each pixel, and reconstruct depth accuracy comparable to rich-resource models after fusion.

## Method

### Overall Architecture

The training phase contains two branches:
- **Upper branch (Rich-resource branch)**: Uses a pretrained ManyDepth-HR (multi-frame + high-resolution) with fixed parameters that are not updated. It provides reference features $f_r$ and reference depth $D_r$.
- **Lower branch (Single-image branch)**: The target model is based on DIFFNet, receiving a single low-resolution image and improving performance through prior fusion.

During inference, only the single-image branch and the compressed prior data (compressed from 2.6 million pixels to 25k pixels) are retained.

### Key Designs

1. **Prior Depth Fusion Module**: Contains two fusion schemes:

   **Pixel-wise Fusion**: Retrieves the most similar pixels from the reference features via an affinity matrix. Specifically, target features $F_s$ and reference features $F_r$ are aligned in dimension, and the affinity is calculated as:

    $\mathcal{A} = \text{Softmax}(F_s \otimes F_r)$

   The affinity matrix is then used to construct pixel-wise reference features and reference depths:

    $F_c = \mathcal{A} \times f_r, \quad D_c = \mathcal{A} \times D_r$

   Here, $F_c$ provides spatial geometric priors, and $D_c$ provides direct depth priors.

   **Depth-hint Fusion**: Uses a Transformer multi-head attention mechanism with target features $F_s$ as Query, and reference features $f_r$ as Key and Value:

    $F_d = \text{MHA}(Q, K, V)$

   This global attention fusion captures macro-level prior information, complementing the pixel-wise fusion.

   The results of both fusion methods are concatenated with the single-image features and compressed to the original channel count via convolution, yielding the feature $F_o$ containing rich information.

2. **Rich-resource Guided Loss**: Composed of two parts:

   **View-reconstruction Loss**: Uses the rich-resource input (high-resolution multi-frame) as the reconstruction target, providing a more precise supervision signal than the low-resolution source image:

    $\mathcal{L}_{\text{vp}} = l_{vp}(\text{Resize}(D_o), I_r)$

   **Consistency Loss**: Uses the depth prediction of the rich-resource model as pseudo-labels. Since self-supervised models predict relative disparity rather than absolute depth, scale differences exist across models. Therefore, the **gradient difference** is minimized instead of the direct value difference:

    $\mathcal{L}_c = \|\tilde{G}_{x,y}(D_o) - \tilde{G}_{x,y}(D_p)\|_1$

   where $\tilde{G}_{x,y}(\cdot)$ represents the normalized sum of gradients in the x and y directions. This forces the model to generate depth discontinuities at edges consistent with the rich-resource model.

   **Auxiliary Loss**: Guides the learning of the affinity matrix: $\mathcal{L}_{\text{aux}} = \|D_p - \text{Resize}(D_c)\|_1$

   Total loss: $\mathcal{L} = \alpha \mathcal{L}_{\text{vp}} + \beta \mathcal{L}_c + \mathcal{L}_{\text{aux}}$

3. **Attention Guided Feature Selection**: During inference, retrieving reference features from the complete reference set poses a large computational overhead. The solution is:

    - Calculate the average attention weights of all reference pixels on the validation set: $W_{\text{avg}} = \frac{1}{N}\sum_{i=1}^{N}(\mathcal{A}_i + \mathcal{A}_{\text{MHA},i})$
    - Select the subset of pixels with the highest weights as the compressed prior data.
    - Compress from 2.6 million pixels to 25k pixels (**1%**), replacing the prior set, and fine-tune for several epochs.
    - Surprising key finding: The performance improves rather than drops after compression, as it filters out interference from irrelevant pixels.

### Loss & Training

- Uses DIFFNet (based on HR-Net) as the single-image baseline model.
- Uses ManyDepth-HR as the rich-resource guidance model (multi-frame + high-resolution + future frames).
- Reference dataset: Randomly selects 2,000 triplets from the training set, extracting features offline (1% pixel sampling × 2,000 images).
- End-to-end training while fixing the parameters of the rich-resource branch.
- Replaces the reference set and fine-tunes after completing feature selection.

## Key Experimental Results

### Main Results (KITTI Eigen Split, 640×192 Resolution, Mono Training)

| Method | Input Frames | Abs Rel↓ | Sq Rel↓ | RMSE↓ | $\delta<1.25$↑ |
|------|----------|----------|---------|-------|-----------------|
| Monodepth2 | 1 | 0.115 | 0.903 | 4.863 | 0.877 |
| DIFFNet (Baseline) | 1 | 0.102 | 0.764 | 4.483 | 0.896 |
| ManyDepth (Multi-frame) | 2 | 0.098 | 0.770 | 4.459 | 0.900 |
| **RPrDepth (Single frame)** | **1** | **0.097** | **0.658** | **4.279** | **0.900** |

High-resolution (1024×320) results:

| Method | Input Frames | Abs Rel↓ | RMSE↓ | $\delta<1.25$↑ |
|------|----------|----------|-------|-----------------|
| ManyDepth-HR (Guidance Model) | 2 | 0.093 | 4.245 | 0.909 |
| **RPrDepth (Single frame)** | **1** | **0.091** | **4.098** | **0.910** |

**Single-frame RPrDepth outperforms its own multi-frame high-resolution teacher model!**

### Ablation Study

| Component | Abs Rel↓ | RMSE↓ | $\delta<1.25$↑ |
|------|----------|-------|-----------------|
| Baseline (DIFFNet) | 0.102 | 4.483 | 0.896 |
| + PDF (Prior Fusion) | 0.098 | 4.284 | 0.898 |
| + AGFS (Feature Selection) | 0.098 | 4.240 | 0.898 |
| + RGL (Guided Loss) | 0.100 | 4.321 | 0.897 |
| + Full (All) | 0.097 | 4.279 | 0.900 |

### Key Findings

- The prior depth fusion module is the primary source of performance improvement (Abs Rel: 0.102 → 0.098).
- Compressing the reference set by 99% via feature selection actually continues to reduce RMSE (4.284 → 4.240), as streamlining the features eliminates noise interference.
- RPrDepth exhibits the best generalization performance, achieving the top result in the Make3D cross-domain test (Abs Rel 0.288 vs BRNet 0.302).
- The gradient consistency loss effectively resolves the scale inconsistency problem between different models.

## Highlights & Insights

- **Training-Inference Decoupling**: Build priors using rich-resource data during training, while requiring only a single low-resolution image during inference, offering high practicality.
- **Student Outperforms Teacher**: The single-frame model surpasses the multi-frame high-resolution teacher model through prior fusion (multi-frame methods are often detrimental in moving object regions in autonomous driving, which can be corrected by a single frame + priors).
- **Philosophy of Feature Selection**: Less is more—selecting the top 1% most representative reference pixels is more effective than retaining all data.
- **Gradient-level rather than Value-level Consistency**: Cleverly bypasses the scale ambiguity problem in self-supervised depth estimation.

## Limitations & Future Work

- The reference feature set requires offline preprocessing and storage, which still introduces extra deployment overhead despite only being 25k pixels.
- Prior quality is constrained by the capability of the rich-resource model; if the teacher model fails in certain scenarios, the prior information will also be unreliable.
- Currently, only ManyDepth has been verified as the rich-resource model. Stronger teachers, such as the Transformer-based DPT, could be explored.
- The diversity of the reference dataset must cover the target scene distribution; cross-domain deployment requires updating the reference set.
- The approach can be extended to knowledge transfer for other dense prediction tasks (e.g., optical flow, semantic segmentation).

## Related Work & Insights

- **ManyDepth**: A representative method for multi-frame self-supervised depth estimation and the main teacher model for RPrDepth, providing cost volume features.
- **DIFFNet**: An efficient single-frame baseline based on HR-Net, serving as the student backbone of RPrDepth.
- **Monodepth2**: Established foundational designs for self-supervised monocular depth estimation, such as the min-reprojection loss.
- **Knowledge Distillation Perspective**: Can be viewed as a more granular teacher-student learning scheme—not only distilling predictions but also distilling spatial priors at the feature level.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The problem formulation of "rich resources during training, low resources during inference" has actual practical value, and the prior retrieval + fusion scheme is more flexible than direct distillation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Conducted experiments on three datasets: KITTI, Make3D, and Cityscapes, covering multiple training modes with complete ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — The motivation is clearly articulated, and the pipeline figures and tables are well-designed.
- **Value**: ⭐⭐⭐⭐⭐ — Directly reduces the sensor requirements for depth estimation in autonomous driving, with no additional overhead during inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] ProDepth: Boosting Self-Supervised Multi-Frame Monocular Depth with Probabilistic Fusion](prodepth_boosting_self-supervised_multi-frame_monocular_depth_with_probabilistic.md)
- [\[ECCV 2024\] Improving Domain Generalization in Self-Supervised Monocular Depth Estimation via Stabilized Adversarial Training](improving_domain_generalization_in_self-supervised_monocular_depth_estimation_vi.md)
- [\[NeurIPS 2025\] Jasmine: Harnessing Diffusion Prior for Self-Supervised Depth Estimation](../../NeurIPS2025/3d_vision/jasmine_harnessing_diffusion_prior_for_self-supervised_depth_estimation.md)
- [\[ECCV 2024\] DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation](diffusiondepth_diffusion_denoising_approach_for_monocular_depth_estimation.md)
- [\[ECCV 2024\] Diffusion Models for Monocular Depth Estimation: Overcoming Challenging Conditions](diffusion_models_for_monocular_depth_estimation_overcoming_challenging_condition.md)

</div>

<!-- RELATED:END -->
