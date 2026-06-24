---
title: >-
  [Paper Note] DeSplat: Decomposed Gaussian Splatting for Distractor-Free Rendering
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] DeSplat proposes decomposing 3D Gaussian Splatting into static scene Gaussians and view-specific distractor Gaussians. It accomplishes scene-distractor separation purely based on volume rendering, requiring no external semantic models. It achieves comparable distractor-free novel view synthesis performance to prior methods across three benchmark datasets without sacrificing rendering speed.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "distractor removal"
  - "scene decomposition"
  - "novel view synthesis"
  - "alpha compositing"
date: 2026-05-08
content_hash: cd058c57c7413cce
---

# DeSplat: Decomposed Gaussian Splatting for Distractor-Free Rendering

**Conference**: CVPR 2025  
**arXiv**: [2411.19756](https://arxiv.org/abs/2411.19756)  
**Code**: [https://github.com/AaltoML/desplat/](https://github.com/AaltoML/desplat/)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, distractor removal, scene decomposition, novel view synthesis, alpha compositing

## TL;DR

DeSplat proposes decomposing 3D Gaussian Splatting into static scene Gaussians and view-specific distractor Gaussians. It accomplishes scene-distractor separation purely based on volume rendering, requiring no external semantic models. It achieves comparable distractor-free novel view synthesis performance to prior methods across three benchmark datasets without sacrificing rendering speed.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) enables rapid novel view synthesis of static 3D scenes. However, during real-world scene acquisition, distractors such as pedestrians, vehicles, and illumination changes violate the multi-view consistency assumption, leading to artifacts in the 3D reconstruction.

**Limitations of Prior Work**: Most existing distractor removal methods rely on external pre-trained semantic models to identify distractors. For example, SpotLessSplats uses features from vision foundation models like DINO or MAE to detect inconsistent regions, while WildGaussians leverages DINO features and appearance embeddings. Although effective, these methods introduce additional computational overhead (preprocessing or semantic inference during optimization) and suffer from dependency on the quality and generalization of pre-trained models.

**Key Challenge**: Classifying which image content belongs to the static scene and which to transient distractors purely using RGB information and multi-view geometry, without relying on external models, is an under-constrained problem.

**Goal**: To design a method based purely on volume rendering of Gaussian primitives to achieve explicit separation of distractors and the static scene in the alpha compositing stage, without requiring any pre-trained semantic models.

**Key Insight**: It is observed that the critical characteristic of distractors is that they are "view-specific"—they appear only in a subset of the training views. Therefore, a set of Gaussian primitives can be initialized independently for each camera view to model the distractors specific to that view, while globally shared Gaussian primitives model the view-consistent static scene.

**Core Idea**: To decompose rendering into two stages during alpha compositing—rendering view-specific distractor Gaussians first and then rendering global static Gaussians. The two stages are cascaded via transmittance, allowing distractor Gaussians to "occlude" the static scene behind them, thereby achieving a natural scene decomposition.

## Method

### Overall Architecture

The input consists of a set of multi-view images with distractors and their corresponding camera parameters. The system maintains two sets of Gaussian primitives: (1) global static Gaussians $\mathcal{G}_{static}$ to model the 3D scene shared across all views, and (2) view-specific distractor Gaussians $\{\mathcal{G}_{dyn}^k\}_{k=1}^K$, with an independent set for each camera view $k$. During rendering, the alpha compositing contribution of distractor Gaussians is computed first to obtain the color and accumulated opacity of the distractor layer. The remaining transmittance is then passed to the static Gaussian layer to continue the composition. The final image is a fusion of the two layers.

### Key Designs

1. **View-Specific Distractor Gaussian Initialization**:

    - **Function**: To independently create a set of 2D Gaussian primitives for each training view, specifically modeling distractors in that view.
    - **Mechanism**: For each training camera view $k$, a set of Gaussians $\mathcal{G}_{dyn}^k$ is initialized on the image plane of that view. These Gaussians are initialized on the near plane in front of the camera, and their depth and spatial extent are constrained within a thin layer so that they only affect the corresponding view. Unlike global static Gaussians, distractor Gaussians do not participate in rendering other views.
    - **Design Motivation**: The key characteristic of distractors is view-dependence—a pedestrian may only appear in a few frames. Modeling each view independently naturally captures this unshared information while preventing the distractor Gaussians from being incorrectly "generalized" to other views.

2. **Decomposed Alpha Compositing**:

    - **Function**: To achieve explicit separation of distractors and the static scene during the alpha compositing stage of volume rendering.
    - **Mechanism**: When rendering each ray, all involved Gaussians (including distractor and static Gaussians) are first sorted by depth. For rays in the current view $k$, the distractor Gaussians $\mathcal{G}_{dyn}^k$ participate in alpha compositing before the static Gaussians. Specifically, the color of the distractor layer is formulated as $\hat{C}_{dyn} = \sum_i T_i^{dyn} \alpha_i^{dyn} c_i^{dyn}$, and the accumulated transmittance of the distractor layer $T_{dyn}$ is passed to the static layer: $\hat{C}_{static} = T_{dyn} \sum_j T_j^{static} \alpha_j^{static} c_j^{static}$. The final pixel color is $\hat{C} = \hat{C}_{dyn} + \hat{C}_{static}$. This cascaded compositing allows distractors to naturally "occlude" the static scene.
    - **Design Motivation**: Compared to removing distractors via post-processing, decomposition inside the rendering pipeline enables correct gradient propagation, allowing the two sets of Gaussians to automatically separate roles during training—view-specific signals are assigned to distractor Gaussians, while view-consistent signals are assigned to static Gaussians.

3. **Self-Supervised Separation Without External Semantic Models**:

    - **Function**: To automatically separate distractors from the static scene using only rendering reconstruction loss.
    - **Mechanism**: The training loss of the entire system consists only of standard photometric reconstruction loss (L1 + SSIM/LPIPS), without any additional segmentation or classification supervision. Separation is achieved solely through architectural constraints: distractor Gaussians can only contribute to the view they were created for, meaning view-consistent content is naturally explained only by static Gaussians. During optimization, distractor Gaussians automatically absorb view-specific content (distractors, illumination changes, etc.), while static Gaussians learn view-consistent geometry and appearance.
    - **Design Motivation**: This avoids the computational overhead and domain adaptation challenges introduced by external models, making the method simpler and more generalizable.

### Loss & Training

The training loss is a combination of standard L1 reconstruction loss and SSIM loss. During training, the static Gaussians and distractor Gaussians for all views are optimized jointly. Distractor Gaussians employ standard 3DGS densification and pruning strategies. During novel view synthesis, only the static Gaussians $\mathcal{G}_{static}$ are rendered, while all distractor Gaussians are discarded.

## Key Experimental Results

### Main Results (RobustNeRF Dataset)

| Method | PSNR↑| SSIM↑ | LPIPS↓ | Requires External Model |
|------|-------|-------|--------|------------|
| Splatfacto (Baseline 3DGS) | ~25 | ~0.85 | ~0.20 | No |
| SpotLessSplats | ~28 | ~0.90 | ~0.15 | Yes (DINO/MAE) |
| WildGaussians | ~27 | ~0.88 | ~0.16 | Yes (DINO) |
| **DeSplat (Ours)** | **~28** | **~0.90** | **~0.15** | **No** |

### Ablation Study

| Configuration | PSNR | Description |
|------|------|------|
| Full DeSplat | Optimal | Complete model with view-specific distractor Gaussians + decomposed alpha compositing |
| w/o distractor Gaussians | Significantly decreased | Degenerates to standard 3DGS with severe distractor artifacts |
| Shared distractor Gaussians (non view-specific) | Decreased | Shared Gaussians fail to capture view-specific distractors |
| Post-processing anomaly detection only | Decreased | Lacks end-to-end optimization, leading to poor separation |

### Key Findings

- **DeSplat achieves comparable performance to SpotLessSplats without using external semantic models**: This indicates that architectural constraints based purely on volume rendering are sufficient for effective scene decomposition, rendering external models unnecessary.
- **View-specific initialization is crucial**: If distractor Gaussians are shared across all views, they fail to distinguish between distractors and the static scene due to mixed gradient signals.
- **Decomposed rendering maintains the speed advantages of 3DGS**: Since no additional semantic inference is required, the rendering speed of DeSplat is nearly identical to standard 3DGS.
- **Consistently effective across three benchmarks**: Including RobustNeRF (synthetic distractors), Phototourism (in-the-wild web images), and custom datasets, demonstrating strong generalization capability.
- **Explicit scene decomposition outputs can be used for downstream tasks**: The rendering results of the distractor Gaussians can be utilized as distractor segmentation masks.

## Highlights & Insights

- **Replacing semantic supervision with architectural constraints is elegant**: By restricting distractor Gaussians to only participate in rendering their corresponding views, automatic separation is achieved without any mask annotations. This "constraint-as-supervision" concept can be transferred to other tasks requiring decomposed representations.
- **Decomposed alpha compositing is the core innovation**: Performing scene decomposition inside the rendering pipeline rather than externally ensures correct gradient propagation and training stability.
- **Method simplicity**: The overall approach does not introduce any additional network modules (such as encoders or classifiers). It achieves distractor removal solely by redesigning the organization of Gaussian primitives and the rendering process, resulting in a clean engineering implementation.

## Limitations & Future Work

- **Assumptions on distractor ratio**: The method implicitly assumes that distractors appear in only a minority of views. If the majority of views contain the same distractor, it might be mistakenly classified as a static element.
- **Storage overhead of view-specific Gaussians**: Maintaining an independent set of distractor Gaussians for each training view can become a GPU memory bottleneck when the number of training views is large.
- **Lack of temporal modeling for dynamic distractors**: The method treats each view independently, without leveraging temporal consistency between adjacent frames in videos.
- **Inability to handle global illumination changes**: Illumination changes affect all pixels rather than local regions, making them difficult to model effectively with view-specific Gaussians.
- **Future directions**: Lightweight frequency-aware features or appearance-embeddings could be introduced to better handle lighting variations. Additionally, substituting 3DGS with 2DGS for modeling distractors could be explored to reduce parameter count.

## Related Work & Insights

- **vs SpotLessSplats**: SpotLessSplats relies on DINO/MAE to detect inconsistent regions and downweight them. In contrast, this work achieves automatic separation via architectural design without requiring external models, making it much simpler.
- **vs WildGaussians**: WildGaussians uses DINO features and appearance embeddings to handle in-the-wild images. While more comprehensive (including modeling of lighting changes), it incurs significantly higher computational overhead.
- **vs HybridGS (Concurrent Work)**: HybridGS uses 2DGS to model distractors and 3DGS to model static scenes. The concept is similar but employs different Gaussian representations.
- **vs NeRF-W**: NeRF-W was the first to propose modeling appearance changes and transient objects for in-the-wild images, but its NeRF-based pipeline is extremely slow. DeSplat implements a similar idea efficiently within the 3DGS framework.

## Rating

- Novelty: ⭐⭐⭐⭐ The concept of achieving scene decomposition purely based on volume rendering is elegant and novel.
- Experimental Thoroughness: ⭐⭐⭐ The evaluation covers three datasets, though it lacks detailed quantitative data tables.
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear and key concepts are intuitively communicated.
- Value: ⭐⭐⭐⭐ It provides a distractor removal solution without external dependencies, which is highly valuable for applying 3DGS to in-the-wild scenes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting](selfsplat_pose-free_and_3d_prior-free_generalizable_3d_gaussian_splatting.md)
- [\[CVPR 2025\] SfM-Free 3D Gaussian Splatting via Hierarchical Training](sfm-free_3d_gaussian_splatting_via_hierarchical_training.md)
- [\[CVPR 2025\] Learnable Infinite Taylor Gaussian for Dynamic View Rendering](learnable_infinite_taylor_gaussian_for_dynamic_view_rendering.md)
- [\[ICCV 2025\] DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction](../../ICCV2025/3d_vision/degauss_dynamic-static_decomposition_with_gaussian_splatting_for_distractor-free.md)
- [\[CVPR 2025\] SVG-IR: Spatially-Varying Gaussian Splatting for Inverse Rendering](svg-ir_spatially-varying_gaussian_splatting_for_inverse_rendering.md)

</div>

<!-- RELATED:END -->
