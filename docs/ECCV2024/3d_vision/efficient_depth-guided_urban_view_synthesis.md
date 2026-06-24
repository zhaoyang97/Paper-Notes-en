---
title: >-
  [Paper Note] Efficient Depth-Guided Urban View Synthesis (EDUS)
description: >-
  [ECCV 2024][3D Vision][Urban View Synthesis] This work proposes EDUS, which leverages noisy geometric priors (monocular/stereo depth) to guide generalizable NeRF. Through a tri-part decomposition consisting of a foreground 3D CNN and background/sky image-based rendering, it achieves fast feed-forward inference and efficient scene-by-scene fine-tuning under sparse urban views.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Urban View Synthesis"
  - "Generalizable NeRF"
  - "Sparse Views"
  - "Depth Guidance"
  - "Autonomous Driving"
date: 2026-05-08
content_hash: 7d229dcafddfd344
---

# Efficient Depth-Guided Urban View Synthesis (EDUS)

**Conference**: ECCV 2024  
**arXiv**: [2407.12395](https://arxiv.org/abs/2407.12395)  
**Code**: [https://xdimlab.github.io/EDUS/](https://xdimlab.github.io/EDUS/)  
**Area**: 3D Vision  
**Keywords**: Urban View Synthesis, Generalizable NeRF, Sparse Views, Depth Guidance, Autonomous Driving

## TL;DR

This work proposes EDUS, which leverages noisy geometric priors (monocular/stereo depth) to guide generalizable NeRF. Through a tri-part decomposition consisting of a foreground 3D CNN and background/sky image-based rendering, it achieves fast feed-forward inference and efficient scene-by-scene fine-tuning under sparse urban views.

## Background & Motivation

**Background**: NeRF-based urban novel view synthesis methods (e.g., Urban Radiance Fields, Block-NeRF) have made progress but rely on dense training images and substantial computational resources.

**Limitations of Prior Work**:
   - Vehicles travel at high speeds in autonomous driving, resulting in most areas being captured by only 2-3 views, leading to a severe lack of overlap between views.
   - Forward camera motion leads to small parallax angles, increasing reconstruction uncertainty.
   - Generalizable NeRF methods (e.g., IBRNet, MVSNeRF) rely on feature matching to recover geometry, performing poorly in sparse, textureless urban scenes.
   - These methods select the nearest reference images for feature matching, which easily overfits to specific camera pose configurations, leading to a drastic performance drop when generalizing to different sparsities.

**Key Challenge**: How to build an efficient and generalizable urban view synthesis method that is robust to varying levels of sparsity?

**Goal**: To achieve novel view synthesis with fast feed-forward inference and efficient fine-tuning under sparse urban views.

**Key Insight**: Leveraging geometric priors (depth estimation) instead of feature matching to acquire geometric information, operating directly in the 3D space to avoid reliance on reference image poses.

**Core Idea**: Integrating depth estimation priors into a global 3D volume representation and processing it directly in world coordinates via a SPADE 3D CNN, making the method insensitive to the reference image pose configurations.

## Method

### Overall Architecture

EDUS decomposes unbounded urban scenes into three components: foreground (within the near volume), background (distant objects), and sky. Each component features an independent generalizable module. The core is the depth-guided foreground field: depth estimation $\rightarrow$ point cloud accumulation $\rightarrow$ 3D SPADE CNN volume feature extraction $\rightarrow$ combining with 2D image features $\rightarrow$ decoding color and density. Training is conducted on multiple scenes, while inference can be performed feed-forward or with fast fine-tuning.

### Key Designs

1. **Depth-Guided Generalizable Foreground Fields**:

    - **Function**: Leveraging depth estimation to construct 3D point clouds and extracting volume features via a 3D CNN to represent the near foreground regions.
    - **Mechanism**:
        - **Point Cloud Accumulation**: For $N$ input images, a depth estimator predicts depth maps $\{D_i\}$, which are back-projected to the 3D world coordinate system to form a point cloud $\mathcal{P} \in \mathbb{R}^{N_p \times 3}$: $\mathbf{x} = d\mathbf{R}_i\mathbf{K}^{-1}\mathbf{u} + \mathbf{t}_i$. Depth consistency checks are used to filter noise (threshold $\sigma = 0.2m$).
        - **SPADE 3D CNN**: Discretizes the point cloud into a volume $\mathbf{P} \in \mathbb{R}^{H \times W \times D \times 3}$ and extracts feature volume $\mathbf{F} = f_\theta^{3D}(\mathbf{P})$ via a SPADE CNN. The SPADE CNN contains 3 SPADE residual blocks and upsampling layers, maintaining appearance information through multi-resolution modulation.
        - **2D Feature Retrieval**: Selects $K=3$ nearest reference views and projects 3D points back to reference frames to retrieve as color features $\mathbf{f}_{fg}^{2D} \in \mathbb{R}^{3K}$.
        - **Decoding**: Density is determined solely by the 3D features: $\sigma_{fg} = g_\theta(\mathbf{f}_{fg}^{3D})$, while color is jointly predicted from 3D and 2D features: $\mathbf{c}_{fg} = h_\theta(\mathbf{f}_{fg}^{3D}, \mathbf{f}_{fg}^{2D}, \gamma(\mathbf{x}), \mathbf{d})$.
    - **Design Motivation**: Geometric priors are unaffected by reference image poses, making the method robust to density/sparsity variations. The 3D CNN processes the global volume independently, avoiding local cost volume overfitting to specific pose configurations. The SPADE CNN preserves appearance details better than a traditional U-Net.

2. **Generalizable Background Fields**:

    - **Function**: Handling far-away objects outside the foreground volume using image-based rendering.
    - **Mechanism**: Distant objects occupy small regions in the image and exhibit minor relative depth variations, meaning image-based rendering is sufficient for faithful reconstruction:
    $\sigma_{bg}, \mathbf{c}_{bg} = h_\theta^{bg}(\mathbf{f}_{bg}^{2D}, \gamma(\mathbf{x}), \mathbf{d})$
    - **Design Motivation**: Distant depth estimations are unreliable, and perspective projections cause minimal variation in background appearance, making image-based rendering sufficient.

3. **Generalizable Sky Fields**:

    - **Function**: Modeling the sky as a view-dependent environment map.
    - **Mechanism**: The sky has no physical collision, and appearance variation across frames is extremely small:
    $\mathbf{c}_{sky} = h_\theta^{sky}(\mathbf{f}_{sky}^{2D}, \mathbf{d})$
    - **Design Motivation**: The sky is an infinitely distant region that does not require positional information.

4. **Scene Decomposition**:

    - **Function**: Combining volume-rendered foreground/background results with sky color.
    - **Mechanism**: Sampled points along rays are assigned to either foreground or background modules based on their locations, and accumulated to obtain color and alpha values:
    $\mathbf{C} = \mathbf{C}^{(fg+bg)} + (1 - \alpha^{(fg+bg)})\mathbf{c}_{sky}$
   A pre-trained segmentation model provides sky mask overview/supervision.

### Loss & Training

- **Training Loss**: $\mathcal{L}_{training} = \mathcal{L}_{rgb} + \lambda_1\mathcal{L}_{lidar} + \lambda_2\mathcal{L}_{sky} + \lambda_3\mathcal{L}_{entropy}$
- **Fine-tuning Loss**: $\mathcal{L}_{fine-tuning} = \mathcal{L}_{rgb} + \lambda_2\mathcal{L}_{sky} + \lambda_3\mathcal{L}_{entropy}$ (without LiDAR)
- **LiDAR Loss**: Modified line-of-sight loss using exponentially decaying bound width $\epsilon$ (decaying from 0.5m to 0.1m):
    - $\mathcal{L}_{empty}$: Suppressing weights in proximal space.
    - $\mathcal{L}_{near}$: Encouraging concentration of density near the surface.
    - $\mathcal{L}_{dist}$: Suppressing weights in distal space.
- **Entropy Regularization**: Penalizes semi-transparent reconstructions, encouraging opaque rendering.
- **Training Tricks**: Randomly masking the input volume (similar to MAE to enhance completion capability), stratified sampling, and per-frame appearance embedding.
- **Training Details**: Adam optimizer, learning rate of $5 \times 10^{-3}$, $\lambda_1=0.1, \lambda_2=1, \lambda_3=0.002$, trained for 500k steps on an RTX 4090 for approximately 2 days.

## Key Experimental Results

### Main Results

Trained on KITTI-360 (80 scenes), tested on 5 validation scenes + 5 Waymo scenes.

| Method | Setting | KITTI-360 drop50% PSNR↑ | drop80% PSNR↑ | Waymo drop50% PSNR↑ |
|------|------|------------------------|---------------|---------------------|
| IBRNet | Feed-forward | 19.99 | 15.96 | 21.28 |
| MVSNeRF | Feed-forward | 17.73 | 16.50 | 19.58 |
| MuRF | Feed-forward | 22.19 | 18.69 | 23.12 |
| **EDUS** | **Feed-forward** | **21.93** | **19.63** | **23.16** |
| MuRF | Fine-tuning | 23.71 | 19.70 | 28.30 |
| **EDUS** | **Fine-tuning** | **24.43** | **20.91** | **28.45** |

### Comparison with Per-Scene Optimization Methods

| Method | drop50% PSNR | drop80% PSNR | drop90% PSNR | Time Cost |
|------|-------------|-------------|-------------|------|
| MixNeRF | 21.50 | 18.89 | 17.89 | ~51min |
| SparseNeRF | 21.34 | 19.18 | 17.94 | ~35min |
| 3DGS | **24.37** | 19.80 | 17.46 | ~29min |
| **EDUS (Fine-tuning)** | 24.43 | **20.91** | **19.16** | **~5min** |

### Ablation Study

| Configuration | Key Metrics | Note |
|------|---------|------|
| SPADE CNN vs U-Net | SPADE is better | U-Net produces blurry artifacts in novel scenes |
| 3D features only | Lacks high-frequency details | Point cloud discretization limits resolution |
| 2D features only | Poor geometry | Feature matching is unreliable in sparse scenes |
| Random masking | Enhances completion capability | MAE-like training strategy |

### Key Findings

- EDUS exhibits a more pronounced advantage under high sparsity (drop 80%/90%), because the geometric prior is independent of reference image poses.
- The global volume based method converges much faster than the local volume based method in MuRF (5 minutes vs 50 minutes).
- Strong cross-dataset generalization: The model trained on KITTI-360 performs well on Waymo.
- High memory efficiency: Full-resolution inference requires only 6GB of VRAM, whereas MuRF requires 16.2GB.

## Highlights & Insights

- **Geometric Priors Instead of Feature Matching**: The core insight is that although depth prediction is noisy, it is independent of reference image poses, and thus more robust than feature matching-based methods.
- **Global vs. Local Volume**: The global volume only needs to be updated once to adapt to a new scene. During fine-tuning, only the feature volume is updated rather than the entire network, significantly accelerating convergence.
- **Divide and Conquer**: The tripartite strategy of separating foreground, background, and sky allows each module to utilize the most appropriate representation (3D geometry vs. 2D image-based rendering).
- **Fine-tuning Efficiency**: State-of-the-art performance can be achieved with only 5 minutes of fine-tuning, which is 5-10x faster than other methods.

## Limitations & Future Work

- The foreground volume range is fixed ($\pm12.6\text{m} \times [-3, 9.8\text{m}] \times [-20, 31.2\text{m}$), which may not adapt to all scene layouts.
- The 3D CNN still exhibits a smoothing bias, where high-frequency details rely heavily on 2D features for supplementation.
- LiDAR supervision is used during training, but an RGB-only setting is assumed during testing.
- The voxel resolution of 0.2m limits fine-grained reconstruction.

## Related Work & Insights

- **vs. IBRNet/MVSNeRF**: These methods recover geometry based on feature matching, whereas EDUS operates directly in the 3D space using geometric priors.
- **vs. MuRF**: MuRF constructs a cost volume in the target view, which generalizes well but fine-tunes slowly. EDUS uses a global volume, enabling extremely fast fine-tuning.
- **vs. PointNeRF**: PointNeRF also uses point clouds but directly as a radiance field, whereas EDUS uses a 3D CNN to refine the noisy point clouds.
- **vs. 3DGS**: 3DGS performs excellently with dense views but suffers from drastic performance drops under high sparsity, while EDUS remains robust.
- **Insight**: The paradigm of combining geometric priors with generalizable architectures can be extended to indoor or dynamic scenes.

## Rating

- Novelty: ⭐⭐⭐⭐ The concept of using geometric priors instead of feature matching is simple yet effective, and the scene decomposition design is reasonable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Thorough ablation studies and comprehensive comparisons across multiple datasets, sparsity levels, and baselines.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with well-explained motivations.
- Value: ⭐⭐⭐⭐ Achieving SOTA results with 5-minute fine-tuning holds practical significance for autonomous driving applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Depth-Guided Bundle Sampling for Efficient Generalizable Neural Radiance Field Reconstruction](../../CVPR2025/3d_vision/depth-guided_bundle_sampling_for_efficient_generalizable_neural_radiance_field_r.md)
- [\[ECCV 2024\] MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images](mvsplat_efficient_3d_gaussian_splatting_from_sparse_multi-view_images.md)
- [\[ECCV 2024\] MegaScenes: Scene-Level View Synthesis at Scale](megascenes_scene-level_view_synthesis_at_scale.md)
- [\[ECCV 2024\] MVDD: Multi-View Depth Diffusion Models](mvdd_multi-view_depth_diffusion_models.md)
- [\[CVPR 2025\] Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](../../CVPR2025/3d_vision/multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)

</div>

<!-- RELATED:END -->
