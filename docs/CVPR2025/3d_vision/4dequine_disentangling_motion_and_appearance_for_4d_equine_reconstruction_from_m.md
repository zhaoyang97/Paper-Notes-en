---
title: >-
  [Paper Note] 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video
description: >-
  [CVPR 2025][3D Vision][4D Reconstruction] Disentangles 4D equine reconstruction from monocular video into motion estimation (AniMoFormer spatio-temporal Transformer) and appearance reconstruction (EquineGS single-image feed-forward 3DGS). Leveraging the VAREN parametric model and two large-scale synthetic datasets, it achieves SOTA geometry + appearance reconstruction results on real-world data and generalizes zero-shot to donkeys and zebras.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "4D Reconstruction"
  - "Equine Reconstruction"
  - "3D Gaussian Splatting"
  - "VAREN Model"
  - "Motion-Appearance Disentanglement"
date: 2026-05-08
content_hash: c20cd8e5759f41cb
---

# 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video

**Conference**: CVPR 2025  
**arXiv**: [2603.10125](https://arxiv.org/abs/2603.10125)  
**Code**: [Project Page](https://luoxue-star.github.io/4DEquine_Project_Page/)  
**Area**: 3D Vision / Animal Reconstruction / 4D Reconstruction  
**Keywords**: 4D Reconstruction, Equine Reconstruction, 3D Gaussian Splatting, VAREN Model, Motion-Appearance Disentanglement

## TL;DR
Disentangles 4D equine reconstruction from monocular video into motion estimation (AniMoFormer spatio-temporal Transformer) and appearance reconstruction (EquineGS single-image feed-forward 3DGS). Leveraging the VAREN parametric model and two large-scale synthetic datasets, it achieves SOTA geometry + appearance reconstruction results on real-world data and generalizes zero-shot to donkeys and zebras.

## Background & Motivation

**Background**: Monocular 4D animal reconstruction primarily follows two paradigms: (a) template-free methods (BANMo, RAC, etc.) that learn deformable radiance fields but lack geometric priors, resulting in coarse shapes; (b) SMAL/hSMAL parametric model-based methods (SMALR, Dessie, etc.) that possess geometric geometric priors but exhibit poor texture quality and require per-video optimization. More recently, methods like GART utilize 3D Gaussian Splatting for animal avatars, yet still require 360° videos + per-instance optimization.

**Limitations of Prior Work**: (a) Per-video optimization methods (GART) are computationally expensive and demand complete observational coverage; real-world videos often have limited viewpoints, leading to failed optimization. (b) Feed-forward methods (3D-Fauna, MagicPony, etc.) bypass per-instance optimization but must sacrifice shape realism to achieve generalizability. (c) Existing approaches rely on the coarser SMAL instead of using the latest highly accurate VAREN equine model.

**Key Challenge**: 4D reconstruction simultaneously estimates motion and appearance, which are highly coupled: inaccurate motion prevents correct appearance alignment, while poor appearance conversely degrades motion estimation. Joint optimization is slow and highly sensitive to incomplete observation.

**Goal**:
   - How to efficiently and accurately recover frame-by-frame equine motion (pose + shape sequence) from monocular video?
   - How to generate a high-fidelity animatable 3D Gaussian avatar feed-forward from minimal input (even a single image)?
   - How to train these networks under the shortage of real-world annotated data?

**Key Insight**: Explicitly disentangle 4D reconstruction into two independent sub-problems—dynamic motion recovery + static appearance reconstruction, using the VAREN parametric model as a bridge to connect them. Separate synthetic datasets are constructed to train the respective networks.

**Core Idea**: Once motion and appearance are decoupled, they can be solved individually using specialized networks and synthetic data. The appearance is then driven to each frame's pose via the skinning mechanism of the VAREN model, achieving efficient and high-quality 4D equine reconstruction.

## Method

### Overall Architecture
Given a monocular video, the pipeline splits into two branches:
1. **AniMoFormer** (motion branch): A spatio-temporal Transformer combined with post-optimization predicts the shape $\beta \in \mathbb{R}^{39}$, pose $\theta \in \mathbb{R}^{38\times3}$, and camera parameters of the VAREN model for each frame.
2. **EquineGS** (appearance branch): Selects a representative frame from the video to generate a 3D Gaussian avatar in the canonical space (55,486 Gaussian points) feed-forward.
3. **Fusion**: Deforms the canonical Gaussians into each frame's pose space using VAREN's LBS to render final 4D outputs.

### Key Designs

1. **VarenPoser Synthetic Motion Dataset**

    - **Function**: Creates a large-scale synthetic equine video dataset that provides ground truth VAREN parameters for motion network training.
    - **Mechanism**: Obtains pose parameters by fitting the VAREN model to the marker-based motion capture dataset PFERD, randomly assigns shape parameters to increase diversity, and generates textures using MV-Adapter. A key innovation is simulating three realistic camera trajectories (fixed, dolly, and orbit).
    - **Design Motivation**: Annotating real-world 4D equine data is extremely challenging. The synthetic dataset provides 1,171 video clips at 512×512 resolution and 60 FPS with precise ground-truth VAREN parameters.

2. **AniMoFormer Spatio-Temporal Transformer**

    - **Function**: Regresses temporally consistent VAREN parameters from a 16-frame video window.
    - **Mechanism**: A two-stage design: (1) a Spatial Transformer extracts per-frame spatial features; (2) a Temporal Transformer models temporal relationships via self-attention over an $N$-frame window; (3) a VAREN Transformer Decoder regresses the parameters. The training loss is:
    $$\mathcal{L} = \lambda_{varen}\mathcal{L}_{varen} + \lambda_{smooth}\mathcal{L}_{smooth} + \lambda_{2D}\mathcal{L}_{2D} + \lambda_{3D}\mathcal{L}_{3D}$$
      where $\mathcal{L}_{smooth}$ constrains parameter smoothness between adjacent frames.
    - **Design Motivation**: Single-frame methods (AniMer) cannot exploit temporal information, leading to inter-frame jitter. Spatio-temporal Transformers naturally support sliding windows to handle videos of arbitrary length.

3. **Post-Optimization**

    - **Function**: Accurately aligns the predicted mesh from the Transformer with 2D images.
    - **Mechanism**: Projects the 3D mesh onto the 2D plane via a differentiable renderer, compares it with pseudo-GT keypoints extracted by ViTPose++ and masks from Samurai, and optimizes the pose parameters to make them pixel-aligned.
    - **Design Motivation**: The predicted mesh from the Transformer may suffer from offsets relative to 2D evidence. Post-optimization closes this gap using image-level supervision. They are mutually indispensable (ablation confirms performance drops if either is removed).

4. **EquineGS Feed-forward Appearance Reconstruction**

    - **Function**: Generates an animatable 3DGS avatar feed-forward from a single image.
    - **Mechanism**:
        - **Point Initialization**: Subdivides the VAREN template mesh (midpoint of each edge + dividing each face into four) to upsample 13,873 vertices to 55,486 initial Gaussian positions.
        - **Dual-Stream Feature Extraction**: DINOv3 ViT-L extracts multi-scale image features $F_I \in \mathbb{R}^{784\times1024}$, while a Point Transformer encodes 3D points into $F_P \in \mathbb{R}^{N_G\times1024}$.
        - **DSTG Decoder**: An improved MMDiT block containing a three-step fusion: global context vector extraction $\rightarrow$ joint attention over image + point features $\rightarrow$ MLP prediction of each Gaussian's attributes (offset, rotation, scale, color, opacity).
    - **Design Motivation**: An equine's appearance remains constant during a video, and real-world videos have limited viewpoints $\rightarrow$ single-frame feed-forward is more reasonable than per-frame optimization. DSTG achieves better fusion than standard cross-attention (validated in ablation).

5. **VarenTex Synthetic Appearance Dataset**

    - **Function**: Generates 150K high-quality multi-view equine images for training EquineGS.
    - **Mechanism**: Renders normal maps and CCM from VarenPoser meshes $\rightarrow$ generates reference images via ControlNet $\rightarrow$ generates consistent multi-view training images using the UniTex multi-view diffusion model.
    - **Design Motivation**: The texture quality of VarenPoser is not high-fidelity enough, and the appearance network requires multi-view data rather than monocular videos.

### Loss & Training
- **AniMoFormer**: L2 loss on VAREN parameters + smoothness loss + L1 loss on 2D/3D keypoints.
- **Post-Optimization**: 2D keypoint loss + mask L1 loss + smoothness loss + regularization.
- **EquineGS**: $\mathcal{L} = \lambda_{image}(\|I-\hat{I}\|_1 + \text{LPIPS}) + \lambda_{mask}\|M-\hat{M}\|_1 + \lambda_{reg}\mathcal{L}_{reg}$.
- Trained entirely on synthetic data and evaluated directly on real-world data to demonstrate generalization.

## Key Experimental Results

### Main Results (Motion Estimation)

| Method | APT36K PCK@0.05↑ | APT36K PCK@0.1↑ | APT36K Accel↓ | AiM PCK@0.05↑ | VarenPoser CD↓ |
|------|-----------------|-----------------|--------------|---------------|---------------|
| 3D-Fauna | 20.1 | 51.4 | 189.3 | 33.3 | 43.0 |
| 4D-Fauna | 25.5 | 53.5 | 177.7 | 46.5 | 38.5 |
| AniMer | 44.5 | 76.6 | 130.5 | 55.5 | 15.2 |
| **AniMoFormer** | **61.8** | **83.9** | **128.6** | **84.2** | **3.4** |

### Appearance Reconstruction (Novel View/Pose on AiM)

| Method | Horse PSNR↑ | Horse SSIM↑ | Horse LPIPS↓ | Zebra PSNR↑ | Zebra LPIPS↓ |
|------|------------|------------|-------------|------------|-------------|
| 3D-Fauna | 12.20 | 0.7205 | 0.2782 | 12.33 | 0.3318 |
| GART (full opt.) | 16.19 | 0.7819 | 0.2308 | 15.21 | 0.2287 |
| GART* (few-shot) | 15.42 | 0.7550 | 0.2452 | 14.31 | 0.2973 |
| **4DEquine** | 15.66 | **0.8364** | **0.1720** | **15.54** | **0.2000** |

### Ablation Study

| Configuration | APT36K PCK@0.05 | AiM PCK@0.05 | AiM Accel |
|------|----------------|--------------|-----------|
| w/o PO & Temporal | 37.1 | 45.1 | 30.6 |
| w/o PO | 37.7 | 47.8 | 25.7 |
| w/o Temporal | 57.9 | 82.9 | 24.7 |
| Full AniMoFormer | **61.8** | **84.2** | **21.8** |

### Key Findings
- **AniMoFormer significantly outperforms all baselines**: PCK@0.05 rises from AniMer's 44.5 to 61.8 (APT36K), and CD drops from 15.2 to 3.4. This success stems from combining the VAREN model, the spatio-temporal Transformer, and post-optimization.
- **The Temporal module remarkably reduces Accel** (yielding smoother motion), while Post-Optimization dramatically boosts PCK (ensuring tighter alignment)—the two components are highly complementary.
- **4DEquine surpasses GART on SSIM and LPIPS**: Although its PSNR is slightly lower than fully-optimized GART (15.66 vs 16.19), the perceptual metrics are superior, indicating more accurate geometry and structures with only a single frame of input.
- **Zero-shot generalization to zebras**: Outperforms all methods, including GART, on the zebra subset, proving that the model learns generalizable features rather than memorizing training textures.
- **Efficiency advantage**: Takes 11 seconds per frame (A100), whereas GART requires an intensive 15-minute optimization process.

## Highlights & Insights
- **Disentangled Motion-Appearance Philosophy**: Splitting 4D reconstruction into two independent sub-problems bypasses the difficulty of joint optimization. This paradigm is transferable to other 4D reconstruction tasks—as long as a parametric model acts as a bridge, motion and appearance can be trained independently. For instance, human 4D reconstruction could utilize SMPL similarly.
- **DSTG Dual-Stream Transformer Decoder**: Modifying the MMDiT block to fuse image features and 3D point features is more effective than standard cross-attention. The core lies in extracting a global context vector for modulation before dual-stream joint attention.
- **Synthetic Training & Real-world Generalization**: Even though VarenPoser and VarenTex are entirely synthetic, the model generalizes excellently to real data. Crucial factors include diverse camera trajectories (fixed/dolly/orbit), randomized shapes, and high-quality textures from multi-view diffusion.
- **Single-Image Canonical Avatar Generation**: EquineGS generates an animatable 360° Gaussian avatar from just a single image, bypassing the bottleneck of per-video optimization.

## Limitations & Future Work
- **Inadequate Tail and Mane Reconstruction**: The VAREN model has limited representation for tails and manes, which impacts the quality of appearance in these regions.
- **No Dynamic Lighting Handling**: Assumes static appearance across frames, failing to handle scenes with changing illumination.
- **Equidae Specificity Only**: Although it generalizes zero-shot to donkeys and zebras, it cannot process other animal categories (e.g., cats, dogs) without their corresponding parametric models.
- **Future Work**: (1) Introduce physical simulation representations for tails/manes; (2) add a relighting module to handle dynamic lighting; (3) integrate multi-keyframe information to enhance appearance completeness.

## Related Work & Insights
- **vs GART**: GART demands per-video optimization (15 mins) and complete visibility. 4DEquine is feed-forward (11s/frame), requires only a single-frame appearance input, and yields better SSIM/LPIPS.
- **vs 4D-Fauna**: Template-free methods lack geometric priors, leading to a PCK@0.05 of only 46.5 vs 4DEquine's 84.2. On the other hand, 4D-Fauna can process 100+ species.
- **vs AniMer**: 4DEquine's AniMoFormer builds directly upon AniMer by introducing the spatio-temporal Transformer, bringing PCK@0.05 from 44.5 $\rightarrow$ 61.8.

## Rating
- Novelty: ⭐⭐⭐⭐ The motion-appearance disentanglement bridged by VAREN is elegant, and the construction of the two synthetic datasets is highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across multiple datasets and baselines with comprehensive ablations and zero-shot generalization experiments.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured, well-illustrated, with intuitive workflow diagrams.
- Value: ⭐⭐⭐⭐ Driving SOTA performance in the niche domain of 4D equine reconstruction; the disentanglement strategy is highly generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SplineGS: Robust Motion-Adaptive Spline for Real-Time Dynamic 3D Gaussians from Monocular Video](splinegs_robust_motion-adaptive_spline_for_real-time_dynamic_3d_gaussians_from_m.md)
- [\[ICCV 2025\] Shape of Motion: 4D Reconstruction from a Single Video](../../ICCV2025/3d_vision/shape_of_motion_4d_reconstruction_from_a_single_video.md)
- [\[CVPR 2025\] FluidNexus: 3D Fluid Reconstruction and Prediction from a Single Video](fluidnexus_3d_fluid_reconstruction_and_prediction_from_a_single_video.md)
- [\[CVPR 2025\] Reconstructing Close Human Interaction with Appearance and Proxemics Reasoning](reconstructing_close_human_interaction_with_appearance_and_proxemics_reasoning.md)
- [\[CVPR 2025\] EnvGS: Modeling View-Dependent Appearance with Environment Gaussian](envgs_modeling_view-dependent_appearance_with_environment_gaussian.md)

</div>

<!-- RELATED:END -->
