---
title: >-
  [Paper Note] Motion Matters: Compact Gaussian Streaming for Free-Viewpoint Video Reconstruction
description: >-
  [NeurIPS 2025][3D Vision][3D Gaussian Splatting] This paper proposes ComGS, a framework that exploits the locality and consistency of motion in dynamic scenes to drive the motion of all Gaussians in moving regions using…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Free-Viewpoint Video"
  - "Online Reconstruction"
  - "Motion Modeling"
  - "Streaming"
date: 2026-05-08
content_hash: e8396d0fd2e89d12
---

# Motion Matters: Compact Gaussian Streaming for Free-Viewpoint Video Reconstruction

**Conference**: NeurIPS 2025
**arXiv**: [2505.16533](https://arxiv.org/abs/2505.16533)  
**Code**: [Project Page](https://chenjiacong-1005.github.io/ComGS/)  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Free-Viewpoint Video, Online Reconstruction, Motion Modeling, Streaming

## TL;DR
This paper proposes ComGS, a framework that exploits the locality and consistency of motion in dynamic scenes to drive the motion of all Gaussians in moving regions using only ~200 keypoints. ComGS achieves 159× storage compression over 3DGStream and 14× over QUEEN while maintaining competitive visual quality and rendering speed.

## Background & Motivation
Free-viewpoint video (FVV) reconstruction is an important research direction in computer vision and graphics, enabling immersive and interactive viewing experiences with broad applications in VR/AR. 3DGS has become the dominant paradigm for FVV reconstruction due to its high fidelity and real-time rendering capabilities.

Existing online FVV methods (e.g., 3DGStream, QUEEN) face a core contradiction: **prohibitively high storage requirements** that impede real-time transmission. These methods typically produce reconstruction data exceeding 20 MB/s, as they adopt a per-point modeling strategy that independently updates attribute residuals for each Gaussian in the motion region, overlooking two key observations: (1) the majority of dynamic scenes are static and require no updates; and (2) Gaussians belonging to the same object typically share the same or similar motion, resulting in substantial motion redundancy.

The core idea of this paper is grounded in two insights: **motion locality**—only the attribute residuals of Gaussians in moving regions need to be modeled; and **motion consistency**—a shared motion representation can be used to model attribute residuals for Gaussians with similar motion. By using only ~200 keypoints (far fewer than ~200K Gaussians) to holistically drive moving regions, ComGS eliminates motion redundancy at its root and achieves extreme storage compression.

## Method

### Overall Architecture
ComGS adopts an online frame-by-frame streaming reconstruction framework. The first frame is reconstructed independently using standard 3DGS, and subsequent frames are organized into Groups of Frames (GoF). Non-keyframes are reconstructed via a keypoint-driven motion representation, while keyframes employ an error-aware correction strategy to eliminate accumulated errors. The entire pipeline requires transmitting only keypoint attributes, enabling efficient storage.

### Key Designs
1. **Motion-Sensitive Keypoint Selection**:

    - **Function**: Precisely identify a small number of keypoints from moving regions while avoiding redundant modeling of static areas.
    - **Mechanism**: A view-space gradient difference strategy is employed. The gradient of the rendering loss from the previous frame is computed under both the current and previous frame images; points with large gradient changes are identified as dynamically salient. The top-$k$ Gaussians with the highest dynamic saliency scores are selected as keypoints $\mathcal{K}_t$.
    - Key formula: dynamic saliency score $\Delta\mathcal{G}_t = \frac{1}{V}\sum_{v=1}^{V}|\mathcal{G}_t^{(v)} - \mathcal{G}_{t-1}^{(v)}|$
    - **Design Motivation**: Selecting the top-$k$ not only ensures keypoints reside in motion regions but also naturally allocates more keypoints to regions with complex motion. The paper sets $k=200$ to balance training efficiency and reconstruction quality.

2. **Adaptive Motion-Driven Mechanism**:

    - **Function**: Determine which neighboring Gaussians each keypoint controls and propagate motion accordingly.
    - **Mechanism**: Each keypoint is initialized with a spatial influence field $\Sigma_{adap}^i$ (defined by a learnable quaternion $q_{adap}^i$ and scaling vector $s_{adap}^i$). The influence weight from neighboring Gaussians to the keypoint is computed as $w_{ij} = \exp(-\frac{1}{2}d_{ij}^\top(\Sigma_{adap}^i)^{-1}d_{ij})$. When $w_{ij} \geq \tau_{adap}$, the Gaussian is controlled by that keypoint. Each keypoint carries a learnable translation offset $\Delta\mu$ and rotation quaternion $\Delta q$; Gaussians controlled by multiple keypoints obtain their final motion via weighted aggregation.
    - **Design Motivation**: Compared to fixed-scale KNN approaches, the spatial influence field adapts to the complexity and variability of motion structures in dynamic scenes. Each keypoint requires storing only 14 parameters.

3. **Error-Aware Corrector**:

    - **Function**: Mitigate error accumulation arising from the rigid-motion assumption of keypoint-driven representations.
    - **Mechanism**: A keyframe is designated every $s$ frames for correction. A learnable attribute residual $\Delta\theta_i^t$ and mask $m_i$ are introduced per Gaussian. A hard mask is obtained via sigmoid mapping and straight-through estimator (STE) binarization: $m_i^{hard} = sg(\mathbb{1}(m_i^{soft} > \phi_{thres}) - m_i^{soft}) + m_i^{soft}$. Only Gaussians with a hard mask of 1 participate in attribute updates: $\theta_i^t = \theta_i^{t-1} + m_i^{hard}\Delta\theta_i^t$.
    - **Design Motivation**: Avoids unnecessary storage overhead from updating all Gaussians. A sparsity regularizer $\mathcal{L}_{error} = \frac{1}{N}\sum_i m_i^{soft}$ encourages updates only in regions with genuine errors.

### Loss & Training
- The first frame and non-keyframes use a reconstruction loss: $\mathcal{L}_{recon} = (1-\lambda_{D-SSIM})\mathcal{L}_1 + \lambda_{D-SSIM}\mathcal{L}_{D-SSIM}$, where $\lambda_{D-SSIM}=0.2$.
- Keyframe optimization jointly applies reconstruction loss and error-aware loss: $\mathcal{L}_{total} = \mathcal{L}_{recon} + \lambda_{error}\mathcal{L}_{error}$, where $\lambda_{error}=0.001$.
- After optimization, quantization and entropy coding are applied to the initialized Gaussians and keyframe residuals for further compression.

## Key Experimental Results

### Main Results

| Dataset | Metric | ComGS-s | ComGS-l | QUEEN-s | QUEEN-l | 3DGStream |
|--------|------|---------|---------|---------|---------|-----------|
| N3DV | PSNR (dB) | 31.87 | 32.12 | 31.89 | 32.19 | 31.67 |
| N3DV | SSIM | 0.943 | 0.945 | 0.945 | 0.946 | 0.941 |
| N3DV | Storage (MB) | **0.049** | **0.106** | 0.68 | 0.75 | 7.80 |
| MeetRoom | PSNR (dB) | **31.49** | - | 31.14 | - | 30.79 |
| MeetRoom | Storage (MB) | **0.028** | - | 0.45 | - | 4.1 |

### Ablation Study

| Configuration | PSNR (dB) | Storage (KB) | Notes |
|------|---------|-------------|------|
| Random keypoint selection | 33.27 | 46.7 | Quality degrades without motion-sensitive selection |
| No adaptive driving | 32.82 | 36.4 | Keypoints only, no propagation to neighbors |
| No keypoint motion | 31.26 | 37.9 | Relies solely on keyframe correction; significant degradation |
| No error-aware corrector | 31.67 | 26.9 | Error accumulates without keyframe correction |
| Full ComGS | **33.49** | 46.5 | All modules cooperating optimally |

### Key Findings
- ComGS-s reduces storage by 159× over 3DGStream and 14× over QUEEN, enabling real-time transmission.
- Only 200 keypoints are sufficient to effectively drive the motion of ~200K Gaussians.
- The adaptive spatial influence field (PSNR 31.87) substantially outperforms the KNN control strategy (PSNR 31.39).
- Full per-Gaussian correction without error awareness requires 373 KB storage, whereas the error-aware corrector requires only 49 KB.
- ComGS achieves a 0.7 dB PSNR gain over 3DGStream on MeetRoom with 146× smaller storage.
- On the long video Flame Salmon (1200 frames), storage is only 0.053 MB, competitive with the offline method TGH (0.075 MB).

## Highlights & Insights
- Framing "motion redundancy elimination" as the entry point for online FVV compression is an intuitive and effective design philosophy. The 1000:1 ratio between keypoints (200) and Gaussians (200K) yields extremely efficient compression.
- The spatial influence field design elegantly reuses the Gaussian function form to define the control range, which is conceptually and implementationally consistent with the core representation of 3DGS.
- The error-aware correction strategy implements sparse updates via learnable masks with STE binarization, cleverly balancing gradient propagation and binary selection.
- Long-video experiments (1200-frame Flame Salmon) demonstrate the temporal robustness of the framework, achieving 29.56 dB PSNR with only 0.053 MB storage.
- Each keypoint requires only 14 parameters (3 translation + 4 rotation + 3 scale + 4 rotation for the influence field), making the data volume minimal and naturally suited for real-time streaming.
- Visual quality comparisons show that ComGS effectively reconstructs both motion and static regions, avoiding the static-region artifacts caused by global updates in 3DGStream.

## Limitations & Future Work
- The framework relies on high-quality initialization of the first frame; poor first-frame quality leads to error propagation in subsequent frames.
- Dense multi-view video input is required, making direct application to sparse-view or monocular settings difficult.
- The training and encoding efficiency (37–43 seconds/frame) still lags significantly behind QUEEN (4.65–7.9 seconds/frame).

## Related Work & Insights
- **vs. 3DGStream**: 3DGStream encodes per-frame transformations with a hash MLP, resulting in high storage demand (7.8 MB/frame); ComGS reduces this to 0.049 MB via keypoint-shared motion, achieving a 159× compression ratio.
- **vs. QUEEN**: QUEEN performs per-point residual optimization followed by quantization-sparsity compression as post-processing; ComGS eliminates redundancy at the modeling level, representing a structural compression that is more fundamental. ComGS-s is 14× smaller than QUEEN-s.
- **vs. SC-GS/SP-GS**: These offline methods use KNN to select control points, which is motion-agnostic and scale-invariant; ComGS incorporates motion sensitivity and adaptive optimization for online scenarios.
- **vs. HiCoM**: HiCoM accelerates training with a hierarchical motion mechanism but offers limited storage compression; ComGS focuses on extreme storage efficiency.
- **vs. V3**: V3 compresses Gaussian attributes into 2D video leveraging hardware codecs, an orthogonal compression approach that could theoretically be combined with ComGS.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Keypoint-driven motion has precedents in offline methods, but the exploitation of motion locality and consistency for online streaming scenarios is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two datasets, long-video testing, and detailed ablations are provided, though the variety of datasets is somewhat limited.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear, the method is well-organized, and figures and tables are of high quality.
- **Value**: ⭐⭐⭐⭐⭐ A 159× compression ratio carries strong practical value and significant implications for real-time online FVV transmission.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video](../../AAAI2026/3d_vision/streamstgs_streaming_spatial_and_temporal_gaussian_grids_for_real-time_free-view.md)
- [\[NeurIPS 2025\] OnlineSplatter: Pose-Free Online 3D Reconstruction for Free-Moving Objects](onlinesplatter_pose-free_online_3d_reconstruction_for_free-moving_objects.md)
- [\[NeurIPS 2025\] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [\[ICCV 2025\] Shape of Motion: 4D Reconstruction from a Single Video](../../ICCV2025/3d_vision/shape_of_motion_4d_reconstruction_from_a_single_video.md)
- [\[NeurIPS 2025\] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
