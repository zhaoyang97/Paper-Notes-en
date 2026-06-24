---
title: >-
  [Paper Note] MV-DUSt3R(+): Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds
description: >-
  [CVPR 2025][3D Vision][Multi-View Reconstruction] MV-DUSt3R proposes a single-stage feed-forward network to jointly process an arbitrary number of pose-free input views via multi-view decoder blocks. It completely eliminates the global optimization required by DUSt3R, achieving scene reconstruction 48–78 times faster than DUSt3R while reducing the Chamfer Distance by 1.6–3.2 times. Furthermore, MV-DUSt3R+ introduces cross-reference-view attention blocks…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Multi-View Reconstruction"
  - "Pose-Free Reconstruction"
  - "Feed-Forward Networks"
  - "Gaussian Splatting"
  - "Novel View Synthesis"
date: 2026-05-08
content_hash: 4c4135c397326886
---

# MV-DUSt3R(+): Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds

**Conference**: CVPR 2025  
**arXiv**: [2412.06974](https://arxiv.org/abs/2412.06974)  
**Code**: [https://mv-dust3rp.github.io/](https://mv-dust3rp.github.io/) (Project Page)  
**Area**: 3D Vision  
**Keywords**: Multi-View Reconstruction, Pose-Free Reconstruction, Feed-Forward Networks, Gaussian Splatting, Novel View Synthesis

## TL;DR
MV-DUSt3R proposes a single-stage feed-forward network to jointly process an arbitrary number of pose-free input views via multi-view decoder blocks. It completely eliminates the global optimization required by DUSt3R, achieving scene reconstruction 48–78 times faster than DUSt3R while reducing the Chamfer Distance by 1.6–3.2 times. Furthermore, MV-DUSt3R+ introduces cross-reference-view attention blocks, further improving reconstruction quality in large-scale scenes.

## Background & Motivation

1. **Background**: Recent methods such as DUSt3R and MASt3R do not require camera calibration or pose estimation, directly inferring pixel-aligned 3D point maps from unordered RGB images. However, these methods only process a pair of views at a time, requiring a combinatorial number of pairwise inferences followed by a global optimization step when dealing with multiple views.

2. **Limitations of Prior Work**: (a) The combinatorial explosion of pairwise reconstruction results in extremely long inference times; (b) reconstruction results between different view pairs often conflict, and global optimization can only rotate pairwise predictions but cannot correct erroneous matches; (c) when scenes contain objects with similar appearances (e.g., multiple chairs, multiple windows), pairwise methods are prone to generating incorrect spatial relationships.

3. **Key Challenge**: Stereo cues from two views are inherently ambiguous, especially when viewpoint variations between different views are large in extensive scenes. As a post-processing step, global optimization cannot fundamentally correct errors in pairwise reconstruction.

4. **Goal**: (a) Process multiple views in a single forward pass to eliminate expensive global optimization; (b) maintain high-quality reconstruction under sparse multi-view setups in large-scale scenes; (c) make the method robust to the choice of the reference view.

5. **Key Insight**: Feed the tokens of all views together into a decoder for cross-view information fusion, and directly align the predicted point maps in the same reference camera coordinate system, thereby eliminating subsequent global alignment.

6. **Core Idea**: Use a single-stage feed-forward network to simultaneously process token interactions across all input views, replacing the two-stage paradigm of "pairwise inference + global optimization" in DUSt3R.

## Method

### Overall Architecture
Inputting $N$ pose-free RGB images, visual tokens are extracted via a weight-sharing ViT encoder and then fed into multi-view decoder blocks (categorized into reference and source views) for cross-view token fusion. Finally, regression heads predict the 3D point map and confidence map for each view in the reference view's coordinate system. All views are processed in a single forward pass without requiring subsequent optimization.

### Key Designs

1. **Multi-View Decoder Blocks**:

    - **Function**: Jointly exchange information among all views, rather than independently fusing between pairs of views.
    - **Mechanism**: Use two types of decoders (ref/src) that share the architecture but have different weights. Each block performs self-attention on the primary tokens, followed by cross-attention with the secondary tokens of all other views, and finally passes through an MLP. The formula is $F_d^v = \text{DecBlock}_d(F_{d-1}^v, \mathcal{F}_{d-1}^{-v})$, where $\mathcal{F}^{-v}$ contains the tokens of all views except view $v$. During training, a confidence-weighted point map regression loss forces the predictions to automatically align to the reference camera coordinate system.
    - **Design Motivation**: DUSt3R only performs cross-attention with tokens from two views at a time, whereas MV-DUSt3R utilizes tokens from all views as secondary tokens to obtain richer matching cues from multiple views. Since the architecture differs only minutely from DUSt3R (additional skip connections and conv nets), the parameter size is nearly identical, allowing direct initialization with DUSt3R pretrained weights.

2. **Cross-Reference-View Blocks (MV-DUSt3R+)**:

    - **Function**: Address the issue of insufficient stereo cues between a single reference view and distant views in large-scale scenes.
    - **Mechanism**: Select $M$ reference views, each corresponding to a path, and append a CrossRefViewBlock after each decoder block. For the same input view $v$, its intermediate representations $G_d^{v,m}$ across different reference view paths are fused via cross-reference-view attention: $F_d^{v,m} = \text{CrossRefViewBlock}_d(G_d^{v,m}, \mathcal{G}_d^{v,-m})$. During training, $M$ reference views are randomly selected, and the loss is averaged over all paths. During inference, $M$ reference views are uniformly selected, and the final point map is output from the head of the first path.
    - **Design Motivation**: Different reference views provide varying reconstruction qualities for different input views—combinations with smaller viewpoint changes yield better quality, while larger ones yield poorer quality. Through information fusion across multiple reference view paths, each view can acquire information from the most favorable reference view, improving overall reconstruction quality.

3. **Gaussian Splatting Heads**:

    - **Function**: Extend the model to support novel view synthesis.
    - **Mechanism**: Add lightweight prediction heads on top of the existing point map head to regress per-pixel Gaussian properties (scale $S^{v,m}$, rotation quaternion $q^{v,m}$, opacity $\alpha^{v,m}$), treating the predicted point map as the Gaussian centers and pixel colors as the Gaussian colors. During training, differentiable splatting rendering is employed, and the L2 pixel loss + LPIPS perceptual similarity loss are combined as the rendering loss $\mathcal{L}_{\text{render}}$, which is jointly trained with the point map regression loss $\mathcal{L}_{\text{conf}}$.
    - **Design Motivation**: While the original DUSt3R only has geometric reconstruction capabilities, jointly training Gaussian heads kills two birds with one stone—more accurate point maps directly translate to better Gaussian position predictions, achieving high-quality novel view synthesis.

### Loss & Training
- Confidence-weighted point map regression loss $\mathcal{L}_{\text{conf}} = \sum_v \sum_p C_p^{v,r} \ell_{\text{regr}}(v,p) - \beta \log C_p^{v,r}$, where the regression error is normalized for both prediction and ground truth to resolve scale ambiguity.
- Additional rendering loss $\mathcal{L}_{\text{render}} = \text{L2} + \text{LPIPS}$ is added for novel view synthesis.
- Trained for 100 epochs using 64 H100 GPUs, with 150k trajectories per epoch, totaling 180 hours.
- Input resolution is $224 \times 224$; trained with 8-view inputs, and MV-DUSt3R+ uses $M = 4$ reference views.

## Key Experimental Results

### Main Results

| Dataset | Views | Method | CD↓ | ND↓ | DAc↑ | Inference Time |
|--------|--------|------|-----|-----|------|----------|
| HM3D | 4 | DUSt3R+GO | 5.6 | 1.9 | 75.1% | 2.42s |
| HM3D | 4 | MV-DUSt3R | 2.0 | 1.1 | 92.2% | **0.05s** |
| HM3D | 4 | MV-DUSt3R+ | **1.5** | **1.0** | **95.2%** | 0.29s |
| HM3D | 24 | DUSt3R+GO | 32.4 | 6.8 | 7.3% | 27.21s |
| HM3D | 24 | MV-DUSt3R | 10.0 | 3.4 | 36.7% | 0.35s |
| HM3D | 24 | MV-DUSt3R+ | **3.9** | **2.1** | **64.5%** | 1.97s |
| MP3D (zero-shot) | 24 | DUSt3R+GO | 80.9 | 11.4 | 2.5% | 27.21s |
| MP3D (zero-shot) | 24 | MV-DUSt3R+ | **22.0** | **4.3** | **26.7%** | 1.97s |

### Ablation Study (Pose Estimation mAE@30)

| Configuration | HM3D 4v | HM3D 12v | HM3D 24v |
|------|---------|----------|----------|
| DUSt3R+GO | 12.5 | 20.1 | 30.9 |
| MV-DUSt3R | 5.5 | 8.4 | 23.7 |
| MV-DUSt3R+ | **4.9** | **5.2** | **15.8** |
| MV-DUSt3R oracle | 2.8 | 4.9 | 14.7 |
| MV-DUSt3R+ oracle | 2.4 | 3.4 | 11.1 |

### Key Findings
- MV-DUSt3R is **48 times** faster than DUSt3R and achieves a **2.8 times** lower Chamfer Distance on 4-view small-scale scenes; it is **78 times** faster with a **3.2 times** lower CD on 24-view large-scale scenes.
- Compared to MV-DUSt3R, MV-DUSt3R+ further reduces the CD by **2.6 times** on 24-view large-scale scenes, indicating that the cross-reference-view mechanism is crucial for extensive scenes.
- As the number of views increases from 4 to 24, the performance of DUSt3R degrades (due to more pairwise conflicts), whereas MV-DUSt3R(+) consistently improves.
- Spann3R fails completely on sparse views because its design targets continuous video frames rather than sparse sampling.

## Highlights & Insights
- **Architecture borrowing DUSt3R weights**: The network can be directly initialized with DUSt3R pretrained weights as they share nearly identical parameter size, leading to highly efficient training. This paradigm of making minimal architectural modifications to yield maximum performance gains is highly elegant.
- **Eliminating global optimization**: By training the model to directly predict computationally consistent global point maps in the reference coordinate system, it resolves the uncorrectability of errors inherent in two-stage methods at the architectural level.
- **Multipath and cross-reference-view fusion**: This design can be transferred to any multi-view task that requires selecting an "anchor/reference frame", such as keyframe selection in video understanding.

## Limitations & Future Work
- The input resolution is limited to $224 \times 224$, which restricts the level of detail in reconstruction.
- Training requires 64 H100 GPUs for 180 hours, incurring a substantial computational overhead.
- For 24 views, inference of MV-DUSt3R+ still takes around 2 seconds, and its memory footprint grows quadratically with the number of views due to cross-attention.
- Image priors from diffusion models are not utilized to handle unobserved regions.
- The strategy for selecting reference views (uniform sampling) may not be optimal.

## Related Work & Insights
- **vs DUSt3R**: DUSt3R processes 2 views at a time followed by a global optimization alignment. This work processes all views simultaneously without any optimization, boosting speed by 1–2 orders of magnitude.
- **vs Spann3R**: Spann3R uses spatial memory for online incremental reconstruction but is prone to drifting on sparse views and large-scale scenes; MV-DUSt3R+ processes all views offline to prevent accumulative drift.
- **vs NoPoSplat**: NoPoSplat also predicts Gaussians in the reference coordinate system but is limited to 2 views, whereas this work scales up to multi-view scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The core idea (multi-view token fusion replacing pairwise inferencing with global optimization) is natural and effective, though not strictly revolutionary.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ It covers three datasets, three tasks (MVS/MVPE/NVS), and 4–24 views comprehensively, with a thorough comparison against oracle settings.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, with excellent coordination between equations and illustrations.
- Value: ⭐⭐⭐⭐⭐ Highly practical, achieving large-scale scene reconstruction within 2 seconds, which makes it directly applicable to MR, robotics, and autonomous driving scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] HandOS: 3D Hand Reconstruction in One Stage](handos_3d_hand_reconstruction_in_one_stage.md)
- [\[CVPR 2025\] WonderWorld: Interactive 3D Scene Generation from a Single Image](wonderworld_interactive_3d_scene_generation_from_a_single_image.md)
- [\[ICML 2025\] PhysicsNeRF: Physics-Guided 3D Reconstruction from Sparse Views](../../ICML2025/3d_vision/physicsnerf_physics-guided_3d_reconstruction_from_sparse_views.md)
- [\[CVPR 2025\] MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photorealism From Sparse Views](matcha_gaussians_atlas_of_charts_for_high-quality_geometry_and_photorealism_from.md)
- [\[CVPR 2025\] FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views](flare_sparse_view_reconstruction.md)

</div>

<!-- RELATED:END -->
