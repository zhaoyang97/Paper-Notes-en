---
title: >-
  [Paper Note] ShowMak3r++: Compositional Entertainment Video Reconstruction
description: >-
  [CVPR 2025][Human Understanding][Dynamic Radiance Fields] This paper proposes ShowMak3r++, a compositional pipeline to reconstruct dynamic radiance fields from TV shows and web videos. The core innovations include a depth-prior-based spatio-temporal positioning module, ShotMatcher for cross-shot actor association, and an implicit face-fitting network, supporting post-production editing applications such as actor repositioning, insertion, and deletion.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Dynamic Radiance Fields"
  - "3D Gaussian Splatting"
  - "TV Show Reconstruction"
  - "Human Positioning"
  - "Face Fitting"
date: 2026-05-08
content_hash: bbeef9856556e260
---

# ShowMak3r++: Compositional Entertainment Video Reconstruction

**Conference**: CVPR 2025  
**arXiv**: [2504.19584](https://arxiv.org/abs/2504.19584)  
**Code**: [https://nstar1125.github.io/showmak3r/](https://nstar1125.github.io/showmak3r/) (Project Page)  
**Area**: Human Understanding  
**Keywords**: Dynamic Radiance Fields, 3D Gaussian Splatting, TV Show Reconstruction, Human Positioning, Face Fitting

## TL;DR
This paper proposes ShowMak3r++, a compositional pipeline to reconstruct dynamic radiance fields from TV shows and web videos. The core innovations include a depth-prior-based spatio-temporal positioning module, ShotMatcher for cross-shot actor association, and an implicit face-fitting network, supporting post-production editing applications such as actor repositioning, insertion, and deletion.

## Background & Motivation

**Background**: Significant progress has been made in 4D scene reconstruction recently. NeRF and 3DGS can reconstruct dynamic scenes from synchronized multi-view or monocular videos. Data-driven methods (such as MonST3R, Shape of Motion) generate dynamic point clouds from monocular videos, but their output is non-parametric, rendering them incapable of photorealistic post-production editing.

**Limitations of Prior Work**: TV shows and web videos present unique challenges for reconstruction: (1) multiple actors occluding each other with rich facial expressions; (2) shot transitions leading to abrupt viewpoint jumps; (3) narrow camera baselines that only capture the frontal view of the scene, leaving back-view information missing. Existing 4D reconstruction methods fail in these scenarios due to inaccurate human-scene alignment and inconsistent body deformation.

**Key Challenge**: The predecessor ShowMak3r only applies to controlled environments (where extra background images can be acquired from other episodes), and its 3DLocator module relies solely on depth alignment to position actors, causing actors to clip into the stage or experience motion jitter. A unified pipeline is required that works for both controlled environments and uncontrolled web videos.

**Goal**: To build a comprehensive reconstruction pipeline capable of reconstructing dynamic radiance fields from TV shows and web videos, enabling scene post-production editing like a studio control room.

**Key Insight**: To decompose the problem into four sub-problems: stage reconstruction, actor positioning, cross-shot tracking, and expression recovery, utilizing a compositional 3D Gaussian representation to make each part independently editable.

**Core Idea**: To propose a spatio-temporal positioning module that simultaneously considers 2D image alignment and 3D motion naturalness to accurately place actors, and use ShotMatcher to solve actor association across shot transitions.

## Method

### Overall Architecture
The input consists of TV shows or web videos. The pipeline is divided into: (1) Preprocessing—estimating camera poses using GLOMAP/Pi3 and obtaining guided depth maps via monocular depth estimation; (2) Stage reconstruction—reconstructing the background $\mathcal{G}^{\text{stage}}$ using 3DGS with a depth-guided loss; (3) Spatio-temporal positioning—locating the SMPL body model at the correct position on the stage; (4) ShotMatcher—associating actor identities across shots; (5) Actor appearance reconstruction + face fitting—recovering dynamic facial expressions frame by frame. Finally, all 3D Gaussians are combined as $\mathcal{G}^{\text{composite}} = \mathcal{G}^{\text{stage}} \cup \{\mathcal{G}_n^{\text{actor}}\}$ to render new viewpoints.

### Key Designs

1. **Spatio-Temporal Positioning Module**:

    - **Function**: Accurately places the estimated SMPL body model onto the 3D stage while ensuring both 2D projection alignment and 3D motion naturalness.
    - **Mechanism**: Unlike the predecessor's 3DLocator which solely relies on depth alignment, the new module jointly optimizes three objectives: (1) 2D image alignment—matching the rendered SMPL with the human silhouette in the original image; (2) depth consistency—fitting the human body to the stage's depth map; (3) 3D trajectory smoothness—preventing inter-frame jittering and clipping issues via temporal regularization. It can also solve invisible poses caused by occlusion through interpolation.
    - **Design Motivation**: Relying solely on depth in 3DLocator easily leads to actors embedding into the stage or experiencing unnatural motion. Jointly optimizing the three constraints provides robust 3D positioning.

2. **ShotMatcher (Cross-Shot Actor Tracking)**:

    - **Function**: Associates the same actor across different shots during shot transitions.
    - **Mechanism**: Calculates pairwise distances (based on appearance features and spatial positions) between actors in adjacent shots at shot boundaries, and solves the optimal association using the Hungarian algorithm. It handles cases where some actors are unseen in specific shots by propagating tracking information across preceding and succeeding shots to maintain continuous tracking.
    - **Design Motivation**: Multi-shot editing in TV shows represents a continuous narrative flow, which requires maintaining actor identity consistency across discontinuous viewpoint jumps.

3. **Implicit Face-Fitting Network**:

    - **Function**: Dynamically recovers the frame-by-frame facial expression changes of the actors.
    - **Mechanism**: Rather than using pre-trained expression encoders (which require multi-view face images) or directly using SMPL-X expression parameters (which struggle to capture subtle expressions), an implicit deformation network is designed to learn the deformation field from the SMPL canonical space to each frame's expression. It is trained end-to-end via rendering loss without extra expression annotations.
    - **Design Motivation**: Actor expressions are key performance elements in TV shows. A simple yet effective implicit deformation scheme avoids dependency on multi-view data and expression encoders.

### Loss & Training
Stage reconstruction utilizes $\mathcal{L}_{\text{background}} = (1-\lambda)\mathcal{L}_{\text{color}} + \lambda\mathcal{L}_{\text{D-SSIM}} + \lambda_d\mathcal{L}_{\text{depth}} + \lambda_s\mathcal{L}_{\text{TV}}$, where the depth loss adopts a log-L1 formulation to improve convergence. SAM is used to segment actors to obtain masks, masking transient objects during training. For web video scenarios, Pi3 replaces GLOMAP for camera pose estimation, and sampling is used instead of additional background images.

## Key Experimental Results

### Main Results

| Method | Sitcoms3D | CMU Panoptic | Web Videos |
|------|-----------|-------------|---------|
| ShowMak3r++ | Best novel-view rendering quality | Successfully reconstructs multi-person dynamic scenes | Applicable to action movies / dance / movie clips |
| ShowMak3r (Prev.) | Controlled environments only | - | Not supported |
| MonST3R | Point cloud lacks photorealism | - | Point-cloud output only |
| Shape of Motion | Cannot handle shot transitions | - | Limited support |

### Ablation Study

| Configuration | Key Effect |
|------|---------|
| W/o Spatio-Temporal Positioning $\rightarrow$ Depth Alignment Only | Actor clipping, motion jitter |
| W/o ShotMatcher | Confused actor identities across shots |
| W/o Face Fitting | Stiff expressions, lack of detail |
| W/o Depth Guidance | Sparse and incomplete stage reconstruction |
| W/o Transient Object Removal | Floating artifacts in the background |

### Key Findings
- The spatio-temporal positioning module significantly improves the alignment quality between the actors and the stage, eliminating clipping and jitter.
- Depth guidance is critical for successful stage reconstruction in narrow-baseline scenarios.
- The method successfully scales to web video scenarios (requiring no extra background images) by leveraging Pi3 for camera poses.
- It supports diverse editing applications: composite shot production, actor repositioning/insertion/deletion, and pose manipulation.

## Highlights & Insights
- Explicitly modeling the "scene-shot-frame" hierarchy of TV shows into the reconstruction pipeline leverages domain knowledge to enhance system practicality.
- The compositional 3D Gaussian representation (stage + multiple independent actor Gaussian sets) naturally supports scene editing with a simple yet powerful design.
- The joint optimization strategy of spatio-temporal positioning with multiple constraints (2D alignment + depth consistency + 3D smoothness) can be transferred to other tasks requiring anchoring detections to 3D scenes.

## Limitations & Future Work
- It relies on the parametric SMPL model, making it inapplicable to non-human dynamic objects (e.g., pets, props).
- The accuracy of camera pose estimation bounds the upper limit of the entire pipeline, especially in scenarios with fast motion and severe blur.
- For heavily occluded scenes, pose interpolation in invisible regions may not be sufficiently accurate.
- Future work could combine diffusion priors to recover the appearance and geometry of invisible regions.

## Related Work & Insights
- **vs Sitcoms3D**: Only uses NeRF-W to reconstruct the background and optimises SMPL parameters, lacking human body textures and requiring the same actor across adjacent shots. ShowMak3r++ provides fully-textured actor reconstruction and does not depend on multiple shots.
- **vs OmniRe**: Relies on LiDAR sensors to reconstruct outdoor scenes, which is inapplicable to shot transition scenarios. ShowMak3r++ requires only monocular videos.
- **vs Feed-forward methods (MonST3R, etc.)**: Directly generate point clouds but lack photorealism. ShowMak3r++ provides high-quality renderable outputs via 3DGS.

## Rating
- Novelty: ⭐⭐⭐⭐ The designs of spatio-temporal positioning and ShotMatcher are practical and effective; scaling to web videos represents a significant engineering contribution.
- Experimental Thoroughness: ⭐⭐⭐ Primarily dominated by qualitative results, lacking quantitative comparison metrics.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed pipeline illustrations.
- Value: ⭐⭐⭐⭐ Promising practical application prospects for video post-production and virtual production.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] 3D Face Reconstruction From Radar Images](3d_face_reconstruction_from_radar_images.md)
- [\[CVPR 2025\] WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild](wilor_end-to-end_3d_hand_localization_and_reconstruction_in-the-wild.md)
- [\[CVPR 2025\] FRESA: Feedforward Reconstruction of Personalized Skinned Avatars from Few Images](fresa_feedforward_reconstruction_of_personalized_skinned_avatars_from_few_images.md)
- [\[CVPR 2025\] D3-Human: Dynamic Disentangled Digital Human from Monocular Video](d3-human_dynamic_disentangled_digital_human_from_monocular_video.md)
- [\[CVPR 2025\] Efficient Video Face Enhancement with Enhanced Spatial-Temporal Consistency](efficient_video_face_enhancement_with_enhanced_spatial-temporal_consistency.md)

</div>

<!-- RELATED:END -->
