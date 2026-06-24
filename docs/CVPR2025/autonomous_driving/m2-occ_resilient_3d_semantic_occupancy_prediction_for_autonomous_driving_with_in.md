---
title: >-
  [Paper Note] M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs
description: >-
  [CVPR 2025][Autonomous Driving][Semantic occupancy prediction] M²-Occ addresses the problem of semantic occupancy prediction under incomplete multi-camera inputs. It proposes a Multi-view Masked Reconstruction (MMR) module to recover missing view features using overlapping regions of adjacent cameras, and a Feature Memory Module (FMM) to refine uncertain voxel features through class-level semantic prototypes, improving IoU by 4.93% under the missing rear-view setup. The signi…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Semantic occupancy prediction"
  - "missing view robustness"
  - "masked reconstruction"
  - "semantic prototype memory"
  - "sensor failure"
  - "surround-view camera"
date: 2026-05-08
content_hash: ea0809b7bce09e57
---

# M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs

**Conference**: CVPR 2025  
**arXiv**: [2603.09737](https://arxiv.org/abs/2603.09737)  
**Code**: [GitHub](https://github.com/qixi7up/M2-Occ)  
**Area**: Autonomous Driving / 3D Occupancy Prediction  
**Keywords**: Semantic occupancy prediction, missing view robustness, masked reconstruction, semantic prototype memory, sensor failure, surround-view camera

## TL;DR
M²-Occ addresses the problem of semantic occupancy prediction under incomplete multi-camera inputs. It proposes a Multi-view Masked Reconstruction (MMR) module to recover missing view features using overlapping regions of adjacent cameras, and a Feature Memory Module (FMM) to refine uncertain voxel features through class-level semantic prototypes, improving IoU by 4.93% under the missing rear-view setup. The significance of this work lies not only in the method itself but also in providing the first systematic study of occupancy prediction behavior under sensor failures.

## Background & Motivation
**Background**: 3D semantic occupancy prediction (e.g., SurroundOcc, TPVFormer) provides dense voxel-level geometric and semantic understanding for autonomous driving, with multi-camera solutions attracting attention due to cost advantages.

**Limitations of Prior Work**: Existing methods implicitly assume all surround-view cameras are fully functional—however, camera occlusion, hardware failure, and communication interruption occur frequently in real-world deployment. Supposing even a single key view is lost (e.g., rear view), the IoU of SurroundOcc drops sharply from 32.38% to 23.94% (-26%).

**Key Challenge**: Fields of view overlap between surround-view cameras, but existing methods fail to utilize this redundancy; missing views not only result in geometric blanks but also cause inconsistent cross-view correspondences, degrading feature fusion quality.

**Goal**: How to make occupancy prediction models robust to camera failures without sacrificing full-view performance?

**Key Insight**: Drawing inspiration from the human capability to infer invisible regions from context and memory—(a) utilizing adjacent camera overlapping regions to recover missing features (spatial redundancy); (b) using global semantic prototypes as prior knowledge to refine uncertain features (memory).

**Core Idea**: Adjacent overlapping region feature reconstruction + class-level semantic prototype memory = sensor failure-robust occupancy prediction.

## Method

### Overall Architecture
Two modules are added to the standard 2D→3D pipeline: (1) MMR utilizes adjacent unmasked view overlap during the feature extraction stage to recover missing view features; (2) FMM enhances uncertain voxels using global semantic prototypes during the voxel refinement stage. The entire framework is trained end-to-end.

### Key Designs

1. **Multi-view Masked Reconstruction (MMR)**

    - **Function**: Utilizes overlapping boundary features of adjacent unmasked views to reconstruct the feature representation of missing views.
    - **Mechanism**: Models surround-view cameras as a cyclic graph $\mathcal{N}(v_i) = \{v_{(i-1) \bmod N}, v_{(i+1) \bmod N}\}$ $\rightarrow$ crops overlapping boundary regions $w_{ov}$ of left and right neighbors $\rightarrow$ concatenates with a learnable mask token $\rightarrow$ reconstructs via a lightweight Transformer decoder.
    - **Design Motivation**: Operating in the feature space (rather than pixel space) avoids the uncertainty of generative hallucinations; overlapping regions provide structural priors.

2. **Feature Memory Module (FMM) - Single-Proto**

    - **Function**: Maintains one global centroid prototype per class to provide semantic stability.
    - **Mechanism**: Updates via momentum moving average $\mathbf{m}_k^{(t)} = (1-\lambda)\mathbf{m}_k^{(t-1)} + \lambda \cdot \bar{\mathbf{f}}_k$, where $\lambda=0.1$.
    - **Design Motivation**: The Single-Proto captures core category features and provides a stable prior under incomplete observations—ensuring that the semantic representation of a "car" remains intact even if its visual features are partially damaged.

3. **Feature Memory Module (FMM) - Multi-Proto**

    - **Function**: Maintains $N_p$ sub-prototypes per class to model intra-class variation.
    - **Mechanism**: Computes cosine similarity between query voxel features and all sub-prototypes $\rightarrow$ takes weighted average via softmax + temperature parameter $\tau$ $\rightarrow$ dynamically retrieves the most relevant sub-prototype for feature refinement.
    - **Design Motivation**: Multi-Proto captures intra-class diversity (e.g., "truck" includes pickups and semi-trucks), achieving finer-grained semantic refinement.

### Loss & Training
$\mathcal{L} = \mathcal{L}_{occ} + \mathcal{L}_{MMR}$.
$\mathcal{L}_{MMR} = \frac{1}{|\mathcal{M}|}\sum_{i\in\mathcal{M}} \|\hat{\mathbf{f}}_i - \mathbf{f}_i^{gt}\|_2^2$, where the MSE reconstruction loss is calculated only on masked views.
During training, views are randomly masked to simulate failures, ensuring the model learns various missing view patterns.
For FMM, the momentum coefficient is set to $\lambda=0.1$ to ensure smooth prototype updates and filter noise.
In Multi-Proto, the temperature parameter $\tau$ controls the sharpness of the retrieval distribution, with a lower temperature emphasizing the most relevant prototypes.
Using ResNet-101 + FPN as the encoder on the SurroundOcc baseline, maintaining the same settings as the original implementation.

## Key Experimental Results

### Main Results (nuScenes/SurroundOcc benchmark)

| Missing Setup | SurroundOcc IoU | M²-Occ IoU | Gain |
|---------|----------------|-----------|------|
| Standard (No Missing) | 32.38 | 31.08~31.25 | ~Comparable |
| Missing Front | 25.03 | 30.40 | **+5.37** |
| Missing Rear | 23.94 | 28.87 | **+4.93** |
| Missing Front Right | 30.56 | 31.17 | +0.61 |

### Extreme Missing Scenarios

| Number of Missing Views | Baseline IoU | M²-Occ IoU | Description |
|-----------|-------------|-----------|------|
| 5 Missing | 13.35 | 18.36 | **+5.01** IoU |

### Key Findings
- Missing front and rear views have the greatest impact (IoU drops by 7.35 and 8.44 respectively) because they cover critical areas in front of and behind the ego-vehicle and have the least overlap with adjacent cameras.
- M²-Occ improves IoU by 4.93% under the missing rear-view setup without degrading full-view performance—robustness is achieved for free.
- Missing side views have less impact (IoU only drops by 1.6~2.0) due to larger overlap with adjacent views.
- Under extreme scenarios with 5 missing views, M²-Occ still maintains 18.36% IoU (vs baseline 13.35%), demonstrating FMM's capability to maintain basic structures using prior knowledge when observations are almost non-existent.

## Highlights & Insights
- **First systematic study of sensor failure robustness in multi-camera occupancy prediction**: Not only is a method proposed, but more importantly, a comprehensive evaluation protocol for missing views is established (deterministic failures of single views + random drops of multiple views).
- **Feature-space repair in MMR**: Operating in the feature space instead of the pixel space—more efficient and independent of generative models, utilizing a cyclic graph to model the natural redundancy of surround-view configurations.
- **Semantic prototypes as a safety fallback**: FMM acts similarly to a "prior knowledge base"—even if all visual evidence is lost, the model can still provide a minimal level of semantic understanding using class-level prototypes.
- **No degradation under full views for all improvements**: This demonstrates that MMR and FMM do not trade normal performance to achieve robustness.

## Limitations & Future Work
- Validated based on the SurroundOcc framework; generalizability has not been tested on more advanced occupancy prediction models (e.g., OccFormer, VoxFormer).
- MMR relies on sufficient overlap between adjacent cameras—the effect may be limited if the camera configuration has very little overlap.
- FMM prototypes may lack relevant representations for scenarios outside the training set (e.g., rare objects).
- Temporal information is not considered (e.g., utilizing results from the previous frame to compensate for the current missing frame), which is a promising direction for improvement.

## Related Work & Insights
- **vs M-BEV**: M-BEV uses full-view masking + cross-view reconstruction but for BEV perception (20); M²-Occ extends this idea to 3D occupancy prediction and incorporates semantic prototypes.
- **vs MetaBEV**: MetaBEV handles sensor corruption and missing modalities but targets joint BEV tasks; M²-Occ targets denser 3D voxel semantics.
- **vs SafeMap/FlexMap**: These methods enhance the robustness of BEV map construction; M²-Occ focuses on 3D voxel-level semantics.
- **vs MAE Paradigm**: MAE randomly masks patches in a single image for self-supervised learning; MMR masks entire views and reconstructs using cross-view context, naturally extending the MAE concept to multi-sensor scenarios.
- **vs Multi-sensor Fusion Methods** (e.g., OccFusion, FusionOcc): These methods provide redundancy by fusing LiDAR + camera, but increase hardware costs; M²-Occ achieves failure robustness using cameras alone.
- **Inspiration for Autonomous Driving Safety Certification**: The missing-view evaluation protocol of M²-Occ can serve as a standardized test framework for certifying the robustness of perception systems.
- **Temporal Extension Direction**: Leveraging occupancy prediction results from the previous time step to compensate for the missing view in the current frame is a natural and promising direction for improvement.

## Rating
- Novelty: ⭐⭐⭐⭐ The combined design of MMR+FMM is highly targeted, and the missing-view evaluation protocol is pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation covering single-view, multi-view, and extreme missing scenarios.
- Writing Quality: ⭐⭐⭐⭐ Clear paper structure, well-defined motivation, and detailed tables.
- Value: ⭐⭐⭐⭐⭐ Solves critical safety issues in autonomous driving deployment—perception robustness during sensor failures.
- Overall: ⭐⭐⭐⭐ Contributions in problem definition and evaluation protocols are as important as the method itself.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SDGOcc: Semantic and Depth-Guided BEV Transformation for 3D Multimodal Occupancy Prediction](sdgocc_semantic_and_depth-guided_birds-eye_view_transformation_for_3d_multimodal.md)
- [\[CVPR 2025\] OccMamba: Semantic Occupancy Prediction with State Space Models](occmamba_semantic_occupancy_prediction_with_state_space_models.md)
- [\[CVPR 2025\] GDFusion: Rethinking Temporal Fusion with a Unified Gradient Descent View for 3D Semantic Occupancy Prediction](gdfusion_temporal_fusion_occupancy.md)
- [\[ICCV 2025\] SA-Occ: Satellite-Assisted 3D Occupancy Prediction in Real World](../../ICCV2025/autonomous_driving/sa-occ_satellite-assisted_3d_occupancy_prediction_in_real_world.md)
- [\[CVPR 2026\] TT-Occ: Test-Time 3D Occupancy Prediction](../../CVPR2026/autonomous_driving/test-time_3d_occupancy_prediction.md)

</div>

<!-- RELATED:END -->
