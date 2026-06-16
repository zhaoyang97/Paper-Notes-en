---
title: >-
  [Paper Note] PFAvatar: Pose-Fusion 3D Personalized Avatar Reconstruction from Real-World Outfit-of-the-Day Photos
description: >-
  [AAAI 2026][3D Vision][3D avatar reconstruction] PFAvatar is proposed as a two-stage pipeline—comprising pose-aware diffusion model fine-tuning (ControlBooth) and NeRF distillation (BoothAvatar)—that reconstructs high-qu…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D avatar reconstruction"
  - "OOTD photos"
  - "diffusion models"
  - "NeRF"
  - "Score Distillation Sampling"
date: 2026-05-08
content_hash: a3b1cefc8e21cd8d
---

# PFAvatar: Pose-Fusion 3D Personalized Avatar Reconstruction from Real-World Outfit-of-the-Day Photos

**Conference**: AAAI 2026
**arXiv**: [2511.12935](https://arxiv.org/abs/2511.12935)  
**Code**: N/A  
**Area**: 3D Vision
**Keywords**: 3D avatar reconstruction, OOTD photos, diffusion models, NeRF, Score Distillation Sampling

## TL;DR

PFAvatar is proposed as a two-stage pipeline—comprising pose-aware diffusion model fine-tuning (ControlBooth) and NeRF distillation (BoothAvatar)—that reconstructs high-quality 3D personalized avatars from real-world Outfit-of-the-Day (OOTD) photos, completing personalization within 5 minutes and achieving a 48× speedup over prior methods.

## Background & Motivation

Converting everyday photos into personalized 3D human models is a novel and practically valuable task. OOTD photos exhibit several distinctive characteristics: (1) consistent identity, clothing, hairstyle, and accessories across images; (2) diverse poses and proportions; (3) frequent occlusions and severe truncations; and (4) varying viewpoints against complex backgrounds. These properties pose significant challenges to existing 3D reconstruction methods.

The representative prior work PuzzleAvatar adopts a "decompose-and-assemble" strategy: OOTD photos are segmented into multiple semantic assets (clothing, accessories, face, hairstyle), each associated with a Stable Diffusion token, and then assembled into a 3D avatar. However, this approach suffers from **three critical issues**:

**Segmentation inconsistency**: Fine-grained segmentation readily introduces visual inconsistencies (e.g., misaligned boundaries, mislabeled parts), leading to seams and artifacts in the assembled 3D model.

**Lack of pose-controllable generation**: Because individual components are learned separately, the method cannot generate complete human images at specified poses, causing the Janus problem during SDS optimization.

**Low training efficiency**: Learning multiple independent components significantly increases training time (approximately 4 hours), limiting practical applicability.

**Limitations of mesh representation**: DMTet's topology is constrained by the initial mesh structure, making it difficult to represent complex topological variations such as fine hair strands and clothing textures.

PFAvatar addresses these issues with a novel end-to-end framework that avoids decomposition and directly models full-body appearance.

## Method

### Overall Architecture

PFAvatar consists of two stages:
1. **ControlBooth**: Fine-tunes a pose-aware diffusion model $\mathcal{M}_b$ on a small set of OOTD photos.
2. **BoothAvatar**: Distills a NeRF-based 3D avatar from the fine-tuned $\mathcal{M}_b$.

### Key Designs

#### 1. **ControlBooth: Pose-Aware Diffusion Model**

**Function**: Trains a diffusion model capable of generating personalized human images conditioned on arbitrary poses.

**Data preprocessing pipeline**:
- Grounded-SAM is used to separate the foreground person from the background (isolating only the human region, avoiding inconsistencies from fine-grained part segmentation).
- A pretrained ControlNet is used to estimate the pose $\{\mathcal{P}_i\}$ for each image.
- GPT-4V generates detailed textual descriptions $\mathcal{T}_i$ covering body orientation, hairstyle, clothing, and other attributes.

**Training loss** consists of two components:

**Reconstruction diffusion loss**:

$$\mathcal{L}_{\text{rec}} = \mathbb{E}\left[\|\mathcal{D}_\theta(\alpha_t \mathcal{I}_i + \sigma_t \epsilon, \mathbf{c}_{t_i}, \mathbf{c}_{p_i}) - \mathcal{I}_i\|_2^2\right]$$

**Conditional Prior Preservation Loss (CPPL)**: This is one of the core contributions of the paper. During few-shot fine-tuning, the model tends to overfit to the training poses, losing its ability to generate diverse poses (as illustrated by the pose rigidity in the middle row of Figure 3). CPPL regularizes training by using a frozen pretrained model to generate prior data:

$$\mathcal{L}_{\text{cppl}} = \mathbb{E}\left[\lambda w'_t \|\mathcal{D}_\theta(\alpha_t \mathcal{I}_{pr_i} + \sigma_t \epsilon, \mathbf{c}_{prt_i}, \mathbf{c}_{prp_i}) - \mathcal{I}_{pr_i}\|_2^2\right]$$

**Design Motivation**: CPPL enables the model to learn a new identity while preserving its capacity to generate diverse poses and viewpoints, preventing both language drift and control drift. Personalization is completed in only 5 minutes—48× faster than PuzzleAvatar.

#### 2. **BoothAvatar: NeRF Representation and 3D-SDS Distillation**

**Function**: Distills a 3D NeRF avatar in the canonical A-pose from the fine-tuned diffusion model.

**Rationale for choosing NeRF over meshes**:
- The volumetric density of NeRF naturally handles occlusions via transmittance, avoiding the generation of spurious surfaces.
- Continuous volume rendering in NeRF, combined with high-frequency positional encodings such as hash grids, preserves fine-grained textures (e.g., hair strands, patterns).
- Mesh representations suffer from resolution-dependent discretization, performing poorly on high-frequency details.

Instant-NGP is adopted as the canonical avatar representation, optimized via 3D-consistent SDS:

$$\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{3D-SDS}} = \mathbb{E}\left[w(t)(\boldsymbol{\epsilon}_\phi(\mathbf{x}_t; y, t, c) - \boldsymbol{\epsilon})\frac{\partial \mathbf{z}_t}{\partial \mathbf{x}}\frac{\partial \mathbf{x}}{\partial \boldsymbol{\theta}}\right]$$

The conditioning image $c$ uses a skeleton map, providing minimal structural prior to facilitate complex avatar generation.

#### 3. **Local Geometry Loss**

**Function**: Addresses degradation of fine structures such as hands and faces caused by unstable SDS optimization.

Based on predefined body-part meshes, a margin ranking loss aligns NeRF density with part meshes:

$$\mathcal{L}_{\text{geo}} = \begin{cases} (\max(0, \tau_{\max} - \tau(\mathbf{p})))^2 & \text{if } \mathbf{p} \text{ on mesh} \\ (\max(0, \tau(\mathbf{p}) - \tau_{\min}))^2 & \text{if } \mathbf{p} \text{ not on mesh} \end{cases}$$

**Design Motivation**: SDS optimization inherently lacks human body priors and tends to produce blurry fingers and faces. Constraining local region density via predefined meshes preserves fine structures without restricting global optimization.

### Loss & Training

- **ControlBooth stage**: $\mathcal{L}^{\text{CB}}_{\text{total}} = \mathcal{L}_{\text{rec}} + \lambda_{\text{cppl}} \mathcal{L}_{\text{cppl}}$, with $\lambda_{\text{cppl}}=1$
- **BoothAvatar stage**: $\mathcal{L}^{\text{BA}}_{\text{total}} = \mathcal{L}_{\text{3D-SDS}} + \lambda_{\text{geo}} \mathcal{L}_{\text{geo}}$, with $\lambda_{\text{geo}}=1.0$
- **Multi-resolution progressive sampling**: The upsampling resolution is gradually increased for more stable SDS training.

The sampling strategy combines two spaces: (1) canonical SMPL-X space sampling to generate more pose-conditioned images ensuring 3D consistency; and (2) observation space sampling to obtain higher-quality appearance details.

## Key Experimental Results

### Main Results

#### Identity Preservation Comparison (ControlBooth Stage)

| Method | CLIP-I (body) | CLIP-I (head) | DINO (body) | DINO (head) | CLIP-T (body) | CLIP-T (head) |
|--------|-------------|-------------|-----------|-----------|-------------|-------------|
| **PFAvatar** | **0.9016** | **0.9432** | **0.7282** | **0.9352** | **0.3036** | **0.2996** |
| PuzzleAvatar | 0.8147 | 0.7705 | 0.6257 | 0.6096 | 0.2340 | 0.1849 |
| FreeCustom | 0.8573 | 0.9337 | 0.7022 | 0.9222 | 0.2583 | 0.2811 |
| InstantID | 0.7687 | 0.8164 | 0.5977 | 0.8302 | 0.2164 | 0.2711 |

#### Reconstruction Quality on PuzzleIOI Benchmark

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| **PFAvatar** | **27.576** | **0.952** | **0.041** |
| PuzzleAvatar | 24.687 | 0.930 | 0.062 |
| TECH | 23.635 | 0.919 | 0.065 |
| AvatarBooth | 16.431 | 0.758 | 0.153 |

PFAvatar achieves significant improvements over all baselines across every metric.

### Ablation Study

| Configuration | CLIP-I (body) | DINO (body) | CLIP-T (body) | Note |
|---------------|-------------|-----------|-------------|------|
| Full | 0.9125 | 0.8072 | 0.3546 | Complete model |
| w/o Head Part Data | 0.8702 | 0.7154 | 0.2912 | Noticeable facial degradation |
| w/o ControlBooth | 0.8352 | 0.7091 | 0.2314 | Consistency and color shift |
| w/o 3D-SDS | 0.8021 | 0.7281 | 0.2281 | A-pose generation impaired |
| w/o $\mathcal{L}_{\text{geo}}$ | 0.8929 | 0.8011 | 0.3257 | Blurry hand geometry |
| w/o Multi-sampling | 0.8654 | 0.7486 | 0.2812 | Slow convergence, poor detail |

Removing each component leads to visible degradation, validating the necessity of each design choice.

### Key Findings

1. **CPPL effectively prevents overfitting**: Figure 3 qualitatively demonstrates that without CPPL the model overfits to training poses, while CPPL enables diverse and controllable pose generation.
2. **NeRF outperforms mesh representation**: NeRF shows clear advantages over DMTet meshes in handling occlusions and preserving high-frequency textures.
3. **Rich downstream applications**: The reconstructed NeRF avatar supports virtual try-on, animation, facial animation, and human video reenactment.

## Highlights & Insights

1. **End-to-end design sidesteps the segmentation bottleneck**: By directly modeling full-body appearance, the method elegantly circumvents the segmentation inconsistency problem inherent to PuzzleAvatar.
2. **5-minute personalization**: A 48× speedup over PuzzleAvatar substantially increases practical utility.
3. **Regularization insight behind CPPL**: Using self-generated data from a pretrained model as regularization is an elegant solution to few-shot fine-tuning overfitting.
4. **Local geometry constraints**: Body-part priors are leveraged to stabilize local structures during SDS optimization.

## Limitations & Future Work

1. NeRF representations currently lack the mature toolchain available for traditional mesh-based methods (though animation capabilities are demonstrated in the paper).
2. The reliance on GPT-4V for text description generation introduces additional cost and dependency on an external API.
3. Robustness to extreme occlusions and severe truncations still has room for improvement, despite outperforming baselines.
4. The current method relies solely on skeleton conditioning; incorporating multi-condition control combining depth maps and skeletons could further improve quality.

## Related Work & Insights

- **Relationship to DreamBooth**: ControlBooth can be viewed as a significant extension of DreamBooth toward pose-aware generation; CPPL addresses the degradation problem that arises when applying DreamBooth to few-shot human fine-tuning.
- **NeRF vs. mesh representation**: The detailed comparative analysis of the two representations (occlusion handling, high-frequency detail, topological flexibility) provides useful reference for the broader 3D human reconstruction community.
- The conditional prior preservation idea underlying CPPL is generalizable to other few-shot fine-tuning scenarios beyond avatar generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — CPPL and the end-to-end pose-aware framework are novel; the specific NeRF distillation strategy introduces meaningful innovations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluation on two datasets, comparisons against multiple baselines, user study, and comprehensive ablation.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is well-articulated, comparisons are clear, and figures are informative.
- **Value**: ⭐⭐⭐⭐⭐ — 5-minute personalization with support for diverse downstream applications makes this highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning 3D Shape Fidelity Metric from Real-world Distortions](../../CVPR2026/3d_vision/learning_3d_shape_fidelity_metric_from_real-world_distortions.md)
- [\[AAAI 2026\] RTGaze: Real-Time 3D-Aware Gaze Redirection from a Single Image](rtgaze_real-time_3d-aware_gaze_redirection_from_a_single_image.md)
- [\[CVPR 2026\] Learning a Particle Dynamics Model with Real-world Videos](../../CVPR2026/3d_vision/learning_a_particle_dynamics_model_with_real-world_videos.md)
- [\[CVPR 2026\] Intrinsic Image Fusion for Multi-View 3D Material Reconstruction](../../CVPR2026/3d_vision/intrinsic_image_fusion_for_multi-view_3d_material_reconstruction.md)
- [\[CVPR 2026\] Changes in Real Time: Online Scene Change Detection with Multi-View Fusion](../../CVPR2026/3d_vision/changes_in_real_time_online_scene_change_detection_with_multi-view_fusion.md)

</div>

<!-- RELATED:END -->
