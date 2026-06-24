---
title: >-
  [Paper Note] PFAvatar: Pose-Fusion 3D Personalized Avatar Reconstruction from Real-World Outfit-of-the-Day Photos
description: >-
  [AAAI 2026][3D Vision][3D Avatar Reconstruction] Proposes PFAvatar, a two-stage approach (pose-aware diffusion model fine-tuning + NeRF distillation) to reconstruct high-quality 3D human avatars from real-world "Outfit-of-the-Day" (OOTD) photos, achieving personalization in just 5 minutes, representing a 48x speedup compared to prior methods.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D Avatar Reconstruction"
  - "OOTD Photos"
  - "Diffusion Models"
  - "NeRF"
  - "Score Distillation Sampling"
date: 2026-05-08
content_hash: 8e604b7d99f75f10
---

# PFAvatar: Pose-Fusion 3D Personalized Avatar Reconstruction from Real-World Outfit-of-the-Day Photos

**Conference**: AAAI 2026  
**arXiv**: [2511.12935](https://arxiv.org/abs/2511.12935)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: 3D Avatar Reconstruction, OOTD Photos, Diffusion Models, NeRF, Score Distillation Sampling

## TL;DR

Proposes PFAvatar, a two-stage approach (pose-aware diffusion model fine-tuning + NeRF distillation) to reconstruct high-quality 3D human avatars from real-world "Outfit-of-the-Day" (OOTD) photos, achieving personalization in just 5 minutes, representing a 48x speedup compared to prior methods.

## Background & Motivation

Reconstructing a personalized 3D human model from daily photos is a novel and practical task. "Outfit-of-the-Day" (OOTD) photos exhibit several distinct characteristics: (1) consistent identity, clothing, hairstyle, and accessories across images, (2) diverse poses and scales, (3) frequent occlusions and severe truncations, and (4) different perspectives in complex backgrounds. These characteristics pose severe challenges to existing 3D reconstruction methods.

A representative prior work, PuzzleAvatar, adopts a "disassemble-reassemble" strategy: segmenting OOTD photos into multiple semantic assets (clothing, accessories, face, hair), associating each with a Stable Diffusion token, and then assembling them into a 3D avatar. However, this method suffers from **three key limitations**:

**Inconsistent Segmentation**: Fine-grained segmentation easily introduces visual inconsistencies (e.g., misaligned segmentation boundaries, incorrect part labeling), leading to seams and artifacts in the assembled 3D model.

**Lack of Pose-Controlled Generation**: Since individual parts are learned separately, the model cannot generate complete human images in specific poses, easily causing the Janus problem during SDS optimization.

**Low Training Efficiency**: Learning multiple independent components significantly increases training time (~4 hours), limiting its utility.

**Limitations of Mesh Representation**: The topology of DMTet is constrained by the initial mesh structure, making it difficult to represent complex topological changes (e.g., hair strands, clothing textures).

To address these issues, PFAvatar proposes a novel end-to-end framework that directly models full-body appearance without disassembly.

## Method

### Overall Architecture

PFAvatar consists of two stages:
1. **ControlBooth**: Fine-tuning a pose-aware diffusion model $\mathcal{M}_b$ on a few OOTD photos.
2. **BoothAvatar**: Distilling a NeRF-based 3D avatar from the fine-tuned $\mathcal{M}_b$.

### Key Designs

#### 1. **ControlBooth: Pose-Aware Diffusion Model**

**Function**: Train a diffusion model capable of generating personalized human images conditioned on arbitrary poses.

**Data Preprocessing Pipeline**:
- Use Ground-SAM to isolate the foreground human from the background (only isolating the body region, avoiding the inconsistencies of fine-grained part segmentation).
- Use a pre-trained ControlNet to predict the pose $\{\mathcal{P}_i\}$ for each image.
- Use GPT-4V to generate detailed textual descriptions $\mathcal{T}_i$ (including attributes like body orientation, hairstyle, and clothing).

**Training Loss** consists of two components:

**Reconstruction Diffusion Loss**:

$$\mathcal{L}_{\text{rec}} = \mathbb{E}\left[\|\mathcal{D}_\theta(\alpha_t \mathcal{I}_i + \sigma_t \epsilon, \mathbf{c}_{t_i}, \mathbf{c}_{p_i}) - \mathcal{I}_i\|_2^2\right]$$

**Conditional Prior Preservation Loss (CPPL)**: This is one of the core innovations of this work. During few-shot fine-tuning, the model is prone to overfitting to the training poses, losing its ability to generate diverse poses (as illustrated by the pose rigidity in the middle row of Figure 3). CPPL regularizes the training by using prior data generated from the frozen pre-trained model:

$$\mathcal{L}_{\text{cppl}} = \mathbb{E}\left[\lambda w'_t \|\mathcal{D}_\theta(\alpha_t \mathcal{I}_{pr_i} + \sigma_t \epsilon, \mathbf{c}_{prt_i}, \mathbf{c}_{prp_i}) - \mathcal{I}_{pr_i}\|_2^2\right]$$

**Design Motivation**: CPPL essentially allows the model to learn new identities while maintaining the ability to generate diverse poses and viewpoints, preventing language drift and control drift. Personalization is completed in just 5 minutes, which is 48x faster than PuzzleAvatar.

#### 2. **BoothAvatar: NeRF Representation and 3D-SDS Distillation**

**Function**: Distill a 3D NeRF avatar in a canonical A-pose from the fine-tuned diffusion model.

**Reasons for Choosing NeRF over Mesh**:
- NeRF's volume density naturally handles occlusions (via transmittance), preventing the generation of fake surfaces.
- NeRF's continuous volume rendering can leverage high-frequency positional encodings like hash grids to preserve fine textures (e.g., hair strands, patterns).
- Mesh representation is limited by resolution-dependent discretization, performing poorly on high-frequency details.

Instant-NGP is adopted as the canonical avatar representation and optimized via 3D-consistent SDS:

$$\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{3D-SDS}} = \mathbb{E}\left[w(t)(\boldsymbol{\epsilon}_\phi(\mathbf{x}_t; y, t, c) - \boldsymbol{\epsilon})\frac{\partial \mathbf{z}_t}{\partial \mathbf{x}}\frac{\partial \mathbf{x}}{\partial \boldsymbol{\theta}}\right]$$

where the conditioning image $c$ uses a skeleton image to provide minimal structural priors to facilitate complex avatar generation.

#### 3. **Local Geometry Loss**

**Function**: Address the degradation of fine structures, such as hands and faces, caused by the instability of SDS optimization.

Based on a predefined body part mesh, NeRF density is aligned with the part mesh using a margin ranking loss:

$$\mathcal{L}_{\text{geo}} = \begin{cases} (\max(0, \tau_{\max} - \tau(\mathbf{p})))^2 & \text{if } \mathbf{p} \text{ on mesh} \\ (\max(0, \tau(\mathbf{p}) - \tau_{\min}))^2 & \text{if } \mathbf{p} \text{ not on mesh} \end{cases}$$

**Design Motivation**: SDS optimization lacks human body priors, easily leading to blurry fingers and faces. Constraining local region densities with a predefined mesh preserves fine structures without restricting global optimization.

### Loss & Training

- **ControlBooth Stage**: $\mathcal{L}^{\text{CB}}_{\text{total}} = \mathcal{L}_{\text{rec}} + \lambda_{\text{cppl}} \mathcal{L}_{\text{cppl}}$, with $\lambda_{\text{cppl}}=1$
- **BoothAvatar Stage**: $\mathcal{L}^{\text{BA}}_{\text{total}} = \mathcal{L}_{\text{3D-SDS}} + \lambda_{\text{geo}} \mathcal{L}_{\text{geo}}$, with $\lambda_{\text{geo}}=1.0$
- **Multi-resolution Progressive Sampling**: Gradually increasing the upsampling resolution facilitates more stable SDS training.

The sampling strategy combines two spaces: (1) sampling in the canonical SMPL-X space to generate more pose-conditional images to ensure 3D consistency; (2) sampling in the observation space to capture higher-quality appearance details.

## Key Experimental Results

### Main Results

#### Identity Preservation Comparison (ControlBooth Stage)

| Method | CLIP-I (body) | CLIP-I (head) | DINO (body) | DINO (head) | CLIP-T (body) | CLIP-T (head) |
|------|-------------|-------------|-----------|-----------|-------------|-------------|
| **PFAvatar** | **0.9016** | **0.9432** | **0.7282** | **0.9352** | **0.3036** | **0.2996** |
| PuzzleAvatar | 0.8147 | 0.7705 | 0.6257 | 0.6096 | 0.2340 | 0.1849 |
| FreeCustom | 0.8573 | 0.9337 | 0.7022 | 0.9222 | 0.2583 | 0.2811 |
| InstantID | 0.7687 | 0.8164 | 0.5977 | 0.8302 | 0.2164 | 0.2711 |

#### PuzzleIOI Benchmark Reconstruction Quality

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| **PFAvatar** | **27.576** | **0.952** | **0.041** |
| PuzzleAvatar | 24.687 | 0.930 | 0.062 |
| TECH | 23.635 | 0.919 | 0.065 |
| AvatarBooth | 16.431 | 0.758 | 0.153 |

PFAvatar significantly outperforms all baselines across all evaluation metrics.

### Ablation Study

| Configuration | CLIP-I (body) | DINO (body) | CLIP-T (body) | Description |
|------|-------------|-----------|-------------|------|
| Full | 0.9125 | 0.8072 | 0.3546 | Full model |
| w/o Head Part Data | 0.8702 | 0.7154 | 0.2912 | Significant facial degradation |
| w/o ControlBooth | 0.8352 | 0.7091 | 0.2314 | Consistency and color shift |
| w/o 3D-SDS | 0.8021 | 0.7281 | 0.2281 | Impaired A-pose generation |
| w/o $\mathcal{L}_{\text{geo}}$ | 0.8929 | 0.8011 | 0.3257 | Blurry hand geometry |
| w/o Multi-sampling | 0.8654 | 0.7486 | 0.2812 | Slow convergence, poor details |

Removing any component leads to visible degradation, proving the necessity of the proposed design.

### Key Findings

1. **CPPL Effectively Prevents Overfitting**: Figure 3 qualitatively demonstrates that without CPPL, the model overfits to the training poses, whereas injecting CPPL enables the generation of diverse and controllable poses.
2. **NeRF Representation Outperforms Mesh**: NeRF is significantly superior to DMTet mesh in handling occlusions and retaining high-frequency textures.
3. **Abundant Downstream Applications**: The reconstructed NeRF avatars support virtual try-ons, animation, facial animation, and human video reenactment.

## Highlights & Insights

1. **End-to-End Design Sidesteps Segmentation Bottleneck**: Directly modeling the full-body appearance elegantly bypasses the segmentation inconsistencies in PuzzleAvatar.
2. **5-Minute Personalization**: Achieves personalization 48x faster than PuzzleAvatar, significantly boosting practical utility.
3. **CPPL Regularization Concept**: Utilizing self-generated data from the pre-trained model for regularization is an elegant solution to address overfitting in few-shot fine-tuning.
4. **Local Geometry Constraint**: Smartly exploits human part priors to stabilize local structures during SDS optimization.

## Limitations & Future Work

1. NeRF representation is relatively new and lacks the rich manipulation toolchains of traditional mesh-based methods (though its animation capability is demonstrated in this paper).
2. The reliance on GPT-4V to generate textual descriptions increases cost and dependency on external APIs.
3. There is still room for improvement in robustness against extreme occlusions and severe truncations (though it already outperforms baselines).
4. Currently, only skeleton conditions are used; exploring multi-conditional control, such as depth maps combined with skeletons, could further improve quality.

## Related Work & Insights

- **Relationship with DreamBooth**: ControlBooth in this paper can be viewed as an important extension of DreamBooth toward pose awareness; CPPL solves the degradation problem of DreamBooth in few-shot human fine-tuning.
- **NeRF vs. Mesh**: The detailed comparative analysis of these two representations (handing of occlusions, high-frequency details, topological flexibility) provides valuable references for the 3D human reconstruction community.
- The conditional prior preservation idea of CPPL can be extended to other tasks requiring few-shot fine-tuning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The CPPL and the end-to-end pose-aware scheme are novel, and the specific strategies for NeRF distillation are innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated on two datasets, with comparisons against multiple baselines, user studies, and a comprehensive ablation study.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-motivated, clear comparisons, and rich illustrations.
- **Value**: ⭐⭐⭐⭐⭐ — 5-minute personalization and support for various downstream applications, making it highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Towards High-fidelity 3D Talking Avatar with Personalized Dynamic Texture](../../CVPR2025/3d_vision/towards_high-fidelity_3d_talking_avatar_with_personalized_dynamic_texture.md)
- [\[AAAI 2026\] RTGaze: Real-Time 3D-Aware Gaze Redirection from a Single Image](rtgaze_real-time_3d-aware_gaze_redirection_from_a_single_image.md)
- [\[ECCV 2024\] 3D Reconstruction of Objects in Hands without Real World 3D Supervision](../../ECCV2024/3d_vision/3d_reconstruction_of_objects_in_hands_without_real_world_3d.md)
- [\[CVPR 2026\] Learning 3D Shape Fidelity Metric from Real-world Distortions](../../CVPR2026/3d_vision/learning_3d_shape_fidelity_metric_from_real-world_distortions.md)
- [\[CVPR 2026\] Iris: Bringing Real-World Priors into Diffusion Model for Monocular Depth Estimation](../../CVPR2026/3d_vision/iris_bringing_realworld_priors_into_diffusion_model_for_monocular_depth_estimation.md)

</div>

<!-- RELATED:END -->
