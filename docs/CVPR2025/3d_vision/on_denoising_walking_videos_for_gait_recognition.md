---
title: >-
  [Paper Note] On Denoising Walking Videos for Gait Recognition
description: >-
  [CVPR 2025][3D Vision][Gait Recognition] This paper proposes DenoisingGait, which combines "knowledge-driven denoising" (utilizing generative diffusion models at specific timesteps to filter out gait-irrelevant information) and "geometry-driven denoising" (compressing multi-channel diffusion features into 2D direction vectors via the Feature Matching module) to generate a novel Gait Feature Field representation, achieving state-of-the-art (SOTA) performance on multiple RGB ga…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Gait Recognition"
  - "Diffusion Model"
  - "Feature Denoising"
  - "Optical Flow Field"
  - "Robustness to Clothing Changes"
date: 2026-05-08
content_hash: 326bcaac063970b4
---

# On Denoising Walking Videos for Gait Recognition

**Conference**: CVPR 2025  
**arXiv**: [2505.18582](https://arxiv.org/abs/2505.18582)  
**Code**: [https://github.com/ShiqiYu/OpenGait](https://github.com/ShiqiYu/OpenGait)  
**Area**: 3D Vision  
**Keywords**: Gait Recognition, Diffusion Model, Feature Denoising, Optical Flow Field, Robustness to Clothing Changes

## TL;DR

This paper proposes DenoisingGait, which combines "knowledge-driven denoising" (utilizing generative diffusion models at specific timesteps to filter out gait-irrelevant information) and "geometry-driven denoising" (compressing multi-channel diffusion features into 2D direction vectors via the Feature Matching module) to generate a novel Gait Feature Field representation, achieving state-of-the-art (SOTA) performance on multiple RGB gait datasets.

## Background & Motivation

**Background**: Gait recognition is a non-invasive biometric identification method that recognizes identities through body shape and limb movements in walking videos. Existing methods are mainly divided into "hard denoising" (using predefined representations like silhouettes, skeletons, and SMPL to remove background and texture interference) and "soft denoising" (directly using human priors on RGB videos to suppress non-gait information).

**Limitations of Prior Work**: Hard denoising methods (such as silhouettes, skeletons) have sparse inputs with limited information, losing many structural details beneficial for identity recognition. While soft denoising methods retain more information, they still struggle to completely remove gait-irrelevant factors such as clothing textures and colors. This is especially challenging in **clothing-changing scenarios**, where texture and color information encoded in RGB becomes the primary source of noise for recognition.

**Key Challenge**: Gait recognition requires extracting features that are "invariant to clothing and background, but sensitive to body shape and movement", whereas RGB images naturally encode a large amount of identity-irrelevant visual information. How to filter out this "noise" while preserving structural information is the core challenge.

**Goal**: Design a denoising framework combining both knowledge-driven and geometry-driven paradigms to extract clean gait representations from RGB videos.

**Key Insight**: Inspired by "what I cannot create, I do not understand", this paper explores the potential of generative diffusion models as gait representation learners. It discovers that by controlling the timestep $t$ of the diffusion model, RGB details of different granularities can be selectively filtered—larger $t$ retains global shape and smaller $t$ reconstructs fine textures. At $t=700$, gait recognition achieves optimal performance (gaining 5.3% on CCPG), but residual RGB noise still exists, requiring further geometry-driven denoising.

**Core Idea**: Use diffusion models for coarse denoising (selectively filtering RGB details), and then employ a Feature Matching module for fine denoising (compressing features to direction vectors) to generate a flow-like Gait Feature Field as the final representation.

## Method

### Overall Architecture

DenoisingGait's pipeline: (1) Input RGB frames are projected into the latent space via a VAE encoder, and a pre-trained Stable Diffusion performs a single-step denoising at timestep $t$ to obtain diffusion features $F_l$; (2) The Feature Matching module performs intra-frame matching ($\Delta l=0$) and inter-frame matching ($\Delta l>0$) on $F_l$ to generate static and dynamic Gait Feature Fields, respectively; (3) The two Feature Fields are fed in parallel into GaitBase for gait recognition, trained using triplet loss + cross-entropy loss.

### Key Designs

1. **Diffusion-based Denoising**:

    - **Function**: Filters out gait-irrelevant details in RGB images using a pre-trained diffusion model.
    - **Mechanism**: Given a frame $I_l$, the latent variable $z = \mathcal{E}(I_l)$ is first obtained using a VAE encoder, and then the UNet $\epsilon_\theta$ of a pre-trained SD 1.5 performs a single-step denoising at timestep $t$ without adding random noise: $F_l = \epsilon_\theta(\mathcal{E}(I_l), t)$. The key lies in the choice of the timestep $t$: a too-large $t$ causes excessive blurring and loss of structure, while a too-small $t$ retains too much texture detail. Experiments identify $t=700$ as the optimal value.
    - **Design Motivation**: Diffusion models capture information of different granularities at different timesteps—early timesteps correspond to global shape, while later ones correspond to fine textures. Gait recognition happens to require medium-grained shape information. This concept of "using a generative model for a discriminative task" is highly creative.

2. **Feature Matching + Gait Feature Field**:

    - **Function**: Compresses multi-channel diffusion features into 2D direction vectors to further remove noise encoded in RGB.
    - **Mechanism**: For the feature $f^Q_{\langle i,j \rangle}$ of a query pixel $\langle i,j \rangle$, key features $\mathcal{M}^K_{\langle i,j \rangle}$ are searched in its neighborhood to compute a Softmax similarity distribution $\mathcal{P}$. Then, a weighted sum is computed using a fixed direction template $\mathcal{T}$ (comprising relative displacements of neighbors $[\hat{i}, \hat{j}]$) to obtain the direction vector $G_{\langle i,j \rangle} = \mathcal{P} \cdot \mathcal{T}$. Intra-frame matching ($\Delta l=0$) yields the static Gait Feature Field (similar to a SIFT gradient field), and inter-frame matching ($\Delta l>0$) yields the dynamic Gait Feature Field (similar to an optical flow field). The background is removed using a silhouette mask.
    - **Design Motivation**: Compressing multi-channel features into 2D direction vectors naturally filters out high-dimensional texture information encoded in RGB, retaining only local structural and directional movement features. This is inspired by SIFT descriptors and optical flow estimation, but is fully end-to-end learnable.

3. **Texture Suppression**:

    - **Function**: Randomly masks high-texture regions during training to encourage the model to learn texture-invariant gait features.
    - **Mechanism**: It is observed that the magnitude of direction vectors $\|G^{\text{Static}}_{\langle i,j \rangle}\|_2$ in the static Gait Feature Field reflects texture intensity. During training, pixels with magnitudes greater than a threshold $m=0.5$ are set to zero with probability $p$, forcing the model not to rely on texture info for recognition.
    - **Design Motivation**: In clothing-changing scenarios, texture information is the primary source of interference. This operation tells the model that "texture is unreliable," forcing it to focus on stable features such as body shape and movement.

### Loss & Training

- Uses a standard combination of triplet loss + cross-entropy loss to train GaitBase.
- SGD optimizer, initial learning rate of 0.1, weight decay of 0.0005.
- An ordered sampling strategy is adopted, processing 20 frames per training step.
- Trained for 60k steps on the CCPG dataset with a batch size of (8, 4).

## Key Experimental Results

### Main Results

| Method | Input | CCPG-CL (Clothing-Changing) | CCPG-Mean | Protocol |
|------|------|----------------|-----------|------|
| GaitBase | Sils | 71.6 | 75.5 | Gait |
| DeepGaitV2 | Sils | 78.6 | 83.3 | Gait |
| BigGait | RGB | 82.6 | 87.2 | Gait |
| SkeletonGait++ | Sils+Skeleton | 79.1 | 83.7 | Gait |
| MultiGait++ | Sils+Parsing+Flow | 83.9 | 87.6 | Gait |
| **DenoisingGait** | **RGB+Sils** | **84.0** | **89.5** | **Gait** |
| **DenoisingGait** | **RGB+Sils** | **91.8** | **95.7** | **ReID** |

### Ablation Study

| Configuration | CCPG-CL | Description |
|------|---------|------|
| Diffusion Baseline (No $\epsilon_\theta$) | ~78.7 | VAE encoding + GaitBase only |
| Diffusion Baseline ($t=700$) | ~84.0 | Added diffusion denoising, +5.3% |
| + Feature Matching (Static) | Gain | Added geometry-driven denoising |
| + Feature Matching (Dynamic) | Further Gain | Added motion field info |
| + Texture Suppression | Final | Texture suppression further enhances robustness |

### Key Findings

- The choice of timestep $t$ is crucial: $t=700$ is the optimal point on CCPG; either too large or too small values decrease performance, validating the multi-granularity property of diffusion models.
- The static Gait Feature Field automatically avoids texture-rich regions (e.g., clothing patterns) and focuses on body contours and joint structures.
- The activation focus of the dynamic Gait Feature Field is concentrated on moving limb parts, showing high consistency with the kinematic features of gait.
- Cross-domain evaluation (training on one dataset and testing on another) also yields outstanding performance, demonstrating the strong generalization ability of diffusion features.

## Highlights & Insights

- **Application of semantic meaning of diffusion model timestep to discriminative tasks**: Discovering that the timestep can serve as an "information granularity controller" provides insights that can be transferred to other discriminative tasks requiring multi-grain features (e.g., person re-identification, fine-grained classification).
- **Design of the Gait Feature Field**: The core idea of compressing multi-channel features into 2D direction vectors is both elegant and effective—it naturally filters out high-dimensional texture information while preserving directional information of shape and motion. This Feature Matching module can serve as a plug-and-play feature denoising tool.
- **Correlation discovery between texture intensity and direction vector magnitude**: Finding that $\|G^{\text{Static}}\|_2$ reflects texture intensity provides a natural metric for designing texture suppression operations.

## Limitations & Future Work

- Relies on the pre-trained SD 1.5 model for feature extraction, which involves high computational overhead during inference (requiring one diffusion forward pass per frame), making it less suitable for real-time scenarios.
- The timestep $t=700$ is obtained by tuning on CCPG; it may need to be re-searched when transferred to other datasets.
- Robustness has only been tested on upper-body clothing-changing scenarios, and its efficacy under more extreme appearance changes (e.g., raincoats, helmets) remains to be validated.
- The accuracy of silhouette masks affects the background removal performance, occasionally leading to performance drops in highly occluded conditions.

## Related Work & Insights

- **vs BigGait**: BigGait is also an end-to-end RGB method, using feature smoothing for soft denoising. DenoisingGait's dual-driven (diffusion + structural) denoising is more thorough, gaining 1.4% in CL scenarios.
- **vs SkeletonGait++**: The multi-modal fusion of skeleton and silhouette is effective (83.7%), but is limited by the information loss in pre-extracted representations. DenoisingGait directly extracts features from RGB, retaining richer information.
- **vs MultiGait++**: Fusing three representations (silhouette, parsing, and optical flow) yields results close to DenoisingGait under the Gait protocol (87.6 vs 89.5), but DenoisingGait is simpler and significantly outperforms it under the ReID protocol.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to apply diffusion models to gait recognition; proposed the novel Gait Feature Field representation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on three datasets, cross-domain scenarios, with sufficient ablation and visualization analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, with a smooth logical transition from diffusion models to Feature Matching.
- Value: ⭐⭐⭐⭐ Provides a brand-new paradigm for gait recognition; the multi-granularity control idea of diffusion features has broad implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Denoising Functional Maps: Diffusion Models for Shape Correspondence](denoising_functional_maps_diffusion_models_for_shape_correspondence.md)
- [\[ICCV 2025\] Hierarchical Material Recognition from Local Appearance](../../ICCV2025/3d_vision/hierarchical_material_recognition_from_local_appearance.md)
- [\[CVPR 2025\] Recovering Dynamic 3D Sketches from Videos](recovering_dynamic_3d_sketches_from_videos.md)
- [\[NeurIPS 2025\] Walking the Schrödinger Bridge: A Direct Trajectory for Text-to-3D Generation](../../NeurIPS2025/3d_vision/walking_the_schrödinger_bridge_a_direct_trajectory_for_text-to-3d_generation.md)
- [\[CVPR 2025\] GenFusion: Closing the Loop between Reconstruction and Generation via Videos](genfusion_closing_the_loop_between_reconstruction_and_generation_via_videos.md)

</div>

<!-- RELATED:END -->
