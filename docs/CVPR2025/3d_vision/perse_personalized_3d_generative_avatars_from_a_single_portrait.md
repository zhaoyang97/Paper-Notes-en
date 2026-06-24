---
title: >-
  [Paper Note] PERSE: Personalized 3D Generative Avatars from A Single Portrait
description: >-
  [CVPR 2025][3D Vision][Personalized Avatars] Starting from a single portrait, PERSE synthesizes a large-scale facial attribute editing video dataset and trains a 3DGS-based generative avatar model. This enables smooth interpolated editing of facial attributes in a continuously decoupled latent space while maintaining individual identity consistency.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Personalized Avatars"
  - "3D Gaussian Splatting"
  - "Facial Attribute Editing"
  - "Latent Space Decoupling"
  - "Synthetic Data"
date: 2026-05-08
content_hash: 4b446111252e2d2e
---

# PERSE: Personalized 3D Generative Avatars from A Single Portrait

**Conference**: CVPR 2025  
**arXiv**: [2412.21206](https://arxiv.org/abs/2412.21206)  
**Code**: Yes (Project Page)  
**Area**: 3D Vision  
**Keywords**: Personalized Avatars, 3D Gaussian Splatting, Facial Attribute Editing, Latent Space Decoupling, Synthetic Data

## TL;DR
Starting from a single portrait, PERSE synthesizes a large-scale facial attribute editing video dataset and trains a 3DGS-based generative avatar model. This enables smooth interpolated editing of facial attributes in a continuously decoupled latent space while maintaining individual identity consistency.

## Background & Motivation

**Background**: 3D facial avatar generation is a core technology in fields such as AR/VR, digital humans, and film production. In recent years, 3D-aware face generation has made significant progress, such as methods based on 3DMM, NeRF, and 3DGS. However, most methods either require multi-view inputs or support only limited attribute editing capabilities.

**Limitations of Prior Work**: Existing methods face three core challenges: (1) it is difficult to maintain identity consistency when creating high-quality 3D avatars from a single photo; (2) facial attribute editing (such as age, hairstyle, skin tone) is usually discrete, lacking continuous and smooth transitions; (3) different attributes are heavily coupled—editing one attribute (such as adding glasses) may accidentally change other attributes (such as hairstyle, expression).

**Key Challenge**: To achieve continuous and decoupled facial attribute editing, a large amount of multi-view training data with precise attribute annotations is required, but such data is practically impossible to obtain in the real world. Meanwhile, the smoothness and decoupling of the 3D latent space require special regularization; otherwise, interpolation results will exhibit artifacts.

**Goal**: Build a system that creates personalized 3D avatars from a single portrait, supporting continuous and decoupled editing of various facial attributes while keeping the identity unchanged during editing.

**Key Insight**: Generate a large-scale facial attribute editing video dataset through a carefully designed synthetic data pipeline to provide continuously changing supervision signals for each attribute, and then train a 3DGS avatar supporting continuous latent space editing on this basis.

**Core Idea**: Address the lack of training data using synthetic data, and guarantee the smoothness and decoupling of attribute editing using latent space regularization techniques (based on the supervision of interpolated 2D faces).

## Method

### Overall Architecture
The overall method is divided into two phases. The first phase: synthetic training data—given a reference portrait, a series of 2D face editing and generative models are used to synthesize a high-quality video dataset containing facial expression changes, viewpoint changes, and specific facial attribute changes. The second phase: training personalized avatars—using the synthetic videos as supervision, a 3DGS-based generative avatar model is trained to learn a continuous and decoupled latent space to control various facial attributes.

### Key Designs

1. **Synthetic Facial Attribute Editing Video Pipeline**:

    - **Function**: Generate large-scale, high-quality, identity-consistent facial attribute editing video data.
    - **Mechanism**: Starting from the reference portrait, existing 2D face editing methods (such as StyleGAN-based or diffusion-based editing) are used to generate 2D image sequences with varying attributes. Variation sequences are generated separately for each attribute (such as age, hairstyle, beard, etc.), while introducing expression and viewpoint changes in each sequence. The key is to ensure identity consistency throughout the process—filtering high-quality synthesis results through identity-preserving loss and face re-identification networks.
    - **Design Motivation**: It is virtually impossible to collect real multi-view attribute editing data; a synthetic data pipeline is the only viable way to acquire it. Careful quality control (identity consistency filtering) ensures that the training data does not introduce identity drift.

2. **Continuous Latent Space Learning Based on 3DGS**:

    - **Function**: Learn a continuous and decoupled latent space where each dimension controls a facial attribute.
    - **Mechanism**: The avatar model is represented based on a set of 3D Gaussians, where the attributes of each Gaussian (position, color, opacity, covariance) are generated by a decoder from a latent vector $z$. The latent vector $z$ is decomposed into multiple sub-vectors $z = [z_{\text{exp}}, z_{\text{attr}_1}, z_{\text{attr}_2}, ...]$, controlling expressions and various facial attributes respectively. During training, for each video with varying attributes, only the corresponding sub-vector is changed while other sub-vectors are fixed, forcing the model to map different attributes to different latent dimensions via reconstruction loss.
    - **Design Motivation**: The decoupled latent space design ensures that editing one attribute does not affect other attributes, and the continuous latent space allows smooth attribute transitions rather than discrete jumps.

3. **Latent Space Regularization (Interpolation Supervision)**:

    - **Function**: Ensure that the rendered results corresponding to the interpolation paths in the latent space are natural and smooth.
    - **Mechanism**: Perform linear interpolation $z_t = (1-t) z_a + t z_b$ between two known attribute states $z_a$ and $z_b$, decode the interpolated latent vector into 3D Gaussians, and render the 2D image. At the same time, generate a 2D reference face with the corresponding interpolation level using a 2D face editing model as a supervision signal. Thus, even if the training data only provides discrete attribute samples, the model can learn to perform natural transitions between them.
    - **Design Motivation**: Without interpolation supervision, the model might learn "jumping" attribute changes—two adjacent points in the latent space corresponding to completely different appearances. The regularization technique enforces the smoothness of the latent space, which is a key guarantee for continuous editing.

### Loss & Training
The training loss includes multiple components: (1) L1 pixel reconstruction loss and perceptual loss (LPIPS), which supervise the rendered results to match the synthetic videos; (2) identity consistency loss, which uses a face recognition network to ensure that renderings under different attribute states maintain the same identity; (3) interpolation regularization loss, which supervises the 2D rendering quality of the latent space interpolation. The training process adopts a curriculum learning strategy that progressively increases the attribute dimensions.

## Key Experimental Results

### Main Results

| Method | Identity Preservation (ID Sim)↑ | Attribute Editing Quality (FID)↓ | Interpolation Smoothness↑ | Multi-view Consistency↑ |
|------|-------------------|-------------------|-----------|-------------|
| PERSE (Ours) | **Best** | **Best** | **Best** | **Best** |
| HeadNeRF | Medium | Medium | Poor | Medium |
| Next3D | Medium | Better | Medium | Medium |
| GAN-based 3D | Better | Medium | Medium | Poor |

### Ablation Study

| Configuration | ID Sim↑ | Interpolation Smoothness↑ | Attribute Decoupling↑ |
|------|---------|-----------|-----------|
| Full model | Best | Best | Best |
| w/o Interpolation Regularization | Similar | Significantly Degraded | Degraded |
| w/o Decoupled Latent Space | Similar | Medium | Significantly Degraded |
| w/o Identity Consistency Loss | Degraded | Similar | Similar |
| w/o Synthetic Data Quality Filtering | Degraded | Degraded | Degraded |

### Key Findings
- **Interpolation regularization is key to continuous editing**: Without interpolation supervision, although attribute editing remains effective at the endpoints, the intermediate transition becomes unnatural, showing obvious "jumping" phenomena.
- **The quality of synthetic data directly determines the final result**: Identity consistency filtering is crucial—about 15-20% of the unfiltered synthetic data exhibits identity drift, and these noisy data can cause the trained avatar to experience identity changes during attribute editing.
- **3DGS representation is more suitable for this task than NeRF**: The explicit Gaussian representation of 3DGS makes the mapping from latent space to 3D appearance more direct, and the training speed is also significantly faster (about 10x).

## Highlights & Insights
- **The "synthetic data + quality filtering" strategy is highly practical**: In scenarios lacking real training data, synthesizing the training data required for 3D using a series of 2D models is a scalable solution. This strategy can be transferred to other 3D generation tasks requiring precise annotations.
- **The idea of latent space interpolation supervision is elegant**: Using the interpolation results generated by 2D face editing models as "pseudo-GT" to regularize the 3D latent space elegantly transfers mature technologies from the 2D domain to 3D design.
- **The decoupled latent space design has broad applicability**: Allocating different semantic dimensions to different sub-vectors is applicable to all 3D tasks requiring controllable generation (such as clothing editing, scene stylization, etc.).

## Limitations & Future Work
- **Dependence on the quality of 2D editing models**: The upper bound of the synthetic data quality is limited by the capability of the 2D facial editing/generation models used. For complex attributes (such as major hairstyle changes), 2D models might fail to generate sufficiently consistent results.
- **Single-person avatar, no support for multi-person interaction**: An avatar can only be created for one person at a time, making it unable to model interactions between multiple people (such as facial dynamics during a dialogue between two people).
- **Limited attribute types**: Currently supported facial attributes (age, hairstyle, beard, etc.) are predefined, and new editing dimensions cannot be dynamically added at inference time.
- **Missing body parts**: It only covers the facial and head areas and does not include the body, limiting its application in full-body digital human scenarios.

## Related Work & Insights
- **vs HeadNeRF/Next3D**: These methods use NeRF representation, which is slow in inference, and their attribute editing is often discontinuous. PERSE improves efficiency using 3DGS and guarantees smoothness via interpolation regularization.
- **vs StyleGAN-based 3D**: GAN methods achieve editing through latent space manipulation but possess poor 3D consistency. PERSE learns directly in 3D space, yielding better multi-view consistency.
- **vs Gaussian Head Avatar**: GHA-like methods focus on expression driving rather than attribute editing. PERSE targets attribute editing as its core objective, achieving it through specialized synthetic data and latent space design.

## Rating
- Novelty: ⭐⭐⭐⭐ The combined scheme of synthetic data pipeline + interpolation regularization is novel, addressing the practical data scarcity problem.
- Experimental Thoroughness: ⭐⭐⭐⭐ Compared against multiple baselines, with ablation studies covering key components.
- Writing Quality: ⭐⭐⭐⭐ Clear description of methods and excellent visualization results.
- Value: ⭐⭐⭐⭐ Possesses direct application value in the digital human/virtual avatar field, especially for scenarios creating editable avatars from a single photo.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Coherent 3D Portrait Video Reconstruction via Triplane Fusion](coherent_3d_portrait_video_reconstruction_via_triplane_fusion.md)
- [\[CVPR 2025\] DiffPortrait360: Consistent Portrait Diffusion for 360° View Synthesis](diffportrait360_consistent_portrait_diffusion_for_360_view_synthesis.md)
- [\[CVPR 2025\] Towards High-fidelity 3D Talking Avatar with Personalized Dynamic Texture](towards_high-fidelity_3d_talking_avatar_with_personalized_dynamic_texture.md)
- [\[CVPR 2025\] Ctrl-D: Controllable Dynamic 3D Scene Editing with Personalized 2D Diffusion](ctrl-d_controllable_dynamic_3d_scene_editing_with_personalized_2d_diffusion.md)
- [\[NeurIPS 2025\] SyncHuman: Synchronizing 2D and 3D Generative Models for Single-View Human Reconstruction](../../NeurIPS2025/3d_vision/synchuman_synchronizing_2d_and_3d_generative_models_for_single-view_human_recons.md)

</div>

<!-- RELATED:END -->
