---
title: >-
  [Paper Note] SimVS: Simulating World Inconsistencies for Robust View Synthesis
description: >-
  [CVPR 2025][3D Vision][Robust View Synthesis] SimVS leverages video diffusion models to simulate inconsistencies (e.g., changes in lighting, object motion) in real-world casual captures. It uses this simulated data to train a multi-view harmonization network that converts inconsistent, sparse observations into consistent multi-view images, thereby enabling high-quality static 3D reconstruction from in-the-wild casually captured scenes.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Robust View Synthesis"
  - "Video Diffusion Models"
  - "Simulating World Inconsistencies"
  - "Multi-View Harmonization"
  - "Data Augmentation"
date: 2026-05-08
content_hash: c2238c4312f4067b
---

# SimVS: Simulating World Inconsistencies for Robust View Synthesis

**Conference**: CVPR 2025  
**arXiv**: [2412.07696](https://arxiv.org/abs/2412.07696)  
**Code**: None  
**Area**: 3D Vision / Novel View Synthesis  
**Keywords**: Robust View Synthesis, Video Diffusion Models, Simulating World Inconsistencies, Multi-View Harmonization, Data Augmentation

## TL;DR

SimVS leverages video diffusion models to simulate inconsistencies (e.g., changes in lighting, object motion) in real-world casual captures. It uses this simulated data to train a multi-view harmonization network that converts inconsistent, sparse observations into consistent multi-view images, thereby enabling high-quality static 3D reconstruction from in-the-wild casually captured scenes.

## Background & Motivation

**Background**: Novel view synthesis has made impressive progress. Multi-view diffusion models like CAT3D can generate high-quality novel views given sparse, consistent inputs. Mature solutions also exist for handling scene dynamics (4D representations) and illumination changes (appearance embeddings) under dense captures.

**Limitations of Prior Work**: Real casual capture presents two simultaneous challenges: inputs are extremely sparse (a few photos) and contain inconsistencies (lighting changes, moving objects). Methods like CAT3D assume consistent inputs and collapse when encountering inconsistencies, averaging observations from different states and leading to severe blurriness and artifacts. Robust view synthesis methods require dense videos and are inapplicable to sparse scenes.

**Key Challenge**: There is a lack of paired training data to train robust sparse view synthesis models. Existing multi-view datasets are consistent, and heuristic augmentations (random tinting, optical flow warping) cannot simulate realistic 3D inconsistencies. Synthetic data (e.g., Objaverse) exhibits a large domain gap.

**Goal**: (1) Establish a method to generate realistic and diverse inconsistent training data; (2) Train a harmonization model that can recover consistent multi-views from sparse, inconsistent inputs.

**Key Insight**: Video diffusion models can "simulate the world"—generating physically plausible dynamic changes and illumination variations given a static image. Instead of using them directly for generation, they are utilized to "create training data."

**Core Idea**: Use video diffusion models to generate "inconsistent variants" for each viewpoint of an existing consistent multi-view dataset, producing a paired dataset of (inconsistent inputs, consistent targets). A harmonization network is trained on this data to convert real inconsistent sparse captures into consistent observations, which are then used for standard 3D reconstruction.

## Method

### Overall Architecture

The pipeline consists of three stages: (1) **Simulating Inconsistencies**: For each viewpoint in a consistent multi-view dataset, a video diffusion model (Lumiere) is used to generate scene-changing videos, from which inconsistent frames are sampled; (2) **Harmonization Network Training**: Finetuned on the CAT3D architecture, this network takes inconsistent images + a reference image (marking the desired state) as input, and outputs multi-view images consistent with the reference; (3) **3D Reconstruction**: The harmonized consistent images are densified using CAT3D and used to train Zip-NeRF, yielding the final 3D representation.

### Key Designs

1. **Video Model Data Augmentation**:

    - **Function**: Generate paired inconsistent training data from consistent multi-view data.
    - **Mechanism**: For each viewpoint image $x_i$, Gemini generates a prompt describing scene dynamics (e.g., "the woman swings the pillow"). A "static shot, " prefix is prepended to keep the camera stationary, and Lumiere generates the video. Negative prompts such as "panning view" and "orbit shot" are used to suppress camera motion. Around 640 frames of inconsistent variants are generated per scene, totaling ~6 million frames for the dynamics dataset and ~12 million frames for the illumination dataset. The same prompt is applied across all viewpoints of a scene to ensure inconsistencies are highly correlated across views (matching real-world physics).
    - **Design Motivation**: Video models naturally understand 3D physics—object motion follows physical laws, and lighting changes possess scene semantics. Heuristic augmentations only perform 2D transformations, which leads the model to learn simple "pixel-copying" strategies.

2. **Multi-View Harmonization Model**:

    - **Function**: Convert inconsistent sparse inputs into multi-view outputs consistent with the reference image state.
    - **Mechanism**: Finetuned on the latent space multi-view diffusion model of CAT3D. The VAE encoding $\tilde{z}_i$ of the inconsistent image is concatenated with the target raymap and noisy latent as an extra conditioning signal. The reference image is marked with a binary mask of 1s, and inconsistent images with 0s. The training objective is to denoise and predict the consistent target latents $z_{1:7}$. During training, a uniform dropout is applied to the number of conditioning frames (1-8 frames) to make the model robust to varying input counts.
    - **Design Motivation**: The harmonization network does not explicitly model motion or lighting—it only needs to learn to "unify all observations to the reference state." This avoids constraints on specific types of inconsistency, making the method effective for arbitrary inconsistencies.

3. **Cascaded 3D Reconstruction**:

    - **Function**: Reconstruct high-quality NeRF from the harmonized 8 consistent images.
    - **Mechanism**: 7 harmonized outputs + 1 reference image = 8 consistent images. A finetuned CAT3D (conditioned on 5 images for more context compared to the original 3) densifies the views to generate sufficient viewpoints, followed by training Zip-NeRF. The conditioning set for the densified CAT3D always includes the reference image.
    - **Design Motivation**: 8 images are too sparse for high-quality standard 3D reconstruction, requiring multi-view diffusion for expansion. The cascaded design allows the processes of harmonization and densification to be decoupled and individually optimized.

### Loss & Training

The harmonization network utilizes the standard diffusion denoising loss: $\mathbb{E}[w(t)\|f(\alpha_t z_{1:7} + \sigma_t \epsilon; z_0, \tilde{z}_{1:7}) - z_{1:7}\|^2]$. The dynamic model is trained on the Mannequin Challenge dataset (static multi-view + simulated dynamics), and the illumination model is trained on the RealEstate10k dataset (diverse illumination scenes). Prompts for illumination changes are uniformly sampled from a predefined list (as Lumiere's lighting generation is unstable on non-generic prompts), whereas prompts for dynamic changes are generated per-scene by Gemini.

## Key Experimental Results

### Main Results

**Scene Dynamics (DyCheck Dataset)**

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| CAT3D (Single Image) | 14.61 | 0.382 | 0.473 |
| CAT3D (All Images) | 15.59 | 0.448 | 0.462 |
| **SimVS** | **16.73** | **0.463** | **0.413** |

**Illumination Changes (Custom Dataset)**

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| CAT3D (Single Image) | 15.06 | 0.526 | 0.552 |
| CAT3D (All Images) | 18.26 | 0.625 | 0.419 |
| **SimVS** | **20.98** | **0.707** | **0.357** |

### Ablation Study

| Augmentation Strategy | PSNR↑ | SSIM↑ | LPIPS↓ |
|---------|-------|-------|--------|
| Heuristic (Dynamics, optical flow warp) | 15.52 | 0.448 | 0.466 |
| Objaverse Synthetic Data | 14.92 | 0.380 | 0.524 |
| **Video Model Augmentation (Dynamics)** | **16.60** | **0.462** | **0.409** |
| Heuristic (Illumination, random color jittering) | 18.96 | 0.645 | 0.406 |
| **Video Model Augmentation (Illumination)** | **20.98** | **0.707** | **0.357** |

### Key Findings

- Video model augmentation significantly outperforms heuristic augmentations and synthetic data—video models simulate physically plausible 3D motion and illumination changes, whereas heuristics only perform 2D transformations.
- Models trained with heuristic enhancements (such as optical flow warping) learn a "pixel-copying" strategy where they copy undeformed pixels instead of understanding movement.
- CAT3D (All Images) attempts to average all states, leading to severe blurriness, while CAT3D (Single Image) loses multi-view geometric cues.
- COLMAP achieves a 100% registration success rate on SimVS samples, compared to 4/28 failures on CAT3D baselines—validating that SimVS outputs have better geometric consistency.
- The improvement in illumination-change scenes is more substantial than in dynamic scenes (PSNR +2.7 vs. +1.1), likely because illumination variations are more systematic and better simulated by the video model.

## Highlights & Insights

- **Video models as world simulators**: Instead of directly utilizing the video model for generation, it is used as a "data factory." This is an elegant indirect leverage—converting the physical understanding of video models into realistic training data.
- **Abstraction of harmonization**: By avoiding separate modeling for motion and lighting, the task is abstracted into an "inconsistent $\rightarrow$ consistent" mapping. This concept is simple and generalizes well to unseen types of inconsistencies.
- **Crucial role of negative prompts**: Appending negative prompts like "panning view" is critical to keeping the camera stationary—this seemingly minor detail actually dictates the usability of the generated data.
- **Scalability with video models**: As stronger video models like Sora emerge, the quality of generated data will scale automatically without altering the core methodology.

## Limitations & Future Work

- Accurate camera poses are required—COLMAP might fail under sparse + inconsistent inputs (which can be mitigated by newer methods like MonST3R).
- Harmonization is difficult when view overlap is minimal (due to a lack of co-visible regions).
- The dynamics and illumination models are trained separately, preventing them from handling motion and lighting changes simultaneously.
- The inconsistencies generated by the video model are biased towards common scene variations, with insufficient coverage of rare dynamics.
- As work from Google DeepMind, it utilizes internal Lumiere and Gemini models, posing a barrier to exact reproduction.

## Related Work & Insights

- **vs. CAT3D**: Multi-view diffusion assumes consistent inputs; SimVS builds upon CAT3D by introducing inconsistency handling, making them complementary (furthermore, CAT4D has already adopted the data strategy proposed in this work).
- **vs. Dynamic NeRF (Nerfies/CasualSAM)**: These require dense videos + explicit motion modeling; SimVS operates on sparse inputs without explicit motion modeling.
- **vs. NeRF in the Wild**: Handles lighting changes via per-image appearance embeddings but requires dense inputs; SimVS operates generatively in sparse scenarios.
- **vs. InstructPix2Pix**: Useful for image editing, but the generated variations are minor (retaining layouts) and cannot simulate realistic dynamics.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Leveraging video models to simulate world inconsistencies for training data augmentation is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across both dynamics and illumination dimensions; the ablation study fully validates the necessity of video model augmentation, though the illumination dataset is relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ Flowing storytelling, clear problem formulation, and a natural, elegant methodology.
- Value: ⭐⭐⭐⭐⭐ Provides a practical solution for casual in-the-wild 3D reconstruction, and the concept is transferable to other robustness tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Novel View Synthesis with Pixel-Space Diffusion Models](novel_view_synthesis_with_pixel-space_diffusion_models.md)
- [\[CVPR 2025\] AerialMegaDepth: Learning Aerial-Ground Reconstruction and View Synthesis](aerialmegadepth_learning_aerial-ground_reconstruction_and_view_synthesis.md)
- [\[CVPR 2025\] DiffPortrait360: Consistent Portrait Diffusion for 360° View Synthesis](diffportrait360_consistent_portrait_diffusion_for_360_view_synthesis.md)
- [\[CVPR 2025\] MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion](zero-shot_novel_view_and_depth_synthesis_with_multi-view_geometric_diffusion.md)
- [\[CVPR 2025\] Sharp-It: A Multi-view to Multi-view Diffusion Model for 3D Synthesis and Manipulation](sharp-it_a_multi-view_to_multi-view_diffusion_model_for_3d_synthesis_and_manipul.md)

</div>

<!-- RELATED:END -->
