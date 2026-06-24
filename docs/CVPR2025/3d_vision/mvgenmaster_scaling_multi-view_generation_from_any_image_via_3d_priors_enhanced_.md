---
title: >-
  [Paper Note] MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model
description: >-
  [CVPR 2025][3D Vision][Novel View Synthesis] MVGenMaster proposes a multi-view diffusion model that integrates metric depth geometric priors. Combined with the MvD-1M dataset containing 1.6 million scenes and a training-free key-rescaling technique, it can generate up to 100 novel views from an arbitrary reference view in a single forward pass, comprehensively outperforming CAT3D and ViewCrafter on both in-distribution and out-of-distribution NVS benchmarks.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Novel View Synthesis"
  - "Multi-view Diffusion Model"
  - "3D Priors"
  - "Metric Depth"
  - "Large-scale Dataset"
date: 2026-05-08
content_hash: a7c84769a640f19b
---

# MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model

**Conference**: CVPR 2025  
**arXiv**: [2411.16157](https://arxiv.org/abs/2411.16157)  
**Code**: [https://ewrfcas.github.io/MVGenMaster](https://ewrfcas.github.io/MVGenMaster) (includes code, models, and data)  
**Area**: 3D Vision  
**Keywords**: Novel View Synthesis, Multi-view Diffusion Model, 3D Priors, Metric Depth, Large-scale Dataset

## TL;DR
MVGenMaster proposes a multi-view diffusion model that integrates metric depth geometric priors. Combined with the MvD-1M dataset containing 1.6 million scenes and a training-free key-rescaling technique, it can generate up to 100 novel views from an arbitrary reference view in a single forward pass, comprehensively outperforming CAT3D and ViewCrafter on both in-distribution and out-of-distribution NVS benchmarks.

## Background & Motivation

1. **Background**: Novel view synthesis (NVS) based on diffusion models has made significant progress. Methods like CAT3D and ViewCrafter can generate novel view images from sparse reference images. While 3D reconstruction methods (NeRF, 3DGS) require dense multi-view inputs, NVS methods aim to complete views from sparse observations.

2. **Limitations of Prior Work**: (a) **Data limitations**: Most methods rely on synthetic datasets (Objaverse), primarily focusing on object-level 3D generation, which is difficult to generalize to complex scene-level tasks; (b) **Lack of 3D priors**: Many methods rely purely on 2D generational capability, making it hard to ensure 3D consistency, especially in out-of-distribution scenes; (c) **Lack of flexibility**: They require cumbersome workflows such as anchor-based iterative generation, dataset updates, or test-time optimization, and cannot handle arbitrary reference and target views in a one-pass manner.

3. **Key Challenge**: Scaling multi-view diffusion models to a larger number of views suffers from attention dilution—when the sequence is extremely long, the guidance from reference views is diluted by a massive number of target views, causing the model to over-rely on unreliable 3D priors and leading to degraded generation quality.

4. **Goal**: How to build a multi-view diffusion model that supports flexible inputs (1-to-many reference views) and generates a large number of novel views in a single pass while ensuring 3D consistency and cross-domain generalization?

5. **Key Insight**: Introduce geometric warping based on metric depth and camera poses as an explicit 3D prior, enabling the diffusion model to both "generate" and "reconstruct". Simultaneously, construct a large-scale dataset of 1.6 million scenes to scale up training.

6. **Core Idea**: Use RGB pixels and Canonical Coordinate Maps (CCM) warped via metric depth as 3D priors to inject into the multi-view diffusion model, combined with key-rescaling to resolve attention dilution in long sequences, achieving flexible, scalable, and highly consistent multi-world generation.

## Method

### Overall Architecture
Built on StableDiffusion2. Inputs are split into reference views (image + camera pose) and target views (camera pose only). During training: Monocular depth is extracted from the reference view and aligned with SfM to obtain metric depth, which is used to warp CCMs and RGB pixels as 3D priors fed into the diffusion model. During inference: For single-view inputs, Depth-Pro is used to obtain metric depth and focal length, while DUSt3R is used for multi-view inputs. Camera poses are encoded using Plücker rays, and all reference and target view latent features are processed via 3D full attention.

### Key Designs

1. **Metric Depth 3D Priors**:

    - **Function**: Provide explicit geometric constraints and 3D structural information for multi-view generation.
    - **Mechanism**: During training, a monocular depth estimation model (e.g., DepthAnything v2) predicts depth, which is aligned with SfM sparse points via RANSAC to obtain the metric depth $\hat{D} = D \cdot r + s$. RGB pixels and CCM of the reference views are warped to the target view coordinate systems using this metric depth. Key Decision: Warp 1:1 RGB pixels rather than 1/8 latent features—although latent warping is more efficient, pixel-level warping performs better in camera zoom-in scenarios and avoids a 20% training overhead from additional VAE encoding. CCM provides precise location and occlusion information, offsetting the blurriness of RGB warping under large viewpoint changes.
    - **Design Motivation**: Pure 2D diffusion models exhibit poor 3D consistency in out-of-distribution scenes. Metric depth warping allows the model to "pre-visualize" the potential target view, reducing generation difficulty and enhancing structural consistency.

2. **Key-Rescaling View Expansion Technique**:

    - **Function**: Scale the model to ultra-long sequences (up to 158 views) without training, resolving the attention dilution issue.
    - **Mechanism**: In the 3D full self-attention module, the key features of the reference views are multiplied by a constant $\gamma$ (empirically $\gamma=1.2$), giving the reference views higher attention weights after softmax. Thus, even with a large number of target views, the aggregation operation still prioritizes guidance from the reference views. This is a training-free trick directly compatible with FlashAttention2. The authors experimentally found that without rescaling, severe degradation occurs with 25+ target views—e.g., generating multiple duplicated objects (over-relying on unreliable 3D priors).
    - **Design Motivation**: Methods like CAT3D require iterative generation to expand the number of views, which accumulates errors at each iteration. Key-rescaling enables a single forward pass to generate 100 views with superior consistency.

3. **MvD-1M Large-Scale Multi-View Dataset**:

    - **Function**: Provide diverse training data of 1.6 million scenes with metric depth.
    - **Mechanism**: Consolidates 12 data sources (Co3Dv2, MVImgNet, DL3DV, Objaverse, etc.), covering object-level, scene-level, indoor, and outdoor settings. Monocular depth estimation and SfM alignment are performed on all real-world data to obtain metric depth. A domain switcher is used to unify different data domains—by adding class embeddings to the diffusion model's timestep embeddings, with Megascenes (inconsistent lighting), Objaverse (simple background), and other datasets labeled as different classes. During inference, the class label of normal multi-view datasets is used uniformly.
    - **Design Motivation**: Existing multi-view datasets are dominated by synthetic objects and lack scene-level real-world data. The diversity of MvD-1M directly determines the model's generalization capabilities.

### Loss & Training
- Standard diffusion training loss in v-prediction mode.
- ZeroSNR and linear noise schedule are used instead of scaled linear schedule (multi-view images have a high SNR when noise is added, making training overly simple).
- qk-norm is applied to improve training stability, along with multi-scale training (from 320x768 to 768x320) and EMA (decay 0.9995).
- Trained on 16 A800 GPUs for 600k steps. The first 350k steps fixed 3 reference views, after which the learning rate was decreased and reference views were dynamically adjusted between 1 and 3.
- 3D priors and Plücker rays are subjected to 15% and 10% dropout rates, respectively, for CFG training.

## Key Experimental Results

### Main Results (NVS, 1-view→24 targets)

| Method | CO3D+MVImgNet PSNR↑ | DL3DV+Real10k PSNR↑ | Zero-shot PSNR↑ |
|------|---------------------|---------------------|-------------|
| ViewCrafter | 15.347 | 13.279 | 11.431 |
| CAT3D* (w/o 3D prior) | 17.296 | 13.650 | 10.865 |
| MVGenMaster (1-view) | 18.484 | 15.476 | 12.593 |
| MVGenMaster (3-view) | **18.964** | **16.177** | **13.718** |

3DGS Reconstruction (3-view→100 targets):

| Method | T&T PSNR↑ | DTU PSNR↑ | MipNeRF-360 PSNR↑ |
|------|-----------|-----------|-------------------|
| CAT3D* | 11.758 | 11.268 | 13.609 |
| **MVGenMaster** | **14.669** | **15.856** | **15.543** |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| w/o 3D prior (≈CAT3D*) | 15.348 | 0.479 | 0.462 |
| +qk-norm | 15.521 | 0.483 | 0.451 |
| +CCM+RGB warp (pixel) | **17.651** | **0.554** | **0.346** |
| +CCM+RGB warp (latent) | 17.521 | 0.550 | 0.345 |
| w/o CCM (RGB warp only) | 17.514 | 0.553 | 0.352 |

Dataset Scalability (Zero-shot evaluation):

| Dataset Scale | PSNR↑ | LPIPS↓ |
|-----------|-------|--------|
| 6 Datasets | 14.869 | 0.354 |
| 8 Datasets | 15.126 | 0.351 |
| 10 Datasets | 15.081 | 0.345 |
| **12 Datasets (Full)** | **15.641** | **0.326** |

### Key Findings
- **The contribution of 3D priors is the most significant**: Adding warping boosts PSNR from 15.5 to 17.7 (+2.2), accounting for the vast majority of the total improvement. Pixel-level warping slightly outperforms latent-level warping.
- **CCM is helpful but provides marginal gains**: CCM provides position and occlusion information, which is particularly useful under large viewpoint changes, but RGB warping is sufficient in most scenarios.
- **Dataset scaling consistently yields improvements**: Expanding from 6 to 12 datasets improves zero-shot PSNR from 14.869 to 15.641, with domain switcher, multi-scale training, and EMA each contributing significantly.
- **Key-rescaling is crucial for long sequences**: Generating 100 views severely degrades without it, whereas high-quality results can be stably generated with it.

## Highlights & Insights
- **The paradigm of injecting 3D priors into 2D diffusion models** is highly elegant—it doesn't alter the core generative capability of the diffusion model but rather provides geometric prompts on "what the target view roughly looks like." This "warp then inpaint" idea is transferrable to tasks like video generation and scene editing.
- **Key-rescaling is a general attention-modulation trick**—applicable to any scenario where the guidance of reference signals needs to be maintained in long sequences. It is simple to implement (one multiplication line), highly effective, and compatible with FlashAttention.
- **The MvD-1M dataset itself is a major contribution**—1.6 million scenes with metric depth across 12 domains. The domain switcher design is also worth emulating—training on data from different domains with different domain class tags and unifying labels during inference to eliminate domain gaps.

## Limitations & Future Work
- During inference, single-view and multi-view inputs use different depth estimation methods (Depth-Pro vs. DUSt3R), introducing additional inconsistencies.
- The inconsistency of monocular depth alignment during training is deemed by the authors as "beneficial regularization," which lacks strict rigor.
- No direct comparison with closed-source methods (such as the official version of CAT3D), only with their own implementation CAT3D*.
- The 1.6 million scenes are still dominated by object-level data (Objaverse accounts for 620k), and the proportion of scene-level data needs improvement.
- Generating 100 views requires 80G GPU VRAM, posing high deployment costs.

## Related Work & Insights
- **vs. CAT3D**: CAT3D also performs multi-view diffusion but lacks 3D priors and requires iterative generation to extend views. MVGenMaster achieves better results in a single forward pass via metric depth warping and key-rescaling.
- **vs. ViewCrafter**: ViewCrafter generates novel views based on video diffusion models, but is constrained by sequential generation and limited frame counts. MVGenMaster supports unordered viewpoints and a larger number of target views.
- **vs. ReconX**: ReconX also utilizes DUSt3R 3D priors to assist video generation, but its method is more limited. MVGenMaster's metric depth warping is more direct and effective.

## Rating
- Novelty: ⭐⭐⭐⭐ Metric depth warping as a 3D prior and key-rescaling are both practical and novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 training domains, comprehensive in-distribution and out-of-distribution evaluations, and thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Globally coherent, though containing many details that require careful digestion.
- Value: ⭐⭐⭐⭐⭐ The MvD-1M dataset and the entire methodology establish a strong baseline for the NVS domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Sharp-It: A Multi-view to Multi-view Diffusion Model for 3D Synthesis and Manipulation](sharp-it_a_multi-view_to_multi-view_diffusion_model_for_3d_synthesis_and_manipul.md)
- [\[CVPR 2025\] MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation](midi_multi-instance_diffusion_for_single_image_to_3d_scene_generation.md)
- [\[CVPR 2025\] PCDreamer: Point Cloud Completion Through Multi-view Diffusion Priors](pcdreamer_point_cloud_completion_through_multi-view_diffusion_priors.md)
- [\[CVPR 2025\] MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion](zero-shot_novel_view_and_depth_synthesis_with_multi-view_geometric_diffusion.md)
- [\[CVPR 2025\] Ouroboros3D: Image-to-3D Generation via 3D-aware Recursive Diffusion](ouroboros3d_image-to-3d_generation_via_3d-aware_recursive_diffusion.md)

</div>

<!-- RELATED:END -->
