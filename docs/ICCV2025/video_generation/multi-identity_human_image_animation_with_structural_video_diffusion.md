---
title: >-
  [Paper Note] Multi-identity Human Image Animation with Structural Video Diffusion
description: >-
  [ICCV 2025][Video Generation][Multi-identity human animation] This paper proposes the Structural Video Diffusion framework, which maintains multi-person appearance consistency via mask-guided identity-specific embeddings…
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "Multi-identity human animation"
  - "video diffusion model"
  - "identity embedding"
  - "joint depth-normal learning"
  - "human interaction"
date: 2026-05-08
content_hash: e5854f3538f06d5c
---

# Multi-identity Human Image Animation with Structural Video Diffusion

**Conference**: ICCV 2025
**arXiv**: [2504.04126](https://arxiv.org/abs/2504.04126)  
**Code**: [GitHub](https://github.com/AvatarAnything/StructuralVideoDiffusion)  
**Area**: Video Generation
**Keywords**: Multi-identity human animation, video diffusion model, identity embedding, joint depth-normal learning, human interaction

## TL;DR
This paper proposes the Structural Video Diffusion framework, which maintains multi-person appearance consistency via mask-guided identity-specific embeddings, jointly learns RGB/depth/normal tri-modal geometric structure to model human–object interactions, and introduces the Multi-HumanVid dataset of 25K multi-person interaction videos to enable multi-identity human video generation.

## Background & Motivation

1. **Background**: Human image animation, represented by Animate Anyone, MagicAnimate, and CamAnimate, employs pose-guided diffusion models to generate high-fidelity human videos from single images.
2. **Limitations of Prior Work**: Existing methods focus on single-person animation and perform poorly in multi-identity scenarios (multi-person interaction, human–object interaction)—they cannot associate correct appearance–pose pairs and lack modeling of 3D spatial relationships. Identity confusion arises during two-person dancing, and held objects become blurry, floating, or disappear.
3. **Key Challenge**: (1) Multi-person scenes require trackable identity-specific features to maintain appearance consistency, yet existing frameworks lack an identity-discrimination mechanism; (2) pose guidance alone is insufficient to model the 3D spatial relationships in human–object interactions.
4. **Goal**: How to preserve each person's appearance consistency in multi-person scenes? How to model complex 3D human–object interactions?
5. **Key Insight**: Designing identity embeddings inspired by DETR's learnable query paradigm; jointly predicting RGB + depth + normal as implicit 3D structural supervision.
6. **Core Idea**: Mask-guided identity embeddings associate each person's appearance with their pose, while joint RGB/depth/normal learning enables the model to understand 3D spatial interactions.

## Method

### Overall Architecture
The inputs consist of a reference image $C$ containing $N$ persons, per-frame identity masks $M^f$, 2D skeletons $P^f$, and camera parameters $R^f$. The video diffusion model is built upon Stable Diffusion 1.5 and AnimateDiff, comprising a Denoising UNet and a Reference UNet.

### Key Designs

1. **ID-Specific Embedding Learning**:
    - **Function**: Maintains cross-frame appearance consistency for each person in the scene.
    - **Mechanism**: Introduces $N$ learnable ID embeddings $E_{query} \in \mathbb{R}^{N \times C}$. For each frame $f$, the mask $M^f$ is converted into a spatial ID embedding map $E^f \in \mathbb{R}^{H \times W \times C}$ by copying the $n$-th embedding to all spatial positions where $M^f(h,w)=n$. The final $E \in \mathbb{R}^{F \times H \times W \times C}$ is added to the noisy latent via zero-initialized convolution (ControlNet style): $\widetilde{x}_t = x_t + \text{zero\_conv}(E)$.
    - **Design Motivation**: Zero initialization ensures that multi-person training initializes equivalently to the single-person model. SAM2-generated masks provide reliable identity tracking. Even when persons swap positions, embeddings correctly associate each identity. The framework flexibly supports up to $N$ identities, ignoring unused embeddings when fewer are present.

2. **Latent Structural Video Diffusion**:
    - **Function**: Captures 3D structural information by jointly predicting RGB, depth, and normal modalities.
    - **Mechanism**: Rather than using depth/normal as inputs (impractical to obtain per frame), they are treated as **output modalities** predicted jointly with RGB. Both the Denoising UNet and Reference UNet duplicate the conv_in/conv_out layers and the first/last DownBlock/UpBlock into three modality-specific paths, while intermediate layers are shared. During training, DepthCrafter provides depth supervision and Sapiens provides normal supervision. The three modalities share the same timestep $t$ with independent noise and a joint loss: $\mathcal{L} = \|v_{rgb} - \hat{v}_{rgb}\|^2 + \|v_{depth} - \hat{v}_{depth}\|^2 + \|v_{normal} - \hat{v}_{normal}\|^2$.
    - **Design Motivation**: Depth provides occlusion and relative distance cues; normals preserve object/clothing shape. Joint denoising enables the model to learn appearance–geometry coupled dynamics, improving human–object interaction quality. No explicit object-level conditioning is required; spatial relationships between objects and persons are inferred through depth and normals.

3. **Multi-HumanVid Dataset**:
    - **Function**: Provides large-scale multi-person interaction training data.
    - **Mechanism**: The Pexels API is queried with interaction keywords (e.g., *party*); videos are filtered via 2D pose detection (upper-body confidence > 0.5, subject ratio > 0.07, person count ≤ 5), yielding 25K new videos that extend the total to 45K. The annotation pipeline proceeds as: Grounding-DINO localizes persons → SAM2 tracks masks → TRAM estimates cameras → DepthCrafter and Sapiens provide depth/normal annotations.
    - **Design Motivation**: The existing HumanVid dataset contains only 20K interaction-free videos; training for multi-person interaction scenarios requires dedicated data. The fully automated annotation pipeline is highly scalable.

### Loss & Training
Two-stage training: Stage 1 trains all network parameters (Denoising UNet + ReferenceNet + Pose Guider), with batch size determined by the number of modalities. Stage 2 freezes Stage 1 parameters and trains only the camera encoder and motion module. Training runs for 40K + 20K iterations on 8 A100 GPUs.

## Key Experimental Results

### Main Results

| Method | SSIM↑ | PSNR↑ | LPIPS↓ | FVD↓ | FID↓ |
|--------|------|------|----------|------|------|
| MimicMotion | 0.628 | 19.878 | 0.258 | 1042.6 | 59.11 |
| CamAnimate | 0.649 | 19.552 | 0.265 | 982.1 | 54.09 |
| **Ours** | **0.691** | **20.685** | **0.233** | **878.2** | **30.57** |

User study: the proposed method achieves 91.25% preference over CamAnimate.

### Ablation Study

| Configuration | SSIM↑ | PSNR↑ | LPIPS↓ | FVD↓ | FID↓ |
|------|------|------|------|------|------|
| Baseline (CamAnimate) | 0.649 | 19.552 | 0.265 | 982.1 | 54.09 |
| + ID-embedding | 0.686 | 20.374 | 0.237 | 873.5 | 33.75 |
| + Multi-modality | 0.668 | 20.139 | 0.240 | 907.8 | 47.67 |
| + Both | **0.691** | **20.685** | **0.233** | **878.2** | **30.57** |

Modality ablation: adding depth alone yields the best results (+Depth: FID 30.57), while adding normals degrades performance (+Normal: FID 60.58).

### Key Findings
- ID embeddings and multi-modal structural learning are each individually effective, with their combination yielding further gains—demonstrating complementarity between the two components.
- Depth contributes far more than normals: DepthCrafter (fine-tuned from a video model) offers substantially better temporal consistency than the frame-wise normals from Sapiens.
- Normals are only effective within human body regions; incomplete supervision limits their contribution.
- Cross-identity motion transfer: motion templates from one video can be transferred to edited characters while preserving appearance consistency.

## Highlights & Insights
- The ID embedding design is simple yet effective: zero initialization ensures backward compatibility, and mask guidance naturally associates each identity with its spatial location.
- The "predict rather than input" strategy for geometric information avoids the impractical requirement of per-frame depth/normal inputs at inference time.
- The multi-modal branch design, which shares the backbone, achieves cross-modal correlation learning with minimal parameter overhead.
- The fully automated annotation pipeline for Multi-HumanVid is highly scalable.

## Limitations & Future Work
- The SD 1.5 backbone limits visual quality and motion stability.
- Insufficient normal estimation quality introduces noise; a temporally consistent normal estimator is needed.
- The method has not been implemented on large-scale DiT models (e.g., HunyuanVideo, CogVideoX).
- No explicit object-level conditioning is included, limiting performance on complex physical interactions.

## Related Work & Insights
- **vs. Animate Anyone / MagicAnimate**: These methods support only single-person animation, resulting in appearance confusion in multi-person scenarios.
- **vs. CamAnimate**: Camera control is added but no multi-identity mechanism is present.
- **vs. Champ**: Uses SMPL + rendered depth/normal maps as input conditions, but is limited to single persons and requires explicit geometric inputs.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of multi-identity embeddings and tri-modal joint learning is the first to address multi-person interaction animation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Quantitative + qualitative + user study + comprehensive ablation.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear; method description is detailed.
- **Value**: ⭐⭐⭐⭐ Pioneers the direction of multi-identity human video generation; the dataset has lasting value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] OmniHuman-1: Rethinking the Scaling-Up of One-Stage Conditioned Human Animation Models](omnihuman-1_rethinking_the_scaling-up_of_one-stage_conditioned_human_animation_m.md)
- [\[CVPR 2026\] Vanast: Virtual Try-On with Human Image Animation via Synthetic Triplet Supervision](../../CVPR2026/video_generation/vanast_virtual_try-on_with_human_image_animation_via_synthetic_triplet_supervisi.md)
- [\[ICCV 2025\] Versatile Transition Generation with Image-to-Video Diffusion](versatile_transition_generation_with_image-to-video_diffusion.md)
- [\[ICCV 2025\] Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis](adversarial_distribution_matching_for_diffusion_distillation_towards_efficient_image_and_video_synthesis.md)
- [\[ICCV 2025\] DualReal: Adaptive Joint Training for Lossless Identity-Motion Fusion in Video Customization](dualreal_adaptive_joint_training_for_lossless_identity-motion_fusion_in_video_cu.md)

</div>

<!-- RELATED:END -->
