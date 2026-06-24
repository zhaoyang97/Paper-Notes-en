---
title: >-
  [Paper Note] FoundHand: Large-Scale Domain-Specific Learning for Controllable Hand Image Generation
description: >-
  [CVPR 2025][Image Generation][Hand Image Generation] This paper proposes FoundHand, a domain-specific diffusion model trained on a tens-of-millions-scale hand dataset (FoundHand-10M). By employing 2D keypoint heatmaps as a universal control representation, FoundHand achieves precise control over hand poses/viewpoints and preserves identity appearance, demonstrating zero-shot emergent capabilities such as correcting deformed hands, video generation…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Hand Image Generation"
  - "Domain-Specific Diffusion Models"
  - "2D Keypoint Control"
  - "Hand Gesture Transfer"
  - "Novel View Synthesis"
date: 2026-05-08
content_hash: 235d8ca4de6082f9
---

# FoundHand: Large-Scale Domain-Specific Learning for Controllable Hand Image Generation

**Conference**: CVPR 2025  
**arXiv**: [2412.02690](https://arxiv.org/abs/2412.02690)  
**Code**: [https://ivl.cs.brown.edu/research/foundhand.html](https://ivl.cs.brown.edu/research/foundhand.html)  
**Area**: Diffusion Models / Image Generation / 3D Vision  
**Keywords**: Hand Image Generation, Domain-Specific Diffusion Models, 2D Keypoint Control, Hand Gesture Transfer, Novel View Synthesis

## TL;DR
This paper proposes FoundHand, a domain-specific diffusion model trained on a tens-of-millions-scale hand dataset (FoundHand-10M). By employing 2D keypoint heatmaps as a universal control representation, FoundHand achieves precise control over hand poses/viewpoints and preserves identity appearance, demonstrating zero-shot emergent capabilities such as correcting deformed hands, video generation, and hand-object interaction videos.

## Background & Motivation

**Background**: General text-to-image (T2I) models (e.g., Stable Diffusion, Midjourney) have made significant progress in generating human faces and bodies, yet they still perform poorly on hand generation—frequently producing extra fingers, deformed joints, and incorrect articulations. Existing hand generation methods either rely on the 3D MANO model as a condition (which is unreliable and costly to obtain) or are designed specifically for individual sub-tasks, lacking generality.

**Limitations of Prior Work**: (1) General datasets suffer from a severe under-representation of hand samples, which are either too small in pixel area or lack complex articulation diversity; (2) 3D hand representations (e.g., MANO meshes) are poorly estimated in in-the-wild scenarios, causing error cascading when used as conditioning; (3) Existing hand generation methods (e.g., HandRefiner, CosHand) only target a single task (inpainting/interaction) and cannot natively unify diverse needs like pose transfer, view synthesis, and video generation.

**Key Challenge**: The lack of a combined "large-scale hand dataset + reliable and scalable control representation". Insufficient data leads to poor generalization, while unreliable 3D representations lead to imprecise control.

**Goal**: To build a foundational model for general hand image generation that simultaneously addresses data scale, control precision, and multi-task applicability.

**Key Insight**: 2D keypoints are a natural, lightweight representation that encodes articulated poses and camera viewpoints, and can be reliably extracted at scale using MediaPipe. By replacing the high computational overhead of video models with an image-to-image training paradigm, the model learns pose and viewpoint transformations on video frame pairs and multi-view frame pairs.

**Core Idea**: Utilizing 2D keypoints as a universal hand representation to train an image-to-image diffusion model on a dataset of ten million diverse hand images, establishing precise and controllable hand generation.

## Method

### Overall Architecture
FoundHand is based on a latent DiT architecture. During training, a reference frame and a target frame are sampled from video sequences or multi-view images. The VAE-encoded image latent, a 42-channel 2D keypoint Gaussian heatmap, and a hand segmentation mask are spatially aligned, encoded via a shared-weight embedder, and fed into a 3D self-attention transformer to generate the target frame. At inference time, the model iteratively denoises from pure noise, conditioned on the reference image and target keypoints.

### Key Designs

1. **FoundHand-10M Dataset**:

    - Function: Provides large-scale, diverse, and uniformly annotated hand training data.
    - Mechanism: Integrates 12 existing hand datasets (DexYCB, ARCTIC, InterHand, Ego4D, HAGRID, etc.), covering single/both hands, hand-object interactions, sign language, and first/third-person perspectives. MediaPipe is used for uniform extraction of 2D keypoints, and SAM for segmentation masks. It totals 10 million images.
    - Design Motivation: Existing individual hand datasets are too small or limited in scenarios to support foundational model training. Standardizing annotations with 2D keypoints resolves incompatible annotation formats across different datasets.

2. **Multimodal Spatially Aligned Input**:

    - Function: Enables the model to learn the spatial relationships between hand appearance, silhouettes, and joint structures.
    - Mechanism: The VAE latent (image features), 42-channel keypoint heatmap (one channel per keypoint, eliminating occlusion ambiguity), and binary hand mask are concatenated and aligned along the spatial dimension, then mapped to unified patch tokens via a shared embedder. The reference and target frame tokens interact through 3D self-attention.
    - Design Motivation: Unlike OpenPose RGB skeletons (which are ambiguous during finger occlusion), 42 independent heatmaps eliminate ambiguity completely. Hand masks provide silhouette priors to help the model distinguish the hand from the background. The two-frame 3D self-attention offers manageable computational cost compared to multi-frame video models.

3. **Training Strategy and Conditional Dropout**:

    - Function: Enhances robustness and multi-task generalization.
    - Mechanism: (1) Data augmentation includes random gamma correction, hand left-right swapping, horizontal flipping (to learn chirality transformation), and cropping; (2) Conditional dropout randomly masks out all reference frame tokens or target keypoints, enabling the model to learn both conditional and marginal distributions; (3) A binary flag $y$ distinguishes between pose transformation (video sequence frame pairs) and viewpoint transformation (synchronized multi-view frame pairs); (4) REPA alignment—aligning intermediate features with DINOv2 self-supervised representations.
    - Design Motivation: Conditional dropout makes reference keypoints and masks optional during inference, significantly increasing practical flexibility. REPA alignment accelerates convergence and improves generation quality.

### Loss & Training
Standard diffusion model noise prediction loss: $\mathcal{L} = \mathbb{E}[\|\epsilon_\theta(z_\tau; \tau, c) - \epsilon_\tau\|_2^2]$. At inference time, Classifier-Free Guidance (CFG) is used to balance quality and control accuracy. The backbone is initialized from an ImageNet-pretrained DiT.

## Key Experimental Results

### Main Results (Pose Transfer - Identity Consistency)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ |
|------|-------|-------|--------|------|
| GestureGAN | 11.18 | 0.43 | 0.52 | 12.90 |
| Uni-ControlNet | 9.41 | 0.32 | 0.48 | 11.01 |
| AnyControl | 10.59 | 0.42 | 0.40 | 7.46 |
| CosHand | 26.21 | 0.75 | 0.22 | 3.60 |
| **FoundHand** | **30.96** | **0.82** | **0.20** | **2.58** |

### Ablation Study / Multi-task Comparison

| Task | Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|------|-------|-------|--------|
| Novel View Synthesis | ZeroNVS | 19.21 | 0.74 | 0.24 |
| Novel View Synthesis | ImageDream | 19.97 | 0.80 | 0.17 |
| **Novel View Synthesis** | **FoundHand** | **27.72** | **0.88** | **0.10** |
| Video Generation | ControlNeXt | 17.64 | 0.73 | 0.29 |
| Video Generation | AnimateAnyone | 15.76 | 0.74 | 0.35 |
| **Video Generation** | **FoundHand** | **24.08** | **0.83** | **0.17** |

### Key Findings
- In pose transfer, FoundHand's PSNR is 4.75dB higher than the strongest competitor CosHand, and the FID is reduced by 28%.
- In novel view synthesis, it outperforms methods relying on 3D representations (NeRF + SDS) by 39% PSNR—indicating that 2D priors learn 3D consistency.
- For zero-shot video generation, it surpasses specialized video diffusion models (ControlNeXt/AnimateAnyone), demonstrating the effectiveness of the image-pair training paradigm.
- In hand-object interaction videos, it displays a physical understanding of both rigid motion (cup movement) and non-rigid deformation (sponge squeezing), which emerges without any explicit target object supervision.

## Highlights & Insights
- **2D keypoints as a unified representation insight**: 2D keypoints naturally encode articulation + viewpoint information. This simple observation brings immense practical value—ease of acquisition, cross-dataset unification, and avoidance of 3D reconstruction errors. A similar idea can be scaled to whole-body or animal pose control.
- **Emergent hand-object interaction understanding**: Despite only seeing hand motion training data, during inference, the model correctly predicts the movement and deformation of objects manipulated by the hand—indicating that large-scale training allows the model to "understand" the causal relationships between hands and objects.
- **Stochastic Conditioning Strategy**: Randomly selecting reference frames (including previously generated frames) during NVS and video generation balances long-term consistency and short-term coherence, serving as a clever and practical engineering trick.

## Limitations & Future Work
- Input and output resolutions are limited to 256×256, constrained by the 32×32 latent resolution, which limits practical utility.
- FoundHand-10M is primarily sourced from laboratory settings and specific scenarios, potentially lacking in-the-wild diversity.
- Video generation is autoregressive frame-by-frame, which might accumulate errors over long sequences.
- Future improvements: incorporating super-resolution modules to support high-resolution outputs; exploring joint control via text conditions; scaling up to whole-body generation.

## Related Work & Insights
- **vs CosHand**: CosHand requires a precise target hand mask as input, which is challenging in practice; FoundHand only requires 2D keypoints, offering higher flexibility. Additionally, FoundHand is more robust in generating interaction videos, avoiding CosHand's random object hallucinations.
- **vs HandRefiner / RealisHuman**: These methods rely on off-the-shelf 3D hand reconstruction models, which are inherently unreliable for deformed hands. FoundHand directly corrects and paints hands using 2D keypoints, leading to greater stability.
- **vs Video Diffusion Models**: FoundHand performs zero-shot video generation and outperforms video models despite not being trained on continuous video objectives, showing that domain-specific high-quality image models paired with clever inference strategies can replace naive video expansions.

## Rating
- Novelty: ⭐⭐⭐⭐ The dataset scale and the choice of 2D keypoint representations are the core contributions; the model architecture is built upon existing DiT.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprising 6 downstream tasks with comprehensive quantitative and qualitative evaluations.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear and task demonstrations are diverse, though some descriptions are slightly verbose.
- Value: ⭐⭐⭐⭐⭐ A landmark work in hand generation; both the dataset and the model provide high value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Learning Flow Fields in Attention for Controllable Person Image Generation](learning_flow_fields_in_attention_for_controllable_person_image_generation.md)
- [\[CVPR 2025\] InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation](interact_advancing_large-scale_versatile_3d_human-object_interaction_generation.md)
- [\[CVPR 2025\] BootComp: Controllable Human Image Generation with Personalized Multi-Garments](controllable_human_image_generation_with_personalized_multi-garments.md)
- [\[CVPR 2025\] CTRL-O: Language-Controllable Object-Centric Visual Representation Learning](ctrl-o_language-controllable_object-centric_visual_representation_learning.md)
- [\[CVPR 2025\] ZoomLDM: Latent Diffusion Model for Multi-Scale Image Generation](zoomldm_latent_diffusion_model_for_multi-scale_image_generation.md)

</div>

<!-- RELATED:END -->
