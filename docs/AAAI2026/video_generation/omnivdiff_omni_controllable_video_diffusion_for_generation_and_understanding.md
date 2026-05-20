---
title: >-
  [Paper Note] OmniVDiff: Omni Controllable Video Diffusion for Generation and Understanding
description: >-
  [AAAI 2026][Video Generation][Video diffusion models] This paper proposes OmniVDiff, a unified controllable video diffusion framework that jointly models multiple visual modalities (RGB, depth, segmentation…
tags:
  - "AAAI 2026"
  - "Video Generation"
  - "Video diffusion models"
  - "multimodal generation"
  - "controllable video generation"
  - "video understanding"
  - "unified framework"
date: 2026-05-08
content_hash: 1a87da7a27218406
---

# OmniVDiff: Omni Controllable Video Diffusion for Generation and Understanding

**Conference**: AAAI 2026
**arXiv**: [2504.10825](https://arxiv.org/abs/2504.10825)  
**Code**: [https://tele-ai.github.io/OmniVDiff/](https://tele-ai.github.io/OmniVDiff/) (project page available)  
**Area**: Video Generation
**Keywords**: Video diffusion models, multimodal generation, controllable video generation, video understanding, unified framework

## TL;DR
This paper proposes OmniVDiff, a unified controllable video diffusion framework that jointly models multiple visual modalities (RGB, depth, segmentation, Canny) in color space and introduces an Adaptive Modality Control Strategy (AMCS). Within a single diffusion model, OmniVDiff simultaneously supports three task types—text-conditioned generation, X-conditioned generation, and video understanding—achieving state-of-the-art performance on VBench.

## Background & Motivation

Video diffusion models have achieved remarkable progress in text-to-video generation, yet **controllable video generation** faces two core bottlenecks:

**Task-specific fine-tuning**: Introducing each new control signal (depth, segmentation, Canny, etc.) requires dedicated fine-tuning of large-scale diffusion architectures, incurring prohibitive computational costs and poor scalability.

**Reliance on external expert models**: Most methods depend on standalone expert models (depth estimators, segmentation models, etc.) to extract conditioning signals before passing them to a separate diffusion model, forming a multi-step, non-end-to-end pipeline.

Limitations of prior work:
- **VideoJAM**: Jointly models only RGB and optical flow; does not support conditional generation or understanding.
- **UDPDiff**: Supports joint generation of RGB + depth or RGB + segmentation, but cannot synthesize all modalities simultaneously.
- **Aether**: Unifies RGB + depth + camera pose, but primarily targets geometric world modeling.

**Core Idea**: Treat all visual modalities as parallel signals in color space, concatenate them along the channel dimension, and feed them into a unified diffusion Transformer. By dynamically assigning each modality's role (generation vs. conditioning), the framework flexibly supports diverse downstream tasks.

## Method

### Overall Architecture

OmniVDiff is built upon the pretrained CogVideoX text-to-video model and comprises three core components:
1. **Multimodal video encoding**: A shared 3D-VAE encoder encodes each of the four modalities—RGB, depth, segmentation, and Canny—into latent space independently.
2. **OmniVDiff diffusion network**: Noisy latent representations of all modalities are concatenated along the channel dimension and jointly denoised by the diffusion Transformer.
3. **Multimodal video decoding**: Modality-Specific Projection Heads (MSPH) disentangle the denoised output into per-modality representations, which are then reconstructed by the 3D-VAE decoder.

### Key Designs

1. **Multimodal Video Diffusion Architecture**:

    - **Function**: Extends CogVideoX's input space to accommodate multiple modalities and equips the output with independent projection heads per modality.
    - **Mechanism**: Each modality is encoded into a latent representation $x_m$ via 3D-VAE, mixed with noise, and concatenated to form the unified input $x_i = \text{Concat}(x_r^t, x_d^t, x_s^t, x_c^t)$.
    - **Design Motivation**: Projection heads are replicated rather than shared, as different modalities (depth vs. segmentation vs. Canny) exhibit fundamentally distinct distributional characteristics.

2. **Adaptive Modality Control Strategy (AMCS)**:

    - **Function**: Dynamically determines whether each modality acts as a "generation modality" or a "conditioning modality."
    - **Mechanism**: Generation modalities are mixed with noise before input; conditioning modalities are concatenated directly as clean signals. Learnable modality embeddings $e_g / e_c$ further differentiate the roles.
    - **Design Motivation**: Eliminates the need for separate fine-tuning per conditioning task; a single unified architecture can flexibly switch between tasks.

3. **Two-Stage Training Strategy**:

    - **Function**: Stage one learns joint multimodal video generation; stage two incorporates conditional generation and video understanding tasks.
    - **Mechanism**: Each stage runs for 20K steps with independent denoising losses; conditioning modalities are excluded from loss computation.
    - **Design Motivation**: Progressive training avoids instability that would arise from introducing all tasks simultaneously.

### Loss & Training

Multimodal diffusion loss:
$$\mathcal{L} = \sum_{m, m \notin Cond} \mathbb{E}_{x_m, t, \epsilon, m} [\|\epsilon - \epsilon_\theta(x_m^{t,'}, t, e_m)\|^2]$$

- Denoising loss is computed independently for each generation modality.
- Conditioning modalities are excluded from loss computation and serve only as guidance.
- Training data: 400K videos sampled from Koala-36M; pseudo-labels generated using Video Depth Anything and Semantic-SAM + SAM2.

## Key Experimental Results

### Main Results

**Text-conditioned video generation (VBench)**:

| Method | subject cons. | b.g. cons. | motion smooth. | dynamic deg. | weighted avg. |
|--------|--------------|-----------|----------------|-------------|--------------|
| CogVideoX | 95.68 | 96.00 | 98.21 | 53.98 | 72.25 |
| **OmniVDiff** | **97.78** | **96.26** | **99.21** | 49.69 | **72.78** |

**Depth-conditioned video generation (VBench)**:

| Method | subject cons. | dynamic deg. | weighted avg. |
|--------|--------------|-------------|--------------|
| Make-your-video | 90.04 | 51.95 | 70.17 |
| VideoX-Fun | 96.25 | 50.43 | 72.85 |
| **OmniVDiff** | **97.96** | **53.32** | **73.45** |

**Zero-shot video depth estimation (ScanNet)**:

| Method | AbsRel ↓ | δ1 ↑ |
|--------|----------|------|
| DepthCrafter | 0.169 | 0.730 |
| VDA-S (teacher) | 0.110 | 0.876 |
| OmniVDiff | 0.125 | 0.852 |
| OmniVDiff-Syn | **0.100** | **0.894** |

### Ablation Study

| Configuration | subject cons. | dynamic deg. | weighted avg. | Note |
|---------------|--------------|-------------|--------------|------|
| w/o modality embedding | 97.11 | 41.80 | 71.54 | No role differentiation across modalities |
| w/o AMCS | 97.31 | 33.28 | 71.21 | No adaptive control; dynamic degree drops sharply |
| w/o MSPH | 96.76 | 41.41 | 71.35 | Shared projection head; modality features entangled |
| **Full OmniVDiff** | **97.78** | **49.69** | **72.78** | All components working in concert |

### Key Findings

- Trained solely on pseudo-labels, OmniVDiff achieves video depth estimation performance approaching or even surpassing the expert teacher model VDA-S.
- With a small amount (10K) of high-quality synthetic data, OmniVDiff-Syn outperforms the teacher model on AbsRel (0.100 vs. 0.110).
- AMCS has the largest impact on dynamic degree (removal causes a drop from 49.69 to 33.28), demonstrating that adaptive control is critical for motion dynamics modeling.
- In terms of inference efficiency, OmniVDiff adds only 11.8M parameters and 3 seconds of latency while simultaneously outputting RGB + depth + segmentation + Canny.

## Highlights & Insights

- **Elegant unified framework design**: Three concise designs—channel concatenation, modality embeddings, and adaptive control—unify generation and understanding within a single model.
- **Effectiveness of pseudo-label training**: Demonstrates that pseudo-labels generated by expert models are sufficient to train a unified model that approaches or even surpasses the experts themselves.
- **Flexible task adaptability**: New modalities or tasks (e.g., super-resolution) can be adapted with only 2K fine-tuning steps, showcasing strong extensibility.
- **Elimination of external expert dependencies**: The end-to-end pipeline reduces the complexity and inconsistency of multi-model deployment.

## Limitations & Future Work

- Dynamic degree in text-conditioned generation is slightly lower than CogVideoX (49.69 vs. 53.98), suggesting that joint multimodal training may mildly compromise motion dynamics.
- Currently limited to four modalities; extension to additional modalities (optical flow, surface normals, semantic labels, etc.) remains to be validated.
- Segmentation quality depends on Semantic-SAM and SAM2 pseudo-labels; annotation noise may constrain the performance ceiling.
- Validation on higher resolutions or longer videos has not been conducted.

## Related Work & Insights

- The choice of CogVideoX as the base model is critical; its spatiotemporal compression capability in the 3D-VAE makes multimodal concatenation feasible.
- Analogous approaches in the image domain (OneDiff, UniReal) unify tasks via a "multi-view" formulation; the proposed channel concatenation approach elegantly avoids the token explosion problem in video settings.
- The design of AMCS is generalizable to other multimodal generation tasks (e.g., audio-visual generation).

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MotionCharacter: Fine-Grained Motion Controllable Human Video Generation](motioncharacter_fine-grained_motion_controllable_human_video_generation.md)
- [\[CVPR 2026\] LAMP: Language-Assisted Motion Planning for Controllable Video Generation](../../CVPR2026/video_generation/lamp_language-assisted_motion_planning_for_controllable_video_generation.md)
- [\[CVPR 2026\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](../../CVPR2026/video_generation/flashmotion_fewstep_controllable_video_generation.md)
- [\[CVPR 2026\] PoseGen: In-Context LoRA Finetuning for Pose-Controllable Long Human Video Generation](../../CVPR2026/video_generation/posegen_in-context_lora_finetuning_for_pose-controllable_long_human_video_genera.md)
- [\[CVPR 2026\] AutoCut: End-to-end Advertisement Video Editing Based on Multimodal Discretization and Controllable Generation](../../CVPR2026/video_generation/autocut_end-to-end_advertisement_video_editing_based_on_multimodal_discretizatio.md)

</div>

<!-- RELATED:END -->
