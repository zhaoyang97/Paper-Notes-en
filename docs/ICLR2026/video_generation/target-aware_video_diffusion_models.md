---
title: >-
  [Paper Note] Target-Aware Video Diffusion Models
description: >-
  [ICLR 2026][Video Generation][Video Diffusion Models] This paper proposes a target-aware video diffusion model that generates videos of an actor interacting with a specified target object…
tags:
  - "ICLR 2026"
  - "Video Generation"
  - "Video Diffusion Models"
  - "Target-Aware"
  - "Cross-Attention Loss"
  - "Human Interaction"
  - "Action Planning"
date: 2026-05-08
content_hash: f2c0a5c4d2538e21
---

# Target-Aware Video Diffusion Models

**Conference**: ICLR 2026
**arXiv**: [2503.18950](https://arxiv.org/abs/2503.18950)  
**Code**: [taeksuu.github.io/tavid](https://taeksuu.github.io/tavid/)  
**Area**: Video Generation / Human-Object Interaction
**Keywords**: Video Diffusion Models, Target-Aware, Cross-Attention Loss, Human Interaction, Action Planning

## TL;DR
This paper proposes a target-aware video diffusion model that generates videos of an actor interacting with a specified target object, given only a single input image and a segmentation mask of the target. The core innovations are the introduction of a special [TGT] token and a selective cross-attention loss that guides the model to attend to the spatial location of the target, achieving comprehensive improvements over baselines in both target alignment and video quality.

## Background & Motivation
Video diffusion models have demonstrated remarkable capabilities in simulating complex scenes, but precise control over content and actions is required for practical applications. Existing controllable video generation methods typically rely on dense structural or motion cues (depth maps, edge maps, optical flow, drag trajectories, etc.) to guide actor motion. While effective for simple translations or viewpoint changes, these approaches face fundamental challenges in **actor–target interaction scenarios**, where providing structural motion guidance—such as how to reach for a cup on a table—is inherently difficult.

A further motivation is to leverage video diffusion models as **high-level action planners**. Rather than treating video models as "renderers" (requiring dense motion inputs), this work positions them as "planners" capable of generating plausible interaction actions given only a target location. This has significant implications for downstream applications such as robotic manipulation.

The core idea is to mark a target object with a single segmentation mask and allow the generative prior of the video diffusion model to autonomously infer reasonable interaction actions for the actor.

## Method

### Overall Architecture
**Inputs**: a single image $I$, a segmentation mask $M$ of the target object, and a text prompt describing the desired action. **Output**: a video in which the actor accurately interacts with the target specified by the mask. The method is built upon CogVideoX-5B-I2V and fine-tuned via LoRA.

### Key Designs

1. **Mask Condition Injection**: The binary segmentation mask $M$ is downsampled and concatenated with the input image as an additional channel fed into the diffusion model. The input channels of the image projection layer are extended accordingly, with newly added weights initialized to zero to preserve pretrained parameters. This allows the model to perceive the spatial location of the target, though this alone is insufficient to guarantee target awareness—the model may still ignore the mask information.

2. **[TGT] Token and Cross-Attention Loss**: This is the core contribution of the paper.

    - The phrase "The person interacts with [TGT] object." is appended to the text prompt, introducing a special token [TGT] to encode the spatial information of the target.
    - A cross-attention loss is designed to align the cross-attention map of the [TGT] token with the input mask:
     $$\mathcal{L}_{\text{attn}} = \mathbb{E}[\|A(\mathbf{z}_t^0, [\text{TGT}]) - M\|_2^2]$$
     where $A(\mathbf{z}_t^0, [\text{TGT}])$ denotes the cross-attention weights between the first-frame video latent and the [TGT] token.
    - The total training objective is: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + \lambda_{\text{attn}} \mathcal{L}_{\text{attn}}$ (with $\lambda_{\text{attn}} = 0.1$).
    - During inference, [TGT] is prepended to the word referring to the target in the text, enabling the model to leverage the spatial cues provided by the mask.

3. **Selective Cross-Attention Loss**: Rather than applying the loss indiscriminately across all transformer blocks and attention regions, the paper applies it selectively:

    - **Selective Transformer Blocks**: Empirical evaluation identifies blocks 5–23 (out of 42) as having cross-attention maps most semantically aligned with the segmentation mask. At each training step, 7 blocks are sampled from this range.
    - **Selective Attention Regions**: The MM-DiT architecture contains four attention types (text-to-text, T2V, V2T, video-to-video). **V2T (video-to-text) cross-attention** directly influences the values of video latent representations and yields the best results. T2V also encodes semantic information but has an indirect effect.

4. **Dataset Construction**: A total of 1,290 video clips are extracted from the BEHAVE (simple human-object interactions) and Ego-Exo4D (complex scenarios such as cooking and car repair) datasets. Each clip satisfies: (1) the actor is present but not yet interacting with the target in the initial frame, and (2) the actor engages with the target in subsequent frames. Target masks are obtained using SAM, and text captions are generated using CogVLM2.

### Training Details
- Built on CogVideoX-5B-I2V; LoRA rank=128, α=64.
- Only LoRA layers and the image projection layer are trained; all other parameters are frozen.
- Trained for 2,000 steps with AdamW, lr=1e-4, batch size=4.
- 4× NVIDIA A100 GPUs; approximately 6 hours of training.
- Inference: DPM sampler, 50 steps, CFG=6; approximately 4 minutes per video on a single A100.

## Key Experimental Results

### Main Results — Target Alignment and Video Quality

| Method | Hum. Eval. ↑ | User Pref. ↑ | Contact Score ↑ | SS | BC | DD | MS |
|--------|-------------|-------------|----------------|----|----|----|----|
| CogVideoX | 0.592 | 0.456 | 28.4% | 0.914 | 0.903 | 0.950 | 0.988 |
| CogVideoX w. data | 0.692 | 0.596 | 36.2% | 0.915 | 0.900 | 0.956 | 0.990 |
| Attn. Mod. | 0.613 | 0.508 | 22.2% | 0.878 | 0.887 | 0.827 | 0.984 |
| **Ours** | **0.896** | **0.892** | **Highest** | **0.938** | **0.914** | 0.956 | 0.905 |

### Ablation Study

| Configuration | Contact Score | Notes |
|---------------|--------------|-------|
| $\lambda_{\text{attn}} = 0.0$ (no attention loss) | 0.688 | ≈ CogVideoX w. data; confirms attention loss is critical |
| $\lambda_{\text{attn}} = 0.01$ | 0.756 | Improvement but insufficient |
| $\lambda_{\text{attn}} = 0.1$ (ours) | **0.896** | Optimal balance |
| $\lambda_{\text{attn}} = 1.0$ | 0.904 | Marginally higher contact score but degraded video quality |
| Random block selection | 0.840 | Inferior to semantic selection |
| Uniform block selection | 0.839 | Inferior to semantic selection |
| T2V Cross-Attn. | 0.784 | Inferior to V2T |
| V2T Cross-Attn. (ours) | **0.896** | V2T directly influences video latent representations |

### Key Findings
- The cross-attention loss is critical for achieving target awareness: performance at $\lambda=0$ nearly equals the data-only fine-tuning baseline.
- V2T cross-attention is the correct application site, as it directly influences video latent representations through value dot products.
- Semantic block selection (every 3rd block from blocks 5–23) yields the best results.
- The advantage of mask-based control is especially pronounced when multiple objects of the same category are present in the scene, as text cannot disambiguate them while masks can.
- The model generalizes to non-human subjects (e.g., animals).

## Highlights & Insights
- **Minimizing control input while maximizing generative prior**: Using only a single segmentation mask—without dense trajectories or multi-frame guidance—the model autonomously infers plausible interaction actions, fully leveraging the generative capacity of video diffusion models.
- **Elegant [TGT] token design**: Spatial information is carried via a text token in a principled manner without any architectural modifications; only an additional training loss is introduced.
- **In-depth analysis of selective loss**: The paper systematically characterizes the semantic properties of different blocks and attention regions in MM-DiT, yielding a principled design rather than empirical black-box tuning.
- **Two compelling downstream applications**: Video content creation (navigation + interaction composition) and zero-shot 3D HOI motion synthesis demonstrate the potential of the model as an action planner.

## Limitations & Future Work
- Video quality is bounded by the underlying open-source model (CogVideoX); closed-source commercial models may yield better results.
- The training data is captured with static cameras, leading the model to favor fixed viewpoints during generation.
- The dataset contains only 1,290 clips; scaling up data volume may further improve generalization.
- Currently only a single target mask is supported; multi-target simultaneous interaction (with preliminary exploration of [SRC]+[TGT]) remains to be fully addressed.
- Generated motions are plausible but may lack physical precision in areas such as contact mechanics.
- 3D pose and scene scale are not fully aligned in the physical simulation experiments.

## Related Work & Insights
- **Distinction from ControlNet-style methods**: ControlNet requires dense per-frame conditions (depth/edge maps), making it suitable for precise control of simple motions; this paper uses a single-frame mask, better suited to HOI scenarios.
- **Comparison with DragDiffusion**: Drag-based methods fail under large motion and cannot generate complex interactions.
- **Distinction from Direct-a-Video**: Attention modulation methods require no training but perform poorly—in MM-DiT, the row-normalization of softmax causes amplified cross-attention values to corrupt self-attention, leading to temporal inconsistencies.
- **Broader insight**: Video diffusion models inherently encode rich priors over physical world interactions; the key challenge lies in releasing these priors with minimal signals (a single mask).

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models](frame_guidance_training-free_guidance_for_frame-level_control_in_video_diffusion.md)
- [\[CVPR 2026\] TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models](../../CVPR2026/video_generation/tear_temporal-aware_automated_red-teaming_for_text-to-video_models.md)
- [\[ICLR 2026\] Geometry-aware 4D Video Generation for Robot Manipulation](geometry-aware_4d_video_generation_for_robot_manipulation.md)
- [\[CVPR 2026\] Goal-Driven Reward by Video Diffusion Models for Reinforcement Learning](../../CVPR2026/video_generation/goal-driven_reward_by_video_diffusion_models_for_reinforcement_learning.md)
- [\[CVPR 2026\] Diff4Splat: Repurposing Video Diffusion Models for Dynamic Scene Generation](../../CVPR2026/video_generation/diff4splat_controllable_4d_scene_generation_with_latent_dynamic_reconstruction_m.md)

</div>

<!-- RELATED:END -->
