---
title: >-
  [Paper Note] GameFactory: Creating New Games with Generative Interactive Videos
description: >-
  [ICCV 2025][Image Generation][game generation] This paper proposes GameFactory, a multi-stage training strategy that **decouples game style from action control** on top of a pretrained video diffusion model…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "game generation"
  - "video diffusion models"
  - "action control"
  - "scene generalization"
  - "autoregressive video generation"
  - "Minecraft"
  - "world model"
date: 2026-05-08
content_hash: bd8889cc7fe8f7e7
---

# GameFactory: Creating New Games with Generative Interactive Videos

**Conference**: ICCV 2025
**Code**: [https://yujiwen.github.io/gamefactory/](https://yujiwen.github.io/gamefactory/) (project page)  
**Area**: Image Generation
**Keywords**: game generation, video diffusion models, action control, scene generalization, autoregressive video generation, Minecraft, world model

## TL;DR

This paper proposes GameFactory, a multi-stage training strategy that **decouples game style from action control** on top of a pretrained video diffusion model, enabling action control learned from small-scale Minecraft data to **generalize to arbitrary open-domain scenes** for interactive game video generation. This is the first method with a complete technical paper that validates scene generalization over a complex action space (7 keys + mouse).

## Background & Motivation

Video generation models are emerging as strong candidates for generative game engines, yet current approaches share three fundamental limitations:

**Game-specific binding**: DIAMOND (Atari/CS:GO), GameNGen (DOOM), and Oasis (Minecraft) can only generate content within the specific games on which they were trained.

**Lack of scene generalization**: None can create content beyond existing games, which limits their utility as engines for "creating new games."

**Impractical action annotation at scale**: Annotating open-domain video with action labels is prohibitively expensive.

**Core Insight**: The internet provides vast quantities of open-domain video, and pretrained video generation models already encode rich scene generation priors. Transferring action control learned from small-scale game data to arbitrary scenes could enable the creation of entirely new games.

**Key Challenge**: Directly fine-tuning a pretrained model on game data causes **domain collapse** — outputs inherit Minecraft's pixelated aesthetic and the model loses open-domain generation capability. Style and action control are **entangled**.

## Method

### GF-Minecraft Dataset

The dataset is designed around three key requirements:

1. **Unbiased action distribution**: Keyboard and mouse inputs are decomposed into atomic actions with uniform distribution. Compared to the VPT dataset, the W key accounts for 50.11% vs. 13.56% (uniform) here, while the S key accounts for only 0.32% vs. 13.56%.
2. **Diverse scenes**: 70 hours of gameplay video captured across different scenes, weather conditions, and times of day.
3. **Text annotations**: MiniCPM is used to generate textual descriptions for video segments.

### Action Control Module

An action control module is injected into each block of the video diffusion Transformer:

**Mouse control (continuous signal)**: Uses concatenation — grouped mouse actions $\mathbf{M}_{repeat} \in \mathbb{R}^{(n+1) \times l \times rwd_1}$ are concatenated with features $\mathbf{F}$ along the channel dimension, followed by an MLP and temporal self-attention.

**Keyboard control (discrete signal)**: Uses cross-attention — keyboard action embeddings are learned, and the grouped action embedding $\mathbf{K}_{group}$ serves as key/value while features $\mathbf{F}$ serve as query.

**Key Design — Sliding Window Grouping**: Due to the temporal compression ratio $r=4$, the number of actions ($rn$) does not match the number of feature tokens ($n+1$). A sliding window of size $w$ is introduced to align them while capturing delayed action effects (e.g., a jump command influencing multiple frames).

### Autoregressive Long Video Generation

The core idea is borrowed from Diffusion Forcing: different frames are allowed to have different noise levels.

**Training**: Among $N+1$ frames, $k+1$ frames are randomly selected as noise-free conditioning frames; the noise prediction loss is computed only on the remaining $N-k$ frames.

**Inference**: The most recent $k+1$ frames are used as conditioning to generate $N-k$ new frames; this is repeated to produce arbitrarily long videos.

**Training loss**:
$$\mathcal{L}_{\mathbf{a}}(\phi) = \mathbb{E}[||\boldsymbol{\epsilon}_\phi(\mathbf{Z}_t, \mathbf{p}, \mathbf{A}, t) - \boldsymbol{\epsilon}||_2^2]$$

Key finding: computing the loss only on predicted frames (rather than all frames) yields significantly better performance (Flow: 85.45 vs. 148.73).

### Style–Action Decoupled Training (Core Contribution)

**Multi-stage training strategy**:

- **Phase #0**: Pretrained video generation model (open-domain data).
- **Phase #1**: Fine-tune with LoRA (rank=128, lr=1e-4) to adapt to Minecraft style — LoRA absorbs game-specific visual style.
- **Phase #2**: Freeze pretrained parameters and LoRA; train only the action control module (lr=1e-5) — since Phase #1 has already handled style adaptation via LoRA, the training loss in this phase is dominated by action control.
- **Phase #3**: At inference, **remove the LoRA weights** and retain only the action control module → action control generalizes to open-domain scenes.

**Decoupling Principle**: Phase #1 lets LoRA learn game style; Phase #2 lets the control module learn action control. Because the two are carried by separate parameters, removing LoRA at inference preserves action control capability while eliminating the style constraint.

## Key Experimental Results

### Main Results: Action Control Mechanism Ablation

| Keyboard Control | Mouse Control | Only-Key Cam↓ | Only-Key Flow↓ | Mouse-Small Cam↓ | Mouse-Large Flow↓ |
|---|---|---|---|---|---|
| Cross-Attn | Cross-Attn | 0.0527 | 8.67 | 0.0798 | 325.18 |
| Concat | Concat | 0.0853 | 22.37 | 0.0756 | 258.93 |
| **Cross-Attn** | **Concat** | **0.0439** | **7.79** | **0.0685** | **249.54** |

Optimal combination: **cross-attention for keyboard, concatenation for mouse**. Discrete signals suit similarity-based cross-attention; continuous signals suit concatenation, which preserves magnitude information.

### Scene Generalization Comparison

| Strategy | Domain | Cam↓ | Flow↓ | Dom↑ | CLIP↑ | FID↓ | FVD↓ |
|---|---|---|---|---|---|---|---|
| Multi-Phase | In-domain | 0.0839 | 43.48 | — | — | — | — |
| **Multi-Phase** | **Open-domain** | **0.0997** | **54.13** | **0.7565** | **0.3181** | **121.18** | **1256.94** |
| One-Phase | Open-domain | 0.1134 | 76.02 | 0.7345 | 0.3111 | 167.79 | 1323.58 |

The multi-stage strategy significantly outperforms the single-stage baseline across all metrics, particularly in:
- Action following (Flow: 54.13 vs. 76.02)
- Domain consistency (Dom: 0.7565 vs. 0.7345)
- Generation quality (FID: 121.18 vs. 167.79)

### Dataset Comparison (GF-Minecraft vs. VPT)

| Dataset | Cam↓ | Flow↓ | CLIP↑ | FID↓ | FVD↓ |
|---|---|---|---|---|---|
| VPT (biased) | 0.1324 | 107.67 | 0.3174 | 156.69 | 1233.15 |
| **GF-Minecraft (unbiased)** | **0.0839** | **43.48** | **0.3135** | **125.85** | **1047.59** |

The unbiased dataset yields a substantial advantage in action following (Flow: 43.48 vs. 107.67), validating the importance of removing human behavioral bias.

### Ablation Study: Long Video Generation

| Loss Scope | Cam↓ | Flow↓ | CLIP↑ | FID↓ | FVD↓ |
|---|---|---|---|---|---|
| All frames | 0.1547 | 148.73 | 0.2965 | 176.07 | 1592.43 |
| **Predicted frames only** | **0.0924** | **85.45** | **0.3190** | **136.95** | **1154.45** |

### Key Findings

1. The model successfully generates long videos exceeding **300 frames**, covering a complex action space including forward/backward/left/right movement, jumping, acceleration/deceleration, and mouse movement.
2. Action control learned from Minecraft generalizes to entirely different scenes such as deserts, forests, cities, and interiors.
3. Single-stage training leads to "style leakage" — generated open-domain videos exhibit visible Minecraft visual artifacts.

## Highlights & Insights

1. **Elegant style–action decoupling**: Using LoRA to capture style, an independent module to capture action control, and removing LoRA at inference — the approach is conceptually simple yet highly effective.
2. **Importance of unbiased datasets**: The extreme skew in VPT data (W key at 50%, S key at only 0.32%) prevents the model from executing "backward" movement — data bias directly translates into capability deficiency.
3. **Paradigm shift from "simulating existing games" to "creating new games"**: This is the first serious attempt to generalize game action control from a specific domain to open-domain scenes.
4. **Engineering value of autoregressive design**: Generating multiple frames per step (rather than frame-by-frame) substantially reduces long video generation time.

## Limitations & Future Work

1. **Proprietary base model**: The 11B internal text-to-video model used is not publicly available, making reproduction impossible.
2. **Limited resolution**: 360×640 remains far below commercial game quality standards.
3. **Absence of unified benchmark comparisons**: Fair quantitative comparison is infeasible given that different methods use different game sources, resolutions, and control granularities.
4. **Physical plausibility not evaluated**: Whether generated videos adhere to reasonable physical rules (e.g., collision, gravity) is not validated.
5. **Latency and frame rate not reported**: Real-time gaming requires 30+ FPS, yet inference efficiency is not discussed.
6. **Complex interactions absent**: Object manipulation, inventory management, and other core game mechanics are not supported.

## Related Work & Insights

- **Genie 2**: Achieves control generalization through large-scale action-annotated data — a data-scaling approach; GameFactory takes a pretrained-prior + small-data transfer approach. The two are complementary.
- **Diffusion Forcing**: GameFactory's autoregressive generation draws on this work's idea of assigning different noise levels to different frames.
- **Novel use of LoRA**: Typically used for adaptation, LoRA is here used for "isolation" — absorbing specific style during training and being removed at inference. This paradigm is generalizable to other domain transfer scenarios.
- **Implications for the game industry**: If scene generalization and physical plausibility are further improved, generative game engines could constitute an entirely new form of gaming — one in which players instantly create and explore any imagined game world.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐⭐ — The multi-stage style–action decoupled training strategy is the core highlight; the scene generalization direction is forward-looking.
- **Experimental Thoroughness**: ⭐⭐⭐ — Ablations are sufficient, but quantitative comparison with comparable methods is lacking and the base model is not public.
- **Value**: ⭐⭐⭐ — Proof-of-concept stage; substantial gaps remain before practical game applications.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear and method description is detailed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SDMatte: Grafting Diffusion Models for Interactive Matting](sdmatte_grafting_diffusion_models_for_interactive_matting.md)
- [\[ICLR 2026\] Market Games for Generative Models: Equilibria, Welfare, and Strategic Entry](../../ICLR2026/image_generation/market_games_for_generative_models_equilibria_welfare_and_strategic_entry.md)
- [\[ICCV 2025\] StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation](streamdiffusion_a_pipeline-level_solution_for_real-time_interactive_generation.md)
- [\[ICCV 2025\] EC-Flow: Enabling Versatile Robotic Manipulation from Action-Unlabeled Videos via Equivariant Flow Matching](ec-flow_enabling_versatile_robotic_manipulation_from_action-unlabeled_videos_via.md)
- [\[ICCV 2025\] MotionDiff: Training-Free Zero-Shot Interactive Motion Editing via Flow-Assisted Multi-View Diffusion](motiondiff_training-free_zero-shot_interactive_motion_editing_via_flow-assisted_.md)

</div>

<!-- RELATED:END -->
