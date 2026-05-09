---
title: >-
  [Paper Note] Gloria: Consistent Character Video Generation via Content Anchors
description: >-
  [CVPR 2026][Video Generation][character video generation] Gloria introduces a compact set of "Content Anchors" to represent a character's multi-view appearance and expression identity. Through two key mechanisms—superset content anchoring (to prevent copy-paste artifacts) and RoPE weak conditioning (to distinguish multiple anchor frames)—the method enables consistent character video generation exceeding 10 minutes in duration.
tags:
  - CVPR 2026
  - Video Generation
  - character video generation
  - consistency
  - content anchor frames
  - diffusion models
  - long video
date: 2026-05-08
content_hash: 874efaed7da597d0
---

# Gloria: Consistent Character Video Generation via Content Anchors

**Conference**: CVPR 2026
**arXiv**: [2603.29931](https://arxiv.org/abs/2603.29931)
**Code**: [https://yyvhang.github.io/Gloria_Page/](https://yyvhang.github.io/Gloria_Page/)
**Area**: Video Understanding / Video Generation
**Keywords**: character video generation, consistency, content anchor frames, diffusion models, long video

## TL;DR

Gloria introduces a compact set of "Content Anchors" to represent a character's multi-view appearance and expression identity. Through two key mechanisms—superset content anchoring (to prevent copy-paste artifacts) and RoPE weak conditioning (to distinguish multiple anchor frames)—the method enables consistent character video generation exceeding 10 minutes in duration.

## Background & Motivation

Digital character video generation faces a triple challenge: long-duration consistency, multi-view appearance consistency, and expression identity consistency. Existing methods rely on a single reference image or text prompt, which carry insufficient character information to sustain long-term consistency. Some approaches introduce pre-selected or generated frames as "memory," but these frames are typically not character-centric and lack semantic grounding.

**Core Insight**: Character video generation is fundamentally an "appearance look-up" scenario—the visual attributes of a character can be compactly represented by a structured set of anchor frames, while motion is learned from short video clips.

**Technical Challenges**: (1) How to inject anchor frames without inducing trivial copy-paste behavior; (2) How to use multiple anchor frames simultaneously without conflict; (3) How to efficiently extract anchor frames from large-scale video data.

## Method

### Overall Architecture

Anchor frame extraction pipeline (offline) → unified content anchor injection mechanism (anchor frame tokens concatenated with video tokens for self-attention) → superset content anchoring + RoPE weak conditioning training → inference supporting text/image/audio inputs.

### Key Designs

1. **Superset Content Anchoring**:

    - Function: Prevents the model from trivially copying anchor frame content.
    - Mechanism: During training, each video clip is provided with "superset" anchor frames—including both intra-clip frames (from within the clip) and extra-clip frames (from outside the clip). This compels the model to adaptively extract useful information from multiple potentially relevant anchor frames rather than directly copying the most similar one.
    - Design Motivation: If anchor frames always closely correspond to the target during training, the model takes the shortcut of direct copying. The superset introduces redundancy, requiring the model to genuinely understand the semantic content of anchor frames.

2. **RoPE as Weak Condition**:

    - Function: Distinguishes multiple simultaneously injected anchor frames to avoid conflicts.
    - Mechanism: Different anchor frames are shifted to different positional ranges within RoPE, enabling the model to reliably differentiate them. This is a "weak" condition—it does not enforce strict one-to-one correspondence but provides positional disambiguation cues. Combined with mixed-ratio training (varying numbers of anchor frames), the model learns to adapt flexibly.
    - Design Motivation: When multiple anchor frames are directly concatenated into a sequence, the model cannot distinguish one from another. RoPE provides the least intrusive means of differentiation.

3. **Automated Anchor Frame Extraction Pipeline**:

    - Function: Efficiently extracts viewpoint and expression anchor frames from large-scale video data.
    - Mechanism: Viewpoint anchor frames are obtained by analyzing the character's orientation relative to the camera to determine viewpoint categories. Expression anchor frames are extracted via emotion recognition to detect distinct expressions, then refined by an MLLM. The entire pipeline is automated and scalable to large video datasets.
    - Design Motivation: Manual anchor frame selection is not scalable; an automated pipeline is a prerequisite for practical deployment.

### Loss & Training

Standard diffusion training loss (denoising objective), fine-tuned on top of a video diffusion model. Mixed-ratio training randomly selects 0–N anchor frames as conditions.

## Key Experimental Results

### Main Results

| Method | Max Duration | Multi-view Consistency | Expression Consistency | Identity Preservation |
|--------|-------------|----------------------|----------------------|----------------------|
| WanS2V/FramePack | ~1 min | Moderate | Moderate | Moderate |
| Gloria | **10+ min** | **Excellent** | **Excellent** | **Excellent** |

Gloria generates character videos exceeding 10 minutes, surpassing existing methods in multi-view appearance consistency and expression identity consistency.

### Ablation Study

| Configuration | Consistency | Copy-Paste Issue | Note |
|--------------|------------|-----------------|------|
| w/o superset anchoring | Poor | Severe | Directly copies most similar anchor frame |
| w/o RoPE weak condition | Moderate | Moderate | Multiple anchor frames cause confusion |
| Full Gloria | Best | None | Both mechanisms work synergistically |

### Key Findings

- Superset anchoring is critical for preventing copy-paste behavior—without it, the model degrades to nearest-neighbor retrieval and direct copying.
- RoPE weak conditioning outperforms strong conditioning (e.g., separate cross-attention heads) in positional disambiguation; strong conditioning limits flexibility.
- The automated anchor frame extraction pipeline makes large-scale training data construction feasible.

## Highlights & Insights

- **Anchor Frames as Character "Identity Cards"**: A small set of representative frames captures all visual attributes of a character—more interpretable than embedding vectors and more compact than full videos.
- **Superset Anchoring Prevents Shortcut Learning**: By providing redundant and imperfectly aligned conditions, the model is forced to develop semantic-level understanding rather than pixel-level copying.
- **10-Minute Long Video Generation**: A significant duration breakthrough in current character video generation research.

## Limitations & Future Work

- The limited number of anchor frames may be insufficient for highly complex clothing details (e.g., intricate pattern variations).
- The current approach primarily targets single-character scenarios; multi-character settings remain underexplored.
- Audio-driven lip synchronization quality is constrained by the underlying model.
- Future work may explore 3D-aware anchor frame representations.

## Related Work & Insights

- **vs. WanS2V (FramePack)**: WanS2V aggregates multiple frames but lacks a structured character representation; Gloria introduces semantically explicit anchor frame concepts.
- **vs. Animate Anyone / MagicAnimate**: These methods rely on a single reference image, which carries insufficient information to maintain long-term consistency.
- **vs. ConsisID / UniAnimate**: These methods focus on short-term consistency; Gloria achieves long-term consistency at the 10-minute scale.

## Rating

- Novelty: ⭐⭐⭐⭐ The content anchor frame concept and superset anchoring mechanism are creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Qualitative results are rich, but quantitative evaluation could be more comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Concepts are clearly articulated.
- Value: ⭐⭐⭐⭐⭐ Directly applicable to the digital human / virtual character industry.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] First Frame Is the Place to Go for Video Content Customization](first_frame_is_the_place_to_go_for_video_content_customization.md)
- [\[CVPR 2026\] Towards Realistic and Consistent Orbital Video Generation via 3D Foundation Priors](orbital_video_3d_foundation_priors.md)
- [\[CVPR 2026\] Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context](geometry-as-context_modulating_explicit_3d_in_scene-consistent_video_generation_.md)
- [\[ICLR 2026\] BindWeave: Subject-Consistent Video Generation via Cross-Modal Integration](../../ICLR2026/video_generation/bindweave_subject-consistent_video_generation_via_cross-modal_integration.md)
- [\[AAAI 2026\] FilmWeaver: Weaving Consistent Multi-Shot Videos with Cache-Guided Autoregressive Diffusion](../../AAAI2026/video_generation/filmweaver_weaving_consistent_multi-shot_videos_with_cache-guided_autoregressive.md)

<!-- RELATED:END -->
