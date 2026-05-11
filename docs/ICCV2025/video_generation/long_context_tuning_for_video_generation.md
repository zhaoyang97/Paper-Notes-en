---
title: >-
  [Paper Note] Long Context Tuning for Video Generation
description: >-
  [ICCV 2025][Video Generation][Scene-level video generation] This paper proposes Long Context Tuning (LCT), which extends the context window of pretrained single-shot video diffusion models to the scene level. By introduc…
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "Scene-level video generation"
  - "multi-shot consistency"
  - "long context tuning"
  - "asynchronous timestep"
  - "causal attention"
date: 2026-05-08
content_hash: 5d15598cc07dae47
---

# Long Context Tuning for Video Generation

**Conference**: ICCV 2025
**arXiv**: [2503.10589](https://arxiv.org/abs/2503.10589)
**Code**: None (project page available)
**Area**: Video Generation
**Keywords**: Scene-level video generation, multi-shot consistency, long context tuning, asynchronous timestep, causal attention

## TL;DR
This paper proposes Long Context Tuning (LCT), which extends the context window of pretrained single-shot video diffusion models to the scene level. By introducing interleaved 3D positional embeddings and an asynchronous noise strategy, LCT achieves cross-shot visual and temporal consistency without additional parameters, supporting both joint and autoregressive multi-shot generation, and exhibits emergent capabilities such as compositional generation.

## Background & Motivation

1. **Background**: DiT-based video generation models (Sora, Kling, HunyuanVideo, etc.) can synthesize high-quality single-shot videos lasting up to one minute. However, real narrative videos consist of multiple shots requiring cross-shot consistency.
2. **Limitations of Prior Work**: Existing scene-level generation approaches fall into two categories: (1) appearance-conditioned generation (e.g., VideoStudio), which relies on predefined conditions and specific datasets and struggles to maintain abstract elements such as lighting and color tone; (2) keyframe generation + I2V (e.g., StoryDiffusion), where shots are synthesized independently without guaranteeing temporal consistency, and sparse keyframes limit conditioning effectiveness.
3. **Key Challenge**: Scene-level consistency requires visual consistency (character identity, background, lighting, color tone) as well as temporal consistency (motion, camera movement). Both existing categories exhibit deficiencies along different consistency dimensions.
4. **Goal**: To learn cross-shot consistency directly from data without relying on predefined conditions or auxiliary networks.
5. **Key Insight**: Extend the context window of pretrained single-shot models so that full attention covers all tokens across all shots within a scene, enabling the model to learn cross-shot correlations directly from scene-level video data.
6. **Core Idea**: Distinguish shots via interleaved 3D RoPE positional embeddings, unify conditioning and diffusion samples via asynchronous timesteps, and support efficient autoregressive generation via context causal attention.

## Method

### Overall Architecture
Built upon a 3B-parameter MMDiT video diffusion model trained with Rectified Flow, with a maximum context window of 9 shots. Data includes global prompts (characters, environments, story) and per-shot prompts. Joint training on both single-shot and scene-level data is performed to preserve pretrained capabilities.

### Key Designs

1. **Interleaved 3D RoPE (Interleaved 3D Positional Embeddings)**:
    - **Function**: Distinguishes tokens from different shots while preserving intra-shot text–video alignment.
    - **Mechanism**: The relative positional relationship between text tokens and video tokens within a single shot (along the spatial diagonal) is preserved. For multiple shots, the text–video groups of each shot are appended sequentially, forming an interleaved `[text]-[video]-[text]-[video]-...` sequence. The global prompt is assigned dummy video tokens and treated as a standard text–video pair.
    - **Design Motivation**: Preserving relative positions allows each shot to inherit pretrained text–visual alignment; different absolute positions distinguish tokens and their corresponding shots. Conceptually similar to M-RoPE (Qwen2-VL), but applied to diffusion models for the first time.

2. **Asynchronous Timestep Strategy**:
    - **Function**: Unifies visual conditioning inputs and diffusion samples by independently sampling noise levels for each shot.
    - **Mechanism**: During training, each shot independently samples a diffusion timestep from a logit-normal distribution rather than using a shared timestep across all shots. When a shot has lower noise, it naturally serves as an appearance information source to guide denoising of noisier shots. During inference, timesteps can be synchronized across shots for joint generation, or selected shots can be set to low noise levels to serve as visual conditions.
    - **Design Motivation**: Eliminates the need for auxiliary networks for visual conditioning. A single model simultaneously supports joint generation, visually conditioned generation, and autoregressive generation—an exceptionally clean design.

3. **Context Causal Attention Fine-tuning**:
    - **Function**: Converts bidirectional attention to efficient causal attention, enabling KV-cache-based autoregressive generation.
    - **Mechanism**: Fine-tuned on top of the LCT bidirectional model: bidirectional attention is retained within each shot, but tokens only attend to the context of all preceding shots (causal mask). During inference, K/V features from historical shots are cached to avoid redundant computation. Only 9K fine-tuning iterations are required.
    - **Design Motivation**: Information flow in autoregressive generation is inherently directional—clean historical samples do not require information from subsequent noisy samples, making bidirectional attention redundant. Causal attention combined with KV-cache substantially reduces computational overhead.

### Loss & Training
Rectified Flow loss: $\mathcal{L} = \mathbb{E}_{t,z_0,\epsilon}\|v_\Theta(z_t, t, c_{text}) - (\epsilon - z_0)\|_2^2$. The loss is computed independently per shot and then averaged. Trained on 128 H800 GPUs for 135K iterations (LCT stage); causal attention fine-tuning requires 9K additional iterations. Training resolution is $480 \times 480$ pixels by area.

## Key Experimental Results

### Main Results

| Method | Aesthetic↑ | Quality↑ | Consistency (avg.)↑ | Text↑ | User Rank (AHR)↑ |
|--------|------|------|----------|------|------|
| VideoStudio | 61.68 | 73.13 | 95.25 | 28.00 | 2.14 |
| StoryDiffusion+Kling | 60.40 | 74.04 | 96.57 | 27.33 | 2.50 |
| IC-LoRA+Kling | 57.88 | 69.07 | 96.27 | 27.90 | 1.57 |
| **LCT (Ours)** | 60.79 | 67.44 | 95.65 | **30.14** | **3.79** |

In the user study, LCT achieves a mean rank of 3.79 (out of 4), significantly outperforming all baselines.

### Ablation Study

| Configuration | Performance | Notes |
|------|---------|------|
| Bidirectional attention | Joint + conditioned generation | Versatile but computationally expensive |
| Causal attention | Efficient autoregressive generation | Accelerated via KV-cache |
| Without interleaved RoPE | Degraded consistency | Cannot distinguish shot membership |
| Synchronized timestep | Joint generation only | Loses conditioning capability |

### Key Findings
- The text alignment score (30.14) substantially surpasses all baselines, demonstrating LCT's superior cross-shot semantic understanding.
- Emergent capabilities: compositional generation (character + environment image → video) and interactive shot extension, despite the model never being explicitly trained on these tasks.
- "Re-appearance" problem: baseline methods suffer consistency collapse when a character reappears after several intervening shots; LCT mitigates this via a history pool strategy.
- Baseline methods exhibit poor compositional diversity, whereas LCT can generate rich combinations of wide, medium, and close-up shots.

## Highlights & Insights
- Exceptionally clean and elegant design: no additional parameters, no auxiliary networks—multi-modal generation is achieved solely through positional embeddings, timestep strategy, and attention patterns.
- The asynchronous timestep is the core innovation: a single mechanism unifies joint generation, conditioned generation, and autoregressive generation into three distinct inference modes.
- Emergent capabilities are particularly impressive: compositional generation (never explicitly trained) and interactive extension demonstrate strong generalization of scene-level understanding.
- Human-selection-based generation strategy: rather than strict autoregressive generation, conditioning shots are selected from a history pool based on relevance.

## Limitations & Future Work
- Training resolution of $480 \times 480$ is relatively low.
- The context window is limited to 9 shots; longer narratives may require segmented processing.
- Video quality and aesthetic scores are slightly below some baselines, possibly due to differences in training data.
- The causal attention fine-tuning uses only 9K iterations; full-scale training could yield further improvements.

## Related Work & Insights
- **vs. VideoStudio**: VideoStudio uses entity embeddings to preserve appearance but yields limited compositional diversity; LCT learns richer consistency through full attention.
- **vs. MoviDreamer/VGoT**: Keyframe-based methods are constrained by independent I2V generation and cannot guarantee temporal consistency.
- **vs. MinT/DFoT**: Long video generation via temporally dependent prompts or history guidance, but without handling multi-shot structure.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The asynchronous timestep strategy elegantly unifies multiple generation paradigms.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Qualitative results are outstanding; automatic metrics and user studies are comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Well-structured, with the Titanic example vividly illustrating the scene concept.
- **Value**: ⭐⭐⭐⭐⭐ Represents a paradigm shift from single-shot to scene-level generation, with significant implications for video content creation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PoseGen: In-Context LoRA Finetuning for Pose-Controllable Long Human Video Generation](../../CVPR2026/video_generation/posegen_in-context_lora_finetuning_for_pose-controllable_long_human_video_genera.md)
- [\[ICCV 2025\] MagicDrive-V2: High-Resolution Long Video Generation for Autonomous Driving with Adaptive Control](magicdrive-v2_high-resolution_long_video_generation_for_autonomous_driving_with_.md)
- [\[CVPR 2026\] Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context](../../CVPR2026/video_generation/geometry-as-context_modulating_explicit_3d_in_scene-consistent_video_generation_.md)
- [\[NeurIPS 2025\] Scaling RL to Long Videos](../../NeurIPS2025/video_generation/scaling_rl_to_long_videos.md)
- [\[NeurIPS 2025\] Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation](../../NeurIPS2025/video_generation/radial_attention_onlog_n_sparse_attention_with_energy_decay_for_long_video_gener.md)

</div>

<!-- RELATED:END -->
