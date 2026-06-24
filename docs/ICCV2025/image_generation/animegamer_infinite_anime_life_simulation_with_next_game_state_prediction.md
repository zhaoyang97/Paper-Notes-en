---
title: >-
  [Paper Note] AnimeGamer: Infinite Anime Life Simulation with Next Game State Prediction
description: >-
  [ICCV 2025][Image Generation][infinite game generation] This paper proposes AnimeGamer, an infinite anime life simulation system built upon a multimodal large language model (MLLM). By predicting the next game state via action-aware multimodal representations — comprising dynamic animation shots and character state updates — the system achieves a continuously consistent interactive anime gaming experience.
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "infinite game generation"
  - "anime life simulation"
  - "multimodal large language model"
  - "video diffusion model"
  - "game state prediction"
date: 2026-05-08
content_hash: 72b6c1feabf32dac
---

# AnimeGamer: Infinite Anime Life Simulation with Next Game State Prediction

**Conference**: ICCV 2025
**arXiv**: [2504.01014](https://arxiv.org/abs/2504.01014)  
**Code**: [https://github.com/TencentARC/AnimeGamer](https://github.com/TencentARC/AnimeGamer)  
**Area**: Image/Video Generation
**Keywords**: infinite game generation, anime life simulation, multimodal large language model, video diffusion model, game state prediction

## TL;DR

This paper proposes AnimeGamer, an infinite anime life simulation system built upon a multimodal large language model (MLLM). By predicting the next game state via action-aware multimodal representations — comprising dynamic animation shots and character state updates — the system achieves a continuously consistent interactive anime gaming experience.

## Background & Motivation

Recent advances in generative AI have achieved notable progress in anime production, yet existing approaches exhibit clear limitations:

**Finite vs. Infinite Games**: Existing game generation methods (e.g., GameNGen simulating DOOM) are confined to predefined environments and limited instructions, constituting "finite games." An ideal anime life simulator should instead be an "infinite game" — with no preset boundaries, open-ended language interaction, and a continuously evolving storyline.

**Limitations of the Predecessor Work Unbounded**:
   - Relies solely on an LLM for pure-text dialogue, ignoring historical visual context, resulting in poor game coherence.
   - Generates only static images, incapable of producing the dynamic effects required to portray character actions.

**Core Challenge**: How to maintain character consistency and contextual coherence across multi-turn interactions while generating high-quality dynamic video?

## Method

### Overall Architecture

AnimeGamer is trained in three stages: (a) learning tokenization and decoding of animation shots; (b) training the MLLM to predict the next game state; (c) decoder adaptation training. During inference, a sliding window strategy enables theoretically infinite game generation.

### Key Designs

1. **Action-aware Multimodal Representation**

    - Each animation shot is decomposed into three components:
        - **Visual reference** $f_v$: CLIP embedding of the first frame of the animation clip, capturing overall appearance.
        - **Motion description** $f_{md}$: a short textual action prompt (e.g., "Softly talk") encoded by T5.
        - **Motion magnitude** $f_{ms}$: motion intensity of the character estimated via optical flow.
    - The encoder $\mathcal{E}_a$ fuses visual and textual features via MLP + LayerNorm + Concat:
    $s_a = \mathcal{E}_a(f_{md}, f_v) = \text{Concat}(\text{LN}(\text{MLP}(x)), \text{LN}(\text{MLP}(y)))$
    - **Design Motivation**: Existing methods predict only text or image representations, which are insufficient to preserve the visual and motion information inherent in video.

2. **Animation Shot Decoder $\mathcal{D}_a$**

    - Built upon the video diffusion model CogVideoX, replacing the original text features with action-aware multimodal representations.
    - Motion magnitude $f_{ms}$ is injected into the timestep embedding $f_t$ via sinusoidal embedding + FC + SiLU activation.
    - Training proceeds in two steps: the decoder is first frozen while only the encoder is trained (warm-up), followed by joint training.
    - The training objective is the standard diffusion loss:
    $\mathcal{L} = \mathbb{E}_{z,c,s_a,\epsilon \sim \mathcal{N}(0,1),t}\left[\|\epsilon - \epsilon_\theta(z_t, t, c, s_a)\|_2^2\right]$

3. **MLLM-based Game State Prediction**

    - The MLLM is initialized from Mistral-7B and serves as the "game engine."
    - Input: historical multimodal context + current player instruction.
    - Output: $N=226$ action-aware multimodal representations ($s_a$) + character state $s_c$ (stamina/social/entertainment values) + motion magnitude.
    - $s_a$ is supervised with MSE loss; $s_c$ and $f_{ms}$ are supervised with cross-entropy loss:
    $\mathcal{L} = \mathcal{L}_{CE} + \alpha \mathcal{L}_{MSE}$

4. **Decoder Adaptation**

    - Separate training of the MLLM and the decoder may lead to latent space misalignment.
    - The MLLM is frozen while only the decoder is fine-tuned to adapt to the MLLM's output embeddings.
    - During inference, a sliding window combined with a train-short-test-long strategy supports infinite generation.

### Data Construction

Approximately 20,000 video clips (16 frames, 480×720) are extracted from 10 popular anime films. InternVL is used to automatically annotate character motion, background, and character states, with support for player-defined characters.

## Key Experimental Results

### Main Results

| Model | CLIP-I↑ | DreamSim↑ | CLIP-T↑ | ACC-F↑ | MAE-F↓ | Inference Time (s/turn)↓ |
|-------|---------|-----------|---------|--------|--------|--------------------------|
| GSC | 0.786 | 0.502 | 0.333 | 0.316 | 0.826 | 50 |
| GFC | 0.766 | 0.580 | 0.333 | 0.292 | 1.021 | 63 |
| GC | 0.796 | 0.642 | 0.334 | 0.425 | 0.722 | 25 |
| **AnimeGamer** | **0.813** | **0.740** | **0.416** | **0.674** | **0.424** | **24** |

GPT-4V and human evaluation (10-point scale):

| Model | Overall (GPT/Human) | Instruction Following (GPT/Human) | Context Consistency (GPT/Human) | Character Consistency (GPT/Human) |
|-------|--------------------|------------------------------------|----------------------------------|-----------------------------------|
| GC | 6.42/7.38 | 7.29/7.37 | 6.58/6.89 | 7.49/7.55 |
| **AnimeGamer** | **8.36/10.0** | **9.14/9.95** | **8.41/9.95** | **9.11/9.86** |

### Ablation Study

| Configuration | CLIP-I↑ | DreamSim↑ | ACC-F↑ | MAE-F↓ |
|---------------|---------|-----------|--------|--------|
| w/ random frame | 0.845 | 0.450 | 0.474 | 0.562 |
| w/o warm-up | 0.831 | 0.511 | 0.703 | 0.458 |
| w/o $f_{ms}$ | 0.853 | 0.689 | 0.182 | 1.219 |
| w/o adapt | 0.683 | 0.494 | 0.365 | 0.847 |
| **Ours (full)** | **0.867** | **0.793** | **0.729** | **0.403** |

### Key Findings

- AnimeGamer outperforms all baselines across every automatic metric, with particularly strong gains in DreamSim (+15.4%) and CLIP-T (+24.6%), demonstrating the critical role of multimodal context for consistency.
- Human evaluation yields near-perfect scores (10/10), substantially surpassing methods that rely solely on textual context.
- Removing motion magnitude control causes ACC-F to collapse to 0.182, confirming that text alone is insufficient to reliably control motion intensity.
- Decoder adaptation is essential — omitting it reduces CLIP-I from 0.786 to 0.683.
- Inference efficiency is optimal at 24 s/turn, since no additional LLM API calls are required.

## Highlights & Insights

- **"MLLM as Game Engine"**: Employing the MLLM as a game engine to directly predict game states — rather than merely acting as a text router — constitutes an innovative paradigm.
- **Elegant action-aware multimodal representation design**: Decoupling visual reference, motion description, and motion magnitude enables high-quality video generation while preserving controllability.
- **End-to-end design**: The paper provides a complete technical stack, spanning data collection, model training, and evaluation benchmarks.

## Limitations & Future Work

- Training and evaluation are conducted only in the closed-domain setting (custom characters); open-domain generalization remains unexplored.
- The training data is sourced from only 10 anime films, limiting scale and diversity.
- Only short 16-frame video clips are generated per turn.
- The character state design (stamina/social/entertainment) is relatively simple and does not cover more complex game mechanics.
- Comparisons with additional game generation baselines (e.g., GameNGen, DIAMOND) are absent.

## Related Work & Insights

- The work extends the infinite game concept from Unbounded with significant improvements, upgrading from static image generation to dynamic video generation.
- The action-aware representation design is generalizable to other multimodal video-controlled generation tasks.
- The automated data collection pipeline enables rapid adaptation to any anime IP.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The first MLLM-based infinite anime life simulator; both the problem formulation and methodology are highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Automatic and human evaluations are comprehensive, ablations are thorough, but additional baselines are lacking.
- Writing Quality: ⭐⭐⭐⭐ — The framework is described clearly and supported by rich illustrations.
- Value: ⭐⭐⭐⭐ — Strong commercial potential, though substantial improvements to actual gameplay experience are still needed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] PanoLlama: Generating Endless and Coherent Panoramas with Next-Token-Prediction LLMs](panollama_generating_endless_and_coherent_panoramas_with_next-token-prediction_l.md)
- [\[CVPR 2026\] FVAR: Next-Focus Prediction for Visual Autoregressive Modeling](../../CVPR2026/image_generation/fvar_next-focus_prediction_for_visual_autoregressive_modeling.md)
- [\[ICML 2025\] Generative Audio Language Modeling with Continuous-Valued Tokens and Masked Next-Token Prediction](../../ICML2025/image_generation/generative_audio_language_modeling_with_continuous-valued_tokens_and_masked_next.md)
- [\[ICCV 2025\] AID: Adapting Image2Video Diffusion Models for Instruction-guided Video Prediction](aid_adapting_image2video_diffusion_models_for_instruction-guided_video_predictio.md)
- [\[NeurIPS 2025\] State-Covering Trajectory Stitching for Diffusion Planners](../../NeurIPS2025/image_generation/state-covering_trajectory_stitching_for_diffusion_planners.md)

</div>

<!-- RELATED:END -->
