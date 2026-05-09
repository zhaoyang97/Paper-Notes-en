---
title: >-
  [Paper Note] Customized Visual Storytelling with Unified Multimodal LLMs
description: >-
  [CVPR 2026][Multimodal VLM][Visual story generation] This paper proposes the VstoryGen framework and its core component CustFilmer, which leverages a unified multimodal large language model (UMLLM) to enable customized multimodal story generation with joint conditioning on text descriptions, character/scene reference images, and shot types. Two new benchmarks, MSB and M2SB, are also introduced.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Visual story generation
  - multimodal customization
  - unified multimodal LLM
  - shot type control
  - keyframe generation
date: 2026-05-08
content_hash: a7456a443baea87d
---

# Customized Visual Storytelling with Unified Multimodal LLMs

**Conference**: CVPR 2026
**arXiv**: [2603.27690](https://arxiv.org/abs/2603.27690)
**Code**: None (not explicitly provided on project page)
**Area**: Multimodal VLM / Visual Storytelling Generation
**Keywords**: Visual story generation, multimodal customization, unified multimodal LLM, shot type control, keyframe generation

## TL;DR
This paper proposes the VstoryGen framework and its core component CustFilmer, which leverages a unified multimodal large language model (UMLLM) to enable customized multimodal story generation with joint conditioning on text descriptions, character/scene reference images, and shot types. Two new benchmarks, MSB and M2SB, are also introduced.

## Background & Motivation

**Background**: Text-to-video generation has advanced rapidly, yet generating long-sequence coherent narrative videos remains challenging. Existing visual story generation methods (ConsiStory, StoryDiffusion, CharaConsist) primarily rely on text-only inputs, with only a few supporting character identity preservation.

**Limitations of Prior Work**: (1) Existing methods use only text inputs and cannot leverage reference images for character and scene customization; (2) background consistency is often overlooked in favor of foreground characters; (3) generated viewpoints are monotonous, lacking cinematic shot language (wide/medium/close-up, etc.); (4) multi-character interaction scene generation remains insufficient.

**Key Challenge**: How to achieve flexible multimodal conditional control (text + reference images + shot types) while maintaining character and scene consistency?

**Goal**: Leverage the multimodal understanding and generation capabilities of UMLLMs to construct a visual storytelling pipeline supporting rich multimodal conditions.

**Key Insight**: Extend the image editing capability of UMLLMs into keywise autoregressive story generation, enhancing consistency and cinematic quality through structured retrieval and shot-type prompt tuning.

**Core Idea**: UMLLM + structured multimodal script + visual reference memory bank + shot-type prompt tuning = customizable visual storytelling.

## Method

### Overall Architecture
VstoryGen three-stage pipeline:
1. **Multimodal Script Generation**: GPT-4o generates structured scripts (text prompts + character/background reference images + shot types) from free-form text descriptions.
2. **CustFilmer Keyframe Generation**: Consistent keyframes are generated based on the script.
3. **TI2V Video Extension**: Existing text-and-image-to-video models extend keyframes into video clips.

### Key Designs

1. **Text Prompt Consolidation (TPC)**:

    - **Function**: All prompts $P = \{p_1, \ldots, p_n\}$ from the same story are jointly encoded in a single batch, and the LLM autoregressively generates hidden states $H = \{h_1, \ldots, h_n\}$.
    - **Mechanism**: The contextual consistency of the LLM is exploited — descriptions of different events encoded within the same context window naturally maintain semantic and identity coherence across their hidden states.
    - **Design Motivation**: Compared to encoding each prompt independently, joint encoding keeps text conditions for different frames consistent in the embedding space, thereby preserving character and scene coherence during generation.

2. **Visual Reference Memory Bank and Retrieval**:

    - **Function**: Stores initial reference images (character portraits, background scenes) and previously generated keyframes, organized as a structured key-value dictionary. At each timestep $t$, relevant visual references are precisely retrieved using character/background mentions in the script as queries.
    - **Mechanism**: $z_t = \text{VAE}[\mathcal{R}_t, \{\text{Scale}_\alpha(I_{t-i})\}_{i=1}^\mu]$
    - **Design Motivation**: Rather than embedding-based retrieval (which may be ambiguous), structured script annotations ensure precise and interpretable reference selection. Retrieving the most recent $\mu$ frames provides temporal coherence. The $\alpha$ parameter balances consistency and diversity.

3. **Shot-type Prompt Tuning**:

    - **Function**: A set of shot-type embeddings $E_{\text{shot}}(k_t) \in \mathbb{R}^{d \times N}$ is learned on the Condensed Movie Dataset (CMD) and prepended to the hidden states as a prefix: $h_t' = [E_{\text{shot}}(k_t); h_t]$
    - **Mechanism**: Parameter-efficient prompt tuning that learns only shot-relevant embeddings without modifying the base model.
    - **Design Motivation**: General-purpose UMLLMs lack compositional priors for cinematic shot language. A small number of learnable parameters (4,000 iterations) inject shot-type knowledge, enabling diverse viewpoints such as wide shots, medium shots, and close-ups.

4. **Keyframe-wise Autoregressive Generation**:

    - **Function**: Extends standard UMLLM single-pass image editing to keyframe-level autoregressive generation: $I_t = \text{DiT}(h_t, z_t)$
    - **Design Motivation**: Avoids multi-turn dialogue (low efficiency and error accumulation); low-level visual information is preserved by injecting VAE-encoded reference images directly into the DiT decoder.

### Loss & Training
- Shot-type prompt tuning: trained for 4,000 iterations on CMD movie data.
- OmniGen2 is used as the backbone UMLLM during inference.
- $\alpha=0.75$, $d=2048$, $N=30$

## Key Experimental Results

### Main Results — MSB Benchmark (Consistency Metrics)

| Method | Backbone | CLIP-I-fg (Inter)↑ | CLIP-I-bg (Inter)↑ | Avg Consistency↑ |
|------|---------|-------------------|-------------------|------------------|
| IP-Adapter | SDXL | 0.901 | 0.936 | 0.846 |
| ConsiStory | SDXL | 0.868 | 0.884 | 0.812 |
| StoryDiffusion | SDXL | 0.857 | 0.900 | 0.831 |
| CharaConsist | Flux.1 | 0.904 | 0.945 | 0.852 |
| **CustFilmer** | **OmniGen2** | **0.905** | **0.961** | **0.858** |

Text alignment and quality metrics:

| Method | CLIP-T↑ | IAS↑ | IQS↑ | STA (Shot)↑ |
|------|--------|------|------|-----------|
| ConsiStory | **0.303** | 0.431 | 0.385 | 0.406 |
| CharaConsist | 0.265 | 0.448 | 0.415 | 0.247 |
| **CustFilmer** | 0.285 | **0.450** | **0.423** | **0.418** |

### Ablation Study

| Configuration | Avg-Consistency↑ | Note |
|------|-----------------|------|
| w/o TPC + w/o Retrieval | 0.854 | Baseline |
| + TPC | 0.855 | Marginal gain |
| + Retrieval | 0.856 | Marginal gain |
| + TPC + Retrieval | **0.858** | Complementary |

$\alpha$ parameter ablation:

| $\alpha$ | CLIP-T↑ | Avg-Consistency↑ | Note |
|---------|--------|-----------------|------|
| 0.125 | **0.289** | 0.850 | High diversity but inconsistent |
| 0.75 | 0.285 | 0.858 | Balanced choice |
| 1.00 | 0.284 | **0.860** | Most consistent but less diverse |

### Key Findings
- CustFilmer achieves the best overall consistency, particularly excelling in background consistency (CLIP-I-bg), significantly outperforming all baselines.
- Shot type control accuracy (STA=0.418) substantially surpasses non-customized methods.
- The slightly lower CLIP-T compared to ConsiStory is attributed to different backbone models (SDXL naturally benefits from training with a CLIP encoder).
- $\alpha=0.75$ achieves the best balance between consistency and diversity.

## Highlights & Insights
- **Complete multimodal storytelling pipeline**: End-to-end usable, from free-form text descriptions → structured scripts → keyframes → video.
- **Shot type control**: The first work to introduce cinematic shot language into visual story generation, significantly enhancing narrative expressiveness.
- **New benchmark contributions**: MSB and M2SB fill the evaluation gap for multimodal story customization.
- **UMLLM-based paradigm**: Leverages the unified understanding and generation capabilities of UMLLMs, representing a new paradigm for story generation.

## Limitations & Future Work
- Relies on GPT-4o for script generation (cost and latency concerns).
- The consistency gains from TPC and Retrieval are modest (0.854→0.858), indicating limited marginal benefit of these designs.
- Advantages in multi-character scenarios (M2SB) are less pronounced than in single-character settings.
- No direct comparison with the latest dedicated video generation models (e.g., Veo3).

## Related Work & Insights
- Comparison with CharaConsist demonstrates that text-only input limits customization flexibility.
- UMLLMs (particularly OmniGen2) as a backbone for story generation represent a promising direction.
- The shot-type prompt tuning approach is generalizable to other generation tasks requiring compositional control.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of multimodal customization and shot control is innovative, though individual components are incremental.
- Experimental Thoroughness: ⭐⭐⭐⭐ New benchmarks, multi-baseline comparisons, and ablations are provided, but in-depth evaluation at the video level is lacking.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with detailed method descriptions.
- Value: ⭐⭐⭐⭐ Offers meaningful contributions to visual storytelling generation; both the benchmarks and the framework are likely to be adopted by future work.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Widget2Code: From Visual Widgets to UI Code via Multimodal LLMs](widget2code_from_visual_widgets_to_ui_code_via_multimodal_llms.md)
- [\[ICCV 2025\] Multimodal LLMs as Customized Reward Models for Text-to-Image Generation](../../ICCV2025/multimodal_vlm/multimodal_llms_as_customized_reward_models_for_text-to-image_generation.md)
- [\[CVPR 2026\] PersonaVLM: Long-Term Personalized Multimodal LLMs](personavlm_long_term_personalized_multimodal_llms.md)
- [\[CVPR 2026\] UniGame: Turning a Unified Multimodal Model Into Its Own Adversary](unigame_turning_a_unified_multimodal_model_into_its_own_adversary.md)
- [\[CVPR 2026\] TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs](timelens_rethinking_video_temporal_grounding_with_multimodal_llms.md)

<!-- RELATED:END -->
