---
title: >-
  [Paper Note] SAO-Instruct: Free-form Audio Editing using Natural Language Instructions
description: >-
  [NeurIPS 2025][Image Generation][Audio Editing] This paper proposes SAO-Instruct, the first audio editing model supporting fully free-form natural language instructions. Training data consisting of editing triplets is constructed via three pipelines — Prompt-to-Prompt, DDPM inversion, and manual editing — and Stable Audio Open is fine-tuned to achieve context-preserving, targeted audio modification.
tags:
  - NeurIPS 2025
  - Image Generation
  - Audio Editing
  - Natural Language Instructions
  - Stable Audio Open
  - Prompt-to-Prompt
  - Diffusion Models
date: 2026-05-08
content_hash: 295d602afcb44fdc
---

# SAO-Instruct: Free-form Audio Editing using Natural Language Instructions

**Conference**: NeurIPS 2025
**arXiv**: [2510.22795](https://arxiv.org/abs/2510.22795)
**Code**: [GitHub](https://eth-disco.github.io/sao-instruct)
**Area**: Image Generation
**Keywords**: Audio Editing, Natural Language Instructions, Stable Audio Open, Prompt-to-Prompt, Diffusion Models

## TL;DR

This paper proposes SAO-Instruct, the first audio editing model supporting fully free-form natural language instructions. Training data consisting of editing triplets is constructed via three pipelines — Prompt-to-Prompt, DDPM inversion, and manual editing — and Stable Audio Open is fine-tuned to achieve context-preserving, targeted audio modification.

## Background & Motivation

Generative audio models have achieved remarkable progress in synthesizing high-fidelity audio, yet **audio editing** remains a largely unsolved challenge. Existing approaches suffer from several critical bottlenecks:

**Zero-shot inversion methods** (e.g., ZETA) require a complete textual description of the target audio, but accurately capturing the unique acoustic characteristics of an audio clip in concise text is inherently difficult.

**Supervised methods such as AUDIT** support only a predefined set of editing operations (add, remove, replace, inpaint, super-resolution), making them unable to handle flexible and diverse user instructions.

**Editing granularity**: Users may say "make the birdsong louder," "add reverb," or "remove the vocals" — instructions that vary greatly in scope and complexity and are ill-suited to hard-coded taxonomies.

Core motivation: **enabling users to edit audio with a single free-form natural language instruction**, where the system autonomously understands the editing intent and executes it precisely while preserving the background context of the original audio.

## Method

### Overall Architecture

A three-stage pipeline: (1) an LLM generates (input description, editing instruction, output description) triplets from input descriptions; (2) three complementary methods produce corresponding audio editing sample pairs; (3) Stable Audio Open is fine-tuned to yield SAO-Instruct. At inference time, the model receives an input audio clip and a free-form editing instruction, and produces the edited audio.

### Key Designs

1. **Prompt-to-Prompt Audio Synthesis (Fully Synthetic Data)**

   The Prompt-to-Prompt approach from the image domain is adapted to audio. The core idea is to inject attention maps (cross-attention maps) from the input description into the generation process guided by the output description, thereby achieving local edits while preserving global contextual consistency.

   Three key parameters control editing strength:
   - $\lambda_{\text{frac}}^{\text{attn}}$: attention injection ratio (0 = no influence, 1 = identical)
   - $\lambda_{\text{delay}}^{\text{attn}}$: injection delay (skip the first $N\%$ of attention maps)
   - $\lambda_{\text{weight}}^{\text{attn}}$: attention weight amplification for changed tokens

   Since different edits require different parameter configurations, **Bayesian optimization** (10 trials/sample) is employed to automatically search for optimal settings. The objective function is:

   $$\mathcal{L}_{\text{obj}} = \omega_1 \cdot M_{\text{CLAP}}^{\text{out}} + \omega_2 \cdot M_{\text{CLAP}}^{\text{dir}} + \omega_3 \cdot M_{\text{CLAP}}^{\text{sim}} - \omega_4 \cdot M_{\text{MEL}}^{\text{sim}}$$

   Weights are determined via ELO rankings from small-scale human listening tests: $\omega_1=8,\ \omega_2=14,\ \omega_3=0.5,\ \omega_4=1.5$.

   A candidate search procedure is also designed: seven audio pairs are generated using different seeds/CFG combinations, Gemini 2.0 Flash performs perceptual quality evaluation (threshold: 6 points), and CLAP similarity is used to select the best candidate from those passing the filter.

2. **DDPM Inversion (Semi-Synthetic Data)**

   Real input audio is encoded into the latent space via DDPM inversion and then denoised using a modified description to generate the edited audio. The key parameter $T_{\text{start}}$ controls inversion depth (low value = high consistency/low editing flexibility; high value = the reverse). Bayesian optimization (7 trials/sample) is again used for automatic hyperparameter tuning. The advantage is that the input audio is real, increasing data diversity.

3. **Manual Editing (Fully Real Data)**

   Twelve deterministic audio editing operations are implemented: ADD, REPLACE, DROP, SWAP, LOOP, PITCH, SPEED, LOW_PASS, HIGH_PASS, INPAINT, SUPER_RES, and DENOISE. GPT-4.1 mini generates natural language instructions for each operation, and a two-stage post-processing pipeline (variant paraphrasing + concise compression, each with 50% probability) is applied to increase instruction diversity.

### Loss & Training

Fine-tuning proceeds under Stable Audio Open's diffusion objective. The model employs three conditioning inputs: (1) the text condition is replaced by the free-form editing instruction; (2) the duration condition is set to the length of the input audio; (3) the encoded input audio is concatenated to the model's input channels. At inference time, Gaussian noise is added to the latent representation of the input audio as the denoising starting point, with 100 denoising steps and a CFG value of 5.

## Key Experimental Results

### Ablation Study — Contribution of Different Data Sources

| Training Data | Samples | FD (orig.)↓ | FD (regen.)↓ | IS↑ | CLAP↑ |
|---|---|---|---|---|---|
| Prompt-to-Prompt | 50k | 18.71 | 18.29 | 7.94 | 0.38 |
| DDPM Inversion | 50k | 20.50 | 20.72 | 6.82 | 0.34 |
| Manual Editing | 50k | **14.60** | 21.21 | 7.50 | 0.35 |
| Mixed | 50k | 19.11 | 19.24 | 7.69 | 0.38 |
| **Mixed-Large** | **150k** | 18.38 | **18.97** | 7.59 | **0.38** |

### Main Results — Comparison with Baselines

| Model | Inference Time↓ | FD (orig.)↓ | CLAP↑ | Quality MOS↑ | Relevance MOS↑ | Fidelity MOS↑ |
|---|---|---|---|---|---|---|
| AudioEditor | 79.49s | 17.21 | **0.48** | 3.22 | 3.33 | 2.75 |
| ZETA ($T=50$) | 24.65s | **15.31** | 0.38 | 3.56 | 3.25 | 2.95 |
| ZETA ($T=75$) | 27.91s | 17.78 | 0.36 | 3.28 | 3.04 | 2.75 |
| **SAO-Instruct** | **9.94s** | 18.38 | 0.38 | 3.54 | **3.83** | **3.99** |

### Key Findings

1. **Consistently superior subjective evaluation**: SAO-Instruct significantly outperforms all baselines on editing relevance (3.83 vs. 3.33) and fidelity (3.99 vs. 2.95), while requiring only free-form editing instructions (baselines require complete audio descriptions).
2. **Highest inference efficiency**: 9.94 seconds per sample, approximately 8× faster than AudioEditor.
3. **Complementary data sources**: Prompt-to-Prompt excels in editing precision (low regeneration FD), manual editing achieves the highest fidelity (low original FD), and combining both balances the two advantages.
4. **150k samples marginally outperform 50k**: Scaling data volume yields marginal improvements across most metrics, indicating further room for data scaling.
5. **Competitive under information asymmetry**: SAO-Instruct uses only editing instructions, whereas baselines use complete target descriptions, yet SAO-Instruct still achieves superior subjective scores.

## Highlights & Insights

1. **First fully free-form audio editing model**: Removes the constraint of predefined operation sets, allowing users to express editing intent in arbitrary natural language.
2. **Elegant data engine design**: Three complementary data pipelines — fully synthetic, semi-synthetic, and fully real — are combined, with Bayesian optimization for automatic hyperparameter tuning to avoid manual search.
3. **Instruction diversification strategy**: A three-stage instruction generation process (initial → variants → condensed) effectively simulates the diverse expression styles of real users.
4. **Human-in-the-loop objective weight design**: Weights are determined via ELO rankings from small-scale listening tests, yielding a more reliable design than manual tuning.

## Limitations & Future Work

- The data generation pipeline incurs substantial computational overhead (Bayesian optimization × multiple denoising runs × multiple samples).
- Performance is bounded by the generation quality of the underlying Stable Audio Open model, which may fail in certain complex scenarios.
- Only English instructions and general audio are supported; music editing is not addressed.
- Multi-step editing is not supported — only one instruction can be executed at a time.
- Prompt-to-Prompt attention injection has limited fidelity in complex scenarios.

## Related Work & Insights

- **Audio Generation**: Stable Audio Open, AudioLDM, AudioGen
- **Audio Editing**: AUDIT (predefined operations), ZETA (zero-shot inversion), AudioEditor
- **Inspiration from Image Editing**: InstructPix2Pix (instruction fine-tuning), Prompt-to-Prompt (attention injection)

## Rating

- **Novelty**: ⭐⭐⭐⭐☆ — First fully free-form audio editing model, though core techniques are adapted from image editing.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Ablation, comparison, and subjective listening tests cover comprehensive evaluation dimensions.
- **Writing Quality**: ⭐⭐⭐⭐☆ — Method is described in detail with a clear pipeline presentation.
- **Value**: ⭐⭐⭐⭐☆ — Fills an important gap in audio editing; the open-source model is directly usable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Describe, Don't Dictate: Semantic Image Editing with Natural Language Intent](../../ICCV2025/image_generation/describe_dont_dictate_semantic_image_editing_with_natural_language_intent.md)
- [\[CVPR 2026\] Language-Free Generative Editing from One Visual Example](../../CVPR2026/image_generation/language-free_generative_editing_from_one_visual_example.md)
- [\[NeurIPS 2025\] CAMILA: Context-Aware Masking for Image Editing with Language Alignment](camila_contextaware_masking_for_image_editing_with_language.md)
- [\[NeurIPS 2025\] DiffEye: Diffusion-Based Continuous Eye-Tracking Data Generation Conditioned on Natural Images](diffeye_diffusion-based_continuous_eye-tracking_data_generation_conditioned_on_n.md)
- [\[NeurIPS 2025\] SplitFlow: Flow Decomposition for Inversion-Free Text-to-Image Editing](splitflow_flow_decomposition_for_inversion-free_text-to-image_editing.md)

</div>

<!-- RELATED:END -->
