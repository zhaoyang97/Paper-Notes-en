---
title: >-
  [Paper Note] ETTA: Elucidating the Design Space of Text-to-Audio Models
description: >-
  [ICML 2025][Audio & Speech][Text-to-Audio] ETTA systematically elucidates the design space (data, architecture, training objectives, and sampling strategies) of text-to-audio (TTA) models through large-scale experiments, and constructs the current state-of-the-art TTA model under public data based on these findings.
tags:
  - "ICML 2025"
  - "Audio & Speech"
  - "Text-to-Audio"
  - "Diffusion Models"
  - "Flow Matching"
  - "Design Space"
  - "Synthetic Data"
date: 2026-05-08
content_hash: 22e831cadfa47ccd
---

# ETTA: Elucidating the Design Space of Text-to-Audio Models

**Conference**: ICML 2025  
**arXiv**: [2412.19351](https://arxiv.org/abs/2412.19351)  
**Code**: [https://github.com/NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN)  
**Area**: Image Generation (Audio Generation)  
**Keywords**: Text-to-Audio, Diffusion Models, Flow Matching, Design Space, Synthetic Data

## TL;DR
ETTA systematically elucidates the design space (data, architecture, training objectives, and sampling strategies) of text-to-audio (TTA) models through large-scale experiments, and constructs the current state-of-the-art TTA model under public data based on these findings.

## Background & Motivation

**Background**: Text-to-audio (TTA) synthesis has made significant progress in recent years, allowing users to generate audio using natural language prompts, which enriches creative workflows. Representative works include AudioLDM, Make-An-Audio, Tango, and others.

**Limitations of Prior Work**: Although numerous TTA models exist, the specific impacts of data, model architectures, training objectives, and sampling strategies on target benchmarks remain unclear. There is a lack of unified, large-scale ablation studies to guide design choices.

**Key Challenge**: Prior works tend to modify individual components in isolation, making it difficult to determine which design choices are truly critical. While models trained on proprietary data show good performance, they are not reproducible, and models trained on public data lack competitiveness.

**Goal**: To provide a comprehensive understanding of the design space of TTA models, including best practices for architecture, training, and inference, and to construct the most competitive model on public data.

**Key Insight**: Focus on diffusion and Flow Matching models, designing large-scale experiments to systematically compare choices across each dimension.

**Core Idea**: Identify the optimal configuration for each design dimension of TTA models through systematic experiments, supplemented by a large-scale synthetic caption dataset (AF-Synthetic) to mitigate data scarcity.

## Method

### Overall Architecture

ETTA is a TTA model based on latent diffusion / flow matching:
- **Input**: Text descriptions (natural language prompts).
- **Intermediate Representation**: Audio is encoded into a latent space via a VAE, where latent diffusion / flow matching generation is performed.
- **Output**: The corresponding audio waveform is generated.
- A text encoder extracts conditional information to guide the generation process.

### Key Designs

1. **AF-Synthetic Large-Scale Synthetic Caption Dataset**:

    - Uses audio understanding models to generate high-quality synthetic captions for large volumes of audio.
    - Expands the scale of training data to alleviate the shortage of public TTA datasets.
    - **Design Motivation**: High-quality captions are a key bottleneck for TTA model performance; synthetic captions can scale labeled data at lower costs.

2. **Systematic Comparison of Architectural Choices**:

    - Compares the performance of UNet versus Transformer (DiT) architectures in TTA tasks.
    - Investigates the impact of different text encoders (CLAP, T5, FLAN-T5).
    - Analyzes structural design choices of VAEs (compression ratio, channel size, etc.).
    - **Design Motivation**: Architectural choices vary widely in literature and require a fair comparison under a unified setup.

3. **Comparison of Training Objectives**:

    - Systematically compares diffusion models (DDPM) and Flow Matching objectives.
    - Analyzes the impact of different noise schedules and weighing functions.
    - Investigates the optimal configuration for Classifier-Free Guidance (CFG).
    - **Design Motivation**: The training objective directly determines generation quality, but a unified comparison has been lacking in the TTA domain.

4. **Pareto Analysis of Sampling Strategies**:

    - Analyzes the performance of different sampling methods (DDIM, DPM-Solver, Euler) on the quality-speed Pareto curve.
    - Identifies the optimal sampling configuration under a given inference budget.
    - **Design Motivation**: Practical deployment requires balancing inference speed and quality.

### Loss & Training

- Training is performed in the latent space based on Flow Matching or diffusion objectives.
- Classifier-Free Guidance is used during inference to enhance adherence to text conditions.
- The training data features a mixture of real captions and AF-Synthetic captions.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA (Public Data) | Prev. SOTA (Proprietary Data) |
|--------|------|------|---------------------|---------------------|
| AudioCaps | FD↓ | Significantly improved | Higher | Optimal |
| AudioCaps | KL↓ | Significantly improved | Higher | Close |
| MusicCaps | FD↓ | Improved | Higher | Better |
| MusicCaps | KL↓ | Improved | Higher | Close |

ETTA achieves state-of-the-art performance among models trained on public data, matching the competitiveness of models trained on proprietary data.

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| W/o AF-Synthetic | FD increases | Synthetic caption data is crucial for performance |
| UNet vs DiT | Each has advantages | Selection should depend on scale and scenario |
| DDPM vs Flow Matching | FM is slightly superior | FM is advantageous in sampling efficiency |
| Different Sampling Steps | Pareto curve | 25-50 steps represent the optimal balance point |

### Key Findings

- Data quality (especially caption quality) is the biggest performance bottleneck.
- Synthetic captions (AF-Synthetic) can effectively compensate for the lack of manual annotations.
- Flow Matching outperforms traditional diffusion in sampling efficiency.
- Appropriate CFG scaling is crucial for quality.
- ETTA performs better in generating audio corresponding to complex and imaginative captions.

## Highlights & Insights

1. **Systematic**: This work presents the first comprehensive and quantitative ablation study on the design space of TTA models.
2. **Dataset Contribution**: AF-Synthetic provides a high-quality synthetic caption dataset.
3. **Pareto Analysis**: It serves as a practical guide for choosing sampling methods.
4. **Reproducibility**: All experiments are based on public data, making the results fully reproducible.

## Limitations & Future Work

1. The quality of synthetic captions is limited by the capabilities of the audio understanding models used.
2. Evaluation is primarily limited to AudioCaps and MusicCaps, showcasing restricted task diversity.
3. Multi-modal conditioning (such as text + audio editing) was not explored in depth.
4. Long-audio generation capability has not been fully verified.

## Related Work & Insights

- Analogous to EDM's illumination of the image diffusion design space, ETTA does the same for audio diffusion.
- The synthetic caption strategy can be transferred to other conditional generation tasks (e.g., text-to-video).
- The Pareto analysis methodology can be extended to sampling strategy selection in other generative models.

## Rating
- Novelty: ⭐⭐⭐⭐ The systematic study of the design space is methodologically valuable, though the individual components are not entirely net-new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale systematic ablations with extensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with comprehensive experiments.
- Value: ⭐⭐⭐⭐ Provides a practical design guide and a strong baseline for the TTA community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers](../../NeurIPS2025/audio_speech/physics_of_language_models_part_41_architecture_design_and_the_magic_of_canon_la.md)
- [\[ICML 2025\] IMPACT: Iterative Mask-based Parallel Decoding for Text-to-Audio Generation with Diffusion Modeling](impact_iterative_mask-based_parallel_decoding_for_text-to-audio_generation_with_.md)
- [\[ACL 2025\] SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models](../../ACL2025/audio_speech/speechweave_diverse_multilingual_synthetic_text_audio_data_generation_pipeline_f.md)
- [\[NeurIPS 2025\] Latent Space Factorization in LoRA](../../NeurIPS2025/audio_speech/latent_space_factorization_in_lora.md)
- [\[ICLR 2026\] MambaVoiceCloning: Efficient and Expressive Text-to-Speech via State-Space Modeling and Diffusion Control](../../ICLR2026/audio_speech/mambavoicecloning_efficient_and_expressive_text-to-speech_via_state-space_modeli.md)

</div>

<!-- RELATED:END -->
