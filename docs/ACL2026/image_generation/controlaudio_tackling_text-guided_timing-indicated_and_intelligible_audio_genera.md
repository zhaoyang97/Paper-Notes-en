---
title: >-
  [Paper Note] ControlAudio: Tackling Text-Guided, Timing-Indicated and Intelligible Audio Generation via Progressive Diffusion Modeling
description: >-
  [ACL 2026][Image Generation][Text-to-Audio] This paper proposes ControlAudio, a unified progressive diffusion modeling framework that achieves three capabilities—text-guided generation, precise temporal control, and intelligible speech synthesis—within a single diffusion model through three-stage progressive training (TTA pretraining → temporal control fine-tuning → joint temporal+intelligible speech training) and progressive guidance sampling, significantly outperforming existing methods in temporal precision and speech intelligibility.
tags:
  - ACL 2026
  - Image Generation
  - Text-to-Audio
  - Temporal Control
  - Intelligible Speech
  - Progressive Diffusion
  - DiT
date: 2026-05-08
content_hash: e26a4d1d69917451
---

# ControlAudio: Tackling Text-Guided, Timing-Indicated and Intelligible Audio Generation via Progressive Diffusion Modeling

**Conference**: ACL 2026  
**arXiv**: [2510.08878](https://arxiv.org/abs/2510.08878)  
**Code**: [Project Page](https://control-audio.github.io/Control-Audio/)  
**Area**: Image Generation  
**Keywords**: Text-to-Audio, Temporal Control, Intelligible Speech, Progressive Diffusion, DiT

## TL;DR

This paper proposes ControlAudio, a unified progressive diffusion modeling framework that achieves three capabilities—text-guided generation, precise temporal control, and intelligible speech synthesis—within a single diffusion model through three-stage progressive training (TTA pretraining → temporal control fine-tuning → joint temporal+intelligible speech training) and progressive guidance sampling, significantly outperforming existing methods in temporal precision and speech intelligibility.

## Background & Motivation

**Background**: Text-to-audio (TTA) generation has achieved significant progress through large-scale diffusion models. Recent research has begun exploring fine-grained control: one line of work achieves precise temporal control (e.g., "bird chirping, 2-5 seconds"), while another enables intelligible speech generation (audio containing clearly distinguishable speech content).

**Limitations of Prior Work**: (1) Due to scarce large-scale annotated data (datasets containing both temporal annotations and speech transcriptions are extremely limited), controllable TTA remains constrained in performance even after scaling; (2) No prior work has achieved both temporal control and intelligible speech generation within a unified framework; (3) Adding fine-grained control signals often sacrifices generation quality under pure text conditions (catastrophic forgetting); (4) Natural language descriptions of complex multi-event scenarios contain ambiguities.

**Key Challenge**: Controllable TTA requires handling condition signals at multiple granularities (text → temporal → phoneme), but training data at different granularities varies drastically in scale (millions of text-audio pairs vs. tens of thousands of temporal annotations), making direct joint training ineffective.

**Goal**: Unify text-guided generation, temporal indication, and intelligible speech capabilities within a single diffusion model without sacrificing individual capabilities.

**Key Insight**: Model controllable TTA as a multi-task learning problem through progressive diffusion modeling—adopting coarse-to-fine progressive strategies across data construction, model training, and guidance sampling.

**Core Idea**: Progressive modeling naturally aligns with the hierarchical nature of control granularity (text → temporal → phoneme) and the coarse-to-fine characteristics of diffusion sampling—emphasizing coarse-grained temporal structure in early diffusion stages and introducing fine-grained phoneme content in later stages.

## Method

### Overall Architecture

Three-stage progressive training: Stage 1 pretrains DiT on large-scale text-audio pairs → Stage 2 fine-tunes on temporal annotation data (switching between text/text+temporal conditions to avoid forgetting) → Stage 3 jointly trains on full multi-source data (unfreezing text encoder for joint optimization). During inference, progressive guidance sampling is employed: early diffusion steps use temporal condition guidance, while later steps introduce phoneme conditions.

### Key Designs

1. **Unified Semantic Modeling of Structured Prompts**:

    - Function: Unify encoding of text, temporal, and phoneme condition signals using a single text encoder
    - Mechanism: Design structured prompt format using special tokens to separate event descriptions and precise start/end times (e.g., `<event>bird chirping<start>2.0<end>5.0`), eliminating natural language ambiguities. Temporal windows naturally define speech duration, and phoneme tokens are extended into the text encoder's vocabulary for unified encoding
    - Design Motivation: Natural language descriptions of complex scenarios create ambiguity where "from...to" could indicate pitch change or time range; structured format eliminates ambiguity and is extensible

2. **Progressive Diffusion Training**:

    - Function: Progressively acquire finer-grained control capabilities while maintaining learned abilities
    - Mechanism: Stage 1 text-only conditional pretraining establishes high-fidelity TTA prior. Stage 2 fine-tunes on temporal data while randomly switching between pure text/text+temporal conditions to avoid catastrophic forgetting. Stage 3 unfreezes text encoder for joint optimization, switching among text/text+temporal/text+temporal+phoneme conditions
    - Design Motivation: Direct training with all conditions leads to performance degradation due to imbalanced data scales and task complexity; progressive strategy enables models to gradually build coarse-to-fine control capabilities

3. **Progressive Guidance Sampling**:

    - Function: Guide diffusion sampling hierarchically by granularity during inference
    - Mechanism: Diffusion sampling is inherently coarse-to-fine—early steps generate large-scale structure, later steps synthesize fine-grained details. Progressive guidance emphasizes temporal conditions early (determining event time windows) and introduces phoneme conditions later (determining speech content), aligning with diffusion sampling's natural characteristics
    - Design Motivation: Fixed guidance signals cannot match the requirements of different diffusion stages; progressive guidance achieves natural alignment between condition granularity and sampling granularity

### Loss & Training

Standard conditional diffusion training objective (noise prediction). Data construction: Annotated data extracted from AudioSet-SL containing speech segments transcribed using Gemini 2.5 Pro; synthetic data from LibriTTS-R combining single/multi-speaker scenarios and mixing with non-speech backgrounds (SNR 2-10 dB), generating 171,246 complex audio scenarios.

## Key Experimental Results

### Main Results

**Temporal Control Evaluation on AudioCondition Test Set**

| Method | Eb ↑ | At ↑ | FAD ↓ | CLAP ↑ | Temporal(subjective) ↑ | OVL(subjective) ↑ |
|------|------|------|-------|--------|----------------|------------|
| Ground Truth | 43.37 | 67.53 | - | 0.377 | 4.52 | 4.48 |
| Stable Audio | 11.28 | 51.67 | 1.93 | 0.318 | 1.94 | 3.44 |
| PicoAudio | 29.96 | 57.70 | 3.43 | 0.296 | 2.70 | 2.44 |
| **ControlAudio** | **38.50** | **67.87** | **0.98** | **0.347** | **4.01** | **3.74** |

### Ablation Study

**Training Strategy Ablation**

| Config | At ↑ | FAD ↓ | Speech WER ↓ |
|------|------|-------|----------|
| Stage 1 only | baseline | baseline | no speech capability |
| + Stage 2 (temporal) | significant improvement | slight decrease | no speech capability |
| + Stage 3 (temporal+speech) | best | best | **best** |
| No progressive guidance (fixed) | decrease | increase | increase |

### Key Findings

- ControlAudio achieves temporal precision approaching Ground Truth (At 67.87 vs 67.53), far exceeding other methods
- Stage 3 joint training not only unlocks speech capability but further improves temporal precision—benefiting from richer time-content alignment signals in temporal-annotated speech data
- Unfreezing text encoder for joint optimization is critical—enabling condition encoding and generation backbone to synergistically adapt to complex multi-objective tasks
- Progressive guidance sampling significantly outperforms fixed guidance—alignment of condition granularity with sampling granularity improves generation quality
- CoT LLM planning can transform free text into structured prompts, expanding practical use cases

## Highlights & Insights

- Progressive design permeates data → training → inference, forming a consistent coarse-to-fine paradigm
- Structured prompts + phoneme extension enable handling three condition types with a single text encoder, avoiding multi-module complexity
- Stage 3 joint training counterintuitively improving temporal precision—positive transfer in multi-task learning

## Limitations & Future Work

- SNR range (2-10 dB) of synthetic data may not cover all real-world scenarios
- Speaker identity control for speech generation remains unexplored
- Long audio generation beyond 10 seconds not evaluated
- Relies on external LLMs to convert free text into structured prompts

## Related Work & Insights

- **vs PicoAudio/AudioComposer**: Only achieve temporal control without speech capability; ControlAudio first unifies both
- **vs VoiceLDM/VoiceDiT**: Focus on speech synthesis but lack temporal control of general audio events
- **vs Progressive modeling in video generation**: ControlAudio first introduces progressive modeling to controllable TTA domain

## Rating

- Novelty: ⭐⭐⭐⭐ Novel framework design combining progressive diffusion modeling + unified semantic encoding
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive subjective and objective evaluations, thorough ablation studies
- Writing Quality: ⭐⭐⭐⭐ Systematic and clear method description, well-articulated progressive design motivation
- Value: ⭐⭐⭐⭐ First to unify temporal control and intelligible speech generation, advancing controllable audio generation

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Visual Autoregressive Modeling for Instruction-Guided Image Editing](../../ICLR2026/image_generation/visual_autoregressive_modeling_for_instruction-guided_image_editing.md)
- [\[CVPR 2026\] CoLoGen: Progressive Learning of Concept-Localization Duality for Unified Image Generation](../../CVPR2026/image_generation/cologen_progressive_learning_of_concept-localization_duality_for_unified_image_g.md)
- [\[AAAI 2026\] ReAlign: Text-to-Motion Generation via Step-Aware Reward-Guided Alignment](../../AAAI2026/image_generation/realign_text-to-motion_generation_via_step-aware_reward-guided_alignment.md)
- [\[CVPR 2026\] BiMotion: B-spline Motion for Text-guided Dynamic 3D Character Generation](../../CVPR2026/image_generation/bimotion_b-spline_motion_for_text-guided_dynamic_3d_character_generation.md)
- [\[NeurIPS 2025\] MGAudio: Model-Guided Dual-Role Alignment for High-Fidelity Open-Domain Video-to-Audio Generation](../../NeurIPS2025/image_generation/model-guided_dual-role_alignment_for_high-fidelity_open-domain_video-to-audio_ge.md)

<!-- RELATED:END -->
