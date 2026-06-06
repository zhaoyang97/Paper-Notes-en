---
title: >-
  [Paper Note] ControlAudio: Tackling Text-Guided, Timing-Indicated and Intelligible Audio Generation via Progressive Diffusion Modeling
description: >-
  [ACL 2026][Audio & Speech][Text-to-Audio] This paper proposes ControlAudio, a unified progressive diffusion modeling framework. Through three-stage progressive training (TTA pre-training → temporal control fine-tuning →…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Text-to-Audio"
  - "Temporal Control"
  - "Intelligible Speech"
  - "Progressive Diffusion"
  - "DiT"
date: 2026-05-08
content_hash: 2b28ef1a82179bf5
---

# ControlAudio: Tackling Text-Guided, Timing-Indicated and Intelligible Audio Generation via Progressive Diffusion Modeling

**Conference**: ACL 2026  
**arXiv**: [2510.08878](https://arxiv.org/abs/2510.08878)  
**Code**: [Project Page](https://control-audio.github.io/Control-Audio/)  
**Area**: Image Generation  
**Keywords**: Text-to-Audio, Temporal Control, Intelligible Speech, Progressive Diffusion, DiT

## TL;DR

This paper proposes ControlAudio, a unified progressive diffusion modeling framework. Through three-stage progressive training (TTA pre-training → temporal control fine-tuning → joint temporal and intelligible speech training) and progressive guided sampling, it achieves text-guided, precisely timed, and intelligible speech generation within a single diffusion model, significantly outperforming existing methods in timing precision and speech clarity.

## Background & Motivation

**Background**: Text-to-Audio (TTA) generation has achieved significant progress through large-scale diffusion models. Recent research has begun exploring fine-grained control: one line of work implements precise temporal control (e.g., "bird chirping, 2-5 seconds"), while another achieves intelligible speech generation (containing clearly discernible spoken content).

**Limitations of Prior Work**: (1) Due to the scarcity of large-scale annotated data (data containing both temporal labels and speech transcripts is extremely rare), controllable TTA performance remains limited at scale; (2) Prior work has not simultaneously achieved both temporal control and intelligible speech within a unified framework; (3) Adding fine-grained control signals often sacrifices generation quality under pure text conditions (catastrophic forgetting); (4) Natural language descriptions of complex multi-event scenes are often ambiguous.

**Key Challenge**: Controllable TTA requires processing multiple granularities of conditional signals (text → timing → phonemes) simultaneously, but the data scales for different granularities vary drastically (millions of text-audio pairs vs. tens of thousands of temporal annotations), making direct joint training ineffective.

**Goal**: To implement text-guided, timing-indicated, and intelligible speech capabilities within a single diffusion model without sacrificing individual performance.

**Key Insight**: Modeling controllable TTA as a multi-task learning problem using progressive diffusion modeling—adopting a coarse-to-fine progressive strategy across data construction, model training, and guided sampling levels.

**Core Idea**: Progressive modeling naturally aligns with the hierarchical nature of control granularity (text → timing → phoneme) and the coarse-to-fine characteristics of diffusion sampling—emphasizing coarse-grained temporal structures in the early stages of the diffusion trajectory and introducing fine-grained phoneme content in the later stages.

## Method

### Overall Architecture

Three-stage progressive training: Stage 1 pre-trains DiT on large-scale text-audio pairs → Stage 2 fine-tunes on temporal annotated data (switching between text and text+temporal conditions to avoid forgetting) → Stage 3 joint training on all multi-source data (unfreezing the text encoder for joint optimization). Inference utilizes progressive guided sampling: temporal conditions lead in early diffusion steps, while phoneme conditions are introduced in later steps.

### Key Designs

1.  **Unified Semantic Modeling for Structured Prompts**:
    *   **Function**: Encodes text, timing, and phoneme conditional signals using a single unified text encoder.
    *   **Mechanism**: A structured prompt format is designed utilizing special tokens to separate event descriptions and precise timestamps (e.g., `<event>bird chirping<start>2.0<end>5.0`), eliminating natural language ambiguity. It leverages temporal windows to naturally define speech duration and extends phoneme tokens into the text encoder vocabulary for unified encoding.
    *   **Design Motivation**: Natural language descriptions for complex scenes like "from...to" can refer to either pitch changes or time ranges; structured formats eliminate ambiguity and are scalable.

2.  **Progressive Diffusion Training**:
    *   **Function**: Gradually acquires finer-grained control capabilities while maintaining learned priors.
    *   **Mechanism**: Stage 1 establishes a high-fidelity TTA prior via text-only pre-training. Stage 2 fine-tunes on temporal data but randomly switches between text-only and text+temporal conditions to prevent catastrophic forgetting. Stage 3 unfreezes the text encoder for joint optimization, switching among text, text+temporal, and text+temporal+phoneme conditions.
    *   **Design Motivation**: Direct training on all conditions simultaneously leads to performance degradation due to data imbalance and task complexity; the progressive strategy allows the model to build control from coarse to fine steps.

3.  **Progressive Guided Sampling**:
    *   **Function**: Guides diffusion sampling according to the granularity hierarchy during inference.
    *   **Mechanism**: Diffusion sampling is inherently coarse-to-fine—early steps generate large-scale structures, while later steps synthesize fine-grained details. Progressive guidance emphasizes temporal conditions early (determining event windows) and introduces phoneme conditions later (determining speech content), aligning with the natural properties of diffusion.
    *   **Design Motivation**: Fixed guidance signals fail to match the needs of different diffusion stages; progressive guidance achieves natural alignment between condition granularity and sampling granularity.

### Loss & Training

Standard conditional diffusion training objective (noise prediction). Data construction: Annotated data is extracted from AudioSet-SL containing speech segments transcribed by Gemini 2.5 Pro; simulated data is synthesized from LibriTTS-R by combining single/multi-speaker scenarios and mixing with non-speech backgrounds (SNR 2-10 dB), generating 171,246 complex audio scenes.

## Key Experimental Results

### Main Results

**Temporal Control Evaluation on AudioCondition Test Set**

| Method | Eb ↑ | At ↑ | FAD ↓ | CLAP ↑ | Temporal (Subj) ↑ | OVL (Subj) ↑ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Ground Truth | 43.37 | 67.53 | - | 0.377 | 4.52 | 4.48 |
| Stable Audio | 11.28 | 51.67 | 1.93 | 0.318 | 1.94 | 3.44 |
| PicoAudio | 29.96 | 57.70 | 3.43 | 0.296 | 2.70 | 2.44 |
| **ControlAudio** | **38.50** | **67.87** | **0.98** | **0.347** | **4.01** | **3.74** |

### Ablation Study

**Training Strategy Ablation**

| Configuration | At ↑ | FAD ↓ | Speech WER ↓ |
| :--- | :--- | :--- | :--- |
| Stage 1 only | Baseline | Baseline | No speech |
| + Stage 2 (Timing) | Significant Gain | Slight Drop | No speech |
| + Stage 3 (Timing + Speech) | Best | Best | **Best** |
| W/O Progressive Guidance | Decrease | Increase | Increase |

### Key Findings

*   ControlAudio approaches Ground Truth in temporal accuracy (At 67.87 vs. 67.53), far exceeding other methods.
*   Joint training in Stage 3 not only unlocks speech capabilities but further improves temporal accuracy—this is attributed to the richer temporal-content alignment signals provided by annotated speech data.
*   Unfreezing the text encoder for joint optimization is crucial—it allows the condition encoding and generation backbone to co-adapt to complex multi-objective tasks.
*   Progressive guided sampling is significantly better than fixed guidance—aligning condition granularity with sampling granularity improves generation quality.
*   CoT LLM planning can transform free-form text into structured prompts, extending practical use cases.

## Highlights & Insights

*   Progressive design spans data → training → inference, forming a consistent coarse-to-fine paradigm.
*   Structured prompts combined with phoneme expansion enable a single text encoder to handle three types of conditions, avoiding multi-module complexity.
*   The finding that Stage 3 joint training actually improves temporal accuracy is counter-intuitive—suggesting positive transfer in multi-task learning.

## Limitations & Future Work

*   The SNR range of simulated data (2-10 dB) may not cover all real-world scenarios.
*   Speaker identity control for speech generation has not yet been explored.
*   Long audio generation exceeding 10 seconds has not been evaluated.
*   Dependency on external LLMs to transform free-form text into structured prompts.

## Related Work & Insights

*   **vs PicoAudio/AudioComposer**: These only implement temporal control without speech capabilities; ControlAudio unifies both for the first time.
*   **vs VoiceLDM/VoiceDiT**: These focus on speech synthesis but lack temporal control for general audio events.
*   **vs Progressive Modeling in Video Generation**: ControlAudio is the first to introduce progressive modeling into the controllable TTA domain.

## Rating

*   Novelty: ⭐⭐⭐⭐ Framework design of progressive diffusion + unified semantic encoding is novel.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive subjective and objective evaluations with thorough ablation studies.
*   Writing Quality: ⭐⭐⭐⭐ Systematic and clear method descriptions; motivations for progressive design are well-articulated.
*   Value: ⭐⭐⭐⭐ First to unify temporal control and intelligible speech generation, advancing controllable audio generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text](../../CVPR2026/audio_speech/omnisonic_towards_universal_and_holistic_audio_generation_from_video_and_text.md)
- [\[ACL 2026\] Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval](omni-embed-audio_leveraging_multimodal_llms_for_robust_audio-text_retrieval.md)
- [\[ACL 2026\] ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Representation Alignment](immersivetts_environment-aware_text-to-speech_with_multimodal_diffusion_transfor.md)
- [\[ICLR 2026\] AC-Foley: Reference-Audio-Guided Video-to-Audio Synthesis with Acoustic Transfer](../../ICLR2026/audio_speech/ac-foley_reference-audio-guided_video-to-audio_synthesis_with_acoustic_transfer.md)
- [\[NeurIPS 2025\] Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video](../../NeurIPS2025/audio_speech/node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)

</div>

<!-- RELATED:END -->
