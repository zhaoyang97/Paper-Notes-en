---
title: >-
  [Paper Note] LiveStar: Live Streaming Assistant for Real-World Online Video Understanding
description: >-
  [NeurIPS 2025][Video Understanding][online video understanding] This paper proposes LiveStar, an always-on live streaming video understanding assistant that achieves adaptive response timing via a Streaming Causal Attent…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "online video understanding"
  - "streaming decoding"
  - "video-language alignment"
  - "live streaming"
  - "response timing"
date: 2026-05-08
content_hash: f43ad32290a6823f
---

# LiveStar: Live Streaming Assistant for Real-World Online Video Understanding

**Conference**: NeurIPS 2025
**arXiv**: [2511.05299](https://arxiv.org/abs/2511.05299)  
**Code**: [yzy-bupt/LiveStar](https://github.com/yzy-bupt/LiveStar)  
**Area**: Video Understanding
**Keywords**: online video understanding, streaming decoding, video-language alignment, live streaming, response timing

## TL;DR

This paper proposes LiveStar, an always-on live streaming video understanding assistant that achieves adaptive response timing via a Streaming Causal Attention Masks (SCAM) training strategy and a Streaming Verification Decoding (SVeD) inference framework, improving semantic correctness by 19.5% and reducing temporal deviation by 18.1% on the OmniStar benchmark.

## Background & Motivation

Existing online Video-LLMs (e.g., VideoLLM-online, VideoLLM-MoD) rely on EOS tokens to mark "silent" intervals, which introduces four critical issues:

1. **Response–silence imbalance**: Frames requiring EOS output far outnumber those requiring normal responses (e.g., in a 1-minute video at 3 FPS, the response-to-silence ratio is approximately 1:35).
2. **Consecutive frame inconsistency**: Visually similar adjacent frames may produce contradictory outputs—one frame generating a full narration while the next outputs only EOS.
3. **Pretraining misalignment**: Pretraining aligns image–text pairs, but the silence state forces frame tokens to map to EOS, violating the visual–language correspondence established during pretraining.
4. **Vocabulary confusion**: Embedding EOS as a regular vocabulary token causes frequent occurrences that pollute semantic coherence.

Furthermore, existing training data and evaluation scopes are limited, with most work focusing on Ego4D egocentric videos and lacking coverage of diverse real-world scenarios and multi-task settings.

## Core Problem

1. How to establish an effective response–silence training and inference framework without degrading foundational video understanding capabilities?
2. How to construct a comprehensive dataset and benchmark covering diverse real-world scenarios and tasks?

## Method

### 1. Streaming Causal Attention Masks (SCAM) Training Strategy

**Streaming video–language alignment**: The standard image/video–text pair alignment objective is reformulated as a frame-by-frame multi-turn instruction fine-tuning objective:

$$\max P([Txt^k] \mid [Ctx^{<t_i}], [Frm^{t_i}]), \forall t_i \in C_k$$

where $C_k = \{t_i\}_{i=m}^n$ is the semantic segment sharing the semantic text $[Txt^k]$. Consecutive frames within the same semantic segment share the same semantic caption, with random sampling from a paraphrase pool of size $M$ to avoid overfitting.

**Interleaved frame–caption sequences**: A dialogue-style format is adopted, with each turn containing one frame $[Frm^{t_i}]$ and its corresponding caption $[Cap^k]$, enabling incremental visual input while maintaining temporal awareness.

**Streaming causal attention mask**: A dedicated mask matrix is designed to replace standard causal attention, addressing three challenges:
- Preventing leakage of already-generated captions within the current semantic segment (avoiding trivial copying).
- Maintaining visibility of previously predicted tokens during current caption generation.
- Allowing the last caption of each semantic segment to persist across subsequent frames to mark semantic boundaries.

### 2. Streaming Verification Decoding (SVeD) Inference Framework

SVeD determines the optimal response timing via a single forward-pass verification:

- At each triggered decoding step $t_i$, the perplexity of the generated caption $\text{PPL}^{t_i}([Dec])$ is computed.
- For each new frame $[Frm^{t_j}]$, $\text{PPL}^{t_j}([Dec])$ is recomputed.
- If $\text{PPL}^{t_j}([Dec]) > \alpha \cdot \text{PPL}^{t_i}([Dec])$ (where $\alpha$ is a tunable scaling factor, default 1.03), decoding is triggered to generate a new caption.
- Otherwise, the model remains silent and the current caption is appended to the end of the context.

Compared to predicting EOS tokens to indicate silence, SVeD achieves faster inference under the same model architecture.

### 3. Peak-End Memory Compression

Inspired by the Peak-End rule from cognitive psychology, stale frames beyond a window $W$ (default 40 frames) are probabilistically pruned:
- Pre-computed PPL values are used to identify key frames (low PPL = high semantic importance).
- The caption of the last frame in each semantic segment is retained as an event summary.
- The deletion probability is proportional to the relative PPL within a semantic segment and the elapsed time.

### 4. Streaming KV Cache

A two-level cache architecture is employed: an intra-dialogue KV cache for frame-level processing, and a cross-dialogue streaming cache for maintaining long context. This achieves a 1.53× speedup on 5-minute video inference.

### 5. OmniStar Dataset

The dataset covers 15 real-world scene categories (46 subcategories), 20,137 videos, and 5 online evaluation tasks:
- **RNG**: Real-time Narration Generation
- **OTG**: Online Temporal Grounding
- **FDQ**: Frame-level Dense QA
- **COQ**: Contextual Online QA
- **MIQ**: Multi-turn Interactive QA

A semi-automatic, temporally dense annotation pipeline is adopted, with captions forming narratively coherent storylines.

## Key Experimental Results

| Model | RNG SemCor↑ | RNG TimDiff↓ | FDQ SemCor↑ | FPS↑ |
|-------|-------------|-------------|-------------|------|
| VideoLLM-online | 1.68 | 2.67 | 2.35 | 3.37 |
| VideoLLM-MoD | 1.66 | 2.54 | 2.11 | 3.41 |
| MMDuet | 1.63 | 2.32 | 4.78 | 0.91 |
| **LiveStar** | **3.19** | **1.91** | **6.44** | **3.82** |
| Human | 6.09 | 1.08 | 9.12 | - |

- Averaged across five OmniStar tasks: SemCor improves by 19.5%, TimDiff decreases by 18.1%, and FPS increases by 12.0%.
- On the Ego4D offline benchmark: TokAcc reaches 61.1%, surpassing the second-best LION-FS by 8.7%.
- Ablation study: Peak-End compression outperforms Uniform Dropout and FIFO Forgetting; KV cache achieves 1.53× speedup with negligible performance loss.

## Highlights & Insights

1. **Paradigm shift**: SCAM + SVeD replaces the EOS mechanism, fundamentally resolving the response–silence imbalance without disrupting the pretrained visual–language alignment.
2. **Efficient inference**: SVeD requires only a single forward-pass verification (rather than full decoding), and combined with Peak-End memory compression, supports video streams exceeding 10 minutes.
3. **OmniStar benchmark**: The first comprehensive dataset covering 15 real-world scene categories × 5 online tasks, filling a critical gap in online video understanding evaluation.
4. **Substantial gains**: LiveStar outperforms all existing online Video-LLMs across all 5 tasks while achieving the fastest inference speed.

## Limitations & Future Work

1. Each frame is compressed to 16 visual tokens, sacrificing fine-grained visual detail, which is unfavorable for subtle motion changes or complex scenes.
2. Only vision–text modalities are supported; audio information is not incorporated, limiting multimodal reasoning capacity.
3. Online evaluation relies on GPT-4o scoring (SemCor, SumFluen), which may introduce evaluation bias.
4. A substantial gap remains compared to human performance (SemCor: 3.19 vs. 6.09).

## Related Work & Insights

| Dimension | VideoLLM-online | MMDuet | LiveStar |
|-----------|----------------|--------|----------|
| Response timing | EOS token prediction | EOS token prediction | SVeD perplexity verification |
| Training strategy | Standard fine-tuning | Standard fine-tuning | SCAM streaming alignment |
| Output mode | Nearly every frame | Sparse output | Adaptive and balanced |
| Long video support | Limited | Limited | Peak-End compression + KV cache |
| Data diversity | Primarily Ego4D | Limited scenes | 15 scene categories, 20K videos |

- The perplexity verification mechanism of SVeD can be generalized to other streaming generation tasks (e.g., real-time translation, live commentary) for output timing decisions.
- The interleaved frame–caption training strategy of SCAM offers a new approach to streaming multimodal alignment, potentially applicable to audio streams, sensor streams, and other continuous signals.
- Peak-End memory compression draws on cognitive science, representing an interesting application of cognitive psychology principles to LLM inference optimization.
- The multi-scene, multi-task design of OmniStar provides a standard benchmark for subsequent online video understanding research.

## Rating
- **Novelty**: 8/10 — The SCAM + SVeD paradigm replacing the EOS mechanism is genuinely innovative.
- **Experimental Thoroughness**: 9/10 — Three benchmarks + five tasks + comprehensive ablations.
- **Writing Quality**: 8/10 — Problem motivation is clear and the framework is well-structured.
- **Value**: 8/10 — Both the method and the dataset offer high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Lattice Boltzmann Model for Learning Real-World Pixel Dynamicity](lattice_boltzmann_model_for_learning_real-world_pixel_dynamicity.md)
- [\[NeurIPS 2025\] egoEMOTION: Egocentric Vision and Physiological Signals for Emotion and Personality Recognition in Real-World Tasks](egoemotion_egocentric_vision_and_physiological_signals_for_emotion_and_personali.md)
- [\[NeurIPS 2025\] InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding](infinipot-v_memory-constrained_kv_cache_compression_for_streaming_video_understa.md)
- [\[ICCV 2025\] Online Dense Point Tracking with Streaming Memory](../../ICCV2025/video_understanding/online_dense_point_tracking_with_streaming_memory.md)
- [\[CVPR 2026\] Real-World Point Tracking with Verifier-Guided Pseudo-Labeling](../../CVPR2026/video_understanding/realworld_point_tracking_with_verifierguided_pseud.md)

</div>

<!-- RELATED:END -->
