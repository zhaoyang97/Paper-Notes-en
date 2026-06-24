---
title: >-
  [Paper Note] PAVE: Patching and Adapting Video Large Language Models
description: >-
  [CVPR 2025][Video Understanding][Video LLM adaptation] PAVE proposes a framework to adapt pre-trained Video LLMs via lightweight "patches," integrating side-channel signals such as audio, 3D cues, and multi-view videos into the base model with only about 0.1% additional parameters and computation, outperforming specialized models on tasks like audio-visual QA and 3D QA.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Video LLM adaptation"
  - "side-channel signals"
  - "lightweight adapter"
  - "audio-visual QA"
  - "3D scene understanding"
date: 2026-05-08
content_hash: 20d53d7f2f882267
---

# PAVE: Patching and Adapting Video Large Language Models

**Conference**: CVPR 2025  
**arXiv**: [2503.19794](https://arxiv.org/abs/2503.19794)  
**Code**: [https://github.com/dragonlzm/PAVE](https://github.com/dragonlzm/PAVE)  
**Area**: Multimodal VLM / Video Understanding  
**Keywords**: Video LLM adaptation, side-channel signals, lightweight adapter, audio-visual QA, 3D scene understanding

## TL;DR

PAVE proposes a framework to adapt pre-trained Video LLMs via lightweight "patches," integrating side-channel signals such as audio, 3D cues, and multi-view videos into the base model with only about 0.1% additional parameters and computation, outperforming specialized models on tasks like audio-visual QA and 3D QA.

## Background & Motivation

**Background**: Video Large Language Models (Video LLMs) such as LLaVA-OneVision and Qwen2-VL have demonstrated strong reasoning capabilities in video understanding, but they typically rely solely on video-text inputs. Many downstream tasks require additional modalities or data types (e.g., audio, 3D depth, multi-view video).

**Limitations of Prior Work**: Addressing these tasks usually requires training specialized models from scratch (such as CAT for audio-visual tasks, LLaVA-3D for 3D QA), which is costly and fails to leverage the pre-trained knowledge of Video LLMs. Interestingly, the authors discover that even without using audio/3D data, simply fine-tuning Video LLMs on target datasets achieves comparable or superior performance to these specialized models—indicating that Video LLMs themselves already possess strong reasoning capabilities.

**Key Challenge**: How to efficiently inject side-channel signal information without altering the architecture and weights of pre-trained Video LLMs?

**Goal**: Design a flexible and parameter-efficient framework that adapts pre-trained Video LLMs to various downstream tasks by adding lightweight "patches."

**Key Insight**: Drawing an analogy to the success of LoRA in LLMs and diffusion models—small patches can customize large models and are easy to share. Cross-attention is used to integrate side-channel signals into visual tokens without increasing the number of input tokens to the LLM.

**Core Idea**: Learn a fusion function $g(\cdot)$ that merges side-channel tokens with visual tokens through temporally-aligned cross-attention in a residual manner to update visual tokens, accompanied by LoRA to adapt the LLM.

## Method

### Overall Architecture

Given a video and side-channel signals, their respective tokens are extracted using separate encoders. The "patch" of PAVE consists of two components: (1) a fusion function $g(\cdot)$—a lightweight module based on temporally-aligned cross-attention that merges side-channel tokens into visual tokens; (2) a LoRA module—the low-rank adaptation for the LLM. The fusion result is added to the original visual tokens in a residual manner: $\hat{z}^v(k) = z^v(k) + z^{v|s}(k)$, which are then fed into the LLM.

### Key Designs

1. **Temporal-Aligned Cross-Attention**:

    - **Function**: Merges side-channel signal info into the visual features of the video.
    - **Mechanism**: Uses visual tokens $z^v(k)$ as queries to attend only to side-channel tokens $z^s(k_s)$ ($k_s \in N(k)$) within their temporal neighborhood, achieving local temporal alignment. RoPE positional encodings are incorporated, followed by an MLP and LayerNorm to form a complete Transformer block.
    - **Design Motivation**: Side-channel signals (e.g., audio, 3D point clouds) align naturally in time with video frames. Local attention requires significantly less computation than global attention while maintaining temporal alignment accuracy.

2. **Residual Fusion + Unchanged Token Count**:

    - **Function**: Ensures that the adaptation process is non-intrusive to the base model.
    - **Mechanism**: The output shape of the fusion function is identical to the visual tokens, allowing visual tokens to be updated via simple addition. The number of tokens received by the LLM remains unchanged, meaning computational overhead is virtually unaffected.
    - **Design Motivation**: The LLM is the most computationally expensive part of a Video LLM. Keeping its input size constant means the extra overhead is negligible (~0.1% FLOPs).

3. **Unified Framework for Multiple Side-Channels**:

    - **Function**: Handles diverse signals such as audio, 3D, multi-view, and high-frame-rate videos using the same architecture.
    - **Mechanism**: Represents all side-channel signals uniformly as temporally aligned token sequences $\{z^s(k_s)\}$, varying only the encoders. Audio uses ImageBind, 3D uses the LLaVA-3D encoder, and multi-view/high-frame-rate videos use the Video LLM's own visual encoder.
    - **Design Motivation**: A unified interface enables PAVE to address various downstream tasks in a single framework, even supporting multi-task joint training.

### Loss & Training

Uses standard autoregressive negative log-likelihood loss. Only the fusion function $g(\cdot)$ (~9M parameters) and the LoRA module (~0.5M parameters) are trained, while all parameters of the base model are frozen. Training is conducted for 1-2 epochs.

## Key Experimental Results

### Main Results

| Task | Method | Key Metrics | Extra Params | Extra FLOPs |
|------|------|---------|---------|-----------|
| Audio-Visual QA (AVSD) | COST (SOTA) | CIDEr=108.5 | - | - |
| | **PAVE** | **CIDEr=152.9** | 9M | 0.1 TB |
| Audio-Visual QA (AVQA) | CAT-7B-FT | Acc=92.0 | - | - |
| | **PAVE** | **Acc=93.8** | 9M | 0.1 TB |
| 3D QA (SQA3D) | LLaVA-3D | EM@1=55.6 | - | - |
| | **PAVE** | **EM@1=59.0** | 9M | 0.15 TB |
| Video QA (VideoMME) | LLaVA-OV-7B | Avg=58.2 | - | - |
| | **PAVE** | **Avg=59.9** | 9M | 0.1 TB |

### Ablation Study

| Configuration | AVQA Acc | Description |
|------|---------|------|
| LLaVA-OV-7B (zero-shot) | 85.6 | No fine-tuning, no audio |
| LLaVA-OV-7B-FT (LoRA only) | 90.8 | Fine-tuned but no audio |
| PAVE (w/ audio) | 93.8 | Full model |

### Key Findings

- **Fine-tuning Video LLMs with only video is already very strong**: LLaVA-OV-7B-FT achieves performance close to specialized audio-visual models without audio, indicating that the reasoning capabilities of Video LLMs have been significantly underestimated.
- **Integrating side-channel signals brings consistent improvements**: PAVE consistently outperforms video-only fine-tuning by 2-3%, demonstrating the value of side-channel information.
- **Extremely low overhead**: Achieves SOTA performance with only 0.1% extra parameters and FLOPs.
- PAVE underperforms CAT-7B-FT on the "audio-visual" splitting of Music-AVQA, because Video LLMs tend to prioritize visual cues, struggling when audio-visual conflicts occur.

## Highlights & Insights

- The finding that **"Video LLMs are stronger than you think"** is highly valuable—fine-tuning using only video data yields results close to specialized multimodal models, implying that Video LLMs have implicitly learned cross-modal reasoning.
- The **"patch as plugin"** design philosophy is highly practical: small patches are easy to share and combine, similar to the LoRA ecosystem in the diffusion model community. This paradigm can foster a "Video LLM patch ecosystem."
- The design of **temporally-aligned cross-attention** is simple yet effective—leveraging the natural alignment along the timeline to avoid the high computational cost of global attention.

## Limitations & Future Work

- In scenarios requiring strong audio reasoning (such as audio-visual conflicts), PAVE—which is pre-trained primarily on video—still exhibits weaknesses.
- The parameter budget of LoRA + patch is fixed, which might be insufficient for highly complex new modalities (such as point clouds or tactile data).
- Currently, the patches for each task must be trained independently; a unified multi-task patch remains an open question.
- Future work could explore combining the patch with more Video LLM backbones (such as Qwen2-VL).

## Related Work & Insights

- **vs CAT**: CAT embeds audio-visual understanding directly into model training, requiring a large amount of multimodal data; PAVE adapts existing Video LLMs with just a small patch, presenting a lighter and more flexible approach.
- **vs LLaVA-3D**: LLaVA-3D specifically designs a 3D encoder and training pipeline; PAVE treats its 3D encoder as a side-channel input, achieving superior results with a unified framework.
- **vs SeViLA**: SeViLA adapts to video tasks through a Localizer + Answerer, but requires significant architectural modifications; PAVE's residual patch design is much simpler.

## Rating

- Novelty: ⭐⭐⭐⭐ The unified side-channel adaptation framework is elegantly designed, though the core techniques (cross-attention + LoRA) are not entirely brand new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across four tasks, thorough comparisons, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and solid motivation.
- Value: ⭐⭐⭐⭐⭐ Highly versatile framework, facilitating the direct deployment of Video LLMs in various downstream tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Video Summarization with Large Language Models](video_summarization_with_large_language_models.md)
- [\[CVPR 2025\] On the Consistency of Video Large Language Models in Temporal Comprehension](on_the_consistency_of_video_large_language_models_in_temporal_comprehension.md)
- [\[CVPR 2025\] VoCo-LLaMA: Towards Vision Compression with Large Language Models](voco-llama_towards_vision_compression_with_large_language_models.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](../../NeurIPS2025/video_understanding/fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [\[CVPR 2025\] Efficient Transfer Learning for Video-language Foundation Models](efficient_transfer_learning_for_video-language_foundation_models.md)

</div>

<!-- RELATED:END -->
