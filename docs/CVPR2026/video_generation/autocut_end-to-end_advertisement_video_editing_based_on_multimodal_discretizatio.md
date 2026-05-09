---
title: >-
  [Paper Note] AutoCut: End-to-end Advertisement Video Editing Based on Multimodal Discretization and Controllable Generation
description: >-
  [CVPR 2026][Video Generation][video editing] AutoCut proposes an end-to-end advertisement video editing framework that unifies video, audio, and text into a shared discrete token space via Residual Vector Quantization (RQVAE), performs multimodal alignment and supervised fine-tuning on Qwen3-8B, and enables four tasks—clip selection, ordering, script generation, and background music selection—within a single unified model, surpassing GPT-4o baselines across multiple metrics.
tags:
  - CVPR 2026
  - Video Generation
  - video editing
  - Multimodal LLM
  - Residual VQ
  - Advertisement
  - Controllable Generation
date: 2026-05-08
content_hash: e8521f6102daecfa
---

# AutoCut: End-to-end Advertisement Video Editing Based on Multimodal Discretization and Controllable Generation

**Conference**: CVPR 2026
**arXiv**: [2603.28366](https://arxiv.org/abs/2603.28366)
**Code**: [https://github.com/AdAutoCut/Autocut](https://github.com/AdAutoCut/Autocut)
**Area**: Video Understanding / Video Editing
**Keywords**: video editing, Multimodal LLM, Residual VQ, Advertisement, Controllable Generation

## TL;DR
AutoCut proposes an end-to-end advertisement video editing framework that unifies video, audio, and text into a shared discrete token space via Residual Vector Quantization (RQVAE), performs multimodal alignment and supervised fine-tuning on Qwen3-8B, and enables four tasks—clip selection, ordering, script generation, and background music selection—within a single unified model, surpassing GPT-4o baselines across multiple metrics.

## Background & Motivation

**State of the Field**: Short-form video has become the dominant medium for digital advertising, yet the production pipeline—covering scripting, shooting, editing, and post-processing—remains costly and technically demanding.

**Three Major Obstacles in Prior Work**:
   - **Loose multimodal coupling**: Weak alignment among video, audio, and text representations prevents unified reasoning.
   - **Lack of interpretable control**: Models provide no structured or discrete representations, making it difficult to adjust narrative pacing and content emphasis.
   - **Disconnected understanding and generation**: Multimodal understanding and generation are treated as separate processes with inconsistent optimization objectives.

**Opportunities and Limitations of MLLMs**: Multimodal large language models hold promise for unifying perception, understanding, and creation, but are **constrained by context window length**, making large-scale video retrieval and editing difficult.

**Core Idea**: Video and audio features are discretized into tokens via RQVAE and unified with text tokens into a shared vocabulary, enabling the LLM to perform multimodal reasoning and generation within a single token space.

## Method

### Overall Architecture
Two-stage training:
1. **Multimodal Alignment**: The LLM backbone is frozen; only the newly introduced multimodal embedding layers are updated (~700K samples).
2. **Supervised Fine-Tuning (SFT)**: Full-parameter fine-tuning for task-specific behavior learning (~100K curated samples).

At inference: The LLM generates a token sequence → video tokens retrieve nearest-neighbor clips; audio tokens are decoded; text is output directly → ffmpeg compositing produces the final video.

### Key Designs

1. **Multimodal Encoding and Discretization**:

    - **Video Encoder**: ResNet-50 (pretrained with contrastive learning), extracting frame-level semantic embeddings.
    - **Audio Encoder**: PANNs (Wavegram-Logmel-CNN), pretrained on AudioSet.
    - **RQVAE Discretization**: Residual vector quantization compresses continuous embeddings into discrete tokens.
        - Codebook size: $256 \times 8$ (each frame/audio segment encoded as 8 tokens).
        - Reconstruction quality: video cosine similarity 0.89, audio 0.96.
        - Training loss: $\mathcal{L}_{rec} = 1 - \cos(\hat{f}, f)$
    - **Design Motivation**: RQVAE achieves efficient compression through successive residual approximation; the 8-token configuration strikes a favorable balance between reconstruction quality and sequence length.

2. **Unified Token Space**:

    - Video tokens, audio tokens, and text tokens share an expanded vocabulary.
    - The multimodal alignment stage is trained with standard NTP loss: $\mathcal{L}_{NTP} = -\sum_t \log P(x_t | x_{<t})$
    - The LLM backbone is frozen during alignment; only the new embedding layers are updated, ensuring stable training.
    - **Design Motivation**: A unified token space reduces cross-modal reasoning to a sequence modeling problem.

3. **Unified Modeling of Four Tasks**:

    - **Clip Selection**: Selecting relevant segments from a candidate pool (CSA metric).
    - **Clip Ordering**: Arranging segments into a coherent temporal sequence (CRA metric).
    - **Script Generation**: Generating advertisement copy aligned with visual content (SQ + WCD metrics).
    - **Background Music Selection**: Retrieving BGM matching the multimodal context (MSS metric).

4. **Retrieval and Rendering**:

    - Video tokens → FAISS nearest-neighbor search to match clips in the asset library.
    - Audio tokens → decoded or retrieved.
    - ffmpeg splicing, transitions, and subtitle overlay → final MP4 output.

### Training Data
- **Alignment Data**: ~700K filtered advertisement videos (high engagement, with voiceover).
- **SFT Data**: ~100K high-quality curated samples (duration <120s, clips 2–60s, high visual-text relevance assessed by Qwen-VL).
- Data processing: ASR extraction of aligned timestamps, 1fps frame sampling, pydub audio separation.

## Key Experimental Results

### Main Results (364 test videos)

| Method | CSA↑ | CRA↑ | VSC↑ | SQ↑ | WCD↓ | MSS↑ |
|------|------|------|------|-----|------|------|
| Qwen3-8B (Caption) | 0.137 | 0.016 | 0.931 | 80.0 | 5.26 | – |
| Qwen3-8B (Caption+SFT) | 0.569 | 0.030 | 1.123 | 59.2 | 6.82 | – |
| Qwen2.5-VL-32B | 0.665 | 0.025 | **0.998** | 78.3 | 12.51 | – |
| GPT-4o + MGSV | 0.269 | 0.078 | 1.136 | 83.0 | 7.75 | 0.266 |
| **AutoCut** | **0.659** | **0.107** | 1.036 | **84.6** | **3.02** | **0.348** |

### Ablation Study

| Configuration | CSA↑ | CRA↑ | VSC↑ | SQ↑ | WCD↓ |
|------|------|------|------|-----|------|
| SFT only | 0.478 | 0.082 | 1.004 | 83.2 | 4.43 |
| emb+full+sft | **0.717** | 0.058 | 0.967 | 79.0 | 4.50 |
| **emb+sft (Ours)** | 0.659 | **0.107** | **1.036** | **84.6** | **3.02** |

### Key Findings
- AutoCut achieves substantially higher CRA (clip ordering accuracy) than all baselines (0.107 vs. 0.078), demonstrating that tokenized multimodal representations better capture temporal structure.
- WCD (script–video temporal consistency) of 3.02 far outperforms GPT-4o's 7.75, reflecting the temporal alignment advantage of joint multimodal modeling.
- In human evaluation, AutoCut outperforms GPT-4o on all 5 dimensions (88% overall win rate).
- Adding an extra pretraining stage (emb+full+sft) degrades CRA and SQ, indicating that limited-quality pretraining corpora introduce noise.
- Significant cost advantage: processing 100 videos costs AutoCut ~$0.015 vs. GPT-4o ~$2.5.

## Highlights & Insights
- **"Discretization as unification"**: By mapping all modalities into a shared token space via RQVAE, the problem reduces elegantly to next-token prediction.
- The two-stage training strategy (alignment + SFT) outperforms the three-stage variant (alignment + pretraining + SFT), confirming that data quality matters more than data quantity.
- The dual-track design—low-frame-rate tokens for reasoning and high-frame-rate frames for retrieval—balances efficiency and precision.
- The inclusion of BGM selection is a notable contribution, as audio selection has long been neglected in video editing research.

## Limitations & Future Work
- Fine-grained synchronization between video motion and audio rhythm remains imperfect, with occasional desynchronization.
- Control granularity is limited to the clip level; frame-level or emotion-level editing is not supported.
- Although RQVAE achieves high cosine similarity in reconstruction, the effect of information loss may be amplified in downstream tasks.
- Evaluation relies on GPT-4o as a judge (VSC, SQ metrics), introducing potential assessment bias.

## Related Work & Insights
- VC-LLM also employs MLLMs for advertisement video generation but relies on multi-resolution spatiotemporal reasoning.
- MGSV is the only baseline with audio matching capability.
- Compared to "any-modality" LLMs such as NExT-GPT, AutoCut is more focused on the practical constraints of editing scenarios.
- The discretization + retrieval paradigm is generalizable to other video creation contexts (short dramas, vlogs, etc.).

## Rating
- Novelty: ⭐⭐⭐⭐ The unified multimodal discretization framework is a novel approach to advertisement editing, though the core components are relatively mature.
- Experimental Thoroughness: ⭐⭐⭐⭐ Automatic metrics + human evaluation + ablation study, though the test set comprises only 364 videos.
- Writing Quality: ⭐⭐⭐⭐ The framework is described clearly, but definitions of evaluation metrics are somewhat scattered.
- Value: ⭐⭐⭐⭐ The work has direct practical value for automated advertisement video production, with a significant cost advantage.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] U-Mind: A Unified Framework for Real-Time Multimodal Interaction with Audiovisual Generation](u-mind_a_unified_framework_for_real-time_multimodal_interaction_with_audiovisual.md)
- [\[CVPR 2026\] VideoCoF: Unified Video Editing with Temporal Reasoner](videocof_unified_video_editing_with_temporal_reasoner.md)
- [\[ICLR 2026\] LoRA-Edit: Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning](../../ICLR2026/video_generation/lora-edit_controllable_first-frame-guided_video_editing_via_mask-aware_lora_fine.md)
- [\[CVPR 2026\] LAMP: Language-Assisted Motion Planning for Controllable Video Generation](lamp_language-assisted_motion_planning_for_controllable_video_generation.md)
- [\[CVPR 2026\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](flashmotion_few-step_controllable_video_generation_with_trajectory_guidance.md)

<!-- RELATED:END -->
