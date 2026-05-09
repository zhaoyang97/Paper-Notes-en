---
title: >-
  [Paper Note] ViLL-E: Video LLM Embeddings for Retrieval
description: >-
  [ACL 2026][Video Understanding][Video Retrieval] This paper proposes ViLL-E, the first Video LLM unified architecture that simultaneously supports text generation and embedding generation. Through three-stage joint generative-contrastive training and an adaptive KV-Former embedding head, it approaches specialist model performance on video retrieval and temporal grounding while maintaining competitive VideoQA results.
tags:
  - ACL 2026
  - Video Understanding
  - Video Retrieval
  - Video LLM
  - Embedding Generation
  - Contrastive Learning
  - Temporal Grounding
date: 2026-05-08
content_hash: 1e1c65080aeb83a2
---

# ViLL-E: Video LLM Embeddings for Retrieval

**Conference**: ACL 2026  
**arXiv**: [2604.12148](https://arxiv.org/abs/2604.12148)  
**Code**: None  
**Area**: Video Understanding  
**Keywords**: Video Retrieval, Video LLM, Embedding Generation, Contrastive Learning, Temporal Grounding

## TL;DR
This paper proposes ViLL-E, the first Video LLM unified architecture that simultaneously supports text generation and embedding generation. Through three-stage joint generative-contrastive training and an adaptive KV-Former embedding head, it approaches specialist model performance on video retrieval and temporal grounding while maintaining competitive VideoQA results.

## Background & Motivation

**State of the Field** Video LLMs (e.g., VideoLLaVA, VideoChat2) excel at text generation tasks like video question answering and captioning, but significantly lag behind specialized models (e.g., QD-DETR, SigLIP, VidLA) on tasks requiring embedding matching, such as text-to-video retrieval (T2V) and moment retrieval.

**Limitations of Prior Work** Current video understanding requires maintaining two separate model stacks: Video LLMs for generation tasks and specialized dual-encoders for retrieval tasks. This not only increases deployment complexity but also prevents sharing representation learning across task types. While NLP research has shown that LLMs can be converted into strong retrieval models through contrastive fine-tuning (e.g., GRIT, E5), such work remains absent in the video domain.

**Root Cause** The autoregressive generation architecture of Video LLMs is inherently unsuited for producing dense embeddings, yet specialized embedding models lack the reasoning and generation capabilities of LLMs. Unifying these two capabilities within a single model presents a key challenge.

**Paper Goals** Design a unified VideoLLM architecture that can both generate text responses and produce high-quality video/text embeddings, achieving competitive performance on retrieval, grounding, and QA tasks.

**Starting Point** Build upon the PaliGemma multimodal LLM by adding a learnable embedding head, and optimize both generation and discrimination capabilities through a three-stage joint training strategy (large-scale pretraining → high-quality pretraining → multi-task fine-tuning).

**Core Idea** The key innovation is an EOS-triggered adaptive embedding generation mechanism—the model first autoregressively generates a variable number of tokens, which are then aggregated by the embedding head into dense embeddings. This allows the model to "think longer" for complex videos and return quickly for simple ones.

## Method

### Overall Architecture
ViLL-E builds upon PaliGemma-3B multimodal LLM, comprising a vision encoder, LLM backbone, and newly added embedding head. Visual tokens and input prompts use bidirectional attention, while autoregressively generated suffixes use causal attention. Upon encountering an `<EOS>` token, all generated tokens are collected and fed into the embedding head to produce dense embeddings. Training proceeds in three stages: large-scale contrastive-generative joint pretraining, high-quality data continuation training, and multi-task fine-tuning.

### Key Designs

1. **KV-Former Embedding Head**:
    - Function: Aggregate variable-length token sequences from LLM output into fixed-dimension dense embeddings
    - Mechanism: Uses LLM output tokens as queries, with $P$ learnable keys and values ("pooling tokens") as a dictionary, performing adaptive weighted aggregation through attention mechanisms. Final embeddings are obtained through MLP projection and mean pooling
    - Design Motivation: Compared to Q-Former's fixed output length, KV-Former supports variable-length input and can adaptively adjust based on video complexity; compared to simple mean pooling or self-attention, it provides a bottleneck representation capacity independent of generation tasks while maintaining parameter efficiency

2. **EOS-Triggered Adaptive Embedding Generation**:
    - Function: Enables the model to automatically determine how many intermediate tokens to generate before producing embeddings based on video complexity
    - Mechanism: The model autoregressively generates tokens until `<EOS>` before extracting embeddings, with the number of generated tokens naturally varying with video complexity. Complex videos require more analysis steps, while simple videos can converge quickly
    - Design Motivation: Fixed-step embedding generation methods cannot adapt to videos of varying complexity; the adaptive mechanism achieves better balance between efficiency and representation quality

3. **Three-Stage Joint Generative-Contrastive Training**:
    - Function: Progressively enhance model capabilities on both generation and embedding tasks
    - Mechanism: Stage 1 jointly trains next-token prediction (generation) and CLIP-style contrastive loss (embedding) on 10M Shutterstock video-caption pairs; Stage 2 continues training on 200K high-quality long captions generated by Claude-3-Sonnet; Stage 3 performs four-task fine-tuning on 100K samples (QA, retrieval, matching, grounding)
    - Design Motivation: Stage 1 establishes basic video-language alignment; Stage 2 addresses overly brief original captions through high-quality detailed descriptions; Stage 3 unlocks downstream task capabilities through multi-task fine-tuning. Ablation studies confirm significant contributions from each stage

### Loss & Training
Four tasks correspond to four losses: (1) Retrieval tasks use CLIP-style in-batch contrastive loss; (2) Captioning/QA use next-token prediction loss; (3) Matching tasks use binary classification cross-entropy; (4) Temporal grounding uses contrastive loss + sliding window hard negative mining (segments with IoU < 0.2 as negatives). The fine-tuning stage uses LoRA for parameter efficiency, with full training for vision projection modules and embedding head.

## Key Experimental Results

### Main Results

| Task/Dataset | Metric | ViLL-E | Prev. Best VideoLLM | Specialist Model |
|------------|------|--------|------------------|---------|
| ActivityNet (grounding) | R@1,IoU=0.5 | **39.4** | 31.2 (LLaVA-ST) | 33.2 (QD-DETR) |
| Charades-STA (grounding) | R@1,IoU=0.5 | **51.5** | 44.8 (LLaVA-ST) | 57.3 (QD-DETR) |
| MSR-VTT (retrieval) | R@1 | **62.5** | N/A | 58.0 (VidLA) |
| DiDeMo (retrieval) | R@1 | **61.4** | N/A | 61.1 (VidLA) |
| MSR-VTT QA | Acc | **65.2** | 63.2 (ST-LLM) | - |
| Composed Retrieval (zero-shot) | R@1 | **53.1** | - | 47.5 (SOTA) |

### Ablation Study

| Config | MSR QA | MSR Retr. | ANet Loc. | Note |
|------|--------|-----------|-----------|------|
| G+C+M (full model) | 65.1 | 62.8 | 39.4 | Joint three supervision signals |
| G+C (no matching) | 63.9 | 60.3 | 39.1 | Matching loss helps retrieval |
| G only (generation only) | 61.3 | 25.1 | 28.7 | Retrieval collapses without contrastive learning |
| C only (contrastive only) | 45.5 | 54.7 | 29.3 | QA significantly drops without generation loss |
| No pretraining | 55.9 | 49.3 | 32.3 | Pretraining crucial for retrieval |

### Key Findings
- ViLL-E achieves 77% average improvement (8+ percentage points) over specialized VideoLLMs on temporal grounding, and exceeds fine-tuned specialist models by 4% on video retrieval
- Generation and contrastive training are complementary: joint training outperforms isolated training on both task types
- Zero-shot novel task capabilities: composed video retrieval exceeds SOTA by 5%, long-text retrieval exceeds SOTA by 2%
- KV-Former design performs best among all embedding head variants
- Two-stage retrieval (embedding retrieval + LLM reranking) provides additional 2% R@1 improvement over single-stage

## Highlights & Insights
- First demonstration that a single VideoLLM can excel at both generation and embedding tasks, breaking the "two model stack" paradigm
- Adaptive embedding generation mechanism elegantly addresses video complexity variation
- Three-stage training strategy is well-designed with clear objectives for each stage and strong ablation study support
- Unlocks novel tasks previously infeasible for VideoLLMs (composed retrieval, long-text retrieval)

## Limitations & Future Work
- Based on PaliGemma-3B with relatively small parameter count, lacks multi-turn dialogue capability
- Training data primarily in English, potentially losing multilingual capabilities
- Missing comparisons with latest general VideoLLMs (e.g., Qwen2.5-VL-72B), with significant model scale gap
- Future extensions could include larger backbones and audio modality

## Related Work & Insights
- NLP works like GRIT and E5 demonstrated that LLMs can be adapted into strong retrieval models; this paper successfully extends this approach to the video domain
- Concurrent works like VLM2Vec and GME are limited to images; ViLL-E is the first unified solution for video
- Provides affirmative experimental evidence for the discussion of "whether a single large model can replace multiple specialized models"

## Rating
- Novelty: ⭐⭐⭐⭐ First unified generation+embedding VideoLLM, ingenious KV-Former design
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 benchmarks, detailed ablations, multiple zero-shot novel task validations
- Writing Quality: ⭐⭐⭐⭐ Clear structure, informative figures and tables
- Value: ⭐⭐⭐⭐ Provides feasible path toward model unification in video understanding

## Rating
- Novelty: To be evaluated
- Experimental Thoroughness: To be evaluated
- Writing Quality: To be evaluated
- Value: To be evaluated

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Log Probability Tracking of LLM APIs](../../ICLR2026/video_understanding/log_probability_tracking_of_llm_apis.md)
- [\[ICCV 2025\] TimeExpert: An Expert-Guided Video LLM for Video Temporal Grounding](../../ICCV2025/video_understanding/timeexpert_an_expert-guided_video_llm_for_video_temporal_grounding.md)
- [\[AAAI 2026\] APVR: Hour-Level Long Video Understanding with Adaptive Pivot Visual Information Retrieval](../../AAAI2026/video_understanding/apvr_hour-level_long_video_understanding_with_adaptive_pivot.md)
- [\[NeurIPS 2025\] MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence](../../NeurIPS2025/video_understanding/muvr_a_multi-modal_untrimmed_video_retrieval_benchmark_with_multi-level_visual_c.md)
- [\[ICLR 2026\] NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks](../../ICLR2026/video_understanding/nerve_nonlinear_eigenspectrum_dynamics_in_llm_feed-forward_networks.md)

<!-- RELATED:END -->
