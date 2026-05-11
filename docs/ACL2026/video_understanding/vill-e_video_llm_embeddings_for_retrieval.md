---
title: >-
  [Paper Note] ViLL-E: Video LLM Embeddings for Retrieval
description: >-
  [ACL 2026][Video Understanding][Video Retrieval] This paper proposes ViLL-E, the first unified Video LLM architecture supporting both text generation and embedding generation. Through a three-stage joint generative-contr…
tags:
  - "ACL 2026"
  - "Video Understanding"
  - "Video Retrieval"
  - "Video LLM"
  - "Embedding Generation"
  - "Contrastive Learning"
  - "Temporal Grounding"
date: 2026-05-08
content_hash: 63881a3b6a8f538a
---

# ViLL-E: Video LLM Embeddings for Retrieval

**Conference**: ACL 2026
**arXiv**: [2604.12148](https://arxiv.org/abs/2604.12148)
**Code**: None
**Area**: Video Understanding
**Keywords**: Video Retrieval, Video LLM, Embedding Generation, Contrastive Learning, Temporal Grounding

## TL;DR
This paper proposes ViLL-E, the first unified Video LLM architecture supporting both text generation and embedding generation. Through a three-stage joint generative-contrastive training strategy and an adaptive KV-Former embedding head, ViLL-E approaches expert models on video retrieval and temporal grounding while maintaining competitive performance on VideoQA.

## Background & Motivation

**State of the Field** Video LLMs (e.g., VideoLLaVA, VideoChat2) excel at text generation tasks such as video question answering and captioning, but fall far behind specialized models (e.g., QD-DETR, SigLIP, VidLA) on embedding-based tasks such as text-to-video retrieval (T2V) and moment retrieval.

**Limitations of Prior Work** Current video understanding requires maintaining two separate model stacks: a Video LLM for generative tasks and a dedicated dual-encoder for retrieval tasks. This not only increases deployment complexity but also prevents shared representation learning across the two task categories. Although NLP research has demonstrated that LLMs can be adapted into strong retrieval models via contrastive fine-tuning (e.g., GRIT, E5), no analogous work exists for the video domain.

**Root Cause** The autoregressive generation architecture of Video LLMs is inherently ill-suited for producing dense embeddings, whereas dedicated embedding models lack the reasoning and generation capabilities of LLMs. Unifying both capabilities within a single model is therefore the central challenge.

**Paper Goals** To design a unified Video LLM architecture that can both generate textual responses and produce high-quality video/text embeddings, achieving competitive performance across retrieval, temporal grounding, and QA tasks.

**Starting Point** A learnable embedding head is added on top of the PaliGemma multimodal LLM, and a three-stage joint training strategy (large-scale pretraining → high-quality pretraining → multi-task fine-tuning) is employed to simultaneously optimize generative and discriminative capabilities.

**Core Idea** The key innovation is an EOS-triggered adaptive embedding generation mechanism: the model first autoregressively generates a variable number of tokens, which are then fed into the embedding head and aggregated into a dense embedding. This allows the model to "think longer" for complex videos while returning quickly for simpler ones.

## Method

### Overall Architecture
ViLL-E is built upon PaliGemma-3B and comprises a visual encoder, an LLM backbone, and a newly introduced embedding head. Visual tokens and input prompts attend bidirectionally, while autoregressively generated suffixes use causal attention. Upon encountering an `<EOS>` token, all generated tokens are collected and passed to the embedding head to produce a dense embedding. Training proceeds in three stages: large-scale joint contrastive-generative pretraining, high-quality data continued training, and multi-task fine-tuning.

### Key Designs

1. **KV-Former Embedding Head**:
    - **Function**: Aggregates the variable-length token sequence output by the LLM into a fixed-dimensional dense embedding.
    - **Mechanism**: The LLM output tokens serve as queries, while $P$ learnable keys and values ("pooling tokens") act as a dictionary; an attention mechanism performs adaptive weighted aggregation. The result is then projected via an MLP and mean-pooled to obtain the final embedding.
    - **Design Motivation**: Unlike Q-Former's fixed output length, KV-Former supports variable-length inputs and can adaptively adjust to video complexity. Compared to simple mean pooling or self-attention, it provides a bottleneck representation capacity decoupled from the generation task while remaining parameter-efficient.

2. **EOS-Triggered Adaptive Embedding Generation**:
    - **Function**: Allows the model to automatically determine how many intermediate tokens to generate before producing an embedding, based on video complexity.
    - **Mechanism**: Before extracting an embedding, the model autoregressively generates tokens until `<EOS>`; the number of generated tokens varies naturally with video complexity. Complex videos require more analytical steps, while simple videos converge quickly.
    - **Design Motivation**: Fixed-step embedding generation cannot adapt to videos of varying complexity; the adaptive mechanism achieves a better balance between efficiency and representation quality.

3. **Three-Stage Joint Generative-Contrastive Training**:
    - **Function**: Progressively improves the model's capabilities on both generative and embedding tasks.
    - **Mechanism**: Stage 1 jointly trains next-token prediction (generation) and CLIP-style contrastive loss (embedding) on 10M Shutterstock video-caption pairs. Stage 2 continues training on 200K high-quality long captions generated by Claude-3-Sonnet. Stage 3 performs four-task fine-tuning (QA, retrieval, matching, and grounding) on 100K samples.
    - **Design Motivation**: Stage 1 establishes foundational video-language alignment; Stage 2 compensates for the overly brief nature of raw captions through high-quality detailed descriptions; Stage 3 unlocks downstream task capabilities via multi-task fine-tuning. Ablation studies confirm that each stage contributes substantially.

### Loss & Training
Four task-specific losses are employed: (1) CLIP-style in-batch contrastive loss for retrieval; (2) next-token prediction loss for captioning/QA; (3) binary cross-entropy for video-text matching; and (4) contrastive loss with sliding-window hard negative mining (segments with IoU < 0.2 serve as negatives) for temporal grounding. During fine-tuning, LoRA is applied for parameter efficiency, while the visual projection module and embedding head are trained in full.

## Key Experimental Results

### Main Results

| Task / Dataset | Metric | ViLL-E | Prev. Best VideoLLM | Expert Model |
|---|---|---|---|---|
| ActivityNet (Grounding) | R@1, IoU=0.5 | **39.4** | 31.2 (LLaVA-ST) | 33.2 (QD-DETR) |
| Charades-STA (Grounding) | R@1, IoU=0.5 | **51.5** | 44.8 (LLaVA-ST) | 57.3 (QD-DETR) |
| MSR-VTT (Retrieval) | R@1 | **62.5** | N/A | 58.0 (VidLA) |
| DiDeMo (Retrieval) | R@1 | **61.4** | N/A | 61.1 (VidLA) |
| MSR-VTT QA | Acc | **65.2** | 63.2 (ST-LLM) | — |
| Composed Retrieval (Zero-shot) | R@1 | **53.1** | — | 47.5 (SOTA) |

### Ablation Study

| Configuration | MSR QA | MSR Retr. | ANet Loc. | Note |
|---|---|---|---|---|
| G+C+M (Full) | 65.1 | 62.8 | 39.4 | Joint supervision from all three signals |
| G+C (w/o Matching) | 63.9 | 60.3 | 39.1 | Matching loss benefits retrieval |
| G only (Generation only) | 61.3 | 25.1 | 28.7 | Retrieval collapses without contrastive learning |
| C only (Contrastive only) | 45.5 | 54.7 | 29.3 | QA drops sharply without generation loss |
| w/o Pretraining | 55.9 | 49.3 | 32.3 | Pretraining is critical for retrieval |

### Key Findings
- ViLL-E outperforms dedicated Video LLMs on temporal grounding by an average of 77% (8+ percentage points) and surpasses fine-tuned expert models on video retrieval by up to 4%.
- Generative and contrastive training are complementary: joint training outperforms either objective alone on both task categories.
- Zero-shot generalization to new tasks: composed video retrieval exceeds SOTA by 5%, and long-text retrieval exceeds SOTA by 2%.
- The KV-Former design outperforms all other embedding head variants.
- Two-stage retrieval (embedding retrieval + LLM reranking) yields an additional R@1 gain of 2% over single-stage retrieval.

## Highlights & Insights
- This work is the first to demonstrate that a single Video LLM can excel at both generative and embedding tasks simultaneously, challenging the prevailing paradigm of maintaining two separate model stacks.
- The adaptive embedding generation mechanism elegantly addresses the problem of variable video complexity.
- The three-stage training strategy is well-motivated, with each stage having a clear objective supported by ablation evidence.
- The approach unlocks new tasks previously inaccessible to Video LLMs, including composed retrieval and long-text retrieval.

## Limitations & Future Work
- ViLL-E is built on PaliGemma-3B, which has a relatively small parameter count and lacks multi-turn dialogue capability.
- Training data is primarily in English, potentially compromising multilingual generalization.
- Comparisons with the latest general-purpose Video LLMs (e.g., Qwen2.5-VL-72B) are absent, leaving a substantial gap in model scale unaddressed.
- Future work could extend the approach to larger backbones and incorporate the audio modality.

## Related Work & Insights
- GRIT and E5 in NLP have demonstrated that LLMs can be repurposed as strong retrieval models; this paper successfully extends that paradigm to the video domain.
- Concurrent works such as VLM2Vec and GME are limited to images; ViLL-E is the first unified solution for the video domain.
- The paper provides affirmative experimental evidence for the broader question of whether a single large model can replace multiple specialized models.

## Rating
- Novelty: ⭐⭐⭐⭐ First unified generation + embedding Video LLM; KV-Former design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Eight benchmarks, detailed ablations, and multiple zero-shot task evaluations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with informative figures and tables.
- Value: ⭐⭐⭐⭐ Provides a viable path toward model unification in video understanding.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] RARE: Redundancy-Aware Retrieval Evaluation Framework for High-Similarity Corpora](rare_redundancy-aware_retrieval_evaluation_framework_for_high-similarity_corpora.md)
- [\[ICLR 2026\] Log Probability Tracking of LLM APIs](../../ICLR2026/video_understanding/log_probability_tracking_of_llm_apis.md)
- [\[ICCV 2025\] TimeExpert: An Expert-Guided Video LLM for Video Temporal Grounding](../../ICCV2025/video_understanding/timeexpert_an_expert-guided_video_llm_for_video_temporal_grounding.md)
- [\[ICLR 2026\] NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks](../../ICLR2026/video_understanding/nerve_nonlinear_eigenspectrum_dynamics_in_llm_feed-forward_networks.md)
- [\[AAAI 2026\] APVR: Hour-Level Long Video Understanding with Adaptive Pivot Visual Information Retrieval](../../AAAI2026/video_understanding/apvr_hour-level_long_video_understanding_with_adaptive_pivot.md)

</div>

<!-- RELATED:END -->
