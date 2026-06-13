---
title: >-
  [Paper Note] ViLL-E: Video LLM Embeddings for Retrieval
description: >-
  [ACL 2026][Video Understanding][Video Retrieval] This paper proposes ViLL-E, the first unified Video LLM architecture that simultaneously supports text generation and embedding generation. Through a three-stage generativ…
tags:
  - "ACL 2026"
  - "Video Understanding"
  - "Video Retrieval"
  - "Video LLM"
  - "embedding generation"
  - "contrastive learning"
  - "temporal localization"
date: 2026-05-08
content_hash: 9c7df79859e0e64d
---

# ViLL-E: Video LLM Embeddings for Retrieval

**Conference**: ACL 2026  
**arXiv**: [2604.12148](https://arxiv.org/abs/2604.12148)  
**Code**: None  
**Area**: Video Understanding  
**Keywords**: Video Retrieval, Video LLM, embedding generation, contrastive learning, temporal localization

## TL;DR
This paper proposes ViLL-E, the first unified Video LLM architecture that simultaneously supports text generation and embedding generation. Through a three-stage generative-contrastive joint training strategy and an adaptive KV-Former embedding head, it approaches the performance of expert models in video retrieval and temporal localization while maintaining competitive VideoQA capabilities.

## Background & Motivation

**Background** Video LLMs (e.g., VideoLLaVA, VideoChat2) perform excellently on generative tasks like video question answering and captioning, but lag significantly behind specialized models (e.g., QD-DETR, SigLIP, VidLA) in tasks requiring embedding matching, such as text-to-video (T2V) retrieval and temporal moment retrieval.

**Limitations of Prior Work** Current video understanding requires maintaining two independent model stacks: Video LLMs for generative tasks and specialized dual-encoders for retrieval tasks. This increases deployment complexity and prevents shared representation learning between the two categories of tasks. While research in the NLP field has shown that LLMs can be transformed into strong retrieval models through contrastive fine-tuning (e.g., GRIT, E5), no such work exists in the video domain.

**Key Challenge** The autoregressive generative architecture of Video LLMs is naturally unsuited for producing dense embeddings, whereas specialized embedding models lack the reasoning and generative capabilities of LLMs. Unifying these two capabilities within a single model is a critical challenge.

**Goal** To design a unified Video LLM architecture capable of generating both textual responses and high-quality video/text embeddings, achieving competitive performance across retrieval, localization, and QA tasks.

**Key Insight** Building upon the PaliGemma multimodal LLM by adding a learnable embedding head and optimizing both generative and discriminative capabilities through a three-stage joint training strategy (large-scale pre-training → high-quality pre-training → multi-task fine-tuning).

**Core Idea** The key innovation is an EOS-triggered adaptive embedding generation mechanism—the model first autoregressively generates a variable number of tokens, which are then fed into the embedding head to be aggregated into a dense embedding. This allows the model to "think longer" for complex videos and return quickly for simple ones.

## Method

### Overall Architecture
ViLL-E is based on the PaliGemma-3B multimodal LLM, consisting of a vision encoder, an LLM backbone, and a newly added embedding head. Bidirectional attention is used for visual tokens and input prompts, while causal attention is used for the autoregressively generated suffixes. When an `<EOS>` token is encountered, all generated tokens are collected and sent to the embedding head to produce a dense embedding. Training is divided into three stages: large-scale generative-contrastive joint pre-training, continued training on high-quality data, and multi-task fine-tuning.

### Key Designs

1.  **KV-Former Embedding Head**:
    - **Function**: Aggregates the variable-length token sequences output by the LLM into fixed-dimensional dense embeddings.
    - **Mechanism**: Uses the LLM output tokens as queries, and $P$ learnable keys and values ("pooling tokens") as a dictionary to perform adaptive weighted aggregation via an attention mechanism. This is followed by MLP projection and mean pooling to obtain the final embedding.
    - **Design Motivation**: Compared to the fixed output length of Q-Former, KV-Former supports variable-length inputs, allowing for adaptive adjustments based on video complexity; compared to simple mean pooling or self-attention, it provides bottleneck representation capacity independent of the generation task while maintaining parameter efficiency.

2.  **EOS-triggered Adaptive Embedding Generation**:
    - **Function**: Enables the model to automatically decide how many intermediate tokens to generate before producing an embedding based on video complexity.
    - **Mechanism**: The model autoregressively generates tokens until `<EOS>` before extracting the embedding. The number of tokens naturally varies with video complexity. Complex videos require more analysis steps, while simple videos can converge quickly.
    - **Design Motivation**: Fixed-step embedding generation methods cannot adapt to videos of different complexities. The adaptive mechanism achieves a better balance between efficiency and representation quality.

3.  **Three-stage Generative-Contrastive Joint Training**:
    - **Function**: Progressively enhances the model's capabilities in both generative and embedding tasks.
    - **Mechanism**: Stage 1 performs joint training of next-token prediction (generative) and CLIP-style contrastive loss (embedding) on 10M Shutterstock video-caption pairs. Stage 2 involves continued training on 200K high-quality long captions generated by Claude-3-Sonnet. Stage 3 performs four-task fine-tuning (QA, retrieval, matching, localization) on 100K samples.
    - **Design Motivation**: Stage 1 establishes foundational video-language alignment. Stage 2 compensates for short original captions with high-quality detailed descriptions. Stage 3 unlocks downstream task capabilities through multi-task fine-tuning. Ablation studies confirm significant contributions from each stage.

### Loss & Training
Four tasks correspond to four losses: (1) Retrieval tasks use a CLIP-style in-batch contrastive loss; (2) Captioning/QA use a next-token prediction loss; (3) Matching tasks use binary cross-entropy; (4) Temporal localization uses contrastive loss combined with sliding window hard negative mining (segments with IoU < 0.2 serve as negatives). LoRA is used during the fine-tuning stage for parameter efficiency, while the vision projection module and embedding head are fully trained.

## Key Experimental Results

### Main Results

| Task/Dataset | Metric | Ours | Prev. SOTA VideoLLM | Expert Model |
| :--- | :--- | :--- | :--- | :--- |
| ActivityNet (Loc.) | R@1, IoU=0.5 | **39.4** | 31.2 (LLaVA-ST) | 33.2 (QD-DETR) |
| Charades-STA (Loc.) | R@1, IoU=0.5 | **51.5** | 44.8 (LLaVA-ST) | 57.3 (QD-DETR) |
| MSR-VTT (Retr.) | R@1 | **62.5** | N/A | 58.0 (VidLA) |
| DiDeMo (Retr.) | R@1 | **61.4** | N/A | 61.1 (VidLA) |
| MSR-VTT QA | Acc | **65.2** | 63.2 (ST-LLM) | - |
| Composed Retr. (ZS) | R@1 | **53.1** | - | 47.5 (SOTA) |

### Ablation Study

| Config | MSR QA | MSR Retr. | ANet Loc. | Description |
| :--- | :--- | :--- | :--- | :--- |
| G+C+M (Full) | 65.1 | 62.8 | 39.4 | Combination of three supervision signals |
| G+C (No Match) | 63.9 | 60.3 | 39.1 | Matching loss assists retrieval |
| G only | 61.3 | 25.1 | 28.7 | Retrieval collapses without contrastive learning |
| C only | 45.5 | 54.7 | 29.3 | QA drops significantly without generative loss |
| w/o Pre-train | 55.9 | 49.3 | 32.3 | Pre-training is crucial for retrieval |

### Key Findings
- ViLL-E improves by an average of 77% (8+ percentage points) over specialized VideoLLMs in temporal localization and surpasses fine-tuned expert models by 4% in video retrieval.
- Generative and contrastive training are complementary: joint training outperforms individual training on both types of tasks.
- Zero-shot capability for new tasks: exceeds SOTA by 5% in composed video retrieval and by 2% in long-text retrieval.
- The KV-Former design performs best among all embedding head variants.
- Two-stage retrieval (embedding retrieval + LLM reranking) provides an additional 2% gain in R@1 over single-stage retrieval.

## Highlights & Insights
- It is demonstrated for the first time that a single Video LLM can excel at both generative and embedding tasks, breaking the "two model stack" paradigm.
- The adaptive embedding generation mechanism elegantly solves the issue of varying video complexity.
- The design of the three-stage training strategy is rational, with clear objectives for each stage supported by thorough ablation studies.
- New tasks previously inaccessible to Video LLMs (e.g., composed retrieval, long-text retrieval) are unlocked.

## Limitations & Future Work
- Based on PaliGemma-3B, the parameter count is relatively small, and it lacks multi-turn dialogue capabilities.
- Training data is primarily English, which may result in a loss of multilingual capability.
- No comparison was made with the latest large-scale general Video LLMs (e.g., Qwen2.5-VL-72B), indicating a gap in model scale.
- Future work could extend to larger backbones and incorporate the audio modality.

## Related Work & Insights
- GRIT and E5 in the NLP field proved that LLMs can be adapted into strong retrieval models; this paper successfully extends this idea to the video domain.
- Concurrent works like VLM2Vec and GME are limited to images; ViLL-E is the first unified solution in the video domain.
- Provides empirical evidence for the discussion on whether a single large model can replace multiple specialized models.

## Rating
- Novelty: ⭐⭐⭐⭐ First unified generative + embedding VideoLLM; ingenious KV-Former design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across 8 benchmarks with detailed ablations and multiple zero-shot tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with information-rich charts.
- Value: ⭐⭐⭐⭐ Provides a feasible path for model unification in the field of video understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] TimeExpert: An Expert-Guided Video LLM for Video Temporal Grounding](../../ICCV2025/video_understanding/timeexpert_an_expert-guided_video_llm_for_video_temporal_grounding.md)
- [\[ICML 2026\] Revisiting Uncertainty: On Evidential Learning for Partially Relevant Video Retrieval](../../ICML2026/video_understanding/revisiting_uncertainty_on_evidential_learning_for_partially_relevant_video_retri.md)
- [\[AAAI 2026\] APVR: Hour-Level Long Video Understanding with Adaptive Pivot Visual Information Retrieval](../../AAAI2026/video_understanding/apvr_hour-level_long_video_understanding_with_adaptive_pivot.md)
- [\[NeurIPS 2025\] MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence](../../NeurIPS2025/video_understanding/muvr_a_multi-modal_untrimmed_video_retrieval_benchmark_with_multi-level_visual_c.md)
- [\[CVPR 2026\] RAGTrack: Language-aware RGBT Tracking with Retrieval-Augmented Generation](../../CVPR2026/video_understanding/ragtrack_language-aware_rgbt_tracking_with_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->
