---
title: >-
  [Paper Note] Goldfish: Vision-Language Understanding of Arbitrarily Long Videos
description: >-
  [ECCV 2024][Video Understanding][Long Video Understanding] This work introduces the Goldfish framework, which achieves efficient understanding of arbitrarily long videos by segmenting them into short clips and utilizing a text-similarity-based retrieval mechanism to select the top-k segments most relevant to the question. It also presents the MiniGPT4-Video short video model and the TVQA-long benchmark for long video evaluation.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Long Video Understanding"
  - "Retrieval-Augmented Generation"
  - "Vision-Language Model"
  - "Video QA"
  - "MiniGPT4-Video"
date: 2026-05-08
content_hash: 95dff71b62752398
---

# Goldfish: Vision-Language Understanding of Arbitrarily Long Videos

**Conference**: ECCV 2024  
**arXiv**: [2407.12679](https://arxiv.org/abs/2407.12679)  
**Code**: [Yes (GitHub)](https://github.com/Vision-CAIR/Goldfish)  
**Area**: Video Understanding  
**Keywords**: Long Video Understanding, Retrieval-Augmented Generation, Vision-Language Model, Video QA, MiniGPT4-Video

## TL;DR

This work introduces the Goldfish framework, which achieves efficient understanding of arbitrarily long videos by segmenting them into short clips and utilizing a text-similarity-based retrieval mechanism to select the top-k segments most relevant to the question. It also presents the MiniGPT4-Video short video model and the TVQA-long benchmark for long video evaluation.

## Background & Motivation

Current LLM-based video understanding models mainly target short videos (on the scale of minutes) and face three core challenges when processing long videos (such as movies and TV shows):

**Noise and Redundant Information**: Similar to the "needle in a haystack" problem in NLP, LLMs tend to overlook key information in excessively long contexts. A large portion of segments in long videos are irrelevant to the given question, making it harder for models to extract meaningful content, particularly after spatial and temporal resolution compression.

**Computational and Memory Limits**: Processing longer videos scales up computational and memory overhead. Existing video LLMs (such as Video-LLaMA and Video-ChatGPT) have inherent constraints on the maximum video length they can handle.

**Lack of Effective Long Video Benchmarks**: Existing long video benchmarks (e.g., LLaMA-VID) generate questions primarily by feeding movie summaries and scripts into language models, ignoring visual content and leading to questions that can be answered using text alone.

Existing attempts, such as MovieChat, expand the context window via memory consolidation, while LLaMA-VID compresses each frame into two tokens. However, these compression strategies suffer from the loss of spatial and temporal visual details, and they still use full video features to predict answers, failing to effectively filter out noise. The key insight of this work is that **accurately identifying query-related video segments is the key to understanding long videos**, leading to the proposal of a retrieval-based framework to address all of the aforementioned challenges.

## Method

### Overall Architecture

The Goldfish framework consists of three core modules: (1) Video Descriptor, (2) Retrieval Module, and (3) Answer Module. The workflow is as follows: segmenting the long video into short clips $\rightarrow$ generating detailed descriptions for each clip $\rightarrow$ retrieving the top-k most relevant clips based on the user query $\rightarrow$ feeding the retrieved results into the answer module to generate the final response.

### Key Designs

1. **Video Descriptor (MiniGPT4-Video)**:

    - **Function**: Segmenting long videos into non-overlapping short clips, generating detailed textual descriptions for each clip, and encoding them into embedding vectors.
    - **Mechanism**: The video frame sequence $V = \{v_1, v_2, \ldots, v_T\}$ is divided into $m$ chunks, with each chunk containing at most $L$ frames (determined by the context window of MiniGPT4-Video). Each frame is encoded using EVA-CLIP, and the visual features are mapped into the LLM text space via a projection layer, where every 4 adjacent visual tokens are concatenated into 1 (reducing tokens from 256 to 64 per frame, a 75% reduction). The generated descriptions $S_1, S_2, \ldots, S_m$ and corresponding subtitles are encoded into embeddings $\{T_{s_i}\}$ and $\{T_{u_i}\}$ using a text encoder (OpenAI text-embedding-3-small).
    - **Design Motivation**: A short video model capable of processing multi-frame inputs is required to provide high-quality semantic descriptions for each clip, while token compression allows more frames to fit within the limited context window.

2. **Retrieval Module**:

    - **Function**: Retrieving the top-k most relevant clips from all clips based on the user query.
    - **Mechanism**: The user query $Q$ is encoded as $T_Q \in \mathbb{R}^d$, and its cosine similarity with all clip descriptions and subtitle embeddings is calculated as: $\frac{K_i \cdot T_Q}{|K_i||T_Q|}$, where $K_i \in \{T_{u_1}, \ldots, T_{u_m}, T_{s_1}, \ldots, T_{s_m}\}$. The top-k clips with the highest similarity are then selected.
    - **Design Motivation**: Filtering out irrelevant content through a retrieval mechanism retains only question-relevant information. This effectively addresses noise redundancy and computational cost issues, allowing the framework to scale to videos of arbitrary length.

3. **Answer Module**:

    - **Function**: Fusing the retrieved clip information with the original query to generate the final answer.
    - **Mechanism**: The original user query and the retrieved clip descriptions (and subtitles) are input as context to Llama2-chat to generate the final response. Ablation studies show that directly feeding descriptions + subtitles to the LLM (Option A, accuracy 41.78%) outperforms passing them through the video model again to obtain new descriptions (Options B/C, around 27.6%), as the latter introduces hallucinations.
    - **Design Motivation**: Directly utilizing existing high-quality descriptions avoids hallucination issues caused by repeatedly processing video frames.

### Loss & Training

MiniGPT4-Video adopts a three-stage training process:
1. **Large-scale Image-Text Pre-training**: Training the linear projection layer on LAION, Conceptual Captions, and SBU to align the visual and LLM text spaces.
2. **Large-scale Video-Text Pre-training**: Utilizing CMD and WebVid datasets, sampling up to 45 frames per video, to learn short video understanding.
3. **Video QA Training**: Fine-tuning with instruction tuning using high-quality QA pairs from the Video-ChatGPT dataset.

## Key Experimental Results

### Main Results

**Long Video Benchmark Comparison (Table 3)**:

| Dataset | Method | Modality | Acc.↑ | Score↑ |
|--------|------|------|-------|--------|
| TVQA-Long | LLaMA-VID | V | 24.63 | 2.16 |
| TVQA-Long | **Ours** | **V** | **28.61** | **2.78** |
| TVQA-Long | LLaMA-VID | V+T | 26.81 | 2.21 |
| TVQA-Long | **Ours** | **V**+**T** | **41.78** | **3.21** |
| Movie QA | LLaMA-VID | V | 24.42 | 2.19 |
| Movie QA | **Ours** | **V** | **28.49** | **2.80** |
| MovieChat | LLaMA-VID | V | 53.2 | 3.81 |
| MovieChat | **Ours** | **V** | **67.6** | **4.23** |

**Short Video Benchmark Comparison (Table 5)**:

| Dataset | Ours (Ours-7B) | LLaMA-VID-7B | Video-ChatGPT | Gain |
|--------|---------------|-------------|----------------|------|
| MSVD (Acc.) | **72.93** | 69.7 | 64.9 | +3.23% |
| MSRVTT (Acc.) | **58.83** | 57.7 | 49.3 | +2.03% |
| TGIF (Acc.) | **67.9** | — | 51.4 | +16.5% |
| TVQA (Acc.) | **36.45** | — | 23.35 | +23.59% |

### Ablation Study

**Importance of the Retrieval Module**:

| Configuration | TVQA-Long Acc. | Description |
|------|---------------|------|
| No Retrieval (direct downsampling to 45 frames) | 25.07% | Close to random guessing (20% baseline for 1-out-of-5 choice) |
| **With Retrieval (top-k)** | **41.78%** | Retrieval yields a +16.71% absolute gain |

**Ablation of Retrieval Inputs (Table 1)**:

| Retrieval Input | TVQA | TVR-Text | TVR-Vision |
|----------|------|----------|------------|
| Subtitles Only | **39.7** | 66.4 | 48.4 |
| Summaries Only | 12.1 | 41.2 | 51.2 |
| Subtitles \| Summaries | 39.5 | **67.2** | **50.8** |

**Ablation of Text Encoders (Table 2)**:

| Text Encoder | Retrieval Acc. | Overall Acc. |
|-----------|---------|---------|
| bert-base-nli | 19.0 | 28.4 |
| paraphrase-MiniLM | 31.9 | 38.03 |
| all-mpnet-base-v2 | 32.5 | 38.33 |
| **OpenAI-text-embedding-3-small** | **46.6** | **41.78** |

### Key Findings

- Retrieval accuracy is linearly and positively correlated with the final accuracy; a better text encoder directly leads to better overall performance.
- Subtitles are more effective for text-based questions, and video summaries are more effective for visual questions; combining both (in an "or" manner) yielding the best results.
- The framework demonstrates robustness across varying video lengths (6/12/24 minutes), as retrieval and overall accuracy do not degrade significantly with increasing video length.

## Highlights & Insights

- **Transferring RAG Concepts to Video Understanding**: Creatively applying the Retrieval-Augmented Generation (RAG) paradigm from the NLP domain to long video understanding serves as an elegant engineering approach.
- **Flexibility of Modular Design**: The Video Descriptor (MiniGPT4-Video) can serve either as a component of Goldfish or independently for short video tasks, achieving state-of-the-art performance in short video scenarios as well.
- **High Practical Value**: The framework can process videos of arbitrary length, supporting real-world scenarios such as movies and TV shows.

## Limitations & Future Work

- Retrieval relies on textual similarity, which might perform poorly for purely visual questions lacking any text clues.
- The answer module uses Llama2-chat instead of a video model, making it unable to leverage the visual information of the retrieved video frames (ablation studies showed that directly using video frames led to a performance drop).
- The clip-level accuracy on TVQA-Long is 41.78%, which, despite outperforming previous SOTA by 14.94%, still leaves considerable room for absolute performance improvement.
- The impact of clip segmentation granularity and top-k configurations across different question types has not been thoroughly validated.

## Related Work & Insights

- Compared to MovieChat (memory consolidation) and LLaMA-VID (token compression), Goldfish takes the path of "retrieval" rather than "compression," holding a distinct preference for maintaining information integrity.
- The three-stage training of MiniGPT4-Video continues and improves upon the architectural design of MiniGPT-v2.
- The decoupled architecture of retrieval and generation facilitates independent upgrades for each submodule.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The concept of applying RAG to long video understanding is novel, though the overall design represents an engineering combination innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covered 4 long video benchmarks and 5 short video benchmarks, with comprehensive ablation studies across retrieval inputs, text encoders, answer modules, and video lengths.
- **Writing Quality**: ⭐⭐⭐⭐ — Clearly structured with well-articulated motivation, although some mathematical notations are slightly redundant.
- **Value**: ⭐⭐⭐⭐ — High practical value; the framework is simple yet effective, providing significant reference value for long video understanding research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] AMEGO: Active Memory from Long EGOcentric Videos](amego_active_memory_from_long_egocentric_videos.md)
- [\[ECCV 2024\] RGNet: A Unified Clip Retrieval and Grounding Network for Long Videos](rgnet_a_unified_clip_retrieval_and_grounding_network_for_long_videos.md)
- [\[CVPR 2025\] ReWind: Understanding Long Videos with Instructed Learnable Memory](../../CVPR2025/video_understanding/rewind_understanding_long_videos_with_instructed_learnable_memory.md)
- [\[CVPR 2025\] BehaviorVLM: Unified Finetuning-Free Behavioral Understanding with Vision-Language Reasoning](../../CVPR2025/video_understanding/behaviorvlm_unified_finetuning-free_behavioral_understanding_with_vision-languag.md)
- [\[ICCV 2025\] Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers](../../ICCV2025/video_understanding/vamba_understanding_hour-long_videos_with_hybrid_mamba-transformers.md)

</div>

<!-- RELATED:END -->
