---
title: >-
  [Paper Note] Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension
description: >-
  [NeurIPS 2025][Object Detection][Retrieval-Augmented Generation] This paper proposes Video-RAG, a training-free, plug-and-play RAG pipeline that extracts visually-aligned auxiliary texts (OCR, ASR…
tags:
  - "NeurIPS 2025"
  - "Object Detection"
  - "Retrieval-Augmented Generation"
  - "Long Video Understanding"
  - "Auxiliary Text"
  - "Plug-and-Play"
  - "Multimodal Alignment"
date: 2026-05-08
content_hash: 36f4c953565d83be
---

# Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension

**Conference**: NeurIPS 2025
**arXiv**: [2411.13093](https://arxiv.org/abs/2411.13093)  
**Code**: [https://github.com/Leon1207/Video-RAG-master](https://github.com/Leon1207/Video-RAG-master)  
**Area**: Object Detection
**Keywords**: Retrieval-Augmented Generation, Long Video Understanding, Auxiliary Text, Plug-and-Play, Multimodal Alignment

## TL;DR

This paper proposes Video-RAG, a training-free, plug-and-play RAG pipeline that extracts visually-aligned auxiliary texts (OCR, ASR, object detection) from video, retrieves relevant content, and feeds it into LVLMs. With an overhead of only ~2K tokens, it improves average Video-MME performance by 2.8% across seven open-source LVLMs, and the 72B model surpasses GPT-4o.

## Background & Motivation

Existing large video-language models (LVLMs) are constrained by limited context lengths when understanding long videos. Two technical approaches have emerged to address this challenge:

**Fine-tuning long-context LVLMs**: LongVA, for example, expands token capacity via pre-training on extended text, but requires massive high-quality data and substantial GPU resources. Experiments show that simply increasing the number of sampled frames can actually hurt performance (LongVA drops from 52.6% to 51.8% when frames increase from 128 to 384).

**GPT-based Agent methods**: Approaches such as VideoAgent and DrVideo employ multi-round interactions with proprietary models, but incur prohibitive computational costs (running VideoAgent on Video-MME takes approximately 20 days and ~\$2,000 in API fees) and depend on closed-source models.

The motivation of this work is to find a **training-free, low-cost solution compatible with arbitrary LVLMs**. The core idea is: rather than increasing the number of visual tokens, supplement insufficient visual information with refined auxiliary texts that are both visually aligned and capable of providing additional information beyond the visual modality (e.g., audio content).

## Method

### Overall Architecture

Video-RAG comprises three stages:
1. **Query Decoupling**: Decompose the user question into retrieval requests for auxiliary texts.
2. **Auxiliary Text Generation & Retrieval**: Generate three types of auxiliary texts in parallel and retrieve relevant content via RAG.
3. **Integration & Generation**: Feed the retrieved auxiliary texts together with the query and video frames into the LVLM.

### Key Designs

1. **Query Decoupling**:

    - The LVLM processes only text input (without accessing video frames) and decomposes the user query into three types of retrieval requests:
        - $R_{asr}$: Requests for speech recognition (extracting audio information)
        - $R_{det}$: Object detection requests (identifying physical entities in the video)
        - $R_{type}$: Object information type requests (location, count, relationships)
    - Output is in JSON format; a NULL value indicates that a particular type of information is not needed.

2. **Auxiliary Text Generation & RAG Retrieval**:

    - **OCR Database**: EasyOCR is applied to each frame for text recognition; texts are encoded into vectors using Contriever and stored in a FAISS index.
    - **ASR Database**: Whisper transcribes the audio; transcripts are chunked, encoded, and stored in FAISS.
    - **DET Database**: Key frames are first filtered by CLIP similarity (threshold $t=0.3$), then APE (open-vocabulary object detection) detects query-relevant objects on the selected frames.
    - At retrieval time, the query and request are encoded with Contriever; FAISS computes similarity scores and retains text chunks exceeding the threshold.

3. **Scene Graph Processing for Object Detection Information**:

    - Raw detection results ("category: [bbox]") are processed into three types of structured information:
        - Object location ($A_{loc}$): Precise description of object categories and coordinates
        - Object count ($A_{cnt}$): Statistics of the number of objects per category
        - Relative spatial relationships ($A_{rel}$): Descriptions of spatial relations among objects
    - Scene graph organization facilitates the LVLM's understanding of object relationships.

4. **Integration & Generation**:

    - OCR, ASR, and DET auxiliary texts are merged in chronological order.
    - They are fed together with the user query and video frames into the LVLM to generate the answer.
    - The entire pipeline operates in a single retrieval round, requiring no multi-turn interaction.

## Key Experimental Results

### Main Results: Video-MME Benchmark

| Model | Params | Frames | w/o Subs | w/ Subs | +Video-RAG | Gain |
|-------|--------|--------|----------|---------|------------|------|
| Video-LLaVA | 7B | 8 | 39.9% | 41.6% | 45.0% | **+3.4%** |
| LLaVA-NeXT-Video | 7B | 16 | 43.0% | 47.7% | 50.0% | +2.3% |
| LongVA | 7B | 128 | 52.6% | 56.0% | 62.0% | **+6.0%** |
| Long-LLaVA | 7B | 64 | 52.9% | 57.8% | 62.6% | **+4.8%** |
| Qwen2-VL | 72B | 32 | 64.9% | 71.9% | 72.9% | +1.0% |
| LLaVA-Video | 72B | 64 | 70.3% | 75.9% | **77.4%** | +1.5% |
| GPT-4o | - | 384 | 71.9% | 77.2% | - | - |

### Ablation Study on Auxiliary Texts

| RAG | DET | OCR | ASR | Short | Medium | Long | Overall |
|-----|-----|-----|-----|-------|--------|------|---------|
| - | - | - | - | 60.3 | 51.4 | 44.1 | 52.0 |
| ✓ | ✓ | - | - | 62.2 | 55.4 | 54.4 | 57.4 |
| ✓ | ✓ | ✓ | - | 64.0 | 56.2 | 55.0 | 58.4 |
| ✓ | - | - | ✓ | 63.0 | 57.3 | 56.4 | 58.9 |
| ✓ | ✓ | ✓ | ✓ | **66.4** | **60.2** | **59.8** | **62.1** |
| - | ✓ | ✓ | ✓ | 64.3 | 58.8 | 56.3 | 59.8 |

### Cross-Benchmark Performance

| Benchmark | Model | Baseline | +Video-RAG | Gain | Reference |
|-----------|-------|----------|------------|------|-----------|
| MLVU | LLaVA-Video-7B | 70.8% | 72.4% | +1.6% | > Oryx-1.5 (32B) |
| MLVU | LLaVA-Video-72B | 73.1% | 73.8% | +0.7% | New SOTA |
| LongVideoBench | LLaVA-Video-7B | 56.6% | 58.7% | +2.1% | - |
| LongVideoBench | LLaVA-Video-72B | 61.9% | 65.4% | +3.5% | > Gemini-1.5-Pro |

### Key Findings

- Video-RAG adds only ~2K tokens (approximately equivalent to 14 frames) and achieves an average performance gain of 2.8%.
- 72B LLaVA-Video + Video-RAG surpasses GPT-4o (77.4% vs. 77.2%).
- ASR yields the largest gain on long videos (+14.7% on the Long category); OCR and DET are more effective on short videos.
- RAG retrieval outperforms feeding all auxiliary texts directly (62.1% vs. 59.8%), demonstrating that retrieval filtering reduces noise.
- The additional GPU memory overhead is only 8 GB, with approximately 5 seconds of extra inference time per question.

## Highlights & Insights

- The design philosophy is highly pragmatic: no fine-tuning, no dependence on closed-source models, and exclusive use of open-source tools (EasyOCR, Whisper, APE, Contriever).
- The role of auxiliary texts is not merely to supply additional information; more importantly, they **facilitate cross-modal alignment** — Grad-CAM visualizations confirm that auxiliary texts help LVLMs focus attention on query-relevant key frames.
- The query decoupling design avoids the waste of generating all types of auxiliary texts for every question.
- The single-round retrieval design is far more efficient than multi-turn agent approaches while maintaining competitive performance.

## Limitations & Future Work

- The quality of auxiliary texts is upper-bounded by the performance of the visual tools — errors from OCR, ASR, and object detection propagate to the final results.
- Adaptive frame selection strategies remain unexplored; the current uniform sampling may miss critical frames.
- For visually dominant tasks (e.g., action recognition), auxiliary texts may offer limited benefit.
- The generalizability of the similarity threshold ($t=0.3$) across different LVLMs and task types has not been thoroughly validated.
- The representation of relative spatial relationships in scene graph processing remains relatively coarse.

## Related Work & Insights

- The approach is conceptually consistent with RAG in NLP, but innovatively extends the retrieval target from text documents to multimodal auxiliary texts automatically extracted from video.
- Compared to agent-based methods such as VideoAgent, Video-RAG replaces multi-turn interaction with single-round retrieval, achieving a favorable balance between efficiency and performance.
- Insight: For multimodal models, cross-modal alignment may be more important than increasing the volume of information within the same modality.
- The idea of using auxiliary texts as "semantic bridges" to help LVLMs better understand visual content can be generalized to other multimodal tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ — Innovatively applies RAG to video understanding with a clean and effective design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Evaluated on 7 LVLMs, 3 benchmarks, with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with detailed method descriptions.
- Value: ⭐⭐⭐⭐⭐ — The plug-and-play nature confers strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Beyond Caption-Based Queries for Video Moment Retrieval](../../CVPR2026/object_detection/beyond_caption-based_queries_for_video_moment_retrieval.md)
- [\[ICCV 2025\] Large-scale Pre-training for Grounded Video Caption Generation](../../ICCV2025/object_detection/large-scale_pre-training_for_grounded_video_caption_generation.md)
- [\[NeurIPS 2025\] MSTAR: Box-Free Multi-Query Scene Text Retrieval with Attention Recycling](mstar_box-free_multi-query_scene_text_retrieval_with_attention_recycling.md)
- [\[ICCV 2025\] VOccl3D: A Video Benchmark Dataset for 3D Human Pose and Shape Estimation under Real Occlusions](../../ICCV2025/object_detection/voccl3d_a_video_benchmark_dataset_for_3d_human_pose_and_shape_estimation_under_r.md)
- [\[AAAI 2026\] Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection](../../AAAI2026/object_detection/temporal_object-aware_vision_transformer_for_few-shot_video_object_detection.md)

</div>

<!-- RELATED:END -->
