---
title: >-
  [Paper Note] VGEnt: Graph-Based Retrieval-Reasoning-Augmented Generation for Long Video Understanding
description: >-
  [NeurIPS 2025][Video Understanding][long video understanding] This paper proposes VGEnt, a graph-based retrieval-reasoning-augmented generation framework that constructs a video knowledge graph to preserve cross-segment…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "long video understanding"
  - "graph RAG"
  - "structured reasoning"
  - "retrieval-augmented generation"
  - "video language model"
date: 2026-05-08
content_hash: f7bc185ae19ad681
---

# VGEnt: Graph-Based Retrieval-Reasoning-Augmented Generation for Long Video Understanding

**Conference**: NeurIPS 2025
**arXiv**: [2510.14032](https://arxiv.org/abs/2510.14032)  
**Code**: [GitHub](https://xiaoqian-shen.github.io/Vgent)  
**Area**: Video Understanding
**Keywords**: long video understanding, graph RAG, structured reasoning, retrieval-augmented generation, video language model

## TL;DR

This paper proposes VGEnt, a graph-based retrieval-reasoning-augmented generation framework that constructs a video knowledge graph to preserve cross-segment semantic relationships, and introduces structured reasoning steps to filter noise and aggregate information. VGEnt consistently improves open-source LVLMs by 3.0%–5.4% across multiple long video understanding benchmarks and outperforms existing video RAG methods by 8.6%.

## Background & Motivation

Core challenges in long video understanding:

**Context window limitations**: A 30-minute video can exceed 200K tokens, surpassing the context limits of most models. Existing approaches address this via sparse sampling or token compression, but inevitably lose fine-grained temporal information.

**Limitations of naive RAG**: Naive RAG splits videos into independent segments for retrieval, disrupting entity continuity and temporal dependencies. In approximately 40% of failure cases, the correct segments are successfully retrieved, yet the model still produces incorrect answers—due to interference from irrelevant information.

**Dependence on closed-source models**: Methods such as VideoAgent and DrVideo rely on GPT-4 for multi-round interaction, incurring high costs and limited flexibility.

## Method

### Overall Architecture

VGEnt comprises four stages: (1) offline video graph construction; (2) graph-based retrieval; (3) structured reasoning; and (4) multimodal augmented generation. The entire pipeline is training-free and can be directly applied to any open-source LVLM.

### Key Designs

1. **Video Graph Construction**:

    - The video is segmented into clips of $K=64$ frames each, with each clip serving as a node in the graph.
    - An LVLM extracts key entities (subjects, actions, scenes) and their descriptions from each clip.
    - Cross-segment entity merging is performed via text embedding similarity (threshold $\tau=0.7$): semantically equivalent entities are unified, and edges are established between nodes sharing the same entities.
    - Graph construction is offline and query-agnostic: once built, it can be reused for multiple questions on the same video.

2. **Graph-based Retrieval**:

    - Keywords $\mathcal{K}$ are extracted from the user query.
    - The similarity between each keyword and every entity description in the global entity set $\mathcal{U}$ is computed; all nodes associated with entities exceeding threshold $\theta=0.5$ are treated as candidates.
    - Top-$N$ ($N=20$) most relevant segments are selected via reranking.
    - Compared to naive RAG's independent per-segment retrieval, the graph structure naturally preserves temporal associations among entities.

3. **Structured Reasoning**:

    - Core finding: in approximately 40% of failure cases, correct segments are retrieved but the model still answers incorrectly (information overload problem).
    - **Divide-and-verify**: the LVLM generates structured sub-queries (yes/no or numerical), and each retrieved segment is verified against them individually.
    - **Noise filtering**: only segments passing at least one sub-query verification are retained (up to $r=5$), effectively eliminating hard negatives.
    - **Information aggregation**: for the filtered segments, all sub-query results are aggregated to produce auxiliary context.

### Loss & Training

VGEnt is a training-free framework with no additional fine-tuning or loss functions. Graph construction uses BAAI/bge-large-en-v1.5 embeddings for similarity computation, and caption extraction uses openai/whisper-large.

## Key Experimental Results

### Main Results

| Model | Size | MLVU Gain | VideoMME (w/ sub.) Gain | LVB Gain |
|------|------|----------|----------------------|---------|
| InternVL2.5 + VGEnt | 2B | +4.4 | +1.6 | +2.8 |
| Qwen2.5-VL + VGEnt | 3B | +4.2 | +2.0 | +3.6 |
| LongVU + VGEnt | 7B | +5.4 | +2.8 | +2.5 |
| Qwen2-VL + VGEnt | 7B | +4.6 | +2.0 | +2.8 |
| LLaVA-Video + VGEnt | 7B | +3.0 | +1.9 | +2.9 |
| Qwen2.5-VL + VGEnt | 7B | +3.3 | +3.2 | +3.7 |

Highlight: Qwen2.5-VL (3B) + VGEnt achieves 70.4% on MLVU, **surpassing its 7B counterpart** (68.8%).

### Ablation Study

| Configuration | MLVU | VideoMME | LVB | Notes |
|------|------|---------|-----|------|
| Qwen2.5-VL baseline | 68.8 | 71.1 | 56.0 | No RAG |
| + NaïveRAG | 65.4 | 68.3 | 56.2 | Naive RAG degrades MLVU |
| + GraphRAG | 69.5 | 72.7 | 57.1 | Graph retrieval outperforms naive RAG |
| + NaïveRAG + SR | 68.6 | 69.8 | 57.3 | Structured reasoning provides limited help |
| + GraphRAG + SR | **72.1** | **74.3** | **59.7** | Synergy of both yields best results |

### Key Findings

- **Naive RAG can be harmful**: MLVU drops from 68.8 to 65.4, indicating that inappropriate retrieval introduces more noise than no retrieval at all.
- **GraphRAG vs. NaïveRAG**: average improvement of 2.9%, with a gap of 4.1% on MLVU.
- **Necessity of structured reasoning**: GraphRAG + SR improves over GraphRAG alone by approximately 2.6%.
- Gains are most significant on long video scenarios (VideoMME long subset), reaching 5.4%.

## Highlights & Insights

- The graph structure's advantage lies in preserving entity-level temporal dependencies—something chunk-based RAG fundamentally cannot achieve.
- The divide-and-verify strategy in structured reasoning is particularly elegant: it decomposes complex questions into simple yes/no sub-problems, allowing the model to respond segment by segment, thereby reducing the reasoning burden on the LVLM.
- The modular design of the framework enables plug-and-play integration with any open-source LVLM, with graph construction incurring only a one-time offline cost.

## Limitations & Future Work

- Graph construction relies on an LVLM to extract entities and descriptions, placing demands on the model's visual comprehension capability.
- Entity merging uses a fixed similarity threshold, which may cause synonymous entities with different surface forms to remain unmerged.
- Structured reasoning introduces multiple rounds of LVLM calls, resulting in higher inference latency.
- For abstract reasoning tasks lacking explicit entities (e.g., sentiment or style analysis), the advantages of graph structure may be limited.

## Related Work & Insights

- **Relation to GraphRAG (NLP)**: The paper transfers graph-augmented RAG ideas from the text domain to video, with the addition of visual entity merging and multimodal verification.
- **Relation to VideoAgent**: Addresses a similar problem without relying on closed-source APIs, achieving a self-contained open-source solution.
- **Inspiration**: The post-hoc structured verification paradigm can be generalized to other multimodal RAG scenarios, such as document understanding and multi-image reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of graph RAG and structured reasoning represents a novel attempt in the video domain.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 models × 3 benchmarks × comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-integrated figures and text.
- Value: ⭐⭐⭐⭐ Plug-and-play framework with strong practical utility; the result of a 3B model surpassing a 7B model is highly compelling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](adavideorag_omnicontextual_adaptive_retrievalaugmented_effic.md)
- [\[CVPR 2026\] RAGTrack: Language-aware RGBT Tracking with Retrieval-Augmented Generation](../../CVPR2026/video_understanding/ragtrack_language-aware_rgbt_tracking_with_retrieval-augmented_generation.md)
- [\[NeurIPS 2025\] VideoLucy: Deep Memory Backtracking for Long Video Understanding](videolucy_deep_memory_backtracking_for_long_video_understanding.md)
- [\[NeurIPS 2025\] Unleashing Hour-Scale Video Training for Long Video-Language Understanding](unleashing_hour-scale_video_training_for_long_video-language_understanding.md)
- [\[NeurIPS 2025\] Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task](toolaugmented_spatiotemporal_reasoning_for_streamlining_vide.md)

</div>

<!-- RELATED:END -->
