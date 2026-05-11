---
title: >-
  [Paper Note] APVR: Hour-Level Long Video Understanding with Adaptive Pivot Visual Information Retrieval
description: >-
  [AAAI 2026][Video Understanding][long video understanding] This paper proposes APVR, a training-free dual-granularity visual information retrieval framework. At the frame level…
tags:
  - "AAAI 2026"
  - "Video Understanding"
  - "long video understanding"
  - "training-free"
  - "keyframe retrieval"
  - "token compression"
  - "dual-granularity retrieval"
date: 2026-05-08
content_hash: a69bc1106365da65
---

# APVR: Hour-Level Long Video Understanding with Adaptive Pivot Visual Information Retrieval

**Conference**: AAAI 2026
**arXiv**: [2506.04953v3](https://arxiv.org/abs/2506.04953v3)
**Code**: No public link
**Area**: Video Understanding / Multimodal LLM
**Keywords**: long video understanding, training-free, keyframe retrieval, token compression, dual-granularity retrieval

## TL;DR
This paper proposes APVR, a training-free dual-granularity visual information retrieval framework. At the frame level, it iteratively retrieves keyframes (up to 1024) via query expansion and spatiotemporal semantic confidence scoring; at the token level, it compresses visual tokens through query-aware attention-driven selection. APVR overcomes memory limitations to process hour-long videos, achieving improvements of up to 9.5%, 4.6%, and 9.7% on LongVideoBench, VideoMME, and MLVU, respectively.

## Background & Motivation
Existing video MLLMs face three major challenges in processing long videos: (1) **Uniform sampling**: dilutes critical information, with a large proportion of irrelevant frames; (2) **Sparse keyframe retrieval**: loses temporal-semantic relationships, making temporal reasoning tasks intractable; (3) **Dense frame processing**: hits the memory wall. Training-based approaches (sequence parallelism, feature compression) require costly multi-stage retraining and are tightly coupled to specific architectures.

**Key Challenge**: The fundamental trade-off between **temporal coverage and computational feasibility**.

## Core Problem
**How can a model efficiently retrieve query-relevant frames and tokens from hour-long videos without retraining, while overcoming MLLM memory constraints and preserving semantic integrity?**

## Method

### Overall Architecture
APVR = Pivot Frame Retrieval (PFR) + Pivot Token Retrieval (PTR), integrated as a plug-in module into arbitrary MLLMs.
- **PFR**: Query expansion → dual-model scoring with CLIP + Grounding-DINO → temporal diffusion → adaptive resampling → selection of $K$ frames
- **PTR**: Query-aware attention scoring → dynamic chunk selection + head-wise soft voting → compressed visual tokens

### Key Designs

1. **Semantic Information Expansion**: An LLM expands the original query into four categories: Objects (detectable entities), Descriptions (entity descriptions/hypernyms), Relations (spatiotemporal/causal relation triples among objects), and Semantics (knowledge-graph semantics). This substantially improves recall in frame retrieval.

2. **Spatiotemporal Semantic Confidence Scoring**: Two complementary models are employed — CLIP computes semantic similarity (cosine distance between text and image embeddings), while Grounding-DINO detects specific objects and models spatial relationships (intra-frame co-occurrence, temporal appearance, etc.). The final score is $\mathcal{S}_t = (1-\lambda) \cdot s_t^{CLIP} + \lambda \cdot s_t^{GD}$. Temporal diffusion propagates high scores to neighboring frames.

3. **Iterative Adaptive Resampling**: Rather than a single scoring pass, multiple rounds of iteration are performed (default: 3), with the sampling stride reduced at each round. The candidate set comprises a high-confidence subset and a high-uncertainty subset (regions with high Shannon entropy), simultaneously exploiting existing knowledge and exploring unvisited regions.

4. **Query-Aware Token Selection**: Within the MLLM, cross-modal attention scores from the text query to visual tokens are used to dynamically select important tokens at varying chunk granularities. Head-wise soft voting mitigates discrepancies across attention heads.

### Loss & Training
Entirely training-free. CLIP and Grounding-DINO use off-the-shelf pretrained weights. Only the LLM is used for query expansion.

## Key Experimental Results

| Base Model | Method | LVB val | VideoMME Long | VideoMME Overall | MLVU dev |
|---|---|---|---|---|---|
| Qwen2-VL-7B | Vanilla | 55.6 | 53.8 | 63.3 | 66.9 |
| Qwen2-VL-7B | +APVR | **60.9**(+9.5%) | **55.1**(+2.4%) | **65.2**(+3.0%) | **73.4**(+9.7%) |
| Qwen2.5-VL-7B | Vanilla | 59.5 | 55.6 | 65.4 | 70.2 |
| Qwen2.5-VL-7B | +APVR | **64.9**(+9.1%) | **59.1**(+6.3%) | **68.4**(+4.6%) | **76.1**(+8.4%) |
| VideoLLaMA3-7B | Vanilla | 59.8 | 54.9 | 66.2 | 73.0 |
| VideoLLaMA3-7B | +APVR | **63.5**(+6.2%) | **58.7**(+6.9%) | **68.1**(+2.9%) | **77.2**(+5.5%) |

APVR + Qwen2.5-VL-7B surpasses GPT-4V (59.1) and Gemini-1.5-Pro (64.0) on LVB.

### Ablation Study
- **Both PFR and PTR are indispensable**: Removing PFR reverts to uniform sampling; removing PTR limits processing to 256 frames (vs. 1024).
- **Contribution of query expansion**: Removing expanded semantic information reduces LVB by approximately 1–2%.
- **Effect of frame count $K$**: Performance increases monotonically as $K$ grows from 32 to 1024; the ability to process 1024 frames is central to APVR's gains.
- **Optimal iteration count $P=3$**: $P=2$ is insufficiently fine-grained; $P=5$ leads to overfitting.
- **Optimal $\lambda=0.5$**: Balanced weighting of CLIP and Grounding-DINO scores yields the best performance.

## Highlights & Insights
- **Training-free and plug-and-play**: No modification to MLLM parameters; compatible with arbitrary backbone models, making it highly practical for the rapidly evolving MLLM ecosystem.
- **Complementarity of dual-granularity retrieval**: Frame-level retrieval addresses *which frames to observe*, while token-level retrieval addresses *what to attend to within each frame*; together they overcome the memory wall.
- **Iterative resampling with uncertainty exploration**: Keyframes are progressively refined rather than selected in a single pass; high-uncertainty regions are also explored to avoid local optima.
- **Query expansion substantially enhances retrieval**: Expanding a simple question into objects, relations, and semantics improves the precision of CLIP/Grounding-DINO matching.
- **7B model surpasses GPT-4V**: The training-free approach with a 7B model achieves 64.9% on LVB, exceeding GPT-4V at 59.1%.

## Limitations & Future Work
- **Dependency on external models**: The pipeline requires three additional models — CLIP, Grounding-DINO, and an LLM — for query expansion and frame scoring, increasing system complexity.
- **Latency**: Processing an hour-long video takes approximately two minutes per query, which may be insufficient for real-time applications.
- **Restricted to multiple-choice QA**: All benchmarks adopt multiple-choice formats; performance on open-ended question answering remains unexplored.
- **Query expansion quality depends on the LLM**: Erroneous objects or relations generated during expansion can mislead frame retrieval.
- **Grounding-DINO detection limitations**: Retrieval of abstract concepts or actions (rather than physical objects) may be suboptimal.

## Related Work & Insights

| Method | Type | Mechanism | Key Difference from APVR |
|---|---|---|---|
| **AKS** | Training-free | Keyframe selection + information pre-filtering | Frame-level only; no token-level compression |
| **QuoTA** | Training-free | Query-aware token allocation | Token-level only; no frame-level retrieval |
| **LongVILA** | Training-based | 5-stage training + sequence parallelism | Requires substantial training resources; architecture-specific |
| **Video-XL-Pro** | Training-based | Learned compression module | Requires retraining |

The core advantage of APVR lies in its **dual-granularity** design — joint optimization at both the frame and token levels — while remaining entirely training-free.

**Broader Insights**:
- The iterative resampling and uncertainty-guided exploration paradigm is transferable to other information retrieval tasks in agentic settings (e.g., RAG, document understanding).
- The query expansion strategy (objects + relations + semantics) offers general utility for video retrieval and video grounding tasks.
- Training-free methods serve as a compelling alternative to parameter scaling in an era of rapid model iteration.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The dual-granularity retrieval framework and iterative resampling strategy are noteworthy, though individual components rely on existing tools.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three benchmarks, three baseline MLLMs, comprehensive ablation analysis, and qualitative comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Architecture diagrams and algorithmic descriptions are clear; motivation is well articulated.
- **Value**: ⭐⭐⭐⭐⭐ Training-free, plug-and-play, and surpassing GPT-4V — the practical value is exceptionally high.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding](../../ICLR2026/video_understanding/floc_facility_location-based_efficient_visual_token_compression_for_long_video_u.md)
- [\[AAAI 2026\] MambaMia: State-Space Hierarchical Compression for Hour-Long Video Understanding in Large Multimodal Models](state-space_hierarchical_compression_with_gated_attention_an.md)
- [\[CVPR 2026\] VSI: Visual-Subtitle Integration for Keyframe Selection to Enhance Long Video Understanding](../../CVPR2026/video_understanding/vsi_visual-subtitle_integration_for_keyframe_selection_to_enhance_long_video_un.md)
- [\[NeurIPS 2025\] AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](../../NeurIPS2025/video_understanding/adavideorag_omnicontextual_adaptive_retrievalaugmented_effic.md)
- [\[AAAI 2026\] ReaSon: Reinforced Causal Search with Information Bottleneck for Video Understanding](reason_reinforced_causal_search_with_information_bottleneck_for_video_understand.md)

</div>

<!-- RELATED:END -->
