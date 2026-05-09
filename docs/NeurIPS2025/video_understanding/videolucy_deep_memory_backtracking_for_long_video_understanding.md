---
title: >-
  [Paper Note] VideoLucy: Deep Memory Backtracking for Long Video Understanding
description: >-
  [NeurIPS 2025][Video Understanding][Long video understanding] This paper proposes VideoLucy, a framework that simulates the human coarse-to-fine recall process via a hierarchical memory structure and an agent-based iterative backtracking mechanism. VideoLucy substantially outperforms existing methods on multiple long video understanding benchmarks, surpassing even commercial models such as GPT-4o.
tags:
  - NeurIPS 2025
  - Video Understanding
  - Long video understanding
  - hierarchical memory
  - agent system
  - memory backtracking
  - video question answering
date: 2026-05-08
content_hash: 49eae87b8b6f95fc
---

# VideoLucy: Deep Memory Backtracking for Long Video Understanding

**Conference**: NeurIPS 2025
**arXiv**: [2510.12422](https://arxiv.org/abs/2510.12422)
**Code**: [https://videolucy.github.io](https://videolucy.github.io)
**Area**: Video Understanding
**Keywords**: Long video understanding, hierarchical memory, agent system, memory backtracking, video question answering

## TL;DR

This paper proposes VideoLucy, a framework that simulates the human coarse-to-fine recall process via a hierarchical memory structure and an agent-based iterative backtracking mechanism. VideoLucy substantially outperforms existing methods on multiple long video understanding benchmarks, surpassing even commercial models such as GPT-4o.

## Background & Motivation

Long video understanding requires a system to maintain comprehensive memory and awareness of nearly all details in a video, as any information loss can lead to inaccurate answers. Although existing agent systems circumvent the long-context limitations of conventional video MLLMs, they face two critical challenges:

**Challenge 1: Frame-level modeling fails to capture temporal context.** Existing agent systems typically generate textual descriptions for individual frames and retrieve key frames through information retrieval loops. However, many questions are closely tied to the temporal context of consecutive frames, making frame-level processing weak at temporal reasoning.

**Challenge 2: Sparse sampling causes loss of critical information.** To reduce the computational cost of dense frame-by-frame captioning, existing systems adopt sparse frame sampling (e.g., VideoTree uses only 0.125 FPS on Video-MME). Even at 1 FPS, a one-hour video requires 3,600 captions, forcing systems to compromise with sparse sampling and thereby losing substantial fine-grained detail.

VideoLucy draws inspiration from the film *Lucy*, in which the protagonist acquires perfect memory and can recall every detail of her life. VideoLucy emulates the human process of moving from vague to precise recollection, achieving comprehensive and in-depth understanding of long videos through hierarchical memory and iterative backtracking.

## Method

### Overall Architecture

VideoLucy comprises three core components: (1) a hierarchical memory structure that provides multi-level video representations with comprehensive information coverage from coarse to fine; (2) agents assigned distinct roles for description, localization, instruction, and answering; and (3) an iterative backtracking mechanism that dynamically retrieves question-relevant deep memories through multi-stage loops.

### Key Designs

1. **Hierarchical Memory Structure**: Three memory levels are defined, with decreasing temporal scope and increasing granularity:

    - **Coarse-grained long-range memory**: Summarized descriptions over large temporal windows (e.g., every 60 seconds)
    - **Fine-grained short-range memory**: Detailed descriptions over shorter temporal segments
    - **Frame-level ultra-fine memory**: Precise descriptions of individual frames or very short clips

   The memory for each video segment is obtained via $m_k = VidCap(v_k, p_k)$, where $VidCap$ denotes a video MLLM (e.g., Qwen2.5-VL-7B) and $p_k$ is the instruction prompt. When $K=1$, this reduces to a global overview; when $K=N$, it corresponds to frame-by-frame description. This structure simultaneously enables multi-level representation and comprehensive information coverage.

2. **Four Agent Types**:

    - **Captioning Agent**: The system's "eyes," converting video segments into textual descriptions using an MLLM.
    - **Localization Agent**: Uses an LLM (DeepSeek-R1) to identify the temporal segments most relevant to the query within the current memory.
    - **Instruction Agent**: Analyzes missing critical information in the current memory and generates guided captioning instructions.
    - **Answering Agent**: Determines whether the current memory is sufficient for a confident answer; if not, outputs an uncertainty flag.

3. **Iterative Backtracking Mechanism (Algorithm 1)**:

    - **Sparse coarse-grained initialization**: Global coarse-grained memory is first generated; the Localization Agent then selects the most query-relevant temporal segments.
    - **Depth and breadth exploration**: In each iteration, the Localization Agent identifies relevant segments → the Instruction Agent analyzes missing information and generates instructions → the Captioning Agent regenerates memory at the current depth (update) and at a deeper level (drill-down) → the current memory list is updated.
    - **Agent-driven loop**: Iteration continues until the Answering Agent deems the information sufficient for a confident response, or until the maximum number of iterations (default: 5) is reached.

### Loss & Training

VideoLucy is a training-free inference-time agent system that directly leverages off-the-shelf open-source models (Qwen2.5-VL-7B for captioning; DeepSeek-R1 for text reasoning). The temporal scope parameters $T_c, T_f, T_{uf}$ are configured differently for each benchmark. Role specialization among agents is achieved through carefully designed prompts.

## Key Experimental Results

### Main Results

| Benchmark | Metric | VideoLucy | Prev. SOTA | Gain |
|-----------|--------|-----------|------------|------|
| Video-MME (Long) | Acc | **66.8** | 65.0 (AdaReTake-72B) | +1.8 |
| Video-MME (Overall) | Acc | **72.5** | 71.9 (GPT-4o) | +0.6 |
| LVBench (Overall) | Acc | **58.8** | 53.3 (AdaReTake-72B) | +5.5 |
| LVBench (KIR) | Acc | **75.6** | 62.2 (AdaReTake-72B) | +13.4 |
| MLVU | M-Avg | **76.1** | 74.7 (VideoChat-Flash-7B) | +1.4 |
| EgoMem (Overall) | Acc | **56.7** | 46.4 (VideoChat-Flash-7B) | +10.3 |

*Note: VideoLucy uses a 7B open-source model yet surpasses methods based on 72B models and commercial systems such as GPT-4o.*

### Ablation Study

| Memory Depth Configuration | Video-MME Long Acc | Notes |
|----------------------------|--------------------|-------|
| Video summary only | ~52 | Severely insufficient information |
| Coarse-grained memory | ~58 | Basic improvement |
| + Fine-grained memory | ~63 | Significant gain |
| + Frame-level ultra-fine memory | **~67** | Best; each memory level contributes |

| Max Iterations | Video-MME Long Acc | Notes |
|----------------|--------------------|-------|
| 1 | ~60 | Insufficient exploration |
| 3 | ~64 | Notable improvement |
| 5 | **~67** | Optimal (default) |
| 7 | ~66 | No additional gain from excess iterations |

### Key Findings

- VideoLucy substantially outperforms prior agent systems on Video-MME (overall +8.5% vs. MemVid), approaching or exceeding closed-source commercial models.
- Performance on the Key Information Retrieval (KIR) task is particularly strong (75.6%), demonstrating the effectiveness of iterative backtracking for precise information retrieval.
- The "Needle-in-A-Video-Haystack" experiment shows that VideoLucy's detail perception is largely invariant to video length.
- Information richness and query relevance improve consistently across backtracking iterations, validating the efficacy of hierarchical exploration.
- The EgoMem benchmark reveals that existing MLLMs perform near random-chance levels on ultra-long video understanding.

## Highlights & Insights

- The hierarchical memory design closely mirrors human cognitive processes, specifically the coarse-to-fine recall pattern.
- VideoLucy requires no additional training; as a purely inference-time framework built on 7B open-source models, it surpasses both 72B and commercial model baselines.
- The proposed EgoMem benchmark (average video length: 6.33 hours) fills a gap in evaluation for ultra-long video understanding.
- The iterative backtracking mechanism provides an interpretable reasoning process, enhancing user trust.

## Limitations & Future Work

- The iterative backtracking incurs high computational costs, requiring multiple MLLM and LLM calls per iteration.
- The maximum number of iterations is a fixed hard limit; more intelligent stopping criteria warrant further investigation.
- System performance is heavily dependent on the captioning quality of the Captioning Agent; captioning errors can propagate through subsequent stages.
- Text-based memory introduces information loss; directly processing visual features may be more effective.

## Related Work & Insights

- **vs. DrVideo**: DrVideo also employs an agent loop to iteratively update key frame information, but operates exclusively at the frame level; VideoLucy models information across multiple temporal scales via hierarchical memory.
- **vs. VideoTree**: VideoTree constructs hierarchical video representations but relies on sparse sampling (0.125 FPS); VideoLucy achieves dense coverage through on-demand drill-down.
- **vs. Conventional Video MLLMs**: Traditional approaches are constrained by maximum context length and sparse sampling; VideoLucy overcomes these limitations through textualized memory and iterative retrieval.
- **vs. LangRepo**: LangRepo maintains a structured language repository but lacks on-demand depth exploration; VideoLucy's backtracking mechanism is more flexible and efficient.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of hierarchical memory and iterative backtracking is effective and intuitive, though the individual components (agent loops, hierarchical structures) are not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across four benchmarks, a new EgoMem benchmark, needle-in-a-haystack experiments, extensive ablations, and qualitative comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, compelling motivation (the *Lucy* film analogy), and rich illustrations.
- Value: ⭐⭐⭐⭐⭐ Establishes a new performance benchmark in long video understanding; the EgoMem benchmark makes an important contribution to the research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] VideoLLaMB: Long Streaming Video Understanding with Recurrent Memory Bridges](../../ICCV2025/video_understanding/videollamb_long_streaming_video_understanding_with_recurrent_memory_bridges.md)
- [\[NeurIPS 2025\] Unleashing Hour-Scale Video Training for Long Video-Language Understanding](unleashing_hour-scale_video_training_for_long_video-language_understanding.md)
- [\[NeurIPS 2025\] VGEnt: Graph-Based Retrieval-Reasoning-Augmented Generation for Long Video Understanding](vgent_graph-based_retrieval-reasoning-augmented_generation_for_long_video_unders.md)
- [\[NeurIPS 2025\] AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](adavideorag_omnicontextual_adaptive_retrievalaugmented_effic.md)
- [\[NeurIPS 2025\] InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding](infinipot-v_memory-constrained_kv_cache_compression_for_streaming_video_understa.md)

</div>

<!-- RELATED:END -->
