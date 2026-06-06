---
title: >-
  [Paper Note] VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding
description: >-
  [CVPR 2026][Video Understanding][Long-form video understanding] VideoARM proposes an agentic reasoning paradigm built upon a Hierarchical Multimodal Memory (HM3) structure. Through an adaptive observe–think–act–memorize…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Long-form video understanding"
  - "agentic reasoning"
  - "hierarchical memory"
  - "coarse-to-fine reasoning"
  - "token efficiency"
date: 2026-05-08
content_hash: 716bea15557b3b86
---

# VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding

**Conference**: CVPR 2026
**arXiv**: [2512.12360](https://arxiv.org/abs/2512.12360)  
**Code**: [https://milvlg.github.io/videoarm/](https://milvlg.github.io/videoarm/)  
**Area**: Video Understanding / LLM Agent
**Keywords**: Long-form video understanding, agentic reasoning, hierarchical memory, coarse-to-fine reasoning, token efficiency

## TL;DR
VideoARM proposes an agentic reasoning paradigm built upon a Hierarchical Multimodal Memory (HM3) structure. Through an adaptive observe–think–act–memorize loop and a coarse-to-fine tool-calling strategy, it surpasses state-of-the-art methods on long-form video understanding benchmarks while reducing token consumption to 1/34 of DVD.

## Background & Motivation

1. **Background**: Long-form video understanding requires capturing fine-grained spatiotemporal details and reasoning over long-range dependencies across videos spanning tens of minutes to hours. Recent advances in long-context MLLMs and cross-modal alignment have provided a foundation for this task. Existing LLM-driven approaches fall into two categories: hand-crafted reasoning pipelines (e.g., LLoVi, VideoTree) and autonomous agentic reasoning (e.g., DVD).

2. **Limitations of Prior Work**: (a) Hand-crafted methods (e.g., VideoTree) follow a fixed pipeline of segmentation → clustering → scoring → tree construction → reasoning, which constrains autonomy and fails to fully leverage the reasoning capacity of stronger backbone models. (b) Agentic methods (e.g., DVD) perform exhaustive preprocessing on all 10-second clips to build a static database, incurring prohibitive token costs (~4M tokens for a 30-minute video); the database cannot be updated during inference.

3. **Key Challenge**: Exhaustive preprocessing wastes tokens and introduces query-irrelevant redundancy, while hand-crafted pipelines suppress the model's autonomous reasoning potential. The core tension is: how to maintain reasoning quality while dramatically reducing token consumption?

4. **Goal**: To design an adaptive, on-demand agentic reasoning paradigm that replaces static exhaustive preprocessing, enabling efficient and flexible long-form video understanding.

5. **Key Insight**: Replace the prebuilt database with a hierarchical memory (sensory → result → working) that the agent constructs dynamically on demand; replace the retrieval paradigm with a coarse-to-fine toolset that allows the agent to progressively narrow the search space through temporal focusing and local analysis.

6. **Core Idea**: Replace the static database with a dynamically constructed three-level memory (HM3), enabling the MLLM agent to explore video content on demand within an observe–think–act–memorize loop, achieving token-efficient long-form video reasoning.

## Method

### Overall Architecture
VideoARM consists of two core components: (1) **Hierarchical Multimodal Memory (HM3)**—a three-level structure (sensory memory, result memory, working memory) that dynamically records the agent's observations and reasoning states; and (2) **a coarse-to-fine video reasoning agent**—driven by a Controller (OpenAI o3), equipped with temporal scoping tools and multimodal understanding tools, performing autonomous reasoning within the observe–think–act–memorize loop. The maximum number of reasoning steps is $N=10$.

### Key Designs

1. **Hierarchical Multimodal Memory (HM3)**

    - **Function**: Serves as the agent's contextual knowledge base, constructed dynamically and continuously updated throughout execution.
    - **Mechanism**: Three-level design—**Sensory Memory** comprises a long-term sensory pool $P_l$ (frames from the currently attended time interval, compressed into 3×2 grids) and a short-term sensory pool $P_s$ (frames and audio from local exploration, cleared after analysis); **Result Memory** records each tool's output and its corresponding time interval, forming a chronologically ordered evidence history; **Working Memory** records the Controller's reasoning trajectory and intent prior to each tool call, externalizing the chain of thought to relieve context pressure.
    - **Design Motivation**: Sensory memory provides the current visual context; result memory allows the agent to reflect on history and avoid redundant actions; working memory addresses context overflow. The three levels abstract from perception → semantics → cognition, forming a complete reasoning scaffold.

2. **Temporal Scoping Tools**

    - **Function**: Adaptively narrow the agent's focus to query-relevant regions of the video.
    - **Mechanism**: **Interval Localizer** uses contextual signals in HM3 to identify the frame interval $T_{long}$ most relevant to the query, adaptively determines the number of sampled frames $N_1$ (30–150), composes frames into compact 3×2 grid images, and updates the long-term sensory pool. **Clip Explorer** performs short-duration fine-grained probing within a local interval $T_{local}$ of the long-term focus (without altering the global focus), samples a fixed number of frames $N_2$ into the short-term sensory pool, and stores the corresponding audio clip.
    - **Design Motivation**: Interval Localizer implements coarse-grained temporal funneling to narrow the focus region; Clip Explorer implements fine-grained hypothesis verification by rapidly collecting local evidence. Together, they realize a coarse-to-fine exploration strategy.

3. **Multimodal Understanding Tools**

    - **Function**: Extract and verify query-relevant evidence from complementary perspectives.
    - **Mechanism**: Three complementary tools—**Scene Snapper** summarizes frames in the long-term sensory pool to produce a scene description $V_C$, providing global semantic abstraction (implemented via GPT-4.1/4o). **Audio Transcriber** transcribes audio in the short-term sensory pool using whisper-1, supplying semantic information when visual cues are insufficient. **Clip Analyzer** analyzes frames in the short-term sensory pool with respect to a sub-question $Q_{sub}$, returning an answer $A_{sub}$ and confidence score $S_{sub}$, providing fine-grained local semantic evidence. After use, results are written to result memory and the short-term sensory pool is cleared.
    - **Design Motivation**: The three tools cover global overview, auditory supplementation, and local detail respectively, allowing the agent to flexibly combine them as needed to balance breadth and depth.

### Controller and Reasoning Loop

The Controller is implemented using OpenAI o3 and follows a streamlined observe–think–act–memorize loop (similar to ReAct but supported by HM3). No rigid workflow or tool-usage rules are predefined, maximizing the exploitation of the MLLM's intrinsic reasoning capacity. In each iteration: observe the global–local context in HM3 → think and generate a reasoning plan $R_t$ → select a tool and execute with specified parameters → write results to HM3. Execution terminates when the step budget $N$ is reached or the Answer action is selected, at which point the final answer is generated.

## Key Experimental Results

### Main Results

| Method | Video-MME Overall | Video-MME Long | LongVideoBench | EgoSchema |
|------|-------------------|----------------|----------------|-----------|
| GPT-4o | 71.9 | 65.3 | 66.7 | 72.2 |
| OpenAI o3 | — | 63.2 | 67.5 | 63.2 |
| DVD | — | 67.3 | 71.6 | 76.6 |
| VideoLucy | 72.5 | 66.8 | — | — |
| **VideoARM (o3+GPT-4.1)** | **80.1** | **75.3** | **73.7** | **78.2** |
| **VideoARM (o3+GPT-4o)** | **82.8** | **81.2** | **78.0** | 76.2 |

### Token Efficiency Comparison

| Method | Theoretical Estimate (30 min / 1 query) | Measured (10 videos / 30 queries) |
|------|------------------------|---------------------------|
| DVD | 3.98M tokens | 64.21M tokens |
| **VideoARM** | **0.08M (1/50 of DVD)** | **1.89M (1/34 of DVD)** |

### Ablation Study

| Configuration | Video-MME Long |
|------|----------------|
| Full (o3 + GPT-4.1) | 76.5 |
| w/o short-term sensory pool | 72.5 (−4.0) |
| w/o long-term sensory pool | 67.0 (−9.5) |
| w/o result memory | Invalid (repetitive loops) |
| w/o working memory | 75.5 (−1.0) |
| Controller context only | 74.5 (−2.0) |
| Controller: GPT-4o | 40.5 |
| Controller: Qwen3-VL | 54.9 |

### Key Findings
- VideoARM achieves 81.2% on Video-MME Long, substantially surpassing DVD's 67.3% (+13.9 pp) while consuming only 1/34 of the tokens.
- The long-term sensory pool is the most critical component; its removal causes a 9.5% drop, indicating that temporal focusing substantially reduces the search space.
- The Controller's reasoning capability is paramount—replacing o3 with GPT-4o as the Controller yields only 40.5%, demonstrating that complex multi-step reasoning requires a strong reasoning model (o3/GPT-5).
- The adaptive frame sampling strategy outperforms fixed sampling (76.5 vs. 74.0), using an average of only 49.8 frames.
- A step budget of $N=10$ is optimal for long videos; short videos do not require as many steps.

## Highlights & Insights
- **Dynamic memory vs. static database is the core innovation**: DVD spends a large number of tokens prebuilding a database and then retrieves from it; VideoARM constructs memory on demand, processing only query-relevant content. This is analogous to lazy evaluation vs. eager evaluation in database systems.
- **The three-level memory design has a cognitive science basis**: The hierarchy of sensory → working → long-term memory mirrors the human cognitive model; the externalization of working memory elegantly addresses LLM context length limitations.
- **The Controller's "degrees of freedom" design philosophy is instructive**: Rather than prescribing a fixed tool-calling order, the system delegates autonomous decision-making to a strong reasoning model, fully unleashing the reasoning potential of o3.

## Limitations & Future Work
- The system relies entirely on API calls (o3 + GPT-4.1/4o + whisper-1), incurring non-negligible costs and introducing dependency on API availability.
- A 10-step reasoning budget may be insufficient for very long videos (>1 hour), yet increasing the budget raises API costs.
- Frame sampling and grid stitching strategies may discard spatial detail.
- Deployment with open-source models is not considered; the poor performance of Qwen3-VL as the Controller indicates that the approach places high demands on model capability.
- Real-time streaming video is not supported.

## Related Work & Insights
- **vs. DVD**: VideoARM essentially replaces "exhaustive preprocessing + retrieval" with "on-demand reasoning + memory," achieving a 34× improvement in token efficiency and +13.9 pp in performance.
- **vs. VideoTree**: VideoTree uses fixed hierarchical clustering, whereas VideoARM employs adaptive tool invocation, affording greater flexibility unconstrained by predefined strategies.
- **vs. VideoLucy**: VideoLucy relies on fixed textual summarization and backtracking mechanisms, while VideoARM maintains a hierarchical multimodal evidence buffer, providing richer information granularity.
- The underlying approach is generalizable to long-document understanding, multimodal RAG, and other scenarios requiring efficient exploration of large-scale information.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — HM3 hierarchical memory and the on-demand reasoning paradigm exhibit solid innovation, though the observe–think–act loop itself is not novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers five benchmarks (Video-MME / LongVideoBench / EgoSchema / MLVU / LVBench) with highly detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with thorough tool design descriptions, though some sections are slightly redundant.
- **Value**: ⭐⭐⭐⭐ — The substantial improvement in token efficiency has practical application value, though reliance on high-end APIs limits deployability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)
- [\[CVPR 2026\] DIvide, then Ground: Adapting Frame Selection to Query Types for Long-Form Video Understanding](divide_then_ground_adapting_frame_selection_to_query_types_for_long-form_video_u.md)
- [\[ACL 2026\] HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding](../../ACL2026/video_understanding/hermes_kv_cache_as_hierarchical_memory_for_efficient_streaming_video_understandi.md)
- [\[AAAI 2026\] TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding](../../AAAI2026/video_understanding/tspo_temporal_sampling_policy_optimization_for_long-form_video_language_understa.md)
- [\[CVPR 2026\] LensWalk: Agentic Video Understanding by Planning How You See in Videos](lenswalk_agentic_video_understanding_by_planning_how_you_see_in_videos.md)

</div>

<!-- RELATED:END -->
