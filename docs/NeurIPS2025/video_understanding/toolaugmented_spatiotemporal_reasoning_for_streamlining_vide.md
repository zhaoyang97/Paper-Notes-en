---
title: >-
  [Paper Note] Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task
description: >-
  [NeurIPS2025][Video Understanding][VideoQA] This paper proposes a video toolkit comprising 22 tools and the STAR (Spatiotemporal Reasoning) framework, which progressively localizes a 3D Region of Interest (RoI) via an alternating temporal–spatial tool scheduling strategy. The approach improves GPT-4o by 8.2% on VideoMME while substantially reducing the number of processed frames and computational overhead.
tags:
  - NeurIPS2025
  - Video Understanding
  - VideoQA
  - tool-augmented LLM
  - spatiotemporal reasoning
  - agentic framework
  - video toolkit
date: 2026-05-08
content_hash: 3aa0f0f383c996c4
---

# Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task

**Conference**: NeurIPS2025
**arXiv**: [2512.10359](https://arxiv.org/abs/2512.10359)
**Code**: [GitHub](https://github.com/fansunqi/VideoTool)
**Area**: Video Understanding / Multimodal Reasoning
**Keywords**: [VideoQA, tool-augmented LLM, spatiotemporal reasoning, agentic framework, video toolkit]

## TL;DR

This paper proposes a video toolkit comprising 22 tools and the STAR (Spatiotemporal Reasoning) framework, which progressively localizes a 3D Region of Interest (RoI) via an alternating temporal–spatial tool scheduling strategy. The approach improves GPT-4o by 8.2% on VideoMME while substantially reducing the number of processed frames and computational overhead.

## Background & Motivation

**State of the Field**: Video Question Answering (VideoQA) is a critical benchmark for evaluating a model's ability to understand dynamic scenes. Existing approaches fall into two categories: Video-LLMs (e.g., Qwen-VL), which directly process large numbers of frames at high computational redundancy, and tool-augmented LLMs (e.g., DoraemonGPT), which incorporate external tools to assist reasoning.

**Limitations of Prior Work**: Existing tool-augmented methods suffer from three fundamental deficiencies: (1) *unidimensional tooling*—tools focus exclusively on either the temporal or spatial dimension, failing to jointly model intra-frame spatial relationships and inter-frame temporal causality; (2) *imbalanced tool quantity and diversity*—naively stacking tools leads to disordered LLM invocation; (3) *insufficient scheduling strategies*—the absence of effective scheduling mechanisms causes *Toolchain Shortcuts*, whereby the LLM bypasses step-by-step reasoning and directly invokes a general-purpose tool to answer.

**Root Cause**: Progressive refinement is simultaneously required along both the temporal and spatial dimensions, yet unconstrained tool scheduling induces the LLM to take shortcuts.

**Paper Goals**: To construct a comprehensive video toolkit and design an effective spatiotemporal alternating scheduling strategy that resolves the Toolchain Shortcut problem.

**Starting Point**: Inspired by Chain-of-Thought (CoT) reasoning, video understanding is decomposed into alternating iterations of temporal localization and spatial analysis.

**Core Idea**: Temporal and spatial tools are invoked alternately to progressively narrow the spatiotemporal search space toward a 3D RoI—analogous to System 2 thinking applied to visual reasoning.

## Method

### Overall Architecture

STAR is a training-free, extensible agentic reasoning framework. It comprises three tool sets (temporal toolset $T_t$, spatial toolset $T_s$, and general toolset $T_g$) and a core LLM Planner. At initialization, frames are sparsely and uniformly sampled and stored in a visible-frame dictionary $V$. Temporal and spatial tools are then invoked alternately: temporal tools select or reduce the set of frame indices, while spatial tools process designated frames and update the information in $V$. At each step, the LLM Planner assesses whether the accumulated information is sufficient; if not, it continues invoking tools. General-purpose tools are reserved as a final fallback for answer generation.

### Key Designs

1. **22-Tool Video Toolkit**:

    - *Function*: Constructs a comprehensive toolset spanning temporal, spatial, and general dimensions, including Frame Selector (LLM-based frame selection), Temporal Grounding, Object Detection (YOLO/Grounding DINO), Patch Zoomer (region magnification), OCR, Image Captioner, Person ReID, and 15 additional tools totaling 22.
    - *Mechanism*: All tools are encapsulated with a standardized Tool Card interface for plug-and-play integration. Spatial tools support three bounding-box utilization modes: textual description, region magnification, and Set-of-Mark annotation. Temporal tools support both frame-level and segment-level operations (single-frame selection, video clipping, and consecutive segment extraction).
    - *Design Motivation*: Video processing tasks decompose naturally into temporal and spatial dimensions; tools must provide fine-grained capabilities in both. The Tool Card interface ensures extensibility.

2. **Spatiotemporal Alternating Scheduling Strategy (STAR Algorithm)**:

    - *Function*: Constrains the tool chain so that temporal and spatial tools are invoked in alternation, with general-purpose tools serving only as a last resort.
    - *Mechanism*: The algorithm maintains a visible-frame dictionary $V$ (keys: frame indices; values: information collected by each tool). At the first step, either a temporal or spatial tool is automatically selected; subsequent steps alternate between the two dimensions. At each step, the LLM Planner generates a Thought, selects a tool, executes it, observes the output, and updates $V$. When a temporal tool narrows the temporal scope, spatial tools can perform finer analysis over fewer frames; the results of spatial analysis in turn influence subsequent temporal tool selection, forming a closed loop that progressively localizes the 3D RoI.
    - *Design Motivation*: Addresses the Toolchain Shortcut problem—without constraints, LLMs tend to invoke a VLM directly for a one-step answer, bypassing multi-step reasoning. The alternating constraint forces the LLM to engage in progressive spatiotemporal reasoning, analogous to the System 2 mode in CoT.

### Loss & Training

STAR is a completely training-free framework in which all tools are plug-and-play. The full STAR variant uses GPT-4o as the Planner paired with open-source model tools of at most 3B parameters (e.g., QwenVL-2.5-3B). The STAR-mini variant uses GPT-3.5-turbo as the Planner with tools not exceeding 500M parameters (e.g., BLIP), enabling execution on consumer-grade hardware.

## Key Experimental Results

### Main Results

| Method | Params | Frames↓ | Runtime↓ | VideoMME↑ | LongVideoBench↑ |
|--------|--------|---------|----------|-----------|----------------|
| GPT-4o (32 frames) | - | 32 | <30s | 61.8 | 52.6 |
| GPT-4o (1fps/384) | - | 384 | >10min | 71.9 | 66.7 |
| Qwen2.5-VL-7B | 7B | - | - | 65.1 | 53.7 |
| InternVL3-8B | 8B | 64 | <30s | 66.3 | 48.9 |
| Qwen2-VL-72B | 72B | 2fps | 6–8min | 71.2 | - |
| **STAR (ours)** | - | **30.2** | **15.8s** | **70.0 (+8.2)** | **57.2 (+4.6)** |

NExT-QA test set (STAR vs. best baseline AKeyS 78.1%):

| Method | Frames↓ | Causal↑ | Temporal↑ | Descriptive↑ | Overall↑ |
|--------|---------|---------|-----------|-------------|---------|
| AKeyS (GPT-4o) | 7.6 | 72.9 | 79.0 | 86.1 | 78.1 |
| **STAR (GPT-4o)** | **7.2** | **81.1** | **81.5** | **86.3** | **82.1 (+1.2)** |

### Ablation Study

| Scheduling Strategy | Accuracy↑ | Frames↓ | Chain Length↑ | Tool Variety↑ |
|--------------------|----------|---------|--------------|--------------|
| Unconstrained | 61.2 | 112.6 | 2.9 | 1.3 |
| Prompting | 60.4 | 98.7 | 3.6 | 1.9 |
| In-Context Learning | 63.2 | 50.1 | 5.4 | 3.2 |
| Spatiotemporal Decoupled | 68.6 | 40.6 | 5.6 | 3.4 |
| **STAR (Alternating)** | **70.0** | **30.2** | **8.7** | **6.3** |

### Key Findings

- The spatiotemporal alternating strategy outperforms spatiotemporal decoupling by 1.4% while reducing frame usage by 10.4 frames, demonstrating that bidirectional information feedback between the two dimensions is critical.
- The unconstrained strategy results in extremely short tool chains (2.9 steps) utilizing only 1.3 distinct tools, yet consuming 112.6 frames—a textbook instance of the Toolchain Shortcut.
- STAR approaches the performance of GPT-4o with 384 frames and over 10 minutes of runtime using only ~30 frames in ~15 seconds.

## Highlights & Insights

- **Toolchain Shortcut Concept**: This work is the first to formally define and analyze the Toolchain Shortcut phenomenon, exposing a characteristic failure mode of unconstrained LLM agents.
- **Efficiency–Accuracy Pareto Optimality**: STAR achieves 70% on VideoMME with 30 frames, approaching the performance of 72B models that consume thousands of frames.
- **Extensible Architecture**: All 22 tools are plug-and-play; new tools can be integrated through the standardized Tool Card interface.
- **EgoSchema Scalability**: STAR's performance continues to grow as the number of frames increases, demonstrating strong scaling behavior.

## Limitations & Future Work

- The framework still relies on the GPT-4o API as the Planner, incurring cost and latency; replacing it with a lightweight open-source model is an important future direction.
- The current approach focuses solely on visual content and does not incorporate subtitle or audio information, which is a key advantage of models such as Gemini.
- Tool invocation counts scale linearly with API costs; multi-round interaction overhead for long videos may be substantial.
- Tool output quality is constrained by the underlying models (YOLO, BLIP, etc.), introducing error propagation risks.

## Related Work & Insights

- **DoraemonGPT**: Employs text-to-SQL queries over a database of tool outputs, but SQL queries frequently fail; STAR circumvents this issue through the visible-frame dictionary.
- **AKeyS / T***: LLM-driven frame selection methods; this work extends them by incorporating tool support along the spatial dimension.
- **ReAct Framework**: STAR builds upon the ReAct Thought–Action–Observation loop but introduces spatiotemporal alternating constraints.
- **Insight**: The key to tool-augmented agents lies not merely in the number of tools but in the design of the scheduling strategy—appropriate constraints can unlock stronger reasoning capabilities.

## Rating

- Novelty: ⭐⭐⭐⭐ The spatiotemporal alternating scheduling strategy and the Toolchain Shortcut concept offer genuine insight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across four benchmarks with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with intuitive figures and tables.
- Value: ⭐⭐⭐⭐ Provides a reproducible framework and principled design guidelines for video agent systems.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] EgoGazeVQA: Egocentric Gaze-Guided Video Question Answering Benchmark](egogazevqa_egocentric_gaze_guided_video_question_answering.md)
- [\[NeurIPS 2025\] VGEnt: Graph-Based Retrieval-Reasoning-Augmented Generation for Long Video Understanding](vgent_graph-based_retrieval-reasoning-augmented_generation_for_long_video_unders.md)
- [\[NeurIPS 2025\] DSAS: A Universal Plug-and-Play Framework for Attention Optimization in Multi-Document Question Answering](dsas_a_universal_plug-and-play_framework_for_attention_optimization_in_multi-doc.md)
- [\[NeurIPS 2025\] AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](adavideorag_omnicontextual_adaptive_retrievalaugmented_effic.md)
- [\[NeurIPS 2025\] Video Finetuning Improves Reasoning Between Frames](video_finetuning_improves_reasoning_between_frames.md)

<!-- RELATED:END -->
