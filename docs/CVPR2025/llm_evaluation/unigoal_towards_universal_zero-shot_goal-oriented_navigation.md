---
title: >-
  [Paper Note] UniGoal: Towards Universal Zero-shot Goal-oriented Navigation
description: >-
  [CVPR 2025][LLM Evaluation][Goal-oriented Navigation] This paper proposes UniGoal, a unified zero-shot goal-oriented navigation framework. By representing both scenes and goals uniformly as graph structures and combined with a graph matching-driven multi-stage exploration strategy, it achieves zero-shot navigation for three goal types—object categories, instance images, and text descriptions—within a single model, outperforming task-specific methods.
tags:
  - "CVPR 2025"
  - "LLM Evaluation"
  - "Goal-oriented Navigation"
  - "Zero-shot"
  - "Scene Graph"
  - "Graph Matching"
  - "Large Language Models"
date: 2026-05-08
content_hash: b82b5e9d4ac4b703
---

# UniGoal: Towards Universal Zero-shot Goal-oriented Navigation

**Conference**: CVPR 2025  
**arXiv**: [2503.10630](https://arxiv.org/abs/2503.10630)  
**Code**: None  
**Area**: Robotics  
**Keywords**: Goal-oriented Navigation, Zero-shot, Scene Graph, Graph Matching, Large Language Models

## TL;DR

This paper proposes UniGoal, a unified zero-shot goal-oriented navigation framework. By representing both scenes and goals uniformly as graph structures and combined with a graph matching-driven multi-stage exploration strategy, it achieves zero-shot navigation for three goal types—object categories, instance images, and text descriptions—within a single model, outperforming task-specific methods.

## Background & Motivation

Goal-oriented navigation is divided into three subtasks based on the goal type: Object Goal Navigation (ON), Instance Image Navigation (IIN), and Text-specified Goal Navigation (TN). Existing zero-shot methods (e.g., ESC, Mod-IIN) design task-specific reasoning pipelines for each task, preventing cross-task reuse. Although InstructNav proposed a unified framework, it only supports language-related tasks and cannot handle visual goals (IIN).

Learning-based methods (e.g., GOAT, PSL) achieve general navigation by training policy networks using RL, but they overfit to simulation environments and exhibit poor generalization in real-world settings.

**Core Problem**: How to design a unified reasoning framework that can handle object, image, and text goals in a zero-shot manner without any modifications? The key challenge lies in the massive discrepancy in information format and volume among different goal types.

## Method

### Overall Architecture

UniGoal uniformly represents the scene and the goal as graph structures. It constructs a 3D scene graph $\mathcal{G}_t$ online and converts the three types of goals into a goal graph $\mathcal{G}_g$. At each time step, graph matching is performed, and the exploration strategy is selected based on the matching score. Three stages progress incrementally: zero match $\rightarrow$ partial match $\rightarrow$ perfect match. A blacklist mechanism is employed to prevent redundant exploration. The entire pipeline is based on LLM reasoning and is completely zero-shot.

### Key Design 1: Unified Graph Representation and Three-Metric Graph Matching

**Function**: Uniformly represent different goal types as graph structures and design multi-dimensional matching evaluations.

**Mechanism**: In the graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, nodes represent objects, and edges represent relationships between objects. Three matching metrics are introduced: (1) Node matching $S_N$—obtained through embedding similarity and bipartite matching to get matching node pairs $\mathcal{M}_N$; (2) Edge matching $S_E$—similarly matching edge pairs $\mathcal{M}_E$; (3) Topological matching $S_T$—comparing graph topologies using normalized graph edit distance. The final matching score is defined as $S = (S_N + S_E + S_T)/3$.

**Design Motivation**: The graph structure unifies the information representation of the three goal types (category $\rightarrow$ single-node graph, image $\rightarrow$ graph containing objects and relations, text $\rightarrow$ parsed graph) while preserving more structural information than pure text. The tri-metric matching covers semantics, relations, and topology, providing reliable matching scores to guide exploration.

### Key Design 2: Multi-stage Exploration Strategy

**Function**: Select the most appropriate exploration strategy based on the matching degree.

**Mechanism**: Three-stage progressive exploration—Stage 1 (Zero Match, $S < \sigma_1$): Decompose the goal graph into internally related subgraphs, search for objects related to each subgraph sequentially, and select frontiers using LLM reasoning. Stage 2 (Partial Match, $\sigma_1 \leq S < \sigma_2$): Leverage anchor pairs (matched node pairs) for coordinate projection and alignment, projecting the goal graph into the scene coordinate system through LLM reasoning on spatial relationships to locate unobserved parts. Stage 3 (Perfect Match, $S \geq \sigma_2$): Navigate to the matched central object while performing scene graph correction and goal validation.

**Design Motivation**: The volume of information varies significantly across different matching degrees. Under zero match, it is necessary to maximize the exploration of unknown areas; under partial match, existing anchors can be leveraged to narrow down the search range using spatial reasoning; under perfect match, validation is required to ensure correctness (avoiding false matches). The progressive strategy decomposes the complex navigation problem into step-by-step solvable subproblems.

### Key Design 3: Blacklist Mechanism

**Function**: Prevent redundant exploration in mismatched areas.

**Mechanism**: Maintain a blacklist to record failed matches. When all anchor pairs in Stage 2 fail to proceed to Stage 3, the associated nodes and edges are added to the blacklist; when goal validation fails in Stage 3, all matched pairs are blacklisted. Elements in the blacklist do not participate in subsequent graph matching. If the scene graph correction updates certain nodes/edges, they are removed from the blacklist.

**Design Motivation**: Graph matching yields the maximum similarity result, but the highest-scoring match may still be false. Without a blacklist, the agent would repeatedly navigate to the same incorrect location. The blacklist forces the exploration of new areas and allows corrected elements to "get a second chance."

### Loss & Training

No training loss. A fully zero-shot method that relies on LLM reasoning for decision-making.

## Key Experimental Results

### Main Results (Object/Instance/Text Navigation)

| Method | Training | Universal | ON-MP3D SR | IIN-HM3D SR | TN-HM3D SR |
|------|------|------|-----------|------------|------------|
| SemEXP | Yes | No | 36.0 | — | — |
| OVRL-v2 | Yes | No | — (HM3D: 64.7) | — | — |
| GOAT | Yes | Yes | — | — | — |
| ESC (Zero-shot) | No | No | — | — | — |
| SG-Nav (Zero-shot) | No | No | — | — | — |
| **UniGoal** | **No** | **Yes** | **SOTA** | **SOTA** | **SOTA** |

### Ablation Study

| Component | ON SR | IIN SR | TN SR |
|------|-------|--------|-------|
| Baseline (FBE) | — | — | — |
| + Graph Matching | +Gain | +Gain | +Gain |
| + Multi-stage | +Gain | +Gain | +Gain |
| + Blacklist | +Gain | +Gain | +Gain |

### Key Findings

1. **Single Model with Three-Task SOTA**: UniGoal achieves zero-shot SOTA on all three tasks (ON, IIN, and TN) within the same framework, outperforming task-specific zero-shot methods.
2. **Even Outperforming Supervised Methods**: In some benchmarks, it outperforms generalist methods that require extensive training (e.g., GOAT).
3. **Graph Representation Outperforms Pure Text**: The unified graph representation preserves the structural information of spatial and semantic relations between objects, making it more suitable for LLM reasoning than pure text descriptions.
4. **Coordinate Projection is Effective**: Stage 2 projects coordinates through LLM reasoning on spatial relationships, significantly narrowing down the search space.
5. **Blacklist Prevents Infinite Loops**: Ablation studies show that the blacklist mechanism is crucial for long-horizon navigation.

## Highlights & Insights

- **Choice of Unified Abstraction Layer**: The graph structure is the most natural representation for object-relation information, bridging perception and reasoning better than text (which loses structure) and raw vision (which is difficult to reason over).
- **Explicit Graph Reasoning**: Instead of compressing all information into the LLM context for implicit reasoning, UniGoal fully utilizes spatial information through explicit geometric reasoning such as graph matching and coordinate projection.
- **Three-stage Progressive Strategy**: It elegantly decomposes the navigation problem into "search $\rightarrow$ reason $\rightarrow$ verify," aligning with human cognitive processes for finding objects.

## Limitations & Future Work

- **LLM Reasoning Overhead**: Each step requires multiple LLM calls (graph decomposition, frontier selection, coordinate reasoning), which limits real-time performance.
- **Scene Graph Construction Quality**: Performance depends heavily on the quality of object detection and relation extraction by VLMs, leading to the cascade of perceptual errors.
- **Limited to Static Scenes**: Dynamic objects or scene changes are not considered.
- Future directions include optimizing LLM call efficiency, introducing learnable scene graph construction, and extending to dynamic scenes.

## Related Work & Insights

- **SG-Nav**: An ON method based on scene graphs. This paper extends it to a universal framework of goal graph + graph matching.
- **InstructNav**: A chain-of-thought-driven universal navigation method, but limited to language goals and unable to handle visual goals.
- **Insight**: The "graph matching + LLM reasoning" paradigm provides a unified and scalable solution for multimodal navigation.

## Rating

⭐⭐⭐⭐ — The design of unifying three navigation tasks into a graph matching + multi-stage exploration framework is elegant and effective. The zero-shot performance outperforming supervised methods is convincing. The main limitations are the LLM reasoning overhead and dependency on scene graph quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] A Conditional Probability Framework for Compositional Zero-shot Learning](../../ICCV2025/llm_evaluation/a_conditional_probability_framework_for_compositional_zerosh.md)
- [\[NeurIPS 2025\] Benchmarking Large Language Models for Zero-Shot and Few-Shot Phishing URL Detection](../../NeurIPS2025/llm_evaluation/benchmarking_large_language_models_for_zero-shot_and_few-shot_phishing_url_detec.md)
- [\[ACL 2025\] Language Complexity Measurement as a Noisy Zero-Shot Proxy for Evaluating LLM Performance](../../ACL2025/llm_evaluation/language_complexity_measurement_as_a_noisy_zero-shot_proxy_for_evaluating_llm_pe.md)
- [\[ACL 2026\] Zero-shot Large Language Models for Automatic Readability Assessment](../../ACL2026/llm_evaluation/zero-shot_large_language_models_for_automatic_readability_assessment.md)
- [\[ECCV 2024\] Spherical Linear Interpolation and Text-Anchoring for Zero-shot Composed Image Retrieval](../../ECCV2024/llm_evaluation/spherical_linear_interpolation_and_text-anchoring_for_zero-shot_composed_image_r.md)

</div>

<!-- RELATED:END -->
