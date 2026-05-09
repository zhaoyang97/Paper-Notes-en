---
title: >-
  [Paper Note] Verification-Guided Context Optimization for Tool Calling via Hierarchical LLMs-as-editors
description: >-
  [AAAI 2026][Tool Calling] This paper proposes the VGCO framework, which employs LLMs as hierarchical editors to iteratively optimize tool documentation and knowledge base context through verification-guided signals, achieving significant improvements in retrieval recall, tool selection, and parameter filling accuracy in large-scale tool calling scenarios.
tags:
  - AAAI 2026
  - Tool Calling
  - Context Optimization
  - LLM Editor
  - Knowledge Base Retrieval
  - Document Optimization
date: 2026-05-08
content_hash: 450791fa7616791c
---

# Verification-Guided Context Optimization for Tool Calling via Hierarchical LLMs-as-editors

**Conference**: AAAI 2026
**arXiv**: [2512.13860](https://arxiv.org/abs/2512.13860)
**Code**: None
**Area**: Other
**Keywords**: Tool Calling, Context Optimization, LLM Editor, Knowledge Base Retrieval, Document Optimization

## TL;DR

This paper proposes the VGCO framework, which employs LLMs as hierarchical editors to iteratively optimize tool documentation and knowledge base context through verification-guided signals, achieving significant improvements in retrieval recall, tool selection, and parameter filling accuracy in large-scale tool calling scenarios.

## Background & Motivation

1. **Background**: Tool calling has become a key mechanism for extending LLM capabilities, enabling models to invoke external APIs/tools for real-time information retrieval or complex task execution. Existing approaches fall into two broad categories: fine-tuning-based methods (e.g., xLAMs, ToolBench) and tuning-free methods (relying on in-context learning).

2. **Limitations of Prior Work**: Tool documentation and knowledge base content are typically written for human readers and are poorly aligned with the comprehension needs of LLMs. In industrial-scale settings, enterprises manage hundreds of API tools, and issues such as overlapping functionality and ambiguous boundaries are particularly severe, leading to frequent tool calling failures.

3. **Key Challenge**: Existing context optimization frameworks (e.g., DSPy, SAMMO, Promptim) suffer from three fundamental limitations: stateless operation (no awareness of the current context state), lack of a structured action space (relying solely on token-level adjustments), and the absence of domain-specific evaluation mechanisms.

4. **Goal**: To systematically optimize tool-related documentation and knowledge base context so that it better aligns with LLM tool calling requirements, especially in large-scale industrial scenarios.

5. **Key Insight**: The authors observe that many tool calling failures stem not from insufficient model reasoning capabilities, but from incomplete, inconsistent, or improperly structured contextual information.

6. **Core Idea**: Tool context is decomposed into a retrieval layer, a tool layer, and a parameter layer. Verification signals guide LLM editors to perform targeted iterative optimization at each layer.

## Method

### Overall Architecture

VGCO operates in two phases: (1) **Evaluation Phase**: structured signals are collected to identify and diagnose contextual failure cases; (2) **Optimization Phase**: hierarchical LLM editors generate revised tool documentation and context, which are integrated into the production system following offline verification. The overall pipeline forms a closed loop: user query → validation dataset evaluation → error identification → LLM editor correction → verified improvement → knowledge base update.

### Key Designs

1. **Hierarchical LLMs-as-Editors**:
    - **Function**: Decomposes tool context optimization into three layers — retrieval ($D_r$), tool ($D_t$), and parameter ($D_p$) — each handled by a corresponding editor $\mathcal{M}_r, \mathcal{M}_t, \mathcal{M}_p$.
    - **Mechanism**: Optimization proceeds top-down: retrieval content is first refined to improve candidate tool recall, followed by tool descriptions to improve selection accuracy, and finally parameter schemas to improve argument filling correctness. Each layer's editor proposes modifications based on failure cases, and updates are retained only when the post-edit evaluation metric $eval(\mathcal{M}(\tilde{D},X),Y)$ improves over the pre-edit baseline.
    - **Design Motivation**: The context for tool use is inherently hierarchical. Existing methods treat documentation as a monolithic unit and ignore inter-layer dependencies; hierarchical editing isolates the scope of each edit and reduces unintended side effects.

2. **Verification-Guided Instruction Editor**:
    - **Function**: Provides structured guidance to editors at each layer, overcoming the stateless and unconstrained nature of existing optimizers.
    - **Mechanism**: Each editor's behavior is governed by a guided system prompt consisting of three key components — a **state space** (tracking prior edits and their effects to prevent repeated degradation), an **action space** (defining task-specific editing operations such as addition, deletion, and terminology unification), and a **reward signal** (providing per-query verification against ground truth to deliver targeted feedback). The action space is customized per layer: retrieval-layer editors optimize query-tool alignment, tool-layer editors enhance semantic precision, and parameter-layer editors refine schema definitions.
    - **Design Motivation**: Directly addresses the three limitations of existing optimizers; structured constraints confine editing to high-level semantic operations, avoiding boundary issues introduced by token-level adjustments.

3. **Iterative Verification and Update Mechanism**:
    - **Function**: Ensures that every edit constitutes a verifiable and incremental improvement.
    - **Mechanism**: At each iteration, the inference model $\mathcal{M}$ runs the validation set over the current documentation and collects failure cases $E_c$. Editors generate modification proposals for each failure case, and updates are accepted only when the global evaluation metric improves. The process continues until a predefined performance threshold is reached.
    - **Design Motivation**: Unlike self-refinement methods that rely on model-generated feedback, VGCO leverages explicit verification signals to ensure that improvements are interpretable and robust.

### Loss & Training

VGCO is a tuning-free framework and does not involve a conventional loss function. "Training" is realized through iterative optimization by LLM editors. The paper also discusses post-training strategies for the editors: performance can be further enhanced via SFT (maximizing the likelihood of correct edits), DPO (preference learning from positive/negative edit pairs), or GRPO (group relative preference optimization).

## Key Experimental Results

### Main Results

Results on the xLAM dataset (100 most frequently used tools) using Claude Sonnet 3.5 as the inference model:

| Editing Method | Retrieval Recall | Tool Selection | Param. Filling | Final Accuracy |
|---|---|---|---|---|
| Raw | 78.98 | 71.5 | 57.5 | 32.5 |
| ReAct | 91.64 | 75.0 | 56.0 | 38.5 |
| DRAFT | 95.62 | 78.0 | 62.0 | 46.2 |
| **VGCO (Claude-4)** | **97.48** | **80.0** | **65.0** | **50.7** |
| **VGCO (Claude-4.5)** | **98.30** | **84.0** | **74.0** | **61.0** |

Results on the BFCL dataset (Claude Sonnet 4 editor + Claude Sonnet 3.7 inference):

| Editing Method | Tool Selection | Param. Filling |
|---|---|---|
| Raw | 69.1 | 55.8 |
| DRAFT | 92.1 | 88.3 |
| **VGCO** | **96.9** | **94.7** |

### Ablation Study

| Configuration | Tool Selection | Param. Filling | Notes |
|---|---|---|---|
| Full Guided Instruction | 96.94 | 94.73 | Full model |
| w/o Common Issues | 82.25 | 81.43 | Remove error type list |
| w/o ICL Examples | 74.89 | 65.43 | Remove in-context examples; largest impact |
| w/o Requirements | 91.18 | 89.35 | Remove editing constraints |
| w/o Analysis Task | 87.32 | 85.41 | Remove structured analysis task |

### Key Findings

- ICL examples have the greatest impact on performance; removing them causes tool selection and parameter filling to drop by 22% and 29% respectively, indicating that in-context examples are central to aligning editor behavior.
- The largest gains from iterative optimization occur in the first two rounds (e.g., Claude Sonnet 3.5 with Claude Sonnet 4.5 editor improves from 0.599 to 0.801), after which improvements gradually converge.
- Stronger editor models (Claude Sonnet 4.5) yield the best and most consistent improvements across all inference models.
- VGCO demonstrates the most pronounced advantages on tool selection and parameter filling — the two subtasks that require fine-grained contextual reasoning.

## Highlights & Insights

- The **hierarchical decoupling design** is elegant: decomposing document optimization into retrieval, selection, and parameter layers, optimizing each independently and propagating improvements in cascade, reduces optimization complexity while preserving inter-layer consistency. This paradigm is transferable to any system optimization problem involving hierarchical decision-making.
- **Verification-guided vs. self-reflection**: Relying on external verification signals rather than model-generated feedback for optimization guidance is more robust than self-refinement. The core insight is that "most tool calling failures are context problems, not reasoning problems."
- **Context engineering perspective**: The problem framing is elevated from "prompt optimization" to "context engineering," covering not only prompt refinement but also retrieval content, tool descriptions, and parameter schemas — encompassing the entire tool calling pipeline.

## Limitations & Future Work

- Validation is currently limited to single-turn tool calling scenarios; context optimization for multi-turn dialogues is an important direction for future work.
- The editors depend on high-quality validation datasets; optimization effectiveness may be limited when validation data coverage is insufficient.
- Scalability as the number of tools continues to grow (beyond hundreds) requires further investigation.
- Robustness under domain shift and in low-resource settings requires additional validation.

## Related Work & Insights

- **vs. DRAFT**: DRAFT also leverages LLM trial-and-error interactions to improve tool documentation, but does not decompose tool calling subtasks and edits the complete task at each iteration. VGCO achieves finer-grained control and more efficient optimization through hierarchical editing.
- **vs. DSPy/SAMMO**: These general-purpose prompt optimization frameworks underperform in industrial-scale scenarios because they are stateless and lack a structured action space. VGCO introduces state-awareness and action constraints specifically designed for tool calling.
- **vs. ReAct**: ReAct improves tool use through alternating reasoning and action steps, but does not modify the underlying tool documentation. VGCO improves document quality at the source, making it complementary to ReAct.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of hierarchical editors and verification-guided optimization represents a solid contribution to tool calling optimization, though the overall framework leans toward engineering.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-model, multi-dataset, and multi-baseline comparisons with clear ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly articulated and the framework is systematically described, though the extensive formalism is occasionally verbose.
- Value: ⭐⭐⭐⭐ Strong practical value for industrial-scale tool calling scenarios, though academic novelty is moderate.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] In-Context Algebra](../../ICLR2026/others/in-context_algebra.md)
- [\[AAAI 2026\] CAE: Hierarchical Semantic Alignment for Image Clustering](hierarchical_semantic_alignment_for_image_clustering.md)
- [\[AAAI 2026\] LILAD: Learning In-context Lyapunov-stable Adaptive Dynamics Models](lilad_learning_in-context_lyapunov-stable_adaptive_dynamics_models.md)
- [\[AAAI 2026\] Forget Less by Learning from Parents Through Hierarchical Relationships](forget_less_by_learning_from_parents_through_hierarchical_relationships.md)
- [\[AAAI 2026\] LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval](leanrag_knowledge-graph-based_generation_with_semantic_aggregation_and_hierarchi.md)

</div>

<!-- RELATED:END -->
