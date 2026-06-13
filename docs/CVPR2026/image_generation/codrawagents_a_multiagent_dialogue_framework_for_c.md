---
title: >-
  [Paper Note] coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation
description: >-
  [CVPR 2026][Image Generation][compositional T2I generation] This paper proposes coDrawAgents, an interactive multi-agent dialogue framework in which four specialized agents — Interpreter, Planner, Checker…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "compositional T2I generation"
  - "multi-agent dialogue"
  - "layout planning"
  - "visual context grounding"
  - "error correction"
date: 2026-05-08
content_hash: 788cee59e0d5157c
---

# coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation

**Conference**: CVPR 2026
**arXiv**: [2603.12829](https://arxiv.org/abs/2603.12829)  
**Code**: To be released  
**Area**: Image Generation / Multi-Agent Systems
**Keywords**: compositional T2I generation, multi-agent dialogue, layout planning, visual context grounding, error correction

## TL;DR

This paper proposes coDrawAgents, an interactive multi-agent dialogue framework in which four specialized agents — Interpreter, Planner, Checker, and Painter — collaborate in a closed loop. A divide-and-conquer strategy incrementally plans layouts group by group according to semantic priority, grounding reasoning in canvas visual context with explicit error correction. The framework achieves an Overall Score of 0.94 on GenEval, substantially outperforming GPT Image 1 (0.84), and reaches 85.17 SOTA on DPG-Bench.

## Background & Motivation

**Background**: Text-to-image (T2I) generation faces compositional fidelity challenges in complex multi-object scenes. Existing approaches include LLM-assisted layout generation (LayoutLLM-T2I, LMD), diffusion attention guidance (Attend-and-Excite), generation chain-of-thought (GoT), and early multi-agent frameworks (MCCD, T2I-Copilot).

**Limitations of Prior Work**:

1. Single-agent methods delegate parsing, planning, and verification to one model, making early spatial errors difficult to detect and correct.
2. Existing multi-agent frameworks are essentially fixed pipelines lacking negotiation and visual feedback, allowing errors to propagate.
3. Global layout planning incurs quadratic complexity $O(N^2)$ over inter-object relationships, making simultaneous planning of $N$ objects extremely difficult.
4. The vast majority of methods predict layouts without visual context, relying solely on "imagination" of the scene.

**Key Challenge**: The layout reasoning complexity required for complex scenes grows quadratically with the number of objects, yet single-pass planning and fixed pipelines are fundamentally unable to handle this combinatorial explosion.

**Goal**: Achieve faithful compositional T2I generation in complex multi-object scenes while addressing three core challenges: layout complexity, lack of visual perception, and the inability to correct early errors.

**Key Insight**: A four-agent closed-loop dialogue — divide-and-conquer to reduce complexity + canvas visual grounding + explicit checking and error correction.

**Core Idea**: Enable the Planner to observe the canvas being generated when planning the next step, allow the Checker to retrospect over all historical layouts for error correction, and reduce per-round complexity by grouping objects according to semantic priority.

## Method

### Overall Architecture

The Interpreter assesses text complexity → simple scenes directly invoke T2I (layout-free mode) → complex scenes activate layout-aware mode: the Interpreter parses text into attribute-enriched object descriptions with semantic priority ordering and grouping → iterative rounds proceed by priority → each round: the Planner incrementally plans layouts via VCoT → the Checker performs two-stage checking and correction → the Painter incrementally renders on the canvas → the canvas serves as visual context for the next round → after $N$ rounds the final image is output.

### Key Designs

1. **Interpreter + Divide-and-Conquer Strategy**

    - **Function**: Determines the generation mode, decomposes text into attribute-rich object descriptions, and ranks and groups them by semantic salience.
    - **Mechanism**: An LLM with CoT prompting executes three steps: (i) identify and decompose text into semantic units; (ii) rank by semantic salience and group objects of equal priority; (iii) CoT-guided attribute enhancement and background description generation.
    - **Design Motivation**: By grouping, each round processes only objects of the same priority, reducing $O(N^2)$ global planning to multiple rounds of $O(k^2)$ local planning ($k \ll N$). On DPG-Bench, with an average of 2.79 objects per scene, only 1.52 Planner calls are required on average.

2. **Planner + Visualization Chain-of-Thought (VCoT)**

    - **Function**: Incrementally plans the layout of current-priority objects conditioned on the visual context of the current canvas.
    - **Mechanism**: GPT-5 as the MLLM executes a three-step VCoT: (1) *Canvas State Analysis* — receives the canvas image $I_{i-1}$ and existing layouts, analyzing the spatial state of present objects; (2) *Context-Aware Planning* — reasons about plausible interactions between new objects and the existing scene using world knowledge; (3) *Physics Constraint Enforcement* — ensures physical plausibility (no floating objects, reasonable contact surfaces). Object grounding establishes correspondence between textual entities and canvas regions, compensating for LLMs' insensitivity to absolute coordinates.
    - **Design Motivation**: Planning based on the actual canvas rather than imagined scenes fundamentally resolves layout–visual inconsistency.

3. **Two-Stage Checker**

    - **Function**: Validates spatial consistency and attribute alignment of layouts, and corrects errors.
    - **Mechanism**: Stage one checks the current layout $L_i$ at object level (size/proportion/coverage) and global level (relative position/relationships) and applies corrections. Stage two retrospects over all historical layouts $\{L_1, \ldots, L_i\}$, detects cross-object conflicts (overlap/occlusion/scale drift), and iteratively repairs and propagates corrections.
    - **Design Motivation**: Diffusion models fix coarse structure in early denoising steps and are difficult to correct afterward. The Checker performs explicit error correction at the layout stage, preventing errors from being baked into generation.

4. **Painter — Plug-and-Play Rendering**

    - **Function**: Incrementally renders the canvas each round, providing visual context for subsequent iterations.
    - Layout-free mode uses Flux (T2I); layout-aware mode uses 3DIS (L2I); no additional training is required.
    - **Design Motivation**: Decoupling rendering capability from planning and verification logic allows the framework to naturally benefit from improvements in underlying generative models.

### Loss & Training

No additional training is required. The framework leverages pre-trained LLMs (GPT-5) and existing T2I (Flux) / L2I (3DIS) models, constituting a training-free and plug-and-play framework.

## Key Experimental Results

### Main Results

**GenEval Benchmark Comparison**

| Model | Single | Two Obj. | Counting | Colors | Position | Color Attr. | Overall↑ |
|-------|--------|----------|----------|--------|----------|-------------|----------|
| DALL-E 3 | 0.96 | 0.87 | 0.47 | 0.83 | 0.43 | 0.45 | 0.67 |
| FLUX.1-dev | 0.99 | 0.81 | 0.79 | 0.74 | 0.20 | 0.47 | 0.67 |
| GoT | 0.99 | 0.69 | 0.67 | 0.85 | 0.34 | 0.27 | 0.64 |
| UniWorld-V1 | 0.99 | 0.93 | 0.79 | 0.89 | 0.49 | 0.70 | 0.80 |
| GPT Image 1 [High] | 0.99 | 0.92 | 0.85 | 0.92 | 0.75 | 0.61 | 0.84 |
| **coDrawAgents** | **1.00** | **0.96** | **0.94** | **0.97** | **0.95** | **0.81** | **0.94** |

**DPG-Bench Comparison**

| Model | Global | Entity | Relation | Overall↑ |
|-------|--------|--------|----------|----------|
| DALL-E 3 | 90.97 | 89.61 | 90.58 | 83.50 |
| SD3-Medium | 87.90 | 91.01 | 80.70 | 84.08 |
| OmniGen2 | 88.81 | 88.83 | 89.37 | 83.57 |
| **coDrawAgents** | 84.78 | **90.15** | **92.92** | **85.17** |

### Ablation Study

| Configuration | DPG Overall↑ | Notes |
|---------------|-------------|-------|
| Layout-free baseline | 77.60 | Direct T2I only |
| + Layout-aware | 82.61 (+5.01) | Divide-and-conquer reduces complexity |
| + Visual context | 84.51 (+1.90) | Canvas grounding improves spatial consistency |
| + Checker (full) | **85.17** (+0.66) | Explicit error correction improves faithfulness |

**Efficiency Statistics (DPG-Bench, 1074 images)**

| Agent | Avg. calls / image |
|-------|--------------------|
| Interpreter | 1.00 |
| Planner | 1.52 |
| Checker | 1.62 |
| Painter | 1.95 |
| Avg. objects per scene | 2.79 |

### Key Findings

- GenEval Overall improves from 0.84 (GPT Image 1) to 0.94 (+11.9%), achieving the highest scores across all sub-metrics.
- The Position metric surges from 0.75 to 0.95, demonstrating that canvas visual grounding combined with divide-and-conquer substantially enhances spatial reasoning.
- Counting improves from 0.85 to 0.94, confirming that grouped generation effectively addresses the counting problem.
- Average agent invocation counts are far fewer than the number of scene objects (1.52 vs. 2.79), as the grouping strategy reduces the number of iterative rounds.

## Highlights & Insights

- The divide-and-conquer strategy decomposes global $N$-object layout planning into group-wise planning ordered by semantic priority, elegantly reducing complexity.
- Using canvas visual context as Planner input is the core innovation — shifting layout reasoning from "imagining" to "observing the canvas."
- The Checker's cross-iteration retrospective correction handles cascading effects of early errors, which is impossible in fixed-pipeline systems.
- The three-step VCoT reasoning (state analysis → context-aware planning → physics constraint enforcement) is structurally clear and generalizable to other generation tasks requiring spatial reasoning.

## Limitations & Future Work

- Multi-agent invocations introduce computational overhead (multiple LLM inferences + multiple image generation steps), resulting in longer inference times than single-pass methods.
- Painter performance depends on the underlying T2I/L2I model capabilities; imperfect attribute rendering (e.g., "black radish") may propagate across rounds.
- Both Planner and Checker rely on GPT-5 as the MLLM, introducing risks of hallucination and overconfidence.
- The framework supports only 2D synthesis and has not been extended to 3D scene generation.
- The DPG-Bench Global metric (84.78) falls below some single models (e.g., DALL-E 3 at 90.97), suggesting that stepwise generation may compromise global coherence.

## Related Work & Insights

- **vs. GoT**: GoT infers all bounding boxes in a single pass without visual feedback (Overall 0.64 vs. 0.94), validating the fundamental advantage of closed-loop interactive collaboration.
- **vs. T2I-Copilot**: A fixed pipeline with no dialogue negotiation or visual grounding (Overall 74.34 vs. 85.17).
- **vs. MCCD**: Performs only text decomposition without canvas awareness, essentially constituting parallel generation followed by fusion.
- **Insights**: The closed-loop multi-agent collaboration paradigm is generalizable to video generation (per-frame planning + consistency checking), 3D scene construction (per-object placement + collision detection), and other tasks requiring incremental compositional assembly.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The closed-loop multi-agent dialogue framework and VCoT visual grounding planning are innovative, though the core techniques largely constitute prompt engineering over LLMs/MLLMs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparisons on GenEval and DPG-Bench, with ablations and efficiency analysis; qualitative comparisons are clear.
- **Writing Quality**: ⭐⭐⭐⭐ The roles and responsibilities of the four agents are described clearly; the architecture diagram is intuitive.
- **Value**: ⭐⭐⭐ Compositional generation results are impressive, but the approach is highly engineering-driven; cost and reproducibility constraints stemming from GPT-5 dependence are primary concerns.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] JointDiff: Bridging Continuous and Discrete in Multi-Agent Trajectory Generation](../../ICLR2026/image_generation/jointdiff_bridging_continuous_and_discrete_in_multi-agent_trajectory_generation.md)
- [\[CVPR 2026\] MultiBanana: A Challenging Benchmark for Multi-Reference Text-to-Image Generation](multibanana_a_challenging_benchmark_for_multi_reference_text_to_image_generation.md)
- [\[ICML 2026\] Offline Multi-agent Reinforcement Learning via Sequential Score Decomposition](../../ICML2026/image_generation/offline_multi-agent_reinforcement_learning_via_sequential_score_decomposition.md)
- [\[AAAI 2026\] Conditional Diffusion Model for Multi-Agent Dynamic Task Decomposition](../../AAAI2026/image_generation/conditional_diffusion_model_for_multi-agent_dynamic_task_dec.md)
- [\[CVPR 2026\] Intrinsic Concept Extraction Based on Compositional Interpretability](intrinsic_concept_extraction_based_on_compositional_interpretability.md)

</div>

<!-- RELATED:END -->
