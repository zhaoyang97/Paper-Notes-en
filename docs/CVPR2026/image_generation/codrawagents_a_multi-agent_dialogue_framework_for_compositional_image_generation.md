---
title: >-
  [Paper Note] coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation
description: >-
  [CVPR 2026][Image Generation][Multi-agent collaboration] Ours proposes coDrawAgents, an interactive multi-agent dialogue framework (Interpreter-Planner-Checker-Painter). It significantly enhances the faithfulness of compositional text-to-image generation in complex scenarios through divide-and-conquer incremental layout planning, visual context-driven spatial reasoning, and an explicit error correction mechanism.
tags:
  - CVPR 2026
  - Image Generation
  - Multi-agent collaboration
  - Compositional image generation
  - Layout planning
  - Text-content alignment
  - Diffusion models
date: 2026-05-08
content_hash: ecfdaa774e3b4f38
---

# coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation

**Conference**: CVPR 2026  
**arXiv**: [2603.12829](https://arxiv.org/abs/2603.12829)  
**Code**: None yet (the paper states it will be public after publication)  
**Area**: Image Generation / Compositional Text-to-Image Generation  
**Keywords**: Multi-agent collaboration, Compositional image generation, Layout planning, Text-content alignment, Diffusion models

## TL;DR

Ours proposes coDrawAgents, an interactive multi-agent dialogue framework (Interpreter-Planner-Checker-Painter). It significantly enhances the faithfulness of compositional text-to-image generation in complex scenarios through divide-and-conquer incremental layout planning, visual context-driven spatial reasoning, and an explicit error correction mechanism.

## Background & Motivation

**Background**: Existing T2I models face three core challenges when handling complex scenes with multiple objects and attributes:

**Limitations of Prior Work**:
- **Layout Complexity Explosion**: The complexity of relationships between objects in global layout planning is quadratic; a single planner struggles to capture all dependencies.
- **Lack of Visual Context**: Most methods predict layouts in pure text space without reference to the actual image, leading to spatial implausibility.
- **No Explicit Error Correction**: Diffusion models determine rough structures early in the denoising process; once spatial errors are introduced, they are difficult to fix.

**Key Challenge**: Existing solutions (including single-agent and fixed-pipeline multi-agent systems) lack closed-loop reasoning capabilities—they cannot form iterative feedback between planning, verification, and synthesis.

## Method

### Overall Architecture

coDrawAgents consists of a closed-loop dialogue system with four specialized agents:

- **Interpreter**: Decides the generation mode (layout-free or layout-aware), parses text into structured object descriptions, and ranks/groups them by semantic significance.
- **Planner**: Performs incremental layout reasoning in layout-aware mode.
- **Checker**: Verifies spatial consistency and semantic alignment + error correction.
- **Painter**: Synthesizes images step-by-step, providing visual context.

**Mechanism**: The Interpreter determines scene complexity → Simple scenes are generated directly by the Painter → Complex scenes enter the Planner-Checker-Painter loop, iterating layer-by-layer according to semantic priority.

### Key Designs

1. **Divide-and-conquer Planning**: The Interpreter groups objects by semantic significance (objects of the same priority form a group). The Planner processes the layout of only one group at a time, decomposing global $O(n^2)$ complexity into multiple rounds of $O(k^2)$ local problems. The **Design Motivation** is to reduce the spatial relationship complexity for LLM inference in a single pass.

2. **Visualization Chain-of-Thought (VCoT)**: The Planner uses GPT-5 as an MLLM. Inputs include the global text prompt, current priority object descriptions, historical layouts, the partially rendered image $I_{i-1}$ from the Painter, and object grounding information. Reasoning involves three steps:
    - **Canvas State Analysis**: Analyzes the spatial distribution of existing objects in the current frame.
    - **Context-aware Planning**: Reasons about plausible interactions between new objects and the existing scene based on world knowledge.
    - **Physics Constraint Enforcement**: Applies physical constraints (avoiding floating, irrational contact, etc.).

3. **Check-then-Refine**: The Checker performs two-level verification in each iteration:
    - **Single-object level**: Checks size, ratio, and boundary coverage.
    - **Global level**: Reviews cross-object conflicts (overlap, occlusion order, scale drift) across all historical layouts $\{L_1, \ldots, L_i\}$ and propagates corrections to subsequent layouts.

4. **Plug-and-play Painter**: Supports any T2I (layout-free mode, using Flux in this paper) and L2I (layout-aware mode, using 3DIS) models without additional training. The evolving image provides real visual context for subsequent planning.

### Loss & Training

This framework is a training-free inference-time method and does not involve model training. Agents collaborate dynamically during inference via carefully designed prompts and CoT guidance. The Painter uses off-the-shelf pre-trained T2I/L2I models.

## Key Experimental Results

### Main Results

| Dataset | Metric | coDrawAgents | Prev. SOTA | Gain |
|--------|------|-------------|-----------|------|
| GenEval | Overall Score ↑ | **0.94** | 0.84 (GPT Image 1) | +0.10 |
| GenEval | Counting ↑ | **0.94** | 0.85 (GPT Image 1) | +0.09 |
| GenEval | Position ↑ | **0.95** | 0.75 (GPT Image 1) | +0.20 |
| GenEval | Color Attri. ↑ | **0.81** | 0.70 (UniWorld-V1) | +0.11 |
| DPG-Bench | Overall ↑ | **85.17** | 84.08 (SD3-Medium) | +1.09 |
| DPG-Bench | Relation ↑ | **92.92** | 90.87 (FLUX.1-dev) | +2.05 |

### Ablation Study

| Configuration | DPG-Bench Overall ↑ | Function |
|------|---------------------|------|
| Layout-free mode only | 77.60 | Direct T2I generation only |
| + Layout-aware mode | 82.61 | Adds divide-and-conquer layout planning, +5.01 |
| + Visual context | 84.51 | Planner utilizes image context, +1.90 |
| + Checker (Full coDrawAgents) | **85.17** | Explicit error correction mechanism, +0.66 |

### Key Findings

- **Efficiency**: On DPG-Bench, an average image requires only 1.00 Interpreter call, 1.52 Planner calls, 1.62 Checker calls, and 1.95 Painter calls, which is far fewer than the average number of objects (2.79) in the scene.
- **Spatial Accuracy**: Reaching 0.95 on the GenEval Position sub-metric (compared to 0.75 for GPT Image 1) proves the critical role of incremental visual grounding planning for spatial precision.
- **Error Handling**: The Checker's cross-iteration global review resolves the issue of cumulative error propagation.

## Highlights & Insights

- **Closed-loop vs. Pipeline**: Unlike fixed pipeline systems, the four agents form a true iterative dialogue with mutual feedback between planning, checking, and synthesis.
- **Semantic Priority Grouping** is a key novelty—it not only reduces single-step reasoning complexity but also ensures consistent layout treatment for objects at the same semantic level.
- **VCoT** (State Analysis → Context Planning → Physics Constraints) provides a structured and interpretable reasoning framework for MLLM layout planning.
- The **Plug-and-play** design allows the framework to naturally benefit from future, more powerful T2I/L2I models.

## Limitations & Future Work

1. Multi-agent calls introduce additional computational overhead, resulting in higher inference time than single-pass generation methods.
2. Painter performance is limited by the underlying T2I/L2I models (e.g., attribute rendering biases can propagate).
3. Planner and Checker rely on MLLMs, which are subject to hallucinations and overconfidence issues.
4. Currently only supports 2D scenes; extending to 3D controllable generation is an important future direction.

## Related Work & Insights

- Compared to GoT (one-time global reasoning for all bboxes), coDrawAgents' incremental local planning leads significantly on GenEval (0.94 vs. 0.64).
- Compared to T2I-Copilot (fixed multi-agent pipeline), the closed-loop dialogue mechanism yields a 10+ point improvement on DPG-Bench.
- **Insight**: The key to a multi-agent system is not the number of agents, but the combination of closed-loop feedback and divide-and-conquer strategies.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Combines divide-and-conquer strategy with a visual context-driven closed-loop multi-agent framework to systematically solve compositional generation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Verified on both GenEval and DPG-Bench; comprehensive ablation and efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem definition and logical framework design.
- **Value**: ⭐⭐⭐⭐ Achieves breakthrough results (0.94) on GenEval; training-free and plug-and-play with high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Intrinsic Concept Extraction Based on Compositional Interpretability](intrinsic_concept_extraction_based_on_compositional_interpretability.md)
- [\[CVPR 2026\] Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models](erasure_or_erosion_evaluating_compositional_degradation_in_unlearned_text-to-ima.md)
- [\[CVPR 2026\] Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework](taming_score-based_denoisers_in_admm_a_convergent_plug-and-play_framework.md)
- [\[CVPR 2026\] Smoothing the Score Function for Generalization in Diffusion Models: An Optimization-based Explanation Framework](smoothing_score_function_generalization_diffusion_models.md)
- [\[CVPR 2026\] Image Generation as a Visual Planner for Robotic Manipulation](image_generation_as_a_visual_planner_for_robotic_manipulation.md)

</div>

<!-- RELATED:END -->
