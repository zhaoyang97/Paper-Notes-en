---
title: >-
  [Paper Note] coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation
description: >-
  [CVPR 2025][Image Generation][Multi-Agent] This paper proposes coDrawAgents, an interactive multi-agent dialogue framework consisting of four expert agents: Interpreter, Planner, Checker, and Painter. Through divide-and-conquer incremental layout planning, visual context-aware reasoning, and explicit error correction, it achieves 0.94 (SOTA) on GenEval and 85.17 (SOTA) on DPG-Bench.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Multi-Agent"
  - "Compositional Generation"
  - "Layout Planning"
  - "Text-to-Image"
  - "Closed-Loop Reasoning"
date: 2026-05-08
content_hash: 5d26c3d922d9637c
---

# coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation

**Conference**: CVPR 2025  
**arXiv**: [2603.12829](https://arxiv.org/abs/2603.12829)  
**Code**: To be confirmed  
**Area**: Image Generation / Compositional Generation  
**Keywords**: Multi-Agent, Compositional Generation, Layout Planning, Text-to-Image, Closed-Loop Reasoning

## TL;DR

This paper proposes coDrawAgents, an interactive multi-agent dialogue framework consisting of four expert agents: Interpreter, Planner, Checker, and Painter. Through divide-and-conquer incremental layout planning, visual context-aware reasoning, and explicit error correction, it achieves 0.94 (SOTA) on GenEval and 85.17 (SOTA) on DPG-Bench.

## Background & Motivation

**Background**: T2I models still struggle to correctly compose multiple objects and maintain attribute consistency in complex scenes.

**Limitations of Prior Work**: (1) Global layout planning faces $O(n^2)$ relational complexity; (2) Most methods lack visual feedback for spatial reasoning; (3) It is difficult to correct errors once a rough structure is settled early in the diffusion pipeline.

**Key Challenge**: The capacity bottleneck of a single agent is apparent, while existing multi-agent methods mostly rely on fixed pipelines and lack closed-loop reasoning.

**Goal**: To solve compositional image generation in complex scenes via a multi-agent closed-loop dialogue.

**Key Insight**: Decompose the generation process into four specialized roles, interacting through dynamic dialogue rather than sequential pipelining.

**Core Idea**: Compositional image generation requires "planning-checking-rendering" closed-loop collaboration.

## Method

### Overall Architecture

Interpreter decides generation mode (layout-free/layout-aware). In layout-aware mode: parse text into structured object descriptions -> group by semantic priority -> for each group: Planner performs incremental planning -> Checker validates and corrects -> Painter renders the canvas -> loop to the next group.

### Key Designs

1. **Interpreter**

    - Function: Decides the generation mode and framework, decomposing complex prompts into structured object descriptions.
    - Mechanism: LLM + CoT in three steps (identification, priority grouping, attribute enrichment).
    - Design Motivation: Generate simple prompts directly; adaptive selection avoids unnecessary overhead.

2. **Planner + Visual Chain-of-Thought (VCoT)**

    - Function: Incrementally plans the layout of current high-priority objects.
    - Mechanism: Employs GPT-5 for multimodal VCoT, taking global text, object descriptions, historical layouts, partial images, and object localization as input.
    - Three-step Reasoning: Canvas state analysis -> Context-aware planning -> Physical constraint execution.
    - Design Motivation: Divide-and-conquer reduces complexity, and visual context eliminates "imagining from thin air".

3. **Checker**

    - Function: Two-stage checking and correction (current proposal check + full history backtracking).
    - Mechanism: Performs object-level (size, ratio) and global-level (position, occlusion) checks, backtracking to correct history.
    - Design Motivation: Explicit correction compensates for the "hard-to-change once decided" drawback of diffusion models.

4. **Painter**

    - Function: Calls Flux for layout-free generation and 3DIS for layout-aware generation.
    - Designed as plug-and-play, allowing replacement with any T2I/L2I models.

### Loss & Training

Training-free framework. Planner/Checker utilize GPT-5 for inference.

## Key Experimental Results

### Main Results

GenEval comparison:

| Model | Overall |
|------|---------|
| DALL-E 3 | 0.67 |
| FLUX.1-dev | 0.67 |
| SD3-Medium | 0.74 |
| GPT Image 1 | 0.84 |
| coDrawAgents | **0.94** |

DPG-Bench: Overall 85.17 (SOTA), Relation 92.92 (best).

### Ablation Study

| Configuration | Overall |
|------|---------|
| Layout-free only | 77.60 |
| + Layout-aware | 82.61 |
| + Visual context | 84.51 |
| + Checker | **85.17** |

Agent efficiency: On average, only 1.52 Planner calls and 1.62 Checker calls per image, which is far fewer than the average object count of 2.79.

### Key Findings

- GenEval score of 0.94 is 10 percentage points higher than GPT Image 1.
- Position metric increases from FLUX's 0.20 to 0.95, indicating that incremental planning + checking is highly effective for spatial localization.
- Counting metric improves from 0.79 to 0.94, showing that the divide-and-conquer strategy effectively resolves object counting inaccuracy.
- Each component has independent contributions (divide-and-conquer +5pp, visual context +2pp, Checker +0.7pp).

## Highlights & Insights

- Closed-loop multi-agent dialogue allows the Checker to backtrack and correct historical iterative errors.
- Visual context-aware planning avoids "imagining from thin air".
- Semantic priority grouping effectively reduces agent invocation times.
- The GenEval score of 0.94 is highly impressive.

## Limitations & Future Work

- Heavily relies on GPT-5, resulting in high computational costs and susceptibility to hallucinations.
- The Painter is constrained by the performance of the underlying base models.
- Limited to 2D generation.

## Related Work & Insights

- GoT performs one-shot global reasoning, whereas coDrawAgents conducts incremental visual-aware reasoning.
- The divide-and-conquer + visual CoT paradigm can be generalized to video and 3D generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The closed-loop multi-agent dialogue framework is a meaningful innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation on two benchmarks with thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear architectural diagram.
- **Value**: ⭐⭐⭐⭐ The 0.94 SOTA on GenEval holds system-level significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MCCD: Multi-Agent Collaboration-based Compositional Diffusion for Complex Text-to-Image Generation](mccd_multi-agent_collaboration-based_compositional_diffusion_for_complex_text-to.md)
- [\[CVPR 2025\] Unified Uncertainty-Aware Diffusion for Multi-Agent Trajectory Modeling](unified_uncertainty-aware_diffusion_for_multi-agent_trajectory_modeling.md)
- [\[CVPR 2025\] Multi-Group Proportional Representation for Text-to-Image Models](multi-group_proportional_representations_for_text-to-image_models.md)
- [\[CVPR 2025\] LaTexBlend: Scaling Multi-concept Customized Generation with Latent Textual Blending](latexblend_scaling_multi-concept_customized_generation_with_latent_textual_blend.md)
- [\[CVPR 2025\] Concept Lancet: Image Editing with Compositional Representation Transplant](concept_lancet_image_editing_with_compositional_representation_transplant.md)

</div>

<!-- RELATED:END -->
