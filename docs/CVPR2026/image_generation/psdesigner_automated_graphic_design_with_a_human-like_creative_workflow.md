---
title: >-
  [Paper Note] PSDesigner: Automated Graphic Design with a Human-Like Creative Workflow
description: >-
  [CVPR 2026][Image Generation][Automated graphic design] This paper presents PSDesigner, an automated graphic design system that simulates the creative workflow of human designers. It operates through three collaborative…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Automated graphic design"
  - "PSD file manipulation"
  - "tool invocation"
  - "reinforcement learning"
  - "creative workflow"
date: 2026-05-08
content_hash: f235a8e30ff15d6f
---

# PSDesigner: Automated Graphic Design with a Human-Like Creative Workflow

**Conference**: CVPR 2026
**arXiv**: [2603.25738](https://arxiv.org/abs/2603.25738)
**Code**: [https://henghuiding.com/PSDesigner/](https://henghuiding.com/PSDesigner/)
**Area**: Image Generation / Automated Design
**Keywords**: Automated graphic design, PSD file manipulation, tool invocation, reinforcement learning, creative workflow

## TL;DR

This paper presents PSDesigner, an automated graphic design system that simulates the creative workflow of human designers. It operates through three collaborative modules — AssetCollector (resource collection), GraphicPlanner (tool-call planning), and ToolExecutor (PSD operation execution) — and is trained on CreativePSD, the first PSD-format design dataset, enabling the system to learn professional design workflows and directly generate editable PSD files.

## Background & Motivation

1. **Background**: Graphic design plays a critical role in e-commerce and advertising. Existing automated approaches fall into two categories: (a) text-to-image models (e.g., FLUX, Glyph-Byt5) that generate design images; and (b) MLLM-driven methods (e.g., LaDeCo, COLE) that directly generate editable design files in JSON format.

2. **Limitations of Prior Work**: Text-to-image methods produce non-editable outputs with inaccurate text rendering (particularly for Chinese); MLLM-based methods group layers by predefined categories (underlay/text) and predict all attributes in a single pass, resulting in an unintuitive design process with limited flexibility.

3. **Key Challenge**: Existing methods significantly oversimplify professional design workflows — (a) grouping by category is less intuitive than grouping by visual concept; (b) one-shot prediction of all layer attributes lacks the flexibility of progressive design; (c) only simple layer hierarchies and a limited set of attribute types are supported, falling far short of production-level design requirements.

4. **Goal**: To construct an automated design system that simulates the workflow of human designers, handles complex PSD layer hierarchies (averaging 48.35 layers), supports diverse layer types and 60+ attribute types, and generates editable, professional-grade PSD files.

5. **Key Insight**: By observing how human designers work — first collecting thematic assets, then iteratively integrating resources grouped by visual concept, and refining defects after each integration step — the paper formalizes this process as a tool-call prediction problem for VLMs.

6. **Core Idea**: Graphic design is modeled as a sequential tool-call prediction task for VLMs, with SFT and GRPO training enabling the model to learn iterative asset integration and defect correction operations.

## Method

### Overall Architecture

Given a user instruction, PSDesigner operates in three stages: (1) AssetCollector employs an LLM to identify visual concepts and gather relevant assets (images/text) for each concept; (2) the nested layer hierarchy is traversed bottom-up, with GraphicPlanner predicting tool calls for asset integration ($\mathcal{X}_{gen}$ mode) followed by defect correction ($\mathcal{X}_{edt}$ mode) at each iteration; (3) ToolExecutor executes these tool calls in Adobe Photoshop via the UXP API, supporting 70+ PSD operations.

### Key Designs

1. **CreativePSD Dataset (the first PSD-format design dataset)**:

    - **Function**: Provides supervised training data on professional design operations for GraphicPlanner.
    - **Mechanism**: Constructed in three stages — Stage I: high-quality PSD files are collected from the internet and commercial sources, with professional annotators grouping layers by visual concept; Stage II: PSD files are parsed to extract raw assets, metadata, and intermediate rendering results; Stage III: training samples are constructed in both $\mathcal{X}_{gen}$ (asset integration) and $\mathcal{X}_{edt}$ (defect correction) modes from the extracted information. Each training sample is a triplet $(a, \mathcal{C}, x)$: assets + observations + tool-call sequence.
    - **Design Motivation**: Existing datasets (CGL/Crello/Design39K) average only 4–5 layers, 2 layer types, and limited attributes. CreativePSD contains 10,454 samples with an average of 48.35 layers, 5 layer types, and 60+ attribute types, enabling learning of realistic design workflows.

2. **GraphicPlanner (Dual-Mode VLM Tool-Call Predictor)**:

    - **Function**: Predicts the next PSD operation tool call based on the current design state.
    - **Mechanism**: Built on Qwen2.5-VL-7B with mode-specific LoRA modules injected. The $\mathcal{X}_{gen}$ mode receives assets $a$ and observations $(M, R)$ (layer metadata + current rendering) to predict integration tool calls; the $\mathcal{X}_{edt}$ mode receives observations $(M, R, G)$ (perturbed metadata + rendering + pre-group rendering) to predict correction tool calls. Training proceeds in two stages: SFT first establishes basic tool–parameter mapping, followed by GRPO reinforcement learning to refine the precision of parameter values.
    - **Design Motivation**: Asset integration and defect correction are fundamentally different operations; mode-specific LoRA modules prevent task interference. The GRPO reward function directly evaluates the correctness of tool names and parameter values, improving the accuracy of tool-call predictions.

3. **ToolExecutor (UXP-Based PSD Operation Executor)**:

    - **Function**: Translates tool calls predicted by GraphicPlanner into actual PSD file operations.
    - **Mechanism**: Implements 70+ Photoshop operations using JavaScript APIs on the Adobe UXP framework, including inserting images/text/adjustment layers, applying effects (inner glow, drop shadow, etc.), setting blending modes, and clipping masks.
    - **Design Motivation**: Direct manipulation of PSD files — rather than JSON representations — enables support for the complex layer attributes and effect configurations required in production-level design.

### Loss & Training

The SFT stage uses standard autoregressive cross-entropy loss. The GRPO reinforcement learning stage employs a dedicated reward function $r$ that compares generated tool calls against ground truth on tool names, parameter names, and parameter values. SFT training runs for 15,000 steps ($\mathcal{X}_{gen}$) / 12,000 steps ($\mathcal{X}_{edt}$) with batch size 64, lr $= 2\text{e-}4$, and LoRA rank $= 32$. GRPO training runs for 6,000 steps with group size 8, using 4,000 PSD files.

## Key Experimental Results

### Main Results

User intent → design (VLM score, out of 10):

| Method | Quality | Layout | Relevance | Harmony | Creativity | Editable |
|--------|---------|--------|-----------|---------|------------|----------|
| **PSDesigner** | 7.62 | **8.68** | 7.78 | 8.02 | **8.45** | ✓ PSD |
| CanvaGPT | **8.52** | 8.15 | 4.72 | 7.21 | 7.52 | ✓ |
| FLUX | 8.18 | 6.88 | 6.92 | 6.82 | 6.95 | ✗ |
| PosterCraft | 7.95 | 8.35 | **8.42** | **8.05** | 5.87 | ✗ |
| OpenCOLE | 5.12 | 3.66 | 5.25 | 6.68 | 6.08 | Partial |

Design composition on Crello-v5 (VLM score):

| Method | Quality | Layout | Harmony | Creativity |
|--------|---------|--------|---------|------------|
| **Ours** | **7.85** | **7.43** | 6.77 | **6.94** |
| LaDeCo | 5.95 | 6.03 | **7.22** | 5.75 |
| Ground Truth | 8.13 | 9.18 | 8.90 | 7.12 |

### Ablation Study

| Configuration | Quality | Layout | Harmony | Creativity |
|---------------|---------|--------|---------|------------|
| Full model (Crello) | **7.85** | **7.43** | **6.77** | **6.94** |
| w/o $\mathcal{X}_{edt}$ | 6.05 | 5.88 | 5.90 | 6.75 |
| w/o layer info $M$ | 6.25 | 6.10 | 6.18 | 6.02 |
| w/o RL (GRPO) | 6.38 | 6.00 | 6.35 | 6.20 |
| Full model (PSD) | **6.28** | **6.15** | **7.02** | **6.88** |
| w/o $\mathcal{X}_{edt}$ (PSD) | 5.32 | 5.15 | 6.22 | 6.05 |

### Key Findings

- Removing the $\mathcal{X}_{edt}$ mode (defect correction) has the largest impact on quality and layout (drops of 1.80 and 1.55 points respectively on Crello), demonstrating that iterative refinement is critical to design quality.
- Removing layer metadata $M$ prevents the model from perceiving other elements within the same group, leading to significant degradation in layout and harmony.
- GRPO reinforcement learning is essential for accurately predicting tool-call parameter values; its removal causes a 1.43-point drop in layout score.
- PSDesigner is the only system capable of simultaneously generating editable PSD files and accurately rendering Chinese text.

## Highlights & Insights

- **Human-Like Workflow Design**: Decomposing the design process into an iterative "collect → integrate → refine" pipeline closely mirrors how human designers think. This problem decomposition paradigm is transferable to other multi-step creative tasks (e.g., slide authoring, video editing).
- **CreativePSD Dataset**: The first PSD-format training dataset for design, with an average of 48 layers and 60+ attributes, substantially expanding the complexity level that automated design systems can handle. The three-stage data construction pipeline (collection → parsing → training data construction) also constitutes a reusable methodology.
- **GRPO for Tool-Call Refinement**: Applying reinforcement learning to improve the precision of tool-call parameters is an elegant design choice. SFT can only approximate the parameter value distribution, whereas GRPO's reward signal directly optimizes output–GT alignment.

## Limitations & Future Work

- Evaluation relies primarily on VLM scoring and user studies, lacking more objective quantitative metrics (e.g., layer attribute prediction accuracy).
- AssetCollector depends on the quality of external image search/generation models; failures in asset collection cascade into degraded design quality.
- Only static designs are supported; dynamic or interactive designs (e.g., web pages, animations) are not addressed.
- The dataset scale (10,454 samples) remains small compared to LLM training corpora, potentially limiting generalization.
- Deep coupling with Photoshop via the UXP API restricts extensibility to other design tools.

## Related Work & Insights

- **vs. LaDeCo**: LaDeCo groups layers by predefined categories (image/text) and predicts all attributes for each category in a single pass, failing to handle complex hierarchies. PSDesigner groups by visual concept and integrates resources iteratively, achieving 1.9 points higher quality on Crello.
- **vs. COLE/OpenCOLE**: COLE constructs a pipeline of multiple task-specific models, but its editability is limited (only a single image layer plus text layers). PSDesigner unifies the process under VLM-driven tool calls, supporting full PSD layer hierarchies.
- **vs. T2I methods (FLUX/PosterCraft)**: These methods produce visually high-quality but non-editable raster images, with frequent errors in rendering Chinese or complex text.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First application of the VLM tool-call paradigm to PSD-level automated design; CreativePSD is also a first-of-its-kind contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive comparisons against multiple baselines with thorough ablations, though objective metrics such as tool-call accuracy are absent.
- **Writing Quality**: ⭐⭐⭐⭐ — The comparison figure between human designer workflows and PSDesigner is highly intuitive; problem motivation is clearly articulated.
- **Value**: ⭐⭐⭐⭐ — Significant contribution to the automated design field, demonstrating the potential of VLM + tool invocation for creative tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ShowTable: Unlocking Creative Table Visualization with Collaborative Reflection and Refinement](showtable_unlocking_creative_table_visualization_with_collaborative_reflection_a.md)
- [\[ICCV 2025\] Rethinking Layered Graphic Design Generation with a Top-Down Approach](../../ICCV2025/image_generation/rethinking_layered_graphic_design_generation_with_a_top-down_approach.md)
- [\[CVPR 2026\] GIST: Towards Design Compositing](gist_towards_design_compositing.md)
- [\[CVPR 2026\] Ar2Can: An Architect and an Artist Leveraging a Canvas for Multi-Human Generation](ar2can_an_architect_and_an_artist_leveraging_a_canvas_for_multi-human_generation.md)
- [\[CVPR 2026\] PosterIQ: A Design Perspective Benchmark for Poster Understanding and Generation](posteriq_a_design_perspective_benchmark_for_poster_understanding_and_generation.md)

</div>

<!-- RELATED:END -->
