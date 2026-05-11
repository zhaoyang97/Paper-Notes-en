---
title: >-
  [Paper Note] CAD-Assistant: Tool-Augmented VLLMs as Generic CAD Task Solvers
description: >-
  [ICCV 2025][Multimodal VLM][CAD Agent] This paper proposes CAD-Assistant, the first tool-augmented vision-language model framework for generic CAD tasks. By integrating a CAD-specific toolset (sketch parameterizer…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "CAD Agent"
  - "Tool-Augmentation"
  - "VLLM"
  - "Geometric Reasoning"
  - "FreeCAD"
date: 2026-05-08
content_hash: ce7332c9f99ab8fc
---

# CAD-Assistant: Tool-Augmented VLLMs as Generic CAD Task Solvers

**Conference**: ICCV 2025
**arXiv**: [2412.13810](https://arxiv.org/abs/2412.13810)
**Code**: [https://github.com/dimitrismallis/CAD-Assistant](https://github.com/dimitrismallis/CAD-Assistant)
**Area**: Multimodal VLM
**Keywords**: CAD Agent, Tool-Augmentation, VLLM, Geometric Reasoning, FreeCAD

## TL;DR

This paper proposes CAD-Assistant, the first tool-augmented vision-language model framework for generic CAD tasks. By integrating a CAD-specific toolset (sketch parameterizer, rendering module, constraint checker, etc.) and the FreeCAD Python API, it surpasses supervised task-specific methods in a zero-shot setting.

## Background & Motivation

The CAD domain has long faced automation bottlenecks. Existing research focuses on fixed workflows (e.g., 3D reverse engineering, CAD generation), while general-purpose CAD agents remain largely unexplored. Despite VLLMs demonstrating strong capabilities across many domains, they face three core challenges in CAD scenarios:

**Insufficient geometric reasoning**: VLLMs struggle to accurately understand the semantics, spatial arrangements, and primitive localization of rendered objects.

**Unpredictable CAD command effects**: High-level CAD operations (chamfering, filleting, geometric constraints, etc.) have complex and non-intuitive effects on model topology that VLLMs cannot reliably predict.

**Lack of real CAD interaction**: Existing methods cannot directly interact with CAD software, and generated designs cannot be validated.

Tool-augmentation has been shown to effectively mitigate the shortcomings of foundation models, but has yet to be explored in the CAD domain. This paper fills that gap.

## Method

### Overall Architecture

CAD-Assistant adopts a three-component architecture of "planner–environment–toolset." At each timestep $t$, the planner analyzes the current context to generate a plan $p_t$ and action $a_t$ (Python code), which is executed in the environment and returns feedback to drive the next iteration:

$$p_t \leftarrow \mathcal{P}(x_0; c_{t-1}, \mathcal{T})$$
$$a_t \leftarrow \mathcal{P}(p_t; c_{t-1}, x_0, \mathcal{T})$$
$$(f_t, e_t) \leftarrow \mathcal{E}(a_t; e_{t-1}, \mathcal{T}, x_0)$$

where $x_0$ is the multimodal user query, $c_t$ is the context, $f_t$ is the code execution output, and $e_t$ is the new state of the CAD design. The process iterates until the planner generates a TERMINATE signal.

### Key Designs

1. **VLLM Planner**:

    - Uses GPT-4o as the core planner
    - Accepts multimodal inputs (text, sketches, drawing commands, 3D scans)
    - Generates actions as Python code (rather than natural language instructions), directly usable with the FreeCAD API
    - Maintains long-term memory via context concatenation $c_{t+1} \leftarrow \text{concat}(f_t, \{c_s\}_{s=1}^t)$

2. **CAD-Specific Toolset (7 tools)**:

    - **Python**: Logical operations and action formatting
    - **FreeCAD Integration**: Direct CAD software invocation via Python API
    - **Sketch Parameterizer**: Converts freehand sketch images into parameterized CAD sketches using the Davinci model
    - **Sketch Recognizer**: Renders sketches and visualizes parameters to help the planner understand 2D geometry
    - **Solid Recognizer**: Renders 3D CAD models with annotated parameters to enhance 3D understanding
    - **Constraint Checker**: Analyzes the effect of applied geometric constraints and determines whether they compromise geometric integrity
    - **Crosssection Extractor**: Generates cross-sectional images from 3D meshes for reverse engineering of 3D scans

3. **Geometric Reasoning Enhancement Strategies**:

    - **Parameterization strategy**: Point-based representation outperforms SGPBench's implicit representation (accuracy from 0.674 to 0.748)
    - **Serialization strategy**: Schema-embedded format (JSON, 0.748) outperforms tabular formats (HTML/CSV/Markdown, ~0.71)
    - **Rendering augmentation**: Precise rendering (0.754) outperforms textual descriptions (0.748) and freehand sketches (0.616)
    - **Over-parameterization**: Redundant parameter sets incur negligible accuracy loss compared to point-based (0.747 vs. 0.748), and may benefit certain tasks

### Loss & Training

CAD-Assistant is a **training-free** framework requiring no fine-tuning or optimization. It relies on:
- Tool docstrings as usage instructions
- Zero-shot prompting as the primary strategy, with few-shot (5-shot) providing further gains
- The FreeCAD geometric solver for constraint correctness validation
- Iterative correction via multimodal feedback (rendered images + JSON parameters)

## Key Experimental Results

### Main Results

| Task | Metric | CAD-Assistant | GPT-4o Baseline | Supervised Method | Gain |
|------|--------|---------------|-----------------|-------------------|------|
| 2D CQA (SGPBench) | Accuracy | **0.791** | 0.686 | - | +15.3% |
| 3D CQA (SGPBench) | Accuracy | **0.857** | 0.782 | - | +9.6% |
| Auto-constraining | PF1 | **0.979** | 0.693 | 0.706 (Vitruvion) | +38.7% |
| Auto-constraining | CF1 | **0.484** | 0.274 | 0.238 (Vitruvion) | +103.4% |
| Freehand Sketch Parameterization | Accuracy | 0.784 | - | 0.789 (Davinci) | Comparable |
| Freehand Sketch Parameterization | CD↓ | **0.680** | - | 1.184 (Davinci) | −42.6% |

### Ablation Study

| Configuration | PF1 | CF1 | Notes |
|---------------|-----|-----|-------|
| No tools + no docstring (0-shot) | 0.726 | 0.318 | Baseline |
| + Multimodal recognizer (MMrecog) | 0.747 | 0.329 | Rendering-assisted understanding |
| + Constraint checker (ConstrCheck) | **0.979** | **0.484** | Key improvement source |
| Full + 5-shot | 0.981 | 0.514 | Few-shot examples are helpful |
| Full + 5-shot + docstring | **0.984** | **0.529** | Best configuration |

### Key Findings

- The most critical contribution of tool-augmentation comes from the **Constraint Checker**, which enables the agent to evaluate geometric changes after constraint application and avoid destructive operations.
- GPT-4 mini benefits only marginally from tool-augmentation (2D: 0.614 vs. 0.594), indicating that a capable VLLM is a prerequisite for tool-augmentation to be effective.
- Human evaluation shows 98.5% of tool usages are valid; the minority of errors primarily stem from incorrect FreeCAD API calls.
- Failure case analysis reveals that most errors originate from VLLM reasoning mistakes (e.g., confusing trapezoids with triangles).

## Highlights & Insights

- **Generalist design**: No training required; new tools can be integrated via docstrings without being constrained by the command sets of existing CAD datasets.
- **Real CAD interaction**: Generated FreeCAD code is editable, interpretable, and directly usable in production.
- **Multimodal input support**: Covers diverse usage scenarios from textual descriptions to freehand sketches, 3D scans, and drawing commands.
- **Evaluation framework contribution**: Defines evaluation standards for general-purpose CAD agents by integrating multiple existing CAD benchmarks.

## Limitations & Future Work

- Relies on the closed-source GPT-4o as the planner, which entails high cost and precludes local deployment.
- JSON serialization may produce excessively long contexts for complex, large-scale designs.
- Current evaluation focuses primarily on 2D sketches and simple 3D models; capabilities on industrial-grade complex CAD models remain insufficiently validated.
- The toolset is fixed at 7 tools; additional specialized tools (e.g., finite element analysis, tolerance analysis) could be explored.

## Related Work & Insights

- Contrasts with fine-tuning approaches such as CadVLM/CADLLM, demonstrating the competitiveness of the training-free paradigm.
- The tool-augmentation paradigm is generalizable to other engineering software interaction scenarios (e.g., EDA, simulation).
- Vitruvion, trained on large-scale data, is outperformed by zero-shot CAD-Assistant, highlighting the critical importance of constraint solver feedback.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First tool-augmented VLLM framework in the CAD domain, filling an important gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers three major tasks—CQA, constraint reasoning, and sketch parameterization—with human evaluation and failure analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Framework description is clear, tool design motivation is well-justified, and visualizations are rich.
- **Value**: ⭐⭐⭐⭐⭐ Transformative significance for CAD automation and AI-assisted design, with exceptionally high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ReCAD: Reinforcement Learning Enhanced Parametric CAD Model Generation with Vision-Language Models](../../AAAI2026/multimodal_vlm/recad_reinforcement_learning_enhanced_parametric_cad_model_generation_with_visio.md)
- [\[ICCV 2025\] Dynamic Group Detection using VLM-augmented Temporal Groupness Graph](dynamic_group_detection_using_vlm-augmented_temporal_groupness_graph.md)
- [\[ICLR 2026\] Error Notebook-Guided, Training-Free Part Retrieval in 3D CAD Assemblies via Vision-Language Models](../../ICLR2026/multimodal_vlm/error_notebook-guided_training-free_part_retrieval_in_3d_cad_assemblies_via_visi.md)
- [\[CVPR 2026\] From Intuition to Investigation: A Tool-Augmented Reasoning MLLM Framework for Generalizable Face Anti-Spoofing](../../CVPR2026/multimodal_vlm/from_intuition_to_investigation_a_tool-augmented_reasoning_mllm_framework_for_ge.md)
- [\[ICCV 2025\] DWIM: Towards Tool-aware Visual Reasoning via Discrepancy-aware Workflow Generation & Instruct-Masking Tuning](dwim_towards_tool-aware_visual_reasoning_via_discrepancy-aware_workflow_generati.md)

</div>

<!-- RELATED:END -->
