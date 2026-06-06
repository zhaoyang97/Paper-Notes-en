---
title: >-
  [Paper Note] Learning From Design Procedure To Generate CAD Programs for Data Augmentation
description: >-
  [NeurIPS 2025 (Workshop: Deep Learning for Code in the Agentic Era)][Code Intelligence][CAD program generation] This paper proposes a CAD program data augmentation paradigm inspired by industrial design workflows. By pro…
tags:
  - "NeurIPS 2025 (Workshop: Deep Learning for Code in the Agentic Era)"
  - "Code Intelligence"
  - "CAD program generation"
  - "data augmentation"
  - "LLM prompting strategy"
  - "B-Spline geometry"
  - "industrial design"
date: 2026-05-08
content_hash: e76775149adc629b
---

# Learning From Design Procedure To Generate CAD Programs for Data Augmentation

**Conference**: NeurIPS 2025 (Workshop: Deep Learning for Code in the Agentic Era)
**arXiv**: [2603.06894](https://arxiv.org/abs/2603.06894)  
**Code**: None  
**Area**: Code Intelligence
**Keywords**: CAD program generation, data augmentation, LLM prompting strategy, B-Spline geometry, industrial design

## TL;DR
This paper proposes a CAD program data augmentation paradigm inspired by industrial design workflows. By providing reference surface programs and design procedure descriptions as LLM prompts, the method guides the generation of CAD programs containing B-Spline organic shapes, substantially narrowing the geometric complexity gap between public CAD datasets and industrial-grade designs.

## Background & Motivation

**Background**: CAD program generation (e.g., DeepCAD, GenCAD, CAD-MLLM) is an important direction in 3D design automation, with recent progress accelerated by combining Python scripting libraries (CadQuery, Build123d) with LLMs.

**Limitations of Prior Work**: Existing public CAD program datasets (DeepCAD, ABC, etc.) almost exclusively contain primitive operations such as sketch-extrude, producing geometries dominated by planes, cylinders, and other standard primitives, with virtually no B-Spline freeform surfaces—far removed from industrial-grade CAD designs.

**Key Challenge**: CAD data annotation requires domain experts, and data is siloed across different CAD tools, making large-scale collection difficult. Even when image-conditioned generation is employed (GenCAD, CAD-MLLM), the conditioning images themselves originate from simple datasets, creating a hard ceiling on geometric distribution complexity.

**Goal**: To enable LLMs to generate CAD programs containing complex organic shapes (B-Spline surfaces/curves) for high-quality data augmentation.

**Key Insight**: The actual workflow of industrial designers—first selecting a reference surface (typically a B-Spline freeform surface), then building the model upon it incrementally, so that the geometric characteristics of all subsequent operations are influenced by the curvature of the reference surface.

**Core Idea**: A B-Spline reference surface program expressed as a Python script, combined with a natural-language design procedure description, is used as the LLM prompt to guide the generation of CAD programs with industrial-grade geometric complexity.

## Method

### Overall Architecture
The system comprises four stages: (1) design procedure prompt construction—combining a design description and a reference surface program into a prompt; (2) LLM program generation—using an off-the-shelf LLM (OpenAI o3) to generate CadQuery Python scripts; (3) program validation—executing the script and checking whether it can be converted to a B-rep STEP file; (4) structural validation—verifying that the generated B-rep is a watertight solid with feasible topology.

### Key Designs

1. **Design Procedure Prompting**:

    - **Function**: Constructs a multi-part composite text prompt that simulates the industrial designer's modeling workflow starting from a reference surface.
    - **Mechanism**: The prompt template consists of [pre-system prompt, design description, design context, post-system prompt]. The pre-system prompt specifies the use of the CadQuery library; the design description provides a textual description of the target shape (e.g., "a rectangular bracket with two circular holes"); the design context instructs the generated object to conform to the curvature of the reference surface and to remove the reference surface upon completion; the post-system prompt requires a watertight solid and specifies the output path.
    - **Design Motivation**: In industrial design, designers typically begin with a freeform surface as an initial constraint (e.g., a wall or shell surface), and subsequent operations naturally produce B-Spline geometry. This "start-from-reference-surface" workflow is an industry-standard practice that has never been exploited in existing data augmentation methods.

2. **Reference Surface Program**:

    - **Function**: Represents the reference surface as a Python script (rather than an image or point cloud) to be fed into the LLM as part of the prompt.
    - **Mechanism**: Four types of B-Spline surfaces are prepared—Gaussian, saddle, wave, and ripple—each parametrically expressed as a CadQuery script. Diversity is further increased by varying parameters (e.g., saddle curvature from shallow to deep). At each generation step, a reference surface is randomly paired with a design description.
    - **Design Motivation**: Script-form reference surfaces provide the LLM with precise and interpretable parametric geometric descriptions, avoiding geometric inaccuracies introduced by cross-modal inputs such as images or point clouds.

3. **Program Generation and Iterative Validation**:

    - **Function**: Generated programs undergo two-stage validation; failures are fed back to the LLM for self-correction.
    - **Mechanism**: Program validation executes the Python script to check whether a B-rep STEP file can be exported; structural validation uses DTGBrepGen's validity checker to verify watertightness and structural feasibility. Error messages from failed validation are injected back into the prompt for iterative correction.
    - **Design Motivation**: The introduction of reference surfaces increases program complexity, necessitating feedback-driven iterative generation to maintain an acceptable success rate.

### Evaluation Metric

The B-Spline ratio $\beta_i$ is used to measure the proportion of organic shapes in each CAD object:

$$\beta_i = \frac{1}{2}\left(\frac{f_i^b}{f_i} + \frac{e_i^b}{e_i}\right)$$

where $f_i$ is the number of faces, $f_i^b$ is the number of B-Spline faces, $e_i$ is the number of edges, and $e_i^b$ is the number of B-Spline edges. Higher values indicate a greater proportion of organic shapes.

## Key Experimental Results

### Main Results

Using the bracket category as the target class, the proposed method is compared against a commercial industrial dataset and three public CAD program datasets:

| Dataset | Avg. STEP Lines | Avg. Faces | Avg. Edges | B-Spline Faces | B-Spline Edges | B-Spline Ratio |
|---------|----------------|-----------|-----------|----------------|----------------|----------------|
| Industry (commercial) | 10099 | 82.91 | 259.4 | 100% | 100% | 0.535 |
| DeepCAD-b | 1783 | 21.48 | 50.30 | 0% | 1% | 0.0004 |
| GenCAD* | 991 | 12.55 | 27.86 | 0% | 1% | 0.0009 |
| CAD-MLLM* | 2402 | 28.65 | 69.26 | 0% | 1% | 0.0008 |
| **Ours** | **4494** | **26.57** | **67.57** | **77%** | **89%** | **0.2217** |

Key finding: 77% of the CAD objects generated by the proposed method contain B-Spline faces and 89% contain B-Spline edges, compared to approximately 0% for DeepCAD/GenCAD/CAD-MLLM. The B-Spline ratio improves from below 0.001 to 0.22, substantially approaching the industrial-grade value of 0.54.

### Ablation Study

The effect of removing the reference surface prompt is evaluated:

| Method | Avg. STEP Lines | Avg. Faces | B-Spline Faces | B-Spline Edges | B-Spline Ratio |
|--------|----------------|-----------|----------------|----------------|----------------|
| Ours(-RT): no design context + no reference surface | 1225 | 14.86 | 2% | 6% | 0.0085 |
| Ours(-R): text-guided "smooth organic" | 2992 | 34.56 | 18% | 27% | 0.0478 |
| **Ours: full method** | **4494** | **26.57** | **77%** | **89%** | **0.2217** |

### Key Findings
- The reference surface program is the critical factor for generating B-Spline geometry: its removal causes the B-Spline ratio to drop sharply from 0.22 to 0.008.
- Guiding with only the text phrase "smooth and organic" yields only a marginal improvement (0.048), demonstrating the limited capacity of natural language to specify complex geometry.
- Notably, Ours(-R) exhibits higher average face and edge counts than the full method—the LLM tends to approximate smooth shapes using a greater number of standard primitives rather than employing concise B-Spline representations.
- The complexity introduced by reference surfaces causes 21% of samples to require more than 5 iterative corrections, compared to 12% without reference surfaces.

## Highlights & Insights
- **Script as Prompt**: Using executable Python scripts rather than images or point clouds as the reference surface representation. Processing geometric information within the text domain is far more precise for LLMs than cross-modal inputs, and this idea is transferable to any code generation scenario requiring accurate geometric constraints.
- **Design Procedure Knowledge Injection**: Encoding the tacit knowledge of industrial designers (the reference-surface-first modeling workflow) as a prompting strategy is essentially a form of domain-aware prompt engineering.
- **No Multimodal Model Required**: Unlike GenCAD/CAD-MLLM, which require image conditioning, the proposed method operates on purely textual inputs and is compatible with any code-generation LLM, lowering the deployment barrier.

## Limitations & Future Work
- Only four types of reference surfaces (Gaussian, saddle, wave, ripple) are used; diversity relies on parameter variation. Additional surface families could be incorporated through programmatic extension.
- Precise control over which part of the bracket aligns with the reference surface (sometimes the base, sometimes the legs) is not achievable; finer-grained control would require a deeper understanding of CAD parametrization.
- Validation is currently limited to the bracket and wheel categories; generalization to a broader range of industrial design categories remains to be explored.
- Evaluation metrics are proxy-based (e.g., B-Spline ratio); real industrial quality criteria such as curvature continuity and manufacturing tolerances are not assessed.

## Related Work & Insights
- **vs. DeepCAD/GenCAD/CAD-MLLM**: These methods rely on limited operation sets and simple datasets, resulting in low geometric complexity. The proposed work addresses this from a data augmentation perspective, generating training data closer to industrial standards.
- **vs. CAD-Recode**: Both employ Python scripts combined with LLMs, but CAD-Recode performs conditional generation from point clouds to CAD, whereas this paper focuses on unconditional data augmentation—a fundamentally different angle.
- **vs. Image-conditioned methods**: GenCAD/CAD-MLLM use rendered images as prompts and are constrained by the geometric distribution of the source dataset; the use of script-form reference surfaces in this work avoids that bottleneck.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The idea of encoding industrial design workflows as LLM prompting strategies is both simple and effective, though the method does not introduce a new model architecture.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comparisons against multiple baselines and an industrial dataset, with thorough ablation studies, though quantitative evaluation is limited to the bracket category.
- **Writing Quality**: ⭐⭐⭐⭐ — The paper is clearly structured with a complete motivation chain and intuitive explanations of the design workflow.
- **Value**: ⭐⭐⭐⭐ — Offers direct practical value to the CAD program generation community; the generated data can be directly used to train downstream models.

## Related Work & Insights

## Highlights & Insights

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization](preserving_llm_capabilities_through_calibration_data_curation_from_analysis_to_o.md)
- [\[ICLR 2026\] Sharing State Between Prompts and Programs](../../ICLR2026/code_intelligence/sharing_state_between_prompts_and_programs.md)
- [\[ICLR 2026\] CARD: Towards Conditional Design of Multi-agent Topological Structures](../../ICLR2026/code_intelligence/card_towards_conditional_design_of_multi-agent_topological_structures.md)
- [\[ACL 2026\] ChatHLS: Towards Systematic Design Automation and Optimization for High-Level Synthesis](../../ACL2026/code_intelligence/chathls_towards_systematic_design_automation_and_optimization_for_high-level_syn.md)
- [\[ACL 2026\] CodeWiki: Evaluating AI's Ability to Generate Holistic Documentation for Large-Scale Codebases](../../ACL2026/code_intelligence/codewiki_evaluating_ai39s_ability_to_generate_holistic_documentation_for_large-s.md)

</div>

<!-- RELATED:END -->
