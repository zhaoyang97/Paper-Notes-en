---
title: >-
  [Paper Note] FloorplanQA: A Benchmark for Spatial Reasoning in LLMs Using Structured Representations
description: >-
  [ICML 2026][LLM Reasoning][Spatial Reasoning] FloorplanQA systematically diagnoses the "symbolic spatial reasoning" capabilities of 15 leading LLMs using 2,000 2D indoor layouts in JSON/XML formats and 16…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Spatial Reasoning"
  - "Structured Representation"
  - "JSON Layout"
  - "Geometric Inference"
  - "LLM Diagnostic Benchmark"
date: 2026-05-08
content_hash: 46309b6cb0c32a3e
---

# FloorplanQA: A Benchmark for Spatial Reasoning in LLMs Using Structured Representations

**Conference**: ICML 2026  
**arXiv**: [2507.07644](https://arxiv.org/abs/2507.07644)  
**Code**: https://OldDelorean.github.io/FloorplanQA/ (Project Page)  
**Area**: LLM Evaluation / Spatial Reasoning / Indoor Layout  
**Keywords**: Spatial Reasoning, Structured Representation, JSON Layout, Geometric Inference, LLM Diagnostic Benchmark  

## TL;DR
FloorplanQA systematically diagnoses the "symbolic spatial reasoning" capabilities of 15 leading LLMs using 2,000 2D indoor layouts in JSON/XML formats and 16,000 geometric problems (covering distance, visibility, pathing, and placement). The study finds that while models can handle simple distance calculations, they consistently fail at set unions, planning, and constraint satisfaction. Furthermore, Python tool enhancement mitigates arithmetic errors but does not resolve failures at the algorithmic level.

## Background & Motivation

**Background**: LLMs have demonstrated remarkable potential in structured reasoning. In scenarios such as architectural design, auxiliary planning, and embodied interaction, models are increasingly required to process layouts directly in JSON format (coordinates, dimensions, orientations) rather than relying solely on image perception. This necessitates reasoning over symbolic geometry.

**Limitations of Prior Work**: Existing spatial reasoning benchmarks are often limited to qualitative relationships ("left of," "above"), require VLMs to evaluate layouts as images, or embed navigation tasks implicitly within embodied frameworks like ALFRED or R2R. There is a lack of a clean benchmark specifically designed to probe whether a model can calculate a 1.7m hallway, determine if a $2 \times 3$m table fits in a space, or find an obstacle-avoidance path from a stove to a door given a JSON layout.

**Key Challenge**: In real-world deployments, models often serve as "front-ends" for geometric solvers or code generators. The viability of these toolchains depends on the model possessing basic spatial intuition—if a model cannot maintain geometric invariants without tools, tool enhancement may simply amplify underlying errors. While works like LayoutGPT and FirePlace evaluate the realism of generated results, none systematically test geometric consistency under symbolic input.

**Goal**: To construct a **purely symbolic, automatically scorable diagnostic benchmark covering metric, topological, and dynamic categories** to quantitatively determine what current LLMs can achieve on structured layouts, where they fail, and why.

**Key Insight**: Feed models the same abstractions used by architects (room polygons + object bounding boxes + openings). All ground truth values are calculated using deterministic geometric algorithms (independent of model-based scoring) to eliminate circular evaluation logic.

**Core Idea**: Utilize a triplet of "**Structured JSON Layout + Templated Geometric Problems + Rule-based Scoring**" to transform spatial reasoning from an open task into a diagnostic protocol with itemized scoring across "Metric-Topology-Dynamic" capability tiers.

## Method

### Overall Architecture
The benchmark consists of three components: (1) 2,000 layouts—1,800 synthesized using Gemini 2.5 Pro and 200 extracted from real HSSD scenes; (2) 8 questions per layout, totaling 16,000 questions across 8 geometric tasks; (3) a zero-shot evaluation protocol requiring a `Final answer` line, parsed via regex and scored using specific metrics. All layouts share a right-handed 2D coordinate system, with objects represented as polygons with semantic labels (axis-aligned rectangles for synthetic data, arbitrary polygons for HSSD).

The layout generation pipeline follows two stages: first, room geometry is constrained via prompts (shape, adjacency, corridor clearance, symmetry), followed by furniture placement based on stylized templates (e.g., a bedroom must contain a bed and storage). Approximately one-third of candidates are filtered by a rule-validator for unreasonable placements (e.g., sofas blocking doors). 3D scenes from HSSD are projected to 2D and simplified using the Douglas-Peucker algorithm ($\epsilon = 0.01$m).

### Key Designs

1.  **Three-tier Task Taxonomy (Metric / Topology / Dynamic)**:
    - **Function**: Categorizes 8 task types by reasoning intensity. Metric (Pair Distance, View Angle) requires coordinate calculation; Topology (Free Space, Max Box, Placement, Visibility) requires set operations and constraint satisfaction; Dynamic (Repositioning, Shortest Path) requires reasoning over changes in the layout.
    - **Mechanism**: Each question specifies a fixed output format: N (scalar, relative error $\leq 2\%$, relaxed to $5\%$ for Free Space), B (boolean, set equality), L (list, set equality), or S (sequence, requiring obstacle avoidance and Fréchet distance $\leq 0.6$m). Templates replace object names with specific furniture instances (e.g., `fridge_1`) to ensure unique references.
    - **Design Motivation**: Traditional single-difficulty QA fails to identify the specific threshold where a model fails. This three-tier classification maps directly to "arithmetic / constraint reasoning / multi-step planning" failure modes.

2.  **Deterministic Scoring to Eliminate Evaluation Loops**:
    - **Function**: All ground truth values are calculated accurately using geometric libraries like `shapely` (e.g., polygon union area, A* paths, shoelace formula for centroids), independent of any LLM.
    - **Mechanism**: Model outputs must include a brief structured reasoning section followed by a `Final answer:` line. Scores are assigned via automated comparison; failures to follow format result in a zero. Token truncation rates are also tracked to differentiate between reasoning failures and efficiency limits.
    - **Design Motivation**: Using a model like Gemini to both generate layouts and evaluate them introduces circular bias. This framework decouples generation and scoring, ensuring objective evaluation.

3.  **Multi-dimensional Ablation Probes**:
    - **Function**: Four ablation groups determine whether models understand JSON syntax or geometric semantics.
    - **Mechanism**: (a) JSON ↔ XML equivalence testing showed fluctuations within $\pm 3$ pp, suggesting models focus on layout semantics; (b) semantic label swapping (e.g., swapping "sofa" and "chair" labels while keeping geometry constant) caused Repositioning accuracy to plummet from 60.5% to 40.0% (gpt-oss-120B), revealing a reliance on linguistic association; (c) GPT-4.1 + Python interpreter improved arithmetic tasks by $+30$ to $+43$ pp but decreased performance in Max Box ($-4.5$) and Shortest Path ($-12.5$); (d) Visual rendering (icons, boxes, AI photos) remained within $\pm 4$ pp of JSON-only baselines, while pure image input dropped to 19%-40%.
    - **Design Motivation**: These probes separate arithmetic, algorithmic, visual, and semantic bias dimensions.

### Loss & Training
This is an evaluation paper; no models were trained. 15 LLMs were evaluated in a zero-shot setting with `temperature=0` using a unified prompt schema.

## Key Experimental Results

### Main Results
The 15 models (including reasoning models like gpt-oss-120B and DeepSeek-R1, and general models like Claude Sonnet 4) were tested on 16,000 questions. The results followed a three-tier gradient:

| Task Category | Representative Task | Synthetic Avg. Acc | HSSD Acc | Primary Failure Mode |
| :--- | :--- | :--- | :--- | :--- |
| Metric | Pair Distance / View Angle | 75-95% | 35-60% | HSSD polygon centroid errors (shoelace formula failure) |
| Topology (Constraint) | Placement / Visibility / Repositioning | 60-85% | 40-70% | Multi-constraint joint verification failure |
| Topology (Opt) + Dynamic | Free Space / Max Box / Shortest Path | 5-45% | < 30% | Treating "Union of Area" as "Sum of Area"; Search failure |

### Ablation Study
Comparison of Tool Enhancement (GPT-4.1 + Python interpreter) on HSSD:

| Task | Raw | + Tools | $\Delta$ | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| Pair Distance | 56.0 | 99.0 | $+43.0$ | Arithmetic bottleneck resolved |
| View Angle | 55.0 | 96.0 | $+41.0$ | Arithmetic bottleneck resolved |
| Visibility | 46.5 | 86.5 | $+40.0$ | Geometric checks improved |
| Repositioning | 47.0 | 83.5 | $+36.5$ | Better collision detection |
| Placement | 64.5 | 95.0 | $+30.5$ | Benefit from precision |
| Free Space | 16.0 | 44.0 | $+28.0$ | Set union remains difficult |
| Max Box | 7.5 | 3.0 | $-4.5$ | Incorrect algorithmic generation |
| Shortest Path | 25.0 | 12.5 | $-12.5$ | Boundary contact misinterpreted as collision |

### Key Findings
- Kitchen layouts yielded the highest accuracy due to low object overlap (0.52 pairs avg.), whereas bedrooms and HSSD scenes (up to 4.39 pairs) saw significant performance drops in union and clearance tasks.
- Reasoning models outperformed general models by $+10$ to $+40$ pp in Free Space and Max Box, but showed negligible gains in Shortest Path, suggesting that reasoning budgets help with geometric unions but not with search-based planning.
- Tool enhancement caused degradation in planning tasks (Max Box/Shortest Path), as models generated incorrect Python logic (e.g., treating boundary contact as a collision), proving the bottleneck is spatial-algorithmic, not numerical.

## Highlights & Insights
- **Diagnostic Protocol over Leaderboards**: By grouping tasks by reasoning type, the benchmark reveals specific "capability profiles" rather than a single average score.
- **Counter-intuitive Tool Degradation**: The fact that Python interpreters can reduce performance in planning tasks suggests that LLM failures are not merely "calculation errors." Even with a calculator, models struggle to formulate correct search algorithms.
- **JSON vs. XML Irrelevance**: The negligible difference between formats suggests LLMs operate at a semantic layout level, implying future research should focus on geometric inductive biases rather than prompt engineering.

## Limitations & Future Work
- Restricted to 2D indoor scenes; does not cover 3D height, multi-story buildings, or outdoor dynamics.
- Lacks few-shot evaluation, which may underestimate the benefit of in-context geometric examples.
- Future directions: (Short-term) Integration with A*/shapely solvers; (Mid-term) Training with geometric consistency objectives; (Long-term) Multi-step interaction for agent self-correction.

## Related Work & Insights
- **vs. CLEVR / SpatialSense**: Those focus on qualitative visual relations; FloorplanQA targets quantitative geometric reasoning from symbolic coordinates.
- **vs. ScanQA / 3DSRBench**: Those focus on VLM understanding of 3D point clouds/meshes; this work focuses on pure LLM reasoning with symbolic 2D input.
- **vs. LayoutGPT / Holodeck**: Those evaluate the realism of generated layouts; this work evaluates the model's ability to infer geometric properties from given layouts (the inverse direction).

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] CoCoReviewBench: A Completeness- and Correctness-Oriented Benchmark for AI Reviewers](cocoreviewbench_a_completeness-_and_correctness-oriented_benchmark_for_ai_review.md)
- [\[ICML 2026\] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)
- [\[CVPR 2026\] EagleVision: A Dual-Stage Framework with BEV-grounding-based Chain-of-Thought for Spatial Intelligence](../../CVPR2026/llm_reasoning/eaglevision_a_dual-stage_framework_with_bev-grounding-based_chain-of-thought_for.md)
- [\[NeurIPS 2025\] TimE: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios](../../NeurIPS2025/llm_reasoning/time_a_multilevel_benchmark_for_temporal_reasoning_of_llms_i.md)
- [\[ICML 2026\] Evaluating Relational Reasoning in LLMs with REL](evaluating_relational_reasoning_in_llms_with_rel.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] CoCoReviewBench: A Completeness- and Correctness-Oriented Benchmark for AI Reviewers](cocoreviewbench_a_completeness-_and_correctness-oriented_benchmark_for_ai_review.md)
- [\[ICML 2026\] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)
- [\[CVPR 2026\] EagleVision: A Dual-Stage Framework with BEV-grounding-based Chain-of-Thought for Spatial Intelligence](../../CVPR2026/llm_reasoning/eaglevision_a_dual-stage_framework_with_bev-grounding-based_chain-of-thought_for.md)
- [\[ICML 2025\] Adversarial Manipulation of Reasoning Models using Internal Representations](../../ICML2025/llm_reasoning/adversarial_manipulation_of_reasoning_models_using_internal_representations.md)
- [\[ICML 2026\] Evaluating Relational Reasoning in LLMs with REL](evaluating_relational_reasoning_in_llms_with_rel.md)

</div>

<!-- RELATED:END -->
