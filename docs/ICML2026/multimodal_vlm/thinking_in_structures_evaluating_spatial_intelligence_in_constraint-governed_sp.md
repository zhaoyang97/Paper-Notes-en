---
title: >-
  [Paper Note] Thinking in Structures: Evaluating Spatial Intelligence in Constraint-Governed Spaces
description: >-
  [ICML 2026][Multimodal VLM][Spatial Intelligence] The authors construct SSI-Bench, a benchmark consisting of 1,000 ranking-based VQA questions focusing on "constraint-governed structured spaces" (real 3D structures such…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Spatial Intelligence"
  - "Structured Reasoning"
  - "Ranking VQA"
  - "VLM Benchmark"
  - "3D Constraints"
date: 2026-05-08
content_hash: 4139531a89ad91d3
---

# Thinking in Structures: Evaluating Spatial Intelligence in Constraint-Governed Spaces

**Conference**: ICML 2026  
**arXiv**: [2602.07864](https://arxiv.org/abs/2602.07864)  
**Code**: https://ssi-bench.github.io  
**Area**: Multimodal VLM  
**Keywords**: Spatial Intelligence, Structured Reasoning, Ranking VQA, VLM Benchmark, 3D Constraints

## TL;DR
The authors construct SSI-Bench, a benchmark consisting of 1,000 ranking-based VQA questions focusing on "constraint-governed structured spaces" (real 3D structures such as roofs, bridges, and towers). It requires VLMs to provide a complete permutation of 3-4 candidate components based on geometric or topological criteria. Evaluations of 31 VLMs reveal that the strongest proprietary model, Gemini-3-Flash, achieves only 33.6%, and the best open-source model, GLM-4.6V, reaches 22.2%, whereas humans achieve 91.6%. This highlights the lack of consistent spatial reasoning capabilities in current VLMs within real 3D scenes jointly constrained by geometry, connectivity, and physical feasibility.

## Background & Motivation

**Background**: Spatial intelligence benchmarks are expanding along multiple axes—single-view vs. multi-view (SpatialRGPT, ViewSpatial-Bench), image vs. video (VSI-Bench, STI-Bench), and manual vs. automatic annotation (MMSI-Bench, Spatial457). Most of these works model spatial reasoning as "scene-centric," meaning they measure distance and direction based on unconstrained indoor/outdoor daily environments.

**Limitations of Prior Work**: Scene-centric benchmarks suffer from fundamental ambiguity—3D relationships are often underdetermined in a single image (the same object could be smaller or further away), and multiple 3D configurations can explain the same 2D observation. Consequently, models can "guess" correctly using appearance priors or dataset biases, making it impossible to discern whether they truly recover 3D structures.

**Key Challenge**: Reliable spatial reasoning in the real world often occurs in *structure-constrained* scenarios (bridges, roofs, towers), where geometric laws, connectivity constraints, and physical feasibility strictly narrow down candidate 3D states. However, existing benchmarks either focus on completely unconstrained daily scenes or minimalist synthetic shapes (CLEVR, Spatial457), failing to preserve the combination of "real visual complexity + strong structural constraints."

**Goal**: (i) Formally define Structure-Centric Spatial Reasoning (SCSR); (ii) construct a VQA benchmark that preserves real 3D complexity while ensuring candidate relationships are uniquely determinable; (iii) use ranking tasks as the evaluation format to force models to parse relative 3D relationships among all candidates; (iv) systematically evaluate 31 VLMs and diagnose typical failure modes.

**Key Insight**: Represent the scene as a node-component graph $\mathbf{s}=(V,E,\mathbf{G},\mathbf{A})$, where geometric degrees of freedom $\mathbf{G}$ and discrete attributes $\mathbf{A}$ are restricted by *explicit equality constraints* $\mathbf{c}(\mathbf{s})=\mathbf{0}$ and *inequality constraints* $\mathbf{h}(\mathbf{s})\leq\mathbf{0}$. These constraints are not directly provided to the model but are used to *construct* samples where the candidate ranking is uniquely determinable. This preserves real-world visual complexity while strictly defining the ground truth.

**Core Idea**: Upgrade spatial intelligence evaluation from "measuring distance/direction" to "ranking all candidate 3D relationships," and ensure the ranking is uniquely determinable through structural constraints, thereby decoupling the model's spatial reasoning ability from 2D pixel shortcuts.

## Method

### Overall Architecture
The construction and evaluation of SSI-Bench follow a five-step pipeline: (1) Data Recall—scanning ~20,000 structural images from copyright-free libraries like Unsplash/Pexels/Pixabay and self-taken photos, with 10 researchers spending 400+ hours filtering 2,000+ candidates covering common structures like spatial trusses, steel towers, cable-stayed bridges, timber trusses, reinforced frames, and piping systems; (2) Task Design—10 categories divided into geometric and topological groups; (3) Metadata Labeling—using Label Studio to record rankings and mark polygons to highlight target components; (4) Question Generation—rendering a separate highlighted image for each candidate to avoid occlusion and color bias; (5) Quality Control—reviews by independent inspectors, third-party arbitration for disagreements, and difficulty labeling for each question.

### Key Designs

1.  **SCSR Formalization + Three Types of Structural Constraints**:
    *   Function: Formalizes "spatial reasoning in structure-constrained spaces" as a constrained inference problem, giving the evaluation target a mathematically clear semantics.
    *   Mechanism: Each image corresponds to a structural state $\mathbf{s}=(V,E,\mathbf{G},\mathbf{A})$, with a feasible set $\mathcal{M}=\{\mathbf{s}:\mathbf{c}(\mathbf{s})=\mathbf{0},\,\mathbf{h}(\mathbf{s})\leq\mathbf{0}\}$. The three types of constraints are geometric laws (e.g., symmetry constraints on member length/orientation), topological connectivity (the graph $\mathcal{G}=(V,E)$ determines which nodes are collinear/coplanar), and physical feasibility (inequalities such as non-intersection and support conditions). These constraints are used during sample construction to *filter out ambiguous samples*, leaving questions with uniquely determined candidate rankings; models only see the image and do not explicitly receive the constraints during inference.
    *   Design Motivation: Use constraints to make the ground truth uniquely determinable, fundamentally avoiding the ambiguity of "multiple 3D configurations explaining the same image," and forcing models to truly recover 3D structure rather than relying on appearance priors.

2.  **Ranking-style VQA Evaluation Protocol**:
    *   Function: Replaces traditional binary or multiple-choice questions with full-ranking questions of $K \in \{3,4\}$ candidates, requiring the model to parse the relationship between all pairs of candidates.
    *   Mechanism: Each question provides a candidate set $\mathcal{C}=\{c_i\}_{i=1}^K$ and a criterion function $f_\tau(\mathbf{s}, c)$ (e.g., "centroid height relative to the ground," "angle between the main direction and the ground," "convex hull volume of node groups"). The ground truth is $\pi^\star=\arg\mathrm{sort}_{\pi\in S_K}\bigl(f_\tau(\mathbf{s}, c_{\pi(1)}), \dots, f_\tau(\mathbf{s}, c_{\pi(K)})\bigr)$; the model must output a parsable Python list representing the complete permutation. Both Taskwise Accuracy (full ranking exact match) and Pairwise Accuracy (pairwise consistency) metrics are reported.
    *   Design Motivation: Compared to binary questions, the random baseline for full-ranking questions is only $\sim 12.85\%$ ($1/24$ when $K=4$), significantly increasing the difficulty of guessing correctly; making the right choice requires parsing all $\binom{K}{2}$ pairwise relationships, marginalizing strategies based on "guessing one or two."

3.  **10 Task Categories Covering Geometry + Topology + Multi-View**:
    *   Function: Covers key dimensions of spatial reasoning within "structure-constrained spaces" as much as possible.
    *   Mechanism: 6 categories in the Geometric group—Ground Height (ranking 4 components by centroid height), Ground Angle (ranking by the angle of the main direction with the ground), Dimension (ranking by main direction length), Relative Distance (ranking 3 component groups by minimum axial distance), Area (ranking 3 node groups by planar convex hull area), Volume (ranking 3 node groups by 3D convex hull volume); 2 categories in the Topological group—Hop Distance (ranking by shortest path hops in the connectivity graph), Cycle Length (ranking by minimum cycle length); plus two Multi-View subsets, each providing two images—one highlighting a reference Member 0 and one highlighting the target—forcing cross-view correspondence.
    *   Design Motivation: A single task (like distance estimation) is easily solved by existing priors. The combination of 10 categories forces the model to simultaneously possess various abilities such as mental rotation, cross-section reasoning, occlusion reasoning, and load-path reasoning within the same benchmark, allowing for fine-grained diagnosis of model deficiencies.

### Loss & Training
The benchmark is for evaluation only and does not train any models. All 31 VLMs perform zero-shot inference under a unified protocol (temperature=0, longest image side resized to 512 pixels) using task-specific prompt templates.

## Key Experimental Results

### Main Results
Table 2 excerpts the Taskwise Accuracy of representative models on SSI-Bench (Geom. Mean, Topo. Mean, and Total Mean). Full results for all 10 tasks are available in the original paper.

| Model | Geom. Mean | Topo. Mean | Total Mean | vs Random (12.85%) |
| :--- | :--- | :--- | :--- | :--- |
| Human (Average) | ~91 | ~89 | **91.60** | +78.75 |
| Gemini-3-Flash (proprietary) | ~33 | ~32 | **33.60** | +20.75 |
| GPT-5.2 | ~30 | ~26 | 29.10 | +16.25 |
| Gemini-3-Pro | ~29 | ~29 | 29.50 | +16.65 |
| Seed-1.8 | ~25 | ~29 | 25.90 | +13.05 |
| GLM-4.6V (best open-source) | ~22 | ~23 | 22.20 | +9.35 |
| Qwen3-VL-235B-A22B | ~21 | ~24 | 21.90 | +9.05 |
| InternVL3.5-2B (worst large) | ~12 | ~7 | 11.10 | −1.75 |
| Random Guessing | 12.85 | 12.85 | 12.85 | 0 |

### Ablation Study
The authors conducted a comparison of Gemini-3-Pro (high vs. low thinking) and Qwen3-VL-30B-A3B (Thinking vs. Instruct).

| Setting | w/o Thinking | w/ Thinking | Gain |
| :--- | :--- | :--- | :--- |
| Gemini-3-Pro (low → high) | 27.1% | 29.5% | +2.4 |
| Qwen3-VL-30B-A3B (Instruct → Thinking) | 20.6% | 22.5% | +1.9 |

### Key Findings
*   Huge gap between VLMs and humans: The strongest proprietary model, Gemini-3-Flash, only achieves 33.60%, and the best open-source GLM-4.6V 22.20%, leaving a 60+ point chasm compared to the human 91.60%; many open-source models are close to the 12.85% random baseline, indicating that SCSR cannot be bypassed by shallow 2D heuristics.
*   Significant proprietary vs. open-source rift: The ceiling for all open-source models is around 22%, trailing the Gemini-3 series by 10+ points; meanwhile, GLM-4.5V to 4.6V only gained +0.8 points, suggesting that *scaling up alone is insufficient*.
*   Limited and non-monotonic thinking gains: Thinking token usage vs. accuracy does not increase monotonically; it peaks at moderate usage and decreases as more tokens are used. The correlation between token usage and effective reasoning is weak, as redundant tokens often correspond to "repeatedly perseverating on incorrect 3D hypotheses."
*   Negative thinking gains on Multi-View and Volume tasks: For tasks relying on globally consistent 3D reconstruction, longer reasoning often amplifies errors.

### Error Analysis (Human Diagnosis of 100 Gemini-3-Pro Questions)
The authors summarized four typical failure modes: component extent errors (treating visible fragments as the whole under occlusion), object recognition errors (confusing stair treads with diagonal braces, or treating diagonal bars as horizontal), calculation and comparison logic errors (optimizing projected area instead of volume, using vertical height instead of slanted height), and view fusion errors (failing to find the Member 0 correspondence in multi-view settings).

## Highlights & Insights
*   Using "structural constraints" as implicit priors for sample construction—rather than explicit inputs—cleverly turns the benchmark itself into a probe for 3D grounding. Models must infer 3D from images while the authors guarantee ground truth uniqueness; this approach can be directly transferred to fields like robotic grasping and medical anatomical reasoning.
*   Ranking tasks are an underrated choice for evaluation: their low random baseline, avoidance of single-item guessing, and requirement for full relationship parsing make them more suitable than common binary or multiple-choice formats for measuring "true understanding."
*   The finding that "Thinking gains are only marginal and token usage vs. accuracy is non-monotonic" is crucial—it suggests that the bottleneck for current reasoning-enhanced VLMs lies in 3D representation rather than reasoning length; simply adding chain-of-thought cannot solve SCSR.
*   Error classifications (extent/recognition/calculation/view) can be directly used to design targeted improvements: e.g., extent errors could be aided by part segmentation, and view fusion by geometric correspondence learning.
*   The comparison across 31 models + human baseline + random baseline is very thorough, making the paper's "diagnostic value" far greater than its "leaderboard value."

## Limitations & Future Work
*   The scale of 1,000 questions is relatively small, with a high proportion of geometric tasks; topological tasks (Hop Distance, Cycle Length) only have a few hundred samples, leading to weaker statistical power for fine-grained trends within small groups.
*   Existing images primarily come from Unsplash/Pexels/Pixabay, so structural types are still dominated by "aesthetically pleasing" bridges, towers, and roofs; industrial-grade CAD/BIM scenarios (piping layouts, load flow paths) are not yet covered.
*   In the Multi-View subset, some images were supplemented by the authors' own photos, introducing potential view-pairing bias; expanding to 6-view or even NeRF/3DGS rendered images could provide a more comprehensive diagnosis of 3D consistency.
*   The evaluation is entirely zero-shot, without exploring whether providing the model with "an additional sketch or point cloud as an aid" could cross the 33% threshold; this is a clear direction for follow-up.
*   The error analysis is represented only by Gemini-3-Pro; whether these conclusions generalize to other model families requires further verification.

## Related Work & Insights
*   Complementary to *scene-centric* spatial benchmarks like VSI-Bench, SpatialRGPT, and SpatialVLM: while those evaluate distance/direction in unconstrained daily environments, SSI-Bench evaluates ranking in constrained scenes, allowing for a cross-diagnostic of VLM spatial capability spectrums.
*   Linked with multi-view benchmarks like MMSI-Bench, ViewSpatial-Bench, and MindCube: the Multi-View subset in Ours directly aligns with this direction and reports differences from single-view tasks.
*   Compared to structural understanding benchmarks like PartNet, 3DCoMPaT++, ABC, and GeoQA: those works provide explicit part labels or geometric outputs, whereas SSI-Bench acts as an implicit probe—requiring only answers to spatial relationship questions, letting the model reconstruct the structure internally, which is closer to real reasoning scenarios.
*   Inspiration for future VLM training: component-level segmentation supervision, cross-view correspondence learning, and explicit 3D intermediate representations (e.g., NeRF/3DGS distillation) might be more effective at breaking the SCSR bottleneck than simply adding chain-of-thought; meanwhile, the *constrained sample construction* method can be extended to medical imaging (anatomical constraints), autonomous driving (road geometry constraints), and other high-certainty domains as a next-generation benchmark paradigm.

## Rating
*   Novelty: TBD
*   Experimental Thoroughness: TBD
*   Writing Quality: TBD
*   Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SpatialScore: Towards Comprehensive Evaluation for Spatial Intelligence](../../CVPR2026/multimodal_vlm/spatialscore_towards_comprehensive_evaluation_for_spatial_intelligence.md)
- [\[ICLR 2026\] On the Generalization Capacities of MLLMs for Spatial Intelligence](../../ICLR2026/multimodal_vlm/on_the_generalization_capacities_of_mllms_for_spatial_intelligence.md)
- [\[CVPR 2026\] Scaling Spatial Intelligence with Multimodal Foundation Models](../../CVPR2026/multimodal_vlm/scaling_spatial_intelligence_with_multimodal_foundation_models.md)
- [\[ICML 2026\] ReVSI: Rebuilding Visual Spatial Intelligence Evaluation for Accurate Assessment of VLM 3D Reasoning](revsi_rebuilding_visual_spatial_intelligence_evaluation_for_accurate_assessment_.md)
- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](../../ICLR2026/multimodal_vlm/spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)

</div>

<!-- RELATED:END -->
