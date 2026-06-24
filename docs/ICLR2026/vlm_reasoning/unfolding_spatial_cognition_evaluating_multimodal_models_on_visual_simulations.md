---
title: >-
  [Paper Note] Unfolding Spatial Cognition: Evaluating Multimodal Models on Visual Simulations
description: >-
  [ICLR 2026][VLM Reasoning][Spatial Cognition] The authors propose the STARE benchmark, which systematically evaluates Multimodal Large Language Models (MLLMs) using approximately 4,000 spatial problems requiring "multi-step visual simulation" (2D/3D transformations, cube folding, tangrams, viewpoint, and temporal reasoning). The study finds that while models perform near human levels on simple 2D transformations, their performance drops to near-random on tasks requiring step-…
tags:
  - "ICLR 2026"
  - "VLM Reasoning"
  - "Spatial Cognition"
  - "Visual Simulation"
  - "Multimodal Benchmark"
  - "Mental Rotation"
  - "MLLM"
date: 2026-05-08
content_hash: 2369587632ae4314
---

# Unfolding Spatial Cognition: Evaluating Multimodal Models on Visual Simulations

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=fbGmSV6tUw](https://openreview.net/forum?id=fbGmSV6tUw)  
**Area**: Multimodal VLM / Spatial Reasoning Evaluation  
**Keywords**: Spatial Cognition, Visual Simulation, Multimodal Benchmark, Mental Rotation, MLLM

## TL;DR
The authors propose the STARE benchmark, which systematically evaluates Multimodal Large Language Models (MLLMs) using approximately 4,000 spatial problems requiring "multi-step visual simulation" (2D/3D transformations, cube folding, tangrams, viewpoint, and temporal reasoning). The study finds that while models perform near human levels on simple 2D transformations, their performance drops to near-random on tasks requiring step-by-step "mental imagery" such as folding or tangrams. Furthermore, models fail to consistently utilize intermediate visual steps—revealing a fundamental gap in current MLLMs regarding non-verbal, serialized visual simulation capabilities.

## Background & Motivation

**Background**: Spatial reasoning is a cornerstone of human intelligence. Cognitive psychology provides long-standing evidence that humans solve spatial problems by "running a visual simulation in the brain." Shepard & Metzler discovered that the time required to determine if two 3D perspective drawings represent the same object increases linearly with the rotation angle, suggesting a "simulative mental rotation." Hegarty found that when humans understand mechanical diagrams, they perform "mental animation" by deducing motion component by component. This ability to transform objects step-by-step in the mind and predict physical interactions supports daily tasks such as assembling furniture, reading maps, and interpreting assembly drawings.

**Limitations of Prior Work**: Although dynamic visual simulation is ubiquitous in the real world, existing benchmarks for evaluating MLLMs mostly focus on **static recognition** or problems that can be "reformulated into linguistic reasoning" (e.g., CLEVR, various VQA tasks). These benchmarks measure perception—"answering at a glance"—rather than simulation, which requires "reasoning through several steps after viewing." Even VSI-Bench, which emphasizes spatial memory, focuses on estimating spatial relationships from video rather than explicit step-by-step simulation.

**Key Challenge**: Humans excel specifically at **non-verbal, multi-step visual simulation** (folding a 2D grid into a 3D cube, assembling fragments into a target shape, imagining a scene from a different perspective). This capability is largely absent from existing benchmarks, primarily because the cost of labeling intermediate visual states is high. Consequently, "simulation processes" have long been treated as optional and are rarely explicitly provided or evaluated. Thus, it remains unclear whether models can perform visual simulation or if providing intermediate steps actually assists them.

**Goal**: To construct a spatial reasoning benchmark specifically designed to evaluate tasks "better suited for multi-step visual simulation," covering a continuous difficulty spectrum from basic geometric transformations to real-world spatial cognition, while distinguishing whether model failures stem from perceptual defects or high-level reasoning deficits.

**Key Insight**: The authors categorize tasks into two types based on whether "intermediate steps can be explicitly visualized." The first two levels (basic geometric transformations, folding/tangrams) allow every step to be rendered; humans indeed draw or imagine intermediate states when solving these. The final level (viewpoint, temporal reasoning) requires more abstract, implicit mental simulation without clear intermediate visual cues. By evaluating both "provided intermediate visual steps" and "not provided" settings across task categories, the ability of the model to "mentally simulate" can be decoupled from its ability to "utilize visualization."

**Core Idea**: To quantify the "serialized visual simulation" capability—neglected by existing benchmarks—using a set of procedurally synthesized spatial tasks with controllable difficulty and step-by-step intermediate visualizations, highlighting the gap between MLLMs and human performance.

## Method

### Overall Architecture
STARE (Spatial Transformations and Reasoning Evaluation) is a benchmark rather than a model, centered on "task design + evaluation protocol." It organizes spatial cognition into three levels of complexity: **Basic Geometric Transformations** (2D/3D rotation, translation, scaling, reflection, shearing), **Integrated Spatial Reasoning** (cube folding, tangram puzzles), and **Real-world Spatial Reasoning** (temporal frame completion, viewpoint reasoning). Each problem is formatted as a multiple-choice or yes/no question, accompanied by carefully designed image-text prompts, totaling approximately 4,000 tasks.

The critical evaluation dimension is the "visual simulation" axis. For synthetic tasks where every step can be visualized (the first two levels), the benchmark constructs two settings: **No Intermediate Visual Steps** (only the initial image + optional text instructions, forcing the model to rely on internal simulation) and **Provided Intermediate Visual Steps** (all intermediate visualizations except the final step are fed in an interleaved image-text format). Real-world tasks are evaluated in a single-image setting. All data is generated procedurally (Matplotlib for 2D/folding, Blender for 3D rendering, Objectron for temporal frames, and HM3D + Habitat simulator for viewpoint images). Difficulty is explicitly controlled via distractor similarity and the number of simulation steps. Finally, accuracy (multiple-choice) and $F1$ (yes/no) are aggregated into a macro-averaged total score. Five undergraduate students performed the same tasks to establish a human baseline for accuracy and **response time**.

### Key Designs

**1. A Three-Tier Hierarchical Task System: Spreading Spatial Cognition into a Continuous Difficulty Spectrum**

To address the issue that existing benchmarks are either purely static or linguistically reducible, STARE deliberately ranks tasks into three levels based on their "dependency on visual simulation." The first level, Basic Geometric Transformations, serves as the foundation. it includes "visual analogy" (applying the $A \to A'$ transformation sequence to $B$) and "instruction-based" (selecting the result of text descriptions like "rotate clockwise $90^\circ$ and scale up") tasks. The second level, Integrated Reasoning, sequences multiple basic operations: Cube Folding requires determining if a 2D grid with labeled faces can form a specific cube, and Tangrams require determining if given pieces can fill a target grid. The third level, Real-world Tasks, simulates daily scenarios: Temporal Frame Reasoning requires completing a masked frame from a four-frame video sequence, and Viewpoint Reasoning requires selecting a first-person perspective based on an agent's position and orientation in a top-down map.

The value of this design lies in the fact that intermediate steps in the first two levels "can be explicitly drawn," allowing for the simultaneous evaluation of explicit vs. implicit simulation. The third level requires more abstract implicit simulation due to the lack of clear intermediate cues. Together, they reveal how model performance decays as the "required simulation steps/abstraction" increase.

**2. Dual-Axis Evaluation Protocol (Explicit/Implicit): Decoupling "Internal Simulation" from "Visual Utilization"**

This is the most critical design distinguishing STARE from previous benchmarks. It does not merely ask if the model is correct; for each synthetic task, it runs two settings: **No Intermediate Visual Simulation** (question only, or question + text steps) and **Provided Intermediate Visual Simulation** (interleaved visualizations of transformation steps, omitting the final one). The difference measures the model's ability to "utilize intermediate visual information." The authors chose interleaved sequences over a single large composite image to better reflect real-world usage.

This comparison reveals a counter-intuitive phenomenon: while visual simulation is consistently beneficial for humans (and significantly reduces response time), the gain for models is **inconsistent**. GPT-4o gained $+11.5\%$ in 2D, and Claude gained $+8.6\%$ in Tangrams, yet Gemini-2.0 Flash dropped by $-2.1\%$ in Folding, and InternVL2.5-78B dropped by $12.5\%$ in Tangrams. In other words, models cannot "relay" intermediate visual steps into their reasoning like humans do.

**3. Procedural Synthesis + Controllable Difficulty Knobs: Making "Simulation Steps" and "Distractor Similarity" Experimental Variables**

To solve the issues of high labeling costs and uncontrollable difficulty, STARE uses procedural generation throughout. 2D/3D shapes are rendered via Matplotlib/Blender. Cube folding utilizes a step-by-step algorithm (starting from a static base, folding connected faces, and detecting overlap/disconnection errors) to produce visualizations with face boundaries. Tangrams are generated by randomly partitioning $3 \times 3$ or $4 \times 4$ grids, followed by random rotations for valid solutions or modifications for invalid ones. Temporal frames are extracted from Objectron, and viewpoint images are generated using Habitat in HM3D indoor environments.

Controllability is implemented via two knobs: **Three Difficulty Tiers** defined by distractor similarity—Easy (two distractors look significantly different), Medium (one obvious distractor), and Hard (all distractors are visually similar, forcing focus on the transformation itself); and **Simulation Steps** ($N=1, 2, 3$), used to observe performance decay. These allow for fine-grained analysis, such as identifying accuracy drops as difficulty increases or the anomalous peak at $N=2$ without visual simulation.

**4. Perception vs. Reasoning Attribution Probes: Proving the Bottleneck is Multi-Step Simulation, Not Low-Level Perception**

Mere performance tracking is insufficient; the authors designed probes to attribute failures to "perception" vs. "reasoning." The core method involves **providing the model with the fully simulated final result**, reducing the task to "matching the final state to the correct option." In 2D/3D transformations, accuracy only rose slightly ($+4.2\%/+2.8\%$), but in Cube Folding and Tangrams, it soared to $100\%$ and $91.6\%$, respectively. This proves that once perceptual complexity is minimized, the models can solve the task—the failure is not in final-step recognition. Further sub-probes for Folding targeted 2D perception (color, connectivity) and 3D perception (whether a face is folded): color was $100\%$, connectivity $94.1\%$, but "judging if a face is folded" was only $57.4\%$, identifying 3D perception as a weak link.

Additionally, three control experiments were conducted: translating visual tasks into pure text descriptions (e.g., "red square at $(3,4)$ with size $2$"), which improved 2D performance from $75\%$ to $87\%$ but left 3D Folding at $\sim 57\%$, showing text cannot replace 3D perception; providing text-only reasoning steps, which yielded almost no gain for Folding and caused a crash in Tangrams performance ($62.4\% \to 34.7\%$), revealing that models rely on shortcuts like "summing piece areas" rather than true spatial simulation.

## Key Experimental Results

### Main Results

Evaluations were conducted on 6 closed-source models, 5 open-source models, and 5 humans. Total scores (macro-average, $F1$ for some tasks) and representative sub-tasks are as follows (↑/↓ denotes change after adding Visual Simulation - VSim):

| Model | 2D Transfer (✗/✓VSim) | 3D Transfer (✗/✓) | Cube Folding (✗/✓) | Tangram (✗/✓) | Viewpoint | Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Random | $25.0 / 25.0$ | $25.0 / 25.0$ | $50.0 / 50.0$ | $50.0 / 50.0$ | $25.0$ | $34.8$ |
| GPT-4o | $71.2 / 82.7$ (↑11.5) | $65.5 / 68.4$ | $50.3 / 52.2$ | $52.5 / 51.5$ (↓1.0) | $38.7$ | $53.9$ |
| Claude-3.5 Sonnet | $65.9 / 71.4$ | $51.5 / 57.8$ | $52.3 / 51.6$ (↓0.7) | $59.0 / 67.6$ (↑8.6) | $26.1$ | $53.1$ |
| o1 | $81.8 / 87.7$ | $67.9 / 71.6$ | $51.3 / 53.4$ | $55.3 / 53.2$ (↓2.1) | $36.8$ | $57.2$ |
| o3 | $87.5 / 89.3$ | $75.2 / 78.4$ | $68.4 / 79.4$ (↑11.0) | $68.6 / 82.1$ (↑13.5) | $42.8$ | $68.1$ |
| Qwen2.5-VL-72B | $45.2 / 48.5$ | $43.0 / 49.1$ | $35.2 / 53.4$ (↑18.2) | $61.2 / 56.9$ | $26.0$ | $42.3$ |
| Human (Acc) | $96.8 / 98.6$ | $94.6 / 96.9$ | $98.3 / 98.9$ | $91.5 / 95.8$ | $98.1$ | $97.1$ |
| Human (Time/s) | $14.2 / 11.0$ | $17.1 / 12.5$ | $13.7 / 5.2$ | $28.0 / 10.1$ | $18.4$ | – |

The strongest model, o3, scored 68.1, far below the human 97.1. Among non-reasoning models, GPT-4o (53.9) performed best but remained near random on Folding/Tangrams. Humans achieved near-perfect scores, but Tangrams without VSim took 28.0s, dropping to 10.1s with intermediate steps—proving the tasks are multi-step and cognitively demanding, with visual simulation acting as a consistent "accelerator" for humans.

### Ablation Study

| Probe/Setting | 2D Transfer | 3D Transfer | Cube Folding | Tangram | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Provide Final State (GPT-4o) | $86.9$ (↑4.2) | $71.2$ (↑2.8) | $100.0$ | $91.6$ | Performance near perfect when reduced to matching, indicating the bottleneck is not final recognition. |
| Text-only | $87.5$ | $64.7$ | $57.0$ | $72.6$ | Text helps in 2D but fails in 3D Folding. |
| Image-only | $75.1$ | $67.7$ | $56.0$ | $62.5$ | Contrast for text-only. |
| Question-only $\to$ +Steps | – | – | $50.2 \to 50.4$ | $62.4 \to 34.7$ | Text steps provide no benefit or hurt performance (shortcuts). |
| Folding 3D Perception | – | – | $57.4$ | – | Color/Connectivity high; "is face folded" is the weak point. |

### Key Findings
- **The primary bottleneck is multi-step visual simulation, not low-level perception**: Models achieve $100\%/91.6\%$ when perception is minimized but struggle in normal settings, showing they "see clearly but cannot simulate."
- **Models fail to reliably utilize intermediate visual steps**: While VSim consistently benefits and saves time for humans, its impact on models is inconsistent (e.g., GPT-4o gained $11.5\%$ in 2D but Gemini-2.0 Flash dropped $2.1\%$ in Folding).
- **3D perception and folding judgment are specific weaknesses**: Connectivity perception is acceptable ($94.1\%$), but "judging if a face is folded" ($57.4\%$) explains the limited gain from VSim in Folding tasks.
- **Synthetic tasks predict real-world performance**: The Pearson correlation between synthetic and real-world task averages across 11 models is high ($r \approx 0.88$, rising to $0.97$ with humans), validating the transferability of abstract spatial capabilities.
- **Anomalous peak at $N=2$ without VSim**: Two-step transformations often combine a simple operation (scaling) with a difficult one (shearing); models may "guess correctly" via the simple operation, with performance only collapsing as complexity compounds at $N=3$.

## Highlights & Insights
- **The "Dual-Axis Evaluation" is the most ingenious design**: By contrasting "provided" vs. "not provided" VSim for every task, the authors decouple "internal mental imagery" from "utilizing external visualization." This methodology reveals gaps that accuracy scores alone cannot.
- **Using "Final State Feeding" to attribute bottlenecks**: Reducing the task to pure visual matching is a clean trick to exclude "final-step misrecognition" as an explanation for failure, a method transferable to any "perception vs. reasoning" evaluation.
- **Human response time as a metric**: Recording human seconds (e.g., Tangrams dropping from $28s$ to $10.1s$) quantifies the multi-step nature of the tasks, providing hard evidence that these are not simple "at-a-glance" recognition problems.
- **Synthetic-to-Real correlation analysis**: The $r \approx 0.88$ correlation provides real-world validity to spatial capabilities measured on procedurally synthesized problems, mitigating concerns about the "artificiality" of synthetic benchmarks.

## Limitations & Future Work
- **Evaluation only, no methodology provided**: STARE diagnoses the visual simulation gap but does not propose how models might acquire such capabilities (e.g., generating intermediate steps or integrating explicit simulators).
- **Dominance of multiple-choice/binary formats**: While easy to evaluate, these allow "guessing" and shortcuts (e.g., Tangrams solved via area summation reaching $\sim 75\%$). Open-ended generative spatial reasoning is not covered.
- **Smaller scale and single-image format for real-world tasks**: Temporal (471) and Viewpoint (250) tasks are small and lack intermediate steps, making the characterization of "implicit mental simulation" relatively coarse.
- **Limited human baseline scale**: Only 5 undergraduate students were evaluated. While they achieved near-perfect accuracy, the robustness across populations with varying spatial skills remains undiscussed.
- **Future directions**: Exploring training or inference paradigms that force models to explicitly "draw or predict intermediate frames" before reasoning, and using 3D folding perception ($57.4\%$) as a target for improvement.

## Related Work & Insights
- **vs. Traditional Visual Reasoning (CLEVR / Raven)**: These focus on static recognition or linguistically reducible tasks. STARE targets non-verbal tasks requiring multi-step simulation and explicitly provides intermediate visualizations.
- **vs. VSI-Bench**: VSI-Bench emphasizes mental imagery for spatial memory and estimation from video. STARE provides procedurally synthesized tasks with controllable difficulty to evaluate "step-by-step simulation" itself.
- **vs. Cognitive Science Research**: This work operationalizes classic paradigms (Shepard-Metzler mental rotation, Hegarty mental animation) into a large-scale benchmark for MLLMs, replicating the "increased difficulty, increased time" phenomenon found in human studies.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first spatial cognition benchmark to treat "multi-step visual simulation" as a standalone dimension and decouple simulation capabilities via a dual-axis protocol.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 models + human baseline, with 6 attribution probes (Q1–Q6) to pinpoint bottlenecks.
- Writing Quality: ⭐⭐⭐⭐ Clear task stratification and evaluation settings; cognitive scientific motivation is well articulated, though tables are slightly dense.
- Value: ⭐⭐⭐⭐⭐ Exposes a critical flaw in current MLLMs regarding spatial simulation and provides a metric for future models capable of mental imagery.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] VisuLogic: A Benchmark for Evaluating Visual Reasoning Capabilities of Multimodal Large Models](visulogic_a_benchmark_for_evaluating_visual_reasoning_in_multi-modal_large_langu.md)
- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[CVPR 2026\] EgoMind: Activating Spatial Cognition through Linguistic Reasoning in MLLMs](../../CVPR2026/vlm_reasoning/egomind_activating_spatial_cognition_through_linguistic_reasoning_in_mllms.md)
- [\[ICLR 2026\] GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models](gtr-bench_evaluating_geo-temporal_reasoning_in_vision-language_mod.md)
- [\[CVPR 2026\] SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models](../../CVPR2026/vlm_reasoning/spatialqa_a_benchmark_for_evaluating_spatial_logical_reasoning_in_vision-languag.md)

</div>

<!-- RELATED:END -->
