---
title: >-
  [Paper Note] FloorplanQA: A Benchmark for Spatial Reasoning in LLMs Using Structured Representations
description: >-
  [ICML 2026][LLM Reasoning][Paper Note] FloorplanQA systematically diagnoses the "pure symbolic spatial reasoning" capabilities of 15 frontier LLMs using 2,000 JSON/XML-formatted 2D indoor layouts and 16,000 geometric questions (distance, visibility, path, placement, etc.). The study finds that while models can compute simple distances, they consistently fai
tags:
  - ICML 2026
  - LLM Reasoning
date: 2026-05-08
content_hash: 03da6f85385d38bb
---
# FloorplanQA: A Benchmark for Spatial Reasoning in LLMs Using Structured Representations

**Conference**: ICML 2026  
**arXiv**: [2507.07644](https://arxiv.org/abs/2507.07644)  
**Code**: https://OldDelorean.github.io/FloorplanQA/ (Project Page)  
**Area**: LLM Evaluation / Spatial Reasoning / Indoor Layouts  
**Keywords**: Spatial Reasoning, Structured Representation, JSON Layout, Geometric Inference, LLM Diagnostic Benchmark  

## TL;DR
FloorplanQA systematically diagnoses the "pure symbolic spatial reasoning" capabilities of 15 frontier LLMs using 2,000 JSON/XML-formatted 2D indoor layouts and 16,000 geometric questions (distance, visibility, path, placement, etc.). The study finds that while models can compute simple distances, they consistently fail in union operations, planning, and constraint satisfaction; furthermore, Python tool augmentation fixes arithmetic errors but fails to resolve algorithmic failures.

## Background & Motivation

**Background**: LLMs have demonstrated surprising capabilities in structured reasoning. In scenarios such as architectural design, assisted planning, and embodied interaction, models are increasingly required to directly process layouts in JSON format (coordinates, dimensions, orientations) rather than relying on images. This implies that models must perform reasoning on symbolic geometry rather than perception on pixels.

**Limitations of Prior Work**: Existing spatial reasoning benchmarks either remain at the level of qualitative relations ("to the left of" / "above"), convert layouts back into images for VLM evaluation, or implicitly examine navigation within embodied tasks like ALFRED/R2R. There is no clean benchmark to independently probe whether a model, given a JSON layout, can calculate a 1.7-meter corridor, determine if a $2 \times 3$ meter table fits, or find an obstacle-avoiding path from a stove to a door.

**Key Challenge**: In real-world deployment, models are often used as "front-ends" for geometric solvers or code generators. However, the viability of these toolchains depends on the model possessing basic spatial intuition—if the model cannot maintain geometric invariants without tools, tool augmentation will only amplify errors. Prior works like LayoutGPT and FirePlace evaluate the realism of generated results, but none systematically test the geometric consistency of models under symbolic input.

**Goal**: Construct a **purely symbolic, automatically scorable diagnostic benchmark covering metric, topological, and dynamic categories** to quantitatively answer what today’s LLMs can do with structured layouts, where they fail, and why.

**Key Insight**: Feed the abstractions actually used by architects (room polygons + object bounding boxes + openings) directly to the model. All ground truth answers are calculated using deterministic geometric algorithms (independent of another model's scoring) to eliminate circular evaluation logic.

**Core Idea**: Utilize a triplet of "**structured JSON layouts + templated geometric questions + rule-based scoring**" to transform spatial reasoning from an open-ended task into a diagnostic protocol with itemized scoring, proposing a three-tier capability gradient: Metric, Topology, and Dynamic.

## Method

### Overall Architecture
FloorplanQA aims to achieve one objective: given a purely symbolic 2D indoor layout (JSON/XML containing only coordinates, dimensions, and semantic labels), it measures which geometric properties an LLM can calculate and where it fails. The benchmark consists of three components: 2,000 layouts (1,800 self-generated using Gemini 2.5 Pro, 200 extracted from real HSSD scenes), 8 geometric questions per layout totaling 16,000 questions, and a zero-shot evaluation protocol (requiring a `Final answer` line, parsed via regex and scored automatically using corresponding metrics). All layouts share a right-handed 2D coordinate system, with objects represented as polygons with semantic labels: synthetic data uses 4-vertex axis-aligned rectangles, while HSSD uses arbitrary polygons.

Layout generation follows a two-stage pipeline: first, a prompt constrains the room geometry (shape, adjacency, corridor clearance, symmetrical partitioning), then furniture is filled based on stylized templates (e.g., a bedroom must contain a bed and storage). Approximately one-third of candidates are filtered by a rule-based validator (e.g., unreasonable placements like sofas blocking doors). 3D scenes from HSSD are projected to 2D and simplified using the Douglas-Peucker algorithm ($\epsilon=0.01$m).

### Key Designs

**1. Three-Tier Task Taxonomy (Metric / Topology / Dynamic): Turning Leaderboards into Capability Profiles**

Traditional single-difficulty QA yields a mean score that fails to reveal where a model's reasoning breaks down. FloorplanQA categorizes 8 question types into three tiers based on reasoning intensity: **Metric** (pairwise distance, view angle) requiring basic coordinate arithmetic; **Topology** (free space, maximum contained rectangle, placement feasibility, visibility) requiring set operations and constraint satisfaction; and **Dynamic** (repositioning, shortest path) requiring reasoning during layout changes. These tiers correspond to typical failure modes: arithmetic, constraint reasoning, and multi-step planning. The leaderboard thus becomes a capability profile. For deterministic scoring, each question is assigned a format code: N (scalar, relative error $\leq 2\%$, relaxed to $5\%$ for free space), B (boolean), L (list, set equality), and S (sequence, requiring obstacle avoidance and Fréchet distance $\leq 0.6$m). Object names in templates are replaced with actual instances (e.g., `fridge_1`), ensuring unambiguous referencing.

**2. Deterministic Scoring to Remove Evaluation Loops: Completely Decoupling Generation from Scoring**

Using a model like Gemini to both generate layouts and evaluate responses introduces circular leakage. FloorplanQA severs this path at the protocol level: layouts are pre-generated and frozen, and all ground truth is precisely calculated by geometric libraries such as `shapely` (polygon union area, A* paths, shoelace formula for centroids), independent of any LLM output. During evaluation, models are forced to output concise structured reasoning followed by a `Final answer:` line. Responses are regex-extracted and compared to reference answers; missing answers or format errors are scored as 0. The protocol also records token truncation rates and reports both "raw accuracy" and "completed response accuracy"—the former prevents reasoning models from inflating scores with hyper-long outputs, while the latter serves as a performance upper bound.

**3. Multi-Dimensional Ablation Probes (Input Format / Semantic Perturbation / Tool Augmentation / VLM Rendering): Decomposing Failures into Four Independent Dimensions**

A single accuracy metric hides "why" a model fails. This paper designs four sets of ablations: (a) JSON ↔ XML equivalent rewriting shows accuracy fluctuations within $\pm 3$ pp, indicating models capture layout semantics rather than serialization syntax; (b) Object label permutation (e.g., swapping "sofa" for "chair" labels while keeping geometry constant) causes Repositioning accuracy to plummet from $60.5\%$ to $40.0\%$ (gpt-oss-120B), revealing that movement tasks rely on linguistic associations rather than geometric properties; (c) GPT-4.1 with a Python interpreter improves arithmetic tasks by $+30$ to $+43$ pp but regresses on Max Box ($-4.5$) and Shortest Path ($-12.5$); (d) Three rendering types (bounding boxes, icons, AI-generated photos) plus JSON stay within $\pm 4$ pp of the JSON-only baseline, while pure image accuracy drops to $19\%$-$40\%$. These probes isolate "arithmetic error / algorithmic error / visual irrelevance / semantic bias."

## Loss & Training
This work is an evaluation paper and does not train any models. All 15 LLMs were evaluated at `temperature=0`, zero-shot, under a unified prompt schema. Large models were allocated 12,288 tokens, mid-sized models 8,192 tokens, and GPT-5 was allocated 4,096 tokens due to its forced reasoning mode.

## Key Experimental Results

### Main Results
A total of 15 models (7 reasoning: GPT-5, gpt-oss-120B, etc.; 8 general: Claude Sonnet 4, GPT-4.1, etc.) were evaluated on 16,000 questions. The results exhibit a three-tier gradient:

| Task Category | Representative Type | Mean Accuracy (Synthetic) | HSSD Accuracy | Primary Failure Mode |
|----------|----------|--------------------|-------------|-------------|
| Metric | Pair Distance / View Angle | 75-95\% | 35-60\% | HSSD polygon centroid calculation (shoelace formula) |
| Topology (Constraint) | Placement / Visibility / Repositioning | 60-85\% | 40-70\% | Multi-constraint joint verification failure |
| Topology (Opt.) + Dynamic | Free Space / Max Box / Shortest Path | 5-45\% | < 30\% | Mistaking "sum of areas" for "union of areas", failed obstacle search |

### Ablation Study

Comparison of tool augmentation (GPT-4.1 + Python interpreter) on HSSD:

| Task | Raw | + Tools | $\Delta$ | Interpretation |
|------|-----|---------|----------|------|
| Pair Distance | 56.0 | 99.0 | $+43.0$ | Arithmetic bottleneck resolved |
| View Angle | 55.0 | 96.0 | $+41.0$ | Arithmetic bottleneck resolved |
| Visibility | 46.5 | 86.5 | $+40.0$ | Explicit geometric checks succeed |
| Repositioning | 47.0 | 83.5 | $+36.5$ | Tools enable collision detection |
| Placement | 64.5 | 95.0 | $+30.5$ | Benefits from precise determination |
| Free Space | 16.0 | 44.0 | $+28.0$ | Unions remain difficult |
| Max Box | 7.5 | 3.0 | $-4.5$ | Tools generate incorrect algorithms |
| Shortest Path | 25.0 | 12.5 | $-12.5$ | Boundary contact misidentified as collision |

### Key Findings
- Kitchen accuracy is the highest—each kitchen layout averages only 0.52 object overlaps, whereas bedrooms/living rooms range from 1.52-1.82, and HSSD reaches 4.39. Higher overlap significantly complicates union calculations and clearance determination.
- Reasoning models score $+10$ to $+40$ pp higher than general models on Free Space and Max Box, but show negligible gains on Shortest Path—indicating extra reasoning budget helps geometric union but not search-based planning.
- Repositioning accuracy drops significantly after object label swapping (e.g., gpt-oss-120B drops from $60.5\%$ to $40.0\%$), proving models rely on linguistic common sense (e.g., "sofa = large furniture") rather than true coordinates.
- Tool augmentation leads to regression in Max Box / Shortest Path. Error cases show that model-generated Python code often treats "boundary contact as collision" or fails to enumerate rectangle orientations. The bottleneck is algorithmic spatial reasoning, not numerical precision.

## Highlights & Insights
- **Diagnostic Protocol vs. Leaderboard**: By grouping questions by reasoning type rather than difficulty, the benchmark identifies which capability dimension a model lacks, rather than providing a single mean score. This "capability profile" approach can be transferred to other symbolic reasoning evaluations in code, math, or logic.
- **Counter-intuitive Degradation of Tool Augmentation**: Intuition suggests a Python interpreter should provide universal improvements; however, it causes regressions in planning tasks. This proves LLM failures are not merely arithmetic—with a calculator, they get geometric problems right but still cannot write correct search algorithms. This is a critical warning for the "agent + tool" paradigm.
- **JSON vs. XML Impact Minimized, Images Help Little**: These clean negative results suggest LLMs understand space at the layout semantic level. Serialized syntax and visual rendering are merely surface packaging. Future research should focus on "geometric inductive biases" rather than "prompt formatting."

## Limitations & Future Work
- Layouts are limited to 2D indoor scenes with mostly axis-aligned furniture; they do not cover 3D height differences, multi-story buildings, outdoor scenes, or dynamic environments.
- Evaluation is zero-shot only—this may underestimate the potential aid of in-context geometric examples.
- 1,800 synthetic layouts were generated by Gemini 2.5 Pro; although models are scored independently, the layout distribution carries that specific model's geometric biases.
- Future Work: (Short-term) Integration with specialized solvers like A*/shapely; (Mid-term) Incorporation of geometric consistency objectives and violation samples during training; (Long-term) Multi-step interactive evaluation—allowing agents to self-verify and correct plans.

## Related Work & Insights
- **vs CLEVR / SpatialSense**: These assess qualitative spatial relations in vision ("left/right/above/below"). FloorplanQA tests quantitative geometry under symbolic coordinates (distance, area, path), providing complementary and more precise difficulty.
- **vs ScanQA / 3DSRBench**: These target VLM Q&A for 3D scenes, emphasizing point cloud/mesh understanding. Ours focuses on pure LLM geometric reasoning under 2D symbolic input.
- **vs LayoutGPT / Holodeck / FirePlace**: Those works use LLMs to generate layouts and evaluate output realism. Ours acts as a probe—given a layout, it tests if a model can infer its geometric properties, effectively the inverse direction.
- **vs Yamada et al. 2023**: They use natural language descriptions of qualitative relations. Ours uses structured JSON + coordinates for precise geometry, refining vague "spatial understanding" into quantifiable "metric/topology/dynamic" tiers.

## Rating
- Novelty: ⭐⭐⭐⭐ The task taxonomy and "diagnostic protocol" are clear, though as a resource, it follows standard benchmark structures.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 15 models × 8 task types × 4 layout categories + 4 ablation sets; comprehensive enough to cover future comparison needs for a year.
- Writing Quality: ⭐⭐⭐⭐ The three-tier taxonomy is well-explained; insights regarding tool-driven regression and the correlation between overlap and accuracy are particularly strong.
- Value: ⭐⭐⭐⭐⭐ Provides the first diagnostic benchmark capable of precise itemized scoring for "LLM + Spatial Reasoning," with direct value for agents, architectural design, and embodied AI.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] CoCoReviewBench: A Completeness- and Correctness-Oriented Benchmark for AI Reviewers](cocoreviewbench_a_completeness-_and_correctness-oriented_benchmark_for_ai_review.md)
- [\[ICML 2026\] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)
- [\[CVPR 2026\] EagleVision: A Dual-Stage Framework with BEV-grounding-based Chain-of-Thought for Spatial Intelligence](../../CVPR2026/llm_reasoning/eaglevision_a_dual-stage_framework_with_bev-grounding-based_chain-of-thought_for.md)
- [\[ICML 2025\] Adversarial Manipulation of Reasoning Models using Internal Representations](../../ICML2025/llm_reasoning/adversarial_manipulation_of_reasoning_models_using_internal_representations.md)
- [\[ICML 2026\] Evaluating Relational Reasoning in LLMs with REL](evaluating_relational_reasoning_in_llms_with_rel.md)

</div>

<!-- RELATED:END -->
