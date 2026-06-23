---
title: >-
  [Paper Note] FloorplanQA: A Benchmark for Spatial Reasoning in LLMs Using Structured Representations
description: >-
  [ICML 2026][LLM Reasoning][Paper Note] FloorplanQA systematically diagnoses the "pure symbolic spatial reasoning" capabilities of 15 cutting-edge LLMs using 2,000 JSON/XML-formatted 2D indoor layouts and 16,000 geometric problems (distance, visibility, pathing, placement, etc.). The study reveals that while models can calculate simple distances, they consis
tags:
  - ICML 2026
  - LLM Reasoning
date: 2026-05-08
content_hash: 49bc0d91eafce75d
---
# FloorplanQA: A Benchmark for Spatial Reasoning in LLMs Using Structured Representations

**Conference**: ICML 2026  
**arXiv**: [2507.07644](https://arxiv.org/abs/2507.07644)  
**Code**: https://OldDelorean.github.io/FloorplanQA/ (Project Page)  
**Area**: LLM Evaluation / Spatial Reasoning / Indoor Layout  
**Keywords**: Spatial Reasoning, Structured Representation, JSON Layout, Geometric Inference, LLM Diagnostic Benchmark  

## TL;DR
FloorplanQA systematically diagnoses the "pure symbolic spatial reasoning" capabilities of 15 cutting-edge LLMs using 2,000 JSON/XML-formatted 2D indoor layouts and 16,000 geometric problems (distance, visibility, pathing, placement, etc.). The study reveals that while models can calculate simple distances, they consistently fail at set unions, planning, and constraint satisfaction. Furthermore, Python tool augmentation fixes arithmetic errors but cannot salvage failures at the algorithmic level.

## Background & Motivation

**Background**: LLMs have demonstrated impressive capabilities in structured reasoning. Scenarios such as architectural design, assistive planning, and embodied interaction have begun requiring models to directly ingest layouts in JSON format (coordinates, dimensions, orientations) rather than relying on images. This implies that models must perform reasoning on symbolic geometry rather than perception on pixels.

**Limitations of Prior Work**: Existing spatial reasoning benchmarks either remain at the level of qualitative relationships ("to the left of" / "above"), re-render layouts into images for VLM evaluation, or implicitly examine navigation within embodied tasks like ALFRED or R2R. There is no clean benchmark that independently probes: "Given a JSON layout, can the model calculate a 1.7m corridor, determine if a 2×3m table fits, or find an obstacle-avoidance path from the stove to the door?"

**Key Challenge**: In real-world deployment, models are often used as "front-ends" for geometric solvers or code generators. However, these toolchains only function if the model itself possesses basic spatial intuition—if the model cannot maintain geometric invariants without tools, tool augmentation will only amplify errors. While works like LayoutGPT and FirePlace evaluate the realism of generated results, none systematically test the geometric consistency of models under symbolic input.

**Goal**: To construct a **purely symbolic, automatically gradable diagnostic benchmark covering Metric, Topology, and Dynamic categories** to quantitatively answer "what today's LLMs can do, where they fail, and why" regarding structured layouts.

**Key Insight**: Abstract representations used by professional architects (room polygons + object bounding boxes + openings) are fed directly to the model. The ground truth for all questions is calculated using deterministic geometric algorithms (avoiding reliance on another model for scoring) to eliminate evaluation circularity.

**Core Idea**: Utilizing a "Structured JSON Layout + Templated Geometric Problems + Rule-based Scoring" framework to transform spatial reasoning from an open-ended task into a diagnostic protocol with itemized scoring, proposing a three-tier capability gradient: Metric-Topology-Dynamic.

## Method

### Overall Architecture
FloorplanQA aims to determine exactly which geometric properties an LLM can calculate and where it fails, given a purely symbolic 2D indoor layout (JSON/XML containing only coordinates, dimensions, and semantic labels, without pixels). The benchmark consists of 2,000 layouts (1,800 self-generated using Gemini 2.5 Pro, 200 extracted from real HSSD scenes), with each layout paired with 8 geometric questions for a total of 16,000 questions. It employs a zero-shot evaluation protocol requiring a `Final answer` line, which is parsed via regular expressions and automatically scored using corresponding metrics. All layouts share a right-handed 2D coordinate system, with objects represented as polygons with semantic labels: synthetic data uses 4-vertex axis-aligned rectangles, while HSSD data uses arbitrary polygons.

Layout generation follows a two-stage pipeline: first, a prompt constrains the room geometry (shape, adjacency, corridor clearance, symmetrical partitioning), then furniture is filled based on stylized templates (e.g., a bedroom must contain a bed and storage). Approximately one-third of candidates are filtered by a rule-validator for unrealistic placements (e.g., sofas blocking doors). 3D scenes from HSSD are projected to 2D, and polygons are simplified using the Douglas-Peucker algorithm ($\epsilon=0.01$m).

### Key Designs

**1. Three-tier Task Taxonomy (Metric / Topology / Dynamic): Turning Leaderboards into Capability Profiles**

Traditional single-difficulty Q&A provides an average score that fails to identify where a model's logic breaks down. FloorplanQA categorizes 8 task types into three tiers based on reasoning intensity: Metric (pairwise distance, view angle) requiring basic coordinate arithmetic; Topology (free area, max contained rectangle, placement feasibility, visibility) requiring set operations and constraint satisfaction; and Dynamic (repositioning, shortest path) requiring reasoning during layout modification. These tiers map to typical failure modes: "Arithmetic / Constraint Reasoning / Multi-step Planning." Each question includes a fixed output format code: N (scalar, relative error $\leq 2\%$, relaxed to $5\%$ for free area), B (boolean), L (list/set equality), and S (sequence, requiring obstacle avoidance and Fréchet distance $\leq 0.6$m). Object names in templates are replaced with actual instances (e.g., `fridge_1`), ensuring unambiguous referencing.

**2. Deterministic Scoring to Remove Evaluation Circularity**

Using an LLM (such as Gemini) to both generate layouts and evaluate results introduces leakage via circular referencing. FloorplanQA severs this path: layouts are pre-generated and frozen, and all ground truths are calculated precisely using geometric libraries like `shapely` (polygon union area, A* paths, shoelace formula for centroids), independent of any LLM output. During evaluation, models are forced to output short structured reasoning followed by a `Final answer:` line. Responses are compared directly against the reference; missing answers or formatting errors receive a zero. The protocol records token truncation rates and reports both "Raw Accuracy" and "Finished Accuracy" to prevent reasoning models from inflating scores through excessive output length.

**3. Multi-dimensional Ablation Probes (Input Format / Semantic Perturbation / Tool Augmentation / VLM Rendering)**

To understand why models fail, four sets of ablations were designed: (a) JSON ↔ XML rewritings showed accuracy fluctuations within $\pm 3$ pp, suggesting models capture layout semantics rather than serialization syntax; (b) Object label swapping (e.g., swapping "sofa" and "chair" labels while keeping geometry constant) caused Repositioning accuracy to plummet from $60.5\%$ to $40.0\%$ (gpt-oss-120B), exposing a reliance on linguistic associations rather than geometric attributes; (c) GPT-4.1 with a Python interpreter improved arithmetic tasks by $+30$ to $+43$ pp but regressed in Max Box ($-4.5$) and Shortest Path ($-12.5$); (d) Evaluations with three types of rendering (bounding boxes, icons, AI-generated photos) alongside JSON remained within $\pm 4$ pp of the JSON-only baseline, while pure image input dropped to $19\%$-$40\%$. These probes isolate "Arithmetic Error / Algorithmic Error / Visual Inutility / Semantic Bias."

### Loss & Training
This work is an evaluation paper and does not train any models. All 15 LLMs were evaluated at `temperature=0`, zero-shot, under a unified prompt schema. Large models were allocated 12,288 tokens, while mid-to-small models were allocated 8,192 tokens. GPT-5 was allocated 4,096 tokens specifically for its reasoning process.

## Key Experimental Results

### Main Results
15 models were evaluated (7 reasoning models: GPT-5, gpt-oss-120B, DeepSeek-R1-0528, etc.; 8 general models: Claude Sonnet 4, GPT-4.1, Qwen3-Coder-480B, etc.). The overall results exhibit a three-tier gradient:

| Task Category | Representative Task | Average Accuracy (Synthetic) | Accuracy (HSSD) | Primary Failure Mode |
|---------------|---------------------|-----------------------------|-----------------|---------------------|
| Metric        | Pair Distance / View Angle | 75-95\% | 35-60\% | HSSD polygon centroid errors (shoelace formula failure) |
| Topology (Constraint) | Placement / Visibility / Repositioning | 60-85\% | 40-70\% | Multi-constraint joint verification failure |
| Topology (Opt.) + Dynamic | Free Space / Max Box / Shortest Path | 5-45\% | < 30\% | Confusing "sum of areas" with "union of areas", failed obstacle-avoidance search |

### Ablation Study

Comparison of Tool Augmentation (GPT-4.1 + Python interpreter) on HSSD:

| Task | Raw | + Tools | $\Delta$ | Interpretation |
|------|-----|---------|----------|----------------|
| Pair Distance | 56.0 | 99.0 | $+43.0$ | Arithmetic bottleneck resolved |
| View Angle | 55.0 | 96.0 | $+41.0$ | Same as above |
| Visibility | 46.5 | 86.5 | $+40.0$ | Success of explicit geometric checks |
| Repositioning | 47.0 | 83.5 | $+36.5$ | Tool handles collision detection |
| Placement | 64.5 | 95.0 | $+30.5$ | Benefits from precise determination |
| Free Space | 16.0 | 44.0 | $+28.0$ | Polygon union remains difficult |
| Max Box | 7.5 | 3.0 | $-4.5$ | Tool generates incorrect algorithm |
| Shortest Path | 25.0 | 12.5 | $-12.5$ | Boundary contact misjudged as collision |

### Key Findings
- Kitchen accuracy is the highest—each kitchen layout contains an average of only 0.52 overlapping object pairs, compared to 1.52-1.82 in bedrooms/living rooms and 4.39 in HSSD. High overlap significantly complicates union calculations and clearance determination.
- Reasoning models score $+10$ to $+40$ pp higher than general models on Free Space and Max Box, but gain very little in Shortest Path—suggesting that additional reasoning budget can solve geometric unions but not search-based planning.
- Repositioning scores drop significantly after object label swapping (e.g., gpt-oss-120B from 60.5% to 40.0%), proving models rely on linguistic common sense (e.g., "sofa = large furniture") rather than actual coordinates.
- Tool augmentation actually caused regressions in Max Box and Shortest Path. Analysis shows the LLM-generated Python code contained errors such as "treating boundary contact as collision" or "incomplete enumeration of rectangle orientations," indicating the bottleneck is spatial reasoning at the algorithmic level, not numerical precision.

## Highlights & Insights
- **Diagnostic Protocol vs. Leaderboard**: By grouping questions by reasoning type rather than difficulty, the benchmark identifies specific capability dimensions where a model falls short, rather than providing a single average score. This "capability profile" approach is transferable to evaluations of code, mathematics, and logic.
- **Counter-intuitive Degradation of Tool Augmentation**: While one might expect a Python interpreter to improve all scores, it led to regressions in planning tasks. This proves LLM failures are not merely arithmetic; given a calculator, the model can solve geometric problems but still fails to write a correct search algorithm. This observation serves as a warning for the "agent + tool" paradigm.
- **JSON vs. XML Irrelevance & Visual Inutility**: These two clean negative results indicate that LLMs understand space at a layout-semantic level; serialization syntax and visual rendering are merely surface-level packaging. Future research should focus on "geometric inductive bias" rather than "prompt formatting."

## Limitations & Future Work
- Layouts are limited to 2D indoor scenes with mostly axis-aligned furniture; they do not cover 3D height differences, multi-story buildings, outdoor scenes, or dynamic environments.
- The evaluation was conducted in a zero-shot setting only; few-shot examples might provide helpful geometric context.
- The 1,800 synthetic layouts were generated by Gemini 2.5 Pro; although scoring is independent, the layout distribution may carry the geometric biases of that model.
- Future Work: (Short-term) Integration with dedicated solvers like A*/shapely; (Mid-term) Incorporating geometric consistency objectives and violation samples during training; (Long-term) Multi-step interactive evaluation for agent self-verification and correction.

## Related Work & Insights
- **vs. CLEVR / SpatialSense**: These assess qualitative spatial relations in vision ("left/right/top/bottom"). FloorplanQA tests quantitative geometry under symbolic coordinates (distance, area, path), providing complementary and more focused difficulty.
- **vs. ScanQA / 3DSRBench**: These target VLM Q&A for 3D scenes, emphasizing point cloud/mesh understanding. This work focuses on pure LLM geometric reasoning under 2D symbolic input.
- **vs. LayoutGPT / Holodeck / FirePlace**: These works use LLMs to generate layouts and evaluate output realism. This paper acts as a probe—given a layout, it tests if the model can infer its properties, effectively the inverse direction.
- **vs. Yamada et al. 2023 (Evaluating Spatial Understanding)**: While they use natural language to evaluate qualitative relations, this paper uses structured JSON and coordinates to evaluate precise geometry, refining "spatial understanding" into quantifiable Metric, Topology, and Dynamic tiers.

## Rating
- Novelty: ⭐⭐⭐⭐ The task taxonomy and "diagnostic protocol" are clear, though the structure remains typical for a benchmark paper.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 15 models × 8 task types × 4 layout categories + 4 sets of ablations; comprehensive enough to cover comparative needs for the foreseeable future.
- Writing Quality: ⭐⭐⭐⭐ The three-tier classification is well-explained, and the insights regarding tool-driven regression and overlap correlations are particularly valuable.
- Value: ⭐⭐⭐⭐⭐ Provides the first diagnostic benchmark for "LLM + spatial reasoning" with precise sub-item scoring, offering direct value to the fields of Agents, Architectural Design, and Embodied AI.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] CoCoReviewBench: A Completeness- and Correctness-Oriented Benchmark for AI Reviewers](cocoreviewbench_a_completeness-_and_correctness-oriented_benchmark_for_ai_review.md)
- [\[ICML 2026\] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)
- [\[ICML 2025\] Adversarial Manipulation of Reasoning Models using Internal Representations](../../ICML2025/llm_reasoning/adversarial_manipulation_of_reasoning_models_using_internal_representations.md)
- [\[ICML 2026\] Evaluating Relational Reasoning in LLMs with REL](evaluating_relational_reasoning_in_llms_with_rel.md)
- [\[NeurIPS 2025\] TimE: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios](../../NeurIPS2025/llm_reasoning/time_a_multilevel_benchmark_for_temporal_reasoning_of_llms_i.md)

</div>

<!-- RELATED:END -->
