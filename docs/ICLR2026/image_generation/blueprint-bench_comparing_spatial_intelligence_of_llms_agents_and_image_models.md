---
title: >-
  [Paper Note] Blueprint-Bench: Comparing Spatial Intelligence of LLMs, Agents and Image Models
description: >-
  [ICLR 2026][Image Generation][Spatial Intelligence] Blueprint-Bench evaluates AI spatial reasoning through the task of "generating 2D floorplans from apartment interior photographs": the inputs (photos) are fully within…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Spatial Intelligence"
  - "Floorplan Generation"
  - "Benchmark"
  - "LLM Evaluation"
  - "Image Generation Model Evaluation"
  - "AI Safety"
date: 2026-05-08
content_hash: 664526763e5fbbad
---

# Blueprint-Bench: Comparing Spatial Intelligence of LLMs, Agents and Image Models

**Conference**: ICLR 2026
**arXiv**: [2509.25229](https://arxiv.org/abs/2509.25229)  
**Code**: [GitHub](https://github.com/AndonLabs/Blueprint-Bench-generation) (includes generation code and dataset samples)  
**Area**: Image Generation
**Keywords**: Spatial Intelligence, Floorplan Generation, Benchmark, LLM Evaluation, Image Generation Model Evaluation, AI Safety

## TL;DR

Blueprint-Bench evaluates AI spatial reasoning through the task of "generating 2D floorplans from apartment interior photographs": the inputs (photos) are fully within the training distribution, while the task (spatial reconstruction) is out-of-distribution. The benchmark evaluates LLMs including GPT-5, Claude 4 Opus, Gemini 2.5 Pro, and Grok-4; image generation models including GPT-Image and NanoBanana; and agent systems including Codex CLI and Claude Code. Results show that the vast majority of models perform at or below a random baseline, revealing a systematic blind spot in current AI spatial intelligence.

## Background & Motivation

**Background**: LLMs continue to demonstrate emergent capabilities beyond their training scope, and next-generation image generation models (GPT-Image, NanoBanana/Gemini 2.5 Flash Image) are beginning to exhibit reasoning abilities (e.g., solving geometry problems). Nevertheless, the "intelligence" of image generation models lacks quantitative evaluation—GPT-Image was released without a single quantitative chart.

**Limitations of Prior Work**: (1) LLM benchmarks focus on text, code, and mathematics, with no systematic benchmark for spatial reasoning; (2) the ARC benchmark features inputs (grid patterns) and tasks that are both out-of-distribution for LLMs, making it impossible to distinguish between "failing to understand the input" and "failing to perform the task"; (3) no evaluation framework exists for cross-architecture comparison of intelligence across LLMs, image models, and agents.

**Key Challenge**: Apartment photographs as inputs are fully within the training distribution of modern multimodal models, yet inferring a floorplan from photographs requires genuine spatial reasoning—deducing room layout, understanding connectivity, and maintaining consistent scale—a task models have not been trained to perform. This "in-distribution input + out-of-distribution task" design enables precise localization of spatial reasoning deficiencies.

**Goal**: To introduce the first quantitative benchmark capable of cross-architecture comparison of spatial intelligence (LLMs / image generation models / agents), while providing the first quantitative intelligence evaluation tool for image generation models.

**Key Insight**: The benchmark is designed to be model-agnostic: any system capable of generating an image from a sequence of images may participate (LLMs generate SVG which is then rasterized; image models generate directly; agents iteratively write and execute code within a Docker environment).

**Core Idea**: By means of a benchmark where inputs are in-distribution but the task is out-of-distribution, the work quantitatively reveals—for the first time—systematic deficiencies in AI spatial reasoning: most state-of-the-art models perform at or below a random baseline.

## Method

### Overall Architecture

The dataset comprises 50 apartments, each with approximately 20 interior photographs paired with a standardized ground-truth floorplan. Models receive the photographs along with 9 strict formatting specifications (black walls at 3 px / green doors / red dots marking room centers / pure white background, etc.) and must output a specification-compliant floorplan image. Evaluation proceeds in two stages: (1) automatic extraction of a room connectivity graph and area ranking from the standardized image; (2) computation of a weighted similarity score against the ground truth. Three participant categories are included: LLMs (GPT-5 / Claude 4 Opus / Gemini 2.5 Pro / Grok-4 / GPT-4o / GPT-5-mini), image generation models (GPT-Image / NanoBanana), and agents (Codex CLI / Claude Code), together with human and random baselines.

### Key Design 1: Standardized Dataset and Formatting Specifications

- **Function**: Ensures the scoring algorithm can robustly extract spatial structure from any participant's output.
- **Mechanism**: Nine strict rules are enforced—black walls (3 px wide), green doors (overlaid on black lines), red circular dots (10×10 px marking room centers), pure white background, fully enclosed rooms, and prohibition of furniture, windows, and other details.
- **Design Motivation**: Expressiveness is sacrificed in favor of scoring reliability. At the current level of model capability—where most models perform near chance—reliable scoring is more important than rich expressiveness.

### Key Design 2: Two-Stage Automatic Evaluation Algorithm

- **Function**: Quantifies the similarity between two floorplans as a score in $[0, 1]$.
- **Mechanism**: **Extraction stage**—HSV color filtering detects red dot centers (room positions); a binarization mask excludes walls and doors; flood-fill from each red center segments room boundaries; wall scans detect green doors and their orientation (horizontal/vertical); rooms are assigned IDs by area ranking. **Scoring stage**—a weighted average of 6 similarity components is computed: edge-overlap Jaccard (50%), degree correlation (20%), graph density matching (10%), room count accuracy (10%), door count accuracy (5%), and door orientation distribution (5%).
- **Design Motivation**: Graph-based rather than pixel-based matching avoids spurious penalties from minor spatial offsets. LLM-based extraction was attempted but abandoned, as LLMs proved highly unreliable at interpreting floorplans, frequently misidentifying room connectivity and area rankings.

### Key Design 3: Fair Cross-Architecture Comparison

- **Function**: Enables the first fair comparison of spatial intelligence among LLMs, image generation models, and agents on a unified task.
- **Mechanism**: LLMs generate SVG code which is then rasterized; image models receive photographs and generate floorplans directly; agents operate in a Docker Linux environment with the freedom to view images, write and execute code, and iterate.
- **Design Motivation**: The agent setting (with iterative viewing and modification) mirrors the human workflow, testing whether "iteration can compensate for single-pass reasoning deficiencies."

### Baseline Design

- **Random Baseline**: Models are prompted to generate a typical floorplan without any image input, serving as a lower bound.
- **Human Baseline**: Human participants draw floorplans under the same conditions (photographs only, no site visit).

## Key Experimental Results

### Main Results: Average Similarity Score per Model (50 Apartments)

| Model Type | Model Name | Relative Performance | Key Characteristics |
|-----------|-----------|---------------------|-------------------|
| Human | Human | **Significantly above all AI models** | All floorplans exhibit correct room connectivity |
| LLM | GPT-5 | Statistically significant > random baseline | Best among LLMs |
| LLM | Gemini 2.5 Pro | Statistically significant > random baseline | Close to GPT-5 |
| LLM | GPT-5-mini | Statistically significant > random baseline | Smaller model still effective |
| LLM | Grok-4 | Statistically significant > random baseline | Only marginally above baseline |
| LLM | Claude 4 Opus | ≈ random baseline | No significant improvement |
| LLM | GPT-4o | **Well below** random baseline | Severe instruction-following failure |
| Image Generation | GPT-Image | ≈ random baseline | Good instruction compliance, poor spatial reasoning |
| Image Generation | NanoBanana | **Well below** random baseline | Consistently includes furniture and other details; extremely poor instruction compliance |
| Agent | Codex CLI (GPT-5) | ≈ random baseline | Does not exploit iterative capability |
| Agent | Claude Code (Claude 4 Opus) | ≈ random baseline | Exhibits iterative behavior but with negligible effect |

*Note: Scores are presented as figures in the paper; exact numerical values are not provided. All models score substantially below the human baseline.*

### Scoring Weight Breakdown

| Similarity Component | Weight | What It Measures |
|--------------------|--------|-----------------|
| Edge-overlap Jaccard | 50% | Correctness of room connectivity |
| Degree correlation | 20% | Match of door-count distribution per room |
| Graph density matching | 10% | Ratio of actual to possible connections |
| Room count accuracy | 10% | Correctness of total room count |
| Door count accuracy | 5% | Correctness of total door count |
| Door orientation distribution | 5% | Match of horizontal/vertical door ratio |

### Key Findings

- **Spatial intelligence is a significant blind spot in current AI**: Only 4 LLMs (GPT-5, Gemini 2.5 Pro, GPT-5-mini, Grok-4) statistically significantly exceed the random baseline, and only by a small margin—most top-performing models perform at or below chance.
- **Humans lead by a wide margin**: All human-drawn floorplans exhibit correct room connectivity (a frequent failure point for AI), and even when area rankings are occasionally incorrect, human scores far exceed those of AI. The authors argue that under a more lenient scoring scheme, the human advantage would be even larger.
- **Image generation models struggle particularly**: NanoBanana consistently violates formatting rules (including furniture, windows, and decorative details); GPT-Image exhibits better instruction compliance but equally poor spatial reasoning.
- **Agent iteration is surprisingly ineffective**: Codex CLI (GPT-5) does not leverage iterative capability at all—it views all images, writes a script in a single pass, and submits without reviewing the output. Claude Code exhibits iterative behavior but with negligible effect, ultimately asserting that "all rooms are properly enclosed"—which is factually incorrect.
- **Anomalous behavior of GPT-4o**: As the weaker LLM tested, it fails instruction following (omitting red dot room markers), causing its score to fall well below the random baseline.
- **GPT-Image vs. its underlying LLM**: GPT-Image does not demonstrate stronger spatial intelligence than GPT-5 (approximately at the random baseline vs. marginally above it), suggesting that the image generation training stage may not enhance spatial reasoning capability.

## Highlights & Insights

- **"In-distribution input + OOD task" evaluation paradigm**: Unlike ARC (where both input and task are OOD), Blueprint-Bench uses everyday photographs—data highly familiar to modern multimodal models—as inputs, thereby precisely isolating "spatial reasoning" as the specific capability under scrutiny. Models are not failing to understand the images; they are failing to infer spatial structure from them.
- **Pioneering cross-architecture comparison**: Blueprint-Bench is the first benchmark to quantitatively compare LLMs, image generation models, and agents on a unified task, filling the gap in quantitative intelligence evaluation for image generation models.
- **Revelation of agent iteration failure**: Claude Code's iterative process demonstrates that current agents, despite possessing a degree of self-review capability, remain unable to effectively self-correct—claiming "all rooms properly enclosed" when the actual output is incorrect.
- **AI safety perspective**: While spatial intelligence is not inherently dangerous, it is a prerequisite for hazardous applications (e.g., military robotics, autonomous navigation). Blueprint-Bench has potential value as a monitoring tool for tracking the emergence of spatial intelligence, offering an early-warning function for AI safety.

## Limitations & Future Work

- **Score depends on area-ranking-based ID assignment**: Rooms are not labeled by type (bedroom, kitchen, etc.); area-ranking errors cascade into connectivity scoring errors, imposing unfair false-positive penalties on humans and some AI models.
- **Room geometry is not considered**: Only the connectivity graph and area ranking are compared; the geometric shape of rooms is entirely ignored. Bidirectional nearest-neighbor wall-sampling distances were explored as a shape metric but were found to penalize minor errors too harshly and unpredictably.
- **Dataset limited to 50 apartments**: The scale may be insufficient to fully support statistical significance analysis.
- **Formatting rules constrain expressiveness**: The 9 strict rules unfairly penalize models with poor instruction-following ability—Blueprint-Bench should measure spatial intelligence, not instruction compliance.
- **Specialized spatial AI systems not evaluated**: NeRF-based indoor reconstruction methods, for example, are outside the scope of evaluation—though this is intentional, as the benchmark targets general-purpose model spatial intelligence.
- Results are presented as figures in the paper without precise numerical values, limiting the reproducibility of quantitative comparisons.

## Related Work & Insights

- **vs. ARC**: ARC features both OOD inputs (grid patterns) and OOD tasks (transformation rule inference); Blueprint-Bench makes only the task OOD, enabling more precise localization of spatial reasoning deficits rather than general OOD reasoning ability.
- **vs. specialized architectural AI systems (LayoutGPT, PosterLLaVA)**: These works pursue optimal floorplan generation systems; Blueprint-Bench does not seek state-of-the-art performance but instead measures the spatial intelligence of general-purpose models—a fundamentally different evaluation perspective.
- **vs. image generation benchmarks (FID / IS / GenEval)**: Existing benchmarks focus on aesthetics and semantic consistency; Blueprint-Bench focuses on spatial reasoning accuracy, filling the gap in intelligence evaluation for image generation models.
- **Implications**: As image generation models become increasingly "intelligent" (e.g., solving mathematics), benchmarks must measure reasoning ability rather than generation quality—Blueprint-Bench opens this direction.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First cross-architecture spatial intelligence benchmark; the "in-distribution input + OOD task" evaluation paradigm is elegantly designed and fills the gap in intelligence evaluation for image generation models.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers three architectural categories (LLMs / image models / agents) plus human and random baselines; however, the dataset contains only 50 apartments and results are not reported as precise numerical values.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated, evaluation methodology is described in detail, and the analysis of agent behavior is illuminating.
- Value: ⭐⭐⭐⭐ — Reveals an important blind spot in spatial intelligence, offers reference value for AI safety assessment, and supports ongoing tracking of new model performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Everything in Its Place: Benchmarking Spatial Intelligence of Text-to-Image Models](everything_in_its_place_benchmarking_spatial_intelligence_of_text-to-image_model.md)
- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](../../CVPR2026/image_generation/micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)
- [\[ICLR 2026\] MVAR: Visual Autoregressive Modeling with Scale and Spatial Markovian Conditioning](mvar_visual_autoregressive_modeling_with_scale_and_spatial_markovian_conditionin.md)
- [\[NeurIPS 2025\] Is Artificial Intelligence Generated Image Detection a Solved Problem?](../../NeurIPS2025/image_generation/is_artificial_intelligence_generated_image_detection_a_solved_problem.md)
- [\[ICLR 2026\] SSG: Scaled Spatial Guidance for Multi-Scale Visual Autoregressive Generation](ssg_scaled_spatial_guidance_for_multi-scale_visual_autoregressive_generation.md)

</div>

<!-- RELATED:END -->
