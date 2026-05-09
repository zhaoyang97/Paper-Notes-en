---
title: >-
  [Paper Note] ShowTable: Unlocking Creative Table Visualization with Collaborative Reflection and Refinement
description: >-
  [CVPR 2026][Image Generation][Table Visualization] ShowTable introduces a novel task termed *creative table visualization* (generating infographics from structured data tables) and proposes a progressive self-correction pipeline in which an MLLM (for reasoning and reflection) and a diffusion model (for generation and refinement) collaborate iteratively. Through a dedicated fine-tuned rewriting module and an RL-optimized refinement module, the framework consistently and substantially improves visualization quality over all baseline models on the newly constructed TableVisBench benchmark.
tags:
  - CVPR 2026
  - Image Generation
  - Table Visualization
  - Self-Correction
  - MLLM Reasoning
  - Diffusion Models
  - Reinforcement Learning
date: 2026-05-08
content_hash: be240aca546997f2
---

# ShowTable: Unlocking Creative Table Visualization with Collaborative Reflection and Refinement

**Conference**: CVPR 2026
**arXiv**: [2512.13303](https://arxiv.org/abs/2512.13303)
**Code**: [https://lntzm.github.io/showtable-page/](https://lntzm.github.io/showtable-page/)
**Area**: Diffusion Models / Image Generation
**Keywords**: Table Visualization, Self-Correction, MLLM Reasoning, Diffusion Models, Reinforcement Learning

## TL;DR
ShowTable introduces a novel task termed *creative table visualization* (generating infographics from structured data tables) and proposes a progressive self-correction pipeline in which an MLLM (for reasoning and reflection) and a diffusion model (for generation and refinement) collaborate iteratively. Through a dedicated fine-tuned rewriting module and an RL-optimized refinement module, the framework consistently and substantially improves visualization quality over all baseline models on the newly constructed TableVisBench benchmark.

## Background & Motivation

1. **Background**: Image generation models have achieved high quality in general-purpose scenarios, and recent research has progressively shifted toward more complex structured generation tasks such as poster design and text rendering. Nevertheless, data-driven visualization—e.g., generating charts or infographics from tabular data—remains a significant challenge for existing models.

2. **Limitations of Prior Work**: When a markdown table is directly used as a prompt, generation models tend to *render the table as text* rather than *visualize the underlying data*. Existing unified models achieve near-zero Data Accuracy scores (e.g., Bagel: 0.1; Blip3o-Next: 0.4), failing to correctly map data values to visual elements such as bar heights or pie-chart angles.

3. **Key Challenge**: Creative table visualization demands two seemingly conflicting capabilities—creative aesthetic design (requiring freedom) and strict data-fidelity mapping (requiring precision). Generative models excel at the former but frequently fail at the latter.

4. **Goal**: To enable generative models to accurately and aesthetically visualize structured tabular data as infographics, while automatically detecting and correcting generation errors.

5. **Key Insight**: MLLMs are leveraged for reasoning and planning (rewriting) and for error auditing (reflection), while diffusion models handle execution (generation and refinement), forming an iterative self-correction loop. Dedicated modules are trained separately to address the two identified bottlenecks.

6. **Core Idea**: A collaborative paradigm of *MLLM coordination + diffusion model execution* realizes high-fidelity infographic generation from tables through a Rewriting → Generation → Reflection → Refinement self-correction cycle.

## Method

### Overall Architecture
ShowTable is a four-stage pipeline: (1) **Rewriting**—an MLLM transforms the data-dense markdown table into a detailed descriptive prompt encompassing data points, layout, color, and background planning; (2) **Generation**—a diffusion model produces an initial image from the rewritten prompt; (3) **Reflection**—an MLLM audits the generated image against the original table, identifying data-mapping errors, text-rendering errors, proportion errors, etc., and produces precise editing instructions; (4) **Refinement**—an image-editing model corrects identified errors according to the instructions. The Reflection → Refinement cycle iterates for up to three rounds.

### Key Designs

1. **Rewriting Module**:

    - **Function**: Converts tabular data into detailed descriptive prompts that generation models can execute.
    - **Mechanism**: A dedicated rewriting model is fine-tuned from Qwen3-8B. Training data construction: (a) Gemini-2.5-Pro generates detailed descriptions of collected ground-truth table visualization images; (b) chain-of-thought rationales explaining the conversion logic are subsequently generated. This yields 30K SFT samples of the form {table, rationale} → {description}, trained with standard next-token prediction.
    - **Design Motivation**: General-purpose LLMs (GPT-5, Gemini) still omit data points or plan layouts poorly when confronted with complex multi-level tables. The fine-tuned rewriting module surpasses even the Reference-Caption upper bound on Data Accuracy (51.2 vs. 50.3).

2. **Reflection Module**:

    - **Function**: Audits the data accuracy of generated images and produces editing instructions.
    - **Mechanism**: GPT-5 serves as the reflection model (best performance), conducting a dimension-by-dimension audit of generated images against the original table—verifying data-point correctness, text clarity, proportional accuracy, and the reasonableness of supplementary information—and outputting precise, actionable editing instructions (e.g., "reduce the height of the third bar by 20%").
    - **Design Motivation**: Although MLLMs cannot directly generate perfect visualizations, their comprehension and auditing capabilities compensate for the shortcomings of generative models. Decoupling generation from auditing allows each component to operate within its domain of strength.

3. **Refinement Module (via RL Training)**:

    - **Function**: Precisely corrects errors in generated images according to editing instructions.
    - **Mechanism**: Experiments reveal that the base editing model (Qwen-Image-Edit) degrades progressively across refinement rounds, whereas Wan2.5-I2I-Preview improves consistently, confirming that the pipeline logic is sound but the refinement model is the bottleneck. Accordingly, RL training via the GRPO algorithm is applied to the refinement module. A reward model (RM) is first trained—based on Qwen2.5-VL-3B fine-tuned with Bradley-Terry loss on 30K preference pairs to output scalar quality scores. The RM combined with ImageReward then serves as a composite reward for GRPO training (applied to a distilled version of Qwen-Image-Edit-2509) on 5K quality-filtered refinement samples.
    - **Design Motivation**: Iterative refinement with off-the-shelf editing models leads to error accumulation; dedicated training is necessary to adapt to the iterative error-correction setting. RL training is better suited than SFT for optimizing tasks that require balancing multiple quality dimensions.

### Data Construction

Three automated data pipelines are employed: (1) *Rewriting data*: 30K table → description pairs (filtered by dual-reviewer consistency); (2) *Reward data*: 30K preference pairs (voted by GPT-5 and Gemini); (3) *Refinement data*: 5K samples—five refinement candidates are generated per sample, and extreme cases (all good or all bad) are discarded, retaining only discriminative samples. Data are sourced from SlideVQA, OpenImages, and Cambrian-10M.

## Key Experimental Results

### Main Results (TableVisBench; higher scores are better)

| Baseline | Original Score | +RW Score | +RW+REF Score | Gain |
|---------|----------|-----------|---------------|------|
| Flux | 29.3 | 32.1 | 36.4 | +7.1 |
| Bagel | 10.1 | 19.5 | 32.7 | **+22.6** |
| Blip3o-Next | 10.8 | 14.1 | 34.8 | **+24.0** |
| UniWorld-V1 | 14.8 | 18.6 | 33.5 | +18.7 |
| OmniGen2 | 14.4 | 21.9 | 29.9 | +15.5 |
| Qwen-Image | 44.3 | 54.3 | 54.9 | +10.6 |

### Ablation Study

**Rewriting Module**:

| Configuration | DA | RR | Score |
|------|----|----|-------|
| No Rewriting | 47.5 | 26.1 | 44.3 |
| Qwen3-8B | 30.6 | 46.6 | 46.8 |
| GPT-5 | 35.9 | 47.8 | 51.2 |
| Gemini-2.5-Pro | 40.8 | 53.9 | 53.3 |
| Qwen3-8B* (fine-tuned) | **51.2** | 50.1 | **54.3** |

**Refinement Module (multi-round performance)**:

| Refinement Model | Round 0 | Round 1 | Round 2 | Round 3 |
|---------|---------|---------|---------|---------|
| Qwen-Image-Edit (base) | 54.3 | 51.8 | 50.1 | 49.4 ↓ |
| Qwen-Image-Edit* (ours) | 54.3 | 53.7 | 54.8 | **54.9** ↑ |
| Wan2.5-I2I-Preview | 54.3 | 61.3 | 62.8 | **63.4** ↑ |

### Key Findings
- Weaker baseline models benefit the most—Bagel improves from 10.1 to 32.7 (+22.6); Blip3o-Next from 10.8 to 34.8 (+24.0).
- The rewriting module contributes most substantially to the Relative Relationship (RR) dimension, with Qwen-Image jumping from 26.1 to 50.1.
- The progressive degradation of the base refinement model (54.3 → 49.4) confirms that refinement capability is the bottleneck; RL training reverses this trend to yield consistent improvement (54.3 → 54.9).
- The fine-tuned rewriting module's Data Accuracy (51.2) even surpasses the Reference-Caption upper bound (50.3), indicating that task-specific planning is more suitable for generative models than human-written descriptions.
- Using Wan2.5 as the refinement model achieves 63.4; open-source models trained with RL also show notable gains (+5.5).

## Highlights & Insights
- **Discovery and Resolution of the Refinement Bottleneck**: Controlled experiments swapping the refinement model demonstrate that the pipeline logic is correct while the model capacity is insufficient; RL training is then applied in a targeted manner, yielding a methodologically clear problem-solution chain.
- **Reusable Reward Model Construction**: Rather than relying on unstable direct MLLM scoring, a compact RM trained on preference pairs serves as an intermediate bridge—a pattern applicable to any RL scenario requiring MLLM-based evaluation.
- **Introduction of a Practical and Challenging New Task**: Creative table visualization has clear downstream applications (automated slide/report generation), and both the benchmark and training pipelines are made available to the community.

## Limitations & Future Work
- The Reflection module relies on GPT-5, incurring high costs and limiting open-source reproducibility.
- The iterative refinement is capped at three rounds, which may be insufficient for highly complex tables.
- Aesthetic Quality (AQ) scores exhibit little variation across methods (4.3–4.6), suggesting that the current aesthetic evaluation granularity may be inadequate.
- Only static infographic generation is supported; interactive charts and animations are not addressed.
- Data filtering relies on consensus between GPT-5 and Gemini, potentially introducing systematic bias.

## Related Work & Insights
- **vs. AnyText / Glyph-ByT5**: These works focus on text-rendering accuracy; ShowTable addresses a more complex task requiring not only accurate text rendering but also correct mapping of data proportions to visual elements.
- **vs. AutoPoster / PosterMaker**: Poster generation emphasizes aesthetic layout; ShowTable additionally enforces data fidelity.
- **vs. RPG / SynTalker and other reflection-refinement works**: Existing reflection loops are primarily applied to general-purpose instruction following. ShowTable is the first to apply this paradigm to structured data visualization with high information density.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The new task definition is valuable; the self-correction framework combining MLLMs with diffusion models is insightful; RL training for refinement is creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six baseline models × three configurations, detailed ablations, a five-dimension evaluation framework, and extensive case analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Figures and tables are rich and intuitive; the pipeline description is clear; the problem-discovery-to-solution logical chain is complete.
- **Value**: ⭐⭐⭐⭐ The task has clear application scenarios (automated slide/report generation); the benchmark and training pipeline are made available for community use.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] PSDesigner: Automated Graphic Design with a Human-Like Creative Workflow](psdesigner_automated_graphic_design_with_a_human-like_creative_workflow.md)
- [\[CVPR 2026\] ViStoryBench: Comprehensive Benchmark Suite for Story Visualization](vistorybench_comprehensive_benchmark_suite_for_story_visualization.md)
- [\[CVPR 2026\] RAISE: Requirement-Adaptive Evolutionary Refinement for Training-Free Text-to-Image Alignment](raise_requirement-adaptive_evolutionary_refinement_for_training-free_text-to-ima.md)
- [\[ACL 2026\] VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval](../../ACL2026/image_generation/visret_visualization_improves_knowledge-intensive_text-to-image_retrieval.md)
- [\[ICLR 2026\] From Prediction to Perfection: Introducing Refinement to Autoregressive Image Generation](../../ICLR2026/image_generation/from_prediction_to_perfection_introducing_refinement_to_autoregressive_image_gen.md)

<!-- RELATED:END -->
