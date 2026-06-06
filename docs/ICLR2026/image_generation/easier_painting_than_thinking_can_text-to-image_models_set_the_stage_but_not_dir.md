---
title: >-
  [Paper Note] Easier Painting Than Thinking: Can Text-to-Image Models Set the Stage, but Not Direct the Play?
description: >-
  [ICLR 2026][Image Generation][T2I evaluation] This paper proposes T2I-CoReBench, the first comprehensive benchmark that systematically evaluates both **compositional ability** (Composition) and **reasoning ability** (Rea…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "T2I evaluation"
  - "compositional generation"
  - "reasoning ability"
  - "benchmark"
  - "scene graph"
date: 2026-05-08
content_hash: 79e8be52a755d2b3
---

# Easier Painting Than Thinking: Can Text-to-Image Models Set the Stage, but Not Direct the Play?

**Conference**: ICLR 2026
**arXiv**: [2509.03516](https://arxiv.org/abs/2509.03516)  
**Code**: [GitHub](https://github.com/) (available, with Leaderboard and Benchmark)  
**Area**: Text-to-Image Generation / Evaluation Benchmark
**Keywords**: T2I evaluation, compositional generation, reasoning ability, benchmark, scene graph

## TL;DR

This paper proposes T2I-CoReBench, the first comprehensive benchmark that systematically evaluates both **compositional ability** (Composition) and **reasoning ability** (Reasoning) of T2I models. It covers 12 evaluation dimensions, 1,080 high-difficulty prompts, and approximately 13,500 checklist questions. Large-scale evaluation of 38 models reveals that reasoning capability lags far behind compositional capability, constituting the primary bottleneck in current T2I generation.

## Background & Motivation

In T2I generation, text prompts simultaneously contain **explicit descriptions** (content requiring compositional generation) and **implicit cues** (content requiring reasoning to generate correctly). For instance, "a ripe tomato squeezed tightly in a fist" implicitly entails "tomato juice bursting out." This corresponds to two core capabilities: Composition and Reasoning.

**Two major deficiencies of existing benchmarks**:

**Lack of comprehensiveness**: Most benchmarks evaluate either composition or reasoning, with heuristic taxonomies that fail to systematically cover all dimensions.

**Lack of complexity**: Compositional scenarios test only a small number of visual elements (≤5), and reasoning tasks test only simple one-to-one causality, failing to reflect high-density real-world scenes.

**Key Insight**: The paper draws on scene graph structures to formalize compositional ability, and on the philosophical tripartite classification of reasoning (deductive/inductive/abductive) to formalize reasoning ability, constructing a 12-dimensional evaluation framework that is both comprehensive and complex.

## Method

### Overall Architecture

T2I-CoReBench consists of three components: (1) a 12-dimensional evaluation taxonomy; (2) a high-complexity prompt and checklist construction pipeline; and (3) an MLLM-based automatic evaluation protocol.

### Key Designs

1. **Compositional Ability Evaluation (4 dimensions)**: Based on the three elements of scene graphs:

    - **MI (Multi-Instance)**: Generating approximately 25 instances in a single image, e.g., "a busy modern kitchen"
    - **MA (Multi-Attribute)**: Binding approximately 20 attributes to a single subject
    - **MR (Multi-Relation)**: Approximately 15 relational connections within a scene
    - **TR (Text Rendering)**: Precise rendering of content and layout for approximately 15 text items

   Compared to existing benchmarks such as DPG-Bench, visual element density is increased by 4–5× (~20 vs. ~5).

2. **Reasoning Ability Evaluation (8 dimensions)**: Based on the philosophical tripartite classification of reasoning:

    - **Deductive Reasoning** (premise → conclusion): LR (Logical Reasoning), BR (Behavioral Reasoning), HR (Hypothetical Reasoning), PR (Process Reasoning)
    - **Inductive Reasoning** (pattern → rule): GR (Generalization Reasoning), AR (Analogical Reasoning)
    - **Abductive Reasoning** (observation → explanation): CR (Commonsense Reasoning), RR (Reconstruction Reasoning)

   The framework introduces complex reasoning chains of one-to-many (one action → multiple consequences) and many-to-one (multiple premises → one conclusion), surpassing the one-to-one reasoning limitation of existing benchmarks.

3. **Prompt and Checklist Construction Pipeline**:

    - Three SOTA large reasoning models (Claude Sonnet 4, Gemini 2.5 Pro, OpenAI o3) are used to assist in generating prompts, ensuring diversity and complexity.
    - Each prompt is paired with a checklist of independent yes/no questions that verify explicit and implicit visual elements point by point.
    - All samples undergo rigorous human review.

### Loss & Training

This paper presents an evaluation benchmark and involves no training process. The evaluation protocol employs Gemini 2.5 Flash as the MLLM evaluator, converting each checklist question into a binary VQA task ("0" = No / "1" = Yes), leveraging the atomic design of checklists to ensure compatibility with MLLM-based evaluation.

## Key Experimental Results

### Main Results (Comprehensive Evaluation of 38 Models)

| Model | Parameters | Composition Avg. | Reasoning Avg. | Overall Avg. |
|------|--------|---------|---------|--------|
| FLUX.2-dev | 32B | **84.7** | **54.2** | **64.4** |
| Qwen-Image-2512 | 20B | 83.7 | 51.7 | 62.4 |
| FLUX.2-klein-9B | 9B | 78.0 | 52.0 | 60.6 |
| LongCat-Image | 6B | 70.8 | 54.1 | 59.6 |
| HunyuanImage-3.0 | 80B | 78.9 | 48.6 | 58.7 |
| BAGEL w/ Think | 14B | 39.6 | 41.9 | 41.1 |
| OmniGen2-7B | 7B | 49.9 | 39.4 | 42.9 |
| PixArt-α | 0.6B | 25.0 | 23.7 | 24.1 |
| Janus-Pro-1B | 1B | 35.5 | 13.0 | 20.5 |

| Per-Dimension Breakdown (FLUX.2-dev) | MI | MA | MR | TR | LR | BR | HR | PR | GR | AR | CR | RR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Score | 89.5 | 83.4 | 72.7 | 93.3 | 48.4 | 33.5 | 53.4 | 83.1 | 62.3 | 64.3 | 62.1 | 26.9 |

### Ablation Study

- Evaluation results from different MLLM evaluators (Qwen2.5-VL-72B, Qwen3-VL series) are highly consistent with those from Gemini 2.5 Flash.
- Using three different LRMs for prompt generation avoids single-model bias.

### Key Findings

1. **Compositional ability is improving steadily**: Open-source models are gradually closing the gap with closed-source counterparts; FLUX.2-dev achieves a compositional average of 84.7.
2. **Reasoning ability severely lags behind**: Even the strongest model shows a reasoning average far below its compositional score; FLUX.2-dev scores only 54.2 in reasoning vs. 84.7 in composition.
3. **Chain-of-thought in unified models is helpful but limited**: BAGEL w/ Think improves reasoning from 34.1 to 41.9, but still falls substantially short of diffusion models.
4. **Text Rendering shows large variance**: FLUX.2-dev reaches 93.3, while most unified/autoregressive models score below 10.
5. **Model scale is not a decisive factor**: LongCat-Image at 6B achieves a reasoning average of 54.1, surpassing HunyuanImage-3.0 at 80B (48.6).

## Highlights & Insights

- This is the first work to systematically introduce the philosophical tripartite reasoning taxonomy (deductive/inductive/abductive) into T2I evaluation, providing a theoretically grounded classification framework.
- The high-density compositional design (~20 elements/prompt) and high-intensity reasoning design (one-to-many/many-to-one) better reflect real-world application scenarios.
- The work surfaces an important finding: **T2I models are better at "painting" (composition) than "thinking" (reasoning)**—precisely as the title suggests.
- The checklist design decomposes complex evaluation into atomic questions, improving assessment reliability.

## Limitations & Future Work

- The MLLM evaluator may introduce its own biases, despite human validation efforts.
- The difficulty definition for reasoning dimensions (one-to-many/many-to-one) is relatively coarse-grained; further refinement of reasoning chain length is warranted.
- Only static image generation is evaluated; video or interactive generation is not addressed.
- The benchmark's 1,080 prompts are moderate in scale, with only 90 prompts per dimension, potentially limiting statistical power.

## Related Work & Insights

- **DPG-Bench / ConceptMix**: Pioneers in compositional evaluation, but insufficiently complex.
- **R2I-Bench / WISE**: Early explorations of reasoning evaluation, but covering only a subset of reasoning types.
- **GenAI-Bench**: General-purpose evaluation but lacking a systematic taxonomy.
- Insight: The reasoning dimensions of this benchmark can inform the training of T2I models to improve reasoning capability.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic integration of composition and reasoning evaluation with a theoretically grounded taxonomy, though the benchmark construction methodology itself is not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Broad coverage of 38 models, cross-validation with multiple evaluators, and in-depth analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, polished figures, and thorough explanation of the taxonomy.
- Value: ⭐⭐⭐⭐⭐ Fills a critical gap in T2I reasoning evaluation with significant implications for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Aligning Text to Image in Diffusion Models is Easier Than You Think](../../NeurIPS2025/image_generation/aligning_text_to_image_in_diffusion_models_is_easier_than_you_think.md)
- [\[ICLR 2026\] Everything in Its Place: Benchmarking Spatial Intelligence of Text-to-Image Models](everything_in_its_place_benchmarking_spatial_intelligence_of_text-to-image_model.md)
- [\[CVPR 2026\] CRAFT: Aligning Diffusion Models with Fine-Tuning Is Easier Than You Think](../../CVPR2026/image_generation/craft_aligning_diffusion_models_with_finetuning_is_easier_than_you_think.md)
- [\[ICLR 2026\] Direct Reward Fine-Tuning on Poses for Single Image to 3D Human in the Wild](direct_reward_fine-tuning_on_poses_for_single_image_to_3d_human_in_the_wild.md)
- [\[ICLR 2026\] RNE: plug-and-play diffusion inference-time control and energy-based training](rne_plug-and-play_diffusion_inference-time_control_and_energy-based_training.md)

</div>

<!-- RELATED:END -->
