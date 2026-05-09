---
title: >-
  [Paper Note] Everything in Its Place: Benchmarking Spatial Intelligence of Text-to-Image Models
description: >-
  [ICLR 2026][Image Generation][Spatial Intelligence] This paper proposes SpatialGenEval, a benchmark comprising 1,230 long, information-dense prompts spanning 10 spatial sub-domains, for systematically evaluating the spatial intelligence of 23 state-of-the-art T2I models. The benchmark reveals that spatial reasoning is the primary bottleneck. The authors additionally construct the SpatialT2I dataset to enable data-centric improvement of spatial intelligence.
tags:
  - ICLR 2026
  - Image Generation
  - Spatial Intelligence
  - Text-to-Image Generation
  - Benchmark
  - Information-Dense Prompts
  - Data-Centric Paradigm
date: 2026-05-08
content_hash: 8c4c9689d505fb66
---

# Everything in Its Place: Benchmarking Spatial Intelligence of Text-to-Image Models

**Conference**: ICLR 2026
**arXiv**: [2601.20354](https://arxiv.org/abs/2601.20354)
**Code**: Available ([GitHub](https://github.com/AMAP-ML/SpatialGenEval))
**Area**: Image Generation
**Keywords**: Spatial Intelligence, Text-to-Image Generation, Benchmark, Information-Dense Prompts, Data-Centric Paradigm

## TL;DR

This paper proposes SpatialGenEval, a benchmark comprising 1,230 long, information-dense prompts spanning 10 spatial sub-domains, for systematically evaluating the spatial intelligence of 23 state-of-the-art T2I models. The benchmark reveals that spatial reasoning is the primary bottleneck. The authors additionally construct the SpatialT2I dataset to enable data-centric improvement of spatial intelligence.

## Background & Motivation

Current T2I models excel at generating high-fidelity images and can accurately render *what* appears in a scene, yet they frequently fail to correctly depict *where* objects are, *how* they are arranged, and *why* they interact spatially. Even state-of-the-art models such as GPT-Image-1 and Qwen-Image suffer from object misplacement, incorrect orientation, failed quantitative comparisons, and inaccurate causal interaction rendering.

Limitations of existing benchmarks:

**Sparse prompts**: Benchmarks such as T2I-CompBench and GenEval rely on short prompts that can only verify object presence and simple attributes.

**Coarse evaluation granularity**: Most adopt classification or Yes/No question answering, which cannot capture higher-order spatial capabilities.

**Lack of systematic spatial intelligence stratification**: No distinction is drawn among perception, reasoning, and interaction as different levels of spatial competence.

## Method

### Overall Architecture

SpatialGenEval is designed around four core principles:

1. **Long, information-dense prompts**: Each prompt contains approximately 60 words and densely encodes 10 mutually related spatial constraints.
2. **Full-dimensional multiple-choice evaluation**: Each prompt is paired with 10 multiple-choice questions covering all spatial sub-domains.
3. **Image-dependent answers (no answer leakage)**: The generation prompt is not forwarded to the evaluator.
4. **Refuse-to-answer option**: Each multiple-choice question includes an "E: None" option to prevent forced selection of incorrect answers.

### Key Designs

#### 1. Spatial Intelligence Stratification (10 Sub-domains)

SpatialGenEval organizes spatial intelligence into 4 levels and 10 sub-domains:

**Spatial Foundations (S1/S2)**:

| Sub-domain | Evaluation Content |
|---|---|
| S1 Object Category | Compositional completeness — whether all mentioned objects are generated |
| S2 Object Attribute | Attribute binding — whether color/shape/material is correctly associated |

**Spatial Perception (S3/S4/S5)**:

| Sub-domain | Evaluation Content |
|---|---|
| S3 Spatial Position | Absolute/relative positional localization |
| S4 Spatial Orientation | Rotational alignment (e.g., facing left, inverted) |
| S5 Spatial Layout | Multi-object arrangement (linear sequences, circular, etc.) |

**Spatial Reasoning (S6/S7/S8)**:

| Sub-domain | Evaluation Content |
|---|---|
| S6 Spatial Comparison | Relative quantitative attributes (e.g., three times larger) |
| S7 Spatial Proximity | Fine-grained physical distance (contact, nearest, far apart) |
| S8 Spatial Occlusion | 3D depth and object layering |

**Spatial Interaction (S9/S10)**:

| Sub-domain | Evaluation Content |
|---|---|
| S9 Motion Interaction | Dynamic states or moments of motion |
| S10 Causal Interaction | Causal physical relationships |

#### 2. Benchmark Construction Pipeline

**Prompt generation**: Gemini 2.5 Pro is prompted with 25 real-world scenes and the definitions of 10 spatial sub-domains to generate information-dense prompts. Each prompt seamlessly integrates all 10 spatial constraints.

**Human-in-the-loop review**:
- Merging unnatural short sentences (e.g., "There is a robot. It is rusty." → "A rusty robot")
- Correcting logical contradictions (e.g., circular layout constraints)
- Replacing obscure vocabulary (e.g., *vermilion* → *bright red*)

**QA generation and validation**:
- 10 multiple-choice questions are automatically generated per prompt
- Manual inspection for answer leakage: ensuring questions do not contain explicit answers
- Programmatic addition of the "E: None" option

#### 3. SpatialT2I Dataset (Data-Centric Paradigm Beyond Evaluation)

- An additional 1,100 prompts are constructed, with images generated by 14 top open-source models.
- Qwen2.5-VL-72B evaluates image quality, and Gemini 2.5 Pro rewrites prompts to ensure consistency.
- The final dataset comprises 15,400 text–image pairs.
- Used to fine-tune SDXL, UniWorld-V1, and OmniGen2.

### Loss & Training

Evaluation procedure:
- Primary evaluator: Qwen2.5-VL-72B (open-source, ensuring reproducibility)
- 5-round voting mechanism: a response is counted as correct only when the MLLM selects the correct answer in at least 4 out of 5 rounds
- Final score: accuracy per spatial sub-domain

## Key Experimental Results

### Main Results

**Table 2: SpatialGenEval Leaderboard (23 Models)**

| Model | Scale | Overall | Foundations (S1/S2) | Perception (S3–S5) | Reasoning (S6–S8) | Interaction (S9/S10) |
|---|---|---|---|---|---|---|
| SD-1.5 | 0.86B | 28.5 | 8.5/33.7 | 19.5/29.2/38.2 | 12.8/37.7/15.6 | 42.0/47.6 |
| FLUX.1-dev | 12B | 56.5 | 51.7/73.8 | 50.0/55.5/66.7 | 28.2/62.9/28.9 | 73.1/73.8 |
| Qwen-Image | 20B | 60.6 | 61.0/77.2 | 55.6/56.7/69.7 | 28.6/67.7/30.8 | 78.1/80.2 |
| GPT-Image-1 | — | 60.5 | 56.3/74.1 | 53.3/58.9/70.4 | 31.4/66.8/30.2 | 80.9/82.2 |
| **Seed Dream 4.0** | — | **62.7** | 59.9/80.2 | 57.2/58.9/70.1 | 32.1/68.3/33.8 | 83.0/83.8 |

**Table 6: Fine-tuning Effects with SpatialT2I**

| Model | Overall (Before) | Overall (After) | Gain |
|---|---|---|---|
| SD-XL | 41.2 | 45.4 | +4.2% |
| UniWorld-V1 | 54.2 | 59.9 | +5.7% |
| OmniGen2 | 56.4 | 60.8 | +4.4% |

### Ablation Study

**Evaluator consistency**: GPT-4o and Qwen2.5-VL-72B produce identical model rankings, validating the robustness of the evaluation protocol.

**Human alignment study**: Gemini-2.5-Pro achieves a balanced accuracy of 84.2%, while Qwen2.5-VL-72B achieves 80.4%.

### Key Findings

1. **Spatial reasoning is the primary bottleneck**: Comparison and Occlusion sub-tasks frequently score below 30%, approaching the random-choice baseline of 20%.
2. **Open-source models are closing the gap**: Qwen-Image (60.6%) vs. Seed Dream 4.0 (62.7%).
3. **Text encoders are critical**: Models using LLM-based encoders (e.g., Qwen-Image) substantially outperform those relying solely on CLIP.
4. **Unified architectures are more parameter-efficient**: The 7B Bagel (57.0%) approaches the performance of the 12B FLUX.1-krea (58.5%).
5. **The data-centric paradigm is effective**: Fine-tuning on SpatialT2I consistently yields improvements of 4–6 percentage points.

## Highlights & Insights

1. **Information-dense prompt design**: Integrating 10 spatial constraints into a single 60-word prompt avoids the discriminability limitations of simple prompts.
2. **Hierarchical spatial intelligence definition**: The progression from Foundations → Perception → Reasoning → Interaction is both conceptually clear and extensible.
3. **Engineering design of the "E: None" option**: Prevents forced selection, improving evaluation accuracy.
4. **Data flywheel via SpatialT2I**: The benchmark's by-product can be directly leveraged for model improvement, forming a closed loop.

## Limitations & Future Work

1. The highest score is only ~63%, barely above a passing threshold — indicating the task remains highly challenging.
2. Prompts of approximately 60 words may exceed the effective processing length (77 tokens) of some CLIP-based encoders.
3. Scene coverage spans 25 categories; further expansion to more complex interaction scenarios is warranted.
4. Evaluation relies on MLLM judgment, which may introduce biases inherent to the evaluator model.
5. The quality of the SpatialT2I dataset is bounded by the current capabilities of the generative models used.

## Related Work & Insights

- **T2I-CompBench**: Short prompts with Yes/No evaluation; insufficient coverage.
- **DPG-Bench**: Long prompts but scored with ratings; limited discriminability.
- **TIIF-Bench**: Mixed prompt lengths but Yes/No evaluation.
- Insight: The paradigm of information-dense prompts combined with full-dimensional evaluation can be extended to spatial intelligence benchmarking in video generation, 3D generation, and other domains.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic spatial intelligence benchmark for T2I evaluation
- Technical Contribution: ⭐⭐⭐⭐⭐ — Integrates benchmark design, dataset construction, and large-scale evaluation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 23 models, multi-evaluator validation, human alignment
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, though table-heavy
- Overall Recommendation: ⭐⭐⭐⭐⭐ — A comprehensive and high-impact perspective on spatial capabilities in T2I generation

## Background & Motivation

## Core Problem

## Method

## Key Experimental Results

## Highlights & Insights

## Limitations & Future Work

## Related Work & Insights

## Related Work

## Rating

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Blueprint-Bench: Comparing Spatial Intelligence of LLMs, Agents and Image Models](blueprint-bench_comparing_spatial_intelligence_of_llms_agents_and_image_models.md)
- [\[ICLR 2026\] Easier Painting Than Thinking: Can Text-to-Image Models Set the Stage, but Not Direct the Play?](easier_painting_than_thinking_can_text-to-image_models_set_the_stage_but_not_dir.md)
- [\[ICCV 2025\] CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models](../../ICCV2025/image_generation/compass_enhancing_spatial_understanding_in_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](../../CVPR2026/image_generation/micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)
- [\[ICLR 2026\] Diverse Text-to-Image Generation via Contrastive Noise Optimization](diverse_text-to-image_generation_via_contrastive_noise_optimization.md)

<!-- RELATED:END -->
