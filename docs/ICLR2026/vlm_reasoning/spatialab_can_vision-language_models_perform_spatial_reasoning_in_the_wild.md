---
title: >-
  [Paper Note] SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?
description: >-
  [ICLR 2026][Multimodal VLM][Spatial Reasoning] This paper introduces SpatiaLab, a real-world spatial reasoning benchmark comprising 1…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "Spatial Reasoning"
  - "VLM Benchmark"
  - "MCQ Evaluation"
  - "Open-ended Evaluation"
  - "Real-world Scenarios"
date: 2026-05-08
content_hash: e86f277a9b0db7a3
---

# SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?

**Conference**: ICLR 2026  
**arXiv**: [2602.03916](https://arxiv.org/abs/2602.03916)  
**Code**: [spatialab-reasoning.github.io](https://spatialab-reasoning.github.io/)  
**Area**: Multimodal VLM  
**Keywords**: Spatial Reasoning, VLM Benchmark, MCQ Evaluation, Open-ended Evaluation, Real-world Scenarios

## TL;DR
This paper introduces SpatiaLab, a real-world spatial reasoning benchmark comprising 1,400 visual QA pairs spanning 30 subcategories across 6 major spatial task categories. Supporting both MCQ and open-ended evaluation formats, SpatiaLab reveals a substantial gap between the strongest current VLMs (InternVL3.5-72B: 54.93% MCQ) and humans (87.57%), with the gap widening further under open-ended settings.

## Background & Motivation

**Background**: Spatial reasoning is a foundational cognitive ability for humans and is critical to robotics, autonomous driving, and AR/VR. While VLMs have made progress in multimodal representation and language grounding, spatial judgment in real-world environments remains fragile.

**Limitations of Prior Work**:
   - Existing spatial reasoning benchmarks are overly simplified: most focus on binary spatial relations, coarse depth categorization, or synthetic/puzzle-style scenes.
   - Controlled environments reduce perceptual and reasoning difficulty, causing apparent saturation that masks failures under distribution shift.
   - Critical challenges such as occlusion reasoning, cross-view scale consistency, and path planning under partial observability are severely undersampled.
   - Models that perform well on synthetic benchmarks such as ScanQA and BLINK frequently fail in real-world settings.

**Key Challenge**: Humans seamlessly integrate multidimensional spatial information—relative position, depth, orientation, scale, navigation, and 3D geometry—whereas VLMs fall far short of human performance on any single dimension, let alone joint multi-dimensional reasoning.

**Goal**:
   - Construct a real-world benchmark covering all core axes of spatial reasoning.
   - Employ dual-format evaluation (MCQ and open-ended) to avoid format bias.
   - Evaluate 25+ VLMs and establish a human baseline.
   - Conduct in-depth failure analysis and provide actionable directions for improvement.

**Key Insight**: Drawing from cognitive psychology's taxonomy of spatial cognition, the paper systematically decomposes spatial reasoning into $6 \times 5 = 30$ fine-grained task types, constructing the benchmark from real photographs rather than synthetic data.

**Core Idea**: SpatiaLab employs dual-format evaluation across 30 real-world spatial reasoning tasks to systematically expose fundamental deficiencies of VLMs in depth perception, occlusion reasoning, navigation planning, and 3D geometry.

## Method

### Overall Architecture
SpatiaLab = Benchmark Dataset + Evaluation Protocol + Improvement Strategy Exploration

- 6 major categories: Relative Positioning, Depth & Occlusion, Orientation, Size & Scale, Spatial Navigation, and 3D Geometry.
- Each major category contains 5 subcategories → 30 task types in total.
- Each subcategory contains ≥25 questions, each major category ≥200 questions → 1,400 verified QA pairs in total.
- Dual format: MCQ (4-choice) + open-ended generation.

### Key Designs

1. **Multi-source Image Collection**

    - Function: Construct a visually diverse real-world image repository.
    - Mechanism: Three complementary sources—automated web crawling, targeted online retrieval, and manual indoor/outdoor photography. Systematic coverage along 6 meta-dimensions: illumination, texture complexity, edge complexity, spatial relations, material type, and gravity constraints.
    - Design Motivation: Ensure the benchmark reflects real-world visual noise and complexity rather than controlled laboratory conditions.
    - Complexity statistics: average 21.48 objects per image, 11.88 partially visible, 3.23 depth layers, and 2.07 spatial reasoning steps per chain.

2. **Three-stage Annotation and Quality Control**

    - Function: Ensure semantic validity, answer correctness, and task clarity of all QA pairs.
    - Mechanism: Phase 1 annotator training → Phase 2 paired spatial QA generation per image → Phase 3 dual-format encoding. Three rounds of review: semantic validation → independent verification → gold-standard establishment.
    - Design Motivation: Error rates in spatial reasoning QA are high under complex scenes; three rounds of review ensure the reliability of the final 1,400 questions.

3. **Improvement Strategy Exploration**

    - Function: Systematically test multiple approaches for enhancing VLM spatial reasoning.
    - Methods covered: intrinsic reasoning, CoT prompting, CoT + self-reflection, SFT fine-tuning (40% data / 60% evaluation), and a multi-agent system (SpatioXolver).
    - Design Motivation: Beyond exposing problems, the paper provides actionable improvement directions. SFT yields the best results on navigation and orientation; multi-agent reasoning helps on orientation but stagnates or degrades on other categories.

### Loss & Training
(This is a benchmark paper; no training loss is defined. SFT experiments fine-tune Qwen-VL2.5-3B-Instruct with a standard cross-entropy loss.)

## Key Experimental Results

### Main Results (MCQ Format, 25+ Models)

| Model | 3D Geometry | Depth & Occlusion | Orientation | Relative Positioning | Size & Scale | Navigation | Overall |
|-------|-------------|-------------------|-------------|----------------------|--------------|------------|---------|
| Human Baseline | 93.70 | 74.13 | 91.58 | 91.51 | 88.89 | 87.76 | **87.57** |
| InternVL3.5-72B | 50.00 | 57.14 | 53.47 | 66.04 | 49.21 | 54.85 | 54.93 |
| GPT-5-mini | 48.74 | 54.83 | 60.40 | 62.74 | 44.84 | 56.54 | 54.29 |
| o4-mini-medium | 51.26 | 58.30 | 54.95 | 64.15 | 40.87 | 51.48 | 53.21 |
| Spatial-specialized Models | ~42 | ~38 | ~48 | ~38 | ~43 | ~39 | ~41 |
| Random Baseline | 25.00 | 25.00 | 25.00 | 25.00 | 25.00 | 25.00 | 25.00 |

### Open-ended Format Comparison

| Model | MCQ Overall | Open-ended Overall | Performance Drop |
|-------|-------------|-------------------|-----------------|
| GPT-5-mini | 54.29 | **40.93** | −13.36 |
| o4-mini-medium | 53.21 | 37.86 | −15.35 |
| InternVL3.5-72B | 54.93 | 23.36 | **−31.57** |
| Human Baseline | 87.57 | 64.93 | −22.64 |
| Avg. MCQ→Open Gap | — | — | **−23.0%** |

### Key Findings
- **Top models achieve only 55% (MCQ) / 41% (open-ended)**: A large gap remains relative to human performance of 88% / 65%. Spatial-specialized models perform even worse (~41%), indicating that current specialization approaches are ineffective.
- **Open-ended evaluation exposes true capability**: The average MCQ-to-open-ended drop is 23%; spatial-specialized models drop the most (~27%), suggesting that MCQ scores can overestimate true spatial reasoning ability.
- **Three most challenging categories**: Size & Scale, Depth & Occlusion, and Spatial Navigation consistently emerge as bottlenecks, with most models scoring below 50% (or 30% in open-ended settings).
- **Model scale ≠ spatial reasoning ability**: Llama-3.2-11B achieves only 30.5%, worse than many 4B models, indicating that spatial reasoning requires specialized capabilities beyond raw parameter count.
- **Limited gains from reasoning augmentation**: CoT helps on the Orientation category; SFT improves Navigation (+7.69%); however, multi-agent systems degrade performance on Occlusion and Size & Scale.
- **Systematic failure patterns**: Tasks involving object rotation (2%), reflective surfaces (<20%), and tool handedness (<30%) result in near-total failure.

## Highlights & Insights
- **Well-designed real-world dual-format evaluation**: The 30-task taxonomy across 1,400 questions represents the most fine-grained categorization in spatial reasoning research; the MCQ + open-ended dual format addresses format bias, a critical issue overlooked by prior benchmarks.
- **Counterintuitive finding—spatial-specialized models underperform general models**: SpaceOm, SpaceThinker, and SpaceQwen all lag behind InternVL3.5-72B on real-world scenes, demonstrating that spatial capabilities acquired from synthetic training data do not transfer to real-world settings.
- **Diagnostic value of error analysis**: Cluster analysis reveals that failures concentrate in three patterns: spatial mislocalization, perspective/scale errors, and occlusion ordering failures—directly attributable to the lack of geometric supervision in VLM training.
- **Necessity of open-ended evaluation**: The average MCQ-to-open-ended drop of 23% is largest on Navigation (which requires the most multi-step reasoning), indicating that current VLMs rely on elimination strategies rather than genuine spatial understanding.

## Limitations & Future Work
- Although high in quality, the 1,400 questions are limited in quantity; as few as 25+ questions per subcategory may be insufficient for stable evaluation.
- Open-ended evaluation relies on an LLM judge (Gemini-2.5-Flash); while Cohen's kappa = 0.738, the judging process itself remains imperfect.
- Temporal spatial reasoning in video settings is not covered.
- **Directions for improvement**: Developing spatial reasoning pre-training data based on physics engines, or incorporating explicit geometric encoding modules into VLMs to address spatial reasoning deficiencies.

## Related Work & Insights
- **vs. BLINK-Spatial (2024)**: 14 task types / 3.8K questions but mixes synthetic and real data; best performance 59%. SpatiaLab focuses on 30 real-world task types, offering finer granularity and greater difficulty.
- **vs. OmniSpatial (2025)**: 50 categories but only 1.5K questions in a puzzle setting; best performance 56%. SpatiaLab emphasizes real-world scenes over puzzle-style settings.
- **vs. VSI-Bench (2025)**: An indoor video benchmark with 8 categories; best performance 45%. SpatiaLab covers a broader range of scene types and image modalities.

## Rating
- Novelty: ⭐⭐⭐⭐ The 30-category taxonomy and dual-format evaluation design are novel, though the core methodology (benchmark construction) is not an entirely new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly comprehensive: 25+ models, human baselines, improvement strategy exploration, and error analysis.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with in-depth analysis, though somewhat lengthy.
- Value: ⭐⭐⭐⭐⭐ Fills a critical gap in real-world spatial reasoning evaluation, quantifies the VLM–human gap, and provides important guidance for the VLM community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[ICLR 2026\] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)
- [\[ICLR 2026\] Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation](spatial_captcha_generatively_benchmarking_spatial_reasoning_for_human-machine_di.md)
- [\[CVPR 2026\] SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models](../../CVPR2026/multimodal_vlm/spatialqa_a_benchmark_for_evaluating_spatial_logical_reasoning_in_vision-languag.md)
- [\[ICLR 2026\] SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs](spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)

</div>

<!-- RELATED:END -->
