---
title: >-
  [Paper Note] Beyond Recognition: Evaluating Visual Perspective Taking in Vision Language Models
description: >-
  [CVPR 2026][Multimodal VLM][Visual Perspective Taking] By constructing the Isle-Brick-V2 benchmark using psychologically inspired controlled LEGO scenes, this work systematically exposes significant deficiencies in current VLMs' Visual Perspective Taking (VPT) capabilities—even when scene understanding is near-perfect, spatial reasoning and perspective-taking performance degrade substantially, accompanied by persistent directional biases.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Visual Perspective Taking
  - Theory of Mind
  - Spatial Reasoning
  - VLM Evaluation
  - Cognitive Science
date: 2026-05-08
content_hash: b4d56ee132ebf35b
---

# Beyond Recognition: Evaluating Visual Perspective Taking in Vision Language Models

**Conference**: CVPR 2026
**arXiv**: [2505.03821](https://arxiv.org/abs/2505.03821)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: Visual Perspective Taking, Theory of Mind, Spatial Reasoning, VLM Evaluation, Cognitive Science

## TL;DR
By constructing the Isle-Brick-V2 benchmark using psychologically inspired controlled LEGO scenes, this work systematically exposes significant deficiencies in current VLMs' Visual Perspective Taking (VPT) capabilities—even when scene understanding is near-perfect, spatial reasoning and perspective-taking performance degrade substantially, accompanied by persistent directional biases.

## Background & Motivation

**Background**: VLMs (GPT-4o, Gemini, Claude, etc.) demonstrate strong performance on visual tasks such as object recognition and counting, with several models claiming spatial understanding capabilities. Benchmarks such as 3D-PC have begun evaluating VLMs' perspective-taking abilities, but they largely rely on natural scenes that make variable control difficult.

**Limitations of Prior Work**: Existing VLM evaluations primarily focus on the "recognition" level (what can be seen), lacking systematic assessment of the "reasoning" level (how a scene appears from another's viewpoint). Natural-scene benchmarks are susceptible to data contamination and cannot precisely isolate failure factors (i.e., whether failures stem from recognition or reasoning).

**Key Challenge**: VLMs achieve near-perfect performance at the surface level of object recognition, yet exhibit significant performance degradation when spatial reasoning and perspective transformation are required. This reflects a deep mismatch between recognition and reasoning—models may rely on linguistic priors (e.g., defaulting to "facing East") rather than genuine visual-spatial reasoning.

**Goal**: To systematically address whether VLMs can perform Visual Perspective Taking (VPT), and to isolate specific failure points through hierarchical diagnostic evaluation.

**Key Insight**: Drawing on two levels of VPT from psychology—Level-1 (understanding whether another agent can see an object) and Level-2 (adopting another's viewpoint to determine relative object positions)—the work designs minimal-contrast experiments in which only one cognitively relevant factor varies at a time.

**Core Idea**: Using controlled LEGO scenes paired with seven hierarchically structured diagnostic questions, the work disentangles scene understanding, spatial reasoning, and perspective-taking as three distinct cognitive levels, thereby revealing systematic VPT deficiencies in VLMs.

## Method

### Overall Architecture
The Isle-Brick-V2 benchmark comprises 144 visual tasks: 9 figurine types × 9 object types × 4 spatial positions (left/right/front/back) × 2 figurine orientations × 2 viewpoints (bird's-eye/eye-level). Each task is accompanied by 7 open-ended diagnostic questions, evaluated under zero-shot, zero-temperature conditions.

### Key Designs

1. **Three-Tier Diagnostic Question Design**:

    - **Function**: Decomposes VPT into three progressively demanding cognitive levels.
    - **Mechanism**: Q1–Q3 assess scene understanding (object counting, figurine counting, co-planarity judgment); Q4–Q5 assess spatial reasoning (object relative direction, figurine orientation); Q6–Q7 assess visual perspective-taking (Level-1 visibility judgment, Level-2 egocentric localization).
    - **Design Motivation**: Q6 and Q7 can be viewed as logical compositions of the answers to Q4 and Q5 (e.g., Q6: object in the same direction = figurine can see it), enabling precise localization of failure points.

2. **Minimal-Contrast Methodology**:

    - **Function**: Controls variables by varying only one cognitive factor at a time.
    - **Mechanism**: Analogous to the dot-perspective paradigm in psychology and the COMPS conceptual minimal pairs framework; all stimuli share the same scene but differ on a single factor (e.g., figurine orientation).
    - **Design Motivation**: LEGO elements enable precise scene control, eliminating confounds present in natural images and mitigating risks of data contamination.

3. **Open-Ended Evaluation Format**:

    - **Function**: Avoids the guessing bias and positional bias inherent in multiple-choice formats.
    - **Mechanism**: All questions are answered in free-form text; evaluation uses averaged prediction correctness—computing the proportion of correct components in the model's response (e.g., if the prediction is "northeast" and the correct answer is "north," the score is 0.5).
    - **Design Motivation**: Matches realistic user interaction patterns and avoids handcrafted model-specific prompt engineering strategies.

### Loss & Training
This work presents an evaluation benchmark and involves no model training. All models are evaluated zero-shot at temperature 0 with a maximum of 128 tokens; each question is assessed independently with context cleared between evaluations.

## Key Experimental Results

### Main Results

| Model | Scene Understanding | Spatial Reasoning | Perspective Taking |
|-------|--------------------|--------------------|-------------------|
| GPT-4o | 100.0% | 85.8% | 73.3% |
| Gemini Robotics-ER 1.5 | 100.0% | 80.2% | 49.3% |
| Claude 3.5 Sonnet | 96.5% | 72.8% | 45.7% |
| Qwen3-4B-Instruct | 99.8% | 71.9% | 45.9% |
| Llama-3.2-11B | 92.4% | 61.7% | 40.6% |
| Random Baseline | 38.9% | 31.7% | 41.1% |

Note: Most open-source models exceed the random baseline by only a small margin on VPT tasks (+4.75 pp), while GPT-4o leads substantially (+32.15 pp).

### Ablation Study (Directional Bias Intervention, GPT-4-Turbo)

| Intervention | Q5 Accuracy | Bias Change |
|--------------|-------------|-------------|
| Original | 41.7% | Strong bias toward East |
| Object removed | ~44.4% | East still predicted in 31/36 cases |
| Enlarged by 10%/30%/50% | 41.7%–47.2% | Bias persists |
| NESW visual markers added | 34.3% | East still predicted in 27/36 cases |
| Real human replaces figurine | N/A | All 8/8 predictions are East |

### Key Findings
- **Scene understanding ≠ spatial reasoning ≠ perspective-taking**: There are pronounced performance drops across the three levels; GPT-4o declines from 100% to 73%, while open-source models drop to near-random performance.
- **Directional bias is remarkably persistent**: GPT-4-Turbo consistently favors the East direction regardless of object removal, image magnification, addition of directional markers, or substitution with real human photographs, indicating that the bias originates from the model's linguistic priors rather than visual perception.
- **Providing correct orientation does not resolve VPT failures**: When the model is supplied with the ground-truth answer to Q5 (figurine orientation), performance on Q6 (VPT) improves only marginally, demonstrating that VPT difficulties extend beyond directional misjudgment.

## Highlights & Insights
- **Transfer of cognitive science methodology**: The work systematically imports the Level-1/Level-2 VPT framework and minimal-contrast methodology from psychology into VLM evaluation. This interdisciplinary approach is highly instructive and may be extended to evaluating other cognitive capabilities such as causal or counterfactual reasoning.
- **Discovery of directional bias**: The finding that VLMs may rely on linguistic priors (e.g., "facing East") rather than genuine visual-spatial reasoning has far-reaching implications for VLM reliability and safety—potentially causing systematic errors in applications such as autonomous driving that depend on spatial reasoning.
- **Lower-bound argument**: The controlled LEGO scenes represent the "simplest possible" VPT conditions (ideal lighting, no occlusion, isolated objects). VLM failures under these conditions indicate that the underlying problem is fundamental.

## Limitations & Future Work
- Only single figurine–single object configurations are used; multi-agent, dynamic, and complex occlusion scenarios are not addressed.
- Spatial coverage is limited to four cardinal directions and two orientations; finer-grained angular variation may reveal additional failure modes.
- Intervention experiments are conducted primarily on GPT-4-Turbo; bias characteristics may differ across other models.
- No solutions are proposed—future work could explore explicit geometric representations, mental rotation training protocols, or hybrid approaches combining symbolic spatial reasoning with learned representations.

## Related Work & Insights
- **vs. 3D-PC**: 3D-PC evaluates depth ordering and line-of-sight classification in natural scenes, but is susceptible to data contamination and cannot isolate failure factors. Isle-Brick-V2 enables more precise diagnosis through controlled scenes.
- **vs. Omni-Perspective**: Omni-Perspective scales to large-scale multimodal Theory-of-Mind evaluation, but its multiple-choice format and reliance on natural scenes limit control precision.
- **vs. SpatialVLM/SpatialRGPT**: These works augment spatial understanding via 3D data, but do not systematically evaluate VPT capabilities.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to systematically import the psychological VPT framework into VLM evaluation; the directional bias finding is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Nine models, 144 tasks, and multiple intervention experiments; additional models and proposed solutions would strengthen the work.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is logically structured, experimentally rigorous, and presents its interdisciplinary narrative with clarity.
- Value: ⭐⭐⭐⭐⭐ Provides a fundamental re-examination of VLM spatial reasoning capabilities, with important cautionary implications for applications in autonomous driving, robotics, and related domains.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Think360: Evaluating the Width-centric Reasoning Capability of MLLMs Beyond Depth](think_360_evaluating_the_width-centric_reasoning_capability_of_mllms_beyond_dept.md)
- [\[CVPR 2026\] SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models](spatialqa_a_benchmark_for_evaluating_spatial_logical_reasoning_in_vision-languag.md)
- [\[CVPR 2026\] Beyond Static Artifacts: A Forensic Benchmark for Video Deepfake Reasoning in Vision Language Models](beyond_static_artifacts_a_forensic_benchmark_for_video_deepfake_reasoning_in_vis.md)
- [\[CVPR 2026\] TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition](trivia_self-supervised_fine-tuning_of_vision-language_models_for_table_recogniti.md)
- [\[CVPR 2026\] Taxonomy-Aware Representation Alignment for Hierarchical Visual Recognition with Large Multimodal Models](taxonomy-aware_representation_alignment_for_hierarchical_visual_recognition_with.md)

<!-- RELATED:END -->
