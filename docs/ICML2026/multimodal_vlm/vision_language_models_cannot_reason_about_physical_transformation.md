---
title: >-
  [Paper Note] Vision Language Models Cannot Reason Physical Transformations
description: >-
  [ICML 2026][Multimodal VLM][Vision Language Models] This paper introduces the ConservationBench benchmark to reveal that 112 VLMs, while claiming strong perception and reasoning capabilities…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Vision Language Models"
  - "Physical Reasoning"
  - "Invariance"
  - "Conservation"
  - "Benchmarking"
date: 2026-05-08
content_hash: f4808b469f071b4a
---

# Vision Language Models Cannot Reason Physical Transformations

**Conference**: ICML 2026  
**arXiv**: [2603.07109](https://arxiv.org/abs/2603.07109)  
**Code**: TBD  
**Area**: Multimodal VLM / Physical Understanding / Visual Reasoning  
**Keywords**: Vision Language Models, Physical Reasoning, Invariance, Conservation, Benchmarking

## TL;DR
This paper introduces the ConservationBench benchmark to reveal that 112 VLMs, while claiming strong perception and reasoning capabilities, systematically fail when judging conservation during physical transformations (such as volume invariance when pouring water), relying on text priors rather than true visual understanding.

## Background & Motivation

**Background**: VLMs have demonstrated significant capabilities in perception, reasoning, and commonsense understanding, and have been applied to tasks requiring physical world understanding, such as embodied intelligence.

**Limitations of Prior Work**: Despite the impressive performance of VLMs across various benchmarks, there remains a lack of deep understanding regarding whether they truly grasp physical principles and can operate reliably in dynamic environments. Existing physical understanding evaluations mostly focus on static scenes or outcome prediction.

**Key Challenge**: The high accuracy of VLMs may stem from surface heuristics (such as a preference for "invariant" in text) rather than a true mastery of physical principles. Diagnostic benchmarks are needed to distinguish true reasoning from shallow shortcuts.

**Goal**: Design a benchmark inspired by cognitive science to evaluate whether VLMs understand conservation; systematically analyze the root causes of failure modes.

**Key Insight**: Drawing from developmental psychology conservation tasks (Piaget's classical experiments), the authors created ConservationBench, consisting of 192 conservation videos and 192 non-conservation controls across four dimensions: quantity, length, volume, and size.

**Core Idea**: Through a paired design (conservation vs. non-conservation, identical visual background, only changing the target quantity), models are forced to demonstrate an understanding of transformations under matched conditions. If a model exhibits high accuracy on conservation tasks but low accuracy on non-conservation tasks (inverse correlation $r=-0.510$), it exposes a reliance on fixed heuristics rather than flexible reasoning.

## Method

### Overall Architecture
ConservationBench utilizes a hierarchical evaluation: the bottom layer consists of 4 conservation types × 48 video variants = 192 conservation tasks + 192 non-conservation controls; the middle layer combines frame extraction/frame count/prompts into $3 \times 5 \times 4 = 60$ experimental conditions; the top layer evaluates 112 VLMs, totaling 23,040 trials.

### Key Designs

1.  **Conservation vs. Non-conservation Controls**:
    - **Function**: Identify whether models are based on heuristics or true reasoning through paired tasks.
    - **Mechanism**: Conservation tasks require judging if "volume remains unchanged after pouring water into different shaped cups" (Yes); non-conservation controls introduce adding/removing water within the same visual background (No). An inverse correlation of $r=-0.510$ indicates that high-scoring models often gain points on conservation tasks while failing non-conservation tasks.
    - **Design Motivation**: Address the limitations of single-truth benchmarks—if a model unconditionally answers "invariant," it would receive a high score, masking its inability to recognize transformations.

2.  **Multi-frame Temporal Resolution Conditions**:
    - **Function**: Test whether models can reason physical transformations from temporal evidence.
    - **Mechanism**: Five conditions (3/5/7/9/16 frames) are set to compare whether models improve as the number of frames increases. Frames are extracted via fixed counts (uniform, manual selection, or model selection).
    - **Design Motivation**: Frame extraction methods produced a significant main effect ($F(2,222)=8.75, p=0.0002$) for volume and size tasks, where uniform sampling actually outperformed manual/model selection—suggesting models perform worse even with high-quality visual evidence.

3.  **Text/Vision Dissociation Experiments**:
    - **Function**: Separate whether the root cause of failure is text priors or visual deficiencies.
    - **Mechanism**: Three experiments: (1) Standard conditions; (2) Cleared image content (white canvas) while retaining text; (3) Text-only without images. If pure text also yields high accuracy, it proves a strong text bias; if accuracy rises after clearing images, it indicates that actual visual content disrupts rather than enhances conservation judgment.
    - **Design Motivation**: Reveal that the core defect of models is not "not knowing physics" but "inability to extract and maintain object state representations from dynamic visuals."

## Key Experimental Results

### Main Results

| Performance Dimension | Value | Description |
|-----------|------|------|
| VLM Accuracy Range | 20-69% | Distribution across 112 VLMs |
| Human Baseline | 98.35% | 6 subjects |
| Conservation-Non-conservation Correlation | r = -0.510 | Strong negative correlation |
| Strict Pair Accuracy | <10% (82/112 models) | Only 3 models exceeded chance |

### Dissociation Experiments

| Experimental Condition | Conservation Accuracy | Non-conservation Accuracy | Interpretation |
|---------|----------|-----------|------|
| Standard (7 frames) | ~60% | ~30% | Baseline: Overestimation of conservation |
| Cleared Image (White) | 85.7% | 14.3% | Text prior forces "Same" answer |
| Text-only | 73.7% | 26.3% | Text alone drives bias |
| 16 Frames (Highest Res) | ~60% | ~30% | Temporal resolution provides no help |
| CoT Prompting | Decreased | Worsened | Forced step-by-step reasoning worsens heuristic reliance |

### Key Findings
- **Heuristic Inversion**: Models show systematically low accuracy (~30% mean) on non-conservation controls, with a tendency to answer "Same."
- **Temporal Resolution Ineffectiveness**: Increasing frames from 3 to 16 yielded no significant improvement ($F(4,444)=0.98, p=0.416$).
- **Model Scale Ineffectiveness**: No correlation exists between model parameters and conservation accuracy ($R^2=0.019$).
- **Mechanism Analysis**: Qwen2.5-VL-7B answers "Same" with high confidence in non-conservation failure cases, with attention excessively focused on the initial frame.

## Highlights & Insights
- **Ingenuity of Diagnostic Benchmark Design**: The paired tasks force the exposure of heuristic biases—using conservation tasks alone would allow the "default to same" bias to mask true reasoning defects.
- **Discovery via Text-Vision Dissociation**: The finding that clearing images actually improves accuracy provides deep insight into the tension between VLM text and visual encoders.
- **Reusable ConservationBench Framework**: The four-dimensional, frame-variable, and paired-control framework can be extended to other transformation invariance tasks.

## Limitations & Future Work
- **Scenario Simplification**: Controlled laboratory conditions do not cover occlusions, deformable objects, or noisy observations.
- **Preliminary Mechanism Analysis**: Causal verification and intervention experiments across model families are still lacking.
- **Impact on Downstream Tasks**: Whether conservation reasoning defects harm embodied tasks (planning, tool use) still requires empirical evidence.
- **Improvement**: Next-generation models should adopt predictive, state-based visual abstractions (such as latent object state tracking) rather than purely static semantic features.

## Related Work & Insights
- **vs General VLM Benchmarks (MMMU/MMBench)**: Those focus on broad perception and reasoning; ConservationBench is a micro-benchmark specifically diagnosing "transformation-invariant representation."
- **vs Physics Benchmarks (PhysBench/BLINK)**: These cover diverse daily physical scenes but often embed complex contexts; ConservationBench ensures observed failures are directly attributable by controlling non-task-related features.
- **vs Developmental Psychology**: Draws from over 50 years of cognitive science literature on Piaget's conservation tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First systematic benchmarking of VLM understanding of conservation; paired design cleverly exposes text-visual interference mechanisms.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 112 VLMs + 23,040 trials + multi-factor manipulation + human baseline + dissociation experiments + mechanism analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear logic, rooted in cognitive science principles.
- **Value**: ⭐⭐⭐⭐⭐ Direct warning implications for embodied AI and VLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World](../../CVPR2026/multimodal_vlm/thinking_in_dynamics_how_multimodal_large_language_models_perceive_track_and_rea.md)
- [\[ICCV 2025\] Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models](../../ICCV2025/multimodal_vlm/physics_context_builders_a_modular_framework_for_physical_reasoning_in_vision-la.md)
- [\[ICML 2026\] Find, Fix, Reason: Context Repair for Video Reasoning](find_fix_reason_context_repair_for_video_reasoning.md)
- [\[ICML 2026\] Neutral-Reference Prompting for Vision-Language Models](neutral-reference_prompting_for_vision-language_models.md)
- [\[ICCV 2025\] LLaVA-CoT: Let Vision Language Models Reason Step-by-Step](../../ICCV2025/multimodal_vlm/llava-cot_let_vision_language_models_reason_step-by-step.md)

</div>

<!-- RELATED:END -->
