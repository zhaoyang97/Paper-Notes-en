---
title: >-
  [Paper Note] Vision Language Models Cannot Reason About Physical Transformations
description: >-
  [ICML 2026][Multimodal VLM][Vision Language Models] By introducing the ConservationBench benchmark, this paper reveals that while 112 VLMs claim powerful perception and reasoning capabilities, they systematically fail to judge conservation in physical transformations (e.g., constant liquid volume after pouring), relying on textual priors rather than genuine visual understanding.
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Vision Language Models"
  - "Physical Reasoning"
  - "Invariance"
  - "Conservation"
  - "Benchmark"
date: 2026-05-08
content_hash: 44a37ce956a75d98
---

# Vision Language Models Cannot Reason About Physical Transformations

**Conference**: ICML 2026  
**arXiv**: [2603.07109](https://arxiv.org/abs/2603.07109)  
**Code**: To be confirmed  
**Area**: Multimodal VLM / Physical Understanding / Visual Reasoning  
**Keywords**: Vision Language Models, Physical Reasoning, Invariance, Conservation, Benchmark

## TL;DR
By introducing the ConservationBench benchmark, this paper reveals that while 112 VLMs claim powerful perception and reasoning capabilities, they systematically fail to judge conservation in physical transformations (e.g., constant liquid volume after pouring), relying on textual priors rather than genuine visual understanding.

## Background & Motivation

**Background**: VLMs have demonstrated significant capabilities in perception, reasoning, and commonsense understanding, and are being applied to tasks requiring physical world understanding, such as embodied intelligence.

**Limitations of Prior Work**: Despite impressive performance on various benchmarks, there remains a lack of deep understanding regarding whether VLMs truly grasp physical principles and can operate reliably in dynamic environments. Existing physical understanding evaluations mostly focus on static scenes or outcome prediction.

**Key Challenge**: The high accuracy of VLMs may stem from surface heuristics (e.g., a "no change" preference in text) rather than true mastery of physical principles. Diagnostic benchmarks are needed to distinguish genuine reasoning from shallow shortcuts.

**Goal**: Design cognitive-science-inspired benchmarks to evaluate whether VLMs understand conservation; systematically analyze the root causes of failure modes.

**Key Insight**: Drawing on conservation tasks from developmental psychology (Piaget's classic experiments), the authors created ConservationBench, consisting of 192 conservation videos and 192 non-conservation controls across four dimensions: quantity, length, volume, and size.

**Core Idea**: Use a paired design (conservation vs. non-conservation, identical visual background, only changing the target quantity) to force models to demonstrate understanding of transformations under matched conditions. If a model shows high accuracy on conservation tasks but low accuracy on non-conservation tasks (negative correlation $r=-0.510$), it exposes a reliance on fixed heuristics rather than flexible reasoning.

## Method

### Overall Architecture
ConservationBench performs hierarchical evaluation—Lower level: 4 conservation types $\times$ 48 video variants = 192 conservation tasks + 192 non-conservation controls; Middle level: combinations of frame extraction/frame count/prompts forming $3 \times 5 \times 4 = 60$ experimental conditions; Top level: evaluation of 112 VLMs, totaling 23,040 trials.

### Key Designs

**1. Conservation vs. Non-conservation Controls: Debunking "Unchanged" Heuristics through Paired Tasks**

If models are tested only on conservation tasks, a model that unconditionally answers "unchanged" would receive a high score, masking its inability to recognize transformations. ConservationBench addresses this by pairing each conservation video with a non-conservation control: the conservation task asks if the volume remains constant when water is poured into differently shaped cups (Answer: Yes), while the non-conservation control involves secretly adding or removing water within the same visual background (Answer: No). Only the change in the target quantity differs. Truly reasoning models should be accurate in both cases, whereas models relying on fixed biases will perform well on conservation but poorly on non-conservation—resulting in a negative correlation between accuracy types. The measured strong negative correlation of $r=-0.510$ confirms that high-scoring models primarily "game" conservation tasks by defaulting to "unchanged."

**2. Multi-frame Temporal Resolution Conditions: Testing Reasoning from Sequential Evidence**

Physical transformations are dynamic processes; theoretically, more frames provide more evidence and should lead to higher accuracy. The authors conducted factorial experiments on two factors: **Temporal Resolution** (5 conditions: 3/5/7/9/16 frames) and **Frame Sampling Strategy** (uniform sampling, manual selection, model-based selection/SeViLA). Results contradicted expectations: increasing the number of frames did not reliably improve accuracy (for quantity and length tasks, $F(4,444)=0.98,\ p=0.416$). For tasks like volume and size that "require seeing the process," sampling strategies showed a significant main effect ($F(2,222)=8.75,\ p=0.0002$), yet simple uniform sampling outperformed "high-quality" frames selected manually or by models—refined frames tended to highlight misleading static features. In other words, providing more and better evidence did not improve performance, indicating that models are not utilizing temporal visual evidence for reasoning.

**3. Text/Visual Dissociation Experiments: Separating "Textual Priors" from "Visual Deficiencies"**

The previous designs proved models are taking shortcuts; this experiment identifies whether they rely on textual shortcuts or suffer from visual failure. Three conditions were compared: (1) Standard multimodal input; (2) Clearing image content to a white canvas (text only + blank image); (3) Text only without an image container. If pure text yields high accuracy, models are dominated by textual priors; if accuracy increases after clearing the image, actual visual content is disrupting judgment. Experimentally, conservation accuracy surged to 85.7% (while non-conservation dropped to 14.3%) after clearing images. This confirms the core defect: models cannot extract and maintain object state representations from dynamic visuals and thus retreat to the textual prior that "objects should be conserved."

## Key Experimental Results

### Main Results

| Model Performance Dimension | Value | Description |
|-----------|------|------|
| VLM Accuracy Range | 20-69% | Distribution of 112 VLMs |
| Human Baseline | 98.35% | 6 participants |
| Conservation vs. Non-conservation Correlation | r = -0.510 | Strong negative correlation |
| Strict Paired Accuracy | <10% (82/112 models) | Only 3 models exceeded chance |

### Dissociation Study

| Experimental Condition | Conservation Acc | Non-conservation Acc | Interpretation |
|---------|----------|-----------|------|
| Standard (7 frames) | ~60% | ~30% | Baseline: Conservation task overestimated |
| Blank Image (White) | 85.7% | 14.3% | Textual prior forces "unchanged" answer |
| Text-only No Image | 73.7% | 26.3% | Text alone drives the bias |
| 16 Frames (Highest Res) | ~60% | ~30% | Temporal resolution provides no help |
| CoT Prompting | Decrease | Worsened | Forced step-by-step reasoning exacerbates heuristic reliance |

### Key Findings
- **Heuristic Reversal**: Models systematically tend to answer "unchanged," leading to low accuracy (~30%) on non-conservation controls.
- **Ineffective Temporal Resolution**: Increasing frames from 3 to 16 yielded no significant improvement ($F(4,444)=0.98, p=0.416$).
- **Model Scale Does Not Help**: There is no correlation between model parameters and conservation accuracy ($R^2=0.019$).
- **Mechanism Analysis**: In non-conservation failures, Qwen2.5-VL-7B showed high confidence in "same" responses, with attention focusing excessively on the initial frame.

## Highlights & Insights
- **Ingenuity of Diagnostic Benchmark Design**: Forced exposure of heuristic bias through paired tasks—using only conservation tasks would mask true reasoning deficiencies with a "default to unchanged" bias.
- **Text-Visual Dissociation Discovery**: The finding that clearing images improves accuracy provides deep insight into the tension between VLM text and visual encoders.
- **Reusable ConservationBench Framework**: The four-dimensional, variable-frame, paired-control framework can be extended to other transformation invariance tasks.

## Limitations & Future Work
- **Scenario Simplification**: Controlled laboratory conditions lack occlusion, deformable objects, or noisy observations.
- **Preliminary Mechanism Analysis**: Causal verification across model families and intervention experiments are still needed.
- **Unknown Downstream Impact**: Whether conservation reasoning defects impair embodied tasks (planning, tool use) requires empirical study.
- **Improvements**: Next-generation models should adopt predictive, state-based visual abstractions (e.g., latent object state tracking) rather than purely static semantic features.

## Related Work & Insights
- **vs. General VLM Benchmarks (e.g., MMMU, MMBench)**: These focus on broad perception and reasoning; ConservationBench is a specialized diagnostic benchmark for "transformation-invariant representation."
- **vs. Physics Benchmarks (e.g., PhysBench, BLINK)**: These cover diverse daily scenarios but remain embedded in complex contexts; ConservationBench ensures failures are directly attributable by controlling non-task-related features.
- **vs. Developmental Psychology Inspiration**: Leverages over 50 years of cognitive science literature on Piaget’s conservation tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic benchmarking of VLM conservation understanding; paired design cleverly exposes text-visual interference.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 112 VLMs + 23,040 trials + multi-factor manipulation + human baseline + dissociation experiments + mechanism analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Logically clear, grounded in cognitive science.
- Value: ⭐⭐⭐⭐⭐ Direct cautionary implications for embodied AI and VLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Vision-Speech Models: Teaching Speech Models to Converse about Images](../../CVPR2026/multimodal_vlm/vision-speech_models_teaching_speech_models_to_converse_about_images.md)
- [\[ECCV 2024\] The Hard Positive Truth About Vision-Language Compositionality](../../ECCV2024/multimodal_vlm/the_hard_positive_truth_about_visionlanguage_compositionalit.md)
- [\[CVPR 2026\] PhyCritic: Multimodal Critic Models for Physical AI](../../CVPR2026/multimodal_vlm/phycritic_multimodal_critic_models_for_physical_ai.md)
- [\[ICML 2026\] Neutral-Reference Prompting for Vision-Language Models](neutral-reference_prompting_for_vision-language_models.md)
- [\[ICML 2026\] Jailbreaking Vision-Language Models Through the Visual Modality](jailbreaking_vision-language_models_through_the_visual_modality.md)

</div>

<!-- RELATED:END -->
