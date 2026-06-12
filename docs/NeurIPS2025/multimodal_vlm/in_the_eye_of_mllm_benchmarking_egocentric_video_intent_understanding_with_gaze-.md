---
title: >-
  [Paper Note] In the Eye of MLLM: Benchmarking Egocentric Video Intent Understanding with Gaze-Guided Prompting
description: >-
  [Multimodal VLM] This paper proposes the EgoGazeVQA benchmark and three gaze-guided prompting strategies (textual / visual / salience map)…
tags:
  - "Multimodal VLM"
date: 2026-05-08
content_hash: fd40106af42601f9
---

# In the Eye of MLLM: Benchmarking Egocentric Video Intent Understanding with Gaze-Guided Prompting

## TL;DR

This paper proposes the EgoGazeVQA benchmark and three gaze-guided prompting strategies (textual / visual / salience map), providing the first systematic validation of eye-gaze signals for improving egocentric video intent understanding in MLLMs. The best configuration, Qwen2.5-VL-72B + GazeS, achieves a 5.8 percentage-point gain in average accuracy.

## Background & Motivation

**Background**: Multimodal large language models (MLLMs) have achieved remarkable progress on video understanding tasks. Egocentric video, naturally aligned with the user's perspective, is considered an ideal medium for building proactive, personalized AI assistants.

**Limitations of Prior Work**: Existing egocentric video QA benchmarks (e.g., QaEgo4D, EgoSchema, EgoTextVQA) rely solely on global visual frame information, entirely neglecting eye-gaze signals—a core cue that reflects user attention and intent. MLLMs perform substantially below human level when reasoning about users' spatial and temporal intentions.

**Key Challenge**: MLLMs construct visual tokens from global frames, which provides broad contextual coverage but fails to capture the wearer's explicit attention signal directed at specific objects or regions, leading to frequent failures on questions that require understanding "what the user is looking at."

**Goal**: (1) A benchmark for egocentric video intent understanding that integrates gaze signals is lacking; (2) effective methods for leveraging gaze signals to enhance MLLM intent understanding are lacking.

**Key Insight**: Video clips with gaze data are extracted from three large-scale datasets—Ego4D, EgoExo4D, and EGTEA Gaze+. High-quality QA pairs are constructed via MLLM-assisted generation followed by human review, and multiple gaze-guided prompting strategies are designed.

**Core Idea**: Eye-gaze signals, as direct indicators of user intent, can be effectively injected into MLLMs through textual, visual, and salience-map prompting forms, significantly improving egocentric video intent understanding.

## Method

### Overall Architecture

EgoGazeVQA consists of two major components: a **benchmark construction pipeline** and **gaze-guided prompting strategies**. The pipeline extracts video clips, frame descriptions, and gaze coordinates from three egocentric datasets; leverages Qwen2.5-VL to generate spatial, temporal, and causal intent QA pairs (5 options each); and applies human review across 6 dimensions to ensure quality. During inference, three gaze-guided prompting strategies are applied on top of standard MLLM inputs to enhance intent understanding.

### Key Designs

**1. Gaze as Textual Prompt (GazeT)**

- **Function**: Encodes per-frame gaze coordinates as normalized $(x, y)$ text coordinates, concatenated directly into the text prompt.
- **Mechanism**: Exploits the strong text comprehension capability of the underlying LLM in MLLMs, converting spatial gaze information into linguistic descriptions and avoiding scale mismatches between vision and gaze.
- **Design Motivation**: All gaze data is normalized to the $[0,1]$ range to abstract away resolution differences across videos. This strategy performs best on temporal intent understanding, as MLLMs are more adept at reasoning about temporal relations from text than from visually annotated frames.

**2. Gaze as Visual Prompt (GazeV)**

- **Function**: Draws a 25-pixel red circle at the gaze coordinate in each frame, accompanied by a text prompt indicating that "the red circle denotes a high-attention region."
- **Mechanism**: Embeds gaze signals into the visual modality, simulating human visual attention patterns and guiding the model to focus on salient regions within frames.
- **Design Motivation**: Communicates spatial attention information more intuitively through the visual channel, leading the model to attend to regions of interest in a manner analogous to human gaze.

**3. Sequential Gaze Salience Map (GazeS)**

- **Function**: Cumulatively accumulates gaze salience maps along the temporal axis of the video, merging multi-frame gaze trajectories into a single heatmap, with regions revisited more frequently receiving higher intensity.
- **Mechanism**: Models the temporal correlation of gaze by progressively reinforcing gaze salience, encoding both spatial and temporal intent simultaneously.
- **Design Motivation**: Achieves the best overall performance (especially for spatial intent) because it jointly captures "where the user looks" and "how long / how often the user looks there."

### Loss & Training

- **Fine-tuning Strategy**: LoRA is applied for lightweight fine-tuning of Qwen2.5-VL-7B (via LLaMA-Factory); approximately 500 gaze-guided QA pairs are sufficient to yield substantial improvements.
- **Cross-dataset Transfer**: Fine-tuning on EGTEA and testing on Ego4D+EgoExo raises average accuracy from 54.0% to 69.5%; the reverse transfer is equally effective (52.0% → 65.4%), demonstrating cross-domain generalizability of gaze-guided understanding.
- **Evaluation Metric**: 5-way multiple-choice accuracy (random baseline: 20%), evaluated separately across spatial, temporal, and causal dimensions.

## Key Experimental Results

### Main Results

Performance of various MLLMs on EgoGazeVQA (best GazeS strategy vs. no gaze signal):

| Model | Strategy | Spatial | Temporal | Causal | Avg |
|---|---|---|---|---|---|
| Human | — | 80.7 | 75.6 | 95.2 | 83.8 |
| Qwen2.5-VL-72B | w/o | 57.1 | 45.2 | 79.3 | 60.5 |
| Qwen2.5-VL-72B | GazeS | **64.3** | **50.3** | **84.3** | **66.3** |
| InternVL2.5-8B | w/o | 50.4 | 51.1 | 73.3 | 58.3 |
| InternVL2.5-8B | GazeV | **55.0** | 50.6 | **76.2** | **60.6** |
| MiniCPM-o 2.6 | w/o | 40.8 | 34.5 | 32.5 | 35.9 |
| MiniCPM-o 2.6 | GazeS | **43.2** | **43.5** | **74.4** | **53.7** |

### Ablation Study

Cross-dataset transfer via LoRA fine-tuning (Qwen2.5-VL-7B):

| Train Set | Test Set | Spatial | Temporal | Causal | Avg |
|---|---|---|---|---|---|
| — (Zero-shot) | Ego4D+EgoExo | 40.4 | 43.3 | 78.2 | 54.0 |
| EGTEA (~500 pairs) | Ego4D+EgoExo | **67.7** | **56.5** | **84.4** | **69.5** |
| — (Zero-shot) | EGTEA | 45.7 | 32.9 | 77.3 | 52.0 |
| Ego4D+EgoExo | EGTEA | **66.5** | **47.0** | **82.8** | **65.4** |

### Key Findings

1. **GazeS is the overall best strategy**: It yields the largest gains on spatial intent (Qwen2.5-VL-72B: +7.2 pp), whereas GazeT performs better on temporal intent, indicating that MLLMs are more capable of reasoning about temporal relations from text than from visually annotated frames.
2. **Model scale determines gaze utilization efficiency**: Larger models (72B) benefit substantially more from gaze signals than smaller ones (7B), whose limited capacity hinders full interpretation of gaze-related instructions.
3. **Limits of MLLM self-estimated gaze**: When Qwen2.5-VL-72B is used to predict gaze points for self-prompting, lower MSE correlates with larger gains (MSE = 0.038 yields +2.8 pp on EgoExo), but inaccurate estimates can even hurt performance.
4. **Scene complexity has a strong effect**: Kitchen scenes (many cluttered targets) yield the lowest accuracy, while medical and garage scenes (fewer targets) perform relatively better; multi-person social interactions substantially lower performance.
5. **A small amount of gaze-annotated data suffices**: Only ~500 LoRA fine-tuning pairs improve spatial reasoning by approximately 27 pp, with cross-dataset generalizability.

## Highlights & Insights

- **First gaze-guided egocentric video QA benchmark**: Fills the gap of gaze signals in egocentric VQA research, comprising 913 videos and 1,757 QA pairs across diverse scenes and activities.
- **Complementarity of three prompting strategies**: GazeS is optimal for spatial intent and GazeT for temporal intent, revealing MLLM preferences for different modalities of information.
- **Sophisticated distractor design**: Distractors include counter-causal options, spatially proximate traps, social influence distractors, and salient but irrelevant objects.
- **Validates the efficient "small gaze data + LoRA" pathway**: Deployment-friendly and does not require large-scale gaze annotation.

## Limitations & Future Work

1. **Limited video duration**: Clips average 30–60 seconds; the effectiveness of gaze guidance in longer videos remains unvalidated.
2. **Gaze acquisition requires hardware**: Real-world deployment depends on eye-tracking devices, constraining applicability; inaccurate MLLM-estimated gaze can even degrade performance.
3. **Limited scene coverage**: Kitchen and cooking activities constitute approximately 60% of the dataset, potentially introducing distributional bias.
4. **Fusion with other sensors unexplored**: EgoExo4D provides IMU, audio, and other multimodal streams that could be further incorporated.
5. **Only multiple-choice format tested**: The effect of gaze signals on open-ended question answering remains unexamined.

## Related Work & Insights

- **GazeGPT (Konrad et al. 2024)**: Demonstrates that gaze data can improve MLLM UI design, but lacks a standardized benchmark evaluation → this paper fills that gap.
- **IntentQA (Li et al. 2023)**: Addresses intent reasoning but uses third-person video → shifting to egocentric video with gaze provides more natural intent signals.
- **EgoSchema / EgoMemoria**: Egocentric long-video reasoning benchmarks without gaze signals → gaze can serve as a strong prior to substantially reduce reasoning difficulty.
- **Insight**: Gaze signals are analogous to referring expressions in visual grounding; future work could explore incorporating gaze as a form of visual prompt tuning.

## Rating

| Dimension | Score | Rationale |
|---|---|---|
| Novelty | ⭐⭐⭐⭐ | First work to introduce gaze signals into MLLM video understanding benchmarks; distinctive and practically valuable perspective. |
| Technical Depth | ⭐⭐⭐ | The three prompting strategies are well-motivated but not technically demanding; the LoRA fine-tuning experiments are relatively standard. |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Comprehensive evaluation across 7 models × 3 strategies × multiple scenes and activities, with in-depth analysis of fine-tuning and self-estimated gaze. |
| Practical Value | ⭐⭐⭐⭐ | Directly relevant to AR glasses and egocentric AI assistants; the "small data + LoRA" pathway is highly actionable. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video](rtv-bench_benchmarking_mllm_continuous_perception_understanding_and_reasoning_th.md)
- [\[AAAI 2026\] Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](../../AAAI2026/multimodal_vlm/plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)
- [\[ICCV 2025\] OrderChain: Towards General Instruct-Tuning for Stimulating the Ordinal Understanding Ability of MLLM](../../ICCV2025/multimodal_vlm/orderchain_towards_general_instruct-tuning_for_stimulating_the_ordinal_understan.md)
- [\[ICCV 2025\] Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models](../../ICCV2025/multimodal_vlm/physics_context_builders_a_modular_framework_for_physical_reasoning_in_vision-la.md)
- [\[ICCV 2025\] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning](../../ICCV2025/multimodal_vlm/sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)

</div>

<!-- RELATED:END -->
