---
title: >-
  [Paper Note] INST-IT: Boosting Instance Understanding via Explicit Visual Prompt Instruction Tuning
description: >-
  [NeurIPS 2025][Video Understanding][Instance-level Understanding] This work presents the complete Inst-IT framework: a GPT-4o-assisted automatic annotation pipeline for generating instance-level fine-grained data…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "Instance-level Understanding"
  - "Visual Prompting"
  - "Instruction Tuning"
  - "Large Multimodal Models"
  - "Spatiotemporal Understanding"
date: 2026-05-08
content_hash: 3f834ab5f6f1a8e0
---

# INST-IT: Boosting Instance Understanding via Explicit Visual Prompt Instruction Tuning

**Conference**: NeurIPS 2025
**arXiv**: [2412.03565](https://arxiv.org/abs/2412.03565)  
**Code**: [GitHub](https://github.com/inst-it/inst-it) | [HuggingFace](https://huggingface.co/Inst-IT)  
**Area**: Video Understanding / Multimodal Learning
**Keywords**: Instance-level Understanding, Visual Prompting, Instruction Tuning, Large Multimodal Models, Spatiotemporal Understanding

## TL;DR

This work presents the complete Inst-IT framework: a GPT-4o-assisted automatic annotation pipeline for generating instance-level fine-grained data, an Inst-IT Bench evaluation benchmark, a 335K QA-pair instruction tuning dataset, and a continual fine-tuning paradigm that effectively enhances instance-level understanding in LMMs while also improving general image and video comprehension.

## Background & Motivation

Large multimodal models (LMMs) have achieved remarkable progress in holistic image and video understanding, yet they remain significantly limited in **instance-level understanding**:

**What is instance-level understanding**: Recognizing the attributes, behaviors, relationships, and temporal changes of specific instances (e.g., a particular person or object) within images or videos.

**Strong real-world demand**: Users typically focus on specific targets in a scene rather than the overall context — e.g., "What is the person in the red shirt doing?"

**Limitations of current models**: Existing LMMs perform well at global description but frequently confuse or omit details when asked to focus on a particular instance.

A key observation motivates this work: **state-of-the-art LMMs show substantially improved instance understanding when provided with explicit visual cues (e.g., bounding boxes, arrows, ID labels)**. This suggests that models already possess the latent capacity for instance-level understanding, but lack the appropriate training data to activate it.

Accordingly, the core idea behind Inst-IT is to **construct large-scale instance-level visual prompt instruction tuning data**, guiding models to learn instance-level understanding through explicit visual marking.

## Method

### Overall Architecture

Inst-IT consists of three components:

1. **Inst-IT Bench**: An evaluation benchmark for diagnosing instance-level understanding in LMMs.
2. **Inst-IT Dataset**: A large-scale instruction tuning dataset.
3. **Continual Instruction Tuning Paradigm**: An effective training strategy.

### Key Designs

**1. Automatic Annotation Pipeline**

GPT-4o is employed as the annotation engine to process videos frame by frame:

- **Instance annotation**: Explicit visual markers (e.g., `[1]`, `[2]`, `[3]`) are applied to label each instance in the image.
- **Frame-level description**: Three-level descriptions are generated per frame — (a) independent description of each instance, (b) holistic scene description, and (c) temporal changes relative to the previous frame.
- **Video-level description**: All frame-level annotations are aggregated into a chronologically organized overall video description.
- **QA generation**: Instance-centric open-ended question-answer pairs are generated from the annotations.

**2. Inst-IT Bench (Evaluation Benchmark)**

- **Scale**: ~1,000 image QA pairs + ~1,000 video QA pairs.
- **Evaluation dimensions**: Image branch (instance attributes, instance relations) + Video branch (temporal tracking, action understanding).
- **Format**: Supports both open-ended and multiple-choice questions.
- **Distinctive design**: Uses `[ID]` format to reference instances and `<timestamp>` to reference temporal moments, evaluating fine-grained spatiotemporal understanding.

**3. Inst-IT Dataset (Fine-tuning Dataset)**

- 21K videos + 51K images
- 21K video-level descriptions
- 207K frame-level descriptions (51K image frames + 156K video frames)
- 335K open-ended QA pairs
- Currently the largest instance-level visual prompt annotation dataset.

**4. Continual Instruction Tuning Paradigm**

- Inst-IT Dataset is mixed with the original general-purpose instruction tuning data.
- A continual training strategy is adopted rather than training from scratch.
- Adding only a modest volume of instance-level data (~155K) yields substantial gains in instance-level understanding without degrading general capabilities.

### Loss & Training

- Standard autoregressive language modeling loss.
- Built upon the LLaVA-Next framework with two-stage training.
- Mixing ratio: original LLaVA-Next data (~765K) + Inst-IT data (~155K) = ~920K total.

## Key Experimental Results

### Main Results: Inst-IT Bench Evaluation

| Model | Backbone | Image OE | Image MC | Video OE | Video MC |
|-------|----------|----------|----------|----------|----------|
| Random Guess | — | — | 25.0 | — | 25.0 |
| GPT-4o | — | 74.1 | 84.8 | 65.5 | 81.0 |
| Gemini-1.5-pro | — | 69.9 | 79.7 | 61.4 | 76.7 |
| LLaVA-1.5 | Vicuna-7B | 41.6 | 32.1 | — | — |
| LLaVA-Next | Vicuna-7B | 46.0 | 42.4 | — | — |
| LLaVA-OV | Qwen2-7B | 48.0 | 71.7 | 33.2 | 45.6 |
| InternVL2 | InternLM2.5-7B | 58.6 | 66.5 | 39.8 | 45.5 |
| Qwen2-VL | Qwen2-7B | 48.3 | 64.9 | 38.2 | 59.4 |
| **LLaVA-Next-Inst-IT** | **Vicuna-7B** | **68.6** | **63.0** | **49.3** | **42.1** |
| **LLaVA-Next-Inst-IT** | **Qwen2-7B** | **67.9** | **75.3** | **45.7** | **53.3** |

Key findings:
- After Inst-IT fine-tuning, LLaVA-Next (Vicuna-7B) improves from 46.0 to 68.6 on Image OE (+22.6), approaching GPT-4o performance.
- On Video OE, the score improves from 25.8 to 49.3 (+23.5), a substantial gain.

### Performance on General Benchmarks

Inst-IT fine-tuning not only improves instance-level understanding but also enhances general image and video comprehension:

| Benchmark | LLaVA-Next (Original) | +Inst-IT | Gain |
|-----------|----------------------|----------|------|
| AI2D | 65.2 | 68.7 | +3.5 |
| TextVQA | 63.8 | 65.1 | +1.3 |
| EgoSchema | 42.1 | 48.5 | +6.4 |
| MVBench | 56.3 | 60.8 | +4.5 |

### Ablation Study

**Importance of data composition**:

| Data Configuration | Inst-IT Bench (MC) | AI2D | EgoSchema |
|-------------------|-------------------|------|-----------|
| LLaVA-Next baseline | 42.4 | 65.2 | 42.1 |
| + Image instance data only | 56.8 | 67.1 | 43.5 |
| + Video instance data only | 48.2 | 65.8 | 47.2 |
| + Image + Video instance data | 63.0 | 68.7 | 48.5 |

The combination of image and video instance data yields the best overall performance; video data contributes more to temporal understanding tasks such as EgoSchema.

**Effect of visual prompt type**:

| Visual Prompt Type | Inst-IT Bench (Image MC) | Inst-IT Bench (Video MC) |
|-------------------|--------------------------|--------------------------|
| No visual prompt | 42.4 | 24.8 |
| Bounding Box | 55.1 | 35.2 |
| ID-labeled markers ([ID]) | 63.0 | 42.1 |

Explicit `[ID]` label markers outperform simple bounding boxes, as the ID system naturally enables cross-frame tracking of the same instance.

### Key Findings

1. **Instance understanding is a notable weakness of LMMs**: Even GPT-4o achieves only 74–85 on Inst-IT Bench, far from perfect.
2. **Explicit visual cues are highly effective**: Introducing ID labels leads to a dramatic leap in instance-level understanding.
3. **Instance data enhances general capabilities**: Instance-level training does not conflict with general-purpose capabilities; rather, the two are mutually reinforcing.
4. **High efficiency with limited data**: Only ~155K instance-level samples are sufficient to produce substantial improvements.

## Highlights & Insights

1. **Complete ecosystem**: Bench + Dataset + Training constitutes a closed loop from evaluation to data to training.
2. **Automated annotation**: GPT-4o is leveraged to automatically generate high-quality instance-level annotations at scale.
3. **General capability enhancement**: This finding is counterintuitive — fine-grained instance-level training improves holistic understanding, suggesting that instance-level understanding is a foundational capability.
4. **Simple yet powerful visual prompt design**: The `[ID]` labeling system is straightforward but effective, with native support for cross-frame instance tracking.

## Limitations & Future Work

1. **Annotation cost**: The pipeline relies on GPT-4o for annotation, which incurs high cost and may introduce GPT-4o-specific biases.
2. **Limited model scale**: Validation is conducted only on 7B-scale models; effectiveness on larger models remains to be confirmed.
3. **Dependency on instance detection**: The annotation pipeline requires existing detection/segmentation models to provide instance locations; detection failures cascade downstream.
4. **Open-world generalization**: Current data covers a limited range of scene types; generalization to rare scenarios is unclear.
5. **Training code not open-sourced**: Only the evaluation toolkit and model weights are currently released.

## Related Work & Insights

- **ViP-LLaVA / SoM-LLaVA**: Early works on visual prompt LMMs, but lacking large-scale instance-level data.
- **LLaVA-OneVision**: A representative multi-stage instruction tuning approach that inspired the continual fine-tuning strategy in Inst-IT.
- **VideoGLaMM**: Pixel-level video grounding focusing on finer-grained localization.
- **GPT-4o**: Serves both as the annotation tool and as the practical upper bound, demonstrating the potential of well-trained instance understanding.
- Insight: **Explicit visual marking is a low-cost, high-efficiency means of "unlocking" latent model capabilities**, a principle that generalizes to other fine-grained tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ (first large-scale instance-level visual prompt fine-tuning framework)
- Technical Depth: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐ (data and models open-sourced)
- Writing Quality: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs](tempsampr1_effective_temporal_sampling_with_reinforcement_fi.md)
- [\[CVPR 2026\] VRR-QA: Visual Relational Reasoning in Videos Beyond Explicit Cues](../../CVPR2026/video_understanding/vrr-qa_visual_relational_reasoning_in_videos_beyond_explicit_cues.md)
- [\[ICCV 2025\] RainbowPrompt: Diversity-Enhanced Prompt-Evolving for Continual Learning](../../ICCV2025/video_understanding/rainbowprompt_diversity-enhanced_prompt-evolving_for_continual_learning.md)
- [\[NeurIPS 2025\] PreFM: Online Audio-Visual Event Parsing via Predictive Future Modeling](prefm_online_audio-visual_event_parsing_via_predictive_future_modeling.md)
- [\[NeurIPS 2025\] PixFoundation 2.0: Do Video Multi-Modal LLMs Use Motion in Visual Grounding?](pixfoundation_20_do_video_multi-modal_llms_use_motion_in_visual_grounding.md)

</div>

<!-- RELATED:END -->
