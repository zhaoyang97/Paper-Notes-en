---
title: >-
  [Paper Note] Flex-Judge: Text-Only Reasoning Unleashes Zero-Shot Multimodal Evaluators
description: >-
  [NeurIPS 2025][Image Generation][LLM-as-a-Judge] This paper proposes Flex-Judge, which fine-tunes a multimodal large language model on only 1K text-only reasoning samples to achieve zero-shot generalization across image, video, audio, and molecular evaluation tasks, matching or surpassing commercial APIs such as GPT-4o and specialized evaluators trained on large-scale annotated data.
tags:
  - NeurIPS 2025
  - Image Generation
  - LLM-as-a-Judge
  - Multimodal Evaluation
  - Reasoning-Guided Training
  - Cross-Modal Transfer
  - Zero-Shot Generalization
  - Preference Optimization
date: 2026-05-08
content_hash: 5ae82e25ad6be89d
---

# Flex-Judge: Text-Only Reasoning Unleashes Zero-Shot Multimodal Evaluators

**Conference**: NeurIPS 2025
**arXiv**: [2505.18601](https://arxiv.org/abs/2505.18601)
**Code**: [https://flex-judge.github.io](https://flex-judge.github.io)
**Area**: Image Generation
**Keywords**: LLM-as-a-Judge, Multimodal Evaluation, Reasoning-Guided Training, Cross-Modal Transfer, Zero-Shot Generalization, Preference Optimization

## TL;DR

This paper proposes Flex-Judge, which fine-tunes a multimodal large language model on only 1K text-only reasoning samples to achieve zero-shot generalization across image, video, audio, and molecular evaluation tasks, matching or surpassing commercial APIs such as GPT-4o and specialized evaluators trained on large-scale annotated data.

## Background & Motivation

### Core Problem

As generative models expand from text to multimodal domains including images, video, and audio, efficiently evaluating generation quality has become a critical challenge. Traditional human evaluation is costly and difficult to scale. While the LLM-as-a-Judge paradigm reduces cost, it faces two fundamental challenges:

**Uncontrollability of commercial APIs**: Closed-source models such as GPT-4V suffer from transparency, controllability, and cost issues; silent model updates can substantially degrade evaluation quality (e.g., GPT-4V has exhibited significant performance drops on the MLLM-as-a-Judge benchmark).

**Insufficient multimodal coverage**: Existing judge models are largely limited to text or vision-language tasks, with poor support for modalities such as audio, molecular structures, heatmaps, and 3D point clouds, while constructing modality-specific annotated data is extremely difficult.

### Core Hypothesis

The authors draw inspiration from cross-lingual transfer in multilingual LLMs: fine-tuning on a downstream task in one language can improve performance in other languages. By analogy, if a model has learned unified cross-modal representations, fine-tuning on a single modality (text) may enable generalization to other modalities. The key intuition is that **structured textual reasoning explanations inherently encode generalizable decision patterns**, enabling the model to transfer the logic of "why A is preferred over B" to multimodal evaluation scenarios.

## Method

### Overall Architecture

The core mechanism of Flex-Judge is remarkably concise: **Think Once, Judge Anywhere**.

1. **Data Generation**: JudgeLRM-7B (a specially trained reasoning-oriented judge LM) is used to generate structured reasoning annotations enclosed in `<think></think>` tags.
2. **Quality Filtering**: Only 1K high-quality text-only reasoning samples are selected from the JudgeLM-100K dataset as the training set.
3. **MLLM Fine-Tuning**: Qwen2.5-VL-7B or Qwen2.5-Omni-7B is fine-tuned on these 1K text samples, yielding Flex-VL-7B and Flex-Omni-7B.
4. **Zero-Shot Inference**: At inference time, multimodal inputs are directly provided; the model performs evaluation without any additional fine-tuning.

### Key Designs: Data Curation Strategy

Training data curation is central to Flex-Judge's success. The authors design the pipeline along four dimensions:

- **Quality and Difficulty Filtering**: Samples where JudgeLRM-7B's predicted scores agree with GPT-4o annotations are retained; samples with longer reasoning chains are prioritized (longer reasoning implies higher difficulty), consistent with findings in the literature that high-quality, high-difficulty samples improve sample efficiency.
- **Sample Count Control**: Excessive training samples lead to catastrophic forgetting—language-side judging ability improves while multimodal comprehension degrades. Experiments show that 1K is the optimal sweet spot.
- **On-Policy Sampling**: Training data is generated using low-temperature decoding, since JudgeLRM-7B and Flex-Judge share the same LLM backbone; low-loss on-policy samples help prevent catastrophic forgetting.
- **Format Diversity**: JudgeLRM-7B's original 1–10 pairwise scoring format is post-processed into multiple formats supporting single-score rating, pairwise comparison, and 1–5/1–10 scales, improving generalization to diverse evaluation instructions.

### Loss & Training

Standard supervised fine-tuning (SFT) loss is applied, optimizing the language modeling objective of the MLLM on text-only reasoning annotation data. Only the language model component is updated during training; modality encoders are kept frozen.

### Inference-Time Scaling

Flex-Judge supports two inference-time scaling strategies to further improve evaluation quality:

- **Majority Voting**: Multiple samples are drawn and the majority judgment is taken, consistently improving performance on the reasoning subset of VL-RewardBench.
- **Budget Forcing**: Injecting the "Wait" keyword triggers deeper reasoning, yielding stable gains on score-based evaluation.

## Key Experimental Results

### Main Results: Image Understanding Evaluation (MLLM-as-a-Judge)

| Model | Training-Free? | Score ↑ | Pair w. Tie ↑ | Pair w.o. Tie ↑ | Batch ↓ |
|------|---------|---------|---------------|-----------------|---------|
| GPT-4V | ✗ | 0.424 | 0.538 | 0.717 | 0.361 |
| Gemini-2.5-Pro | ✗ | 0.390 | 0.556 | 0.668 | 0.512 |
| LLaVA-Critic-7B (113K samples) | ✗ | 0.314 | 0.556 | 0.689 | 0.565 |
| Qwen2.5-VL-7B (baseline) | ✓ | 0.165 | 0.423 | 0.425 | 0.585 |
| **Flex-VL-7B (1K samples)** | **✓** | **0.332** | **0.538** | **0.655** | **0.426** |
| **Flex-Omni-7B (1K samples)** | **✓** | **0.306** | **0.532** | **0.650** | **0.425** |

### VL-RewardBench and MJ-Bench

| Model | VL-RewardBench Overall ↑ | MJ-Bench Safety w. Tie ↑ |
|------|--------------------------|--------------------------|
| GPT-4o | 65.8 | 35.3 |
| LLaVA-Critic-7B | 43.7 | — |
| Flex-VL-7B | **48.60** | **57.51** |
| Flex-Omni-7B | **48.02** | **47.69** |

Flex-VL-7B surpasses all open-source models at the 72B scale (e.g., Qwen2-VL-72B at 39.5) on VL-RewardBench, and substantially outperforms GPT-4o on MJ-Bench safety detection.

### Video Generation Evaluation (GenAI-Bench)

| Model | Image Gen. | Image Edit. | Video Gen. | Overall |
|------|---------|---------|---------|------|
| GPT-4o | 45.59 | 53.54 | 48.46 | 49.20 |
| Flex-VL-7B + Majority Voting | **46.34** | **54.19** | **47.34** | **49.29** |

Flex-VL-7B with majority voting achieves overall performance exceeding GPT-4o.

### Ablation Study

| Ablation Dimension | Result |
|---------|------|
| Reasoning order (reason-first vs. answer-first) | Reason-first consistently outperforms answer-first; Score improves from 0.290 to 0.332 |
| Data quality vs. modality alignment | High-quality text data > low-quality vision-language data (VL-Reward: 48.60 vs. 43.84) |
| Inference-time scaling | Majority voting consistently improves pairwise comparison; budget forcing yields stable modest gains on score-based evaluation |
| Training sample count | 1K is optimal; more samples degrade multimodal performance |

### Molecular Domain Extension (Flex-Mol-LLaMA)

| Method | Default | CoT | w/ Task Info |
|------|---------|-----|-------------|
| Mol-LLaMA baseline | 63.55 | 64.37 | 72.48 |
| + Best-of-16 Sampling | 68.85 | 69.83 | 77.49 |
| + DPO Preference Optimization | **76.41** | **75.92** | **80.10** |

On the PAMPA permeability prediction task, DPO-trained Mol-LLaMA achieves 80.10%, substantially surpassing the previous state of the art.

### Key Findings

1. **Remarkable data efficiency**: 1K text reasoning samples vs. 113K vision-language pairs for LLaVA-Critic vs. 150K for Prometheus-Vision — Flex-Judge achieves comparable or superior performance with 1/100 of the data.
2. **Reasoning length correlates with task difficulty**: Harder tasks elicit longer reasoning chains, indicating that reasoning ability is critical for accurate evaluation.
3. **Cross-modal transfer is effective**: Text-only reasoning training transfers effectively to image, video, audio, and even molecular modalities.

## Highlights & Insights

1. **A minimalist yet highly effective paradigm**: The central finding — that large-scale multimodal annotation is unnecessary for training effective multimodal judges, and that a small amount of high-quality reasoning data suffices — carries paradigm-shifting implications for the LLM-as-a-Judge field.
2. **Empirical validation of cross-modal reasoning transfer**: This work provides the first systematic evidence that textual reasoning ability can transfer to evaluation tasks spanning visual, audio, and molecular modalities.
3. **Unique advantage of inference-time scaling**: Unlike prior work, Flex-Judge benefits from inference-time scaling strategies such as majority voting, as reasoning-oriented training produces more diverse reasoning trajectories.
4. **Pioneering application to the molecular domain**: Flex-Mol-LLaMA is proposed as the first judge model for the molecular modality, demonstrating practical value in domains lacking dedicated reward models.
5. **Data quality > modality alignment**: Experiments clearly show that high-quality text data outperforms low-quality modality-aligned data, challenging the conventional assumption that modality matching is necessary.

## Limitations & Future Work

1. **Only 7B-scale models are evaluated**: Performance on larger or smaller models remains unexplored, and generalizability across scales is unverified.
2. **Remaining gap in audio evaluation**: Flex-Omni-7B still lags considerably behind single-task fine-tuned specialist models on audio quality assessment.
3. **Narrow molecular task coverage**: Validation is limited to the PAMPA permeability prediction task, without broader coverage of molecular property prediction tasks.
4. **Dependence on JudgeLRM for data generation**: Seed data quality is heavily dependent on JudgeLRM-7B; biases inherent to JudgeLRM may propagate to Flex-Judge.
5. **More challenging generation evaluation scenarios unexplored**: Complex settings such as long-video understanding and multi-turn dialogue evaluation are not addressed.

## Related Work & Insights

### Positioning Relative to Prior Work

- **LLM-as-a-Judge**: Builds upon evaluation paradigms such as MT-Bench and AlpacaEval, extending them from text to multimodal settings.
- **Multimodal evaluators**: Compared to LLaVA-Critic (trained on 113K vision-language pairs) and Prometheus-Vision (150K samples), competitive performance is achieved with far less data.
- **Reasoning-guided supervision**: Complements reasoning-oriented judge models such as JudgeLRM by converting reasoning supervision signals into cross-modal capabilities.
- **Cross-lingual/cross-modal transfer**: Draws on cross-lingual transfer mechanisms from multilingual LLMs, providing the first validation of cross-modal transfer in evaluation tasks.

### Insights

- **Toward universal evaluators**: Could this paradigm be applied to additional "data-desert" domains such as remote sensing or medical image evaluation?
- **Reasoning as a general alignment signal**: High-quality reasoning annotations may constitute a more efficient source of alignment signal than preference data.
- **Extreme data efficiency**: The finding that 1K samples suffice suggests that MLLMs already encode substantial latent judging priors that can be activated with minimal supervision.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Fine-tuning on text-only reasoning to achieve cross-modal evaluation is a concise yet profound idea.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers four major modalities (image understanding/generation, video, audio, molecular) across seven benchmarks with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with rich figures and tables; motivation is clearly articulated.
- Value: ⭐⭐⭐⭐⭐ — Provides a universal multimodal evaluation solution at minimal cost, with significant practical value for resource-constrained domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Robust Zero-Shot Reinforcement Learning](towards_robust_zero-shot_reinforcement_learning.md)
- [\[NeurIPS 2025\] Evaluating the Evaluators: Metrics for Compositional Text-to-Image Generation](evaluating_the_evaluators_metrics_for_compositional_text-to-image_generation.md)
- [\[NeurIPS 2025\] Semantic Surgery: Zero-Shot Concept Erasure in Diffusion Models](semantic_surgery_zero-shot_concept_erasure_in_diffusion_models.md)
- [\[ICCV 2025\] AnyPortal: Zero-Shot Consistent Video Background Replacement](../../ICCV2025/image_generation/anyportal_zero-shot_consistent_video_background_replacement.md)
- [\[ICCV 2025\] DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability](../../ICCV2025/image_generation/dynamicid_zero-shot_multi-id_image_personalization_with_flexible_facial_editabil.md)

</div>

<!-- RELATED:END -->
