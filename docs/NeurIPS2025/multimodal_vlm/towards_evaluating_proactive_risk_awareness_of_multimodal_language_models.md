---
title: >-
  [Paper Note] Towards Evaluating Proactive Risk Awareness of Multimodal Language Models
description: >-
  [NeurIPS 2025][Multimodal VLM][Proactive safety] This paper introduces PaSBench, a benchmark for evaluating the proactive risk awareness of multimodal language models — requiring models to autonomously observe environmen…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Proactive safety"
  - "risk detection"
  - "LLM evaluation"
  - "benchmark"
  - "anticipatory reasoning"
date: 2026-05-08
content_hash: 2f5387678fd6cfe8
---

# Towards Evaluating Proactive Risk Awareness of Multimodal Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.17455](https://arxiv.org/abs/2505.17455)
**Code**: [HuggingFace](https://huggingface.co/datasets/Youliang/PaSBench)
**Area**: Multimodal VLM
**Keywords**: Proactive safety, risk detection, LLM evaluation, benchmark, anticipatory reasoning

## TL;DR

This paper introduces PaSBench, a benchmark for evaluating the proactive risk awareness of multimodal language models — requiring models to autonomously observe environments and issue safety warnings without any user query. An evaluation of 36 models reveals that the strongest model (Gemini-2.5-pro) achieves only 71% accuracy, with 45% of risks failing to be detected consistently. The core bottleneck is identified as unstable proactive reasoning rather than a lack of safety knowledge.

## Background & Motivation

In everyday life, people frequently face risks due to insufficient safety knowledge or awareness. An ideal AI safety system should be **proactive** — automatically observing the environment and user behavior, detecting potential hazards, and issuing timely warnings without waiting for the user to ask.

Existing AI safety research suffers from a fundamental paradigm limitation:

**Reactive Paradigm**: Existing safety benchmarks (SafeText, HealthBench, LabSafetyBench, etc.) presuppose that users have already recognized a risk and are actively seeking guidance, making them essentially QA-style evaluations.

**Focus on AI-Generated Harm**: A large body of work studies whether LLMs produce harmful outputs (toxicity, bias), rather than whether LLMs can protect human safety.

**Absence of Proactive Capability Evaluation**: No benchmark specifically assesses models' ability to proactively detect risks.

This paper defines a novel task — **proactive risk detection**: given a sequence of observations (text logs or image sequences), the model must determine, without any user query, whether a person is currently in or about to encounter an unsafe situation, and proactively issue a warning. This is far more realistic than conventional safety evaluations.

## Method

### Overall Architecture

PaSBench is constructed following a pipeline of knowledge collection → sample generation → quality control, yielding two evaluation sets: a text log set (288 instances) and an image set (128 instances), spanning five major safety domains.

### Key Designs

1. **Knowledge Collection and Filtering**

   Safety knowledge is collected from Chinese safety education books and official government websites, governed by five strict principles:
   - **User specificity**: Focus on risks arising from individual behavior (e.g., consuming wild mushrooms), excluding group- or societal-level risks.
   - **Risk determinism**: A clear causal link must exist between the risk and the harm.
   - **Knowledge currency**: Only safety knowledge that remains currently valid is retained.
   - **Consequence severity**: The risk must result in significant harm.
   - **Verifiability**: Uncertain knowledge must be verifiable via Google within five minutes.

   Three annotators performed cross-validation, filtering 288 knowledge entries from an initial pool of 495.

2. **Image Observation Generation**

   For each knowledge entry, GPT-4o generates 1–4 text-to-image prompt drafts; these are manually refined, and GPT-4o-image then generates image sequences progressively (each image conditioned on the previous to maintain consistency). Each image undergoes manual inspection for consistency, realism, and semantic accuracy, with up to 10 retries permitted. A final set of 128 image samples is collected.

3. **Text Log Observation Generation**

   Character profiles (name, gender, residence) are randomly generated; occupations and hobbies are produced by DeepSeek-R1 conditioned on the safety knowledge (requiring relevance to the risk type), followed by full log generation. The log format is: [time]...[location]...[environmental observation]...[behavioral observation]. A key constraint is that observations must conclude **before** the safety incident occurs, so that model warnings carry genuine preventive value.

4. **Evaluation Protocol**

   Each sample is evaluated $N=16$ times per model, with three reported metrics:
   - **Accuracy (Average-of-N)**: Proportion of runs in which the risk is correctly identified and explained.
   - **Potential (Best-of-N)**: Proportion of samples for which at least one run is correct.
   - **Robustness (Worst-of-N)**: Proportion of samples for which all 16 runs are correct.

   Correctness requires: (1) *Identification* — warning the user to cease the dangerous behavior; and (2) *Explanation* — providing a sound rationale for the risk. GPT-4.1 is used as the judge model (human-validated accuracy: 94.5%).

### Loss & Training

PaSBench is a pure evaluation benchmark; no training is involved.

## Key Experimental Results

### Main Results: Risk Detection Rate

| Model | Image Accuracy | Image Robustness | Text Accuracy | Text Robustness |
|-------|---------------|-----------------|--------------|----------------|
| Gemini-2.5-pro | 71% | 55% | 64% | ~45% |
| Gemini-2.0-pro | ~65% | ~45% | **Best** | ~50% |
| GPT-4.1 | ~60% | ~35% | ~58% | ~38% |
| Claude-3.5-sonnet | ~55% | ~30% | ~55% | ~35% |
| o1 | — | — | ~50% | ~25% |
| Qwen2.5-VL-7B | 23% | <5% | — | — |
| GPT-4.1-nano | ~25% | <5% | 20% | <5% |

### Knowledge vs. Detection Capability

| Evaluation Type | Gemini-2.5-pro | GPT-4.1-nano |
|----------------|---------------|--------------|
| Multiple-choice knowledge test | 87%–94.5% | >80% |
| Passive mode (given knowledge, asked if violated) | 93% (552/596 failed samples) | 75% (1047/1393) |
| Proactive mode (detection without prompting) | ~71% | ~25% |

### Ablation Study: Proactive vs. Passive Capability

| Model | # Proactive Failures | Passive Success Rate | Notes |
|-------|---------------------|---------------------|-------|
| Gemini-2.5-pro (image) | 596 | 552/596 (93%) | Failures predominantly due to insufficient proactive capability |
| GPT-4.1-nano (image) | 1393 | 1047/1393 (75%) | Same pattern |
| Gemini-2.5-pro (text) | 1646 | 1217/1646 (74%) | Knowledge present but application is unstable |

### Key Findings

- **Models possess safety knowledge but cannot apply it proactively in a stable manner**: multiple-choice accuracy exceeds 80%, yet proactive detection rates range from only 20% to 71%.
- **Reasoning models are not necessarily superior**: the non-reasoning model Gemini-2.0-pro achieves the best performance on the text set, while reasoning models such as o1 perform comparatively poorly.
- **Model scale matters**: larger models consistently outperform smaller counterparts across nearly all comparisons (pro > flash, sonnet > haiku).
- **High potential but low robustness**: GPT-4.1-nano can cover 91.4% of risks with 128 samples via Best-of-N sampling, yet achieves only ~30% accuracy in a single run.
- Detection rates on image and text sets are highly correlated (Pearson $r = 0.897$), indicating that the bottleneck is a modality-agnostic deficiency in proactive analytical capability.
- Observation sequence length does not significantly affect performance within the ranges tested.

## Highlights & Insights

- **A novel evaluation dimension is defined**: "proactive safety" represents a critically underexplored yet highly important AI capability.
- The diagnostic analysis is notably rigorous: by progressively decomposing performance across knowledge testing → passive detection → proactive detection, the bottleneck is precisely localized.
- The three-dimensional evaluation scheme (Average-of-N, Best-of-N, Worst-of-N) provides substantially richer insight than a single accuracy metric.
- The paper explicitly establishes that **the current bottleneck is not knowledge deficiency but rather the instability of proactive reasoning**, providing clear direction for future improvement.

## Limitations & Future Work

- The dataset is relatively small (416 instances); each image sample contains only 2–3 sub-images and each text sample 4–8 observation segments, which may be insufficient for evaluating long-sequence comprehension.
- Risk severity levels and false-positive rates are not considered; real-world deployment would require balancing warning frequency against user experience.
- Source materials are primarily drawn from Chinese safety education resources, potentially introducing cultural bias.
- Proposed improvement directions — including GRPO-based reinforcement learning to encourage proactive alerting and a "propose-then-verify" pipeline — warrant further exploration.
- Continuous data stream scenarios are not covered; practical deployment would require resolving how to segment streams for evaluation.

## Related Work & Insights

- Passive safety benchmarks such as SafeText and HealthBench serve as the primary points of contrast for this work.
- ProAgent and related proactive LLM studies focus on asking clarifying questions within dialogue; this paper extends the concept to safety monitoring scenarios.
- GRPO (the training method used in DeepSeek-R1) is proposed as a candidate approach for training proactive capabilities.
- Broader implication: proactive AI represents a core capability for next-generation safety assistants, with potential integration into wearable devices and smart home sensors for real-time protection.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Defines an entirely new "proactive safety" evaluation task, filling a significant gap in the field.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Large-scale evaluation across 36 models with in-depth diagnostic analysis; the relatively small dataset size is a limitation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is convincingly articulated; experimental analysis proceeds in a well-structured, layered manner with strong analytical insight.
- **Value**: ⭐⭐⭐⭐⭐ Carries directional significance for the AI safety field; the dataset is publicly released.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Evaluating Multimodal Large Language Models on Core Music Perception Tasks](evaluating_multimodal_large_language_models_on_core_music_perception_tasks.md)
- [\[NeurIPS 2025\] Adapting Vision-Language Models for Evaluating World Models](adapting_visionlanguage_models_for_evaluating_world_models.md)
- [\[NeurIPS 2025\] MME-VideoOCR: Evaluating OCR-Based Capabilities of Multimodal LLMs in Video Scenarios](mme-videoocr_evaluating_ocr-based_capabilities_of_multimodal_llms_in_video_scena.md)
- [\[NeurIPS 2025\] Advancing Compositional Awareness in CLIP with Efficient Fine-Tuning](advancing_compositional_awareness_in_clip_with_efficient_fin.md)
- [\[NeurIPS 2025\] Beyond Greedy Exits: Improved Early Exit Decisions for Risk Control and Reliability](beyond_greedy_exits_improved_early_exit_decisions_for_risk_control_and_reliabili.md)

</div>

<!-- RELATED:END -->
