---
title: >-
  [Paper Note] Towards Rationale-Answer Alignment of LVLMs via Self-Rationale Calibration
description: >-
  [ICML2025][Multimodal VLM][Large Vision-Language Models] This paper proposes the Self-Rationale Calibration (SRC) framework, which guides LVLMs to output intermediate reasoning processes through lightweight rationale fine-tuning. It leverages sentence-level beam search to generate diverse candidate responses, and employs a specially designed R-Scorer with a pairwise scoring strategy to select positive and negative rationale-answer pairs. Finally…
tags:
  - "ICML2025"
  - "Multimodal VLM"
  - "Large Vision-Language Models"
  - "Rationale Alignment"
  - "Preference Optimization"
  - "Self-Calibration"
  - "Rationale Generation"
date: 2026-05-08
content_hash: f6d32ac062d96811
---

# Towards Rationale-Answer Alignment of LVLMs via Self-Rationale Calibration

**Conference**: ICML2025  
**arXiv**: [2509.13919](https://arxiv.org/abs/2509.13919)  
**Authors**: Yuanchen Wu, Ke Yan, Shouhong Ding, Ziyin Zhou, Xiaoqiang Li  
**Code**: Not publicly available  
**Area**: Multimodal VLM  
**Keywords**: Large Vision-Language Models, Rationale Alignment, Preference Optimization, Self-Calibration, Rationale Generation  

## TL;DR

This paper proposes the Self-Rationale Calibration (SRC) framework, which guides LVLMs to output intermediate reasoning processes through lightweight rationale fine-tuning. It leverages sentence-level beam search to generate diverse candidate responses, and employs a specially designed R-Scorer with a pairwise scoring strategy to select positive and negative rationale-answer pairs. Finally, DPO-based preference alignment is used to iteratively calibrate the model's rationale-answer consistency, achieving significant improvements across multiple perception, reasoning, and generalization benchmarks.

## Background & Motivation

### Background
Large Vision-Language Models (LVLMs) have achieved remarkable progress in tasks such as Visual Question Answering (VQA), but alignment issues between visual and textual modalities (e.g., hallucination, factual errors) remain prominent. Current post-training strategies can be broadly categorized into two types:

**Supervised Fine-Tuning (SFT)**: Instruction tuning using additional annotated data.

**Preference Alignment (e.g., DPO)**: Re-aligning models by constructing positive/negative sample pairs to guide the model away from counterfactual descriptions.

Although these methods are effective for visual description tasks, they all **neglect the quality of the reasoning process (rationale)**—specifically, whether the model's final answer genuinely stems from a reasonable, factually-grounded reasoning chain.

### Key Observations
The authors conducted an in-depth analysis of LVLMs (e.g., LLaVA-1.5-7B) and discovered that:

- **Correct Answer $\neq$ Correct Reasoning**: Models frequently generate correct answers while the underlying reasoning processes contain severe errors.
- Three typical patterns of reasoning flaws:
    - **Counterfactual Reasoning**: The rationale contains descriptions inconsistent with the image.
    - **Insufficient Reasoning**: The reasoning process lacks support from key information.
    - **Unreasonable Reasoning**: The logical chain of reasoning is inherently flawed.
- Current instruction tuning heavily relies on short-answer datasets, lacking explicit supervision of the reasoning process. This causes models to establish spurious causal correlations from instruction to answer, rather than the correct path from visual perception to rationale to answer.

### Core Problem
> "Does a correct answer stem from a reasonable and factually grounded rationale?"

## Method

### Overall Architecture
The SRC framework consists of four core stages, which can be executed iteratively to continuously improve the model:

**Stage 1: Rationale Fine-Tuning**
- Augment selected VQA samples to construct Rationale-Answer Pairs (RAPs), which represent a complete response format containing both the reasoning process and the final answer.
- Use lightweight LoRA to fine-tune the LVLM, enabling the model to automatically output responses in a "reasoning-then-answering" format without explicit prompting.
- Core Value: Transforms the model from directly providing answers to first rendering rationales and then answering, establishing the prerequisite condition for subsequent evaluation and calibration.

**Stage 2: Diverse Candidate Generation**
- Utilize the rationale-fine-tuned model as the seed model.
- Perform **sentence-level beam search** in the output space for each visual instruction sample to retrieve diverse candidate responses.
- Key Design: Sentence-level instead of token-level search to generate candidate sets with more distinct reasoning paths.

**Stage 3: Pairwise Scoring with R-Scorer**
- Design a dedicated, lightweight LLM scoring model, **R-Scorer**.
- Adopt a **pairwise scoring strategy**: Pair candidates up for relative comparison rather than scoring them independently.
- Evaluation Dimensions: Rationale quality and factual consistency.
- Advantage of Pairwise Comparison: Since open-ended reasoning is hard to quantify absolutely, a relative judgment of "A is better than B" is much more reliable.
- Incorporate the LLM-as-judge paradigm to capture relative superiorities in reasoning among candidates.

**Stage 4: Confidence-Weighted Preference Construction and Alignment**
- Aggregate confidence scores from pairwise evaluations to identify the best and worst candidates.
- Confidence weighting addresses the ambiguity of neutral ratings and scoring bias.
- Construct preference pairs containing the best and worst candidates, followed by Direct Preference Optimization (DPO).
- Stages 2-4 can be executed iteratively, using the updated model to generate new candidates in each round.

### Key Designs

#### Rationale-Answer Pair (RAP) Format
RAP structures the model output into two components: Rationale (visual analysis and reasoning process) and Answer (final answer). This makes the reasoning process auditable and evaluable, serving as the foundational design of the framework.

#### R-Scorer Scoring Model
- Built upon a lightweight LLM, specifically trained to evaluate rationale quality rather than simply looking at answer correctness.
- Evaluation Criteria: Whether the reasoning is consistent with the image facts, whether the logical chain is complete, and whether the rationale sufficiently supports the answer.
- Pairwise scoring bypasses the difficulty of absolute quantification in open-ended reasoning.

#### Sentence-Level Beam Search
Departing from token-level beam search, searching at the sentence granularity branch enables generating candidates with fundamentally distinct reasoning paths rather than mere phrasing differences, thus assuring candidate-set diversity.

#### Iterative Calibration Mechanism
In each round, the improved model generates new candidates, and R-Scorer provides finer distinctions to construct higher-quality preference data, further upgrading the model. This establishes a virtuous cycle: a better model produces better candidates, and more precise preferences yield a better model.

## Key Experimental Results

### Main Results: Performance on MMStar Benchmark

| Model | Method | MMStar Performance | Gain |
|---|---|---|---|
| LLaVA-1.5-7B | Baseline | Baseline level | — |
| LLaVA-1.5-7B | + SRC | Significant improvement | Multiple percentage points |
| LLaVA-Next-8B | Baseline | Baseline level | — |
| LLaVA-Next-8B | + SRC | Significant improvement | Multiple percentage points |

Figure 1 of the paper displays the effects of SRC on MMStar, validating the framework's universality across different-sized LVLMs.

### Comparison with Existing Post-Training Methods

| Methodology Category | Representative Methods | Core Idea | Rationale-focused |
|---|---|---|---|
| Visual Description Preference Alignment | RLHF-V, POVID | Perturbing visual descriptions to construct preference pairs | No, description accuracy only |
| Output Sampling Preference Alignment | LLaVA-RLHF, STIC | Sampling from output to construct preferences | No, answer correctness only |
| Expert Model-assisted | DRESS, HA-DPO | Introducing external expert evaluation | Partially focused |
| **SRC (Ours)** | **R-Scorer + Iterative Calibration** | **Evaluating rationale quality + answer consistency** | **Core focus on rationale quality** |

### Multi-Dimensional Capability Improvement

| Evaluation Dimension | Representative Benchmark | SRC Gain |
|---|---|---|
| Perception | VQA series benchmarks | Significant improvement |
| Reasoning | Reasoning benchmarks like MMStar | Most prominent improvement |
| Generalization | Cross-domain testing | Strong generalization performance |

The paper emphasizes that compared to post-training methods that solely focus on vision-language alignment, the improvement of SRC in QA scenarios is particularly outstanding.

## Highlights & Insights

- **Precise Problem Definition**: This work is the first to systematically study the phenomenon where correct answers do not equal correct reasoning in LVLMs, establishing rationale-answer alignment as an independent research problem.
- **Self-Supervised Closed Loop**: The model is calibrated utilizing only the quality differences in its own output space, without relying on external annotations or expert models.
- **Pairwise Scoring Philosophy**: Relative comparison is inherently more reliable than absolute scoring, aligning with the Bradley-Terry model in RLHF.
- **Iterative Progressive Improvement**: Avoids the issue of underutilizing information typical in single-round training.
- **Lightweight Implementation**: Employs LoRA fine-tuning and a lightweight R-Scorer, ensuring the overall computational overhead is manageable.
- **CoT Reasoning Embedding**: Ensures reasoning quality through preference learning instead of pure SFT, representing a deep integration of Chain-of-Thought into VLM training.

## Limitations & Future Work

- **Reliability of R-Scorer**: Biases in the scoring model may propagate incorrect preference signals.
- **Iterative Computational Overhead**: The cumulative cost of multi-round candidate generation, scoring, and DPO is relatively high.
- **Diversity Ceiling**: When the seed model has low reasoning capabilities, the candidate set may lack high-quality samples.
- **Subjectivity in Evaluation**: Pairwise comparisons cannot completely eliminate noise in the evaluation of rationale quality.
- **Limited Model Coverage**: Primarily validated on the LLaVA series; its applicability to other architectures (e.g., Qwen-VL, InternVL) remains to be confirmed.
- **Format Rigidity**: A fixed RAP format may impact the naturalness of free-form conversation scenarios.

## Related Work & Insights

### LVLM Preference Alignment
Works like RLHF-V, POVID, LLaVA-RLHF, and STIC focus on the accuracy of descriptions, whereas SRC goes a step further to focus on whether the reasoning is logical and sound.

### LLM-as-Judge
R-Scorer is an innovative application of the LLM-as-judge paradigm in visual reasoning assessment; using a pairwise instead of independent scoring scheme boosts judgment quality.

### Chain-of-Thought and Reasoning
While classic CoT only guides the output of reasoning processes during inference, SRC fundamentally elevates reasoning quality through preference learning during training.

### Self-Improvement Paradigm
Self-play and self-reward paradigms iteratively improve models using their own capabilities; SRC's closed loop aligns with this concept but strictly focuses on rationale-answer alignment.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to systematically study rationale-answer alignment in LVLMs, featuring an innovative pairwise scoring strategy.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated across multiple benchmarks and models over three dimensions: perception, reasoning, and generalization.
- Writing Quality: ⭐⭐⭐⭐ — Clear problem motivation, logically self-consistent framework, and intuitive examples.
- Value: ⭐⭐⭐⭐ — Identifies a major blind spot in current LVLM post-training; the methodology is practical and highly generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Self-Prophetic Decoding to Unlock Visual Search in LVLMs](../../ICML2026/multimodal_vlm/self-prophetic_decoding_to_unlock_visual_search_in_lvlms.md)
- [\[ICML 2025\] CoMemo: LVLMs Need Image Context with Image Memory](comemo_lvlms_need_image_context_with_image_memory.md)
- [\[CVPR 2026\] Multimodal Learning on Low-Quality Data with Conformal Predictive Self-Calibration](../../CVPR2026/multimodal_vlm/multimodal_learning_on_low-quality_data_with_conformal_predictive_self-calibrati.md)
- [\[CVPR 2025\] VladVA: Discriminative Fine-tuning of LVLMs](../../CVPR2025/multimodal_vlm/vladva_discriminative_fine-tuning_of_lvlms.md)
- [\[NeurIPS 2025\] Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs](../../NeurIPS2025/multimodal_vlm/video-safetybench_a_benchmark_for_safety_evaluation_of_video_lvlms.md)

</div>

<!-- RELATED:END -->
