---
title: >-
  [Paper Note] FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding
description: >-
  [AAAI 2026][Video Anomaly Understanding] This paper proposes the FineVAU benchmark, which decomposes Video Anomaly Understanding (VAU) into three dimensions — Event (What), Entity (Who), and Location (Where) — introduces the FV-Score metric with high alignment to human perception, and constructs the FineW³ dataset via a fully automated LVLM-assisted pipeline. Experiments reveal critical shortcomings of current LVLMs in fine-grained anomalous event perception.
tags:
  - AAAI 2026
  - Video Anomaly Understanding
  - Benchmark
  - LLM-as-Judge
  - Fine-Grained Evaluation
  - Human Alignment
date: 2026-05-08
content_hash: 5d29809e7e4acf4e
---

# FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding

**Conference**: AAAI 2026
**arXiv**: [2601.17258](https://arxiv.org/abs/2601.17258)
**Code**: [https://finevau.github.io](https://finevau.github.io)
**Area**: Interpretability
**Keywords**: Video Anomaly Understanding, Benchmark, LLM-as-Judge, Fine-Grained Evaluation, Human Alignment

## TL;DR
This paper proposes the FineVAU benchmark, which decomposes Video Anomaly Understanding (VAU) into three dimensions — Event (What), Entity (Who), and Location (Where) — introduces the FV-Score metric with high alignment to human perception, and constructs the FineW³ dataset via a fully automated LVLM-assisted pipeline. Experiments reveal critical shortcomings of current LVLMs in fine-grained anomalous event perception.

## Background & Motivation

### State of the Field
Video Anomaly Understanding (VAU) is a core task in video surveillance. With the rise of large vision-language models (LVLMs), VAU has evolved from simple binary classification (normal/anomalous) to richer task formulations, including dense captioning, video question answering, and chain-of-thought reasoning.

### Limitations of Prior Work
Existing VAU evaluation methods suffer from two categories of critical problems:

**N-gram-based metrics** (e.g., BLEU, ROUGE-L) measure only lexical overlap and fail to capture semantic equivalence in free-form responses. A factually correct description that uses different wording is incorrectly penalized.

**LLM-based metrics** (e.g., AnomEVAL, VAU-EVAL) focus on linguistic fluency and reasoning consistency, lacking fine-grained detection of anomaly-specific visual elements. These metrics produce vague, subjective scores that are poorly aligned with human perception of anomalies.

### Root Cause
A fundamental gap exists between evaluation metrics and human priorities. When judging the quality of anomaly descriptions, humans primarily focus on three core questions: "What event occurred," "Who was involved," and "Where did it happen" — rather than textual fluency or lexical matching.

### Starting Point
The authors formulate VAU as a three-dimensional structured problem, assessing quality by detecting whether key visual elements from the ground truth are covered in LVLM outputs, rather than relying on subjective scoring.

## Method

### Overall Architecture
FineVAU comprises three core contributions:
1. **Problem Formulation**: Formalizing VAU as a What/Who/Where three-dimensional evaluation problem
2. **FV-Score Metric**: A structured LLM-based evaluation metric
3. **FineW³ Dataset**: A finely annotated dataset constructed via an automated pipeline

### Key Designs

#### 1. **Three-Dimensional Structured Evaluation Framework**
- **What (Event Dimension)**: Captures key actions (e.g., "arson"), interactions (e.g., "fighting"), and isolated state changes (e.g., "explosion"), scored on a three-level scale (0 = missing/incorrect, 0.5 = partially correct, 1 = fully accurate)
- **Who (Entity Dimension)**: Describes entities involved in the anomaly and their visual attributes (clothing, age, gender, etc.), scored on a binary scale (0/1)
- **Where (Location Dimension)**: Covers physical environment, time, lighting conditions, crowd density, etc., scored on a binary scale (0/1)
- **Design Motivation**: Humans instinctively attend to these three dimensions when perceiving anomalies. Simplifying to binary/ternary scoring — unlike the complex rubrics of existing metrics — reduces the difficulty of LLM-based judgment and improves interpretability

#### 2. **FV-Score and FineVAU-Judge**
- A structured scoring function is defined as: $\mathcal{S}(R) = \lambda_{what} \cdot \mathcal{J}_{what}(R) + \lambda_{who} \cdot \mathcal{J}_{who}(R) + \lambda_{where} \cdot \mathcal{J}_{where}(R)$
- Gemini-2.5-Flash is used as the LLM judge to evaluate semantic membership of each GT element in the model response
- **Mechanism**: Evaluation is recast as a "multi-part detection problem" — verifying whether key elements from the GT are covered in the generated report
- Weight ablations show that $\lambda_{who}=2.0$ yields the best alignment with human judgments, indicating that humans place particularly high importance on entity identification

#### 3. **FineW³ Dataset Construction**
- A **two-stage fully automated annotation pipeline** that augments high-quality human annotations from the UCA dataset:
    - **Stage 1 (Event Decomposition & Entity Linking)**: An LVLM decomposes complex event descriptions into causal-chain atomic events, supplements missing events, and identifies and links participating entities
    - **Stage 2 (Entity Grounding & Scene Description)**: Fine-grained physical attributes are added for each entity, along with scene characteristic descriptions
- Gemini-2.5-Pro is used with 1fps sampled frames and original UCA annotations as input
- Final dataset: 1,544 videos, 17,813 events (13,393 normal + 4,420 anomalous), 59,392 entities, 74,593 attributes, and 7,669 location attributes

### Human Alignment Validation
- 60 videos, 8 human experts, 180 ranking judgments
- Four correlation measures: PCC, 1-R², Kendall τ, Spearman τ

## Key Experimental Results

### Main Results

| Model | Overall | Where | What | Who | Attributes |
|------|------|------|------|------|------|
| InternVL3-9B | 40.5 | 71.8 | 18.0 | 51.2 | 25.5 |
| LLaVA-VID-7B | 35.0 | 65.7 | 14.4 | 44.0 | 21.0 |
| LLaVA-OV-7B | 32.2 | 58.3 | 13.0 | 41.1 | 19.9 |
| Qwen2.5-VL-7B | 32.9 | 70.8 | 9.1 | 38.3 | 20.3 |
| VideoLLaMA3-7B | 19.3 | 40.3 | 6.5 | 24.3 | 10.2 |
| **Average** | **32.0** | **61.3** | **12.2** | **39.8** | **19.4** |

### Ablation Study (Metric Human Alignment)

| Metric | PCC ρ↑ | 1-R²↓ | Kendall τ↑ | Spearman τ↑ |
|------|--------|-------|------------|-------------|
| FV-Score (Ours) | **0.61** | **0.63** | **0.56** | **0.56** |
| VAU-EVAL | 0.53 | 0.72 | 0.49 | 0.47 |
| ROUGE-L | 0.47 | 0.78 | 0.43 | 0.44 |
| AnomEVAL | 0.42 | 0.82 | 0.39 | 0.37 |
| BLEU | 0.19 | 0.96 | 0.17 | 0.17 |
| CIDEr | -0.63 | 0.60 | -0.59 | -0.58 |

### FV-Score Weight Ablation

| λ_what | λ_who | λ_where | PCC ρ↑ | Kendall τ↑ |
|--------|-------|---------|--------|------------|
| 1.0 | **2.0** | 1.0 | **0.61** | **0.56** |
| 2.0 | 1.0 | 1.0 | 0.56 | 0.50 |
| 1.0 | 1.0 | 1.0 | 0.51 | 0.46 |
| 1.0 | 1.0 | 2.0 | 0.47 | 0.42 |

### Key Findings
1. **LVLMs excel at static coarse-grained information**: The Where dimension averages 61.3%, far exceeding the What dimension at 12.2%
2. **Event understanding is critically weak**: Only 12.2% average accuracy, particularly for anomalies lacking strong visual cues (e.g., shoplifting)
3. **LVLMs exhibit a "normalcy bias"**: Models tend to describe anomalous events as normal behavior (e.g., describing a fight as a conversation)
4. **Entity recognition is easier than event understanding**: 39.8% vs. 12.2%, though substantial room for improvement remains
5. **InternVL3 leads across all dimensions**: Achieves best performance on every evaluated dimension

## Highlights & Insights
- **Evaluation paradigm innovation**: Shifting from subjective scoring to structured element detection substantially improves interpretability
- **Who dimension carries the highest weight**: Ablation experiments reveal that humans prioritize accurate identification of participating entities when evaluating anomaly descriptions — a counterintuitive yet well-supported finding
- **Exposing LVLM blind spots**: Current models exhibit fundamental deficiencies in understanding fine-grained spatiotemporal events, a problem unlikely to be resolved through simple scaling
- **Fully automated annotation pipeline**: The approach is extensible to additional datasets and provides a reusable solution for VAU data construction

## Limitations & Future Work
- The dataset is sourced from CCTV surveillance footage, limiting scene diversity
- Evaluation relies on a single LLM (Gemini-2.5-Flash), which may introduce judgment bias
- Only open-source 7–9B models are evaluated; larger-scale or proprietary models are not covered
- The three-dimensional framework may omit "Why" (the cause of anomalies) as an important dimension
- In the three-level scoring of the What dimension, the assignment of 0.5 scores may still carry some subjectivity

## Related Work & Insights
- UCA (Yuan et al., CVPR 2024) introduced dense human-annotated descriptions but relied on N-gram evaluation
- HAWK (Tang et al., NeurIPS 2024) proposed synthesized descriptions but still evaluated primarily on language quality
- Holmes-VAU (Zhang et al., CVPR 2025) introduced multi-granularity descriptions but lacked entity and scene information
- The structured decomposition + element detection paradigm proposed here can generalize to evaluation of other visual understanding tasks

## Rating
- Novelty: ⭐⭐⭐⭐ (Innovative evaluation paradigm, though fundamentally a benchmark contribution)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Comprehensive human alignment validation and multi-dimensional ablations)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear logic, rich figures and tables)
- Value: ⭐⭐⭐⭐ (Establishes a stronger standard for VAU evaluation, though the application scope is relatively narrow)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Text-guided Fine-Grained Video Anomaly Understanding](../../CVPR2026/interpretability/text-guided_fine-grained_video_anomaly_understanding.md)
- [\[AAAI 2026\] Can LLMs Truly Embody Human Personality? Analyzing AI and Human Behavior Alignment in Dispute Resolution](can_llms_truly_embody_human_personality_analyzing_ai_and_human_behavior_alignmen.md)
- [\[AAAI 2026\] ElementaryNet: A Non-Strategic Neural Network for Predicting Human Behavior in Normal-Form Games](elementarynet_a_non-strategic_neural_network_for_predicting_human_behavior_in_no.md)
- [\[AAAI 2026\] GateRA: Token-Aware Modulation for Parameter-Efficient Fine-Tuning](gatera_token-aware_modulation_for_parameter-efficient_fine-tuning.md)
- [\[AAAI 2026\] FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer](flashkat_understanding_and_addressing_performance_bottlenecks_in_the_kolmogorov-.md)

</div>

<!-- RELATED:END -->
