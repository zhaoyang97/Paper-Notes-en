---
title: >-
  [Paper Note] FinMME: Benchmark Dataset for Financial Multi-Modal Reasoning Evaluation
description: >-
  [ACL 2025][VLM Reasoning][Multimodal Evaluation] This work constructs FinMME, an evaluation benchmark containing over 11,000 high-quality financial multimodal samples across 18 financial domains and 10 chart types. It proposes the FinScore evaluation framework, which integrates hallucination penalties with domain normalization. Experimental results show that even GPT-4o scores only 15.34 (with an average accuracy of 46.56%), revealing significant deficiencies of MLLMs in the…
tags:
  - "ACL 2025"
  - "VLM Reasoning"
  - "Multimodal Evaluation"
  - "Financial Benchmark"
  - "Hallucination Penalty"
  - "Chart Understanding"
  - "MLLM"
date: 2026-05-08
content_hash: 840bb719ac2da930
---

# FinMME: Benchmark Dataset for Financial Multi-Modal Reasoning Evaluation

**Conference**: ACL 2025  
**arXiv**: [2505.24714](https://arxiv.org/abs/2505.24714)  
**Code**: [https://github.com/luo-junyu/FinMME](https://github.com/luo-junyu/FinMME)  
**Area**: Multimodal Large Language Model Evaluation / Financial NLP  
**Keywords**: Multimodal Evaluation, Financial Benchmark, Hallucination Penalty, Chart Understanding, MLLM

## TL;DR

This work constructs FinMME, an evaluation benchmark containing over 11,000 high-quality financial multimodal samples across 18 financial domains and 10 chart types. It proposes the FinScore evaluation framework, which integrates hallucination penalties with domain normalization. Experimental results show that even GPT-4o scores only 15.34 (with an average accuracy of 46.56%), revealing significant deficiencies of MLLMs in the financial domain.

## Background & Motivation

Multimodal Large Language Models (MLLMs) have advanced rapidly in recent years, performing exceptionally well on general benchmarks (such as MME, MMMU, and MMBench) with accuracies reaching 80–90%. However, the financial domain presents the following unique challenges:

**Highly Knowledge-Intensive**: Financial data contains professional terminology, complex quantitative relations, and domain-specific reasoning logic.

**Zero Tolerance for Inaccuracy**: Unlike general scenarios, errors in financial decision-making (especially hallucinations) can lead to severe economic losses.

**Abundant Multimodal Data**: Financial research reports extensively utilize charts (K-lines, pie charts, heatmaps, etc.), requiring strong cross-modal comprehension capabilities.

The key challenge is that existing general multimodal benchmarks cannot effectively differentiate the capabilities of models in knowledge-intensive financial scenarios. Although MMMU has a financial subset (only 390 questions), its scale is too small. While the parallel work MME-Finance targets finance, its data volume is limited (1,171 questions) and its annotation quality is questionable (not fully human-annotated).

The proposed solution in this work is to construct FinMME—a large-scale (11,099 questions), high-quality (annotated by a 20-person team with a key error rate $<1\%$), and comprehensive (covering 18 financial domains, 6 asset classes, and 10 chart types) finance-specific multimodal benchmark, alongside an innovative FinScore evaluation system designed to assess models fairly and rigorously.

## Method

### Overall Architecture

The construction of FinMME follows a three-stage pipeline:
1. **Data Collection and Cleaning**: Extract high-quality financial charts and text from professional research reports and webpage screenshots.
2. **Annotation**: Conduct parallel human and LLM annotation, ensuring quality through internal and external consistency checks.
3. **Quality Control**: Agreeing annotations are verified by a single expert, whereas disagreeing annotations are adjudicated by multiple experts.

### Key Designs

1. **Fine-grained Data Labeling System**:

    - **Knowledge Domain**: 18 core financial domains (TMT, consumer goods, healthcare, finance, real estate, etc.), comprehensively covering the modern financial knowledge system.
    - **Asset Class**: 6 classes (equities, foreign exchange, interest rates, commodities, credit, cross-asset), supporting targeted evaluation of different market sectors.
    - **Data Category**: 10 main categories and 21 subcategories (time series, distribution charts, proportion charts, relation charts, financial statements, etc.), reflecting the chart diversity in real-world financial analysis.
    - Each sample is accompanied by an image, image caption, research report description, hierarchical metadata, and QA annotations.

2. **Three-layer Cognitive Evaluation Framework**:

    - **Comprehensive Perception**: Evaluates time-series identification, horizontal comparison, and multi-chart analysis capabilities (2,333 questions).
    - **Fine-grained Perception**: Examines numerical extraction and local variation analysis capabilities (6,466 questions).
    - **Cognition and Reasoning**: Covers advanced reasoning such as data inference, cross-modal understanding, trend prediction, and causal analysis (2,300 questions).
    - Question types include single-choice, multiple-choice, and calculation questions, with a higher proportion of multiple-choice questions than existing benchmarks to better detect hallucinations.

3. **Innovative Quality Control Mechanism**:

    - Employs a "dual-track" strategy of parallel human and multi-LLM annotation.
    - Utilizes external consistency of LLMs (predictions from multiple LLMs) and internal consistency (comparison with human annotations) to detect errors.
    - When human and machine annotations agree, a single expert validates them; when they disagree, a joint review by multiple experts is conducted.
    - Consists of 20 annotators (12 junior + 8 experts), with a cumulative effort of approximately 800 hours.

### FinScore Evaluation Framework

FinScore combines domain-normalized performance with an hallucination penalty:

1. **Per-question Score (Multiple-choice)**: $S_q = \max(0, \frac{c}{n} - \frac{i}{s})$, where $c$ is the number of correct selections, $n$ is the total number of options, $i$ is the number of incorrect selections, and $s$ is the total number of selections made by the model. This penalizes hallucination behaviors of over-selection.

2. **Domain-Normalized Score**: $F = \frac{1}{K}\sum_{k=1}^{K}\frac{1}{N_k}\sum_{i=1}^{N_k}S_{k,i}$, averaged within domains first and then across domains, ensuring fair contributions from domains of different difficulties.

3. **Final FinScore**: $\mathcal{F} = F \cdot (1 - P_H)$, where $P_H = \text{mean}(\frac{i}{s})$ represents the hallucination penalty rate. The multiplicative combination ensures that models with high hallucination rates are severely penalized.

## Key Experimental Results

### Main Results

| Model | Comprehensive Perception | Fine-grained Perception | Reasoning | Average | FinScore |
|------|---------|-----------|------|------|----------|
| Gemini Flash 2.0 | 49.89 | 59.07 | 48.71 | 51.85 | 20.10 |
| Claude 3.5 Sonnet | 45.99 | 55.28 | 43.35 | 48.20 | 15.61 |
| GPT-4o | 44.33 | 53.49 | 42.24 | 46.56 | 15.34 |
| GPT-4o Mini | 41.91 | 48.47 | 42.88 | 43.72 | 11.70 |
| **Qwen2.5-VL 72B** | **49.64** | **60.25** | **49.44** | **52.54** | **20.87** |
| InternVL 2.5-8B | 37.96 | 51.83 | 35.33 | 41.90 | 10.42 |
| Qwen2.5-VL 3B | 32.53 | 52.55 | 30.70 | 39.87 | 6.95 |
| Phi-3.5 V | 25.73 | 43.37 | 26.46 | 33.13 | 2.85 |

The open-source Qwen2.5-VL 72B outperforms all closed-source models, achieving the best performance.

### Ablation Study (Dimensional Contrastive Analysis)

| Evaluation Dimension | Single-choice | Multiple-choice | Calculation | Description |
|---------|-------|-------|-------|------|
| Comprehensive Perception | - | - | - | Requires global chart understanding |
| Fine-grained Perception | Higher | Lower | - | Large disparity in numerical localization capability |
| Cognitive Reasoning | - | - | 35.59 (Best) | Calculation questions are the most challenging |

**Analysis by Question Type**:
- Single-choice (58-65%) is significantly higher than multiple-choice (25-55%), indicating that multiple-choice questions effectively check for hallucinations.
- Calculation questions score the lowest overall (7-37%), exposing weakness in mathematical reasoning.

**Compression Effect of FinScore**:
- GPT-4o's average score is 46.56 $\rightarrow$ FinScore is only 15.34, representing a penalty rate of approximately 67%.
- This demonstrates that even if a model answers many questions correctly, a large number of incorrect selections (hallucinations) will severely drag down the final score.

### Key Findings

- **Closed-source vs. Open-source**: Qwen2.5-VL 72B is the only open-source model that outperforms all closed-source models (FinScore of 20.87 vs. GPT-4o's 15.34).
- **Scale Effect**: In the DeepSeekVL-2 series, Full > Small > Tiny, showing a clear positive correlation between parameter size and performance.
- **Robustness**: The standard deviation of predictions under different prompts is below 1%, which is significantly better than existing general benchmarks.
- **Domain Discrepancies**: Scores in TMT and consumer goods are generally higher (where information is more common), while scores in derivatives and fixed income/quantitative finance are lower (requiring higher professional expertise).
- **Fine-grained Perception Paradox**: Small models sometimes achieve scores close to large models in fine-grained perception but show huge gaps in FinScore due to their higher hallucination rates.

## Highlights & Insights

- **Design Philosophy of FinScore**: In financial scenarios, "not knowing" is more valuable than "guessing wrong." FinScore embodies this concept through a multiplicative penalty mechanism, making a significant contribution to financial AI evaluation.
- **Innovative Use of LLM-assisted Annotation**: Instead of letting LLMs fully replace human annotators, the consistency/inconsistency across multiple LLMs is leveraged to assist in quality control. This human-AI collaborative annotation paradigm is highly generalizable.
- **Multiple-Choice Questions as Hallucination Detectors**: Increasing the proportion of multiple-choice questions and introducing a penalty mechanism effectively distinguishes "random guessing" from "true understanding."
- **Scale Effect Reversal**: The phenomenon wherein an open-source 72B model outperforms top-tier closed-source models suggests that, in vertical domains, model architecture and training data may be of greater importance than brand prestige.

## Limitations & Future Work

- **Data Source Bias**: The data primarily originates from Chinese financial research reports, lacking coverage of European, American, and emerging markets.
- **Static Evaluation**: Finance is a dynamic environment, and the current benchmark cannot evaluate a model's ability to process time-sensitive information.
- **Absence of Open-ended Generation Tasks**: The benchmark consists entirely of multiple-choice and calculation questions, making it impossible to evaluate capabilities such as financial text generation and report writing.
- **Potentially Harsh Hallucination Penalties**: The multiplicative combination results in extremely low scores for models with moderate accuracy but minor hallucinations, which may underestimate their practical utility.
- **Extendable Directions**: Introducing time-decay weights, incorporating real-time data tasks, and supporting multi-turn interactive financial QA evaluation.

## Related Work & Insights

- This is a parallel work with MME-Finance (Gan et al., 2024), but FinMME holds a significant advantage in data volume (11K vs. 1.2K) and annotation quality.
- The financial subset of MMMU contains only 390 questions, failing to provide stable and reliable evaluation.
- The design philosophy of FinScore can be extended to other high-precision domains (e.g., medicine, law).
- The hierarchical evaluation framework (perception $\rightarrow$ reasoning) resembles the application of Bloom's Taxonomy in AI evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐ Multimodal financial evaluation is a blank field, and the FinScore design is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 17 models are evaluated, with detailed multidimensional analyses and rigorous robustness validation.
- Writing Quality: ⭐⭐⭐⭐ Structured and clear, though some statistical descriptions are redundant.
- Value: ⭐⭐⭐⭐⭐ Fills the gap in financial MLLM evaluation, and FinScore offers practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] FCMR: Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning](fcmr_robust_evaluation_of_financial_cross-modal_multi-hop_reasoning.md)
- [\[ACL 2025\] Mitigating Visual Forgetting via Take-along Visual Conditioning for Multi-modal Long CoT Reasoning](tvc_mitigating_visual_forgetting.md)
- [\[ICCV 2025\] From Easy to Hard: The MIR Benchmark for Progressive Interleaved Multi-Image Reasoning](../../ICCV2025/vlm_reasoning/from_easy_to_hard_the_mir_benchmark_for_progressive_interleaved_multi-image_reas.md)
- [\[ICCV 2025\] ToolVQA: A Dataset for Multi-step Reasoning VQA with External Tools](../../ICCV2025/vlm_reasoning/toolvqa_a_dataset_for_multistep_reasoning_vqa_with_external.md)
- [\[CVPR 2025\] Beyond Final Answers: CRYSTAL Benchmark for Transparent Multimodal Reasoning Evaluation](../../CVPR2025/vlm_reasoning/beyond_final_answers_crystal_benchmark_for_transparent_multimodal_reasoning_eval.md)

</div>

<!-- RELATED:END -->
