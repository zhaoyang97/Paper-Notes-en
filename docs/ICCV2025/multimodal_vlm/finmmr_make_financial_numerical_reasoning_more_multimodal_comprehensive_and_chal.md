---
title: >-
  [Paper Note] FinMMR: Make Financial Numerical Reasoning More Multimodal, Comprehensive, and Challenging
description: >-
  [ICCV 2025][Multimodal VLM][financial numerical reasoning] This paper proposes FinMMR, a bilingual (Chinese–English) multimodal financial numerical reasoning benchmark containing 4,300 questions and 8,700 images spanning 14 financial sub-domains, requiring models to perform multi-step precise numerical computation. Evaluation of 15 state-of-the-art MLLMs shows that the best model achieves only 53% accuracy on the Hard subset, exposing fundamental bottlenecks in current MLLMs for professional-domain multimodal reasoning.
tags:
  - ICCV 2025
  - Multimodal VLM
  - financial numerical reasoning
  - multimodal benchmark
  - LLM evaluation
  - chain-of-thought reasoning
  - knowledge augmentation
date: 2026-05-08
content_hash: 1da98f14c6e31679
---

# FinMMR: Make Financial Numerical Reasoning More Multimodal, Comprehensive, and Challenging

**Conference**: ICCV 2025
**arXiv**: [2508.04625](https://arxiv.org/abs/2508.04625)
**Code**: None (online evaluation platform provided)
**Area**: Multimodal VLM
**Keywords**: multimodal reasoning, financial numerical reasoning, benchmark, LLM evaluation, visual perception

## TL;DR

This paper proposes FinMMR, a bilingual (Chinese–English) multimodal financial numerical reasoning benchmark comprising 4,300 questions, 8,700+ financial charts, and 14 financial sub-domains. It systematically evaluates 15 MLLMs to identify bottlenecks in complex domain-specific reasoning, and proposes three improvement strategies: visual filtering, knowledge augmentation, and model collaboration.

## Background & Motivation

Large reasoning models (LRMs) have demonstrated strong performance on pure-text reasoning tasks, yet they remain significantly challenged in real-world scenarios requiring the integration of visual perception and domain expertise. Finance is a prototypical high-stakes domain, where analysts must read visually rich financial documents, extract key metrics from tables, charts, and text, and perform multi-step precise numerical calculations.

Existing financial reasoning benchmarks suffer from three major shortcomings:

**Lack of multimodality**: Benchmarks such as FinanceMath and CodeFinQA support text-only input.

**Narrow domain coverage**: Most benchmarks cover only 6–10 sub-domains.

**Insufficient reasoning difficulty**: FAMMA draws from textbooks/exams, and MMMU is limited to multiple-choice format, failing to genuinely test precise numerical reasoning ability.

## Method

### Overall Architecture

The FinMMR benchmark is constructed from two sources:
- **Adapted public benchmarks**: Questions are extracted from MMMU, MMMU-Pro, FinanceMath, CodeTAT-QA, CodeFinQA, and DocMath-Eval, then converted into multimodal form by rendering tabular text as images.
- **Newly constructed CRRQA**: 2,150 new questions are built from 90 Chinese financial research reports.

Each question includes rich image inputs, a clear problem description, a Python-format solution, and a precise numerical answer.

### Key Designs

1. **Multimodality**: All tables and charts are rendered as images rather than structured text, forcing models to perform visual perception. In Chinese questions, distractor images from adjacent pages of the same report are mixed in (3,938 distractor images for 2,150 questions), simulating the real-world challenge of filtering relevant images from multiple inputs. Fourteen image categories are covered, including bar charts, line charts, equity structure diagrams, and candlestick charts.

2. **Comprehensiveness**: The benchmark spans 14 financial sub-domains (corporate finance, banking, industry analysis, asset management, etc.), substantially exceeding the knowledge breadth of existing benchmarks. Both Chinese and English are supported, with 2,150 questions per language.

3. **Difficulty**: All questions require precise numerical answers (no multiple choice), evaluated under a strict 0.2% error tolerance. Difficulty is graded using a heuristic formula: $rc = \ln(\max(o,1)) + \ln(\max(l+p,1))$, where $o$ is the number of operators, $l$ is the number of code lines, and $p$ is the number of parenthesis pairs. The Hard subset requires on average 5.34 operators and 7.34 lines of code.

### Loss & Training

As a benchmark paper, no model training is involved. The evaluation framework includes:
- **Prompting methods**: IO (no prompting), CoT (chain-of-thought), and PoT (program-of-thought).
- **Answer extraction**: GPT-4o-mini extracts numerical values for CoT/IO; PoT directly executes the generated Python program.
- **Evaluation criterion**: Strict 0.2% numerical error tolerance.

## Key Experimental Results

### Main Results

| Model | Reasoning Augmentation | Hard (CoT) | Hard (PoT) | Medium (CoT) | Easy (CoT) | Avg (CoT) |
|---|---|---|---|---|---|---|
| Claude 3.7 Sonnet (64K) | ✔ | 53.00 | 51.40 | 62.50 | 78.50 | 64.00 |
| OpenAI o1 | ✔ | 48.40 | 44.70 | — | — | — |
| GPT-4o | ✘ | 45.40 | 47.80 | 63.33 | 78.00 | 62.24 |
| Llama 4 Maverick (17B) | ✘ | 48.70 | 47.80 | 63.25 | 77.83 | 63.26 |
| Qwen2.5-VL-72B | ✘ | 43.30 | 46.20 | 63.42 | 77.42 | 61.38 |
| QVQ-72B-Preview | ✔ | 40.30 | 6.20 | 55.67 | 75.42 | 57.13 |

The best-performing model achieves only 53% on the Hard subset, far below a passing threshold.

### Ablation Study (Knowledge Augmentation & Model Collaboration)

| Strategy | PoT Baseline | RAG + PoT | Gain |
|---|---|---|---|
| Gemini 2.0 Flash Thinking | 78.71 | 83.02 | +4.31 |
| GPT-4o | 80.60 | 83.62 | +3.02 |
| Claude 3.7 Sonnet | 81.21 | 85.43 | +4.22 |
| Claude 3.7 Sonnet (64K) | 83.53 | 86.29 | +2.76 |

| Distractor Image Experiment (Qwen2.5-VL-72B) | Ground Images | Distractor Images | Performance Drop |
|---|---|---|---|
| Hard | 57.18% | 47.23% | ↓9.95 |
| Medium | 73.01% | 61.36% | ↓11.65 |
| Easy | 61.59% | 53.64% | ↓7.95 |

The two-stage visual filtering–reasoning pipeline improves accuracy on the Medium subset from 64.73% to 71.56% (+6.83).

### Key Findings

1. **Reasoning-augmented models are costly**: Claude 3.7 Sonnet's 64K thinking mode consumes nearly 12× more tokens (4.06M vs. 0.34M) for only a 2.2-point accuracy gain.
2. **PoT outperforms CoT**: PoT is superior when precise computation is required; Qwen2.5-VL-72B with PoT reduces token consumption by 58.88% while improving accuracy.
3. **Catastrophic degradation of QVQ**: Reinforcement learning training causes a loss of code generation capability, resulting in a PoT execution success rate of only 10.9% and an accuracy collapse from 40.3% (CoT) to 6.2% (PoT).
4. **Model collaboration is effective**: The combination of GPT-4o (visual parser) + DeepSeek-R1 (reasoner) achieves 86.72%, surpassing the best single-model result of 83.53%.
5. **Error type analysis**: 38% knowledge reasoning errors, 32% numerical computation errors, and 30% visual perception errors.

## Highlights & Insights

- The **distractor image mechanism** closely mirrors real-world financial analysis scenarios and reveals the fragility of MLLMs in multi-image filtering (performance drops exceeding 10%).
- The **two-stage decoupling of visual filtering and reasoning** is concise and effective, offering a practical paradigm for complex multimodal reasoning.
- The **knowledge augmentation experiments** demonstrate the substantial potential of structured domain knowledge (a library of 3,133 Python financial functions) to enhance MLLM reasoning.
- **Data quality assurance**: 16 finance graduate students and 2 CFA charterholders spent three months on annotation and validation.

## Limitations & Future Work

- Test set answers are not publicly released (to prevent leakage), relying on an online evaluation platform, which may limit reproducibility and in-depth analysis.
- Distractor images are only used in the Chinese subset; the English subset lacks this challenge.
- The difficulty grading is based on a heuristic formula derived from code complexity, which may not fully reflect cognitive difficulty.
- Open-source reasoning-augmented models (e.g., multimodal variants of DeepSeek-R1) were not evaluated.

## Related Work & Insights

- FinMMR is complementary to FAMMA (textbook/exam-sourced) and FinMME (multiple-choice format), as it more closely reflects real financial work scenarios.
- The model collaboration paradigm of visual parser + reasoner is general and can be extended to other professional domains such as medicine and law.
- MLLM-instructed knowledge retrieval (having the MLLM generate retrieval queries) is a practical RAG improvement strategy.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First large-scale multimodal financial numerical reasoning benchmark; the distractor image design and precise numerical evaluation are distinctive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluates 15 models with three prompting methods, multiple improvement strategies, and in-depth error analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with logically organized experiments, though some tables are extremely information-dense.
- **Value**: ⭐⭐⭐⭐⭐ Provides an important evaluation benchmark and actionable improvement directions for MLLM applications in specialized domains.

---

# FinMMR: Make Financial Numerical Reasoning More Multimodal, Comprehensive, and Challenging

**Conference**: ICCV 2025
**arXiv**: [2508.04625](https://arxiv.org/abs/2508.04625)
**Code**: None (online evaluation platform provided)
**Area**: Multimodal Vision-Language Models
**Keywords**: financial numerical reasoning, multimodal benchmark, LLM evaluation, chain-of-thought reasoning, knowledge augmentation

## TL;DR

This paper proposes FinMMR, a bilingual (Chinese–English) multimodal financial numerical reasoning benchmark containing 4,300 questions and 8,700 images spanning 14 financial sub-domains, requiring models to perform multi-step precise numerical computation. Evaluation of 15 state-of-the-art MLLMs shows that the best model achieves only 53% accuracy on the Hard subset, exposing fundamental bottlenecks in current MLLMs for professional-domain multimodal reasoning.

## Background & Motivation

Large reasoning models (LRMs) have made significant advances in code, mathematics, and scientific reasoning, and MLLMs have demonstrated strong performance on general multimodal reasoning. Nevertheless, challenges faced by MLLMs in high-stakes professional domains such as finance remain poorly understood:
- Financial analysis requires extracting key metrics from visually rich documents and performing multi-step precise numerical calculations.
- Existing benchmarks have notable limitations: FAMMA draws from textbooks and exam questions, MathVista does not involve financial knowledge, and MMMU is restricted to multiple-choice format.

FinMMR offers three key advantages:
1. **Multimodality**: All tables and charts are presented as images, including distractor images.
2. **Comprehensiveness**: Covers 14 financial sub-domains (corporate finance, banking, industry analysis, etc.).
3. **Difficulty**: Requires precise numerical answers, eliminating the guessing bias inherent in multiple-choice formats.

## Method

### Overall Architecture

FinMMR is constructed via two pathways:
1. Questions are extracted from publicly available text-based financial reasoning benchmarks (MMMU, MMMU-Pro, FinanceMath, CodeTAT-QA, CodeFinQA, DocMath-Eval) and converted into multimodal form.
2. A new dataset, CRRQA (Chinese Research Report QA, 2,150 questions), is constructed from recent Chinese financial research reports.

Both sources are merged into FinMMR; each question is paired with an executable Python solution and a precise numerical answer.

### Key Designs

1. **Multimodal Conversion**: Tabular data is rendered as images and the corresponding textual table content is removed, ensuring that MLLMs cannot rely solely on text. The core innovation is the introduction of **distractor images**—semantically related but task-irrelevant images selected from adjacent positions within the same report—to simulate real-world information overload.

2. **Difficulty Grading System**: Questions are graded heuristically based on complexity metrics of the Python solution, considering the number of operators $o$, code lines $l$, and parenthesis pairs $p$: $rc = \ln(\max(o,1)) + \ln(\max(l+p,1))$. Questions are divided into Easy (1,300), Medium (1,500), and Hard (1,500) subsets.

3. **Evaluation Framework**: Three prompting methods are employed—CoT (chain-of-thought), PoT (program-of-thought), and IO (no prompting). PoT generates and executes Python code; strict numerical evaluation applies a 0.2% error tolerance, requiring precision in units, percentages, and decimal places.

### Loss & Training

This paper presents a benchmark and involves no model training. The core methodological contributions to the evaluation strategy are:
- **Visual filtering–reasoning pipeline**: MLLMs first assess image relevance, filter out distractors, and then perform reasoning.
- **Knowledge augmentation**: A financial function library containing 3,133 Python functions is constructed; MLLM-guided knowledge retrieval augments reasoning.
- **Model collaboration**: GPT-4o serves as a visual parser to convert images into structured text, which is then processed by an LRM for reasoning.

## Key Experimental Results

### Main Results

| Model | Extended Thinking | Hard (CoT) | Hard (PoT) | Medium (CoT) | Easy (CoT) | Avg (CoT) |
|---|---|---|---|---|---|---|
| Claude 3.7 Sonnet | ✔ (64K) | 53.00 | 51.40 | 62.50 | 78.50 | 64.00 |
| Claude 3.7 Sonnet | ✘ | 50.80 | 48.50 | 62.25 | 77.00 | 63.35 |
| OpenAI o1 | ✔ | 48.40 | 44.70 | — | — | — |
| GPT-4o | ✘ | 45.40 | 47.80 | 63.33 | 78.00 | 62.24 |
| Llama 4 Maverick | ✘ | 48.70 | 47.80 | 63.25 | 77.83 | 63.26 |
| Qwen2.5-VL-72B | ✘ | 43.30 | 46.20 | 63.42 | 77.42 | 61.38 |
| QVQ-72B-Preview | ✔ | 40.30 | 6.20 | 55.67 | 75.42 | 57.13 |

### Ablation Study / Knowledge Augmentation Effect

| Model | PoT Baseline | + Knowledge Augmentation (RAG) | Gain |
|---|---|---|---|
| Gemini 2.0 Flash Thinking | 78.71 | 83.02 | +4.31 |
| GPT-4o | 80.60 | 83.62 | +3.02 |
| Claude 3.7 Sonnet | 81.21 | 85.43 | +4.22 |
| Claude 3.7 Sonnet (64K) | 83.53 | 86.29 | +2.76 |

(Based on 1,160 table-QA instances.)

**Effect of Distractor Images** (Qwen2.5-VL-72B, PoT):

| Subset | Ground Images | Distractor Images | Drop |
|---|---|---|---|
| Hard | 57.18% | 47.23% | ↓9.95 |
| Medium | 73.01% | 61.36% | ↓11.65 |
| Easy | 61.59% | 53.64% | ↓7.95 |

The visual filtering–reasoning pipeline improves accuracy on the Medium validation set from 64.73% to 71.56% (+6.83).

### Key Findings

1. **All MLLMs perform poorly on FinMMR**: The strongest model, Claude 3.7 Sonnet (64K thinking), achieves only 53% on the Hard subset, well below the 60% passing threshold.
2. **PoT outperforms CoT**: PoT achieves an average accuracy of 37.64% vs. 36.20% for CoT, with lower token consumption. However, QVQ-72B's PoT accuracy collapses to 6.2% due to reinforcement learning bias, exposing a training strategy issue.
3. **Distractor images severely impair reasoning**: Performance drops of over 10% indicate insufficient visual filtering capability in current MLLMs.
4. **Error analysis (100 failure cases)**: 30% visual perception errors, 38% knowledge reasoning errors, 32% numerical computation errors.
5. **Model collaboration is effective**: GPT-4o parsing + DeepSeek-R1 reasoning achieves 86.72%, outperforming the single-model best of 83.53% from Claude 3.7 Sonnet.

## Highlights & Insights

- **Practice-oriented benchmark design**: The introduction of distractor images to simulate real-world information overload is a feature absent from other benchmarks.
- **Comprehensive error attribution**: Decomposing failures into visual perception, knowledge reasoning, and numerical computation errors provides a clear roadmap for future improvement.
- **Effectiveness of knowledge augmentation**: Structured financial function libraries combined with MLLM-guided retrieval and MLLM-based judgment allow weaker models to approach SOTA performance.
- **Insight on extended thinking**: The modest accuracy gain (+2.2 pp) at 12× token cost raises important questions about the efficiency–effectiveness trade-off.

## Limitations & Future Work

- Only zero-shot settings are evaluated; few-shot and fine-tuning scenarios are not explored.
- The CRRQA portion relies on Qwen-VL-Max for initial question generation, potentially introducing model bias.
- The financial function library is manually constructed, limiting coverage and scalability.
- Recently released models such as GPT-o3 are not evaluated.
- The distractor image construction method (selecting adjacent images) is relatively simple; more complex distractor patterns warrant further exploration.

## Related Work & Insights

- Compared to general benchmarks such as MathVista and MMMU, FinMMR offers significant advantages in domain depth and reasoning complexity.
- The visual filtering–reasoning pipeline (decoupling perception from reasoning) is transferable to other domains.
- The model collaboration framework (visual parser + text reasoner) offers new directions for multimodal system design.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First large-scale multimodal financial numerical reasoning benchmark; the distractor image design and three-dimensional error analysis are novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 15 models, 3 prompting methods, error analysis, visual filtering, knowledge augmentation, and model collaboration.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured; the research-question-driven experimental organization facilitates comprehension.
- **Value**: ⭐⭐⭐⭐⭐ Fills the gap in multimodal reasoning evaluation for the financial domain and provides clear guidance for MLLM improvement.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] GRAB: A Challenging GRaph Analysis Benchmark for Large Multimodal Models](grab_a_challenging_graph_analysis_benchmark_for_large_multimodal_models.md)
- [\[AAAI 2026\] FinMMDocR: Benchmarking Financial Multimodal Reasoning with Scenario Awareness, Document Understanding, and Multi-Step Computation](../../AAAI2026/multimodal_vlm/finmmdocr_benchmarking_financial_multimodal_reasoning_with_scenario_awareness_do.md)
- [\[ICCV 2025\] Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation](is_less_more_exploring_token_condensation_as_training-free_test-time_adaptation.md)
- [\[AAAI 2026\] CrossVid: A Comprehensive Benchmark for Evaluating Cross-Video Reasoning in Multimodal Large Language Models](../../AAAI2026/multimodal_vlm/crossvid_a_comprehensive_benchmark_for_evaluating_cross-vide.md)
- [\[NeurIPS 2025\] MMPerspective: Do MLLMs Understand Perspective? A Comprehensive Benchmark for Perspective Perception, Reasoning, and Robustness](../../NeurIPS2025/multimodal_vlm/mmperspective_do_mllms_understand_perspective_a_comprehensive_benchmark_for_pers.md)

<!-- RELATED:END -->
