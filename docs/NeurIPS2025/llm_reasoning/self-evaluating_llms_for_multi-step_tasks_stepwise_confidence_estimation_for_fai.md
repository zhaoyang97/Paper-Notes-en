---
title: >-
  [Paper Note] Note 8: PolyMath — Evaluating Mathematical Reasoning in a Multilingual Context
description: >-
  [NeurIPS 2025][LLM Reasoning][Multilingual benchmark] PolyMath introduces a mathematical reasoning benchmark spanning 18 languages, 4 difficulty levels, and 500 problems, revealing that: (1) reasoning performance varies by up to 10 points across languages; (2) reasoning models exhibit low input–output language consistency, which may affect performance; and (3) thinking length varies substantially across languages — offering new perspectives for multilingual reasoning research.
tags:
  - NeurIPS 2025
  - LLM Reasoning
  - Multilingual benchmark
  - difficulty stratification
  - reasoning transfer
  - language consistency
  - thinking length
date: 2026-05-08
content_hash: a0cf7b8ee0952c8f
---

# Note 8: PolyMath — Evaluating Mathematical Reasoning in a Multilingual Context

**Conference**: NeurIPS 2025
**arXiv**: [2511.07364](https://arxiv.org/abs/2511.07364)
**Code**: [GitHub](https://github.com/QwenLM/PolyMath)
**Area**: Multilingual LLM, Mathematical Reasoning, Benchmark Evaluation
**Keywords**: Multilingual benchmark, difficulty stratification, reasoning transfer, language consistency, thinking length

## TL;DR
PolyMath introduces a mathematical reasoning benchmark spanning 18 languages, 4 difficulty levels, and 500 problems, revealing that: (1) reasoning performance varies by up to 10 points across languages; (2) reasoning models exhibit low input–output language consistency, which may affect performance; and (3) thinking length varies substantially across languages — offering new perspectives for multilingual reasoning research.

## Background & Motivation

### Root Cause

**Key Challenge**: **Background**: Multilingual benchmarks lag behind: although multilingual datasets such as MGSM and XSVAMP exist, their difficulty is too low (K-12 level) to evaluate the true capabilities of reasoning models.

**Lack of evidence for the "language as a tool of thought" hypothesis**: Nearly all challenging mathematics benchmarks are English-only, leaving the relationship between multilingual reasoning and English-based thinking unclear.

**Data contamination risk**: State-of-the-art models may have been exposed to English benchmarks, and multilingual variants lack data isolation guarantees.

**Key gap**: Does the choice of language (for thinking and output) affect reasoning performance? How much thinking does each language require?

## Method

### Overall Architecture
**Two-dimensional difficulty stratification** for scalable benchmark construction:
- **Reasoning depth** (IQ dimension): K-12 → High School → Olympiad → Frontier
- **Knowledge breadth** (coverage dimension): ⭐ to ⭐⭐⭐⭐

### Key Designs
**1. Difficulty Stratification System** — based on cognitive complexity rather than problem source

| Difficulty | Reasoning Depth | Knowledge Breadth | Problem Type | Examples |
|:---:|:---:|:---:|:---|:---|
| Low ⭐ | ⭐ | ⭐ | K-12 math, MWP | Basic algebra, probability |
| Medium ⭐⭐ | ⭐⭐ | ⭐⭐ | College entrance / university exercises, preliminary competitions | AMC, competition preliminaries |
| High ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | Preliminary competition problems, critical thinking | AIME, CNMO |
| Top ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐→∞ | IMO shortlist, frontier mathematics | IMO/Putnam, HLE |

**2. Multilingual Coverage and Translation Quality**:
- **18-language coverage**: >75% of global first-language speakers, including high-resource languages (English, Chinese, Spanish) and low-resource languages (Bengali, Urdu, etc.)
- **Human translation**: LLM-based automatic translation is rejected; native-speaker experts verify each problem individually to ensure logical and terminological accuracy.

**3. Difficulty-Weighted Accuracy Metric**:
$$\text{Difficulty-Weighted Accuracy} = \frac{1}{N}\sum_{i=1}^N\frac{\text{acc}_i}{w_i}$$
where $w_i\in\{1, 1.5, 2.5, 4\}$ according to difficulty level, preventing easy problems from dominating the overall score.

## Key Experimental Results

### Multilingual Performance of State-of-the-Art Models — Difficulty-Weighted Accuracy


### Main Results

| Model | English | Chinese | Spanish | Bengali | Japanese | Cross-lingual Avg. | Max–Min Gap |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen3-235B-A22B-Think** | 58.2 | 57.3 | 52.1 | 44.8 | 51.6 | **54.6** | 13.4 |
| **Gemini-2.5-pro** | 56.1 | 51.2 | 50.4 | 42.1 | 48.3 | **52.2** | 14.0 |
| **o1-preview** | 62.5 | 55.8 | 49.2 | 38.6 | 46.9 | **50.6** | 23.9 |
| GPT-4o | 55.3 | 48.1 | 47.2 | 35.4 | 42.1 | 45.6 | 20.0 |

### Per-Difficulty-Level Accuracy


### Ablation Study

| Difficulty | Low (⭐) | Medium (⭐⭐) | High (⭐⭐⭐) | Top (⭐⭐⭐⭐) |
|:---:|:---:|:---:|:---:|:---:|
| Qwen3-Think | 72% | 61% | 41% | **~40%** |
| Gemini-2.5 | 68% | 58% | 38% | **~40%** |
| o1-preview | 75% | 62% | 44% | **~44%** |

### Language Consistency Analysis — Input–Output Language Alignment

| Model Type | Forced Input Language | Forced Thinking Language | Natural Output Language Deviation |
|:---:|:---:|:---:|:---|
| Non-reasoning models (instruction-tuned) | 95% compliance | — | High fidelity |
| **Reasoning models** | **Surface-level compliance only** | **Strong bias toward English** | **30–50% reversal** |
| Qwen-QwQ | 68% (Chinese input) | English-first | 56% of answers in English |
| o1-preview | 52% (Chinese input) | English-first | 71% of answers in English |

### Key Findings
1. **Cross-lingual performance cliff**: The largest gap reaches 23.9 points (o1 in English vs. Bengali), indicating non-isotropic capabilities that disadvantage low-resource languages.
2. **Heterogeneous thinking length**: Non-reasoning models show less than 2% variation across languages, whereas reasoning models exhibit over 20% variation, with English consistently producing longer chains (25%–50% longer).
3. **Double-edged effect of language forcing**: Forcing English thinking improves accuracy (Chinese-input cases: +5–8%), but reduces framework consistency; forcing the target language degrades performance, revealing the underlying "English-first" nature of these models.
4. **High-difficulty ceiling risk**: Top-level accuracy converges around 40%; even human mathematicians cannot achieve perfect scores, yet the gap between state-of-the-art models and average human performance suggests room for further improvement.

## Highlights & Insights
1. **Benchmark design innovation**: The two-dimensional stratification (reasoning depth × knowledge breadth) better captures cognitive complexity than single-tier difficulty schemes, yielding finer-grained difficulty calibration.
2. **In-depth multilingual analysis**: The work goes beyond performance comparison to investigate the language–thought interaction mechanism, uncovering an English-biased thinking phenomenon in reasoning models.
3. **Commitment to data quality**: Human verification avoids translation errors introduced by LLMs; expert review across 18 languages represents an industry-leading quality standard.
4. **Practical implications**: Language control may improve multilingual reasoning, providing a new direction for fine-tuning frontier models.

## Limitations & Future Work
1. The sample size (500 problems) is relatively small, limiting the scope for fine-grained cross-lingual analyses (e.g., by language family).
2. Coverage of frontier-level problems (HLE) is limited (only 25 items), resulting in uneven representation at the top difficulty tier.
3. The effects of code-mixed language and symbol-intensive (purely formal) mathematical notation remain unexplored.

## Related Work & Insights
- Multilingual NLP benchmarks (MGSM, P-MMeval) and cross-lingual transfer
- Multilingual adaptation of large-scale reasoning models (o1 / DeepSeek-R1)
- Scaling laws relating reasoning length to performance

## Rating
⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Note 6: Self-Evaluating LLMs - Step-Level Confidence Estimation for Multi-Step Tasks](value-guided_search_for_efficient_chain-of-thought_reasoning.md)
- [\[NeurIPS 2025\] Note 1: Is CoT a Hallucination? A Data Distribution Perspective](is_chain-of-thought_reasoning_of_llms_a_mirage_a_data_distribution_lens.md)
- [\[NeurIPS 2025\] Unlocking Multimodal Mathematical Reasoning via Process Reward Model](unlocking_multimodal_mathematical_reasoning_via_process_reward_model.md)
- [\[NeurIPS 2025\] The Hawthorne Effect in Reasoning Models: Evaluating and Steering Test Awareness](the_hawthorne_effect_in_reasoning_models_evaluating_and_steering_test_awareness.md)
- [\[NeurIPS 2025\] Beyond Chemical QA: Evaluating LLM's Chemical Reasoning with Modular Chemical Operations](beyond_chemical_qa_evaluating_llms_chemical_reasoning_with_modular_chemical_oper.md)

<!-- RELATED:END -->
