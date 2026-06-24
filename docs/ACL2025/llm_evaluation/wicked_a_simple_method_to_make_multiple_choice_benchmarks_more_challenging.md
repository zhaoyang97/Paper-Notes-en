---
title: >-
  [Paper Note] WiCkeD: A Simple Method to Make Multiple Choice Benchmarks More Challenging
description: >-
  [ACL 2025][LLM Evaluation][Multiple-Choice Benchmarks] This paper proposes the WiCkeD method, which increases the difficulty of existing benchmarks by randomly replacing one option of multiple-choice questions with "None of the above". This leads to an average performance drop of 12.1 percentage points across 18 LLMs, and even Chain-of-Thought reasoning cannot compensate for this decrease.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Multiple-Choice Benchmarks"
  - "None of the Above"
  - "Benchmark Saturation"
  - "Reasoning Evaluation"
  - "Wild-Card Distractor"
date: 2026-05-08
content_hash: efab2d0981fd0e9c
---

# WiCkeD: A Simple Method to Make Multiple Choice Benchmarks More Challenging

**Conference**: ACL 2025  
**arXiv**: [2502.18316](https://arxiv.org/abs/2502.18316)  
**Code**: [github.com/ahmedselhady/wicked-benchmarks](https://github.com/ahmedselhady/wicked-benchmarks)  
**Area**: LLM Evaluation  
**Keywords**: Multiple-Choice Benchmarks, None of the Above, Benchmark Saturation, Reasoning Evaluation, Wild-Card Distractor

## TL;DR

This paper proposes the WiCkeD method, which increases the difficulty of existing benchmarks by randomly replacing one option of multiple-choice questions with "None of the above". This leads to an average performance drop of 12.1 percentage points across 18 LLMs, and even Chain-of-Thought reasoning cannot compensate for this decrease.

## Background & Motivation

Multiple-choice question (MCQ) benchmarks are the mainstream approach for evaluating LLMs, but they face several challenges:

**Benchmark Saturation**: Mainstream benchmarks such as MMLU and ARC-Challenge are rapidly saturated in the LLM era, making it difficult to distinguish differences in model capabilities.

**High Cost of Building New Benchmarks**: Creating high-quality benchmarks requires substantial human annotation and verification.

**Limited Existing Enhancement Methods**: MMLU-Pro increases difficulty by expanding the number of distractors, but creating plausible distractors is inherently difficult and often requires human validation.

Prior studies have identified inherent limitations in MCQ evaluations:
- Some LLMs can answer correctly by looking only at the options (without reading the questions).
- LLMs exhibit prior preferences for specific answer keys (A/B/C/D).
- LLMs cannot effectively identify cases where they lack knowledge.

The field of educational assessment has long utilized the "None of the Above" (NOTA) option to increase test difficulty and encourage deep thinking, which inspires the WiCkeD method.

## Method

### Overall Architecture

WiCkeD (Wild-Card Distractor) is an automated method that can transform any MCQ benchmark into a more challenging version:
- Keeps the questions unchanged.
- Randomly replaces one option with "None of the above".
- Does not increase the number of options (replacement rather than addition).

Core hypothesis: Detecting that the correct answer is not among the options is more difficult than selecting the correct answer from existing options.

### Key Designs

**WiCkeD Algorithm**:
Given a benchmark containing $M$ samples, with each sample having $N$ options (1 correct + $N-1$ distractors):
1. Uniformly sample one option to be replaced.
2. Append the "None of the above" option (replacing the sampled option).
3. If the correct option is replaced, the new correct answer becomes "None of the above".
4. If a distractor is replaced, the original correct answer remains unchanged.

**SBA/SCA Categorization**:
Not all replacements yield consistent WiCkeD samples. When a "Single Best Answer" (SBA) exists rather than a "Single Correct Answer" (SCA):
- If the best answer is removed, the second-best answer becomes correct, but the algorithm would incorrectly label NOTA as correct.
- Solution: Train a BERT classifier to detect SBA samples (yielding a 98.9% recall rate), and copy SBA samples to the WiCkeD version without modification.

**Classifier Training**:
- Sample 4,000 instances from 4 benchmarks (MMLU, MMLU-Pro, TruthfulQA, CommonsenseQA).
- Use GPT-4o-mini to automatically label SBA/SCA.
- Use 75% for training the BERT classifier and 25% for evaluation (including human annotation).
- SBA proportion: ~20% in MMLU/MMLU-Redux/MMLU-Pro, and $<$ 5% in other benchmarks.

**Handling Randomness**:
- Since the option replacement is random, 5 WiCkeD variants are generated for each benchmark.
- Report mean and standard deviation.

### Loss & Training

WiCkeD itself does not involve model training. The BERT classifier for SBA/SCA detection uses standard binary cross-entropy loss.

## Key Experimental Results

### Main Results

**WiCkeD Performance Drop of 18 Models across 6 Benchmarks**:

| Model | Original Avg. | WiCkeD Avg. | Drop $\Delta$ |
|------|---------|-----------|--------|
| Qwen-2.5-72B | 84.6 | 72.6 | -12.0 |
| Qwen-2.5-72B-IT | 82.6 | 69.3 | -13.3 |
| Llama-3.1-70B-IT | 77.1 | 64.5 | -12.6 |
| Qwen-2.5-7B | 74.7 | 54.9 | **-19.7** |
| DS-R1-Qwen7B | 60.8 | 53.4 | **-7.3** |
| Total Avg. | 70.78 | 58.52 | -12.2 |

Key Findings:
- All models exhibit a significant drop in performance (7.2 to 19.7 percentage points).
- **Qwen-2.5-7B suffers the largest drop (19.7%)**, whereas its DeepSeek-R1 distilled counterpart only drops by 7.3%.
- WiCkeD **shuffles model rankings**: Qwen-2.5-7B is originally close to Llama-3.1-70B, but lags behind by 13% on WiCkeD.

**WiCkeD reveals hidden capability differences among models**:
- Gemma-2-9B-IT and Gemma-2-27B-IT trail Llama-3.1-70B by 9.5% and 5.3%, respectively, on WiCkeD.
- Instruction tuning offers no consistent advantage, with performance varying across model families.

### Ablation Study

**Chain-of-Thought (CoT) Experiments** (MMLU, MMLU-Pro, MMLU-Redux):

| Model | Direct WiCkeD $\Delta$ | CoT WiCkeD $\Delta$ |
|------|-------------|-------------|
| DS-R1-Llama 8B | -4.1 | -2.0 |
| DS-R1-Qwen 7B | -4.3 | -2.5 |
| Llama-3.1-8B | -3.2 | -5.8 |
| Qwen-2.5-7B | -6.9 | -14.6 |
| Gemma-2-9B | -12.2 | -8.9 |
| Avg. | -5.62 | -5.24 |

Key Observations:
- CoT cannot eliminate the performance degradation brought by WiCkeD—there is still an average decrease of ~5%.
- **Reasoning-enhanced models (distilled versions of DeepSeek-R1) are least affected** (~2%), validating that WiCkeD indeed probes reasoning capabilities.
- With instruction tuning + CoT, Qwen-2.5-7B-IT and 14B-IT drop by less than 2%.
- Base models might suffer even larger performance drops under CoT (e.g., Llama-3.1-8B drops from -3.2 to -5.8).

### Key Findings

1. WiCkeD is a zero-cost benchmark enhancement method—fully automated with no human annotation required.
2. It induces a large range of performance drops (7-20%), effectively addressing the benchmark saturation problem.
3. WiCkeD exposes model discrepancies hidden by original benchmarks, particularly in reasoning capabilities.
4. Models with stronger reasoning capabilities (e.g., DeepSeek-R1) are least affected by WiCkeD.
5. "Detecting the absence of the correct answer" indeed constitutes an independent and valuable evaluation dimension.

## Highlights & Insights

1. **Simplistic yet Profound**: A single simple action (replacement with NOTA) exposes deep-seated issues in LLM evaluation.
2. **Inspiration from Educational Assessment**: NOTA is a mature evaluation technique in education, successfully transferred here to the LLM evaluation paradigm.
3. **SBA Detection Mechanism**: Meticulously handles "multiple-correct-answer" edge cases, ensuring the internal consistency of the WiCkeD variants.
4. **Precise Probing of Reasoning Capability**: The magnitude of performance degradation under WiCkeD exhibits a clear negative correlation with model reasoning capability.
5. **Composability**: Can be overlaid on top of any MCQ benchmark, orthogonal to other enhancement methods (e.g., expanding distractor pools).

## Limitations & Future Work

1. The position of "None of the above" is always the last option, which may introduce positional bias.
2. SBA detection relies on GPT-4o-mini annotations, introducing approximately 1.1% noise.
3. Experiments were only conducted on 6 benchmarks and 18 models; this can be extended to more scenarios.
4. Other wild-card options such as "All of the above" are not explored.
5. Educational assessment research indicates that NOTA can reduce examinee confidence—whether a similar effect exists in LLMs remains to be studied.
6. It does not analyze on which types of questions LLMs are more sensitive to WiCkeD.

## Related Work & Insights

- **MMLU-Pro (Wang et al., 2024)**: Enhances difficulty by increasing the number of distractors, but requires human efforts to build plausible distractors.
- **MMLU-Redux (Gema et al., 2024)**: Corrects erroneous questions in MMLU and re-annotates them.
- **Balepur et al. (2024)**: Reveals that LLMs can answer questions by looking only at options—WiCkeD partially mitigates this "shortcut".
- **DiBattista & Fortuna (2014)**: Studies NOTA in educational assessment—WiCkeD introduces this concept to AI evaluation.
- Insight: Innovations in evaluation methods can sometimes be more efficient and insightful than constructing entirely new benchmarks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Elegantly transfers educational assessment techniques to LLM evaluation; the idea is simple yet exquisite.
- **Utility**: ⭐⭐⭐⭐⭐ — Zero-cost automation, immediately applicable to any MCQ benchmark.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 18 models × 6 benchmarks + CoT analysis, 5 random seeds.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, meticulous handling of SBA, and proper experimental presentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] FinanceReasoning: Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging](financereasoning_benchmarking_financial_numerical_reasoning_more.md)
- [\[ACL 2025\] Right Answer, Wrong Score: Uncovering the Inconsistencies of LLM Evaluation in Multiple-Choice QA](right_answer_wrong_score_uncovering_the_inconsistencies_of_llm_evaluation_in_mul.md)
- [\[ACL 2026\] BenchMarker: An Education-Inspired Toolkit for Highlighting Flaws in Multiple-Choice Benchmarks](../../ACL2026/llm_evaluation/benchmarker_an_education-inspired_toolkit_for_highlighting_flaws_in_multiple-cho.md)
- [\[ACL 2025\] CoPrUS: Consistency Preserving Utterance Synthesis Towards More Realistic Benchmark](coprus_consistency_preserving_utterance_synthesis_towards_more_realistic_benchma.md)
- [\[ACL 2025\] CulturalBench: A Robust, Diverse, and Challenging Cultural Benchmark by Human-AI CulturalTeaming](culturalbench_a_robust_diverse_and_challenging_cultural_benchmark_by_human-ai_cu.md)

</div>

<!-- RELATED:END -->
