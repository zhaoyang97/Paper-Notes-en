---
title: >-
  [Paper Note] NegVQA: Can Vision Language Models Understand Negation?
description: >-
  [ACL 2025][Multimodal VLM][Negation understanding] Proposes the NegVQA benchmark (7,379 binary-choice VQA questions) to systematically evaluate the negation understanding capabilities of 20 VLMs, revealing a sharp performance drop across all models (averaging 29.7%) and uncovering a "U-shaped" scaling trend.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Negation understanding"
  - "VQA benchmark"
  - "Vision-language models"
  - "Scaling trends"
  - "Diagnostic evaluation"
date: 2026-05-08
content_hash: 58deed72058b7b22
---

# NegVQA: Can Vision Language Models Understand Negation?

**Conference**: ACL 2025  
**Code**: [https://yuhui-zh15.github.io/NegVQA/](https://yuhui-zh15.github.io/NegVQA/)  
**Area**: Multimodal VLMs  
**Keywords**: Negation understanding, VQA benchmark, Vision-language models, Scaling trends, Diagnostic evaluation

## TL;DR

Proposes the NegVQA benchmark (7,379 binary-choice VQA questions) to systematically evaluate the negation understanding capabilities of 20 VLMs, revealing a sharp performance drop across all models (averaging 29.7%) and uncovering a "U-shaped" scaling trend.

## Background & Motivation

- **Negation is a fundamental linguistic phenomenon**: A single "not" can completely reverse the meaning of a sentence, such as "Who wrote this book?" $\rightarrow$ "Who did **not** write this book?"
- **VLMs are deployed in high-risk scenarios**: In systems like robotics and embodied AI, the failure of a model to correctly understand negative instructions (e.g., "do not do X") can lead to dangerous actions contrary to user intent.
- **Existing VQA datasets are almost exclusively affirmative**: Current benchmarks do not systematically test negation understanding, lacking targeted diagnostic tools.
- **Prior research exists on the text side, but the vision side remains blank**: Previous works primarily evaluate the negation capabilities of LLMs and CLIP, leaving a gap in systematic evaluations for generative VLMs.

## Method

### 1. Data Construction Pipeline

NegVQA is built upon VMCBench (a multiple-choice VQA benchmark) and involves two core steps:

**Step 1: Negation Question Generation**

Uses GPT-4o to transform original questions into their negated versions while retaining syntactic structure and semantics. For each original question, the LLM determines if it is negatable (e.g., "Find the value of x" cannot be meaningfully negated), ultimately filtering out 7,379 questions from an initial pool of 9,018. Human verification on 100 samples shows a conversion accuracy of 97%.

**Step 2: Answer Option Adjustment**

Reduces the original four-option multiple-choice format to a binary-choice format: keeps the correct answer and a randomly chosen incorrect answer, then **reverses their correctness**. Formally represented as:

$$
\text{NegVQA}(q, a^+, a^-) = (\text{Negate}(q), a^-, a^+)
$$

where $a^+$ is the original correct answer and $a^-$ is the original incorrect answer. After negation, $a^-$ becomes the correct answer, and $a^+$ becomes the incorrect answer. This ensures that the model must truly understand negation to answer correctly.

### 2. Dataset Coverage

NegVQA covers four major domains across 20 VQA datasets:

- **General VQA**: VQAv2, OKVQA, MMVet, VizWiz, A-OKVQA, MMStar, SEEDBench
- **Reasoning Tasks**: MathVision, GQA, MMMU, RealWorldQA, MathVista, ScienceQA
- **OCR**: OCRVQA, TextVQA
- **Document/Chart**: DocVQA, InfoVQA, ChartQA, TableVQABench, AI2D

Negation types cover object absence, attribute negation (color/size/position), action negation (unoccurred events), and complex negations requiring deep reasoning.

### 3. Theoretical Explanation of the U-Shaped Scaling Trend

The authors model the performance on NegVQA as a composition of two capabilities:

$$
\text{Perf}_{\text{NegVQA}}(s) = f_{\text{QA}}(s) \cdot g_{\text{neg}}(s)
$$

where $s$ is the model scale, $f_{\text{QA}}(s)$ is the original VQA capability (monotonically increasing with scale), and $g_{\text{neg}}(s)$ is the negation understanding capability (resembling a tanh curve, which is initially flat and then rises sharply).

- **Small models**: Weak reasoning capability, leading to random guessing on both negated and non-negated questions, with similar performance.
- **Medium models**: Improved original VQA capability, but still ignoring negation terms and treating negative questions as affirmative, which paradoxically degrades performance.
- **Large models**: Negation understanding is activated once the scale is sufficiently large, allowing performance to recover and improve.

## Key Experimental Results

### Table 1: Average performance of various model families on NegVQA

| Model | PosVQA (%) | NegVQA (%) | Performance Drop |
|------|-----------|-----------|---------|
| Qwen2-VL-72B | 92.2 | 72.7 | -19.5 |
| Molmo-72B | 87.5 | 74.5 | -13.0 |
| Qwen2-VL-7B | 88.8 | 57.2 | -31.6 |
| Cambrian-34B | 87.4 | 59.9 | -27.5 |
| VILA1.5-40B | 85.7 | 56.6 | -29.1 |
| VILA1.5-8B | 78.5 | 56.2 | -22.3 |
| Molmo-7B-D | 83.0 | 55.3 | -27.7 |
| Cambrian-8B | 83.8 | 55.7 | -28.1 |
| Qwen2-VL-2B | 85.4 | 53.4 | -32.0 |
| LLaVA-1.5-7B | 73.3 | 47.9 | -25.4 |
| LLaVA-1.5-13B | 74.3 | 40.3 | -34.0 |
| DeepSeek-VL-7B | 79.8 | 41.9 | -37.9 |
| DeepSeek-VL-1.3B | 75.0 | 37.2 | -37.8 |
| InstructBLIP-7B | 55.3 | 28.9 | -26.4 |
| InstructBLIP-13B | 67.0 | 35.2 | -31.8 |
| Human Baseline | — | 89.0 | — |

### Table 2: NegVQA performance by domain (best model Qwen2-VL-72B)

| Domain | PosVQA (%) | NegVQA (%) | Performance Drop |
|------|-----------|-----------|---------|
| General | 93.6 | 71.7 | -21.9 |
| Reasoning | 83.4 | 64.1 | -19.3 |
| OCR | 99.0 | 91.8 | -7.2 |
| Doc & Chart | 94.8 | 72.4 | -22.4 |
| Average | 92.2 | 72.7 | -19.5 |

## Key Findings

1. **All VLMs struggle with negation**: There is an average performance drop of 29.7 percentage points; the best-performing model, Qwen2-VL-72B, is still 16.3% below the human baseline (89%).
2. **U-shaped scaling trend**: As model size increases, NegVQA performance first decreases and then increases, a trend most prominent in the Cambrian and VILA families.
3. **Lack of negation samples in training data**: Only 1.1% of conversations in the LLaVA fine-tuning data contain the word "not", which is likely the core cause.
4. **OCR tasks are least affected**: This is likely because OCR tasks focus more on text matching than on semantic reasoning.
5. **Reasoning and document tasks display a more pronounced U-shaped effect**: These tasks require deeper semantic understanding, making them more sensitive to negation misunderstandings.
6. **Model scale is not a silver bullet**: LLaVA-1.5-13B (40.3%) performing worse than LLaVA-1.5-7B (47.9%) illustrates the downward phase of the U-shaped curve.

## Highlights & Insights

- **Simple yet effective benchmark design**: Constructs a high-quality diagnostic dataset at a low cost through LLM automatic negation + binary-choice reversal, achieving a 97% conversion accuracy.
- **U-shaped scaling discovery**: Reveals that negation understanding may be an emergent capability, providing a new perspective on VLM scaling research.
- **Broad coverage**: Evaluation spans 20 datasets, 4 main domains, and 20 models across 7 model families, ensuring the generalizability of the findings.
- **Strong safety warnings**: Highlights the severe potential risks of negation understanding failures in embodied AI scenarios.

## Limitations & Future Work

- **Limited to multiple-choice format**: The binary-choice format may not fully reflect negation understanding capabilities in open-ended scenarios.
- **Zero-shot evaluation only**: The potential of few-shot prompting to improve negation understanding remains unexplored.
- **No mitigation strategies proposed**: The paper diagnoses the problem but does not provide specific training resolutions (only suggests data augmentation).
- **Lack of granular analysis on negation types**: Does not categorize and compare different forms of negation (lexical vs. syntactic vs. implicit negation).
- **GPT-4o conversion bias**: The 3% error rate may introduce systematic noise into large-scale data.

## Related Work & Insights

- **VQA Benchmarks**: VQAv2, OKVQA, MMMU, and VMCBench provide the raw evaluation data.
- **Negation Understanding (NLP side)**: Kassner & Schütze (2020) evaluate negation in PLMs; Zhang et al. (2023) discover a U-shaped negation scaling in language models.
- **Negation Understanding (Vision side)**: Alhamoud et al. (2025) and Singh et al. (2024) evaluate CLIP's negation understanding, but lack evaluations on generative VLMs.
- **Scaling Laws**: Kaplan et al. (2020) on LM scaling laws; McKenzie et al. (2023) on inverse scaling; Wei et al. (2022) on U-shaped scaling.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐ |
| Technical Depth | ⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

**Overall**: ⭐⭐⭐⭐ — A diagnostic benchmark paper with comprehensive experiments and insightful findings (U-shaped scaling), posing an important warning for the safe deployment of VLMs. While the technical depth is relatively limited, the topic selection is precise and the conclusions are solid.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Vision-Language Models Do Not Understand Negation](../../CVPR2025/multimodal_vlm/vision-language_models_do_not_understand_negation.md)
- [\[ACL 2025\] Can Vision Language Models Understand Mimed Actions?](can_vision_language_models_understand_mimed_actions.md)
- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](spatialmqa_mllm_spatial_relations.md)
- [\[ACL 2025\] CORDIAL: Can Multimodal Large Language Models Effectively Understand Coherence Relations?](cordial_can_multimodal_large_language_models_effectively_understand_coherence_re.md)
- [\[ACL 2025\] Can MLLMs Understand the Deep Implication Behind Chinese Images?](can_mllms_understand_the_deep_implication_behind_chinese_images.md)

</div>

<!-- RELATED:END -->
