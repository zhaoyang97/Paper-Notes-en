---
title: >-
  [Paper Note] Right Answer, Wrong Score: Uncovering the Inconsistencies of LLM Evaluation in Multiple-Choice QA
description: >-
  [ACL 2025][LLM Evaluation][Multiple-Choice QA Evaluation] This study systematically exposes inconsistencies in LLM evaluation within Multiple-Choice Question Answering (MCQA). It demonstrates that different combinations of evaluation strategies (RegEx, Logprobs, xFinder) and prompting setups (constrained vs. free generation) lead to substantial discrepancies in reported model performance. Furthermore, even state-of-the-art LLM-based answer extractors fail to reliably identify…
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Multiple-Choice QA Evaluation"
  - "LLM Evaluation Consistency"
  - "Answer Extraction Strategies"
  - "CoT Reasoning"
  - "Adversarial Evaluation"
date: 2026-05-08
content_hash: fa383b5ac3bff0c5
---

# Right Answer, Wrong Score: Uncovering the Inconsistencies of LLM Evaluation in Multiple-Choice QA

**Conference**: ACL 2025  
**arXiv**: [2503.14996](https://arxiv.org/abs/2503.14996)  
**Authors**: Francesco Maria Molfese, Luca Moroni, Luca Gioffré, Alessandro Scirè, Simone Conia, Roberto Navigli (Sapienza University of Rome / Babelscape)
**Code**: [github.com/SapienzaNLP/mcqa-eval](https://github.com/SapienzaNLP/mcqa-eval)  
**Area**: LLM Evaluation  
**Keywords**: Multiple-Choice QA Evaluation, LLM Evaluation Consistency, Answer Extraction Strategies, CoT Reasoning, Adversarial Evaluation

## TL;DR

This study systematically exposes inconsistencies in LLM evaluation within Multiple-Choice Question Answering (MCQA). It demonstrates that different combinations of evaluation strategies (RegEx, Logprobs, xFinder) and prompting setups (constrained vs. free generation) lead to substantial discrepancies in reported model performance. Furthermore, even state-of-the-art LLM-based answer extractors fail to reliably identify reasoning contradictions, highlighting the urgent need for standardized evaluation protocols.

## Background & Motivation

### Background
MCQA is one of the most widely used tasks for evaluating LLMs, covering areas such as commonsense reasoning, scientific knowledge, and multi-domain challenges. While evaluation seems straightforward—the model simply selects an answer from predefined options—the introduction of techniques like Chain-of-Thought (CoT) prompts models to generate substantial free-text reasoning before outputting the final answer. This makes extracting the intended answer from the model's output complex and unreliable.

### Limitations of Prior Work
- Prior studies have focused on "task format" issues, such as option ordering bias and label binding bias, but have rarely systematically analyzed the reliability of the **evaluation strategies themselves**.
- RegEx-based methods exhibit high miss rates when models generate complex reasoning chains, while Logprobs-based methods cannot handle free-text generation.
- Although existing LLM-based answer extractors (e.g., xFinder) perform better, their systematic failure modes have not been thoroughly investigated.
- There is a lack of work providing a comprehensive alignment analysis between automatic evaluation and **human judgment**.

### Core Motivation
With models increasingly adopting test-time scaling and "thinking" mechanisms (e.g., DeepSeek-R1), free-generation reasoning text has become the prevailing trend. The reliability of evaluation strategies directly impacts fair comparisons between models. This study aims to answer: How reliable are existing evaluation methods, and where do they diverge from human judgment?

## Method

### Overall Architecture
Our study designs a three-dimensional analysis framework to investigate the impact of the combination of **Evaluation Strategy $\times$ Prompting Setup $\times$ Benchmark Domain** on reported performance under controlled variables:

- **Evaluation Strategy**: RegEx (18 regular expression patterns), Logprobs (first-token probability), xFinder-Llama (8B) / xFinder-Qwen (500M).
- **Prompting Setup**: Zero-Shot (ZS), Zero-Shot CoT (ZS-CoT), Zero-Shot Constrained (ZS-Const), and Few-Shot (FS).
- **Benchmark Dataset**: MMLU-Redux (5,700 questions across 57 domains), OpenBookQA, and ARC-Challenge.
- **Evaluated Models**: 8 LLMs spanning 1B to 8B parameter scales (Llama series, Phi series, Qwen, Mistral, and SmolLM2).

### Key Design 1: Human Annotation and Consistency Analysis
From all the outputs of the 8 models $\times$ 4 prompting setups on MMLU-Redux, 1,000 instances were randomly sampled, and 4 annotators extracted the intended answers. The annotation options included labels A–D or `[No valid answer]` (used for cases of reasoning contradictions, label binding issues, refusals to answer, etc.). A subset of 200 shared samples was used to calculate inter-annotator agreement (Cohen's kappa = 98.5), establishing a highly reliable gold standard.

### Key Design 2: Adversarial Dataset MMLU-Adversarial
To target two typical failure modes of xFinder, specialized adversarial datasets were constructed:
1. **Reasoning Inconsistency**: The reasoning chain supports answer C, but the final conclusion states "Answer: A"—Gemini-1.5-Flash was used to retain the original reasoning while replacing the final answer.
2. **Recall/Multi-Answer Conflict**: The model provides "correct" arguments for multiple options—prompting the model to generate plausible arguments for multiple choices.

Each mode contains 1,000 high-quality, human-verified instances to test the robustness of answer extractors.

### Key Design 3: Detecting "Problem-Solving" Tendencies in Answer Extractors
Three adversarial prompts were designed to deliberately generate ambiguous outputs, testing whether xFinder bypasses the answer extraction task to directly "solve the problem" itself. Two metrics were defined:
- **Adversarial Rate**: The proportion of times xFinder assigns a valid label instead of `[No valid answer]`.
- **Relative Accuracy**: The proportion of times the label selected by xFinder matches the ground truth.

## Key Experimental Results

### Table 1: Agreement between Evaluation Strategies and Human Judgment (Cohen's kappa)

| Evaluation Strategy | ZS | ZS-CoT | ZS-Const | FS | Average |
|---------|-----|--------|----------|-----|------|
| RegEx | 90.7 | 84.3 | 97.9 | 97.3 | 92.5 |
| Logprobs | 74.7 | — | 94.1 | 90.4 | 86.4 |
| xFinder-Llama | 95.8 | 89.7 | 98.4 | 97.3 | 95.3 |
| xFinder-Qwen | 94.8 | 90.3 | 98.4 | 97.3 | 95.2 |
| Human | 98.2 | 97.0 | 98.7 | 100.0 | 98.5 |

*Key Finding*: Going from constrained prompts (ZS-Const) to free-form generation (ZS), agreement drops substantially across all automatic strategies. Agreement for xFinder-Llama decreases by 2.6%, RegEx by 7.2%, and Logprobs by 19.2%, whereas human annotation agreement only drops by 0.5%.

### Table 2: Detection Capability of xFinder on Adversarial Datasets

| Failure Mode | xFinder-Llama Correct Marking Rate | xFinder-Qwen Correct Marking Rate |
|---------|----------------------|----------------------|
| Reasoning Inconsistency | 1.9% | 0.6% |
| Multi-Answer Conflict | 10.9% | 0.9% |

xFinder fails almost entirely to recognize reasoning inconsistencies and multi-answer conflicts, indicating that it fundamentally performs "final answer matching" rather than understanding the reasoning chain.

### Table 3: "Problem-Solving" Tendency of xFinder (MMLU-Redux)

| Adversarial Prompt | xFinder-Llama Adversarial Rate / Relative Accuracy | xFinder-Qwen Adversarial Rate / Relative Accuracy |
|---------|------------------------------|------------------------------|
| Prompt A | 58.9% / 68.2% | 45.7% / 29.3% |
| Prompt B | 54.0% / 69.6% | 43.2% / 28.0% |
| Prompt C | 15.3% / 74.8% | 96.9% / 23.1% |

On OBQA, the relative accuracy of xFinder-Llama reaches up to 89.9%, strongly suggesting that it may covertly leverage the base model's knowledge to "solve the problem" while performing "answer extraction."

## Key Findings

1. **Fundamental Trade-off between Constrained Prompting and Free Generation**: Constrained prompting (ZS-Const/FS) makes answer extraction more reliable but may suppress reasoning; free generation (ZS/ZS-CoT) enhances actual model capabilities but renders evaluation unreliable.
2. **Domain Sensitivity**: STEM-related tasks benefit the most under free-generation settings (with ZS scoring 2.5 accuracy points higher than ZS-Const); humanities tasks remain stable across configurations, relying more on factual recall than reasoning.
3. **Diminishing Returns of Answer Offset**: Longer model outputs (i.e., larger character offsets to the answer) correlate with higher accuracy, but increasing from $10^2$ to $10^3$ characters yields only a 0.6% improvement, demonstrating diminishing marginal returns.
4. **xFinder is Not a Pure Answer Extractor**: Under adversarial settings, it bypasses the extraction functionality to solve the problem directly, thereby inheriting the biases of its base model.

## Highlights & Insights

- **First Comprehensive Alignment Analysis**: Systematically cross-references 4 automatic evaluation strategies and 4 prompting setups against human judgment, providing a reliable baseline with 1,000 human-annotated samples.
- **Precise Problem Formulation**: Instead of generic claims that "MCQA evaluation is problematic," this work accurately identifies specific failure modes (reasoning inconsistency, multi-answer conflict) and quantifies their impact via custom adversarial datasets.
- **High Practical Impact**: MCQA scores reported in current leaderboards and papers may undergo systematic biases due to differing evaluation strategies, directly affecting fair comparisons across models.
- **MMLU-Adversarial Dataset**: Provides an off-the-shelf benchmark for future studies, facilitating the development of more robust answer extraction methods.
- **Exposing "Role Confusion" in LLM-based Evaluators**: Key insight on a design level indicating that answer extractors should not possess problem-solving capabilities, as doing so introduces systematic biases.

## Limitations & Future Work

- **English Only**: The study does not cover multilingual or cross-lingual scenarios, where answer extraction for multilingual LLMs might face even greater challenges.
- **Unexplored Large-Scale Models**: Only models with 1B-8B parameters were evaluated; larger models (>8B, e.g., Llama-70B, GPT-4) were not covered.
- **Limited Number of Benchmarks**: The work covers only 3 MCQA datasets and does not address more complex reasoning tasks or knowledge-intensive benchmarks.
- **Adversarial Dataset Dependency on Gemini**: The "reasoning inconsistency" pattern in MMLU-Adversarial was generated by Gemini-1.5-Flash, which might introduce bias intrinsic to the generator model.
- **Lack of Evaluation on Latest Reasoning Models**: Models utilizing test-time scaling, such as DeepSeek-R1 or o1, were not evaluated, whose outputs can be significantly longer and more complex.

## Related Work & Insights

- **Robinson et al. (2023)**: Investigated symbol binding issues in MCQA, where models show inconsistent performance under different option permutations. This work further finds that this issue harms the reliability of answer extraction.
- **Zheng et al. (2024)**: Documented positional bias in LLMs. In contrast, this study focuses on the evaluation side rather than the model side.
- **Wang et al. (2024a)**: Found that first-token probabilities do not match textual answers. This work systematically quantifies the scale and impact of this mismatch.
- **Yu et al. (2024) xFinder**: The primary object of comparison. Although xFinder is a state-of-the-art answer extractor, this study uncovers its vulnerability under adversarial scenarios and its "problem-solving" tendencies.
- **Insights**: Evaluation methods themselves must be evaluated—a meta-level research direction. Future work may need to introduce mechanisms such as reasoning chain consistency detection and multi-answer conflict recognition.

## Rating

- Novelty: ⭐⭐⭐⭐ — The first to systematically align MCQA evaluation strategies with human judgments and quantify failure modes using an adversarial dataset.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive cross-evaluations spanning 8 models $\times$ 4 prompting setups $\times$ 3 datasets $\times$ 4 strategies, augmented with 1,000 human annotations.
- Writing Quality: ⭐⭐⭐⭐⭐ — Structurally progressive across 4 RQs, supported by detailed charts and statistics, with highly logical argumentation.
- Value: ⭐⭐⭐⭐ — Offers direct, cautionary insights for current LLM evaluation practices, while the MMLU-Adversarial dataset will benefit future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] WiCkeD: A Simple Method to Make Multiple Choice Benchmarks More Challenging](wicked_a_simple_method_to_make_multiple_choice_benchmarks_more_challenging.md)
- [\[ACL 2026\] BenchMarker: An Education-Inspired Toolkit for Highlighting Flaws in Multiple-Choice Benchmarks](../../ACL2026/llm_evaluation/benchmarker_an_education-inspired_toolkit_for_highlighting_flaws_in_multiple-cho.md)
- [\[ICLR 2026\] TrustJudge: Inconsistencies of LLM-as-a-Judge and How to Alleviate Them](../../ICLR2026/llm_evaluation/trustjudge_inconsistencies_of_llm-as-a-judge_and_how_to_alleviate_them.md)
- [\[ACL 2025\] HomeBench: Evaluating LLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multiple Devices](homebench_evaluating_llms_in_smart_homes_with_valid_and_invalid_instructions_acr.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)

</div>

<!-- RELATED:END -->
