---
title: >-
  [Paper Note] How Does Response Length Affect Long-Form Factuality
description: >-
  [ACL 2025 (Findings)][LLM Safety][Long-Form Factuality] This paper systematically studies the relationship between LLM response length and factual precision, proposing an efficient two-tier factuality evaluation framework Bafe (which achieves 89.31% agreement with human annotations). It confirms the existence of length bias and proves that "fact exhaustion" is the primary cause of factuality decline by ruling out the error propagation and long context hypotheses.
tags:
  - "ACL 2025 (Findings)"
  - "LLM Safety"
  - "Long-Form Factuality"
  - "Length Bias"
  - "Fact Exhaustion"
  - "Factuality Evaluation"
  - "Hallucination"
date: 2026-05-08
content_hash: 4a82c94c3c273047
---

# How Does Response Length Affect Long-Form Factuality

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2505.23295](https://arxiv.org/abs/2505.23295)  
**Code**: Yes  
**Area**: LLM Safety  
**Keywords**: Long-Form Factuality, Length Bias, Fact Exhaustion, Factuality Evaluation, Hallucination

## TL;DR

This paper systematically studies the relationship between LLM response length and factual precision, proposing an efficient two-tier factuality evaluation framework Bafe (which achieves 89.31% agreement with human annotations). It confirms the existence of length bias and proves that "fact exhaustion" is the primary cause of factuality decline by ruling out the error propagation and long context hypotheses.

## Background & Motivation

**Background**: Large language models (LLMs) are widely used for long-form text generation (e.g., long-form QA, biography generation), but factual errors in the generated text severely undermine their reliability. FActScore and Safe are two mainstream methods for evaluating long-form factuality.

**Limitations of Prior Work**: Regarding the impact of response length on factuality, prior studies have presented contradictory conclusions—some find that longer responses contain more errors, while others find that length does not affect hallucinations. Crucially, no systematic investigation has been conducted on this relationship. Meanwhile, existing evaluation methods have limitations: FActScore relies solely on a single Wikipedia page for verification, leading to insufficient coverage; Safe uses Google Search but requires 28 minutes and $0.5 per fact, which is extremely inefficient.

**Key Challenge**: If length bias indeed exists, deploying LLMs requires a trade-off between information quantity and accuracy. To investigate this issue, an accurate and efficient evaluation tool is a prerequisite.

**Goal**: To address two research questions—RQ1: Does response length affect factual precision (does length bias exist)? RQ2: If it exists, what is its root cause?

**Key Insight**: First construct an efficient evaluation tool, then isolate the causal impact of length on factuality through carefully controlled experiments, and finally identify the root cause by systematically validating three hypotheses.

**Core Idea**: The decline in long-form factuality is not primarily caused by error propagation or long contexts, but by "fact exhaustion"—when continuously generating content on a single topic, the model gradually depletes its reliable knowledge and is forced to introduce uncertain information.

## Method

### Overall Architecture

The overall work is divided into two major parts: (1) Constructing the Bafe evaluation framework—decomposing long-form text into atomic facts and verifying the correctness of each fact through a two-tier verification process (Wikipedia + Google Search); (2) Conducting controlled experiments using Bafe—observing changes in factual precision by varying the requested output length, and then validating three hypotheses individually to find the root cause of the length bias.

### Key Designs

1. **Bafe Two-Tier Factuality Evaluation Framework**:

    - **Function**: Efficiently and accurately evaluate factual precision in long-form text.
    - **Mechanism**: First, an LLM (gpt-3.5-turbo-instruct) decomposes the long-form text into atomic facts (each containing only one piece of information). First-tier verification: compare each atomic fact with retrieved Wikipedia pages, where LLaMA determines if it is supported. Second-tier verification: performed only on facts that failed the first tier—first reformulate them into self-contained statements (resolving coreference issues), and then conduct a single Google Search to find supporting evidence. A fact is determined as incorrect if both tiers fail. $\text{Factual Precision} = \text{Supported Facts} / \text{Total Facts}$.
    - **Design Motivation**: Wikipedia has broad and reliable coverage. Filtering with Wikipedia first can drastically reduce the number of necessary Google Searches, lowering cost and time. Only a single search is conducted instead of Safe's five, as experiments show that multiple search results are highly redundant and can introduce noise. Unnecessary pertinence filtering steps in Safe are removed. Ultimately, Bafe is 7 times cheaper and 4 times faster than Safe, while achieving higher accuracy.

2. **Controlled Experimental Design (Proving Length Bias)**:

    - **Function**: Verify length bias while controlling for confounding factors.
    - **Mechanism**: Experiments are conducted using GPT-4o on two tasks: biography generation and long-form factual description. Key control variable: the response length is varied solely via the system prompt instruction "Generate with around x words" ($x \in \{100, 200, 300, 400, 500\}$), while keeping everything else constant. GPT-4o's strong instruction-following capability is leveraged to ensure effective length control.
    - **Design Motivation**: Without strict control, it is difficult to disentangle the length effect from confounding factors such as topic difficulty or entity rarity.

3. **Three-Hypothesis Validation Experiments**:

    - **Function**: Identify the root cause of length bias.
    - **Mechanism**: (a) **Error Propagation Hypothesis**: dependence within the error sequence is analyzed via autocorrelation, yielding a lag-1 correlation coefficient of only around 0.1 (with larger lags close to zero), indicating that errors do not accumulate; counterfactual analysis (modifying the first sentence's factuality to observe subsequent effects) reveals that first-sentence errors do not affect the factual precision of subsequent text. (b) **Long Context Hypothesis**: a segmented generation experiment is designed (fixing the evaluation segment Topic B as "Career" with 200 words, and varying the context segment Topic A's length from 100 to 500 words), showing that context length does not affect the factuality of the newly generated content. (c) **Fact Exhaustion Hypothesis**: a single-topic setting (generating 400 words for one topic) is compared against a multi-topic setting (generating 200 words each for two topics), evaluating factual precision under the same total word count. The multi-topic setting consistently outperforms the single-topic setting by 2.25% - 2.86%, proving that deep-diving into a single topic exhausts reliable knowledge.
    - **Design Motivation**: Causal inference requires controlled experiments rather than simple correlation analysis. Autocorrelation analysis combined with counterfactual analysis forms a complementary chain of evidence.

### Loss & Training

This paper does not involve model training; the core contributions are the evaluation methodology and experimental analysis.

## Key Experimental Results

### Main Results (Bafe vs. Existing Methods)

| Evaluation Method | Human Agreement | Cost per Response ($) | Time per Response (min) |
|---|---|---|---|
| FActScore | 69.97% | 0.021 | 0.67 |
| Safe | 84.48% | 0.493 | 28.70 |
| **Bafe (Ours)** | **89.31%** | **0.067** | **7.17** |

### Ablation Study (Three-Hypothesis Validation)

| Validation Experiment | Key Results | Conclusion |
|---|---|---|
| Autocorrelation Analysis | lag-1 coefficient $\approx 0.1$, lag > 1 $\approx 0$ | Error propagation has only a minor short-term effect |
| Counterfactual Analysis | Precision of subsequent text after modifying the first sentence: 91.17% vs. original 90.79% | First-sentence errors do not propagate |
| Long Context Experiment | Context 100 $\rightarrow$ 400 words, evaluation segment precision 92.50% $\rightarrow$ 92.26% | Long context does not affect factuality |
| Single vs. Multi-Topic Comparison | Early life + Career: Single 86.02% $\rightarrow$ Multi 88.27% | **Fact exhaustion is the primary cause** |
| Qualitative Analysis | 100-word response is accurate, 200-word response adds untruthful details | Model is forced to fill in uncertain information |

### Key Findings

- **Length bias indeed exists**: In the biography task, factual precision decreases from 94.5% for 100 words to 90.5% for 500 words (a 4% drop); in the long factual description task, it drops from 98.1% to 96.9%.
- **Error propagation is only a superficial phenomenon**: The autocorrelation coefficient is extremely small ($\sim 0.1$), showing only weak dependence between adjacent facts, which is far from sufficient to explain the systematic decline.
- **Long context is not the culprit**: Even when the context reaches 500 words, the factuality of the newly generated content is barely affected (change < 0.3%).
- **Fact exhaustion is the root cause**: When continuously generating on a single topic, the model gradually depletes its reliable knowledge store and is forced to introduce speculative, unverified details. Switching topics can alleviate this issue.
- **Bafe outperforms existing methods across all three dimensions**: Achieving the highest accuracy while maintaining the lowest cost and time, demonstrating the superiority of the two-tier design.

## Highlights & Insights

- **The concept of "fact exhaustion"** precisely characterizes the core problem in LLM long-form generation: models do not make errors randomly, but rather have varying depths in their "knowledge reserve"—outputting high-confidence knowledge first, and as generation continues, being forced to introduce low-confidence information. This insight has far-reaching implications for understanding and mitigating LLM hallucinations.
- **The design philosophy of two-tier verification** is highly elegant: first using low-cost Wikipedia to filter facts with high certainty, and only launching high-cost search verification for uncertain facts. This "funnel-like" design ensures coverage while controlling cost, making it transferable to other scenarios requiring multi-level verification.
- **The causal inference approach in experimental design** is highly instructive: instead of merely observing correlations, hypotheses are proposed individually and ruled out via controlled experiments. The validation logic for the three hypotheses is clear and straightforward.

## Limitations & Future Work

- **Validated only on GPT-4o**: The length control experiments rely on strong instruction-following capabilities, which might not directly apply to open-source models. Future work needs validation across more models.
- **Bafe is only applicable to fact-dense tasks**: Extensions are needed for texts containing subjective judgments, numerical reasoning, etc.
- **Inherent limitations of black-box analysis**: Direct observation of internal knowledge utilization processes within the model is impossible, making "fact exhaustion" more of an empirical observation rather than a mechanical explanation.
- **No mitigation solution proposed**: The paper only diagnoses the issue without providing a solution. An obvious direction is to introduce retrieval augmentation or knowledge refresh mechanisms in long-form generation.
- **Future direction**: Developing long-form generation strategies that are aware of "knowledge boundaries", actively switching topics or halting generation when approaching depletion.

## Related Work & Insights

- **vs. FActScore (Min et al., 2023)**: FActScore only uses a single Wikipedia page, leading to insufficient coverage and coreference resolution issues. Bafe resolves both problems through two-tier verification + self-contained modifications.
- **vs. Safe (Wei et al., 2024)**: Safe uses Google Search five times per fact, which is redundant and expensive. Bafe demonstrates that a single search is sufficient (as multiple searches yield duplicate results), boosting efficiency by 7 times.
- **vs. Hallucination Studies (Zhang et al., 2024)**: Previous snowballing effect studies focused on error propagation in short-form QA, whereas this work shows the propagation effect is very weak in long-form scenarios, pointing to a different root cause.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The concept of "fact exhaustion" is novel and persuasive; the design of the two-tier evaluation framework is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Rigorous controlled experimental design, systematic validation of three hypotheses, with extensive human evaluation and statistical analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ A clear structure driven by research questions, with distinct hypotheses and conclusions for each experiment.
- **Value**: ⭐⭐⭐⭐ Significant contribution to understanding long-form LLM hallucination mechanisms, with Bafe ready for immediate community adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Ewe: Improving Factuality with Explicit Working Memory](improving_factuality_with_explicit_working_memory.md)
- [\[ACL 2025\] Improving Model Factuality with Fine-grained Critique-based Evaluator](improving_model_factuality_with_fine-grained_critique-based_evaluator.md)
- [\[ACL 2025\] Chinese SimpleQA: A Chinese Factuality Evaluation for Large Language Models](chinese_simpleqa_a_chinese_factuality_evaluation_for_large_language_models.md)
- [\[ACL 2025\] Real-time Factuality Assessment from Adversarial Feedback](real-time_factuality_assessment_from_adversarial_feedback.md)
- [\[ICLR 2026\] LLMs Can Hide Text in Other Text of the Same Length](../../ICLR2026/llm_safety/llms_can_hide_text_in_other_text_of_the_same_length.md)

</div>

<!-- RELATED:END -->
