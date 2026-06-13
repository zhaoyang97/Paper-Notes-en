---
title: >-
  [Paper Note] Lost in the Prompt Order: Revealing the Limitations of Causal Attention in Language Models
description: >-
  [ACL 2026][NLP Understanding][Causal Attention] This paper investigates the sensitivity of large language models (LLMs) to the order of prompt components in multiple-choice question answering (MCQA). Through systematic e…
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "Causal Attention"
  - "Prompt Order Sensitivity"
  - "Multiple-Choice Question Answering"
  - "Information Bottleneck"
  - "Mechanistic Interpretation"
date: 2026-05-08
content_hash: 79039f8967555fb6
---

# Lost in the Prompt Order: Revealing the Limitations of Causal Attention in Language Models

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.14152](https://arxiv.org/abs/2601.14152)  
**Code**: None  
**Area**: NLP Understanding / Prompt Sensitivity  
**Keywords**: Causal Attention, Prompt Order Sensitivity, Multiple-Choice Question Answering, Information Bottleneck, Mechanistic Interpretation

## TL;DR

This paper investigates the sensitivity of large language models (LLMs) to the order of prompt components in multiple-choice question answering (MCQA). Through systematic experiments, it rules out the training bias and memory decay hypotheses, revealing that the causal attention mask is the fundamental mechanism causing significant performance degradation in the QOC (Question-Options-Context) order.

## Background & Motivation

**Background**: The sensitivity of LLMs to prompt structures has been widely reported—ranging from the ordering of examples in in-context learning to the permutation of options in multiple-choice questions. However, current research mostly remains at the level of phenomenological description; while it is known "what" affects performance, the "why" remains unclear.

**Limitations of Prior Work**: In the MCQA task, a typical prompt consists of three parts: a context passage (C), a question (Q), and options (O). Intuitively, rearranging these components should not affect performance since the semantic content remains identical. However, experiments show that placing the context before the question and options (CQO) consistently and significantly outperforms the reverse order (QOC), with an average gap exceeding 14 percentage points across 21 decoder-only models and 4 datasets.

**Key Challenge**: The significant performance discrepancy across semantically equivalent prompt permutations poses a severe challenge to the reliability of LLMs. Prior works such as lu2022fantastically and pezeshkpour2024 have reported similar phenomena but failed to conduct deep architectural-level root-cause analysis.

**Goal**: To propose three competing hypotheses and verify or exclude them through carefully designed controlled experiments, ultimately identifying the core mechanism of prompt order sensitivity and designing targeted intervention methods to validate the findings.

**Key Insight**: Starting from the architectural level, the root cause is localized by comparing behavioral differences across decoder-only, encoder-only, and encoder-decoder architectures.

## Method

### Overall Architecture

This paper adopts a "propose hypothesis → design experiment → verify/exclude" research paradigm. It first quantifies the performance gap between CQO and QOC across 21 decoder-only LLMs and then proposes three hypotheses: (1) training data bias, (2) option recall failure, and (3) causal attention mechanism. These are tested through controlled experiments to narrow down the cause, followed by targeted intervention experiments to provide causal evidence.

### Key Designs

1.  **Hypothesis 1 Exclusion: Training Data Bias Testing**

    -   **Function**: Verify if the CQO format is more familiar to the model due to higher frequency in training data.
    -   **Mechanism**: Compare the performance gap $\Delta = \text{Acc}_{\text{CQO}} - \text{Acc}_{\text{QOC}}$ across 9 matched base/instruct model pairs and use up to 5-shot ICL to familiarize the model with the QOC format.
    -   **Design Motivation**: If training distribution were the primary cause, instruct models (which see more CQO-formatted instruction data) should exhibit larger gaps, and few-shot learning should significantly narrow the gap.
    -   **Result**: The gap for base and instruct models is nearly identical; 5-shot only improves QOC by 3.1%, which is far from bridging the gap. This hypothesis is excluded.

2.  **Hypothesis 2 Exclusion: Option Recall Test**

    -   **Function**: Examine whether the model "forgets" intermediate options in the QOC format due to excessive context length (similar to the lost-in-the-middle effect).
    -   **Mechanism**: Require LLMs to precisely recall each option after the prompt is given and measure the exact match rate.
    -   **Design Motivation**: If option forgetting were the primary cause, the option recall rate for the QOC format should be significantly lower than for CQO.
    -   **Result**: The option recall accuracy for QOC is comparable to or even higher than CQO. This hypothesis is excluded.

3.  **Hypothesis 3 Verification: Causal Attention Mechanism Analysis**

    -   **Function**: Prove that the causal attention mask prevents option tokens in the QOC format from attending to context tokens.
    -   **Mechanism**: Validation through three sub-experiments: (a) architectural comparison between decoder-only (causal attention), encoder-decoder (bidirectional encoder), and encoder-only (bidirectional attention); (b) context removal test comparing QOC with QO (complete removal of context); (c) attention and attribution analysis tracing layer-wise attention distribution and Gradient×Input attribution.
    -   **Design Motivation**: If causal masking is the root cause, architectures using bidirectional attention should be unaffected, and removing context should not significantly alter QOC performance.
    -   **Result**: Decoder-only shows a 14.72% gap, encoder-decoder only 2.30%, and encoder-only only 0.02%. QOC performance is nearly equal to QO. Context attribution in CQO is 0.797, while it is only 0.335 in QOC.

### Moderating Factors & Intervention Experiments

The authors identify two moderating factors: **context length** (longer contexts lead to larger gaps, e.g., a 20.8% gap in RACE-H with ~305 tokens) and **answer position** (an earlier option A shows a gap of 22.4%, while a later option D shows only 9.9%).

Based on the causal attention explanation, four targeted interventions were designed:

-   **Attention Pruning** (Lowering CQO): Setting $\text{mask}[i,j] = -\infty$ ($i \in \text{Options}, j \in \text{Context}$) reduced CQO accuracy from 69.26% to 42.46%.
-   **Activation Patching** (Improving QOC): Replacing QOC option hidden states with those from CQO ($h_{\text{opt}}^{\text{QOC}} \leftarrow h_{\text{opt}}^{\text{CQO}}$) improved QOC by 6.0 points.
-   **Option Repetition QOCO**: Repeating options after the context allows the new option tokens to access the context, improving QOC by 8.2 points.
-   **CoT Prompting**: Narrowed the gap from 14.72 to 7.47.

## Key Experimental Results

### Main Results

| Method | LogiQA | SciQ | RACE-M | RACE-H | Average |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CQO | 39.08 | 94.16 | 74.32 | 69.48 | 69.26 |
| QOC | 32.94 | 86.89 | 49.57 | 48.76 | 54.54 |
| Gap Δ | 6.14 | 7.27 | 24.75 | 20.72 | 14.72 |

| Architectural Type | Representative Models | Average Gap Δ |
| :--- | :--- | :--- |
| Decoder-only | LLaMA/Qwen/Gemma | 14.72% |
| Encoder-decoder | Flan-T5 | 2.30% |
| Encoder-only | BERT/RoBERTa/ALBERT | 0.02% |

### Ablation Study

| Intervention | Goal | Effect |
| :--- | :--- | :--- |
| Attention Pruning (CQO) | Lower CQO | -26.8% |
| Activation Patching (QOC) | Improve QOC | +6.0% |
| Option Repetition QOCO | Improve QOC | +8.2% |
| CoT Prompting | Reduce Gap | Gap 14.72 → 7.47 |

### Key Findings

-   The causal attention mask is the fundamental mechanism for prompt order sensitivity, rather than training bias or memory decay.
-   In QOC, the hidden states of option tokens are structurally unable to access context information during computation, resulting in mutual information $I(h_O^{\text{QOC}}; C | Q, O) = 0$.
-   Although the final answer token can access both options and context, the option representations are already "context-blind," and single-step decoding cannot compensate for this.
-   Longer contexts and earlier correct answer positions amplify the negative impact of the causal mask.

## Highlights & Insights

-   **Mechanistic rather than descriptive**: Unlike previous work that only reported phenomena, this paper provides a clear causal mechanism, offering causal evidence through architectural comparisons and intervention experiments.
-   **Single-step bottleneck theory**: Proposes an elegant "single-step bottleneck" explanation—even if the final token can see all information, the options have been encoded as context-independent representations, which one-step decoding cannot fix.
-   **Practical Value**: Option repetition (QOCO) and CoT serve as simple prompt engineering strategies that can partially alleviate the problem without modifying the model.
-   **Information-theoretic formalization**: The appendix provides rigorous information-theoretic derivations, proving that the mutual information between option representations and context is structurally zero under QOC.

## Limitations & Future Work

-   The theoretical analysis is relatively fundamental, only establishing structural independence without deeply quantifying the specific magnitude of information loss.
-   This is a diagnostic study that proposes inference-time mitigation solutions but does not explore fundamental training-time fixes.
-   Experiments are limited to models with 0.5B-9B parameters; performance on larger-scale models remains to be verified.
-   Although CoT narrows the gap, the remaining gap is still 7.47%, indicating the limitations of inference-time fixes.

## Related Work & Insights

-   **vs. Lost-in-the-Middle**: Although both involve long-context utilization, this paper proves through option recall experiments that the issue with QOC is structural attention blockage rather than "forgetting."
-   **vs. Option Arrangement Sensitivity**: Previous work focused on the impact of internal option ordering on performance, while this study focuses on the broader arrangement of components, finding a larger performance gap.
-   **vs. Bidirectional Attention Models**: The finding that encoder-only models have almost zero gap provides architectural inspiration for introducing partial bidirectional attention during pre-training.

## Rating

-   Novelty: ⭐⭐⭐⭐ First to systematically explain prompt order sensitivity from an architectural level with a clear hypothesis-driven paradigm.
-   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough, spanning 21 models, 4 datasets, 3 architectural comparisons, and 4 intervention experiments.
-   Writing Quality: ⭐⭐⭐⭐⭐ Rigorous logic with a complete chain of hypothesis-experiment-conclusion and intuitive chart designs.
-   Value: ⭐⭐⭐⭐ Significant guiding implications for LLM reliability and prompt engineering, providing directions for future architectural improvements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] The Imperfective Paradox in Large Language Models](the_imperfective_paradox_in_large_language_models.md)
- [\[ACL 2026\] AdapTime: Enabling Adaptive Temporal Reasoning in Large Language Models](adaptime_enabling_adaptive_temporal_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Table Question Answering in the Era of Large Language Models: A Comprehensive Survey](table_question_answering_in_the_era_of_large_language_models_a_comprehensive_sur.md)
- [\[AAAI 2026\] Language Models and Logic Programs for Trustworthy Tax Reasoning](../../AAAI2026/nlp_understanding/language_models_and_logic_programs_for_trustworthy_tax_reasoning.md)
- [\[NeurIPS 2025\] Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention](../../NeurIPS2025/nlp_understanding/generalization_error_analysis_for_selective_state-space_models_through_the_lens_.md)

</div>

<!-- RELATED:END -->
