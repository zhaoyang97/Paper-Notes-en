---
title: >-
  [Paper Note] Lost in the Prompt Order: Revealing the Limitations of Causal Attention in Language Models
description: >-
  [ACL 2026][LLM/NLP][causal attention] This paper systematically investigates the sensitivity of large language models to the ordering of prompt components in multiple-choice question answering (MCQA). Through controlled experiments, the authors rule out training bias and memory decay hypotheses, identifying the causal attention mask as the fundamental mechanism responsible for the substantial performance degradation observed under the QOC (Question–Options–Context) ordering.
tags:
  - ACL 2026
  - LLM/NLP
  - causal attention
  - prompt order sensitivity
  - multiple-choice QA
  - information bottleneck
  - mechanistic interpretability
date: 2026-05-08
content_hash: f249f374b6c2234f
---

# Lost in the Prompt Order: Revealing the Limitations of Causal Attention in Language Models

**Conference**: ACL 2026
**arXiv**: [2601.14152](https://arxiv.org/abs/2601.14152)
**Code**: None
**Area**: NLP Understanding / Prompt Sensitivity
**Keywords**: causal attention, prompt order sensitivity, multiple-choice QA, information bottleneck, mechanistic interpretability

## TL;DR

This paper systematically investigates the sensitivity of large language models to the ordering of prompt components in multiple-choice question answering (MCQA). Through controlled experiments, the authors rule out training bias and memory decay hypotheses, identifying the causal attention mask as the fundamental mechanism responsible for the substantial performance degradation observed under the QOC (Question–Options–Context) ordering.

## Background & Motivation

**State of the Field**: The sensitivity of LLMs to prompt structure has been widely reported — both the ordering of demonstrations in in-context learning and the arrangement of options in multiple-choice tasks can cause significant performance fluctuations. However, most existing research remains at the phenomenological level: while it is known *what* affects model performance, *why* it does so remains unclear.

**Limitations of Prior Work**: In MCQA tasks, a typical prompt consists of three components: a context passage (C), a question (Q), and options (O). Intuitively, reordering these components should not affect performance, as the semantic content remains identical. However, experiments show that placing the context before the question and options (CQO) consistently and substantially outperforms the reverse ordering (QOC), with an average gap exceeding 14 percentage points across 21 decoder-only models and 4 datasets.

**Root Cause**: Semantically equivalent prompt orderings yield dramatically different performance, posing a serious challenge to LLM reliability. Prior works such as lu2022fantastically and pezeshkpour2024 report similar phenomena but do not investigate the architectural root cause.

**Paper Goals**: The paper proposes three competing hypotheses and evaluates each through carefully designed controlled experiments, ultimately identifying the core mechanism underlying prompt order sensitivity and designing targeted interventions to validate the conclusion.

**Starting Point**: The investigation proceeds from an architectural perspective, localizing the source of the problem by comparing the behavior of decoder-only, encoder-only, and encoder-decoder architectures.

## Method

### Overall Architecture

The paper adopts a "propose hypothesis → design experiment → verify/eliminate" research paradigm. The performance gap between CQO and QOC is first quantified across 21 decoder-only LLMs, after which three hypotheses are proposed: (1) training data bias, (2) option recall failure, and (3) causal attention mechanism. A series of controlled experiments progressively narrows the causal scope, and targeted intervention experiments provide causal evidence.

### Key Designs

1. **Ruling Out Hypothesis 1: Training Data Bias**

    - **Function**: Assess whether the CQO format is more prevalent in training data, causing models to be less familiar with QOC.
    - **Mechanism**: Compare the performance gap $\Delta = \text{Acc}_{\text{CQO}} - \text{Acc}_{\text{QOC}}$ across 9 matched base/instruct model pairs, and apply up to 5-shot ICL to familiarize models with the QOC format.
    - **Design Motivation**: If training distribution were the primary cause, instruct models (exposed to more CQO-formatted instruction data) should exhibit larger gaps, and few-shot examples should substantially close the gap.
    - **Result**: Base and instruct models exhibit nearly identical gaps; 5-shot ICL improves QOC by only 3.1%, far from closing the gap. This hypothesis is eliminated.

2. **Ruling Out Hypothesis 2: Option Recall Failure**

    - **Function**: Test whether models "forget" options positioned in the middle of the QOC prompt due to long context (analogous to the lost-in-the-middle effect).
    - **Mechanism**: After presenting the prompt, models are asked to reproduce each option exactly; exact match rates are measured.
    - **Design Motivation**: If option forgetting were the primary cause, option recall accuracy under QOC should be significantly lower than under CQO.
    - **Result**: Option recall accuracy under QOC is comparable to or higher than under CQO. This hypothesis is eliminated.

3. **Validating Hypothesis 3: Causal Attention Mechanism**

    - **Function**: Demonstrate that the causal attention mask prevents option tokens from attending to context tokens in the QOC ordering.
    - **Mechanism**: Three sub-experiments are conducted: (a) *Architecture comparison*: decoder-only (causal attention) vs. encoder-decoder (bidirectional encoder) vs. encoder-only (bidirectional attention); (b) *Context removal test*: comparing QOC against QO (context entirely removed); (c) *Attention and attribution analysis*: tracking layer-wise attention distributions and Gradient×Input attributions.
    - **Design Motivation**: If the causal mask is the root cause, architectures with bidirectional attention should be unaffected, and removing the context should leave QOC performance largely unchanged.
    - **Result**: The decoder-only gap is 14.72%; encoder-decoder gap is only 2.30%; encoder-only gap is only 0.02%. QOC performance nearly equals QO. Context attribution in CQO is 0.797, versus only 0.335 in QOC.

### Moderating Factors and Intervention Experiments

Two moderating factors are identified: **context length** (longer contexts lead to larger gaps, e.g., RACE-H at ~305 tokens yields a gap of 20.8%) and **answer position** (option A in the first position shows a gap of 22.4%, whereas option D in the last position shows only 9.9%).

Based on the causal attention mechanism explanation, four targeted interventions are designed:

- **Attention pruning** (degrading CQO): Setting $\text{mask}[i,j] = -\infty$ for $i \in \text{Options}, j \in \text{Context}$ reduces CQO accuracy from 69.26% to 42.46%.
- **Activation patching** (improving QOC): Replacing option hidden states in QOC with those from CQO, $h_{\text{opt}}^{\text{QOC}} \leftarrow h_{\text{opt}}^{\text{CQO}}$, improves QOC by 6.0 points.
- **Option repetition (QOCO)**: Repeating options after the context so that new option tokens can attend to the context improves QOC by 8.2 points.
- **CoT prompting**: Reduces the gap from 14.72 to 7.47.

## Key Experimental Results

### Main Results

| Method | LogiQA | SciQ | RACE-M | RACE-H | Average |
|--------|--------|------|--------|--------|---------|
| CQO | 39.08 | 94.16 | 74.32 | 69.48 | 69.26 |
| QOC | 32.94 | 86.89 | 49.57 | 48.76 | 54.54 |
| Gap Δ | 6.14 | 7.27 | 24.75 | 20.72 | 14.72 |

| Architecture | Representative Models | Average Gap Δ |
|---|---|---|
| Decoder-only | LLaMA / Qwen / Gemma | 14.72% |
| Encoder-decoder | Flan-T5 | 2.30% |
| Encoder-only | BERT / RoBERTa / ALBERT | 0.02% |

### Ablation Study

| Intervention | Target | Effect |
|---|---|---|
| Attention pruning (CQO) | Degrade CQO | −26.8% |
| Activation patching (QOC) | Improve QOC | +6.0% |
| Option repetition QOCO | Improve QOC | +8.2% |
| CoT prompting | Reduce gap | 14.72 → 7.47 |

### Key Findings

- The causal attention mask is the fundamental mechanism underlying prompt order sensitivity, not training bias or memory decay.
- In QOC, option token hidden states are computed without any access to context information; the mutual information $I(h_O^{\text{QOC}}; C \mid Q, O) = 0$ structurally.
- Although the final answer token can attend to both options and context, the option representations are already "context-blind" at that point, and single-step decoding cannot compensate.
- Longer contexts and earlier positions of the correct answer amplify the negative impact of the causal mask.

## Highlights & Insights

- **Mechanistic rather than descriptive**: Unlike prior work that merely reports the phenomenon, this paper provides a clear causal mechanism explanation, supported by architectural comparisons and intervention experiments.
- **Single-step bottleneck theory**: The paper offers an elegant explanation — even though the final token can attend to all information, options have already been encoded as context-independent representations, and a single decoding step cannot recover what was lost.
- **Practical value**: Option repetition (QOCO) and CoT serve as simple prompt engineering strategies that can partially mitigate the problem without modifying the model.
- **Information-theoretic formalization**: The appendix provides a rigorous information-theoretic derivation proving that the mutual information between option representations and context is structurally zero under QOC.

## Limitations & Future Work

- The theoretical analysis is relatively basic, establishing only structural independence without quantifying the magnitude of information loss.
- The paper is diagnostic in nature; while inference-time mitigation strategies are proposed, training-time fundamental fixes are not explored.
- Experiments are limited to models with 0.5B–9B parameters; behavior at larger scales remains to be verified.
- CoT narrows the gap but a residual gap of 7.47% remains, highlighting the limits of inference-time remediation.

## Related Work & Insights

- **vs. Lost-in-the-Middle**: Although both involve the utilization of information in long contexts, option recall experiments in this paper demonstrate that the QOC problem is not one of "forgetting" but of structural attention blockage.
- **vs. Option permutation sensitivity studies**: Prior work focuses on how option ordering affects performance; this paper examines the more macro-level ordering of prompt components, uncovering even larger performance gaps.
- **vs. Bidirectional attention models**: The finding that encoder-only models exhibit near-zero gaps provides architectural design inspiration for whether partial bidirectional attention should be incorporated during pretraining.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic architectural-level explanation of prompt order sensitivity; the hypothesis-driven research paradigm is clearly structured.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 21 models, 4 datasets, 3 architectural comparisons, and 4 intervention experiments; extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous argumentation; the hypothesis–experiment–conclusion chain is complete and the figures are intuitive.
- Value: ⭐⭐⭐⭐ — Significant implications for LLM reliability and prompt engineering, and provides direction for future architectural improvements.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] CoSToM: Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment in Large Language Models](costomcausal-oriented_steering_for_intrinsic_theory-of-mind_alignment_in_large_l.md)
- [\[ACL 2026\] Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models](don39t_adapt_small_language_models_for_tools_adapt_tool_schemas_to_the_models.md)
- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](foresight_optimization_for_strategic_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Adam's Law: Textual Frequency Law on Large Language Models](adam39s_law_textual_frequency_law_on_large_language_models.md)
- [\[ACL 2026\] Why Did Apple Fall: Evaluating Curiosity in Large Language Models](why_did_apple_fall_evaluating_curiosity_in_large_language_models.md)

<!-- RELATED:END -->
