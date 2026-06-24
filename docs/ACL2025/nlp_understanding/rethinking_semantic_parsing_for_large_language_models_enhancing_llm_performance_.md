---
title: >-
  [Paper Note] Rethinking Semantic Parsing for Large Language Models: Enhancing LLM Performance with Semantic Hints
description: >-
  [ACL 2025][NLP Understanding][Semantic Parsing] In response to the counterintuitive phenomenon where "directly inputting semantic parsing results into LLMs actually degrades performance," this paper proposes SENSE—a zero-shot method that embeds semantic hints (rather than explicit parsing results) in the prompt, consistently improving LLM performance across GLUE understanding tasks and generation tasks such as machine translation, paraphrasing, and simplification.
tags:
  - "ACL 2025"
  - "NLP Understanding"
  - "Semantic Parsing"
  - "Prompt Engineering"
  - "SENSE"
  - "LLM Enhancement"
  - "Zero-Shot"
date: 2026-05-08
content_hash: 25225d7702b0af7f
---

# Rethinking Semantic Parsing for Large Language Models: Enhancing LLM Performance with Semantic Hints

**Conference**: ACL 2025  
**arXiv**: [2409.14469](https://arxiv.org/abs/2409.14469)  
**Code**: None  
**Area**: NLP Understanding / Prompt Engineering  
**Keywords**: Semantic Parsing, Prompt Engineering, SENSE, LLM Enhancement, Zero-Shot

## TL;DR
In response to the counterintuitive phenomenon where "directly inputting semantic parsing results into LLMs actually degrades performance," this paper proposes SENSE—a zero-shot method that embeds semantic hints (rather than explicit parsing results) in the prompt, consistently improving LLM performance across GLUE understanding tasks and generation tasks such as machine translation, paraphrasing, and simplification.

## Background & Motivation
Semantic Parsing is a fundamental task in NLP that converts natural language into structured semantic representations (such as SRL, AMR, etc.). During the BERT era, injecting semantic parsing results into model inputs was proven to effectively enhance downstream task performance.

**Paradox in the LLM Era**:
- Jin et al. (2024) discovered that directly feeding AMR parsing results into LLM prompts actually **harms** performance.
- Analysis of reasons: Formalized representations like AMR are "foreign symbol systems" to LLMs, introducing unnecessary complexity and potential parsing errors.
- Key Challenge: Semantic information benefits language understanding, but LLMs struggle with handling explicit semantic parsing structures.

**Key Insight**: Given that LLMs themselves possess strong semantic comprehension capabilities, why not instruct LLMs through **prompts** to self-activate their internal semantic parsing abilities, rather than injecting external parsing results? This is the Core Idea of SENSE—replacing "semantic results" with "semantic hints."

## Method

### Overall Architecture
SENSE is a zero-shot prompting method that adds a brief semantic hint sentence in the task prompt (e.g., "please use semantic parsing result which can enhance comprehension of the sentence's structure and semantics"), guiding the LLM to internally activate its semantic parsing capabilities without providing it with any explicit parsing output.

### Key Designs

1. **Semantic Hint Injection (Core of SENSE)**:

    - Injects a single prompt sentence into the standard task prompt, similar to: "please use semantic parsing result which can enhance comprehension of the sentence's structure and semantics"
    - The specific phrasing varies slightly across tasks, but the core message remains consistent: guiding the model to focus on semantic structure.
    - Design Motivation: LLMs, through large-scale pre-training, have already internalized semantic parsing capabilities. SENSE "awakens" this capability via prompts, avoiding noise from external parsing and format incompatibility issues.
    - Difference from CoT: CoT uses "let's think step by step" to guide a reasoning chain, whereas SENSE uses "use semantic parsing" to guide semantic understanding—the former is suitable for reasoning tasks, while the latter fits comprehension and generation tasks.

2. **Comparison with Other Paradigms**:

    - **SP-Input** (Figure 1b): Uses external tools to generate parsing results, then appends them to the input → Performance drops (introduces noise and foreign symbols).
    - **SP-Output** (Figure 1c): Prompts the LLM to output parsing results before performing the task → Unstable performance (inconsistent parsing quality).
    - **SENSE** (Figure 1d): Only provides semantic hints without generating or consuming any parsing results → Stable improvement.
    - Key Insight: LLMs cannot process explicit semantic structures well, but they can be prompted to "internally" leverage semantic information.

3. **Attention Mechanism Analysis**:

    - By visualizing the attention score distribution of LLaMA3-70B, it is observed that SENSE directs the model to focus more intensely on **key semantic elements** (core lexical units and main sentence components) in paraphrasing tasks.
    - Compared to vanilla prompts, the attention distribution of SENSE is more focused rather than uniformly distributed.
    - This mechanistically validates that SENSE indeed guides the model to perform more targeted semantic processing.

### Loss & Training
SENSE is a pure inference-time prompting method, involving no training or fine-tuning. It is applied in a zero-shot manner on GPT-3.5-turbo, GPT-4o-mini, and LLaMA3-70B, with the temperature set to 0.

## Key Experimental Results

### Main Results (GLUE Understanding Tasks)

| Model Configuration | SST-2 | MRPC | QQP | MNLI | QNLI | RTE | CoLA | Average |
|---------|-------|------|-----|------|------|-----|------|------|
| BERT_LARGE | 93.20 | 88.00 | 91.30 | 86.60 | 92.30 | 70.40 | 60.60 | 83.20 |
| GPT-4o-mini | 91.63 | 72.30 | 73.00 | 73.90 | 92.30 | 87.36 | 65.49 | 79.43 |
| GPT-4o-mini + SENSE | **92.08** | **76.47** | 73.00 | **78.20** | **93.30** | **88.45** | **67.22** | **81.25** |
| GPT-3.5 | 91.86 | 73.28 | 73.40 | 61.80 | 82.40 | 81.81 | 63.50 | 75.44 |
| GPT-3.5 + SENSE | 92.20 | 75.49 | 77.20 | 64.60 | 83.20 | 84.12 | 64.57 | 77.34 |

GPT-4o-mini + SENSE improves the average score from 79.43 to 81.25, closing the gap with BERT. The gains on MRPC and MNLI are the most significant.

### Ablation Study (Comparison of Different Paradigms, GPT-3.5 Baseline)

| Method | SST-2 | MRPC | QQP | MNLI | QNLI | RTE | CoLA |
|------|-------|------|-----|------|------|-----|------|
| Baseline | 91.86 | 73.28 | 73.40 | 61.80 | 82.40 | 81.81 | 63.50 |
| + CoT | 89.11(-2.75) | 73.28 | 77.00(+3.60) | 56.20(-5.60) | 82.70 | 82.54 | 64.32 |
| + SP-Input | 87.50(-4.36) | 74.26 | 74.30 | **50.50(-11.30)** | 78.40(-4.00) | 84.11 | 58.37(-5.13) |
| + SP-Output | 89.11(-2.75) | 73.52 | 71.90 | 62.00 | 78.40(-4.00) | 81.59 | 64.44 |
| + SENSE | **92.20(+0.34)** | **75.49(+2.21)** | **77.20(+3.80)** | **64.60(+2.80)** | **83.20(+0.80)** | **84.12(+2.31)** | **64.57(+1.07)** |

SENSE consistently achieves positive improvements across all 7 tasks, whereas SP-Input, SP-Output, and CoT exhibit significant negative impacts.

### Generation Tasks

| Task | Metric | GPT-4o-mini | + SENSE | Gain |
|------|------|-------------|---------|------|
| Paraphrase | Semantic Similarity↑ | 89.71 | 90.26 | +0.55 |
| Paraphrase | Lexical Overlap↓ | 39.00 | 34.00 | -5.00 |
| Paraphrase | Syntactic Diversity↑ | 7.25 | 8.08 | +0.83 |
| Simplification (TurkCorpus) | BLEU↑ | 58.16 | 63.42 | +5.26 |
| Simplification (TurkCorpus) | SAMSA↑ | 31.42 | 37.03 | +5.61 |
| Machine Translation (DE→EN) | COMET22↑ | 85.71 | 86.44 | +0.73 |
| Machine Translation (EN→DE) | BLEU↑ | 33.42 | 34.18 | +0.76 |

### Key Findings
- Explicitly injecting semantic parsing results harms LLMs (MNLI drops by 11.30 points), while SENSE consistently maintains positive improvements.
- CoT performs poorly on comprehension tasks (as CoT is better suited for reasoning tasks), whereas SENSE is more appropriate for tasks requiring semantic understanding.
- In generation tasks, SENSE promotes richer syntactic transformations and lexical diversity while maintaining semantic faithfulness.
- Attention visualization confirms that SENSE focuses the model on key semantic components.

## Highlights & Insights
- **Extremely Simple Yet Effective**: Requires only one additional sentence in the prompt to obtain stable improvements, with zero extra computational overhead and zero testing/training requirements.
- **Deep Cognitive Inspiration**: The success of SENSE implies that LLMs have already encoded semantic parsing capabilities within, and the key lies in how to "awaken" them—this offers important insights into understanding the internal representations of LLMs.
- **Complementarity with CoT**: CoT is suitable for reasoning chains, while SENSE is suited for semantic understanding; the two may each excel in different task types.
- **Attention Analysis** provides evidence for interpretability, rather than being a black-box enhancement.

## Limitations & Future Work
- Validated only on LLaMA and GPT series; generalizability needs to be confirmed across more model architectures.
- Task coverage is mainly focused on tasks where semantic parsing is known to be beneficial; testing on more diverse tasks is necessary.
- The mechanism of how SENSE affects the internal decision-making process of LLMs remains unclear (LLMs are black boxes).
- The combined usage of SENSE and CoT has not been explored.
- The expression "semantic parsing result" may be understood differently by different models, and the robustness of the prompt phrasing has not been fully analyzed.

## Related Work & Insights
- **Jin et al. (2024)** proposed AMR-CoT and found that directly injecting AMR into LLMs has negative impacts; this work proposes a solution based on this finding.
- **The Dual Nature of Semantic Parsing**: Beneficial for small models (BERT) but harmful for LLMs; this discrepancy itself is an interesting reflection of the capacity characteristics of LLMs.
- **Broader Prompt Engineering Inspiration**: Can similar strategies of "prompting the model to utilize intrinsic capabilities" be generalized to other domains? For example, "please use your knowledge graph reasoning ability" or "please use your causal reasoning ability"?

## Rating
- Novelty: ⭐⭐⭐⭐ Novel perspective—having LLMs "self-activate" semantic capabilities rather than injecting external ones.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers both comprehension and generation tasks, with a complete ablation analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear argumentation logic and highly informative tables/figures.
- Value: ⭐⭐⭐⭐ Provides a plug-and-play LLM enhancement strategy, inspiring thinking on the activation of LLMs' intrinsic capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Disambiguate First, Parse Later: Generating Interpretations for Ambiguity Resolution in Semantic Parsing](disambiguate_first_parse_later_generating_interpretations_for_ambiguity_resoluti.md)
- [\[ACL 2025\] BQA: Body Language Question Answering Dataset for Video Large Language Models](bqa_body_language_question_answering_dataset_for_video_large_language_models.md)
- [\[ACL 2025\] Generating Diverse Training Samples for Relation Extraction with Large Language Models](generating_diverse_training_samples_for_relation_extraction_with_large_language_.md)
- [\[ACL 2026\] The Imperfective Paradox in Large Language Models](../../ACL2026/nlp_understanding/the_imperfective_paradox_in_large_language_models.md)
- [\[ACL 2026\] LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines](../../ACL2026/nlp_understanding/llm-guided_semantic_bootstrapping_for_interpretable_text_classification_with_tse.md)

</div>

<!-- RELATED:END -->
