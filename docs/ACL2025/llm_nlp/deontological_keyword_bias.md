---
title: >-
  [Paper Note] Deontological Keyword Bias: The Impact of Modal Expressions on Normative Judgments of Language Models
description: >-
  [ACL 2025][LLM (Other)][Deontological Keyword Bias] This paper reveals that LLMs exhibit "Deontological Keyword Bias" (DKB)—when prompts contain modal deontic expressions such as "must" and "ought to", the models misclassify over 90% of commonsense scenarios as obligations. The authors propose debiasing strategies based on few-shot examples and reasoning prompts.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Deontological Keyword Bias"
  - "Modal Expressions"
  - "Normative Judgments"
  - "LLM Bias"
  - "Debias Methods"
date: 2026-05-08
content_hash: 84cffef4800629ff
---

# Deontological Keyword Bias: The Impact of Modal Expressions on Normative Judgments of Language Models

**Conference**: ACL 2025  
**arXiv**: [2506.11068](https://arxiv.org/abs/2506.11068)  
**Area**: LLM Alignment / AI Safety  
**Keywords**: Deontological Keyword Bias, Modal Expressions, Normative Judgments, LLM Bias, Debias Methods  

## TL;DR

This paper reveals that LLMs exhibit "Deontological Keyword Bias" (DKB)—when prompts contain modal deontic expressions such as "must" and "ought to", the models misclassify over 90% of commonsense scenarios as obligations. The authors propose debiasing strategies based on few-shot examples and reasoning prompts.

## Background & Motivation

**The moral reasoning capability of LLMs is increasingly important**: As the application of LLMs in the real world expands, the normative decisions they make as agents may affect society's understanding of "right" and "wrong."

**The criticality of deontic/obligation judgments**: Obligation judgments are core elements of behavioral decision-making in LLMs. Unlike factual judgments, the criteria for obligation judgments are often ambiguous, even for humans.

**Key differences between humans and LLMs**:
   - **Humans** learn normative judgments through real-world interactions and consequence simulation/imagination.
   - **LLMs** learn concepts of obligation indirectly through textual patterns, lacking direct interaction with real-world consequences.
   - This leads to LLMs potentially over-relying on linguistic cues (such as modal expressions) rather than contextual understanding.

**Core Hypothesis**: LLM obligation judgments are primarily influenced by modal deontic expressions (Modal Expressions), even in scenarios where no obligation judgment is required.

**Real-world Risk Example**: "You should have an umbrella when it rains" — carrying an umbrella is a reasonable recommendation but not a true obligation. LLMs might falsely judge it as an obligation.

## Method

### Overall Architecture

The study unfolds around two core concepts:

- **DKE (Deontological Keyword Effect)**: The general phenomenon where modal deontic expressions lead to an increase in obligation judgments.
- **DKB (Deontological Keyword Bias)**: A special case of DKE—where the model incorrectly judges a scenario as an obligation due to the presence of modal expressions in contexts where humans do not believe an obligation exists.

Mathematical definition: Given a semantic frame $S$, modal enhancement $Z$, and question format $Q$, DKE holds when `$f(Y_with_ME) > f(Y_without_ME)$` holds consistently across instances. DKB refers specifically to DKE when $S$ lacks obligation-related semantics.

### Key Designs

1. **Experimental Dataset Design**:

    - **Deontological Dataset (Positive Labels)**: The deontology dataset from Hendrycks et al. (2021).
    - **Commonsense Dataset (Negative Labels)**: Serves as the control group for non-deontic contexts.
    - **Moral Dataset**: From Scherrer et al. (2023), containing high and low ambiguity sub-datasets.
    - Each dataset has 445 samples, using four modal expressions: "must", "ought to", "should", and "have to".

2. **Multi-dimensional Validation**:

    - Three levels of questions: general, explicit, and strict.
    - Two answer formats: binary and continuous rating.
    - Impact of negative modal expressions (e.g., "must not").
    - Comparison of different modal expression strengths.

3. **Debiasing Method — In-Context Reasoning**:

    - A hybrid method combining few-shot learning and reasoning prompts.
    - Few-shot examples are annotated based on deontological semantics (rather than keywords).
    - Modal expressions are removed from examples in the deontological dataset.
    - Negative examples from the commonsense dataset contain modal expressions.

## Key Experimental Results

### Main Results

**Comparison of Humans vs. GPT-4o on Obligation Judgments** (0-5 scale):

| Condition | Dataset | Human | GPT-4o |
|------|--------|------|--------|
| With Modal Expression | Deontological | 4.17 (0.50) | 4.95 (0.25) |
| Without Modal Expression | Deontological | 3.11 (1.44) | 0.30 (0.05) |
| With Modal Expression | Commonsense | 3.33 (1.84) | **4.90 (0.10)** |
| Without Modal Expression | Commonsense | 1.90 (1.05) | 0.10 (0.10) |

**Key Findings**: On the commonsense dataset with modal expressions, GPT-4o gives a score of 4.90 (compared to only 3.33 for humans) with extremely low variance (0.10), indicating that the model almost mechanically classifies all sentences with modal expressions as obligations.

**Cross-Model Existence Validation of DKB** (Commonsense dataset, ratio of positive obligation judgments):

| Model | Without ME | With ME | With Negative ME |
|------|------|------|----------|
| GPT-4o | 0.02 | **0.98** | 0.98 |
| GPT-4o-mini | 0.04 | **0.96** | 0.97 |
| Llama-3.1-70B | 0.01 | **0.86** | 0.87 |
| Llama-3.1-8B | 0.00 | 0.54 | 0.59 |
| Gemma-9B | 0.01 | **0.89** | 0.69 |
| Qwen-7B | 0.02 | **0.88** | 0.92 |

**Bias Strength of Different Modal Expressions** (Commonsense dataset, cross-model average):

| Modal Expression | Ratio of Positive Judgments |
|----------|-----------|
| must | 0.86 |
| ought to | 0.83 |
| should | 0.79 |
| have to | 0.64 |

The bias strength is consistent with the modal strength in deontic logic.

### Key Findings

1. **Universality of DKB**: Across almost all tested LLMs on the commonsense dataset, the ratio of positive judgments skyrocketed from less than 5% to 50–98% after adding modal expressions.

2. **Negative Modal Expressions Also Induce Bias**: Negative forms (such as "must not") are also judged as containing obligation semantics, with the bias being even more severe than affirmative forms on the commonsense dataset.

3. **Consistency Across Question Formats**: DKB consistently exists across general, explicit, and strict question levels, as well as binary/continuous rating formats.

4. **Limited Impact in Reasoning Tasks**: In the Opposing Contexts Scenario (OCS) experiments, the impact of modal expressions on reasoning outcomes is small and inconsistent, suggesting that keywords might affect judgment and reasoning differently.

5. **De-biasing Performance**: The hybrid few-shot + reasoning prompting method reduced the ratio of positive judgments on the commonsense dataset from 0.88 to 0.28 (using 2-shot + reasoning), demonstrating effective debiasing potential.

## Highlights & Insights

1. **Discovery and Definition of a New Phenomenon**: This study is the first to systematically identify and formally define "Deontological Keyword Bias" (DKB), filling an important gap in the research of LLM moral reasoning.

2. **Connection with Instruction Tuning**: LLMs are frequently instruction-tuned to follow user prompts, making them particularly sensitive to modal deontic expressions. When expressions like "must follow the instruction" appear, the model may over-generalize its authority.

3. **Bias Source in Training Data**: Taking the Alpaca RLHF dataset as an example, non-deontic usages such as "A picnic list should include items such as sandwiches" also reinforce the association between modal expressions and deontic semantics.

4. **Reflections on Practical Impact**: As LLMs act as agent systems making real-world decisions, the ability to distinguish legal enforcement, social norms, and recommendations is crucial.

5. **Simple and Effective Debiasing Solution**: The proposed training-free debiasing method is simple and practical, serving as a quick patch for actual deployment.

## Limitations & Future Work

1. **Limited Dataset Scale**: Each dataset contains only 445 samples, and the experiments did not cover all available models.

2. **Linguistic Limitations**: Only English data was used; whether similar biases exist in other languages or cultural backgrounds remains to be verified.

3. **Insufficient Quantification of Debiasing Effects**: Although the debiasing method is effective, quantitative evaluation of the extent of its adjustment is lacking.

4. **Lack of Mechanistic Analysis**: The internal mechanisms of DKB generation in LLM knowledge representations were not thoroughly analyzed.

5. **Limited Types of Modal Expressions**: Only four modal expressions were tested; other types (such as "need to", "required to") were not covered.

## Related Work & Insights

- **Deontic Logic**: Von Wright (1951) symbolic deontic logic; Kant's Categorical Imperative.
- **Obligation Detection in NLP**: Chalkidis et al. (2018) RNN-based regulatory obligation detection; Sun et al. (2023) DeonticBERT.
- **LLM Bias**: Solaiman et al. (2019) social fairness bias; Ladhak et al. (2023) name-culture entanglement.
- **LLM Moral Reasoning**: Zhou et al. (2023), Rao et al. (2023) ethical reasoning.

## Rating

| Dimension | Rating (1-10) | Explanation |
|------|-------------|------|
| Novelty | 9 | Identifies and defines the important phenomenon of DKB for the first time. |
| Technical Depth | 7 | Primarily empirical, with a clear formal definition. |
| Experimental Thoroughness | 8 | Systematic validation across multiple models, dimensions, and conditions. |
| Writing Quality | 8 | Clear conceptual definitions and structured layout. |
| Value | 8 | Direct guiding significance for AI safety and alignment research. |
| **Overall Rating** | **8.0** | A high-quality empirical study revealing an important bias phenomenon. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Comparing Linguistic Acceptability Judgments of Autoregressive Language Models](comparing_linguistic_acceptability_judgments_of_autoregressive_language_models.md)
- [\[ACL 2025\] Detecting Referring Expressions in Visually Grounded Dialogue with Autoregressive Language Models](detecting_referring_expressions_in_visually_grounded_dialogue_with_autoregressiv.md)
- [\[ACL 2025\] Attention Speaks Volumes: Localizing and Mitigating Bias in Language Models](attention_speaks_volumes_localizing_and_mitigating_bias_in_language_models.md)
- [\[ACL 2025\] Information Locality as an Inductive Bias for Neural Language Models](information_locality_as_an_inductive_bias_for_neural_language_models.md)
- [\[ACL 2025\] AI as a Novel Ethical Agent: Exploring Moral Judgments by Large Language Models](ai_as_a_novel_ethical_agent_exploring_moral_judgments_by_large_language_models.md)

</div>

<!-- RELATED:END -->
