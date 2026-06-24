---
title: >-
  [Paper Note] Meaning Beyond Truth Conditions: Evaluating Discourse Level Understanding via Anaphora Accessibility
description: >-
  [ACL 2025][LLM (Other)][Discourse Semantic Understanding] This paper proposes a hierarchical framework for semantic NLU capabilities (lexical, sentential, and discourse levels) and constructs an evaluation dataset based on anaphora accessibility. It is found that while LLMs align with humans on certain structures, they systematically diverge on others—LLMs rely on lexical cues rather than structural abstractions.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Discourse Semantic Understanding"
  - "Anaphora Accessibility"
  - "Dynamic Semantics"
  - "LLM Evaluation"
  - "Formal Semantics"
date: 2026-05-08
content_hash: 8a875fcb3b64cb58
---

# Meaning Beyond Truth Conditions: Evaluating Discourse Level Understanding via Anaphora Accessibility

**Conference**: ACL 2025  
**arXiv**: [2502.14119](https://arxiv.org/abs/2502.14119)  
**Code**: None  
**Area**: Other  
**Keywords**: Discourse Semantic Understanding, Anaphora Accessibility, Dynamic Semantics, LLM Evaluation, Formal Semantics

## TL;DR

This paper proposes a hierarchical framework for semantic NLU capabilities (lexical, sentential, and discourse levels) and constructs an evaluation dataset based on anaphora accessibility. It is found that while LLMs align with humans on certain structures, they systematically diverge on others—LLMs rely on lexical cues rather than structural abstractions.

## Background & Motivation

The success of LLMs relies on natural language understanding capabilities, but existing evaluation tasks rarely investigate whether LLMs can accurately represent and update discourse states. Successful discourse interpretation requires using pronouns to refer to entities already introduced in the text—this is the problem of anaphora.

Anaphoric appropriateness is constrained by the semantic scope of the antecedent. This is illustrated by classic examples:

- "**A** farmer worked in his field. **He** dreamed of the harvest." ✓ (An entity introduced by an existential quantifier can be referred to in subsequent sentences.)
- "**Every** farmer worked in his field. **He** dreamed of the harvest." ✗ (An entity introduced by a universal quantifier is inaccessible outside its scope.)

This phenomenon is rigorously formalized in Dynamic Semantics—discourse meaning is not merely static truth conditions, but an update operation on the discourse state. different quantifiers and logical connectives determine the accessibility scope of a discourse referent.

Limitations of prior work:
- Schuster and Linzen (2022) only consider the interaction of negation with discourse referents.
- Kim and Schuster (2023) use overly simplified language (e.g., "Box 1 contains the book").
- There is a lack of systematic evaluation of the interaction between various scopes (universal quantifiers, conditionals, disjunctions, etc.) and anaphora.

## Method

### Overall Architecture

The authors propose a three-level hierarchy of semantic understanding capabilities:

1. **Lexical Level**: Understanding individual word meanings—synonymy, antonymy, entailment, etc.
2. **Sentence Level**: Integrating lexical meanings to form a truth-conditional representation of sentences.
3. **Discourse Level**: Integrating meanings across consecutive sentences to update discourse representations.

Focusing on the discourse level, this paper uses anaphora accessibility as a diagnostic tool to evaluate whether LLMs understand how different semantic operators affect discourse state updates.

### Key Designs

**The experiments cover three types of semantic constructions:**

**1. Universal Quantifiers**

- Simple contrast: A farmer vs. Every farmer + cross-sentence anaphora
- Donkey Conditionals:
    - "John owns a donkey, and he beats **it**. **It** is a big one." ✓ (existential quantifier)
    - "If John owns a donkey, he beats **it**. **It** is a big one." ✗ (conditionals imply universal quantification)
    - "Whenever John owns a donkey, he beats **it**. **It** is a big one." ✗ (same as above)

**2. Negation**

- Existential quantifier: The farmer owned a cow. -> It was away on the meadow. ✓
- Negation: The farmer didn't own a cow. -> It was away on the meadow. ✗
- Double negation: It was not the case that the farmer didn't own a cow. -> It was away on the meadow. ✓
- Double negation resolution: Two negations cancel each other out, semantically equivalent to an existential quantifier.

**3. Disjunction**

- Evans (1977)'s finding: An existential quantifier in the first disjunct does not license cooperative anaphora in the second, but a negative quantifier can.
- "Either there was **a** manuscript, or **it** was hidden..." ✗
- "Either there was **no** manuscript, or **it** was hidden..." ✓
- Presence of "either" does not affect semantics (or vs. Either...or are equivalent).
- Negative quantifiers do not have the same effect in conjunctions.

**Evaluation Metrics**:

- **Difference-of-Difference metric**: Compares the probability difference between in-scope (within-sentence) and cross-sentence anaphora under existential and universal quantifiers, controlling for confounding factors such as sentence complexity.
- **Conditional Probability Metric**: Compares the joint surprisal of the same subsequent sentence under different contexts.
- **SLOR (Syntactic Log-Odds Ratio)**: Used for disjunction experiments to control for sentence length and word frequency.

**Models and Human Experiments**:
- 4 open-source LLMs: Llama3-2-1B/3B, Llama3-1-8B, Llama3-1-8B-Instruct
- 2 closed-source LLMs: GPT babbage-002, davinci-002
- Human experiment: 104 participants recruited via Prolific, 66 forced-choice trials.

**Corpus Construction**:
- Generated from structural templates, manually constructing 32 semantically plausible sentence frames.
- Manually checked by linguistics experts to ensure acceptability/unacceptability.
- 9,816 experimental sentences in total.

### Loss & Training

This paper does not involve model training. Instead, it treats LLMs as psycholinguistic subjects, measuring discourse comprehension capabilities through their surprisal (negative log-probability) on target tokens.

## Key Experimental Results

### Main Results

**Experiment 1: Universal Quantifiers**

- Simple contrast (Exi > Every): Llama family achieves around 75% accuracy, GPT family is slightly lower, while humans are close to the ceiling.
- Conditionals (Exi > If, Exi > Whenever): All LLMs are close to the ceiling (>90%), but human accuracy is unexpectedly lower.
- **Interesting Divergence**: Humans prefer if/whenever conditionals in the "he"-continuation sentence (reversing the expected direction), possibly due to the "telescoping" effect—humans tend to interpret "he" as being within the scope of the conditional.

**Experiment 2: Negation**

- Exi > Neg: All models successfully distinguish, achieving high accuracy.
- DN > Neg (Double Negation > Single Negated): 3 models failed; the Llama3-1-8B series even preferred negation over double negation (reversing the expected direction).
- After adding "in fact": DN > Neg accuracy improved, but Exi > Neg accuracy reversed.
- **Key Finding**: LLMs' understanding of negation scope is unsystematic and heavily relies on lexical cues like "in fact".

**Experiment 3: Disjunction**

- EitherOr > Conjunction and EitherOr > EitherPosOr: All models hit the ceiling, aligning with humans.
- or > Conjunction: Model accuracy is close to random, whereas humans show the expected preference.
- EitherPosOr vs. or: Models prefer EitherPosOr (reversing the expected direction), while humans show no clear preference.
- **Key Finding**: Though EitherOr and or are semantically equivalent, model performance heavily depends on the presence of the word "either".

### Ablation Study

**The lexical influence experiment of "in fact"** is the core ablation:
- Adding "in fact" increased preference for double negation $\rightarrow$ indicating LLMs rely on lexical co-occurrence patterns rather than semantic understanding.
- Meanwhile, it decreased preference for existential quantifiers $\rightarrow$ suggesting "in fact" usually co-occurs with negative/reversing contexts, leading to erroneous inferences by LLMs.
- Human performance remains stable across both conditions $\rightarrow$ human understanding is based on structural abstraction rather than lexical cues.

### Key Findings

1. **LLMs and humans align on certain tasks**: Basic scope constraints of universal quantifiers are correctly learned by all LLMs.
2. **LLMs do not understand double negation resolution**: They fail to equate double negation with existential quantification correctly.
3. **LLM discourse understanding relies on lexical cues rather than structure**: The presence of "either" and the addition of "in fact" affect decisions when they semantically should not.
4. **Humans exhibit structural sensitivity that LLMs lack**: Particularly shown in the telescoping effect and stable understanding of negation scope.
5. **Discourse-level understanding is a systematic weakness of LLMs**: Even when performing well at the sentence level, fundamental deficiencies remain at the discourse level.

## Highlights & Insights

- **Theory-driven evaluation design**: Test items are constructed strictly based on dynamic semantic theory rather than simple empiricism.
- **Three-level semantic understanding framework**: Provides a systematic framework for evaluating the semantic capabilities of LLMs.
- **Profound insights from human-model comparison**: Not only demonstrates where LLMs fail, but also explains **why** they fail—lexical dependency vs. structural abstraction.
- **Bridging formal semantics and NLP**: Introduces classic theories such as Heim (1983) and Groenendijk and Stokhof (1991) into LLM evaluation.

## Limitations & Future Work

- **Limited model scope**: Failed to test the latest models like GPT-4o (due to API restrictions on log-probability access).
- **Only English considered**: Rules of anaphora accessibility may differ across languages.
- **Template-generated stimulus materials**: While checked by experts, they may lack naturalness.
- **Only surprisal metric used**: Fails to probe the internal representations of the models directly.
- **Incomplete coverage of relevant semantic constructions**: e.g., other variants of conditionals, quantifier scope interactions, etc.
- **Sample size for human experiments (104) may be insufficient** for certain effects.

## Related Work & Insights

- Inherits the discourse entity recognition paradigm from Schuster and Linzen (2022) and Zhu and Frank (2024), significantly expanding the coverage of test constructions.
- Complements the entity tracking work of Li et al. (2021): while they use explicit state descriptions, this paper uses implicit semantic scope.
- Implications for LLM training: **Current pre-training objectives may be insufficient for learning discourse-level semantics**, requiring dedicated training signals.
- Implications for benchmark design: Discourse-level evaluation should be incorporated into the mainstream NLU evaluation system.

## Rating

- **Novelty**: ★★★★★ (Theory-driven systematic evaluation, filling an important gap)
- **Experimental Thoroughness**: ★★★★☆ (Three sets of experiments + human controls, but limited model scope)
- **Value**: ★★★☆☆ (Main contributions lie in theoretical insights, with limited engineering applications)
- **Writing Quality**: ★★★★★ (Rigorous arguments, excellent integration of theory and experiments)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Evaluating Language Models as Synthetic Data Generators](evaluating_lms_synthetic_data_gen.md)
- [\[ACL 2025\] Logical Forms Complement Probability in Understanding Language Model (and Human) Performance](logical_forms_complement_probability_in_understanding_language_model_and_human_p.md)
- [\[ACL 2025\] Evaluating Implicit Bias in Large Language Models by Attacking from a Psychometric Perspective](evaluating_implicit_bias_in_large_language_models_by_attacking_from_a_psychometr.md)
- [\[ACL 2025\] UAQFact: Evaluating Factual Knowledge Utilization of LLMs on Unanswerable Questions](uaqfact_evaluating_factual_knowledge_utilization_of_llms_on_unanswerable_questio.md)
- [\[ACL 2025\] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)

</div>

<!-- RELATED:END -->
