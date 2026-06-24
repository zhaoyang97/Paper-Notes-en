---
title: >-
  [Paper Note] Meaning Beyond Truth Conditions: Evaluating Discourse Level Understanding via Anaphora Accessibility
description: >-
  [ACL 2025][LLM (Other)][Discourse Semantics] This paper proposes a three-level hierarchical system of natural language understanding capabilities (lexical/sentential/discourse), utilizing anaphora accessibility as a diagnostic task for discourse-level understanding. Through an evaluation dataset inspired by dynamic semantics, it systematically investigates LLMs' discourse understanding capabilities under three linguistic structures: universal quantifiers, negation…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Discourse Semantics"
  - "Anaphora Accessibility"
  - "Dynamic Semantics"
  - "Quantifier Scope"
  - "LLM Evaluation"
date: 2026-05-08
content_hash: 13d4782db848c451
---

# Meaning Beyond Truth Conditions: Evaluating Discourse Level Understanding via Anaphora Accessibility

**Conference**: ACL 2025  
**arXiv**: [2502.14119](https://arxiv.org/abs/2502.14119)  
**Code**: None  
**Area**: NLP Understanding  
**Keywords**: Discourse Semantics, Anaphora Accessibility, Dynamic Semantics, Quantifier Scope, LLM Evaluation

## TL;DR

This paper proposes a three-level hierarchical system of natural language understanding capabilities (lexical/sentential/discourse), utilizing anaphora accessibility as a diagnostic task for discourse-level understanding. Through an evaluation dataset inspired by dynamic semantics, it systematically investigates LLMs' discourse understanding capabilities under three linguistic structures: universal quantifiers, negation, and disjunction.

## Background & Motivation

1. **Background**: Existing NLU evaluations primarily focus on the lexical level (word sense disambiguation, analogical reasoning) and sentential level (NLI, entailment judgment), while assessments of discourse-level semantic understanding remain far from adequate.
2. **Limitations of Prior Work**: The few existing discourse-level evaluation works either target simplistic scenarios (such as entity tracking) or only consider the scope of negation, lacking a systematic investigation into the interaction of various logical connectives (e.g., universal quantifiers, conditionals, disjunctions) with discourse referents (entities).
3. **Key Challenge**: Discourse understanding requires not only sentential truth-conditional semantics but also the capability to dynamically update discourse states; however, it remains unknown whether LLMs truly possess this structure-based capability for state updates.
4. **Goal**: Systematically evaluate whether LLMs understand how the scope of various semantic operators affects the accessibility (referentiality) of discourse referents.
5. **Key Insight**: Utilize precise predictions regarding anaphora accessibility from formal semantics (specifically, dynamic semantics) to design minimal-pair evaluation stimuli.
6. **Core Idea**: If LLMs possess discourse-level understanding, they should assign higher probabilities to licensing continuations—i.e., allowing anaphoric reference after existential quantifiers, but rendering it inaccessible after universal quantifiers.

## Method

### Overall Architecture

Three sets of experiments are designed to test the effects of universal quantifiers (every/if/whenever), negation (single/double negation), and disjunction (either...or/and) on the accessibility of discourse referents. Surprisal (negative log-probability) is used to measure LLMs' acceptability of continuations, which is then compared with human experiments using a forced-choice paradigm.

### Key Designs

1. **Universal Quantifiers Experiment**:

    - **Function**: Test whether LLMs distinguish the discourse referent accessibility between existential and universal quantifiers
    - **Mechanism**: Compare pronoun resolution following "A farmer worked..." (which should be accessible) vs. "Every farmer worked..." (which should be inaccessible). A difference-of-difference metric is employed to compare whether the probability difference between inter-sentential and intra-sentential continuation conditions differs under existential vs. universal environments, thereby factoring out simple lexical biases.
    - **Design Motivation**: Merely comparing absolute probabilities is too weak of a test—even if an LLM prefers the existential condition, it could be due to lexical preferences rather than genuine discourse understating.

2. **Negation/Double Negation Experiment**:

    - **Function**: Test whether LLMs understand that negation blocks accessibility while double negation restores it
    - **Mechanism**: Compare the probabilities of continuations under three conditions: "The farmer owned a cow" (accessible) vs. "didn't own a cow" (inaccessible) vs. "It was not the case that...didn't own" (double negation = accessible). Furthermore, "in fact" is introduced to probe the influence of lexical cues.
    - **Design Motivation**: Double negation elimination is a sophisticated semantic inference—two negations canceling each other out should restore anaphoric accessibility. This serves as a litmus test for genuine structural understanding vs. surface lexical pattern matching.

3. **Disjunction Experiment**:

    - **Function**: Test Evans' observation regarding the special behavior of negative quantifiers in disjunctions
    - **Mechanism**: Compare "Either there was no manuscript, or it was hidden" (licensing/valid) vs. "Either there was a manuscript, or it was hidden" (unlicensing/invalid). SLOR (Syntactic Log-Odds Ratio) scores are used to normalize the effects of sentence length and word frequency.
    - **Design Motivation**: Anaphora accessibility in disjunctions is one of the most subtle predictions in formal semantics, providing a rigorous test for LLMs' discourse understanding.

### Loss & Training

This paper is an evaluation study with no training. It evaluates the Llama 3 series (1B/3B/8B/8B-Instruct) and GPT babbage-002/davinci-002. For the human baseline, 104 participants were recruited on the Prolific platform to conduct a forced-choice experiment.

## Key Experimental Results

### Main Results

| Experiment | LLM Performance | Human Performance | Description |
|------|---------|---------|------|
| exi > every | ~75% | Close to 100% | LLMs succeed but underperform humans |
| exi > if | ~100% | ~70% | LLMs outperform humans (telescoping effect) |
| exi > neg | Above chance | Above chance | Both succeed |
| DN > neg | Reversal in some models | Above chance | LLMs struggle with double negation |
| EitherOr > And | ~100% | Above chance | LLMs succeed |
| or > EitherPosOr | Reversal! | No distinct preference | LLMs over-rely on the lexical cue "either" |

### Ablation Study (Adding "in fact")

| Configuration | Effect | Description |
|------|------|------|
| DN>Neg + "in fact" | Accuracy increases | The co-occurrence pattern of "in fact" and double negation helps LLMs |
| Exi>Neg + "in fact" | Direction reversal! | "in fact" is less likely to appear after existential sentences, lowering the probability |
| Human + "in fact" | No change | Humans are not distracted by lexical cues |

### Key Findings

- LLMs perform well on simple existential/universal contrasts, but their success partially relies on lexical cues rather than structural understanding.
- Double negation is a weak spot for LLMs—most models fail to correctly handle double negation elimination.
- Key finding in disjunctions: LLMs rely excessively on the word "either"—exhibiting a preference when "either" is present and none when it is absent, even when the sentences are semantically equivalent.
- Humans and LLMs exhibit opposite behaviors in conditionals containing "he"-continuations (vs. "it"-continuations), likely due to the human tendency for telescoping.

## Highlights & Insights

- The **three-level semantic understanding hierarchical system** (lexical $\to$ sentential $\to$ discourse) is a valuable conceptual framework that provides systematic guidance for evaluating NLU.
- Ingenious experimental design: It utilizes precise predictions from formal semantics to design minimal pairs and incorporates human subject experiments, enabling a direct comparison between LLMs and humans.
- The finding that **LLMs rely on lexical cues rather than structural abstraction** is highly significant, suggesting that current "understanding" of discourse by LLMs might be mostly surface pattern matching.

## Limitations & Future Work

- Evaluated only relatively small models from the Llama 3 and GPT series; SOTA models like GPT-4o could not be tested because their APIs do not support logprobs.
- The evaluation construction is relatively simple and does not cover more complex linguistic structures (such as modal subordination).
- Behavioral-level evaluation cannot reveal the internal mechanisms of the models, requiring mechanistic interpretability methods as a complement.

## Related Work & Insights

- **vs. Schuster & Linzen (2022)**: Tested only the effect of negation on discourse referents, whereas this study expands to universal quantifiers and disjunctions.
- **vs. Kim & Schuster (2023)**: Evaluated state tracking using simple scenarios (such as moving objects within boxes), whereas this work focuses on the semantic structures of natural language.
- **vs. Li et al. (2021)**: Probed entity tracking representations in internal states of transformers, whereas this work provides a complementary evaluation at the behavioral level.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematically applies formal semantics predictions to LLM evaluation, offering a unique perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Human baseline experiments + multiple LLMs + various linguistic structures, though the range of models is somewhat limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-established theoretical background and a clear conceptual hierarchy.
- Value: ⭐⭐⭐⭐ Highlights the limitations of LLMs' discourse understanding, providing valuable insights for NLU research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Evaluating Language Models as Synthetic Data Generators](evaluating_lms_synthetic_data_gen.md)
- [\[ACL 2025\] Logical Forms Complement Probability in Understanding Language Model (and Human) Performance](logical_forms_complement_probability_in_understanding_language_model_and_human_p.md)
- [\[ACL 2025\] Evaluating Implicit Bias in Large Language Models by Attacking from a Psychometric Perspective](evaluating_implicit_bias_in_large_language_models_by_attacking_from_a_psychometr.md)
- [\[ACL 2025\] UAQFact: Evaluating Factual Knowledge Utilization of LLMs on Unanswerable Questions](uaqfact_evaluating_factual_knowledge_utilization_of_llms_on_unanswerable_questio.md)
- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)

</div>

<!-- RELATED:END -->
