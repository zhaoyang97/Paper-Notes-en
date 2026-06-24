---
title: >-
  [Paper Note] RULEBREAKERS: Challenging LLMs at the Crossroads between Formal Logic and Human-like Reasoning
description: >-
  [ICML 2025][LLM (Other)][rulebreaker] Constructs the first large-scale "rulebreaker" dataset, RULEBREAKERS (25,600 instances), to systematically evaluate the performance of 7 LLMs when formal logical reasoning conflicts with factual knowledge. The study reveals that models generally tend to apply logical rules over-rigidly while ignoring common sense, deviating significantly from human reasoning behavior.
tags:
  - "ICML 2025"
  - "LLM (Other)"
  - "rulebreaker"
  - "formal logic"
  - "human-like reasoning"
  - "modus tollens"
  - "disjunctive syllogism"
date: 2026-05-08
content_hash: a3d05a300cd8dba9
---

# RULEBREAKERS: Challenging LLMs at the Crossroads between Formal Logic and Human-like Reasoning

**Conference**: ICML 2025  
**Authors**: Jason Chan, Robert Gaizauskas, Zhixue Zhao (University of Sheffield)  
**arXiv**: [2410.16502](https://arxiv.org/abs/2410.16502)  
**Code**: [GitHub](https://github.com/jasonchanly/rulebreakers)  
**Area**: LLM/NLP  
**Keywords**: rulebreaker, formal logic, human-like reasoning, modus tollens, disjunctive syllogism  

## TL;DR

Constructs the first large-scale "rulebreaker" dataset, RULEBREAKERS (25,600 instances), to systematically evaluate the performance of 7 LLMs when formal logical reasoning conflicts with factual knowledge. The study reveals that models generally tend to apply logical rules over-rigidly while ignoring common sense, deviating significantly from human reasoning behavior.

## Background & Motivation

**Background**: Formal logic is a classic approach to reasoning in NLP—converting natural language into symbolic forms and then applying predefined rules to derive conclusions. An increasing number of studies leverage formal logic to enhance LLM reasoning capabilities, including logical data augmentation, logically constrained reasoning, and symbolic solver integration.

**Limitations of Prior Work**: Cognitive science research (such as Johnson-Laird et al.) has long established that reasoning based purely on logical form fundamentally differs from how humans actually reason. The core issue lies in "rulebreaker" scenarios: although conclusions derived from formal logic are valid at the symbolic level, they factually contradict the premises, and humans typically do not accept such conclusions. For example: "Anne is in Stockholm or somewhere in Sweden" and "Anne is not in Sweden"; disjunctive syllogism derives "Anne is in Stockholm"—but anyone who knows Stockholm is in Sweden would not accept this conclusion.

**Key Challenge**: Formal logic rules assume that connectives (if, or, and) have fixed semantics, but the interpretation of connectives in natural language is highly dependent on the semantic content of the sentences. Existing LLM reasoning evaluations assume that as long as the logical form is correct, the conclusion is correct, neglecting situations where semantic content should influence reasoning judgments.

**Goal**: (1) Build a strictly controlled rulebreaker evaluation benchmark; (2) Evaluate whether LLMs can distinguish between rulebreakers and non-rulebreakers like humans; (3) Analyze potential reasons for failure.

**Key Insight**: Starting from experimental paradigms in cognitive science, design minimally different rulebreaker/non-rulebreaker pairs, controlling for the same logical form but different semantic content, to isolate and evaluate semantic awareness.

**Core Idea**: Use a cognitive-science-inspired rulebreaker dataset to test whether LLMs can integrate commonsense knowledge in reasoning instead of mechanically applying logical rules.

## Method

### Overall Architecture

RULEBREAKERS is an evaluation dataset containing 25,600 instances, consisting of 12,800 rulebreakers and 12,800 paired non-rulebreakers. The dataset is systematically generated via 4 templates (2 logical rules $\times$ 2 entity types) and evaluated using two types of metrics: paired accuracy and model confidence.

### Key Designs

1. **Four-Template Data Generation System**:

    - **Function**: Systematically generates minimally different rulebreaker/non-rulebreaker pairs.
    - **Mechanism**: Combines two logical rules (Modus Tollens MT: "if P then not Q; Q; ∴ not P" and Disjunctive Syllogism DS: "P or Q; not Q; ∴ P") with two entity types (geographic pairs: 183 country-capital pairs from WikiData; category pairs: 91 category-instance pairs from ConceptNet, covering birds, fish, musical instruments, etc.). Each entity pair is further combined with 5-6 verbs and 5 person names. Non-rulebreakers are created by randomly replacing country/category names so that the conclusions no longer contradict the premises.
    - **Design Motivation**: Template-based generation ensures large-scale, minimal-difference, and controlled variables, avoiding inconsistencies of manual construction.

2. **Paired Accuracy**:

    - **Function**: Evaluates the model's ability to correctly handle both rulebreakers and non-rulebreakers simultaneously.
    - **Mechanism**: $\tau = \frac{1}{|D^{paired}|}\sum_{(x^R, x^N)} \mathbb{1}[T(x^R) \wedge T(x^N)]$. The model is considered correct only when it answers "no" to the rulebreaker and "yes" to the paired non-rulebreaker. The random guessing baseline is 0.25, and a strategy of always answering "yes" scores 0.
    - **Design Motivation**: Eliminates the confounding effect of response bias (e.g., always answering "yes") on evaluation, providing a more rigorous measure than accuracy alone.

3. **Model Confidence Analysis**:

    - **Function**: Detects whether the model possesses a latent ability to distinguish rulebreakers.
    - **Mechanism**: Extracts the output probability $p^+(x)$ for "yes/true" from the model, and compares the positive (and correct) response confidence on non-rulebreakers $\Pi_{D^N}^+$ with the positive (but incorrect) response confidence on rulebreakers $\Pi_{D^R}^+$, testing for significance using Welch's t-test.
    - **Design Motivation**: Even if the final output of the model is incorrect, differences in confidence may reveal latent signals of discrimination.

### Loss & Training

This work is an evaluative study and does not involve model training.

## Experiments

### Main Results

| Model | Paired Accuracy $\tau$ | Rulebreaker Accuracy | Non-rulebreaker Accuracy |
|------|------------|-------------------|----------------------|
| LLaMA-3-8B-Instruct | **~0.609** | Medium | High |
| LLaMA-3-70B-Instruct | ~0.497 | Medium | High |
| Mistral-7B-Instruct | ~0.476 | Medium | High |
| Phi-3-medium-128k | ~0.292 | Low | High |
| Phi-3-mini-128k | ~0.208 | Low | Very High |
| GPT-4o | ~0.14 | Very Low | **100%** |
| Gemma-2-27b-it | ~0.071 | Very Low | **100%** |

### Ablation Study — Model Confidence and Failure Analysis

| Model | $\Pi_{D^R}^+$ (%) | $\Pi_{D^N}^+$ (%) | Statistical Significance |
|------|-------------------|-------------------|-----------|
| Phi-3-mini | 92.05 | 96.22 | p<0.0001 |
| Phi-3-medium | 93.96 | 97.16 | p<0.0001 |
| LLaMA-3-8B | 77.20 | 90.46 | p<0.0001 |
| LLaMA-3-70B | 96.34 | 99.95 | p<0.0001 |
| Mistral-7B | 92.55 | 98.11 | p<0.0001 |
| Gemma-2-27b | 97.99 | 100.00 | p<0.0001 |

### Key Findings

- **Counter-intuitive negative correlation**: The models performing best on non-rulebreakers (Gemma-2-27b, GPT-4o reaching 100%) perform the worst on rulebreakers, indicating that these models over-generalize logical rules.
- **Latent capability to distinguish exists**: All models express significantly higher positive response confidence on non-rulebreakers than on rulebreakers, indicating that latent discriminative signals exist but are not utilized.
- **Failure correlates with entity familiarity**: Except for LLaMA-3-8B and GPT-4o, the remaining models show higher entity familiarity in the correctly answered groups than in the incorrectly answered groups.
- **Attention allocation impact**: LLaMA-3-8B/70B, which achieve the highest accuracy, also show the highest ratio of attention allocated to the second premise (the factual premise).
- **DS outperforms MT, category outperforms location**: Model performance is generally better on Disjunctive Syllogism than on Modus Tollens, and better on category entity pairs than on geographic entity pairs.

## Highlights & Insights

- The first large-scale rulebreaker dataset, featuring an exquisite design, strict minimal-difference control, and evaluation metrics that eliminate response bias interference.
- Unveils a deep blind spot in LLM reasoning: models can be simultaneously proficient in fact retrieval and logical rule application, yet fail to recognize conflicts between the two.
- Issues an important warning to the current trend of relying on formal logic to enhance LLM reasoning—this may further exacerbate the divergence between LLMs and human reasoning.
- Confidence analysis shows that suppressed discriminative signals exist within the models, providing clues for future improvements.

## Limitations & Future Work

- Only covers two logical rules (MT and DS), omitting other reasoning patterns (such as hypothetical syllogism, reductio ad absurdum, etc.).
- Limited entity set (183 geographic pairs + 91 category pairs), which may not sufficiently represent the diversity of real-world knowledge.
- Evaluation is limited to multiple-choice formats, without fully testing reasoning behavior in open-ended generation scenarios.
- Does not propose concrete methods to improve LLM performance on rulebreakers, remaining at the diagnostic level.

## Related Work & Insights

- **Cognitive Science Foundations**: Directly inherits the experimental paradigm of the mental models theory (Johnson-Laird 1983) and systemizes the rulebreaker concept into an NLP evaluation tool.
- **Relation to Content Effects**: Lampinen et al. (2024) found that LLMs are also influenced by semantic content in formal logic tasks, but in the opposite direction—this paper focuses on scenarios where models should utilize semantic content but fail to do so.
- **Insights**: Training strategies (such as contrastive learning) can be designed to teach models when to apply logical rules versus when to rely on common sense.

## Rating

| Dimension | Score | Rationale |
|------|------|------|
| Novelty | ⭐⭐⭐⭐⭐ | First to systematically introduce the cognitive science concept of rulebreaker to LLM evaluation |
| Technical Depth | ⭐⭐⭐⭐ | Rigorous metric design and deep multi-dimensional analysis, though without model-level improvements |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Evaluates 7 models across 10 formulation variants, with a three-layer analysis: paired, confidence-based, and attribution-based |
| Writing Quality | ⭐⭐⭐⭐⭐ | Clearly articulates problem motivation with transparent dataset construction |
| Value | ⭐⭐⭐⭐ | The dataset is open-sourced, and the warnings regarding logic-enhancement methods have high practical guiding value |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] HumT DumT: Measuring and Controlling Human-like Language in LLMs](../../ACL2025/llm_nlp/humt_dumt_measuring_and_controlling_human-like_language_in_llms.md)
- [\[ACL 2025\] Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning](../../ACL2025/llm_nlp/problem-solving_logic_guided_curriculum_in-context_learning_for_llms_complex_rea.md)
- [\[ACL 2025\] Can We Further Elicit Reasoning in LLMs? Critic-Guided Planning with Retrieval-Augmentation for Solving Challenging Tasks](../../ACL2025/llm_nlp/can_we_further_elicit_reasoning_in_llms_critic-guided_planning_with_retrieval-au.md)
- [\[ICML 2025\] Regress, Don't Guess — A Regression-like Loss on Number Tokens for Language Models](regress_dont_guess_--_a_regression-like_loss_on_number_tokens_for_language_model.md)
- [\[ACL 2025\] Enhancing Conversational Agents with Theory of Mind: Aligning Beliefs, Desires, and Intentions for Human-Like Interaction](../../ACL2025/llm_nlp/enhancing_conversational_agents_with_theory_of_mind_aligning_beliefs_desires_and.md)

</div>

<!-- RELATED:END -->
