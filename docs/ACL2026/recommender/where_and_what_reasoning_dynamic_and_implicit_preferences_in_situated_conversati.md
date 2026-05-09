---
title: >-
  [Paper Note] Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation
description: >-
  [ACL 2026][Recommender Systems][Situated Conversational Recommendation] SiPeR addresses the challenge of dynamically shifting and implicitly expressed user preferences in situated conversational recommendation (SCR) via two mechanisms — Scene Transition Estimation ("Where") and Bayesian Inverse Inference ("What") — achieving improvements of 10.9% and 10.6% on SIMMC 2.1 and SCREEN, respectively.
tags:
  - ACL 2026
  - Recommender Systems
  - Situated Conversational Recommendation
  - Scene Transition
  - Bayesian Inverse Inference
  - Implicit Preference
  - Multimodal
date: 2026-05-08
content_hash: 5238eba7ec3af7dd
---

# Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation

**Conference**: ACL 2026
**arXiv**: [2604.20749](https://arxiv.org/abs/2604.20749)
**Code**: [https://github.com/DongdingLin/SiPeR](https://github.com/DongdingLin/SiPeR)
**Area**: Recommender Systems / Conversational Recommendation
**Keywords**: Situated Conversational Recommendation, Scene Transition, Bayesian Inverse Inference, Implicit Preference, Multimodal

## TL;DR

SiPeR addresses the challenge of dynamically shifting and implicitly expressed user preferences in situated conversational recommendation (SCR) via two mechanisms — Scene Transition Estimation ("Where") and Bayesian Inverse Inference ("What") — achieving improvements of 10.9% and 10.6% on SIMMC 2.1 and SCREEN, respectively.

## Background & Motivation

**State of the Field**: Conversational recommender systems (CRS) provide recommendations through natural language interaction, but most existing work focuses solely on textual exchanges, neglecting visual information and environmental context. Situated Conversational Recommendation (SCR) leverages visual scenes alongside dialogue to deliver context-aware recommendations, more closely mirroring real-world shopping scenarios.

**Limitations of Prior Work**: SCR presents two unique challenges — (1) User preferences are dynamic and shift with the scene: when a user in the formal wear section expresses interest in "outdoor hiking," the system must proactively navigate to the outdoor section, yet existing work ignores scene transition decisions; (2) User preferences are often implicit: a user saying "the size is right" but requesting other options signals that the recommended blue jeans do not match their true preference, and the system must infer that the user actually wants gray pants.

**Root Cause**: SCR requires simultaneously resolving two decisions — "Where" (in which scene to recommend) and "What" (which item to recommend) — yet prior research has focused primarily on dataset construction rather than framework design.

**Paper Goals**: (1) Design a scene transition estimation mechanism to determine when and where to transition between scenes; (2) Apply Bayesian inverse inference to deduce users' true implicit preferences from dialogue.

**Starting Point**: Users are modeled as rational agents (inspired by Bayesian inverse planning), whose utterances are "actions" executed to achieve latent goals. Preferences are inferred by comparing the likelihood ratio of two hypotheses — "like" versus "dislike."

**Core Idea**: Scene transitions are handled via a generate-then-retrieve strategy (first generating a target scene description, then retrieving the best-matching scene); item preference is inferred via Bayesian inverse inference (treating user utterances as observational signals of latent goals).

## Method

### Overall Architecture

SiPeR comprises two core mechanisms: (1) **Scene Transition Estimation (STE)** — employs an MLLM to determine whether a scene transition is needed and to predict the target scene, using a coarse-to-fine retrieval strategy to identify the best-matching scene; (2) **Bayesian Inverse Inference (BI-INF)** — formalizes preference reasoning as a POMDP, extracts user intent via dialogue state tracking, and ranks candidate items by comparing their preference probabilities through likelihood ratios.

### Key Designs

1. **Scene Transition Estimation (STE)**:

    - **Function**: Decides whether to transition scenes and, if so, to which scene.
    - **Mechanism**: A three-step pipeline — (a) each candidate scene is converted into a textual description (situated profile) by an MLLM; (b) given the dialogue history and current scene, an MLLM jointly generates a transition decision (Yes/No) and a target scene description; (c) coarse-to-fine retrieval — Top-N candidates are first retrieved via embedding similarity, then re-ranked by a fine-tuned LLM reranker. The transition probability is computed by normalizing the logits of the Yes/No tokens.
    - **Design Motivation**: Direct semantic reasoning over a large candidate scene pool is computationally intractable; the generate-then-retrieve decomposition reduces complexity.

2. **Bayesian Inverse Inference (BI-INF)**:

    - **Function**: Infers the user's true preferences over candidate items from dialogue.
    - **Mechanism**: Users are formalized as rational agents in a POMDP, where utterances are actions executed to achieve a goal (acquiring a target item). Structured intent tuples are extracted via dialogue state tracking. For each candidate item $m_i$, the likelihood ratio under two hypotheses is compared: $r_i = \mathbb{P}(\text{like} | \text{dialogue}) / \mathbb{P}(\text{dislike} | \text{dialogue})$. Concretely, a fine-tuned MLLM computes the probability of generating the observed dialogue state under the hypotheses that the user does or does not want the item.
    - **Design Motivation**: LLMs struggle to discern nuanced preferences from surface-level dialogue; the Bayesian framework provides more rigorous probabilistic reasoning.

3. **Dialogue State Tracking**:

    - **Function**: Converts natural language dialogue into structured intent representations.
    - **Mechanism**: A capable LLM is directly prompted to extract $\langle\text{intent, slot, value}\rangle$ tuples from dialogue history; manual validation yields 98.8% accuracy.
    - **Design Motivation**: Structured representations reduce the unpredictability of natural language inputs within the Bayesian inference process.

### Loss & Training

The reranker is optimized with negative log-likelihood. MLLM fine-tuning is applied for dialogue state generation and likelihood computation.

## Key Experimental Results

### Main Results

| Method | SIMMC 2.1 R@1 | SCREEN R@1 |
|--------|--------------|------------|
| GPT-4o (CoT) | 28.12 | 33.45 |
| Qwen2.5-VL (CoT) | 16.72 | 21.05 |
| **SiPeR** | **~39** | **~44** |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| SiPeR (full) | Best | STE + BI-INF |
| w/o STE | Degraded | Unable to handle scene transitions |
| w/o BI-INF | Degraded | Unable to infer implicit preferences |
| BI-INF → CoT | Significantly degraded | Validates probabilistic reasoning over heuristic reasoning |

### Key Findings

- SiPeR outperforms the best baselines by an average of 10.9% on SIMMC 2.1 and 10.6% on SCREEN.
- The likelihood ratio approach of Bayesian inverse inference substantially outperforms simple CoT reasoning, validating the advantage of a probabilistic framework for implicit preference reasoning.
- Scene transition estimation is critical for recommendations in dynamic, shifting environments — without STE, the system cannot recommend within the correct scene.

## Highlights & Insights

- Applying Bayesian inverse planning from cognitive science to conversational recommendation — treating user utterances as "actions" rather than "statements" — constitutes an elegant theoretical framework.
- The "Where + What" problem decomposition cleanly maps onto the two core challenges of SCR.
- The generate-then-retrieve scene transition strategy effectively balances semantic reasoning capability with computational efficiency.

## Limitations & Future Work

- Experiments are conducted solely on simulated datasets; real-world e-commerce scenarios entail substantially greater complexity.
- The Bayesian inference framework assumes users are rational agents, whereas actual user behavior may be non-rational.
- Dialogue state tracking relies on a capable LLM, which may be unsuitable for low-resource settings.

## Related Work & Insights

- **vs. Traditional CRS**: Conventional systems operate on text only; SiPeR jointly processes visual scenes and textual dialogue.
- **vs. BIP / Theory of Mind**: SiPeR introduces Bayesian inverse planning from computational cognitive science into recommender systems.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The application of Bayesian inverse inference to SCR and the Where+What problem decomposition are highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two benchmarks, comparisons with multiple baselines, and complete ablations, though real-world validation is absent.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation and methodology are articulated clearly.
- **Value**: ⭐⭐⭐⭐ Provides the first systematic framework for situated conversational recommendation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)
- [\[ACL 2026\] Personalized Benchmarking: Evaluating LLMs by Individual Preferences](personalized_benchmarking_evaluating_llms_by_individual_preferences.md)
- [\[ACL 2026\] IceBreaker for Conversational Agents: Breaking the First-Message Barrier with Personalized Starters](icebreaker_for_conversational_agents_breaking_the_first-message_barrier_with_per.md)
- [\[ACL 2026\] What Makes an Ideal Quote? Recommending "Unexpected yet Rational" Quotations via Novelty](what_makes_an_ideal_quote_recommending_34unexpected_yet_rational34_quotations_vi.md)
- [\[ACL 2026\] What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context](what_makes_llms_effective_sequential_recommenders_a_study_on_preference_intensit.md)

</div>

<!-- RELATED:END -->
