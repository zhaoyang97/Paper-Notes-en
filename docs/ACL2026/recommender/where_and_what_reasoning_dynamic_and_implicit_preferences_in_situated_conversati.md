---
title: >-
  [Paper Note] Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation
description: >-
  [ACL 2026][Recommender Systems][Situated Conversational Recommendation] SiPeR addresses the challenges of dynamic environment-dependent user preferences and implicit expressions in situated conversational recommendation…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "Situated Conversational Recommendation"
  - "Scene Transition"
  - "Bayesian Inverse Reasoning"
  - "Implicit Preference"
  - "Multimodal"
date: 2026-05-08
content_hash: ab180fa5fcd190d9
---

# Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation

**Conference**: ACL 2026  
**arXiv**: [2604.20749](https://arxiv.org/abs/2604.20749)  
**Code**: [https://github.com/DongdingLin/SiPeR](https://github.com/DongdingLin/SiPeR)  
**Area**: Recommender Systems / Conversational Recommendation  
**Keywords**: Situated Conversational Recommendation, Scene Transition, Bayesian Inverse Reasoning, Implicit Preference, Multimodal

## TL;DR

SiPeR addresses the challenges of dynamic environment-dependent user preferences and implicit expressions in situated conversational recommendation via Scene Transition Estimation ("Where") and Bayesian Inverse Reasoning ("What"), achieving improvements of 10.9% and 10.6% on SIMMC 2.1 and SCREEN, respectively.

## Background & Motivation

**Background**: Conversational Recommender Systems (CRS) provide recommendations through natural language interaction. However, most focus solely on text, ignoring visual information and environmental factors. Situated Conversational Recommendation (SCR) utilizes visual scenes and dialogue to provide context-aware recommendations, aligning more closely with real-world shopping scenarios.

**Limitations of Prior Work**: SCR faces two unique challenges: (1) user preferences are dynamic and vary with the scene—when a user expresses interest in "outdoor hiking" while in a formal wear section, the system must proactively transition to the outdoor section, yet existing work ignores scene transition decisions; (2) user preferences are often implicit—if a user says "the size is right" but asks for other options, it implies the recommended blue jeans do not meet the true preference, requiring the system to infer that the user actually wants gray pants.

**Key Challenge**: SCR must simultaneously address "Where" (in which scene to recommend) and "What" (which items to recommend) decisions, but existing research primarily focuses on dataset construction rather than framework design.

**Goal**: (1) Design a Scene Transition Estimation mechanism to determine when and where to transition scenes; (2) use Bayesian Inverse Reasoning to infer the user's true implicit preferences from the dialogue.

**Key Insight**: Users are treated as rational agents (inspired by Bayesian Inverse Planning) whose utterances are "actions" performed to achieve latent goals. Preferences are reasoned by comparing the likelihood ratios of "like" and "dislike" hypotheses.

**Core Idea**: Scene transitions utilize a "generation-retrieval" strategy (generating target scene descriptions before retrieving matching scenes), while item preferences use Bayesian Inverse Reasoning (treating user utterances as observed signals of latent goals).

## Method

### Overall Architecture

SiPeR consists of two core mechanisms: (1) Scene Transition Estimation (STE), which uses an MLLM to determine if a transition is needed and predicts the target scene via a coarse-to-fine retrieval strategy; (2) Bayesian Inverse Inference (BI-INF), which formalizes preference reasoning as a POMDP, extracts user intent through Dialogue State Tracking, and compares preference probabilities for candidate items via likelihood ratios.

### Key Designs

1.  **Scene Transition Estimation (STE)**:
    *   **Function**: Determines whether to transition scenes and identifies the target scene.
    *   **Mechanism**: A three-step process: (a) each candidate scene is converted into a textual "situated profile" using an MLLM; (b) given dialogue history and the current scene, an MLLM jointly generates the transition decision (Yes/No) and a target scene description; (c) coarse-to-fine retrieval—first retrieving Top-N candidates by embedding similarity, then re-ranking with a trained LLM reranker. Transition probability is calculated by normalizing the logits of the Yes/No tokens.
    *   **Design Motivation**: Direct semantic reasoning over large-scale candidate scenes is infeasible; the generation-retrieval decomposition reduces computational complexity.

2.  **Bayesian Inverse Inference (BI-INF)**:
    *   **Function**: Infers true user preferences for candidate items from dialogue.
    *   **Mechanism**: Formalizes the user as a rational agent in a POMDP, where utterances are actions taken to reach a goal (obtaining the target item). Dialogue State Tracking extracts structured intent tuples. For each candidate item $m_i$, the likelihood ratio of two hypotheses is compared: $r_i = \mathbb{P}(\text{like} | \text{dialogue}) / \mathbb{P}(\text{dislike} | \text{dialogue})$. In practice, a fine-tuned MLLM calculates the probability of generating the observed dialogue state under the hypotheses "user wants the item" and "user does not want the item."
    *   **Design Motivation**: LLMs struggle to distinguish subtle preferences from surface-level dialogue; the Bayesian framework provides more rigorous probabilistic reasoning.

3.  **Dialogue State Tracking**:
    *   **Function**: Converts natural language dialogue into structured intent representations.
    *   **Mechanism**: Instructs a strong LLM to extract ⟨intent, slot, value⟩ tuples from the dialogue history, achieving 98.8% manual verification accuracy.
    *   **Design Motivation**: Structured representations reduce the uncontrollability of the natural language space during Bayesian inference.

### Loss & Training

The Reranker is optimized using Negative Log-Likelihood (NLL). MLLM fine-tuning is applied for dialogue state generation and likelihood computation.

## Key Experimental Results

### Main Results

| Method | SIMMC 2.1 R@1 | SCREEN R@1 |
| :--- | :--- | :--- |
| GPT-4o (CoT) | 28.12 | 33.45 |
| Qwen2.5-VL (CoT) | 16.72 | 21.05 |
| **SiPeR (Ours)** | **~39** | **~44** |

### Ablation Study

| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| SiPeR Full | Optimal | STE + BI-INF |
| w/o STE | Decrease | Unable to handle scene transitions |
| w/o BI-INF | Decrease | Unable to infer implicit preferences |
| Replace BI-INF with CoT | Significant Decrease | Verifies probabilistic reasoning outperforming heuristic reasoning |

### Key Findings

*   SiPeR outperforms the best baselines by an average of 10.9% on SIMMC 2.1 and 10.6% on SCREEN.
*   The likelihood ratio approach in Bayesian Inverse Reasoning significantly outperforms simple CoT reasoning, validating the advantage of the probabilistic framework for implicit preference inference.
*   Scene Transition Estimation is critical for recommendation in dynamic scene changes—without STE, the system cannot recommend within the correct context.

## Highlights & Insights

*   Applying Bayesian Inverse Planning from cognitive science to conversational recommendation by treating user utterances as "actions" rather than "statements" is an elegant theoretical framework.
*   The "Where + What" problem decomposition clearly addresses the two core challenges of SCR.
*   The generation-retrieval scene transition strategy effectively balances semantic reasoning capability with computational efficiency.

## Limitations & Future Work

*   Experiments were validated only on simulated datasets; real-world e-commerce scenarios involve higher complexity.
*   Bayesian reasoning assumes users are "rational agents," but real user behavior may be irrational.
*   Dialogue State Tracking relies on strong LLMs, which may not be applicable in low-resource scenarios.

## Related Work & Insights

*   **vs Traditional CRS**: Traditional systems handle only text, while SiPeR processes visual scenes plus textual dialogue.
*   **vs BIP/Theory of Mind**: SiPeR introduces Bayesian Inverse Planning from computational cognitive science into recommender systems.

## Rating

*   **Novelty**: ⭐⭐⭐⭐⭐ Functional application of Bayesian Inverse Reasoning in SCR and the "Where+What" problem decomposition are highly novel.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid across two benchmarks with multiple baselines and complete ablations, though lacking real-world scenario validation.
*   **Writing Quality**: ⭐⭐⭐⭐ Clear exposition of motivation and methodology.
*   **Value**: ⭐⭐⭐⭐ Provides the first systematic framework for situated conversational recommendation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)
- [\[ACL 2026\] Intent-Driven Semantic ID Generation for Grounded Conversational News Recommendation](intent-driven_semantic_id_generation_for_grounded_conversational_news_recommenda.md)
- [\[ACL 2026\] ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning](rerec_reasoning-augmented_llm-based_recommendation_assistant_via_reinforcement_f.md)
- [\[ACL 2026\] IceBreaker for Conversational Agents: Breaking the First-Message Barrier with Personalized Starters](icebreaker_for_conversational_agents_breaking_the_first-message_barrier_with_per.md)
- [\[ACL 2026\] What Makes an Ideal Quote? Recommending "Unexpected yet Rational" Quotations via Novelty](what_makes_an_ideal_quote_recommending_34unexpected_yet_rational34_quotations_vi.md)

</div>

<!-- RELATED:END -->
