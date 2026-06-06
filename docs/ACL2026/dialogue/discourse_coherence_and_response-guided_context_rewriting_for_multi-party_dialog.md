---
title: >-
  [Paper Note] Discourse Coherence and Response-Guided Context Rewriting for Multi-Party Dialogue Generation
description: >-
  [ACL 2026][Dialogue Systems][Multi-party dialogue] This paper proposes DRCR, the first framework to introduce context rewriting into multi-party dialogue generation. It constructs preference data using dual feedback sign…
tags:
  - "ACL 2026"
  - "Dialogue Systems"
  - "Multi-party dialogue"
  - "context rewriting"
  - "discourse coherence"
  - "preference learning"
  - "dynamic self-evolution"
date: 2026-05-08
content_hash: e7a7a096add42ae2
---

# Discourse Coherence and Response-Guided Context Rewriting for Multi-Party Dialogue Generation

**Conference**: ACL 2026  
**arXiv**: [2604.06784](https://arxiv.org/abs/2604.06784)  
**Code**: None  
**Area**: Dialogue Systems / Multi-Party Dialogue  
**Keywords**: Multi-party dialogue, context rewriting, discourse coherence, preference learning, dynamic self-evolution

## TL;DR

This paper proposes DRCR, the first framework to introduce context rewriting into multi-party dialogue generation. It constructs preference data using dual feedback signals of discourse coherence and response quality, enabling the rewriter and responder to mutually enhance each other through iterative training via dynamic self-evolution.

## Background & Motivation

**Background**: Multi-party dialogue generation (MDG) involves multiple interlocutors and complex discourse structures (speaking relationships spanning multiple utterances), which is significantly more challenging than dyadic dialogue. Existing methods typically assist generation by encoding dialogue structure information.

**Limitations of Prior Work**: (1) Colloquial expressions and incomplete utterances (such as anaphora and ellipsis) in dialogues damage discourse coherence, which in turn affects the quality of dialogue structure representation. (2) Previous methods directly encode structures using flawed dialogue contexts without attempting to improve context quality first. (3) These problems are more prominent in multi-party dialogues where multiple speakers increase the complexity of references and omissions.

**Key Challenge**: The quality of dialogue structure encoding depends on the coherence of the context, but colloquial expressions and omissions in the original context disrupt this coherence. Simple rewriting may fail to balance discourse coherence with the quality of downstream response generation.

**Goal**: To improve the quality of multi-party dialogue generation through dialogue context rewriting, while ensuring that the rewriting both enhances discourse coherence and facilitates the generation of high-quality responses.

**Key Insight**: Utilize discourse coherence quality and response generation quality as dual feedback signals to construct preference data, training the rewriter to generate contexts that are both coherent and beneficial for responses.

**Core Idea**: The rewriter and the responder enhance each other through iterative training—better rewriting leads to better responses, and better response feedback guides better rewriting.

## Method

### Overall Architecture

DRCR consists of two modules: a Rewriter and a Responder, trained through three stages: (1) Supervised Fine-Tuning (SFT)—training the fundamental capabilities of the rewriter and responder separately; (2) Preference Data Construction—ranking rewriting results using dual signals of discourse coherence and response quality; (3) Dynamic Self-Evolution—continuous enhancement of the rewriter and responder through mutual feedback in iterative training.

### Key Designs

1.  **Discourse Coherence Feedback**:
    *   **Function**: Evaluates the quality of the discourse structure of the rewritten context.
    *   **Mechanism**: A discourse coherence evaluation model is used to score different rewriting results. Rewritings with higher coherence serve as "preferred" samples in the preference data. Coherence measures whether the rewriting eliminates referential ambiguity, completes omissions, and clarifies discourse relations.
    *   **Design Motivation**: The coherence of the dialogue context directly affects the quality of discourse structure encoding, which subsequently influences response generation.

2.  **Response Quality Feedback**:
    *   **Function**: Ensures that rewriting is conducive to generating high-quality responses.
    *   **Mechanism**: Contexts from different rewritings are input into the responder to compare the quality (relevance, informativeness, coherence) of the generated responses. Rewritings that yield better responses are labeled as "preferred."
    *   **Design Motivation**: The ultimate goal of rewriting is to improve response quality; optimizing for discourse coherence alone may not guarantee performance in downstream generation.

3.  **Dynamic Self-Evolution Learning**:
    *   **Function**: Enables the rewriter and responder to enhance each other during iterations.
    *   **Mechanism**: In each iteration, the rewriter is updated with feedback from the current responder. The updated rewriter generates better contexts, allowing the responder to improve further on superior contexts. Multiple iterations are performed until convergence.
    *   **Design Motivation**: Single-round training may fall into sub-optimal solutions—the rewriter may not know which specific rewritings truly benefit the current responder. Dynamic interaction allows for joint optimization.

### Loss & Training

Both the rewriter and the responder utilize DPO-style preference learning. Preference data is constructed from dual feedback signals (discourse coherence + response quality). Iterative training continues until rewriting and response quality stabilize.

## Key Experimental Results

### Main Results

**BLEU/ROUGE Scores on Four Multi-Party Dialogue Datasets**

| Method | Dataset 1 | Dataset 2 | Dataset 3 | Dataset 4 |
| :--- | :--- | :--- | :--- | :--- |
| SS-MPC (Prev. SOTA) | Baseline | Baseline | Baseline | Baseline |
| LLM Direct Gen | Medium | Medium | Medium | Medium |
| **DRCR** | **Gain** | **Gain** | **Gain** | **Gain** |

### Ablation Study

| Configuration | Effect | Description |
| :--- | :--- | :--- |
| Coherence Feedback Only | Limited Improvement | Lacks downstream signals |
| Response Quality Feedback Only | Improvement | Directly optimizes the goal |
| Dual Feedback | Optimal | Two signals are complementary |
| No Self-Evolution (Single Training) | Sub-optimal | Lacks collaborative optimization |
| With Self-Evolution | Optimal | Iterative enhancement |

### Key Findings

*   DRCR outperforms the Prev. SOTA across all four multi-party dialogue datasets.
*   Dual feedback signals are superior to a single signal—discourse coherence and response quality provide complementary perspectives.
*   Dynamic self-evolution iterative training significantly outperforms single-round training due to the synergy between the rewriter and responder.
*   Context rewriting effectively eliminates comprehension barriers caused by anaphora and ellipsis.

## Highlights & Insights

*   Introduces context rewriting to multi-party dialogue generation for the first time, addressing overlooked colloquialism issues.
*   The design of dual feedback combined with self-evolution forms an elegant closed-loop optimization.
*   As a pre-processing step, rewriting is orthogonal to existing generation methods and can be layered with them.

## Limitations & Future Work

*   Rewriting increases computational overhead during inference (additional rewriting step).
*   The number of iterations and convergence conditions for self-evolution need to be determined experimentally.
*   The method has only been validated on Chinese multi-party dialogue datasets; cross-lingual effectiveness remains to be confirmed.
*   Rewriting may introduce information bias, especially in scenarios involving ambiguous intentions.

## Related Work & Insights

*   **vs SS-MPC**: SS-MPC directly encodes original dialogue structures, while DRCR rewrites before encoding.
*   **vs Query Rewriting**: Query rewriting in search inspired dialogue context rewriting, but the structure of multi-party dialogue is more complex.

## Rating

*   Novelty: ⭐⭐⭐⭐ First application of context rewriting + dual feedback self-evolution in multi-party dialogue.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets and detailed ablations.
*   Writing Quality: ⭐⭐⭐⭐ Clear framework description and intuitive examples.
*   Value: ⭐⭐⭐⭐ Provides a new pre-processing paradigm for multi-party dialogue generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Context-Agent: Dynamic Discourse Trees for Non-Linear Dialogue](context-agent_dynamic_discourse_trees_for_non-linear_dialogue.md)
- [\[ACL 2026\] Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review](author-in-the-loop_response_generation_and_evaluation_integrating_author_experti.md)
- [\[ACL 2026\] SPASM: Stable Persona-driven Agent Simulation for Multi-turn Dialogue Generation](spasm_stable_persona-driven_agent_simulation_for_multi-turn_dialogue_generation.md)
- [\[ICLR 2026\] AQuA: Toward Strategic Response Generation for Ambiguous Visual Questions](../../ICLR2026/dialogue/aqua_toward_strategic_response_generation_for_ambiguous_visual_questions.md)
- [\[ACL 2026\] ETHICMIND: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](ethicmind_a_risk-aware_framework_for_ethical-emotional_alignment_in_multi-turn_d.md)

</div>

<!-- RELATED:END -->
