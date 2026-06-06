---
title: >-
  [Paper Note] Discourse Coherence and Response-Guided Context Rewriting for Multi-Party Dialogue Generation
description: >-
  [ACL 2026][Dialogue Systems][Multi-Party Dialogue] This paper proposes DRCR, the first framework to introduce context rewriting into multi-party dialogue generation…
tags:
  - "ACL 2026"
  - "Dialogue Systems"
  - "Multi-Party Dialogue"
  - "Context Rewriting"
  - "Discourse Coherence"
  - "Preference Learning"
  - "Dynamic Self-Evolution"
date: 2026-05-08
content_hash: 8836fbd3226ca939
---

# Discourse Coherence and Response-Guided Context Rewriting for Multi-Party Dialogue Generation

**Conference**: ACL 2026  
**arXiv**: [2604.06784](https://arxiv.org/abs/2604.06784)  
**Code**: None  
**Area**: Dialogue Systems / Multi-Party Dialogue  
**Keywords**: Multi-Party Dialogue, Context Rewriting, Discourse Coherence, Preference Learning, Dynamic Self-Evolution

## TL;DR

This paper proposes DRCR, the first framework to introduce context rewriting into multi-party dialogue generation, using dual feedback signals of discourse coherence and response quality to construct preference data, and enabling the rewriter and responder to mutually enhance each other through iterative training via dynamic self-evolution.

## Background & Motivation

**Background**: Multi-party dialogue generation (MDG) involves multiple participants and complex discourse structures (utterance relationships spanning multiple turns), making it significantly more challenging than two-party dialogue. Existing methods assist generation by encoding dialogue structure information.

**Limitations of Prior Work**: (1) Colloquial expressions and incomplete utterances in conversations (e.g., references, ellipsis) damage discourse coherence, thereby affecting the quality of dialogue structure representations; (2) Previous methods directly encode structure from flawed dialogue contexts without attempting to improve context quality first; (3) These issues are more prominent in multi-party dialogues—multiple speakers increase the complexity of references and ellipsis.

**Key Challenge**: The quality of dialogue structure encoding depends on context coherence, but colloquial expressions and ellipsis in raw contexts break coherence. Simple rewriting may fail to balance discourse coherence and downstream response generation quality.

**Goal**: Improve multi-party dialogue generation quality through dialogue context rewriting, ensuring the rewriting both enhances discourse coherence and facilitates high-quality response generation.

**Key Insight**: Use discourse coherence quality and response generation quality as dual feedback signals to construct preference data, training the rewriter to generate contexts that are both coherent and conducive to responses.

**Core Idea**: The rewriter and responder mutually enhance each other through iterative training—better rewriting produces better responses, and better response feedback guides better rewriting.

## Method

### Overall Architecture

DRCR consists of two modules: Rewriter and Responder, trained through three stages: (1) Supervised fine-tuning—training basic capabilities of rewriter and responder separately; (2) Preference data construction—ranking rewriting results using dual signals of discourse coherence and response quality; (3) Dynamic self-evolution—rewriter and responder continuously improve through mutual feedback in iterative training.

### Key Designs

1. **Discourse Coherence Feedback**:

    - Function: Evaluates dialogue structure quality of rewritten contexts
    - Mechanism: Uses a discourse coherence evaluation model to score different rewriting results, with more coherent rewrites serving as "preferred" samples in preference data. Coherence measures whether rewriting eliminates referential ambiguity, completes ellipsis, and streamlines discourse relations
    - Design Motivation: Context coherence directly affects dialogue structure encoding quality, thereby impacting response generation

2. **Response Quality Feedback**:

    - Function: Ensures rewriting facilitates high-quality response generation
    - Mechanism: Inputs contexts from different rewrites to the responder, comparing quality of generated responses (relevance, informativeness, coherence). Rewrites producing better responses are marked as "preferred"
    - Design Motivation: The ultimate goal of rewriting is to improve response quality; optimizing only discourse coherence may be insufficient to guarantee downstream generation effectiveness

3. **Dynamic Self-Evolution Learning**:

    - Function: Enables rewriter and responder to mutually enhance through iterations
    - Mechanism: In each iteration, the rewriter updates using current responder feedback, the updated rewriter produces better contexts, and the responder further improves on better contexts. Multiple iterations continue until convergence
    - Design Motivation: Single-round training may fall into suboptimal solutions—the rewriter doesn't know what rewrites truly benefit the current responder; dynamic interaction allows collaborative optimization

### Loss & Training

Both rewriter and responder use DPO-style preference learning. Preference data is constructed from dual feedback signals (discourse coherence + response quality). Iterative training continues until rewriting and response quality stabilize.

## Key Experimental Results

### Main Results

**BLEU/ROUGE scores on four multi-party dialogue datasets**

| Method | Dataset1 | Dataset2 | Dataset3 | Dataset4 |
|------|--------|--------|--------|--------|
| SS-MPC (Prev. SOTA) | baseline | baseline | baseline | baseline |
| LLM direct generation | moderate | moderate | moderate | moderate |
| **DRCR** | **surpasses** | **surpasses** | **surpasses** | **surpasses** |

### Ablation Study

| Config | Performance | Note |
|------|------|------|
| Coherence feedback only | Limited improvement | Lacks downstream signal |
| Response quality feedback only | Improvement | Directly optimizes objective |
| Dual feedback | Optimal | Two signals complement |
| No self-evolution (single training) | Suboptimal | Lacks collaborative optimization |
| With self-evolution | Optimal | Iterative improvement |

### Key Findings

- DRCR surpasses previous SOTA on all four multi-party dialogue datasets
- Dual feedback signals outperform single signals—discourse coherence and response quality provide complementary perspectives
- Dynamic self-evolution iterative training significantly outperforms single-round training—synergy between rewriter and responder
- Context rewriting effectively eliminates understanding barriers caused by references and ellipsis

## Highlights & Insights

- First to introduce context rewriting into multi-party dialogue generation—addresses overlooked colloquial issues
- Dual feedback + self-evolution design forms an elegant closed-loop optimization
- Rewriting as a preprocessing step for generation is orthogonal to existing generation methods and can be combined

## Limitations & Future Work

- Rewriting increases inference-time computational overhead (additional rewriting step)
- Iteration count and convergence criteria for self-evolution require experimental determination
- Validated only on Chinese multi-party dialogue datasets; cross-lingual effectiveness remains to be confirmed
- Rewriting may introduce information bias, especially in scenarios involving ambiguous intent

## Related Work & Insights

- **vs SS-MPC**: SS-MPC directly encodes original dialogue structure; DRCR rewrites before encoding
- **vs Query Rewriting**: Query rewriting in search inspired dialogue context rewriting, but multi-party dialogue has more complex structure

## Rating

- Novelty: ⭐⭐⭐⭐ First application of context rewriting + dual feedback self-evolution in multi-party dialogue
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets, detailed ablations
- Writing Quality: ⭐⭐⭐⭐ Clear framework description, intuitive examples
- Value: ⭐⭐⭐⭐ Provides new preprocessing paradigm for multi-party dialogue generation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SPASM: Stable Persona-driven Agent Simulation for Multi-turn Dialogue Generation](spasm_stable_persona-driven_agent_simulation_for_multi-turn_dialogue_generation.md)
- [\[ACL 2026\] Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review](author-in-the-loop_response_generation_and_evaluation_integrating_author_experti.md)
- [\[ICLR 2026\] AQuA: Toward Strategic Response Generation for Ambiguous Visual Questions](../../ICLR2026/dialogue/aqua_toward_strategic_response_generation_for_ambiguous_visual_questions.md)
- [\[ACL 2026\] ETHICMIND: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](ethicmind_a_risk-aware_framework_for_ethical-emotional_alignment_in_multi-turn_d.md)
- [\[ACL 2026\] ODUTQA-MDC: A Task for Open-Domain Underspecified Tabular QA with Multi-turn Dialogue-based Clarification](odutqa-mdc_a_task_for_open-domain_underspecified_tabular_qa_with_multi-turn_dial.md)

</div>

<!-- RELATED:END -->
