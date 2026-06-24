---
title: >-
  [Paper Note] LLMs Know Their Vulnerabilities: Uncover Safety Gaps through Natural Distribution Shifts
description: >-
  [ACL 2025][LLM (Other)][jailbreak] Proposes ActorBreaker, a multi-turn attack method based on Latour's Actor-Network Theory. By leveraging benign prompts semantically related to harmful content (natural distribution shifts) to bypass safety mechanisms, it achieves state-of-the-art (SOTA) attack success rates on HarmBench, revealing the semantic coverage gap between pre-training and safety training data.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "jailbreak"
  - "actor-network theory"
  - "multi-turn attack"
  - "safety alignment"
  - "natural distribution shift"
date: 2026-05-08
content_hash: c896506d99310594
---

# LLMs Know Their Vulnerabilities: Uncover Safety Gaps through Natural Distribution Shifts

**Conference**: ACL 2025  
**arXiv**: [2410.10700](https://arxiv.org/abs/2410.10700)  
**Code**: [https://github.com/AI45Lab/ActorAttack](https://github.com/AI45Lab/ActorAttack)  
**Area**: LLM/NLP  
**Keywords**: jailbreak, actor-network theory, multi-turn attack, safety alignment, natural distribution shift

## TL;DR
Proposes ActorBreaker, a multi-turn attack method based on Latour's Actor-Network Theory. By leveraging benign prompts semantically related to harmful content (natural distribution shifts) to bypass safety mechanisms, it achieves state-of-the-art (SOTA) attack success rates on HarmBench, revealing the semantic coverage gap between pre-training and safety training data.

## Background & Motivation

**Background**: LLM safety training teaches models to refuse harmful queries, yet adversarial attacks (e.g., GCG, jailbreak) can bypass them. Most attacks utilize adversarial distribution shifts (e.g., ciphertexts, low-resource language translation).

**Limitations of Prior Work**: Existing multi-turn attacks rely on fixed strategies (e.g., role-play, hypothetical scenarios), lacking diversity. Moreover, the use of "unnatural" prompts makes them easily detectable.

**Key Challenge**: The pre-training data contains a vast amount of seemingly benign knowledge semantically related to harmful topics (e.g., the association between "Ted Kaczynski" and bomb-making), but safety training fails to cover these indirect associations.

**Goal**: To construct benign yet effective multi-turn attacks by leveraging natural semantic associations within the pre-training distribution.

**Key Insight**: Latour's Actor-Network Theory (ANT)—decomposing a harmful target into six categories of actors (creator, disseminator, recipient, regulator, etc.), each containing human and non-human entities.

**Core Idea**: Utilizing the LLM's own knowledge to construct a semantic association network of harmful content, and using benign multi-turn dialogues to progressively guide the model to expose unsafe content.

## Method

### Overall Architecture
Two phases: (1) Network Construction—Given a harmful query, the LLM is used to build an actor-network (six categories of actors $\times$ human/non-human entities), where each node acts as a potential attack clue; (2) Attack Generation—Selecting an actor and its semantic relationship with the harmful target as a clue, then generating multi-turn benign prompts to progressively guide the model.

### Key Designs

1. **Actor-Network Construction**

    - Six types of actors: creator, disseminator, recipient, regulator, influencer, or associator.
    - Each type is categorized into human (historical figures, etc.) and non-human (books, media, social movements) entities.
    - Instantiating the network using the LLM's own knowledge.
    - Design Motivation: Latour’s theory ensures comprehensive coverage of potential attack paths.

2. **Attack Chain Generation**

    - Selecting an actor and its semantic relationship with the target.
    - Designing multi-turn seemingly harmless questions to gradually approach the target.
    - Design Motivation: Each turn is benign when viewed individually, but they collectively guide the model.

3. **Natural Distribution Shifts vs. Adversarial Distribution Shifts**

    - The prompts in this work are within the pre-training distribution (naturally benign).
    - Prompts in prior methods are out-of-distribution (encryption/role-play).
    - Design Motivation: Natural prompts do not trigger safety detectors.

## Key Experimental Results

### Main Results — HarmBench Attack Success Rate

| Method | Type | GPT-4o | Claude-3 | Llama-3-70B | Average |
|------|------|--------|---------|------------|------|
| GCG | Single-turn | ~20% | ~15% | ~35% | ~23% |
| PAIR | Single-turn | ~30% | ~25% | ~40% | ~32% |
| Crescendo | Multi-turn | ~40% | ~35% | ~50% | ~42% |
| **ActorBreaker** | **Multi-turn** | **~55%** | **~45%** | **~65%** | **~55%** |

### Safety Detector Bypassing

| Method | Llama-Guard-2 Detection Rate |
|------|-------------------|
| GCG prompt | ~80% (easily detected) |
| PAIR prompt | ~60% |
| Crescendo prompt | ~40% |
| **ActorBreaker prompt** | **~5%** (barely detected) |

### Defense Effectiveness

| Configuration | Attack Success Rate |
|------|----------|
| Baseline (No Defense) | 55% |
| Fine-tuned on ActorBreaker safety data | **25%** (-30%) |
| General capabilities degraded | ~3% (slight trade-off) |

### Key Findings
- **ActorBreaker achieves the highest attack success rate on all aligned LLMs**, including GPT-o1.
- **Llama-Guard-2 barely detects these prompts**, as each individual turn is benign.
- **Attack diversity far exceeds existing methods**, as the six types of actors provide rich attack paths.
- **Defense is effective but has a trade-off**: Fine-tuning on ActorBreaker data reduces the attack success rate by 30%, but slightly affects general capabilities.
- **The semantic gap between pre-training and safety training** resides as the fundamental problem.

## Highlights & Insights
- **Application of Actor-Network Theory in AI safety** is highly novel, transforming a sociological theory into a systematic red-teaming methodology.
- **The conceptual distinction between natural distribution shifts and adversarial distribution shifts** uncovers the fundamental limitations of safety alignment.
- **The recursive nature of using an LLM's own knowledge to attack itself** yields profound research insights.

## Limitations & Future Work
- The attack method could potentially be abused.
- The construction of the actor-network depends on the LLM itself; if safety training reaches near-perfection, it might fail to construct the network.
- Future Directions: Broader coverage in safety training data and safety mechanisms parameterized by semantic distance.

## Related Work & Insights
- **vs GCG/PAIR**: While they rely on adversarial distribution shifts, ActorBreaker utilizes natural distribution shifts.
- **vs Crescendo**: Crescendo relies on fixed templates, whereas ActorBreaker generates diverse paths grounded in network theory.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Innovative combination of Actor-Network Theory and natural distribution shifts.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluation across multiple models, multiple baseline attacks, and defense experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Solid theoretical foundation.
- Value: ⭐⭐⭐⭐⭐ Offers significant insights for AI safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cooperating and Competing Through Natural Language](cooperating_and_competing_through_natural_language.md)
- [\[ACL 2025\] Why Not Act on What You Know? Unleashing Safety Potential of LLMs via Self-Aware Guard Enhancement](why_not_act_on_what_you_know_unleashing_safety_potential_of_llms_via_self-aware_.md)
- [\[ACL 2025\] Can LLMs Ground when they (Don't) Know: A Study on Direct and Loaded Political Questions](can_llms_ground_when_they_dont_know_a_study_on_direct_and_loaded_political_quest.md)
- [\[ACL 2025\] Can LLMs Help Uncover Insights about LLMs? A Large-Scale, Evolving Literature Analysis of Frontier LLMs](can_llms_help_uncover_insights_about_llms_a_large-scale_evolving_literature_anal.md)
- [\[ACL 2025\] Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training](derta_decoupled_refusal.md)

</div>

<!-- RELATED:END -->
