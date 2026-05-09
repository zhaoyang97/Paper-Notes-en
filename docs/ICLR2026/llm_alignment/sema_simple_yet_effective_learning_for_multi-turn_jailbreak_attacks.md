---
title: >-
  [Paper Note] SEMA: Simple yet Effective Learning for Multi-Turn Jailbreak Attacks
description: >-
  [ICLR 2026][LLM Alignment][multi-turn jailbreak attack] This paper proposes SEMA, a two-stage training framework consisting of prefilling self-tuning and RL with an intent-drift-aware reward. Without relying on any existing attack strategies or external data, SEMA trains an attacker capable of automatically generating multi-turn jailbreak attacks, achieving an average ASR@1 of 80.1% across three victim models on AdvBench — surpassing the prior state of the art by 33.9%.
tags:
  - ICLR 2026
  - LLM Alignment
  - multi-turn jailbreak attack
  - reinforcement learning red-teaming
  - intent drift
  - open-loop attack
  - LLM safety
date: 2026-05-08
content_hash: 16a28736d2c36963
---

# SEMA: Simple yet Effective Learning for Multi-Turn Jailbreak Attacks

**Conference**: ICLR 2026
**arXiv**: [2602.06854](https://arxiv.org/abs/2602.06854)
**Code**: [https://github.com/fmmarkmq/SEMA](https://github.com/fmmarkmq/SEMA)
**Area**: Alignment & RLHF
**Keywords**: multi-turn jailbreak attack, reinforcement learning red-teaming, intent drift, open-loop attack, LLM safety

## TL;DR
This paper proposes SEMA, a two-stage training framework consisting of prefilling self-tuning and RL with an intent-drift-aware reward. Without relying on any existing attack strategies or external data, SEMA trains an attacker capable of automatically generating multi-turn jailbreak attacks, achieving an average ASR@1 of 80.1% across three victim models on AdvBench — surpassing the prior state of the art by 33.9%.

## Background & Motivation
Multi-turn jailbreaks represent a more realistic threat model than single-turn attacks, as real-world interactions between users and chatbots are ongoing dialogues in which an attacker can progressively guide a model into lowering its defenses. Existing multi-turn attack methods, however, face serious challenges:

**Challenge 1: Exponential Exploration Complexity.** In multi-turn settings, the action space grows exponentially with the number of turns, making it difficult for RL agents to efficiently explore effective attack trajectories.

**Challenge 2: Intent Drift.** In multi-turn dialogues, an attacker may gradually deviate from the original harmful objective — benign topics introduced in early turns to "set the stage" can prevent the conversation from ever returning to the harmful intent.

**Limitations of Prior Work:** Manually designed multi-turn attack strategies (e.g., Crescendo, PAIR) rely on fixed templates and lack adaptability. RL-based methods require closed-loop interaction with the victim model, resulting in high training costs and sensitivity to unstable feedback.

**Core Idea:** SEMA adopts an open-loop attack paradigm — generating a complete multi-turn attack sequence without requiring intermediate feedback from the victim model. This unifies single-turn and multi-turn settings, substantially reduces exploration complexity, and is complemented by an intent-drift-aware reward that anchors the attacker to the harmful objective throughout the dialogue.

## Method
SEMA trains a multi-turn attacker from scratch in two stages, without relying on any existing attack strategies or external data.

### Overall Architecture
Stage 1: Prefilling Self-Tuning → Stage 2: RL with Intent-Drift-Aware Reward. Stage 1 provides a stable initialization for Stage 2, which further optimizes the attack policy via RL.

### Key Designs
1. **Prefilling Self-Tuning**: The first stage addresses the cold-start problem — an untrained LLM cannot generate multi-turn attack sequences, making direct RL training infeasible due to the poor quality of initial rollouts. The solution provides the model with a minimal prefix (e.g., the beginning of a harmful target) and lets it self-generate non-refusal, well-structured multi-turn adversarial prompts. These self-generated samples are then used for SFT, enabling the model to learn the correct format for multi-turn attacks. Crucially, no external templates or human-annotated data are required.

2. **Open-Loop Attack Paradigm**: SEMA generates a complete multi-turn attack sequence and submits it to the victim model in one pass, rather than waiting for feedback after each turn before generating the next. This eliminates any dependency on victim model feedback, unifies single-turn and multi-turn settings, and allows the attacker to transfer to different victim models without retraining.

3. **Intent-Drift-Aware Reward**: The core innovation of the RL stage. The reward function combines three dimensions: (a) **intent alignment** — whether the multi-turn attack still centers on the original harmful objective at the end; (b) **compliance risk** — whether the victim model produces a harmful response; (c) **level of detail** — the specificity of the harmful response. By anchoring intent alignment, this reward addresses the fundamental problem of goal drift in multi-turn attacks.

### Loss & Training
Stage 1 uses standard SFT loss. Stage 2 applies the GRPO (Group Relative Policy Optimization) algorithm with the intent-drift-aware reward. The base model is Llama-3.1-8B-Instruct. The intent-drift-aware reward is implemented as a weighted combination of the three dimensions described above, each computed independently and then aggregated into a final scalar reward. Training uses 4–8×H100 GPUs; a key hyperparameter in Stage 1 is the length of the minimal prefix. The victim model used during Stage 2 training may differ from the attacker model, which encourages transferability as a training objective.

## Key Experimental Results

### Main Results

| Victim Model | Dataset | SEMA ASR@1 | Prev. SOTA | Gain |
|---|---|---|---|---|
| Average across 3 models | AdvBench | 80.1% | 46.2% | +33.9% |
| Closed-source models | AdvBench | High | Low | Significant |
| Open-source models | AdvBench | High | Moderate | Significant |

### Ablation Study

| Configuration | ASR | Notes |
|---|---|---|
| SEMA (full) | Highest | Two-stage + intent-drift reward |
| SFT-only | Low | Stage 1 only, no RL |
| DPO variant | Moderate | Preference optimization underperforms RL |
| Without intent-drift reward | Degraded | Attacks frequently deviate from target |
| Single-turn setting | Effective but below multi-turn | Validates unified formulation |

### Key Findings
- SEMA outperforms all single-turn baselines, manually scripted multi-turn baselines, and template-driven multi-turn baselines.
- SEMA surpasses both SFT and DPO variants, demonstrating the advantage of RL for multi-turn attack learning.
- Open-loop attacks transfer directly to unseen victim models, including closed-source ones.
- Prefilling self-tuning is a critical prerequisite for RL convergence — without it, Stage 2 fails to converge.
- The method is compact and reproducible; code will be released prior to the ICLR conference.

## Highlights & Insights
- The paper analyzes the realistic threat model of multi-turn jailbreaks and establishes that single-turn attacks are merely a special case, thereby redefining the standard for LLM safety evaluation.
- The open-loop attack paradigm is an elegant design choice — it avoids the complexity of closed-loop interaction while preserving transferability.
- The intent-drift-aware reward is a substantive contribution to multi-turn safety research, as it precisely defines what constitutes a successful multi-turn attack.
- Bootstrapping attack capability from scratch represents an important advance in automated red-teaming.
- Prefilling self-tuning addresses the general cold-start problem in RL for sequential generation and is potentially applicable beyond jailbreaking.
- A 33.9% absolute improvement in ASR@1 over the prior state of the art reveals that existing LLMs are severely underprotected against multi-turn attacks.
- The method's compactness — modest code size and training cost — makes it well-suited as a standardized stress-testing tool for LLM safety.

## Limitations & Future Work
- The code remains under Microsoft Research review and has not yet been fully released.
- Open-loop attacks may underperform closed-loop attacks against specific victim models, as they cannot leverage intermediate feedback to adjust strategy.
- Defenders may be able to detect multi-turn attack patterns through dialogue-level monitoring.
- Ethical considerations apply — the method could be misused, though the authors position it as a stress-testing tool for exposing LLM security vulnerabilities.
- The effectiveness against state-of-the-art defense mechanisms (e.g., Llama Guard 3 and other dedicated safety detectors) has not yet been evaluated.
- Training data containing harmful content requires strict access controls and usage protocols.

## Related Work & Insights
- **vs. Crescendo/PAIR**: These methods rely on manually designed templates, whereas SEMA learns attack strategies entirely from data.
- **vs. AutoDAN-Turbo**: AutoDAN-Turbo depends on an existing library of attack strategies; SEMA is fully self-bootstrapped.
- **vs. Single-turn attacks (GCG/AutoDAN)**: SEMA unifies single-turn and multi-turn settings and better reflects realistic threat scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of open-loop multi-turn attacks and intent-drift-aware reward is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ 37 pages, 13 tables, 7 figures, evaluated across multiple datasets and victim models.
- Writing Quality: ⭐⭐⭐⭐ Well-structured; the design motivation for the two-stage pipeline is clearly articulated.
- Value: ⭐⭐⭐⭐ Directly advances LLM safety red-teaming research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks](jailnewsbench_multi-lingual_and_regional_benchmark_for_fake_news_generation_unde.md)
- [\[ICLR 2026\] Obscure but Effective: Classical Chinese Jailbreak Prompt Optimization via Bio-Inspired Search](obscure_but_effective_classical_chinese_jailbreak_prompt_optimization_via_bio-in.md)
- [\[ICLR 2026\] Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)](toward_universal_and_transferable_jailbreak_attacks_on_vision-language_models.md)
- [\[AAAI 2026\] AlignTree: Efficient Defense Against LLM Jailbreak Attacks](../../AAAI2026/llm_alignment/aligntree_efficient_defense_against_llm_jailbreak_attacks.md)
- [\[ICLR 2026\] Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training](chasing_the_tail_effective_rubric-based_reward_modeling_for_large_language_model.md)

</div>

<!-- RELATED:END -->
