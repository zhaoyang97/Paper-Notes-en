---
title: >-
  [Paper Note] How Should We Enhance the Safety of Large Reasoning Models: An Empirical Study
description: >-
  [ACL 2026][LLM Reasoning][reasoning model safety] This paper systematically investigates how to enhance the safety of large reasoning models (LRMs) via SFT. It identifies five risky reasoning patterns—most notably *weak vacillation*—as the root cause of limited effectiveness in direct safety response distillation, proposes targeted distillation strategies that reduce the PAIR attack success rate from 63% to 13%, and demonstrates that short chain-of-thought and template-based reasoning achieve safety performance comparable to full-length reasoning chains.
tags:
  - ACL 2026
  - LLM Reasoning
  - reasoning model safety
  - risky reasoning patterns
  - weak vacillation
  - safety distillation
  - short chain-of-thought
date: 2026-05-08
content_hash: c11bd6a5e04691c4
---

# How Should We Enhance the Safety of Large Reasoning Models: An Empirical Study

**Conference**: ACL 2026
**arXiv**: [2505.15404](https://arxiv.org/abs/2505.15404)
**Code**: [GitHub](https://github.com/thu-coai/LRM-Safety-Study)
**Area**: LLM Safety / Reasoning Models
**Keywords**: reasoning model safety, risky reasoning patterns, weak vacillation, safety distillation, short chain-of-thought

## TL;DR

This paper systematically investigates how to enhance the safety of large reasoning models (LRMs) via SFT. It identifies five risky reasoning patterns—most notably *weak vacillation*—as the root cause of limited effectiveness in direct safety response distillation, proposes targeted distillation strategies that reduce the PAIR attack success rate from 63% to 13%, and demonstrates that short chain-of-thought and template-based reasoning achieve safety performance comparable to full-length reasoning chains.

## Background & Motivation

**State of the Field**: Large reasoning models such as DeepSeek-R1 have achieved remarkable success on reasoning-intensive tasks like mathematics and programming. However, their enhanced reasoning capabilities do not translate into improved safety—and in some cases even lead to degradation. For instance, LRMs may generate detailed harmful plans within intermediate reasoning steps or final outputs.

**Limitations of Prior Work**: (1) Directly distilling safe responses from DeepSeek-R1 yields far weaker results than expected—fine-tuning on ReasoningShield-filtered data reduces the PAIR ASR only from 66% to 54%. (2) Existing safety filters (e.g., ReasoningShield) fail to detect all risky patterns, particularly *weak vacillation*—cases where the reasoning correctly rejects the core harmful request but hedges over superficially benign elements of a jailbreak prompt (e.g., "Is it okay to play this role?"). (3) No systematic comparative study exists on safety fine-tuning choices for LRMs.

**Root Cause**: Weak vacillation is not directly harmful in itself, making it undetectable by safety classifiers. Nevertheless, including samples exhibiting weak vacillation in training teaches the model to partially comply with harmful instructions, substantially undermining safety.

**Paper Goals**: To systematically investigate best practices for enhancing LRM safety via SFT, covering data construction, reasoning chain length, and training configuration.

**Starting Point**: The paper begins by analyzing failure cases—why do models trained on distilled "safe" data remain unsafe?—and proceeds to identify five risky reasoning patterns and eliminate them one by one.

**Core Idea**: (1) Risky reasoning patterns, especially weak vacillation, are the key reason safety distillation fails and must be actively eliminated at the distillation stage. (2) Long, complex reasoning chains are not necessary for safety—short chains or even template-based reasoning can achieve comparable safety performance.

## Method

### Overall Architecture

The study proceeds in three stages: (1) **Analysis**—identifying five risky reasoning patterns and verifying that weak vacillation is the central issue; (2) **Improved Distillation**—designing targeted prompting strategies to eliminate risky patterns (RealSafe CoT and Improved CoT); (3) **Reasoning Simplification**—verifying the safety equivalence of short chain-of-thought (Short CoT) and template-based reasoning (Template CoT).

### Key Designs

1. **Identification of Five Risky Reasoning Patterns**

    - Function: Explains the root cause of safety distillation failure.
    - Mechanism: (1) *Lack of safety awareness*—the reasoning fails to recognize the harmful nature of the query; (2) *Strong vacillation* during over-thinking—hedging over the core harmful intent; (3) *Weak vacillation* during over-thinking—hedging over superficially benign elements of a jailbreak prompt (e.g., "Is it okay to play this role?"); (4) *Harmful supplementation*—unintentionally providing harmful details within the reasoning; (5) *Reasoning–response inconsistency*—the reasoning decides to refuse, yet the final response still outputs harmful content.
    - Design Motivation: After ReasoningShield filtering, the proportion of weak vacillation actually doubles (0.33→0.66), because filtering removes more conspicuous risky patterns and leaves weak vacillation as the dominant residue. Removing weak vacillation samples reduces ASR from 45% to 21%.

2. **Improved Distillation Prompting Strategy**

    - Function: Actively eliminates risky reasoning patterns at the distillation stage.
    - Mechanism: (a) *RealSafe CoT*—modifies the distillation prompt to instruct the model to "directly identify and refuse harmful queries without being distracted by role-playing or contextual framing"; (b) *Improved CoT*—further instructs the model to "not exhibit vacillation toward any part of a jailbreak prompt." Across four models (7B–32B), the average PAIR ASR drops from 63.0% to 13.0%.
    - Design Motivation: Post-hoc filtering can only remove known risky patterns and has very low data retention (only 6.6% kept); active elimination at the distillation stage is substantially more efficient.

3. **Reasoning Simplification Experiments**

    - Function: Verifies whether long reasoning chains are necessary in safety scenarios.
    - Mechanism: Three simplified settings—(a) *Short CoT*: distills brief reasoning processes from GPT-4o; (b) *Template CoT*: uses a fixed template ("this is a harmful request → refuse"); (c) *No CoT*: refuses directly without any reasoning. Short CoT and Template CoT achieve safety performance comparable to full reasoning chains, whereas No CoT is entirely ineffective.
    - Design Motivation: The weak vacillation phenomenon suggests that long reasoning chains may introduce unnecessary deliberation—"thinking one step more" during reasoning may undermine safety decisions.

### Loss & Training

Standard SFT cross-entropy loss. Training set: 200 harmful queries × 20 jailbreak templates = 4,000 queries; after filtering, 1,000 samples are retained, supplemented with 100 XSTest benign queries to prevent over-refusal.

## Key Experimental Results

### Main Results

**DeepSeek-R1-Distill-Qwen-7B**

| Method | MATH500 | AIME24 | LiveCodeBench | None ASR↓ | PAP ASR↓ | PAIR ASR↓ | Over-refusal↓ |
|--------|---------|--------|---------------|-----------|----------|----------|--------------|
| Original | 93.0 | 49.2 | 33.1 | 60.0 | 64.0 | 66.0 | 0.8 |
| Default CoT | 90.4 | 50.0 | 36.1 | 20.0 | 40.0 | 54.0 | 2.7 |
| Improved CoT | 90.2 | 51.7 | 34.9 | 4.0 | 4.0 | **12.0** | 6.7 |
| Short CoT | 89.6 | 53.3 | 36.1 | 4.0 | 2.0 | 16.0 | 12.0 |
| Template CoT | 89.4 | 49.2 | 33.7 | 12.0 | 0.0 | **0.0** | 10.7 |
| No CoT | 90.8 | 52.5 | 31.9 | 62.0 | 64.0 | 64.0 | 10.0 |

### Ablation Study

| Risky Patterns Removed | PAIR ASR |
|------------------------|----------|
| ReasoningShield filtering (includes weak vacillation) | 45.0% |
| Remove all risky patterns **except** weak vacillation | 45.0% (no improvement) |
| Remove **all** risky patterns (including weak vacillation) | **21.0%** |

### Key Findings

- **Weak vacillation is the central cause of safety distillation failure**—removing all other risky patterns yields no improvement (ASR unchanged), while removing weak vacillation alone reduces ASR by 24 percentage points.
- **No CoT is entirely ineffective**—some form of reasoning is necessary for safety, but it need not be lengthy.
- **Template CoT achieves 0% PAIR ASR**—the simplest reasoning mode proves the safest, as template-based reasoning leaves no room for the model to vacillate.
- **Safety fine-tuning does not impair reasoning capability**—MATH500, AIME24, and LiveCodeBench scores remain essentially unchanged.
- Short CoT exhibits a higher over-refusal rate (12%), necessitating a trade-off between safety and helpfulness.

## Highlights & Insights

- The concept of *weak vacillation* is a significant contribution—appearing harmless on the surface while conveying a signal that partial compliance is acceptable, it represents a subtle yet consequential safety vulnerability.
- The finding that "longer reasoning does not necessarily mean greater safety" is counterintuitive—in the safety domain, more deliberation may introduce more hesitation.
- The inability of safety filters to detect weak vacillation exposes a critical blind spot in existing safety evaluation tooling.

## Limitations & Future Work

- The study focuses on SFT and does not examine the effect of RLHF/DPO or other alignment methods on LRM safety.
- The training set of 1,000 samples is relatively small; effectiveness at larger scales may differ.
- Although Template CoT achieves the lowest ASR, its over-refusal rate is elevated, requiring more refined balancing.
- Evaluation is conducted exclusively in English; safety behavior of multilingual LRMs may differ.

## Related Work & Insights

- **vs. Zhang et al. (2025a)**: Proposes safety-aware response distillation but does not systematically analyze failure causes; this paper identifies five specific risky patterns.
- **vs. Jiang et al. (2025)**: Finds that safety tuning degrades reasoning performance; this paper demonstrates that substantial safety improvements are achievable without sacrificing reasoning ability.
- **vs. Wang et al. (2025)**: Proposes low-cost mitigation strategies; this paper provides a more comprehensive ablation study.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The concept of weak vacillation and the finding that shorter reasoning yields greater safety are highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 models × 6 methods × multi-dimensional evaluation (reasoning, safety, over-refusal) with detailed ablation.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from problem analysis to solution is exceptionally clear.
- Value: ⭐⭐⭐⭐⭐ Provides a systematic practical guide for safety alignment of LRMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure](../../ICLR2026/llm_reasoning/dynamics_within_latent_chain-of-thought_an_empirical_study_of_causal_structure.md)
- [\[ICLR 2026\] Reasoning or Retrieval? A Study of Answer Attribution on Large Reasoning Models](../../ICLR2026/llm_reasoning/reasoning_or_retrieval_a_study_of_answer_attribution_on_large_reasoning_models.md)
- [\[AAAI 2026\] Trade-offs in Large Reasoning Models: An Empirical Analysis of Deliberative and Adaptive Reasoning over Foundational Capabilities](../../AAAI2026/llm_reasoning/trade-offs_in_large_reasoning_models_an_empirical_analysis_of_deliberative_and_a.md)
- [\[ACL 2026\] CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization](cipo_counterfactual_unlearning_for_large_reasoning_models_through_iterative_pref.md)
- [\[ACL 2026\] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](revisiting_entropy_in_reinforcement_learning_for_large_reasoning_models.md)

</div>

<!-- RELATED:END -->
