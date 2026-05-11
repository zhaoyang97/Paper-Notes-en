---
title: >-
  [Paper Note] Dissecting Failure Dynamics in Large Language Model Reasoning
description: >-
  [ACL 2026][LLM Reasoning][Reasoning Failure Analysis] By analyzing LLM reasoning trajectories, this work finds that errors concentrate at a small number of critical turning points in the early stages…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Reasoning Failure Analysis"
  - "Entropy Signal"
  - "Early Failure Onset"
  - "Cognitive Spiral"
  - "Inference-Time Intervention"
date: 2026-05-08
content_hash: 1f6d9eedd4fe6803
---

# Dissecting Failure Dynamics in Large Language Model Reasoning

**Conference**: ACL 2026
**arXiv**: [2604.14528](https://arxiv.org/abs/2604.14528)
**Code**: [GitHub](https://github.com/ZHUWEI-hub/GUARD)
**Area**: LLM Reasoning / Inference-Time Computation
**Keywords**: Reasoning Failure Analysis, Entropy Signal, Early Failure Onset, Cognitive Spiral, Inference-Time Intervention

## TL;DR
By analyzing LLM reasoning trajectories, this work finds that errors concentrate at a small number of critical turning points in the early stages, after which the model enters a "cognitive spiral"—continuously extending the reasoning in a locally coherent but globally erroneous manner. Based on these findings, the paper proposes the GUARD framework, which performs short-range branching repairs at high-risk turning points detected via entropy signals.

## Background & Motivation

**State of the Field**: Large reasoning models (LRMs) such as DeepSeek-R1 and OpenAI o1 improve performance by extending reasoning chains. Existing inference-time scaling strategies focus primarily on "providing more computation"—generating longer chains, sampling multiple trajectories in parallel, or conducting MCTS search.

**Limitations of Prior Work**: Existing methods perform "blind scaling"—they are agnostic to when and where errors occur in a trajectory, allocating computation uniformly across all positions. Multi-path methods (e.g., Best-of-N) require maintaining multiple full parallel trajectories, resulting in severe computational redundancy.

**Root Cause**: The benefit of inference-time scaling depends on whether errors are recoverable, yet existing methods do not distinguish between "early deviations that are still correctable" and "late deviations that are already irreversible"—leading to wasted computation on ineffective late-stage extensions.

**Paper Goals**: To understand the temporal dynamics of reasoning failures within trajectories, and to design targeted intervention mechanisms accordingly.

**Starting Point**: Segment-level analysis of erroneous trajectories reveals four key empirical regularities that guide intervention design.

**Core Idea**: Errors concentrate in early stages + error segments exhibit entropy spikes + some errors are recoverable from the same prefix → perform short-range branching at entropy spikes and truncate hesitation behavior in late stages.

## Method

### Overall Architecture
GUARD maintains a single main reasoning trajectory and monitors token-level entropy in real time. Upon detecting abnormally high entropy at reasoning step boundaries, it triggers short-range branching: three brief alternative continuations are generated (momentum, suppression, and counterfactual branches), and the one with the lowest average entropy is selected to continue. In later stages, when hesitation markers are detected, the reasoning is truncated to prevent ineffective extension.

### Key Designs

1. **Four Findings on Reasoning Failure Dynamics**:

    - Function: Provide an empirical basis for the intervention strategy.
    - Mechanism: (1) *Early failure onset*: over 85% of failure origins appear within the first 30% of the trajectory, and 43.5% of erroneous trajectories contain only a single error segment; (2) *Cognitive spiral*: trajectories after the error point are significantly longer yet remain locally coherent, forming "seemingly plausible but globally erroneous" extended reasoning; (3) *Entropy signal*: token-level entropy exhibits a local spike at failure origins, and the overall entropy of error segments is significantly higher than that of correct segments ($p<0.001$); (4) *Local recoverability*: over 20% of failed trajectories can reach the correct answer via alternative continuations from the same prefix.
    - Design Motivation: These four findings jointly indicate that errors are local, detectable, and partially recoverable → intervening only at critical positions is more efficient than global scaling.

2. **Instance-Adaptive Threshold-Based Failure Detection**:

    - Function: Detect high-risk turning points at reasoning step boundaries.
    - Mechanism: At delimiter positions, the current token entropy is checked against the $q$-th quantile of historical entropy: $\mathbb{I}_{drift}(x_t) = \mathbb{I}[x_{t-1} \in \mathcal{T}_{delim} \land \mathcal{H}(x_t) > \text{Quantile}_q(\mathbf{H}_{<t})]$. Using a quantile rather than an absolute threshold makes detection adaptive to the entropy scale of the current problem.
    - Design Motivation: Absolute thresholds are not robust across problems of varying difficulty—"high entropy" for a simple problem may be "normal entropy" for a hard one; the quantile-based approach eliminates this scale discrepancy.

3. **Short-Range Semantic Branching and Late-Stage Truncation**:

    - Function: Explore local alternatives at detected risk points without maintaining full parallel trajectories.
    - Mechanism: Upon triggering, three short-range continuations are generated—a momentum branch (standard greedy decoding), a suppression branch (prepending "Wait," to interrupt the continuation pattern), and a counterfactual branch (prepending "Let me reconsider:" to encourage re-evaluation). The continuation with the lowest average entropy is selected to proceed along a single trajectory. In later stages, when the remaining capacity satisfies $\rho_t \leq \rho_{min}$, hesitation markers are directly replaced with termination signals.
    - Design Motivation: Motivated by the recoverability finding—full alternative paths need not be explored; providing a few local alternatives at the deviation point and selecting the most confident one is sufficient.

### Loss & Training
GUARD is a purely inference-time framework and involves no training. All branches share precomputed KV caches to minimize latency overhead.

## Key Experimental Results

### Main Results

| Method | AIME24 | AIME25 | AMC23 | MATH500 | Avg. Pass@1 |
|--------|--------|--------|-------|---------|-------------|
| BASE | 20.0 | 13.3 | 57.0 | 78.9 | 36.2 |
| Reflexion | 30.0 | 23.3 | 72.5 | 80.2 | — |
| α1 | 20.0 | 26.7 | 70.0 | 80.4 | 41.2 |
| GUARD | — | — | — | — | Significant improvement |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| No branching (detection only) | Limited performance | Detection alone is insufficient; repair is necessary |
| No late-stage truncation | Increased token waste | Late-stage extensions constitute ineffective computation |
| Fixed absolute threshold | Unstable | Adaptive threshold is more robust |

### Key Findings
- Erroneous trajectories contain significantly more segments than correct ones—the extra segments are almost entirely ineffective extensions following the failure origin.
- Entropy signal is a reliable failure indicator—the average entropy of failure segments is significantly higher than that of correct segments.
- Short-range branching (3 branches × short range) is substantially more token-efficient than maintaining multiple full parallel trajectories.
- GUARD yields especially pronounced gains on smaller models (1.5B), as smaller models are more prone to cognitive spirals.

## Highlights & Insights
- The **"cognitive spiral" concept** precisely characterizes the core pathology of LLM reasoning failures—after an error, the model does not immediately collapse but instead "plausibly deepens into the mistake," which explains why longer reasoning chains are not necessarily better.
- The approach of **performing surgery at the deviation point rather than treating the whole body** is highly efficient—concentrating computation on the 20% of recoverable failures.
- The analytical findings can inform reasoning RL training—if 85% of failures originate in the first 30% of the trajectory, training signals should likewise be concentrated at these early turning points.

## Limitations & Future Work
- Gemini 3 Pro is used as an external oracle to judge segment validity, introducing potential evaluation bias.
- Validation is limited to mathematical and competition reasoning; failure dynamics in natural language reasoning and code generation may differ.
- The three-branch design (momentum / suppression / counterfactual) is relatively hand-crafted; more principled branching strategies warrant further exploration.
- Late-stage truncation may incorrectly terminate correct trajectories that arrive at the right answer after extended deliberation.

## Related Work & Insights
- **vs. Best-of-N**: BoN generates $N$ complete parallel trajectories; GUARD performs short-range exploration only at a small number of risk points along a single trajectory.
- **vs. DTS**: DTS triggers branching based on absolute entropy thresholds; GUARD employs an adaptive threshold derived from the historical entropy quantile.
- **vs. α1**: α1 dynamically adjusts reasoning depth via information-theoretic metrics but does not perform local repair.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The systematic analysis of reasoning failure dynamics offers a fundamentally new perspective; the cognitive spiral concept provides deep insight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple competition reasoning benchmarks with detailed statistical analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from analysis to method is exceptionally coherent, with outstanding visualizations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TROJail: Trajectory-Level Optimization for Multi-Turn Large Language Model Jailbreaks with Process Rewards](trojail_trajectory-level_optimization_for_multi-turn_large_language_model_jailbr.md)
- [\[AAAI 2026\] Incorporating Self-Rewriting into Large Language Model Reasoning Reinforcement](../../AAAI2026/llm_reasoning/incorporating_self-rewriting_into_large_language_model_reasoning_reinforcement.md)
- [\[ACL 2026\] Failure Modes in Multi-Hop QA: The Weakest Link Effect and the Recognition Bottleneck](failure_modes_in_multi-hop_qa_the_weakest_link_effect_and_the_recognition_bottle.md)
- [\[ICLR 2026\] Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models](../../ICLR2026/llm_reasoning/dynamics-predictive_sampling_for_active_rl_finetuning_of_large_reasoning_models.md)
- [\[ICLR 2026\] Why is Your Language Model a Poor Implicit Reward Model?](../../ICLR2026/llm_reasoning/why_is_your_language_model_a_poor_implicit_reward_model.md)

</div>

<!-- RELATED:END -->
