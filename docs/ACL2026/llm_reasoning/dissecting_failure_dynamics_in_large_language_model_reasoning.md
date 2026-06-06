---
title: >-
  [Paper Note] Dissecting Failure Dynamics in Large Language Model Reasoning
description: >-
  [ACL 2026][LLM Reasoning][Reasoning failure analysis] By analyzing LLM reasoning trajectories, it is discovered that errors are concentrated at a few early key turning points…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Reasoning failure analysis"
  - "Entropy signal"
  - "Early onset of failures"
  - "Cognitive spiral"
  - "Test-time intervention"
date: 2026-05-08
content_hash: ec9dad54e289c3a8
---

# Dissecting Failure Dynamics in Large Language Model Reasoning

**Conference**: ACL 2026  
**arXiv**: [2604.14528](https://arxiv.org/abs/2604.14528)  
**Code**: [GitHub](https://github.com/ZHUWEI-hub/GUARD)  
**Area**: LLM Reasoning / Test-time Compute  
**Keywords**: Reasoning failure analysis, Entropy signal, Early onset of failures, Cognitive spiral, Test-time intervention

## TL;DR
By analyzing LLM reasoning trajectories, it is discovered that errors are concentrated at a few early key turning points, after which the model enters a "cognitive spiral"—becoming locally coherent but globally incorrect as it continues to extend. Based on this, the GUARD framework is proposed to perform short-range branch repairs at high-risk turning points detected by entropy signals.

## Background & Motivation

**Background**: Large Reasoning Models (LRMs) such as DeepSeek-R1 and OpenAI o1 improve performance by lengthening reasoning chains. Existing test-time scaling strategies primarily focus on "providing more computation"—generating longer chains, parallel sampling of multiple trajectories, or MCTS search.

**Limitations of Prior Work**: Existing methods involve "blind expansion"—they ignore when and where errors occur in the trajectory and allocate computation equally to all positions. Multi-path methods (such as Best-of-N) require maintaining multiple full parallel trajectories, resulting in severe computational redundancy.

**Key Challenge**: The gains from test-time scaling depend on "whether errors are repairable," but existing methods do not distinguish between "repairable early deviations" and "irreversible late deviations," leading to computational waste on ineffective late extensions.

**Goal**: To understand the temporal dynamics of reasoning failures within trajectories and design targeted intervention mechanisms accordingly.

**Key Insight**: A segment-by-segment analysis of error trajectories revealed four key patterns that provide guidance for interventions.

**Core Idea**: Errors are concentrated in the early stage + error segments exhibit entropy spikes + partial errors are recoverable from the same prefix → conduct short-range branching at entropy spikes and truncate hesitant behavior in the later stages.

## Method

### Overall Architecture
GUARD maintains a single main reasoning trajectory and monitors token-level entropy in real-time. When abnormally high entropy is detected at reasoning step boundaries, a short-range branching is triggered: three short alternative continuations (momentum, suppression, and counterfactual) are generated, and the one with the lowest average entropy is selected to continue. In the later stages, reasoning is truncated upon detecting hesitation markers to prevent ineffective extension.

### Key Designs

1.  **Four Findings on Reasoning Failure Dynamics**:

    - **Function**: To provide an empirical basis for intervention strategies.
    - **Mechanism**: (1) Early onset of failure: Over 85% of failure starts occur within the first 30% of the trajectory, and 43.5% of error trajectories contain only a single error segment; (2) Cognitive spiral: Post-error trajectories significantly lengthen while remaining locally coherent, forming "seemingly plausible but globally incorrect" extended reasoning; (3) Entropy signals: Token-level entropy exhibits local spikes at the start of failures, and the overall entropy of error segments is significantly higher than that of correct segments ($p<0.001$); (4) Local recoverability: Over 20% of failure trajectories can reach the correct answer through alternative continuations from the same prefix.
    - **Design Motivation**: These four findings collectively demonstrate that errors are local, detectable, and partially repairable—intervening only at key positions is more efficient than global expansion.

2.  **Failure Detection via Instance-Adaptive Thresholds**:

    - **Function**: To detect high-risk turning points at reasoning step boundaries.
    - **Mechanism**: At delimiters, it is checked whether the current token entropy exceeds the $q$-quantile of historical entropy: $\mathbb{I}_{drift}(x_t) = \mathbb{I}[x_{t-1} \in \mathcal{T}_{delim} \land \mathcal{H}(x_t) > \text{Quantile}_q(\mathbf{H}_{<t})]$. Using quantiles instead of absolute thresholds makes detection adaptive to the entropy scale of the current problem.
    - **Design Motivation**: Absolute thresholds are not robust across different problems—"high entropy" for a simple problem might be "normal entropy" for a difficult one; the quantile method eliminates this scale difference.

3.  **Short-range Semantic Branching and Late Truncation**:

    - **Function**: To explore local alternatives at detected risk points rather than maintaining full parallel paths.
    - **Mechanism**: Upon triggering, three short-range continuations are generated—Momentum branch (standard greedy), Suppression branch (preceded by "Wait," to break the sequence pattern), and Counterfactual branch (preceded by "Let me reconsider:" to encourage rethinking). The continuation with the lowest average entropy is selected to continue the single trajectory. Later, when the remaining capacity $\rho_t \leq \rho_{min}$, hesitation markers are directly replaced with a termination signal.
    - **Design Motivation**: Inspired by the recoverability finding—there is no need to explore full alternative paths; it is sufficient to provide a few local alternatives at the deviation point and select the most certain one.

### Loss & Training
GUARD is a pure test-time framework and does not involve training. All branches share pre-computed KV caches to minimize latency overhead.

## Key Experimental Results

### Main Results

| Method | AIME24 | AIME25 | AMC23 | MATH500 | Mean Pass@1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BASE | 20.0 | 13.3 | 57.0 | 78.9 | 36.2 |
| Reflexion | 30.0 | 23.3 | 72.5 | 80.2 | - |
| α1 | 20.0 | 26.7 | 70.0 | 80.4 | 41.2 |
| GUARD | - | - | - | - | Significant Gain |

### Ablation Study

| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| Without Branching (Detection Only) | Limited Performance | Detection alone is insufficient; repair is required |
| Without Late Truncation | Increased Token Waste | Late extension constitutes ineffective computation |
| Fixed Absolute Threshold | Unstable | Adaptive thresholds are more robust |

### Key Findings
- The number of segments in error trajectories is significantly higher than in correct trajectories—extra segments are almost entirely ineffective extensions following the failure start.
- Entropy signals are reliable indicators of failure—the average entropy of failure segments is significantly higher than that of correct segments.
- Short-range branching (3 branches × short distance) is much more token-efficient than maintaining multiple full parallel paths.
- GUARD provides especially significant gains for small models (1.5B), as they are more prone to falling into cognitive spirals.

## Highlights & Insights
- The **"Cognitive Spiral"** concept precisely describes the core pathology of LLM reasoning failures—falling deeper into seemingly plausible traps rather than collapsing immediately, which explains why longer reasoning chains are not necessarily better.
- The idea of **"performing surgery at the deviation point rather than systemic treatment"** is highly efficient—concentrating computation on the 20% of repairable failures.
- The analysis findings can guide reasoning RL training—if 85% of failures originate in the first 30% of the trajectory, training signals should also be concentrated on these early turning points.

## Limitations & Future Work
- Using Gemini 3 Pro as an external oracle to judge segment validity involves evaluation bias.
- Validated only on math/competition reasoning; failure dynamics in natural language reasoning and code generation may differ.
- The design of the three branches (momentum/suppression/counterfactual) is somewhat heuristic; better branching strategies are worth exploring.
- Late truncation might prune correct trajectories that "eventually find the answer after long thinking."

## Related Work & Insights
- **vs Best-of-N**: BoN generates $N$ full parallel paths, whereas GUARD only performs short-range exploration at a few risk points on a single path.
- **vs DTS**: DTS triggers branching based on absolute entropy, while GUARD uses adaptive thresholds based on historical quantiles.
- **vs α1**: α1 dynamically adjusts depth through information-theoretic metrics but does not perform local repairs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The systematic analysis of reasoning failure dynamics is a fresh perspective; the cognitive spiral concept offers deep insight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple competition reasoning benchmarks and detailed statistical analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from analysis to method is extremely smooth, with excellent visualization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Language Model as Planner and Formalizer under Constraints](language_model_as_planner_and_formalizer_under_constraints.md)
- [\[AAAI 2026\] Incorporating Self-Rewriting into Large Language Model Reasoning Reinforcement](../../AAAI2026/llm_reasoning/incorporating_self-rewriting_into_large_language_model_reasoning_reinforcement.md)
- [\[ACL 2026\] SeLaR: Selective Latent Reasoning in Large Language Models](selar_selective_latent_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Failure Modes in Multi-Hop QA: The Weakest Link Effect and the Recognition Bottleneck](failure_modes_in_multi-hop_qa_the_weakest_link_effect_and_the_recognition_bottle.md)
- [\[ACL 2026\] TInR: Exploring Tool-Internalized Reasoning in Large Language Models](tinr_exploring_tool-internalized_reasoning_in_large_language_models.md)

</div>

<!-- RELATED:END -->
