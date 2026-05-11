---
title: >-
  [Paper Note] Speculative Actions: A Lossless Framework for Faster AI Agents
description: >-
  [ICLR 2026 Oral][speculative execution] Inspired by CPU speculative execution and LLM speculative decoding, this paper proposes the Speculative Actions framework: while a slow Actor (large model) computes…
tags:
  - "ICLR 2026 Oral"
  - "speculative execution"
  - "AI agents"
  - "latency reduction"
  - "lossless acceleration"
  - "MDP"
date: 2026-05-08
content_hash: 838027f00bc9d97b
---

# Speculative Actions: A Lossless Framework for Faster AI Agents

**Conference**: ICLR 2026 Oral  
**OpenReview**: [P0GOk5wslg](https://openreview.net/forum?id=P0GOk5wslg)  
**Code**: None  
**Area**: Other  
**Keywords**: speculative execution, AI agents, latency reduction, lossless acceleration, MDP  

## TL;DR
Inspired by CPU speculative execution and LLM speculative decoding, this paper proposes the Speculative Actions framework: while a slow Actor (large model) computes, a fast Speculator (small model) predicts future actions and pre-executes them; upon a match, the waiting round is skipped, achieving lossless acceleration. The framework achieves 15–30% latency reduction across Chess, e-commerce, and QA scenarios. A confidence-based dynamic branching strategy attains acceleration comparable to three speculative branches while using 40% fewer tokens.

## Background & Motivation

**Background**: AI agents interacting with environments follow a strict sequential pattern: the agent generates an action → the environment responds → the agent generates the next action. When large models (e.g., GPT-5, Gemini-2.5-Pro) serve as agents, the latency of each API call becomes a bottleneck.

**Limitations of Prior Work**: (a) Speculative decoding only accelerates token generation and does not address agent–environment interaction latency; (b) existing agent acceleration methods mostly sacrifice accuracy (e.g., replacing large models with small ones); (c) no theoretical framework exists to analyze the cost–latency trade-off of parallel speculation in agents.

**Key Challenge**: Large-model agents achieve high accuracy but are slow; small models are fast but insufficiently accurate. Can both be achieved simultaneously—preserving large-model output quality while approaching small-model speed?

**Goal**: Design a lossless acceleration framework that exploits the speed gap between large and small models to speculatively execute actions in parallel, reducing end-to-end latency while fully preserving the output quality of the large model.

**Key Insight**: The key insight from CPU speculative execution—"predict then verify" does not affect correctness, only efficiency. Analogously, in agent interactions, predicted actions are pre-executed; matches are reused and mismatches are discarded, yielding results identical to purely sequential execution.

**Core Idea**: A fast small model predicts agent actions and pre-executes environment steps; when predictions are correct, one round of waiting is skipped, and the resulting trajectory is guaranteed to be identical to sequential execution.

## Method

### Overall Architecture
The Actor (large model) and Speculator (small model) run in parallel. The Speculator rapidly predicts $k$ possible next actions and pre-executes them, while the Actor computes the true action. If the true action matches a predicted action, the pre-executed result is reused directly, skipping one round of environment interaction latency; otherwise, the prediction is discarded and normal execution continues.

### Key Designs

1. **Breadth Speculation**

    - Function: Launch $k$ speculative branches simultaneously from the current state $s_t$.
    - Mechanism: The Speculator predicts $k$ actions $\{\hat{a}_t^{(i)}\}_{i=1}^k$ in parallel; for each predicted action, the next state is pre-computed and an Actor call is pre-initiated. The match probability is $p(k) = 1 - (1-p)^k$; larger $k$ increases match probability but raises token cost.
    - Design Motivation: Breadth speculation is straightforward, and the $k$ branches are mutually independent and fully parallelizable.

2. **Depth Speculation**

    - Function: Continue speculating subsequent steps after a successful match (multi-step speculation chains).
    - Mechanism: A successfully matched speculation chain can extend to the next step, forming a speculation tree. Theoretical analysis shows that the computational overhead of depth speculation is bounded by the speed ratio $a/b$ and does not grow exponentially with horizon $T$.
    - Design Motivation: When the single-step match rate is high, depth speculation can compound the acceleration effect.

3. **Confidence-Based Dynamic Branching**

    - Function: Dynamically decide whether to speculate based on the Speculator's confidence.
    - Mechanism: $\text{Accept speculation at step } t \iff p_t \geq p^\star$, where the threshold $p^\star$ is derived from the cost ratio. This strategy is proven to be theoretically optimal.
    - Design Motivation: Avoids wasting tokens when confidence is low; achieves the acceleration of $k=3$ speculation using 40% fewer tokens.

4. **Lossless Guarantee**

    - Function: Guarantee that the final execution trajectory is identical to purely sequential execution.
    - Mechanism: The Actor reuses cached results only when they exactly match the true action; otherwise, predictions are discarded. The output sequence is identical to sequential execution.
    - Design Motivation: This guarantee is a prerequisite for real-world deployment—users need not worry about acceleration introducing errors.

### Theoretical Results
**Latency Savings:** $\frac{E[T_{\text{seq}} - T_{\text{spec}}]}{E[T_{\text{seq}}]} \to \frac{p(k)}{1+p(k)} \cdot \frac{b}{a+b}$

**Cost Increase:** $\frac{E[M_{\text{spec}} - M_{\text{seq}}]}{E[M_{\text{seq}}]} \to \frac{k}{1+p(k)} - \frac{b}{a+b} \cdot \frac{p(k)}{1+p(k)}$

where Actor and Speculator latencies follow $\text{Exp}(\beta)$ and $\text{Exp}(\alpha)$, respectively.

## Key Experimental Results

### Main Results

| Task | Speculation Count $k$ | Latency Savings | Extra Tokens |
|------|----------------------|----------------|-------------|
| Chess | $k=1$ | 4–8% | ~91% |
| Chess | $k=2$ | 11–18% | ~155% |
| Chess | $k=3$ | 19–31% | ~180% |
| Chess | Confidence Dynamic | **16–25%** | **~88%** |

### Ablation Study

| Analysis Dimension | Key Findings |
|-------------------|-------------|
| Next-step prediction accuracy | Reaches 55% across domains |
| Confidence dynamic vs. fixed $k$ | Achieves $k=3$ acceleration at $k=1$ token cost |
| Lossy mode (OS Tuning) | 93.5% latency reduction, 92% cost reduction |
| Speculator selection | Same-family small models (e.g., GPT-5-nano for GPT-5) perform best |

### Key Findings
- **Cross-domain generality**: Effective across four highly diverse domains—Chess, e-commerce, QA, and OS tuning.
- **Confidence threshold is the core optimization**: Dynamic branching achieves the best trade-off between token efficiency and latency reduction.
- **Self-hosted deployment offers nearly free speculation**: Using idle GPUs for speculation incurs almost no additional cost.
- **Lossy extension has large potential**: When lossiness is permitted (OS Tuning), both latency and cost are substantially reduced simultaneously.

## Highlights & Insights
- **A perfect analogy from CPU speculative execution to AI agents**: CPU speculative execution has a 40+ year history; porting it to AI agent interaction scenarios is a natural yet previously overlooked direction.
- **Lossless guarantee enables direct deployment**: As a backend optimization that is completely transparent to users, it requires no trust in speculative results.
- **Theory guides the optimal strategy**: Beyond proposing the method, the paper derives the theoretically optimal threshold $p^\star$, eliminating the need for hyperparameter search.

## Limitations & Future Work
- **Depends on the predictability of the action space**: If agent actions are highly stochastic or creative (e.g., open-ended writing), prediction accuracy will be low, making speculation wasteful.
- **Applicable only to environments with deterministic verification**: Exact judgment of "predicted action = true action" is required; for continuous action spaces, a matching threshold must be defined.
- **Speculator training not explored**: Off-the-shelf small models are used as Speculators; the possibility of training dedicated Speculators to improve match rates is not investigated.

## Related Work & Insights
- **vs. Speculative Decoding**: Speculative decoding accelerates token-level generation, while Speculative Actions accelerates action-level environment interaction; the two approaches can be used in combination.
- **Connection to LoongRL**: The action sequences generated by LoongRL's plan-retrieve-reason-recheck paradigm are likely highly predictable (especially the plan and retrieve steps), making them naturally amenable to acceleration via Speculative Actions.

## Rating
- Novelty: ⭐⭐⭐⭐ Adapts a classic idea to a new scenario; the concept is elegant, though it does not alter the underlying algorithm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-task validation with theoretical alignment, though larger-scale agent tasks are lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations and clear system design.
- Value: ⭐⭐⭐⭐⭐ High practical value; directly deployable to accelerate existing agent systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DA-AC: Distributions as Actions — A Unified RL Framework for Diverse Action Spaces](distributions_as_actions_a_unified_framework_for_diverse_action_spaces.md)
- [\[ICLR 2026\] ANO: Faster is Better in Noisy Landscapes](ano_faster_is_better_in_noisy_landscape.md)
- [\[AAAI 2026\] Whispering Agents: An Event-Driven Covert Communication Protocol for the Internet of Agents](../../AAAI2026/others/whispering_agents_an_event-driven_covert_communication_protocol_for_the_internet.md)
- [\[AAAI 2026\] Faster Certified Symmetry Breaking Using Orders With Auxiliary Variables](../../AAAI2026/others/faster_certified_symmetry_breaking_using_orders_with_auxiliary_variables.md)
- [\[AAAI 2026\] Enhancing Control Policy Smoothness by Aligning Actions with Predictions from Preceding States](../../AAAI2026/others/enhancing_control_policy_smoothness_by_aligning_actions_with_predictions_from_pr.md)

</div>

<!-- RELATED:END -->
