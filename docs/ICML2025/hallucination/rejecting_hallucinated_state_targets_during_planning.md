---
title: >-
  [Paper Note] Rejecting Hallucinated State Targets during Planning
description: >-
  [ICML 2025][Hallucination Detection][Goal-conditioned RL] This paper systematically identifies the types of "delusional behavior" caused by generators producing unfeasible goals (hallucinatory goals) in goal-conditioned decision planning, and designs a feasibility evaluator as an auxiliary module to identify and reject these unfeasible goals. Combined with off-policy learning rules, a distributional architecture, and hindsight relabeling data augmentation…
tags:
  - "ICML 2025"
  - "Hallucination Detection"
  - "Goal-conditioned RL"
  - "Hallucinatory Goals"
  - "Delusional Behavior"
  - "Hindsight Relabeling"
  - "OOD Generalization"
  - "Feasibility Evaluator"
date: 2026-05-08
content_hash: 203bcb5ec3b98f59
---

# Rejecting Hallucinated State Targets during Planning

**Conference**: ICML 2025  
**arXiv**: [2410.07096](https://arxiv.org/abs/2410.07096)  
**Code**: [https://github.com/mila-iqia/delusions](https://github.com/mila-iqia/delusions)  
**Area**: Hallucination Detection  
**Keywords**: Goal-conditioned RL, Hallucinatory Goals, Delusional Behavior, Hindsight Relabeling, OOD Generalization, Feasibility Evaluator

## TL;DR
This paper systematically identifies the types of "delusional behavior" caused by generators producing unfeasible goals (hallucinatory goals) in goal-conditioned decision planning, and designs a feasibility evaluator as an auxiliary module to identify and reject these unfeasible goals. Combined with off-policy learning rules, a distributional architecture, and hindsight relabeling data augmentation, this approach significantly reduces delusional behavior and enhances OOD generalization performance without modifying the original agent.

## Background & Motivation

**Background**: Goal-conditioned RL (GCRL) uses a generator to produce subgoals/goal states during decision-making to guide behavior, which is a mainstream method for enhancing RL generalization. Hindsight Experience Replay (HER) is the core technique for training such agents.

**Limitations of Prior Work**: Learned generators inevitably produce "hallucinatory goals"—either non-existent states (type G.3.1, e.g., unreachable positions surrounded by lava) or temporarily unreachable states (type G.3.1.1, e.g., being unable to return to an unarmed state after acquiring a sword). When the estimator also fails to correctly identify these problematic goals, the agent chases unreachable goals, resulting in "delusional behavior."

**Key Challenge**: Existing HER strategies (such as "future" and "episode") exhibit a fundamental trade-off between improving sample efficiency under non-delusional scenarios and handling delusional goals—"episode" excels at short-range estimation but fails to expose unreachable goals to the estimator, while "future" only provides temporally ordered data, leading to more blind spots.

**Goal**: (1) How to systematically identify different types of delusional behavior? (2) How to preventively reject unfeasible goals without modifying the original agent? (3) How to learn the ability to reject OOD delusional goals within training tasks?

**Key Insight**: Drawing inspiration from the psychiatric definition of "delusion"—false beliefs that cannot be rejected—this work systematically classifies false beliefs naturally arising during the RL learning process and enables the estimator to recognize unfeasible goals by expanding the training data distribution.

**Core Idea**: By designing a feasibility evaluator that combines off-policy learning, a distributional architecture, and hindsight relabeling data augmentation, the agent is enabled to reject hallucinatory goals produced by the generator during decision-making.

## Method

### Overall Architecture
The goal-conditioned framework comprises two core components: a **generator** that proposes candidate goal states, and an **estimator** that evaluates the feasibility and benefit of the goals. Delusional behavior arises from the misalignment between the two—the generator produces problematic goals (hallucinations), and the estimator fails to correctly reject them.

This paper proposes two auxiliary strategies to expand the training data distribution of the estimator, and introduces mixture strategies and a hybrid approach to satisfy the distinct requirements of the generator and the estimator, respectively.

### Key Designs

1. **The "generate" strategy — helping the estimator learn features of candidate goals**:

    - **Function**: Replaces the relabeled target with candidate goals currently produced by the generator during HER relabeling.
    - **Mechanism**: Since the estimator needs to evaluate the proposals from the generator during decision-making, using the generator's outputs directly as training data exposes the estimator beforehand to all types of problematic goals (including G.3.1).
    - **Implementation**: A Just-In-Time (JIT) approach where, when sampling training data from the buffer, relabeled goals are replaced with conditionally generated goals from the generator with a certain probability.
    - **Design Motivation**: Primarily targets type E.3.2.1 delusions (incorrect evaluation of non-existent goals), as the generator may produce various types of problematic goals.

2. **The "pertask" strategy — helping the estimator learn experienced goals**:

    - **Function**: Samples relabeled targets from all historical observations within the same training task.
    - **Mechanism**: Creates task-level source-target pairs, covering more long-range and temporarily unreachable combinations compared to trajectory-level "episode" and "future" strategies.
    - **Implementation**: Maintains an independent auxiliary experience replay buffer for each task to record all experienced states.
    - **Design Motivation**: Primarily targets type E.3.2.2 delusions (incorrect evaluation of temporarily unreachable goals), while also mitigating type E.3.2 delusions caused by short trajectories.

3. **Mixture Strategies and the Hybrid Method**:

    - **Mixture Strategies**: Blend multiple atomic strategies proportionally under a fixed training budget to achieve a trade-off.
    - **Hybrid Method**: The generator and estimator use independent relabeling processes. For example:
        - $\text{F-(E+G)}$: The generator uses "future", and the estimator uses 50% "episode" + 50% "generate".
        - $\text{F-(E+P)}$: The generator uses "future", and the estimator uses 50% "episode" + 50% "pertask".
        - $\text{F-(E+P+G)}$: The generator uses "future", and the estimator uses 50% "episode" + 25% "pertask" + 25% "generate".
    - **Design Motivation**: The generator should avoid exposure to problematic goals (otherwise it will produce more hallucinations), whereas the estimator precisely requires these exposures to learn rejection.

4. **Three Elements of the Feasibility Evaluator**:

    - **Off-policy compatible learning rules**: $Q$-value estimation based on TD updates, which can penalize the feasibility of unreachable targets upon seeing their training data.
    - **Distributional architecture**: Uses distributional RL architectures to enhance estimation stability.
    - **Hindsight relabeling data augmentation**: Expands training data distribution through the aforementioned "generate" and "pertask" strategies.

### Loss & Training
The estimator utilizes standard Temporal Difference (TD) update rules to learn cumulative rewards and discount estimates between states. A key property is that, when training data contains source-target pairs with unfeasible targets, TD updates naturally penalize the estimated values of these unfeasible paths. The estimator acts as a plug-in attached to the existing agent, learning by observing the agent's interactions with the environment and the targets produced by the generator, without necessitating modifications to the original agent or its generator.

## Key Experimental Results

### Main Results

Experiments were conducted on two MiniGrid environments, SwordShieldMonster (SSM) and RandDistShift (RDS), utilizing two goal-conditioned methods, Skipper and LEAP.

**Frequency of Delusional Behavior of Skipper on SSM (Proportion of E.3.2.2 behavior at the end of training)**:

| Strategy | E.3.2.2 Behavior Frequency ↓ | Non-delusional Estimation Error ↓ | Aggregated OOD Performance ↑ |
|------|------------------|----------------|-------------|
| F-E (baseline) | ~35% | Low (short-range) | ~0.35 |
| F-P | ~20% | High (short-range) | ~0.25 |
| F-G | ~30% | Moderate | ~0.38 |
| F-(E+G) | ~25% | Low (short-range) | ~0.42 |
| F-(E+P) | ~12% | Low (short-range) | ~0.48 |
| F-(E+P+G) | ~10% | Low (short-range) | **~0.50** |

### Ablation Study

**Comparison of Characteristics of Different Atomic Strategies**:

| Strategy | Pros | Cons |
|------|------|------|
| "episode" | Accurate non-delusional short-range estimation | Cannot handle E.3.2.2; performs poorly under short trajectories |
| "future" | Enables learning of temporally abstract conditional generators | Inherits the disadvantages of "episode" and additionally leads to E.3.2 |
| "generate" | Effectively resolves E.3.2.1 | Dependent on generator, extra computational overhead, low non-delusional estimation efficiency |
| "pertask" | Effectively resolves E.3.2.2 and long-range E.3.2 | May introduce G.3.1.1 when training the generator; low short-range efficiency |

### Key Findings
- In the SSM environment, type E.3.2.2 delusions are the primary cause of Skipper's failure, with hybrid strategies containing "pertask" performing best.
- In the RDS environment (which has no temporarily unreachable states), E.3.2.1 is the main issue, and strategies containing "generate" are the most effective.
- The hybrid method allows the generator and estimator to use different training data distributions, simultaneously improving the performance of both.
- All mixture/hybrid strategies significantly outperform single atomic strategies in terms of OOD generalization.

## Highlights & Insights
- **Systematic Delusion Classification**: First to utilize the psychiatric concept of "delusion" to systematically identify and classify failure modes in RL, establishing a comprehensive taxonomy of G.3.1/G.3.1.1 (generator) and E.3.2/E.3.2.1/E.3.2.2 (estimator) delusions.
- **Plug-and-Play Design**: The feasibility evaluator does not require modifications to the original agent or generator and learns solely by observing interactions, offering high engineering practicality.
- **Core Insight on Diverse Data Requirements**: The generator should avoid exposure to problematic goals, whereas the estimator precisely requires such exposure—the hybrid method elegantly resolves this contradiction.

## Limitations & Future Work
- The experimental environments (SSM/RDS) are relatively simple and lack validation on high-dimensional continuous control tasks.
- The mixture ratios currently require manual tuning, lacking an adaptive ratio selection mechanism.
- Improvements to the generator itself are not sufficiently discussed; the current scheme primarily relies on the estimator's rejection capability.
- Computational overhead: the "generate" strategy, in particular, requires extra forward passes through the generator.

## Related Work & Insights
- **vs Original HER**: HER focuses only on sample efficiency, ignoring potential delusion issues caused by relabeling strategies. This paper is the first to systematically analyze this overlooked failure mode.
- **vs Goal Misgeneralization**: Goal misgeneralization studied by Di Langosco et al. is a subclass of delusional behavior, whereas this work provides a more comprehensive taxonomy and solution.
- **vs Non-delusional Q-learning (Lu et al. 2018)**: That work focuses on delusions caused by function approximators in model-free RL, whereas this paper extends it to specific delusion types inside the goal-conditioned framework.
- **Insight**: The core idea of the hybrid method—that different components require different training data distributions—can be transferred to RLHF training for LLMs, where policy models and reward models may also require different data exposure strategies.

## Rating
- Novelty: ⭐⭐⭐⭐ The systematic delusion classification and hybrid method are genuinely novel contributions, though the core technique (HER variants) is relatively incremental.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 sets of experiments (2 environments × 2 methods) offer comprehensive coverage, with 20 seed runs per set and sufficient confidence intervals.
- Writing Quality: ⭐⭐⭐⭐ The classification taxonomy is clear, the naming system is consistent, and the figures and tables contain rich information.
- Value: ⭐⭐⭐⭐ Identifies an important, overlooked failure mode in goal-conditioned RL and proposes a practical solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Monitoring Decoding: Mitigating Hallucination via Evaluating the Factuality of Partial Response during Generation](../../ACL2025/hallucination/monitoring_decoding_mitigating_hallucination_via_evaluating_the_factuality_of_pa.md)
- [\[ACL 2025\] ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs](../../ACL2025/hallucination/icr_probe_tracking_hidden_state_dynamics_for_reliable_hallucination_detection_in.md)
- [\[ACL 2026\] MultiHaluDet: Multilingual Hallucination Detection via LLM Hidden State Probing](../../ACL2026/hallucination/multihaludet_multilingual_hallucination_detection_via_llm_hidden_state_probing.md)
- [\[ACL 2025\] Activation Steering Decoding: Mitigating Hallucination in Large Vision-Language Models through Bidirectional Hidden State Intervention](../../ACL2025/hallucination/activation_steering_decoding_mitigating_hallucination_in_large_vision-language_m.md)
- [\[ICML 2025\] Steer LLM Latents for Hallucination Detection](steer_llm_latents_for_hallucination_detection.md)

</div>

<!-- RELATED:END -->
