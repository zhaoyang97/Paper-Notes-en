---
title: >-
  [Paper Note] Learning Utilities from Demonstrations in Markov Decision Processes
description: >-
  [ICML 2025][Reinforcement Learning][Utility Learning] This paper introduces the Utility Learning (UL) problem to capture agents' risk attitudes by inferring their utility functions from demonstrations, proposes two provably efficient algorithms, and analyzes their sample complexity and identifiability.
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "Utility Learning"
  - "Risk-Sensitive Behavior"
  - "Inverse Reinforcement Learning"
  - "Utility Function"
  - "MDP"
date: 2026-05-08
content_hash: bc1d250b704dabb9
---

# Learning Utilities from Demonstrations in Markov Decision Processes

**Conference**: ICML 2025  
**arXiv**: [2409.17355](https://arxiv.org/abs/2409.17355)  
**Code**: None  
**Area**: Reinforcement Learning / Inverse RL  
**Keywords**: Utility Learning, Risk-Sensitive Behavior, Inverse Reinforcement Learning, Utility Function, MDP

## TL;DR
This paper introduces the Utility Learning (UL) problem to capture agents' risk attitudes by inferring their utility functions from demonstrations, proposes two provably efficient algorithms, and analyzes their sample complexity and identifiability.

## Background & Motivation

### Background

**Background**: Inverse Reinforcement Learning (IRL) infers reward functions from expert demonstrations, serving as a core method for understanding and mimicking behavior. However, classical IRL assumes agents are risk-neutral (maximizing expected cumulative rewards).

**Limitations of Prior Work**: Human behavior commonly exhibits risk sensitivity, demonstrating risk aversion or risk seeking when facing stochasticity. The risk-neutrality assumption not only introduces model misspecification but also fails to capture agents' risk attitudes directly.

**Key Challenge**: The modeling capability of IRL is insufficient to express risk preferences; even with perfect recovery of the reward function, it cannot distinguish between risk-averse and risk-seeking agents.

**Key Insight**: Introduce a utility function $u$ to explicitly represent risk attitudes, formulating its learning as the Utility Learning (UL) problem.

**Core Idea**: Define utility functions in MDPs to encode risk attitudes, analyze their partial identifiability, and design finite-sample algorithms for inference.

### Proposed Approach

**Goal**: ### Overall Architecture
Input: MDP environment structure + agent's demonstration trajectories → Utility learning algorithm → Output utility function (encoding risk attitude)

### Key Designs

1. **Utility Function Modeling**:

    - The utility function $u: \mathbb{R} \to \mathbb{R}$ acts on cumulative rewards to capture risk attitude
    - Concave function → risk aversion, convex function → risk seeking, linear function → risk neutrality
    - Design Motivation: Expected utility in economics.


## Method

### Overall Architecture
Input: MDP environment structure + agent's demonstration trajectories → Utility learning algorithm → Output utility function (encoding risk attitude)

### Key Designs

1. **Utility Function Modeling**:

    - The utility function $u: \mathbb{R} \to \mathbb{R}$ acts on cumulative rewards to capture risk attitude
    - Concave function → risk aversion, convex function → risk seeking, linear function → risk neutrality
    - Design Motivation: Expected utility theory (von Neumann-Morgenstern) in economics provides a solid theoretical foundation for risk modeling

2. **Partial Identifiability Analysis**:

    - Analyze under what conditions the utility function can be uniquely identified from demonstrations
    - Establish a precise characterization of unidentifiability: existence of equivalence classes
    - Design Motivation: Understanding the theoretical boundaries of the problem is crucial for algorithm design

3. **Two Efficient Algorithms**:

    - Algorithms are based on finite-sample guarantees
    - Analyze sample complexity: how many demonstration trajectories are needed for a given accuracy requirement
    - Design Motivation: Provide provable guarantees for finite-sample efficiency, rather than just asymptotic consistency

### Loss & Training
- Estimate the utility function from demonstrations via maximum likelihood or moment matching
- Leverage MDP structure to constrain the search space

## Key Experimental Results

### Main Results

| Scenario | Metric | Performance | Description |
|------|------|------|------|
| Risk-Averse Agent | Utility Function Recovery | Correctly identified concave utility | Verifies the model can distinguish risk attitudes |
| Risk-Seeking Agent | Utility Function Recovery | Correctly identified convex utility | Contrast with risk aversion |
| Risk-Neutral Agent | Utility Function Recovery | Correctly identified linear utility | Degenerates to standard IRL |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Different Sample Sizes | Recovery Error | Monotonically decreases as samples increase, matching theoretical predictions |
| Different MDP Structures | Identifiability | Utilities are unidentifiable under certain simple MDPs |

### Key Findings
- The identifiability of the utility function heavily depends on the structure of the MDP (state space size, degree of stochasticity)
- The two algorithms can effectively distinguish between risk-averse and risk-seeking behaviors under finite data conditions
- Standard IRL methods lead to significant model misspecification on risk-sensitive behavioral data

## Highlights & Insights
- **Novel Problem Definition**: Explicitly modeling risk attitudes as utility functions and defining the UL problem extends the expressive power of IRL
- **Theoretical Rigor**: Provides identifiability analysis and sample complexity guarantees
- **Bridging Economics and RL**: Introduces expected utility theory into the MDP/IRL framework

## Limitations & Future Work
- The experiments are proof-of-concept, lacking large-scale or real-world validation
- The utility function is assumed to belong to a known parametric family; non-parametric settings are more challenging
- Time-varying risk attitudes are not discussed (human risk preferences may change with contexts)

## Related Work & Insights
- Classical IRL (Abbeel & Ng 2004, MaxEnt IRL) assumes risk neutrality
- Risk-sensitive RL (CVaR-RL, Robust MDPs) handles risk from the perspective of agent training
- Prospect theory in behavioral economics (Kahneman & Tversky) provides richer ways to model risk
- Insight: What is learned from demonstrations is not only "what to do" but also "how to face uncertainty"

## Rating
- Novelty: ⭐⭐⭐⭐ The formal definition of the utility learning problem and the identifiability analysis are novel theoretical contributions
- Experimental Thoroughness: ⭐⭐⭐ A theory-driven work where experiments are primarily proof-of-concept
- Writing Quality: ⭐⭐⭐⭐ The mathematical presentation is rigorous and clear
- Value: ⭐⭐⭐⭐ Introduces an important modeling dimension to the IRL/LfD field

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Generalized Bisimulation Metric of State Similarity between Markov Decision Processes: From Theoretical Propositions to Applications](../../NeurIPS2025/reinforcement_learning/a_generalized_bisimulation_metric_of_state_similarity_betwee.md)
- [\[ICLR 2026\] Solving General-Utility Markov Decision Processes in the Single-Trial Regime with Online Planning](../../ICLR2026/reinforcement_learning/solving_general-utility_markov_decision_processes_in_the_single-trial_regime_wit.md)
- [\[ICML 2025\] Solving Zero-Sum Convex Markov Games](solving_zero-sum_convex_markov_games.md)
- [\[ICML 2025\] Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making](counterfactual_effect_decomposition_in_multi-agent_sequential_decision_making.md)
- [\[ICML 2025\] Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical Reinforcement Learning](divide_and_conquer_grounding_llms_as_efficient_decision-making_agents_via_offlin.md)

</div>

<!-- RELATED:END -->
