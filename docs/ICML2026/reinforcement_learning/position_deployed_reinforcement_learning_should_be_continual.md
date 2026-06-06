---
title: >-
  [Paper Note] Position: Deployed Reinforcement Learning should be Continual
description: >-
  [ICML 2026][Reinforcement Learning][Continual Reinforcement Learning] This position paper argues that any RL system that continues to receive evaluative reward signals after deployment…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Continual Reinforcement Learning"
  - "Measurable Deployment"
  - "History Process"
  - "Non-stationarity"
  - "train-then-fix"
date: 2026-05-08
content_hash: 9f6a321038508233
---

# Position: Deployed Reinforcement Learning should be Continual

**Conference**: ICML 2026  
**arXiv**: [2606.04029](https://arxiv.org/abs/2606.04029)  
**Code**: None (position paper)  
**Area**: Reinforcement Learning / Continual Learning / Post-deployment Adaptation  
**Keywords**: Continual Reinforcement Learning, Measurable Deployment, History Process, Non-stationarity, train-then-fix

## TL;DR
This position paper argues that any RL system that continues to receive evaluative reward signals after deployment, and where environment complexity exceeds the agent's representational or computational capacity, is inherently a Continual Reinforcement Learning (CRL) problem. The authors advocate for abandoning the "train-then-freeze" paradigm in favor of policies that update continuously during deployment.

## Background & Motivation
**Background**: Landmark achievements in RL (TD-Gammon, AlphaGo, OpenAI Five, GT Sophy, stratospheric balloons, Tokamak control) almost exclusively follow the train-then-fix paradigm—extensive offline training followed by freezing the policy for deployment. This convention stems from both engineering stability requirements and the mathematical tradition of "converging to $\pi^\star$" under the MDP formalism.

**Limitations of Prior Work**: Frozen policies fail to maintain performance in real-world deployments, necessitating periodic retraining. This results in "sawtooth" performance curves (decay followed by manual retraining triggers). Systems like Cursor Tab and Lyft process hundreds of millions of requests daily, where fixed policies cannot keep pace with changes in user behavior, library versions, or market structures; robotics sim-to-real also demonstrates that fixed policies fail when encountering wear, lighting changes, or sensor drift.

**Key Challenge**: Traditional MDP formalisms assume environment stationarity, state accessibility, and the existence of a fixed point $\pi^\star$, modeling learning as a "one-time solution." However, the Big World Hypothesis suggests that real-world complexity far exceeds any agent's representational capacity, making the optimal policy neither expressible nor reachable. Furthermore, post-deployment environments suffer from four sources of non-stationarity: action-induced changes, dynamic shifts, goal evolution, and emergent novelty. Agents constrained by the "solve-and-freeze" mindset are destined to lose performance to the environment.

**Goal**: (1) Formally define the common scenario where evaluative feedback remains available after deployment as "measurable deployment"; (2) Use the history process formalism to demonstrate that it is inherently a CRL problem; (3) Provide action checklists for both practitioners and researchers.

**Key Insight**: Building on the definition of CRL by Abel et al. (2023)—"a problem where the best agent never stops learning"—and the history process formalism by Bowling et al., the necessity of continual learning is framed as a property of the problem rather than an attribute of the algorithm.

**Core Idea**: When reward signals persist but the optimal policy is not within the reachable policy set, "stopping the search" is a suboptimal behavior. The optimal solution for measurable deployment is to treat deployment itself as a learning process.

## Method
As a position paper, this work does not introduce new algorithms but provides a formal argument, three real-world deployment cases, and action lists for two categories of stakeholders.

### Overall Architecture
The argumentative chain consists of four parts: (1) Reformulating RL using the history process to bypass MDP assumptions of stationarity and resettability; (2) Categorizing four sources of non-stationarity in measurable deployment to prove it is a CRL problem; (3) Mapping Cursor Tab, Lyft, and Sim-to-Real cases to different non-stationarity sources; (4) Introducing the dichotomy of continual vs. non-continual learners to clarify that "continual" refers to whether the learning rule $\sigma$ terminates the search in the policy space.

### Key Designs

1. **Formal Definition of Measurable Deployment**:

    - **Function**: Converts the ambiguous question of "should we continue learning after deployment" into a decidable formal condition.
    - **Mechanism**: The environment is described by a history process $e:\mathcal{H}\times\mathcal{A}\to\Delta(\mathcal{O})$, where $\mathcal{H}=\bigcup_{n=0}^\infty(\mathcal{A}\times\mathcal{O})^n$ represents all finite histories. An agent is a policy $\pi:\mathcal{S}\to\Delta(\mathcal{A})$ coupled with a learning rule $\sigma:\mathcal{H}\to\Delta(\Pi)$. Deployment is "measurable" if and only if (i) the deployment environment is in the "big world regime," where the optimal policy $\pi^\star$ is outside the reachable policy set $\Pi$ or is computationally unreachable; (ii) evaluative reward signals continue to be received post-deployment. When both are met, the best agent must not terminate its search, placing the problem in the CRL domain.
    - **Design Motivation**: The MDP framework implicitly suggests the existence of a fixed point $\pi^\star$, leading researchers into a "train-then-finish" mindset. The history process makes no assumptions about resettability, Markov properties, or state reachability, making it a more suitable mathematical language for real-world deployment.

2. **Four Sources of Post-deployment Non-stationarity**:

    - **Function**: Decomposes "why CRL is necessary" into four identifiable environment characteristics for engineers to assess their systems.
    - **Mechanism**: (i) Action-induced non-stationarity: the agent's actions change future history distributions (e.g., recommenders changing user preferences, trading strategies changing markets); closely related to performative prediction. (ii) Changes in environment dynamics: exogenous changes like seasons, hardware aging, or market structures. (iii) Evolving goals: per the reward hypothesis, goals can evolve, and weights in multi-objective scenarios drift over time. (iv) Emergent novelty: The Big World Hypothesis guarantees that finite-capacity agents will encounter unseen sequences; "black swan" events are extreme forms. The paper maps Cursor Tab, Lyft, and Sim-to-Real cases to show where each source is Primary, Present, or Implicit.
    - **Design Motivation**: Concretizing "non-stationarity" into four dimensions allows the classification of a system as CRL to be a check-box exercise rather than a philosophical debate.

3. **Dichotomy of Continual vs. Non-Continual Learner**:

    - **Function**: Clarifies the confusion between CRL as a problem versus an algorithm by situating the property in the learning rule.
    - **Mechanism**: Under the history process view, learning is a search over policy set $\Pi$. An agent is a non-continual learner if it locks a policy at some history, and a continual learner if the search never terminates. For example, a small network with SGD is non-continual if the step-size anneals to zero, but continual if using meta-gradients (like IDBD) to keep the step-size non-zero. A problem is CRL if and only if its best agent cannot terminate the search. This definition allows shifts in the policy set $\Pi$ or learning rule $\sigma$ to change the problem classification.
    - **Design Motivation**: Many mistake "catastrophic forgetting" or "plasticity loss" as defining features of CRL, whereas they are side effects of algorithms. This work strictly distinguishes problem-side characterization from solution-side challenges.

### Argument Strategy
The paper uses three case studies (Cursor Tab, Lyft dispatching, Sim-to-Real robotics) as existence proofs: in successful industrial CRL systems, every category of non-stationarity is a primary driver in at least one instance, and continual learning provides quantitative gains. The "Rusting Pendulum" toy experiment is also used to show that while train-then-fix fails as joint friction accumulates, a continual learner maintains performance.

## Key Experimental Results

### Case Mapping Table
The three industrial systems are aligned against the four non-stationarity sources:

| Non-stationarity Source | Cursor Tab | Lyft | Sim-to-Real |
|----------|------------|------|-------------|
| Action-induced NS | Implicit | **Primary** | Implicit |
| Dynamic Changes | Implicit | Present | **Primary** |
| Evolving Goals | Present | Implicit | Implicit |
| Emergent Novelty | **Primary** | Present | Present |

*Primary indicates a dominant driver; Present indicates clear presence; Implicit indicates presence without being the main focus.*

### Industrial Deployment Gains
| System | Quantitative Gain | Continual Learning Cadence |
|------|----------|--------------|
| Cursor Tab | 400M daily requests; suggestions −21%, acceptance +28% | Policy updates every 1.5–2 hours |
| Lyft Matching | Millions of extra completed rides/year, +$30M revenue | Online RL + switchback safety validation |
| Rusting Pendulum | Train-then-fix degrades with friction; continual learner maintains | Experimental toy environment |

### Key Findings
- All three industrial systems utilize evaluative rewards (acceptance rate, completion rate, performance metrics) for online updates; the paper emphasizes that these signals often already exist in deployments but are underutilized.
- Cursor Tab's choice of policy gradient → forced on-policy → 1.5–2 hour iteration cycles shows how solution-level constraints shape engineering practice (a "solution challenge").
- Lyft engineers noted the difficulty of "trusting a self-updating algorithm," relying on switchback experiments for safety validation. The paper recommends a three-layer guarantee: pre-deployment validation + continuous online validation + fallback policies.

## Highlights & Insights
- **Defining deployment as a learning process**: Traditional MLOps views deployment as the "end of training and start of serving." This paper flips it: "the deployed model is the learning system, and production data is training data." This perspective shift is a high-leverage conceptual contribution.
- **Engineering significance of history processes**: Moving from MDPs to history processes is not just mathematical aesthetics; it exposes hidden engineering assumptions like resettability and state revisitation that are routinely violated in reality.
- **Problem vs. Solution Distinction**: By framing catastrophic forgetting and plasticity loss as algorithmic challenges rather than problem definitions, it prevents the community from equating "solving forgetting" with "solving CRL."
- **Transferable Trick**: The authors suggest using "controlled non-stationarity" (perturbing rewards, shifting observations, simulating concept drift) to stress-test system adaptability as a standard development practice for CRL, similar to chaos engineering in ML pipelines.

## Limitations & Future Work
- The scope of "measurable deployment" is narrow—scenarios with sparse, delayed, noisy, or unobservable rewards (e.g., a Roomba that cannot evaluate cleaning quality) are not covered.
- The "Rusting Pendulum" is a minimal demo, and the industrial cases are retrospective explanations; there is a lack of controlled comparisons to quantify the "continual vs. fixed" gap across more domains.
- On safety, the authors argue "adaptation is safer than stagnation" but offer only high-level directions for formal safety verification under continual learning (shielded RL, constrained MDPs).
- The paper does not deeply discuss how reward hacking or Goodhart's Law might worsen under continuous deployment, which remains a significant hurdle.

## Related Work & Insights
- **vs. Abel et al. (2023)**: While Abel provided the formal definition of CRL, this paper applies it to industrial deployment and introduces "measurable deployment" to engage practitioners.
- **vs. Big World Hypothesis (Javed & Sutton 2024)**: BWH argues agent capacity is always less than world complexity; this paper uses that hypothesis as evidence for why measurable deployment must be CRL.
- **vs. Khetarpal et al. (2022) CRL survey**: The survey organizes algorithmic challenges, while this paper refocuses on problem settings.
- **vs. Alberta Plan (Sutton et al. 2022)**: The Alberta Plan is a long-term research roadmap; this is a short-term deployment handbook.

## Rating
- Novelty: ⭐⭐⭐⭐ High conceptual synthesis (measurable deployment + 4 non-stationarity sources + industrial case mapping), though base definitions leverage existing work.
- Experimental Thoroughness: ⭐⭐⭐ Primarily relies on industrial retrospective cases and toy demos; lacks controlled scientific benchmarks.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear argumentation; effectively alternates between theory and practice.
- Value: ⭐⭐⭐⭐⭐ Provides a clear direction for the RL deployment community with actionable suggestions; a standout in the ICML 2026 position track.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Shapley Neuron Values for Continual Learning: Which Neurons Matter Most?](shapley_neuron_values_for_continual_learning_which_neurons_matter_most.md)
- [\[NeurIPS 2025\] Continual Knowledge Adaptation for Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/continual_knowledge_adaptation_for_reinforcement_learning.md)
- [\[ICLR 2026\] Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning](../../ICLR2026/reinforcement_learning/principled_fast_and_meta_knowledge_learners_for_continual_reinforcement_learning.md)
- [\[NeurIPS 2025\] Temporal-Difference Variational Continual Learning](../../NeurIPS2025/reinforcement_learning/temporal-difference_variational_continual_learning.md)
- [\[AAAI 2026\] Where to Start Alignment? Diffusion Large Language Model May Demand a Distinct Position](../../AAAI2026/reinforcement_learning/where_to_start_alignment_diffusion_large_language_model_may_demand_a_distinct_po.md)

</div>

<!-- RELATED:END -->
