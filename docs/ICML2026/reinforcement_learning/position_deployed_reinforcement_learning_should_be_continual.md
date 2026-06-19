---
title: >-
  [Paper Note] Position: Deployed Reinforcement Learning should be Continual
description: >-
  [ICML 2026][Reinforcement Learning][train-then-fix] This position paper argues that any RL system that continues to receive evaluative reward signals after deployment in an environment exceeding the agent's representational/computational capacity is fundamentally a Continual Reinforcement Learning (CRL) problem. The authors advocate for abandoning the "train-then-fix" p
tags:
  - ICML 2026
  - Reinforcement Learning
  - train-then-fix
date: 2026-05-08
content_hash: 5e119808fa2e8e01
---
# Position: Deployed Reinforcement Learning should be Continual

**Conference**: ICML 2026  
**arXiv**: [2606.04029](https://arxiv.org/abs/2606.04029)  
**Code**: None (position paper)  
**Area**: Reinforcement Learning / Continual Learning / Post-deployment Adaptation  
**Keywords**: Continual Reinforcement Learning, Measurable Deployment, History Process, Non-stationarity, train-then-fix  

## TL;DR
This position paper argues that any RL system that continues to receive evaluative reward signals after deployment in an environment exceeding the agent's representational/computational capacity is fundamentally a Continual Reinforcement Learning (CRL) problem. The authors advocate for abandoning the "train-then-fix" paradigm in favor of agents that continuously update policies during deployment.

## Background & Motivation
**Background**: Landmark achievements in RL (TD-Gammon, AlphaGo, OpenAI Five, GT Sophy, stratospheric balloons, Tokamak control) almost exclusively follow the "train-then-fix" paradigm—extensive offline training followed by a frozen policy deployment. This convention stems from engineering stability needs and the mathematical tradition of "converging to $\pi^\star$" under MDP formalization.

**Limitations of Prior Work**: Frozen policies cannot maintain performance in real-world deployments and rely on periodic retraining, resulting in a sawtooth performance curve (decay followed by manual retraining). Systems like Cursor Tab and Lyft process hundreds of millions of requests daily; fixed policies fail to keep pace with changes in user behavior, library versions, and market structures. Robotics sim-to-real also demonstrates that fixed policies fail when encountering wear, lighting changes, or sensor drift.

**Key Challenge**: Traditional MDP formalization assumes environmental stationarity, state reachability, and the existence of a fixed point $\pi^\star$, modeling learning as a "one-time solution." However, the Big World Hypothesis suggests that real-world complexity far exceeds any agent's representational capacity, making the optimal policy neither expressible nor reachable. Furthermore, there are four sources of non-stationarity post-deployment: action-induced, dynamic drift, goal evolution, and emergent novelty. An agent constrained by the "solve-then-freeze" mindset is destined to continually lose performance to the environment.

**Goal**: (1) Formally name the common scenario where evaluative feedback remains available after deployment as "measurable deployment"; (2) Use the history process formalization to prove it is essentially a CRL problem; (3) Provide action checklists for both practitioners and researchers.

**Key Insight**: Building on the definition of CRL by Abel et al. (2023)—"problems where the best agent never stops learning"—and the history process formalization by Bowling et al., the necessity of continual learning is shifted from an algorithmic attribute to a problem attribute.

**Core Idea**: When reward signals persist but the optimal policy is not within the reachable policy set, "stopping the search" is a suboptimal behavior. The optimal solution for measurable deployment is to treat deployment itself as a learning process.

## Method
As a position paper, no new algorithms are introduced. Instead, the paper provides a formal argument, three real-world deployment cases, and action checklists for two types of audiences.

### Overall Architecture
The argument chain consists of four parts: (1) Reformulating RL using the history process to bypass MDP stationarity/resettability assumptions; (2) Listing four types of non-stationarity sources in measurable deployment to prove it is a CRL problem; (3) Mapping Cursor Tab, Lyft, and Sim-to-Real cases to different non-stationarity sources; (4) Introducing the dichotomy of continual learner vs. non-continual learner, reducing "continuality" to whether the learning rule $\sigma$ terminates the search in the policy set.

### Key Designs

**1. Formal Definition of Measurable Deployment: Transforming "whether to continue learning" into a decidable condition**

The MDP framework carries the implication of a fixed point $\pi^\star$, inducing researchers into the "end after training" mindset. The authors utilize the history process to describe the environment: $e:\mathcal H\times\mathcal A\to\Delta(\mathcal O)$, where $\mathcal H=\bigcup_{n=0}^\infty(\mathcal A\times\mathcal O)^n$ represents all finite histories. The agent is defined by a policy $\pi:\mathcal S\to\Delta(\mathcal A)$ and a learning rule $\sigma:\mathcal H\to\Delta(\Pi)$. This framework does not assume countability, Markov properties, or state revisitability, making it more suitable for real deployment. A deployment is defined as measurable if and only if: (i) It exists in the big world regime, where $\pi^\star$ is outside the reachable policy set $\Pi$ or computationally unreachable; (ii) Evaluative rewards continue to be received post-deployment. If both are met, the best agent cannot stop searching, and the problem is categorized as CRL.

**2. Four Sources of Post-deployment Non-stationarity: Decomposing "why CRL is necessary" into four dimensions**

The paper identifies four identifiable sources of non-stationarity: (i) Action-induced—the agent’s actions change the future history distribution (e.g., recommendation systems reshaping user preferences), closely related to performative prediction; (ii) Dynamic environmental changes—external factors like seasons, hardware aging, and regulations; (iii) Goal evolution—per the reward hypothesis, goals may change or weights in multi-objective scenarios may drift; (iv) Emergent novelty—the Big World Hypothesis ensures a finite agent will encounter unseen action-observation sequences.

**3. Continual vs. Non-Continual Learner Dichotomy: Reducing "continuality" to the learning rule**

The authors clarify that catastrophic forgetting or plasticity loss are side effects of algorithms in CRL, not defining characteristics of the problem. Learning is viewed as a search over the policy set $\Pi$: an agent either stops searching at some history (non-continual learner) or never terminates (continual learner). For example, a small network with SGD is non-continual if the step-size anneals to zero, but becomes continual if meta-gradients (like IDBD) keep the step-size non-zero. CRL is thus defined as a problem where the best agent cannot terminate the search.

### Mechanism
The paper uses three real-world deployment cases (Cursor Tab, Lyft dispatch, Sim-to-Real robotics) as existence proofs: in successful industrial CRL systems, every category of non-stationarity exists as a primary driver, and continual learning provides quantitative gains. The "Rusting Pendulum" experiment further proves that while "train-then-fix" fails as friction accumulates, a continual learner maintains performance.

## Key Experimental Results

### Case Comparison Table
The paper aligns three deployment systems against the four sources of non-stationarity:

| Source of Non-stationarity | Cursor Tab | Lyft | Sim-to-Real |
|----------|------------|------|-------------|
| Action-induced NS | Implicit | **Primary** | Implicit |
| Environmental Dynamics | Implicit | Present | **Primary** |
| Goal Evolution | Present | Implicit | Implicit |
| Emergent Novelty | **Primary** | Present | Present |

*Primary* indicates a dominant driver, *Present* indicates significant presence, and *Implicit* indicates presence without being a focus.

### Industrial Deployment Gains

| System | Quantitative Gain | Continual Learning Cadence |
|------|----------|--------------|
| Cursor Tab | 400M daily requests; Suggestions −21%, Acceptance +28% | Policy updates every 1.5–2 hours |
| Lyft Matching | Millions of extra completed rides/year, +$30M revenue | Online RL + switchback verification |
| Rusting Pendulum | Fixed policy decays with friction; CRL agent maintains performance | Experimental toy environment |

### Key Findings
- All three industrial systems use evaluative rewards (acceptance rate, completion rate, performance metrics) for online updates; the paper emphasizes that such signals often already exist but are underutilized.
- Cursor Tab's choice of policy gradient forces an on-policy, 1.5–2 hour iteration cycle, showing how solution-level constraints shape engineering.
- Safety is addressed via multi-layer protection: pre-deployment validation, continuous online validation (switchback), and fallback policies.

## Highlights & Insights
- **Repurposing Deployment as a Learning Process**: Traditional MLOps views deployment as the end of training; this paper flips it—the deployed model is the learning system, and production data is the training data.
- **Engineering Significance of History Processes**: Moving to history processes exposes hidden assumptions like "resettability," which are almost always violated in real-world deployments.
- **Problem vs. Solution Distinction**: Catastrophic forgetting is an algorithmic challenge, while non-stationarity is a problem feature. Solving forgetting does not equate to solving the CRL problem.
- **Transferable Tricks**: Using "controlled non-stationarity" (perturbing rewards or shifting observations) as a standard dev practice for stress-testing adaptation.

## Limitations & Future Work
- The scope is limited to measurable deployment; scenarios with sparse, delayed, or unobservable rewards (e.g., home vacuum quality) are not covered.
- The "Rusting Pendulum" is a simplified demo, and industrial cases are retrospective; there is a lack of controlled benchmarks like a "big-world simulator."
- Regarding safety, the authors argue "adaptation is safer than stagnation" but offer directions (Shielded RL, Cautious agents) rather than deployment-ready solutions.
- Potential issues like Reward Hacking or the Goodhart effect in continual deployment were not discussed in detail.

## Related Work & Insights
- **vs. Abel et al. (2023)**: This paper applies Abel's formal definition to industrial deployment and introduces "measurable deployment" to engage the existing RL community.
- **vs. Big World Hypothesis (Javed & Sutton 2024)**: BWH provides the existential proof that CRL is necessary because agent capacity is always less than world complexity.
- **vs. Khetarpal et al. (2022)**: While surveys focus on algorithmic challenges, this paper refocuses on problem settings.
- **vs. Alberta Plan (Sutton et al. 2022)**: Complementary to the long-term Alberta Plan, this paper serves as a short-term deployment manual.

## Rating
- Novelty: ⭐⭐⭐⭐ (Strong synthesis of measurable deployment and non-stationarity sources).
- Experimental Thoroughness: ⭐⭐⭐ (Primary focus on industrial cases and toy demos; lacks controlled comparisons).
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear logic, good mix of theory and case studies).
- Value: ⭐⭐⭐⭐⭐ (Provides a clear direction and actionable checklist for RL deployment).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Position: Lifetime Tuning is Incompatible with Continual Reinforcement Learning](../../ICML2025/reinforcement_learning/position_lifetime_tuning_is_incompatible_with_continual_reinforcement_learning.md)
- [\[ICML 2026\] Shapley Neuron Values for Continual Learning: Which Neurons Matter Most?](shapley_neuron_values_for_continual_learning_which_neurons_matter_most.md)
- [\[ICML 2025\] Continual Reinforcement Learning by Planning with Online World Models](../../ICML2025/reinforcement_learning/continual_reinforcement_learning_by_planning_with_online_world_models.md)
- [\[ICLR 2026\] Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning](../../ICLR2026/reinforcement_learning/principled_fast_and_meta_knowledge_learners_for_continual_reinforcement_learning.md)
- [\[CVPR 2026\] Resolving the Stability-Plasticity Dilemma in Reinforcement Learning via Complementary Continual Critics](../../CVPR2026/reinforcement_learning/resolving_the_stability-plasticity_dilemma_in_reinforcement_learning_via_complem.md)

</div>

<!-- RELATED:END -->
