---
title: >-
  [Paper Note] Position: Deployed Reinforcement Learning should be Continual
description: >-
  [ICML 2026][Reinforcement Learning][Continual RL] This is a position paper: the authors argue that any RL system that still obtains evaluative reward signals after deployment and whose environmental complexity exceeds the agent's representation/computational capacity is essentially a Continual Reinforcement Learning (CRL) problem. It advocates for abandoning the "train-then-fix" paradigm in favor of allowing agents to continuously update policies during deployment.
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Continual RL"
  - "Measurable Deployment"
  - "History Process"
  - "Non-stationarity"
  - "Train-then-fix"
date: 2026-05-08
content_hash: 2ddb702945f06866
---

# Position: Deployed Reinforcement Learning should be Continual

**Conference**: ICML 2026  
**arXiv**: [2606.04029](https://arxiv.org/abs/2606.04029)  
**Code**: None (position paper)  
**Area**: Reinforcement Learning / Continual Learning / Post-deployment Adaptation  
**Keywords**: Continual RL, Measurable Deployment, History Process, Non-stationarity, Train-then-fix

## TL;DR
This is a position paper: the authors argue that any RL system that still obtains evaluative reward signals after deployment and whose environmental complexity exceeds the agent's representation/computational capacity is essentially a Continual Reinforcement Learning (CRL) problem. It advocates for abandoning the "train-then-fix" paradigm in favor of allowing agents to continuously update policies during deployment.

## Background & Motivation
**Background**: Iconic RL achievements (TD-Gammon, AlphaGo, OpenAI Five, GT Sophy, stratus balloons, Tokamak control) almost all follow the train-then-fix paradigm—deploying a frozen policy after massive offline training. This convention stems from both engineering stability requirements and the mathematical tradition of "converging to $\pi^\star$" under MDP formalization.

**Limitations of Prior Work**: Frozen policies fail to maintain performance in real-world deployments and rely on periodic retraining, resulting in a sawtooth performance curve (decay followed by manual retraining). Systems like Cursor Tab and Lyft process hundreds of millions of requests daily; fixed policies cannot keep pace with changes in user behavior, library versions, or market structures. Sim-to-real transitions in robotics also demonstrate that fixed policies fail due to wear and tear, lighting changes, or sensor drift.

**Key Challenge**: Traditional MDP formalization assumes a stationary environment, state accessibility, and the existence of a fixed point $\pi^\star$, thereby modeling learning as a "one-time solution." However, the Big World Hypothesis points out that real-world complexity far exceeds any agent's representation capacity, making the optimal policy neither expressible nor reachable. Furthermore, there are four sources of non-stationarity post-deployment: action-induced non-stationarity, dynamic drift, goal evolution, and emergent novelty. An agent constrained by the "solve-then-freeze" mindset is destined to continuously yield performance to the environment.

**Goal**: (1) Formally name the common scenario "measurable deployment" where evaluative feedback is still received post-deployment; (2) Use the history process formalization to argue that it is essentially a CRL problem; (3) Provide action lists for both practitioners and researchers.

**Key Insight**: Starting from the definition of CRL by Abel et al. (2023)—"the problem where the best agent never stops learning"—combined with the history process formalization proposed by Bowling et al., this paper reframes "whether continual learning is required" from an algorithmic attribute back to a problem property.

**Core Idea**: When reward signals persist while the optimal policy is not within the reachable policy set, "stopping the search" is a suboptimal behavior. The optimal solution for measurable deployment is to treat deployment itself as a learning process.

## Method
As a position paper, this work introduces no new algorithms but provides a set of formal arguments, three real-world deployment cases, and action lists for two categories of audiences.

### Overall Architecture
The argument chain consists of four components: (1) Rewriting RL formalization using history processes to bypass MDP stationarity/reset assumptions; (2) Listing four sources of non-stationarity in measurable deployment to prove it is inherently a CRL problem; (3) Mapping three real-world cases (Cursor Tab, Lyft, Sim-to-Real) to different non-stationarity sources; (4) Introducing the dichotomy of continual vs. non-continual learners to reduce "continuality" to whether a learning rule $\sigma$ terminates the search within a policy set.

### Key Designs

**1. Formal Definition of Measurable Deployment: Making "Should we continue learning" a Decidable Condition**

MDP language carries the implication of the existence of a fixed point $\pi^\star$, inducing researchers into a "train-then-finish" mindset. The authors instead use history processes to describe the environment—$e:\mathcal H\times\mathcal A\to\Delta(\mathcal O)$, where $\mathcal H=\bigcup_{n=0}^\infty(\mathcal A\times\mathcal O)^n$ represents all finite histories. An agent is defined by a policy $\pi:\mathcal S\to\Delta(\mathcal A)$ plus a learning rule $\sigma:\mathcal H\to\Delta(\Pi)$. This language makes no assumptions about resets, Markov properties, or state reachability, making it a better fit for real deployment. A deployment is defined as measurable if and only if: (i) it operates in the big world regime, where the optimal policy $\pi^\star$ is outside the reachable policy set $\Pi$ or is computationally unreachable; (ii) evaluative rewards continue to be received after deployment. Once both conditions are met, the best agent cannot terminate its search, and the problem necessarily falls under CRL.

**2. Four Sources of Post-deployment Non-stationarity: Decomposing "Why CRL is Mandatory" into Four Verifiable Dimensions**

The authors decompose non-stationarity into four identifiable sources: (i) Action-induced—the agent's own actions change the future history distribution (e.g., recommendation systems reshaping user preferences), closely related to performative prediction; (ii) Dynamic environmental changes—external factors like seasonality, hardware aging, or market structures; (iii) Goal evolution—according to the reward hypothesis, the goal itself may change, or weights in multi-objective scenarios may drift; (iv) Emergent novelty—the Big World Hypothesis ensures a finite-capacity agent will encounter action-observation sequences never seen during training. Through cases like Cursor Tab, Lyft, and Sim-to-Real, these sources are identified as Primary, Present, or Implicit.

**3. Continual vs. Non-Continual Learner Dichotomy: Reducing "Continuality" to the Learning Rule**

Author points out that many mistake catastrophic forgetting or plasticity loss for the defining features of CRL, but those are algorithmic side effects. From a history process perspective, learning is a search over a policy set $\Pi$. An agent either stops searching at some history and locks a policy (non-continual learner) or never terminates the search (continual learner). A minimal example is provided: a small network with 64 parameters using SGD. If the step-size anneals to 0, it is non-continual; if using meta-gradients like IDBD to prevent the step-size from reaching zero, it is continual. CRL is thus defined as "a problem where the best agent cannot terminate the search."

## Key Experimental Results

### Case Comparison Table
The paper uses a table to align three real-world deployment systems with the four non-stationarity sources:

| Source of Non-stationarity | Cursor Tab | Lyft | Sim-to-Real |
|----------|------------|------|-------------|
| Action-induced NS | Implicit | **Primary** | Implicit |
| Environmental Dynamic Changes | Implicit | Present | **Primary** |
| Goal Evolution | Present | Implicit | Implicit |
| Emergent Novelty | **Primary** | Present | Present |

*Primary* indicates the dominant driver, *Present* indicates clear existence, and *Implicit* indicates existence without prominence.

### Industrial Deployment Gains

| System | Quantitative Gain | Continual Learning Cadence |
|------|----------|--------------|
| Cursor Tab | 400M daily requests; suggestions −21%, acceptance rate +28% | Policy updates every 1.5–2 hours |
| Lyft Matching | Millions of extra completed trips annually, +$30M revenue | Online RL + switchback safety verification |
| Rusting Pendulum | Train-then-fix degrades with friction accumulation; continual learner maintains performance | Experimental toy environment |

### Key Findings
- All three industrial systems rely on evaluative rewards (acceptance rates, trip completion rates) for online updates; the paper emphasizes that this signal often already exists in deployments but is frequently discarded.
- Cursor Tab's choice of policy gradient forces on-policy updates and a 1.5–2 hour iteration cycle, demonstrating how solution-level constraints shape engineering practices.
- Lyft engineers maintain safety through switchback experiments; the paper recommends a three-layer assurance: pre-deployment verification + continuous online verification + fallback policies.

## Highlights & Insights
- **Defining deployment as a learning process**: Traditional MLOps views deployment as the "end of training and start of service." This paper flips it: "the deployed model is a learning system, and production data is training data."
- **Engineering significance of history processes**: Shifting from MDPs to history processes exposes default engineering assumptions (like resetability) that are almost always violated in real-world deployments.
- **Problem vs. Solution distinction**: By categorizing catastrophic forgetting as an algorithmic challenge and non-stationarity as a problem feature, it prevents the community from equating "solving forgetting" with "solving CRL."
- **Transferable Trick**: Using controlled non-stationarity (perturbed rewards, shifted observations) to stress-test system adaptability is recommended as a standard development practice for CRL.

## Limitations & Future Work
- The scope of "measurable deployment" is narrow; scenarios with sparse, delayed, noisy, or unobservable rewards (e.g., a home Roomba unable to judge cleaning quality) are not covered.
- The "Rusting Pendulum" is a minimal demo, and industrial cases are retrospective; there is a lack of controlled comparisons to quantify the gap between "continual vs. fixed" in academic benchmarks.
- Regarding safety, the authors argue "adaptation is safer than stagnation," but provide only directions (shielded RL, constrained MDPs) rather than deployment-ready formal safety verification solutions.
- The paper does not discuss in detail how reward hacking or the Goodhart effect might worsen under continuous deployment.

## Related Work & Insights
- **vs. Abel et al. (2023)**: While Abel provided a formal definition of CRL, this paper applies it to industrial deployment and introduces "measurable deployment" to engage the research community with existing RL systems.
- **vs. Big World Hypothesis (Javed & Sutton 2024)**: BWH argues agent capacity is always less than world complexity; this paper uses that as an existential argument for why measurable deployment must be CRL.
- **vs. Alberta Plan (Sutton et al. 2022)**: While the Alberta Plan is a long-term research roadmap, this paper acts as a short-term deployment manual.

## Rating
- Novelty: ⭐⭐⭐⭐ Strong concept synthesis (measurable deployment + 4 NS sources), though the underlying definition builds on Abel et al.
- Experimental Thoroughness: ⭐⭐⭐ Primarily relies on industrial cases and toy demos; lacks controlled comparative experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear argument chain with a good balance between theory and case studies.
- Value: ⭐⭐⭐⭐⭐ Provides a clear direction for the RL deployment community with actionable practical advice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Position: Lifetime Tuning is Incompatible with Continual Reinforcement Learning](../../ICML2025/reinforcement_learning/position_lifetime_tuning_is_incompatible_with_continual_reinforcement_learning.md)
- [\[ICML 2026\] Shapley Neuron Values for Continual Learning: Which Neurons Matter Most?](shapley_neuron_values_for_continual_learning_which_neurons_matter_most.md)
- [\[ICLR 2026\] Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning](../../ICLR2026/reinforcement_learning/principled_fast_and_meta_knowledge_learners_for_continual_reinforcement_learning.md)
- [\[NeurIPS 2025\] Continual Knowledge Adaptation for Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/continual_knowledge_adaptation_for_reinforcement_learning.md)
- [\[CVPR 2026\] Resolving the Stability-Plasticity Dilemma in Reinforcement Learning via Complementary Continual Critics](../../CVPR2026/reinforcement_learning/resolving_the_stability-plasticity_dilemma_in_reinforcement_learning_via_complem.md)

</div>

<!-- RELATED:END -->
