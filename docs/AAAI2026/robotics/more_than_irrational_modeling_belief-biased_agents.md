---
title: >-
  [Paper Note] More Than Irrational: Modeling Belief-Biased Agents
description: >-
  [AAAI 2026][Robotics][Computational Rationality] This paper proposes a computational rationality (CR) user model framework that interprets seemingly "irrational" human behavior as optimal decision-making under limited memory (belief bias). A nested particle filter (NPF) is used to online-infer the user's latent memory bound parameter $\theta$ and biased belief state $\tilde{b}$. The posterior mean (PM) error is reduced by 90% within 45 steps, and adaptive AI assistant policies are demonstrated within an assistive POMDP.
tags:
  - AAAI 2026
  - Robotics
  - Computational Rationality
  - Memory Decay
  - Nested Particle Filter
  - Belief Bias
  - Adaptive Assistance
date: 2026-05-08
content_hash: 45f701279a86b867
---

# More Than Irrational: Modeling Belief-Biased Agents

**Conference**: AAAI 2026
**arXiv**: [2511.12359](https://arxiv.org/abs/2511.12359)
**Code**: [GitHub](https://github.com/Yifan-Zhu/More-Than-Irrational-Modeling-Belief-Biased-Agents)
**Area**: User Modeling / Human-AI Collaboration
**Keywords**: Computational Rationality, Memory Decay, Nested Particle Filter, Belief Bias, Adaptive Assistance

## TL;DR

This paper proposes a computational rationality (CR) user model framework that interprets seemingly "irrational" human behavior as optimal decision-making under limited memory (belief bias). A nested particle filter (NPF) is used to online-infer the user's latent memory bound parameter $\theta$ and biased belief state $\tilde{b}$. The posterior mean (PM) error is reduced by 90% within 45 steps, and adaptive AI assistant policies are demonstrated within an assistive POMDP.

## Background & Motivation

**Background**: In human-AI collaboration, AI systems must infer user goals, beliefs, and future actions from past behavior. Computational rationality (CR) theory posits that humans are rational agents operating under cognitive constraints, and that "irrational" behavior stems from limited resources rather than genuine randomness.

**Limitations of Prior Work**: (1) Existing CR research focuses on specific applications such as gaze, typing, and driving, lacking general-purpose modeling of limited memory; (2) prior work either assumes perfect memory given an imperfect internal model, or attributes "irrationality" solely to a limited reasoning budget; (3) online inference of a user's latent cognitive bounds and dynamic belief states from passive observations is computationally intractable under exact inference, with complexity $O(|\mathcal{S}|^t \cdot t!)$.

**Key Challenge**: Memory-decay-induced belief bias renders user behavior apparently irrational, yet an AI assistant must distinguish between "genuine irrationality" and "rational decisions based on corrupted memory."

**Goal**: To construct a general-purpose limited-memory user model and propose a tractable online inference algorithm, enabling AI systems to track user cognitive states in real time and provide adaptive assistance.

**Key Insight**: Memory decay is explicitly modeled as a cognitive process $f_\theta$ that systematically corrupts the user's memory of historical observations, causing the belief state to deviate from the ground truth.

**Core Idea**: Apparently "irrational" behavior is in fact rational decision-making under biased beliefs — once the memory corruption mechanism is modeled, behavior becomes predictable and inferable.

## Method

### Overall Architecture

The framework comprises three layers: (1) **CR User Model**: a cognitive process $f_\theta$ is added on top of a standard POMDP, mapping the true history $h_t$ to a corrupted internal memory $\tilde{h}_t$; the user performs Bayesian filtering over $\tilde{h}_t$ to obtain a biased belief $\tilde{b}_t$ and executes an optimal policy with respect to $\tilde{b}_t$; (2) **Online Inference**: a nested particle filter (NPF) jointly maintains outer particles over $\theta$ and inner particles over $\tilde{h}$, updating weights based on observed user actions; (3) **Assistive POMDP**: the AI assistant leverages the inferred $\theta$ and $\tilde{b}$ to select an optimal intervention strategy (no intervention / memory hint / action hint).

### Key Designs

1. **Explicit Memory Corruption Process $f_\theta$**:

    - **Function**: Models the user's internal memory as a process that degrades dynamically over time.
    - **Mechanism**: At each time step $t$, the internal memory is updated as $\tilde{h}_t \sim f_\theta(\tilde{h}_{t-1}, o_t, a_{t-1})$. Under the memory decay instantiation, $\theta$ denotes the forgetting probability — at each step, each historical observation is replaced with a default value with probability $p = \theta$. The biased belief must be recomputed from scratch: $\tilde{b}_t(s_t) \propto \sum_{s_{:t-1}} p(s_0)\mathcal{O}(\tilde{o}_t^0|s_0) \prod_{i=1}^t \mathcal{O}(\tilde{o}_t^i|s_i)\mathcal{T}(s_i|s_{i-1},\tilde{a}_t^{i-1})$
    - **Design Motivation**: The Markov property no longer holds — when memory modifies $\tilde{o}_t^i$, the belief must be "retroactively" re-evaluated, which is the mathematical expression of human "memory replay."

2. **Nested Particle Filter Inference**:

    - **Function**: Online inference of the user's $\theta$ and $\tilde{h}$ from a passively observed action stream.
    - **Mechanism**: $N_\theta$ outer particles (distinct $\theta$ hypotheses) are maintained; under each outer particle, $N_{\tilde{h}}$ inner particles sample $\tilde{h}$. Weight update: $w^{(i,j)} \leftarrow w^{(i,j)} \cdot \pi_*(a_{t-1}|\tilde{b}_{t-1}^{(i,j)};\theta^i)$. Outer weights are aggregated via inner-particle likelihoods. Total complexity is $O(N_\theta N_{\tilde{h}} t |\mathcal{S}|)$.
    - **Design Motivation**: NPF is naturally suited to the joint estimation structure of "static parameters + dynamic states." The policy $\pi_*(\cdot;\theta)$ can be precomputed, avoiding the high cost of online policy learning.

### Loss & Training

Optimal policies $\pi_*(\cdot;\theta)$ for each $\theta$ are trained with PPO. The NPF inference phase involves no training — it is purely online Bayesian updating. The AI assistant's policy in the assistive POMDP is likewise trained with PPO, with a reward function defined as "user successfully reaches the goal" minus "intervention cost" (action hint cost > memory hint cost > no intervention).

## Key Experimental Results

### Main Results

Online inference accuracy (over 100 steps, averaged across all $\theta_{true}$, $\tau=3.0$):

| Metric | Initial Error | 45 Steps | 78 Steps | Final (100 Steps) |
|--------|:------------:|:--------:|:--------:|:-----------------:|
| PM Error | ~0.15 | −90% | −95% | **0.0087±0.0035** |
| MAP Error | ~0.2 | — | — | Near 0 |

### Ablation Study

Behavioral validity of the CR model (T-maze task):

| Memory Bound $\theta$ | Behavioral Pattern | Description |
|:---------------------:|--------------------|-------------|
| 0.0 (perfect memory) | Shortest-path to goal | One step down → straight → correct terminal |
| 0.4 (moderate decay) | Repeated observation of target | Collects redundant observations to improve memory robustness |
| 0.7 (severe decay) | "Forget-and-revisit" pattern | Returns to previously visited locations to reconfirm the goal |
| 1.0 (no memory) | Random guessing | Does not waste time exploring; selects randomly |

### Key Findings

1. **Different $\theta$ values yield an intuitively coherent behavioral spectrum**: the transition from optimal to random behavior is highly interpretable.
2. **Inference converges rapidly**: most error reduction is achieved within the first 20–30 steps (2–3 episodes).
3. **Adaptive assistance policies are sensible**: the AI learns to refrain from intervening for low-decay users, provide memory hints for moderate-decay users, and provide action hints for high-decay users.
4. **Intervention timing is precise**: the AI provides assistance at critical decision points (e.g., before a turn) rather than intervening continuously.

## Highlights & Insights

- **"Irrationality as rationality under biased beliefs"**: a cognitive-science insight is formalized into a computationally tractable mathematical framework.
- **General-purpose framework**: $f_\theta$ can be replaced by any memory model (decay, interference, capacity limits, etc.).
- **Inference efficiency is practically feasible**: $O(N_\theta N_{\tilde{h}} t |\mathcal{S}|)$ complexity, compared to the exponential cost of exact inference, supports real-time applications.
- **Closed-loop validation via the assistive POMDP**: the work goes beyond modeling and inference to demonstrate downstream application value.

## Limitations & Future Work

- The approach assumes AI access to the environment dynamics model ($\mathcal{T}$, $\mathcal{O}$), which is typically unavailable in real-world settings.
- Experiments are conducted only on the simple T-maze; scalability to complex continuous state spaces remains untested.
- The memory decay model (independent forgetting at each step) is overly simplistic; real human memory exhibits more complex decay curves.
- Policies must be pretrained for each $\theta$; this may become computationally infeasible when $\theta$ is continuous.
- No quantitative comparison is made with alternative user modeling methods (e.g., inverse reinforcement learning, Boltzmann rationality models).

## Related Work & Insights

- **vs. Kwon et al. 2020**: maintains an imperfect internal model but assumes perfect memory — this paper explicitly models memory limitations.
- **vs. Jacob et al. 2023**: explains irrationality via reasoning budget constraints — this paper focuses on memory constraints; the two are complementary.
- **vs. CRTypist (Shi et al. 2024)**: application-specific memory decay — this paper provides a general-purpose framework.
- **Insight**: Modeling *why* users make "wrong" choices is more valuable than merely predicting behavior — understanding cognitive limitations is the true foundation of personalization.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First work to unify a limited-memory CR model with efficient online inference into a general framework.
- Experimental Thoroughness: ⭐⭐⭐ Limited to the T-maze environment; behavioral validity and inference accuracy are well-verified, but large-scale evaluation is absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are clear and the motivational arguments are elegantly presented.
- Value: ⭐⭐⭐⭐ Offers a methodological contribution to human-AI collaboration and adaptive systems, though empirical validation is limited.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Causal Inference Under Threshold Manipulation: Bayesian Mixture Modeling and Heterogeneous Treatment Effects](causal_inference_under_threshold_manipulation_bayesian_mixtu.md)
- [\[AAAI 2026\] Cross Modal Fine-Grained Alignment via Granularity-Aware and Region-Uncertain Modeling](cross_modal_fine-grained_alignment_via_granularity-aware_and_region-uncertain_mo.md)
- [\[AAAI 2026\] When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets](when_hallucination_costs_millions_benchmarking_ai_agents_in_high-stakes_adversar.md)
- [\[NeurIPS 2025\] AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling](../../NeurIPS2025/robotics/autotom_scaling_model-based_mental_inference_via_automated_agent_modeling.md)
- [\[ICLR 2026\] When Agents Persuade: Propaganda Generation and Mitigation in LLMs](../../ICLR2026/robotics/when_agents_persuade_propaganda_generation_and_mitigation_in_llms.md)

<!-- RELATED:END -->
