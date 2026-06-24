---
title: >-
  [Paper Note] General Agents Contain World Models
description: >-
  [ICML 2025][World Models] This work theoretically proves that any agent capable of generalizing across multi-step goal-conditioned tasks must implicitly learn a predictive model of its environment (a world model), and this model can be extracted from the agent's policy—the stronger the agent and the more complex the goals, the more accurate its implicit world model.
tags:
  - "ICML 2025"
  - "World Models"
  - "Goal-Conditioned Agents"
  - "Reinforcement Learning Theory"
  - "Kolmogorov-Arnold Theorem"
  - "Interpretability"
date: 2026-05-08
content_hash: 55ab28f826342c45
---

# General Agents Contain World Models

**Conference**: ICML 2025  
**arXiv**: [2506.01622](https://arxiv.org/abs/2506.01622)  
**Code**: No public code  
**Area**: AI Theory / Reinforcement Learning / Agents  
**Keywords**: World Models, Goal-Conditioned Agents, Reinforcement Learning Theory, Kolmogorov-Arnold Theorem, Interpretability

## TL;DR
This work theoretically proves that any agent capable of generalizing across multi-step goal-conditioned tasks must implicitly learn a predictive model of its environment (a world model), and this model can be extracted from the agent's policy—the stronger the agent and the more complex the goals, the more accurate its implicit world model.

## Background & Motivation

**Background**: In the pursuit of Artificial General Intelligence (AGI), whether a world model is necessary remains a central debate. Explicit model-based methods (e.g., Dreamer, MuZero) directly learn an environment model for planning, while model-free methods (e.g., PPO, RT-2) attempt to bypass world model learning entirely through end-to-end policy learning.
   
**Limitations of Prior Work**:
   - Model-based methods struggle with learning world models—real-world environments are extremely complex, leading to compounding model errors.
   - Model-free methods have demonstrated powerful generalization capabilities across many tasks (e.g., Gato, RT-2), but emerging evidence suggests that these model-free agents actually learn implicit world models (e.g., Othello-GPT).
   - There is a lack of a theoretical framework to answer: "Is a world model *necessary* for general agents?"

**Key Challenge**: Brooks' view of "Intelligence without representation" suggests that all intelligent behavior can emerge through perception-action loops without explicit world representations. However, this creates friction with the fact that finite agents need to generalize—without an environment model, how can correct decisions be made on unseen long-horizon goals?

**Goal**: To provide a *formal proof* to answer:
   - Is a world model necessary for general agents?
   - How accurate does a world model need to be to support a given level of capability?
   - Can a world model be extracted from an agent's policy?

**Key Insight**: Under the framework of controlled Markov processes (cMP), a "bounded goal-conditioned agent" is defined (a policy capable of completing a bounded-depth sequence of goals with a bounded regret rate). Then, a reduction proof is constructed to recover environment transition probabilities from the agent's policy.

**Core Idea**: Any goal-conditioned policy satisfying a regret bound inherently contains a bounded-error world model. Learning such a policy is information-theoretically equivalent to learning a world model.

## Method

### Overall Architecture

This work is a **theory-oriented** study, where the core contributions are two theorems and an accompanying algorithm:

- **Input**: A goal-conditioned policy $\pi(a_t | h_t; \psi)$
- **Output**: An approximation of the environment transition function $\hat{P}_{ss'}(a)$
- **Process**: Infer transition probabilities from the policy's behavior by querying it with a series of carefully designed composite goals (either-or decisions).

### Key Designs

1. **Controlled Markov Process (cMP)**:

    - **Function**: Defines the environmental framework within which the agent operates—state space $\mathbf{S}$, action space $\mathbf{A}$, and transition function $P_{ss'}(a) = P(S_{t+1}=s'|A_t=a, S_t=s)$.
    - **Core Assumption (Assumption 1)**: The environment is finite, irreducible (any state is reachable from any other state), stationary, and $|\mathbf{A}| \geq 2$.
    - **Design Motivation**: This is the most standard environment assumption in reinforcement learning theory, ensuring the theorems are as widely applicable as possible. Irreducibility guarantees that the agent can navigate between any two states, which is crucial for constructing composite goals.

2. **Bounded Goal-Conditioned Agent (Definition 5)**:

    - **Function**: Characterizes a "general agent" using minimal assumptions—the ability to complete goals of bounded complexity with a bounded failure rate.
    - **Core Definition**: A policy $\pi$ satisfies:
    $P(\tau \models \psi | \pi, s_0) \geq \max_\pi P(\tau \models \psi | \pi, s_0)(1 - \delta)$
   for all $\psi \in \Psi_n$, where $\delta \in [0,1]$ is the maximum failure rate and $n$ is the maximum goal depth.
    - **Design Motivation**:
      - It does not assume the agent is optimal ($\delta > 0$ allows sub-optimal behavior).
      - It does not assume rationality (traditional rationality assumptions, such as preference ordering, are not required).
      - It only requires bounded capability over goals of a certain complexity, which is the weakest requirement for "generality."

3. **Theorem 1: General Agents Contain World Models**:

    - **Core Conclusion**: For an agent satisfying Definition 5, its policy completely determines an approximation of the environment transition probability $\hat{P}_{ss'}(a)$, with the error bound:
    $|\hat{P}_{ss'}(a) - P_{ss'}(a)| \leq 2P_{ss'}(a)\sqrt{\frac{1}{1-\delta} \cdot \frac{1}{n}}$
   For $\delta \ll 1, n \gg 1$, the error scales as $\mathcal{O}(\delta/n) + \mathcal{O}(1/n)$.
    - **Key Implications**:
      - The closer the agent is to optimal ($\delta \to 0$), the more accurate the world model.
      - The larger the goal depth the agent can handle ($n \to \infty$), the more accurate the world model.
      - Even sub-optimal agents ($\delta \sim 1$) must learn an accurate world model as long as they can handle sufficiently long goal sequences.

4. **Theorem 2: Myopic Agents Do Not Need World Models**:

    - **Function**: Proves that myopic agents optimizing only immediate outcomes ($n=1$) do not need to learn transition probabilities.
    - **Core Conclusion**: For an optimal myopic agent, the transition probability bound extractable from its policy is trivial ($\epsilon = 1$) and tight.
    - **Design Motivation**: Defines the **boundary condition** for the necessity of world models—only multi-step goals require them. This aligns with intuition: making single-step decisions only requires knowing $\arg\max_a P_{ss'}(a)$, not the exact probability values.

5. **Algorithm 1: Extracting World Models from Policies**:

    - **Function**: Provides a general unsupervised algorithm to recover the transition function from any qualifying agent's policy.
    - **Mechanism**: Construct a dilemma-choice goal $\psi_{a,b}(k,n) = \psi_a(k,n) \vee \psi_b(k,n)$:
        - Goal $\psi_a$: first execute action $a$, then transit $(a,s) \to s'$ at most $k$ times (out of $n$ attempts).
        - Goal $\psi_b$: first execute action $b$, then transit $(a,s) \to s'$ more than $k$ times.
        - The probability of an optimal agent achieving each goal is given by the cumulative binomial distribution, approximately $P_n(X \leq k)$ and $P_n(X > k)$.
        - Iterate $k$ from 0 to $n$ to find the critical point $k^*$ where the agent switches from pursuing $\psi_b$ to pursuing $\psi_a$.
        - $k^*$ is approximately equal to the median $\lfloor P_{ss'}(a)(n+1) \rfloor$, thereby solving for $\hat{P}_{ss'}(a) \approx k^*/n$.
    - **Design Motivation**: This is the constructive part of the theoretical proof. By designing specific goals, the estimation of transition probabilities is reduced to observing the agent's action selection. The algorithm is general (applicable to all qualifying agents and environments) and unsupervised (the only input is the policy $\pi$).

### Loss & Training

This is a theoretical work and does not involve training in the traditional sense. The experimental section trains agents in randomly generated cMP environments (20 states, 5 actions), improving agent capability by increasing the training trajectory length $N_{\text{samples}}$.

## Key Experimental Results

### Main Results: Relationship Between World Model Error and Agent Capability

| Agent Capability ($N_{\max}$ at $\langle\delta\rangle=0.04$) | Average Model Error $\langle\epsilon\rangle$ | Description |
|-------------------------------------------------------|---------------------------------------|------|
| $N_{\max} = 5$ | ~0.25 | Weak agent |
| $N_{\max} = 10$ | ~0.15 | Medium agent |
| $N_{\max} = 20$ | ~0.10 | Stronger agent |
| $N_{\max} = 50$ | ~0.05 | Strong agent |

The error scales as $\mathcal{O}(n^{-1/2})$, consistent with Theorem 1.

### Ablation Study

| Experimental Condition | Key Metric | Description |
|---------|---------|------|
| Agent satisfies strict regret bound | Error $\sim \mathcal{O}(n^{-1/2})$ | Consistent with theory |
| Agent violates regret bound ($\delta=1$ for some goals) | Average error is still $\sim \mathcal{O}(n^{-1/2})$ | Theorem assumptions can be relaxed |
| Different environment scales | Consistent error trend | Algorithm is universally applicable |

### Key Findings

1. **Theory Aligns with Experiments**: Even if the agent completely fails on some goals in the worst-case scenario ($\delta=1$), Algorithm 2 can still accurately recover the transition function as long as the average regret rate is sufficiently low.
2. **Error Scaling**: $\langle\epsilon\rangle \sim \mathcal{O}(n^{-1/2})$, exhibiting the same scaling behavior in both worst-case and average-case environments.
3. **Agents' Learned World Models Become More Precise with Capability Growth**: Increasing training data $\to$ agent can handle longer-horizon goals $\to$ more accurate extractable world model.

## Highlights & Insights

- **Profound Philosophical Implications**: Formally ends the debate of "model-based vs model-free"—if a model-free agent is sufficiently general, it automatically becomes model-based.
- **Architecture-Agnostic Proof**: Does not rely on specific architectures (Transformers, RNNs, etc.) or training methods (PPO, DQN, etc.), as long as the regret bound is satisfied.
- **Explaining Emergent Capabilities**: Provides a mechanism—during training, agents are forced to learn world models to minimize goal regrets, which in turn supports generalization to unseen tasks.
- **Safety Implications**: Precise world models can be extracted from sufficiently strong agents for safety auditing—the more dangerous (stronger) the agent, the more precise the extracted model.
- **Elegant Symmetry with Inverse Reinforcement Learning**: IRL infers goals using (policy + environment model); planning determines policies using (goal + environment model); this work recovers the environment model using (policy + goals).

## Limitations & Future Work

1. **Only Applicable to Fully Observable Environments**: Theorem 1 assumes the environment is fully observable to the agent—whether it holds under partially observable environments (POMDPs) remains unclear.
2. **Existence Proved, Not Usage**: The agent may contain a world model but not use it for planning (e.g., reflex agents).
3. **Scalability**: Algorithm 1 requires querying the policy individually for each $(s,a,s')$ triplet, which is computationally expensive in large state spaces.
4. **Continuous State/Action Spaces**: The current analysis is restricted to discrete and finite state and action spaces.
5. **Extraction of "Objective" World Models**: The extracted model is "objective" and does not necessarily reflect the "subjective" world model actually utilized by the agent.

## Related Work & Insights

- **Good Regulator Theorem** (Conant & Ashby, 1970): Attempted to prove a similar conclusion but had flaws—it only proved the existence of a deterministic policy, which is not equivalent to learning a world model.
- **Mechanistic Interpretability** (Othello-GPT, Li et al., 2022): Discovers implicit world models from activations. This work derives a stronger conclusion from weaker assumptions (requiring only the policy, not internal activations).
- **Inverse Reinforcement Learning** (Ng & Russell, 2000): A complementary relationship—IRL infers rewards from the policy and world model, whereas this work infers the world model from the policy and goals.
- **Causal World Models** (Richens & Everitt): Domain generalization requires causal models (stronger than transition probabilities), while task generalization only requires transition probabilities.
- **Insights**: This theoretical framework can be utilized to: (a) establish capability boundaries for foundation model agents; (b) develop new methods for extracting world knowledge from LLM agents; (c) provide formal guarantees from agent capabilities to world model precision for AI safety.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to rigorously prove that "general agents must contain world models," resolving a fundamental question in the field.
- Experimental Thoroughness: ⭐⭐⭐ Primarily a theoretical work; experiments are small-scale and confirmatory.
- Writing Quality: ⭐⭐⭐⭐⭐ Precise definitions, clear theorems, in-depth discussion, and excellent exposition of philosophical implications.
- Value: ⭐⭐⭐⭐⭐ Has a profound impact on RL theory, AI safety, and interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Beyond World Models: Rethinking Understanding in AI Models](../../AAAI2026/others/beyond_world_models_rethinking_understanding_in_ai_models.md)
- [\[ICML 2025\] Truly Self-Improving Agents Require Intrinsic Metacognitive Learning](truly_self-improving_agents_require_intrinsic_metacognitive_learning.md)
- [\[ICML 2025\] Time-Aware World Model for Adaptive Prediction and Control](time-aware_world_model_for_adaptive_prediction_and_control.md)
- [\[ICML 2025\] Nonparametric Modern Hopfield Models](nonparametric_modern_hopfield_models.md)
- [\[ICML 2026\] iWorld-Bench: A Benchmark for Interactive World Models with a Unified Action Generation Framework](../../ICML2026/others/iworld-bench_a_benchmark_for_interactive_world_models_with_a_unified_action_gene.md)

</div>

<!-- RELATED:END -->
