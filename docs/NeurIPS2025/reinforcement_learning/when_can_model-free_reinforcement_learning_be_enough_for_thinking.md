---
title: >-
  [Paper Note] When Can Model-Free Reinforcement Learning be Enough for Thinking?
description: >-
  [NeurIPS 2025][Reinforcement Learning][Thought MDP] This paper proposes the Thought MDP formalism to characterize the conditions under which "thinking" behavior emerges under model-free RL: policy initialization is the decisive factor; thinking actions are equivalent to the agent performing one step of policy improvement before acting; and open-source LLMs satisfy the necessary conditions for thinking to emerge.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Thought MDP
  - model-free RL
  - emergence of thinking behavior
  - policy initialization
  - dual-process theory
date: 2026-05-08
content_hash: bacae85eb597d59f
---

# When Can Model-Free Reinforcement Learning be Enough for Thinking?

**Conference**: NeurIPS 2025
**arXiv**: [2506.17124](https://arxiv.org/abs/2506.17124)
**Code**: [https://github.com/prediction-action-lab/thinking-as-control](https://github.com/prediction-action-lab/thinking-as-control)
**Area**: Reinforcement Learning
**Keywords**: Thought MDP, model-free RL, emergence of thinking behavior, policy initialization, dual-process theory

## TL;DR
This paper proposes the Thought MDP formalism to characterize the conditions under which "thinking" behavior emerges under model-free RL: policy initialization is the decisive factor; thinking actions are equivalent to the agent performing one step of policy improvement before acting; and open-source LLMs satisfy the necessary conditions for thinking to emerge.

## Background & Motivation

**Background**: Works such as DeepSeek-R1 have demonstrated that model-free RL can elicit "thinking" behavior in LLMs (i.e., generating a reasoning process within `<think>` tags before producing an answer). This phenomenon is surprising, as thinking tokens neither directly yield reward nor alter the external environment state.

**Limitations of Prior Work**: There is a lack of theoretical understanding of the conditions under which model-free RL produces thinking behavior. Existing approaches (e.g., MCTS, learned planning) either require an environment model or study implicit planning within network internals, rather than thinking as an explicit action.

**Core Problem**: Under what conditions will model-free RL select "thinking" as a **reward-maximizing strategy**?

**Key Insight**: The problem is formalized by extending the standard MDP with a thinking state space and a set of thinking actions, followed by an analysis of the behavior of policy iteration.

**Core Idea**: Thinking serves as a "transitional strategy" during learning—the optimal policy never thinks (Proposition 4), yet prior to policy convergence, thinking accelerates learning by **switching to a superior sub-policy**.

## Method

### Overall Architecture
The Thought MDP is defined as $\langle \mathcal{S}, \mathcal{A}, p, r, \gamma, \mathcal{T}, \mathcal{C}, p_{\mathcal{T}} \rangle$, extending the standard MDP with a thinking state space $\mathcal{T}$, a set of thinking actions $\mathcal{C}$, and a thinking transition function $p_{\mathcal{T}}$. The policy maps $\pi: \mathcal{S} \times \mathcal{T} \rightarrow \Delta(\mathcal{A} \cup \mathcal{C})$—the agent may select either environment actions or thinking actions. Thinking actions yield no reward and do not alter the environment state; they only modify the thinking state.

### Key Designs

1. **The Optimal Policy Never Thinks (Proposition 4)**:

    - Mechanism: Suppose the optimal policy $\pi^\star$ selects a thinking action $c$ at some $(s, \tau)$, eventually reaching some $\tilde{\tau}$ after $k$ thinking steps before executing an environment action. Then $v_{\pi^\star}(s,\tau) = \gamma^k v_{\pi^\star}(s,\tilde{\tau})$. A policy that directly executes $\pi^\star(s,\tilde{\tau})$ at $(s,\tau)$ is strictly superior (saving $k$ discounting steps)—a contradiction.
    - Design Motivation: This establishes that thinking is not an end state, but a **stepping stone along the learning trajectory**.

2. **Necessary Condition for the Emergence of Thinking (Theorem 5)**:

    - Mechanism: If a policy improvement step replaces the environment action at $(s, \tau)$ with a thinking action $c$ (transitioning to $\tau'$), then it must hold that $v_\pi(s,\tau') > v_\pi(s,\tau)$—i.e., there exists a thinking state $\tau'$ that yields better policy performance.
    - Design Motivation: This formalizes the key role of policy initialization in whether thinking emerges—the policy must already contain good sub-policies, with thinking actions serving merely as switches among them.

3. **Thinking as a Local Policy Improvement Operator**:

    - Mechanism: When $|\mathcal{T}|=2$, thinking is equivalent to selecting the better of two sub-policies $\pi(\tau_0)$ and $\pi(\tau_1)$. Corollary 8 further proves that each successive thinking step constitutes an improvement.
    - Design Motivation: This is analogous to decision-time planning but requires no forward search—it constitutes a more abstract form of local improvement.

4. **LLMs as Instantiations of the Thought MDP**:

    - Mechanism: Thinking tokens in LLMs correspond to thinking actions $c \in \mathcal{C}$, and changes to the KV cache correspond to thinking state transitions $p_\mathcal{T}$. Pre-training on multi-task language data yields a rich library of sub-policies (different contexts activate different generation modes), and RL teaches the model to switch to a superior sub-policy at the right moment by outputting tokens such as `<think>`.
    - Validation: Across 11 open-source LLMs on a multi-digit addition task, providing intermediate steps (i.e., "thinking") raises accuracy from 0–7% to 28–96%, confirming that thinking actions genuinely increase $v_\pi(s,\tau)$.

### Loss & Training
- The theoretical analysis is grounded in exact policy iteration.
- Non-linguistic experiments employ a 5×5 grid world with a causal transformer trained via REINFORCE.

## Key Experimental Results

### LLM Experiments — Multi-Digit Addition (Validating That Thinking Increases Return)

| Model | Accuracy w/o Thinking | Accuracy w/ Thinking | Gain |
|------|-----------|-----------|------|
| Qwen2.5-1.5B-Instruct | 7.2% | 71.2% | +64.0% |
| Qwen2.5-7B-Instruct | 5.1% | 96.1% | +91.0% |
| Llama-2-7b | 0.0% | 48.8% | +48.8% |
| Gemma-3-4b-it | 4.9% | 91.5% | +86.6% |
| Mistral-7B-v0.3 | 1.5% | 85.2% | +83.7% |

### Grid World Experiments (Validating That Thinking Accelerates Learning)

| Agent | Convergence Speed | Final Success Rate |
|--------|---------|----------|
| Scratch-Think | Does not converge | 0% |
| Scratch-NoThink | Does not converge | 0% |
| Pretrained-NoThink | Slow | ~50% |
| **Pretrained-Think** | **Fast** | **~90%** |

### Policy Iteration Dynamics in a Simple Thought MDP

| Iteration | Policy Characteristics | Remarks |
|------|---------|------|
| 1 | Far from goal → think first (switch sub-policy), then act | Thinking emerges |
| 4 | Near goal → act directly; far away → still think first | Mixed strategy |
| 10 | Act directly throughout; thinking ceases | Converges to optimal |

### Key Findings
- **Pre-training combined with thinking actions** are the two critical ingredients: removing either one causes failure. Pretrained-Think converges significantly faster than Pretrained-NoThink.
- In the grid world, thinking actions stabilize at ~15% after convergence (~2 thinking steps per 14-step episode), close to the theoretical optimum.
- Notably, sequences of environment actions can also serve a thinking-like role (e.g., consecutive downward moves "activate" a pre-trained sub-policy for navigating to the lower-right corner), but such implicit switching is considerably harder to discover than explicit thinking actions.

## Highlights & Insights
- **"The optimal policy never thinks" (Proposition 4)** is a counterintuitive yet mathematically rigorous result—thinking is a suboptimal but learning-path-beneficial transitional strategy. This elegantly explains why thinking should diminish as training converges (though in practice it does not fully disappear due to sampling noise).
- The equivalence **thinking = policy improvement before acting** forges a deep connection between two ostensibly unrelated fields—LLM reasoning and decision-time planning in classical RL.
- The central role of policy initialization: without the rich sub-policy library provided by multi-task pre-training, model-free RL cannot discover thinking behavior—explaining why DeepSeek-R1 requires a strong pre-trained model as its starting point.
- **Non-linguistic validation** (grid world) demonstrates the generality of the theory—thinking is not exclusive to language; any agent with internal state can learn to think when the requisite conditions are met.

## Limitations & Future Work
- The analysis assumes non-negative rewards and reachable positive rewards (Assumptions 2–3); in environments with negative rewards, thinking may degenerate into a "procrastination" strategy.
- The case where the world state changes during thinking in dynamic environments is not addressed (the current framework assumes a static environment).
- Validation is limited to toy domains—more complex non-linguistic thinking scenarios (e.g., robotic planning, game playing) remain to be explored.
- The relationship with the options framework warrants deeper investigation—thinking may be a special form of options, but with an "internal state switching cost."

## Related Work & Insights
- **vs. MCTS/AlphaZero**: These are model-based decision-time planning methods, whereas thinking constitutes model-free policy improvement—more lightweight but requiring good initialization.
- **vs. DeepSeek-R1**: R1 demonstrates the phenomenon (emergence of thinking); this paper provides the theoretical explanation (why it emerges and under what conditions).
- **vs. Chain-of-Thought**: CoT enforces thinking via prompting, whereas this paper studies RL autonomously discovering thinking—the latter is more fundamental.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The Thought MDP formalism is a wholly original and theoretically profound contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across 11 LLMs, a grid world, and theoretical analysis, though more complex non-linguistic scenarios are lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Concepts are precisely defined, theorem derivations are rigorous, and the connection to LLM practice is clearly articulated.
- Value: ⭐⭐⭐⭐⭐ Of far-reaching significance for understanding the emergence of LLM reasoning; provides the first formal answer to "when and why RL produces thinking."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Parameter-Free Algorithms for the Stochastically Extended Adversarial Model](parameter-free_algorithms_for_the_stochastically_extended_adversarial_model.md)
- [\[NeurIPS 2025\] Deep RL Needs Deep Behavior Analysis: Exploring Implicit Planning by Model-Free Agents](deep_rl_needs_deep_behavior_analysis_exploring_implicit_planning_by_model-free_a.md)
- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](learning_interactive_world_model_for_object-centric_reinforcement_learning.md)
- [\[NeurIPS 2025\] Optimizing the Unknown: Black Box Bayesian Optimization with Energy-Based Model and Reinforcement Learning](optimizing_the_unknown_black_box_bayesian_optimization_with_energy-based_model_a.md)
- [\[NeurIPS 2025\] Actor-Free Continuous Control via Structurally Maximizable Q-Functions](actorfree_continuous_control_via_structurally_maximizable_qf.md)

</div>

<!-- RELATED:END -->
