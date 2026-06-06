---
title: >-
  [Paper Note] Provable and Practical In-Context Policy Optimization for Self-Improvement
description: >-
  [ICLR 2026][Optimization][in-context learning] This paper proposes the In-Context Policy Optimization (ICPO) framework, theoretically proving that a single-layer linear self-attention Transformer…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "in-context learning"
  - "policy optimization"
  - "test-time scaling"
  - "self-reflection"
  - "mathematical reasoning"
date: 2026-05-08
content_hash: ec091d4ee86794c9
---

# Provable and Practical In-Context Policy Optimization for Self-Improvement

**Conference**: ICLR 2026
**arXiv**: [2603.01335](https://arxiv.org/abs/2603.01335)  
**Code**: [https://github.com/UNCSciML/ICPO](https://github.com/UNCSciML/ICPO)  
**Area**: Optimization
**Keywords**: in-context learning, policy optimization, test-time scaling, self-reflection, mathematical reasoning

## TL;DR
This paper proposes the In-Context Policy Optimization (ICPO) framework, theoretically proving that a single-layer linear self-attention Transformer, after sufficient pretraining, can simulate a policy optimization algorithm in context. Building on this, the paper designs a practical ME-ICPO algorithm that achieves multi-round test-time self-reflection via minimum-entropy selection and self-evaluation rewards, yielding significant gains on mathematical reasoning tasks (Qwen2.5-Math-7B improves from 11% to 30% on AIME 2024).

## Background & Motivation

**Background**: Test-time scaling has become an important paradigm for improving LLM reasoning—models iteratively refine their answers through multi-round self-reflection without parameter updates. Representative methods include Chain-of-Thought, Tree-of-Thoughts, Best-of-N, and Self-Refine.

**Limitations of Prior Work**: (a) Why does self-reflection emerge from pretraining? Prior work (e.g., Park et al. 2024) directly assumes LLMs possess posterior sampling/policy optimization capabilities without explaining their origin. (b) Theoretical analyses of in-context learning have focused on supervised learning (linear regression) and value function learning (TD learning), with no theory addressing policy optimization. (c) Existing methods such as Tree-of-Thoughts require multi-step search, incurring substantial computational overhead.

**Key Challenge**: How can a model leverage historical attempts and reward feedback in context to optimize its own output policy? Can Transformers theoretically realize such policy optimization without parameter updates?

**Goal**: (1) Provide a theoretical foundation for LLM self-reflection and self-improvement behavior. (2) Design a practical test-time scaling algorithm.

**Key Insight**: Self-reflection is formalized as policy optimization in a K-armed bandit problem—the agent generates a response (action), receives a reward, and accumulates history $\{(\mathbf{x}_1, r_1), \ldots, (\mathbf{x}_t, r_t)\}$ in context to optimize subsequent behavior.

**Core Idea**: The self-attention mechanism of Transformers has a natural inductive bias for simulating FTRL-based policy optimization, and after sufficient pretraining, it can perform policy optimization in context.

## Method

### Overall Architecture
The ICPO framework: given a problem → the model generates response $\mathbf{x}_t$ → receives reward $r_t$ (self-evaluated or external) → appends $(\mathbf{x}_t, r_t)$ to the context history → the model generates an improved response $\mathbf{x}_{t+1}$ based on the updated history → repeat.

Theoretical analysis is conducted on linear self-attention (LSA) Transformers, proving that they can exactly simulate an FTRL-based policy optimization algorithm.

### Key Designs

1. **Fisher-weighted logit-matching pretraining objective**:

    - Function: A novel supervised pretraining loss that enables the Transformer to perform in-context policy optimization.
    - Mechanism: The loss is $\mathcal{L}(\theta) = \frac{1}{2} \mathbb{E}_{\tau \in \mathcal{D}} [\sum_t \| \text{Proj}(\hat{\mathbf{s}}_{\tau,t+1} - \mathbf{s}_{\tau,t+1}^{\text{PO}}) \|_{\Gamma}^2]$, where $\Gamma$ is the Fisher information matrix of the policy and Proj projects out the constant bias (which does not affect the softmax policy).
    - Design Motivation: Fisher weighting makes the loss proportional to the KL divergence (Theorem 4.1), explaining why a standard KL loss suffices to enable Transformers to learn self-reflection.

2. **Population Equivalence and finite-sample guarantees**:

    - Function: Proves that after sufficient pretraining, a single-layer LSA can exactly simulate the target policy optimization algorithm.
    - Mechanism: Theorem 4.2 shows that the optimal parameters $\theta^*$ enable the LSA to exactly reproduce policy optimization behavior over all possible histories; Theorem 4.3 provides a sample complexity of $\tilde{O}(N^2 K / c_\lambda^2)$.
    - Design Motivation: Provides theoretical evidence for LLMs' in-context policy optimization capability—a single attention layer suffices.

3. **Robustness guarantee (Reward Shock Stability)**:

    - Function: Analyzes the stability of the ICPO loop against a single reward perturbation.
    - Mechanism: Theorem 4.8 proves that when the learning rate $\eta_t = c/t$ is sufficiently small, the effect of a one-time reward perturbation $\delta_r$ on the policy decays to zero over time: $\mathbb{E}[\|\Delta \hat{\mathbf{p}}_{t+1}^s\|_2] \leq \frac{a(1+C_b)}{s} (\frac{t}{s})^{b-1} |\delta_r|$.
    - Design Motivation: Provides theoretical support for using noisy self-evaluation rewards.

4. **ME-ICPO practical algorithm**:

    - Function: Translates the theoretical framework into a practical test-time inference algorithm.
    - Mechanism: (1) Sample $k$ candidate responses per round; (2) use Majority Vote to determine the self-evaluation reward $r_j^{(t)} = \mathbb{1}[a_j^{(t)} = \hat{a}_t]$; (3) summarize and compress the CoT; (4) **minimum-entropy selection**—select the candidate that minimizes the entropy of subsequent responses to add to the context.
    - Design Motivation: Minimum-entropy selection follows the "pessimism" principle of offline RL—choosing the direction the agent is most confident about, avoiding being misled by noisy rewards.

### Loss & Training
ME-ICPO involves no parameter updates at test time—it is a pure inference-time algorithm. Core strategy:
- Sample $k=16$ responses per round
- Majority vote as reward estimate
- CoT summarization to compress context length
- Minimum-entropy selection for robustness
- Output the final response after $n$ iterations

## Key Experimental Results

### Main Results

| Model | Benchmark | Base Mean@16 | w/ ME-ICPO Mean@16 | Gain |
|-------|-----------|-------------|-------------------|------|
| Qwen2.5-Math-7B | AIME 2024 | 11.04 | **30.42** | +19.38 |
| Qwen2.5-Math-7B | AMC | 41.42 | **47.06** | +5.64 |
| Qwen2.5-Math-7B | MATH-L5 | 30.58 | **38.71** | +8.13 |
| Qwen2.5-Math-1.5B | AIME 2024 | 6.46 | **9.79** | +3.31 |
| Qwen2.5-Math-1.5B | MATH-L1 | 49.27 | **57.06** | +12.38 |

Gains are most pronounced on AIME 2024: +19.38 for the 7B model and +3.31 for the 1.5B model. ME-ICPO's Mean@16 can exceed the baseline model's Maj@k upper bound.

### Ablation Study

| Configuration | AIME 2024 Accuracy (%) |
|---------------|----------------------|
| w/o Reward | 19.30 |
| w/o Entropy | 5.77 |
| w/o Entropy & Reward | 6.21 |
| **ME-ICPO (full)** | **30.05** |
| ME-ICPO Oracle | 38.19 |

### Key Findings
- **Minimum-entropy selection is the most critical component**: removing it causes accuracy to plummet from 30.05% to 5.77%, worse than doing nothing (6.21%)—indicating that without a principled selection strategy, random context is actually harmful.
- **Reward signal also matters**: removing it reduces accuracy from 30.05% to 19.30%.
- **Theoretical validation experiments**: the policy-matching error of LSA converges rapidly to numerical precision, and the effect of a single reward shock indeed decays over time.
- ME-ICPO's Mean@16 can surpass the baseline's Maj@k upper bound—indicating that in-context policy optimization learns information beyond simple voting.

## Highlights & Insights
- **Closed-loop theory-to-practice**: Starting from a theoretical analysis of linear self-attention, the paper derives practical algorithm design principles (reward guidance + minimum-entropy selection) and validates them on real LLMs—a complete research loop.
- **Insight behind minimum-entropy selection**: Rather than selecting the highest-reward candidate, the algorithm selects the candidate toward which the model is most confident. This is especially important when self-evaluation rewards are noisy—a high reward may be coincidental, but low entropy reflects a stable consensus across the model's outputs.
- **Single-layer sufficiency**: Unlike Lin et al. (2023), which requires $O(\sqrt{T})$ layers, ICPO requires only a single-layer LSA and does not need more layers as context length grows—a result more aligned with the long-context setting of real LLMs.

## Limitations & Future Work
- The theoretical analysis is based on linear self-attention and linear bandit assumptions, which differ substantially from real LLMs and mathematical reasoning problems.
- ME-ICPO samples 16 candidate responses per round, and the computational overhead of multi-round iteration remains considerable.
- Self-evaluation rewards are based on Majority Vote, which may produce incorrect signals when the model makes systematic errors.
- Experiments are limited to mathematical reasoning tasks; code generation, logical reasoning, and other domains have not been evaluated.
- CoT summarization may discard critical reasoning step information.

## Related Work & Insights
- **vs. Self-Refine/Reflexion**: These methods perform self-reflection via natural language feedback but lack a theoretical explanation for their effectiveness; ICPO provides a theoretical foundation from a policy optimization perspective.
- **vs. Tree-of-Thoughts**: ToT searches at every step, whereas ME-ICPO optimizes the entire CoT per round—coarser-grained but more computationally efficient.
- **vs. TTRL**: TTRL performs gradient updates at test time; ME-ICPO is purely in-context with no parameter updates—substantially more lightweight.
- **vs. Best-of-N**: BoN selects the single best response; ME-ICPO leverages multi-round iteration to accumulate contextual information and progressively improve—theoretically superior.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First theoretical analysis of LLM self-reflection from a policy optimization perspective; minimum-entropy selection is a novel design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Theoretical validation is thorough, but LLM experiments are limited to mathematical tasks and the Qwen model family.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous, though the transition from theory to practice could be tighter.
- Value: ⭐⭐⭐⭐ Provides a theoretical foundation for test-time scaling; the minimum-entropy selection strategy has practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning](scaf-grpo_scaffolded_group_relative_policy_optimization_for_enhancing_llm_reason.md)
- [\[ICML 2026\] On the Provable Suboptimality of Momentum SGD in Nonstationary Stochastic Optimization](../../ICML2026/optimization/on_the_provable_suboptimality_of_momentum_sgd_in_nonstationary_stochastic_optimi.md)
- [\[ICLR 2026\] Test-Time Meta-Adaptation with Self-Synthesis](test-time_meta-adaptation_with_self-synthesis.md)
- [\[ICLR 2026\] COLD-Steer: Steering Large Language Models via In-Context One-step Learning Dynamics](cold-steer_steering_large_language_models_via_in-context_one-step_learning_dynam.md)
- [\[ICML 2026\] Test time training enhances in-context learning of nonlinear functions](../../ICML2026/optimization/test_time_training_enhances_in-context_learning_of_nonlinear_functions.md)

</div>

<!-- RELATED:END -->
