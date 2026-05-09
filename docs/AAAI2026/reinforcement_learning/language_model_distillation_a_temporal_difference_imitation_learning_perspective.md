---
title: >-
  [Paper Note] Language Model Distillation: A Temporal Difference Imitation Learning Perspective
description: >-
  [AAAI 2026][Reinforcement Learning][knowledge distillation] This paper revisits language model distillation from an imitation learning / inverse reinforcement learning perspective. It exploits the sparsity of teacher output distributions (top-p tokens concentrate over 96% of probability mass) to construct a top-p MDP for temporal difference (TD) learning, proves that the optimal policy in the reduced action space admits a bounded suboptimality guarantee, and demonstrates that the resulting Bellman Distill method — built on the IQL algorithm — outperforms existing distillation methods across multiple model families.
tags:
  - AAAI 2026
  - Reinforcement Learning
  - knowledge distillation
  - imitation learning
  - temporal difference learning
  - top-p action space
  - inverse reinforcement learning
date: 2026-05-08
content_hash: 619ac080eb52359c
---

# Language Model Distillation: A Temporal Difference Imitation Learning Perspective

**Conference**: AAAI 2026
**arXiv**: [2505.20335](https://arxiv.org/abs/2505.20335)
**Code**: N/A
**Area**: Reinforcement Learning
**Keywords**: knowledge distillation, imitation learning, temporal difference learning, top-p action space, inverse reinforcement learning

## TL;DR

This paper revisits language model distillation from an imitation learning / inverse reinforcement learning perspective. It exploits the sparsity of teacher output distributions (top-p tokens concentrate over 96% of probability mass) to construct a top-p MDP for temporal difference (TD) learning, proves that the optimal policy in the reduced action space admits a bounded suboptimality guarantee, and demonstrates that the resulting Bellman Distill method — built on the IQL algorithm — outperforms existing distillation methods across multiple model families.

## Background & Motivation

- **State of the Field**: LLM distillation has become standard practice for compressing large-capacity teacher models into efficient student models. Most existing distillation methods can be viewed as Behavior Cloning (BC) — directly imitating the teacher's output distribution at each step.

- **Limitations of Prior Work**: BC suffers from **compounding errors**, also known as **exposure bias** in autoregressive models — during training the student observes teacher-generated contexts, whereas at inference it operates on its own (potentially erroneous) contexts, causing errors to accumulate exponentially with sequence length.

- **Root Cause**: The imitation learning literature offers rich tools to address BC's shortcomings, particularly TD learning, which mitigates compounding errors by estimating the long-term impact of actions given a state. However, **directly applying TD learning to LLMs faces the challenge of an enormous action space** — vocabulary size $|\mathcal{V}|$ typically ranges from tens of thousands to over a hundred thousand tokens.

- **Paper Goals**: To identify and exploit structure specific to the distillation setting. The key observation is that LLM output distributions are highly sparse — **the top-50 tokens account for 96% of probability mass, and the top-7 tokens contribute ≥90%** — motivating the idea that only a small set of high-probability candidate actions needs to be considered in TD learning.

- **Starting Point**: Does a distillation-specific exploitable structure exist? The answer is the sparsity of the teacher distribution — a property absent in pure IL settings, where access to the full expert policy distribution is generally unavailable.

## Method

### Overall Architecture

1. Define the MDP for language model generation: state = prompt + generated sequence so far; action = next token.
2. Exploit teacher distribution sparsity to define the top-p candidate set.
3. Construct the top-p MDP and prove bounded suboptimality.
4. Implement the IQL algorithm on the top-p MDP for distillation.

### Key Designs

#### 1. **Top-p Candidate Set and Top-p MDP**

Given teacher policy $\pi^\star$ and state $s$, the top-p candidate set is defined as the subset of tokens whose cumulative probability mass reaches $p$:

$$\mathcal{A}_p^\star(s) = \{a_i : \sum_i \pi^\star(a_i|s) = p; \pi^\star(a_i|s) \geq \pi^\star(a_j|s), \forall i < j\} \subseteq \mathcal{A}$$

The top-p MDP is defined as $\mathcal{M}_p = (\mathcal{S}, \mathcal{A}_p^\star, \mathbb{P}, r, \gamma)$, reducing the action space from the full vocabulary $\mathcal{V}$ to $\mathcal{A}_p^\star$.

#### 2. **Theoretical Suboptimality Guarantee**

A core contribution of this paper is proving that the gap between the optimal policy learned in the reduced action space and the original optimal policy is controllable.

The **Top-p Bellman operator** is defined as:

$$(B_p^\pi \bar{Q})(s,a) = r(s,a) + \gamma \mathbb{E}_{a' \sim \text{proj}_p(\pi)}[\bar{Q}(s',a') - \log \text{proj}_p(\pi)(a'|s')]$$

where $\text{proj}_p(\pi)$ is the projection of policy $\pi$ onto $\mathcal{A}_p^\star$ (via renormalization).

**Proposition 1** (Contraction): $\mathcal{B}_p^\pi$ is a contraction mapping under the supported $\infty$-norm.

**Proposition 2** (Suboptimality Bound): $\|Q^\star - \bar{Q}^{\text{proj}_p \pi^\star}\|_{\infty, \mathcal{A}_p^\star} \leq \kappa(p)$, where $\kappa(p) = -\frac{\gamma}{1-\gamma}\log p$.

**Proposition 3** (Sandwich Condition): $\bar{Q}^{\text{proj}_p \pi^\star}(s,a) \leq \bar{Q}_p^\star(s,a) \leq Q^\star(s,a)$

**Proposition 4** (Bounded Suboptimality, Main Theorem):

$$\|Q^\star - \bar{Q}_p^\star\|_{\infty, \mathcal{A}_p^\star} \leq \kappa(p) = -\frac{\gamma}{1-\gamma}\log p$$

At $p=0.8$, $\kappa(0.8) \approx 0.22 \cdot \gamma/(1-\gamma)$, indicating small suboptimality. This justifies running IRL algorithms on a substantially reduced action space.

#### 3. **Bellman Distill via IQL**

IQL (Inverse soft Q-Learning) is adopted as the base algorithm.

**Q-function masking**: Only Q-values within the top-p set are updated via the top-p mask $\mathcal{F}_p^\star$:

$$(\mathcal{F}_p^\star Q)(s,a) = \begin{cases} Q(s,a) & \text{if } a \in \mathcal{A}_p^\star \\ -\infty & \text{otherwise} \end{cases}$$

**Policy projection**: $(\text{proj}_p \pi_Q)(a|s) = \exp Q(s,a) / \sum_{\mathcal{A}_p^\star} \exp Q(s,a)$

The final objective combines Q-masking and policy projection:

$$\max_Q \mathcal{J}^\star(Q) = \mathbb{E}_{\rho^\star}[\phi((\mathcal{F}_p^\star Q)(s,a) - \gamma V^{\text{proj}_p \pi_Q}(s'))] - \mathbb{E}_{\rho^\star}[V^{\text{proj}_p \pi_Q}(s) - \gamma V^{\text{proj}_p \pi_Q}(s')]$$

**Key implementation details**:
- Student model logits serve directly as Q-values (up to a constant shift), so a single model functions simultaneously as Q-function and policy.
- $\chi^2$ regularization is applied: $\phi(x) = x - x^2/4\alpha$, $\alpha=0.1$.
- Q-value clipping: $Q_{\min} = -10$ for numerical stability.
- Offline training: a dataset is first generated from the teacher, then used to train via Bellman Distill (BD).

### Loss & Training

- A language modeling objective $\mathcal{J}_{PT}$ is jointly maximized to preserve NLP benchmark performance.
- Two-phase training: Phase 1 SFT initialization → Phase 2 BD distillation.
- Batch size 64, learning rate 5e-6, $p=0.8$ ($p=0.5$ for OPT 1.3B).
- Discount factor $\gamma=0.99$.
- The teacher generates 8 responses per query.
- Training is conducted on 4× A40 GPUs.

## Key Experimental Results

### Main Results

**Rouge-L scores (mean over 5 random seeds)**:

| Method | GPT-2 (1.5B→120M) | | | OPT (6.7B→125M) | | | Qwen-2.5 (3B→0.5B) | | |
|--------|-------|----------|--------|-------|----------|--------|-------|----------|--------|
| | Dolly | SelfInst | Vicuna | Dolly | SelfInst | Vicuna | Dolly | SelfInst | Vicuna |
| SFT | 23.2 | 9.9 | 14.3 | 23.2 | 9.9 | 14.3 | 24.5 | 16.5 | 17.6 |
| KD | 22.8 | 10.8 | 13.4 | 21.9 | 9.7 | 14.0 | 24.4 | 14.7 | 17.8 |
| SeqKD | 22.7 | 10.1 | 14.3 | 22.0 | 10.1 | 13.7 | 24.7 | 15.3 | 17.4 |
| MiniLLM | 24.6 | 13.2 | 16.9 | 23.8 | 10.2 | 15.3 | 26.7 | 19.1 | 20.5 |
| **BD (Ours)** | **24.7** | **13.3** | 16.2 | **25.1** | **11.4** | 14.8 | **27.8** | **19.5** | **20.7** |

### Ablation Study

**Effect of top-p value**:

| p | GPT-2 340M | | | OPT 125M | | | Qwen-2.5 0.5B | | |
|---|----------|----------|--------|----------|----------|--------|----------|----------|--------|
| | Dolly | SelfInst | Vicuna | Dolly | SelfInst | Vicuna | Dolly | SelfInst | Vicuna |
| 1.0 (no mask) | 24.9 | 15.2 | 15.7 | 23.6 | 10.5 | 14.2 | 26.6 | 18.1 | 18.6 |
| 0.8 | 25.8(+0.9) | 15.4(+0.2) | 16.5(+0.8) | 25.1(+1.5) | 11.4(+0.9) | 14.8(+0.6) | 27.8(+1.2) | 19.5(+1.4) | 20.7(+2.1) |
| 0.5 | 25.6(+0.7) | 15.5(+0.3) | 16.1(+0.4) | 24.4(+0.8) | 11.0(+0.5) | 14.7(+0.5) | 27.6(+1.0) | 19.2(+1.1) | 20.2(+1.6) |

**Training time comparison**:

| Model | MiniLLM | BD (Ours) | Speedup |
|-------|---------|-----------|---------|
| GPT-2 1.5B→340M | 11.2h | 1.3h | 8.6× |
| OPT 6.7B→350M | 12.8h | 3.5h | 3.7× |
| Qwen-2.5 3B→0.5B | 10.7h | 0.8h | 13.4× |

**$\chi^2$ regularization ablation**:

| Config | GPT-2 760M | | | Qwen-2.5 0.5B | | |
|--------|----------|----------|--------|----------|----------|--------|
| | Dolly | SelfInst | Vicuna | Dolly | SelfInst | Vicuna |
| w/ $\chi^2$ | 26.2 | 16.1 | 17.3 | 27.8 | 19.5 | 20.7 |
| w/o $\chi^2$ | 25.9 | 15.7 | 17.1 | 27.4 | 19.2 | 20.3 |

### Key Findings

1. **Top-p masking is consistently effective**: At $p=0.8$, significant improvements over $p=1.0$ (no mask) are observed across all three model families, with a maximum gain of 2.1 Rouge-L points on Qwen-2.5. This directly validates the value of exploiting teacher distribution sparsity.

2. **Efficiency advantage of offline training**: BD reaches optimal validation performance 3.7–13.4× faster than the online method MiniLLM, by avoiding costly online autoregressive generation.

3. **Win rate under GPT-4o-mini evaluation**: BD consistently achieves higher win rates against KD, SeqKD, and MiniLLM baselines, indicating that generation quality improvements extend beyond Rouge-L.

4. **Scaling consistency**: Rouge-L improves consistently as student model size increases, demonstrating favorable scalability.

5. **$\chi^2$ regularization is effective**: Incorporating $\chi^2$ regularization yields consistent gains of 0.2–0.5 across all settings.

## Highlights & Insights

- **Value of perspective shift**: Recasting distillation as an IL/IRL problem naturally introduces the TD learning toolkit to address BC's compounding error problem — an elegant theoretical contribution.
- **Exploiting LLM-specific structure**: Teacher distribution sparsity is an LLM-specific structural property; converting it into a theoretical guarantee for action space reduction bridges RL theory and LLM practice.
- **Practical significance of the suboptimality bound**: $\kappa(p) = -\gamma\log p/(1-\gamma)$ provides an explicit trade-off — enabling a quantifiable decision on the degree of action space reduction.
- **Clever design of Q-values as logits**: Student model logits serve directly as Q-values, requiring no additional value network — a single model simultaneously serves as both policy and value function.

## Limitations & Future Work

- **Shared vocabulary assumption**: Teacher and student must share a vocabulary for distribution matching, limiting cross-architecture distillation.
- **Performance ceiling of offline training**: While efficient, online training could in principle achieve higher final performance, as the distribution shift problem is not fully resolved.
- **Limited task diversity**: Evaluation is restricted to instruction-following tasks; more complex settings such as code generation and mathematical reasoning are not covered.
- **MiniLLM remains competitive on GPT-2**: BD's advantage over the GPT-2 family is less pronounced than over OPT and Qwen families.
- **Hyperparameter sensitivity**: Performance varies with the choice of $p$ ($p=0.5$ vs. $p=0.8$), and no adaptive strategy for selecting $p$ is proposed.
- **Theory–practice gap**: The theoretical analysis assumes a tabular setting; contraction properties and related guarantees may not hold strictly for parameterized models.

## Related Work & Insights

- SeqKD, KD, and MiniLLM are unified under the IL/IRL framework as offline off-policy BC, online mixed-policy BC, and policy gradient, respectively, providing a clean taxonomy.
- IQL is chosen because it reparameterizes the saddle-point IRL objective as a simple maximization, avoiding complex min-max optimization.
- The top-p candidate set is conceptually aligned with nucleus sampling, but is applied during training rather than inference.
- The theoretical framework is general — any soft IRL algorithm can be adapted to the reduced action space via top-p projection.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The IL perspective combined with the top-p MDP theoretical framework constitutes a genuinely novel contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three model families with multiple ablations, though task diversity is limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are rigorous, motivation is clearly articulated, and the IL/IRL background is well presented.
- **Value**: ⭐⭐⭐⭐ — Provides a new theoretical framework and practical approach for LLM distillation.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Temporal-Difference Variational Continual Learning](../../NeurIPS2025/reinforcement_learning/temporal-difference_variational_continual_learning.md)
- [\[ICLR 2026\] Model Predictive Adversarial Imitation Learning for Planning from Observation](../../ICLR2026/reinforcement_learning/model_predictive_adversarial_imitation_learning_for_planning_from_observation.md)
- [\[AAAI 2026\] Reasoning with Exploration: An Entropy Perspective](reasoning_with_exploration_an_entropy_perspective.md)
- [\[ICLR 2026\] The Sample Complexity of Online Reinforcement Learning: A Multi-Model Perspective](../../ICLR2026/reinforcement_learning/the_sample_complexity_of_online_reinforcement_learning_a_multi-model_perspective.md)
- [\[AAAI 2026\] Do It for HER: First-Order Temporal Logic Reward Specification in Reinforcement Learning](do_it_for_her_first-order_temporal_logic_reward_specification_in_reinforcement_l.md)

<!-- RELATED:END -->
