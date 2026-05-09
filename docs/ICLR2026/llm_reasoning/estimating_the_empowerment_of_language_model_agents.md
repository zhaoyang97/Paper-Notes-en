---
title: >-
  [Paper Note] Estimating the Empowerment of Language Model Agents
description: >-
  [ICLR 2026][LLM Reasoning][empowerment] This paper proposes EELMA, an algorithm that leverages *empowerment* from information theory — defined as the mutual information between an agent's actions and future states — as a goal-agnostic capability metric for LM agents. EELMA achieves strong correlation with task performance ($r=0.83$–$0.94$) in both language games and real-world web navigation scenarios, and can be applied to open-ended agent monitoring and safety evaluation.
tags:
  - ICLR 2026
  - LLM Reasoning
  - empowerment
  - information theory
  - mutual information
  - LM agents
  - goal-agnostic evaluation
  - InfoNCE
  - WebArena
date: 2026-05-08
content_hash: 4076e36f8b693257
---

# Estimating the Empowerment of Language Model Agents

**Conference**: ICLR 2026
**arXiv**: [2509.22504](https://arxiv.org/abs/2509.22504)
**Code**: [GitHub](https://github.com/Jinyeop3110/EELMA)
**Area**: LLM Reasoning
**Keywords**: empowerment, information theory, mutual information, LM agents, goal-agnostic evaluation, InfoNCE, WebArena

## TL;DR

This paper proposes EELMA, an algorithm that leverages *empowerment* from information theory — defined as the mutual information between an agent's actions and future states — as a goal-agnostic capability metric for LM agents. EELMA achieves strong correlation with task performance ($r=0.83$–$0.94$) in both language games and real-world web navigation scenarios, and can be applied to open-ended agent monitoring and safety evaluation.

## Background & Motivation

- **Background**: Current LM agent evaluation relies primarily on goal-centric benchmarks, which require extensive manual task design, are costly to scale, and are blind to capability gains outside their coverage — posing risks for AI safety.
- **Limitations of Prior Work**: As LM agents increasingly engage in long-horizon, multi-turn interactions via tools such as search engines, APIs, and operating systems, milestone-based evaluation methods fail to capture agents' true capabilities in open-ended environments.
- **Core Idea**: Empowerment in information theory measures an agent's influence over future states and is theoretically related to a lower bound on expected returns under arbitrary random goals, making it a natural candidate for a goal-agnostic capability metric.
- **Key Challenge**: Classical empowerment estimation methods are computationally prohibitive and cannot be applied directly in high-dimensional text spaces, necessitating a new scalable algorithm.

## Method

### Overall Architecture

EELMA (Estimating Empowerment of Language Model Agents) is built on the standard MDP framework $(\mathcal{S}, \mathcal{A}, T, R, \gamma)$, modeling LM agent text interactions as state-action sequences and quantifying empowerment via variational mutual information estimation.

### Key Designs

**1. Effective Empowerment Definition**

A future state random variable $s_*$ is introduced (with step count $\tau \sim \text{Geom}(1-\gamma)$), and effective empowerment is defined as the average mutual information between actions and future states:

$$\mathcal{E}(\pi_{LM}) \triangleq \mathbb{E}_{s_t, a_t, s_*}\left[\sum_{t=0}^{\infty} \frac{\gamma^t}{1-\gamma} \log \frac{P(s_{t+\tau}=s_* \mid s_t, a_t)}{P(s_{t+\tau}=s_* \mid s_t)}\right]$$

State-conditioned empowerment $\mathcal{E}(s, \pi_{LM})$ and state-action-conditioned empowerment $\mathcal{E}(s, a, \pi_{LM})$ are further defined to identify high-impact states and actions.

**2. Text Embedding and Projection**

Tuples $(s_t^i, a_t^i, s_*^i)$ are sampled from multi-turn trajectories $\{(s_t^i, a_t^i)\}_{t=1}^{T_i}$. A pretrained embedding model (e.g., Jina Embeddings) paired with a differentiable MLP (parameters $\theta$) maps text to compact embeddings $(z_{s,t}^i, z_{a,t}^i, z_{s_*,t}^i)$.

**3. InfoNCE Mutual Information Estimation**

Two neural encoders $\phi$ (encoding current state/action) and $\psi$ (encoding future states) are trained via a contrastive InfoNCE loss for variational mutual information estimation:

$$I_{\text{NCE}}^{\text{State-action}} \geq \mathbb{E}\left[\log \frac{e^{\phi(z_{s,t}^i, z_{a,t}^i)^\top \psi(z_{s,*}^i)}}{\frac{1}{K}\sum_j e^{\phi(z_{s,t}^i, z_{a,t}^i)^\top \psi(z_{s,*}^j)}}\right]$$

Negative samples are drawn from target states in different trajectories. A state-only variant $I_{\text{NCE}}^{\text{State-only}}$ is computed in parallel.

**4. Empowerment Estimation Formula**

Using the learned representations, effective empowerment is estimated as the difference between two dot products:

$$\mathcal{E}(\pi_{LM}) = \mathbb{E}_{i,t,s^*}\left[\phi(z_{s,t}^i, z_{a,t}^i)^\top \psi(z_{s,*}^i) - \phi(z_{s,t}^i)^\top \psi(z_{s,*}^i)\right]$$

### Loss & Training

Both NCE objectives (state-action and state-only variants) are jointly maximized, simultaneously optimizing encoders $\phi, \psi$ and the embedding projection $\theta$.

### Theoretical Foundation

The relationship between empowerment and agent capability is theoretically grounded: under a uniform reward assumption, empowerment constitutes a lower bound on the average discounted return $\bar{r} = \mathbb{E}_R[\sum_{t=0}^{\infty} \gamma^t r_t]$. Higher empowerment implies that the agent retains greater future optionality across multi-turn interactions, enabling stronger performance on arbitrary tasks.

## Key Experimental Results

### Main Results

**Language Game Validation (Gridworld + Tower of Hanoi)**

| Environment | Method | State RMSE (bits) |
|---|---|---|
| Gridworld | EELMA (fixed format) | 0.056 |
| Gridworld | Direct estimation (NL) | 0.302 |
| Gridworld | EELMA (NL) | 0.048 |
| Tower of Hanoi | EELMA (fixed format) | 0.158 |
| Tower of Hanoi | Direct estimation (NL) | 0.438 |
| Tower of Hanoi | EELMA (NL) | 0.127 |

EELMA maintains robustness under natural language variants, with RMSE even lower than direct estimation under fixed-format conditions.

**WebArena Real-World Web Navigation**

| Domain | Empowerment–Return Correlation ($R_s$) |
|---|---|
| GitLab | 0.94 |
| Reddit | 0.83 |
| Shopping Admin | 0.87 |
| Shopping | Weak correlation (reasoning bottleneck) |

GPT-4o achieves the highest empowerment and discounted return; o3 attains comparable success rates to GPT-4o but incurs more steps, resulting in lower discounted returns.

### Ablation Study

**Effect of Agent Subsystems on Empowerment**

| Ablation Factor | Change in Empowerment |
|---|---|
| Remove CoT | Gridworld: −99% (0.19→0.01 bits); ToH: −65% (0.29→0.09 bits) |
| Memory length m0→m3 | ToH empowerment increases from ~0.3 to 0.4 bits |
| Model scale | Closed-source > open-source; larger > smaller |
| Environment complexity | Empowerment monotonically decreases as boxes increase from 4 to 7 |

### Key Findings

**Authentication Behavior Case Study**

| Action Type | Mean Empowerment (bits) | Significance |
|---|---|---|
| Valid password input | 0.210 | p < 0.001 |
| Invalid password input | −0.152 | — |
| Valid username input | 0.170 | p = 0.32 (n.s.) |
| Overall valid authentication | 0.365 | p < 0.001 |
| Overall invalid authentication | −0.127 | — |

Empowerment rises sharply upon successful authentication, reflecting the agent's acquisition of system administrative access — a "power-seeking" behavior. Password input proves more critical than username input, as a correct username paired with an incorrect password yields no gain in future-state reachability.

## Highlights & Insights

1. **Goal-Agnostic Capability Metric**: Empowerment is the first general-purpose LM agent capability metric that requires no goal annotation, and it correlates strongly with task performance across diverse environments.
2. **Safety Monitoring Value**: High-empowerment actions correspond to critical moments (e.g., gaining authentication), enabling detection of potential power-seeking behavior without requiring a pre-enumerated list of dangerous actions.
3. **Quantifying the Value of CoT**: This work provides the first information-theoretic quantification of CoT's contribution — removing CoT causes a 99% drop in empowerment, offering a theoretically grounded measure of agent reasoning capability.
4. **Linguistic Robustness**: EELMA outperforms direct estimation under natural language variants, which is critical for real-world deployment.
5. **Theory–Experiment Consistency**: The theoretical lower bound relationship of empowerment is empirically supported across settings ranging from toy environments to real-world scenarios.

## Limitations & Future Work

1. **Empowerment ≠ Power**: More options do not necessarily imply greater capability (analogous to "one strong offer beats many weak ones"), and the metric cannot capture indirect influence, such as effects on the beliefs and decisions of other agents.
2. **Weak Correlation in the Shopping Domain**: When the bottleneck lies in numerical reasoning rather than environmental control, the empowerment metric loses effectiveness.
3. **Computational Cost**: Multi-turn trajectory collection and embedding training are required; scaling to more complex open-ended environments remains to be explored.
4. **Text-Only Environments**: Although multimodal extensions are discussed, validation is currently limited to text-based interactions.

## Related Work & Insights

- **Complementarity with Benchmark Evaluation**: EELMA supplements rather than replaces traditional benchmark evaluation, and is particularly well-suited for detecting capability gains not covered by existing benchmarks.
- **Distinction from RL Intrinsic Motivation**: Prior work uses mutual information as an intrinsic training reward; this paper is the first to apply it for evaluating LM agents rather than training them.
- **Connection to AI Safety**: Turner et al.'s "power-seeking" theory predicts that optimal policies tend toward power acquisition; EELMA provides an actionable detection tool grounded in this principle.
- **Implications for Agent Design**: Empowerment analysis yields quantitative insights into the effects of CoT, memory length, and model scale on agent capability, offering guidance for agent architecture design.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First application of information-theoretic empowerment to LM agent evaluation; both the method and perspective are original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers controlled toy environments (with ground-truth validation) through real-world WebArena scenarios, with comprehensive ablations.
- **Value**: ⭐⭐⭐⭐ — Introduces a new paradigm for agent safety monitoring and capability evaluation, though deployment overhead requires further optimization.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical motivation is clearly articulated, figures are informative, and the authentication behavior case study is vivid and persuasive.
- **Overall**: ⭐⭐⭐⭐½ — A high-quality cross-disciplinary contribution that elegantly bridges information theory and LM agent evaluation.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Why is Your Language Model a Poor Implicit Reward Model?](why_is_your_language_model_a_poor_implicit_reward_model.md)
- [\[ICLR 2026\] Adaptive Social Learning via Mode Policy Optimization for Language Agents](adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)
- [\[ICLR 2026\] Scaling Generalist Data-Analytic Agents](scaling_generalist_data-analytic_agents.md)
- [\[ICLR 2026\] Agentified Assessment of Logical Reasoning Agents](agentified_assessment_of_logical_reasoning_agents.md)
- [\[ACL 2026\] JTPRO: A Joint Tool-Prompt Reflective Optimization Framework for Language Agents](../../ACL2026/llm_reasoning/jtpro_a_joint_tool-prompt_reflective_optimization_framework_for_language_agents.md)

<!-- RELATED:END -->
