---
title: >-
  [Paper Note] DEPO: Dual-Efficiency Preference Optimization for LLM Agents
description: >-
  [AAAI 2026][LLM Agent][LLM Agent efficiency optimization] This paper proposes the concept of *dual-efficiency*, decomposing LLM agent efficiency into step-level (reducing tokens per step) and trajectory-level (reducing total number of steps) dimensions. Building on KTO, the authors introduce DEPO, which jointly optimizes efficiency and task performance by incorporating an efficiency bonus into the reward for desirable samples.
tags:
  - AAAI 2026
  - LLM Agent
  - LLM Agent efficiency optimization
  - preference optimization
  - KTO
  - dual-efficiency
  - reinforcement learning
date: 2026-05-08
content_hash: 3a423b842869adee
---

# DEPO: Dual-Efficiency Preference Optimization for LLM Agents

**Conference**: AAAI 2026
**arXiv**: [2511.15392](https://arxiv.org/abs/2511.15392)
**Code**: [https://opencausalab.github.io/DEPO](https://opencausalab.github.io/DEPO)
**Area**: Agent / LLM
**Keywords**: LLM Agent efficiency optimization, preference optimization, KTO, dual-efficiency, reinforcement learning

## TL;DR

This paper proposes the concept of *dual-efficiency*, decomposing LLM agent efficiency into step-level (reducing tokens per step) and trajectory-level (reducing total number of steps) dimensions. Building on KTO, the authors introduce DEPO, which jointly optimizes efficiency and task performance by incorporating an efficiency bonus into the reward for desirable samples.

## Background & Motivation

As LLM reasoning capabilities improve, chain-of-thought outputs grow increasingly lengthy, leading to inefficient agent–environment interactions. Existing efficiency research focuses primarily on token compression in single-turn responses, overlooking two key sources of overhead in agentic settings:
1. **Tokens generated per step**: overthinking produces verbose single-step responses (step-level inefficiency).
2. **Total steps to task completion**: imprecise reasoning requires more interaction steps (trajectory-level inefficiency).

Existing RL methods (PPO, DPO, GRPO, KTO) focus mainly on learning dynamics and performance improvement, with no explicit optimization of agent interaction efficiency.

## Core Problem

How can the per-step token count and total interaction steps of an LLM agent be simultaneously reduced without sacrificing task performance?

## Method

### Overall Architecture

A three-stage pipeline:
1. **MCTS Data Generation**: Monte Carlo Tree Search using DeepSeek-V3 to generate trajectories in ReAct format (Thought + Action).
2. **Behavior Cloning (BC) SFT**: Standard SFT on high-quality desirable trajectories to learn a base policy $\pi_{\text{BC}}$.
3. **DEPO Preference Optimization**: Efficiency-aware preference learning on top of the BC policy.

### Key Designs

**Dual-Efficiency Definition**:
- Step-level efficiency: minimize tokens generated per step.
- Trajectory-level efficiency: minimize total steps required to complete the task.

**Data Annotation and Filtering**:
- Desirable ($\mathcal{D}$): reward $r(\tau) \geq \kappa_0$ (BabyAI: $\geq 0.9$; Webshop: $= 1.0$).
- Undesirable ($\mathcal{U}$): $\kappa_2 \leq r(\tau) < \kappa_1$ (mid-quality range $[0.7, 0.9)$).
- Trajectories below the lower threshold are discarded to ensure a quality margin between desirable and undesirable sets.
- Additional step-count filtering: trajectories with $<7$ steps assigned to $\mathcal{D}$; $\geq 7$ steps to $\mathcal{U}$.
- GPT-4.1 mini is used to rephrase the Thought portion of desirable trajectories, reducing per-step token counts.

**Efficiency Bonus Design** (core contribution):

A parameter-free efficiency offset is added to the KTO implied reward:

$$r_\theta(\tau) = \log \frac{\pi_\theta(a_t | \tau_t)}{\pi_{\text{BC}}(a_t | \tau_t)} + b(\tau)$$

where the bonus $b(\tau)$ is defined as:

$$b(\tau) = \begin{cases} \frac{\alpha_1}{\bar{T}_{\text{token}}(\tau)} + \frac{\alpha_2}{T_{\text{step}}(\tau)}, & \text{if } \tau \in \mathcal{D} \\ 0, & \text{if } \tau \in \mathcal{U} \end{cases}$$

- $\bar{T}_{\text{token}}$: average tokens per step (larger → smaller bonus → penalizes verbosity).
- $T_{\text{step}}$: total number of steps (more steps → smaller bonus → penalizes inefficient trajectories).
- **The bonus is applied only to desirable samples**; ablation experiments confirm that applying penalties to undesirable samples is harmful.

### Loss & Training

Based on the KTO (Kahneman-Tversky Optimization) framework:

$$\mathcal{L}_{\text{KTO}}(\theta) = \mathbb{E}_{\tau \sim \mathcal{D}, \mathcal{U}} [\lambda(\tau) - v(\tau)]$$

The value function applies sigmoid transformations separately for desirable and undesirable samples:
- Desirable: $v(\tau) = \lambda_D \cdot \sigma(\beta(r_\theta(\tau) - z_0(\tau)))$
- Undesirable: $v(\tau) = \lambda_U \cdot \sigma(\beta(z_0(\tau) - r_\theta(\tau)))$

$z_0(\tau)$ is a KL-divergence regularization term between the current policy and the BC policy.

**Training Configuration**:
- LoRA fine-tuning; BC stage lr=1e-4, DEPO stage lr=2e-5, 3 epochs each.
- $\beta=0.2$, $\lambda_D = \lambda_U = 1$.
- Llama3.1-8B: $\alpha_1 = \alpha_2 = 3$; Qwen2.5-7B: $\alpha_1 = \alpha_2 = 2$.
- BabyAI: 512 desirable + 471 undesirable; Webshop: 1567 samples each.
- 8 × A800 80GB GPUs.

## Key Experimental Results

**Main Results (Table 1)**:

| Model | Metric | Webshop Succ.↑ | Webshop T@All↓ | BabyAI Succ.↑ | BabyAI T@All↓ |
|------|------|------|------|------|------|
| Llama3.1-8B-BC | baseline | 0.47 | 840 | 0.77 | 836 |
| + KTO | | 0.48 | 776 | 0.87 | 342 |
| **+ DEPO** | | **0.50** | **633** | **0.88** | **327** |
| Qwen2.5-7B-BC | baseline | 0.44 | 1014 | 0.47 | 2062 |
| + KTO | | 0.54 | 886 | 0.58 | 1199 |
| **+ DEPO** | | **0.56** | **726** | **0.75** | **893** |

**Efficiency Gains Summary** (relative to BC baseline):
- Token usage reduced by up to **60.9%** (Llama, BabyAI T@All).
- Step count reduced by up to **26.9%** (Qwen, BabyAI S@All vs. KTO).
- Task performance improved by up to **29.3%** (Qwen, BabyAI Succ. vs. BC).

**Generalization (Figure 2)**:
- On three OOD math benchmarks (GSM8K, MATH, SimulEq), models trained with DEPO achieve higher average accuracy with reduced token usage.
- Llama3.1-8B-BC+DEPO shows clear cross-domain efficiency gains; Qwen2.5-7B-BC+DEPO shows a slight token increase.

**Sample Efficiency (Figure 3)**:
- Using only 25% of training data (245 BabyAI / 783 Webshop samples), T@All efficiency improves by over 10%.
- With 100% data, T@All improves by nearly 60%.

### Ablation Study

1. **Joint tuning of $\alpha_1$ and $\alpha_2$ is optimal**: optimizing either dimension alone may improve that metric while degrading performance. Setting $\alpha_1=0, \alpha_2>0$ leads to a notable drop in Qwen's Succ. and Reward.
2. **Penalizing undesirable samples is counterproductive**: applying equal-strength penalties to undesirable trajectories causes Llama's T@All on BabyAI to increase by +46.5% and S@All by +39.4% (substantial efficiency degradation), with performance also declining.

## Highlights & Insights

1. **Clear conceptual framing**: decoupling agent efficiency into step-level and trajectory-level dimensions facilitates targeted optimization.
2. **Simple and elegant method**: a single efficiency bonus term added to the KTO reward requires no additional reward model, pairwise annotations, or on-policy sampling, making implementation straightforward and training stable.
3. **Efficiency and performance are not at odds**: rather than trading performance for efficiency, DEPO achieves both simultaneously (up to +29.3% task performance).
4. **Sample efficiency**: significant efficiency gains are achievable with only 25% of training data, making the approach suitable for data-scarce settings.
5. **Cross-domain generalization**: models trained on Webshop/BabyAI transfer to mathematical reasoning tasks.

## Limitations & Future Work

1. **Limited evaluation environments**: training and primary evaluation are conducted only on Webshop (online shopping) and BabyAI (grid world), both relatively simple; more complex real-world settings (e.g., web browsing, code execution) are absent.
2. **Data generation relies on strong external models**: MCTS search uses DeepSeek-V3 and rephrasing uses GPT-4.1 mini, incurring non-trivial data generation costs.
3. **Simplistic efficiency bonus design**: the reciprocal form $1/T$ does not account for task difficulty normalization — simple tasks naturally require fewer steps and should receive less reward, while complex tasks with many steps may be unfairly penalized.
4. **Insufficient comparison with other efficiency methods**: comparisons are limited to Token Budget (TB); direct comparisons with RL-based efficient reasoning methods such as L1 regularization and DAST are absent.
5. **Questionable OOD generalization evaluation**: generalization tests on math tasks report only accuracy and average token counts without step-count comparisons; Qwen's token count does not decrease substantially.
6. **No analysis of reasoning quality degradation**: compressing Thought content may reduce reasoning quality; the paper does not analyze changes in Thought quality.

## Related Work & Insights

| Method | Type | Efficiency Optimization | Requires Paired Data | Requires Reward Model | Online Sampling |
|------|------|------|------|------|------|
| ETO | Offline RL (DPO) | ✗ | ✓ contrastive pairs | ✗ | ✗ |
| DMPO | Offline RL (DPO) | ✗ (length normalization only) | ✓ | ✗ | ✗ |
| RAGen/StarPO | Online RL | ✗ | ✗ | ✗ | ✓ |
| GiGPO | Online RL | ✗ | ✗ | ✗ | ✓ |
| L1/DAST | RL + length penalty | step-level | method-dependent | method-dependent | method-dependent |
| **DEPO** | Offline RL (KTO) | **dual** | **✗** | **✗** | **✗** |

DEPO's advantages are: (1) simultaneous optimization of both efficiency dimensions; (2) KTO-based formulation requiring no paired data; (3) fully offline training.

The efficiency bonus design is transferable to other preference optimization frameworks such as DPO and GRPO. DEPO is complementary to the "thinking budget" line of research: Token Budget constrains the output ceiling, while DEPO optimizes the output distribution from the training side, and the two approaches can be combined. From a system design perspective, API call latency and cost scale linearly with step count in real deployments, suggesting that trajectory-level efficiency may matter more than step-level efficiency in practice. Finally, while the MCTS + rephrasing data construction pipeline is effective, it introduces a dependency on strong external models; self-play approaches for generating efficient data represent a valuable direction for future work.

## Rating (⭐ 1–5)

⭐⭐⭐ (3/5)

**Strengths**: The problem is clearly defined, the method is concise and practical, and the experiments provide reasonable coverage (ablation, generalization, and sample efficiency are all addressed), with substantial efficiency gains reported.

**Weaknesses**: The core technical contribution is limited — the method essentially adds a hand-crafted efficiency bonus to the KTO reward, which offers moderate technical depth. The evaluation environments are relatively simple (Webshop and BabyAI lack complexity), and comparisons with a broader set of efficiency optimization methods are missing. The efficiency bonus does not account for task difficulty normalization, which may cause instability on heterogeneous task sets. Overall, this is a solid but unsurprising piece of work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LPO: Towards Accurate GUI Agent Interaction via Location Preference Optimization](../../ACL2026/llm_agent/lpo_towards_accurate_gui_agent_interaction_via_location_preference_optimization.md)
- [\[ICLR 2026\] Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](../../ICLR2026/llm_agent/solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)
- [\[ICLR 2026\] Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization](../../ICLR2026/llm_agent/exploratory_memory-augmented_llm_agent_via_hybrid_on-_and_off-policy_optimizatio.md)
- [\[AAAI 2026\] COVR: Collaborative Optimization of VLMs and RL Agent for Visual-Based Control](covrcollaborative_optimization_of_vlms_and_rl_agent_for_visu.md)
- [\[ICLR 2026\] ToolTree: Efficient LLM Agent Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning](../../ICLR2026/llm_agent/tooltree_efficient_llm_agent_tool_planning_via_dual-feedback_monte_carlo_tree_se.md)

</div>

<!-- RELATED:END -->
