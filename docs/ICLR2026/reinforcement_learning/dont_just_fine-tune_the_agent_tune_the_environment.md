---
title: >-
  [Paper Note] Don't Just Fine-tune the Agent, Tune the Environment
description: >-
  [ICLR 2026][Reinforcement Learning][Environment Tuning] This paper proposes the Environment Tuning training paradigm, which enables LLM agents to learn complex multi-turn tool use from scratch using only 400 training sam…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Environment Tuning"
  - "LLM Agent"
  - "Multi-turn Tool Use"
  - "Curriculum Learning"
date: 2026-05-08
content_hash: 77c5ffcd4cc3392e
---

# Don't Just Fine-tune the Agent, Tune the Environment

**Conference**: ICLR 2026
**arXiv**: [2510.10197](https://arxiv.org/abs/2510.10197)
**Code**: [https://github.com/inclusionAI/AWorld-RL/tree/main/EnvTuning](https://github.com/inclusionAI/AWorld-RL/tree/main/EnvTuning)
**Area**: Reinforcement Learning / LLM Agent
**Keywords**: Environment Tuning, LLM Agent, Multi-turn Tool Use, Curriculum Learning, Reinforcement Learning

## TL;DR

This paper proposes the Environment Tuning training paradigm, which enables LLM agents to learn complex multi-turn tool use from scratch using only 400 training samples, through structured curriculum learning, actionable environment-augmented feedback, and fine-grained progress rewards, while achieving strong out-of-distribution generalization.

## Background & Motivation

LLM agents face three core challenges in multi-turn tool use tasks: (1) **Extreme data scarcity** — the BFCL V3 multi-turn dataset contains only 800 samples, and high-quality human annotation is prohibitively costly; (2) **Environmental complexity** — 8 distinct domains and 84 tools require cross-domain API calls and sophisticated orchestration; (3) **Long interaction chains** — each task involves multiple rounds of user queries, where failure at any single turn causes overall task failure.

The root cause of existing approaches is as follows: SFT on synthetic trajectories enables rapid capability acquisition but is prone to overfitting and poor generalization; standard RL training suffers from a severe cold-start problem — agents with insufficient initial capability cannot explore effectively in vast action spaces, becoming trapped in a vicious cycle of low-quality rollouts, while long interaction chains further destabilize training and cause gradient explosion. Experiments show that single-stage RL directly applied to 400 samples collapses after approximately 70 steps, achieving only ~10% improvement.

The core idea of this paper is: **rather than imitating on static trajectories, let the agent learn directly within a carefully designed environment**. By "tuning the environment" rather than only "tuning the model," failed explorations are converted into valuable learning signals.

## Method

### Overall Architecture

Environment Tuning models multi-turn tool use as a POMDP and relies on three complementary mechanisms working in concert: (1) a structured curriculum that progressively increases learning difficulty; (2) actionable environment augmentation that transforms ambiguous error messages into pedagogically meaningful feedback; and (3) fine-grained progress rewards that provide dense per-turn learning signals. The input consists of problem instances and tool documentation; the output is a sequence of tool calls and natural language responses produced by the agent.

### Key Designs

1. **Four-Stage Structured Curriculum**: The learning process is divided into four progressive stages, following the principle of "learn syntax first, then reasoning, then remove the training wheels."

    - **Stage 1 (Syntax Mastery)**: Trains the agent to produce correctly formatted outputs and valid tool calls. A dedicated syntax reward is defined as $R_{\text{Stage1}} = I_{\text{tool}} \cdot (R_{\text{format}} + R_{\text{tool}})$, where $R_{\text{format}}$ measures XML format correctness and $R_{\text{tool}}$ measures tool call parameter correctness. This stage rapidly eliminates "empty turns" in which the agent produces irrelevant dialogue instead of tool calls.
    - **Stage 2 (Basic Reasoning + Augmented Feedback)**: Applies progress rewards and environment augmentation on the Base dataset to develop fundamental multi-turn reasoning capabilities.
    - **Stage 3 (Advanced Scenarios)**: Introduces the full training set, including complex scenarios such as missing parameters, missing functions, and long contexts; the agent learns to handle ambiguity and functional gaps with the aid of augmented feedback.
    - **Stage 4 (Alignment with Evaluation Environment)**: Disables environment augmentation, forcing the agent to rely on its own reasoning to handle standard error messages, thereby ensuring out-of-distribution generalization.
    - Stage transition criteria: both validation accuracy convergence and gradient norm stability must be satisfied before advancing to the next stage.

2. **Actionable Environment Augmentation**: Standard environment error messages are replaced with diagnostic and pedagogically informative feedback. The design motivation is to help the agent discover inter-tool dependencies and intra-tool constraint rules.

    - **Discovering inter-tool dependencies**: For example, when booking a flight using a city name instead of an airport code, the standard environment returns "No available route" (ambiguous), whereas the augmented environment returns "Invalid airport code[s]: destination airport 'Pinehaven'. Please use valid airport codes. You can use alternative tool to find the correct airport code for a city." (precise and actionable).
    - **Revealing intra-tool rules**: For example, when the `rm` command does not accept path arguments, the standard environment returns "No such file or directory" (misleading), while the augmented environment returns "Paths are not allowed. Specify only file/directory name in current directory." (directly correcting the misconception).

3. **Fine-Grained Progress Reward**: Replaces sparse binary terminal rewards with dense per-turn signals. The reward at turn $t$ is the product of environment state evaluation $r_t^{\text{state}}$ and execution result evaluation $r_t^{\text{exec}}$, and the total reward is the average success rate across all turns: $R_P = \frac{1}{T}\sum_{t=1}^{T} r_t^{\text{state}} \cdot r_t^{\text{exec}}$. This allows "nearly correct" and "completely incorrect" trajectories to be distinguished.

### Loss & Training

Training is based on a modified GRPO algorithm (PPO-style) with decoupled clipping and KL divergence penalty:

$$\mathcal{L}(\theta) = -\mathbb{E}_t\left[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\hat{A}_t)\right] + \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

Key hyperparameters: $\beta = 0.1$ (a larger KL coefficient is critical for preventing policy collapse), $\epsilon_{\text{low}} = 0.2$, $\epsilon_{\text{high}} = 0.28$. The advantage function is computed via within-group normalization (no critic network).

## Key Experimental Results

### Main Results

In-distribution results on the BFCL V3 multi-turn benchmark (using only 400 training samples):

| Model | Avg (%) | Base (%) | Miss Func (%) | Miss Param (%) | Long Context (%) |
|------|----------|----------|-------------|----------------|-----------------|
| GPT-4o | 51.00 | 59.00 | 54.00 | 41.00 | 50.00 |
| o3 | 49.25 | 47.00 | 55.00 | 47.00 | 48.00 |
| xLAM-2-8b (SFT SOTA) | 70.50 | 77.85 | 69.15 | 65.80 | 69.20 |
| Qwen2.5-7B + EnvTuning | 36.92 | 50.33 | 40.33 | 29.33 | 27.67 |
| watt-tool-8B + EnvTuning | **54.34** | - | - | - | - |
| ToolACE-2 + EnvTuning | 47.18 | - | - | - | - |

Out-of-distribution (OOD) generalization results (BFCL V4 + ACEBench):

| Model | Web Search (%) | ACEBench Agent (%) |
|------|---------------|-------------------|
| xLAM-2-8b (SFT) | 5.00 | 1.65 |
| ToolACE-2 | 9.00 | 8.34 |
| ToolACE-2 + EnvTuning | 14.00 | **15.00** |
| Llama + EnvTuning | **15.00** | 4.17 |

### Ablation Study

| Configuration | Avg Accuracy | Notes |
|------|----------|------|
| Qwen2.5-7B base | 7.00% | Direct inference |
| + Direct GRPO | ~17% | Single-stage RL without curriculum; limited gains |
| + Full EnvTuning | 36.92% | +19.5% over direct GRPO |
| w/o environment augmentation | Drop >20% | Large losses on Missing Param/Func |
| Binary reward replacing progress reward | Stage 3 training fails | Cannot learn on complex tasks at all |

### Key Findings

- **SFT severely overfits**: xLAM-2 achieves 70.50% in-distribution but collapses to 5.00% on OOD Web Search, confirming that trajectory imitation generalizes poorly.
- **Environment augmentation is critical in complex scenarios**: It yields more than 20% improvement on Missing Parameters and Missing Functions.
- **A larger KL coefficient is necessary**: $\beta = 0.1$ substantially outperforms the commonly used 0.001, effectively maintaining policy entropy and preventing premature collapse.
- **Single-stage RL suffers gradient explosion after ~70 steps**, whereas the four-stage curriculum maintains stable gradient norms throughout training.

## Highlights & Insights

- **Paradigm innovation**: Shifting from "imitation on trajectories" to "exploration within an environment" represents an important conceptual shift in LLM Agent training. No expert demonstration trajectories are required — only problem instances.
- **Exceptional data efficiency**: An agent trained on only 400 problem instances surpasses several proprietary models, which is highly significant for data-scarce settings.
- **Importance of environment engineering**: The quality of environment feedback directly determines RL exploration efficiency, offering a methodology for environment design — error messages should be actionable and diagnostic.
- **Compelling case studies**: Three scenarios — file system, travel API, and vehicle control — clearly illustrate how augmented feedback transforms "dead ends" into "learning opportunities."
- **Stage transition strategy** (validation accuracy convergence + gradient norm stability) provides useful engineering guidance for practical implementation.

## Limitations & Future Work

- **Environment augmentation requires manual design**: The current actionable feedback must be hand-crafted for each environment; automation is an important future direction.
- **In-distribution performance gap remains**: There is still a gap relative to SFT methods using large-scale synthetic data (e.g., xLAM-2 at 70.50%), indicating room for improvement in the trade-off between data volume and exploration efficiency.
- **Limited generalization scope**: OOD evaluation is primarily conducted on BFCL V4 and ACEBench; broader multi-modal agent scenarios remain unvalidated.
- **Validated only on 7–8B models**: Performance at larger scales is unknown, and whether curriculum design requires adjustment with model scale is unexplored.
- **Automatic determination of curriculum stage count and data allocation strategy** is a valuable direction for future research.

## Related Work & Insights

- The central finding of **SFT memorizes, RL generalizes** (Chu et al., 2025) is thoroughly validated in this work — OOD collapse of SFT is a general phenomenon.
- **Comparison with ReCall and ARTIST** highlights the limitations of direct RL in complex multi-turn environments, establishing curriculum learning as a necessary component.
- **The environment augmentation idea** is generalizable to other Agent RL domains: engineering environment feedback to guide exploration is more natural than modifying the reward function.
- **Insight**: In automated training pipelines for LLM Agents, environment design and reward engineering are equally important and warrant systematic investigation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TRACED: Transition-aware Regret Approximation with Co-learnability for Environment Design](traced_transition-aware_regret_approximation_with_co-learnability_for_environmen.md)
- [\[ACL 2026\] AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation](../../ACL2026/reinforcement_learning/aj-bench_benchmarking_agent-as-a-judge_for_environment-aware_evaluation.md)
- [\[ICLR 2026\] RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning](rewardmap_tackling_sparse_rewards_in_fine-grained_visual_reasoning_via_multi-sta.md)
- [\[ICLR 2026\] DiVE-k: Differential Visual Reasoning for Fine-grained Image Recognition](dive-k_differential_visual_reasoning_for_fine-grained_image_recognition.md)
- [\[ACL 2026\] ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning](../../ACL2026/reinforcement_learning/rerec_reasoning-augmented_llm-based_recommendation_assistant_via_reinforcement_f.md)

</div>

<!-- RELATED:END -->
