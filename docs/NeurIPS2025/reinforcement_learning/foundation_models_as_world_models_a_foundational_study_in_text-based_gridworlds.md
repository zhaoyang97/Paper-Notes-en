---
title: >-
  [Paper Note] Foundation Models as World Models: A Foundational Study in Text-Based GridWorlds
description: >-
  [NeurIPS 2025][Reinforcement Learning][Foundation Models] This paper systematically evaluates foundation models (LLMs) as zero-shot world models (FWM) and direct decision-making agents (FA) in text-based gridworlds, revealing complementary advantages of the two strategies in deterministic and stochastic environments.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Foundation Models
  - World Models
  - GridWorld
  - Large Language Models
  - Sample Efficiency
date: 2026-05-08
content_hash: d80d4f170962f7de
---

# Foundation Models as World Models: A Foundational Study in Text-Based GridWorlds

**Conference**: NeurIPS 2025
**arXiv**: [2509.15915](https://arxiv.org/abs/2509.15915)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: Foundation Models, World Models, GridWorld, Large Language Models, Sample Efficiency

## TL;DR

This paper systematically evaluates foundation models (LLMs) as zero-shot world models (FWM) and direct decision-making agents (FA) in text-based gridworlds, revealing complementary advantages of the two strategies in deterministic and stochastic environments.

## Background & Motivation

Reinforcement learning (RL) agents excel in complex games but suffer from extremely low sample efficiency, often requiring millions of interactions to learn effective policies. Foundation models (FMs), with their broad pretraining knowledge and reasoning capabilities, are natural candidates for improving sample efficiency. However, how to effectively integrate FMs into RL frameworks remains an open problem.

Prior work has largely used FMs as auxiliary components (e.g., reward shaping, high-level planners). This paper investigates more direct integration strategies:

- **Foundation World Model (FWM)**: Using an FM as a zero-shot simulator to generate pretraining interaction data for conventional RL agents.
- **Foundation Agent (FA)**: Directly using an FM as a low-level action selector to generate actions at each timestep.

To isolate the core simulation and reasoning capabilities of FMs, the authors design a suite of text-based gridworld environments as controlled experimental platforms, avoiding confounds from visual perception.

## Method

### Overall Architecture

The paper evaluates two primary strategies: the FWM approach treats the LLM as an environment simulator for pretraining RL agent policies, while the FA approach directly leverages LLM reasoning for decision-making. Evaluations are conducted under two gridworld settings:

1. **Deterministic + Fully Observable**: The reward location is fixed at the top-right corner and the agent knows the goal position.
2. **Stochastic + Partially Observable**: The reward location is randomly sampled each episode and the agent does not know the goal position.

The gridworld is defined as an $n \times n$ grid; the agent starts at the bottom-left corner $[0,0]$, the maximum number of steps is $2n^2$, and the action space is $\{up, down, left, right\}$.

### Key Designs

#### Foundation World Model (FWM)

The world model consists of a transition function and a reward function. Prompt templates are designed to delegate both functions to the LLM.

**Transition Function Design** — two prompt templates of varying complexity:

| Template | Description | Purpose |
|----------|-------------|---------|
| $T_{\text{minimal}}$ | Minimal description relying on LLM's prior knowledge of gridworlds | Tests lower bound of reasoning ability |
| $T$ | Detailed description of transition rules and mathematical constraints (boundary checks, coordinate directions) | Tests logical reasoning ability |

**Reward Function** — presents two possible outcomes and prompts the LLM to determine the relationship between the current position and the reward position: returning reward=1, terminal=True upon reaching the reward, and reward=0, terminal=False otherwise.

**Temperature Settings** — $\tau=0$ is used for deterministic functions to encourage deterministic outputs; $\tau=1.8$ is used for stochastic elements (e.g., reward position sampling) to introduce diversity.

#### Foundation Agent (FA)

Three prompt strategies with varying levels of guidance are designed:

| Strategy | Abbreviation | Description |
|----------|-------------|-------------|
| Action Only | AO | Provides current state and information; requires direct action output |
| Simple Plan | SP | Encourages reasoning about what to do before deciding on an action |
| Focused Plan | FP | Explicitly instructs the agent to use memory to determine the next target position and analyze the optimal action |

The three strategies test the LLM's ability to leverage reasoning and planning for decision-making under different levels of guidance. In the stochastic setting, the agent must use memory to track visited positions for systematic grid search.

### Loss & Training

**FWM + RL Integration**: Using GPT-4 with the $T+R$ prompt as the best FWM, RL agents are first pretrained on simulated interactions and then fine-tuned in the real environment:

- Deterministic setting: Pretraining a TRPO agent for 1,500 steps.
- Stochastic setting: Pretraining a RecurrentPPO agent (to handle partial observability) for 1 million steps.
- Evaluation: Evaluated every 125 steps using 1 episode, across 5 random seeds.

**Evaluated LLMs**: GPT-3.5, GPT-4, Gemma 2b, Gemma 7b, Gemini 1.0, Gemini 1.5 — covering different model types, scales, and generations. FA evaluation uses 100 independent episodes with $\tau=0$.

## Key Experimental Results

### Main Results (Simulation Accuracy)

$5 \times 5$ grid, $|\mathcal{S}| \cdot |\mathcal{A}| = 25 \times 4 = 100$ transition predictions:

| Model | $T$ Accuracy | $T_{\text{minimal}}$ Accuracy | $T+R$ Accuracy |
|-------|-------------|------------------------------|---------------|
| GPT-4 | **100%** | 97% | **100%** |
| Gemini 1.5 | **100%** | 95% | 99% |
| GPT-3.5 | 96% | 82% | 93% |
| Gemma 7b | 78% | 65% | 72% |
| Gemma 2b | 61% | 48% | 55% |

**Decision Performance Comparison** (Deterministic vs. Stochastic):

| Method | Deterministic | Stochastic |
|--------|--------------|-----------|
| GPT-4 FA (best prompt) | **1.0** (zero-shot optimal) | ~0.6 |
| Gemini 1.5 FA | **1.0** | ~0.5 |
| RL from scratch | 1.0 (requires hundreds of steps) | ~0.8 (requires many interactions) |
| **RL + FWM (GPT-4)** | **1.0** (optimal after pretraining) | **~0.9** (fast convergence) |
| Gemma 2b/7b FA | 0.0 | ~0.1 |

### Ablation Study

**Stochastic Element Simulation Distribution Quality** (1,000 random grid positions):
- Smaller models (Gemma 2b, GPT-3.5, Gemini 1.0): Cover all positions but with uneven distributions.
- Larger models (GPT-4, Gemini 1.5): Concentrate on a small number of positions with incomplete coverage.
- Conclusion: Higher output variance in smaller models is advantageous for simulating large sampling spaces, while larger models are better suited to simulating small sampling spaces.

**Prompt Strategy Ablation** (Stochastic setting):
- Smaller models: Marginal differences across AO/SP/FP; overall performance is low.
- Larger models: SP and FP significantly outperform AO, indicating that enhanced reasoning guidance helps larger models better leverage memory and planning.

### Key Findings

1. **FA dominates in deterministic environments**: FA directly leverages reasoning to solve tasks zero-shot, far outperforming RL agents that require hundreds of interactions.
2. **FWM + RL dominates in stochastic environments**: FA generally fails to systematically search the entire grid; FWM + RL substantially improves sample efficiency.
3. **Model capacity equals performance**: Improvements in LLM capability directly translate to better FWM and FA performance.
4. **FWM transfer is robust**: Even when the FWM cannot perfectly simulate stochastic distributions, pretrained policies transfer smoothly to the real environment.

## Highlights & Insights

- **Complementarity finding**: Simple deterministic tasks favor direct FA decision-making, while complex stochastic tasks favor the FWM + RL combination.
- **Boundary effects**: LLMs are most error-prone when simulating grid boundary transitions — they frequently misapply boundary constraints to other actions.
- **Unexpected advantage of smaller models**: When simulating stochastic elements over large sampling spaces, the higher output variance of smaller models proves beneficial.
- **Zero-shot pretraining paradigm is viable**: FWMs can provide sufficiently accurate simulation data for RL pretraining without any training.

## Limitations & Future Work

1. Experiments are limited to text-based gridworlds and do not extend to visual or embodied environments.
2. No comparison is made against world models trained from scratch; the focus is exclusively on zero-shot capabilities of pretrained FMs.
3. The FWM + RL pipeline adopts only a simple pretrain-then-finetune scheme, without exploring Dyna-style interactive architectures.
4. More recent LLMs (e.g., GPT-4o, Claude) are not evaluated.
5. Future work could explore combining FWM and FA, or using environment data to refine FWMs.

## Related Work & Insights

- **WorldCoder** (Tang et al., 2024): LLMs iteratively generate Python code to represent environment dynamics; a neuro-symbolic approach.
- **Genie 3** (DeepMind, 2025): A multimodal foundation world model capable of generating interactive 3D environments in real time.
- **RAFA** (Liu et al., 2023): LLMs imagine future scenarios and policies execute near-term actions.
- **MOTIF** (Klissarov et al., 2024): An intrinsic reward model based on LLM preferences to improve exploration under sparse rewards.
- Insights: FWMs can be combined with planning algorithms for online decision-making; future visual foundation world models could extend this framework to visual environments.

## Rating

- **Novelty**: ★★★★☆ — First systematic comparison of FWM and FA as two direct integration strategies.
- **Experimental Thoroughness**: ★★★☆☆ — Environments are overly simplistic, limited to gridworlds.
- **Value**: ★★★☆☆ — Provides foundational insights but remains distant from practical applications.
- **Writing Quality**: ★★★★☆ — Well-structured with sound experimental design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Exploration with Foundation Models: Capabilities, Limitations, and Hybrid Approaches](exploration_with_foundation_models_capabilities_limitations_and_hybrid_approache.md)
- [\[NeurIPS 2025\] Self-Improving Embodied Foundation Models](self-improving_embodied_foundation_models.md)
- [\[NeurIPS 2025\] Communicating Plans, Not Percepts: Scalable Multi-Agent Coordination with Embodied World Models](communicating_plans_not_percepts_scalable_multi-agent_coordination_with_embodied.md)
- [\[NeurIPS 2025\] Dynamics-Aligned Latent Imagination in Contextual World Models for Zero-Shot Generalization](dynamics-aligned_latent_imagination_in_contextual_world_models_for_zero-shot_gen.md)
- [\[NeurIPS 2025\] Checklists Are Better Than Reward Models For Aligning Language Models](checklists_are_better_than_reward_models_for_aligning_langua.md)

</div>

<!-- RELATED:END -->
