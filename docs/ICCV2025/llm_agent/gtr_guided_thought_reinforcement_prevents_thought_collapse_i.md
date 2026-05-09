---
title: >-
  [Paper Note] GTR: Guided Thought Reinforcement Prevents Thought Collapse in RL-based VLM Agent Training
description: >-
  [ICCV 2025][LLM Agent][Thought Collapse] This paper identifies the "thought collapse" phenomenon in RL-based VLM Agent training — where CoT reasoning rapidly degenerates into state-agnostic, templated thoughts that lead to ineffective actions — and proposes the GTR framework, which combines a VLM corrector for automatic thought correction (SFT) with PPO-based action optimization in a dual-objective training scheme, achieving 3–5× success rate improvements on the 24-Point Game and ALFWorld.
tags:
  - ICCV 2025
  - LLM Agent
  - Thought Collapse
  - CoT Reasoning
  - Process Guidance
  - PPO
  - VLM Agent
date: 2026-05-08
content_hash: 76dc3c2ab8e3a025
---

# GTR: Guided Thought Reinforcement Prevents Thought Collapse in RL-based VLM Agent Training

**Conference**: ICCV 2025
**arXiv**: [2503.08525](https://arxiv.org/abs/2503.08525)
**Code**: [GitHub](see paper)
**Area**: VLM Agent / Reinforcement Learning
**Keywords**: Thought Collapse, CoT Reasoning, Process Guidance, PPO, VLM Agent

## TL;DR

This paper identifies the "thought collapse" phenomenon in RL-based VLM Agent training — where CoT reasoning rapidly degenerates into state-agnostic, templated thoughts that lead to ineffective actions — and proposes the GTR framework, which combines a VLM corrector for automatic thought correction (SFT) with PPO-based action optimization in a dual-objective training scheme, achieving 3–5× success rate improvements on the 24-Point Game and ALFWorld.

## Background & Motivation

- **Background**: RLVR has successfully scaled CoT capabilities in LLM mathematical reasoning, but shows limited effectiveness in VLM Agent decision-making within visual environments.
- **Limitations of Prior Work**: Under RL training with only outcome rewards, the long-chain thought process is neither evaluated nor supervised, causing CoT reasoning to rapidly degenerate in complex tasks — manifesting as loss of diversity, state-agnostic reasoning, and incomplete inference.
- **Key Challenge**: RL rewards are based solely on action outcomes, whereas CoT thoughts form the foundation of decision-making yet remain entirely unsupervised.
- **Goal**: Prevent thought collapse in RL-based VLM Agent training.
- **Key Insight**: Process guidance — replacing coarse-grained numerical rewards with informative process supervision provided by an external VLM corrector.
- **Core Idea**: Automatically correcting collapsed thought trajectories via a VLM corrector, combined with DAgger to mitigate distribution shift, enabling joint optimization of both thoughts and actions.

## Method

### Overall Architecture

Within the RL training loop: the VLM Agent generates (thought, action) → the VLM corrector evaluates and revises the thought → the environment executes the action and returns a reward → thought tokens are trained with an SFT loss, and action tokens are trained with a PPO loss. DAgger aggregates all historical corrected data.

### Key Designs

**Design 1: VLM Corrector (Process Guidance)**
- **Function**: Evaluates the visual recognition accuracy and reasoning correctness of the Agent's thought at each step, and generates a revised thought.
- **Mechanism**: Leverages an off-the-shelf VLM (e.g., GPT-4o); given an observation and the Agent's thought output, the corrector produces a revised thought after evaluation. No manual annotation is required.
- **Design Motivation**: Numerical rewards (VLM-as-judge / length rewards) carry insufficient information to guide effective RL training; the corrector provides "correct thought examples" rather than scalar scores.

**Design 2: Dual-Objective Training (PPO + SFT)**
- **Function**: Thought tokens are aligned to the corrector's output via SFT; action tokens are optimized for environment rewards via PPO.
- **Mechanism**: $\min_\theta \mathbb{E}[\mathcal{L}_{PPO}(o,a) + \mathcal{L}_{SFT}(o, \pi_{corr}(o,th))]$. PPO ensures action exploration and optimization, while SFT ensures reasoning coherence.
- **Design Motivation**: Pure PPO leads to thought collapse; pure SFT cannot surpass the corrector's level. The dual-objective combination leverages the strengths of both.

**Design 3: DAgger Aggregation + Data Quality Control**
- **Function**: Aggregates all historical corrected data for SFT sampling, preventing distribution shift arising from non-i.i.d. training.
- **Mechanism**: While PPO discards old data each round, the DAgger buffer retains all corrected data. Format rewards and repetition penalties are additionally applied to improve data quality. The corrector may invoke tools (e.g., a Python calculator) to enhance correction accuracy.
- **Design Motivation**: Interactive imitation learning (DAgger) has been proven to converge to the expert policy.

### Loss & Training

PPO: standard clipped objective. SFT: standard autoregressive cross-entropy. A scaling factor $\lambda$ is applied to balance length among thought token log-probabilities. Training runs for 15K steps (24-Point Game) / 5K steps (ALFWorld) with single-GPU LoRA fine-tuning (~30 hours).

## Key Experimental Results

### Main Results

**24-Point Game (GPT-4o Corrector)**

| Model | Success Rate (%) | Return |
|-------|-----------------|--------|
| GPT-4V | 0 | -4.39 |
| GPT-4o | 2.5 | -6.35 |
| GPT-4o+Tool | 13.5 | -3.59 |
| LLaVA-7b-SFT | 3.0 | -15.30 |
| RL4VLM | 2.5 | -12.95 |
| SFT-only | 11.0 | -2.88 |
| **GTR** | **17.5** | **-2.17** |

### Ablation Study

| Process Guidance Method | Success Rate |
|------------------------|-------------|
| No Guidance (RL4VLM) | 2.5% |
| VLM-as-judge | ~3% |
| Length Reward | ~3% |
| SFT-only | 11.0% |
| **GTR (Corrector + RL)** | **17.5%** |

### Key Findings

1. GTR surpasses the corrector model itself (GPT-4o+Tool at 13.5%), demonstrating that RL enables the Agent to exceed imitation.
2. Thought collapse occurs across both 7B and 13B scales and at both 15K and 30K training steps, and does not diminish with increased model scale or training.
3. Numerical rewards from VLM-as-judge are nearly ineffective — they carry insufficient information and are susceptible to reward hacking.
4. On Qwen2.5-VL-7B, GTR enables the Agent to reach o3-level performance.

## Highlights & Insights

1. "Thought collapse" is identified as a core bottleneck in RL-based VLM Agent training — systematically defined and analyzed for the first time.
2. Using a corrector to replace PRMs or numerical rewards provides substantially richer information and requires no annotated data.
3. The Agent can surpass its "teacher" (the corrector) through RL, demonstrating the exploratory and discovery value of reinforcement learning.

## Limitations & Future Work

1. Reliance on external corrector (GPT-4o) introduces API usage costs.
2. The corrector itself has limited domain knowledge in certain areas and requires tool augmentation.
3. Validation is conducted only on a card game and ALFWorld; more complex embodied environments remain untested.

## Related Work & Insights

- RL4VLM pioneered RL-based fine-tuning of VLMs but is limited in complex tasks due to thought collapse.
- Key insight: Process supervision is more important than outcome supervision, but the form of supervision should be "examples" rather than "scores."

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ★★★★★ |
| Practicality | ★★★★☆ |
| Experimental Thoroughness | ★★★★☆ |
| Writing Quality | ★★★★★ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Loss-Guided Auxiliary Agents for Overcoming Mode Collapse in GFlowNets](../../AAAI2026/llm_agent/loss-guided_auxiliary_agents_for_overcoming_mode_collapse_in_gflownets.md)
- [\[AAAI 2026\] PerTouch: VLM-Driven Agent for Personalized and Semantic Image Retouching](../../AAAI2026/llm_agent/pertouch_vlm-driven_agent_for_personalized_and_semantic_image_retouching.md)
- [\[NeurIPS 2025\] MAT-Agent: Adaptive Multi-Agent Training Optimization](../../NeurIPS2025/llm_agent/mat-agent_adaptive_multi-agent_training_optimization.md)
- [\[NeurIPS 2025\] Group-in-Group Policy Optimization for LLM Agent Training](../../NeurIPS2025/llm_agent/groupingroup_policy_optimization_for_llm_agent_training.md)
- [\[AAAI 2026\] COVR: Collaborative Optimization of VLMs and RL Agent for Visual-Based Control](../../AAAI2026/llm_agent/covrcollaborative_optimization_of_vlms_and_rl_agent_for_visu.md)

</div>

<!-- RELATED:END -->
