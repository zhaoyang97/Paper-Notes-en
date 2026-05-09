---
title: >-
  [Paper Note] GTR: Guided Thought Reinforcement Prevents Thought Collapse in RL-Based VLM Agent
description: >-
  [ICCV 2025][Multimodal VLM][Chain-of-Thought Reasoning] This paper identifies that relying solely on outcome rewards during RL training of VLM agents leads to "thought collapse," and proposes the GTR framework, which employs an external VLM corrector to automatically rectify reasoning processes and jointly trains thoughts and actions via PPO + SFT, achieving 3–5× improvement in task success rates on the Game of 24 and ALFWorld benchmarks.
tags:
  - ICCV 2025
  - Multimodal VLM
  - Chain-of-Thought Reasoning
  - VLM Agent
  - Reinforcement Learning
  - Process Supervision
  - Thought Collapse
date: 2026-05-08
content_hash: 8cb72f0f548be1da
---

# GTR: Guided Thought Reinforcement Prevents Thought Collapse in RL-Based VLM Agent

**Conference**: ICCV 2025
**arXiv**: [2503.08525](https://arxiv.org/abs/2503.08525)
**Code**: [GTR](https://github.com/WeihaoTan/GTR)
**Area**: Multimodal VLM
**Keywords**: Chain-of-Thought Reasoning, VLM Agent, Reinforcement Learning, Process Supervision, Thought Collapse

## TL;DR

This paper identifies that relying solely on outcome rewards during RL training of VLM agents leads to "thought collapse," and proposes the GTR framework, which employs an external VLM corrector to automatically rectify reasoning processes and jointly trains thoughts and actions via PPO + SFT, achieving 3–5× improvement in task success rates on the Game of 24 and ALFWorld benchmarks.

## Background & Motivation

Large language models (LLMs) and vision-language models (VLMs) have shown promise in multi-step decision-making tasks. Reinforcement learning with verifiable rewards (RLVR) has successfully elicited chain-of-thought (CoT) reasoning in LLMs. However, this paradigm faces a critical bottleneck when applied to training VLM agents in dynamic visual environments.

### Thought Collapse Phenomenon

Through extensive experiments on the Game of 24 and ALFWorld embodied tasks, the authors identify a key phenomenon: **Thought Collapse**.

- **Manifestation**: The agent's reasoning process rapidly loses diversity, generating incomplete reasoning text irrelevant to the current state and converging toward templated outputs.
- **Consequence**: Although the model continues to produce "thoughts" and "actions," it has effectively lost its reasoning capacity, resulting in invalid actions and negative rewards.
- **Root Cause**: During RL training, rewards are determined entirely by the final action, leaving intermediate reasoning (thought tokens) without any supervision. In complex tasks with long horizons and large state spaces, accumulated errors cause training trajectories to deviate.

Experiments confirm that:
- Thought collapse occurs in both 7B and 13B models.
- Extending training steps from 15k to 30k provides no relief.
- The issue lies not in model capacity or training budget, but in the RL training paradigm itself.

**Root Cause**: How can RL-based VLM agent training leverage environmental rewards to optimize action policies while preventing the degradation of the reasoning process?

## Method

### Overall Architecture

The core idea of GTR (Guided Thought Reinforcement) is to introduce automated process guidance:

1. An external VLM (e.g., GPT-4o) serves as a corrector, evaluating and rectifying the agent's reasoning at each RL training step.
2. SFT loss is applied to thought tokens and PPO loss to action tokens for joint training.
3. The DAgger algorithm is employed to mitigate distributional shift in thought cloning.

### Key Designs

#### 1. VLM Corrector

- An off-the-shelf VLM (GPT-4o) is used as a plug-in corrector $\pi_{\text{corr}}$.
- At each step, it evaluates the agent's thought output, checking visual recognition accuracy and reasoning correctness.
- Upon detecting errors, it generates corrected thoughts based on the original output.

**Key properties**:
- Requires no human annotation or additional model training.
- Provides richer informational supervision than numerical scores (VLM-as-judge).
- Does not require the corrector to be an expert—corrected thoughts need only be reasonable.

#### 2. Joint Optimization Objective

$$\min_{\theta} \mathbb{E}_{o,(th,a)\sim\pi_\theta} \left[ \mathcal{L}_{\text{PPO}}(o,a) + \mathcal{L}_{\text{SFT}}(o, \pi_{\text{corr}}(o, th)) \right]$$

- **PPO loss**: Optimizes action tokens based on environmental rewards.
- **SFT loss**: Optimizes thought tokens to align with corrected reasoning trajectories.
- Only thoughts are cloned (not actions), preventing corrector hallucinations from interfering with action decisions.

#### 3. DAgger for Distributional Shift Mitigation

Since PPO discards old data and resamples at each iteration, performing thought cloning on top of this leads to catastrophic forgetting. The DAgger (Dataset Aggregation) algorithm is adopted:

$$\min_{\theta} \mathbb{E}_{(s,a)\sim\mathcal{B}} \mathcal{L}_{\text{PPO}} + \mathbb{E}_{(s,th)\sim\mathcal{D}} \mathcal{L}_{\text{SFT}}$$

- $\mathcal{B}$: Current on-policy PPO data buffer.
- $\mathcal{D}$: Accumulated historical correction data (DAgger dataset).

#### 4. Data Quality Improvements

- **Format reward**: Explicitly checks the validity of output format.
- **Repetition penalty**: Token-level repetition penalty to prevent format degradation.
- **Tool use**: The corrector can invoke Python code to verify Game of 24 expressions, enhancing correction accuracy.
- **Truncation strategy**: Episodes deemed meaningless or unsolvable by the corrector are truncated.

### Loss & Training

- Base model: LLaVA-v1.6-mistral-7B, pretrained with 1 epoch of SFT before RL.
- LoRA fine-tuning (r=128, α=256, dropout=0.05).
- Cosine learning rate decay: 1e-5 → 1e-9.
- PPO hyperparameters: γ=0.9, GAE λ=0.95, clip=0.1, 4 PPO epochs.
- Thought probability coefficient λ: 0.5 for Game of 24, 0.2 for ALFWorld.

## Key Experimental Results

### Main Results (Game of 24)

| Model | Success Rate (%) | Episode Return |
|-------|-----------------|----------------|
| GPT-4o | 2.5 | -6.35 |
| GPT-4o + Tool | 13.5 | -3.59 |
| Qwen2-VL-72B | 4.5 | - |
| LLaVA-7b-sft | 3.0 | -15.30 |
| RL4VLM | 2.5 | -12.95 |
| SFT-only | 11.0 | -2.88 |
| **GTR** | **17.5** | **-2.17** |

GTR achieves a **7×** improvement in success rate over the SOTA method (RL4VLM) and surpasses the GPT-4o + Tool corrector itself (13.5% → 17.5%), demonstrating that RL enables agents to exceed pure imitation.

### Ablation Study

| Ablation Condition | Effect |
|-------------------|--------|
| Remove DAgger | Early improvement, later stagnation |
| Remove tool use | No performance gain; reasoning becomes illogical |
| Cosine annealing of thought loss weight | Thought collapse re-emerges after guidance relaxation |
| SFT on full response (including actions) | Corrector hallucinations interfere with action decisions |
| Qwen2.5-VL-72B as corrector | 6.5% (insufficient tool-use capability) |
| Qwen2.5-VL-7B as corrector | Failure (unable to follow correction format) |

### Key Findings

1. **Process guidance must be sustained throughout training**: Weight annealing experiments show that relaxing guidance immediately causes collapse to recur.
2. **Corrector capability has a lower bound**: Sufficient analytical ability and tool-use capability are required.
3. **Cloning only thoughts > cloning full responses**: SFT constraints on thoughts and environmental constraints on actions conflict.
4. **GTR on Qwen2.5-VL-7B achieves o3-level performance**, whereas RL4VLM degrades with extended training.
5. **Pure visual input in ALFWorld** (removing text descriptions) better approximates the real world, yet GTR still achieves competitive success rates.

## Highlights & Insights

1. **Introduction of the thought collapse concept**: The paper offers deep insight into the failure mode of RLVR in VLM agents, with significant theoretical implications.
2. **Elegant design philosophy**: Rather than training a process reward model or relying on human annotation, the framework directly uses a corrector to generate corrected text—providing far richer supervision than numerical scores.
3. **Decoupled training of thoughts and actions**: SFT governs thoughts while PPO governs actions, avoiding conflicts between the two supervision signals.
4. **GTR surpasses the corrector itself**: Demonstrating that the combination of RL and process guidance outperforms pure imitation.

## Limitations & Future Work

- o1-style long CoT strategies for action sequence reasoning are not explored.
- Due to resource constraints, validation is limited to 7B models; larger models may yield greater gains.
- The API cost of the GPT-4o corrector is substantial (~\$463.5 / 15k steps).
- Overall success rate in ALFWorld without text observations remains low (18%).
- The DAgger dataset grows continuously; memory management for long-term training requires consideration.

## Related Work & Insights

- RL4VLM is the direct predecessor, establishing the VLM RL fine-tuning framework but showing limited effectiveness on complex tasks.
- Thought Cloning inspired the combination of SFT and RL in GTR.
- Off-policy methods such as TD3+BC integrate supervised signals into RL, but GTR adopts DAgger to address the specific characteristics of on-policy PPO.
- Works such as DeepSeek-R1 demonstrate the strong potential of RL for eliciting LLM reasoning; GTR extends this to visual decision-making.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The discovery of thought collapse and the corrector-guided framework represent clear innovations.
- **Technical Depth**: ⭐⭐⭐⭐ — In-depth analysis and thorough ablation studies.
- **Practical Value**: ⭐⭐⭐⭐ — Directly applicable to RL training of VLM agents.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear and experimental design is well-motivated.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](../../CVPR2026/multimodal_vlm/gtr_turbo_merged_checkpoint_free_teacher.md)
- [\[AAAI 2026\] VipAct: Visual-Perception Enhancement via Specialized VLM Agent Collaboration and Tool-use](../../AAAI2026/multimodal_vlm/vipact_visual-perception_enhancement_via_specialized_vlm_age.md)
- [\[ICCV 2025\] MMAT-1M: A Large Reasoning Dataset for Multimodal Agent Tuning](mmat1m_a_large_reasoning_dataset_for_multimodal_agent_tuning.md)
- [\[NeurIPS 2025\] Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning](../../NeurIPS2025/multimodal_vlm/praxisvlm_visiongrounded_decision_making_via_textdriven_rein.md)
- [\[NeurIPS 2025\] What Can RL Bring to VLA Generalization? An Empirical Study](../../NeurIPS2025/multimodal_vlm/what_can_rl_bring_to_vla_generalization_an_empirical_study.md)

<!-- RELATED:END -->
