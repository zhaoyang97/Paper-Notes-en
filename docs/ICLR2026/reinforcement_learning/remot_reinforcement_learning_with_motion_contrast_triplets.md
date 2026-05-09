---
title: >-
  [Paper Note] ReMoT: Reinforcement Learning with Motion Contrast Triplets
description: >-
  [ICLR 2026][Reinforcement Learning][Vision-Language Models] ReMoT proposes a unified training paradigm that systematically enhances VLM spatiotemporal consistency reasoning through a rule-driven motion contrast triplet dataset (ReMoT-16K) and Group Relative Policy Optimization (GRPO) with composite reward optimization, achieving a 25.1% performance gain on spatiotemporal reasoning benchmarks.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Vision-Language Models
  - Spatiotemporal Reasoning
  - Motion Contrast Triplets
  - GRPO
date: 2026-05-08
content_hash: cf7163025fe0ceac
---

# ReMoT: Reinforcement Learning with Motion Contrast Triplets

**Conference**: ICLR 2026
**arXiv**: [2603.00461](https://arxiv.org/abs/2603.00461)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: Vision-Language Models, Spatiotemporal Reasoning, Motion Contrast Triplets, GRPO, Reinforcement Learning

## TL;DR
ReMoT proposes a unified training paradigm that systematically enhances VLM spatiotemporal consistency reasoning through a rule-driven motion contrast triplet dataset (ReMoT-16K) and Group Relative Policy Optimization (GRPO) with composite reward optimization, achieving a 25.1% performance gain on spatiotemporal reasoning benchmarks.

## Background & Motivation
1. **Background**: Vision-language models (VLMs) have evolved into general-purpose perception systems with broad applications in AIGC, embodied intelligence, and autonomous driving. These tasks fundamentally require models to understand the evolution of physical scenes across both spatial and temporal dimensions.
2. **Limitations of Prior Work**: Current mainstream VLMs (including GPT-4o, Claude-Sonnet-4.5, and Gemini-2.5-Pro) exhibit fundamental deficiencies in spatiotemporal consistency reasoning—confusing camera rotation with object motion, misidentifying robotic arm states, and incorrectly inferring character movement directions.
3. **Key Challenge**: Existing improvement methods (architectural modifications, data augmentation) address issues in isolation and lack a unified framework that systematically resolves the problem across data, training, and evaluation dimensions. Training data predominantly consists of static image-text pairs, with no explicit modeling of fine-grained motion attributes.
4. **Goal**: To systematically address the fundamental shortcomings of VLMs in spatiotemporal consistency reasoning, spanning data construction, training optimization, and evaluation benchmarks.
5. **Key Insight**: Motion contrast triplets (anchor-positive-negative) are employed to explicitly model inter-frame motion attributes, compelling the model to learn fine-grained motion discrimination rather than relying on superficial visual patterns.
6. **Core Idea**: Video meta-annotations (camera pose matrices, robot action logs, etc.) are leveraged through a multi-expert collaborative pipeline to automatically generate large-scale motion contrast triplets, which are then used for training via GRPO reinforcement learning with composite rewards.

## Method

### Overall Architecture
Video meta-annotations → Multi-expert collaborative pipeline generates motion contrast triplets (ReMoT-16K) → SFT/GRPO/hybrid training based on Qwen3-VL-4B-Thinking → Output VLM with enhanced spatiotemporal reasoning capability.

### Key Designs
1. **Multi-Expert Collaborative Data Construction Pipeline**:
    - **Motion Estimation Expert** $g:(I_t, I_{t'}, \mathcal{A}) \to m$: Extracts precise motion attributes from metadata. The navigation expert computes rigid-body transformations from SE(3) pose matrices; the manipulation expert extracts end-effector trajectories from robot telemetry data.
    - **Triplet Construction Expert** $(\phi, \mathcal{N})$: Filters salient positive sample pairs via attribute-specific magnitude thresholds: $\phi(I_t, I_{t'}, m) = (I_{\text{anchor}}, I_{\text{pos}}, m)$ when $\|m\| \in \mathcal{T}_m$. Negative samples are generated through attribute-conditional synthesis: geometric synthesis $\mathcal{T}_{\text{geo}}$ simulates reversed motion, while retrieval $\mathcal{R}$ searches for visually similar but attribute-mismatched frames.
    - **VQA Formulation Expert**: Designs multi-perspective chain-of-thought question-answering for each triplet (multiple-choice, true/false, fill-in-the-blank, comparative reasoning, etc.).
    - Design Motivation: Direct generation via VLMs produces 55% formatting errors and incurs high API costs; the multi-expert pipeline enables scalable, high-quality data generation.

2. **GRPO Composite Reward Training**:
    - Core objective: $J(\theta) = \mathbb{E}_{q,\{o_i\}}\left[\frac{1}{G}\sum_{i=1}^{G}\min(r_i\hat{A}_i, \text{clip}(r_i, 1-\varepsilon, 1+\varepsilon)\hat{A}_i) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right]$
    - **CoT Length Regularization**: $R_{\text{length}}(o_i) = -\max(0, |o_i^{\text{think}}| - L_{\text{target}})$, penalizing excessively verbose reasoning chains.
    - **Logical Consistency Reward**: $R_{\text{logic}}(o) \in \{+1, 0, -1\}$, identifying logical contradictions via transitivity checks (e.g., $L_1 < L_2, L_2 < L_3$ but $L_3 < L_1$).
    - **Composite Reward**: $R_i = R_{\text{task}} + \lambda_1 R_{\text{logic}} + \lambda_2 R_{\text{length}}$, with weight ratio 3.5:3.5:1.3:1.7.
    - Design Motivation: 31.4% of baseline errors manifest as logical inconsistencies; a decoupled logic reward directly corrects this failure mode.

3. **Hybrid Optimization Strategy**:
    - Sequential hybrid (SFT→GRPO): SFT first provides stable initialization, followed by GRPO fine-tuning.
    - Alternating hybrid (SFT↔GRPO): SFT and GRPO steps alternate every few iterations, enabling joint evolution of language alignment and reward alignment. The alternating strategy achieves the best performance.

### Loss & Training
- During SFT, cross-entropy loss is computed only on tokens within `<answer>` tags
- GRPO uses 4 rollouts, batch size 16, and KL regularization coefficient 0.01
- The alternating hybrid strategy achieves peak performance after 2 epochs (Overall 39.9%, Partial 67.7%)
- Training hardware: 8× A800 GPUs, mixed precision

## Key Experimental Results

### Main Results

| Model | Overall Acc. (%) | Partial Acc. (%) | Notes |
|-------|-----------------|-----------------|-------|
| Qwen3-VL-CoT-4B (baseline) | 20.7 | 38.9 | Base model |
| GRPO | 33.6 | 61.6 | RL only |
| SFT→GRPO | 35.0 | 63.3 | Sequential hybrid |
| **SFT↔GRPO** | **38.0** | **64.0** | Alternating hybrid, best |
| SFT↔GRPO (2 epochs) | 39.9 | 67.7 | Extended training |
| GPT-5-Chat | 10.4 | 33.3 | Closed-source |
| Gemini-2.5-Pro | 26.4 | 49.1 | Closed-source |

### Ablation Study

| Configuration | Overall Acc. (%) | Partial Acc. (%) | Notes |
|--------------|-----------------|-----------------|-------|
| No training (baseline) | 20.7 | 38.9 | Qwen3-VL-CoT-4B |
| Manipulation data only | 23.9 | 46.7 | +3.2% |
| + Navigation data | 32.4 | 57.6 | +8.4%, critical for spatial relation reasoning |
| + Simulation data | 38.0 | 64.0 | +5.6% |
| Binary contrast vs. Triplet contrast | 19.4 vs. 38.0 | 39.4 vs. 64.0 | Triplet design yields +18.6% |
| GRPO w/o logic reward | 68.6 (Ov.) | 77.3 (Par.) | Manipulation subset |
| GRPO w/ logic reward | 78.0 (Ov.) | 81.3 (Par.) | +10.6%, logical consistency 99.3% |

### Key Findings
- Alternating SFT-GRPO training outperforms pure SFT or pure GRPO by enabling joint optimization of language fluency and reward alignment
- Triplet contrast outperforms binary contrast by 18.6% (Overall); joint contrastive supervision is critical for fine-grained motion discrimination
- The multi-expert pipeline yields substantially higher data quality than VLM-generated data, which exhibits high variance and plateaus at approximately 0.49 when scaled
- A 4B model can match or surpass Qwen3-VL-30B-CoT (7.5× larger) on spatiotemporal reasoning benchmarks
- Pure SFT training leads to training collapse, causing the model to lose chain-of-thought reasoning capability

## Highlights & Insights
- The work presents a complete framework that systematically addresses VLM spatiotemporal reasoning deficiencies across data, training, and evaluation dimensions
- The multi-expert collaborative pipeline cleverly exploits video meta-annotations, avoiding costly human annotation and unreliable model-generated labels
- The composite reward design—particularly the logical consistency reward—is highly targeted, directly addressing the disconnect between reasoning chains and final answers
- Achieving GPT-4o-level spatiotemporal reasoning capability with a 4B model represents an exceptionally favorable cost-performance trade-off

## Limitations & Future Work
- The ReMoT-16K dataset is relatively limited in scale (16.5K triplets); scaling up may yield further improvements
- Closed-source model evaluation is restricted to a mini-benchmark of approximately 40 samples due to API cost constraints
- Validation is currently limited to 4B/8B models; performance on larger-scale models remains unexplored
- Motion contrast triplets are primarily sourced from navigation, manipulation, and simulation domains; extension to broader domains (e.g., sports, medical imaging) is possible
- Although reasoning chain faithfulness errors are reduced from 60% to 12%, further improvement remains feasible

## Related Work & Insights
- Pose-free NeRF methods such as GeoNLF and BARF inspired the approach of extracting motion information from metadata
- The successful application of GRPO in visual reasoning demonstrates the potential of RL methods in multimodal settings
- The motion contrast triplet design is generalizable to downstream tasks such as video understanding and robot action recognition
- The logical consistency reward design is transferable to other tasks requiring multi-step reasoning consistency

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Entropy-Preserving Reinforcement Learning (REPO / ADAPO)](entropy-preserving_reinforcement_learning.md)
- [\[ICLR 2026\] QuRL: Efficient Reinforcement Learning with Quantized Rollout](qurl_efficient_reinforcement_learning_with_quantized_rollout.md)
- [\[ICLR 2026\] MVR: Multi-view Video Reward Shaping for Reinforcement Learning](mvr_multi-view_video_reward_shaping_for_reinforcement_learning.md)
- [\[ICLR 2026\] Helix: Evolutionary Reinforcement Learning for Open-Ended Scientific Problem Solving](helix_evolutionary_reinforcement_learning_for_open-ended_scientific_problem_solv.md)
- [\[ICLR 2026\] REA-RL: Reflection-Aware Online Reinforcement Learning for Efficient Reasoning](rea-rl_reflection-aware_online_reinforcement_learning_for_efficient_reasoning.md)

<!-- RELATED:END -->
