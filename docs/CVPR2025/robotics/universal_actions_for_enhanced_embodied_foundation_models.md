---
title: >-
  [Paper Note] UniAct: Universal Actions for Enhanced Embodied Foundation Models
description: >-
  [CVPR 2025][Robotics][Embodied AI] UniAct proposes building embodied foundation models in a Universal Action Space, encoding atomic behaviors shared across diverse embodied platforms via a vector-quantized codebook. The 0.5B parameter model outperforms SOTA models 14 times its size and supports rapid adaptation to new robots.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Embodied AI"
  - "Universal Action Space"
  - "Cross-Embodied Transfer"
  - "Vision-Language-Action Models"
  - "Vector Quantization"
date: 2026-05-08
content_hash: 15520fac7468c103
---

# UniAct: Universal Actions for Enhanced Embodied Foundation Models

**Conference**: CVPR 2025  
**arXiv**: [2501.10105](https://arxiv.org/abs/2501.10105)  
**Code**: [Project Page](https://2toinf.github.io/UniAct/)  
**Area**: Robotics  
**Keywords**: Embodied AI, Universal Action Space, Cross-Embodied Transfer, Vision-Language-Action Models, Vector Quantization

## TL;DR

UniAct proposes building embodied foundation models in a Universal Action Space, encoding atomic behaviors shared across diverse embodied platforms via a vector-quantized codebook. The 0.5B parameter model outperforms SOTA models 14 times its size and supports rapid adaptation to new robots.

## Background & Motivation

The core challenge in developing universal embodied foundation models is **action heterogeneity**:

- **Embodied Differences**: Robots with different degrees of freedom (robotic arms, quadrupeds, autonomous vehicles) possess completely different action spaces.
- **Different Control Interfaces**: Even for the same robot, end-effector position control and velocity control carry fundamentally different physical meanings.
- **Behavioral Multimodality**: Data collected by different operators on the same platform exhibit high multimodality.

Limitations of Prior Work:
- **Forced Unification**: Methods like RT-X, Octo, and OpenVLA forcefully treat different action spaces as equivalent, which leads to similar encodings representing completely different physical meanings.
- **Naive Aggregation**: Methods like CrossFormer and RDT aggregate all action spaces but fail to mine commonalities across platforms.
- **Latent Actions**: Frameworks like LAPA infer latent actions through changes in video frames, which inadvertently captures control-irrelevant distractors (such as the appearance of new objects).

The key insight is that although control signals vary drastically across different robots, they should execute similar "move forward" behaviors when facing a target directly in front of them. Such abstract atomic behaviors can be shared across embodiments.

## Method

### Overall Architecture

UniAct is built upon a pretrained VLM (LLaVA-OneVision-0.5B) and consists of three core components: (1) a shared VLM acting as a universal action extractor; (2) a vector-quantized codebook $\mathcal{U} \in \mathbb{R}^{256 \times 128}$ serving as the universal action space; and (3) lightweight heterogeneous decoding heads that translate universal actions into concrete control signals.

### Key Designs

**Design 1: Universal Action Space — Discrete Vector-Quantized Codebook**

- **Function**: Distill heterogeneous actions across embodied platforms into shared atomic behavior representations.
- **Mechanism**: A codebook $\mathcal{U} = (u_1, u_2, \ldots, u_N)$ is constructed with $N=256$ vectors of $D=128$ dimensions, where each code encodes a universal atomic behavior. All robots are forced to share the same codebook, creating a critical information bottleneck that drives the model to discover and exploit shared primitive behaviors across platforms.
- **Design Motivation**: Discrete representations have demonstrated powerful capabilities in complex reasoning, planning, and predictive learning (e.g., the success of LLMs). Restricting the representation to a discrete space forces the model to compress information, extracting the true essence of behaviors shared across platforms.

**Design 2: Universal Action Extractor — Task-Oriented VLM-Based Extraction**

- **Function**: Infer the most relevant universal action $u^* = \arg\max_{u \in \mathcal{U}} p(u|o,g)$ based on current observation $o$ and task goal $g$.
- **Mechanism**: Fine-tune a pretrained VLM to output a probability distribution over the codebook, achieving differentiable action selection via Gumbel-Softmax: $u^* = \sum_{i=1}^n w_i u_i$, where weights $w_i$ are computed via Gumbel-Softmax. The temperature $\tau$ is gradually annealed during training.
- **Design Motivation**: Unlike merely inferring latent actions through changes in video frames, this method extracts universal actions in a task-oriented manner, avoiding the acquisition of control-irrelevant observational changes. Leveraging the vision-language reasoning capabilities and pretrained knowledge of VLMs improves sample efficiency.

**Design 3: Heterogeneous Decoding Heads — Lightweight Embodiment-Specific Translation**

- **Function**: Translate highly abstract universal actions into precise control signals executable by each specific embodied platform.
- **Mechanism**: Design a simple MLP decoding head $h_k$ for each embodiment type, taking the universal action $u^*$ and visual features $o$ as inputs, and outputting embodiment-specific control commands $\hat{a}^{(k)} = h_k(u^*, o)$. Adapting to a new robot only requires adding a new decoding head.
- **Design Motivation**: Keeping the decoding heads lightweight ensures that learning is primarily concentrated within the universal action space, maximizing cross-embodiment generalization. Since universal behaviors are already captured, the decoders only need to incorporate embodiment-specific details.

### Loss & Training

The total training objective is the sum of behavioral cloning losses across all domains: $\min_{\mathcal{U},\theta} \sum_{k=1}^K \mathbb{E}_{a_i \in \tau_i, \tau_i \in \mathcal{D}_k} \mathcal{L}_k(\hat{a}^{(k)}, a_i^{(k)})$, where $\mathcal{L}_k$ can be customized according to the action type (e.g., cross-entropy for discrete actions, MSE/Huber/diffusion loss for continuous actions). The codebook and extractor are updated globally, while the decoding heads are updated per domain.

## Key Experimental Results

### Main Results: Real-World WidowX Robot (19 tasks, 190 rollouts)

| Model | Parameters | Visual | Motion | Physical | Semantic | Language |
|------|--------|--------|--------|----------|----------|----------|
| Octo | 0.1B | Low | Low | Low | Low | Low |
| CrossFormer | 0.1B | Low | Low | Low | Low | Low |
| OpenVLA | 7B | Medium | Medium | Medium | **High** | **High** |
| **UniAct** | **0.5B** | **High** | **High** | **High** | High | High |

UniAct-0.5B outperforms the 14x larger OpenVLA-7B on visual, motion, and physical generalization tasks.

### Ablation Study: Rapid Adaptation to New Robot AIRBOT

| Pretrained Model | Sweep Plate | Fold Towel | Cup on Plate | Transport Pen |
|-----------|-------------|------------|--------------|---------------|
| LLaVa-OV-0.5B | 7.5% | 20% | 2.5% | 15% |
| **UniAct-0.5B** | **45%** | **62.5%** | **50%** | **65%** |

UniAct requires fine-tuning only 0.8% of its parameters (4M/500M) to adapt to a new robot, which is far lower than OpenVLA (1.4%) and Octo (2%).

### Key Findings

- Through manual inspection of the 256 universal actions, at least 40% exhibit completely consistent semantic behaviors across different robots.
- The utilization distribution of universal actions for the same task on different robots is similar (low JS divergence), while it differs across different tasks.
- A robot can be manually controlled to perform complex tasks directly by selecting universal action IDs, without requiring any knowledge of forward/inverse kinematics.

## Highlights & Insights

1. **Ingenious Information Bottleneck Design**: The discrete codebook forces different embodiments to share the same abstract space, naturally driving the model to discover cross-platform commonalities.
2. **Task-Oriented vs. Observation-Oriented**: Universal actions are extracted based on task progress rather than video frame differences, avoiding disturbances from irrelevant visual changes.
3. **Extreme Efficiency**: The 0.5B model outperforms the 7B model, demonstrating that "the right representation space is more important than model scale."

## Limitations & Future Work

- Currently limited by resources, evaluation was only performed using a 0.5B-parameter model and single-arm robots.
- Future work will scale up to larger models and more diverse embodiment types (dual-arm robots, autonomous driving).
- The universal action extractor can serve as an action tokenizer, providing support for planning in future large-scale embodied foundation models.

## Related Work & Insights

- **OpenVLA/RT-X/Octo**: Trained directly on heterogeneous action spaces without resolving action semantic conflicts.
- **VQ-BeT/QueST**: Discrete action encoding under single-embodiment scenarios, handling the multimodality of human demonstrations.
- **LAPA/IGOR**: Inferring latent actions through video frame changes, lacking control causality.
- **Insight**: The key to cross-domain/cross-modal learning lies in finding the correct shared representation space, and discrete bottlenecks are effective tools for achieving this goal.

## Rating

⭐⭐⭐⭐⭐ — This work is highly pioneering. The concept of a universal action space is clean and elegant, and the experimental design is comprehensive (real robots + simulation + new robot adaptation). The 0.5B model outperforming the 7B model is highly impressive. It provides crucial insights for representation learning in the field of Embodied AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-Improving Embodied Foundation Models](../../NeurIPS2025/robotics/self-improving_embodied_foundation_models.md)
- [\[CVPR 2025\] DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness](dexgrasp_anything_towards_universal_robotic_dexterous_grasping_with_physics_awar.md)
- [\[ICML 2025\] FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making](../../ICML2025/robotics/founder_grounding_foundation_models_in_world_models_for_open-ended_embodied_deci.md)
- [\[NeurIPS 2025\] Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics](../../NeurIPS2025/robotics/robot-r1_reinforcement_learning_for_enhanced_embodied_reasoning_in_robotics.md)
- [\[CVPR 2025\] Lift3D Foundation Policy: Lifting 2D Large-Scale Pretrained Models for Robust 3D Robotic Manipulation](lift3d_policy_lifting_2d_foundation_models_for_robust_3d_robotic_manipulation.md)

</div>

<!-- RELATED:END -->
