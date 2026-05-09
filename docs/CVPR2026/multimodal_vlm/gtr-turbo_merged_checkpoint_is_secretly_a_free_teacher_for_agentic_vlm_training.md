---
title: >-
  [Paper Note] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training
description: >-
  [CVPR 2026][Multimodal VLM] This paper proposes GTR-Turbo, a framework that merges historical checkpoints generated during RL training to serve as a free teacher model. Without relying on expensive external API models, GTR-Turbo achieves performance comparable to or better than GTR in multi-turn visual agent training, while reducing training time by 50% and computational cost by 60%.
tags:
  - CVPR 2026
  - Multimodal VLM
date: 2026-05-08
content_hash: 54d18fef45caad5d
---

# GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training

**Conference**: CVPR 2026
**arXiv**: [2512.13043](https://arxiv.org/abs/2512.13043)
**Code**: [GitHub](https://github.com/tongweiICML/GTR-Turbo)
**Area**: Multimodal & VLM

## TL;DR

This paper proposes GTR-Turbo, a framework that merges historical checkpoints generated during RL training to serve as a free teacher model. Without relying on expensive external API models, GTR-Turbo achieves performance comparable to or better than GTR in multi-turn visual agent training, while reducing training time by 50% and computational cost by 60%.

## Background & Motivation

1. **Core challenges in multi-turn RL training**: VLM-based multimodal agents face sparse reward and long-horizon credit assignment problems in multi-turn reinforcement learning, leading to "thought collapse" where outputs become repetitive, incoherent, and templated.
2. **Existing methods rely on expensive teacher models**: Methods such as GTR densify rewards via step-level guidance from external teacher models (e.g., GPT-4o), but training LLaVA-1.6-7B for 15,000 steps requires approximately 4 days and $150, severely limiting scalability.
3. **Weak teacher models fail to provide effective guidance**: Using smaller teacher models reduces overhead but significantly degrades final performance. For example, Qwen2.5-VL-7B as a teacher fails entirely to provide effective thought guidance (success rate of 0%).
4. **Data privacy and accessibility constraints**: Relying on closed-source API models introduces not only cost concerns but also network latency and data privacy risks, and frontier models may be inaccessible in certain scenarios.

## Method

### 3.1 Core Idea: Merged Checkpoint as a Free Teacher

The key insight of GTR-Turbo is that historical checkpoints produced during RL training, after model merging, can constitute a stable and more capable teacher model — entirely without additional training or external model dependencies.

At update step $k$, the merged teacher model is defined as:

$$\pi^{(k)}_{\text{merged}} = \sum_{i=1}^{k-1} w_i \pi_{\theta}^{(i)}$$

### 3.2 TIES Merging Method

To avoid harmful interference caused by directly merging all parameters, the TIES (Trim, Elect Sign, and Merge) method is adopted:

1. **Trimming**: Only parameter changes within the top-$k$% by magnitude are retained; redundant parameters are removed.
2. **Sign Election**: For each parameter, the total magnitude of positive and negative values is computed, and the final sign is determined by majority vote.
3. **Selective Averaging**: Only parameters whose signs match the elected sign are included in the merge computation.

### 3.3 Weight Assignment Strategies

Two strategies are supported:

- **Simple Moving Average (SMA)**: All checkpoints are treated equally, $\pi_{\text{merged}}^{(k)} = \frac{1}{k-1}\sum_{i=1}^{k-1}\pi_{\theta}^{(i)}$
- **Exponential Moving Average (EMA)**: Recent checkpoints are assigned greater weight via smoothing factor $\alpha$, $\pi_{\text{merged}}^{(k)} = \alpha \cdot \pi_{\theta}^{(k-1)} + (1-\alpha) \cdot \pi_{\text{merged}}^{(k-1)}$

### 3.4 Variant 1: GTR-Turbo (SFT)

The merged teacher replaces the external corrector in GTR. At each RL step:
- After the agent generates thoughts and actions, the teacher generates reference thoughts under the same context.
- Online imitation learning is achieved by minimizing the SFT loss.

$$\min_{\theta} \mathbb{E}_{(o,a)\sim\mathcal{B}} \mathcal{L}_{\text{PPO}}(o,a) + \mathbb{E}_{(o,\hat{th})\sim\mathcal{D}} \mathcal{L}_{\text{SFT}}(o,\hat{th})$$

### 3.5 Variant 2: GTR-Turbo (KL) — Soft Logit Distillation

The SFT objective is replaced by KL divergence, where the reverse KL divergence between the agent and the teacher is used as an auxiliary reward:

$$A' = A^{\pi_\theta}(o,a) - \text{RevKL}(\pi_\theta, \pi_{\text{merged}}; th)$$

$$\text{RevKL}(\pi_\theta, \pi_{\text{merged}}; th) = \mathbb{E}_l \left[\log\pi_\theta(th_{[l]}|th_{[<l]}) - \log\pi_{\text{merged}}(th_{[l]}|th_{[<l]})\right]$$

Advantages of the KL variant: (1) only a single forward pass is required, with no need for autoregressive generation by the teacher; (2) it captures probability information across all candidate tokens rather than one-hot supervision; (3) memory consumption is reduced as no additional thought dataset is needed. A clip method is applied to truncate negative KL values to 0 to ensure reward validity.

## Key Experimental Results

### Points24 Card Game

| Model | Success Rate (%) | Episode Return (ER) |
|-------|-----------------|---------------------|
| GPT-4o | 2.5 | -6.35 |
| Qwen2.5-VL-72B | 5.6 | -5.69 |
| Qwen2.5-VL-7B-sft | 22.0 | -3.2 |
| RL4VLM | 3.5 | -13.3 |
| GTR (GPT-4o teacher) | 44.5 | 0.53 |
| **GTR-Turbo (SFT)** | **48.0** | **1.32** |
| **GTR-Turbo (KL)** | **53.5** | **2.39** |

### Computational Cost Comparison

| Environment | Method | Success Rate | Training Time | Estimated Cost |
|-------------|--------|-------------|---------------|----------------|
| Points24 | RL4VLM | 4% | 86h | $0 |
| Points24 | GTR | 41% | 191h | $307.78 |
| Points24 | GTR-Turbo (SFT) | 48% | 168h | $216.72 |
| Points24 | **GTR-Turbo (KL)** | **54%** | **89h** | **$114.81** |
| ALFWorld | RL4VLM | 8% | 70h | $0 |
| ALFWorld | GTR | 16% | 164h | $145.76 |
| ALFWorld | GTR-Turbo (KL) | 15% | 78h | $100.62 |

### Android-in-the-Wild (AitW) Results

| Method | Success Rate | Reasoning Score |
|--------|-------------|-----------------|
| DigiRL | 71.9% | - |
| PPO | 75.0% | 3.26 |
| **GTR-Turbo** | **80.2%** | **3.93** |

### Key Ablation Study

- **TIES merging vs. linear averaging**: TIES significantly outperforms simple linear merging by mitigating interference from redundant parameters.
- **KL estimation method**: The clip method performs best by controlling the magnitude of KL values to provide finer-grained updates.
- **Reverse KL vs. forward KL**: Reverse KL is more effective due to its mode-seeking property.
- **SMA vs. EMA**: SMA already yields strong performance; EMA with $\alpha=0.5$ achieves the best results.
- **Guidance scope**: Guiding thought only outperforms guiding both thought and action, as the latter restricts exploration.

## Highlights & Insights

- **Zero external dependency**: No external API models are required; the teacher model is constructed entirely by merging checkpoints from the model's own training process, addressing the core bottleneck of GTR.
- **Substantial efficiency gains**: The KL variant reduces training time on Points24 from 191h to 89h (−53%) and cost from $308 to $115 (−63%), while achieving superior performance.
- **Self-evolving paradigm**: The teacher model continuously accumulates knowledge and improves as training progresses, overcoming the limitation of GTR's fixed external model which cannot learn further.
- **Two flexible guidance modes**: The SFT variant preserves GTR's online imitation learning; the KL variant further improves efficiency and encourages exploration.
- **Strong generalization**: Consistent advantages are demonstrated across three diverse visual agent tasks: Points24, ALFWorld, and Android-in-the-Wild.

## Limitations & Future Work

- **Failure on weak base models**: When the base model's initial success rate is extremely low (<5%), self-improvement methods may fail, and a stronger external teacher may still be necessary.
- **Limited model scale**: Experiments are primarily conducted on 7B–8B models; the effectiveness at larger scales remains insufficiently explored.
- **Checkpoint storage overhead**: Saving multiple complete checkpoints during training for merging imposes non-trivial storage and memory requirements.

## Rating

- ⭐⭐⭐⭐ **Novelty**: Model merging techniques are cleverly introduced into RL agent training; the "free teacher" insight is novel and intuitively clear.
- ⭐⭐⭐⭐ **Practicality**: External API dependency is entirely eliminated, training costs are substantially reduced, and the practical deployment value is high.
- ⭐⭐⭐⭐ **Experimental Thoroughness**: Coverage spans 3 environments with extensive ablation studies, cost analysis, and compatibility verification on the latest Qwen3-VL.
- ⭐⭐⭐⭐ **Writing Quality**: Motivation is clearly articulated, method description is complete, experimental design is well-structured, and figures are rich and intuitive.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[ICLR 2026\] VLM-SubtleBench: How Far Are VLMs from Human-Level Subtle Comparative Reasoning?](../../ICLR2026/multimodal_vlm/vlm-subtlebench_how_far_are_vlms_from_human-level_subtle_comparative_reasoning.md)
- [\[ICLR 2026\] Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play](../../ICLR2026/multimodal_vlm/vision-zero_scalable_vlm_self-improvement_via_strategic_gamified_self-play.md)
- [\[ICLR 2026\] WebDS: An End-to-End Benchmark for Web-based Data Science](../../ICLR2026/multimodal_vlm/webds_an_end-to-end_benchmark_for_web-based_data_science.md)
- [\[AAAI 2026\] Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](../../AAAI2026/multimodal_vlm/plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)

<!-- RELATED:END -->
