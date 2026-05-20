---
title: >-
  [Paper Note] DriveDPO: Policy Learning via Safety DPO For End-to-End Autonomous Driving
description: >-
  [NeurIPS 2025][Autonomous Driving][End-to-end autonomous driving] DriveDPO is a two-stage framework that first fuses human-imitation similarity and rule-based safety scores into a single supervised distribution via unifi…
tags:
  - "NeurIPS 2025"
  - "Autonomous Driving"
  - "End-to-end autonomous driving"
  - "Safety DPO"
  - "preference optimization"
  - "trajectory planning"
  - "NAVSIM"
date: 2026-05-08
content_hash: 1efc99caa43ef52a
---

# DriveDPO: Policy Learning via Safety DPO For End-to-End Autonomous Driving

**Conference**: NeurIPS 2025
**arXiv**: [2509.17940](https://arxiv.org/abs/2509.17940)  
**Code**: None  
**Area**: Autonomous Driving
**Keywords**: End-to-end autonomous driving, Safety DPO, preference optimization, trajectory planning, NAVSIM

## TL;DR
DriveDPO is a two-stage framework that first fuses human-imitation similarity and rule-based safety scores into a single supervised distribution via unified policy distillation, then applies Safety DPO to construct trajectory preference pairs of the form "human-like but unsafe vs. human-like and safe" for policy fine-tuning — achieving a new state-of-the-art PDMS of 90.0 on NAVSIM.

## Background & Motivation

**Background**: End-to-end autonomous driving predicts future trajectories directly from raw sensor inputs, avoiding the error accumulation of modular pipelines. Dominant approaches (VADv2, UniAD, etc.) rely on imitation learning, minimizing the geometric distance between predicted and human-driven trajectories.

**Limitations of Prior Work**: (a) Imitation learning cannot distinguish trajectories that "look human-like but are unsafe" — even minor deviations from human trajectories may cause collisions or out-of-bound violations; (b) symmetric loss functions such as MSE penalize deviations in both directions equally, despite the asymmetric safety implications (surpassing the human trajectory may cause rear-end collisions, while lagging behind is generally safe); (c) recent score-based methods (e.g., Hydra-MDP) independently regress multiple scores for each candidate trajectory without directly optimizing the policy distribution, leading to suboptimal performance.

**Key Challenge**: Imitation learning optimizes human-likeness but ignores safety; score-based methods incorporate safety signals but decouple them from policy optimization. A unified approach that simultaneously optimizes safety and human consistency at the policy distribution level is needed.

**Goal**: How can human trajectory imitation and rule-based safety signals be jointly incorporated into direct optimization of the policy distribution?

**Key Insight**: Drawing inspiration from RLHF/DPO in large language models — reformulating safety requirements as trajectory-level preference learning.

**Core Idea**: Unified policy distillation (combining imitation and safety into a single distribution) + Safety DPO (trajectory-level preference alignment via carefully constructed preference pairs).

## Method

### Overall Architecture
Two-stage training. **Stage 1** — Unified Policy Distillation: multi-view cameras and LiDAR are encoded via a Transfuser backbone; an Anchor Vocabulary discretizes the action space; imitation similarity and PDMS safety scores are fused into a unified supervised distribution and trained with KL divergence. **Stage 2** — Safety DPO: $K$ candidate trajectories are sampled from the current policy to construct safety preference pairs, and the policy is fine-tuned with the DPO loss.

### Key Designs

1. **Unified Policy Distillation**

    - Function: Merges human imitation signals and rule-based safety signals into a single soft-label distribution.
    - Mechanism: For each anchor trajectory $a_i$, the imitation similarity $\text{Sim}(a_i) = \text{Softmax}(-\|a_i - \hat{a}\|_2)$ and PDMS safety score $\text{PDMS}(a_i)$ are computed. A log transformation amplifies differences at low scores: $p_{unified}(a_i) = \text{Softmax}(w_1 \cdot \log(\text{Sim}(a_i)) + w_2 \cdot \log(\text{PDMS}(a_i)))$. KL divergence is used to align the policy output to this unified distribution.
    - Design Motivation: Unlike score-based methods that independently predict scores for each trajectory, the unified distribution introduces a competitive mechanism across all anchors, directly optimizing the policy distribution. The log transformation exponentially amplifies the penalty for low-safety trajectories.

2. **Safety DPO**

    - Function: Further improves policy safety through trajectory-level preference learning.
    - Mechanism: $K$ candidate trajectories are sampled from the policy distribution; the highest-scoring candidate serves as the chosen trajectory $a_w$. The key lies in selecting the rejected trajectory — **imitation-based selection** chooses the candidate with PDMS below threshold $\tau$ that is closest to the human trajectory (i.e., "the most human-like yet most unsafe" candidate). This preference pair construction directly targets the core failure mode of imitation learning. The policy is fine-tuned using the standard DPO loss to prefer safe trajectories.
    - Design Motivation: Naive DPO (selecting the highest- and lowest-scoring candidates) produces trivially separable pairs. The imitation-based construction of Safety DPO enables the model to precisely learn to distinguish safe from unsafe trajectories among superficially plausible candidates.

3. **Anchor Vocabulary + Transfuser Architecture**

    - Function: Discretizes the continuous trajectory space into $N$ anchor points, enabling the policy to output a discrete distribution.
    - Mechanism: $N = 4096$ anchor points are obtained by k-means clustering over training-set human trajectories. Anchor features are encoded via Fourier positional encoding + MLP, then fused with scene features through a Cross-Attention Transformer Decoder; a softmax produces the final policy distribution.
    - Design Motivation: Discretization naturally accommodates the DPO framework (preference optimization over distributions) and is consistent with the Anchor Vocabulary paradigm.

### Loss & Training
- Perception backbone: ResNet-34 (Transfuser)
- Input: Multi-view cameras + LiDAR
- Stage 1: Unified policy distillation pretraining
- Stage 2: Safety DPO fine-tuning, iterated for 3 rounds

## Key Experimental Results

### Main Results (NAVSIM Benchmark)

| Method | Supervision | NC↑ | DAC↑ | EP↑ | TTC↑ | C↑ | PDMS↑ |
|--------|------------|-----|------|-----|------|-----|-------|
| Transfuser | Human | 97.7 | 92.8 | 79.2 | 92.8 | 100.0 | 84.0 |
| DiffusionDrive | Human | 98.2 | 96.2 | 82.2 | 94.7 | 100.0 | 88.1 |
| Hydra-MDP | H+Rule | 98.3 | 96.0 | 78.7 | 94.6 | 100.0 | 86.5 |
| WOTE | H+Rule | 98.4 | 96.6 | 81.7 | 94.5 | 99.9 | 88.0 |
| **Ours (w/o DPO)** | H+Rule | 97.9 | 97.3 | 84.0 | 93.6 | 100.0 | 88.8 |
| **Ours (full)** | **H+Rule** | **98.5** | **98.1** | **84.3** | **94.8** | **99.9** | **90.0** |

### Ablation Study

| Configuration | PDMS | Note |
|---------------|------|------|
| Pure imitation learning | 84.0 | Transfuser baseline |
| Unified policy distillation | 88.8 | +4.8 PDMS |
| + Naive DPO | 89.2 | +0.4 |
| + Safety DPO (imitation-based) | **90.0** | +1.2 |
| + Safety DPO (distance-based) | 89.8 | +1.0 |

### Key Findings
- Unified policy distillation alone contributes +4.8 PDMS (84.0→88.8), demonstrating that directly optimizing the policy distribution outperforms independent score regression.
- Safety DPO yields a further +1.2 improvement (88.8→90.0); imitation-based rejected selection outperforms distance-based selection.
- DAC (road compliance) shows the largest absolute gain: 92.8→98.1 (+5.3), indicating that safety optimization most effectively reduces out-of-bound behavior.
- EP (route progress) also improves substantially: 79.2→84.3, demonstrating that safety constraints do not sacrifice task completion efficiency.
- In Bench2Drive closed-loop evaluation, DriveDPO also achieves the highest success rate of 30.62% and driving score of 62.02.

## Highlights & Insights
- **Elegant transfer of RLHF to autonomous driving**: Adapting DPO from token-level preferences in LLMs to trajectory-level preferences is conceptually natural and technically straightforward. The Safety DPO preference pair construction — "human-like but unsafe" vs. "human-like and safe" — precisely captures the core problem.
- **Effective use of the log transformation**: Mapping $[0,1]$ safety scores to $(-\infty, 0]$ exponentially down-weights unsafe trajectories in the distribution, providing stronger differentiation between safe and unsafe candidates compared to linear weighting.
- **Unified distribution vs. independent scores**: Results directly demonstrate that constructing a competitive unified distribution over all anchors outperforms independently regressing scores for each trajectory (88.8 vs. Hydra-MDP's 86.5).

## Limitations & Future Work
- NAVSIM relies on open-loop evaluation; more extensive closed-loop validation is needed to confirm real safety improvements (Bench2Drive results remain limited in scale).
- The rejected trajectory selection strategy in Safety DPO still requires manual design; automated methods warrant exploration.
- PDMS acquisition depends on the NAVSIM simulator, which may introduce a sim-to-real gap during real-world deployment.
- Only a ResNet-34 backbone is employed; the scaling behavior of larger models and stronger perception backbones remains unverified.

## Related Work & Insights
- **vs. Hydra-MDP / WOTE**: Both use rule-based supervision but independently regress scores; DriveDPO unifies signals into a distribution and applies DPO fine-tuning, yielding a PDMS improvement of 2.0–3.5 points.
- **vs. DiffusionDrive**: The prior SOTA under pure imitation learning (88.1); DriveDPO improves by +1.9 through the incorporation of safety signals and DPO.
- **vs. TrajHF**: A similarly RLHF-inspired approach, but TrajHF focuses on driving style preferences whereas DriveDPO targets safety preferences.

## Rating
- Novelty: ⭐⭐⭐⭐ Transferring DPO to autonomous driving and constructing safety-oriented preference pairs are notable contributions
- Experimental Thoroughness: ⭐⭐⭐⭐ NAVSIM SOTA + Bench2Drive closed-loop evaluation + comprehensive ablations
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated and the method is systematically described
- Value: ⭐⭐⭐⭐⭐ Opens a new direction for safety preference alignment in end-to-end autonomous driving

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving](model-based_policy_adaptation_for_closed-loop_end-to-end_autonomous_driving.md)
- [\[NeurIPS 2025\] RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving](raw2drive_reinforcement_learning_with_aligned_world_models_for_end-to-end_autono.md)
- [\[NeurIPS 2025\] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)
- [\[NeurIPS 2025\] Prioritizing Perception-Guided Self-Supervision: A New Paradigm for Causal Modeling in End-to-End Autonomous Driving](prioritizing_perception-guided_self-supervision_a_new_paradigm_for_causal_modeli.md)
- [\[ICCV 2025\] Unraveling the Effects of Synthetic Data on End-to-End Autonomous Driving](../../ICCV2025/autonomous_driving/unraveling_the_effects_of_synthetic_data_on_end-to-end_autonomous_driving.md)

</div>

<!-- RELATED:END -->
