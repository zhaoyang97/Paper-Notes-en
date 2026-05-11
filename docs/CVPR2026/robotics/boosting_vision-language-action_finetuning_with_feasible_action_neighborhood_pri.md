---
title: >-
  [Paper Note] Boosting Vision-Language-Action Finetuning with Feasible Action Neighborhood Prior
description: >-
  [CVPR 2026][Robotics][VLA finetuning] This paper proposes a Feasible Action Neighborhood (FAN) regularizer that shapes the output distribution of VLA models into a Gaussian form matching physical action tolerances. The a…
tags:
  - "CVPR 2026"
  - "Robotics"
  - "VLA finetuning"
  - "feasible action neighborhood"
  - "Gaussian regularization"
  - "reinforcement finetuning"
  - "sample efficiency"
date: 2026-05-08
content_hash: 2c5db1e7d278287a
---

# Boosting Vision-Language-Action Finetuning with Feasible Action Neighborhood Prior

**Conference**: CVPR 2026
**arXiv**: [2604.01570](https://arxiv.org/abs/2604.01570)
**Code**: None
**Area**: Robotic Manipulation / VLA Finetuning
**Keywords**: VLA finetuning, feasible action neighborhood, Gaussian regularization, reinforcement finetuning, sample efficiency

## TL;DR
This paper proposes a Feasible Action Neighborhood (FAN) regularizer that shapes the output distribution of VLA models into a Gaussian form matching physical action tolerances. The approach consistently improves success rate, generalization, and sample efficiency under both SFT and RFT finetuning paradigms (RFT requires only 1/3 of training steps to reach 90% success rate).

## Background & Motivation
**Background**: VLA models (e.g., OpenVLA, $\pi_0$) unify visual perception, language understanding, and low-level control into a single model, performing autoregressive prediction over discretized action tokens. In practice, these models are typically pretrained and then finetuned via SFT or RFT.

**Limitations of Prior Work**: Existing VLA training methods directly adopt language model training paradigms (one-hot cross-entropy or PPO), yet **physical actions inherently possess tolerances**—nearby actions may produce entirely equivalent task progress. This fundamental discrepancy has been overlooked.

**Key Challenge**: SFT collapses probability mass onto a single demonstrated action (overfitting), leading to poor generalization; RFT can broaden the distribution but is extremely sample-inefficient, requiring extensive exploration to implicitly discover the tolerance structure.

**Goal**: Explicitly exploit the tolerance structure of the physical action space during VLA finetuning.

**Key Insight**: Formalize the concept of a "Feasible Action Neighborhood" (FAN) and observe that the shape of the policy distribution (sharp vs. smooth) is highly correlated with generalization performance.

**Core Idea**: Introduce a FAN-guided Gaussian regularizer that reshapes the policy distribution from an "overconfident spike" into a "smooth tolerance neighborhood," applicable to both SFT and RFT without modifying model architecture.

## Method

### Overall Architecture
VLA model → predicts action distribution $\pi(a|s)$ at each state $s$ → FAN regularizer pulls this distribution toward a target Gaussian $\mathcal{N}(\mu(s), \Sigma)$ → autoregressive discrete decoding remains unchanged.

### Key Designs

1. **Feasible Action Neighborhood (FAN) Definition**:
    $\mathbb{N}_\delta(s) \subseteq \{a \in A: Q(s, a^*(s)) - Q(s, a) \leq \delta\}$
   This denotes the set of actions whose Q-values are close to optimal for a given state $s$. Physical manipulation tasks naturally exhibit non-trivial FANs.
    - The policy distribution $\pi(a|s)$ serves as a practical observable proxy for the FAN—sharp distribution = small FAN = poor generalization; smooth distribution = large FAN = good generalization.
    - **Design Motivation**: Empirical observation reveals a strong correlation between distribution shape and success rate.

2. **FAN-SFT (Supervised Finetuning Regularization)**:
    $\mathcal{L}_{\text{FAN-SFT}} = -\frac{1}{n}\sum_{i,t}\left(\log\pi_\theta(a_t^i|s_t^i, l^i) + \alpha D_{\text{KL}}(\pi_\theta(\cdot|s_t^i)\|\mathcal{N}(\cdot|\mu(s_t^i), \Sigma(s_t^i)))\right)$
    - The covariance is dynamically defined as the policy's own variance: $\Sigma(s) = \text{diag}(\sum_a \pi(a|s)(a-\mu(s))^2)$
    - **Design Motivation**: SFT is inherently stable and thus amenable to dynamic targets; the adaptive covariance encourages the policy to adopt a Gaussian shape consistent with its current geometry.

3. **FAN-PPO (Reinforcement Finetuning Regularization)**:
    $\max_\pi \mathbb{E}[\frac{\pi(a|s)}{\pi_t(a|s)}A^{\pi_t}] - \alpha \mathbb{E}[D_{\text{KL}}(\pi\|\mathcal{N}(\mu(s), \sigma^2 I))]$
    - Uses a fixed covariance $\Sigma = \sigma^2 I$ (hyperparameter controlling target FAN size).
    - **Closed-form optimal policy**: $\pi_{t+1} \propto \mathcal{N}^{\frac{\alpha}{\alpha+\beta^*}} \cdot \pi_t^{\frac{\beta^*}{\alpha+\beta^*}} \cdot \exp(\frac{Q}{\alpha+\beta^*})$
    - The new policy is a geometric interpolation between the old policy and the target Gaussian, reweighted by Q-values.
    - **Design Motivation**: RFT requires a stable target; the fixed covariance provides a consistent anchor. $\alpha$ controls the Gaussian pull, and $\beta^*$ controls conservatism.

### Loss & Training
- The FAN regularizer is added to the standard SFT/PPO loss, with $\alpha$ controlling the regularization weight.
- OpenVLA: $\sigma=0.3, \alpha=1.0$; OpenVLA-OFT: $\sigma=0.2, \alpha=0.1$
- Advantage functions are estimated via GAE; the value network is trained with MSE loss.

## Key Experimental Results

### Main Results (ManiSkill, Success Rate %)

| Method | In-Dist. | OOD-Visual | OOD-Semantic | OOD-Execution | OOD Avg. |
|--------|----------|------------|--------------|---------------|----------|
| OpenVLA + SFT | 78.1 | 76.6 | 57.4 | 40.4 | 58.1 |
| **OpenVLA + FAN-SFT** | **89.8** | **81.7** | **63.5** | **44.8** | **63.3** |
| Gain | +11.7 | +5.1 | +6.1 | +4.4 | +5.2 |
| OpenVLA + PPO | 95.9 | 80.1 | 79.7 | 85.8 | 81.9 |
| **OpenVLA + FAN-PPO** | **97.4** | **85.0** | **86.7** | **92.6** | **88.1** |
| Gain | +1.5 | +4.9 | +7.0 | +6.9 | +6.2 |

### Ablation Study (Sample Efficiency)

| Configuration | Steps to Reach 90% Success Rate | Note |
|---------------|---------------------------------|------|
| OpenVLA + PPO | ~X steps | Baseline |
| OpenVLA + FAN-PPO | **~X/3 steps** | Requires only ~1/3 of training steps |

| Data Size | SFT | FAN-SFT | Gain |
|-----------|-----|---------|------|
| 1.6K | Lower | Higher | FAN is consistently effective across data scales |
| 16K | Higher | **Even higher** | Further gains persist at large data scale |

### Key Findings
- FAN-PPO yields the largest gains in OOD-Execution scenarios (+6.9–11.1%), demonstrating significantly enhanced action-space generalization.
- The most striking result is sample efficiency—FAN-PPO reaches equivalent performance with only 1/3 of the baseline training steps.
- Real-robot experiments further validate FAN-SFT's spatial generalization, achieving higher success rates at unseen positions.
- FAN is distinct from maximum entropy regularization: the latter encourages unstructured exploration, whereas FAN applies structured regularization grounded in physical priors.

## Highlights & Insights
- The formalization of FAN is conceptually simple yet profound—it reveals a fundamental mismatch between language model training objectives and the physical action space.
- The regularizer requires no architectural changes and does not alter the decoding procedure, making it truly plug-and-play.
- The derivation of the closed-form optimal policy (Proposition 1) provides clear theoretical understanding.
- The unified treatment of both SFT and RFT paradigms ensures broad applicability.

## Limitations & Future Work
- The Gaussian assumption may be overly simplistic—real FANs may be non-convex or multimodal.
- $\sigma$ requires per-task tuning; adaptively learning FAN size is an important future direction.
- Validation is currently limited to simulation and simple real-world tasks; complex dexterous manipulation remains to be explored.
- Combining FAN with value functions to dynamically estimate per-state tolerance is a promising avenue.

## Related Work & Insights
- VLA models such as RT-2 and OpenVLA directly adopt language training paradigms; this paper exposes a fundamental limitation of that approach.
- RFT methods such as RL4VLA and GRPO optimize from the reward side, while FAN optimizes from the geometry of the action space—the two are complementary.
- Label smoothing also regularizes distributions but does not leverage physical structure, and is therefore far less effective than FAN.
- Takeaway: "Physical priors" in robotic control should be more actively incorporated into learning objectives.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The FAN concept is original and insightful, with tight integration of theory and practice.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers SFT and RFT paradigms, multiple VLA backbones, in-distribution and OOD settings, sample efficiency, and real-robot validation.
- Writing Quality: ⭐⭐⭐⭐⭐ The chain from motivation to theory to experiments is complete and coherent.
- Value: ⭐⭐⭐⭐⭐ Establishes a new foundational principle for VLA finetuning with broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] Action–Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation](actiongeometry_prediction_with_3d_geometric_prior.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation](lada_robotic_manipulation.md)
- [\[CVPR 2026\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_active_perception_manipulation_vla_roboti.md)

</div>

<!-- RELATED:END -->
