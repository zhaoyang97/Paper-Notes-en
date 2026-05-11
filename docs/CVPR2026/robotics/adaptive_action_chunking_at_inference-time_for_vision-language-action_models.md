---
title: >-
  [Paper Note] Adaptive Action Chunking at Inference-time for Vision-Language-Action Models
description: >-
  [CVPR 2026][Robotics][Action chunking] This paper proposes Adaptive Action Chunking (AAC), a strategy that leverages action entropy as a signal to dynamically determine the optimal chunk size at inference time, requiring no additional training or architectural modification. AAC consistently improves task success rates of GR00T N1.5 and π0.5 on benchmarks including RoboCasa and LIBERO.
tags:
  - CVPR 2026
  - Robotics
  - Action chunking
  - VLA models
  - adaptive inference
  - action entropy
  - robotic manipulation
date: 2026-05-08
content_hash: 5e26267fdf770436
---

# Adaptive Action Chunking at Inference-time for Vision-Language-Action Models

**Conference**: CVPR 2026
**arXiv**: [2604.04161](https://arxiv.org/abs/2604.04161)
**Code**: [https://lance-lot.github.io/adaptive-chunking.github.io/](https://lance-lot.github.io/adaptive-chunking.github.io/)
**Area**: Robotics / VLA Models
**Keywords**: Action chunking, VLA models, adaptive inference, action entropy, robotic manipulation

## TL;DR
This paper proposes Adaptive Action Chunking (AAC), a strategy that leverages action entropy as a signal to dynamically determine the optimal chunk size at inference time, requiring no additional training or architectural modification. AAC consistently improves task success rates of GR00T N1.5 and π0.5 on benchmarks including RoboCasa and LIBERO.

## Background & Motivation

**Background**: Action chunking — executing a sequence of actions as a unit without intermediate replanning — is a key technique for improving robotic manipulation in VLA models. Current mainstream VLA models (GR00T N1.5, π0, SmolVLA) all adopt fixed chunk sizes.

**Limitations of Prior Work**: (1) Large chunks → poor responsiveness, unable to adapt to new observations in time; (2) Small chunks → mode-jumping, causing jitter from inter-chunk discontinuities; (3) **The optimal chunk size varies across tasks** (experiments show that for the same model, the optimal chunk size ranges from 4 to 16 across different RoboCasa tasks). Existing methods such as ACT apply EMA smoothing and BID searches for the optimal chunk, but both use a fixed size.

**Key Challenge**: A dynamic balance between consistency (large chunks) and reactivity (small chunks) is needed, yet fixed chunk sizes cannot achieve this.

**Key Insight**: Action entropy reflects prediction uncertainty — low entropy → high reliability → larger chunks are feasible; high entropy → low reliability → smaller chunks with frequent replanning are preferred.

**Core Idea**: Compute the average action entropy corresponding to different chunk sizes, then identify the point of maximum discrete difference to determine the optimal chunk size.

## Method

### Overall Architecture
At each observation timestep during inference: (1) sample $N$ candidate action chunks in parallel → (2) compute continuous and discrete action entropy for each timestep → (3) identify the point of maximum difference in the average entropy curve → (4) determine the optimal chunk size → (5) execute the first $h^*$ actions. **No additional training or architectural modification is required.**

### Key Designs

1. **Action Entropy Computation**:

    - **Continuous actions** (translation/rotation): Gaussian differential entropy $E_t = \frac{1}{2}\log[(2\pi e)^d \det(\Sigma_t)]$, with the covariance matrix estimated from $N$ candidate chunks.
    - **Discrete actions** (gripper): Shannon entropy $E_{dis} = -\sum p(a)\log p(a)$, with probabilities estimated from empirical frequencies.
    - Average action entropy: $\bar{E}_h = \frac{1}{h}\sum_{i=t}^{t+h-1}\sum_{j \in \{t,r,g\}} E_j^i$

2. **Adaptive Chunk Size Selection**:

    - Function: Identify the "elbow point" of the average entropy curve.
    - Mechanism: $h^* = \max(\arg\max_h(\bar{E}_{h+1} - \bar{E}_h), \xi)$
    - $\xi$ is a minimum chunk lower bound, ensuring a minimum action magnitude and computational efficiency.
    - Design Motivation: The point of maximum difference marks where further increasing the chunk size leads to a sharp rise in uncertainty — the optimal switching point balancing consistency and reactivity.

3. **Inference-time Behavioral Patterns**:

    - Near target objects → high entropy → small chunks + frequent replanning → fine-grained control.
    - During transport phases → low entropy → large chunks → efficient movement.
    - This aligns fully with human intuition and is confirmed through visualization.

### Loss & Training
AAC requires no training. Entropy is estimated directly at inference time from multiple samples drawn from the flow-matching action head. The method is compatible with all diffusion/flow-matching-based VLA models.

## Key Experimental Results

### Main Results (RoboCasa + LIBERO)

| Method | RoboCasa Avg | LIBERO Avg |
|--------|-------------|------------|
| GR00T (h=16, default) | 59.7% | 94.1% |
| GR00T (h=2) | 47.0% | 90.2% |
| GR00T (h=4) | 56.2% | 92.6% |
| GR00T (h=8) | 61.2% | 94.7% |
| **GR00T + AAC** | **62.0%** | **95.0%** |

LIBERO-Long (hardest subset): 88.8% → **92.8%** (+4.0%)

### Cross-Backbone Validation

| Method | LIBERO Avg |
|--------|------------|
| π0.5 (baseline) | 97.0% |
| **π0.5 + AAC** | **97.9%** |

### OOD Robustness (LIBERO-Pro with Position Perturbation)

| Perturbation Level | GR00T | GR00T+AAC |
|--------------------|-------|-----------|
| ×0.2 | baseline | +improvement |
| ×0.3 | baseline | +improvement |
| ×0.4 | baseline | +improvement |

### Key Findings
- **No single fixed chunk size is optimal across all tasks**: LIBERO-Spatial optimal $h=4$, LIBERO-Goal optimal $h=16$.
- AAC outperforms the average of all fixed chunk sizes without any manual tuning.
- Improvement is most pronounced on long-horizon tasks (LIBERO-Long, +4%), where reactivity demands are highest.
- The temporal distribution of chunk sizes closely aligns with task semantic phases: transport → large chunks, manipulation → small chunks.

## Highlights & Insights
- **Zero-training inference optimization**: AAC operates entirely at inference time with no model architecture changes or retraining — a true plug-and-play solution.
- **Action entropy as a universal uncertainty measure**: A unified entropy computation framework spanning continuous and discrete action spaces, generalizable to diverse robot morphologies (single-arm, dual-arm, humanoid).
- **Alignment with human intuition**: Visualization analysis shows that chunk sizes perfectly correspond to task semantic phases — coarse operations use large chunks, fine operations use small ones — validating the physical plausibility of the method.

## Limitations & Future Work
- Parallel sampling of $N$ candidate chunks introduces additional inference latency (larger $N$ improves estimation accuracy but increases cost).
- The maximum-difference-point strategy is heuristic and does not guarantee global optimality.
- The minimum chunk lower bound $\xi$ is a hyperparameter that may require task-specific tuning.
- Validation is currently limited to tabletop manipulation; more complex mobile manipulation scenarios (e.g., navigation combined with manipulation) remain to be explored.

## Related Work & Insights
- **vs. ACT (EMA smoothing)**: ACT generates a new chunk at each step and fuses it via EMA, but the chunk size remains fixed. AAC adaptively selects the chunk size.
- **vs. BID/TV-BID**: BID selects the best chunk from multiple candidates but with a fixed size; AAC simultaneously adapts the chunk size.
- **vs. RL-based adaptive methods**: These require additional training and task-specific reward signals, whereas AAC requires no training.

## Rating
- Novelty: ⭐⭐⭐⭐ — Entropy-driven chunk selection is concise and effective, though the underlying principle is relatively intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across multiple benchmarks, backbones, OOD settings, real-robot experiments, and qualitative analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear motivation, clean methodology, and excellent visualizations.
- Value: ⭐⭐⭐⭐⭐ — Direct practical value for VLA deployment; zero-overhead, plug-and-play.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_active_perception_manipulation_vla_roboti.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[ICLR 2026\] Real-Time Robot Execution with Masked Action Chunking](../../ICLR2026/robotics/real-time_robot_execution_with_masked_action_chunking.md)
- [\[CVPR 2026\] Boosting Vision-Language-Action Finetuning with Feasible Action Neighborhood Prior](boosting_vision-language-action_finetuning_with_feasible_action_neighborhood_pri.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation](lada_robotic_manipulation.md)

</div>

<!-- RELATED:END -->
