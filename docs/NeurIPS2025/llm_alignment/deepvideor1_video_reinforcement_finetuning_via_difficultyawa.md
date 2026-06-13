---
title: >-
  [Paper Note] DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO
description: >-
  [NeurIPS 2025][LLM Alignment][Video reasoning] This paper proposes DeepVideo-R1, which reformulates GRPO as Reg-GRPO that directly regresses advantage values (eliminating clipping/min safeguards)…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "Video reasoning"
  - "reinforcement fine-tuning"
  - "GRPO"
  - "regression objective"
  - "difficulty-aware augmentation"
date: 2026-05-08
content_hash: fc1b3331c1213652
---

# DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO

**Conference**: NeurIPS 2025
**arXiv**: [2506.07464](https://arxiv.org/abs/2506.07464)  
**Code**: [GitHub](https://github.com/mlvlab/DeepVideoR1)  
**Area**: LLM Alignment / Video Large Language Models
**Keywords**: Video reasoning, reinforcement fine-tuning, GRPO, regression objective, difficulty-aware augmentation

## TL;DR

This paper proposes DeepVideo-R1, which reformulates GRPO as Reg-GRPO that directly regresses advantage values (eliminating clipping/min safeguards), and mitigates the vanishing advantage problem via difficulty-aware data augmentation, achieving up to 10.1 percentage points improvement over standard GRPO on video reasoning tasks.

## Background & Motivation

### State of the Field

**Background**: RL-based post-training (e.g., GRPO) has proven effective for enhancing LLM reasoning, yet its application to Video Large Language Models (VideoLLMs) remains underexplored.

### Limitations of Prior Work

**Limitations of Prior Work**: Applying GRPO to VideoLLMs faces two critical issues:

### Root Cause

**Key Challenge**: **Dependence on safeguard mechanisms**: PPO-style clipping and min operations produce zero gradients when the policy deviates excessively, impeding exploration and convergence.

### Starting Point

**Key Insight**: **Vanishing advantage**: When samples are too easy or too hard, all responses within a group receive identical rewards, causing zero advantage values and loss of training signal.

### Remarks

**Remarks**: Video reasoning requires complex spatiotemporal semantic understanding, making both issues particularly pronounced in video tasks.

### Remarks

**Remarks**: Existing work has primarily focused on reward function design, leaving algorithmic improvements to GRPO itself relatively underexplored.

## Method

### Overall Architecture

DeepVideo-R1 comprises two key innovations: (1) Reg-GRPO reformulates the GRPO objective to directly regress group-relative advantage values, eliminating safeguard mechanisms such as clipping and min operations; (2) difficulty-aware data augmentation dynamically adjusts inputs based on sample difficulty to ensure diverse reward signals.

### Key Designs

1. **Regressive GRPO (Reg-GRPO)**:
    - **Function**: Transforms the RL objective from PPO-style optimization to direct regression of advantage values.
    - **Mechanism**: Leverages a reparameterization of the closed-form solution to the KL-constrained RL objective, defining the predicted advantage as $\hat{A}_\theta^{(i)} = \frac{\rho(\mathbf{x}, \mathbf{y}^{(i)}) - \mu_\rho}{\sigma_\rho}$, where $\rho = \log \frac{\pi_\theta}{\pi_{\theta_{old}}}$, and minimizes MSE against the target advantage.
    - **Design Motivation**: The regression loss is inherently free of clipping truncation, and normalization naturally eliminates the partition function $Z(\mathbf{x})$.

2. **Difficulty-aware Data Augmentation**:
    - **Function**: Dynamically adjusts video-text inputs based on sample difficulty.
    - **Mechanism**: Uses the mean reward from a replay buffer as a reference to compute difficulty $\Delta_\mathcal{R}(\mathbf{x})$.
    - **Design Motivation**: Samples of moderate difficulty yield the most diverse reward distributions, ensuring effective gradient signals.

3. **Bidirectional Difficulty Adjustment**:
    - **Reducing difficulty** (hard samples): Partial reasoning clues extracted from successful reasoning trajectories are injected into the prompt, with intensity adaptively scaled by difficulty.
    - **Increasing difficulty** (easy samples): Gaussian noise or masking is applied to video frames, with intensity proportional to the degree of easiness.

### Loss & Training

$$\mathcal{L}_{\text{Reg-GRPO}}(\theta) = \mathbb{E}\left[(\hat{A}^{(i)} - \hat{A}_\theta^{(i)})^2 - \beta \mathbb{D}_{KL}[\pi_\theta || \pi_{ref}]\right]$$

- The KL divergence constraint prevents excessive deviation from the reference model.
- A replay buffer storing the most recent $W$ steps of data is used for dynamic difficulty baseline computation.

## Key Experimental Results

### Main Results

Performance on SEED-Bench-R1 validation set and LongVideoBench:

| Method | SEED-Bench-R1 (Acc) | LongVideoBench |
|--------|---------------------|----------------|
| Qwen2.5-VL-7B (SFT) | 55.4 | 57.3 |
| + GRPO | 55.8 | 54.1 |
| + Reg-GRPO | 63.2 | 59.4 |
| + **DeepVideo-R1** | **65.9** | **60.7** |

DeepVideo-R1 achieves a 10.1-point gain over GRPO on SEED-Bench-R1.

### Ablation Study

- **Reg-GRPO vs. GRPO**: Reg-GRPO consistently outperforms GRPO across all benchmarks with faster convergence.
- **Contribution of difficulty augmentation**: Provides an additional 2.3-point gain on top of Reg-GRPO.
- **Difficulty reduction vs. difficulty increase**: Each component is individually effective; combining both yields the best results.
- **Zero-advantage ratio**: Difficulty augmentation reduces the zero-advantage ratio from approximately 30% to approximately 10%.

### Key Findings

- The regression objective yields more stable gradients, eliminating zero-gradient regions caused by clipping.
- Difficulty-aware augmentation directly addresses the root cause of vanishing advantage — zero reward variance within a group.
- Consistent improvements on both in-distribution and out-of-distribution tasks indicate enhanced generalization.

## Highlights & Insights

- The derivation of Reg-GRPO is concise: starting from the closed-form RL solution, the partition function is naturally eliminated through group normalization.
- Difficulty-aware augmentation constitutes an RL-native implementation of curriculum learning.
- Extracting reasoning clues from successful trajectories as a difficulty-reduction strategy represents an interesting self-guided approach.
- The method is not restricted to the video domain and is applicable to any setting employing GRPO.

## Limitations & Future Work

- Validation is limited to 7B-scale models; scaling behavior at larger sizes remains unknown.
- Reasoning clue extraction requires additional generation steps, increasing data preparation cost.
- Comparisons with other alignment methods such as DPO are absent.
- Sensitivity analysis of the replay buffer window size $W$ is insufficiently thorough.

## Related Work & Insights

- Reg-GRPO shares conceptual similarities with REBEL (direct reward regression) but is more naturally motivated in the group-relative setting.
- The approach is complementary to NoisyRollout: NoisyRollout improves exploration diversity while this work improves the optimization objective.
- The difficulty-aware paradigm can be integrated with VideoLLM RL works such as VideoChat-R1 and TimeZero.

## Rating

- ⭐⭐⭐⭐ — The RL algorithm improvement is theoretically well-grounded and empirically significant, and the difficulty augmentation strategy is practical; however, validation at larger scales remains limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay](improving_data_efficiency_for_llm_reinforcement_fine-tuning_through_difficulty-t.md)
- [\[NeurIPS 2025\] Mechanism Design for LLM Fine-tuning with Multiple Reward Models](mechanism_design_for_llm_fine-tuning_with_multiple_reward_models.md)
- [\[NeurIPS 2025\] Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs](attack_via_overfitting_10-shot_benign_fine-tuning_to_jailbreak_llms.md)
- [\[NeurIPS 2025\] Strategyproof Reinforcement Learning from Human Feedback](strategyproof_reinforcement_learning_from_human_feedback.md)
- [\[ACL 2026\] Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO](../../ACL2026/llm_alignment/mitigating_selection_bias_in_large_language_models_via_permutation-aware_grpo.md)

</div>

<!-- RELATED:END -->
