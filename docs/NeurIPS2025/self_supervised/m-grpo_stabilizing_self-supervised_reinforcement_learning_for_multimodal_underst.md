---
title: >-
  [Paper Note] M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization
description: >-
  [NeurIPS 2025][Self-Supervised Learning][self-supervised reinforcement learning] To address the pervasive "policy collapse" problem in self-supervised reinforcement learning from verifiable rewards (SS-RLVR) during extended training, this paper proposes M-GRPO: a framework that employs a momentum model to provide stable pseudo-label targets alongside IQR-based low-entropy trajectory filtering to prevent entropy collapse. Training Qwen3-4B-Base on unlabeled MATH data, the final checkpoint directly surpasses the manually selected best checkpoint of SRT, achieving +2.92% on AIME24 and +5.05% on GPQA.
tags:
  - NeurIPS 2025
  - Self-Supervised Learning
  - self-supervised reinforcement learning
  - policy collapse
  - momentum anchoring
  - GRPO
  - entropy filtering
  - pseudo-labels
date: 2026-05-08
content_hash: e4e400f6845cd4a7
---

# M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2512.13070](https://arxiv.org/abs/2512.13070)
**Code**: [https://github.com/M_GRPO](https://github.com/M_GRPO)
**Area**: Self-Supervised Learning
**Keywords**: self-supervised reinforcement learning, policy collapse, momentum anchoring, GRPO, entropy filtering, pseudo-labels

## TL;DR

To address the pervasive "policy collapse" problem in self-supervised reinforcement learning from verifiable rewards (SS-RLVR) during extended training, this paper proposes M-GRPO: a framework that employs a momentum model to provide stable pseudo-label targets alongside IQR-based low-entropy trajectory filtering to prevent entropy collapse. Training Qwen3-4B-Base on unlabeled MATH data, the final checkpoint directly surpasses the manually selected best checkpoint of SRT, achieving +2.92% on AIME24 and +5.05% on GPQA.

## Background & Motivation

1. **Background**: Reinforcement learning from verifiable rewards (RLVR) is a central paradigm for LLM post-training, but it relies on costly human-annotated data and reward model infrastructure. Recent SS-RLVR methods (e.g., SRT, Intuitor, TTRL, CoReward) attempt to construct pseudo-rewards from the model's own consistency signals (e.g., majority voting), eliminating the need for ground-truth labels.
2. **Limitations of Prior Work**:
   - **Policy collapse**: Reproducing SRT and Intuitor on self-supervised training with the MATH dataset, the authors observe that training reward first rises then sharply or gradually drops, with validation accuracy degrading in tandem—a common failure mode across all SS-RLVR methods.
   - **Increasing rollout count only delays collapse**: Scaling rollouts from 16 to 128 improves peak performance but does not prevent collapse; it merely postpones it.
   - **Entropy collapse**: Policy entropy drops sharply in early training, causing the model to become overconfident prematurely and lock into suboptimal strategies.
3. **Key Challenge**: In self-supervised RL, pseudo-labels are derived from the current policy itself—rapid policy changes destabilize pseudo-labels, which in turn exacerbate policy drift, forming a vicious cycle.
4. **Goal**: Break the vicious cycle of "rapidly changing policy → unstable pseudo-labels → policy collapse" while simultaneously preventing the accompanying entropy collapse.
5. **Key Insight**: Inspired by momentum-based contrastive self-supervised visual representation learning (MoCo)—a slowly evolving momentum model serves as a stable anchor.
6. **Core Idea**: A dual-model framework in which the current policy $\pi_{\theta_q}$ is updated via training while the momentum model $\pi_{\theta_k}$ (EMA parameters) provides stable rollouts; outputs from both models are jointly used for majority voting to generate pseudo-labels. IQR-based adaptive filtering of low-entropy trajectories is applied to maintain exploration diversity.

## Method

### Overall Architecture

Built upon GRPO (Group Relative Policy Optimization). A momentum model $\pi_{\theta_k}$ is introduced, whose parameters are the exponential moving average of the current policy $\pi_{\theta_q}$. For each prompt, rollouts are sampled from both models, aggregated, and subjected to majority voting to obtain pseudo-labels, which are then used to update the current policy via the GRPO objective.

### Key Designs

1. **Momentum-Anchored Self-Supervised RL (M-GRPO)**
   - **Function**: Introduces the momentum model $\pi_{\theta_k}$ into pseudo-label generation to stabilize training targets.
   - **Mechanism**:
     - For each prompt $x$, the current policy samples $M$ rollouts $\{y_i^q\}$ and the momentum model samples $N$ rollouts $\{y_j^k\}$, aggregated into $G = M + N$ candidates.
     - Majority voting selects the highest-consensus answer $y_v$ as the pseudo ground truth.
     - Binary rewards are assigned to the $M$ rollouts of the current policy based on agreement with $y_v$ (agree = 1, disagree = 0).
     - Normalized advantage estimates $\hat{A}_i$ are computed and optimized in the GRPO manner.
   - **Momentum update rule**: $\pi_{\theta_k} \leftarrow m \cdot \pi_{\theta_k} + (1-m) \cdot \pi_{\theta_q}$, with $m = 0.99$.
   - **Design Motivation**:
     - The momentum model evolves slowly, providing temporally consistent rollouts that reduce fluctuation in majority voting outcomes.
     - Analogous to the stabilizing role of the momentum encoder in MoCo for contrastive learning.
     - Enlarges the diversity of the voting pool (two slightly different policy perspectives), improving pseudo-label quality.

2. **IQR-Based Trajectory Entropy Filtering**
   - **Function**: Adaptively removes low-entropy trajectories to prevent premature policy convergence.
   - **Mechanism**:
     - For the $G$ trajectories of each prompt, trajectory-level entropy is computed for each.
     - $Q_1$, $Q_3$, and $\text{IQR} = Q_3 - Q_1$ of the entropy distribution are calculated.
     - Trajectories with entropy below $Q_1 - k \cdot \text{IQR}$ ($k = 0.75$) are flagged as low-entropy outliers and removed.
     - Only filtered trajectories participate in voting and policy optimization.
   - **Design Motivation**:
     - Low-entropy trajectories correspond to overconfident policy outputs; their pseudo-labels are of poor quality and suppress exploration.
     - More flexible than static thresholds (e.g., fixed removal of the lowest 10%)—when most trajectories have high entropy early in training, IQR automatically relaxes; as entropy naturally decreases later, IQR automatically tightens.
     - High-entropy trajectories are retained to maintain policy diversity.

3. **Integrated Training Pipeline**
   - Each iteration: sample batch → dual-model rollout → IQR filtering → majority voting → compute advantages → update current policy → EMA update of momentum model.
   - The momentum model contributes $N = G/4$ rollouts (i.e., the current model contributes 3/4 of total rollouts).

### Loss & Training

- Policy objective: $\mathcal{J}(\theta_q) = \mathbb{E}\left[\sum_{i=1}^{M} \hat{A}_i \log \pi_{\theta_q}(y_i^q | x)\right]$
- KL regularization coefficient: 0.005
- Optimizer: AdamW, learning rate $10^{-6}$, cosine warmup (ratio 0.1)
- Clip ratio: 0.2
- Training temperature: 1.1; evaluation temperature: 0.8
- Maximum response length: 3072

## Key Experimental Results

### Main Results: Qwen3-4B-Base Trained on Unlabeled MATH

| Method | MATH500 | AIME24 | AIME25 | GPQA Dia | GPQA | LiveCode |
|--------|---------|--------|--------|----------|------|----------|
| Base Model | 61.50% | 0.83% | 5.00% | 34.41% | 29.91% | 9.61% |
| SRT_Best (manually selected best ckpt) | 79.20% | 12.50% | 11.67% | 38.26% | 35.04% | 19.69% |
| SRT_Final (last ckpt) | 47.50% | 7.50% | 8.75% | 28.54% | 25.89% | 16.12% |
| **M-GRPO+IQR_Final** | **79.75%** | **14.58%** | **14.17%** | **39.65%** | 35.49% | — |

### Rollout Scaling Analysis (M-GRPO+IQR)

| G (rollout count) | MATH500 | AIME24 | AIME25 | GPQA Dia | MMLU-pro | mbpp |
|-------------------|---------|--------|--------|----------|----------|------|
| 8 | 77.60% | 11.25% | 10.42% | 39.02% | 56.05% | 68.60% |
| 16 | 79.75% | 14.43% | 10.00% | 39.65% | 57.05% | 70.40% |
| 32 | 79.75% | 14.58% | 14.17% | 39.65% | 55.47% | 70.60% |
| 256 | 79.50% | 16.67% | 14.17% | 40.66% | 55.08% | 70.40% |

### Key Findings

- **Policy collapse in SRT is catastrophic**: SRT_Final drops from a peak of 79.20% to 47.50% on MATH500—below the base model (61.50%)—demonstrating that uncontrolled self-supervised training can cause forgetting of pre-trained capabilities.
- **M-GRPO completely eliminates policy collapse**: The final checkpoint directly outperforms the manually selected best SRT checkpoint without any human intervention.
- **Rollout scaling is stable under M-GRPO**: Performance improves consistently from $G=8$ to $G=32$, with diminishing returns beyond $G=256$, indicating that M-GRPO makes full use of rollout information.
- **IQR filtering successfully sustains policy entropy**: In contrast to the sharp entropy drop observed early in SRT training, M-GRPO exhibits a slow and steady entropy decline, avoiding premature convergence.
- **Cross-task generalization**: Training exclusively on MATH yields consistent improvements on GPQA (scientific reasoning), AIME (competition mathematics), and LiveCode (code generation).

## Highlights & Insights

- **Rigorous and reproducible diagnosis of policy collapse**: Beyond identifying the problem, systematic rollout scaling experiments reveal the regularity that "more rollouts delay but cannot prevent collapse," establishing a clear baseline for future research.
- **Elegant analogy from MoCo to RL**: The momentum encoder in self-supervised contrastive learning stabilizes negative sample consistency → the momentum policy in self-supervised RL stabilizes pseudo-label consistency; the cross-domain transfer is natural and well-motivated.
- **Adaptivity of IQR filtering**: Unlike static thresholds (e.g., EdgeGRPO), the IQR method adapts to the dynamic entropy distribution throughout training without additional hyperparameter tuning, representing a concise yet effective engineering contribution.
- **Final checkpoint equals best checkpoint**: The need to manually select optimal checkpoints during deployment is eliminated, substantially lowering the barrier to practical adoption of self-supervised RL.

## Limitations & Future Work

- Experiments are conducted solely on Qwen3-4B-Base; evidence from larger-scale models (e.g., 7B, 14B+) is absent.
- The momentum coefficient $m = 0.99$ is fixed; adaptive scheduling of $m$ (e.g., based on training stage or policy change rate) may yield further improvements.
- The momentum model rollout ratio is fixed at $N = G/4$; the optimal ratio may vary with task and model scale.
- Experiments are limited to the MATH dataset; generalizability to other SS-RLVR scenarios (e.g., code generation, dialogue optimization) remains unverified.
- Sensitivity analysis of the IQR coefficient $k = 0.75$ is insufficient; ablation studies examining the effect of different $k$ values are lacking.
- No comparison with other training stabilization methods (e.g., replay buffers, conservative policy updates) is provided.
- The dual-model architecture introduces additional inference overhead (the momentum model requires sampling $N$ additional trajectories), necessitating trade-off considerations in resource-constrained settings.

## Related Work & Insights

- **vs. SRT / Intuitor / TTRL**: These methods rely on the consistency of a single model's own outputs as pseudo-labels; rapid policy drift degrades pseudo-label quality. M-GRPO breaks this vicious cycle by introducing a slowly evolving momentum model.
- **vs. DAPO / GRPO**: DAPO and GRPO are designed for supervised RLVR and rely on ground-truth reward signals; M-GRPO extends them to the self-supervised setting, with the core innovation being a mechanism for stable pseudo-label generation.
- **vs. MoCo (self-supervised visual representation learning)**: MoCo uses a momentum encoder to stabilize the negative sample queue → M-GRPO uses a momentum policy to stabilize majority voting outcomes; both represent successful applications of the same paradigm in different domains.
- **vs. EdgeGRPO (static entropy filtering)**: EdgeGRPO applies a fixed proportion filter for low-entropy trajectories; M-GRPO's IQR method adaptively adjusts the threshold based on the actual entropy distribution, yielding greater robustness.
- **Broader Implications**: The momentum-anchored paradigm generalizes to any iterative optimization process relying on self-generated signals, including self-play, self-distillation, and iterative refinement.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Introducing the MoCo momentum mechanism into self-supervised RL is a natural yet effective innovation; IQR filtering is a more incremental contribution.
- **Experimental Thoroughness**: ⭐⭐⭐ Diagnostic experiments (collapse reproduction, rollout scaling) are strong, but model scale and dataset diversity are limited.
- **Writing Quality**: ⭐⭐⭐⭐ Problem diagnosis is clear, method motivation is well-articulated, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐ Provides a principled solution to training stability in self-supervised RL with meaningful practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning](t-regs_minimum_spanning_tree_regularization_for_self-supervised_learning.md)
- [\[NeurIPS 2025\] Understanding Ice Crystal Habit Diversity with Self-Supervised Learning](understanding_ice_crystal_habit_diversity_with_self-supervised_learning.md)
- [\[NeurIPS 2025\] Adv-SSL: Adversarial Self-Supervised Representation Learning with Theoretical Guarantees](adv-ssl_adversarial_self-supervised_representation_learning_with_theoretical_gua.md)
- [\[NeurIPS 2025\] Continuous Subspace Optimization for Continual Learning (CoSO)](continuous_subspace_optimization_for_continual_learning.md)
- [\[NeurIPS 2025\] STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking](starformer_semi-supervised_task-informed_representation_learning_via_dynamic_att.md)

</div>

<!-- RELATED:END -->
