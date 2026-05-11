---
title: >-
  [Paper Note] RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning
description: >-
  [ICLR 2026][Reinforcement Learning][Multimodal Large Language Models] This paper proposes the RewardMap framework, which addresses the sparse reward problem in fine-grained visual reasoning through difficulty-aware detail reward design and a multi-stage RL curriculum that progresses from simple perception to complex reasoning.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Multimodal Large Language Models
  - Visual Reasoning
  - Sparse Rewards
  - Multi-Stage RL
  - Metro Route Planning
date: 2026-05-08
content_hash: aada1be69448ae97
---

# RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning

**Conference**: ICLR 2026
**arXiv**: [2510.02240](https://arxiv.org/abs/2510.02240)
**Code**: [Project Page](https://fscdc.github.io/RewardMap)
**Area**: Reinforcement Learning
**Keywords**: Multimodal Large Language Models, Visual Reasoning, Sparse Rewards, Multi-Stage RL, Metro Route Planning

## TL;DR

This paper proposes the RewardMap framework, which addresses the sparse reward problem in fine-grained visual reasoning through difficulty-aware detail reward design and a multi-stage RL curriculum that progresses from simple perception to complex reasoning.

## Background & Motivation

Fine-grained visual reasoning (e.g., metro route planning) poses a core challenge for multimodal large language models (MLLMs). The ReasonMap benchmark reveals that even state-of-the-art MLLMs struggle with spatial reasoning in structured, information-dense visual scenes.

Directly applying standard RL methods (e.g., GRPO) to such complex tasks encounters a **sparse reward bottleneck**:
- Success signals are only provided at the end of long reasoning chains (final answer correct/incorrect)
- Task difficulty further amplifies sparsity — most sampled rollouts yield reward $r_i \approx 0$
- In GRPO, when all samples fail, the group-relative advantage $\hat{A}_i$ approaches zero, resulting in weak gradient signals and difficult convergence

Conventional SFT provides dense supervision but cannot equip models with long-chain decision-making capabilities. The root cause is a mismatch between task complexity and the density of supervision signals.

The paper's starting point is two-fold: (1) constructing the ReasonMap-Plus dataset as a dense-reward cold-start source; and (2) designing a multi-stage RL curriculum progressing from easy to hard, transitioning from perception to reasoning.

## Method

### Overall Architecture

RewardMap consists of two core components: (1) difficulty-aware reward design that augments format and correctness rewards with detail rewards; and (2) a multi-stage GRPO training curriculum that progresses from simple VQA to complex route planning.

### Key Designs

1. **ReasonMap-Plus Dataset Construction**:

    - Function: Constructs 4,018 VQA questions covering 5 extended question types across 30 cities in 13 countries.
    - Mechanism: Designs three major question categories — global counting, local counting, and judgment — with answers automatically generated from Metro Data.
    - Design Motivation: VQA questions are simple and reward-dense, making them suitable as an RL cold-start to develop the model's foundational visual understanding.

2. **Difficulty-Aware Detail Rewards**:

    - Function: Introduces partial-credit rewards on top of correctness rewards.
    - Mechanism: $R = W_{\text{difficulty}}(R_{\text{format}} + R_{\text{correctness}} + \alpha \times R_{\text{detail}})$
    - Detail rewards assign credit/penalty separately for start/end stations, line names, transfer stations, and segment counts.
    - Difficulty weight $W_{\text{difficulty}} = W_{\text{map}} + W_{\text{question}}$, integrating map complexity and the number of transfers.
    - Design Motivation: Alleviates sparse rewards in planning tasks, enabling learning from partially correct information even when the final answer is wrong.

3. **Multi-Stage GRPO Curriculum**:

    - Function: Divides training into multiple stages following a global curriculum principle.
    - Mechanism: Judgment questions → Counting questions → Planning questions (visual understanding → visual reasoning). Samples are shuffled randomly within each stage.
    - Design Motivation: (1) Lower-level tasks have dense rewards, supporting effective cold-start; (2) Gradually bridging perception and reasoning avoids training collapse when facing difficult tasks directly; (3) Local randomness prevents overfitting to a fixed curriculum trajectory.

### Loss & Training

The standard policy gradient objective of GRPO is used, driven by group-relative advantages. The key distinctions lie in the reward function design (three-level rewards with difficulty weighting) and the data scheduling strategy (multi-stage curriculum). The cold-start phase uses RL directly rather than SFT, ensuring alignment between the reward signal and task objective from the outset.

## Key Experimental Results

### Main Results (Qwen2.5-VL-7B-Instruct)

| Method | ReasonMap Weighted Acc. (S/L) | ReasonMap-Plus Weighted Acc. |
|--------|-------------------------------|------------------------------|
| Base Model | 13.28% / 7.12% | 44.21% |
| +RL (GRPO) | 26.22% / 26.04% | 44.64% |
| +RL (REINFORCE++) | 27.17% / 27.60% | — |
| +RewardMap (Full) | **Best** | **Best** |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| Format + Correctness reward only | Baseline performance | Difficult to learn under sparse rewards |
| + Detail reward | Significant improvement | Partial credit alleviates sparsity |
| + Difficulty weight | Further improvement | Hard examples contribute more learning signal |
| + Multi-stage curriculum | Best performance | Cold-start strategy is effective |

### Key Findings
- Models trained with RewardMap achieve an average improvement of 3.47% across 6 external benchmarks, demonstrating strong generalization.
- RL cold-start outperforms SFT cold-start, avoiding the overfitting and cognitive rigidity induced by SFT.
- Among reference models, GPT-5 achieves 59.98%/62.50% on ReasonMap, reflecting the extreme difficulty of this task.
- Seed1.5-VL and GPT-4o achieve 73.58% and 64.42% on ReasonMap-Plus, respectively.

## Highlights & Insights

- **Valuable problem formulation**: Metro route planning serves as a natural testbed for MLLM visual reasoning, combining practical utility with scientific significance.
- **Using RL instead of SFT for cold-start** is an insightful design choice that avoids the mismatch between reward signals and the loss function.
- **Detail reward design is elegant**: The structured nature of planning tasks (start station, end station, transfer stations, etc.) allows independent verification of sub-components, enabling principled reward decomposition.

## Limitations & Future Work

- The detail reward design relies on task-specific structural information; adapting it to other visual reasoning tasks requires redesign.
- Hyperparameters for difficulty weighting ($\gamma_e, \gamma_m, \gamma_h, \beta_0, \beta_1$) must be specified in advance.
- Experiments are currently conducted only on the Qwen2.5-VL model family; generalization to other architectures remains unexplored.

## Related Work & Insights

- ReasonMap (Feng et al., 2025b) serves as the benchmark and data foundation for this work.
- GRPO (Shao et al., 2024) provides the RL optimization framework.
- Curriculum RL (Parashar et al., 2025) inspires the multi-stage design via its easy-to-hard strategy.
- Insight: For complex reasoning tasks, reward engineering may be more critical than algorithmic innovation.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of using multi-stage RL cold-start in place of SFT is novel, though individual components are relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark evaluation including external generalization, with ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear and framework diagrams are well-presented.
- Value: ⭐⭐⭐⭐ Provides a practical solution for RL training of MLLMs in visual reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DiVE-k: Differential Visual Reasoning for Fine-grained Image Recognition](dive-k_differential_visual_reasoning_for_fine-grained_image_recognition.md)
- [\[CVPR 2026\] Specificity-aware Reinforcement Learning for Fine-grained Open-world Classification](../../CVPR2026/reinforcement_learning/specificity-aware_reinforcement_learning_for_fine-grained_open-world_classificat.md)
- [\[ICLR 2026\] LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)
- [\[ICLR 2026\] MergeMix: A Unified Augmentation Paradigm for Visual and Multi-Modal Understanding](mergemix_a_unified_augmentation_paradigm_for_visual_and_multi-modal_understandin.md)
- [\[CVPR 2026\] MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning](../../CVPR2026/reinforcement_learning/msrl_scaling_generative_multimodal_reward_modeling.md)

</div>

<!-- RELATED:END -->
