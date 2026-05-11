---
title: >-
  [Paper Note] Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training
description: >-
  [ACL 2026][Reinforcement Learning][Data mixing] This paper proposes the Data Mixing Agent, the first model-based end-to-end domain re-weighting framework. By training a small agent on a large collection of data mixing tr…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Data mixing"
  - "domain re-weighting"
  - "continual pre-training"
  - "catastrophic forgetting"
date: 2026-05-08
content_hash: bc3ae7e0060cce05
---

# Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training

**Conference**: ACL 2026
**arXiv**: [2507.15640](https://arxiv.org/abs/2507.15640)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: Data mixing, domain re-weighting, continual pre-training, reinforcement learning, catastrophic forgetting

## TL;DR

This paper proposes the Data Mixing Agent, the first model-based end-to-end domain re-weighting framework. By training a small agent on a large collection of data mixing trajectories via CQL-based reinforcement learning, the framework learns generalizable data mixing heuristics that balance source- and target-domain performance during continual pre-training for mathematical reasoning. The learned heuristics generalize to unseen source domains, target models, and domain spaces.

## Background & Motivation

**Background**: Although large language models acquire general capabilities through large-scale pre-training, they still require continual pre-training to enhance performance in knowledge-intensive domains such as mathematics and code. However, training directly on target-domain data leads to catastrophic forgetting.

**Limitations of Prior Work**: (1) The common remedy is to mix source- and target-domain data during training, yet the mixing ratio is typically determined by manually designed heuristics or empirical rules. (2) The heuristic space for data mixing is vast—spanning different domains, proportions, and schedules—making manual exploration highly inefficient. (3) Existing methods such as DoReMi and DSIR rely on specific assumptions that limit their generalizability.

**Key Challenge**: The optimal data mixing strategy is high-dimensional, dynamic, and task-dependent, whereas manual heuristics cover only a negligible portion of the policy space, leaving a large number of potentially effective heuristics undiscovered and unexploited.

**Goal**: To train a small agent model that learns generalizable domain re-weighting heuristics from a large collection of data mixing trajectories, enabling automatic adjustment of data mixing ratios during continual pre-training.

**Key Insight**: A large number of data mixing trajectories are first sampled randomly on a small proxy model, and environment feedback (benchmark performance) is collected. An offline RL algorithm then trains the agent to learn the mapping from trajectory states to optimal mixing ratios.

**Core Idea**: Data mixing heuristics can be parameterized as a small agent that learns from trajectory data via RL, and the learned heuristics exhibit cross-model and cross-domain generalization.

## Method

### Overall Architecture

The framework consists of three stages: (1) **Data Collection** — a large number of data mixing trajectories are sampled randomly, trained on a proxy model, and evaluated; (2) **Agent Training** — a data mixing agent is trained on the collected trajectories and feedback using Conservative Q-Learning (CQL); (3) **Deployment** — during continual pre-training of the target model, the agent directly predicts the domain distribution at each re-weighting step.

### Key Designs

1. **Trajectory Collection and Evaluation Environment**:

    - *Function*: Generates experience data required for agent training.
    - *Mechanism*: Twenty data mixing trajectories (each consisting of 80 re-weighting steps) are sampled randomly on a 50M-parameter proxy model. MMLU and MATH benchmark performance is evaluated at each checkpoint. The domain space comprises 52 dimensions (DCLM general-domain data + Dolmino mathematics data).
    - *Design Motivation*: The low training cost of the small proxy model enables extensive exploration of the policy space, and the environment feedback provides supervision that associates mixing strategies with downstream performance.

2. **CQL Reinforcement Learning Training**:

    - *Function*: Learns the optimal mixing strategy from offline trajectory data.
    - *Mechanism*: The state is defined as the current domain distribution concatenated with historical environment feedback; the action is the next-step domain distribution; and the reward is the change in benchmark performance. Conservative Q-Learning is employed to prevent over-optimism toward unseen actions.
    - *Design Motivation*: Online RL would require repeated training of large models, which is computationally infeasible; offline RL enables efficient learning from pre-collected trajectories.

3. **Cross-Setting Generalization**:

    - *Function*: Transfers the learned heuristics to new scenarios.
    - *Mechanism*: The agent is trained on a 50M model in the mathematics domain and deployed directly to target models of different sizes (1B+), different source domains (general → science/code, etc.), and different domain spaces.
    - *Design Motivation*: If the heuristics encode general knowledge about which domain distributions promote balanced performance, they should transfer across settings.

### Loss & Training

The CQL loss combines the standard Q-learning objective with a conservative regularization term that penalizes high Q-value estimates for out-of-dataset actions. The agent is a small MLP that takes the current state as input and outputs a continuous domain distribution.

## Key Experimental Results

### Main Results

**Continual Pre-training for Mathematical Reasoning (balancing MMLU and MATH performance)**

| Method | MMLU Retention | MATH Improvement | Overall |
|--------|---------------|-----------------|---------|
| Uniform Mixing | Moderate | Moderate | Baseline |
| DoReMi | Better | Better | Improved |
| Manual Heuristics | Variable | Variable | Experience-dependent |
| **Data Mixing Agent** | **Best** | **Best** | **Best** |

### Ablation Study

| Generalization Test | Outcome | Notes |
|--------------------|---------|-------|
| Unseen source domains | Effective | Heuristics transfer across domains |
| Different target model sizes | Effective | 50M → 1B+ transfer successful |
| Unseen domain spaces | Effective | Remains effective under different domain taxonomies |
| Code generation domain | Effective | Adapts across target domains |

### Key Findings

- The Data Mixing Agent outperforms all baseline methods in balancing source- and target-domain performance.
- The heuristics learned by the agent closely align with human intuition—e.g., science-domain data benefits MMLU performance.
- The agent achieves better model performance with less source-domain data, indicating that it learns a more efficient data utilization strategy.
- Strategies learned on the 50M model transfer directly to 1B+ models, suggesting that data mixing heuristics are scale-invariant.

## Highlights & Insights

- This work provides the first demonstration that data mixing heuristics can be parameterized and learned via RL.
- The cross-setting generalization is impressive—strategies learned on an extremely small model are applicable to large models.
- The learned strategies are interpretable and aligned with human intuition, lending credibility to the approach.

## Limitations & Future Work

- The trajectory collection stage still incurs considerable computational cost (20 trajectories × 80 steps × evaluation).
- The conservatism of CQL may prevent the agent from exploring more aggressive mixing strategies.
- The evaluation environment uses simplified benchmarks (MMLU/MATH), which may not fully capture real-world performance.
- Domain space partitioning relies on an external classifier, and classification quality affects agent learning.

## Related Work & Insights

- **vs. DoReMi**: DoReMi optimizes domain weights via gradient-based methods, whereas the Data Mixing Agent learns heuristics end-to-end.
- **vs. Manual Tuning**: Manual approaches can only cover a negligible portion of the policy space, while the agent can automatically explore and exploit a vastly larger set of heuristics.
- **vs. Online Methods**: Online methods require repeated training of large models; the agent is trained once and can be deployed multiple times.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First model-based end-to-end data mixing method; heuristics learned via RL.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple generalization tests and ablation analyses, though the benchmark coverage is limited.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and systematic methodology.
- Value: ⭐⭐⭐⭐ Provides an automated tool for data engineering in large-scale pre-training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] unsupervised learning of efficient exploration pre-training adaptive policies vi](../../ICLR2026/reinforcement_learning/unsupervised_learning_of_efficient_exploration_pre-training_adaptive_policies_vi.md)
- [\[ICLR 2026\] $\textbf{Re}^{2}$: Unlocking LLM Reasoning via Reinforcement Learning with Re-solving](../../ICLR2026/reinforcement_learning/textbfre2_unlocking_llm_reasoning_via_reinforcement_learning_with_re-solving.md)
- [\[ACL 2026\] Easy Samples Are All You Need: Self-Evolving LLMs via Data-Efficient Reinforcement Learning](easy_samples_are_all_you_need_self-evolving_llms_via_data-efficient_reinforcemen.md)
- [\[ICLR 2026\] Breaking Barriers: Do Reinforcement Post Training Gains Transfer To Unseen Domains?](../../ICLR2026/reinforcement_learning/breaking_barriers_do_reinforcement_post_training_gains_transfer_to_unseen_domain.md)
- [\[ACL 2026\] Reverse Constitutional AI: A Framework for Controllable Toxic Data Generation via Probability-Clamped RLAIF](reverse_constitutional_ai_a_framework_for_controllable_toxic_data_generation_via.md)

</div>

<!-- RELATED:END -->
