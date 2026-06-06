---
title: >-
  [Paper Note] Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training
description: >-
  [ACL 2026][LLM Pretraining][Data Mixing] This paper proposes Data Mixing Agent, the first model-based end-to-end domain re-weighting framework. By training a small agent using CQL reinforcement learning on extensive data…
tags:
  - "ACL 2026"
  - "LLM Pretraining"
  - "Data Mixing"
  - "Domain Re-weighting"
  - "Continual Pre-training"
  - "Reinforcement Learning"
  - "Catastrophic Forgetting"
date: 2026-05-08
content_hash: 7320f04292b2b57e
---

# Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training

**Conference**: ACL 2026  
**arXiv**: [2507.15640](https://arxiv.org/abs/2507.15640)  
**Code**: None  
**Area**: Reinforcement Learning  
**Keywords**: Data Mixing, Domain Re-weighting, Continual Pre-training, Reinforcement Learning, Catastrophic Forgetting

## TL;DR

This paper proposes Data Mixing Agent, the first model-based end-to-end domain re-weighting framework. By training a small agent using CQL reinforcement learning on extensive data mixing trajectories, it learns generalizable data mixing heuristics. It balances performance between source and target domains in mathematical reasoning continual pre-training and generalizes to unseen source domains, target models, and domain spaces.

## Background & Motivation

**Background**: Although Large Language Models (LLMs) acquire general capabilities through large-scale pre-training, they still require continual pre-training to enhance performance in knowledge-intensive domains (e.g., mathematics, code). However, direct training on target domain data leads to catastrophic forgetting.

**Limitations of Prior Work**: (1) Common solutions involve mixing source and target domain data, but determining mixing ratios typically relies on human-designed heuristics or empirical findings; (2) The heuristic space for data mixing is extremely rich (different domains, ratios, schedules), making manual exploration highly inefficient; (3) Existing methods (e.g., DoReMi, DSIR) are based on specific assumptions and have limited generalization.

**Key Challenge**: The optimal data mixing strategy is high-dimensional, dynamic, and task-dependent, yet manual heuristics cover only a tiny fraction of the strategy space. Many potentially effective heuristics remain undiscovered and underutilized.

**Goal**: Train a small agent model to learn generalizable domain re-weighting heuristics from a large number of data mixing trajectories, automatically adjusting data mixing ratios during continual pre-training.

**Key Insight**: First, randomly sample numerous data mixing trajectories on a small proxy model and collect environmental feedback (benchmark performance). Then, use offline reinforcement learning to train the agent to map trajectory states to optimal mixing ratios.

**Core Idea**: Data mixing heuristics can be parameterized as a small agent, learned from trajectory data via RL, and the learned heuristics demonstrate cross-model and cross-domain generalization capabilities.

## Method

### Overall Architecture

A three-stage framework: (1) Data Collection—randomly sampling numerous data mixing trajectories, training and evaluating on a proxy model; (2) Agent Training—using CQL (Conservative Q-Learning) to train the data mixing agent on collected trajectories and feedback; (3) Deployment—during the continual pre-training of the target model, the agent directly predicts the domain distribution for the next re-weighting step.

### Key Designs

1.  **Trajectory Collection and Evaluation Environment**:
    - **Function**: Generate empirical data required for training the agent.
    - **Mechanism**: Randomly sample 20 data mixing trajectories (80 re-weighting steps each) on a 50M parameter proxy model, evaluating MMLU and MATH benchmark performance at each checkpoint. 52-dimensional domain space (DCLM general data + Dolmino math data).
    - **Design Motivation**: Small proxy models have low training costs and allow for extensive strategy space exploration; environmental feedback provides supervisory signals linking mixing strategies to performance.

2.  **CQL Reinforcement Learning Training**:
    - **Function**: Learn the optimal mixing strategy from offline trajectory data.
    - **Mechanism**: State = current domain distribution + historical environmental feedback, Action = next domain distribution, Reward = change in benchmark performance. Use Conservative Q-Learning to avoid over-optimism toward unseen actions.
    - **Design Motivation**: Online RL requires repeated training of large models (impractical); offline RL can learn efficiently from pre-collected trajectories.

3.  **Cross-Setting Generalization**:
    - **Function**: Enable the learned heuristics to transfer to new scenarios.
    - **Mechanism**: The agent is trained on a 50M model with math domains and deployed directly to target models of different sizes (1B+), different source domains (general $\rightarrow$ science/code, etc.), and different domain spaces.
    - **Design Motivation**: If heuristics represent general knowledge about "what domain distributions help balance performance," they should possess cross-setting transferability.

### Loss & Training

CQL Loss = standard Q-learning loss + conservative regularization term (penalizing high Q-value estimates for actions outside the dataset). The agent is a small MLP model that takes the current state as input and outputs a continuous domain distribution.

## Key Experimental Results

### Main Results

**Continual Pre-training for Mathematical Reasoning (Balancing MMLU and MATH Performance)**

| Method | MMLU Retention | MATH Gain | Overall |
| :--- | :--- | :--- | :--- |
| Uniform Mixing | Medium | Medium | Baseline |
| DoReMi | Good | Good | Improved |
| Manual Heuristic | Variable | Variable | Experience-dependent |
| **Data Mixing Agent** | **Optimal** | **Optimal** | **Optimal** |

### Ablation Study

| Generalization Test | Effect | Description |
| :--- | :--- | :--- |
| Unseen Source Domain | Effective | Heuristic transfer across domains |
| Different Size Target Models | Effective | Successful 50M $\rightarrow$ 1B+ transfer |
| Unseen Domain Space | Effective | Remains effective with different domain classifications |
| Code Generation Domain | Effective | Adaptation across target domains |

### Key Findings

- Data Mixing Agent outperforms all baseline methods in balancing source and target domain performance.
- Heuristics learned by the agent align closely with human intuition—e.g., science domain data benefits MMLU.
- The agent achieves better model performance using less source domain data, indicating it has learned a more efficient data utilization strategy.
- Strategies learned from a 50M model transfer directly to 1B+ models, suggesting that data mixing heuristics are scale-invariant.

## Highlights & Insights

- First to demonstrate that data mixing heuristics can be parameterized and learned via RL.
- Impressive cross-setting generalization—strategies learned on extremely small models apply to large models.
- The strategies learned by the agent are interpretable and align with human intuition, increasing reliability.

## Limitations & Future Work

- The trajectory collection phase still incurs significant computational costs (20 trajectories $\times$ 80 steps $\times$ valuation).
- The conservatism of CQL might limit the agent from exploring more aggressive mixing strategies.
- The evaluation environment uses simplified benchmarks (MMLU/MATH), which may not fully capture real-world performance.
- The division of domain space depends on external classifiers; classification quality affects agent learning.

## Related Work & Insights

- **vs DoReMi**: DoReMi is based on gradient optimization of domain weights; Data Mixing Agent learns heuristics end-to-end.
- **vs Manual Tuning**: Manual methods cover only a tiny strategy space; the agent can automatically explore and utilize a vast number of heuristics.
- **vs Online Methods**: Online methods require repeated training of large models; the agent can be deployed multiple times after learning once.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First model-based end-to-end data mixing method using RL to learn heuristics.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes various generalization tests and ablation analyses, though benchmarks are limited.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and systematic methodology.
- **Value**: ⭐⭐⭐⭐ Provides an automated tool for data engineering in large-scale pre-training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning](forever_forgetting_curve-inspired_memory_replay_for_language_model_continual_lea.md)
- [\[ICLR 2026\] Predicting Training Re-evaluation Curves Enables Effective Data Curriculums](../../ICLR2026/llm_pretraining/predicting_training_re-evaluation_curves_enables_effective_data_curriculums_for_.md)
- [\[ACL 2026\] KoCo: Conditioning Language Model Pre-training on Knowledge Coordinates](koco_conditioning_language_model_pre-training_on_knowledge_coordinates.md)
- [\[ACL 2026\] Working Memory Constraints Scaffold Learning in Transformers under Data Scarcity](working_memory_constraints_scaffold_learning_in_transformers_under_data_scarcity.md)
- [\[ICLR 2026\] Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training](../../ICLR2026/llm_pretraining/common_corpus_ethical_data_for_llm_pretraining.md)

</div>

<!-- RELATED:END -->
