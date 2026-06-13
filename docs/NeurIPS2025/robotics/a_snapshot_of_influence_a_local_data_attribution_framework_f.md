---
title: >-
  [Paper Note] A Snapshot of Influence: A Local Data Attribution Framework for Online Reinforcement Learning
description: >-
  [NeurIPS 2025][Robotics][Data Attribution] This work is the first to introduce data attribution into online reinforcement learning. It proposes a local attribution framework to quantify each training record's contributio…
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "Data Attribution"
  - "Online Reinforcement Learning"
  - "PPO"
  - "Influence Functions"
  - "Experience Filtering"
date: 2026-05-08
content_hash: cfca29f4b9e5dc94
---

# A Snapshot of Influence: A Local Data Attribution Framework for Online Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2505.19281](https://arxiv.org/abs/2505.19281)  
**Code**: [https://github.com/LDAORL/LDA-ORL](https://github.com/LDAORL/LDA-ORL)  
**Area**: Robotics
**Keywords**: Data Attribution, Online Reinforcement Learning, PPO, Influence Functions, Experience Filtering

## TL;DR
This work is the first to introduce data attribution into online reinforcement learning. It proposes a local attribution framework to quantify each training record's contribution to policy updates, and builds upon it an Iterative Influence Filtering (IIF) algorithm that substantially improves sample efficiency and final performance on both classical RL benchmarks and LLM RLHF.

## Background & Motivation
Online RL methods such as PPO have achieved remarkable success in games, robotics, and LLM alignment, yet they suffer from three persistent challenges: low sample efficiency (requiring millions of interactions), training instability (high variance across runs), and poor interpretability. Data attribution is a powerful tool in supervised learning for tracing model predictions back to specific training samples. However, existing attribution methods (e.g., influence functions, Data Shapley, TracIn) all assume a static training dataset — an assumption severely violated in online RL, where each experience both updates the policy and, through the updated policy, influences future data collection, creating a circular dependency.

## Core Problem
How can one quantify the influence of individual training experiences on policy updates under the circular data–model dependency inherent to online RL? And can such attribution information be leveraged to filter harmful experiences and improve training efficiency? This problem is important because not all experiences are equally beneficial in RL — records with inaccurate advantage estimates can actively mislead policy learning.

## Method

### Overall Architecture
The core idea is *local attribution*: rather than tracing influence across the entire training history (where cross-iteration circular dependencies are non-differentiable), the framework focuses on a single iteration, analyzing which records in the current rollout buffer $B^{(k)}$ positively or negatively influence the policy update from $\theta^{(k)}$ to $\theta^{(k+1)}$. The framework comprises three components: the attribution entity, the objective function, and the attribution method.

### Key Designs
1. **Attribution Entity**: A single record $z_i = (s_i, a_i, r_i, \log\pi_i, v_i, \hat{A}_i)$ from the PPO rollout buffer serves as the atomic unit of attribution, consistent with the natural granularity of PPO training.

2. **Dual Objective Function Design**:

    - **Agent Action Objective** $f_{\text{action}}(\theta) = \log\pi_\theta(a|s)$: Used to diagnose why the agent takes a specific action in a given state, primarily serving interpretability analysis.
    - **Cumulative Return Objective** $f_{\text{return}}(\theta) = \mathbb{E}_{\tau\sim\pi_{\text{ref}}} [\log\pi_\theta(a|s) \hat{A}_{\text{ref}}(s,a)]$: Evaluates each record's contribution to overall return. This elegantly uses the current iteration's policy $\pi_{\theta^{(k)}}$ as the reference policy and the rollout buffer itself as the validation set, avoiding policy-dependent distribution shift and high-variance issues. This objective is structurally equivalent to the REINFORCE objective with a baseline.

3. **TracIn-Based Attribution**: For each record $z_i$ in the buffer, an influence score is computed as:
   $$I_i = \sum_{j: z_i \in \mathcal{B}_j^{(k)}} \langle \nabla_\theta f(\theta_j^{(k)}), \nabla_\theta \mathcal{L}_{\text{PPO}}(\theta_j^{(k)}, z_i) \rangle$$
   i.e., the sum of inner products between the objective gradient and the training loss gradient. Positive scores indicate beneficial records (top records); negative scores indicate harmful ones (bottom records).

### Three Applications

1. **Learning Diagnostics**: Bottom records share a common characteristic — inaccurate advantage estimation, where good actions receive negative advantages and poor actions receive positive ones. Quantitative analysis reveals a strong negative correlation between influence scores and $\bar{A} \cdot \hat{A}$ (where $\bar{A}$ is the Monte Carlo estimate of the true advantage), confirming that sign flips and large estimation errors are the root causes of harm.

2. **Behavior Formation Timeline Analysis**: Tracking the top records of specific behaviors across training reveals a three-phase transition — (Phase 1) simple action–advantage association: top records merely correspond to same action + positive advantage or different action + negative advantage, without attending to state semantics; (Phase 2) semantic clustering: top records begin to cluster around semantically similar states, indicating that the agent has learned to generalize; (Phase 3) influence saturation: as convergence approaches, influence scores tend toward zero and become dominated by noise. This phenomenon is quantitatively validated using a roughness metric on weighted graphs.

3. **Targeted Intervention**: Removing negatively influential records within a single iteration and retraining consistently improves performance, validating the practical utility of the framework.

### Iterative Influence Filtering (IIF)
Building on the success of targeted intervention, single-iteration filtering is extended into an iterative algorithm: at each PPO iteration, a filtering step is inserted between data collection and model update — influence scores are computed for all records, the most negative $p\%$ are discarded, and only the filtered data is used to update the policy. Two key efficiency optimizations are employed: (1) gradients are computed only once at the initial parameters $\theta^{(k)}$, rather than iterating over all intermediate checkpoints; (2) a *ghost dot product* is used to efficiently compute gradient inner products. Regarding the hyperparameter $p$: 50% is used for simple environments, while 12.5% or 6.25% is used for complex ones — discarding too many records hurts performance due to the non-additivity of influence scores.

## Key Experimental Results

### Standard RL Benchmarks

| Environment | SEave | SEpeak | RTpeak Reduction |
|-------------|-------|--------|------------------|
| FrozenLake | 34.0%±2.0% | 19.2%±5.9% | 29.5%±2.9% |
| Acrobot | 36.7%±6.5% | 48.5%±0.8% | 55.2%±1.0% |
| MiniGrid | 65.8%±3.3% | 61.7%±4.1% | 69.1%±1.7% |
| Highway | 37.7%±6.1% | 55.1%±2.9% | 59.9%±0.7% |
| LunarLander | 26.0%±1.8% | 39.7%±3.7% | 44.9%±2.5% |
| BipedalWalker | 31.0%±8.7% | 26.2%±8.0% | 29.2%±0.7% |

### RLHF Experiments (GPT-Neo-2.7B Toxicity Mitigation)
- IIF filters approximately 50% of negatively influential records, halving per-iteration optimization time.
- The performance level at which standard PPO diverges is reached in less than half the number of training iterations.
- Total runtime is reduced by approximately 4×.
- Final toxicity is lower and reward is higher.

### Ablation Study
- Advantage-based heuristic: performs comparably to IIF in simple environments (FrozenLake) but fails in complex ones (MiniGrid), as MC advantage estimates are unreliable in large state spaces.
- TD-error-based heuristic (inspired by PER): effective in simple environments but underperforms standard PPO in LunarLander, as PPO's small-batch on-policy data amplifies TD error noise.
- Setting $p$=100% (discarding all negatively influential records) is suboptimal, confirming the non-additivity of influence scores.
- IIF remains effective with the Adam optimizer, though the magnitude of gains may differ.
- Random filtering is significantly worse than standard training, demonstrating that IIF's gains stem from effective data attribution rather than mere data reduction.

## Highlights & Insights
- **Conceptual Innovation**: The first systematic introduction of data attribution into online RL; the local attribution approach elegantly circumvents the circular dependency problem across iterations.
- **Elegant Objective Function Design**: $f_{\text{return}}$ uses the current policy as a dynamic reference and the training buffer as the validation set, simultaneously resolving distribution shift and reducing variance; it is structurally equivalent to REINFORCE.
- **Three-Phase Transition Discovery**: Attribution analysis uncovers an intrinsic mechanism of behavior formation in RL — from simple association to semantic clustering to convergence — a theoretically valuable insight.
- **Practical Utility of IIF**: Adds minimal computational overhead (influence computation takes approximately 0.1–2 seconds per iteration) while achieving 20–70% improvements in sample efficiency and runtime reduction.
- **Extension to RLHF**: A sequence-level objective $f_{\text{seq}}$ is designed to successfully extend the framework to LLM settings, reducing total runtime by 4×.

## Limitations & Future Work
- **Optimizer Assumption**: TracIn is designed for SGD, whereas modern RL and LLM training typically uses Adam. Although IIF remains effective under Adam empirically, theoretical guarantees are lacking.
- **Algorithm Coverage**: The framework currently focuses on PPO; extending it to algorithms such as GRPO commonly used in LLM training is an important direction.
- **Absence of Counterfactual Explanation**: Local attribution cannot answer the question "what would happen if this record were absent" — partly due to inherent limitations of TracIn, and partly because the circular dependency in online RL makes counterfactual tracing extremely difficult.
- **Selection of $p$**: The optimal filtering ratio varies across environments, and an adaptive selection mechanism is lacking.
- **Scalability Validation**: The RLHF experiments use a 2.7B model; whether the approach scales to larger models remains unverified.

## Related Work & Insights
- **vs. Feature-Level RL Interpretability (saliency maps, etc.)**: This work explains from the data level, providing finer-grained per-sample attribution that can directly improve training.
- **vs. Key State Identification (StateMask, RICE, lazy-MDP)**: These methods require modifying the training pipeline or rely on a sufficiently mature policy; the proposed framework requires no pipeline modification and is applicable from the earliest stages of training.
- **vs. Supervised Learning Data Attribution (TracIn, TRAK, Data Shapley)**: This work extends attribution from static datasets to the non-stationary online RL setting; the core contributions are the local attribution framework and RL-specific objective functions.
- **vs. Difficulty/Priority Heuristics (PER, difficulty filtering)**: PER's TD-error prior performs poorly in on-policy settings; difficulty-based filtering (pass@k) is ineffective for PPO. IIF captures richer signals through gradient similarity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First systematic introduction of data attribution into online RL; conceptually clear and highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Six standard RL environments plus RLHF, extensive ablations and baseline comparisons, with rigorous statistical significance testing.
- Writing Quality: ⭐⭐⭐⭐⭐ — Exceptionally clear writing; the problem–method–application logical chain is complete and figures are intuitive.
- Value: ⭐⭐⭐⭐ — The framework is elegant and practical, but currently limited to PPO; extension to broader RL algorithms and larger-scale models remains to be validated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DexFlyWheel: A Scalable Self-Improving Data Generation Framework for Dexterous Manipulation](dexflywheel_a_scalable_and_self-improving_data_generation_framework_for_dexterou.md)
- [\[NeurIPS 2025\] Reinforcement Learning with Action Chunking](reinforcement_learning_with_action_chunking.md)
- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](learning_interactive_world_model_for_object-centric_reinforcement_learning.md)
- [\[NeurIPS 2025\] Real-World Reinforcement Learning of Active Perception Behaviors](real-world_reinforcement_learning_of_active_perception_behaviors.md)
- [\[NeurIPS 2025\] Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning](memo_training_memory-efficient_embodied_agents_with_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
