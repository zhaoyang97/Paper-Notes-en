---
title: >-
  [Paper Note] Preference Optimization for Combinatorial Optimization Problems
description: >-
  [ICML 2025][Reinforcement Learning][Preference Optimization] Introduces the concept of preference optimization from RLHF into Combinatorial Optimization Problems (COPs), transforming quantitative reward signals into qualitative preference signals. Combined with an entropy-regularized objective and local search fine-tuning, the proposed approach achieves a 1.5x-2.5x convergence acceleration and superior solution quality across standard benchmarks such as TSP, CVRP, and FFSP.
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "Preference Optimization"
  - "Combinatorial Optimization"
  - "REINFORCE"
  - "Entropy Regularization"
  - "Local Search"
date: 2026-05-08
content_hash: 3d1e3a2c85071179
---

# Preference Optimization for Combinatorial Optimization Problems

**Conference**: ICML 2025  
**arXiv**: [2505.08735](https://arxiv.org/abs/2505.08735)  
**Code**: None  
**Area**: LLM Alignment  
**Keywords**: Preference Optimization, Combinatorial Optimization, REINFORCE, Entropy Regularization, Local Search

## TL;DR

Introduces the concept of preference optimization from RLHF into Combinatorial Optimization Problems (COPs), transforming quantitative reward signals into qualitative preference signals. Combined with an entropy-regularized objective and local search fine-tuning, the proposed approach achieves a 1.5x-2.5x convergence acceleration and superior solution quality across standard benchmarks such as TSP, CVRP, and FFSP.

## Background & Motivation

**Background**: Reinforcement Learning (RL) has become the dominant paradigm for Neural Combinatorial Optimization (RL4CO). REINFORCE and its variants are widely used to train end-to-end neural solvers (e.g., AM, POMO, Sym-NCO) to learn heuristic policies from environmental interactions without expert demonstrations.

**Limitations of Prior Work**: 
   - **Reward signal decay**: As the policy improves, the difference in rewards |r(x,τ)-b(x)| between sampled solutions approaches zero, leading to vanishing gradients and slow convergence.
   - **Low exploration efficiency**: The combinatorial action space grows exponentially, making traditional trajectory-level entropy regularization computationally intractable (requiring the enumeration of all trajectories).
   - **Extra inference overhead**: Many methods employ Local Search (LS) as a post-processing step to improve solution quality, which introduces additional inference time.

**Key Challenge**: REINFORCE relies on numerical reward differences to drive learning, but this difference becomes extremely small in the middle-to-late stages of COPs; meanwhile, the trajectory space is too vast to perform effective exploration regularization.

**Goal**: To design a general RL optimization framework that does not rely on absolute reward values, possesses built-in exploration capabilities, and seamlessly integrates local search.

**Key Insight**: Drawing inspiration from Direct Preference Optimization (DPO) in RLHF—converting quantitative rewards into qualitative preference comparisons, and using reparameterization to bypass intractable calculations.

**Core Idea**: Replace absolute reward values with preference comparisons, and reparameterize the reward function via the analytical solution of the entropy-regularized objective, thereby transforming policy optimization into a classification problem.

## Method

### Overall Architecture

The PO (Preference Optimization) framework comprises three core modules: (1) Preference Comparison module, which performs pairwise comparisons among multiple sampled solutions for the same problem instance to generate conflict-free preference labels; (2) Policy Optimization module, which directly updates the policy using a preference classification loss based on the reparameterized entropy-regularized target; (3) Local Search Fine-Tuning module, which generates high-quality preference pairs via local search after the policy has converged, further breaking through local optima.

### Key Designs

1. **Preference Signal Generation (Preference Comparison)**: 

    - **Function**: Transforms quantitative rewards r(x,τ) into qualitative preference labels y=𝟙(r(x,τ₁)>r(x,τ₂)).
    - **Mechanism**: Samples N solutions {τ₁,...,τ_N} for each problem instance x, and performs pairwise comparisons using the ground truth reward function (e.g., TSP path length) to generate preference pairs.
    - **Design Motivation**: Preference labels are invariant under affine transformations of the reward—𝟙(k·r+b > k·r'+b) = 𝟙(r > r'). This implies that regardless of how the reward is scaled or shifted, the preference relation is preserved, fundamentally addressing the problem of reward signal decay. Moreover, because the reward function in COPs is an objective physical metric (such as path length), the generated preference labels are naturally conflict-free and transitive.

2. **Reparameterized Entropy-Regularized Objective**:

    - **Function**: Substitutes the analytical optimal policy of the maximum-entropy RL objective into the preference model, eliminating the intractable partition function Z(x).
    - **Mechanism**: 
        - The starting point is the maximum-entropy RL objective: max E[r(x,τ)] + αℋ(π)
        - Its analytical optimal policy is: π*(τ|x) = (1/Z(x))·exp(α⁻¹·r(x,τ))
        - Solving for the reward gives: r̂(x,τ) = α·log π(τ|x) + α·log Z(x)
        - Substituting this reward into the Bradley-Terry preference model yields: p(τ₁≻τ₂|x) = σ(α·[log π(τ₁|x) - log π(τ₂|x)])
        - Key discovery: Z(x) naturally cancels out in the reward difference (as it only depends on x and not on τ). Proposition 3.1 proves that this cancellation does not affect the optimal policy.
    - **Design Motivation**: Directly computing trajectory-level entropy is intractable in COPs (due to the exponential number of trajectories). However, the reparameterization elegantly integrates entropy regularization implicitly into the preference classification objective, eliminating the need to explicitly enumerate any trajectories.

3. **Preference Model Selection and Gradient Analysis**:

    - **Function**: Supports various preference models (Bradley-Terry, Thurstone, Plackett-Luce) and analyzes the properties of their implicit advantage signals.
    - **Mechanism**: The final optimization objective is max E[𝟙(r(x,τ₁)>r(x,τ₂))·log p_θ(τ₁≻τ₂|x)], where the term g(τ,τ',x) - g(τ',τ,x) in its gradient acts as a "magnitude-invariant advantage signal"—when r(x,τ₁)>r(x,τ₂), the gradient always increases π_θ(τ₁) and decreases π_θ(τ₂).
    - **Design Motivation**: Unlike in REINFORCE where the advantage values tend to zero, the advantage signal in PO is controlled by the sigmoid function, maintaining a meaningful distinguishing capability throughout training.

4. **Fine-Tuning with Local Search**:

    - **Function**: Integrates local search (e.g., 2-Opt, swap*) into the training loop rather than at inference, after the policy has converged.
    - **Mechanism**: Performs a few LS iterations on each sampled solution τ to obtain LS(τ), and constructs preference pairs (τ, LS(τ), y), where y=𝟙(r(x,LS(τ))>r(x,τ)). The optimization objective becomes max E[log p_θ(LS(τ)≻τ|x)], which encourages the policy to imitate the solutions improved by LS.
    - **Design Motivation**: 
        - REINFORCE struggles to utilize LS: off-policy samples generated by LS lead to distribution shift, requiring correction via importance sampling.
        - In contrast, the preference objective of PO is naturally compatible with off-policy data, allowing the LS outputs to be treated as expert demonstrations (similar to imitation learning) without requiring importance sampling.
        - At inference, no LS is needed, avoiding extra inference overhead.

### Loss & Training

- **Main training loss**: Preference classification loss based on the BT model:
    - L_PO = -E[𝟙(r(x,τ₁)>r(x,τ₂))·log σ(α·[log π_θ(τ₁|x) - log π_θ(τ₂|x)])]
    - Samples N solutions for each instance in a batch and performs pairwise comparisons with O(N²) complexity.
- **Fine-tuning loss**: L_finetune = -E[log σ(α·[log π_θ(LS(τ)|x) - log π_θ(τ|x)])]
- **Training process**: Train with PO for T steps until convergence → fine-tune with PO+LS for T_FT steps.
- **Hyperparameter $\alpha$**: Use lower α values for models with built-in exploration mechanisms (POMO, Sym-NCO) to prevent excessive exploration from hindering convergence.

## Key Experimental Results

### Main Results

**TSP & CVRP (10k instances evaluation)**:

| Model + Algorithm | TSP-50 Gap | TSP-100 Gap | CVRP-50 Gap | CVRP-100 Gap |
|-----------|-----------|------------|------------|-------------|
| POMO (RF) | 0.04% | 0.15% | 0.94% | 1.76% |
| POMO (PO) | 0.02% | 0.07% | 0.68% | 1.37% |
| POMO (PO+FT) | 0.00% | 0.03% | 0.53% | 1.19% |
| Sym-NCO (RF) | 0.16% | 0.39% | 1.31% | 2.07% |
| Sym-NCO (PO) | 0.08% | 0.28% | 1.20% | 1.88% |

**FFSP (1k instances evaluation)**:

| Model + Algorithm | FFSP20 MS↓ | FFSP50 MS↓ | FFSP100 MS↓ |
|-----------|-----------|-----------|------------|
| MatNet (RF) | 27.3 | 51.5 | 91.5 |
| MatNet (PO) | 27.0 | 51.3 | 91.1 |
| MatNet (RF+Aug) | 25.4 | 49.6 | 89.7 |
| MatNet (PO+Aug) | 25.2 (Best) | 49.5 (Best) | 89.2 (Best) |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Preference Model Selection | BT > PL > Thurstone > Exp (TSP-100) | Bradley-Terry model performs the best |
| Convergence Speed | PO reaches the same level as RF with 40%-60% of the training epochs | 1.5x-2.5x acceleration |
| Advantage Value Distribution | PO: Wide distribution vs. RF: Narrowly concentrated around 0 | PO significantly improves distinguishing capability |
| Policy Consistency | PO >> RF, PO+FT further improves | p(π(τ₁)>π(τ₂) \| r(τ₁)>r(τ₂)) |
| Trajectory Entropy | PO is significantly higher than RF (early training) | Validates the intrinsic exploration capability of PO |
| Generalization (TSPLib) | ELG(PO) 3.00% vs ELG(RF) 3.08% | Consistent zero-shot improvements |

### Key Findings

1. **Significant convergence acceleration**: PO matches the performance of RF trained for 200 epochs within just 80 epochs of training on POMO and Sym-NCO.
2. **Consistent improvement in solution quality**: Across all evaluated neural solvers (AM, POMO, Sym-NCO, Pointerformer, MatNet, ELG), PO consistently performs better than or on par with RF.
3. **Highly effective local search fine-tuning**: POMO+PO+FT reduces the gap from 0.07% to 0.03% on TSP-100, nearing optimality.
4. **Enhanced exploration capability**: The trajectory entropy of PO is significantly higher than that of RF in the early stages of training, demonstrating that it inherits excellent exploration characteristics from the entropy-regularized objective.

## Highlights & Insights

1. **Cross-disciplinary innovation**: Systematically imports the preference optimization concepts of RLHF/DPO into the combinatorial optimization field for the first time, establishing an elegant theoretical bridge.
2. **Concise and complete theoretical derivation**: The derivation path from maximum-entropy RL → analytical solution of the optimal policy → reparameterization → preference model is clear, and Proposition 3.1 guarantees the validity of eliminating Z(x).
3. **Generic & plug-and-play**: PO does not bind to any specific model architecture and can replace the training algorithm of any existing REINFORCE-based methods.
4. **Theoretical justification for LS integration**: The preference framework naturally bypasses the off-policy challenges introduced by LS, which is difficult to achieve under REINFORCE.
5. **Conflict-free preference labels**: Standardizes preference labels naturally by exploiting the objective nature of COP rewards, avoiding the inconsistency issues common in human annotation in RLHF.

## Limitations & Future Work

1. **O(N²) comparison overhead**: Pairwise comparisons bring quadratic complexity, which can be computationally intensive when the sample size N is large.
2. **Selection of hyperparameter $\alpha$**: The hyperparameter α requires manual tuning based on model characteristics (lower α for built-in exploration models), lacking an adaptive strategy.
3. **Dependency on local search**: The LS methods used during fine-tuning (e.g., 2-Opt, swap*) are problem-specific, requiring custom LS operators for new COPs.
4. **Reparameterization assumption**: Approximating π_θ as the optimal policy π* is a strong assumption that may not hold in the early stages of training.
5. **Narrow evaluation**: Only validated on classic COPs; performance on more complex, real-world scenarios (such as VRP with time windows, multi-objective optimization) remains unverified.
6. **Insufficient theoretical analysis of preference models**: The comparative advantages of different preference models (BT vs Thurstone vs PL) across various COPs lack a systematic theoretical explanation.

## Related Work & Insights

- **DPO (Rafailov et al., 2024)**: The core inspiration for PO—reparameterizing the reward in the KL-regularized objective as a policy. However, PO utilizes entropy regularization rather than KL regularization, making it suitable for COPs without reference policies.
- **POMO (Kwon et al., 2020)**: One of the strongest baselines, upon which PO achieves the most significant improvement.
- **SAC (Haarnoja et al., 2017)**: The entropy-regularized objective of PO shares the same origin as SAC, but PO further combines it with preference learning.
- **Inspiration**: The ideas of preference optimization could be applicable to other RL scenarios with unstable reward signals (such as sparse reward problems in robotic control) and multi-objective optimization scenarios where defining a single scalar reward is difficult.

## Rating

- Novelty: ⭐⭐⭐⭐ (Novel cross-domain transfer, but core technology originates from DPO)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Covers 3 types of COPs, 6 neural solvers, comprehensive generalization and ablation studies)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear theoretical derivations, comprehensive experimental results)
- Value: ⭐⭐⭐⭐ (A universal framework with consistent improvements, though mainly focused on training efficiency rather than breakthrough results)

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Complexity Scaling Laws for Neural Models using Combinatorial Optimization](../../NeurIPS2025/reinforcement_learning/complexity_scaling_laws_for_neural_models_using_combinatorial_optimization.md)
- [\[NeurIPS 2025\] PARCO: Parallel AutoRegressive Models for Multi-Agent Combinatorial Optimization](../../NeurIPS2025/reinforcement_learning/parco_parallel_autoregressive_models_for_multi-agent_combinatorial_optimization.md)
- [\[ICML 2025\] Graph-Supported Dynamic Algorithm Configuration for Multi-Objective Combinatorial Optimization](graph-supported_dynamic_algorithm_configuration_for_multi-objective_combinatoria.md)
- [\[ICLR 2026\] Offline Preference-based Value Optimization](../../ICLR2026/reinforcement_learning/offline_preference-based_value_optimization.md)
- [\[ICLR 2026\] DuPO: Enabling Reliable Self-Verification via Dual Preference Optimization](../../ICLR2026/reinforcement_learning/dupo_enabling_reliable_self-verification_via_dual_preference_optimization.md)

</div>

<!-- RELATED:END -->
