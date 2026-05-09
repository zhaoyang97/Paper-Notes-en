---
title: >-
  [Paper Note] Pruning as a Cooperative Game: Surrogate-Assisted Layer Contribution Estimation for Large Language Models
description: >-
  [ICLR 2026][Reinforcement Learning][model pruning] This paper models LLM layer pruning as a cooperative game, employing a lightweight surrogate network to approximate Shapley values that capture inter-layer dependencies, achieving superior deep pruning performance over static heuristic methods.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - model pruning
  - cooperative game
  - Shapley value
  - surrogate network
  - large language models
date: 2026-05-08
content_hash: 2f7785db236f9b93
---

# Pruning as a Cooperative Game: Surrogate-Assisted Layer Contribution Estimation for Large Language Models

**Conference**: ICLR 2026
**arXiv**: [2602.07804](https://arxiv.org/abs/2602.07804)
**Code**: [GitHub](https://github.com/920927/Pruning_As_A_Cooperative_Game)
**Area**: Reinforcement Learning
**Keywords**: model pruning, cooperative game, Shapley value, surrogate network, large language models

## TL;DR

This paper models LLM layer pruning as a cooperative game, employing a lightweight surrogate network to approximate Shapley values that capture inter-layer dependencies, achieving superior deep pruning performance over static heuristic methods.

## Background & Motivation

Deployment of large language models (LLMs) is constrained by substantial computational and memory requirements. Layer pruning—removing entire Transformer layers—is an effective compression strategy. Existing layer pruning methods predominantly rely on **static heuristic rules** (e.g., weight magnitudes, activation norms, sensitivity analysis), assuming that the importance of each layer is fixed and independent.

However, the authors empirically demonstrate that layer importance exhibits **context-dependent behavior**: in single-layer pruning, the ranking of intermediate layers fluctuates significantly; in multi-layer pruning, this instability is further amplified. More critically, sequentially pruning layers according to static importance scores (PPL = 15.4535) does not yield a global optimum—the optimal two-layer combination (Layer 10+11, PPL = 15.4279) may not include the single lowest-ranked layer. This reveals dynamic inter-layer dependencies that static methods cannot capture.

The core idea of this paper is to formalize layer pruning as a **cooperative game**, where each layer is a player, model performance serves as the utility function, and Shapley values quantify the true contribution of each layer after accounting for inter-layer interactions.

## Method

### Overall Architecture

A two-stage framework: Stage 1 generates diverse pruning masks and evaluates their performance; Stage 2 trains a surrogate network to approximate performance degradation, enabling efficient Shapley value estimation.

### Key Designs

1. **Stratified Monte Carlo Mask Sampling (Stage 1)**:

    - Function: Generate diverse binary pruning masks with controlled Hamming weights.
    - Mechanism: Stratified sampling is performed over the number of retained layers $k$, where $N_{k_j}$ masks are sampled per stratum $k_j$: $\mathbf{m}^{(k_j,t)} \sim \text{Uniform}\{\mathbf{m} \in \{0,1\}^L : k(\mathbf{m})=k_j\}$. The performance score is defined as $s(\mathbf{m}) = \text{PPL}_\text{orig} / \text{PPL}(M(\mathbf{m}))$.
    - Design Motivation: Ensures balanced coverage across different pruning ratios, preventing sampling bias toward specific compression rates.

2. **Lightweight Surrogate Network (Stage 2)**:

    - Function: Train a two-layer feedforward network $f_\theta(\mathbf{m})$ to predict the performance score for arbitrary masks.
    - Mechanism: The surrogate is trained with MSE loss $\mathcal{L}(\theta) = \frac{1}{N}\sum_{n=1}^{N}(f_\theta(\mathbf{m}_n) - s(\mathbf{m}_n))^2$.
    - Design Motivation: Avoids full model inference for each mask combination, reducing the computational cost of Shapley value estimation to a tractable level.

3. **Approximate Shapley Value Estimation**:

    - Function: Efficiently compute the marginal contribution of each layer using the surrogate network.
    - Mechanism: $\hat{\phi}_i = \frac{1}{Q}\sum_{q=1}^{Q}(f_\theta(\mathbf{m}^{(k_j,q)} \cup \{i\}) - f_\theta(\mathbf{m}^{(k_j,q)}))$
    - Design Motivation: Exact Shapley values require enumeration over $2^L$ subsets, which is computationally infeasible; the surrogate network enables large-scale sampling-based estimation.

### Loss & Training

The surrogate network is trained with MSE loss, using (mask, performance score) pairs collected in Stage 1 as supervision. After training, Shapley values are estimated over a large set of candidate masks, and layers are removed in ascending order of contribution until the target compression ratio is reached.

## Key Experimental Results

### Main Results (LLaMA-2-7B, WikiText2 PPL)

| Method | Remove 3 | Remove 6 | Remove 9 | Remove 12 |
|--------|----------|----------|----------|-----------|
| SliceGPT | 108.10 | 212.89 | 291.85 | 393.89 |
| SLEB | 14.24 | 19.47 | 27.45 | 58.12 |
| Shortened-LLaMA | 16.65 | 36.37 | 81.96 | 304.52 |
| ShortGPT | 16.65 | 36.37 | 81.96 | 157.99 |
| **Ours** | **14.69** | **18.87** | **24.61** | **38.12** |

### Ablation Study

| Configuration | PPL (WikiText2) | Note |
|---------------|-----------------|------|
| Static single-layer removal | 15.45 | Two layers selected by independent importance |
| Recomputation scheme | 15.45 | Re-evaluate after removing one layer |
| **Optimal combination** | **15.43** | Global optimum accounting for inter-layer interactions |

### Key Findings
- The proposed method yields the greatest advantage under aggressive pruning (12 layers), with PPL substantially lower than baselines (38.12 vs. 58.12–304.52).
- The method generalizes to non-Transformer architectures (RWKV-7B, Mamba-2.8B).
- It integrates seamlessly with quantization techniques for additional efficiency gains.
- Consistent accuracy improvements are observed across 8 zero-shot benchmarks.

## Highlights & Insights

- **Novel game-theoretic perspective**: Shifts layer pruning from independent evaluation to global optimization accounting for inter-layer cooperation.
- **Strong practicality**: Surrogate network training incurs minimal overhead, and once trained, it enables rapid evaluation of a large number of mask combinations.
- **Dynamic nature of layer importance**: Table 1 clearly demonstrates the limitations of static rankings—the least important two-layer combination is not necessarily the globally optimal one.

## Limitations & Future Work

- The generalization capability of the surrogate network depends on the diversity and quantity of masks sampled in Stage 1.
- The current formulation uses only PPL as the utility function, without considering downstream task-specific performance metrics.
- The budget allocation across strata in the stratified sampling strategy is relatively uniform; more adaptive allocation schemes may offer further improvements.

## Related Work & Insights

- GTAP (Diaz-Ortiz Jr et al., 2023) pioneered the application of game theory to neuron importance estimation but was limited by computational complexity.
- The Block Influence (BI) metric from ShortGPT (Men et al., 2024) serves as an important comparison baseline.
- The surrogate network paradigm can be generalized to other scenarios requiring extensive evaluations, such as NAS and hyperparameter search.

## Rating
- Novelty: ⭐⭐⭐⭐ The cooperative game formulation for layer pruning is a creative and well-motivated modeling choice.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across multiple models and benchmarks, with ablation studies and non-Transformer architecture experiments.
- Writing Quality: ⭐⭐⭐⭐ Motivation is articulated clearly; Table 1 is particularly convincing.
- Value: ⭐⭐⭐⭐ Provides a novel and systematic methodological contribution to LLM compression.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Strategic Persuasion with Language Models](towards_strategic_persuasion_with_language_models.md)
- [\[ICLR 2026\] Efficient Estimation of Kernel Surrogate Models for Task Attribution](efficient_estimation_of_kernel_surrogate_models_for_task_attribution.md)
- [\[ICLR 2026\] AWM: Accurate Weight-Matrix Fingerprint for Large Language Models](awm_accurate_weight-matrix_fingerprint_for_large_language_models.md)
- [\[ICLR 2026\] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)
- [\[ICLR 2026\] Robust Multi-Objective Controlled Decoding of Large Language Models](robust_multi-objective_controlled_decoding_of_large_language_models.md)

</div>

<!-- RELATED:END -->
