---
title: >-
  [Paper Note] Provably Efficient Online RLHF with One-Pass Reward Modeling
description: >-
  [NeurIPS 2025][LLM Alignment][online RLHF] This paper proposes a one-pass reward modeling method based on online mirror descent (OMD) that eliminates the computational bottleneck in online RLHF — namely, storing all historical data and re-optimizing from scratch at each iteration — achieving $\mathcal{O}(1)$ time and memory complexity per iteration while also improving upon MLE methods in statistical efficiency.
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "online RLHF"
  - "reward modeling"
  - "online mirror descent"
  - "contextual dueling bandit"
  - "computational efficiency"
date: 2026-05-08
content_hash: ffaa5de46f0e6fc8
---

# Provably Efficient Online RLHF with One-Pass Reward Modeling

**Conference**: NeurIPS 2025
**arXiv**: [2502.07193](https://arxiv.org/abs/2502.07193)  
**Code**: [github.com/ZinYY/Online_RLHF](https://github.com/ZinYY/Online_RLHF)  
**Area**: LLM Alignment
**Keywords**: online RLHF, reward modeling, online mirror descent, contextual dueling bandit, computational efficiency

## TL;DR

This paper proposes a one-pass reward modeling method based on online mirror descent (OMD) that eliminates the computational bottleneck in online RLHF — namely, storing all historical data and re-optimizing from scratch at each iteration — achieving $\mathcal{O}(1)$ time and memory complexity per iteration while also improving upon MLE methods in statistical efficiency.

## Background & Motivation

RLHF (Reinforcement Learning from Human Feedback) is a core technique for aligning LLMs. Conventional RLHF relies on fixed preference datasets, but limited coverage makes it difficult for reward models to generalize to out-of-distribution samples. **Online RLHF** addresses this by iteratively collecting data and continuously improving the model, a strategy whose effectiveness has been demonstrated by Claude and LLaMA-2.

However, online RLHF faces a severe computational bottleneck:
- Each iteration requires incorporating new data into the historical dataset and re-optimizing the reward model over the **entire dataset**
- Under MLE estimation, the computational complexity at round $t$ is $\mathcal{O}(t \log t)$ with $\mathcal{O}(t)$ memory
- This becomes prohibitive over long training horizons, especially on edge devices

**Core Problem**: Can one design an online RLHF algorithm that is both statistically and computationally efficient?

## Method

### Overall Architecture

The paper formalizes RLHF as a **contextual preference bandit** problem and adopts the Bradley-Terry preference model:

$$\mathbb{P}[y=1 \mid x, a, a'] = \sigma(\phi(x,a)^\top \theta^* - \phi(x,a')^\top \theta^*)$$

where $\phi(x,a)$ is a known feature map, $\theta^*$ is an unknown parameter, and $\sigma$ is the sigmoid function. The nonlinearity coefficient $\kappa = \max 1/\dot{\sigma}(\cdot)$ characterizes the learning difficulty ($\kappa$ can be exponentially large).

### Key Designs: One-Pass Reward Modeling

**Problem with conventional MLE**: At round $t$, one must solve

$$\hat{\theta}_{t+1} = \arg\min_{\theta} \sum_{i=1}^t \ell_i(\theta)$$

which requires iterating over all historical data, incurring $\mathcal{O}(t \log t)$ computation per round.

**Proposed approach**: MLE is replaced by OMD with a second-order Taylor expansion under a customized local norm:

$$\widetilde{\theta}_{t+1} = \arg\min_{\theta \in \Theta} \left\{ \langle g_t(\widetilde{\theta}_t), \theta \rangle + \frac{1}{2\eta} \|\theta - \widetilde{\theta}_t\|_{\widetilde{\mathcal{H}}_t}^2 \right\}$$

where the local norm is $\widetilde{\mathcal{H}}_t = \mathcal{H}_t + \eta H_t(\widetilde{\theta}_t)$ and $\mathcal{H}_t = \sum_{i=1}^{t-1} H_i(\widetilde{\theta}_{i+1}) + \lambda I$.

The key closed-form equivalent update is:

$$\widetilde{\theta}_{t+1}' = \widetilde{\theta}_t - \eta \widetilde{\mathcal{H}}_t^{-1} g_t(\widetilde{\theta}_t), \quad \widetilde{\theta}_{t+1} = \text{Proj}_\Theta(\widetilde{\theta}_{t+1}')$$

This update depends only on the current sample and requires no stored historical data, yielding **$\mathcal{O}(1)$ per round**.

**Core design insight**: The local norm $\mathcal{H}_t$ captures second-order information (approximating the Hessian), which is essential for statistical efficiency. Standard OMD with a first-order approximation would sacrifice convergence rates; the second-order Taylor expansion here achieves both a closed-form solution and statistical efficiency.

### Three Online RLHF Settings

**Setting 1: Passive Data Collection** — The algorithm does not control data collection and applies the "pessimism in the face of uncertainty" principle:

$$\widetilde{J}_{T+1}(\pi) = \Phi^\top \widetilde{\theta}_{T+1} - \widetilde{\beta}_{T+1} \|\Phi\|_{\mathcal{H}_{T+1}^{-1}}$$

**Setting 2: Active Data Collection** — The algorithm queries the sample with the highest uncertainty:

$$(x_{t+1}, a_{t+1}, a_{t+1}') = \arg\max \|\phi(x,a) - \phi(x,a')\|_{\mathcal{H}_{t+1}^{-1}}$$

**Setting 3: Deployment-Time Adaptation** — During online deployment, exploitation and exploration are balanced: the first action maximizes the estimated reward, while the second jointly maximizes the reward and its distance from the first action.

### Practical Techniques

- **Hessian-Vector Product (HVP) + Conjugate Gradient**: Reduces the $\mathcal{O}(d^3)$ Hessian inversion to $\mathcal{O}(d)$
- **Rejection Sampling** to approximate model uncertainty: $n$ responses are sampled and reward ranking is used in place of exact confidence bounds
- A time-increasing schedule $\lambda_t = \lambda_0 \cdot \min(1, f(t/T))$ is adopted for $\lambda$ to approximate the cumulative Hessian effect

### Loss & Training

- Backbone models: Llama-3-8B-Instruct and Qwen2.5-7B-Instruct
- Feature dimension: $d = 4096$ (last-layer embeddings)
- Datasets: Ultrafeedback-binarized (64K prompts) and Mixture2
- Passive setting: $T = 30{,}000$ data points sampled randomly
- Active setting: only 6,400 samples selected from the full dataset

## Key Experimental Results

### Main Results

**Passive Data Collection (Llama-3-8B, Ultrafeedback)**:

| Metric | MLE | Ours (OMD) |
|--------|-----|------------|
| Convergence speed | Slow | Fast (especially advantageous when $T < 10{,}000$) |
| Evaluation accuracy | Lower | Higher |
| Evaluation loss | Higher | Lower |

In the low-data regime ($T < 10{,}000$), the OMD method achieves higher evaluation accuracy with fewer samples, validating the improvement in statistical efficiency.

**Active Data Collection (only 6,400 samples)**:

| Method | ACC (%) | Training Time (s) |
|--------|---------|-------------------|
| Rand-MLE | 69.51 ± 0.5 | 4876 ± 47 |
| Active-MLE | 69.82 ± 0.4 | 4982 ± 52 |
| Rand-OMD | 68.97 ± 0.6 | 1456 ± 31 |
| **Ours** | **70.43 ± 0.3** | **1489 ± 36** |

The OMD method reduces training time to approximately **one-third** of MLE while achieving higher accuracy. Active data selection further improves performance.

### Ablation Study

**Deployment-Time Adaptation**: Data is split into 20 chunks and processed sequentially. Multiple action selection strategies are compared:
- The proposed strategy (best + top-1/q percentile) outperforms random selection, best+second-best, and best+worst strategies under both MLE and OMD baselines
- Win rate analysis shows that the OMD method is competitive with MLE while incurring substantially lower computational cost

### Key Findings

1. The OMD method yields at least a $\sqrt{\kappa}$-fold improvement in statistical efficiency (theoretically guaranteed), where $\kappa$ can be exponentially large
2. Training time is reduced by approximately $3\times$ with equal or higher accuracy
3. Memory requirements drop from $\mathcal{O}(T)$ to $\mathcal{O}(1)$, which is highly significant for edge device deployment
4. Active data selection combined with OMD is the optimal configuration: highest accuracy and fastest training

## Highlights & Insights

1. **Strong unity of theory and practice**: Starting from a theoretical analysis of contextual dueling bandits, the paper derives a practically deployable LLM algorithm
2. **Unified treatment of three settings**: Passive, active, and deployment-time settings all follow naturally from the same OMD framework
3. **Practical significance of $\mathcal{O}(1)$ complexity**: The core bottleneck in online RLHF is continuously updating the reward model; this paper reduces the associated complexity from linear growth in the number of iterations to a constant
4. **Elegant local norm design**: Constructing the local norm using the Hessian at a lookahead point preserves second-order information while admitting a closed-form solution
5. **HVP + Conjugate Gradient for practicality**: Reducing complexity from $\mathcal{O}(d^3)$ to $\mathcal{O}(d)$ makes real-time updates feasible in a 4096-dimensional feature space

## Limitations & Future Work

1. Assumes a fixed feature map $\phi$ — in practice, the feature space changes during LLM fine-tuning
2. Based on the Bradley-Terry model — not extended to more general preference models such as Plackett-Luce
3. Assumes a linear reward function — real-world reward functions are typically nonlinear and require neural network approximation
4. Experiments operate only on last-layer embeddings — multi-layer features and full fine-tuning remain unexplored
5. Theoretical guarantees for rejection sampling-based uncertainty approximation are weak — a gap between theory and practice remains

## Related Work & Insights

- **Online RLHF (Dong et al., Guo et al.)**: Demonstrate that online iterative training outperforms offline training, but at high computational cost
- **Faury et al. (Implicit OMD for logistic bandits)**: Provide the theoretical foundation for the OMD approach, which this paper extends to the RLHF setting
- **Zhang & Sugiyama**: Source of the second-order Taylor expansion approximation idea
- **Das et al. (Active RLHF)**: Baseline for the active data collection setting; this paper substantially improves computational efficiency while maintaining equivalent statistical guarantees
- **Foster et al.**: Address the problem of enumerating an exponentially large response space, which is complementary to this paper's focus on iterative computational cost

**Takeaway**: The OMD + local norm combination is generalizable to other online learning problems (e.g., online DPO, online preference optimization) and is particularly valuable for personalized alignment on edge devices.

## Rating

- Novelty: ⭐⭐⭐⭐ — The application of OMD + local norm to RLHF is novel, though the underlying techniques originate from the bandit literature
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers three settings with both theoretical and empirical validation, though scale is limited (Llama-3 8B)
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous theoretical derivations, clear algorithmic descriptions, and well-organized setting taxonomy
- Value: ⭐⭐⭐⭐ — Addresses the core computational bottleneck of online RLHF with practical implications for edge deployment and continual learning

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Greedy Sampling Is Provably Efficient for RLHF](greedy_sampling_is_provably_efficient_for_rlhf.md)
- [\[NeurIPS 2025\] ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning](responserank_data-efficient_reward_modeling_through_preference_strength_learning.md)
- [\[ICML 2026\] Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling](../../ICML2026/llm_alignment/mitigating_reward_hacking_in_rlhf_via_bayesian_non-negative_reward_modeling.md)
- [\[ICML 2025\] Can RLHF be More Efficient with Imperfect Reward Models? A Policy Coverage Perspective](../../ICML2025/llm_alignment/can_rlhf_be_more_efficient_with_imperfect_reward_models_a_policy_coverage_perspe.md)
- [\[NeurIPS 2025\] Capturing Individual Human Preferences with Reward Features](capturing_individual_human_preferences_with_reward_features.md)

</div>

<!-- RELATED:END -->
