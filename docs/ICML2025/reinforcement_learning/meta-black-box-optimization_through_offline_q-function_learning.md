---
title: >-
  [Paper Note] Meta-Black-Box-Optimization through Offline Q-function Learning (Q-Mamba)
description: >-
  [ICML2025][Reinforcement Learning][Meta-BBO] This paper proposes Q-Mamba, the first offline MetaBBO framework. By integrating Q-function decomposition, conservative Q-learning, and the Mamba architecture, it achieves comparable or superior BBO algorithm configuration performance with less than half the training budget of online methods.
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "Meta-BBO"
  - "Offline Reinforcement Learning"
  - "Q-function Decomposition"
  - "Mamba"
  - "Dynamic Algorithm Configuration"
  - "Conservative Q-Learning"
date: 2026-05-08
content_hash: ec843fd4c9c387b3
---

# Meta-Black-Box-Optimization through Offline Q-function Learning (Q-Mamba)

**Conference**: ICML2025  
**arXiv**: [2505.02010](https://arxiv.org/abs/2505.02010)  
**Code**: Open-sourced  
**Area**: Meta-Learning / Black-Box Optimization  
**Keywords**: Meta-BBO, Offline Reinforcement Learning, Q-function Decomposition, Mamba, Dynamic Algorithm Configuration, Conservative Q-Learning

## TL;DR

This paper proposes Q-Mamba, the first offline MetaBBO framework. By integrating Q-function decomposition, conservative Q-learning, and the Mamba architecture, it achieves comparable or superior BBO algorithm configuration performance with less than half the training budget of online methods.

## Background & Motivation

**Meta-Black-Box-Optimization (MetaBBO)** is a bi-level "learning to optimize" paradigm: a meta-level neural network policy $\pi_\theta$ dynamically configures the hyperparameters of a bottom-level BBO algorithm (such as Differential Evolution, DE) to achieve optimal performance over a problem distribution.

Existing MetaBBO methods face two core bottlenecks:

**Learning Effectiveness**: Advanced BBO algorithms have vast configuration spaces (e.g., MadDE has more than 10 hyperparameters). The joint action space grows exponentially with dimension ($M^K$), making it difficult for RL to learn effective policies.

**Training Efficiency**: Current methods all adopt an online RL paradigm, where bottom-level optimization requires hundreds of simulation steps each time, making trajectory sampling extremely time-consuming (training typically takes 25-28 hours).

**Core Problem**: Can offline RL be used to learn DAC policies from pre-collected trajectory data while guaranteeing learning performance?

## Method

### Overall Architecture

Q-Mamba consists of three core designs:

1. **Offline E&E Dataset Collection**: Mixing exploration and exploitation data.
2. **Conservative Q-Learning Loss**: Decomposed Bellman update + conservative regularization.
3. **Mamba-based Q-Learner**: Long-sequence Q-function learning based on State Space Models (SSMs).

### 1. Q-function Decomposition Mechanism

The joint action space $A = (A_1, \ldots, A_K)$ is decomposed into dimension-wise sequential decisions. For the decomposed Q-function of the $i$-th dimension:

$$
Q(a_i^t | s^t) \leftarrow \begin{cases}
\max_{a_{i+1}^t} Q(a_{i+1}^t | s^t, a_{1:i}^t), & \text{if } i < K \\
R(s^t, a_{1:K}^t) + \gamma \max_{a_1^{t+1}} Q(a_1^{t+1} | s^{t+1}), & \text{if } i = K
\end{cases}
$$

- Each hyperparameter $A_i$ is discretized into $M=16$ bins.
- A 5-bit binary code is used to represent the action token (`00000` to `01111`), with `11111` serving as the start token.
- This converts the exponential space $M^K$ to a linear scale $K \times M$.

### 2. Offline E&E Dataset (Exploration & Exploitation)

A dataset of $D = 10\text{K}$ trajectories is collected with a mixing ratio of $\mu = 0.5$:

- **Exploitation Data** ($\mu \cdot D$): High-quality trajectories obtained by rolling out pre-trained MetaBBO methods (RLPSO, LDE, GLEET) on the problem distribution.
- **Exploration Data** ($(1-\mu) \cdot D$): Exploration trajectories obtained by controlling hyperparameters via a random policy.

Mixing these two types of data simultaneously provides high-quality experience and diversity, which reduces offline learning bias.

### 3. Conservative Q-Learning Loss

The training target is a three-branch combined loss (computed for each timestep $t$ in trajectory $\tau$, each action dimension $i$, and each bin $j$):

$$
J(Q_{i,j}^t | \theta) = \begin{cases}
\frac{1}{2}(Q_{i,j}^t - \max_j Q_{i+1,j}^t)^2, & i < K,\ j = a_i^t \\
\frac{\beta}{2}[Q_{i,j}^t - (r^t + \gamma \max_j Q_{1,j}^{t+1})]^2, & i = K,\ j = a_i^t \\
\frac{\lambda}{2}(Q_{i,j}^t - 0)^2, & j \neq a_i^t
\end{cases}
$$

- **First two branches**: Standard decomposed Bellman targets (TD error), where the last dimension is weighted by $\beta=10$ to anchor the backpropagation accuracy.
- **Third branch**: CQL conservative regularization, which pushes the Q-values of bins not selected in the trajectory towards 0 (lower bound), mitigating overestimation caused by distribution shift.
- $\lambda=1$ balances the conservative term.

### 4. Mamba-based Q-Learner Architecture

$Q_i^t$ is output dimension-by-dimension in an autoregressive manner:

$$
\mathbb{O}_i^t, h_i^t = \text{mamba\_block}([s^t, \text{token}(a_{i-1}^t)], h_{i-1}^t | W_{mamba})
$$

$$
Q_i^t = \sigma(\text{Linear}(\mathbb{O}_i^t | W_{head}, b_{head}))
$$

- Input: The current optimization state $s^t \in \mathbb{R}^9$ (6-dimensional population statistical features + 3-dimensional temporal features) concatenated with the action token from the previous step.
- Mamba Block: Processed through a Selective State Space Model (SSM), where the parameters $\bar{B}$ and $C$ are functions of the input (time-varying), enabling flexible learning of long-term and short-term dependencies.
- Q-value Head: A linear layer + Leaky ReLU, outputting a 16-dimensional Q-value vector.
- Choosing $a_i^t = \arg\max_j Q_{i,j}^t$, which is then tokenized and fed into the next step.

**Reasons for choosing Mamba over Transformer**:

- MetaBBO involves $T \times K$ decision steps (long sequences of several thousand steps). Mamba flexibly processes long-term and short-term dependencies through selective SSM.
- Hardware-friendly parallel scan (PrefixSum) algorithm, offering memory efficiency equivalent to FlashAttention.
- Faster training speed compared to Transformer (13h vs. 16h).

Training utilizes AdamW, with a learning rate of $5 \times 10^{-3}$, 300 epochs, and a batch size of 64.

## Key Experimental Results

### In-Distribution Generalization Performance (8 unseen test problems, 19 independent runs, cumulative performance metric Perf)

| Method | Type | Alg0 (K=3) | Alg1 (K=10) | Alg2 (K=16) | Training/Inference Time |
|------|------|-----------|------------|------------|-------------|
| RLPSO | Online | 0.9855 | 0.9953 | 0.9914 | 28h / 11s |
| LDE | Online | 0.9563 | 0.9877 | 0.9904 | 28h / 12s |
| GLEET | Online | 0.9616 | 0.9938 | 0.9910 | 25h / 13s |
| DT | Offline | 0.9325 | 0.6764 | 0.8706 | 13h / 10s |
| DeMa | Offline | 0.9492 | 0.9015 | 0.9159 | 12h / 10s |
| QDT | Offline | 0.9683 | 0.9917 | 0.9919 | 20h / 12s |
| QT | Offline | 0.9729 | 0.9955 | 0.9926 | 20h / 12s |
| Q-Transformer | Offline | 0.9666 | 0.9951 | 0.9895 | 16h / 11s |
| **Q-Mamba** | **Offline** | **0.9889** | **0.9973** | **0.9950** | **13h / 10s** |

**Key Findings**: Q-Mamba achieves optimal performance across three algorithms with different configuration space scales, while requiring less than half the training time of online methods.

### Out-of-Distribution Generalization (Neuroevolution / MuJoCo Continuous Control)

- Trained solely on synthetic BBOB problems ($\le 50$ dimensions), Q-Mamba achieves zero-shot transfer to MLP policy evolution tasks with thousands of parameters.
- It achieves comparable performance to online MetaBBO baselines on MuJoCo tasks such as Ant, HalfCheetah, and Hopper.

### Ablation Study

| Setting | $\lambda=0$ | $\lambda=1$ | $\lambda=10$ |
|------|-----------|-----------|------------|
| $\beta=1$ | 0.9756 | 0.9828 | 0.9855 |
| $\beta=10$ | 0.9833 | **0.9889** | 0.9857 |

- Removing conservative regularization ($\lambda=0$) leads to a remarkable performance drop $\rightarrow$ The conservative term is critical for mitigating distribution shift.
- $\beta=10$ weights the Q-value of the final dimension $\rightarrow$ Improving overall accuracy by anchoring backpropagation.
- A data mixing ratio of $\mu=0.5$ (equal proportion mixing) achieves the best performance.

## Highlights & Insights

1. **First Offline MetaBBO Framework**: Breaks the paradigm of sole reliance on online RL in this field, improving training efficiency by $\ge 2\times$.
2. **Q-function Decomposition**: Elegantly transforms the exponential joint action space into linear sequential decisions, lowering the learning difficulty.
3. **Conservative Q-learning + Decomposed Bellman**: The combined loss is exquisitely designed, and the trick of weighting the last dimension with $\beta$ is theoretically supported (as a backpropagation anchor).
4. **Mamba vs. Transformer**: Empirically demonstrates that Mamba outperforms Transformer in long-sequence Q-learning, leveraging the selective memory advantages of time-varying SSM.
5. **Strong Generalization**: Achieves zero-shot transfer from synthetic problems of $\le 50$ dimensions to Neuroevolution scenarios with thousands of dimensions.

## Limitations & Future Work

1. **Offline Data Reliance on Pre-trained Baselines**: The E&E dataset requires training online methods like RLPSO, LDE, or GLEET beforehand; this initial cost is not fully discussed.
2. **Limited to Evolutionary Algorithms**: Validation has not been conducted on other BBO paradigms like gradient-based optimizers or Bayesian optimization.
3. **Information Loss from Action Discretization**: Uniform discretization into 16 bins might discard the fine-grained details of continuous hyperparameters.
4. **Limited OOD Generalization Experiments**: Only one type of OOD scenario (Neuroevolution) was tested; more real-world scenarios remain to be validated.
5. **Mamba Architecture's Reliance on CUDA**: Hardware acceleration is restricted to specific GPUs, limiting deployment flexibility.

## Related Work & Insights

- **ConfigX** (Guo et al., 2024b): Constructs a large-scale algorithm space; this work selects bottom-level algorithms based on it.
- **Q-Transformer** (Chebotar et al., 2023): Q-decomposition + Transformer; this work gets improvements by replacing it with Mamba.
- **CQL** (Kumar et al., 2020): Its conservative Q-learning ideology is directly absorbed in this paper.
- **Mamba** (Gu & Dao, 2023): Selective State Space Model; this work is the first to apply it to MetaBBO.
- **Decision Transformer** (Chen et al., 2021): Serves as a baseline for the conditional imitation learning paradigm.

## Rating

- Novelty: ⭐⭐⭐⭐ — The first combination of offline MetaBBO + Q-decomposition + Mamba exhibits significant innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Relatively comprehensive, featuring comparisons with 9 baselines, ablation studies, and OOD generalization.
- Writing Quality: ⭐⭐⭐⭐ — Clear mathematical derivations, intuitive framework diagrams, and smooth writing style.
- Value: ⭐⭐⭐⭐ — Introduces a new offline RL paradigm to the MetaBBO community, demonstrating high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MetaBox-v2: A Unified Benchmark Platform for Meta-Black-Box Optimization](../../NeurIPS2025/reinforcement_learning/metabox-v2_a_unified_benchmark_platform_for_meta-black-box_optimization.md)
- [\[NeurIPS 2025\] Optimizing the Unknown: Black Box Bayesian Optimization with Energy-Based Model and Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/optimizing_the_unknown_black_box_bayesian_optimization_with_energy-based_model_a.md)
- [\[ICML 2025\] Graph-Supported Dynamic Algorithm Configuration for Multi-Objective Combinatorial Optimization](graph-supported_dynamic_algorithm_configuration_for_multi-objective_combinatoria.md)
- [\[ICML 2025\] Online Pre-Training for Offline-to-Online Reinforcement Learning](online_pre-training_for_offline-to-online_reinforcement_learning.md)
- [\[ICML 2025\] Wasserstein Policy Optimization](wasserstein_policy_optimization.md)

</div>

<!-- RELATED:END -->
