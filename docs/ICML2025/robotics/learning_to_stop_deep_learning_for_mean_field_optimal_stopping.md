---
title: >-
  [Paper Note] Learning to Stop: Deep Learning for Mean Field Optimal Stopping
description: >-
  [Robotics] First to formalize and computationally solve the Mean Field Optimal Stopping (MFOS) problem in discrete time with a finite state space, proving that MFOS approximates Multi-Agent Optimal Stopping (MAOS) at an $O(1/N)$ rate, and introducing two deep learning algorithms (Direct Approach DA and Dynamic Programming Principle DPP) evaluated on six scenarios with dimensions up to 300.
tags:
  - "Robotics"
date: 2026-05-08
content_hash: 3b234333a02b32e3
---

# Learning to Stop: Deep Learning for Mean Field Optimal Stopping

****Conference****: ICML 2025  
****arXiv****: [2410.08850](https://arxiv.org/abs/2410.08850)  
****Authors****: Lorenzo Magnino, Yuchen Zhu, Mathieu Laurière (NYU Shanghai, Georgia Tech)  
****Area****: Optimization & Control / Machine Learning  
****Keywords****: Optimal Stopping, Mean Field Control, Multi-Agent Systems, Deep Learning, Dynamic Programming  

## TL;DR

First to formalize and computationally solve the Mean Field Optimal Stopping (MFOS) problem in discrete time with a finite state space, proving that MFOS approximates Multi-Agent Optimal Stopping (MAOS) at an $O(1/N)$ rate, and introducing two deep learning algorithms (Direct Approach DA and Dynamic Programming Principle DPP) evaluated on six scenarios with dimensions up to 300.

## Background & Motivation

Optimal Stopping (OS) is a fundamental problem in optimization, focusing on determining when to terminate a stochastic process in an uncertain environment to minimize costs. Classic applications include American option pricing in finance, the secretary problem, and spatio-temporal stopping in robot path planning.

Traditional OS research primarily focuses on **single-agent** scenarios. However, many real-world problems involve **collaborative multi-agent optimal stopping** (MAOS), such as:
- UAV formations needing to stop at different times to match a target distribution
- Multi-robot searching where each robot must decide when to stop exploration

The computational complexity of MAOS scales exponentially with the number of agents $N$. Mean Field Approximation is a classic tool for handling large-scale multi-agent systems—as $N \to \infty$, agents become independent and identically distributed, and the empirical distribution converges to their own law (propagation of chaos).

**Limitations of Prior Work**:
- Talbi et al. (2023, 2024) studied continuous spatio-temporal MFOS from a purely theoretical standpoint, but their PDE method is intractable in infinite-dimensional spaces.
- Deep learning methods have only been applied to **single-agent** OS (Becker et al. 2019; Herrera et al. 2023).
- **No prior work** has investigated MFOS in discrete time and finite state spaces, and no computational methods exist.

The core motivation of this paper is: **to establish the theoretical foundation of MFOS in a discrete-time, finite-space framework and propose the first scalable deep learning solution.**

## Method

### Overall Architecture

The paper constructs a complete theory-algorithm pipeline from finite-agent to the mean field limit:

```
N-agent MAOS → (N→∞) → MFOS → DPP → Deep Learning Solver
                         ↑                    ↓
            O(1/N) Approximation Guarantee   DA / DPP Algorithms
```

**Problem Definition**: $N$ agents move in a finite state space $\mathcal{X}$, where each agent stops at each step with probability $p_n(x)$. Upon stopping, a cost $\Phi(x, \mu)$ is incurred (depending on the current state and the population distribution). The objective is to minimize the social cost:

$$J(p) = \sum_{m=0}^{T} \sum_{(x,a) \in \mathcal{S}} \nu_m^p(x,a) \cdot \Phi(x, \mu_m^p) \cdot a \cdot p_m(x)$$

where $\nu_m^p$ is the distribution of the extended state $(x, a)$, and $a \in \{0,1\}$ indicates whether the agent is still active.

### Key Theoretical Contributions

**Theorem 2.2 ($\epsilon$-Approximation)**: Under Lipschitz conditions, the social cost difference between applying the MFOS optimal strategy $p^*$ to the $N$-agent problem and the $N$-agent optimal strategy is $O(1/N)$:

$$J_N(p^*, \ldots, p^*) - J_N(\hat{p}, \ldots, \hat{p}) = O(1/N)$$

**Theorem 3.1 (Dynamic Programming Principle)**: The value function of MFOS satisfies the Bellman equation:

$$V_n(\nu) = \inf_{h \in \mathcal{H}} \left[ \bar{\Phi}(\nu, h) + V_{n+1}(\bar{F}(\nu, h)) \right]$$

where $\bar{\Phi}(\nu, h) = \sum_{x} \nu(x,1) \Phi(x, \nu^X) h(x)$ represents the single-step cost. This is the **first** work establishing a DPP for discrete-time MFOS.

### Key Designs

| Feature | Direct Approach (DA, Algorithm 1) | Dynamic Programming Principle (DPP, Algorithm 2) |
|------|------|------|
| **Optimization Objective** | Directly minimize the overall social cost $J(p)$ | Backward induction per time step using DPP |
| **Network Input** | $(x, \nu, t)$: state + distribution + time | $(x, \nu)$: state + distribution (no time required) |
| **Output** | Stopping probability $p_n(x)$ | Single-step stopping probability $h(x)$ |
| **Training Mode** | Forward simulation of complete trajectories, end-to-end optimization | Backward step-by-step training from $T$ to $0$ |
| **Memory Requirement** | Scales linearly with time $T$ | Constant, independent of $T$ |
| **Convergence Speed** | Faster (given sufficient compute) | Slower but memory-friendly |
| **Applicable Scenarios** | Short duration / Low-dimensional | Long duration / High-dimensional |

### Network Architecture

- **State Encoding**: One-hot vector processed by an embedding layer.
- **Time Encoding** (DA only): Sinusoidal embedding $\to$ 2-layer MLP + SiLU.
- **Backbone Network**: $k$ residual blocks, each with 4 linear layers + SiLU, hidden dimension $D$.
- **Output Layer**: GroupNorm $\to$ 3-layer MLP + SiLU.
- 1D experiments: $k=3, D=128$; 2D experiments: $k=5, D=256$.

### Loss & Training

**DA Loss**: Directly minimize the social cost
$$\mathcal{L}_{\text{DA}} = \sum_{n=0}^{T} \bar{\Phi}(\nu_n^p, p_n)$$

**DPP Loss**: For each time step $n$, minimize
$$\mathcal{L}_n = \bar{\Phi}(\nu, h) + V_{n+1}(\bar{F}(\nu, h))$$
where $V_{n+1}$ is provided by the network already trained for subsequent time steps.

## Key Experimental Results

### Main Results: Overview of 6 Scenarios

| Experiment | State Space $|\mathcal{X}|$ | Input Dimension $3|\mathcal{X}|$ | Time $T$ | Dynamics | Cost Dependence |
|------|------|------|------|------|------|
| Ex1: Tend to Uniform | 5 | 15 | 4 | Deterministic right shift | Local distribution $\mu(x)$ |
| Ex2: Rolling Dice | 6 | 18 | 5 | Stochastic (uniform) | State $x$ |
| Ex3: Congested Crowd | 6 | 18 | 4 | Stochastic + congestion | State $x$ |
| Ex4: Distributional Cost | 7 | 21 | 3 | Random walk | KL divergence to target distribution |
| Ex5: 2D Uniform | 25 | 75 | 4 | Deterministic right shift | Local distribution $\mu(x)$ |
| Ex6: Drone Swarm | 100 | **300** | **50** | Diffusion + random obstacle | Terminal distribution matching |

### Key Findings

**Example 1 (Tend to Uniform)**: The analytical optimal value is $V^* = \frac{T+2}{2(T+1)} = 0.6$. Both DA and DPP converge to the analytical solution within hundreds of iterations.

**Example 2 (Rolling Dice)**:
- The optimal value for asynchronous stopping is $V^* = 1.6525$, which is recovered by both DA and DPP.
- The optimal value for synchronous stopping is $\tilde{V}^* = 3.25$, which is significantly higher than the asynchronous case—demonstrating that the asynchronous stopping class is crucial for this type of problem.

**Example 3 (Congestion)**: The congestion coefficient $C_{\text{cong}} = 0.8$ significantly slows down population movement. The asynchronous stopping cost is substantially lower than that of synchronous stopping, verifying the necessity of randomized stopping times.

**Example 6 (Drone Swarm, Dimension 300)**:
- DPP successfully drives the initial distribution to match the "M", "F", "O", and "S" letter-shaped target distributions.
- The learned policy exhibits **robust generalization** to initial distributions—different randomized initial distributions are successfully driven to match the targets.
- The memory required for DA training increases drastically with $T=50$, whereas DPP requires constant memory.

### Ablation Study

| Comparison Dimension | DA | DPP |
|------|------|------|
| Ex1 Convergence Iterations | ~500 | ~1000 (per time step) |
| Ex6 Runnability | Memory-constrained | Stable operation |
| Learning Rate Sensitivity | $10^{-4}$ is most stable | $10^{-4}$ is most stable |
| MFOS vs MAOS Approximation (Ex1) | $L_2$ distance decays as $\sim N^{-1/2}$ | — |

**Hyperparameter Sweep** (Appendix F, Example 1): A learning rate of $10^{-2}$ leads to unstable training; $10^{-3}$ is acceptable; $10^{-4}$ is the most stable. The training hyperparameters include the AdamW optimizer, a batch size of 128, and a maximum of $10^4$ iterations.

### Computational Resources

All experiments were completed on an RTX 4090 GPU ($\le 3$ minutes per scenario) or an M2 MacBook Pro ($\le 7$ minutes), demonstrating the high efficiency and scalability of the proposed method.

## Highlights & Insights

1. **Theoretical-Algorithmic Closed Loop**: Establishing a complete pipeline: $O(1/N)$ approximation guarantee from MAOS to MFOS $\to$ DPP $\to$ two deep learning algorithms.
2. **Necessity of Randomized Stopping Times**: A simple counterexample (Example 1 in Appendix) proves that deterministic stopping strategies are suboptimal in the mean field setting, necessitating the introduction of randomization.
3. **Asynchronous vs. Synchronous Stopping**: The asynchronous stopping class significantly outperforms the synchronous counterpart across multiple scenarios, revealing the importance of individualized decision-making in multi-agent stopping problems.
4. **Memory Advantage of DPP**: The constant memory requirement makes long-horizon high-dimensional problems tractable, which is an advantage DA cannot match.
5. **Distributional Robustness of Policies**: The stopping rules learned in Example 6 remain effective for unseen initial distributions, demonstrating excellent generalization capabilities.

## Limitations & Future Work

- The convergence of the deep learning algorithms is not theoretically proven (an inherent difficulty in deep network analysis).
- The theoretical comparison between synchronous and asynchronous stopping is yet to be completed.
- All experiments are performed on synthetic scenarios, lacking large-scale validation on real-world datasets.
- The current framework assumes that the model (transition probabilities) is known and has not been extended to model-free/reinforcement learning settings.
- Only cooperative multi-agent settings are considered, leaving competitive scenarios (Mean Field Games) for future research.

## Related Work & Insights

- **Deep Single-Agent OS**: Becker et al. (2019) on deep optimal stopping; Herrera et al. (2023) on randomized neural network approaches—this paper extends these methods to the mean field setting.
- **Mean Field Control (MFC)**: Carmona & Laurière (2023) on deep learning for MFC; Gu et al. (2023) on reinforcement learning for MFC—transforming the stopping problem into a sub-class of MFC is a conceptual contribution of this work.
- **Continuous MFOS**: Talbi et al. (2023, 2024) established the theoretical framework in continuous space-time but without computability—this work turns to a discrete framework to make computation tractable.
- **Robotics Applications**: Best et al. (2017, 2018) on active search and spatio-temporal optimal stopping for robots—the mean field approach proposed here provides a scalable solution for large-scale robotic swarms.

**Insights**: Interpreting MFOS as a special case of MFC offers a fresh perspective at the intersection of both fields. This framework can be extended to more multi-agent scenarios involving "when to stop," such as early stopping of models in distributed training, termination conditions of multi-robot exploration, etc.

## Rating

| Dimension | Score (1-5) | Comment |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | First to formalize and solve discrete MFOS; the MFC perspective is a pioneering conceptual contribution. |
| Theoretical Depth | ⭐⭐⭐⭐⭐ | Complete approximation theorems and DPP proofs. |
| Experimental Thoroughness | ⭐⭐⭐⭐ | 6 scenarios coverage from simple to high-dimensional, though lacking real-world applications. |
| Value | ⭐⭐⭐ | Elegant framework, but actual application scenarios need to be further developed. |
| Writing Quality | ⭐⭐⭐⭐ | Well-structured, balanced between theory and experiments. |
| **Overall Score** | **⭐⭐⭐⭐** | A theoretically solid, pioneering work that lays the computational foundation for MFOS. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Rolling the DICE on Idiomaticity: How LLMs Fail to Grasp Context](../../ACL2025/robotics/dice_idiomaticity.md)
- [\[NeurIPS 2025\] Time Reversal Symmetry for Efficient Robotic Manipulations in Deep Reinforcement Learning](../../NeurIPS2025/robotics/time_reversal_symmetry_for_efficient_robotic_manipulations_in_deep_reinforcement.md)
- [\[ICLR 2026\] A Primer on SO(3) Action Representations in Deep Reinforcement Learning](../../ICLR2026/robotics/a_primer_on_so3_action_representations_in_deep_reinforcement_learning.md)
- [\[ICCV 2025\] Certifiably Optimal Anisotropic Rotation Averaging](../../ICCV2025/robotics/certifiably_optimal_anisotropic_rotation_averaging.md)
- [\[NeurIPS 2025\] PROFIT: A Specialized Optimizer for Deep Fine Tuning](../../NeurIPS2025/robotics/profit_a_specialized_optimizer_for_deep_fine_tuning.md)

</div>

<!-- RELATED:END -->
