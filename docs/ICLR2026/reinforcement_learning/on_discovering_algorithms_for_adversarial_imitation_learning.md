---
title: >-
  [Paper Note] On Discovering Algorithms for Adversarial Imitation Learning
description: >-
  [ICLR 2026][Reinforcement Learning][adversarial imitation learning] This paper proposes DAIL — the first meta-learned adversarial imitation learning algorithm. It decomposes AIL into two stages (density ratio estimation and reward assignment), and employs LLM-guided evolutionary search to automatically discover an optimal reward assignment (RA) function $r_{\text{disc}}$, achieving generalization to unseen environments and policy optimizers while surpassing all manually designed baselines.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - adversarial imitation learning
  - reward assignment function
  - LLM-guided evolution
  - meta-learning
  - training stability
date: 2026-05-08
content_hash: e25d0cf8c2db4bdc
---

# On Discovering Algorithms for Adversarial Imitation Learning

**Conference**: ICLR 2026
**arXiv**: [2510.00922](https://arxiv.org/abs/2510.00922)
**Code**: None
**Area**: Imitation Learning / Meta-Learning
**Keywords**: adversarial imitation learning, reward assignment function, LLM-guided evolution, meta-learning, training stability

## TL;DR

This paper proposes DAIL — the first meta-learned adversarial imitation learning algorithm. It decomposes AIL into two stages (density ratio estimation and reward assignment), and employs LLM-guided evolutionary search to automatically discover an optimal reward assignment (RA) function $r_{\text{disc}}$, achieving generalization to unseen environments and policy optimizers while surpassing all manually designed baselines.

## Background & Motivation

**Background**: Adversarial imitation learning (AIL) is the most effective imitation learning paradigm under limited expert demonstrations. Inspired by GANs, AIL formalizes the learning process as an adversarial game between a discriminator (distinguishing expert from policy trajectories) and a policy (generating trajectories close to the expert). From a divergence minimization perspective, AIL naturally decomposes into two stages: (1) **Density Ratio (DR) Estimation** — the discriminator estimates the occupancy ratio $\frac{\rho_E}{\rho_\pi}$ of state-action pairs under the expert vs. policy; (2) **Reward Assignment (RA)** — mapping the density ratio to a scalar reward signal for policy optimization.

**Limitations of Prior Work**:
- AIL training is unstable, suffering from GAN-like training difficulties — gradient signal quality directly affects policy improvement.
- Substantial research has focused on improving discriminator training in stage (1) (e.g., C-GAIL, Diffusion-Reward), while the RA function in stage (2) has been largely neglected.
- Existing RA functions (GAIL's softplus, AIRL's log-ratio, FAIRL's exponential decay) are all manually derived from $f$-divergence theory, relying on human intuition and potentially being far from optimal.

**Key Challenge**: Manually designed RA functions face a fundamental tension between theoretical elegance and practical training stability — GAIL over-rewards low-quality state-action pairs, FAIRL's exponential decay leads to unstable training, and AIRL's negative rewards induce premature termination.

**Goal**: Rather than manual design, DAIL uses LLM-guided evolutionary search to directly discover optimal RA functions from a performance-driven perspective, enabling meta-learning of AIL algorithms.

## Method

### Overall Architecture

The DAIL pipeline operates at two levels:
1. **Outer Level: Meta-Learning** — LLM-guided evolutionary search optimizes the RA function $r_f$, minimizing the Wasserstein distance between the trained policy and the expert.
2. **Inner Level: Standard AIL Loop** — given $r_f$, iteratively performs policy rollout → discriminator training (density ratio estimation) → reward assignment → policy improvement.
3. The meta-learning objective is formalized as a bi-level optimization:
$$\min_f \mathcal{W}(\rho_E, \rho_{\pi^*}; f) \quad \text{s.t.} \quad \pi^* = \arg\max_\pi r_f(\rho_E \| \rho_\pi)$$

### Key Design 1: Two-Stage Decomposition of AIL and RA Function Analysis

The core insight is to decompose the reward signal formation process in AIL into two independent stages and analyze the impact of different RA functions on training dynamics:

| Divergence | Algorithm | RA Function $r_f(\ell)$ | Properties |
|---|---|---|---|
| Forward KL | FAIRL | $-\ell \cdot e^{\ell}$ | Exponentially unbounded decay, unstable training |
| Backward KL | AIRL | $\ell$ | Linear, large negative rewards → premature termination |
| Jensen-Shannon | GAIL | $\text{softplus}(\ell)$ | Over-rewards low-quality samples |
| Unnamed $f$-div | GAIL-heuristic | $-\text{softplus}(-\ell)$ | Primarily negative rewards, only incentivizes matching |

Here $\ell = \log \frac{\rho_E(s,a)}{\rho_\pi(s,a)}$ denotes the log density ratio. The response curves of different RA functions to the density ratio differ substantially, directly determining gradient signal informativeness and training stability.

### Key Design 2: LLM-Guided Evolutionary Search

Since bi-level optimization requires backpropagation through the entire AIL training loop — which is computationally infeasible — black-box optimization is adopted instead:

1. **Initial Population**: RA functions from GAIL, AIRL, FAIRL, and GAIL-heuristic serve as the initial population.
2. **Fitness Evaluation**: Each candidate $r_f$ trains a policy to convergence; the Wasserstein distance between rollouts and the expert is computed as the fitness score.
3. **Crossover and Mutation**: A pair of parents $\{r_{f_1}, r_{f_2}\}$ is sampled along with their fitness scores and provided to an LLM (GPT-4.1-mini), which is prompted to combine the strengths of both parents and generate an offspring $r_{f_3}$.
4. **Selection**: Each generation evaluates $M \times N$ candidates, retaining the Top-$K$ for the next generation.
5. RA functions are represented directly as **Python code**, ensuring interpretability and expressiveness.

The search is conducted on Minatar SpaceInvaders, evaluating 200 candidate functions in approximately 3 hours.

### Key Design 3: The Discovered RA Function $r_{\text{disc}}$

The optimal RA function discovered by evolutionary search is:

$$r_{\text{disc}}(x) = 0.5 \cdot \text{sigmoid}(x) \cdot [\tanh(x) + 1]$$

$r_{\text{disc}}$ has the following key properties:
- **Bounded $[0,1]$** — bounded rewards have been shown to stabilize deep RL training.
- **S-shaped curve** with steeper and right-shifted gradients compared to standard sigmoid, providing informative gradients in the $[-1, 0]$ interval.
- **Saturates to zero** for $x \lesssim -1.8$ (near-random policy behavior) — effectively filtering low-quality state-action pairs.
- Two independent evolutionary search runs yield Top-5 functions with highly similar structures, confirming search stability.

## Key Experimental Results

### Main Results: Cross-Environment Generalization

Evaluation on Brax (MuJoCo) and Minatar benchmarks outside the search environment; all methods share the same hyperparameters, differing only in the RA function:

| Method | Brax Mean ↑ | Brax Median ↑ | Minatar Mean ↑ | Minatar Median ↑ |
|---|---|---|---|---|
| GAIL | ~0.65 | ~0.70 | ~0.55 | ~0.60 |
| AIRL | ~0.70 | ~0.75 | ~0.30 | ~0.25 |
| FAIRL | ~0.40 | ~0.35 | ~0.35 | ~0.30 |
| GAIL-heuristic | ~0.55 | ~0.60 | ~0.25 | ~0.20 |
| **DAIL** | **~0.75** | **~0.72** | **~0.85** | **~0.90** |

Key findings:
- DAIL **substantially** outperforms all baselines on Minatar (significant leads in both Mean and Median).
- On Brax, DAIL achieves the best performance on most metrics, with statistically significant superiority in Mean over all baselines.
- AIRL and GAIL-heuristic perform poorly on Minatar — their predominantly negative rewards induce premature termination.

### Ablation Study: Analyzing the Sources of DAIL's Advantage

| Analysis Dimension | Finding | Quantitative Metric |
|---|---|---|
| Search environment generalization | Search on SpaceInvaders → generalize to other environments | $\mathcal{W}$ distance reduced by 20%, normalized return improved by 12.5% |
| Policy optimizer generalization | PPO search → evaluate on A2C | DAIL significantly outperforms GAIL on A2C as well |
| Discriminator regularization robustness | 5 regularization strategies (none/w-decay/entropy/spectral/grad-pen) | DAIL outperforms GAIL in 3/5 settings |
| Policy entropy convergence | DAIL policy converges to lower entropy | Approaches the entropy level of the ground-truth-reward PPO baseline |
| Component ablation | $r_{\text{disc}}$ vs sigmoid vs $0.5[\tanh+1]$ | $r_{\text{disc}}$ > $0.5[\tanh+1]$ > sigmoid |
| Search stability | Two independent evolutionary search runs | Top-5 RA functions exhibit highly similar structures |

Policy entropy analysis reveals the core reason for DAIL's more stable training: $r_{\text{disc}}$ saturates to zero for $x \lesssim -1.8$, filtering noisy rewards from near-random policy behavior. In contrast, GAIL still assigns high positive rewards at $x = -2$, remaining overly sensitive to low-quality actions and producing noisy reward signals.

### Detailed Performance Under Different Discriminator Regularization Strategies

| Algorithm | Environment | None | W-Decay | Entropy | Spectral | Grad-Pen |
|---|---|---|---|---|---|---|
| DAIL | Asterix | 0.88±0.03 | 1.33±0.03 | 0.12±0.01 | 0.92±0.03 | 0.66±0.03 |
| DAIL | Breakout | 0.81±0.07 | 0.74±0.08 | 0.91±0.02 | 0.77±0.07 | 1.01±0.00 |
| DAIL | SpaceInv | 0.71±0.07 | 0.81±0.01 | 0.80±0.01 | 0.70±0.09 | 0.90±0.00 |
| DAIL | **Overall** | **0.80±0.03** | **0.96±0.03** | 0.61±0.01 | **0.80±0.04** | **0.85±0.01** |
| GAIL | Asterix | 1.18±0.03 | 1.44±0.03 | 0.48±0.03 | 0.22±0.03 | 0.52±0.04 |
| GAIL | Breakout | 0.76±0.07 | 0.52±0.10 | 0.89±0.01 | 0.33±0.10 | 0.85±0.07 |
| GAIL | SpaceInv | 0.61±0.09 | 0.34±0.09 | 0.81±0.00 | 0.42±0.08 | 0.81±0.03 |
| GAIL | **Overall** | 0.85±0.04 | 0.76±0.04 | **0.73±0.01** | 0.32±0.05 | 0.73±0.03 |

## Highlights & Insights

### Strengths
- **Precise Problem Identification**: The paper is the first to systematically reveal the critical role of RA functions in AIL training stability, filling a notable gap in the literature.
- **Methodological Innovation**: Introducing meta-learning into RA function discovery for AIL represents a paradigm shift from manual design to data-driven approaches.
- **In-Depth Analysis**: The advantages of DAIL are clearly explained through multiple analytical lenses, including density ratio distributions, policy entropy, and component ablations.
- **Strong Generalization**: Consistent advantages are demonstrated across environments (Brax + Minatar), optimizers (PPO → A2C), and regularization strategies.

### Limitations & Future Work
- The discovered $r_{\text{disc}}$ does not correspond to a valid $f$-divergence, lacking theoretical convergence guarantees.
- The RA function remains static during training and does not leverage training state information (e.g., remaining update steps, observed density ratio distributions) for adaptive adjustment.
- The search is conducted on a single environment (SpaceInvaders); searching over a larger set of environments may yield stronger functions.
- Validation on more complex benchmarks (e.g., Atari-57 / Procgen) is absent.

### Rating
⭐⭐⭐⭐ — Precise problem identification, novel methodology, and thorough analysis, tempered by the lack of theoretical guarantees and limited search scale.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Model Predictive Adversarial Imitation Learning for Planning from Observation](model_predictive_adversarial_imitation_learning_for_planning_from_observation.md)
- [\[ICLR 2026\] Latent Wasserstein Adversarial Imitation Learning](latent_wasserstein_adversarial_imitation_learning.md)
- [\[ICLR 2026\] Near-Optimal Second-Order Guarantees for Model-Based Adversarial Imitation Learning](near-optimal_second-order_guarantees_for_model-based_adversarial_imitation_learn.md)
- [\[ICLR 2026\] Learning to Generate Unit Test via Adversarial Reinforcement Learning](learning_to_generate_unit_test_via_adversarial_reinforcement_learning.md)
- [\[ICLR 2026\] Boolean Satisfiability via Imitation Learning](boolean_satisfiability_via_imitation_learning.md)

</div>

<!-- RELATED:END -->
