---
title: >-
  [Paper Note] MDNS: Masked Diffusion Neural Sampler via Stochastic Optimal Control
description: >-
  [NeurIPS 2025][Optimization][Discrete diffusion models] This paper proposes the Masked Diffusion Neural Sampler (MDNS), a framework grounded in stochastic optimal control theory for continuous-time Markov chains (CTMCs).…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Discrete diffusion models"
  - "neural sampler"
  - "stochastic optimal control"
  - "continuous-time Markov chain"
  - "Ising model"
  - "Potts model"
date: 2026-05-08
content_hash: 91057d2c95fbff9d
---

# MDNS: Masked Diffusion Neural Sampler via Stochastic Optimal Control

**Conference**: NeurIPS 2025
**arXiv**: [2508.10684](https://arxiv.org/abs/2508.10684)
**Authors**: Yuchen Zhu, Wei Guo, Jaemoo Choi (Georgia Tech), Guan-Horng Liu (FAIR at Meta), Yongxin Chen, Molei Tao (Georgia Tech)
**Code**: [github.com/yuchen-zhu-zyc/MDNS](https://github.com/yuchen-zhu-zyc/MDNS)
**Area**: Optimization
**Keywords**: Discrete diffusion models, neural sampler, stochastic optimal control, continuous-time Markov chain, Ising model, Potts model

## TL;DR

This paper proposes the Masked Diffusion Neural Sampler (MDNS), a framework grounded in stochastic optimal control theory for continuous-time Markov chains (CTMCs). By aligning path measures, MDNS trains a discrete neural sampler capable of accurately sampling from Ising/Potts models with state spaces as large as $10^{122}$, substantially outperforming existing learning-based baselines.

## Background & Motivation

### State of the Field
Sampling from an unnormalized target distribution $\pi(x) = \frac{1}{Z} e^{-U(x)}$ is a fundamental problem in statistical physics, Bayesian inference, and combinatorial optimization. Classical MCMC methods (Langevin MC, Metropolis-Hastings, Glauber dynamics) suffer from slow convergence on high-dimensional multimodal distributions. While diffusion-based neural samplers in continuous spaces (e.g., DIS, NETS) have achieved notable progress, diffusion-based sampling methods for **discrete state spaces** remain largely underexplored.

### Limitations of Prior Work
- **LEAPS** [HAJ25]: Relaxes CTMC trajectories into continuous probability vectors via the Gumbel softmax trick to maintain differentiability, but introduces biased gradient estimates and numerical instability, failing to converge to the correct target distribution even in low dimensions.
- **Classical MCMC**: Mixing times grow exponentially for large state spaces and low-temperature (multimodal) distributions.
- Discrete diffusion models have been primarily developed for generative modeling (text, proteins) and have rarely been applied to sampling from distributions with known energy functions.

### Root Cause
The key challenge lies in designing a training framework based on stochastic optimal control that overcomes the optimization difficulties arising from the discontinuous nature of CTMC trajectories, enabling efficient and accurate sampling from high-dimensional discrete distributions.

## Method

### Problem Formulation
- **Objective**: Given a potential function $U$, sample from $\pi(x) = \frac{1}{Z} e^{-U(x)}$ over a discrete state space $\mathcal{X}_0 = \{1,\ldots,N\}^D$.
- **Approach**: Learn the generator $Q^u$ of a CTMC that drives an initial distribution $p_{\text{init}}$ to $\pi$ at terminal time $T$.
- The generator is learned by matching the controlled path measure $\mathbb{P}^u$ to the optimal path measure $\mathbb{P}^*$, which is equivalent to minimizing the KL divergence $\text{KL}(\mathbb{P}^u \| \mathbb{P}^*)$.

### Stochastic Optimal Control Formulation
The sampling problem is formalized as a stochastic optimal control (SOC) problem over CTMCs:

$$\min_u \mathbb{E}_{X \sim \mathbb{P}^u} \left[ \int_0^T \sum_{y \neq X_t} \left( Q_t^u \log \frac{Q_t^u}{Q_t^0} - Q_t^u + Q_t^0 \right)(X_t, y) \, dt - r(X_T) \right]$$

where $r = -U - \log p_{\text{base}}$. The optimal generator takes the form of a multiplicative perturbation of the reference generator: $Q_t^*(x,y) = Q_t^0(x,y) \exp(V_t(y) - V_t(x))$, where $V_t$ is the value function.

### Masked Diffusion Reference Process
The reference path measure $\mathbb{P}^0$ is chosen as the generative process of masked discrete diffusion:
- Initial distribution: fully masked sequence $p_{\text{mask}}$
- Terminal distribution: uniform distribution $p_{\text{unif}}$
- Reference generator: $Q_t^0(x, x^{d \leftarrow n}) = \frac{\gamma(t)}{N} \mathbf{1}_{x^d = \mathbf{M}}$

**Key Property** (Lemma 2): This reference path measure is memoryless, guaranteeing existence and uniqueness of the SOC solution.

**Optimal Generator Structure** (Lemma 3): $Q_t^*(x, x^{d \leftarrow n}) = \gamma(t) \Pr_{X \sim \pi}(X^d = n | X^{\text{UM}} = x^{\text{UM}}) \mathbf{1}_{x^d = \mathbf{M}}$, meaning the score network $s_\theta$ only needs to predict the conditional marginal distribution of $\pi$ given the unmasked positions.

### Four Learning Objectives
Since CTMC trajectories are pure jump processes, the objective function is not differentiable with respect to parameters $\theta$. The paper proposes four learning objectives that bypass differentiability requirements:

1. **RERF (Relative-entropy with REINFORCE)**: Applies the REINFORCE trick to obtain an unbiased gradient estimator of the KL divergence.
   $$\mathcal{F}_{\text{RERF}} = \mathbb{E}_{X \sim \mathbb{P}^{\bar{u}}} W^{\bar{u}}(X) W^u(X)$$

2. **LV (Log-variance)**: Minimizes the variance of the log Radon–Nikodym derivative.
   $$\mathcal{F}_{\text{LV}} = \text{Var}_{X \sim \mathbb{P}^{\bar{u}}} W^u(X)$$

3. **CE (Cross-entropy)**: Reverse KL divergence; convex in $\mathbb{P}^u$ and yields a well-behaved optimization landscape.
   $$\mathcal{F}_{\text{CE}} = \mathbb{E}_{X \sim \mathbb{P}^{\bar{u}}} \frac{1}{Z} e^{W^{\bar{u}}(X)} W^u(X)$$

4. **WDCE (Weighted Denoising Cross-entropy)**: The core innovation — uses the terminal sample $X_T$ of a trajectory as an importance-weighted sample, re-masks it, and computes a denoising cross-entropy loss, avoiding backpropagation through the full trajectory.
   $$\mathcal{F}_{\text{WDCE}} = \mathbb{E}_{X \sim \mathbb{P}^{\bar{u}}} \left[ \frac{e^{W^{\bar{u}}(X)}}{Z} \mathbb{E}_\lambda \left[ w(\lambda) \mathbb{E}_{\mu_\lambda(\tilde{x}|X_T)} \sum_{d: \tilde{x}^d = \mathbf{M}} -\log s_\theta(\tilde{x})_{d, X_T^d} \right] \right]$$

The advantage of WDCE is that all outputs of each score model call are utilized (rather than a single element), and the $O(D)$ computational cost of the Radon–Nikodym derivative is further amortized via a replay buffer and $R$ re-sampling steps.

### Theoretical Guarantees
- **Sampling Guarantee** (Proposition 1): The KL divergence between path measures directly upper-bounds the KL divergence of the sampling distribution.
- **Normalizing Constant Estimation** (Proposition 2): $\hat{Z} = e^{W^u(X)}$ is an unbiased estimator of $Z$; training to $\text{KL} \leq \varepsilon^2/2$ guarantees $|\hat{Z}/Z - 1| \leq \varepsilon$ with probability at least $3/4$.

## Key Experimental Results

### Experiment 1: 4×4 Ising Model — Learning Objective Comparison

$J=1, h=0.1, \beta_{\text{high}}=0.28$; state space $|\mathcal{X}_0| = 2^{16} \approx 65536$. All objectives trained for 1000 steps with batch size 256.

| Objective | ESS ↑ | TV ↓ | KL ↓ | $\chi^2$ ↓ | $\widehat{\text{KL}}(\mathbb{P}^u\|\mathbb{P}^*)$ ↓ | $|\log\hat{Z}|$ error ↓ |
|---------|-------|------|------|-------|---------|---------|
| $\mathcal{F}_{\text{RERF}}$ | 0.9621 | 0.0799 | 0.0380 | 0.0845 | 0.0188 | **3e-5** |
| $\mathcal{F}_{\text{LV}}$ | **0.9713** | **0.0748** | **0.0348** | **0.0714** | **0.0141** | 4.6e-4 |
| $\mathcal{F}_{\text{CE}}$ | 0.9513 | 0.0833 | 0.0393 | 0.0903 | 0.0248 | 9.9e-4 |
| $\mathcal{F}_{\text{WDCE}}$ | 0.9644 | 0.0799 | 0.0382 | 0.0868 | 0.0177 | 3.0e-4 |
| Baseline (MH) | / | 0.0667 | 0.0325 | 0.0628 | / | / |

Across all metrics except $\log\hat{Z}$, the ranking is LV > WDCE > RERF > CE. All four objectives learn samplers that closely approximate the true distribution.

### Experiment 2: 16×16 Ising/Potts Models — High-Dimensional Scaling

State space: Ising $2^{256}$ ($\approx 10^{77}$), Potts ($q=3$) $3^{256}$ ($\approx 10^{122}$). WDCE is used for training.

**Ising Model** ($J=1, h=0$):

| Temperature | Method | Magnetization Error ↓ | 2-Point Correlation Error ↓ | ESS ↑ |
|------|------|-------------|------------|-------|
| $\beta_{\text{low}}=0.6$ | **MDNS** | **9.9e-3** | 2.4e-3 | **0.981** |
| | LEAPS | 2.4e-2 | 5.8e-1 | 0.261 |
| | MH | 1.9e-2 | **7.7e-4** | / |
| $\beta_{\text{critical}}=0.4407$ | **MDNS** | **3.7e-3** | **2.0e-3** | **0.933** |
| | LEAPS | 7.4e-3 | 1.6e-1 | 0.384 |
| | MH | 4.6e-3 | 2.5e-3 | / |
| $\beta_{\text{high}}=0.28$ | **MDNS** | 8.5e-3 | **1.0e-3** | 0.962 |
| | LEAPS | 7.4e-3 | 1.6e-3 | **0.987** |
| | MH | **6.1e-3** | 1.1e-3 | / |

**Potts Model** ($q=3, J=1$):

| Temperature | Method | Magnetization Error ↓ | 2-Point Correlation Error ↓ | ESS ↑ |
|------|------|-------------|------------|-------|
| $\beta_{\text{low}}=1.2$ | **MDNS** | **1.3e-3** | **8.8e-5** | **0.933** |
| | LEAPS | 2.9e-1 | 2.5e-1 | 0.012 |
| | MH | 7.4e-1 | 5.6e-1 | / |
| $\beta_{\text{critical}}=1.005$ | **MDNS** | **4.3e-3** | **2.9e-3** | **0.875** |
| | LEAPS | 2.7e-1 | 2.0e-1 | 0.004 |
| | MH | 5.2e-1 | 3.5e-1 | / |
| $\beta_{\text{high}}=0.5$ | **MDNS** | **2.2e-3** | **5.8e-4** | 0.983 |
| | LEAPS | 2.9e-3 | 1.2e-3 | **0.991** |
| | MH | 3.5e-2 | 1.6e-2 | / |

At low and critical temperatures, MDNS shows a dramatic advantage over LEAPS (ESS 0.933 vs. 0.012). MH fails to mix on the Potts model even after 20+ hours of continuous runtime.

## Highlights & Insights

- **Theory–Algorithm Unification**: Discrete sampling is rigorously formulated as a CTMC stochastic optimal control problem. The structure of the optimal generator naturally corresponds to the score function of masked diffusion, yielding an elegant theoretical framework.
- **Four Unbiased/Low-Variance Learning Objectives**: The approach entirely avoids the biased gradient problem of Gumbel softmax. WDCE enables high-dimensional scalable training through importance weighting and re-sampling.
- **Extreme High-Dimensional Validation**: Successful sampling on a Potts model with state space cardinality $10^{122}$, achieving ESS of 0.93 compared to 0.01 for LEAPS and complete failure of MH.
- **Warm-up Strategy**: Pre-training at high temperatures (simple distributions) before transferring to low temperatures (multimodal distributions) effectively assists the model in locating modes.
- **Normalizing Constant Estimation**: Provides an unbiased estimator of $Z$ as a byproduct, with rigorous probabilistic guarantees.

## Limitations & Future Work

- **Evaluation limited to statistical physics models**: Ising and Potts models have regular lattice structures; performance on graph-structured distributions or combinatorial optimization problems remains unknown.
- **Warm-up strategy lacks systematic study**: Warm-up temperatures and step counts are set manually in current experiments without automation.
- **Computational overhead**: Although WDCE is more efficient than CE/LV, generating full CTMC trajectories to obtain importance weights is still required.
- **Integration with pretrained models**: The paper suggests MDNS could fine-tune pretrained discrete diffusion models given a reward function, but this has not been empirically validated.
- **Incomplete theoretical analysis**: The conjecture that the masked diffusion interpolation path is superior to geometric annealing paths remains speculative.
- **Limited to finite discrete spaces**: The framework is restricted to $\{1,\ldots,N\}^D$ and is not applicable to continuous or countably infinite state spaces.

## Related Work & Insights

- **LEAPS [HAJ25]**: Uses geometric annealing $\pi_\eta \propto e^{-\eta U}$ and Gumbel softmax relaxation, resulting in biased gradients and failure at low temperatures or large state spaces. MDNS uses masked diffusion interpolation with unbiased objectives, comprehensively outperforming LEAPS.
- **DIS/PIS [ZC22, VGD23]**: SDE-based neural samplers in continuous spaces. MDNS is the discrete-space counterpart, sharing the core idea of path measure matching but requiring entirely new technical tools to handle discrete jump processes.
- **NETS [AVE25]**: A non-equilibrium transport sampler in continuous space based on SOC theory. MDNS extends this framework to discrete CTMCs.
- **Masked Diffusion Models [LME24, Ou+25]**: Originally developed for generative modeling (learning from data). MDNS repurposes them in reverse for sampling from distributions with known energy functions.
- **MCMC Methods**: MH fails to mix on the Potts model after 20 hours, whereas MDNS achieves accurate sampling after 100k training steps.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First work to apply the masked diffusion + SOC framework to discrete distribution sampling; the four learning objectives are systematically and comprehensively designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-temperature, multi-scale experiments on Ising/Potts models are thorough and ablation studies are comprehensive, but validation beyond physical models is absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are rigorous; the connection from SOC to masked diffusion is presented clearly and naturally; algorithmic pseudocode is complete.
- **Value**: ⭐⭐⭐⭐⭐ — Establishes a powerful and scalable new paradigm for discrete-space sampling, demonstrating breakthrough performance on extremely high-dimensional problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Estimation of Stochastic Optimal Transport Maps](estimation_of_stochastic_optimal_transport_maps.md)
- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)
- [\[NeurIPS 2025\] Memory-Augmented Potential Field Theory: A Framework for Adaptive Control in Non-Convex Domains](memory-augmented_potential_field_theory_a_framework_for_adaptive_control_in_non-.md)
- [\[NeurIPS 2025\] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification](optimal_rates_for_generalization_of_gradient_descent_for_deep_relu_classificatio.md)
- [\[NeurIPS 2025\] VIKING: Deep Variational Inference with Stochastic Projections](viking_deep_variational_inference_with_stochastic_projections.md)

</div>

<!-- RELATED:END -->
