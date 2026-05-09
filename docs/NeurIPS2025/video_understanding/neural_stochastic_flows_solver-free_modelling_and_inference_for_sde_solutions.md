---
title: >-
  [Paper Note] Neural Stochastic Flows: Solver-Free Modelling and Inference for SDE Solutions
description: >-
  [NeurIPS 2025][Video Understanding][Stochastic Differential Equations] This paper proposes Neural Stochastic Flows (NSF), which directly learns the transition distribution $p(x_t \mid x_s)$ of an SDE via conditional normalising flows. The architecture is constrained to satisfy stochastic flow properties (identity, Markov, Chapman-Kolmogorov), enabling single-step sampling without numerical solvers and achieving up to two orders of magnitude speedup at distant time points.
tags:
  - NeurIPS 2025
  - Video Understanding
  - Stochastic Differential Equations
  - Normalising Flow
  - Solver-Free
  - Transition Distribution
  - State-Space Model
date: 2026-05-08
content_hash: d0caa7cb28d7d1d8
---

# Neural Stochastic Flows: Solver-Free Modelling and Inference for SDE Solutions

**Conference**: NeurIPS 2025
**arXiv**: [2510.25769](https://arxiv.org/abs/2510.25769)
**Code**: [Project Page](https://nkiyohara.github.io/nsf-neurips2025/)
**Area**: Sequence Modelling / Stochastic Differential Equations
**Keywords**: Stochastic Differential Equations, Normalising Flow, Solver-Free, Transition Distribution, State-Space Model

## TL;DR

This paper proposes Neural Stochastic Flows (NSF), which directly learns the transition distribution $p(x_t \mid x_s)$ of an SDE via conditional normalising flows. The architecture is constrained to satisfy stochastic flow properties (identity, Markov, Chapman-Kolmogorov), enabling single-step sampling without numerical solvers and achieving up to two orders of magnitude speedup at distant time points.

## Background & Motivation

**Background**: Stochastic differential equations (SDEs) are widely used in finance, physics, and machine learning for modelling noisy time series. Conventional Neural SDE methods rely on numerical solvers to simulate trajectories step by step, with computational cost proportional to the time interval.

**Limitations of Prior Work**:
- **High solver overhead**: Neural SDE training and inference both depend on step-by-step simulation, making long-horizon prediction computationally expensive.
- **Neural Flow limitations**: Existing solver-free methods (Neural Flow for ODEs) cannot express stochastic dynamics.
- **Non-generality of diffusion-based methods**: Acceleration methods such as Consistency Models apply only to diffusion processes with specific boundary conditions and cannot handle general SDEs.

**Key Challenge**: Transition distributions of general Itô SDEs are needed for uncertainty quantification and probabilistic forecasting, yet numerical solvers are prohibitively costly at distant time points, while existing solver-free methods either do not support stochastic dynamics or lack generality.

**Goal**: Design a neural network architecture that directly parameterises the weak solution (transition distribution) of an SDE, satisfying the key properties of stochastic flows, such that neither training nor inference requires a solver.

**Key Insight**: Beginning from the theory of stochastic flow diffeomorphisms, the conditions for strong solutions are reformulated as conditions on weak solutions (probability distributions), which are then parameterised using conditional normalising flows.

**Core Idea**: Architectural constraints enforce the identity and Markov properties, while a bidirectional KL regularisation loss encourages the Chapman-Kolmogorov flow property, thereby directly learning the transition distribution of the SDE.

## Method

### Overall Architecture

NSF consists of two components:
1. A parameterised Gaussian initialisation conditioned on $(x_{t_i}, \Delta t, t_i)$
2. A sequence of bijective affine coupling layers

Sampling procedure: $\varepsilon \sim \mathcal{N}(0, I) \rightarrow z \rightarrow x_{t_j} = f_\theta(z, c)$

### Key Designs

#### 1. Conditional Normalising Flow Architecture

**Base Gaussian distribution** (analogous to Euler-Maruyama discretisation):

$$z = \underbrace{x_{t_i} + \Delta t \cdot \text{MLP}_\mu(c; \theta_\mu)}_{\mu(c)} + \underbrace{\sqrt{\Delta t} \cdot \text{MLP}_\sigma(c; \theta_\sigma)}_{\sigma(c)} \odot \varepsilon$$

The drift term scales with $\Delta t$ and the diffusion term with $\sqrt{\Delta t}$, consistent with the physical intuition of SDEs.

**Affine coupling layers**: Each layer partitions the state into $(z_A, z_B)$ and applies an affine transformation to $z_B$:

$$f_i(z; c, \theta_i) = \text{Concat}(z_A, z_B \odot \exp(\Delta t \cdot \text{MLP}_{\text{scale}}^{(i)}) + \Delta t \cdot \text{MLP}_{\text{shift}}^{(i)})$$

Crucially, all transformation parameters are multiplied by $\Delta t$, ensuring the mapping reduces to the identity when $\Delta t = 0$ (**identity property**).

#### 2. Flow Property Regularisation Loss

To encourage the Chapman-Kolmogorov property $p(x_{t_k} \mid x_{t_i}) = \int p(x_{t_k} \mid x_{t_j}) p(x_{t_j} \mid x_{t_i}) dx_{t_j}$, a variational upper bound on the bidirectional KL divergence is derived:

- **Forward KL** $\mathcal{L}_{\text{flow}, 1\text{-to-}2}$: uses a bridge distribution $b_\xi(x_{t_j} \mid x_{t_i}, x_{t_k})$ as an auxiliary variational distribution.
- **Reverse KL** $\mathcal{L}_{\text{flow}, 2\text{-to-}1}$: matches in the opposite direction.

Total loss:

$$\mathcal{L}(\theta, \xi) = -\mathbb{E}[\log p_\theta(x_{t_j} \mid x_{t_i})] + \lambda \mathcal{L}_{\text{flow}}(\theta, \xi)$$

#### 3. Latent NSF (Handling Partial Observations)

For noisy or partially observed data, a variational state-space model framework is introduced:
- **Generative model**: $p(x_{t_0:T}, o_{t_0:T}) = p(x_{t_0}) \prod_i p_\theta(x_{t_i} \mid x_{t_{i-1}}) p_\psi(o_{t_i} \mid x_{t_i})$
- **Variational posterior**: GRU encoder $q_\phi(x_{t_i} \mid o_{\leq t_i})$
- **Skip-step KL loss** $\mathcal{L}_{\text{skip}}$: exploits NSF's ability to sample across arbitrary time gaps without recursive transitions.

Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\beta\text{-NELBO}} + \lambda \mathcal{L}_{\text{flow}} + \beta_{\text{skip}} \mathcal{L}_{\text{skip}}$

### Loss & Training

- The bridge distribution $b_\xi$ is optimised separately via $K$ inner-loop steps.
- Main model parameters and bridge parameters are updated alternately.
- Time triplets $(t_i, t_j, t_k)$ are sampled from the data distribution.

## Key Experimental Results

### Main Results: Stochastic Lorenz Attractor

| Method | KL ($t=0.25$) | KL ($t=1.0$) | kFLOPs ($t=1.0$) |
|--------|--------------|-------------|------------------|
| Latent SDE | 2.1±0.9 | 1.5±0.5 | 3,760 |
| Neural LSDE | 1.3±0.4 | 53.1±29.3 | 6,699 |
| SDE matching (Δt=0.0001) | 4.3±0.7 | 3.8±1.0 | 737,354 |
| **NSF (H_pred=1.0)** | **0.8±0.7** | **0.2±0.6** | **53** |

NSF single-step sampling requires only **53 kFLOPs**, approximately **~14,000×** fewer than SDE matching. Runtime: NSF 0.3 ms vs. Latent SDE 124–148 ms/batch.

### Main Results: CMU Motion Capture

| Method | Setup 1 MSE | Setup 2 MSE |
|--------|-------------|-------------|
| Latent ODE | 5.98±0.28 | 31.62±0.05 |
| Latent SDE | 12.91±2.90 | 9.52±0.21 |
| SDE matching | 5.20±0.43 | 4.26±0.35 |
| **Latent NSF** | **8.62±0.32** | — |

Runtime: Latent NSF 3.5 ms vs. Latent SDE 75 ms/batch (~21× speedup).

### Ablation Study

- Effect of different $H_{\text{pred}}$: $H_{\text{pred}}=1.0$ (single step) achieves the lowest KL at long horizons; $H_{\text{pred}}=0.25$ (requires recursion) is more accurate at short horizons but incurs higher FLOPs at long horizons.
- Effect of flow loss $\mathcal{L}_{\text{flow}}$: substantially improves distributional consistency.

### Key Findings

1. NSF surpasses all solver-based baselines in distributional accuracy on the stochastic Lorenz attractor while reducing FLOPs by one to two orders of magnitude.
2. Single-step sampling yields the greatest advantage at long prediction horizons — precisely the regime where numerical solvers are most expensive.
3. The flow loss ensures distributional consistency between single-step and multi-step transitions.

## Highlights & Insights

1. **Theoretical elegance**: The derivation from stochastic flow diffeomorphism theory to weak solution conditions is rigorous and natural.
2. **Architectural ingenuity**: Multiplying all transformation parameters by $\Delta t$ enforces the identity property; omitting $t_i$ for autonomous SDEs enforces stationarity.
3. **Generality**: The method handles general Itô SDEs without restriction to the specific boundary conditions of diffusion models, filling an important gap in the literature.
4. **Bidirectional KL + bridge distribution**: Elegantly resolves the intractability of the marginalisation in the Chapman-Kolmogorov equation.
5. **The speedup is greatest precisely where it is most needed** — at distant time-point predictions.

## Limitations & Future Work

1. **Suboptimal performance on CMU dataset**: Latent NSF MSE (8.62) falls short of SDE matching (5.20) and NCDSSM (5.69), likely bottlenecked by variational inference.
2. **Training complexity**: Inner-loop optimisation of the bridge distribution and time-triplet sampling increase training overhead.
3. **Expressiveness ceiling of conditional normalising flows**: May be insufficiently flexible for high-dimensional or multi-modal transition distributions.
4. **High-dimensional systems not validated**: Experiments are conducted in relatively low-dimensional settings (Lorenz 3D, Motion Capture 50D); further validation on higher-dimensional systems is needed.

## Related Work & Insights

- **Neural Flow (Biloš et al.)**: Solver-free method for ODEs; this work extends the framework to SDEs.
- **Consistency Models**: Diffusion model acceleration, but restricted to specific boundary conditions.
- **Latent SDE (Li et al.)**: Solver-based baseline; NSF achieves comparable accuracy with substantially lower cost.
- **SDE Matching (Bartosh et al.)**: Solver-free training but solver-dependent inference; NSF is solver-free at both training and inference time.
- **Variational state-space models**: Foundational framework for Latent NSF.

## Rating

⭐⭐⭐⭐

The theoretical framework is complete and the method is elegant. Results on synthetic data are excellent, though the performance advantage on real-world data (CMU) is less pronounced. Overall, this work represents a significant contribution to the SDE modelling literature.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Revisiting Bi-Linear State Transitions in Recurrent Neural Networks](revisiting_bi-linear_state_transitions_in_recurrent_neural_networks.md)
- [\[ICCV 2025\] AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning](../../ICCV2025/video_understanding/aim_adaptive_inference_of_multi_modal_llms_via_token_merging_and_pruning.md)
- [\[ICLR 2026\] FuncBenchGen: A Contamination-Free Controllable Evaluation Framework for Reliable Benchmarking](../../ICLR2026/video_understanding/towards_reliable_benchmarking_a_contamination_free_controllable_evaluation_frame.md)
- [\[ACL 2026\] VC-Inspector: Advancing Reference-free Evaluation of Video Captions with Factual Analysis](../../ACL2026/video_understanding/vc-inspector_advancing_reference-free_evaluation_of_video_captions_with_factual_.md)
- [\[CVPR 2026\] VecAttention: Vector-wise Sparse Attention for Accelerating Long Context Inference](../../CVPR2026/video_understanding/vecattention_vector-wise_sparse_attention_for_accelerating_long_context_inferenc.md)

</div>

<!-- RELATED:END -->
