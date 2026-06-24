---
title: >-
  [Paper Note] Multi-Trajectory Physics-Informed Neural Networks for HJB Equations with Hard-Zero Terminal Inventory: Optimal Execution on Synthetic & SPY Data
description: >-
  [NeurIPS 2025 (Workshop on Generative AI in Finance)][Physics & Scientific Computing][Physics-Informed Neural Networks] To address the hard-zero terminal inventory constraint ($X_T=0$) in HJB equations arising from optimal trade execution, this paper proposes Multi-Trajectory PINN (MT-PINN). Through a rollout-based terminal loss and a $\lambda$-curriculum training strategy, MT-PINN significantly outperforms vanilla PINN on both synthetic benchmarks and live SPY backtesting…
tags:
  - "NeurIPS 2025 (Workshop on Generative AI in Finance)"
  - "Physics & Scientific Computing"
  - "Physics-Informed Neural Networks"
  - "Hamilton-Jacobi-Bellman"
  - "Optimal Execution"
  - "Terminal Inventory Constraint"
  - "Quantitative Finance"
date: 2026-05-08
content_hash: 65cfe70c68e82b09
---

# Multi-Trajectory Physics-Informed Neural Networks for HJB Equations with Hard-Zero Terminal Inventory: Optimal Execution on Synthetic & SPY Data

**Conference**: NeurIPS 2025 (Workshop on Generative AI in Finance)
**arXiv**: [2512.12708](https://arxiv.org/abs/2512.12708)  
**Code**: [GitHub](https://github.com/anthimevalin/Multi-Trajectory-PINNs-Zero-Terminal-HJB)  
**Area**: Scientific Computing
**Keywords**: Physics-Informed Neural Networks, Hamilton-Jacobi-Bellman, Optimal Execution, Terminal Inventory Constraint, Quantitative Finance

## TL;DR
To address the hard-zero terminal inventory constraint ($X_T=0$) in HJB equations arising from optimal trade execution, this paper proposes Multi-Trajectory PINN (MT-PINN). Through a rollout-based terminal loss and a $\lambda$-curriculum training strategy, MT-PINN significantly outperforms vanilla PINN on both synthetic benchmarks and live SPY backtesting, achieving a substantial reduction in terminal inventory violation rates.

## Background & Motivation
**Background**: Optimal trade execution is a core problem in quantitative finance, requiring full liquidation ($X_T=0$) within a given time horizon while accounting for market impact and risk aversion. The corresponding optimal control problem is characterized by a Hamilton-Jacobi-Bellman (HJB) PDE.

**Limitations of Prior Work**: Vanilla PINNs are trained via PDE residuals combined with soft boundary penalties, but their enforcement of the hard-zero terminal inventory constraint is insufficient — particularly as $\tau \to 0$ where the value function becomes non-smooth — frequently resulting in nonzero terminal inventory and unstable control policies.

**Key Challenge**: The terminal condition is $\Gamma(0,X) = 0$ if $X=0$, $+\infty$ otherwise — a singular boundary condition that PDE residual losses cannot adequately capture.

**Goal**: To strongly enforce the hard-zero terminal inventory constraint within the PINN framework while maintaining training stability.

**Key Insight**: Rather than training solely at the PDE residual level, the paper adopts a "control-trajectory" perspective, directly simulating execution trajectories and penalizing terminal inventory deviations.

**Core Idea**: Employ a rollout-based trajectory loss with BPTT to propagate the terminal penalty, combined with a $\lambda$-curriculum that gradually transitions from risk-neutral to risk-averse regimes.

## Method

### Overall Architecture
MT-PINN is built upon the Gatheral-Schied optimal execution model, where price $S_t$ follows GBM and inventory $X_t$ evolves at trading rate $v_t$ ($\dot{X}_t = -v_t$). The value function $\Gamma(\tau, X, S)$ satisfies a reduced HJB equation (independent of $S$ under risk neutrality), and the optimal control is $v^* = \frac{1}{2} \partial\Gamma/\partial X$. MT-PINN approximates the value function with an MLP, computes HJB residuals and optimal controls via automatic differentiation, and additionally incorporates a trajectory rollout loss.

### Key Designs
1. **Multi-Trajectory Terminal-Inventory Loss**: Starting from a batch of initial states $\{(X_0^{(p)}, S_0^{(p)})\}_{p=1}^P$ across multiple time horizons $\{T_j\}_{j=1}^J$, trajectories are unrolled using forward Euler discretization (200 steps) to obtain terminal inventories $x_{T_j}^{(p)}$. A hybrid penalty function $\psi(x_T) = |x_T|$ (when $|x_T| \le 1$) or $x_T^2$ (when $|x_T| > 1$) is averaged to form the trajectory loss. Gradients are propagated back to network parameters via BPTT.
2. **$\lambda$-Curriculum Training Strategy**: The risk-aversion parameter $\lambda$ is gradually increased from 0 to the target value $\lambda^*$: training begins at $\lambda=0$ (1D state space) and warm-starts to $\lambda > 0$ (2D state space). Five stages with $\alpha \in (0.25, 0.50, 0.75, 0.9, 1.0)$ are used, each running for 5k epochs. This avoids the instability of training directly under high dimensionality and strong constraints.
3. **Symmetry and Interior Condition Losses**: The symmetry properties of the value function ($\Gamma(\tau,X) = \Gamma(\tau,-X)$ or $\Gamma(\tau,X,S) = \Gamma(\tau,-X,-S)$) and interior conditions ($\Gamma(\tau,0,S) \le 0$) are incorporated as additional constraint terms.
4. **DWA-style Adaptive Weights**: Loss term weights are dynamically adjusted after EMA smoothing to prevent any single term from dominating training.

### Loss & Training
Total loss:
$$\mathcal{L}_{\text{total}} = w_{\text{PDE}}\mathcal{L}_{\text{PDE}} + w_{\text{traj}}\mathcal{L}_{\text{traj}} + w_{\text{IC}}\mathcal{L}_{\text{IC}} + w_{\text{sym}}\mathcal{L}_{\text{sym}} + \mathbf{1}_{\{\lambda>0\}} w_0 \mathcal{L}_{\text{0-term}}$$

- $\mathcal{L}_{\text{PDE}}$: squared HJB equation residual
- $\mathcal{L}_{\text{traj}}$: trajectory rollout terminal inventory penalty (core of MT-PINN)
- $\mathcal{L}_{\text{IC}}$: interior condition at zero inventory
- $\mathcal{L}_{\text{sym}}$: symmetry constraint
- $\mathcal{L}_{\text{0-term}}$: constraint $\Gamma(0,0,S)=0$ (only for $\lambda>0$)

Network architecture: MLP (32×32×32 for MT-PINN, 500×500×500 for baselines), tanh activation, implemented in JAX/FLAX. Optimizer: AdamW, learning rate $5 \times 10^{-4}$. No normalization layers or stochastic regularizers are used to ensure stable derivative computation.

Training proceeds in two phases:
- **Phase A** ($\lambda=0$, 1D): inputs $(\tau, X)$ only, 30k epochs
- **Phase B** ($\lambda>0$, 2D): warm-start to inputs $(\tau, X, S)$, 5 curriculum stages × 5k epochs
- Trajectory rollout parameters: $P=820$ initial states, $J=7$ time horizons $\{T/50, T/10, T/5, 2T/5, 3T/5, 4T/5, T\}$, $N_{\text{dt}}=200$ Euler steps
- DWA weight updates: every 1000 epochs, clip range $[0.1, 2.0]$, EMA $\beta=0.95$

## Key Experimental Results

### Main Results: Synthetic Benchmark (Gatheral-Schied Model)
Setup: $T=5.0$, $X \in [-10,10]$, $S \in [10,100]$, $\sigma=0.1$, $\kappa=0.1$, $N_{\text{PDE}}=30000$, $N_{\text{IC}}=5000$.

| Terminal Inventory Statistics ($\lambda=0.10$) | Mean $|X_T|$ ± Std | 95th pct | $p_\varepsilon$ ($\varepsilon=0.05$) |
|---|---|---|---|
| Vanilla PINN | 0.777 ± 0.444 | 1.407 | 0.055 |
| PINN + $\lambda$-curriculum | 0.164 ± 0.161 | 0.527 | 0.205 |
| **MT-PINN + $\lambda$-curriculum** | **0.073 ± 0.092** | **0.241** | **0.600** |

MT-PINN reduces mean terminal inventory to 1/10 of vanilla PINN, and the probability of satisfying $|X_T| \le 0.05$ improves from 5.5% to 60%.

### SPY Live Backtesting
Setup: 7 days of SPY intraday data (2025.2.10–2.19), three 2-hour windows per day at 5-second intervals, $n=21$ windows. The first and last 15 minutes of each session (high-volatility periods) are excluded. Intraday volatility $\sigma=0.0038$ (≈6% annualized), permanent impact $\kappa=0.2$, inventory normalized to $[-1,1]$, price range $S \in [590, 620]$.

| Model | Mean Exposure | Exposure Std | Mean Cost (bps) | Cost Std (bps) |
|---|---|---|---|---|
| TWAP | 0.334 | 0.0000 | -6.35 | 12.56 |
| MT-PINN $\lambda=0.00$ | 0.336 | 0.0000 | -6.37 | 12.58 |
| MT-PINN $\lambda=0.05$ | 0.231 | 0.0004 | -5.01 | 11.02 |
| MT-PINN $\lambda=0.10$ | 0.164 | 0.0002 | -3.69 | 9.67 |

### Key Findings
- At $\lambda=0$, MT-PINN nearly perfectly replicates TWAP (exposure 0.336 vs. 0.334), validating theoretical consistency
- At $\lambda > 0$, a clear risk-cost frontier emerges: exposure decreases (0.334 → 0.164) while cost increases but with reduced standard deviation
- In declining market windows, MT-PINN with $\lambda > 0$ front-loads execution and outperforms TWAP
- MT-PINN with only 32×32×32 width surpasses vanilla PINN with 500×500×500 width, demonstrating that the trajectory loss is the critical factor
- MT-PINN consistently outperforms baselines across all three settings $\lambda \in \{0, 0.05, 0.10\}$ (see Table 3 in the paper)
- Extremely lightweight compute: 1× TPU v6e-1, typical runtime 1–4 minutes

## Highlights & Insights
- **Trajectory-based constraint enforcement**: Rather than relying on PDE residuals to indirectly satisfy constraints, the method directly unrolls trajectories and applies BPTT to penalize terminal inventory — a concise and effective approach
- **Curriculum regularization**: Warm-starting from low-dimensional/simple problems to high-dimensional/complex ones is a practical technique for stabilizing PINN training
- **Small network outperforms large network**: MT-PINN with width-32 MLP outperforms the width-500 baseline, demonstrating that inductive bias (trajectory loss) matters more than model capacity
- **Theory-practice consistency**: Alignment with TWAP at $\lambda=0$ and with the analytical solution establishes credibility

## Limitations & Future Work
- Only single-asset, linear permanent impact models are considered; multi-asset settings, nonlinear impact, and order book dynamics are not addressed
- Transaction costs and overnight risk are not modeled
- Computational complexity of the trajectory loss scales as $\mathcal{O}(J \times N_{\text{dt}} \times P)$, limiting scalability
- SPY data spans only 7 days and 21 windows, limiting statistical significance
- Market microstructure effects (order book depth, bid-ask spread, order flow imbalance) are not modeled

## Related Work & Insights
- **vs. Vanilla PINN**: The core advantage of MT-PINN lies in the trajectory loss directly constraining terminal inventory, as opposed to soft penalization
- **vs. TWAP**: MT-PINN is equivalent to TWAP at $\lambda=0$ and clearly outperforms TWAP in declining market conditions at $\lambda > 0$
- **vs. Krishnapriyan et al.**: The curriculum learning idea is adapted for PINN training stabilization
- **Insights**: For PDE control problems with hard constraints, "trajectory rollout + BPTT" constitutes a general paradigm transferable to other financial and control settings

## Rating
- **Novelty**: ⭐⭐⭐⭐ Trajectory loss + BPTT for terminal constraint enforcement is a novel combination in the PINN literature
- **Experimental Thoroughness**: ⭐⭐⭐ Synthetic and live experiments are solid but limited in scale; ablation study lacks systematic coverage
- **Writing Quality**: ⭐⭐⭐⭐ Workshop paper is clearly structured with complete mathematical derivations
- **Value**: ⭐⭐⭐⭐ Provides a reproducible and practical solution for constraint enforcement in PINN-based financial control problems

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Differentiable Stellar Atmospheres with Physics-Informed Neural Networks](../../ICML2025/physics/differentiable_stellar_atmospheres_with_physics-informed_neural_networks.md)
- [\[NeurIPS 2025\] Physics-Informed Neural Networks with Fourier Features and Attention-Driven Decoding](physics-informed_neural_networks_with_fourier_features_and_attention-driven_deco.md)
- [\[NeurIPS 2025\] Neuro-Spectral Architectures for Causal Physics-Informed Networks](neuro-spectral_architectures_for_causal_physics-informed_networks.md)
- [\[NeurIPS 2025\] Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints](physics-constrained_flow_matching_sampling_generative_models_with_hard_constrain.md)
- [\[NeurIPS 2025\] TITAN: A Trajectory-Informed Technique for Adaptive Parameter Freezing in Large-Scale VQE](titan_a_trajectory-informed_technique_for_adaptive_parameter_freezing_in_large-s.md)

</div>

<!-- RELATED:END -->
