---
title: >-
  [Paper Note] Empirical Stability Analysis of Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed Discovery
description: >-
  [ICLR 2026][Scientific Computing][KAN] This paper systematically evaluates vanilla KAN as a drop-in replacement for MLP in the residual branch of Hard-Constrained Recurrent Physics-Informed Neural Networks (HRPINN) — through 3 complementary studies × 100 random seeds, it finds that KAN is competitive on univariate separable residuals (Duffing's $-0.3x^3$), but systematically fails on multiplicatively coupled residuals (Van der Pol's $(1-x^2)v$) with extreme hyperparameter fragility, while standard MLP exhibits substantially superior stability across nearly all configurations.
tags:
  - ICLR 2026
  - Scientific Computing
  - KAN
  - physics-informed
  - oscillator
  - HRPINN
  - neural ODE
  - residual discovery
date: 2026-05-08
content_hash: 8a2bf8622b695ec2
---

# Empirical Stability Analysis of Kolmogorov-Arnold Networks in Hard-Constrained Recurrent Physics-Informed Discovery

**Conference**: ICLR 2026
**arXiv**: [2602.09988](https://arxiv.org/abs/2602.09988)
**Code**: Not open-sourced
**Area**: Scientific Computing / Physics-Informed Neural Networks
**Keywords**: KAN, physics-informed, oscillator, HRPINN, neural ODE, residual discovery

## TL;DR

This paper systematically evaluates vanilla KAN as a drop-in replacement for MLP in the residual branch of Hard-Constrained Recurrent Physics-Informed Neural Networks (HRPINN) — through 3 complementary studies × 100 random seeds, it finds that KAN is competitive on univariate separable residuals (Duffing's $-0.3x^3$), but systematically fails on multiplicatively coupled residuals (Van der Pol's $(1-x^2)v$) with extreme hyperparameter fragility, while standard MLP exhibits substantially superior stability across nearly all configurations.

## Background & Motivation

**Background**: Hard-Constrained Recurrent Physics-Informed Neural Networks (HRPINN) embed known physics into a recurrent integrator, leaving the neural network to learn only the residual dynamics — this ensures physical consistency and has been validated in cyber-physical systems. Meanwhile, Kolmogorov-Arnold Networks (KAN), grounded in the Kolmogorov-Arnold representation theorem, decompose multivariate functions into sums of univariate functions $\Phi(\mathbf{x}) = \sum_q \phi_q(\sum_p \psi_{q,p}(x_p))$, replacing MLP's fixed activation functions with learnable B-splines, and have shown promise in scientific ML.

**Limitations of Prior Work**:
- KAN has demonstrated symbolic discovery potential in Neural ODE and gray-box settings (KAN-ODEs, SKANODEs), but these works operate in unconstrained continuous ODE formulations.
- No prior work has tested KAN within hard-constrained recurrent architectures — recurrent settings accumulate errors over time, imposing stricter stability requirements.
- KAN's additive inductive bias ($\phi(x) + \phi(v)$) is theoretically well-suited for separable physical laws, but whether this holds in practice remains an open question.

**Key Challenge**: KAN's additive structure is naturally aligned with additively separable functions, yet many physical laws contain multiplicatively coupled terms (e.g., Van der Pol's $(1-x^2)v$). While KAN can theoretically represent multiplication through deep composition ($xy = \frac{1}{4}((x+y)^2 - (x-y)^2)$), this requires deeper layers — and whether deep KAN remains stable under recurrent error accumulation is unknown.

**Goal**: To establish a baseline empirical evaluation of vanilla KAN within hard-constrained recurrent physics-informed architectures.

**Key Insight**: Two classical oscillators with contrasting residual structures are selected — Duffing (univariate polynomial) and Van der Pol (multiplicative coupling) — as boundary test cases for additive separability.

**Core Idea**: Through carefully controlled experimental design, reveal the practical success boundary of KAN's additive inductive bias under recurrent physical constraints.

## Method

### Overall Architecture: HRPINN + KAN/MLP Residual Branch Comparison

In the HRPINN framework, the residual branch $R_\theta(x, v)$ receives normalized states $[x, v]$ and is implemented separately as a standard ReLU MLP and a B-spline KAN. Known physics and the integrator are fixed within the recurrent update rule; the network learns only the residual manifold. Performance is evaluated by test MSE and Discovery $R^2$ (grid-density-dependent), computed over a $100 \times 100$ grid on phase space $x, v \in [-2.5, 2.5]$. A unified candidate fitting procedure (not KAN-specific symbolic pruning) is used to ensure fair comparison.

### Key Design 1: Configuration Ablation — Systematic Hyperparameter Sensitivity Evaluation

Seven KAN grid configurations (varying grid size $G$ and spline order $k$) are evaluated with fixed training settings and 100 random seeds per configuration. Key findings:

- The coarse-grid configuration (Config F, $G=3, k=3$) achieves $R^2 = 0.862$ on Duffing, narrowing the gap with MLP ($0.957$).
- Most configurations produce negative $R^2$ on Van der Pol (divergent solutions), e.g., Config C yields $R^2 = -5.229 \pm 5.091$.
- MLP (337 parameters) consistently achieves Duffing $R^2 = 0.957$ and VdP $R^2 = 0.768$ with minimal variance.

**Design Motivation**: By exhaustively sweeping the configuration space with large-scale seed statistics, the study distinguishes between "a particular KAN configuration happening to perform well" and "KAN as an architecture being robustly superior" — the evidence supports the former.

### Key Design 2: Parameter Scale Ablation × Two Training Paradigms

With fixed configuration and varying parameter counts (Very Small 120 → Deep 880), models are evaluated under both single-step Teacher Forcing and BPTT:

| Training Paradigm | KAN Behavior | MLP Behavior |
|---|---|---|
| Teacher Forcing | Small KAN competitive on Duffing; VdP rapidly degrades with scale | Scales smoothly |
| BPTT | Smallest KAN achieves best VdP $R^2 \approx 0.74$ (long-horizon supervision helps); deep KAN unstable | Stably superior at all scales |

This contrast reveals a key bottleneck: MLP's dense matrix multiplication achieves variable interaction in the first layer ($w_i x + w_j v$), whereas KAN's additive bias ($\phi(x) + \phi(v)$) requires deep composition to approximate multiplication — and such deep composition is unstable under recurrent error accumulation.

### Key Design 3: Qualitative Validation — Residual Manifold Visualization

Learned residual surfaces from KAN and MLP are visualized against ground truth:
- **Duffing**: KAN accurately reconstructs the cubic manifold; candidate fitting recovers $-0.234x^3$ (true: $-0.3x^3$), $R^2 = 0.91$.
- **Van der Pol**: KAN's surface collapses to a near-linear form, failing to capture the parabolic modulation structure of $(1-x^2)v$.

This qualitative evidence corroborates the quantitative statistics: KAN's additive bias is an advantage for univariate terms and a bottleneck for variable coupling.

## Key Experimental Results

### Configuration Ablation Main Table (95% Bootstrap CI, N=100 seeds)

| Configuration | Duffing $R^2$ | Van der Pol $R^2$ |
|---|:---:|:---:|
| KAN Config A ($G=5, k=3$) | 0.835 ± 0.030 | 0.667 ± 0.037 |
| KAN Config C (Sparse-Low) | 0.595 ± 0.033 | **-5.229 ± 5.091** |
| KAN Config E (Aggressive-Grid) | 0.794 ± 0.067 | 0.699 ± 0.065 |
| KAN Config F (Coarse-Grid) | **0.862 ± 0.037** | 0.639 ± 0.302 |
| KAN Config G (Fine-Grid) | 0.745 ± 0.099 | -0.174 ± 0.691 |
| **MLP (337 params)** | **0.957 ± 0.009** | **0.768 ± 0.015** |

### Parameter Scale Ablation (Mean ± 95% CI, N=100 seeds)

| Architecture | Params | Duffing(TF) | VdP(TF) | Duffing(BPTT) | VdP(BPTT) |
|---|:---:|:---:|:---:|:---:|:---:|
| KAN Very Small | 120 | 0.836±0.032 | 0.464±0.166 | 0.914±0.061 | 0.743±0.061 |
| KAN Small | 240 | 0.777±0.079 | 0.322±0.292 | 0.874±0.080 | 0.785±0.073 |
| KAN Wide | 480 | 0.845±0.025 | 0.232±0.570 | 0.468±0.773 | -0.602±2.842 |
| KAN Deep | 880 | -3.146±7.106 | -0.303±1.579 | (unstable) | 0.754±0.079 |
| MLP Tiny | 105 | 0.914±0.026 | 0.593±0.048 | 0.906±0.092 | 0.622±0.173 |
| MLP Small | 337 | 0.957±0.009 | 0.768±0.015 | 0.937±0.047 | 0.879±0.032 |
| MLP Medium | 1185 | 0.960±0.013 | 0.805±0.014 | 0.951±0.033 | 0.879±0.019 |
| **MLP Large** | **4417** | **0.965±0.009** | **0.843±0.010** | **0.932±0.063** | **0.898±0.017** |

### Key Findings

- KAN can recover the cubic structure on Duffing ($-0.234x^3$, true: $-0.3x^3$, $R^2=0.91$), with 38% of seeds succeeding — promising but unreliable.
- KAN systematically fails on Van der Pol — the additive bias cannot stably learn multiplicative coupling.
- Long-horizon supervision via BPTT helps the smallest KAN partially mitigate VdP failure ($R^2$ rises from 0.464 to 0.743), but MLP remains comprehensively superior.
- KAN's hyperparameter sensitivity far exceeds MLP's — VdP $R^2$ ranges from 0.699 to -5.229 — rendering it impractical.
- Deep KAN (880 parameters) exhibits catastrophic instability in recurrent settings ($R^2 = -3.146$).

## Highlights & Insights

- **Honest negative results** — the paper clearly delineates the practical boundary of current vanilla KAN in physics-constrained recurrent architectures, providing an important cautionary signal to the KAN community.
- **Precise diagnosis of additive bias vs. multiplicative coupling**: Duffing and Van der Pol are chosen as a test pair that straddles the boundary of additive separability — the diagnosis directly targets KAN's core architectural assumption.
- **Credibility through large-scale seed statistics**: 100 random seeds + 95% confidence intervals per experiment — conclusions do not depend on favorable initialization.
- **Unique insight into recurrent error accumulation**: KAN may perform adequately in unconstrained ODE settings, but errors amplify rapidly in hard-constrained recurrent formulations — exposing a critical setting-dependent fragility.

## Limitations & Future Work

- Only vanilla KAN is evaluated — improved variants (SKANODEs, Hybrid KAN-MLP, DeepOKAN) may overcome the multiplicative limitation.
- Only two oscillator systems are tested — more complex or chaotic systems (e.g., Lorenz attractor) remain to be studied.
- No comparison with established symbolic discovery methods such as SINDy.
- KAN's native symbolic pruning capability — directly extracting symbolic expressions from spline structure — is not explored.
- Gradient conditioning and optimization landscape are not analyzed — the paper demonstrates *what* fails but does not fully explain *why*.

## Related Work & Insights

- **vs. KAN-ODEs (Koenig et al., 2024)**: Strong performance in unconstrained continuous ODE settings — this paper reveals fragility under hard-constrained recurrent formulations, highlighting critical setting dependence.
- **vs. SKANODEs (Liu et al., 2025)**: Structured KAN may mitigate the multiplication problem via operator chaining (representing $1-x^2$ and then interacting with $v$ separately) — motivating hybrid approaches.
- **Inspiration**: Can a "multiplication-aware KAN" be designed — introducing explicit multiplicative gates into KAN's base layer — to retain the interpretability of additive bias while handling coupled terms?

## Rating

⭐⭐⭐⭐ (4/5)

Overall assessment: A systematic negative-result paper whose empirical claims about the boundary of KAN's additive bias are rigorously supported by 100 seeds × 3 studies. While no new method is proposed, the work provides an indispensable calibration reference for practitioners considering KAN in physics-informed applications.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Astral: Training Physics-Informed Neural Networks with Error Majorants](astral_training_physics-informed_neural_networks_with_error_majorants.md)
- [\[AAAI 2026\] PIMRL: Physics-Informed Multi-Scale Recurrent Learning for Burst-Sampled Spatiotemporal Dynamics](../../AAAI2026/scientific_computing/pimrl_physics-informed_multi-scale_recurrent_learning_for_burst-sampled_spatiote.md)
- [\[NeurIPS 2025\] Neuro-Spectral Architectures for Causal Physics-Informed Networks](../../NeurIPS2025/scientific_computing/neuro-spectral_architectures_for_causal_physics-informed_networks.md)
- [\[NeurIPS 2025\] Multi-Trajectory Physics-Informed Neural Networks for HJB Equations with Hard-Zero Terminal Inventory: Optimal Execution on Synthetic & SPY Data](../../NeurIPS2025/scientific_computing/multi-trajectory_physics-informed_neural_networks_for_hjb_equations_with_hard-ze.md)
- [\[NeurIPS 2025\] Physics-Informed Neural Networks with Fourier Features and Attention-Driven Decoding](../../NeurIPS2025/scientific_computing/physics-informed_neural_networks_with_fourier_features_and_attention-driven_deco.md)

<!-- RELATED:END -->
