---
title: >-
  [Paper Note] Deep Continuous-Time State-Space Models for Marked Event Sequences
description: >-
  [NeurIPS 2025 Spotlight][Marked Temporal Point Processes] S2P2 unifies linear Hawkes processes with deep state space models by stacking multiple implicit Linear Hawkes (LLH) layers with nonlinear activations, yielding a highly expressive continuous-time MTPP model. It leverages parallel scanning to achieve linear complexity and sub-linear runtime, improving predictive likelihood by an average of 33% across 8 real-world datasets.
tags:
  - "NeurIPS 2025 Spotlight"
  - "Marked Temporal Point Processes"
  - "State Space Models"
  - "Hawkes Process"
  - "Parallel Scan"
  - "Continuous-Time Modeling"
date: 2026-05-08
content_hash: 8d769e2e2e0b4876
---

# Deep Continuous-Time State-Space Models for Marked Event Sequences

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2412.19634](https://arxiv.org/abs/2412.19634)  
**Code**: Integrated into [EasyTPP](https://github.com/ant-research/EasyTPP)  
**Area**: Others (Temporal Point Processes / State Space Models)
**Keywords**: Marked Temporal Point Processes, State Space Models, Hawkes Process, Parallel Scan, Continuous-Time Modeling

## TL;DR
S2P2 unifies linear Hawkes processes with deep state space models by stacking multiple implicit Linear Hawkes (LLH) layers with nonlinear activations, yielding a highly expressive continuous-time MTPP model. It leverages parallel scanning to achieve linear complexity and sub-linear runtime, improving predictive likelihood by an average of 33% across 8 real-world datasets.

## Background & Motivation

**Background**: Marked Temporal Point Processes (MTPPs) model irregular event sequences with applications in e-commerce, healthcare, and finance. RNN-based methods incur $O(N)$ sequential computation; Transformer-based methods scale as $O(N^2)$.

**Limitations of Prior Work**:
   - RNN-MTPP: sequential computation and weak long-range dependency modeling
   - Transformer-MTPP: quadratic complexity, prohibitive for long sequences (e.g., patient medical histories)
   - Classical Hawkes processes: interpretable but limited in expressiveness

**Key Challenge**: How can one simultaneously achieve high expressiveness, long-range dependency capture, and efficient parallel computation?

**Key Insight**: SSMs (State Space Models) have demonstrated efficient parallel computation and long-range dependency modeling for discrete sequences, but direct application to MTPPs is hindered by event-driven discontinuities (jump inputs).

**Core Idea**: Unify the jump stochastic differential equations of Hawkes processes with the state recurrence of SSMs, constructing LLH layers that preserve the inductive biases of event sequences while enabling efficient computation via parallel scanning.

## Method

### From Hawkes to LLH: A Unified Formulation

The differential form of the classical Linear Hawkes Process (LHP) intensity:

$$d\bm{\lambda}_t = -\bm{\beta}(\bm{\lambda}_{t-} - \bm{\nu})dt + \bm{\alpha} d\mathbf{N}_t$$

The SSM state equation:

$$d\mathbf{x}(t) = \mathbf{A}\mathbf{x}(t)dt + \mathbf{Bu}(t)dt$$

Structural analogy between the two: $\bm{\lambda}_t \leftrightarrow \mathbf{x}(t)$, $-\bm{\beta} \leftrightarrow \mathbf{A}$, $\bm{\nu}_t \leftrightarrow \mathbf{Bu}(t)$. However, LHP is restricted to $K$ dimensions (number of marks), while SSMs lack event impulse terms.

The **LLH layer** unifies both:

$$d\mathbf{x}_t = -\mathbf{A}\mathbf{x}_{t-}dt + \mathbf{A}\mathbf{Bu}_{t-}dt + \mathbf{E}\bm{\alpha} d\mathbf{N}_t$$

$$\mathbf{y}_t = \mathbf{C}\mathbf{x}_t + \mathbf{Du}_t$$

where $\mathbf{A} \in \mathbb{R}^{P \times P}$ is a general dynamics matrix (more expressive than LHP's $\bm{\beta}$), $\mathbf{E}\bm{\alpha} d\mathbf{N}_t$ is the event impulse term, and $P$ can be arbitrarily large.

### Diagonalization and Parallel Computation

1. **Diagonalization**: Setting $-\mathbf{A} = \mathbf{V}\bm{\Lambda}\mathbf{V}^{-1}$ and directly parameterizing $\bm{\Lambda}$ in the complex plane (constraining real parts to be negative for stability), avoiding matrix exponentiation.
2. **ZOH Discretization**: A zero-order hold assumption yields closed-form updates:

$$\tilde{\mathbf{x}}_{t'} = \bar{\bm{\Lambda}}\tilde{\mathbf{x}}_t + (\bar{\bm{\Lambda}} - \mathbf{I})\tilde{\mathbf{B}}\mathbf{u}_{t'-} + \tilde{\mathbf{E}}\bm{\alpha}_k$$

where $\bar{\bm{\Lambda}} = \exp(\bm{\Lambda}(t'-t))$ denotes element-wise exponentiation.

3. **Parallel Scan**: The update follows the standard form of a linear recurrence $\mathbf{z}_{i+1} = \mathbf{R}_i \mathbf{z}_i + \mathbf{b}_i$, which can be computed in $O(\log N)$ time via parallel scanning.

### Input-Dependent Dynamics

Inspired by Mamba, the dynamics matrix is allowed to depend on the input:

$$\bm{\Lambda}_i = \text{diag}(\text{softplus}(\mathbf{W}'\mathbf{u}_{t_i} + \mathbf{b}'))\bm{\Lambda}$$

This remains conditionally linear ($\bm{\Lambda}_i$ depends only on input $\mathbf{u}$, not on state $\mathbf{x}$), and is still amenable to parallel scanning.

### S2P2 Architecture

$L$ LLH layers are stacked with position-wise nonlinearities (GELU), LayerNorm, and residual connections:

$$\mathbf{u}_t^{(l+1)} = \text{LayerNorm}^{(l)}(\sigma(\mathbf{y}_t^{(l)}) + \mathbf{u}_t^{(l)})$$

Final intensity: $\bm{\lambda}_t = \mathbf{s} \odot \text{softplus}((\mathbf{W}\mathbf{u}_{t-}^{(L+1)} + \mathbf{b}) \odot \mathbf{s}^{-1})$

Training objective: maximize log-likelihood $\mathcal{L}(\mathcal{H}_T) = \sum_{i=1}^{N_T} \log \lambda_{t_i}^{k_i} - \int_0^T \lambda_s ds$ (integral term estimated via Monte Carlo).

Key feature: no parameterized decoding head is required; the intensity is computed directly from the continuously evolving hidden state.

## Key Experimental Results

### Overall Rankings (8 Datasets, Averaged over 5 Random Seeds)

| Model | Likelihood Rank | Mark Pred. | Time Pred. | Mark Calib. | Time Calib. | **Overall Rank** |
|-------|----------------|------------|------------|-------------|-------------|-----------------|
| RMTPP | 6.8 | 6.9 | 5.0 | 5.8 | 6.1 | 6.1 |
| NHP | 2.4 | 1.8 | 2.4 | 4.9 | 3.6 | 2.9 |
| AttNHP | 2.4 | 2.9 | 6.6 | 3.4 | 3.7 | 3.7 |
| IFTPP | 4.1 | 5.0 | 4.0 | 1.8 | 2.6 | 3.6 |
| **S2P2** | **1.9** | **1.9** | **2.3** | **3.0** | **2.8** | **2.1** |

S2P2 outperforms all baselines by nearly one full rank on average.

### Total Log-Likelihood (nats/event, Selected Datasets)

| Model | Amazon | Taxi | StackOverflow | MIMIC-II | EHRSHOT |
|-------|--------|------|---------------|----------|---------|
| NHP | 0.129 | 0.514 | -2.241 | 0.060 | -3.966 |
| AttNHP | 0.484 | 0.493 | -2.194 | -0.170 | OOM |
| IFTPP | 0.496 | 0.453 | -2.233 | 0.317 | -6.596 |
| **S2P2** | **0.781** | **0.522** | **-2.163** | **0.919** | **-2.512** |

S2P2 demonstrates a particularly pronounced advantage on EHRSHOT (a large-scale medical dataset with the longest sequences and most marks), where AttNHP runs out of memory.

### Synthetic Experiment Validation
- Classical Hawkes / self-correcting processes: ground-truth intensities recovered almost perfectly.
- Non-homogeneous Poisson process with square-wave intensity: models such as NHP fail due to parametric constraints, while S2P2 captures the pattern perfectly.
- Long-range dependency: S2P2 recovers 98% of the true likelihood vs. 88% for NHP.

## Highlights & Insights
- **Elegant framework**: The unified perspective from Hawkes to SSM reveals intrinsic connections between the two model families.
- Theoretical expressiveness results for SSMs (Muca Cirone et al., 2024) apply directly to S2P2.
- **Substantial efficiency gains**: $O(\log N)$ parallel time vs. $O(N)$ for RNNs and $O(N^2)$ for Transformers.
- S2P2 is a **genuinely continuous-time model**—unlike MHP (Mamba for TPP), which relies on discrete encoding with a parameterized decoding head.
- Capable of handling ultra-long sequences such as EHRSHOT, where Transformer-based models run out of memory.

## Limitations & Future Work
- The diagonalization assumption requires the system to be diagonalizable, which may be restrictive under extreme conditions.
- Monte Carlo estimation of the integral term introduces variance.
- Efficiency comparisons with intensity-free methods (normalizing flow TPPs) are absent.
- Performance on discrete event generation and simulation tasks has not been evaluated.
- The ZOH assumption may be insufficiently accurate for very dense event windows.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The unified Hawkes+SSM framework establishes a novel connection; the impulse jump SDE design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 datasets × 6 metrics × synthetic validation × comprehensive baselines.
- Writing Quality: ⭐⭐⭐⭐ Derivations are rigorous and clear; figures are intuitive.
- Value: ⭐⭐⭐⭐⭐ Establishes a new standard tool for the MTPP community, combining efficiency and performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Deep Learning for Continuous-Time Stochastic Control with Jumps](deep_learning_for_continuous-time_stochastic_control_with_jumps.md)
- [\[ICML 2025\] UnHiPPO: Uncertainty-Aware Initialization for State Space Models](../../ICML2025/others/unhippo_uncertainty-aware_initialization_for_state_space_models.md)
- [\[NeurIPS 2025\] Continuous Thought Machines](continuous_thought_machines.md)
- [\[ICLR 2026\] Exploring State-Space Models for Data-Specific Neural Representations](../../ICLR2026/others/exploring_state-space_models_for_data-specific_neural_representations.md)
- [\[NeurIPS 2025\] Addressing Mark Imbalance in Integration-free Neural Marked Temporal Point Processes](addressing_mark_imbalance_in_integrationfree_neural_marked_t.md)

</div>

<!-- RELATED:END -->
