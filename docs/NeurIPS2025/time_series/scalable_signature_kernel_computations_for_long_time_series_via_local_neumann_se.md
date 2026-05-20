---
title: >-
  [Paper Note] Scalable Signature Kernel Computations for Long Time Series via Local Neumann Series Expansions
description: >-
  [NeurIPS 2025][Time Series][Signature Kernel] This paper proposes PowerSig, which efficiently computes signature kernels via locally adaptive truncated Neumann series expansions…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "Signature Kernel"
  - "Neumann Series"
  - "Long Time Series"
  - "Goursat PDE"
  - "Kernel Methods"
date: 2026-05-08
content_hash: d5ca818463272cb6
---

# Scalable Signature Kernel Computations for Long Time Series via Local Neumann Series Expansions

**Conference**: NeurIPS 2025
**arXiv**: [2502.20392](https://arxiv.org/abs/2502.20392)  
**Code**: [https://github.com/geekbeast/powersig](https://github.com/geekbeast/powersig)  
**Area**: Time Series
**Keywords**: Signature Kernel, Neumann Series, Long Time Series, Goursat PDE, Kernel Methods

## TL;DR

This paper proposes PowerSig, which efficiently computes signature kernels via locally adaptive truncated Neumann series expansions, reducing memory from $O(\ell^2)$ to $O(\ell P)$ and enabling signature kernel computation on time series of length exceeding one million on a single GPU.

## Background & Motivation

**Background**: The Signature Kernel is a state-of-the-art tool for analyzing high-dimensional sequential data, rooted in rough path theory, with desirable properties including reparameterization invariance, characteristic property, and robustness to noise. It has been widely applied in financial modeling, signal processing, and related domains.

**Limitations of Prior Work**:
- Existing methods either compute truncated signature kernels via dynamic programming ($O(\ell^2)$ memory) or solve a global Goursat PDE via finite differences (also $O(\ell^2)$ memory).
- The KSig library handles sequences of at most ~$16 \times 10^4$ steps on an RTX 4090, while sigkernel is limited to ~$10^3$ steps.
- In applications such as high-frequency financial data and long-term sensor monitoring, sequence lengths routinely exceed one million, rendering existing methods entirely infeasible.

**Key Challenge**: The signature kernel possesses excellent theoretical properties, yet its computation cannot scale to long or rough time series.

**Goal**: To scale signature kernel computation to time series of length $10^6$+ while preserving accuracy.

**Key Insight**: The paper exploits the geometric structure of piecewise-linear time series to reformulate the global solution of the Goursat PDE as tile-local Neumann series expansions, storing only local series coefficients rather than a global $\ell \times \ell$ grid.

**Core Idea**: The PDE defining the signature kernel is rewritten as a Volterra integral equation; a Neumann series expansion is applied on each tile; recursive boundary propagation connects adjacent tiles, yielding a computation whose memory scales linearly with sequence length.

## Method

### Overall Architecture

The signature kernel is defined as the solution to the Goursat PDE:

$$\frac{\partial^2 K(s,t)}{\partial s \partial t} = \rho_{\boldsymbol{x},\boldsymbol{y}}(s,t) K(s,t), \quad K(0,\cdot) = K(\cdot,0) = 1$$

where $\rho_{\boldsymbol{x},\boldsymbol{y}}(s,t) = \langle \hat{\boldsymbol{x}}'(s), \hat{\boldsymbol{y}}'(t) \rangle$ is piecewise constant.

The equivalent Volterra integral equation is: $K(s,t) = 1 + \int_0^t \int_0^s \rho(u,v) K(u,v) \, du \, dv$

### Key Designs

#### 1. Tile Decomposition and Local Solutions

**Function**: The domain $[0,1]^2$ is partitioned into $(\ell-1)^2$ tiles $T_{k,l}$ according to the time series data points, and the PDE is solved locally on each tile.

**Mechanism**: On tile $T_{k,l}$, $\rho$ is constant $\rho_{k,l} = \Delta_k \boldsymbol{x} \cdot \Delta_l \boldsymbol{y}$. The Volterra-form Neumann series gives:

$$\kappa_{k,l} = \sum_{n=0}^{\infty} \boldsymbol{T}_{k,l}^n \big(\kappa_{k-1,l}(\sigma_k, \cdot) + \kappa_{k,l-1}(\cdot, \tau_l) - \kappa_{k-1,l-1}(\sigma_k, \tau_l)\big)$$

The solution on each tile depends only on the boundary values of its left and bottom neighbors.

**Design Motivation**: The spectral radius of the integral operator $\boldsymbol{T}_\rho$ is zero (Lemma 2.3), guaranteeing convergence of the Neumann series regardless of the magnitude of $\rho$.

#### 2. Tile-Centered Power Series Expansion (Proposition 2.8)

**Function**: The solution on each tile is represented as a power series centered at the tile corner.

**Core Formula**:

$$\kappa_{k,l}(s,t) = \sum_{i,j=0}^{\infty} \tilde{c}_{i,j}^{(k,l)} (s - \sigma_k)^i (t - \tau_l)^j$$

The coefficient matrix is $\tilde{C}^{k,l} = A_{k,l} \odot B_{k,l} \odot W$, where:
- $A_{k,l}$ encodes the local roughness $\rho_{k,l}$
- $B_{k,l}$ encodes boundary conditions propagated recursively from neighboring tiles
- $W$ is the combinatorial weight matrix

**Design Motivation**: Tile-centered expansion avoids the computational overhead of $L_\sigma, R_\tau$ matrices arising in origin-centered expansions, substantially simplifying the recurrence.

#### 3. Adaptive Truncation Strategy

**Function**: The truncation order is adaptively determined for each tile based on the magnitude of $|\rho_{k,l}|$.

**Mechanism**: Truncation error decays as $O((n!)^{-2})$; tiles with small $|\rho_{k,l}|$ require low-order truncation, while tiles with large $|\rho_{k,l}|$ require higher order. A default truncation order of 7 suffices for the vast majority of cases.

#### 4. Parallelization and DAG Scheduling

Tiles on the same anti-diagonal ($k + l = \text{const}$) are mutually independent and can be computed in parallel. The full computation proceeds along the topological order of a directed acyclic graph (DAG).

### Complexity Analysis

| Aspect | Prior Methods (KSig/sigkernel) | PowerSig |
|--------|-------------------------------|----------|
| Time | $O(\ell^2 d)$ | $O(\ell^2 d)$ (same) |
| Space | $O(\ell^2)$ | $O(\ell P)$, $P \ll \ell$ |
| Max sequence length (RTX 4090) | ~$1.6 \times 10^5$ | $> 10^6$ |

## Key Experimental Results

### Accuracy Comparison

| Sequence length $\ell$ | PowerSig MAPE | KSig PDE MAPE |
|------------------------|--------------|---------------|
| 9 | ~$10^{-12}$ | ~$10^{-8}$ |
| 17 | ~$10^{-11}$ | ~$10^{-5}$ |
| 65 | ~$10^{-10}$ | ~$10^{-3}$ |
| 129 | ~$10^{-10}$ | ~$10^{-2}$ |
| 513 | ~$10^{-9}$ | ~$10^{-1}$ |

PowerSig achieves accuracy several orders of magnitude higher than KSig-PDE.

### Accuracy on Rough Sequences (Low Hurst Exponent)

| Hurst exponent | PowerSig MAPE | KSig PDE MAPE |
|----------------|--------------|---------------|
| 0.4 | ~$10^{-10}$ | ~$10^{-2}$ |
| 0.1 | ~$10^{-9}$ | ~$10^{0}$ |
| 0.005 | ~$10^{-8}$ | ~$10^{2}$ |

The advantage of PowerSig grows as the Hurst exponent decreases (i.e., sequences become rougher).

### Memory and Runtime

| Sequence length | PowerSig memory | KSig memory |
|----------------|----------------|-------------|
| 513 | ~10 MB | ~400 MB |
| 4097 | ~30 MB | KSig OOM |
| 524,289 | ~720 MB | Infeasible |

### Downstream Tasks

| Task | PowerSig | KSig-PDE |
|------|----------|----------|
| Bitcoin regression MAPE (test) | **2.81%** | 3.23% |
| Eigenworms classification (L=1024) | **61.1%** | OOM |

### Key Findings

1. PowerSig achieves superior accuracy over PDE and DP methods across all sequence lengths and roughness levels.
2. Memory is reduced from $O(\ell^2)$ to $O(\ell P)$, making sequences of length $10^6$ feasible on a single GPU.
3. Numerical stability is substantially improved for low Hurst exponent (high roughness) sequences.
4. On the Bitcoin regression task, higher-accuracy signature kernels yield better predictive performance.
5. Scalability to high dimensions is strong: runtime grows nearly linearly as $d$ ranges from 2 to 8192.

## Highlights & Insights

- **Space complexity breakthrough**: The reduction $O(\ell^2) \to O(\ell P)$ is the central contribution, bringing signature kernels toward practical applicability.
- **Mathematical elegance**: The derivation chain—Volterra integral equation → Neumann series → recursive tile propagation—is remarkably clean.
- **Adaptive truncation**: Using $|\rho_{k,l}|$ to control the expansion depth per tile effectively balances accuracy and computation.
- **Numerical stability**: PDE methods suffer error explosion on rough paths; PowerSig avoids this through local expansions.
- **Zero spectral radius guarantee**: Lemma 2.3 ensures convergence of the Neumann series for arbitrary $\rho$, without requiring a contraction mapping.

## Limitations & Future Work

1. The time complexity remains $O(\ell^2 d)$; runtime is not improved.
2. Only piecewise-linear interpolation is currently supported; higher-order or learnable interpolation schemes remain unexplored.
3. The $O(N^2)$ complexity in the number of sequences $N$ for kernel matrix computation is a well-known limitation of kernel methods and is not addressed.
4. Inter-tile boundary propagation is sequential (parallelism is only along anti-diagonals), leaving hardware parallelism underutilized.
5. No a posteriori error bounds are provided.

## Related Work & Insights

- Salvi et al. (2021) first formulated the signature kernel as a Goursat PDE; PowerSig builds upon this by introducing tile-local computation.
- A concurrent work by Cass et al. (2025) adopts different mathematical premises but also employs recursive power series.
- The method is applicable to high-frequency financial data analysis, long-term sensor monitoring, astronomical radio signal processing, and other domains requiring extremely long sequence handling.

## Rating

⭐⭐⭐⭐

The mathematical derivations are elegant, and the paper addresses a critical scalability bottleneck of signature kernels. The memory improvement is substantial ($O(\ell^2) \to O(\ell P)$); however, time complexity is unchanged, and the inherent $O(N^2)$ issue of kernel methods remains open.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] L2GTX: From Local to Global Time Series Explanations](../../CVPR2026/time_series/l2gtx_from_local_to_global_time_series_explanations.md)
- [\[NeurIPS 2025\] WaLRUS: Wavelets for Long-range Representation Using SSMs](walrus_wavelets_for_long-range_representation_using_ssms.md)
- [\[NeurIPS 2025\] Synthetic Series-Symbol Data Generation for Time Series Foundation Models](synthetic_series-symbol_data_generation_for_time_series_foundation_models.md)
- [\[NeurIPS 2025\] Time-O1: Time-Series Forecasting Needs Transformed Label Alignment](time-o1_time-series_forecasting_needs_transformed_label_alignment.md)
- [\[NeurIPS 2025\] Selective Learning for Deep Time Series Forecasting](selective_learning_for_deep_time_series_forecasting.md)

</div>

<!-- RELATED:END -->
