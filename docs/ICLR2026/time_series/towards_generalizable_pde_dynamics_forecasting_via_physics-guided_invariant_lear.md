---
title: >-
  [Paper Note] Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning
description: >-
  [ICLR 2026][Time Series][PDE Invariant Learning] This paper proposes iMOOE, a framework that explicitly formalizes two-level physical invariance principles — operator invariance and compositional invariance — within PDE…
tags:
  - "ICLR 2026"
  - "Time Series"
  - "PDE Invariant Learning"
  - "Zero-shot OOD Generalization"
  - "Mixture of Operator Experts"
  - "Frequency Enhancement"
  - "Neural Operators"
date: 2026-05-08
content_hash: b78292aca15d38b6
---

# Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning

**Conference**: ICLR 2026
**arXiv**: [2509.24332](https://arxiv.org/abs/2509.24332)  
**Code**: [GitHub](https://github.com/LSY-Cython/iMOOE)  
**Area**: Time Series / PDE Dynamics Forecasting
**Keywords**: PDE Invariant Learning, Zero-shot OOD Generalization, Mixture of Operator Experts, Frequency Enhancement, Neural Operators

## TL;DR

This paper proposes iMOOE, a framework that explicitly formalizes two-level physical invariance principles — operator invariance and compositional invariance — within PDE systems, and instantiates them via a mixture-of-operator-experts network and a frequency-enhanced risk equalization objective, achieving state-of-the-art zero-shot PDE dynamics forecasting across diverse OOD scenarios without any test-time adaptation.

## Background & Motivation

**Background**: Deep learning-based PDE dynamics forecasting has been widely applied in meteorology, battery design, and chemical synthesis. Neural operators such as FNO and DeepONet can learn unknown PDE dynamics from observed trajectories, yet they generalize poorly under out-of-distribution (OOD) conditions. Existing OOD approaches fall into three categories: (1) meta-learning-based domain-aware methods such as CoDA and GEPS, which partition network parameters into domain-invariant and domain-specific components; (2) parameter-conditioned methods such as CAPE, which encode PDE parameters into the model; and (3) large-scale pretraining methods such as DPOT, which enhance transferability by pretraining on diverse PDE data.

**Limitations of Prior Work**: The zero-shot OOD generalization capability of these methods remains insufficient. Meta-learning methods require few-shot test-time adaptation, parameter-conditioned methods rely on known parameter ranges, and pretraining methods demand large and diverse datasets.

**Key Challenge**: None of these methods explicitly identifies or exploits the fundamental physical invariance principles inherent to PDE systems. They focus on learning domain-generalizable representations without touching the truly invariant structural essence of PDE systems — the operators and their compositional relationships.

**Goal**: Given only limited training trajectories, how can one achieve zero-shot (without accessing any test-time data) OOD generalization for PDE dynamics forecasting? Two sub-problems must be addressed: (1) How to define the fundamental invariance principles within PDE systems? (2) How to design network architectures and training objectives to capture such invariance?

**Key Insight**: The authors draw inspiration from operator splitting methods — complex PDEs can be decomposed into compositions of simpler operators. For example, a reaction-diffusion equation consists of a diffusion process governed by the Laplacian operator and a nonlinear reaction function. Regardless of how system parameters vary, these fundamental operators and their compositional relationships remain invariant. Combined with invariant learning theory (IRM/REx), invariant correlations can be identified by equalizing risks across different training environments.

**Core Idea**: The physical symmetries of PDE systems are formalized as two-level invariance principles — operator invariance and compositional invariance — which are captured by an aligned mixture-of-operator-experts architecture and a frequency-enhanced risk equalization objective, enabling zero-shot OOD generalization.

## Method

### Overall Architecture

The overall iMOOE pipeline takes as input the past $H$-step observed trajectory $\mathbf{I}^e = \{\mathbf{u}^e(t,\mathbf{x})\}_{t=0}^{H-1}$ and outputs the predicted trajectory over the future $N_t - H$ steps. The framework consists of two major components: (1) a **Mixture of Operator Experts (MOOE)** network — a set of parallel neural operator experts that capture distinct physical processes, together with a fusion network that aggregates expert outputs conditioned on physical parameters; and (2) a **frequency-enhanced invariant learning objective** — combining a maximum prediction loss, a risk equalization loss, and a frequency enhancement loss to estimate PDE invariance from multiple training environments. At inference time, an autoregressive scheme is adopted: $\hat{\mathbf{u}}_{t+1} = \int_t^{t+1} h(\{\sigma_i\}_{i=1}^K, \mathbf{p}, \mathbf{f}) dt + \mathbf{u}_t$.

### Key Designs

1. **Two-Level PDE Invariance Principles**:

    - **Function**: Provides a theoretical foundation for OOD generalization by identifying what remains invariant across domains within a PDE system.
    - **Mechanism**: (i) **Operator invariance** — PDE dynamics are governed by a composition of spatial operators $\{\sigma_i(\mathbf{x}, \mathbf{u}, \partial_\mathbf{x}\mathbf{u}, \ldots)\}_{i=1}^K$ representing distinct physical processes (e.g., diffusion, advection, reaction), which remain invariant across different environments and system evolutions. (ii) **Compositional invariance** — the aggregation of operators with external conditions (physical parameters $\mathbf{p}$, forcing terms $\mathbf{f}$), expressed as $F = h(\sigma_1, \ldots, \sigma_K, \mathbf{p}, \mathbf{f})$, is fixed for a given PDE system.
    - **Design Motivation**: Inspired by classical operator splitting, which decomposes complex PDEs into simpler operators (e.g., split-step solvers for Navier-Stokes). Structural causal models (SCMs) are used to formally establish that, regardless of variations in initial conditions, physical parameters, or forcing terms, the operators and their compositional relationships remain invariant.

2. **Mixture of Operator Experts (MOOE)**:

    - **Function**: Captures operator invariance through a parallel expert group and compositional invariance through a fusion network.
    - **Mechanism**: $K$ parallel neural operator experts are designed, where each expert computes $\sigma_i = \text{NO}_i(\mathbf{x}, \mathbf{u}_{t-W+1:t}, \mathbf{m}_i \odot [\partial_\mathbf{x}\mathbf{u}_t, \partial_{\mathbf{xx}}\mathbf{u}_t, \ldots]^\mathbb{T})$, with $\mathbf{m}_i \in \{0,1\}^S$ a binary mask vector enabling each expert to adaptively select relevant spatial derivatives. To encourage experts to learn distinct physical processes, a mask diversity loss $\mathcal{L}_{mask} = \frac{1}{K^2}\sum_{i,j}\exp(-\|\mathbf{m}_i - \mathbf{m}_j\|_2^2)$ is introduced. For strongly nonlinear PDEs, the fusion network learns the composition via an auxiliary network; for additive relationships, a simple summation is used.
    - **Design Motivation**: Analogous to operator splitting but uses a parallel rather than sequential structure to avoid computational bottlenecks. Each expert can be any existing neural operator (FNO/DeepONet/OFormer/VCNeF), enabling plug-and-play compatibility.

3. **Frequency-Enhanced Invariant Learning Objective**:

    - **Function**: Estimates PDE invariance from limited training environments while addressing the spectral bias of neural operators.
    - **Mechanism**: The total loss $\mathcal{L}_{total} = \lambda_{pred}\mathcal{L}_{pred} + \lambda_{inv}\mathcal{L}_{inv} + \lambda_{freq}\mathcal{L}_{freq} + \lambda_{mask}\mathcal{L}_{mask}$ comprises four terms: **(a)** the maximum prediction loss $\mathcal{L}_{pred}$ ensures sufficiency — the cross-environment average of autoregressive prediction errors; **(b)** the risk equalization loss $\mathcal{L}_{inv} = \text{Var}(\{\mathcal{R}_{pred}^e\}_{e \in \mathcal{E}_{tr}})$ ensures invariance — minimizing the variance of per-environment risks; **(c)** the frequency enhancement loss $\mathcal{L}_{freq}$ applies wavenumber weighting $\|\xi\|_2^2$ to amplify supervision on high-frequency modes, addressing the low-frequency bias of neural operators.
    - **Design Motivation**: Supervising only in the spatial domain causes neural operators to neglect high-frequency information; in autoregressive prediction, high-frequency errors propagate across the entire spectral domain, severely degrading OOD generalization. Environments are partitioned not only by physical parameters but also by autoregressive step index, since $p(\mathbf{I}^e)$ shifts across time steps — a particularly critical consideration for fluid dynamics forecasting.

### Loss & Training

The total training loss is $\mathcal{L}_{total} = \lambda_{pred}\mathcal{L}_{pred} + \lambda_{inv}\mathcal{L}_{inv} + \lambda_{freq}\mathcal{L}_{freq} + \lambda_{mask}\mathcal{L}_{mask}$, with $\lambda_{pred}=1, \lambda_{freq}=0.1, \lambda_{mask}=0.001$. A linear scheduling strategy is adopted for $\lambda_{inv}$ (upper bound $0.001$), preserving an initial empirical risk minimization phase using only the prediction loss to learn rich predictive representations. Training runs for 500 epochs using the Adam optimizer with an initial learning rate of $0.001$ on an A100 GPU, with $K=2$ experts and FNO (4 layers, width 64) as the default backbone.

## Key Experimental Results

### Main Results: Zero-Shot OOD Generalization on Five PDE Systems (nMSE)

| PDE System | CoDA | CAPE | DPOT | VCNeF | GEPS | **iMOOE** | Gain |
|------------|------|------|------|-------|------|-----------|------|
| DR (OOD) | 6.05e-1 | 7.16e-2 | 5.67e-2 | 7.84e-2 | 7.94e-2 | **4.23e-2** | ↓25% vs DPOT |
| NS (OOD) | 9.14e-1 | 3.56e-1 | 5.08e-1 | 3.81e-1 | 4.13e-1 | **3.12e-1** | ↓12% vs CAPE |
| BG (OOD) | 9.22e-1 | 3.04e-2 | 8.41e-2 | 4.68e-2 | 7.56e-2 | **1.08e-2** | ↓64% vs CAPE |
| SW (OOD) | n.a. | 6.18e-5 | 4.85e-4 | 6.12e-4 | 2.76e-4 | **3.02e-5** | ↓51% vs CAPE |
| HC (OOD) | 2.37e+0 | 3.65e+0 | 2.12e+0 | 1.42e+0 | 1.35e+0 | **1.22e+0** | ↓10% vs GEPS |

Average OOD improvement: nMSE 40.21%, fRMSE 30.78%.

### Ablation Study / Compatibility: Different Neural Operators + iMOOE (DR-OOD Average nMSE)

| Operator Backbone | Naive | +MOOE | +iMOOE | Gain |
|-------------------|-------|-------|--------|------|
| FNO | 7.94e-2 | 5.16e-2 | **4.23e-2** | ↓47% |
| DeepONet | 6.15e-1 | 6.10e-1 | **5.49e-1** | ↓11% |
| VCNeF | 7.84e-2 | 5.73e-2 | **5.52e-2** | ↓30% |
| OFormer | 5.75e-2 | 4.96e-2 | **4.34e-2** | ↓25% |

### Key Findings

- **iMOOE's improvements are universal**: Regardless of the underlying neural operator type (Fourier/branch-trunk/neural field/Transformer), iMOOE consistently reduces both the mean and variance of OOD errors, validating the general value of PDE invariance learning.
- **Frequency enhancement is critical**: The significant additional gain from +MOOE to +iMOOE (with frequency-enhanced training) demonstrates that architectural alignment alone is insufficient — spectral bias must be addressed at the objective function level.
- **Sensitivity to expert count $K$**: $K=3$ is optimal; $K=1$ is insufficient to capture operator invariance, while $K=4$ introduces redundancy (real PDEs have only a small number of constituent operators), with linearly growing computational overhead.
- **Effectiveness on real-world data**: On two real-world ocean dynamics datasets — SST (sea surface temperature) and SSE (sea surface elevation) — iMOOE achieves the lowest mean and variance, demonstrating the method's ability to capture noisy physical dynamics in practice.
- **Strong temporal extrapolation**: In a temporal extrapolation setting (training on $[0, N_t]$, testing on $[0, 2N_t]$), iMOOE achieves an average nMSE improvement of 32.51%, indicating that the learned operator invariance remains effective over longer time horizons.

## Highlights & Insights

- **Formalization of physical invariance**: Elevating the operator-splitting property of PDE systems into a machine-learning-compatible invariance principle constitutes an elegant bridge between physical PDE theory and OOD generalization theory — a more principled approach than encoding parameters into the network (e.g., CAPE) or partitioning the parameter space (e.g., CoDA).
- **Mask diversity drives expert specialization**: Learnable binary masks allow different experts to select different-order derivatives as inputs, providing greater physical interpretability than conventional MoE routers — advection terms naturally attend to first-order derivatives $\partial_\mathbf{x}\mathbf{u}$, while diffusion terms require second-order derivatives $\partial_{\mathbf{xx}}\mathbf{u}$.
- **Frequency-weighted OOD regularization**: Weighting the frequency-domain loss by $\|\xi\|_2^2$ to compensate for the spectral bias of neural operators is a concise yet effective strategy that is transferable to any spatiotemporal forecasting task based on neural operators.
- **Environment partitioning by autoregressive step**: In autoregressive prediction, the input distribution $p(\mathbf{I}^e)$ inherently drifts across time steps; partitioning environments by step index to impose risk equalization constraints is therefore a highly appropriate and novel design choice.

## Limitations & Future Work

- **Validation limited to a small set of PDE systems**: Only 5 simulated systems and 2 real-world datasets have been tested; applicability to irregular grids, non-periodic boundary conditions, and high-dimensional PDEs (3D+) remains unverified.
- **No automatic mechanism for determining $K$**: The paper fixes $K=2$ (default) or $K=3$ (optimal), but different PDE systems have different numbers of constituent operators, and no principled method for automatically selecting $K$ is provided.
- **Physical parameters must be known**: The fusion network requires $\mathbf{p}$ and $\mathbf{f}$ as inputs, yet physical parameters may not be observable in practice. The SST experiments substitute all-ones vectors, but the robustness of this approach requires further validation.
- **Computational cost scales linearly with expert count**: The memory and time overhead of $K$ parallel neural operators is substantial and may become a bottleneck in resource-constrained settings. Introducing a sparse-activation true MoE routing mechanism could improve efficiency.
- **Applicability of invariance assumptions to chaotic systems**: Whether operator invariance assumptions remain valid for strongly chaotic PDEs (e.g., high Reynolds number turbulence) warrants further investigation.

## Related Work & Insights

- **vs CoDA/GEPS (meta-learning)**: These methods partition network parameters into domain-invariant and domain-specific components and require few-shot test-time adaptation. iMOOE derives invariance from the physical structure of PDEs, requires no test-time adaptation whatsoever, and achieves substantially better OOD performance.
- **vs CAPE (parameter conditioning)**: CAPE directly injects PDE parameters into channel attention modules. iMOOE additionally incorporates an operator-splitting structure and invariant learning objectives, outperforming CAPE in most settings — particularly on the BG and SW systems (improvements of 64% and 51%, respectively).
- **vs DPOT (pretraining)**: DPOT employs a Transformer architecture with denoising pretraining. iMOOE requires neither large-scale nor diverse pretraining data, yet achieves superior OOD generalization from limited training environments alone, suggesting that physical structural priors are more important than data volume.
- **vs REx/IRM (invariant learning)**: Classical invariant learning methods have been validated on vision and graph tasks. This work is the first to extend them to PDE dynamics forecasting; the key contribution lies in defining PDE-specific two-level invariance principles and designing aligned network architectures.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First formalization of two-level physical invariance principles for PDE systems integrated with invariant learning theory; the core idea is clear and substantive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five simulated systems and two real-world datasets, compatibility validation across four neural operator types, and comprehensive coverage of multiple OOD scenarios.
- **Writing Quality**: ⭐⭐⭐⭐ Smooth integration of theory, methodology, and experiments; clear mathematical derivations; SCM diagrams aid comprehension.
- **Value**: ⭐⭐⭐⭐ Provides a general plug-and-play framework for OOD generalization of neural operators; the paradigm of combining physical priors with invariant learning has broad reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting](cpiri_channel_permutation-invariant_relational_interaction_for_multivariate_time_se.md)
- [\[NeurIPS 2025\] IonCast: A Deep Learning Framework for Forecasting Ionospheric Dynamics](../../NeurIPS2025/time_series/ioncast_a_deep_learning_framework_for_forecasting_ionospheric_total_electron_con.md)
- [\[ICML 2026\] Time-series Forecasting Through the Lens of Dynamics](../../ICML2026/time_series/time-series_forecasting_through_the_lens_of_dynamics.md)
- [\[ICLR 2026\] HiVid: LLM-Guided Video Saliency For Content-Aware VOD And Live Streaming](hivid_llm-guided_video_saliency_for_content-aware_vod_and_live_streaming.md)
- [\[NeurIPS 2025\] Learning Time-Scale Invariant Population-Level Neural Representations](../../NeurIPS2025/time_series/learning_time-scale_invariant_population-level_neural_representations.md)

</div>

<!-- RELATED:END -->
