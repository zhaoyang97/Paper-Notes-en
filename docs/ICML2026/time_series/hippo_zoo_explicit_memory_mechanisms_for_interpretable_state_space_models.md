---
title: >-
  [Paper Note] HiPPO Zoo: Explicit Memory Mechanisms for Interpretable State Space Models
description: >-
  [ICML 2026][Time Series][HiPPO] The implicit memory mechanisms in modern SSMs (e.g., Mamba) are **externalized** by proposing "HiPPO Zoo" (5 variants) via an extension of the HiPPO framework. Each variant implements spec…
tags:
  - "ICML 2026"
  - "Time Series"
  - "HiPPO"
  - "State Space Models"
  - "Interpretable Memory"
  - "Polynomial Bases"
date: 2026-05-08
content_hash: d0baeeae5d7a5d0b
---

# HiPPO Zoo: Explicit Memory Mechanisms for Interpretable State Space Models

**Conference**: ICML 2026  
**arXiv**: [2602.21340](https://arxiv.org/abs/2602.21340)  
**Code**: To be confirmed  
**Area**: Time Series / Interpretability / State Space Models  
**Keywords**: HiPPO, State Space Models, Interpretable Memory, Polynomial Bases

## TL;DR
The implicit memory mechanisms in modern SSMs (e.g., Mamba) are **externalized** by proposing "HiPPO Zoo" (5 variants) via an extension of the HiPPO framework. Each variant implements specific modern SSM capabilities (non-linearity, adaptive memory, associative memory, multi-scale, predictive target constraints) using interpretable polynomial representations, achieving 100% accuracy on Selective Copy and Associative Recall tasks.

## Background & Motivation

**Background**: State Space Models (SSMs) have gained attention in sequence modeling for their efficiency and long-range dependency modeling. From S4 to Mamba, SSMs have progressively introduced capabilities such as input-dependent state updates, adaptive memory allocation, non-linear interactions, and implicit associative memory.

**Limitations of Prior Work**: Although modern SSMs are powerful, the mechanisms by which they represent, prioritize, and transform historical information remain a "black box." These capabilities are implicitly encoded in learned state dynamics, making them difficult to analyze or interpret directly.

**Key Challenge**: The HiPPO framework provides explicit and interpretable history representation (via orthogonal polynomial projection), but original HiPPO lacks the key capabilities found in modern SSMs—creating a gap between interpretability and expressivity.

**Goal**: Map the implicit capabilities of modern SSMs back to the HiPPO framework to make them explicit while retaining the interpretable polynomial structure of HiPPO.

**Key Insight**: Many modern SSM capabilities can be achieved through explicit modifications to HiPPO's historical measures, polynomial coefficient dynamics, or objective function constraints.

**Core Idea**: Design "HiPPO Zoo"—five extended variants of HiPPO, each explicitly implementing a modern SSM capability in an interpretable manner. This transforms memory mechanisms from "learned black-box state transitions" into "visualizable and analyzable polynomial structures."

## Method

### Overall Architecture
The core of HiPPO is representing the history of a signal via projection onto orthogonal polynomial bases. Given an input signal $f(t)$, the HiPPO state $\mathbf{s}(t)$ at time $t$ satisfies the linear ODE $\dot{\mathbf{s}}(t) = A \mathbf{s}(t) + \mathbf{b} f(t)$. This work extends HiPPO to support modern SSM capabilities by modifying the measures, dynamics, or objective functions.

### Key Designs

1.  **Volterra HiPPO (Non-linear Interaction)**:
    - **Function**: Implements non-linearity through a polynomial Volterra series readout while maintaining linear state dynamics.
    - **Mechanism**: Volterra kernels are represented as products of orthogonal polynomials. The readout is $y(t) = \beta^{(0)} + \sum_{k=1}^K \sum_{i_1, \ldots, i_k} \beta_{i_1, \ldots, i_k}^{(k)} s_{i_1}(t) \cdots s_{i_k}(t)$, where each coefficient tensor $\beta^{(k)}$ directly encodes the $k$-th order interaction strength.
    - **Design Motivation**: Black-box MLP readouts make it difficult to analyze which historical interactions were learned; the Volterra form makes the interaction structure fully visualizable. Volterra HiPPO converges faster than a single-layer MLP on Volterra tasks and can precisely recover the kernels.

2.  **Salience HiPPO (Adaptive Memory Allocation)**:
    - **Function**: Achieves adaptive weighting of the past via a time-varying scalar "salience" signal, granting more memory resources to critical information.
    - **Mechanism**: The HiPPO ODE is multiplied by a salience signal $g(t)$: $\dot{\mathbf{s}}(t) = g(t) [A \mathbf{s}(t) + \mathbf{b} f(t)]$. Through the time transformation $t_1 = \varphi(t_0) = \int_0^{t_0} g(s) ds$, this is mathematically equivalent to applying standard HiPPO in "warped time," resulting in an explicit deformation of the historical measure.
    - **Design Motivation**: Selective SSMs like Mamba adjust memory allocation implicitly via high-dimensional gating, which is hard to visualize. Salience HiPPO explicitly warps historical measures with a scalar signal. In Selective Copy tasks, salience significantly increases at key points, achieving 100% accuracy (compared to S4D 81%, LSTM 60.8%, and Transformer 22.1%).

3.  **Associative Memory HiPPO (Associative Memory)**:
    - **Function**: Implements explicit key-value associative storage and retrieval, similar to Transformer attention but using polynomial bases.
    - **Mechanism**: Two components are maintained—a HiPPO system for processing the input sequence and an independent associative memory bank for $d_{\text{model}}$ channels. Each channel $m_j(x)$ is a polynomial function defined on the address space $x \in [0, 1]$. Write operations perform minimum-norm updates at address $x_{\text{key}}$ such that $m_j(x_{\text{key}}) = y_t[j]$; read operations evaluate the polynomial at query address $x_{\text{query}}$.
    - **Design Motivation**: Associative behavior in modern SSMs is implemented via implicit state-dependent updates. Explicit Orthogonal Polynomial (OP) based associative memory makes key-value binding fully visualizable. It reaches 100% accuracy on Associative Recall tasks, whereas parameter-matched S4D, LSTM, and Transformer models perform near chance levels (33%).

## Key Experimental Results

### Main Results

| Model | Parameters | Selective Copy Acc. | Assoc. Recall Acc. | Note |
| :--- | :--- | :--- | :--- | :--- |
| Vanilla HiPPO | 25k | 27.6% | 22.8% | Baseline without extensions |
| Vanilla + MLP | 25k | 27.7% | 20.4% | MLP readout performs poorly |
| **Salience HiPPO** | 25k | **100.0%** | 27.1% | Salience adaptation |
| **Assoc. Memory HiPPO** | 25k | 65.7% | **100.0%** | Explicit associative memory |
| S4D (Single layer) | 26k | 81.0% | 33.2% | General SSM baseline |
| LSTM (Single layer) | 25k | 60.8% | 32.7% | RNN baseline |
| Transformer (Single layer) | 27k | 22.1% | 33.9% | Attention baseline |

### Performance Trade-offs in Real Tasks

| Dataset | HiPPO+MLP PPL | Mamba PPL | Performance Gap |
| :--- | :--- | :--- | :--- |
| WikiText-2 Char-level | 6.80 | 5.54 | 23% Decrease |
| Salience HiPPO | 7.52 | 5.54 | 36% Decrease |
| Assoc. Memory | 20.41 | 5.54 | 268% Decrease |

The paper explicitly states that HiPPO Zoo trades raw performance for interpretability. While HiPPO variants excel when task requirements align with explicit mechanisms (synthetic tasks), they remain lower than modern SSMs on general sequence modeling.

### Key Findings
- **Multi-scale Reconstruction**: A single multi-scale HiPPO system matches or exceeds the reconstruction error of a group of single-scale HiPPO systems across a wide range of timescales from $10^1$ to $10^4$.
- **Predictive Memory**: Short-horizon predictors emphasize details at small delays, while long-horizon predictors retain smooth long-term structures—revealing how training objectives implicitly shape memory structures in SSMs.

## Highlights & Insights
- **Deep Trade-off between Interpretability and Performance**: The paper does not claim HiPPO Zoo is superior in performance but systematically demonstrates how to design transparent memory mechanisms for applications where interpretability is crucial (scientific computing, online learning, mechanistic research).
- **Unified Framework Reveals Modular SSM Structure**: Through orthogonal dimensions like measure modification, dynamics modification, and readout modification, the 5 HiPPO variants clearly decompose the capabilities of modern SSMs, providing a foundation for future hybrid architecture design and theoretical analysis.
- **Power of Polynomial Bases**: Using techniques like Volterra projection, orthogonal polynomial reproducing kernels, and multi-scale OP expansion, the paper shows that while HiPPO is a linear system, its polynomial structure can implement many non-linear effects in an explicit and interpretable way.
- **Deriving Memory from Objective Functions**: The insight from Forecasting HiPPO—that the quadratic form $Q = T^\top T$ induced by the prediction target directly defines the geometry of the historical space—is profound, suggesting that different downstream tasks should learn different historical representations.

## Limitations & Future Work
- WikiText-2 experiments show a significant performance drop (23%-36%) for HiPPO Zoo in general sequence modeling, indicating that interpretability comes at a substantial cost to expressivity.
- The continuous address strategy of Associative Memory HiPPO performs poorly on auto-regressive language prediction (20.41 PPL), suggesting that not all explicit mechanisms are suitable for all tasks.
- Computational costs vary: Volterra HiPPO reaches $O(N^k)$ complexity in higher orders; Multiscale HiPPO requires $O(N M^2)$ per update.
- Future Directions: Low-rank tensor approximations could reduce Volterra complexity from $O(N^k)$ to $O(k N r)$; hybrid architectures (embedding interpretable components within expressive SSMs) might achieve the best balance of performance and interpretability for certain sub-tasks.

## Related Work & Insights
- **vs S4 / Mamba**: S4 uses HiPPO for initialization but learns general linear dynamics, losing explicitness; Mamba adds selective updates but increases black-box opacity. This paper employs reverse thinking—designing explicit variants starting from modern capabilities.
- **vs Transformer Attention**: Associative Memory HiPPO achieves a similar key-value mechanism via polynomial reproducing kernels with far fewer parameters (25k vs 27k; 100% vs 33.9% on Associative Recall).
- **vs Interpretable ML**: This work complements existing "post-hoc" explanation methods (LIME, SHAP) by baking interpretability directly into the architecture through "ante-hoc" design.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The idea of reverse-mapping modern SSM implicit capabilities into an explicit polynomial framework is innovative; the 5 variants systematically cover multiple dimensions of modern SSMs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Synthetic tasks are cleverly designed to isolate mechanisms, and qualitative visual summaries fully demonstrate the explicit memory structure. General sequence modeling was only tested on WikiText-2 character-level.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear logic; the derivation from existing pain points to the core conflict and finally the solution is smooth. Mathematical notation is precise.
- **Value**: ⭐⭐⭐⭐⭐ Directly valuable for scientific computing, online learning, and highly regulated fields requiring interpretability; provides new tools for SSM theoretical analysis and hybrid architecture design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] FRACTAL: State Space Model with Fractional Recurrent Architecture for Computational Temporal Analysis of Long Sequences](fractal_ssm_with_fractional_recurrent_architecture_for_computational_temporal_an.md)
- [\[ICML 2026\] Learning Long Range Spatio-Temporal Representations over Continuous Time Dynamic Graphs with State Space Models](learning_long_range_spatio-temporal_representations_over_continuous_time_dynamic.md)
- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](../../NeurIPS2025/time_series/structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[NeurIPS 2025\] Parallelization of Non-linear State-Space Models: Scaling Up Liquid-Resistance Liquid-Capacitance Networks for Efficient Sequence Modeling](../../NeurIPS2025/time_series/parallelization_of_non-linear_state-space_models_scaling_up_liquid-resistance_li.md)
- [\[NeurIPS 2025\] RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting](../../NeurIPS2025/time_series/rivermamba_a_state_space_model_for_global_river_discharge_and_flood_forecasting.md)

</div>

<!-- RELATED:END -->
