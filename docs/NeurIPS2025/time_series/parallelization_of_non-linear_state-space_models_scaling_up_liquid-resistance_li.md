---
title: >-
  [Paper Note] Parallelization of Non-linear State-Space Models: Scaling Up Liquid-Resistance Liquid-Capacitance Networks for Efficient Sequence Modeling
description: >-
  [NeurIPS 2025][Time Series][State Space Models] This paper proposes LrcSSM, which achieves exact and efficient parallelization of nonlinear RNNs by constraining the Jacobian matrix of Liquid-Resistance Liquid-Capacitance…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "State Space Models"
  - "Nonlinear RNN"
  - "Parallelization"
  - "Biologically Inspired"
  - "Diagonal Jacobian"
date: 2026-05-08
content_hash: d4537635d8078c53
---

# Parallelization of Non-linear State-Space Models: Scaling Up Liquid-Resistance Liquid-Capacitance Networks for Efficient Sequence Modeling

**Conference**: NeurIPS 2025
**arXiv**: [2505.21717](https://arxiv.org/abs/2505.21717)  
**Code**: [GitHub](https://github.com/MoniFarsang/LrcSSM)  
**Area**: Time Series
**Keywords**: State Space Models, Nonlinear RNN, Parallelization, Biologically Inspired, Diagonal Jacobian

## TL;DR

This paper proposes LrcSSM, which achieves exact and efficient parallelization of nonlinear RNNs by constraining the Jacobian matrix of Liquid-Resistance Liquid-Capacitance (LRC) networks to be diagonal, surpassing Transformer, LRU, S5, and Mamba on long-sequence classification benchmarks.

## Background & Motivation

Linear state space models (e.g., S4, S5, Mamba) have achieved great success in sequence modeling due to their parallelizability, yet their linear state transitions limit expressiveness. Traditional nonlinear RNNs can capture input correlations more finely through nonlinear state updates, but their inherent sequential dependencies preclude efficient parallelization, leading to their marginalization.

Recent approaches DEER and ELK attempt to parallelize nonlinear RNNs via Newton iteration combined with parallel scans, but suffer from two issues:

**Dense Jacobian does not scale**: DEER employs a full square Jacobian matrix, which cannot scale to long sequences.

**Approximation inaccuracy**: quasi-DEER/quasi-ELK naively extract the diagonal of the Jacobian, discarding potentially important feedback loop information encoded in off-diagonal elements, and exhibit numerical instability.

**Core Idea**: Rather than discarding off-diagonal Jacobian elements post hoc (approximation), the paper constrains the Jacobian to be diagonal by design. The intuition is that the constant synaptic parameter matrices of nonlinear SSMs can themselves be diagonalized, and complex neuronal feedback loops can be well summarized by their complex eigenvalues.

## Method

### Overall Architecture

The LrcSSM architecture consists of: input encoder → normalization layer → multi-layer nonlinear LRC SSM blocks (2/4/6 layers) → MLP with skip connections → post-normalization → decoder. The core component is the parallelized iterative linearization computation within the LRC SSM blocks.

### Key Designs

1. **Diagonalized LRC Model Design**: In the original LRC equations, the forget conductance $f_i$ and update conductance $z_i$ of each neuron depend on the membrane potentials of **all** neurons. LrcSSM modifies these to depend only on the **neuron's own** membrane potential (state-dependent part) and **all external inputs** (input-dependent part):

    $f_i^*(x_i, \mathbf{u}) = \underbrace{g_i^{max,x} \sigma(a_i^x x_i + b_i^x)}_{x_i \text{ state-dependent}} + \underbrace{g_i^{max,u} \sigma(\sum_j a_{ji}^u u_j + b_j^u)}_{\mathbf{u} \text{ input-dependent}} + g_i^{leak}$

   Similarly, $z_i^*$ and the elastance $\epsilon_i^*$ are modified accordingly. This renders the Jacobian matrix $\mathbf{A}(\mathbf{x}, \mathbf{u})$ naturally diagonal:

    $\mathbf{A}(\mathbf{x}, \mathbf{u}) = \text{diag}[-\sigma(f_i^*) \sigma(\epsilon_i^*)]$

2. **Exact Parallelization**: Since the Jacobian is naturally diagonal, no quasi-approximation is needed. The parallel scan algorithm from DEER/ELK is applied directly, achieving $\mathcal{O}(TD)$ computational complexity and $\mathcal{O}(\log T)$ sequential depth. The key advantage is that Line 8 `$J_s \leftarrow \text{Diag}(J_s)$` in Algorithm 1 is no longer required, as $J_s$ is already diagonal by construction.

3. **Gradient Stability Guarantee**: Unlike Liquid-S4 and Mamba, the diagonal structure of LrcSSM admits a formal proof of gradient stability. The $-\sigma(\cdot)$ terms in the state matrix ensure all diagonal elements are negative, guaranteeing system stability.

### Loss & Training

- Cross-entropy loss for classification tasks
- Explicit Euler integration: $\mathbf{x}_t = \mathbf{x}_{t-1} + \Delta t \cdot \dot{\mathbf{x}}_{t-1}$
- Hyperparameters selected via grid search on the validation set
- Test accuracy averaged over 5 different random seeds

## Key Experimental Results

### Main Results (UEA Multivariate Time Series Classification, Test Accuracy %)

| Method | Heart (405) | SCP1 (896) | SCP2 (1152) | Ethanol (1751) | Motor (3000) | Worms (17984) | Avg |
|--------|------------|-----------|------------|---------------|-------------|-------------|-----|
| LRU | 78.1 | 84.5 | 47.4 | 23.8 | 51.9 | 85.0 | 61.8 |
| S5 | 73.9 | 87.1 | 55.1 | 25.6 | 53.0 | 83.9 | 63.1 |
| Mamba | 76.2 | 80.7 | 48.2 | 27.9 | 47.7 | 70.9 | 58.6 |
| LinOSS-IM | 75.8 | 87.8 | 58.2 | 29.9 | **60.0** | **95.0** | 67.8 |
| Transformer | 70.5 | 84.3 | 49.1 | 40.5 | 50.5 | OOM | 59.0 |
| **LrcSSM** | 72.7 | 85.2 | 53.9 | **36.9** | 58.6 | 90.6 | **66.3** |

### Ablation Study (Diagonal Jacobian Comparison Across Nonlinear RNNs, Fixed 64 Units × 6 Layers)

| Model | Heart | SCP1 | SCP2 | Ethanol | Motor | Worms | Avg |
|-------|-------|------|------|---------|-------|-------|-----|
| MguSSM | 74.0 | 78.3 | 49.6 | 31.1 | **56.4** | **90.0** | 63.2 |
| GruSSM | **75.7** | 80.2 | 52.5 | 34.5 | 49.6 | 86.1 | 63.1 |
| LstmSSM | 75.0 | 78.8 | 51.1 | 32.6 | 54.3 | 82.2 | 62.3 |
| **LrcSSM** | 75.0 | **84.8** | **55.4** | **36.1** | 55.7 | 85.6 | **65.4** |

### Key Findings

1. **Significant advantage on EthanolConcentration**: LrcSSM 36.9% vs. LRU 23.8%, Mamba 27.9%, as this dataset contains rich input correlations that nonlinear state dependence captures more effectively.
2. **Strong performance on long sequences**: Particularly competitive on the three datasets with sequence lengths exceeding 1500.
3. **Diagonalization is lossless**: Comparison against the original dense-Jacobian LRC in the appendix ablation shows that enforcing diagonality does not degrade performance.
4. **Biologically inspired model outperforms generic RNNs**: LrcSSM achieves higher average accuracy across 6 datasets than MguSSM, GruSSM, and LstmSSM.
5. **Only trailing LinOSS-IM**: Likely attributable to LrcSSM's use of explicit Euler integration versus LinOSS-IM's implicit integration, suggesting that improved integration schemes could yield further gains.

## Highlights & Insights

- **Design Philosophy**: Rather than "parallelize first, then approximate the diagonal," the paper advocates "design a model whose Jacobian is naturally diagonal"—addressing the problem at its source.
- **Biological Plausibility**: Liquid resistance and liquid capacitance model the saturation effects and membrane capacitance dynamics of real neurons, providing a theoretically grounded basis for nonlinear expressiveness.
- **General Methodology**: The paper explicitly demonstrates how to convert arbitrary nonlinear RNNs into diagonal Jacobian form (validated on GRU/LSTM/MGU), making the approach broadly generalizable.

## Limitations & Future Work

1. Parallelization requires multi-step Newton iteration to converge, an overhead absent in linear SSMs; the practical speed advantage depends on the number of iterations required.
2. The current use of explicit Euler integration may be suboptimal; implicit integration schemes (e.g., as in LinOSS-IM) could potentially improve accuracy.
3. Performance on short-sequence tasks (e.g., Heart) is moderate, suggesting that nonlinear complexity may be excessive for simpler tasks.
4. Evaluation is currently limited to classification tasks; performance on time-series forecasting and generation tasks remains untested.

## Related Work & Insights

This paper bridges biologically inspired Liquid Neural Networks (LTC→STC→LRC) with modern SSM parallelization techniques (DEER/ELK), demonstrating that nonlinear RNNs can simultaneously approach linear SSMs in both efficiency and performance. The instructive comparison with LinOSS—both biologically motivated yet modeling different phenomena—suggests a richer design space for biologically inspired sequence models.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The idea of achieving a diagonal Jacobian through model design rather than post hoc approximation is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Six datasets, multi-model diagonal Jacobian comparisons, and ablation studies, though task diversity is limited.
- **Writing Quality**: ⭐⭐⭐⭐ Biological background and mathematical derivations are clearly presented; architectural diagrams are intuitive.
- **Value**: ⭐⭐⭐⭐ Provides a practical and principled parallelization pathway for the revival of nonlinear RNNs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[ICLR 2026\] WARP: Weight-Space Linear Recurrent Neural Networks](../../ICLR2026/time_series/weight-space_linear_recurrent_neural_networks.md)
- [\[NeurIPS 2025\] RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting](rivermamba_a_state_space_model_for_global_river_discharge_and_flood_forecasting.md)
- [\[ICML 2026\] HiPPO Zoo: Explicit Memory Mechanisms for Interpretable State Space Models](../../ICML2026/time_series/hippo_zoo_explicit_memory_mechanisms_for_interpretable_state_space_models.md)
- [\[NeurIPS 2025\] Simple and Efficient Heterogeneous Temporal Graph Neural Network](simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)

</div>

<!-- RELATED:END -->
