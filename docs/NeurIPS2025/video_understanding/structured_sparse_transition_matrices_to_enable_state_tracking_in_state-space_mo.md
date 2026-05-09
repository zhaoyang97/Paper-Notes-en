---
title: >-
  [Paper Note] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models
description: >-
  [NeurIPS 2025][Video Understanding] This paper proposes PD-SSM, a structured sparse parameterization for the state transition matrix of state-space models (SSMs). The core idea is to factorize the transition matrix as a product of a column-wise one-hot matrix P and a complex diagonal matrix D (i.e., $A = PD$), achieving expressiveness equivalent to unstructured (dense) SSMs while retaining computational efficiency comparable to diagonal SSMs at $\Theta(LN)$. A single layer suffices to simulate any $N$-state finite-state automaton (FSA). The paper provides theoretical guarantees on BIBO stability and optimal state dimensionality, with strong empirical results on FSA simulation, multivariate time-series classification, long-sequence benchmarks, and natural-language state-tracking tasks.
tags:
  - NeurIPS 2025
  - Video Understanding
date: 2026-05-08
content_hash: 104b59f2e448da16
---

# Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models

**Conference**: NeurIPS 2025
**arXiv**: [2509.22284](https://arxiv.org/abs/2509.22284)
**Code**: [https://github.com/IBM/expressive-sparse-state-space-model](https://github.com/IBM/expressive-sparse-state-space-model)
**Area**: Video Understanding

## TL;DR

This paper proposes PD-SSM, a structured sparse parameterization for the state transition matrix of state-space models (SSMs). The core idea is to factorize the transition matrix as a product of a column-wise one-hot matrix P and a complex diagonal matrix D (i.e., $A = PD$), achieving expressiveness equivalent to unstructured (dense) SSMs while retaining computational efficiency comparable to diagonal SSMs at $\Theta(LN)$. A single layer suffices to simulate any $N$-state finite-state automaton (FSA). The paper provides theoretical guarantees on BIBO stability and optimal state dimensionality, with strong empirical results on FSA simulation, multivariate time-series classification, long-sequence benchmarks, and natural-language state-tracking tasks.

## Background & Motivation

1. **Limited expressiveness of diagonal SSMs**: Efficient SSMs such as Mamba employ diagonal transition matrices, which are provably incapable of simulating non-solvable automata, leading to poor performance on algorithmic state-tracking tasks.
2. **Prohibitive cost of dense matrices**: Unstructured transition matrices (SD-SSM) can simulate arbitrary FSAs, but the parallel scan complexity is $\Theta(LN^3)$, making large-scale training infeasible.
3. **Insufficiently compact semi-structured methods**: DPLR (diagonal-plus-low-rank) approaches enhance expressiveness but require either many stacked layers or exponentially large linear readout layers to simulate complex automata.
4. **Fundamental tension between efficiency and expressiveness**: This is the central challenge in SSM research—how to improve state-tracking capacity without sacrificing computational efficiency.
5. **Theoretical limitations of Transformers on FSA tasks**: Transformers face theoretical bottlenecks in automaton state tracking, motivating hybrid architectures with more expressive SSM components.
6. **Demand for hybrid Transformer–SSM architectures**: Large-scale LLMs increasingly adopt hybrid designs, requiring SSM layers that are both efficient and expressive.

### Starting Point

**Goal**: The PD parameterization decomposes the transition matrix as $A(u_t) = P(u_t)D(u_t)$:

- **P matrix** (column-wise one-hot binary matrix): Generated via soft selection over a learnable dictionary $\{M_k\}_{k=1}^K$ followed by column-wise hardmax.

## Method

### PD Parameterization

The transition matrix is factorized as $A(u_t) = P(u_t) D(u_t)$:

- **P matrix** (column-wise one-hot binary matrix): A learnable dictionary $\{M_k\}_{k=1}^K$ is combined via softmax-derived mixing weights $s(u_t)$ computed from the input $u_t$; the weighted sum is then projected column-wise via hardmax to obtain a sparse P.
- **D matrix** (complex diagonal matrix): Two separate feed-forward networks generate the magnitude (constrained to $(0,1)$ via sigmoid) and the phase (mapped to $(0, 2\pi)$), ensuring BIBO stability.

### Key Algebraic Properties

- The set of PD matrices is closed under matrix multiplication (forming a monoid), so any chain product of such matrices remains a PD matrix.
- Matrix multiplication can be performed in $\Theta(N)$ operations via gather-scatter, enabling efficient parallel scans.

### Gradient Approximation

Since the gradient of hardmax is almost everywhere zero, a straight-through estimator is employed: softmax serves as the surrogate gradient during backpropagation while hardmax is retained in the forward pass to preserve sparsity.

### Theoretical Guarantees

- **BIBO Stability** (Proposition 1): The system is bounded whenever the moduli of the D matrix entries are less than 1.
- **Universal Expressiveness** (Proposition 2): A single-layer PD-SSM with $N$-dimensional state and an $N \times N$ readout can exactly represent any $N$-state FSA.
- **Optimality** (Proposition 3): Any $N$-state FSA requires at least $N-1$ state dimensions; PD-SSM approaches this lower bound.

## Key Experimental Results

### Table 1: Length Generalization Accuracy on FSA Simulation Tasks (%)

### Main Results

| Model | Cycle Nav. | Even Pairs | Mod Arith. | Parity | Avg. |
|---|---|---|---|---|---|
| Transformer | 24.4% | 90.4% | 23.6% | 52.2% | 47.7% |
| Mamba | 48.4% | 100.0 | 33.1% | 54.2% | 58.9% |
| Gated DeltaProduct | 46.3% | 100.0 | 78.4% | 98.0% | 80.7% |
| BD-SLiCE | 99.8% | 85.9% | 54.0% | 95.3% | 83.8% |
| D-DE-SLiCE | 73.3% | 84.8% | 98.4% | 83.8% | 85.1% |
| **PD-SSM** | **99.5%** | **99.7%** | **96.2%** | **99.9%** | **98.8%** |

### Table 2: UEA Multivariate Time-Series Classification Accuracy (%)

### Ablation Study

| Model | Worms | SCP1 | SCP2 | Ethanol | Heartbeat | Motor | Avg. |
|---|---|---|---|---|---|---|---|
| Log-NCDE | 85.6% | 83.1% | 53.7% | 34.4% | 75.2% | 53.7% | 64.3% |
| LRU | 87.8% | 82.6% | 51.2% | 21.5% | 78.4% | 48.4% | 61.7% |
| Mamba | 70.9% | 80.7% | 48.2% | 27.9% | 76.2% | 47.7% | 58.6% |
| LinOSS-IM | 95.0% | 87.8% | 58.2% | 29.9% | 75.8% | 60.0% | 67.8% |
| **PD-SSM** | **90.0%** | **80.9%** | **56.1%** | **34.7%** | **80.0%** | **60.0%** | **67.0%** |

### Key Findings
- The principal components and modules contribute the most critical performance gains.

## Highlights & Insights

- **Theoretical elegance**: The PD parameterization simultaneously achieves BIBO stability, universal FSA simulation, and optimal state dimensionality—three theoretical guarantees in a single framework.
- **Efficiency breakthrough**: PD-SSM achieves a 71× speedup over dense SSMs (at $D=5632$), with parallel scan complexity identical to diagonal SSMs at $\Theta(LN)$.
- **Qualitative leap in FSA simulation**: The model achieves an average of 98.8% across four standard FSA tasks, significantly outperforming all existing parallelizable SSMs.
- **Perfect performance on non-solvable groups**: Multiple configurations attain 100% accuracy on $A_5$ and $S_5$ groups, in full agreement with theoretical predictions.
- **Hybrid architecture validation**: Attaching a single PD-SSM layer to a frozen Qwen-2.5 1.5B backbone enables successful generalization on natural-language-encoded FSA state-tracking, demonstrating its potential as a hybrid architecture component.
- **Complementarity of P and D**: P encodes arbitrary permutations while D compactly encodes cyclic groups; together they are both universal and efficient.

## Limitations & Future Work

- **Overhead in P matrix generation**: Although the parallel scan complexity matches that of diagonal SSMs, generating $P(u_t)$ involves dictionary lookup and hardmax operations, resulting in a practical runtime approximately 7× that of diagonal models.
- **Absence of large-scale language modeling validation**: Experiments focus primarily on synthetic tasks and time-series classification; the effectiveness of PD-SSM in large-scale LLM pretraining remains unverified.
- **Limited theoretical guarantees for the surrogate gradient**: The hardmax-to-softmax gradient approximation is heuristic, lacking rigorous convergence analysis.
- **Dictionary size $K$ as a manual hyperparameter**: Optimal values differ across tasks ($K=32$ for FSA, $K=6$ for LRA), with no automatic selection mechanism.
- **Time-varying SSMs underperform time-invariant models on LRA**: On the Long Range Arena benchmark, PD-SSM achieves the best result among time-varying models but still trails time-invariant SSMs such as S4 and LRU by approximately 10 points.

## Related Work & Insights

- **Diagonal SSMs**: S4 (Gu et al., 2022), LRU (Orvieto et al., 2023), and Mamba (Gu & Dao, 2023) achieve computational efficiency via diagonal transition matrices.
- **DPLR-structured SSMs**: DeltaNet (Schlag et al., 2021), GLA (Yang et al., 2024), RWKV-7 (Peng et al., 2025), and DeltaProduct (Siems et al., 2025) augment expressiveness with diagonal-plus-low-rank structures.
- **Unstructured SSMs**: SD-SSM (Terzić et al., 2025) uses dense matrices for complete FSA simulation at the cost of $O(LN^3)$ complexity.
- **FSA and neural networks**: Merrill & Sabharwal (2024) prove that diagonal SSMs cannot simulate non-solvable automata; Sarrof et al. (2024) construct complex diagonal SSMs for solvable automata.
- **Hybrid architectures**: Griffin (De et al., 2024), Jamba (Lenz et al., 2025), and TransXSSM (Wu et al., 2025) combine Transformers with SSMs.
- **Neural CDEs**: Log-NCDE (Walker et al., 2024) and LinOSS (Rusch et al., 2025) are differential-equation-based methods for time-series analysis.

## Rating

| Dimension | Score | Remarks |
|---|---|---|
| Novelty | 9/10 | PD parameterization is an entirely new matrix structure design, elegantly combining sparsity with algebraic closure |
| Theoretical Depth | 9/10 | Complete proofs of stability, expressiveness, and optimality form a rigorous theoretical framework |
| Experimental Thoroughness | 8/10 | Covers synthetic FSA, time-series classification, LRA, and natural-language tasks, but lacks large-scale LM experiments |
| Writing Quality | 9/10 | Clear exposition with well-organized theory and experiments; high-quality figures and tables |
| Value | 7/10 | Well-suited for scenarios requiring precise state tracking, though large-scale practical utility remains to be demonstrated |
| **Overall** | **8.5/10** | A high-quality contribution combining theoretical elegance with empirical validation, with significant impact on SSM expressiveness research |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PASS: Path-Selective State Space Model for Event-Based Recognition](pass_path-selective_state_space_model_for_event-based_recognition.md)
- [\[NeurIPS 2025\] DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products](deltaproduct_improving_state-tracking_in_linear_rnns_via_householder_products.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [\[NeurIPS 2025\] Revisiting Bi-Linear State Transitions in Recurrent Neural Networks](revisiting_bi-linear_state_transitions_in_recurrent_neural_networks.md)
- [\[AAAI 2026\] KineST: A Kinematics-guided Spatiotemporal State Space Model for Human Motion Tracking from Sparse Signals](../../AAAI2026/video_understanding/kinest_a_kinematics-guided_spatiotemporal_state_space_model_for_human_motion_tra.md)

</div>

<!-- RELATED:END -->
