---
title: >-
  [Paper Note] Learning Sparse Approximate Inverse Preconditioners for Conjugate Gradient Solvers on GPUs
description: >-
  [NeurIPS 2025][Optimization][preconditioner] This paper proposes a graph neural network (GNN)-based method for learning sparse approximate inverse (SPAI) preconditioners…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "preconditioner"
  - "conjugate gradient method"
  - "graph neural network"
  - "GPU acceleration"
  - "sparse approximate inverse"
date: 2026-05-08
content_hash: 77f0f62ef0c3ea7a
---

# Learning Sparse Approximate Inverse Preconditioners for Conjugate Gradient Solvers on GPUs

**Conference**: NeurIPS 2025
**arXiv**: [2510.27517](https://arxiv.org/abs/2510.27517)
**Code**: None
**Area**: Optimization
**Keywords**: preconditioner, conjugate gradient method, graph neural network, GPU acceleration, sparse approximate inverse

## TL;DR

This paper proposes a graph neural network (GNN)-based method for learning sparse approximate inverse (SPAI) preconditioners, exploiting the natural compatibility between SPAI's locality and GNN message passing, and introduces a scale-invariant loss function (SAI loss) that achieves 40%–53% reduction in solve time (68%–113% speedup) on GPUs.

## Background & Motivation

**Background**: Solving symmetric positive definite (SPD) sparse linear systems $\mathbf{Ax} = \mathbf{b}$ is a core problem in scientific computing. The conjugate gradient (CG) method is the dominant iterative solver, but its convergence rate critically depends on preconditioning.

**Limitations of Prior Work**:
- Diagonal preconditioners: GPU-friendly but offer limited convergence improvement.
- Incomplete Cholesky (IC): performs well on CPUs but requires triangular solves that are difficult to parallelize on GPUs.
- Existing learning-based methods (GNN→IC): inherit the GPU parallelism bottleneck of triangular solves, and GNNs struggle to model long-range dependencies along the elimination tree.

**Key Challenge**: IC-type preconditioners yield fewer iterations but are GPU-inefficient (triangular solve bottleneck), while diagonal preconditioners are GPU-efficient but require many iterations.

**Goal**: Construct a preconditioner that both effectively reduces the condition number and fully exploits GPU parallelism.

**Key Insight**: The paper targets SPAI, using $\mathbf{M}^{-1} = \mathbf{GG}^\top + \varepsilon\mathbf{I}$ directly as the preconditioner. Each CG step requires only two sparse matrix–vector multiplications (SpMV), making it inherently GPU-friendly.

**Core Idea**: SPAI's locality (output at a node depends only on its two-hop neighborhood) is naturally compatible with GNN local message passing, making GNN-learned SPAI a more principled fit than GNN-learned IC.

## Method

### Overall Architecture

Nonzero entries and node features of input matrix $\mathbf{A}$ → GNN encoder–processor–decoder → output sparse matrix $\mathbf{G}$ → assemble $\mathbf{M}^{-1} = \mathbf{GG}^\top + \varepsilon\mathbf{I}$ → apply as preconditioner in CG.

### Key Designs

1. **Locality Analysis of SPAI Preconditioners**: Assuming $\mathbf{G}$ shares the sparsity pattern of $\mathbf{A}$, the preconditioned output $\mathbf{s}_j = (\mathbf{M}^{-1}\mathbf{r})_j$ depends only on the two-hop neighborhood of node $j$:
$$\mathbf{s}_j = \varepsilon\mathbf{r}_j + \sum_{l: \mathbf{A}_{jl}\neq 0} \mathbf{G}_{jl} \sum_{k: \mathbf{A}_{kl}\neq 0} \mathbf{G}_{kl}\mathbf{r}_k$$
This stands in sharp contrast to IC's global dependencies along the elimination tree, making SPAI naturally suited to GNN's local propagation mechanism.

2. **GNN Architecture**: An encoder–processor–decoder structure is employed, consisting of:

    - Encoder: two MLPs that separately encode node features $\mathbf{v}_i$ and edge features $\mathbf{e}_{ij}$ (i.e., $\mathbf{A}_{ij}$).
    - Processor: $L=4$ message-passing layers, each containing a message function $f_m^{(t)}$, node update $f_v^{(t)}$, and edge update $f_e^{(t)}$, all with residual connections.
    - Decoder: an MLP that maps the final edge features $\mathbf{h}_{ij}^{(L)}$ to $\mathbf{G}_{ij} \in \mathbb{R}^{b \times b}$.

   The total parameter count is only approximately 24k.

3. **Scale-invariant Alignment-to-Identity (SAI) Loss**:

    - Two issues with the conventional SPAI loss $\|\mathbf{AM}^{-1} - \mathbf{I}\|_F^2$: (1) directly computing $\mathbf{AM}^{-1}$ is prohibitively expensive for large matrices; (2) the loss depends on the absolute scale of $\mathbf{A}$, whereas CG convergence rates are scale-invariant.
    - **Stochastic estimation**: random vectors $\mathbf{w} \sim \mathcal{N}(0,1)$ are used to approximate the Frobenius norm, reducing matrix–matrix products to matrix–vector products.
    - **Scale normalization**:
$$\mathcal{L}_{\text{SAI}}(\mathbf{A}, \mathbf{M}^{-1}, \mathbf{w}) = \left\|\left(\frac{1}{\|\mathbf{A}\|}\mathbf{AM}^{-1} - \mathbf{I}\right)\mathbf{w}\right\|_2^2$$
    satisfying $\mathcal{L}_{\text{SAI}}(\mathbf{A}, \mathbf{M}^{-1}) = \mathcal{L}_{\text{SAI}}(\alpha\mathbf{A}, \mathbf{M}^{-1})$ for any $\alpha > 0$.
    - The norm is defined as $\|\mathbf{A}\| = \text{mean}_{\mathbf{A}_{ij}\neq 0}|\mathbf{A}_{ij}|$, which is more robust to matrix size and outliers.
    - Theoretical analysis shows this loss directly bounds the condition number: $\kappa(\mathbf{AM}^{-1}) \leq 1 + 2\|\mathbf{E}\|_2$.

### Loss & Training

- All experiments fix $L=4$, hidden dimension $d=24$, and $\varepsilon = 10^{-4}$.
- Training runs for 500 epochs with batch size 4, AdamW optimizer, and exponential learning rate decay (rate 0.99).
- Training is performed on a single NVIDIA A100 GPU.

## Key Experimental Results

### Main Results — GPU Solve Time ($\text{rtol} = 10^{-8}$)

| Problem | Diag | IC | AINV | **Ours** | Relative Speedup |
|---------|------|-----|------|----------|---------|
| Heat Equation | 77ms (520) | 167ms (204) | 78ms (330) | **36ms (197)** | **113%** |
| Poisson Equation | 45ms (320) | 101ms (128) | 58ms (217) | **26ms (128)** | **73%** |
| Hyperelasticity | 86ms (464) | 202ms (117) | 247ms (266) | **51ms (175)** | **68%** |
| Synthetic System | 445ms (2775) | 1399ms (1808) | 5024ms (10896) | **253ms (1122)** | **75%** |

Numbers in parentheses are CG iteration counts. The proposed method achieves the fastest solve time across all datasets.

### Ablation Study — Performance at Different Convergence Thresholds (Heat Equation, GPU)

| rtol | Diag | IC | AINV | **Ours** |
|------|------|-----|------|----------|
| $10^{-2}$ | 43ms (309) | 92ms (115) | 52ms (199) | **22ms (124)** |
| $10^{-4}$ | 53ms (384) | 114ms (143) | 61ms (246) | **27ms (154)** |
| $10^{-6}$ | 63ms (442) | 132ms (164) | 67ms (280) | **32ms (176)** |
| $10^{-8}$ | 77ms (520) | 167ms (204) | 78ms (330) | **36ms (197)** |

### Key Findings

- On GPUs, the proposed method achieves 68%–113% speedup.
- Iteration counts are comparable to IC (indicating high preconditioner quality), while per-step application cost is far lower than IC's triangular solves.
- Construction time (0.181ms) is on the same order as the diagonal preconditioner (0.196ms), far below IC (1.866ms) and AINV (18.924ms).
- The SAI loss yields a notable condition number improvement over the non-scale-invariant conventional loss.

## Highlights & Insights

- **Natural compatibility of SPAI and GNN**: SPAI's two-hop locality perfectly matches GNN message passing, making it a more principled pairing than IC-GNN.
- **Elegant design of SAI loss**: Scale invariance aligns the preconditioner optimization objective with CG convergence properties, without requiring precomputation of $\mathbf{A}^{-1}$.
- **Extremely lightweight model**: With only 24k parameters, the GNN demonstrates that local structural information suffices to guide high-quality preconditioner construction.
- **Full GPU pipeline**: The entire workflow—from construction (GNN forward pass) to application (SpMV)—executes on the GPU.

## Limitations & Future Work

- Applicable only to SPD matrices; not suitable for nonsymmetric or indefinite systems.
- Separate GNN training is required for different types of PDE problems.
- The sparsity pattern is fixed to match $\mathbf{A}$; adaptive sparsity pattern selection is not explored.
- Comparison with multigrid methods such as AMG is limited.
- Scalability to very large-scale problems (millions of degrees of freedom) remains to be verified.

## Related Work & Insights

- **Li et al., Häusner et al. (2023–2024)**: Pioneering work on GNN→IC preconditioners; this paper identifies the fundamental tension between triangular solves and GNN locality in those approaches.
- **Classical SPAI (Kolotilina et al.)**: This paper addresses the limitations of classical SPAI construction, which relies on sequential algorithms and yields suboptimal performance, through a data-driven approach.
- Insight: In scientific computing, "GPU-friendliness" may matter more than "theoretical optimality"—IC is theoretically superior yet slower in practice on GPUs.

## Rating

- Novelty: ⭐⭐⭐⭐ The GNN+SPAI combination is novel, and the SAI loss design has theoretical depth.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on multiple PDE problems and synthetic datasets, with comprehensive comparison against both classical and learning-based methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured; the comparative analysis of locality is particularly compelling.
- Value: ⭐⭐⭐⭐ High practical value for GPU-based scientific computing; the 68%–113% speedup is highly significant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Natural Gradient VI: Guarantees for Non-Conjugate Models](natural_gradient_vi_guarantees_for_non-conjugate_models.md)
- [\[NeurIPS 2025\] perturbation bounds for low-rank inverse approximations under noise](perturbation_bounds_for_low-rank_inverse_approximations_under_noise.md)
- [\[NeurIPS 2025\] Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry](projecting_assumptions_the_duality_between_sparse_autoencoders_and_concept_geome.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](learning_provably_improves_the_convergence_of_gradient_descent.md)
- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)

</div>

<!-- RELATED:END -->
