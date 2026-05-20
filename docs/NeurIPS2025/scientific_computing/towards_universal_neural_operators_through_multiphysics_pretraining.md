---
title: >-
  [Paper Note] Towards Universal Neural Operators through Multiphysics Pretraining
description: >-
  [NeurIPS 2025][Scientific Computing][neural operator] This paper proposes an adapter-based multiphysics pretraining framework for neural operators. By treating lifting/projection layers as problem-specific adapters and f…
tags:
  - "NeurIPS 2025"
  - "Scientific Computing"
  - "neural operator"
  - "transfer learning"
  - "PDE"
  - "multiphysics"
  - "foundation model"
date: 2026-05-08
content_hash: b37a121c2551c025
---

# Towards Universal Neural Operators through Multiphysics Pretraining

**Conference**: NeurIPS 2025
**arXiv**: [2511.10829](https://arxiv.org/abs/2511.10829)  
**Code**: To be confirmed  
**Area**: Scientific Computing / Neural Operators
**Keywords**: neural operator, transfer learning, PDE, multiphysics, foundation model

## TL;DR

This paper proposes an adapter-based multiphysics pretraining framework for neural operators. By treating lifting/projection layers as problem-specific adapters and freezing shared kernel integration operator layers, the framework enables transfer learning across PDE problems, substantially reducing fine-tuning cost while improving generalization.

## Background & Motivation

**Core Problem:** Neural operators (NOs) have been widely adopted for data-driven physical simulation, yet their training is expensive and each new PDE problem typically requires training an independent model from scratch. The central question is how to build a universal neural operator foundation model that, after pretraining on diverse PDE problems, can efficiently transfer to new ones.

**Limitations of Prior Work:**

**PINNs (Physics-Informed Neural Networks):** Require explicit PDE formulations and guarantee accuracy only at training grid points, offering limited generalization.

**Classical neural operators (FNO/DeepONet):** Although capable of approximating function-space mappings with discretization invariance, each problem is trained independently, precluding reuse of learned physical knowledge.

**Existing pretraining methods:** Most are restricted to specific equation types (e.g., steady-state equations) or perform parameter extrapolation within the same PDE family, lacking the ability to transfer across different physical regimes.

**CoDA-NO:** While codomain attention is introduced for multiphysics transfer, its pretraining and fine-tuning pipeline remains insufficiently flexible.

**Motivation:** Inspired by the adapter fine-tuning paradigm in large language models, this work treats the lifting and projection layers of a neural operator as lightweight adapters and the kernel integration layers (Fourier/Transformer layers) as a shared backbone. During pretraining, all parameters are jointly optimized across multiple physical problems; during fine-tuning, $\theta_\mathcal{F}$ is frozen and only the new problem's adapter $(\theta_{\mathcal{P}_{ft}}, \theta_{\mathcal{L}_{ft}})$ is updated. This design achieves: (a) cross-PDE knowledge transfer; (b) significantly reduced fine-tuning computation; (c) support for PDE problems with different sets of input functions.

## Method

### Overall Architecture

The core architecture follows the standard neural operator **Lifting → Operator Blocks → Projection** three-stage design, augmented with an **adapter-based multiphysics pretraining/fine-tuning** strategy:

- **Lifting layer $\mathcal{L}$:** Maps input functions $\mathbf{a} = \{a_1, \dots, a_{n\_in}\}$ to a high-dimensional latent space, with parameters $\theta_\mathcal{L} = \{A_\mathcal{L}, b_\mathcal{L}\}$.
- **Kernel integration operator layers $\mathcal{F}$:** Comprising $n_{\text{layers}}$ stacked integral kernel operator blocks, each computing $\mathcal{F}_t(x) = \sigma\left(A_t v_t(x) + \int_{D_i} \kappa_t(x,y) v_t(y) dy + b_t(x)\right)$.
- **Projection layer $\mathcal{P}$:** Projects the final latent representation back to the output function space.

**Key Idea:** Different PDE problems maintain their own lifting and projection layers (adapters), while the kernel integration layers are shared across all problems. Pretraining jointly optimizes all parameters $(\theta_{\mathcal{P}_1}, \dots, \theta_{\mathcal{P}_N}, \theta_\mathcal{F}, \theta_{\mathcal{L}_1}, \dots, \theta_{\mathcal{L}_N})$; fine-tuning freezes $\theta_\mathcal{F}$ and trains only the new problem's $(\theta_{\mathcal{P}_{ft}}, \theta_{\mathcal{L}_{ft}})$.

### Key Designs: Two Enhanced Architectures

To improve generalization of neural operators in transfer learning, the paper investigates two architectural modifications:

**1. Mamba-SSM-Enhanced FNO (MambaFNO):**

A Mamba state-space module $\mathcal{M}_\phi$ is inserted after the lifting layer to apply causal convolution over the lifted features:

$$\widetilde{v}_0(x,t) = (\mathcal{M}_\phi v_0)(x,t) = \sum_{\tau \leq t} K_\tau v_0(x, t-\tau)$$

where $K_\tau$ are learnable convolutional kernels. The Mamba module serves as a **latent-space preconditioner**: prior to entering the Fourier layers, it aligns the embedding with dominant dynamical modes (advection, diffusion, oscillation), reducing the spectral rank and variability of the input signal. This stabilizes the training of $\mathcal{F}_t \circ \mathcal{M}_\phi$ and enables more efficient transfer of pretrained representations during fine-tuning.

**2. Perceiver IO-Enhanced Neural Operator:**

The symmetric cross-attention mechanism from Perceiver IO is introduced:
- **Encoding stage:** The input is mapped via FNO to $K_1 = \text{FNO}_{K_1}(X)$ and $V_1 = \text{FNO}_{V_1}(X)$, which are cross-attended with learnable latent variables $Q_1 = L$.
- **Processing stage:** Self-attention over the latent representation.
- **Decoding stage:** Cross-attention between input queries and the transformed latent representation to produce outputs.

The advantage of Perceiver lies in encoding information with a compact latent array, operating on more abstract feature representations with a controllable parameter count.

Additionally, the paper benchmarks **Codomain Attention (CoDA-NO)** and **Swin-v2 Transformer** as baselines. Codomain attention computes dot-product similarity along the feature dimension (rather than the sample dimension), which is better suited to the neural operator setting.

### Loss & Training

Range-normalized mean absolute error (NMAE) is adopted as both the training objective and evaluation metric:

$$\text{NMAE}(\theta) = \frac{1}{|\mathcal{D}^{\text{test}}|} \sum_{(\mathbf{a},u) \in \mathcal{D}^{\text{test}}} \frac{\|\mathcal{G}_\theta(\mathbf{a}) - u\|_{1,G}}{\max_G u - \min_G u + \varepsilon}$$

Normalization by the output value range eliminates the influence of different physical magnitudes, enabling fair comparison across physical regimes.

## Key Experimental Results

### Experiment 1: Out-of-Sample Parameter Values

Comparison of pretrain-then-finetune vs. training from scratch on Burgers' equation, Gray–Scott reaction-diffusion, and incompressible Navier–Stokes:

| Model | MSE | NMAE (%) | Avg. Epoch Time (s) | Parameters |
|-------|-----|----------|---------------------|------------|
| MambaFNO (pretrained) | 1.009×10⁻⁷ | 0.0120 | 21.91 | ~10⁷ |
| MambaFNO (from scratch) | 1.193×10⁻⁷ | 0.0213 | 40.14 | ~10⁷ |
| Perceiver (pretrained) | 1.425×10⁻⁷ | 0.0169 | 3.21 | ~10⁸ |
| Perceiver (from scratch) | 1.981×10⁻⁷ | 0.0219 | 204.73 | ~10⁸ |
| FNO (from scratch) | 1.774×10⁻⁷ | 0.0204 | 7.44 | ~10⁶ |
| Swin-v2 (pretrained+scratch) | 4.391×10⁻⁸ | 0.0092 | 101.3 | ~10⁹ |
| CoDA-NO (pretrained) | 2.881×10⁻⁷ | 0.0343 | 62.91 | ~10⁸ |
| CoDA-NO (from scratch) | 4.912×10⁻⁷ | 0.0712 | 63.29 | ~10⁸ |

**Key Findings:** Pretraining consistently outperforms training from scratch across all architectures. The fine-tuning speedup for Perceiver is most dramatic (from 205 s to 3.2 s/epoch, approximately **64× acceleration**). MambaFNO achieves roughly 44% reduction in NMAE after pretraining.

### Experiment 2: Input Extension + Cross-Physics Transfer

Transfer from advection/Burgers equations to reaction-diffusion (PDEBench dataset), and extension of the heat equation to convection-diffusion:

| Model | MSE | NMAE (%) | Avg. Epoch Time (s) |
|-------|-----|----------|---------------------|
| MambaFNO (pretrained) | 3.91×10⁻⁶ | 0.0041 | 131.2 |
| MambaFNO (from scratch) | 4.291×10⁻⁶ | 0.0054 | 261.1 |
| Perceiver (pretrained) | 4.107×10⁻⁶ | 0.0051 | 20.4 |
| Perceiver (from scratch) | 6.315×10⁻⁶ | 0.0074 | 804.0 |
| FNO (from scratch) | 7.286×10⁻⁶ | 0.0121 | 41.3 |
| CoDA-NO (pretrained) | 1.043×10⁻⁵ | 0.013 | 185.1 |
| CoDA-NO (from scratch) | 1.239×10⁻⁵ | 0.018 | 181.9 |

**Key Findings:** Cross-physics transfer is more challenging, yet pretraining remains consistently beneficial. Perceiver achieves approximately **39× fine-tuning acceleration** (804 s → 20.4 s/epoch) alongside a 31% reduction in NMAE. MambaFNO improves in both accuracy and speed, with per-epoch training roughly 2× faster than training from scratch.

## Highlights & Insights

1. **Elegance of the adapter paradigm:** The analogy between lifting/projection layers and LLM adapters, and between kernel integration layers and a pretrained backbone, is natural and effective — the design is concise and low-cost to implement.
2. **Mamba as a latent-space preconditioner:** Aligning embeddings with universal dynamical modes via causal convolution to reduce spectral variability is a design principle worth extending to other PDE transfer learning settings.
3. **Extreme acceleration from Perceiver:** The 39–64× fine-tuning speedup demonstrates that latent variable representations can effectively compress information, enabling adaptation to new problems by updating only a small number of adapter parameters.
4. **Purely data-driven evaluation:** The deliberate exclusion of physical information (i.e., no PINNs) focuses the evaluation on the neural operator's ability to learn universal dynamics from data, yielding a clean experimental design.

## Limitations & Future Work

1. **Spatial dimensionality constraint:** All experiments involve PDEs of the same spatial dimension; cross-dimensional transfer (e.g., 1D→2D or 2D→3D) has not been validated.
2. **Limited dataset scale and diversity:** The physical scenarios covered remain relatively narrow (advection, Burgers, reaction-diffusion, NS); more complex multiphysics coupling problems are not tested.
3. **Absence of comparison with recent foundation models:** No direct comparison with recent work such as POSEIDON, making it difficult to assess competitiveness in large-scale realistic settings.
4. **Mesh-agnosticism not thoroughly investigated:** Although the paper claims mesh-agnostic behavior, it is unclear whether a unified resolution was used across different PDEs in practice.
5. **Parameter scale imbalance for Swin-v2:** At ~10⁹ parameters, Swin-v2 is not on the same order of magnitude as other methods, raising concerns about the fairness of comparison.

## Related Work & Insights

- **POSEIDON** [Herde et al., 2024]: Hierarchical vision Transformer with shifted windows for transfer between Euler and NS equations; a key reference point for this work.
- **CoDA-NO** [Rahman et al., 2024]: Codomain attention computes similarity along the feature dimension; used as a baseline in this paper.
- **DeepONet Transfer Learning** [Goswami et al., 2022]: Operator transfer under covariate shift, establishing the theoretical feasibility of NO transfer.
- **PDEBench** [Takamoto et al., 2022]: Provides standardized PDE benchmark datasets; the cross-physics experiments in this paper are based on it.

**Insights:** The adapter-based multitask pretraining strategy is generalizable to broader scientific computing settings. Promising future directions include: (a) data augmentation based on Lie symmetries; (b) incorporating physical priors as regularization rather than hard constraints; (c) training a genuine foundation model on larger-scale heterogeneous datasets.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐ |
| Overall | ⭐⭐⭐ |

**Summary:** The paper presents a clear methodology and well-designed experiments, and the adapter-based pretraining and fine-tuning paradigm is demonstrated to be effective for PDE transfer learning. However, the overall contribution leans more toward engineering validation than methodological innovation — the core idea (adapter decoupling + shared backbone) is a direct application of a mature NLP paradigm to scientific computing, with limited novelty. The experimental scale is modest, and the work falls considerably short of the "Universal" ambition stated in the title.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Hamiltonian Neural PDE Solvers through Functional Approximation](hamiltonian_neural_pde_solvers_through_functional_approximation.md)
- [\[NeurIPS 2025\] DeltaPhi: Physical States Residual Learning for Neural Operators in Data-Limited PDE Solving](deltaphi_physical_states_residual_learning_for_neural_operators_in_data-limited_.md)
- [\[NeurIPS 2025\] From Black Hole to Galaxy: Neural Operator Framework for Accretion and Feedback Dynamics](from_black_hole_to_galaxy_neural_operator_framework_for_accretion_and_feedback_d.md)
- [\[NeurIPS 2025\] Stable Minima of ReLU Neural Networks Suffer from the Curse of Dimensionality: The Neural Shattering Phenomenon](stable_minima_of_relu_neural_networks_suffer_from_the_curse_of_dimensionality_th.md)
- [\[ICLR 2026\] Supervised Metric Regularization Through Alternating Optimization for Multi-Regime PINNs](../../ICLR2026/scientific_computing/supervised_metric_regularization_through_alternating_optimization_for_multi-regi.md)

</div>

<!-- RELATED:END -->
