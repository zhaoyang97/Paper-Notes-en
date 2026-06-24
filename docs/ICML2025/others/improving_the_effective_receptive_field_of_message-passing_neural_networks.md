---
title: >-
  [Paper Note] Improving the Effective Receptive Field of Message-Passing Neural Networks
description: >-
  [ICML 2025][MPNN] This paper formalizes the concept of Effective Receptive Field (ERF) in MPNNs, proves that node contributions decay exponentially with distance (modeled as a binomial distribution), and proposes the IM-MPNN architecture. By utilizing multiscale graph coarsening and cross-scale information interleaving, IM-MPNN expands the ERF and achieves significant improvements on long-range dependency benchmarks such as LRGB.
tags:
  - "ICML 2025"
  - "MPNN"
  - "Over-squashing"
  - "Effective Receptive Field"
  - "Multiscale"
  - "Graph Coarsening"
date: 2026-05-08
content_hash: 490d115979646853
---

# Improving the Effective Receptive Field of Message-Passing Neural Networks

**Conference**: ICML 2025  
**arXiv**: [2505.23185](https://arxiv.org/abs/2505.23185)  
**Code**: [https://github.com/BGU-CS-VIL/IM-MPNN](https://github.com/BGU-CS-VIL/IM-MPNN)  
**Area**: LLM Evaluation  
**Keywords**: MPNN, Over-squashing, Effective Receptive Field, Multiscale, Graph Coarsening

## TL;DR
This paper formalizes the concept of Effective Receptive Field (ERF) in MPNNs, proves that node contributions decay exponentially with distance (modeled as a binomial distribution), and proposes the IM-MPNN architecture. By utilizing multiscale graph coarsening and cross-scale information interleaving, IM-MPNN expands the ERF and achieves significant improvements on long-range dependency benchmarks such as LRGB.

## Background & Motivation
**Background**: MPNNs update node representations via local message passing, but they are limited by over-squashing, making it difficult to aggregate long-range information effectively.

**Limitations of Prior Work**: Graph rewiring methods (e.g., SDRF) and Transformer-based methods increase the computational complexity of dense communication; most methods are constrained by the original graph resolution.

**Key Challenge**: How to enable MPNNs to effectively capture long-range dependencies without significantly increasing computational complexity?

**Key Insight**: Analogous to multiscale approaches in CNNs, graph coarsening can be leveraged to achieve long-range communication using only a few layers at coarser scales.

**Core Idea**: For a linear graph, after $\ell$ layers, node contributions follow a $B(\ell, 1/2)$ binomial distribution, which exhibits exponential decay; a coarser scale is equivalent to increasing the value of $\kappa$.

## Method

### Overall Architecture
Input graph → Encoding → Graclus coarsening $S$ times to obtain multiscale graphs → Parallel MP on $S+1$ scales → Scale-mix information interleaving → Repeat $L$ times → Unpooling + Concatenation → Task head.

### Key Designs

1. **ERF Theoretical Analysis**:

    - The feature of the central node on a linear graph after $\ell$ layers of uniform-weight convolution is: $x_0^\ell = \sum_{i=0}^\ell \binom{\ell}{i} v_{2i-\ell}$
    - After normalization, this corresponds to the binomial distribution $B(\ell, 1/2)$; the exponential decay of tail contributions can be derived from Hoeffding's inequality.
    - Continuous diffusion PDE analysis yields Gaussian kernel decay: $x(p,t) \propto \exp(-\|p-p_0\|^2 / 4\kappa t)$

2. **Multiscale Processing**:

    - The Laplacian on a coarser scale is equivalent to a larger $\kappa$ (coarsening factor of 2 $\to$ multiplying $\kappa$ by 4).
    - 1 hop on a coarser scale is equivalent to approximately 4 hops on the original graph, while preventing excessive information compression.

3. **Scale-Mix Layer**:

    - Each node receives information from its parent node (coarser scale) and child nodes (finer scale).
    - Different scales utilize MPNN backbones with independent parameters.

### Loss & Training
Task-related standard loss, which can be combined with different backbones such as GCN, GINE, or GatedGCN.

## Key Experimental Results

### Main Results

| Dataset | Metric | IM-GatedGCN | GatedGCN | Gain |
|--------|------|------------|----------|------|
| Peptides-func | AP↑ | 0.684 | 0.653 | +3.1% |
| Peptides-struct | MAE↓ | 0.244 | 0.256 | -4.7% |
| PascalVOC-SP | F1↑ | 0.397 | 0.367 | +3.0% |

### Ablation Study

| Configuration | Peptides-func AP | Description |
|------|-----------------|------|
| GCN baseline | 0.594 | No multiscale |
| IM-GCN S=1 | 0.623 | 1-level coarsening |
| IM-GCN S=2 | 0.645 | 2-level coarsening |
| IM-GCN S=3 | 0.659 | 3-level coarsening |

### Key Findings
- Increasing the number of scales $S$ continuously improves performance (visualized in Figure 1), indicating that multiscale processing effectively expands the ERF.
- IM-MPNN is a general framework that can boost the performance of various MPNN backbones.
- The computational efficiency is superior to fully-connected Transformer-based methods.

## Highlights & Insights
- The theoretical analysis of ERF is clear and elegant: a dual perspective from discrete binomial distributions to continuous diffusion equations.
- Translating multiscale methods from CNNs to GNNs is a natural yet well-validated idea.
- The framework is highly general and can serve as a plug-and-play enhancement for any MPNN.

## Limitations & Future Work
- Graclus coarsening is a fixed preprocessing step and is not learnable.
- For highly heterogeneous graph topologies, coarsening quality may be inconsistent.
- There is a lack of comprehensive comparison with the latest Graph Transformer methods.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel ERF analysis, incremental methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive LRGB benchmarks.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical motivation.
- Value: ⭐⭐⭐⭐ High practical value for addressing long-range dependency issues in GNNs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Permutation Equivariant Neural Networks for Symmetric Tensors](permutation_equivariant_neural_networks_for_symmetric_tensors.md)
- [\[ICML 2025\] Improved Exploration in GFlowNets via Enhanced Epistemic Neural Networks](improved_exploration_in_gflownets_via_enhanced_epistemic_neural_networks.md)
- [\[ICLR 2026\] Improving Set Function Approximation with Quasi-Arithmetic Neural Networks](../../ICLR2026/others/improving_set_function_approximation_with_quasi-arithmetic_neural_networks.md)
- [\[ICML 2025\] Truly Self-Improving Agents Require Intrinsic Metacognitive Learning](truly_self-improving_agents_require_intrinsic_metacognitive_learning.md)
- [\[CVPR 2025\] NeISF++: Neural Incident Stokes Field for Polarized Inverse Rendering of Conductors and Dielectrics](../../CVPR2025/others/neisf_neural_incident_stokes_field_for_polarized_inverse_rendering_of_conductors.md)

</div>

<!-- RELATED:END -->
