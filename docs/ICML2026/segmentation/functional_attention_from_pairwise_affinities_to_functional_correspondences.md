---
title: >-
  [Paper Note] Functional Attention: From Pairwise Affinities to Functional Correspondences
description: >-
  [ICML 2026][Segmentation][Operator learning] This paper reinterprets the softmax attention in Transformers as a "least-squares linear operator between two learnable functional bases." Borrowing the functional maps concep…
tags:
  - "ICML 2026"
  - "Segmentation"
  - "Operator learning"
  - "functional maps"
  - "spectral attention"
  - "PDE solving"
  - "geometric correspondence"
date: 2026-05-08
content_hash: 8eead3b06ec9e3ca
---

# Functional Attention: From Pairwise Affinities to Functional Correspondences

**Conference**: ICML 2026  
**arXiv**: [2605.31559](https://arxiv.org/abs/2605.31559)  
**Code**: https://github.com/xjffff/FUNCATTN  
**Area**: Scientific Computing / Operator Learning / Attention Mechanism  
**Keywords**: Operator learning, functional maps, spectral attention, PDE solving, geometric correspondence

## TL;DR
This paper reinterprets the softmax attention in Transformers as a "least-squares linear operator between two learnable functional bases." Borrowing the functional maps concept from shape matching, it compresses the $n\times n$ pairwise affinity matrix into a compact $k\times k$ spectral operator, achieving SOTA performance in PDE solving, 3D point cloud segmentation, and OOD generalization simultaneously.

## Background & Motivation
**Background**: Operator learning focuses on learning a mapping $\mathcal{O}:\mathcal{F}\to\mathcal{G}$ between two infinite-dimensional function spaces $\mathcal{F},\mathcal{G}$. There are two main directions: first, methods like FNO/U-NO/WMT that perform mappings in fixed spectral domains (Fourier, Wavelet, Laplacian bases); second, methods like Galerkin Transformer, OFormer, GNOT, and Transolver that discretize fields into tokens and apply attention.

**Limitations of Prior Work**: Fixed-basis methods rely on grid structures and degrade when encountering non-rectangular domains or irregular meshes. Tokenized attention is flexible, but either uses $O(n^2)$ dense affinities or approximates them in token space via linear attention. Both approaches treat "functions" as "sets of sampled points," ignoring the underlying global functional structure, leading to parameter redundancy and resolution sensitivity.

**Key Challenge**: The true function of attention is to induce a linear operator on the value space. However, current mainstream practices materialize this as an $n\times n$ point-level affinity matrix. This intermediate representation ties complexity to the number of tokens and causes a "functionally identical operator" to be redundantly expressed by infinitely many different point-level affinity matrices.

**Goal**: To skip pairwise affinities and directly estimate a compact linear operator $\mathbf{C}\in\mathbb{R}^{k\times k}$ between "learnable functional bases." This operator should be (i) discretization-invariant, (ii) numerically stable/provably Lipschitz, and (iii) compatible with both regular PDEs and point clouds/irregular geometries.

**Key Insight**: The functional maps framework (Ovsjanikov et al., 2012) from geometry processing provides a paradigm: the correspondence between two 3D shapes is not written as $n\times n$ point-wise matches, but as a small linear mapping between the spectral bases of two Laplace-Beltrami operators, solved via regularized least squares. This paper transfers this idea to attention: query and key-value each learn a set of adaptive bases, and attention is the optimal linear operator that pushes $\mathbf{K}$ into the $\mathbf{Q}$ spectral coordinates.

**Core Idea**: Replace the softmax affinity matrix with a $k\times k$ operator $\mathbf{C}^*=\tilde{\mathbf{Q}}\tilde{\mathbf{K}}^\top(\tilde{\mathbf{K}}\tilde{\mathbf{K}}^\top+\lambda\mathbf{I}_k)^{-1}$ solved via Tikhonov-regularized least squares, resulting in FuncAttn.

## Method

### Overall Architecture
FuncAttn follows the standard Transformer backbone: the input function $f$ is sampled at $n$ points to obtain $\mathbf{X}\in\mathbb{R}^{n\times d}$. This is encoded by an MLP, passed through $N$ FuncAttn blocks, and finally decoded back to the output field via another MLP. Inside each FuncAttn block, the traditional $\mathrm{Softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$ is replaced by a three-step pipeline: (a) project $\mathbf{Q},\mathbf{K},\mathbf{V}$ to spectral coordinates $\tilde{\mathbf{Q}},\tilde{\mathbf{K}},\tilde{\mathbf{V}}\in\mathbb{R}^{k\times d}$ using learned bases $\mathbf{\Phi},\mathbf{\Psi}\in\mathbb{R}^{n\times k}$; (b) solve for the compact operator $\mathbf{C}\in\mathbb{R}^{k\times k}$ in spectral space via Tikhonov regularized least squares; (c) project back to $n$ spatial points using $\mathbf{\Phi}\mathbf{C}\tilde{\mathbf{V}}$. Standard LayerNorm and FFN follow.

### Key Designs

1. **Basis Transform**:
    - **Function**: Compresses $n\times d$ discrete samples into $k\times d$ spectral coordinates without pre-defining fixed bases.
    - **Mechanism**: The query and key-value each learn a set of bases $\mathbf{\Phi},\mathbf{\Psi}$, constructed as $\mathcal{B}=\mathrm{Softmax}(\mathrm{Linear}(\mathbf{X}))\in\mathbb{R}^{n\times k}$, applying softmax along the $k$ dimension. The projection is written as $\tilde{\mathbf{Q}}=\mathbf{\Phi}^\dagger\mathbf{Q}$, implemented via the transpose $\mathbf{\Phi}^\top$ for stability. Proposition 4.3 shows that as temperature $\tau\to 0$, each basis function degrades to $\mathbf{1}_{\Lambda_j}$, recovering $P_0$ piecewise constant elements in classical finite element methods; at normal temperatures, it is equivalent to "learned soft partitions."
    - **Design Motivation**: Fixed Fourier bases require regular grids and are insensitive to geometric properties. Learned bases fit the geometry/physics of the input data, and softmax normalization naturally ensures partition-of-unity, preventing basis degradation. Tab. 7 confirms that freely learned bases outperform both "learned + orthogonal constraint" and "Fourier bases."

2. **Optimal Linear Transport**:
    - **Function**: Given $\tilde{\mathbf{Q}},\tilde{\mathbf{K}}$, solves for the $k\times k$ linear operator that best explains the transport from $\mathbf{K}\to\mathbf{Q}$.
    - **Mechanism**: Solves $\min_\mathbf{C}\|\tilde{\mathbf{Q}}-\mathbf{C}\tilde{\mathbf{K}}\|_F^2+\lambda\|\mathbf{C}\|_F^2$, yielding the closed-form solution $\mathbf{C}^*=\tilde{\mathbf{Q}}\tilde{\mathbf{K}}^\top(\tilde{\mathbf{K}}\tilde{\mathbf{K}}^\top+\lambda\mathbf{I}_k)^{-1}$. Substituting this back gives the full attention $\mathbf{\Phi}\mathbf{C}^*\tilde{\mathbf{V}}$. Unlike softmax, elements of $\mathbf{C}$ can be signed, implicitly providing "contrastive" capabilities.
    - **Design Motivation**: The core observation of functional maps is that point-level matching is complex, but in spectral bases, it becomes a convex, small-scale linear problem. The low-rank constraint $k\ll n$ reduces complexity and serves as implicit regularization for structured data generalization. The Tikhonov term $\lambda\|\mathbf{C}\|_F^2$ ensures numerical stability, with Proposition 4.5 proving a Lipschitz upper bound of $C_1/\lambda+C_2/\lambda^2$.

3. **Resolution-Invariant Transport**:
    - **Function**: Enables the same set of trained parameters to transfer across mesh resolutions.
    - **Mechanism**: Because $\mathbf{C}\in\mathbb{R}^{k\times k}$ is independent of $n$, and the spectral projection $\mathbf{\Phi}^\top\mathbf{Q}$ only performs linear projection on the $n$ dimension, the entire attention mechanism is naturally well-defined for different $n$. Training can be done on coarse grids (e.g., 1D Burgers $n=2048$) and tested on 8192 via direct forward pass without fine-tuning.
    - **Design Motivation**: The essence of operator learning should be to "learn the operator, not the sampling." Softmax attention ties the model to resolution. Compressing the correspondence to $k\times k$ makes resolution invariance a free architectural benefit.

### Loss & Training
The setup follows Transolver: single A40, repeated 3 times. The value of $k$ is tuned based on data complexity: $k\in[32,64]$ for smooth fields (Darcy / Pipe), and $k\in[128,256]$ for high-frequency fields (Elasticity / Navier-Stokes). The paper recommends a default of $k=64$.

## Key Experimental Results

### Main Results

Comparison across 6 PDE benchmarks against frequency-domain methods (FNO, WMT, etc.) and attention-based methods (Galerkin, Transolver, etc.) for a total of 14 baselines. Relative $L_2$ loss $\times 100$:

| Dataset | Galerkin | Transolver | LNO | **Ours** | vs Transolver |
|--------|----------|------------|------|---------|----------|
| Elasticity | 2.40 | 0.64 | 0.73 | **0.50** | -21.9% |
| Airfoil | 1.18 | 0.53 | 0.54 | **0.43** | -18.9% |
| Darcy | 0.84 | 0.57 | 0.60 | **0.42** | -26.3% |
| Pipe | 0.98 | 0.31 | **0.25** | 0.29 | -6.5% |
| Navier-Stokes | 14.01 | 9.44 | 8.45 | **8.00** | -15.3% |
| Plasticity | 1.20 | 0.13 | 0.31 | **0.11** | -15.4% |

SOTA was achieved on 5 out of 6 datasets, with relative gains between 6%–26% over Transolver. Other highlights: RNA 3D point cloud segmentation (xyz input) achieved 89.0% vs Transolver's 87.5%; AirfRANS OOD Reynolds showed a -27% relative lift error reduction, and OOD Angles showed -42%.

### Ablation Study

| Configuration | Elasticity | Darcy | Airfoil | Remarks |
|------|-----------|-------|---------|------|
| $k=16$ | 0.65 | 0.49 | 0.51 | Limited expression with too few bases |
| $k=32$ | 0.55 | 0.45 | 0.52 | Sufficient for smooth fields |
| $k=64$ | **0.50** | 0.42 | 0.43 | Recommended default |
| $k=128$ | 0.49 | 0.44 | **0.42** | Slightly better for high-frequency |
| $k=256$ | 0.48 | 0.43 | 0.47 | Diminishing returns |
| $k=512$ | 0.56 | 0.41 | 0.48 | Overfitting begins |
| Fourier basis | / | / | 0.51 | Fixed basis inferior to learned |
| Learnable + Ortho | / | / | 0.50 | Constraint hinders optimization |
| Learnable (Free) | / | / | **0.43** | Free learning is best |

### Key Findings
- $k$ follows a Goldilocks principle: exceeding a threshold leads to overfitting and increased cost; $k=64$ is the optimal performance/cost trade-off.
- Free learnable bases > orthogonal-constrained bases > fixed Fourier bases.
- Substituting FuncAttn with Fourier bases still outperforms all baselines (Airfoil 0.51 vs Galerkin 0.65), indicating that the advantage stems from the "spectral domain + least squares operator" paradigm itself.
- OOD gains (27%–42%) are significantly larger than in-distribution gains (6%–26%), suggesting function-level representations capture physical structures more effectively than token-level attention.

## Highlights & Insights
- **Paradigm Reinterpretation**: Instead of viewing attention as an "approximation of softmax," it is treated as a "linear operator estimation problem between two function spaces." 
- **Closed-form Least Squares**: This bridges the gap in the Galerkin Transformer's approach by explicitly solving for $\mathbf{C}$ as a least-squares solution.
- **Provable Stability**: Proposition 4.5 directly links the Lipschitz constant to Tikhonov $\lambda$, giving the "temperature-like" hyperparameter an explicit mathematical meaning.
- **Signed Affinity Capacity**: The elements of the solved $\mathbf{C}$ can be negative, providing the contrastive capability needed for fine-grained functional classification in 3D tasks.

## Limitations & Future Work
- The learned bases utilize only a single linear layer with softmax; more structured bases (attention/MLP-mixer/GNN) could be explored.
- Lack of formal analysis on approximation error and generalization bounds; current Lipschitz results address only stability.
- $\lambda$ is currently a fixed hyperparameter; data-adaptive $\lambda$ could be beneficial.
- Whether domains without natural function space interpretations (like NLP) can benefit remains an open question.

## Related Work & Insights
- **vs Galerkin Transformer**: Galerkin implicitly treats $\mathbf{Q}/\mathbf{K}/\mathbf{V}$ as Hilbert space functions, but attention remains an inner product; this work explicitly separates "functions" and "bases."
- **vs Transolver**: Transolver's "slice-and-attend" also learns partitions but uses standard softmax on slice tokens; this work replaces slicing with a spectral framework and attention with LS.
- **vs FNO Family**: FNO is restricted by regular grid and periodic boundary assumptions; this work uses learnable, non-grid spectral bases.
- **vs Functional Maps**: This work ports the functional maps framework into attention, replacing LB spectral bases with adaptive ones to enable end-to-end training.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CLIP Is Shortsighted: Paying Attention Beyond the First Sentence](../../CVPR2026/segmentation/clip_is_shortsighted_paying_attention_beyond_the_first_sentence.md)
- [\[NeurIPS 2025\] Attention (as Discrete-Time Markov) Chains](../../NeurIPS2025/segmentation/attention_as_discrete-time_markov_chains.md)
- [\[NeurIPS 2025\] Interpreting ResNet-based CLIP via Neuron-Attention Decomposition](../../NeurIPS2025/segmentation/interpreting_resnet-based_clip_via_neuron-attention_decomposition.md)
- [\[CVPR 2026\] MixerCSeg: An Efficient Mixer Architecture for Crack Segmentation via Decoupled Mamba Attention](../../CVPR2026/segmentation/mixercseg_an_efficient_mixer_architecture_for_crack_segmentation_via_decoupled_m.md)
- [\[AAAI 2026\] Target Refocusing via Attention Redistribution for Open-Vocabulary Semantic Segmentation: An Explainability Perspective](../../AAAI2026/segmentation/target_refocusing_via_attention_redistribution_for_open-vocabulary_semantic_segm.md)

</div>

<!-- RELATED:END -->
