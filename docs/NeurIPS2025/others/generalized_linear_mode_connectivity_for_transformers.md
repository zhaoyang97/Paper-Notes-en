---
title: >-
  [Paper Note] Generalized Linear Mode Connectivity for Transformers
description: >-
  [NeurIPS 2025][linear mode connectivity] This paper proposes a unified symmetry framework (a four-level hierarchy of permutation, semi-permutation, orthogonal, and invertible transformations) to achieve zero or near-zero barrier linear mode connectivity (LMC) on Vision Transformers and GPT-2 for the first time, and further extends the framework to multi-model merging and heterogeneous-width alignment.
tags:
  - NeurIPS 2025
  - linear mode connectivity
  - model merging
  - permutation symmetry
  - orthogonal symmetry
  - ViT
  - GPT-2
date: 2026-05-08
content_hash: 8f4328b5c1d6ee13
---

# Generalized Linear Mode Connectivity for Transformers

**Conference**: NeurIPS 2025
**arXiv**: [2506.22712](https://arxiv.org/abs/2506.22712)
**Code**: Available (link provided in the paper)
**Area**: Model Merging / Transformer Theory
**Keywords**: linear mode connectivity, model merging, permutation symmetry, orthogonal symmetry, ViT, GPT-2
**Authors**: Alexander Theus, Alessandro Cabodi, Sotiris Anagnostidis, Antonio Orvieto, Sidak Pal Singh, Valentina Boeva (ETH Zürich, MPI, et al.)

## TL;DR

This paper proposes a unified symmetry framework (a four-level hierarchy of permutation, semi-permutation, orthogonal, and invertible transformations) to achieve zero or near-zero barrier linear mode connectivity (LMC) on Vision Transformers and GPT-2 for the first time, and further extends the framework to multi-model merging and heterogeneous-width alignment.

## Background & Motivation

**Linear Mode Connectivity (LMC)** refers to the phenomenon where independently trained neural networks can be connected via linear interpolation in parameter space without a significant increase in loss along the path. This is formally characterized by the interpolation barrier:

$$\mathcal{B}[\theta_A, \theta_B] = \sup_{\lambda \in [0,1]} \left[ \mathcal{L}[\lambda \theta_A + (1-\lambda)\theta_B] - (\lambda \mathcal{L}[\theta_A] + (1-\lambda)\mathcal{L}[\theta_B]) \right]$$

When $\mathcal{B} \approx 0$, the models are said to be linearly mode connected. LMC has important implications for model merging, federated learning, and ensemble learning.

**Limitations of Prior Work**:

1. Methods such as Git Re-Basin primarily exploit **discrete permutation symmetry** (neuron reordering) to align models.
2. These approaches have succeeded on MLPs and CNNs (VGG, ResNet), but show limited effectiveness on Transformers.
3. Transformer components—including multi-head attention, LayerNorm/RMSNorm, and QK/OV circuits—admit richer symmetries beyond permutation.
4. Relying solely on permutation symmetry still leaves a non-negligible interpolation barrier in Transformers.
5. Git Re-Basin requires a **32× width increase** on ResNet-20 (CIFAR-10) to reach zero barrier, limiting its practical applicability.

**Core Problem**: How can all symmetries present in Transformer architectures be systematically exploited to achieve LMC?

## Method

### 1. Four-Level Symmetry Hierarchy

The paper organizes network symmetries into a strictly nested four-level structure $\mathcal{S}_1 \subset \mathcal{S}_2 \subset \mathcal{S}_3 \subset \mathcal{S}_4$:

| Level | Class | Transformation Structure | Applicable Scenarios |
|-------|-------|--------------------------|----------------------|
| $\mathcal{S}_1$ | Permutation | Permutation matrices | GELU, sigmoid, softmax, tanh |
| $\mathcal{S}_2$ | Semi-permutation | Sparse random matrices | ReLU, LayerNorm, Multi-Head Attention |
| $\mathcal{S}_3$ | Orthogonal | Orthogonal matrices | RMSNorm |
| $\mathcal{S}_4$ | Invertible | Full-rank matrices | Attention QK/OV circuits |

- **Permutation symmetry**: Element-wise activation functions (e.g., GELU) depend only on the corresponding input position, permitting neuron reordering.
- **Semi-permutation symmetry**: Piecewise linear functions (e.g., ReLU) satisfy $f(x) = f(\alpha x) + f((1-\alpha)x)$, allowing sparse weighted mixing; in MHA, each head contributes independently via summation, and the OV component admits a linear decomposition ($\alpha$-weighted).
- **Orthogonal symmetry**: RMSNorm is invariant under orthogonal transformations, which preserve norms; LayerNorm can be reformulated as RMSNorm combined with a centering matrix $M$ and scale parameters.
- **Invertible symmetry**: Within the QK circuit ($W^Q(W^K)^\top$) and OV circuit ($W^V W^O$) of attention, any invertible transformation can be inserted and algebraically cancelled without affecting the output.

### 2. Component-Level Symmetry Analysis of Transformers

**Feed-forward layers**: $\text{FF}(\mathbf{x}) = W_2 \phi(W_1 \mathbf{x} + b_1) + b_2$
- When $\phi$ is GELU, only permutation symmetry applies.
- When $\phi$ is ReLU, the symmetry extends to the semi-permutation class.

**Multi-head Attention**:
- **Intra-head**: The QK and OV circuits possess invertible symmetry; any invertible transformation can be absorbed algebraically, and the paper directly uses the circuit products as canonical representations.
- **Inter-head**: Independent summation across heads yields permutation symmetry; the linear decomposability of OV yields semi-permutation symmetry.

**Residual Stream**:
- LayerNorm is reformulated as $\text{LayerNorm}(\mathbf{Z}) = \text{RMSNorm}(\mathbf{Z}\mathbf{M}) \cdot \text{diag}(\alpha)\sqrt{D} + \mathbf{1}_N \beta^\top$.
- After absorbing $M$ and the scale, the residual path admits orthogonal symmetry.
- The orthogonal matrix may be **rectangular** ($M \geq N$), enabling alignment across heterogeneous widths.

### 3. Three Alignment Strategies

**Weight Matching (data-free)**:
- FF layers: solved layer-by-layer via SOBLAP (sum of bilinear assignment problems), approximated with the Hungarian algorithm.
- Attention layers: a cost matrix is constructed from the Frobenius norm of QK/OV circuits, and head-level permutations are solved via linear assignment.
- Residual stream: closed-form solution to the Procrustes problem via SVD, $\mathbf{O} = U V^\top$.
- **Advantage**: Entirely data-free and computationally efficient.

**Activation Matching**:
- Models are aligned by comparing intermediate-layer activations on a shared dataset (following existing methods).

**Learned Matching (end-to-end optimization)**:
- Learnable implicit matrices $Z_{\text{FF}}, Z_H, Z_O$ are introduced.
- At each forward pass, they are projected onto the corresponding symmetry class: $P_{\text{FF}} = \text{ProjPerm}(Z_{\text{FF}})$ (Hungarian + STE); $O = \text{ProjOrth}(Z_O)$ (SVD, fully differentiable).
- Weight matching solutions are used as initialization (critical; identity initialization performs substantially worse).
- The interpolation coefficient is sampled as $\lambda \sim \mathcal{U}(0.4, 0.6)$; the task loss is computed on the interpolated model and backpropagated.

### 4. Multi-Model Merging

**Universe Matching**: Iteratively constructs a shared reference model $U^{(t)}$; all models are aligned to this reference and averaged, with convergence achieved over multiple rounds.

**Learned Refinement**: Extended to the M-way setting by sampling $\lambda \sim \text{Dirichlet}(\alpha \mathbf{1}_M)$ with $\alpha=0.1$, directly minimizing the multi-model barrier.

## Key Experimental Results

### Table 1: Two-Model Alignment Loss Barrier Comparison (lower is better)

| Method | CIFAR-10 | CIFAR-100 | Tiny ImageNet | Tiny Shakespeare | BookCorpus |
|--------|----------|-----------|---------------|------------------|------------|
| Vanilla averaging | 1.69±0.07 | 2.46±0.04 | 2.84±0.02 | 2.02±0.12 | 4.34±0.09 |
| Activation matching | 1.27±0.13 | 2.11±0.17 | 1.86±0.10 | 1.43±0.16 | 4.05±0.13 |
| Weight matching (Ours) | 0.36±0.01 | 0.69±0.21 | 0.47±0.04 | 0.34±0.01 | 1.56±0.02 |
| Learned matching (permutation only) | 0.45±0.02 | 0.53±0.07 | 0.29±0.02 | 0.63±0.17 | 1.60±0.04 |
| **Learned matching (Ours)** | **0.00±0.00** | **0.00±0.00** | **0.00±0.00** | **0.02±0.00** | **0.42±0.01** |

### Table 2: Model Architecture and Training Configuration

| Setting | ViT (CIFAR-10/100) | ViT (Tiny ImageNet) | GPT-2 (Tiny Shakespeare) | GPT-2 (BookCorpus) |
|---------|---------------------|---------------------|--------------------------|---------------------|
| Transformer layers | 6 | 8 | 6 | 6 |
| Attention heads | 8 | 8 | 4 | 8 |
| Embedding dim | 256 | 384 | 256 | 512 |
| MLP hidden dim | 512 | 768 | 1024 | 2048 |
| Training epochs | 150 | 150 | 100 | 5 |
| Optimizer | AdamW | AdamW | AdamW | AdamW |
| Hardware | 1× RTX 2060 | 1× RTX 4090 | 1× RTX 4090 | 4× RTX 4090 |

## Highlights & Insights

1. **Strong theoretical contribution**: The paper is the first to systematically organize all Transformer symmetries into a hierarchical framework, unifying previously scattered alignment approaches.
2. **Milestone result**: Zero-barrier LMC is achieved on standard-sized ViT and GPT-2 for the first time, without any width expansion.
3. **Insightful gap analysis**: The performance gap between weight matching and learned matching is **concentrated entirely in the orthogonal matrix $O$ of the residual stream**; the corrections learned are small (eigen-angles cluster near $0 \bmod 2\pi$), and the permutation components remain largely unchanged—suggesting that better data-free orthogonal estimation could close most of the gap.
4. **Heterogeneous-width alignment**: Rectangular orthogonal matrices enable alignment between Transformers of different widths, with the interpolation path still maintaining zero or near-zero barrier.
5. **Natural extension to multi-model merging**: Visualization of the loss surface over the simplex for three models clearly shows that learned matching produces the widest region of near-zero barrier.
6. **Weight matching is already strong**: Even without access to training data, weight matching substantially outperforms activation matching and vanilla averaging, highlighting the critical role of exploiting orthogonal symmetry.

## Limitations & Future Work

1. **Small-scale language model experiments**: GPT-2 experiments use a reduced variant with only 6 layers and embedding dimension up to 512; scalability to full GPT-2 or LLaMA-scale models has not been verified.
2. **Non-zero barrier on BookCorpus (0.42)**: This may reflect fundamentally different generalization strategies learned by NLP models (e.g., lexical-overlap vs. syntactic cues) rather than insufficient alignment.
3. **Requires RMSNorm reparameterization**: Absorbing the centering component of LayerNorm alters the architectural representation (though functionally equivalent), which may be inconvenient for certain architectures.
4. **Restricted to same-task models**: The current framework only considers alignment of models trained on the same task; cross-task scenarios (e.g., in the spirit of ZipIt!) remain unexplored.
5. **Small-scale vision benchmarks**: Experiments are conducted on CIFAR-10/100 and Tiny ImageNet; validation on larger benchmarks such as ImageNet-1K is absent.
6. **Soft permutation potential not fully exploited**: Doubly stochastic relaxations can improve test-time performance, but a systematic study has not been conducted.

## Related Work & Insights

### Directly Related Work

- **Git Re-Basin** (Ainsworth et al.): Proposes weight matching, activation matching, and STE-based alignment; the unified framework in this paper subsumes these as special cases.
- **OT Fusion** (Singh & Jaggi): Employs optimal transport for neuron alignment; this paper extends the class of admissible symmetries.
- **SliceGPT** (Ashkboos et al.): Identifies orthogonal symmetry in the Transformer residual stream for pruning; this paper repurposes it for LMC.
- **Transformer Circuits** (Elhage et al.): Introduces the QK/OV circuit formalism; this paper exploits its invertible symmetry.

### Directions for Future Work

1. **Improved data-free orthogonal estimation**: Section 6 of the paper shows that the gap is primarily attributable to the $O$ matrix; better structural priors or spectral methods could approximate learned matching performance without data.
2. **Federated learning applications**: Weight matching requires no shared data while substantially reducing the barrier, making it naturally suited to federated settings.
3. **Model editing and continual learning**: If models trained at different stages reside in the same loss basin, knowledge can be combined via alignment followed by interpolation.
4. **LMC verification at large scale**: One of the most valuable directions for future work is validating generalized LMC on LLaMA-scale models.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ (First to unify four symmetry classes and achieve zero-barrier LMC on Transformers)
- **Theoretical Depth**: ⭐⭐⭐⭐⭐ (Rigorous hierarchical symmetry analysis integrating Procrustes, Hungarian, and SVD methods)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Ablations are thorough, but model scale is limited; gap analysis is a notable strength)
- **Value**: ⭐⭐⭐⭐ (Weight matching is data-free and efficient, directly applicable to federated learning and model merging)
- **Overall**: ⭐⭐⭐⭐⭐ (A milestone contribution that advances LMC from MLPs/CNNs to Transformers for the first time)

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Entropic Confinement and Mode Connectivity in Overparameterized Neural Networks](../../ICLR2026/others/entropic_confinement_and_mode_connectivity_in_overparameterized_neural_networks.md)
- [\[NeurIPS 2025\] Scalable Inference of Functional Neural Connectivity at Submillisecond Timescales](scalable_inference_of_functional_neural_connectivity_at_submillisecond_timescale.md)
- [\[NeurIPS 2025\] Impact of Layer Norm on Memorization and Generalization in Transformers](impact_of_layer_norm_on_memorization_and_generalization_in_transformers.md)
- [\[NeurIPS 2025\] Infrequent Exploration in Linear Bandits](infrequent_exploration_in_linear_bandits.md)
- [\[NeurIPS 2025\] A Generalized Label Shift Perspective for Cross-Domain Gaze Estimation](a_generalized_label_shift_perspective_for_crossdomain_gaze_e.md)

<!-- RELATED:END -->
