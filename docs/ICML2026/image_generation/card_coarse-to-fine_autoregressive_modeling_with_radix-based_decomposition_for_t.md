---
title: >-
  [Paper Note] CARD: Coarse-to-fine Autoregressive Modeling with Radix-based Decomposition for Transferable Free Energy Estimation
description: >-
  [ICML 2026][Image Generation][Free Energy Estimation] CARD uses "radix $r$ decomposition" to bijectively map molecular 3D coordinates into a coarse-to-fine discrete-continuous mixed token sequence…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Free Energy Estimation"
  - "Autoregressive Transformer"
  - "Radix Decomposition"
  - "Zero Free Energy Proposal"
  - "BAR"
date: 2026-05-08
content_hash: 2a3e0ef7e3408c7a
---

# CARD: Coarse-to-fine Autoregressive Modeling with Radix-based Decomposition for Transferable Free Energy Estimation

**Conference**: ICML 2026  
**arXiv**: [2605.02657](https://arxiv.org/abs/2605.02657)  
**Code**: Public after publication  
**Area**: AI for Science / Molecular Modeling / Autoregressive Generation  
**Keywords**: Free Energy Estimation, Autoregressive Transformer, Radix Decomposition, Zero Free Energy Proposal, BAR

## TL;DR
CARD uses "radix $r$ decomposition" to bijectively map molecular 3D coordinates into a coarse-to-fine discrete-continuous mixed token sequence, enabling a cross-system transferable autoregressive Transformer to serve as a "zero free energy proposal" and directly estimate the absolute free energy of any molecular system via BAR. On solvation tasks for 70 novel systems, it matches the accuracy of classical MFES while being about 40 times faster in inference.

## Background & Motivation
**Background**: Free energy difference $\Delta F = -\beta^{-1} \log Z_b / Z_a$ is central to predicting binding affinity and solvation free energy in drug discovery. Classical approaches like Free Energy Perturbation (FEP) with alchemical intermediates and BAR/MBAR estimators are widely used but require massive MD simulations, resulting in extremely high computational costs.

**Limitations of Prior Work**: (1) Classical methods require extensive sampling of alchemical intermediates to ensure distribution overlap, taking hours to days per system; (2) Data-driven deep methods (e.g., protein-ligand affinity regression) generalize poorly and often fail on out-of-distribution systems; (3) "Zero free energy proposal" methods like DeepBAR use normalizing flows, but their expressiveness is limited by invertibility constraints and fixed input dimensions—requiring retraining for each new molecule.

**Key Challenge**: The ideal proposal model should (a) have analytically tractable probability density (to define $F_\theta = 0$), (b) be as expressive as diffusion/autoregressive models, and (c) generalize across systems. These three are mutually exclusive in existing frameworks (normalizing flow satisfies a but not b/c; diffusion satisfies b but not a; standard AR satisfies a/b but not c).

**Goal**: (1) Construct a generative model that can precisely compute log-density and generalize across systems; (2) Use it as a zero-free-energy proposal so that BAR can estimate the absolute free energy of any system in one step, eliminating the need for alchemical intermediates; (3) Validate zero-shot/few-shot performance on real-world tasks (solvation, endstate correction, tautomerization).

**Key Insight**: Inspired by LLMs ("autoregressive + Transformer + large-scale pretraining → cross-task generalization"), the authors convert 3D coordinates into token sequences for Transformer-based cross-molecule modeling. However, naively unfolding coordinates leads to a "chicken-and-egg" problem of local detail and global geometry interdependence—radix decomposition is proposed to enable a coarse-to-fine generation order.

**Core Idea**: Each coordinate is expanded in base-$r$ into $L$ discrete digits plus a continuous residual, and autoregressively generated in the order "all atoms' most significant digit → next digit → ... → least significant digit → continuous residual," thus determining global structure before refining details.

## Method

### Overall Architecture
CARD's workflow consists of four steps. **(1) Structure Alignment**: PCA is used to remove rotational/translational degrees of freedom, ensuring SE(3) equivariance, yielding $x \in \mathbb{R}^{N \times 3}$. **(2) Atom Ordering**: If topology is available, depth-first search with atom type priority (C→N→O→others→H) is used; otherwise, atoms are sorted by the variance of pairwise distances in a reference structure. **(3) Radix Decomposition**: Each coordinate component in $[0,1)$ is expanded as $\hat{x}_{ij} = (0.\hat{x}_{ij}^1 \hat{x}_{ij}^2 \cdots \hat{x}_{ij}^L \cdots)_r$, resulting in a mixed sequence of $N(L+1)$ tokens $s = (\hat{x}_1^1, ..., \hat{x}_N^1, \hat{x}_1^2, ..., \hat{x}_N^L, y_1, ..., y_N)$, with all atoms' most significant digits first. **(4) Encoder-Decoder Transformer**: The reference structure $u$ and atomic numbers $z$ are encoded into geometry-aware representations; the decoder outputs either a discrete digit (softmax over $r^3$ classes) or a continuous residual $y_i$ (Beta Mixture Model) at each step.

### Key Designs

1. **Radix-based Coordinate Decomposition (Coarse-to-fine Representation)**:

    - **Function**: Bijectively transforms continuous 3D coordinates into a mixed sequence where the first $L$ steps determine grid positions and the $(L+1)$-th step determines the residual, enabling AR to generate "global first, then local."
    - **Mechanism**: Choose a sufficiently large $a$ so that all coordinate components $|x_{ij}| < a/2$, normalize to $[0,1)$, and perform base-$r$ expansion. The first $L$ layers yield $\hat{x}_i^k \in \{0,...,r-1\}^3$ (each atom's sub-cube at level $k$), and the $(L+1)$-th layer gives the continuous residual $y_i \in [0, a/r^L)^3$. The authors prove this is a strict bijection, and the log-density decomposes as $\log q_\theta(x|c) = \sum_{i=1}^{N(L+1)} \log q_\theta(s_i | c, s_{:i})$ since the Jacobian is 1.
    - **Design Motivation**: Directly AR-generating continuous coordinates leads to a paradox where atom $i$'s precise position depends on yet-to-be-generated atom $j$. Coarse-to-fine ensures all atoms' rough spatial locations are set before refinement, analogous to multi-resolution image generation, allowing each step to access global structure.

2. **Geometry-Aware Attention**:

    - **Function**: Each transformer block leverages both "already generated coarse coordinates" and "reference structure distance matrix," making attention weights geometry-aware rather than purely textual.
    - **Mechanism**: Query $q_i = (\text{LN}(h_i + \varphi_1(x'_{i-N}))) W_1$ uses the same atom's previous level coordinate to avoid leaking current step information; key/value use $x'_j$ to access the latest geometry. The attention logit is augmented with a reference structure distance bias $\frac{1}{R}\sum_k \varphi_d^h(d_{ij}^{(k)})$, where $d_{ij}^{(k)} = \|u_{i'}^{(k)} - u_{j'}^{(k)}\|_2$ is the interatomic distance in $R$ reference structures.
    - **Design Motivation**: The fundamental difference between molecular and text modeling is that "position = physical coordinate." Allowing attention to directly access Euclidean distances greatly simplifies learning the physical prior that "distant atoms are independent, nearby atoms are strongly correlated."

3. **Beta Mixture Model for Continuous Residuals**:

    - **Function**: Outputs a bounded continuous distribution for each atom's final continuous token $y_i \in [0, a/r^L)^3$.
    - **Mechanism**: The Beta distribution is naturally defined on $[0,1]$; $y_i$ is scaled to $[0,1)$ and modeled as a weighted mixture of $K$ Beta components: $\text{BMM}(x; \Theta) = \sum_{k=1}^K \pi_k \text{Beta}(x; \alpha_k, \beta_k)$. The three components are modeled in chain rule order $y_{i1} \to y_{i2} \to y_{i3}$.
    - **Design Motivation**: Continuous coordinates cannot use Gaussian (not supported on $[0,1)$) or categorical (loss of precision). Beta mixture maintains closed interval support and enables precise log-density computation, strictly meeting the requirements of the zero-free-energy proposal paradigm.

### Loss & Training
Two-stage training. Stage I: pure NLL $\mathcal{L}_{\text{NLL}} = -\frac{1}{BN}\sum_b \log q_\theta(x^{(b)}|c)$. Stage II: joint optimization of NLL + energy alignment $\mathcal{L}_{\text{energy}} = \frac{1}{B}\sum_b |\tilde{U}_\theta^{(b)} - \tilde{U}^{(b)}|$ (mean-centered), using true force field energies to correct for sample imbalance/incomplete sampling. During inference, BAR is used to estimate free energy difference between $q_\theta$ and the target distribution; since $F_\theta = 0$, absolute free energy is obtained directly.

## Key Experimental Results

### Main Results
Three complementary tasks comprehensively validate generalization:

| Task | Dataset | Metric | CARD | Baseline |
|------|--------|------|------|---------|
| Vacuum→Toluene Solvation | ZINC20 70 testmol | MAE (kcal/mol) | <1, $R^2 > 0.9$ | MFES (ref) |
| Vacuum→Water Solvation | ZINC20 70 testmol | MAE (kcal/mol) | <1, $R^2 > 0.9$ | MFES (ref) |
| MM→NNP Endstate Correction | HiPen 18 mol | MAE (kcal/mol) | **0.90** | MFES (ref) |
| Aqueous Tautomerization | 27 tautomer pairs | MAE↓ | **4.11** | DFT 4.62 / sPhysNet 4.61 |
| Aqueous Tautomerization | 27 tautomer pairs | PCC↑ | **0.64** | DFT 0.36 / sPhysNet 0.35 |

### Ablation Study
Ablation on radix $r$, depth $L$, and training stage for vacuum→toluene solvation:

| Config | MAE↓ | RMSE↓ | $R^2$↑ | Pct(<1)↑ |
|------|------|-------|--------|----------|
| **$r=4, L=3$, Stage I+II (full)** | **0.71** | **1.27** | **0.92** | **82.9** |
| $r=4, L=3$, Stage I only | 0.81 | 1.34 | 0.91 | 77.1 |
| $r=3, L=3$, Stage I only | 2.43 | 3.08 | 0.61 | 26.5 |
| $r=5, L=3$, Stage I only | 1.88 | 2.41 | 0.73 | 22.1 |
| $r=4, L=2$, Stage I only | 5.85 | 14.26 | -0.08 | 17.1 |
| $r=4, L=4$, Stage I only | 1.43 | 2.39 | 0.77 | 61.4 |

### Key Findings
- **40x acceleration + cross-system generalization is a dual breakthrough**: On 70 molecules unseen during training, CARD's per-system inference is about 770 seconds vs MFES's 32,300 seconds, with comparable accuracy; this is unattainable for existing deep methods (which require per-system retraining).
- **$L=2$ collapses** ($R^2 = -0.08$): Coarse-to-fine requires sufficient layers for stable global structure; too shallow degenerates to "directly generating continuous coordinates."
- **$L=4$ slightly underperforms**: Too many layers make high-level $\hat{x}_i^k$'s $r^3$ classes hard to distinguish, reducing effective signal; optimal at $L=3$.
- **$r$ too small (3)** forces BMM to model overly broad residuals, exceeding expressiveness; **$r$ too large (5)** causes the discrete space to balloon, making training difficult. Optimal $r=4$ keeps $r^3=64$ classes in a softmax-friendly range.
- **Stage II energy alignment is crucial**: Reduces MAE from 0.81 to 0.71 (>10% relative improvement), indicating MD sampling bias requires force field label correction.
- On tautomerization, CARD outperforms DFT (B3LYP/6-31G*), since DFT approximates free energy with a single minimum-energy conformation, failing for complex flexible molecules; CARD performs true Boltzmann averaging.

## Highlights & Insights
- **Radix decomposition** is an elegant bridge: It transforms the high-dimensional continuous molecular generation problem into a "coarse-to-fine mixed token AR," leveraging the full Transformer toolchain (KV cache, scaling, cross-task transfer) while strictly maintaining tractable likelihood.
- **"Zero free energy proposal" + BAR** is the core paradigm—proposed by DeepBAR but limited by NF expressiveness; CARD uses AR + BMM to unlock expressiveness, advancing both theory and engineering.
- **Dual query/key split in geometry-aware attention** (query uses previous level coordinates to avoid leakage, key/value use latest coordinates) is worth adopting in any "stepwise geometric object generation" task, such as autoregressive protein folding or 3D mesh generation.
- **Cross-chemical environment (vacuum/toluene/water/NNP/tautomerization) performance is nearly lossless**, representing rare "true generalization" in AI for Chemistry, akin to LLMs' "one model, many tasks" paradigm for molecules.
- The overall approach (radix-based coarse-to-fine + tractable AR) is transferable to all-atom protein modeling, crystal structure generation, and even 3D point cloud generation.

## Limitations & Future Work
- **PCA alignment is unstable for symmetric molecules**: The authors acknowledge that near-symmetric principal axes can cause alignment direction to fluctuate, introducing high variance; more robust equivariant features (e.g., E(3)-equivariant networks) are needed.
- **Dataset mainly consists of drug-like small molecules** (ZINC20, atom count < 50): Protein-ligand complexes often have thousands of atoms; scalability to this regime remains untested.
- **Inference sequence length $\sim N(L+1)$ still grows linearly with atom count**: Empirically, inference complexity is nearly quadratic in $N$ (vanilla Transformer); FlashAttention/Linear Attention is needed for thousand-atom systems.
- **Sampling bias from MD trajectories during training** may propagate to the model; Stage II energy alignment partially mitigates but does not fully resolve this. Future directions: (a) E(3) equivariant CARD; (b) use NNP instead of force field energies for finer labels; (c) extend to protein-ligand docking.

## Related Work & Insights
- **vs DeepBAR (NeurIPS21)**: Pioneered the "zero free energy proposal" concept but used normalizing flow, limiting expressiveness and requiring per-system retraining; CARD uses AR Transformer + radix to unlock expressiveness and achieve cross-system generalization.
- **vs neural TI / FEAT**: These methods use diffusion to learn time-dependent Hamiltonians for thermodynamic integration, still following the alchemical paradigm; CARD directly uses BAR, skipping intermediates.
- **vs Boltzmann Generators (Noé 2019, Tan 2025)**: These aim for equilibrium sampling, but most cannot compute log-density exactly or generalize across systems; CARD achieves both likelihood and transferability.
- **vs MGVAE / MGT / Sequoia (multi-resolution molecular modeling)**: These operate at the graph level for coarse-to-fine, while CARD does so at the atomic coordinate level, maintaining exact AR factorization.
- **vs LLM's BPE tokenization**: Radix decomposition is akin to "BPE for 3D coordinates," inspiring similar tokenization for other continuous physical quantities (e.g., protein dihedrals, crystal lattice parameters) for Transformer-based modeling.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Strictly unifies the LLM AR paradigm with zero free energy proposal; radix decomposition is a truly original "3D→token" bridge.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three complementary tasks, detailed ablation, and inference speed comparisons; lacks protein-scale validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous formula derivations (bijection, Jacobian = 1, log-density decomposition), clear engineering details; a rare "theory + experiment strong" AI4Sci paper.
- Value: ⭐⭐⭐⭐⭐ Transforms per-system hours-long free energy calculations into cross-system, second-level, high-accuracy predictions—a disruptive engineering value for FEP-based drug screening pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](../../NeurIPS2025/medical_imaging/energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)
- [\[AAAI 2026\] Coarse-to-Fine Open-Set Graph Node Classification with Large Language Models](../../AAAI2026/medical_imaging/coarse-to-fine_open-set_graph_node_classification_with_large_language_models.md)
- [\[ICML 2026\] Factored Classifier-Free Guidance](factored_classifier-free_guidance.md)
- [\[AAAI 2026\] PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation](../../AAAI2026/medical_imaging/priorrg_prior-guided_contrastive_pre-training_and_coarse-to-fine_decoding_for_ch.md)
- [\[ICML 2026\] DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning](dp-kfc_data-free_preconditioning_for_privacy-preserving_deep_learning.md)

</div>

<!-- RELATED:END -->
