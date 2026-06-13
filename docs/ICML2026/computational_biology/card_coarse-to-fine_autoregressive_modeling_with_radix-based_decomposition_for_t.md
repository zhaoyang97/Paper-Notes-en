---
title: >-
  [Paper Note] CARD: Coarse-to-fine Autoregressive Modeling with Radix-based Decomposition for Transferable Free Energy Estimation
description: >-
  [ICML 2026][Computational Biology][Free Energy Estimation] CARD utilizes "radix $r$ decomposition" to bijectively map molecular 3D coordinates into a coarse-to-fine sequence of discrete-continuous mixed tokens. This allo…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "Free Energy Estimation"
  - "Autoregressive Transformer"
  - "Radix Decomposition"
  - "Zero-free-energy proposal"
  - "BAR"
date: 2026-05-08
content_hash: 14156b72d7a60eef
---

# CARD: Coarse-to-fine Autoregressive Modeling with Radix-based Decomposition for Transferable Free Energy Estimation

**Conference**: ICML 2026  
**arXiv**: [2605.02657](https://arxiv.org/abs/2605.02657)  
**Code**: To be released after publication  
**Area**: AI for Science / Molecular Modeling / Autoregressive Generation  
**Keywords**: Free Energy Estimation, Autoregressive Transformer, Radix Decomposition, Zero-free-energy proposal, BAR

## TL;DR
CARD utilizes "radix $r$ decomposition" to bijectively map molecular 3D coordinates into a coarse-to-fine sequence of discrete-continuous mixed tokens. This allows a cross-system universal autoregressive Transformer to serve as a "zero-free-energy proposal" to directly estimate the absolute free energy of arbitrary molecular systems via BAR. It achieves classical MFES precision on solvation tasks across 70 new systems with approximately 40x faster inference.

## Background & Motivation
**Background**: The free energy difference $\Delta F = -\beta^{-1} \log Z_b / Z_a$ is a core quantity for predicting binding affinity and solvation free energy in drug discovery. Classical approaches like Free Energy Perturbation (FEP) with alchemical intermediate states and BAR/MBAR estimators are widely used but require massive MD simulations and incur high computational costs.

**Limitations of Prior Work**: (1) Classical methods require sampling many alchemical intermediate states to ensure distribution overlap, taking hours to days per system; (2) Data-driven deep methods (e.g., protein-ligand affinity regression) generalize poorly and often fail on out-of-distribution systems; (3) "Zero-free-energy proposal" methods like DeepBAR use normalizing flows, but their expressivity is limited by invertibility constraints, and input dimensions are tied to specific systems, necessitating retraining for each new molecule.

**Key Challenge**: An ideal proposal model must simultaneously (a) have a tractable probability density to define $F_\theta = 0$, (b) be as expressive as diffusion or autoregressive models, and (c) generalize across systems. These three properties are mutually exclusive in existing frameworks (normalizing flows satisfy a but not b/c; diffusion satisfies b but not a; standard AR satisfies a/b but not c).

**Goal**: (1) Construct a generative model capable of precise log-density calculation and cross-system generalization; (2) Use it as a zero-free-energy proposal to allow BAR to estimate the absolute free energy of any system in one go, eliminating alchemical intermediates; (3) Validate zero-shot/few-shot performance on multiple real-world tasks (solvation, endstate correction, tautomerization).

**Key Insight**: The authors draw inspiration from the "Autoregressive + Transformer + Massive Pre-training $\rightarrow$ Cross-task Generalization" paradigm of LLMs, converting 3D coordinates into token sequences for cross-molecule modeling. However, simply unfolding coordinates encounters a "chicken-and-egg" problem where local details and global geometry are interdependent. To address this, radix decomposition is proposed to achieve a coarse-to-fine ordering.

**Core Idea**: Each coordinate is expanded into $L$ discrete digits plus a continuous residual using a base-$r$ expansion. Generation follows an autoregressive order: "highest-order digits of all atoms $\to$ next highest-order $\to \dots \to$ lowest-order $\to$ continuous residues," fixing global structure before filling in details.

## Method

### Overall Architecture
The CARD workflow consists of 4 steps. **(1) Structural Alignment**: PCA is used to remove rotational/translational degrees of freedom to ensure SE(3) equivalence $\rightarrow$ output $x \in \mathbb{R}^{N \times 3}$. **(2) Atom Ordering**: If topology exists, depth-first search + atom type priority (C→N→O→others→H) is used; otherwise, atoms are sorted by the variance of pairwise distances to a reference structure. **(3) Radix Decomposition**: Each coordinate component is expanded in $[0,1)$ as $\hat{x}_{ij} = (0.\hat{x}_{ij}^1 \hat{x}_{ij}^2 \cdots \hat{x}_{ij}^L \cdots)_r$, resulting in a mixed sequence of $N(L+1)$ tokens $s = (\hat{x}_1^1, ..., \hat{x}_N^1, \hat{x}_1^2, ..., \hat{x}_N^L, y_1, ..., y_N)$, prioritizing the highest level for all atoms. **(4) Encoder-Decoder Transformer**: The reference structure $u$ and atomic numbers $z$ are encoded into geometry-aware representations. The decoder outputs either discrete digits (softmax over $r^3$ classes) or continuous residues $y_i$ (Beta Mixture Model) at each step.

### Key Designs

1.  **Radix-based Coordinate Decomposition (Coarse-to-fine Representation)**:
    - **Function**: Bijectively converts continuous 3D coordinates into a mixed sequence where the first $L$ steps determine grid positions and the $(L+1)$-th step determines residuals, enabling AR to generate "global-to-local."
    - **Mechanism**: A constant $a$ is chosen such that all coordinates $|x_{ij}| < a/2$. After normalization to $[0,1)$, a base-$r$ expansion is performed. The first $L$ levels produce $\hat{x}_i^k \in \{0,...,r-1\}^3$ (indicating which $r^3$ sub-cube atom $i$ falls into at level $k$), and the $(L+1)$-th level provides continuous residuals $y_i \in [0, a/r^L)^3$. The authors prove this is a strict bijection where the log-density decomposes as $\log q_\theta(x|c) = \sum_{i=1}^{N(L+1)} \log q_\theta(s_i | c, s_{:i})$ because the Jacobian is 1.
    - **Design Motivation**: Directly generating continuous coordinates via AR leads to a paradox where the precise position of atom $i$ depends on atom $j$, which hasn't been generated yet. Coarse-to-fine allows all atoms to establish general spatial positions before refinement, analogous to multi-resolution image generation, ensuring each prediction step perceives the coarse global structure.

2.  **Geometry-Aware Attention**:
    - **Function**: Simultaneously utilizes "generated coarse coordinates" and "reference structure distance matrices" in each transformer block, making attention weights aware of geometry rather than just text.
    - **Mechanism**: The query $q_i = (\text{LN}(h_i + \varphi_1(x'_{i-N}))) W_1$ uses coordinates of the same atom from the previous level to avoid leaking current-step information. Key/value pairs use $x'_j$ to access the latest geometry. The attention logit is augmented with a reference structure distance bias $\frac{1}{R}\sum_k \varphi_d^h(d_{ij}^{(k)})$, where $d_{ij}^{(k)} = \|u_{i'}^{(k)} - u_{j'}^{(k)}\|_2$ represents inter-atomic distances across $R$ reference structures.
    - **Design Motivation**: The fundamental difference between molecular and text modeling is that "position = physical coordinates." Allowing attention to directly see Euclidean distances simplifies learning physical priors such as "long-range atoms are irrelevant, nearby atoms are strongly correlated."

3.  **Beta Mixture Model for Continuous Residuals**:
    - **Function**: Outputs a bounded continuous distribution for the final continuous token $y_i \in [0, a/r^L)^3$ of each atom.
    - **Mechanism**: Since the Beta distribution is naturally defined on $[0,1]$, $y_i$ is scaled to $[0,1)$ and modeled as a weighted mixture of $K$ Beta components: $\text{BMM}(x; \Theta) = \sum_{k=1}^K \pi_k \text{Beta}(x; \alpha_k, \beta_k)$. The three components are modeled via the chain rule order $y_{i1} \to y_{i2} \to y_{i3}$.
    - **Design Motivation**: Continuous coordinates cannot use Gaussian (no support on $[0,1)$) or categorical (loss of precision) distributions. Beta mixtures maintain a closed interval and allow for precise log-density calculation, aligning strictly with the requirements of the zero-free-energy proposal paradigm.

### Loss & Training
Two-stage training. Stage I: Pure NLL $\mathcal{L}_{\text{NLL}} = -\frac{1}{BN}\sum_b \log q_\theta(x^{(b)}|c)$. Stage II: Joint optimization of NLL and energy alignment $\mathcal{L}_{\text{energy}} = \frac{1}{B}\sum_b |\tilde{U}_\theta^{(b)} - \tilde{U}^{(b)}|$ (mean-centered), using ground-truth force field energy to correct for sample imbalance or incomplete sampling. During inference, BAR estimates the free energy difference between $q_\theta$ and the target distribution; since $F_\theta = 0$, the absolute free energy is obtained directly.

## Key Experimental Results

### Main Results
Generalization was verified across three complementary tasks:

| Task | Dataset | Metric | CARD | Baseline |
|------|--------|------|------|---------|
| Vacuum $\to$ Toluene Solvation | ZINC20 70 testmol | MAE (kcal/mol) | <1, $R^2 > 0.9$ | MFES (ref) |
| Vacuum $\to$ Water Solvation | ZINC20 70 testmol | MAE (kcal/mol) | <1, $R^2 > 0.9$ | MFES (ref) |
| MM $\to$ NNP Endstate Correction | HiPen 18 mol | MAE (kcal/mol) | **0.90** | MFES (ref) |
| Aqueous Tautomerization | 27 tautomer pairs | MAE↓ | **4.11** | DFT 4.62 / sPhysNet 4.61 |
| Aqueous Tautomerization | 27 tautomer pairs | PCC↑ | **0.64** | DFT 0.36 / sPhysNet 0.35 |

### Ablation Study
Ablations on radix $r$, depth $L$, and training stages for the Vacuum $\to$ Toluene solvation task:

| Configuration | MAE↓ | RMSE↓ | $R^2$↑ | Pct(<1)↑ |
|------|------|-------|--------|----------|
| **$r=4, L=3$, Stage I+II (full)** | **0.71** | **1.27** | **0.92** | **82.9** |
| $r=4, L=3$, Stage I only | 0.81 | 1.34 | 0.91 | 77.1 |
| $r=3, L=3$, Stage I only | 2.43 | 3.08 | 0.61 | 26.5 |
| $r=5, L=3$, Stage I only | 1.88 | 2.41 | 0.73 | 22.1 |
| $r=4, L=2$, Stage I only | 5.85 | 14.26 | -0.08 | 17.1 |
| $r=4, L=4$, Stage I only | 1.43 | 2.39 | 0.77 | 61.4 |

### Key Findings
- **40x Acceleration + Cross-system Generalization is a dual breakthrough**: On 70 molecules not seen in the training set, CARD's single-system inference takes ~770s vs. ~32,300s for MFES, with comparable accuracy. This is unattainable for existing deep methods that require per-system training.
- **$L=2$ failure** ($R^2 = -0.08$): This indicates that coarse-to-fine requires sufficient levels to stabilize the coarse structure; too shallow results in a collapse to "direct continuous coordinate generation."
- **$L=4$ slight degradation**: With too many levels, the $r^3$ categories at higher levels become difficult to distinguish, and the model fails to capture effective signals; $L=3$ is optimal.
- **Small $r$ (3)** forces BMM to model overly wide residuals, exceeding its expressivity; **large $r$ (5)** causes discrete space explosion, making training difficult. The optimal $r=4$ keeps $r^3=64$ classes within a "softmax-friendly" range.
- **Stage II Energy Alignment is significant**: Reducing MAE from 0.81 to 0.71 (>10% relative Gain) shows that MD sampling bias requires force field label correction.
- On the tautomerization task, CARD outperformed DFT (B3LYP/6-31G*) because DFT approximates free energy using a single min-energy conformation, which fails for flexible molecules, whereas CARD performs true Boltzmann averaging.

## Highlights & Insights
- **Radix Decomposition** is an elegant bridge: it transforms the continuous high-dimensional "molecular generation" problem into a "coarse-to-fine mixed token AR" problem, reusing the mature Transformer toolchain (KV cache, scaling, cross-task transfer) while strictly satisfying tractable likelihood.
- The **"Zero-free-energy proposal" + BAR** is the core paradigm of this line of research—DeepBAR introduced the idea but was hampered by NF expressivity. CARD liberates expressivity via AR + BMM, upgrading both theory and engineering.
- The **dual query/key split in geometry-aware attention** (query uses previous level coordinates to prevent leakage, key/value uses latest coordinates) is a technique worth adopting in any "step-by-step geometric object generation" task, such as autoregressive protein folding or 3D mesh generation.
- **Minimal performance degradation across chemical environments** (vacuum/toluene/water/NNP/tautomerization) represents "genuine generalization" rarely seen in AI for Chemistry, effectively porting the LLM "one model, many tasks" paradigm to molecules.
- The general approach (radix-based coarse-to-fine + tractable AR) is transferable to all-atom protein modeling, crystal structure generation, and even 3D point cloud generation.

## Limitations & Future Work
- **PCA alignment is unstable for symmetric molecules**: The authors acknowledge that near-symmetric principal axes can cause alignment directions to flip, introducing high variance; more robust equivariant features (e.g., E(3)-equivariant networks) are needed as replacements.
- **Datasets are primarily drug-like small molecules** (ZINC20, atoms < 50): Whether this scales to protein-ligand complexes with thousands of atoms remains unverified.
- **Inference sequence length $\sim N(L+1)$ still grows linearly with the number of atoms**: Complexity is approximately quadratic in $N$ (vanilla Transformer); FlashAttention or Linear Attention is required to scale to thousands of atoms.
- **Sampling bias in training MD trajectories** may propagate to the model; Stage II energy alignment partially mitigates this but is not a definitive cure. Potential directions include: (a) E(3) equivariant CARD; (b) using NNP to replace force field energy for finer labels; (c) extension to protein-ligand docking.

## Related Work & Insights
- **vs. DeepBAR (NeurIPS21)**: Pioneered "zero-free-energy proposal" but used normalizing flows, lacking expressivity and requiring per-system retraining. CARD uses AR Transformer + radix to liberate expressivity and enable cross-system generalization.
- **vs. neural TI / FEAT**: Those methods use diffusion to learn time-varying Hamiltonians for thermodynamic integration, remaining within the alchemical paradigm. CARD uses BAR to bypass intermediate states entirely.
- **vs. Boltzmann Generators (Noé 2019, Tan 2025)**: These seek equilibrium sampling; most cannot calculate precise log-density or fail at cross-system transfer. CARD possesses both likelihood and transferability.
- **vs. MGVAE / MGT / Sequoia (Multi-resolution molecular modeling)**: Those operate coarse-to-fine at the graph level, whereas CARD operates at the atomic coordinate level while maintaining exact AR factorization.
- **vs. LLM BPE tokenization**: Radix decomposition is effectively "BPE for 3D coordinates," suggesting that other continuous physical quantities (e.g., protein dihedral angles, crystal lattice parameters) can be tokenized similarly for Transformer processing.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Strictly unifies LLM AR paradigms with zero-free-energy proposals; radix decomposition is a truly original "3D-to-token" bridge.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three complementary tasks + detailed ablations + inference speed comparisons; lacks protein-scale validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Derivations (bijection, Jacobian = 1, log-density decomposition) are rigorous, and engineering details are clear—a rare combination of "hard theory + hard experiments" in AI4Sci.
- Value: ⭐⭐⭐⭐⭐ Transforming hours-long single-system free energy calculations into cross-system seconds-level predictions with high precision provides transformative engineering value for FEP-based drug screening pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Protein Autoregressive Modeling via Multiscale Structure Generation](protein_autoregressive_modeling_via_multiscale_structure_generation.md)
- [\[NeurIPS 2025\] Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](../../NeurIPS2025/computational_biology/energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)
- [\[ICML 2026\] Learning Protein Structure-Function Relationships through Knowledge-guided Representation Decomposition](learning_protein_structure-function_relationships_through_knowledge-guided_repre.md)
- [\[ICML 2026\] Constrained Flow Optimization via Sequential Fine-Tuning for Molecular Design](constrained_flow_optimization_via_sequential_fine_tuning_for_molecular_design.md)
- [\[ICML 2026\] Learning the Neighborhood: Contrast-Free Multimodal Self-Supervised Molecular Graph Pretraining](learning_the_neighborhood_contrast-free_multimodal_self-supervised_molecular_gra.md)

</div>

<!-- RELATED:END -->
