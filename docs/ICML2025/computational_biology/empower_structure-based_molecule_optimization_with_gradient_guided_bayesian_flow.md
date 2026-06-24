---
title: >-
  [Paper Note] Empower Structure-Based Molecule Optimization with Gradient Guided Bayesian Flow Networks
description: >-
  [ICML2025][Computational Biology][Bayesian Flow Network] This paper proposes the MolJO framework, which leverages the continuously differentiable parameter space $\boldsymbol{\theta}$ of Bayesian Flow Networks (BFNs) to achieve joint gradient-guided optimization of both molecular coordinates (continuous) and atom types (discrete). It incorporates a sliding-window backward correction strategy to balance exploration and exploitation, outperforming existing methods with a 51.3%…
tags:
  - "ICML2025"
  - "Computational Biology"
  - "Bayesian Flow Network"
  - "Structure-Aware Molecule Optimization"
  - "Gradient Guidance"
  - "Joint Continuous-Discrete Optimization"
  - "SE(3)-Equivariance"
date: 2026-05-08
content_hash: af14166ee7d704b6
---

# Empower Structure-Based Molecule Optimization with Gradient Guided Bayesian Flow Networks

**Conference**: ICML2025  
**arXiv**: [2411.13280](https://arxiv.org/abs/2411.13280)  
**Code**: [AlgoMole/MolCRAFT](https://github.com/AlgoMole/MolCRAFT)  
**Area**: Computational Biology  
**Keywords**: Bayesian Flow Network, Structure-Aware Molecule Optimization, Gradient Guidance, Joint Continuous-Discrete Optimization, SE(3)-Equivariance

## TL;DR

This paper proposes the MolJO framework, which leverages the continuously differentiable parameter space $\boldsymbol{\theta}$ of Bayesian Flow Networks (BFNs) to achieve joint gradient-guided optimization of both molecular coordinates (continuous) and atom types (discrete). It incorporates a sliding-window backward correction strategy to balance exploration and exploitation, outperforming existing methods with a 51.3% Success Rate on CrossDocked2020.

## Background & Motivation

**Problem Definition**: Structure-aware molecule optimization (SBMO) aims to simultaneously optimize the 3D coordinates $\mathbf{x} \in \mathbb{R}^{N \times 3}$ and discrete atom types $\mathbf{v} \in \{1,\dots,K\}^N$ of ligand molecules given a target protein pocket, to satisfy various drug-like property indicators such as binding affinity and synthetic accessibility.

**Limitations of Prior Work**:

- **Generative Models** (e.g., TargetDiff, DecompDiff, MolCRAFT) primarily maximize likelihood to fit training data, lacking targeted optimization capability for specific molecular properties.
- **Oracle-based Methods** (e.g., DecompOpt) require repeated calls to docking simulations for top-of-N screening, which is computationally expensive.
- **Gradient Guidance Methods** face a continuous-discrete challenge:
   - Discrete atom types cannot directly backpropagate gradients. Existing approximations (e.g., adding Gaussian noise, assuming Gaussian-distributed classifiers) are inaccurate.
   - TAGMol only guides continuous coordinates and ignores discrete types, leading to cross-modal inconsistency—binding affinity gets improved, but synthetic accessibility collapses.

**Design Motivation**: The parameter space $\boldsymbol{\theta}$ of BFN is an aggregation of noisy samples via Bayesian inference, which is naturally continuously differentiable and covers both continuous and discrete modalities, serving as an ideal vehicle for joint gradient guidance.

## Method

### Overall Architecture: MolJO

MolJO (Molecule Joint Optimization) applies gradient guidance to the parameter space $\boldsymbol{\theta} = [\boldsymbol{\theta}^x, \boldsymbol{\theta}^v]$ of BFN. In contrast to diffusion models that apply guidance on noisy latent variables $\mathbf{y}$, MolJO operates on the low-variance Bayesian posterior $\boldsymbol{\theta}$, ensuring a smooth gradient flow.

### Joint Gradient Guidance

The guided distribution is formulated as a Product of Experts:

$$\pi(\boldsymbol{\theta}_i | \boldsymbol{\theta}_{i-1}) \propto p_\phi(\boldsymbol{\theta}_i | \boldsymbol{\theta}_{i-1}) \cdot p_E(\boldsymbol{\theta}_i)$$

where $p_E(\boldsymbol{\theta}_i) = \exp[-E(\boldsymbol{\theta}_i, t_i)]$ represents the Boltzmann distribution corresponding to the energy function. The guided transition kernel is approximated via first-order Taylor expansion:

**Continuous Coordinate Guidance**:

$$\boldsymbol{\theta}_i^x \sim \mathcal{N}\left(\boldsymbol{\theta}_\phi^x + \sigma^x \mathbf{g}_{\boldsymbol{\theta}^x},\; \sigma^x \mathbb{I}\right)$$

where $\mathbf{g}_{\boldsymbol{\theta}^x} = -\nabla_{\boldsymbol{\theta}^x} E(\boldsymbol{\theta}, t_i)|_{\boldsymbol{\theta}=\boldsymbol{\theta}_{i-1}}$. This is equivalent to guiding the noisy latent variable $\mathbf{y}^x$ using an uncertainty-adjusted gradient $(\rho_i / \alpha_i)^2 \mathbf{g}_{\mathbf{y}^x}$.

**Discrete Type Guidance**:

$$\mathbf{y}_i^v \sim \mathcal{N}\left(\mathbf{y}_\phi^v + \sigma^v \mathbf{g}_{\mathbf{y}^v},\; \sigma^v \mathbb{I}\right)$$

The guidance operates on the discrete data through the Gaussian-distributed latent variable $\mathbf{y}^v$. Its final effect is re-weighting the class probabilities of the categorical distribution $\boldsymbol{\theta}^v$—increasing the probability of the classes pointed to by the gradient while decreasing others correspondingly. Key advantage: It avoids assuming a Gaussian-distributed classifier and guarantees that discrete variables always reside on the probability simplex.

**SE(3)-Equivariance**: When both the network $\boldsymbol{\Phi}$ and energy function $E$ are SE(3)-equivariant, and the protein's center of mass is zeroed, the guided sampling process remains SE(3)-equivariant.

### Backward Correction Sampling Strategy

In standard BFN sampling, the update at step $i$ depends only on the preceding step $\boldsymbol{\theta}_{i-1}$. Backward correction maintains a sliding window of size $k$ that backtracks and replaces the historical inputs of the past $k$ steps with the current step's optimized prediction $\hat{\mathbf{x}}_i$, re-aggregating $\boldsymbol{\theta}$:

$$p_\phi(\boldsymbol{\theta}_n | \boldsymbol{\theta}_{n-1}, \boldsymbol{\theta}_{n-k}) = \mathbb{E}_{p_O(\hat{\mathbf{x}}_n | \boldsymbol{\Phi}(\boldsymbol{\theta}_{n-1}, t_n))} \; p_U\!\left(\boldsymbol{\theta}_n \,|\, \boldsymbol{\theta}_{n-k}, \hat{\mathbf{x}}_n;\; \sum_{i=n-k+1}^{n} \alpha_i\right)$$

Specifically, the update formula for the continuous part is:

$$\boldsymbol{\theta}_n^x \sim \mathcal{N}\!\left(\frac{\Delta\beta \hat{\mathbf{x}}_n + \boldsymbol{\theta}_{n-k}^x \rho_{n-k}}{\rho_n},\; \frac{\Delta\beta}{\rho_n^2}\mathbb{I}\right)$$

**Exploration-Exploitation Trade-off**: $k=1$ degenerates to the standard single-step update (maximum exploration), whereas $k=n$ uses the entire history (maximum exploitation). An intermediate value of $k$ allows rapid exploration of the molecular space during the early stages of optimization and utilizes more coherent gradient signals for fine-tuning in the later stages. This is validated empirically by visualizing the gradient cosine similarity.

### Loss & Training

The BFN training objective minimizes the KL divergence between the sender and receiver distributions:

$$L^n(\mathbf{x}) = \mathbb{E}_{\prod_{i=1}^n p_U(\boldsymbol{\theta}_i | \boldsymbol{\theta}_{i-1}, \mathbf{x}; \alpha_i)} \sum_{i=1}^n D_{\text{KL}}\!\left(p_S(\mathbf{y}_i | \mathbf{x}; \alpha_i) \,\|\, p_R(\mathbf{y}_i | \boldsymbol{\theta}_{i-1}, t_i, \alpha_i)\right)$$

During inference, a gradient scaling factor $s$ is introduced as a temperature parameter, which is equivalent to $p_E^s(\boldsymbol{\theta},t) \propto \exp[-sE(\boldsymbol{\theta},t)]$.

## Key Experimental Results

### Datasets and Settings

- **Dataset**: CrossDocked2020, filtered for RMSD > 1Å, clustered by 30% sequence identity, with 100K training poses + 100 test proteins.
- **Evaluation Metrics**: Vina Score/Min/Dock (binding affinity ↓), QED (drug-likeness ↑), SA (synthetic accessibility ↑), Success Rate (Vina Dock < -8.18, QED > 0.25, and SA > 0.59).
- 100 molecules are generated for each protein.

### Main Results (Table 1)

| Method | Category | Vina Dock ↓ | QED ↑ | SA ↑ | Success Rate ↑ |
|------|------|-------------|-------|------|----------------|
| Reference | — | -7.45 | 0.48 | 0.73 | 25.0% |
| TargetDiff | Gen | -7.80 | 0.48 | 0.58 | 10.5% |
| MolCRAFT | Gen | -7.67 | 0.50 | 0.67 | 26.8% |
| DecompOpt | Oracle | -7.63 | 0.56 | 0.73 | 39.4% |
| TAGMol | Grad | -8.59 | 0.55 | 0.56 | 11.1% |
| **MolJO** | **Grad** | **-9.05** | **0.56** | **0.78** | **51.3%** |
| MolJO† (N=10) | G+O | -10.50 | 0.67 | 0.79 | 70.3% |

**Key Findings**:

- MolJO achieves state-of-the-art (SOTA) performance in Vina Dock, SA, and Success Rate.
- Compared to the only other gradient guidance baseline, TAGMol, the Success Rate increases from 11.1% to 51.3% (approximately a **4.6× gain**).
- TAGMol's SA is only 0.56 (one of the lowest), confirming the assertion that guiding only continuous coordinates leads to a drop in synthetic accessibility.
- MolJO's "Me-Better" ratio (the proportion of improved molecules) is **2×** higher than that of other 3D baselines.

### Constrained Optimization (R-group Optimization & Scaffold Hopping)

MolJO can be flexibly extended to practical drug design scenarios such as R-group redesign (replacing substituents on a fixed core) and scaffold hopping, further demonstrating the versatility of the method.

## Highlights & Insights

1. **First principled joint gradient guidance framework for continuous-discrete data**: By leveraging the continuous differentiability of the BFN parameter space, it sidesteps the fundamental difficulty of guiding discrete data in diffusion models.
2. **Novel and practical backward correction strategy**: The sliding window size $k$ offers a flexible control knob for the exploration-exploitation trade-off. Visualization of the gradient cosine similarity serves as an intuitive and effective analysis.
3. **Cross-modal consistency**: Joint guidance simultaneously improves both binding affinity and synthetic accessibility, resolving the disconnect between discrete and continuous modalities found in methods like TAGMol.
4. **Plug-and-play capability**: As a guidance method, MolJO can be combined with various pre-trained generative models and supports multi-objective optimization.

## Limitations & Future Work

1. **Limitations of the energy function**: The current framework relies on differentiable surrogate energy functions for gradient computation, which have a gap compared to actual docking scores (the quality of the guidance signal depends heavily on surrogate accuracy).
2. **Computational overhead**: The sliding window in backward correction increases the computational and storage requirements of each step, leading to higher costs for larger values of $k$.
3. **Evaluation limitations**: The primary evaluation was conducted only on the CrossDocked2020 benchmark and has yet to be validated in real-world drug discovery pipelines.
4. **Approximation in discrete guidance for atom types**: While more principled than TAGMol, it still relies on a first-order Taylor expansion approximation, which might be inaccurate when the distribution deviates from Gaussian.
5. **Molecular validity**: The paper does not focus heavily on chemical validity checks and post-processing, leaving the practical usability of the generated molecules to be fully verified.

## Related Work & Insights

- **MolCRAFT** (Qu et al., 2024): The base BFN model for this work, upon which MolJO incorporates gradient guidance.
- **TAGMol** (Dorna et al., 2024): A gradient-based method that only guides continuous coordinates, serving as the direct baseline of comparison.
- **Classifier Guidance** (Dhariwal & Nichol, 2021): Classifier guidance for diffusion models, which MolJO extends to BFN + discrete data.
- **BFN** (Graves et al., 2023): The original Bayesian Flow Networks paper, providing the theoretical foundation of the $\boldsymbol{\theta}$ space.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The idea of performing joint guidance in the BFN parameter space is novel, and the backward correction strategy is backed by mathematical derivation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive coverage of unconstrained/constrained/multi-objective/R-group/scaffold hopping scenarios, though lacking real-world drug validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear mathematical derivations, well-articulated motivations, and high-quality illustrations.
- **Value**: ⭐⭐⭐⭐⭐ — Addresses the core challenge of continuous-discrete gradient guidance in SBMO, delivering a significant boost in Success Rate.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Flexibility-conditioned Protein Structure Design with Flow Matching](flexibility-conditioned_protein_structure_design_with_flow_matching.md)
- [\[NeurIPS 2025\] Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number](../../NeurIPS2025/computational_biology/prior-guided_flow_matching_for_target-aware_molecule_design_with_learnable_atom_.md)
- [\[NeurIPS 2025\] Is Sequence Information All You Need for Bayesian Optimization of Antibodies?](../../NeurIPS2025/computational_biology/is_sequence_information_all_you_need_for_bayesian_optimization_of_antibodies.md)
- [\[NeurIPS 2025\] Curly Flow Matching for Learning Non-gradient Field Dynamics](../../NeurIPS2025/computational_biology/curly_flow_matching_for_learning_non-gradient_field_dynamics.md)
- [\[ICLR 2026\] BioBO: Biology-informed Bayesian Optimization for Perturbation Design](../../ICLR2026/computational_biology/biobo_biology-informed_bayesian_optimization_for_perturbation_design.md)

</div>

<!-- RELATED:END -->
