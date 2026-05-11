---
title: >-
  [Paper Note] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design
description: >-
  [NeurIPS 2025][Medical Imaging][Diffusion Models] This paper proposes an uncertainty-aware multi-objective reinforcement learning framework that guides a 3D molecular diffusion model (EDM) to simultaneously optimize drug…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Diffusion Models"
  - "Reinforcement Learning"
  - "Multi-Objective Optimization"
  - "Uncertainty Quantification"
  - "3D Molecular Generation"
  - "Drug Discovery"
date: 2026-05-08
content_hash: 002ff4bcb9661e96
---

# Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design

**Conference**: NeurIPS 2025
**arXiv**: [2510.21153](https://arxiv.org/abs/2510.21153)
**Code**: [Kyle4490/RL-Diffusion](https://github.com/Kyle4490/RL-Diffusion)
**Area**: Medical Imaging
**Keywords**: Diffusion Models, Reinforcement Learning, Multi-Objective Optimization, Uncertainty Quantification, 3D Molecular Generation, Drug Discovery

## TL;DR
This paper proposes an uncertainty-aware multi-objective reinforcement learning framework that guides a 3D molecular diffusion model (EDM) to simultaneously optimize drug-likeness (QED), synthetic accessibility (SAS), and binding affinity. The framework dynamically shapes the reward function using predictive uncertainty from surrogate models, consistently outperforms baselines across three benchmark datasets, and validates candidate molecules through molecular dynamics simulations and ADMET analysis.

## Background & Motivation
Designing novel 3D molecules with desired properties is a central challenge in drug discovery. While diffusion models have demonstrated strong capabilities in 3D molecular generation, most existing approaches only enforce basic chemical validity constraints and lack explicit control over multiple drug-relevant attributes.

**Limitations of Prior Work**:
- **Flow matching / Energy-guided methods**: Require explicitly differentiable reward functions and cannot handle black-box objectives such as QED, SAS, and binding affinity.
- **RL-guided generative models**: Have been applied to RNNs, VAEs, and Transformers, but primarily operate on 1D SMILES or 2D molecular graphs; RL-guided 3D molecular diffusion models remain underexplored.
- **RL-diffusion models in image domains** (SFT-PG, DDPO, DPOK): Target single-objective optimization and do not transfer well to multi-objective molecular design.
- **Classical multi-objective optimization** (weighted sum, constraint-based, gradient-based methods): Require careful weight tuning and cannot model the global Pareto front.

**Core Motivation**: 3D molecular geometry is critical for downstream tasks such as molecular docking and molecular dynamics simulations—1D/2D representations are insufficient. An end-to-end framework is needed to unify RL, diffusion models, and uncertainty quantification for multi-objective 3D molecular generation.

## Method

### Overall Architecture
The framework consists of three components: a conditional EDM backbone → surrogate model uncertainty quantification → RL-guided optimization.

### 1. Conditional EDM Backbone
An E(3)-equivariant diffusion model (EDM) serves as the backbone:
- **Forward process**: Gradually adds noise to atomic coordinates $\mathbf{r} \in \mathbb{R}^{M \times 3}$ and features $\mathbf{h} \in \mathbb{R}^{M \times d}$: $q(\mathbf{z}_t | \mathbf{x}) = \mathcal{N}(\mathbf{z}_t; \alpha_t \mathbf{x}, \sigma_t^2 \mathbf{I})$
- **Reverse process**: Parameterizes the denoising distribution $p_\theta(\mathbf{z}_{t-1} | \mathbf{z}_t, c) = \mathcal{N}(\mathbf{z}_{t-1}; \mu_\theta(\mathbf{z}_t, t, c), \sigma_t^2 \mathbf{I})$, where $c$ is the target property conditioning vector.
- The noise predictor employs an E(n)-equivariant GNN (EGNN).

### 2. Surrogate Models and Multi-Objective Uncertainty Quantification
Chemprop's directed message-passing neural network (D-MPNN) is used as the surrogate predictor, with independent models trained for each property.

**Single-property uncertainty reward**: Estimates the probability that molecule $m$ satisfies threshold $\delta$:
$$U_{\text{single}}(m; \delta) = \eta \int_\delta^\infty \frac{1}{\sigma(m)\sqrt{2\pi}} \exp\left(-\frac{1}{2}\left(\frac{x - \mu(m)}{\sigma(m)}\right)^2\right) dx$$
where $\eta = +1$ denotes higher-is-better (e.g., QED) and $\eta = -1$ denotes lower-is-better (e.g., SAS, binding affinity).

**Multi-objective reward aggregation**: Assuming conditional independence among properties, the joint satisfaction probability is the product of individual property probabilities:
$$U_{\text{multi}}(m; \delta_1, \ldots, \delta_k) = \prod_{i=1}^k U_{\text{single}}^i(m; \delta_i)$$

### 3. RL-Guided Optimization

**Trajectory sampling**: At each iteration, $n$ molecules are sampled and their complete denoising trajectories $\{\mathbf{z}_T, \ldots, \mathbf{z}_0\}$ are recorded. The reverse denoising process is reformulated as a probability density to support gradient estimation.

**Reward design**: The total reward incorporates three auxiliary components:
$$R_{\text{total}}(m) = U_{\text{multi}}(m) \cdot R_{\text{bonus}}(m) - \lambda(t_{\text{episode}}) \cdot D(m)$$
- **Reward boosting** $R_{\text{bonus}}$: Provides incremental bonuses based on molecular validity, uniqueness, and novelty.
- **Diversity penalty** $D(m)$: Penalizes intra-batch Tanimoto similarity to prevent mode collapse.
- **Dynamic cutoff**: Property thresholds $\delta_i$ are updated dynamically based on a moving average of historically generated molecules.
- The penalty weight decays over training as $\lambda(t) = \lambda_0 e^{-\alpha t}$, encouraging exploration early and exploitation later.

**Policy update**: A PPO-style clipped policy gradient loss is adopted:
$$\mathcal{L}_{\text{PPO}} = -\mathbb{E}_{m \sim p_\theta}\left[\min\left(r(m) \cdot R_{\text{total}}(m), \text{clip}(r(m), 1-\epsilon, 1+\epsilon) \cdot R_{\text{total}}(m)\right)\right]$$

## Key Experimental Results

### Experimental Setup
- **Datasets**: QM9 (small organic molecules), ZINC15 (drug-like molecules), PubChem (large complex molecules)
- **Target properties**: QED > 0.4, SAS < 8, binding affinity < −4.5 (EGFR target)
- **Baselines**: Vanilla EDM (without RL), SFT-PG, DDPO-SF, DDPO-IS, DPOK
- **Evaluation**: 2,000 molecules generated per run, averaged over 3 independent runs

### Table 1: Main Results — Performance Across Three Datasets

| Dataset | Method | Val (%) | Uni (%) | VUN (%) | MSta (%) | Top (%) |
|---------|--------|---------|---------|---------|----------|---------|
| QM9 | W/O RL | 88.55 | 97.57 | 86.19 | 95.90 | 25.17 |
| QM9 | SFT-PG | 88.57 | 96.80 | 85.57 | 95.62 | 25.58 |
| QM9 | DDPO-IS | 88.82 | 96.59 | 85.27 | 86.10 | 25.77 |
| QM9 | **Ours** | **98.17** | 90.90 | **88.90** | **99.17** | **28.33** |
| ZINC15 | W/O RL | 30.05 | 100.00 | 30.05 | 12.00 | 8.02 |
| ZINC15 | SFT-PG | 41.25 | 100.00 | 41.25 | 25.55 | 10.43 |
| ZINC15 | **Ours** | **99.02** | 99.75 | **98.77** | **98.08** | **33.40** |
| PubChem | W/O RL | 7.18 | 99.67 | 7.17 | 38.18 | 2.23 |
| PubChem | DDPO-IS | 10.50 | 99.90 | 10.48 | 45.37 | 2.52 |
| PubChem | **Ours** | **16.23** | 100.00 | **16.23** | **88.65** | **2.97** |

Key findings:
- The most substantial improvement is observed on ZINC15: validity increases from 30.05% to **99.02%**, and Top from 8.02% to **33.40%**.
- On QM9, validity exceeds all baselines by more than **9%**.
- On PubChem, molecular stability (MSta) improves from 38.18% to **88.65%**.

### Table 2: Ablation Study — Multi-Objective Strategy Comparison (QM9 Dataset)

| Category | Method | Val (%) | VUN (%) | Top (%) |
|----------|--------|---------|---------|---------|
| Scalarization | WS | 91.78 | 87.86 | 27.02 |
| Scalarization | POO | 89.13 | 77.88 | 24.60 |
| Constraint | NMD | 93.30 | 77.67 | 25.75 |
| Constraint | PFM | 91.98 | 88.75 | 24.68 |
| Gradient | GradVac | 88.50 | 84.83 | 24.43 |
| Uncertainty | UCB | 86.10 | 82.28 | 13.40 |
| Uncertainty | BORE | 89.33 | 86.57 | 23.73 |
| Ours W/O Reward Boost | — | 90.00 | 86.95 | 25.92 |
| Ours W/O Diversity Penalty | — | 83.55 | 65.77 | 25.43 |
| Ours W/O Dynamic Cutoff | Static | 95.73 | 90.65 | 24.88 |
| **Ours (Full)** | — | **98.17** | **88.90** | **28.33** |

Key findings:
- Removing the diversity penalty causes VUN to drop sharply from 88.90% to **65.77%**, confirming its critical role in preventing mode collapse.
- The full method consistently leads all alternative strategies on the Top metric.
- All 16 alternative multi-objective strategies across 4 categories underperform the proposed joint probability uncertainty approach.

### MD and ADMET Validation
- Generated candidate molecules exhibit RMSD values of 0.20–0.30 nm in molecular dynamics simulations, comparable to known EGFR inhibitors.
- ADMET analysis confirms favorable absorption, low CYP inhibition, and low toxicity.
- The framework is further extended to GeoLDM and GFMDiff architectures, validating its generalizability.

## Highlights & Insights
- **First end-to-end RL + diffusion + uncertainty quantification framework**: Unifies all three components for 3D multi-objective molecular generation with a clear methodological contribution.
- **Uncertainty-driven joint probability reward**: Converts surrogate model predictive uncertainty into a smooth, interpretable reward signal in $[0, 1]$, naturally handling black-box objectives.
- **Complete reward engineering**: The three components—reward boosting, diversity penalty, and dynamic cutoff—are each indispensable, as demonstrated by thorough ablation evidence.
- **Validation in realistic drug discovery settings**: Goes beyond generative metrics by verifying candidate molecules through MD simulations and ADMET analysis benchmarked against known EGFR inhibitors.

## Limitations & Future Work
- **Limited performance on large PubChem molecules**: Validity reaches only 16.23%, primarily constrained by the backbone diffusion model's capacity to handle complex large molecules rather than the RL framework itself.
- **Surrogate model dependence**: Reward quality depends on the predictive accuracy and uncertainty calibration of surrogate models; the binding affinity surrogate achieves R² of only 0.86–0.88.
- **Property independence assumption**: The multi-objective reward assumes conditional independence among properties, whereas drug-relevant attributes often exhibit correlations (e.g., QED and SAS tend to be negatively correlated).
- **Computational overhead**: The RL training phase requires iterative molecule generation, property evaluation, and policy updates, resulting in substantial training cost.
- **Lenient evaluation thresholds**: The Top metric adopts relatively relaxed thresholds (QED > 0.4, SAS < 8), which may not reflect the stricter standards required in practical drug development.

## Related Work & Insights
- **3D molecular generation**: G-SchNet (autoregressive) → E-NF (equivariant flows) → EDM (equivariant diffusion) → GeoLDM (latent space diffusion) → GFMDiff (physics-constrained); this work builds on EDM and validates generalizability on GeoLDM and GFMDiff.
- **RL-guided diffusion**: SFT-PG (reducing distributional mismatch), DDPO-IS/SF (denoising as multi-step decision-making), DPOK (KL regularization)—all address single-objective optimization in image domains; this work represents the first transfer to 3D molecular multi-objective settings.
- **Multi-objective optimization**: Scalarization (WS/POO/MMM), constraint-based (NMD/CP), gradient-based (PCGrad/CAGrad), uncertainty-based (UCB/EI/BORE)—the proposed joint probability method outperforms all 16 alternative strategies.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First framework integrating RL, diffusion models, and uncertainty quantification for 3D molecular generation; the idea of converting surrogate uncertainty into a joint probability reward is concise and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three datasets, five baselines, ablation over 16 alternative multi-objective strategies, three-component ablation, comparison across three diffusion architectures, and MD + ADMET validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with clear mathematical derivations, though the high content density makes the main text somewhat dense.
- **Value**: ⭐⭐⭐⭐ — Practically valuable for both RL-guided molecular generation and multi-objective drug design; credibility is enhanced by MD/ADMET validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Multi-Objective Reinforcement Learning with Max-Min Criterion: A Game-Theoretic Approach](multi-objective_reinforcement_learning_with_max-min_criterion_a_game-theoretic_a.md)
- [\[NeurIPS 2025\] Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)
- [\[NeurIPS 2025\] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)
- [\[NeurIPS 2025\] De novo generation of functional terpene synthases using TpsGPT](de_novo_generation_of_functional_terpene_synthases_using_tpsgpt.md)
- [\[NeurIPS 2025\] Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology](semantic_and_visual_crop-guided_diffusion_models_for_heterogeneous_tissue_synthe.md)

</div>

<!-- RELATED:END -->
