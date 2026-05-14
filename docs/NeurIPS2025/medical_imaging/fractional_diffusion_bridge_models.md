---
title: >-
  [Paper Note] Fractional Diffusion Bridge Models
description: >-
  [NeurIPS 2025][Medical Imaging][Diffusion bridge models] This paper proposes Fractional Diffusion Bridge Models (FDBM), which incorporate fractional Brownian motion (fBM) into the generative diffusion bridge framework. T…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Diffusion bridge models"
  - "fractional Brownian motion"
  - "protein conformation prediction"
  - "image translation"
  - "Schrödinger bridge"
date: 2026-05-08
content_hash: b597b93855af77ef
---

# Fractional Diffusion Bridge Models

**Conference**: NeurIPS 2025
**arXiv**: [2511.01795](https://arxiv.org/abs/2511.01795)
**Code**: [GitHub-paired](https://github.com/GabrielNobis/FDBM_paired) / [GitHub-unpaired](https://github.com/mspringe/FDBM_unpaired) / [SBFlow](https://github.com/mspringe/Schroedinger-Bridge-Flow)
**Area**: Medical Imaging
**Keywords**: Diffusion bridge models, fractional Brownian motion, protein conformation prediction, image translation, Schrödinger bridge

## TL;DR

This paper proposes Fractional Diffusion Bridge Models (FDBM), which incorporate fractional Brownian motion (fBM) into the generative diffusion bridge framework. The Hurst exponent $H$ controls the roughness and long-range dependence of trajectories, yielding improvements over Brownian motion baselines on protein conformation prediction and image translation tasks.

## Background & Motivation

**Background**: Diffusion bridge models construct stochastic interpolation paths between two distributions by conditioning stochastic processes, and are widely applied to paired/unpaired data translation tasks such as protein conformation prediction and image translation. Existing models uniformly employ standard Brownian motion (BM) as the driving noise.

**Limitations of Prior Work**: Standard BM is a Markov process with independent increments, and thus cannot capture **memory effects, long-range dependence, roughness, or anomalous diffusion** present in real data. For complex systems such as proteins, the assumption of temporally independent increments may lead to insufficient modeling.

**Key Challenge**: The motivation for choosing BM in existing work is mathematical convenience rather than physical fidelity — adopting a more expressive driving noise should better match the true data dynamics.

**Goal**: To introduce fractional Brownian motion (non-Markovian, with long-range correlations) into diffusion bridges while maintaining trainability and efficient inference.

**Key Insight**: The paper exploits the Markovian approximation of fBM (MA-fBM), which approximates fBM via a linear superposition of $K$ Ornstein-Uhlenbeck (OU) processes, making the augmented system Markovian.

**Core Idea**: Replace BM-driven diffusion bridges with the Markovian approximation of fBM, using the Hurst exponent $H$ to flexibly control the properties of generated trajectories, improving performance in both paired and unpaired settings.

## Method

### Overall Architecture

The core of FDBM is to replace the BM in standard diffusion bridges with a scaled MA-fBM $X = \sqrt{\varepsilon}\hat{B}^H$, where $\hat{B}^H_t = \sum_{k=1}^K \omega_k Y_t^k$ is a weighted sum of $K$ OU processes. The reference process $X$ itself is non-Markovian, but the augmented process $Z = (X, Y^1, \ldots, Y^K)$ is Markovian.

### Key Designs

#### 1. MA-fBM as Driving Noise

**Type II fBM definition**: $B_t^H = \frac{1}{\Gamma(H+1/2)} \int_0^t (t-s)^{H-1/2} dB_s$

- $H > 0.5$: positively correlated increments (super-diffusion), smoother trajectories
- $H < 0.5$: negatively correlated increments (sub-diffusion), rougher trajectories
- $H = 0.5$: reduces to standard BM

**MA-fBM**: $K$ OU processes with rates $\gamma_k = r^{k-n}$ are selected; optimal weights $\omega$ are given by the closed-form linear system $A\omega = b$, minimizing the $L^2(\mathbb{P})$ error. Experiments fix $K=5$.

#### 2. MA-fBB (Markovian Approximation Fractional Brownian Bridge)

Conditioning the augmented process $Z$ on its endpoints yields a partially pinned process $Z_{|x_0,x_1}$, whose SDE drift includes a guidance term $u(t,z)$ that is analytically computable. The conditional mean $\mu_{1|t}(z)$ and conditional variance $\sigma^2_{1|t}$ are both available in closed form.

#### 3. Paired Data Translation

**Proposition 5 (Coupling Preservation)**: There exists a stochastic process $Z^\star$ that preserves the training data coupling $\Pi_{0,1}$, satisfying $(X_0^\star, X_1^\star) \sim \Pi_{0,1}$.

**Training objective**:
$$\mathcal{L}_{\text{FDBM}}^{\text{paired}}(\theta) = \int_0^1 \mathbb{E}_{\mathbb{P}^\star}\left[\left\|\frac{X_1^\star - \mu_{1|t}(Z_t^\star)}{\sigma_{1|t}^2} - \tilde{u}^\theta(t, X_0, \mu_{1|t}(Z_t^\star))\right\|^2\right] dt$$

**Key point**: The output dimension of the neural network $\tilde{u}^\theta$ matches the data dimension $d$ (not the augmented dimension), and is mapped to the augmented space via a scaling transformation. FDBM can therefore reuse the network architecture of ABM with negligible additional computational overhead.

#### 4. Unpaired Data Translation (Schrödinger Bridge)

The reference process in the Schrödinger bridge (SB) problem is replaced with MA-fBM, defining the SB problem over the augmented space, and introducing augmented reciprocal classes and augmented Markovian projections. An $\alpha$-IMF training scheme (pre-training + fine-tuning) is adopted.

### Loss & Training

- **Paired**: Minimize the $L^2$ distance between neural network predictions and the fractional bridge drift targets.
- **Unpaired**: An analogous objective that does not condition on the starting value $X_0$, using a forward-forward training strategy.
- **Note**: In the unpaired setting, the MA-fBM reference process becomes unstable during fine-tuning when $H$ deviates substantially from 0.5.

## Key Experimental Results

### Paired Data: Protein Conformation Prediction (D3PM Dataset)

| Method | Median RMSD(Å)↓ | Mean RMSD(Å)↓ | RMSD<2Å(%)↑ | RMSD<5Å(%)↑ | Δ RMSD Mean↑ |
|------|-----------------|---------------|-------------|-------------|-------------|
| SBALIGN | 3.67 | 4.82 | 0% | 71% | 1.92 |
| Sesame | 2.87 | 3.65 | 38% | 82% | 3.11 |
| ABM (baseline) | 2.40 | 3.49 | 43% | 84% | 3.35 |
| **FDBM (H=0.2)** | **2.12** | **3.34** | **48%** | **86%** | 3.39 |
| FDBM (H=0.3) | 2.33 | 3.42 | 43% | 85% | **3.49** |
| FDBM (H=0.1) | 2.20 | 3.44 | 46% | 83% | 3.45 |

### Unpaired Data: AFHQ Image Translation

Evaluation of cat↔wild translation on AFHQ-256 and AFHQ-512:

| Method/Setting | FID (wild→cat)↓ | FID (cat→wild)↓ |
|-----------|----------------|----------------|
| SBFlow (AFHQ-256) | baseline | baseline |
| FDBM (H=0.6, AFHQ-256) | 19.42 | 11.62 |
| FDBM (H=0.4, AFHQ-512) | 30.11 | 14.27 |

### Synthetic Data: Moons & T-Shape

| Dataset | ABM WSD | FDBM Best WSD |
|--------|---------|--------------|
| Moons | 0.015±0.019 | **0.012±0.002** (H=0.7) |
| T-Shape | 0.082±0.028 | **0.048±0.039** (H=0.2) |

### Ablation Study

- **Choice of $H$ is task-dependent**: Moons benefits from smooth trajectories ($H=0.6$–$0.7$), while T-Shape and protein tasks favor rough trajectories ($H=0.1$–$0.3$).
- **$K=5$ OU processes** are sufficient; increasing $K$ yields marginal improvements.
- **Diffusion coefficient $\sqrt{\varepsilon}$**: the optimal value for the protein task is 0.2.

### Key Findings

1. **Rough trajectories ($H<0.5$) are superior for protein prediction**: the proportion of RMSD<2Å increases from 43% (ABM) to 48%, and median RMSD decreases from 2.40Å to 2.12Å.
2. **ABM is already a strong baseline**, outperforming all prior methods (SBALIGN, Sesame).
3. **FDBM preserves coupling** (Proposition 5), unlike SBALIGN.
4. **Fine-tuning instability in the unpaired setting when $H$ deviates far from 0.5**, indicating forward-backward asymmetry challenges in the fBM SB problem.
5. **Minimal computational overhead**: FDBM adds only the MA-fBM input/output transformations relative to ABM.

## Highlights & Insights

- **This is the first work to introduce fBM into diffusion bridge modeling** — conceptually elegant, as BM is merely the special case $H=0.5$, and $H$ can now be tuned to match data characteristics.
- **The theoretical proof of coupling preservation** (Proposition 5) is a significant theoretical contribution.
- **Experimental findings suggest that protein conformational changes suit rough trajectories**, implying memory effects in molecular motion.
- **The architecture reuse design is elegant**: the neural network dimensionality is unchanged, with only a scaling mapping applied, minimizing implementation complexity.
- **The finding that Moons prefers smooth $H$ while T-Shape prefers rough $H$** suggests that the optimal $H$ has an interpretable structural basis.

## Limitations & Future Work

1. **Instability when $H$ deviates far from 0.5 in the unpaired setting**: SB fine-tuning fails to converge, limiting the flexibility of FDBM in unpaired tasks.
2. **$H$ requires manual search**: no automatic method for selecting $H$ currently exists; treating it as a learnable parameter is a natural direction.
3. **MA-fBM is an approximation**: the $K=5$ approximation has limited precision and may be insufficient in certain scenarios.
4. **Lack of adaptive $H$**: different regions may require different $H$ values, and a globally fixed $H$ may be suboptimal.
5. **Protein experiments are conducted only on a subset of D3PM**: larger-scale validation is needed.

## Related Work & Insights

- Builds upon ABM (Bortoli et al.), SBFlow (Bortoli et al.), and SBALIGN (Somnath et al.).
- The MA-fBM technique derives from Harms & Stefanovits and Daems et al.
- **Core insight**: the statistical properties of the driving noise (memory, roughness) should be matched to the physical characteristics of the target data, rather than chosen solely for mathematical convenience.

## Rating

⭐⭐⭐⭐ (4/5)

**Rationale**: The work is conceptually novel (the first to introduce fBM into diffusion bridges), theoretically sound (coupling preservation proof), and experimentally comprehensive (synthetic + protein + image). The improvements on protein tasks are meaningful. The instability in the unpaired setting is a clear limitation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)
- [\[NeurIPS 2025\] Posterior Sampling by Combining Diffusion Models with Annealed Langevin Dynamics](posterior_sampling_by_combining_diffusion_models_with_annealed_langevin_dynamics.md)
- [\[NeurIPS 2025\] Atomic Diffusion Models for Small Molecule Structure Elucidation from NMR Spectra](atomic_diffusion_models_for_small_molecule_structure_elucidation_from_nmr_spectr.md)
- [\[NeurIPS 2025\] Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology](semantic_and_visual_crop-guided_diffusion_models_for_heterogeneous_tissue_synthe.md)
- [\[NeurIPS 2025\] FAPEX: Fractional Amplitude-Phase Expressor for Robust Cross-Subject Seizure Prediction](fapex_fractional_amplitude-phase_expressor_for_robust_cross-subject_seizure_pred.md)

</div>

<!-- RELATED:END -->
