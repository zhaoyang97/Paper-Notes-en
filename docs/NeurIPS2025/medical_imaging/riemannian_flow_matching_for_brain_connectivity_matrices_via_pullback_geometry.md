---
title: >-
  [Paper Note] Riemannian Flow Matching for Brain Connectivity Matrices via Pullback Geometry
description: >-
  [NeurIPS 2025][Medical Imaging][brain connectivity matrices] This paper proposes DiffeoCFM, which leverages pullback metrics induced by global diffeomorphisms to equivalently reformulate conditional flow matching on Riemannian manifolds as standard CFM in Euclidean space. The method enables efficient generation of brain connectivity matrices (SPD/correlation) while strictly preserving manifold constraints, achieving state-of-the-art performance on 3 fMRI and 2 EEG datasets.
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "brain connectivity matrices"
  - "Riemannian flow matching"
  - "pullback geometry"
  - "conditional flow matching"
  - "symmetric positive definite matrices"
  - "correlation matrices"
  - "fMRI"
  - "EEG"
date: 2026-05-08
content_hash: dcd3569ab272d232
---

# Riemannian Flow Matching for Brain Connectivity Matrices via Pullback Geometry

**Conference**: NeurIPS 2025
**arXiv**: [2505.18193](https://arxiv.org/abs/2505.18193)  
**Authors**: Antoine Collas (Inria), Ce Ju (Inria), Nicolas Salvy (Inria), Bertrand Thirion (Inria, CEA, Université Paris-Saclay)
**Code**: [github.com/antoinecollas/DiffeoCFM](https://github.com/antoinecollas/DiffeoCFM)  
**Area**: Image Generation
**Keywords**: brain connectivity matrices, Riemannian flow matching, pullback geometry, conditional flow matching, symmetric positive definite matrices, correlation matrices, fMRI, EEG

## TL;DR

This paper proposes DiffeoCFM, which leverages pullback metrics induced by global diffeomorphisms to equivalently reformulate conditional flow matching on Riemannian manifolds as standard CFM in Euclidean space. The method enables efficient generation of brain connectivity matrices (SPD/correlation) while strictly preserving manifold constraints, achieving state-of-the-art performance on 3 fMRI and 2 EEG datasets.

## Background & Motivation

### State of the Field
Brain functional connectivity matrices (covariance or correlation matrices) are central representations in neuroimaging analyses involving fMRI, EEG, and MEG, and are widely used for motor imagery classification, brain age prediction, and disease diagnosis. These matrices naturally belong to the symmetric positive definite (SPD) manifold $\mathbb{S}_d^{++}$ or the correlation matrix manifold $\text{Corr}_d$—non-Euclidean Riemannian manifolds.

### Limitations of Prior Work
- **Riemannian CFM [Chen & Lipman 2024]**: Trains vector fields directly on the manifold, requiring computation of geodesics, Riemannian norms, and ODE integration on the manifold, resulting in **prohibitively high computational cost** (8–10× slower than Euclidean counterparts).
- **SPD-DDPM [Huang & Han 2023]**: Requires a dedicated SPDNet architecture and suffers from slow training.
- **CorrGAN [Marti 2020]**: Trains a GAN in Euclidean space and projects onto the manifold via post-processing, which severely degrades sample quality ($\alpha,\beta$-F1 drops by up to 0.76).
- **TriangDDPM/TriangCFM**: Model the lower-triangular elements of matrices directly; generated matrices frequently contain negative eigenvalues, causing severe structural distortion after projection.
- Existing methods are either **geometrically faithful but computationally expensive**, or **computationally efficient but violating manifold constraints**.

### Root Cause
The central question is whether a method can simultaneously retain the simplicity and efficiency of Euclidean-space training while strictly guaranteeing that generated samples satisfy manifold constraints. The key insight is that, given a global diffeomorphism $\phi:\mathcal{M}\to E$, all computations can be performed in Euclidean space $E$, which is mathematically equivalent to operating on the Riemannian manifold $\mathcal{M}$.

## Method

### Theoretical Foundation: Pullback Manifold

Given a smooth manifold $\mathcal{M}$ and a global diffeomorphism $\phi:\mathcal{M}\to E$ (where $E$ is a Euclidean space), the Euclidean metric $g_E$ can be pulled back to $\mathcal{M}$ via $\phi$:

$$(\phi^*g_E)_x(\xi,\eta) = g_E(\mathrm{D}\phi(x)[\xi], \mathrm{D}\phi(x)[\eta])$$

Under this pullback metric, geodesics are precisely the preimages of Euclidean straight lines: $\gamma(t)=\phi^{-1}((1-t)\phi(x_0)+t\phi(x_1))$.

### DiffeoCFM: Equivalence Theorems

**Core Proposition 1 (Training Equivalence)**: Under the pullback metric, the Riemannian CFM loss reduces to the standard Euclidean CFM loss:

$$\mathcal{L}(\theta) = \mathbb{E}_{t,y,z_0|y,z_1|y}\|u_\theta^E(t,(1-t)z_0+tz_1,y)-(z_1-z_0)\|_E^2$$

where $z_i=\phi(x_i)$, with no need to compute geodesics, Riemannian norms, or manifold gradients.

**Core Proposition 2 (Sampling Equivalence)**: The solution to the Euclidean ODE $\dot{z}(t)=u_\theta^E(t,z(t),y)$ corresponds exactly to the solution of the Riemannian ODE via $\phi$: $x(t)=\phi^{-1}(z(t))$.

**Core Proposition 3 (Discrete Equivalence)**: For any explicit Runge–Kutta scheme, the Euclidean iterations and the Riemannian iterations correspond exactly through $\phi$.

### Two Diffeomorphism Instantiations

**SPD matrices (EEG covariance)**:  The matrix logarithm map is employed:
$$\phi_{\mathbb{S}_d^{++}}(\Sigma) = \text{vec}_{\text{lt}}(\log(\Sigma))$$
This induces the Log-Euclidean metric and maps to $\mathbb{R}^{d(d+1)/2}$.

**Correlation matrices (fMRI connectivity)**: The normalized Cholesky decomposition is used:
$$\phi_{\text{Corr}_d}(\Sigma) = \text{vec}_{\text{sl}}(\text{nchol}(\Sigma))$$
where $\text{nchol}(\Sigma)=\text{diag}(\text{chol}(\Sigma))^{-1}\text{chol}(\Sigma)$, mapping to $\mathbb{R}^{d(d-1)/2}$.

### Training and Sampling Pipelines

**Training** (Algorithm 1): For each class label $y$, data are mapped to Euclidean space via $z=\phi(x)$. A class-conditional Gaussian is fitted as the source distribution and the empirical distribution serves as the target. A two-layer MLP (512 hidden units) is trained to minimize the standard CFM loss.

**Sampling** (Algorithm 2): A sample $z_0$ is drawn from the source distribution. The Euclidean ODE is integrated using the dopri5 solver to obtain $z_L$, which is then mapped back to the manifold via $\phi^{-1}(z_L)$. **Generated matrices inherently satisfy SPD/correlation matrix constraints.**

## Key Experimental Results

### Experiment 1: Quality Metrics and Classification Accuracy (Comprehensive Comparison across 5 Datasets)

Evaluation is conducted on 3 fMRI datasets (ABIDE: 900 subjects; ADNI: 1,900 scans; OASIS-3: 1,800 sessions) and 2 EEG datasets (BNCI2014-002: 13 subjects; BNCI2015-001: 12 subjects). Quality metrics include $\alpha$-precision (fidelity), $\beta$-recall (diversity), and their harmonic mean $\alpha,\beta$-F1; classification metrics include ROC-AUC and F1.

| Dataset | Method | $\alpha,\beta$-F1 ↑ | ROC-AUC ↑ | F1 ↑ | Training Time (s) |
|--------|------|---------------------|-----------|------|------------|
| ABIDE (fMRI) | TriangCFM | 0.00 | 0.52 | 0.40 | 48.78 |
| ABIDE (fMRI) | DiffeoGauss | 0.38 | 0.66 | 0.53 | 0.07 |
| ABIDE (fMRI) | **DiffeoCFM** | **0.59** | **0.64** | **0.58** | 32.78 |
| ADNI (fMRI) | TriangCFM | 0.01 | 0.56 | 0.34 | 87.37 |
| ADNI (fMRI) | DiffeoGauss | 0.04 | 0.60 | 0.29 | 0.14 |
| ADNI (fMRI) | **DiffeoCFM** | **0.68** | **0.63** | **0.47** | 88.01 |
| OASIS-3 (fMRI) | **DiffeoCFM** | **0.44** | **0.67** | **0.53** | 67.83 |
| BNCI2014 (EEG) | RiemCFM | 0.63 | 0.81 | 0.72 | **1983.58** |
| BNCI2014 (EEG) | **DiffeoCFM** | 0.62 | 0.81 | 0.74 | 253.04 |
| BNCI2015 (EEG) | RiemCFM | 0.88 | 0.73 | 0.66 | **2753.93** |
| BNCI2015 (EEG) | **DiffeoCFM** | **0.89** | 0.73 | 0.65 | 319.83 |

DiffeoCFM achieves substantially higher $\alpha,\beta$-F1 than all baselines on all fMRI datasets. On EEG datasets, it matches RiemCFM in generation quality while being **8× faster in training and 10× faster in sampling**.

### Experiment 2: Destructive Effect of Projection on TriangCFM

| Dataset | $\Delta\alpha$-precision | $\Delta\beta$-recall | $\Delta\alpha,\beta$-F1 |
|--------|--------------------------|----------------------|------------------------|
| ABIDE | -0.34 | -0.69 | **-0.50** |
| ADNI | -0.63 | -0.74 | **-0.69** |
| OASIS-3 | -0.52 | -0.76 | **-0.64** |
| BNCI2014-002 | +0.13 | -0.56 | -0.19 |
| BNCI2015-001 | +0.00 | -0.19 | -0.09 |

After projection onto the manifold, TriangCFM's $\beta$-recall drops by up to 0.76 and F1 by up to 0.69, demonstrating that post-processing projection is practically unusable for fMRI correlation matrices. DiffeoCFM fundamentally avoids this issue through the diffeomorphism.

### Neurophysiological Plausibility Validation
- **fMRI connectome**: The class-conditional Fréchet mean connectomes generated by DiffeoCFM on ADNI are consistent with real data—the Alzheimer's disease group exhibits the characteristic pattern of reduced interhemispheric and anterior-posterior connectivity.
- **EEG topographies**: CSP spatial filters derived from DiffeoCFM-generated data concentrate in contralateral sensorimotor regions within the $\alpha$ (8–12 Hz) and $\beta$ (13–30 Hz) frequency bands, closely matching the discriminative patterns of real EEG motor imagery data.

## Highlights & Insights

- **Mathematical elegance**: Three equivalence propositions (training, continuous sampling, and discrete integration) rigorously establish that Euclidean CFM is equivalent to Riemannian CFM, yielding a theoretically complete framework.
- **Unified framework**: The same framework handles both SPD matrices and correlation matrices by simply swapping the diffeomorphism $\phi$, making it the only method currently supporting generation for both matrix types.
- **Computational efficiency**: By avoiding all manifold-specific operations (geodesics, Riemannian exponential maps, parallel transport), DiffeoCFM trains 8× faster and samples 10× faster than RiemCFM without quality degradation.
- **Constructive constraint guarantee**: Mapping back to the manifold via $\phi^{-1}$ ensures that generated samples inherently satisfy SPD/correlation matrix constraints, with no post-processing projection required.
- **Large-scale experiments**: Coverage of 5 datasets, 4,600+ scans, and 30,000+ EEG trials constitutes the most comprehensive evaluation to date in the brain connectivity matrix generation literature.

## Limitations & Future Work

- **Reliance on the existence of a global diffeomorphism**: While natural diffeomorphisms exist for SPD and correlation matrices, this approach does not extend to compact manifolds such as the Stiefel manifold, limiting the generalizability of the method.
- **Curse of dimensionality**: The manifold dimension grows quadratically with the number of brain regions $d$ (as $d(d-1)/2$), and sample complexity increases exponentially for high-resolution brain parcellations (e.g., 400+ regions).
- **Sensitivity to connectivity definition**: Only OAS-estimated covariance/correlation matrices are evaluated; the impact of alternative definitions such as partial correlations or graphical Lasso precision matrices remains unexplored.
- **Geometry-agnostic evaluation metrics**: $\alpha$-precision and $\beta$-recall are based on One-Class SVM and do not account for Riemannian geometric structure, potentially missing neurophysiologically relevant subtle differences.
- **Limited data scale**: The maximum number of brain regions evaluated is approximately 80; scalability to higher-dimensional parcellations (e.g., Schaefer 400) has not been verified.

## Related Work & Insights

- **Riemannian CFM [Chen & Lipman 2024]**: Performs Riemannian CFM directly under the affine-invariant metric, achieving the highest geometric fidelity but at prohibitive computational cost (training: 2,754 s vs. DiffeoCFM's 320 s); supports only SPD matrices, not correlation matrices.
- **SPD-DDPM [Huang & Han 2023]**: Requires a dedicated SPDNet architecture and suffers from extremely slow training.
- **CorrGAN [Marti 2020]**: Euclidean GAN with post-processing projection; projection severely degrades generation quality.
- **TriangCFM/TriangDDPM**: Directly model lower-triangular elements; $\alpha,\beta$-F1 drops by 0.5–0.7 after projection.
- **Normalizing Flows on Lie Groups [Falorsi et al. 2019]**: Learns probability densities on Lie groups via reparameterization; can be viewed as an early conceptual precursor to DiffeoCFM.
- **DiffeoGauss** (ablation baseline in this paper): Also employs a diffeomorphism but fits only a Gaussian; $\beta$-recall is acceptable but $\alpha$-precision is extremely low, demonstrating the indispensability of CFM's nonlinear modeling capacity.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of pullback geometry and CFM is elegant, and the equivalence proofs are rigorous; however, the core idea (performing Euclidean generative modeling after a change of variables) is relatively intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Five real-world datasets, multiple baselines, and a three-tier evaluation system encompassing quality, classification, and neurophysiological plausibility.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear, figures and tables are professional, and experimental descriptions are detailed.
- Value: ⭐⭐⭐⭐ — Provides a practical state-of-the-art solution for brain connectivity matrix generation with clear application prospects such as privacy-preserving data sharing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Continual Learning for fMRI-Based Brain Disorder Diagnosis via Functional Connectivity Matrices Generative Replay](../../CVPR2026/medical_imaging/forge_continual_learning_for_fmri_based_brain_disorder_diagnosis.md)
- [\[NeurIPS 2025\] GeoDynamics: A Geometric State-Space Neural Network for Understanding Brain Dynamics on Riemannian Manifolds](geodynamics_a_geometric_state-space_neural_network_for_understanding_brain_dynam.md)
- [\[NeurIPS 2025\] Surf2CT: Cascaded 3D Flow Matching Models for Torso 3D CT Synthesis from Skin Surface](surf2ct_cascaded_3d_flow_matching_models_for_torso_3d_ct_synthesis_from_skin_sur.md)
- [\[NeurIPS 2025\] Variational Autoencoder with Normalizing Flow for X-ray Spectral Fitting](variational_autoencoder_with_normalizing_flow_for_x-ray_spectral_fitting.md)
- [\[NeurIPS 2025\] Modeling X-ray Photon Pile-up with a Normalizing Flow](modeling_x-ray_photon_pile-up_with_a_normalizing_flow.md)

</div>

<!-- RELATED:END -->
