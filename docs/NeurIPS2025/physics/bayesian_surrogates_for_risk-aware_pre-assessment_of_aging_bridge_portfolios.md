---
title: >-
  [Paper Note] Bayesian Surrogates for Risk-Aware Pre-Assessment of Aging Bridge Portfolios
description: >-
  [NeurIPS 2025][Physics & Scientific Computing][Bayesian Neural Networks] A Bayesian neural network (BNN)-based surrogate model is proposed to replace expensive nonlinear finite element analysis (NLFEA), enabling rapid, uncertainty-aware structural safety pre-assessment of aging bridge portfolios. In a real-world railway case study, the approach saves approximately $370,000 per bridge.
tags:
  - "NeurIPS 2025"
  - "Physics & Scientific Computing"
  - "Bayesian Neural Networks"
  - "Structural Engineering"
  - "Uncertainty Quantification"
  - "Surrogate Models"
  - "Infrastructure Assessment"
date: 2026-05-08
content_hash: 41a4ac047f30e617
---

# Bayesian Surrogates for Risk-Aware Pre-Assessment of Aging Bridge Portfolios

**Conference**: NeurIPS 2025  
**arXiv**: [2509.25031](https://arxiv.org/abs/2509.25031)  
**Code**: None  
**Area**: Scientific Computing / Bayesian Deep Learning  
**Keywords**: Bayesian Neural Networks, Structural Engineering, Uncertainty Quantification, Surrogate Models, Infrastructure Assessment

## TL;DR
A Bayesian neural network (BNN)-based surrogate model is proposed to replace expensive nonlinear finite element analysis (NLFEA), enabling rapid, uncertainty-aware structural safety pre-assessment of aging bridge portfolios. In a real-world railway case study, the approach saves approximately $370,000 per bridge.

## Background & Motivation

**Background**: A large number of bridges worldwide are aging, with many having exceeded their design service life. Asset managers must decide which bridges require strengthening or replacement. Current assessment practice follows a levels-of-approximation approach—starting with conservative simplified analyses, then commissioning detailed NLFEA if compliance cannot be verified.

**Limitations of Prior Work**: Although NLFEA is accurate, each bridge requires approximately 25 minutes of computation and substantial manual modeling effort, making it unscalable to the portfolio level. Moreover, the benefit of NLFEA is unknown a priori—it sometimes only confirms, at high cost, what the simplified method already concluded.

**Key Challenge**: A fundamental tension exists between assessment accuracy and scalability. Managers must either apply simplified analyses to all bridges (potentially triggering unnecessary strengthening) or selectively commission NLFEA without knowing which bridges would benefit most.

**Goal**: To rapidly predict the structural compliance factor for each bridge without running expensive NLFEA, while quantifying predictive uncertainty to enable risk-based prioritization.

**Key Insight**: BNNs are employed as surrogates for NLFEA, leveraging a parametric simulation pipeline to generate large-scale training data, with posterior uncertainty guiding decision-making.

**Core Idea**: Train a BNN surrogate to predict the probability distribution of bridge compliance factors, and implement a three-color triage strategy (red/orange/green) for rapid portfolio-level screening.

## Method

### Overall Architecture
The input is a bridge parameter vector $\mathbf{x} \in \mathbb{R}^k$ (geometry, materials, loads, etc.). The BNN surrogate outputs the predictive distribution $p(\hat{y}|\mathbf{x}, \mathcal{D})$ of the compliance factor $\eta$, with mean $\mu$ as the point estimate and standard deviation $\sigma$ quantifying epistemic uncertainty. A triage strategy classifies bridges into three categories: red ($\mu < 1$, requiring immediate detailed analysis), orange ($\mu > 1$ but $\mu - 2\kappa\sigma < 1$, recommending refined analysis), and green (high-confidence compliance).

### Key Designs

1. **Parametric NLFEA Simulation Pipeline**:

    - **Function**: Automatically generates reinforced concrete frame bridge models from parameter vectors and executes NLFEA.
    - **Mechanism**: Links Rhino/Grasshopper (geometry generation) with Ansys Mechanical APDL (NLFEA), employing the Cracked Membrane Model as a user material for layered shell elements, with workflow automation via the StrucEngLib plugin. Each simulation produces three global compliance factors $\boldsymbol{\eta} = (\eta_{M,c,\min}, \eta_{M,s,\min}, \eta_{V,\min})$, corresponding to safety margins for concrete bending, steel bending, and shear, respectively ($\eta \geq 1$ indicates compliance).
    - **Design Motivation**: Large-scale high-fidelity training data are required, but manual modeling is not scalable; hence parametric automated generation is necessary.

2. **Adaptive Sampling Strategy**:

    - **Function**: Intelligently allocates the simulation budget across the parameter space.
    - **Mechanism**: Latin hypercube sampling (LHS) is first applied to cover the full input space, followed by adaptive resampling guided by kernel density estimation, concentrating samples near the safety-critical region $\eta \approx 1$. Parameter ranges are derived from analysis of the Swiss Federal Railways (SBB) bridge database.
    - **Design Motivation**: Predictive accuracy near the decision boundary $\eta = 1$ is most critical; adaptive sampling maximizes data density in this region under a limited simulation budget. Approximately 11k samples are generated in total.

3. **BNN Surrogate Model and Calibration**:

    - **Function**: Predicts the distribution of compliance factors using a BNN.
    - **Mechanism**: Stochastic variational inference is used, approximating the weight posterior as a product of independent Gaussians (mean-field approximation), with prior $p(\theta) = \mathcal{N}(0, 0.1I)$. The loss is derived from the ELBO and includes KL divergence regularization and a weighted MSLE: $L(\theta, D) = \lambda \cdot \text{KL}(q_\phi(\theta) \| p(\theta)) + \text{wMSLE}(y, \mu(\hat{y}))$, where wMSLE applies heavier penalties for values near $\eta \approx 1$. Twenty forward passes estimate mean and uncertainty during training; 1,000 passes are used at inference.
    - **Post-hoc Calibration**: Raw BNN outputs are slightly overconfident; posterior scaling factors $\kappa$ (steel: 1.3, concrete: 1.7, shear: 2.8) are applied to align nominal coverage with empirical coverage.

4. **Reduced-Dimension Input Deployment**:

    - **Function**: Supports prediction using only a small number of key parameters.
    - **Mechanism**: Kernel SHAP ranks feature importance; the top five most important and readily available features (span, slab thickness, height, width) are selected. Missing features are handled by LHS sampling followed by propagation through the BNN.
    - **Design Motivation**: Complete structural parameter information is often unavailable at early screening stages; reduced-dimension deployment enables pre-assessment under limited information.

### Loss & Training
The core loss decomposes as an ELBO: KL divergence constrains weight distributions toward the prior, while weighted MSLE focuses learning on the decision boundary. Three independent BNNs are trained for the three compliance factors, followed by post-hoc posterior standard deviation scaling for calibration.

## Key Experimental Results

### Main Results

| Compliance Factor | RMSE | MAPE | Calibration Bias (CB) |
|---------|------|------|-------------|
| $\eta_{M,s,\min}$ (Steel Bending) | 0.10 | 4.8% | -0.8 |
| $\eta_{M,c,\min}$ (Concrete Bending) | 0.40 | 37.5% | -0.9 |
| $\eta_{V,\min}$ (Shear) | 0.60 | 46.7% | -0.6 |

Results are reported within the safety-critical region $\eta \in [0.5, 1.5]$; calibration bias is reported after posterior scaling.

### Ablation Study

| Configuration | Key Performance | Notes |
|------|---------|------|
| Full-input BNN | RMSE 0.10 (steel) | Highest accuracy with all parameters |
| Reduced input (Top-5 features) | Steel/shear well maintained | Geometry-dominated factors less affected |
| Reduced input (Top-5 features) | Concrete accuracy degrades noticeably | Concrete compliance depends on material parameters |
| Without posterior scaling | CB > 0 (overconfident) | Insufficient prediction interval coverage |
| With posterior scaling | CB < 0 (slightly conservative) | Preferable in safety-critical settings |

### Key Findings
- The steel bending compliance factor is predicted most reliably (MAPE 4.8%), while shear is most challenging (MAPE 46.7%) due to its brittle, locally dominated behavior.
- Posterior scaling is critical for calibration; the largest scaling factor for shear (2.8) reflects the inherent physical uncertainty in shear behavior.
- The reduced-dimension model maintains good accuracy for geometry-dominated factors but degrades substantially for the concrete factor, which is sensitive to material parameters.
- In the real-world case study, the surrogate correctly flagged shear as the critical verification item; subsequent NLFEA confirmed compliance, avoiding approximately $370,000 in unnecessary strengthening.

## Highlights & Insights
- **Domain-Adaptive Loss Design**: Combining wMSLE with heavy penalties near $\eta \approx 1$ and adaptive sampling concentrates model accuracy precisely at the decision boundary. This principle of "allocating accuracy where decisions are made" is transferable to any threshold-based decision problem.
- **Posterior Scaling Calibration**: A simple constant scaling factor $\kappa$ substantially improves BNN calibration, and deliberately biases toward under-confidence in safety-critical settings—a more pragmatic approach than pursuing perfect calibration.
- **Progressive Information Acquisition**: Ranking feature importance via SHAP, performing initial prediction with few features, and acquiring additional parameters only when uncertainty is high constitutes an elegant human-in-the-loop progressive assessment strategy.

## Limitations & Future Work
- The current approach targets only reinforced concrete frame bridges; extension to arch bridges, prestressed bridges, and other types would require rebuilding the pipeline and retraining.
- Shear compliance factor prediction remains insufficiently accurate (MAPE 47%), potentially requiring more expressive model architectures or physics-informed constraints.
- Posterior scaling uses a global constant, whereas different input regions may warrant different scaling factors—learning heteroscedastic uncertainty could be explored.
- Training data are generated from parametric simulations rather than real measurements, so the model cannot capture the simulation-to-reality gap inherent in NLFEA.

## Related Work & Insights
- **vs. Gaussian Processes**: GPs also provide uncertainty estimates, but scale poorly to high-dimensional inputs (10+ parameters) and large datasets (11k samples); BNNs are better suited to this regime.
- **vs. Deep Ensembles**: Ensemble methods offer frequentist uncertainty estimates, but BNNs provide a more principled epistemic uncertainty quantification from a Bayesian perspective.
- **vs. Deterministic Surrogates**: Most ML methods in structural engineering produce deterministic predictions without uncertainty quantification, which is unacceptable in safety-critical applications.

## Rating
- Novelty: ⭐⭐⭐ BNN surrogates are not new, but the application scenario of bridge portfolio assessment and the end-to-end pipeline design represent genuine contributions.
- Experimental Thoroughness: ⭐⭐⭐ Includes real-world case validation and calibration analysis, though ablation experiments lack systematic coverage.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with clear articulation of problem motivation and practical value.
- Value: ⭐⭐⭐⭐ Demonstrates clear practical impact; the $370,000 savings per bridge is a compelling illustration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GyroSwin: 5D Surrogates for Gyrokinetic Plasma Turbulence Simulations](gyroswin_5d_surrogates_for_gyrokinetic_plasma_turbulence_simulations.md)
- [\[NeurIPS 2025\] 3DID: Direct 3D Inverse Design for Aerodynamics with Physics-Aware Optimization](3did_direct_3d_inverse_design_for_aerodynamics_with_physics-aware_optimization.md)
- [\[ICLR 2026\] Geometric Autoencoder Priors for Bayesian Inversion: Learn First Observe Later](../../ICLR2026/physics/geometric_autoencoder_priors_for_bayesian_inversion_learn_first_observe_later.md)
- [\[ICLR 2026\] Uncertainty-Aware Diagnostics for Physics-Informed Machine Learning](../../ICLR2026/physics/uncertainty-aware_diagnostics_for_physics-informed_machine_learning.md)
- [\[ICLR 2026\] SAQ: Stabilizer-Aware Quantum Error Correction Decoder](../../ICLR2026/physics/saq_stabilizer-aware_quantum_error_correction_decoder.md)

</div>

<!-- RELATED:END -->
