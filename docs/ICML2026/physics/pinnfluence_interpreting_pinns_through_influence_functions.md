---
title: >-
  [Paper Note] PINNfluence: Interpreting PINNs Through Influence Functions
description: >-
  [ICML 2026][Physics & Scientific Computing][PINN Diagnostics] This work extends Influence Functions (IF), a training data attribution method, to Physics-Informed Neural Networks (PINNs). The proposed framework…
tags:
  - "ICML 2026"
  - "Physics & Scientific Computing"
  - "PINN Diagnostics"
  - "Influence Functions"
  - "Training Data Attribution"
  - "Loss Decomposition"
  - "Temporal Causality Metrics"
date: 2026-05-08
content_hash: 2ddd4109628bbd05
---

# PINNfluence: Interpreting PINNs Through Influence Functions

**Conference**: ICML 2026  
**arXiv**: [2409.08958](https://arxiv.org/abs/2409.08958)  
**Code**: https://github.com/aleks-krasowski/pinnfluence  
**Area**: Scientific Computing / Physics-Informed Neural Networks / Interpretability  
**Keywords**: PINN Diagnostics, Influence Functions, Training Data Attribution, Loss Decomposition, Temporal Causality Metrics

## TL;DR
This work extends Influence Functions (IF), a training data attribution method, to Physics-Informed Neural Networks (PINNs). The proposed framework, PINNfluence, utilizes linearized leave-one-out perturbations to attribute PINN predictions, losses, and physical quantities to individual training points and loss components simultaneously. Based on this, a set of diagnostic metrics (loss component ratios, cancellation scores, temporal causality indicators, etc.) is constructed. These metrics reliably distinguish between "well-trained" and "failed" PINNs across five time-dependent PDEs, providing structural diagnostics that traditional residual analysis fails to capture.

## Background & Motivation
**Background**: PINNs (Raissi 2019) embed PDE residuals as soft constraints in the NN training objective, learning a function $\phi(x;\theta)$ to approximate the solution of an IBVP using a unified loss $\mathcal{L}(θ) = \lambda_\text{pde} \sum L_\text{pde} + \sum_k \lambda_{bc,k} \sum L_{bc,k}$. While widely applied in fluids, electromagnetism, and optics, training failures (propagation failure, dominant initial conditions, ill-conditioned loss landscapes) are common, and the community often relies on "phenomenological diagnostics."

**Limitations of Prior Work**: (1) Most understanding of PINNs is based on **training dynamics** (gradient flow analysis, NTK, loss reweighting) rather than **post-hoc interpretability**—identifying "why this prediction comes from this specific training data" or "which loss term dominates it." (2) Traditional verification fails in PINNs: a low training loss **does not equal** a correct PDE solution; PINNs often converge to trivial solutions with low residuals but incorrect physics (Daw 2023, Rohrhofer 2023). (3) While XAI is mature in vision/NLP (LRP, IF, SAE), **no specialized post-hoc explanation methods exist for PINNs**.

**Key Challenge**: PINN failures are often **structural** (over-reliance on initial conditions, failed boundary propagation, information disconnection at certain time steps), but scalar metrics like total loss or residuals flatten these structures. To reveal these structures, one must decompose attribution across three axes: "which training point," "which loss term," and "which spatio-temporal region."

**Goal**: (1) Generalize the classical Koh & Liang (2017) Influence Functions to PINN **composite losses** and **arbitrary differentiable targets**; (2) Provide influence-based diagnostic metrics capable of distinguishing "well-trained" from "failed" models at a structural level; (3) Verify the stability of these metrics across various PDEs and optimizers.

**Key Insight**: PINN losses are naturally weighted sums of multiple components, and IF is **linear** with respect to the loss terms. This implies that a single Hessian-vector product (HVP) calculation can "automatically" decompose influence into each $L_i$. By combining "training point → region" and "region → training point" attribution directions, one can construct attribution maps at point-to-point, point-to-region, and region-to-region granularities.

**Core Idea**: Extend IF from "loss-to-loss" attribution to the attribution of "any differentiable composite loss $L$ to any differentiable output $f$": $\operatorname{Inf}_{\theta_0}^{L\to f}(x,z)=-\nabla_\theta f(z;\theta_0)^\top \mathcal{H}_{\theta_0}^{-1}\nabla_\theta L(x;\theta_0)$. Leverage **linearity and additivity** to directly decompose influence into individual loss components and spatio-temporal regions.

## Method
PINNfluence is a post-hoc analysis framework that does not modify the PINN training process. Given a **pre-trained** PINN $\phi(\cdot;\theta_0)$ and its training set $\mathcal{X}=\mathcal{X}_\text{pde}\cup\bigcup_k\mathcal{X}_{bc,k}$, it computes $\operatorname{Inf}$ to aggregate diagnostic metrics.

### Overall Architecture
- **Input**: A trained PINN $\phi$, training set $\mathcal{X}$ (PDE collocation points and IC/BC points), target quantity of interest $f$ (prediction $\hat{u}$, a loss component $L_i$, or a physical observable), and test points/regions.
- **Mechanism**: Couples the Inverse Hessian-Vector Product (IHVP) $\mathcal{H}_{\theta_0}^{-1}\nabla_\theta L(x;\theta_0)$ with $\nabla_\theta f(z;\theta_0)$. It utilizes low-rank Arnoldi approximation + HVP to avoid explicit Hessian construction.
- **Three Granularities**:
  - Point-to-Point: $\operatorname{Inf}_{\theta_0}^{L\to f}(x,z)$;
  - Point-to-Region / Region-to-Point: Summation over $z$ or $x$ within a region;
  - Region-to-Region: Double summation with normalization to obtain ratio metrics.
- **Output**: (1) Point-to-point influence heatmaps; (2) Loss component decomposition ratios $r_{L_i}$ and cancellation scores $\kappa$; (3) Spatio-temporal metrics, such as the temporal causality index $\eta$.

### Key Designs

1. **Extension from "Loss-to-Loss" to "Arbitrary Differentiable Targets + Composite Losses"**:
    - **Function**: Enables IF to explain the effect of adding/removing a point or adjusting weights on **any output quantity** $f(z;\theta)$ at a test point $z$, including the prediction itself or specific loss components.
    - **Mechanism**: Defines $\operatorname{Inf}_{\theta_0}^{L\to f}(x,z)=-\nabla_\theta f(z;\theta_0)^\top \mathcal{H}_{\theta_0}^{-1}\nabla_\theta L(x;\theta_0)$ (Definition 2.1) and proves it first-order approximates leave-one-out effects: $f(z;\theta_1)-f(z;\theta_0)=\pm\operatorname{Inf}^{L\to f}(x^\pm,z)/N+O(1/N^2)$ (Theorem 2.2). Theoretical assumptions are relaxed from "strict minima + strong convexity" to "non-degenerate stationary points + invertible Hessian" to account for PINN training realities.
    - **Design Motivation**: PINN composite losses $\mathcal{L}=\lambda_\text{pde}L_\text{pde}+\sum_k\lambda_{bc,k}L_{bc,k}$ must be treated as decomposable objects to answer whether predictions are dominated by BCs, ICs, or residuals.

2. **Loss Component Decomposition + Cancellation Score $\kappa$**:
    - **Function**: Linearly decomposes the total influence at any pair $(x,z)$ into individual loss components $r_{L_i}(x,z)\in[0,1]$. The cancellation score $\kappa(x,z)\in[0,1]$ characterizes whether influences from different components **reinforce** ($\kappa\approx 0$) or **cancel** ($\kappa\approx 1$) each other.
    - **Mechanism**: Linearity of IF (Corollary 2.3) yields $\operatorname{Inf}^{\sum_i\alpha_i L_i\to f}=\sum_i\alpha_i \operatorname{Inf}^{L_i\to f}$. Ratios are defined as $r_{L_i}(x,z)=\frac{|\operatorname{Inf}^{L_i\to f}|}{\sum_j|\operatorname{Inf}^{L_j\to f}|}$, and $\kappa(x,z)=1-\frac{|\sum_j \operatorname{Inf}^{L_j\to f}|}{\sum_j|\operatorname{Inf}^{L_j\to f}|}$.
    - **Design Motivation**: Failures where ICs dominate or BCs are ignored are **structural info invisible to residual analysis**. $\kappa$ quantifies the reliability of the decomposition: high $\kappa$ suggests misleading $r_{L_i}$ due to cancellation.

3. **Temporal Causality Index $\eta$ + Region-Normalized Metric $\rho$**:
    - **Function**: Encodes the temporal relationship between a training point $x_t$ and a test point $z_t$ to quantify if the PINN "looks from the past to the future."
    - **Mechanism**: $\eta_{\theta_0}^{L\to f}(R_\text{tr},R_\text{te})=1-\frac{1}{|R_\text{te}|}\sum_{z\in R_\text{te}}\frac{\sum_{x\in R_\text{tr}: x_t\le z_t}|\operatorname{Inf}^{L\to f}(x,z)|}{\sum_{x\in\mathcal{X}_\text{train}}|\operatorname{Inf}^{L\to f}(x,z)|}$. A small $\eta$ indicates influence primarily from the past ("causal alignment"). $\rho_{\theta_0}^{L\to f}(R_\text{tr},R_\text{te})$ is the ratio of influence from a subset region vs. the whole domain.
    - **Design Motivation**: Empirically, well-trained models show $\eta \approx \bar{t}$ (near-uniform influence), while failed models exhibit excessive "past dominance" (unnatural IC persistence), indicating PINNs learn solutions **globally** rather than sequentially.

### Loss & Training
**No new training loss is introduced**—PINNfluence performs post-hoc analysis. Key implementation details: $\mathcal{H}^{-1}$ is estimated using PyHessian-style HVP combined with Arnoldi low-rank approximation (Schioppa 2022) to avoid $O(p^2)$ costs. Metrics are validated using PBRF (Bae 2022) for reliability and across specialized optimizers like NNCG and SOAP.

## Key Experimental Results

Setup: 5 time-dependent PDEs (Heat, Allen-Cahn, Burgers', Wave, Drift-Diffusion) and 2 steady-state PDEs (Poisson, Navier-Stokes). Each problem includes "well-trained" vs. "poorly-trained" configurations (10 seeds each).

### Main Results

| Problem | $\bar{t}$ (Baseline) | Well-trained $\eta$ (pred) | Poorly-trained $\eta$ (pred) | Diagnostic Conclusion |
| :--- | :---: | :---: | :---: | :--- |
| Heat | 0.46 | 0.33 ± 0.02 | 0.26 ± 0.06 | Failed model biased toward past (IC dominance) |
| Allen-Cahn | 0.43 | 0.50 ± 0.02 | 0.32 ± 0.05 | Failed $\eta$ significantly lower |
| Burgers' | 0.43 | 0.41 ± 0.02 | 0.28 ± 0.02 | Same as above |
| Drift-Diffusion | 0.46 | 0.46 ± 0.04 | 0.21 ± 0.06 | IC influence overwhelming; correlates with propagation failure |
| Wave | 0.43 | 0.41 ± 0.03 | **0.11 ± 0.02** | Largest gap; failed model completely anchored by IC |

→ Across all 5 problems, **$\eta$ for failed models is significantly lower than for well-trained models**, with small standard deviations across seeds.

### Ablation Study

| Dimension | Experimental Setup | Key Finding |
| :--- | :--- | :--- |
| Loss Ratio $\bar{r}_{L_\text{ic}}$ | 5 PDEs × 50 time bins | Well-trained: IC share decays over time; Poorly-trained: IC share remains high or plateaus, coinciding with trivial outputs. |
| Training Data Scaling | 5 PDEs | PINNfluence metrics transition smoothly from "failure mode" to "success mode" as points increase; the curve shape is problem-specific. |
| Optimizer Invariance | Adam / NNCG / SOAP | The distinction pattern between good/bad models using $\eta$ and $\bar{r}_{L_i}$ is robust across different optimizers. |
| Hessian Reliability | Arnoldi vs. PBRF vs. Grad-only | Arnoldi faithfully recovers the projected inverse Hessian and aligns with PBRF results; it significantly outperforms pure gradient baselines. |
| Cancellation Score $\kappa$ | All PDEs | Most test points exhibit low $\kappa$, making $r_{L_i}$ robust; high $\kappa$ points indicate "competing constraints" and potential loss weight issues. |

### Key Findings
- **Structural Signals >> Residual Signals**: Residuals show "that" a model failed, but PINNfluence shows "why" (e.g., IC influence over-persisting or BCs being ignored).
- **Anti-intuitive Causality**: Well-trained PINNs do **not** strictly follow physical causality (near-uniform $\eta$); failed PINNs are "causally aligned" due to IC overfitting. This highlights that PINNs optimize solutions globally.
- **Cross-task Consistency**: The IC influence decay pattern consistently distinguishes model quality across diverse PDEs, suggesting universal diagnostic potential.
- **Robustness**: Despite the reputation of IF being fragile in deep networks, it remains reliable in PINNs when using low-rank Arnoldi + PBRF validation.

## Highlights & Insights
- **Generalizing IF to arbitrary $L \to f$ is the pivotal step**: It transforms IF from a classifier debugging tool into a scientific computing diagnostic tool accessible via a single IHVP line. This is applicable to any composite-objective SciML model (Neural Operators, DeepONets).
- **$\kappa$ as a Self-check**: It quantifies "attribution reliability" by accounting for sign cancellation in linear decomposition—a detail often missed in LRP or Shapley-based methods.
- **Structural Diagnosis $\neq$ Causality**: Sensitivities represent "influence," not "physical causality." This distinction is crucial for avoiding over-interpretation in scientific applications.
- **Nature of PINNs**: The finding that well-trained PINNs are not "causal" suggests they behave as global variational approximators rather than sequential integrators.

## Limitations & Future Work
- **First-order Approximation**: The $O(1/N^2)$ residual means large perturbations or extreme ill-conditioning may limit accuracy.
- **Hessian Ill-conditioning**: While low-rank Arnoldi helps, singular directions in complex PDEs may still amplify errors.
- **Computational Cost**: Costs scale linearly with $|\mathcal{X}_\text{train}| \times |R_\text{te}|$, which may be prohibitive for large-scale 3D time-dependent domains.
- **Absolute Value Aggregation**: Using $|\operatorname{Inf}|$ prevents signal cancellation but loses information on whether a point "promotes" or "inhibits" a prediction.
- **Future Directions**: (1) Diagnostic-driven training (resampling high-influence points); (2) Automated loss weight tuning using $\bar{r}_{L_i}$; (3) Extension to inverse problems and non-time-dependent systems.

## Related Work & Insights
- **vs. Koh & Liang (2017)**: Original IF did "loss-to-loss" for strongly convex minima; PINNfluence handles composite losses and non-degenerate stationary points for any differentiable target.
- **vs. Training Dynamics (Wang 2022, Rathore 2024)**: These study *why* training fails; PINNfluence studies *how* the trained model fails from a data perspective.
- **vs. Neural Operator Interpretability**: Where others use SAE/Probing for internal representations, PINNfluence focuses on "data + constraints," providing a complementary post-hoc view.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First data attribution framework for PINNs; universal generalization of IF.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid across multiple PDEs and optimizers; lacks 3D large-scale cases.
- Writing Quality: ⭐⭐⭐⭐⭐ Mature structure from theory to metrics to diagnostics.
- Value: ⭐⭐⭐⭐⭐ Bridge the gap in SciML interpretability with significant downstream potential for training optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Supervised Metric Regularization Through Alternating Optimization for Multi-Regime PINNs](../../ICLR2026/physics/supervised_metric_regularization_through_alternating_optimization_for_multi-regi.md)
- [\[ICML 2026\] Generative Neural Operators Through Diffusion Last Layer](generative_neural_operators_through_diffusion_last_layer.md)
- [\[NeurIPS 2025\] One-Shot Transfer Learning for Nonlinear PDEs with Perturbative PINNs](../../NeurIPS2025/physics/oneshot_transfer_learning_nonlinear_pdes_perturbative_pinns.md)
- [\[ICLR 2026\] HyperKKL: Enabling Non-Autonomous State Estimation through Dynamic Weight Conditioning](../../ICLR2026/physics/hyperkkl_enabling_non-autonomous_state_estimation_through_dynamic_weight_conditi.md)
- [\[NeurIPS 2025\] Towards Universal Neural Operators through Multiphysics Pretraining](../../NeurIPS2025/physics/towards_universal_neural_operators_through_multiphysics_pretraining.md)

</div>

<!-- RELATED:END -->
