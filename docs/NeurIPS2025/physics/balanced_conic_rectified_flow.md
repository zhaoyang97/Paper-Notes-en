---
title: >-
  [Paper Note] Balanced Conic Rectified Flow
description: >-
  [NeurIPS 2025][Physics & Scientific Computing][rectified flow] To address the distribution drift induced by the reflow step in k-rectified flow…
tags:
  - "NeurIPS 2025"
  - "Physics & Scientific Computing"
  - "rectified flow"
  - "reflow"
  - "conic interpolation"
  - "Slerp"
  - "distribution drift"
date: 2026-05-08
content_hash: b1ec54fe1f0c9cfd
---

# Balanced Conic Rectified Flow

**Conference**: NeurIPS 2025
**arXiv**: [2510.25229](https://arxiv.org/abs/2510.25229)  
**Code**: [Project Page](https://grainsack.github.io/BC_rectified_flow_project_page/)  
**Area**: Image Generation / Flow Matching
**Keywords**: rectified flow, reflow, conic interpolation, Slerp, distribution drift

## TL;DR

To address the distribution drift induced by the reflow step in k-rectified flow, this paper proposes conic reflow: constructing conic supervisory trajectories from the inverted noise of real images and their Slerp-perturbed neighbors, substantially reducing the number of required fake pairs while achieving superior generation quality and straighter ODE trajectories.

## Background & Motivation

**Background**: Rectified Flow learns an ODE velocity field from noise to data for efficient generation; k-rectified flow iteratively applies reflow to straighten trajectories, enabling few-step or even single-step generation. Current SOTA models (Flux, SD3, AuraFlow) adopt 1-rectified flow with ~30 NFE.

**Limitations of Prior Work**:
   - Reflow requires large numbers of fake pairs (4M pairs in the original method), incurring substantial generation cost.
   - Fake pairs are produced by imperfect models and inherently deviate from the true data distribution, introducing systematic bias into reflow supervision signals.
   - Repeated reflow rounds cause error accumulation, progressively driving the model away from the real data distribution.

**Key Challenge**: Reflow aims to straighten trajectories for few-step generation, yet the fake-pair supervision it relies upon introduces distribution drift, causing generation quality under full-step inference to actually degrade—creating a fundamental tension between trajectory straightening and distributional fidelity.

**Goal**: To expose the distribution drift phenomenon caused by reflow and to design a new reflow strategy that straightens trajectories while maintaining fidelity to the true data distribution.

**Key Insight**: Reconstruction error is used to quantitatively reveal drift—fake images exhibit substantially lower reconstruction error than real images, indicating that the model overfits the fake distribution and departs from the real one. The paper then proposes using inverted noise of real images combined with Slerp perturbations to construct "conic" supervision as a corrective mechanism.

**Core Idea**: Construct conic supervisory trajectories from the inverted noise of real images and their Slerp neighborhood, and alternate training between real pairs and fake pairs to simultaneously correct distribution drift and ensure trajectory straightness.

## Method

### Overall Architecture

Balanced Conic Rectified Flow comprises three key components:
1. **Real pair**: Invert a real image $X_1$ using the trained model to obtain noise $Z_{0,R} = v^{-1}(X_1)$, forming the pair $(Z_{0,R}, X_1)$.
2. **Conic reflow**: Apply Slerp perturbations to the inverted noise, expanding the supervisory signal from a single trajectory to a conic neighborhood.
3. **Balanced training**: Alternate between real pairs (conic reflow) and fake pairs (standard reflow) to balance distributional fidelity and domain coverage.

### Key Designs

1. **Real Pair Construction**:

    - Mechanism: Instead of relying on fake pairs of the form $(Z_0, v(Z_0))$, real pairs are constructed from a real image $X_1$ and its inversion $Z_{0,R} = v^{-1}(X_1)$.
    - Design Motivation: The endpoint $Z_1 = v(Z_0)$ of fake pairs deviates from the true distribution $\pi_1$, whereas the endpoint of a real pair is drawn directly from real data, anchoring the target distribution.
    - The inversion exploits the deterministic nature of the ODE, requiring no additional stochasticity and remaining conceptually simple.

2. **Conic Reflow (Slerp Perturbation)**:

    - Mechanism: Slerp interpolation perturbations are applied to the inverted noise $Z_{0,R}$, extending supervision to its neighborhood.
    - Slerp formula: $$\text{Slerp}(Z_{0,R}, \epsilon, \zeta) = \frac{\sin((1-\zeta)\phi)}{\sin(\phi)} Z_{0,R} + \frac{\sin(\zeta\phi)}{\sin(\phi)} \epsilon$$
    - Here $\phi = \arccos(Z_{0,R} \cdot \epsilon)$, $\epsilon \sim \mathcal{N}(0, I)$, and $\zeta$ is the interpolation ratio.
    - Conic interpolation: $$\text{Conic}(X_1, \epsilon, \zeta, t) = t X_1 + (1-t) \cdot \text{Slerp}(Z_{0,R}, \epsilon, \zeta)$$
    - Multiple samples of $\epsilon$ and $\zeta$ form a bundle of conic trajectories, hence the name "conic."
    - Design Motivation: A single real-pair trajectory has limited coverage; Slerp preserves vector norms on the Gaussian hypersphere, better respecting the geometry of the noise space than Lerp.

3. **Slerp Noise Schedule**:

    - $\zeta_{\max}$ is determined automatically based on the point of maximum divergence between real and fake sample perturbation reconstruction errors.
    - The noise magnitude decays during training: $\zeta(t') = \zeta^{\max} \cdot \frac{2t'^2}{1+t'^2}$.
    - The inverted noise of real pairs is periodically refreshed to maintain alignment with the latest model.
    - $\zeta^{\max} = 0.13$ on CIFAR-10 and $\zeta^{\max} = 0.23$ on ImageNet.

4. **Balanced Training Strategy**:

    - In the first half of training, conic reflow (real pairs) and standard reflow (fake pairs) are applied in alternation.
    - In the second half, only standard reflow is used to compensate for the numerical asymmetry between fake and real pairs.
    - For example, with 100 total steps: $U_{\text{real}} = \{1, 3, 5, \ldots, 49\}$ and $U_{\text{fake}} = \{2, 4, \ldots, 50, 51, \ldots, 100\}$.

### Loss & Training

The overall training objective combines MSE losses from both pair types, with indicator functions $\chi_{\text{fake}}$ and $\chi_{\text{real}}$ ensuring that only one is active at each training step. Timestep $t$ is sampled from an exponential (U-shaped) distribution, emphasizing regions near $t \approx 0$ and $t \approx 1$ where trajectory crossings are most frequent. The fake-pair branch uses the standard rectified flow velocity-field MSE loss; the real-pair branch uses the velocity-field MSE loss evaluated at conic interpolation points.

## Key Experimental Results

### Main Results

Comparison of one-step and full-step generation quality on CIFAR-10:

| Method | NFE | IS ↑ | FID ↓ |
|--------|-----|------|-------|
| 1-Rectified Flow | 1 | 1.13 | 378 |
| 2-RF Original (+Distill) | 1 | 8.08 (9.01) | 12.21 (4.85) |
| **2-RF Ours (+Distill)** | **1** | **8.79 (9.11)** | **5.98 (4.16)** |
| RF++† | 1 | 8.87 | 4.43 |
| RF++† + Ours | 1 | 8.87 | **4.22** |
| 3-RF Original (+Distill) | 1 | 8.47 (8.79) | 8.15 (5.21) |
| **3-RF Ours (+Distill)** | **1** | **8.84 (8.96)** | **5.48 (4.68)** |
| 1-RF (RK45) | 127 | 9.60 | 2.58 |
| 2-RF Original (RK45) | 110 | 9.24 | 3.36 |
| **2-RF Ours (RK45)** | **104** | **9.30** | **3.24** |

Key figures: on 2-rectified flow, one-step FID drops from 12.21 to 5.98 (over 50% improvement), using only 300K fake pairs (vs. 4M originally, a 92.8% reduction).

Generalizability is further confirmed on ImageNet 64×64 and LSUN Bedroom 256×256:

| Dataset | Method | Euler 1-step FID | RK45 FID |
|---------|--------|-----------------|----------|
| ImageNet 64×64 | Original | 39.7 | 31.2 |
| ImageNet 64×64 | **Ours** | **37.8** | **28.2** |
| LSUN Bedroom 256 | Original | 139.98 | 24.76 |
| LSUN Bedroom 256 | **Ours** | **26.54** | **24.14** |

On LSUN, 1-step FID drops dramatically from 139.98 to 26.54.

### Ablation Study

Contribution of each component to 2-rectified flow performance (CIFAR-10, 1-step):

| Configuration | FID ↓ | IS ↑ | Curvature ↓ | Recon Real ↓ | Recon Fake |
|--------------|-------|------|-------------|-------------|------------|
| Original | 12.21 | 8.08 | 0.002837 | 0.033668 | 0.024106 |
| No Slerp (real pair only) | 6.60 | 8.57 | 0.002322 | 0.023380 | 0.020154 |
| Fixed Real Pair (no refresh) | 6.69 | 8.59 | 0.002313 | 0.020227 | 0.020607 |
| **Ours (full)** | **5.98** | **8.79** | **0.002295** | **0.019404** | 0.023139 |

Slerp vs. Lerp noise comparison:

| Method | IS ↑ | FID ↓ |
|--------|------|-------|
| **Slerp (Ours schedule)** | **8.72** | **6.63** |
| Slerp increasing | 8.48 | 6.64 |
| Slerp decreasing | 8.45 | 6.70 |
| Lerp linear interpolation | 8.46 | 7.50 |

### Key Findings

1. **Distribution drift is an inherent problem of reflow**: On the two-moons toy dataset, successive reflow rounds cause KL divergence to grow continuously, with the fake distribution progressively departing from the target.
2. **Real pairs effectively correct drift**: Adding real pairs alone (without Slerp) reduces FID from 12.21 to 6.60, demonstrating the strong anchoring effect of real data.
3. **Slerp outperforms Lerp**: Slerp preserves norms on the Gaussian hypersphere, achieving an FID approximately 0.9 lower than Lerp (6.63 vs. 7.50).
4. **Periodic refresh of real pairs is important**: Fixed (non-refreshed) inverted noise underperforms periodic updates (6.69 vs. 5.98).
5. **Recall improves substantially**: On ImageNet, 1-step Recall rises from 0.4604 to 0.5325, indicating improved coverage of the true distribution.
6. **Effective even at the extreme of k=4**: On 4-rectified flow, 1-step FID drops from 6.58 to 5.66 with lower curvature as well.

## Highlights & Insights

- **Rigorous analysis**: Beyond identifying distribution drift in reflow, the paper provides quantitative evidence through reconstruction error and perturbed reconstruction error gaps between real and fake samples, offering a novel analytical perspective.
- **Simple and plug-and-play design**: Conic reflow requires no architectural modifications or discriminators; it only changes how training data are constructed, and is directly compatible with existing methods such as RF++ and InstaFlow.
- **High efficiency**: The required number of fake pairs is reduced from 4M to 300K (only 7.2%), substantially lowering the computational cost of reflow.
- **Introduction of the IVD metric**: In addition to curvature, Initial Velocity Delta is introduced to assess initial velocity accuracy, more directly correlating with 1-step generation quality.
- **Adaptive determination of Slerp noise schedule**: $\zeta^{\max}$ is selected automatically based on the real/fake reconstruction error discrepancy, avoiding manual hyperparameter tuning.

## Limitations & Future Work

1. **Validation limited to unconditional generation**: The method has not been evaluated on text-conditional generation (e.g., SD3/Flux), which typically employs only 1-rectified flow; applicability requires further investigation.
2. **Insufficient real-pair coverage on ImageNet**: 60K real pairs provide limited coverage of ImageNet's class diversity; additional real pairs may yield further improvements.
3. **Inversion quality depends on the base model**: The quality of the noise endpoint $v^{-1}(X_1)$ in real pairs is contingent on the inversion accuracy of the preceding model; a poor model may produce low-quality real pairs.
4. **Resolution constraints**: Experiments reach at most 256×256 (LSUN); higher-resolution settings remain unvalidated.
5. **Insufficient theoretical grounding for the Slerp schedule**: The U-shaped schedule (large → small → large) is primarily empirical and lacks theoretical proof of optimality.

## Related Work & Insights

- **Rectified Flow / Flow Matching**: This work builds on the rectified flow framework of Liu et al., improving its reflow step; the approach generalizes to all flow matching models requiring reflow.
- **Rectified Flow++**: Improves RF by modifying the time distribution and loss function; the proposed method is orthogonal to RF++ and can be directly combined (RF++† + Ours = FID 4.22).
- **PerFlow**: Stabilizes reflow via piecewise linear paths, contrasting with the continuous conic paths proposed here.
- **Self-consuming Training**: The distribution drift identified in this work is related to model collapse in the self-consuming training literature; conic reflow can be viewed as a strategy that counteracts self-consuming degradation by injecting real data anchors.
- **Implications for method design**: In any setting where model outputs are used as training data, distribution drift warrants caution and can be mitigated by anchoring to real data.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The analytical perspective on reflow distribution drift is original, and the design of conic reflow is natural and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-dataset validation, comprehensive ablations, and quantitative drift analysis; however, high-resolution and conditional generation experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clearly articulated, figures are intuitive (especially Figures 1–4), and mathematical derivations are complete.
- **Value**: ⭐⭐⭐⭐ — Exposes a fundamental flaw in reflow and provides a simple remedy; the plug-and-play nature adds practical engineering value for the flow matching community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints](physics-constrained_flow_matching_sampling_generative_models_with_hard_constrain.md)
- [\[NeurIPS 2025\] High-order Equivariant Flow Matching for Density Functional Theory Hamiltonian Prediction](high-order_equivariant_flow_matching_for_density_functional_theory_hamiltonian_p.md)
- [\[NeurIPS 2025\] Why Is Attention Sparse in Particle Transformer?](why_is_attention_sparse_in_particle_transformer.md)
- [\[NeurIPS 2025\] DeltaPhi: Physical States Residual Learning for Neural Operators in Data-Limited PDE Solving](deltaphi_physical_states_residual_learning_for_neural_operators_in_data-limited_.md)
- [\[NeurIPS 2025\] From Black Hole to Galaxy: Neural Operator Framework for Accretion and Feedback Dynamics](from_black_hole_to_galaxy_neural_operator_framework_for_accretion_and_feedback_d.md)

</div>

<!-- RELATED:END -->
