---
title: >-
  [Paper Note] Riemannian Consistency Model
description: >-
  [NeurIPS 2025][Image Generation][Consistency Model] This work is the first to extend Consistency Models (CM) to Riemannian manifolds. By leveraging exponential map parameterization and covariant derivatives…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Consistency Model"
  - "Riemannian Manifold"
  - "Flow Matching"
  - "Few-Step Generation"
  - "Covariant Derivative"
date: 2026-05-08
content_hash: b0f8c486c333b898
---

# Riemannian Consistency Model

**Conference**: NeurIPS 2025
**arXiv**: [2510.00983](https://arxiv.org/abs/2510.00983)
**Code**: [GitHub](https://github.com/ccr-cheng/riemannian-consistency-model)
**Area**: Diffusion Models / Generative Models / Manifold Learning
**Keywords**: Consistency Model, Riemannian Manifold, Flow Matching, Few-Step Generation, Covariant Derivative

## TL;DR

This work is the first to extend Consistency Models (CM) to Riemannian manifolds. By leveraging exponential map parameterization and covariant derivatives, it derives both discrete- and continuous-time RCM objectives, enabling high-quality few-step generation on non-Euclidean geometries such as spheres, flat tori, and SO(3).

## Background & Motivation

- **Background**: Diffusion models and flow matching models have achieved remarkable success in image generation, protein design, and related domains, yet inference requires hundreds of iterative sampling steps, incurring substantial computational cost. Consistency Models (CM) bypass this bottleneck by "short-circuiting" the probability-flow ODE, enabling high-quality sample generation in 1–2 steps, and have demonstrated superior performance in Euclidean settings such as image synthesis.
- **Limitations of Prior Work**: Many important scientific applications require generative modeling over non-Euclidean spaces. For instance, protein generation must describe the three-dimensional orientations of amino acids (the SO(3) rotation group) and torsion angles (flat tori), and existing methods typically require 200–1000 sampling steps. Achieving few-step generation on Riemannian manifolds would significantly accelerate drug discovery and enzyme design pipelines.
- **Key Challenge**: Extending CM to Riemannian manifolds faces two fundamental challenges: (1) curved manifolds require that the consistency parameterization remain on the manifold, rendering simple linear interpolation inapplicable; (2) manifold constraints require that vector fields at different points lie in their respective tangent spaces, necessitating additional geometric corrections (covariant derivatives) when computing time derivatives.

## Method

### Overall Architecture

The RCM framework comprises: (1) exponential-map-based consistency parameterization to enforce manifold constraints; (2) closed-form derivation of both discrete- and continuous-time training objectives; (3) theoretical proofs of equivalence between Riemannian Consistency Distillation (RCD) and Riemannian Consistency Training (RCT); and (4) a simplified training objective that eliminates the complex computation of exponential map differentiation.

### Key Designs

1. **Riemannian Consistency Parameterization**: The model learns directly on the vector field $v_\theta(x_t, t)$ and constructs the consistency function via the exponential map: $f_\theta(x_t, t) := \exp_{x_t} \kappa_t v_\theta(x_t, t)$. Since $\kappa_1 = 0$, the boundary condition $f_\theta(x_1, 1) = x_1$ is satisfied naturally. The loss measures consistency via geodesic distance: $\mathcal{L}^N_{\text{RCM}} = N^2 \mathbb{E}_{t,x_t}[w_t d^2_g(f_\theta(x_t,t), f_{\theta^-}(x_{t+\Delta t}, t+\Delta t))]$. Unlike Euclidean CM, which uses the $L^2$ norm directly, geodesic distance is employed here to preserve geometric consistency.

2. **Continuous-Time Limit and Covariant Derivative**: As $N \to \infty$, the continuous-time loss becomes $\mathcal{L}^{\infty}_{\text{RCM}} = \mathbb{E}_{t,x_t}[w\|d(\exp_x)_u(\dot{\kappa}v + \kappa\nabla_{\dot{x}}v) + d(\exp u)_x(\dot{x})\|^2_g]$, where $\nabla_{\dot{x}}$ denotes the covariant derivative along the PF-ODE (via the Levi-Civita connection). The introduction of the covariant derivative is the defining distinction of RCM from Euclidean CM—it captures the change in tangent spaces induced by curved geometry and is a necessary condition for correctly differentiating vector fields on manifolds.

3. **Equivalence Proof from RCD to RCT**: The proof exploits the linearity of $\dot{f}$ with respect to $\dot{x}$ (arising from the linearity of the covariant derivative and the differential of the exponential map), together with a generalization of Bayes' rule for conditional-to-marginal vector fields on Riemannian manifolds: $\dot{x} = \mathbb{E}[(\dot{x}|x_1)|x_t]$. By moving the expectation outside the gradient operation, it is shown that training with the conditional vector field achieves the same optimization effect as distillation.

4. **Simplified Loss (sRCM)**: The key approximation $d(\exp_x)_u \approx d(\exp u)_x$ eliminates the need to distinguish between two exponential map differentials, yielding the simplified loss $\mathcal{L}^{\infty}_{\text{sRCM}} = \mathbb{E}_{t,x_t}[w\|\dot{x} + \dot{\kappa}v + \kappa\nabla_{\dot{x}}v\|^2_g]$. This approximation is exact on flat tori; for general manifolds, it remains accurate when the pretrained model is of high quality.

5. **Kinematic Interpretation**: The RCM objective decomposes into three physically meaningful components: (a) the discrepancy between the predicted and marginal vector fields; (b) the intrinsic change of the vector field (time derivative); and (c) the extrinsic change induced by geometric constraints (the covariant derivative term). This decomposition provides an intuitive physical picture of how curvature affects the consistency objective—for instance, the acceleration formula on the sphere corresponds to uniform circular motion.

### Loss & Training

- **Distillation mode (RCD/sRCD)**: Approximates the marginal vector field using a pretrained RFM model.
- **Training mode (RCT/sRCT)**: Uses the conditional vector field directly, without a pretrained teacher.
- Both modes adopt a linear schedule $\kappa_t = 1-t$ and weight function $w_t = t^2/(1-t)^2$.
- Magnitude-preserving fully connected layers and forced weight normalization are employed to stabilize Jacobian-vector product (JVP) computation.

## Key Experimental Results

### Main Results

Two-step generation results on 2-sphere geographic datasets (KL divergence ↓):

| Dataset | RFM-100 | RFM-2 | sRCD | RCD | CDnaive | sRCT | RCT |
|--------|---------|-------|------|-----|---------|------|-----|
| Earthquake (6124) | 1.51 | 10.99 | **2.13** | 2.22 | 6.20 | 3.66 | 2.38 |
| Volcano (829) | 1.77 | 35.40 | **3.36** | 3.84 | 17.19 | 5.44 | 4.47 |
| Fire (4877) | 0.53 | 9.79 | **1.65** | 1.71 | 8.01 | 3.39 | 1.74 |
| Flood (12810) | 1.33 | 8.17 | **2.27** | 2.41 | 6.21 | 2.81 | 2.39 |

SO(3) dataset MMD ↓ (×10⁻²):

| Dataset | RFM-2 | sRCD | CDnaive | sRCT |
|--------|-------|------|---------|------|
| Swiss Roll | 19.64 | **1.51** | 2.75 | 4.17 |
| Cone | 19.96 | **5.47** | 21.46 | 7.53 |
| Line | 15.50 | **3.06** | 9.36 | 3.75 |

### Ablation Study

Fréchet distance on high-dimensional tori (scalability with dimension):

| Torus Dimension | 2 | 8 | 32 | 64 | 128 |
|---------|---|---|----|----|-----|
| RFM (2 steps) | 0.52 | 1.01 | 1.95 | 1.47 | 1.83 |
| RCT | **0.22** | **0.54** | **0.46** | **0.58** | **0.62** |
| CTnaive | 0.73 | 1.58 | 2.41 | 24.80 | 35.16 |

### Key Findings

- The naïve Euclidean CM performs worst across all manifolds, demonstrating the necessity of the covariant derivative formulation.
- The simplified losses sRCD/sRCT achieve comparable or superior performance to their exact counterparts while eliminating complex exponential map differentiation.
- RCT can outperform RCD when the pretrained model is of low quality (e.g., on the Cone dataset).
- CTnaive degrades sharply in high-dimensional manifold settings, whereas RCT remains stable, underscoring the importance of manifold constraints.
- Discrete-time RCD performs relatively poorly on curved manifolds (e.g., SO(3)), likely due to larger discretization errors.

## Highlights & Insights

- The kinematic interpretation is particularly elegant: decomposing the consistency loss into three physically interpretable components—vector field discrepancy, intrinsic change, and geometric extrinsic change—provides clear conceptual grounding.
- The work exhibits high theoretical rigor, with complete derivations of the discrete-to-continuous limit, RCD↔RCT equivalence, and the justification for the simplified loss.
- The simplified loss is of significant engineering value: it eliminates the need for symbolic computation of exponential map differentials across different manifolds, substantially reducing implementation complexity.

## Limitations & Future Work

- Experiments are conducted only on relatively low-dimensional (up to 128-dimensional) simple manifolds, without validation on high-dimensional complex applications such as actual protein design.
- Although two-step sample quality greatly surpasses RFM-2, it generally remains below RFM-100; single-step generation quality requires further improvement.
- The current framework assumes that the exponential and logarithmic maps admit closed-form expressions, limiting applicability to more general manifolds.
- A systematic comparison with other accelerated sampling methods (e.g., distillation, rectified flow) is absent.

## Related Work & Insights

- This work extends the Consistency Model framework of Song et al., with the core innovation being the introduction of Riemannian geometric tools.
- The relationship with Riemannian Flow Matching (Chen & Lipman, 2024): RFM provides the teacher model, while RCM realizes few-step generation.
- This work may inspire extensions of consistency models to other non-standard spaces, such as Lorentz manifolds and Grassmann manifolds.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First Riemannian consistency model, with outstanding theoretical contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers diverse manifolds and datasets, but lacks large-scale real-world application validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Mathematical derivations are rigorous; the kinematic interpretation is intuitive.
- **Value**: ⭐⭐⭐⭐ Establishes a theoretical foundation for efficient generation on Riemannian manifolds, with potential impact on drug design and related fields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] How to Build a Consistency Model: Learning Flow Maps via Self-Distillation](how_to_build_a_consistency_model_learning_flow_maps_via_self-distillation.md)
- [\[NeurIPS 2025\] FreqPolicy: Efficient Flow-based Visuomotor Policy via Frequency Consistency](freqpolicy_efficient_flow-based_visuomotor_policy_via_frequency_consistency.md)
- [\[NeurIPS 2025\] Toward a Unified Geometry Understanding: Riemannian Diffusion Framework for Graph Generation and Prediction](toward_a_unified_geometry_understanding_riemannian_diffusion_framework_for_graph.md)
- [\[NeurIPS 2025\] Riemannian Flow Matching for Brain Connectivity Matrices via Pullback Geometry](riemannian_flow_matching_for_brain_connectivity_matrices_via_pullback_geometry.md)
- [\[NeurIPS 2025\] SceneDecorator: Towards Scene-Oriented Story Generation with Scene Planning and Scene Consistency](scenedecorator_towards_scene-oriented_story_generation_with_scene_planning_and_s.md)

</div>

<!-- RELATED:END -->
