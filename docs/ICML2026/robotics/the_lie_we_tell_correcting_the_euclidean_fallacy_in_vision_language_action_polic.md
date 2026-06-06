---
title: >-
  [Paper Note] The Lie We Tell: Correcting the Euclidean Fallacy in Vision-Language-Action Policies via Score Matching on Tangent Space
description: >-
  [ICML 2026][Robotics][SE(3) manifolds] The Lie Diffuser Actor (LDA) corrects the diffusion process from the "Euclidean Fallacy"—which flattens SE(3) poses into $\mathbb{R}^{12}$—back to being manifold-native. By injectin…
tags:
  - "ICML 2026"
  - "Robotics"
  - "SE(3) manifolds"
  - "Lie group diffusion"
  - "left-invariant SDE"
  - "tangent space score matching"
  - "CALVIN"
date: 2026-05-08
content_hash: 5d28d624daf0f298
---

# The Lie We Tell: Correcting the Euclidean Fallacy in Vision-Language-Action Policies via Score Matching on Tangent Space

**Conference**: ICML 2026  
**arXiv**: [2606.01847](https://arxiv.org/abs/2606.01847)  
**Code**: Not declared by the authors (only NSTC/NVIDIA acknowledgments, no repository link provided)  
**Area**: Robotics / VLA Policies / Diffusion Models  
**Keywords**: SE(3) manifolds, Lie group diffusion, left-invariant SDE, tangent space score matching, CALVIN

## TL;DR
The Lie Diffuser Actor (LDA) corrects the diffusion process from the "Euclidean Fallacy"—which flattens SE(3) poses into $\mathbb{R}^{12}$—back to being manifold-native. By injecting noise into the Lie algebra $\mathfrak{se}(3)$ via a left-invariant SDE, pulling perturbations back to the manifold via the exponential map, and predicting scores in the tangent space, it theoretically achieves manifold closure, coordinate equivariance, and geodesic optimality. It improves the average task length on CALVIN ABC→D from 3.27 to 3.51.

## Background & Motivation
**Background**: Diffusion-based VLA policies (3D Diffuser Actor, Diffusion Policy, Octo series) have become the mainstream approach in robotic manipulation due to their ability to capture multi-modal behaviors and long-range consistency. A common practice is to flatten the SE(3) pose sequence $\mathbf{g} = (g^1, \dots, g^H)$ into a $\mathbb{R}^{12 \times H}$ vector (9D rotation matrix + 3D translation), inject Gaussian noise in Euclidean space, train a denoising network, and then project the output back to SO(3) using SVD or quaternion normalization.

**Limitations of Prior Work**: The authors label this practice the "Euclidean Fallacy" and point out three specific flaws: (1) Manifold drift: Gaussian noise plus rotation matrices almost inevitably violates $R^\top R = I$, forcing the network to waste capacity learning SVD post-processing; (2) Equivariance destruction: Euclidean noise distributions do not covary when the workspace undergoes a global rigid body transformation, tying the score function to a specific coordinate system; (3) Non-geodesic trajectories: Euclidean interpolation traverses physically unfeasible intermediate poses, losing the screw motion structure defined by Chasles' Theorem and resulting in higher angular jerk. Figure 2 in the paper shows that the orthogonality error of 3D Diffuser Actor is significantly higher than manifold-native methods, with high inter-step variance due to SVD projection magnifying effects near degenerate matrices.

**Key Challenge**: Score-based diffusion requires adding Gaussian noise at each step, but Gaussians are designed for flat vector spaces. SE(3) is a 6D Riemannian manifold with non-zero curvature; additive noise is fundamentally incompatible with manifold geometry. Post-processing projections are mere remedies that cause training-inference mismatch, sensitivity to near-degenerate matrices, and redefinition of the reverse SDE during inference via non-differentiable projections.

**Goal**: To construct an SE(3) diffusion framework such that: (i) intermediate samples at any time $t$ stay $\in SE(3)$; (ii) the score function is equivariant under global rigid transformations of the workspace; (iii) the deterministic limit of the reverse process converges to geodesics on the manifold.

**Key Insight**: The Lie algebra $\mathfrak{se}(3)$ is the tangent space of SE(3) at the identity, which is a flat 6D vector space where Gaussian noise is well-defined. The exponential map $\exp: \mathfrak{se}(3) \to SE(3)$ is a surjection that maps any twist back to a valid rigid transformation. Thus, "adding noise on the manifold" can be redefined as "adding noise in the tangent space and then composing the perturbation with the pose via $\exp$," closing the loop on geometric structure.

**Core Idea**: Formulate the forward diffusion as a left-invariant SDE $g_t = g_0 \cdot \exp(\sigma_t \boldsymbol{\xi})$. The score network outputs a twist $\boldsymbol{\xi} \in \mathfrak{se}(3)$ instead of $\mathbb{R}^{12}$ noise, and reverse updates use $\exp$ pullbacks. All geometric invariants are maintained by the group structure rather than post-processing.

## Method

### Overall Architecture
LDA consists of a geometric context encoder, an iterative denoising Transformer, a tangent-space prediction head, and a tangent-space score matching training objective. Inputs are $K$-view RGB-D observations and a language instruction $\mathcal{L}$; the output is a pose trajectory of horizon $H$, $\mathbf{g} = (g^1, \dots, g^H) \in SE(3)^H$, plus a binary gripper sequence. Point clouds are back-projected and processed via a GAT (Graph Attention Transformer) to extract geometric features $\mathbf{F}_{\text{geo}}$. A CLIP text encoder produces language features $\mathbf{F}_{\text{lang}}$, which are fused into context $\mathcal{C}$ via cross-attention. At each diffusion step $t$, the denoising Transformer receives the noisy trajectory $\mathbf{g}_t$ and time embedding $\tau(t)$, using self-attention for temporal dependencies and cross-attention to inject $\mathcal{C}$. Finally, the tangent-space head outputs a 6D twist $\boldsymbol{\xi}^h = (\boldsymbol{\omega}^h, \mathbf{v}^h)$ for each waypoint. The denoising update follows $g_{t-1}^h = g_t^h \cdot \exp(-\beta_t \boldsymbol{\xi}^h)$, ensuring all intermediate states $\in SE(3)$.

### Key Designs

1.  **Left-invariant SDE Forward Diffusion + Exponential Map Pullback**:
    - **Function**: Transforms "adding noise on SE(3)" into adding Gaussian noise in the tangent space followed by group multiplication, eliminating manifold drift and granting equivariance.
    - **Mechanism**: The forward process takes the Stratonovich form $\mathrm{d}g_t = g_t \cdot (\sigma_t \sum_{i=1}^6 E_i \circ \mathrm{d}W_t^i)$, where $\{E_i\}$ is an orthonormal basis of $\mathfrak{se}(3)$ and $W_t^i$ are independent Wiener processes. Discretization yields $g_t = g_0 \cdot \exp(\sigma_t \boldsymbol{\xi})$, where $\boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_6)$. Since SE(3) is closed under group multiplication and $\exp$ maps any twist to a valid rigid transformation, Proposition 4.1 states that $g_t \in SE(3)$ a.s. throughout. Combined with time-reversal theory, the reverse SDE is $\mathrm{d}g_t = g_t \cdot (\sigma_t^2 s_\theta(g_t, t) \mathrm{d}t + \sigma_t \mathrm{d}\bar{\mathbf{B}}_t)$, implemented discretely as $g_{t-\Delta t} = g_t \cdot \exp(\sigma_t^2 s_\theta(g_t, t) \Delta t + \sigma_t \sqrt{\Delta t} \boldsymbol{\zeta})$.
    - **Design Motivation**: SVD projection after Euclidean noise injection alters the reverse SDE, is sensitive to near-degenerate matrices (magnifying small prediction errors into large rotation errors), and breaks training-inference consistency. By pushing geometric constraints into the group structure, the network no longer needs to learn "outputting a bad matrix and letting SVD save it," focusing instead on manipulation semantics.

2.  **Tangent-space Score Matching + Adjoint Equivariance**:
    - **Function**: Forces the score network to output a 6D twist in $\mathfrak{se}(3)$ rather than $\mathbb{R}^{12}$ noise, automatically gaining coordinate equivariance.
    - **Mechanism**: The prediction head uses two MLPs to regress angular velocity $\boldsymbol{\omega}^h \in \mathbb{R}^3$ and linear velocity $\mathbf{v}^h \in \mathbb{R}^3$, following the semi-direct product structure of SE(3). The output twist is immediately used by $\exp$ for the denoising update. The training objective is tangent-space denoising score matching: sample $t \sim \mathcal{U}(0, T)$ and $\boldsymbol{\xi}^h \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_6)$ from the expert trajectory $\mathbf{g}_0$, construct $g_t^h = g_0^h \cdot \exp(\sigma_t \boldsymbol{\xi}^h)$, and minimize $\|s_\theta(g_t^h, t) - \boldsymbol{\xi}^h\|^2$. Theorem 4.2 proves the optimal score satisfies $s_\theta(h \cdot g, t) = \mathrm{Ad}_h(s_\theta(g, t))$, where the adjoint is $\mathrm{Ad}_{(R, \mathbf{p})}(\boldsymbol{\omega}, \mathbf{v}) = (R\boldsymbol{\omega}, R\mathbf{v} + [\mathbf{p}]_\times R\boldsymbol{\omega})$, meaning the output covaries via the adjoint under global rigid transformations.
    - **Design Motivation**: Equivariance is a hard requirement for robot deployment—a policy should not need to relearn if camera extrinsics or table positions change. By defining the score in the body-fixed tangent space, the network learns the "geometry of the task" rather than the "coordinate system of the lab," which is specifically tested in CALVIN ABC→D zero-shot transfer.

3.  **Geometric Deterministic ODE → Geodesic Bias**:
    - **Function**: Aligns the deterministic limit of the reverse process with geodesics under the bi-invariant metric on SE(3), making generated trajectories resemble screw motions rather than Euclidean lines.
    - **Mechanism**: The probability flow ODE corresponding to the reverse SDE is $\mathrm{d}g_t/\mathrm{d}t = g_t \cdot \sigma_t^2 s_\theta(g_t, t)$. Proposition 4.3 shows that if the score is approximately a constant vector $\boldsymbol{\xi}^*$ along the trajectory, the solution to the ODE is a geodesic on the manifold—a screw motion with constant angular and linear velocities (per Chasles’ Theorem). Even if the score varies over time, the intrinsic form biases the trajectory toward geodesic directions.
    - **Design Motivation**: Angular jerk caused by Euclidean interpolation across non-orthogonal matrices leads to actuator jitter. The "look-ahead consistency" experiment shows that the geodesic jitter of $\hat{x}_0^{(t)}$ vs $\hat{x}_0^{(t-1)}$ for LDA is nearly an order of magnitude lower than the Euclidean baseline, representing a direct gain in steady-state control for real-world deployment.

### Loss & Training
The total loss is $\mathcal{L} = \lambda_s \mathbb{E}_{t, \boldsymbol{\xi}}\left[\sum_h \|s_\theta(g_t^h, t) - \boldsymbol{\xi}^h\|^2\right] + \lambda_p \mathcal{L}_{\text{pos}} + \lambda_g \mathcal{L}_{\text{grip}}$. The score matching term is the primary objective, with MSE for position and binary cross-entropy for the gripper as auxiliary losses. On CALVIN, models are trained for 300K–600K steps to align with baselines at 600K–800K steps.

## Key Experimental Results

### Main Results
Success rate and average chain length on CALVIN ABC→D (train A/B/C, zero-shot test D) and ABCD→D:

| Setting | Method | SR1 | SR2 | SR3 | SR4 | SR5 | Avg Len |
|---------|------|-----|-----|-----|-----|-----|---------|
| ABC→D | 3D Diffuser Actor (600K) | 92.2 | 78.7 | 63.9 | 51.2 | 41.2 | 3.27 |
| ABC→D | LDA (600K) w/o GAT | 89.6 | 78.0 | 66.6 | 55.7 | 46.9 | 3.368 |
| ABC→D | LDA (300K) w/o Lie | 90.2 | 80.3 | 69.6 | 58.5 | 48.8 | 3.474 |
| ABC→D | **LDA (300K) full** | **93.7** | **83.4** | **70.3** | 57.6 | 46.2 | **3.512** |
| ABCD→D | 3D Diffuser Actor (800K) | 90.3 | 77.3 | 65.8 | 53.8 | 41.6 | 3.288 |
| ABCD→D | LDA (300K) w/o GAT | 90.8 | 77.3 | 66.4 | 57.6 | 48.3 | 3.404 |
| ABCD→D | LDA (400K) w/o Lie | 91.0 | 76.1 | 63.4 | 51.6 | 41.8 | 3.239 |
| ABCD→D | **LDA (300K) full** | 90.6 | 80.4 | **71.1** | **62.6** | **53.7** | **3.584** |

Both modules are independently effective, and their combination outperforms the baseline despite using only 1/2 of the training budget. On OpenVLA-OFT, switching the score matching to SE(3) improved LIBERO Long success rates from 92.20 to 94.13, indicating the gain stems from the Lie formulation itself rather than the specific encoder.

### Key Findings
- **Manifold Constraint Violations (Fig. 5)**: During reverse diffusion, the Euclidean baseline exhibits orthogonality errors up to $\mathcal{O}(10^0)$; determinant and quaternion norm deviations are significant (between 0.5–2.0). LDA remains within floating-point precision ($\sim 10^{-7}$). While Euclidean trajectories for quaternions cut through the interior of the $\mathbb{S}^3$ sphere, LDA trajectories strictly follow geodesics on the surface.
- **Look-ahead Consistency (Fig. 6)**: The geodesic jitter between $\hat{x}_0^{(t)}$ and $\hat{x}_0^{(t-1)}$ (predicted final clean actions at adjacent denoising steps) is nearly an order of magnitude lower for LDA. The Euclidean baseline shows large jumps, especially in early noisy steps. This means LDA's target estimation converges monotonically, directly benefiting predictable real-time control.
- **Benefits of Equivariance**: Zero-shot transfer in CALVIN ABC→D requires the model to work in unseen layouts. LDA's adjoint equivariance ensures the score function covaries with global shifts/rotations of the workspace, preventing policy failure. In ABCD→D, the baseline's training was destabilized by environmental diversity, while LDA improved steadily with data scaling.

## Highlights & Insights
- The term "Euclidean Fallacy" is highly effective for communication, grouping the hidden engineering debts of previous diffusion policies (SVD post-processing, quaternion normalization) into a single measurable geometric error. It identifies a clear attack surface: rotation representation choice vs. noise distribution covariance.
- Moving equivariance from the architectural layer (GNN/equivariant CNN) to the "generation process" layer is a clean generational leap. Previous works only guaranteed equivariance at the encoder; the generation process broke it as soon as noise was added. LDA maintains equivariance from encoder to sampling via the left-invariant SDE.
- LDA is a plug-and-play "geometric head"—one only needs to replace the 12D Euclidean head with a 6D twist head + $\exp$ pullback, without redesigning the encoder or RL training loops.

## Limitations & Future Work
- The code repository is not public; reproducibility depends on fine implementation details (especially the numerical handling of $V(\boldsymbol{\omega})$ in the left Jacobian and small angle approximations).
- In real-world "Put Block in Box" tests, LDA slightly trailed the baseline (75% vs 80%), suggesting that for tasks dominated by translation with almost no rotation requirements, unconstrained Euclidean exploration might be more lenient.
- Comparison with large-scale VLA foundation models (e.g., $\pi_0$, full OpenVLA) is limited to sub-tasks; it remains to be seen if Lie formulations remain a dominant source of gain at massive data scales.
- Proposition 4.3 requires the score to be a constant vector along the trajectory, which is not true in practice; the paper describes this as a "bias toward geodesics" without a quantitative metric for the deviation.

## Related Work & Insights
- **vs 3D Diffuser Actor (Ke et al., 2024)**: Shares the point cloud + Transformer + diffusion framework. The difference is Euclidean vs Manifold: 3DDA uses 12D + SVD post-processing, while LDA moves noise, score, and updates into SE(3).
- **vs Equivariant Policy Learning**: These works embed equivariance into the encoder, but the generation side remains Euclidean diffusion, creating an "equivariant-front, broken-back" asymmetry. LDA ensures the entire generation process is left-invariant.
- **vs Riemannian Generative Models (De Bortoli et al., Lou et al.)**: Provides the mathematical foundation, but LDA prefers SDEs over Riemannian flow matching to maintain multi-modal sampling capabilities essential for complex manipulation.
- **vs Traditional Quaternion/6D Rotation Reps**: While those focus on how to represent rotations in the output layer, they still inject noise in Euclidean space. LDA addresses "where the noise lives," a more upstream correction that simultaneously solves manifold drift, equivariance, and geometric jitter.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Rotation representation and Lie diffusion are not individually new, but the "Euclidean Fallacy" systematic naming + Propositions 4.1/4.2/4.3 + real-world validation pushes Riemannian diffusion to the VLA frontline.
- **Experimental Thoroughness**: ⭐⭐⭐ Covers CALVIN, real-robot tasks, and OpenVLA-OFT. However, it lacks benchmarks against the latest VLA baselines and heterogenous benchmarks like RLBench.
- **Writing Quality**: ⭐⭐⭐⭐ Propositions and geometric motivations are clear; Theorems correspond well with empirical phenomena.
- **Value**: ⭐⭐⭐⭐ Provides a low-overhead, high-gain upgrade path for any diffusion-based policy; "replacing the Euclidean head" is a module directly reusable by successor works.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies](discrete_diffusion_vla_bringing_discrete_diffusion_to_action_decoding_in_vision-.md)
- [\[ICML 2026\] Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action](spatial_memory_for_out-of-vision_manipulation_in_vision-language-action.md)
- [\[ICML 2026\] LangForce: Bayesian Decomposition of Vision-Language-Action Models via Latent Action Queries](langforce_bayesian_decomposition_of_vision_language_action_models_via_latent_act.md)
- [\[ICML 2026\] Contrastive Representation Regularization for Vision-Language-Action Models](contrastive_representation_regularization_for_vision-language-action_models.md)
- [\[ICML 2026\] Neural Implicit Action Fields: From Discrete Waypoints to Continuous Functions for Vision-Language-Action Models](neural_implicit_action_fields_from_discrete_waypoints_to_continuous_functions_fo.md)

</div>

<!-- RELATED:END -->
