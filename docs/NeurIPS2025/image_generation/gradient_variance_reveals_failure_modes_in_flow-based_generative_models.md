---
title: >-
  [Paper Note] Gradient Variance Reveals Failure Modes in Flow-Based Generative Models
description: >-
  [NeurIPS 2025][Image Generation][Rectified Flow] By analyzing the gradient variance of the CFM loss, this paper demonstrates that Rectified Flow inevitably memorizes training pairs under deterministic interpolation rather than learning an optimal transport map, and proves that introducing stochasticity (stochastic interpolants) breaks this memorization channel and restores generalization.
tags:
  - NeurIPS 2025
  - Image Generation
  - Rectified Flow
  - Gradient Variance
  - Memorization
  - Optimal Transport
  - Stochastic Interpolant
  - Conditional Flow Matching
date: 2026-05-08
content_hash: 70c96bcd416c620e
---

# Gradient Variance Reveals Failure Modes in Flow-Based Generative Models

**Conference**: NeurIPS 2025  
**arXiv**: [2510.18118](https://arxiv.org/abs/2510.18118)  
**Code**: None  
**Area**: Generative Models / Flow Matching Theory  
**Keywords**: Rectified Flow, Gradient Variance, Memorization, Optimal Transport, Stochastic Interpolant, Conditional Flow Matching

## TL;DR
By analyzing the gradient variance of the CFM loss, this paper demonstrates that Rectified Flow inevitably memorizes training pairs under deterministic interpolation rather than learning an optimal transport map, and proves that introducing stochasticity (stochastic interpolants) breaks this memorization channel and restores generalization.

## Background & Motivation
**Background**: ODE-based flow matching models (Flow Matching / Rectified Flow) represent the dominant paradigm in generative modeling, learning vector fields between source and target distributions to enable sampling. Rectified Flow iteratively "straightens" transport paths to achieve approximate one-step inference.

**Limitations of Prior Work**: The "straightening" objective of Rectified Flow appears intuitive but conceals a fundamental failure mode—under deterministic training, the model tends to memorize paired relationships in the training set rather than learning a generalizable transport map. Prior work claims that 1 or 2 rectification steps suffice to obtain straight paths, but rigorous proofs are lacking and counterexamples exist.

**Key Challenge**: Low gradient variance is intuitively regarded as a sign of good optimization progress; however, this paper shows that under deterministic interpolation, low variance corresponds precisely to a memorized solution—a vector field that perfectly fits the training pairs—rather than to the optimal transport solution.

**Goal**: (1) Clarify under what conditions gradient variance is a reliable indicator of solution quality; (2) prove that the global minimizer of deterministic ReFlow is the memorization solution; (3) show that introducing noisy interpolation restores generalization.

**Key Insight**: Beginning from an analytically tractable Gaussian-to-Gaussian setting, the paper derives closed-form gradient variance expressions and then generalizes to finite datasets.

**Core Idea**: Gradient variance reveals vector field quality—deterministic interpolation drives ReFlow toward memorization, while adding noise restores generalization.

## Method

### Overall Architecture
This is a theoretical analysis paper that does not propose a new method, but rather dissects the failure mechanism of the existing Rectified Flow framework. The central tool is the gradient variance of the CFM training loss, $\text{Var}[\nabla_\theta L_{\text{MC}}]$, used to diagnose which solution is preferred by optimization under different training configurations (coupling $T$, interpolation scheme $I$, vector field class $v$).

### Key Theoretical Results

1. **Lemma 1 — Closed-Form Optimal Vector Field in the Gaussian Setting**:
    - Let $X_0 \sim \mathcal{N}(0, \mathbf{I}_d)$, $X_1 \sim \mathcal{N}(\mu, \mathbf{M}_d)$
    - OT vector field: $\hat{v}_{OT}(X_t, t) = \hat{\theta} + \hat{\Theta}[\mathbf{I}_d + t\hat{\Theta}]^{-1}(X_t - t\hat{\theta})$
    - Key finding: Due to the matrix inverse $[\mathbf{I} + t\hat{\Theta}]^{-1}$, MLP/CNN/Transformer architectures cannot represent this vector field exactly.
    - Implication: Even for the simplest Gaussian-to-Gaussian transport, standard neural network parameterizations incur an irreducible approximation error.

2. **Proposition 1 — Gradient Variance Is Not Determined by Trajectory Crossings** (informal):
    - For OT coupling $T_{OT}$ and rotational coupling $T_{rOT}^R$, the pair-optimal vector field simultaneously achieves zero loss and zero variance.
    - Even when the rotation angle is not 0° (i.e., non-OT), as long as the vector field is consistent with the coupling (pair-optimal), the variance remains zero.
    - **Counter-intuitive conclusion**: Variance does not increase due to interpolation line crossings or dense regions. Variance is determined by the mismatch between the vector field and the coupling structure, not by geometric proximity.
    - This corrects a long-standing misconception in the community that "gradient variance mainly arises at interpolation line crossing points."

3. **Proposition 2 — Minimizers Memorize** (core contribution):
    - Given a finite training set $\{(Z_0^{(i)}, Z_1^{(i)} = T(Z_0^{(i)}))\}_{i=1}^N$ with deterministic interpolation $Z_t = (1-t)Z_0 + tZ_1$
    - **Theorem**: There exists a (deterministic) vector field $v$ such that the empirical loss $L_{\text{MC}}^{\text{det}}(v) = 0$.
    - ODE integration from a training source point recovers its paired target point $\hat{X}_1 = X_1$—i.e., perfect memorization.
    - Intuition: Because the probability of hitting an interpolation line crossing exactly when sampling $t$ continuously is zero, the vector field is unconstrained at crossing points; numerical integration during inference effectively "bypasses" these crossings.

4. **Counter Example 1 — 1-ReFlow Does Not Guarantee Straight Paths**:
    - Counterexample construction: $T(x_0) = R_{180°} x_0 + 5$, where all interpolation lines cross at $t = 1/2$.
    - Even when the transport map $T$ is invertible, the induced interpolation is not invertible, and a subsequent ReFlow step cannot reconstruct $(x_0, x_1)$.

5. **Remark 2 — Noise Breaks Memorization**:
    - Stochastic interpolation $x_t = (1-t)x_0 + tx_1 + f(t,\sigma)Z$ destroys the bijection between $(x_t, t)$ and $(x_0, T(x_0))$.
    - This directly invalidates the assumptions underlying Lemma 2 (idempotence) and Proposition 2 (memorization).
    - Even small noise $\sigma = 0.05$ is sufficient to restore generalization.

### Key Experimental Observations (Gaussian Setting)
- **Figure 6 core experiment**: A 180° rotation transport causes all interpolation lines to cross at $t = 1/2$; nevertheless, under deterministic training the model learns the rotation map and perfectly reproduces training pairs at inference—confirming memorization.
- **Figure 4**: After adding noise, the gradient variance of the OT vector field is significantly lower than that of the pair-optimal vector field ($p < 0.01$), indicating that noise biases optimization toward the OT solution.
- **Figure 5**: Under a random coupling, the variance at the OT field is lower than under a structured 120° coupling—variance is not dominated by trajectory density.

## Key Experimental Results

### Gaussian Mixture Model Experiments (Table 1)

| Dimension | Method | Generalization MMD↓ | Memorization MMD↓ |
|-----------|--------|--------------------|--------------------|
| d=3  | CFM($\sigma=0$) | 0.0034 | **1.758e-6** |
| d=3  | CFM($\sigma=0.05$) | **0.0018** | 3.105e-5 |
| d=50 | CFM($\sigma=0$) | 0.0021 | **9.089e-6** |
| d=50 | CFM($\sigma=0.05$) | **0.0020** | 6.09e-5 |

- Deterministic CFM achieves extremely low memorization distance (nearly perfect pairing) but poor generalization.
- CFM with $\sigma=0.05$ generalizes better and does not rely on memorization.

### CelebA — Adversarial Coupling Experiment (Table 2)

| Metric | CFM($\sigma=0.05$) | CFM($\sigma=0$) |
|--------|--------------------|-----------------|
| 5K Generalization (L2 to OT)↓ | **34.25** ± 7.54 | 50.40 ± 16.73 |
| 5K Memorization (L2 to Shuffled)↓ | 55.02 ± 16.51 | **28.57** ± 5.49 |
| 50K Generalization (L2 to OT)↓ | **30.05** ± 6.77 | 46.78 ± 14.87 |
| 50K Memorization (L2 to Shuffled)↓ | 56.48 ± 18.55 | **45.98** ± 11.85 |

- After shuffling the OT coupling, deterministic CFM still memorizes the incorrect shuffled pairs.
- CFM with noise rejects memorization of incorrect pairs and generalizes toward the true OT map.

### CelebA — Simulated 1-ReFlow (Table 3)

| Metric | CFM($\sigma=0.05$) | CFM($\sigma=0$) |
|--------|--------------------|-----------------|
| 5K Generalization (L2 to OT)↓ | **31.35** ± 7.38 | 43.35 ± 14.21 |
| 5K Memorization (L2 to Generated)↓ | 25.08 ± 8.59 | **8.63** ± 1.76 |

- 1-ReFlow further exacerbates the tendency toward memorization.
- Increasing dataset size (50K vs. 5K) mitigates but does not eliminate memorization.

## Highlights & Insights
1. **Correcting a community misconception**: The paper proves that gradient variance does not arise from geometric crossings of interpolation lines, but from the mismatch between the vector field and the coupling structure—challenging a widely held intuition in the flow matching community.
2. **Rigorous proof of memorization**: Proposition 2 formally proves for the first time that the global minimizer of deterministic ReFlow is the memorization solution, and that ODE integration can bypass crossing points to exactly recover training pairs.
3. **Simple and effective remedy**: Interpolation noise at the level of $\sigma = 0.05$ suffices to break the memorization channel, requiring no modification to the network architecture or training procedure.
4. **Combined analytical and empirical validation**: Closed-form analysis in the Gaussian-to-Gaussian setting is complemented by multi-level validation on GMMs and real CelebA data.
5. **Architectural limitations revealed by Lemma 1**: MLPs and Transformers cannot exactly represent the matrix inverse term in the Gaussian OT vector field.

## Limitations & Future Work
1. **Limited experimental scale**: Validation is restricted to CelebA (64×64); experiments on high-resolution images and other modalities (video, audio, molecular data) are absent.
2. **No new method proposed**: Beyond "adding noise," no dedicated solution to memorization is offered; repeated noise injection converges to entropy-regularized OT rather than standard OT.
3. **Behavior on large datasets**: The paper acknowledges that memorization effects weaken at 50K samples (as model capacity becomes insufficient for perfect memorization), but behavior at the million-scale is not explored.
4. **Noise level selection**: No theoretical guidance is provided for choosing the optimal $\sigma$; different tasks may require different noise levels.
5. **Relation to fast sampling methods**: The paper does not analyze whether methods such as Consistency Models are similarly susceptible to memorization.

## Related Work & Insights

| Aspect | Ours | Rectified Flow (Liu 2022) | SBM (Shi et al. 2024) |
|--------|------|---------------------------|----------------------|
| Training scheme | Analyzes deterministic vs. stochastic interpolation | Deterministic interpolation with iterative straightening | Bidirectional stochastic interpolation |
| Memorization | Theoretically proves deterministic training necessarily memorizes | Not discussed | Naturally avoided via noise |
| Coupling | Identifies deterministic coupling as the root of memorization | Uses deterministic coupling | Independent sampling each iteration |
| Variance analysis | In-depth analysis of gradient variance as a diagnostic tool | Not addressed | Not addressed |
| Optimal transport | Proves low variance $\neq$ OT optimality | Targets OT | Approximates entropy-regularized OT |

The memorization analysis in this paper overlaps with studies of memorization in diffusion models (e.g., Bamberger et al., 2025; Buchanan et al., 2025), but the key distinction is that this paper focuses on the deterministic coupling setting of Rectified Flow rather than the independent sampling setting of standard CFM.

This paper's findings carry several broader implications:
1. **Warning for Rectified Flow practitioners**: Those using deterministic ReFlow iterations should be aware of memorization risk; noise should be introduced at minimum after the first rectification.
2. **Gradient variance as a diagnostic tool**: Monitoring gradient variance can be incorporated into flow matching training pipelines to detect memorization.
3. **Implications for latent and video diffusion**: Large-scale models (e.g., Stable Diffusion, Sora) may exhibit analogous implicit memorization; the framework developed here provides analytical tools to investigate this.
4. **The nature of noise injection**: Adding noise not only acts as a regularizer but fundamentally reshapes the optimization landscape—from a single memorization solution to one that favors the OT solution.
5. **Connection to the SDE vs. ODE debate**: This paper provides a rigorous theoretical foundation for the empirical observation that SDEs (with noise) are less prone to overfitting than ODEs (without noise).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First rigorous proof of the memorization mechanism in Rectified Flow
- Experimental Thoroughness: ⭐⭐⭐ — Theoretically rigorous but experiments limited to CelebA
- Writing Quality: ⭐⭐⭐⭐ — Proofs are clear and figures are intuitive
- Value: ⭐⭐⭐⭐ — Carries important cautionary and practical implications for the flow matching community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Failure Prediction at Runtime for Generative Robot Policies](failure_prediction_at_runtime_for_generative_robot_policies.md)
- [\[NeurIPS 2025\] A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models](a_gradient_flow_approach_to_solving_inverse_problems_with_latent_diffusion_model.md)
- [\[NeurIPS 2025\] Value Gradient Guidance for Flow Matching Alignment](value_gradient_guidance_for_flow_matching_alignment.md)
- [\[NeurIPS 2025\] Rectified-CFG++ for Flow Based Models](rectified-cfg_for_flow_based_models.md)
- [\[NeurIPS 2025\] Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints](physics-constrained_flow_matching_sampling_generative_models_with_hard_constrain.md)

</div>

<!-- RELATED:END -->
