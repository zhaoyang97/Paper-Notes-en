---
title: >-
  [Paper Note] Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems
description: >-
  [CVPR 2026][Image Generation][Diffusion Models] This paper proves that the DDIM deterministic reverse chain is essentially a Partitioned Iterated Function System (PIFS)…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Diffusion Models"
  - "Partitioned Iterated Function Systems (PIFS)"
  - "Fractal Geometry"
  - "Jacobian Analysis"
  - "Kaplan-Yorke Dimension"
date: 2026-05-08
content_hash: 5df71f19273506d7
---

# Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems

**Conference**: CVPR 2026
**arXiv**: [2603.13069](https://arxiv.org/abs/2603.13069)  
**Code**: None  
**Area**: Image Generation
**Keywords**: Diffusion Models, Partitioned Iterated Function Systems (PIFS), Fractal Geometry, Jacobian Analysis, Kaplan-Yorke Dimension

## TL;DR

This paper proves that the DDIM deterministic reverse chain is essentially a Partitioned Iterated Function System (PIFS), and derives from this framework three computable geometric quantities requiring no model evaluation. It provides a unified, first-principles explanation for the two-phase denoising dynamics of diffusion models, the effectiveness of self-attention, and four empirical design choices (cosine schedule offset, resolution-dependent logSNR shift, Min-SNR loss weighting, and Align Your Steps sampling).

## Background & Motivation

### 1. State of the Field
Diffusion models generate high-quality images through sequential denoising processes, grounded theoretically in continuous-time SDEs or probability-flow ODEs, with global $\mathcal{W}_2$ distributional convergence guarantees. However, the continuous perspective treats the learned score network as a black box.

### 2. Limitations of Prior Work
Existing theory fails to structurally explain two key phenomena: (a) why early steps assemble global spatial context while later steps synthesize local detail; and (b) why self-attention is so effective as a generative primitive. Numerous empirical design choices (cosine offset, Min-SNR weighting, etc.) also lack a unified geometric explanation.

### 3. Root Cause
The continuous SDE/ODE framework provides elegant distributional convergence guarantees but cannot reveal how the discrete sampling chain assembles image structure at each step — a tension between theoretical elegance and structural interpretability.

### 4. Paper Goals
- Provide a structural proof for the two-phase dynamics of the DDIM reverse chain
- Explain the geometric role of self-attention in diffusion models
- Derive practical design principles from a unified framework that explain existing empirical techniques

### 5. Starting Point
The paper reinterprets the DDIM deterministic reverse chain $\Phi = \Phi_1 \circ \cdots \circ \Phi_T$ as a Partitioned Iterated Function System (PIFS) — a classical mathematical structure from fractal image compression that is naturally suited for handling local self-similarity.

### 6. Core Idea
The Jacobian of each DDIM operator $\Phi_t$ decomposes into diagonal blocks (intra-patch) and cross blocks (inter-patch), whose contraction/expansion behavior is fully characterized by three closed-form constants depending only on the noise schedule and patch covariance, enabling analysis of denoising dynamics without running the model.

## Method

### Overall Architecture

The paper establishes a mathematical mapping from DDIM to PIFS, with the following core pipeline:

1. **Contraction Structure Analysis** (§3): Derives contraction conditions for the single-step DDIM operator
2. **Two-Phase Structure Analysis** (§4): Computes constants from data statistics and architectural properties to explain the two denoising phases
3. **Attractor Geometry** (§5): Computes the fractal dimension of the PIFS attractor via the Lyapunov spectrum
4. **Practical Design Principles** (§6): Derives three optimization criteria from the PIFS framework, providing a unified explanation for four empirical designs

### Key Designs

#### Design 1: Dual Contraction Conditions (EC) and (PC)

**Function**: Establishes two contraction conditions for the single-step DDIM operator $\Phi_t(x) = \frac{\sqrt{\bar\alpha_{t-1}}}{\sqrt{\bar\alpha_t}} x + b_t \hat\varepsilon_\theta(x,t)$.

**Mechanism**: The Jacobian $J_x\Phi_t = \frac{\sqrt{\bar\alpha_{t-1}}}{\sqrt{\bar\alpha_t}} I + b_t J_x\hat\varepsilon_\theta$ contains an expansion term (identity scaling $>1$) and a contraction term (score correction with $b_t < 0$); contractivity depends on the algebraic properties of the score Jacobian.

- **(EC) Euclidean Contraction**: A global condition defining the contraction threshold $L_t^* = \frac{\sqrt{\bar\alpha_{t-1}/\bar\alpha_t} - 1}{|b_t|}$, depending solely on the noise schedule
- **(PC) Block-Max-Norm Contraction**: A patch-level condition that decomposes the Jacobian into diagonal blocks $\kappa_t^{\mathrm{diag}}$ and cross blocks $\delta_t^{\mathrm{cross}}$, requiring $\kappa_t^{\mathrm{diag}} + \delta_t^{\mathrm{cross}} < 1$

**Design Motivation**: Natural images exhibit local (not global) self-similarity, necessitating patch-level rather than global contraction guarantees — precisely the advantage of classical PIFS over IFS.

#### Design 2: Directional Suppression Field and Hierarchical Release

**Function**: Introduces a directional suppression field $S_{k,t}(x) = |b_t| \langle v_k^{(1)}, [\nabla_x \Delta_t(x)]_{kk} v_k^{(1)} \rangle$ to quantify the non-Gaussian correction applied by the trained score network to each patch.

**Mechanism**: Under the Gaussian baseline, the diagonal block spectral norm $f_t(\lambda_k)$ is $>1$ (expansive) for all CIFAR-10 patches; however, after training, the network learns a suppression field $S_{k,t} > 0$ that drives the effective Rayleigh quotient below 1. The key theorem (Stratified Crossover, Thm 22) proves that under the Margin Monotonicity condition (MM), low-variance patches release suppression first and high-variance patches later, producing a strictly variance-ordered hierarchical release.

**Design Motivation**: Explains why diagonal blocks remain $\approx 1$ in Regime I rather than expanding as the Gaussian baseline predicts, and how patches sequentially "unlock" detail synthesis in Regime II.

#### Design 3: Kaplan-Yorke Dimension Formula for the Attractor

**Function**: Derives the fractal dimension of the PIFS attractor and establishes the discrete Moran equation $\prod_t f_t(\lambda^{**}) = 1$ to solve for the global expansion threshold $\lambda^{**}$.

**Mechanism**: Under the assumptions of Gaussian data and block-diagonal covariance, the Lyapunov spectrum is fully determined by the per-step diagonal expansion function $f_t(\lambda)$. Diagonal directions with $\lambda_k > \lambda^{**}$ are expansive, and the KY dimension formula is:

$$d_{\mathrm{KY}} = N^+ + \frac{\sum_{k:\lambda_k > \lambda^{**}} n_k \Lambda(\lambda_k)}{|\Lambda_{k^*}^-|}$$

For non-Gaussian data, the suppression-corrected version satisfies $d_{\mathrm{KY}}^{\mathrm{eff}} \leq d_{\mathrm{KY}}$, as suppression can only reduce attractor dimensionality.

**Design Motivation**: Provides model-free predictions of attractor dimension, connecting noise schedule design to the geometric properties of the generative manifold.

### Loss & Training

- **Collage Analogy** (Thm 12): The DSM training objective is equivalent to minimizing PIFS collage error (up to SNR weighting)
- **$L^2$–$\mathcal{W}_1$ Bridge** (Thm 14): The training loss controls the Wasserstein-1 distance to the PIFS fixed point
- **PIFS Regularizer** (Thm 15): $\mathcal{L}_{\mathrm{PIFS}}(\theta) = \mathcal{L}(\theta) + \mu_{\mathrm{reg}} \sum_{t,k,j\neq k} \|[J_x\hat\varepsilon_\theta]_{kj}\|_F^2$, which directly enforces the (PC) condition and can be computed efficiently via JVP/VJP

## Key Experimental Results

### Main Results: Block-Jacobian Decomposition Validates Two-Phase Structure

Theoretical predictions are validated on a pretrained DDPM CIFAR-10 model using 8×8 patches ($M=16$, $n_k=192$) and a 50-step DDIM sampler.

| Training step $t$ | $\hat\kappa_t^{\mathrm{diag}}$ | $\hat\delta_t^{\mathrm{cross}}$ | Global $\hat\kappa_t$ | Phase |
|---|---|---|---|---|
| 980 | 1.0004 | 0.0007 | 1.0011 | High noise |
| 800 | 1.0002 | 0.0008 | 1.0010 | High noise |
| 600 | 1.0000 | 0.0853 | 1.0853 | Regime I |
| 400 | 1.0026 | 0.1273 | 1.1300 | Regime I |
| 220 | 1.0325 | 0.0768 | 1.1092 | Regime II |
| 20 | 1.2111 | 0.1858 | 1.3969 | Detail |

**Key Finding**: In Regime I, global expansion is driven entirely by cross-patch coupling (diagonal blocks $\approx 1$); in Regime II, diagonal blocks begin expanding and attention localizes.

### Attention Entropy and Cross-Patch Coupling

| $t$ | Attention Entropy $H(A_t)$ (nats) | $\hat\delta_t^{\mathrm{cross}}$ | Phase |
|---|---|---|---|
| 980 | 4.963 | 0.00946 | High noise |
| 560 | 4.662 | 0.09463 | Regime I |
| 160 | 4.541 | 0.42899 | Regime II |
| 20 | 4.063 | 2.06175 | Detail |

$\hat\delta_t^{\mathrm{cross}}$ increases 218-fold from $t=980$ to $t=20$. Spearman $\rho(H, \hat\delta^{\mathrm{cross}}) = -1.000$ ($p < 0.001$), indicating a perfect inverse correlation between coupling and entropy.

### Ablation Study

#### (PC) Condition Crossover Validation

| $t$ | Phase | Mean margin slack | Violation rate |
|---|---|---|---|
| 700 | Regime I | $-0.003942$ | 16/16 |
| 200 | I/II transition | $-0.000304$ | 14/16 |
| 160 | Regime II | $+0.001382$ | 0/16 |
| 40 | Regime II | $+0.006412$ | 0/16 |

The crossover occurs at $t \in [160, 200]$, within approximately a 40-step window. (PC) is universally violated in Regime I and universally satisfied in Regime II.

#### Spearman Correlation for Hierarchical Release

In the crossover interval ($t=240, 260$), $\rho(\hat\lambda_k, \hat\kappa_t^{\mathrm{diag}})$ is negative and significant ($p \leq 0.047$), confirming that low-variance patches are released first. Deep in Regime II ($t=40$), $\rho$ returns to positive at $0.771$ ($p=0.001$), recovering the Gaussian ordering.

#### Suppression-Corrected KY Dimension (CelebA-HQ Experiment)

On the google/ddpm-celebahq-256 model, with $\lambda_k \in [38.7, 231.7]$, the Gaussian baseline predicts $d_{\mathrm{KY}} = 12288$ (full-dimensional expansion). However, the suppression-corrected Moran threshold $\lambda^{***} = 500 \gg \lambda_{\max}$ yields $d_{\mathrm{KY}}^{\mathrm{eff}} = 0$. Predicted Lyapunov exponent signs match measured signs for all 16 patches (100% agreement).

### Key Findings

1. **Score Deviation Scaling**: In the high-noise regime, $\|\Delta_t\|_2 = O(\sqrt{\bar\alpha_t})$; OLS-fitted slope is $0.95$ (95% CI $[0.88, 1.02]$), consistent with theoretical predictions
2. **Information Gain–KY Dimension Proportionality**: $\rho(\mathrm{IG}_t, |\Delta d_t|) \geq 0.9999$, with the ratio CV only 3.4%, nearly perfectly satisfying the Cauchy-Schwarz equality condition
3. **Noise Schedule Comparison**: The linear schedule achieves the most uniform $L_t^*$ (CV 0.341); the cosine schedule yields more uniform information gain (CV 0.867 vs. 1.107)

## Highlights & Insights

1. **Deep Mathematical Connection**: The paper bridges diffusion models and fractal image compression via the PIFS framework, revealing the elegant correspondence that score matching = collage error minimization
2. **Three Model-Free Constants**: $L_t^*$ (contraction threshold), $f_t(\lambda)$ (diagonal expansion function), and $\lambda^{**}$ (global expansion threshold) are fully determined by the schedule and data statistics, forming a universal design language
3. **Unified Theory Explaining Four Empirical Designs**: The cosine offset improves the weakest link $L_1^*$ (4×); Min-SNR balances KY dimension growth; the resolution shift preserves the Moran ratio; AYS concentrates steps where $L_t^*$ is smallest
4. **Geometric Role of Self-Attention**: Query tokens = range blocks; key/value tokens = domain blocks; $A_{kj}$ = soft domain-range pairing. The hard-attention limit exactly recovers classical PIFS structure, and cross-patch coupling $\delta_t^{\mathrm{cross}}$ is bounded by the attention weights

## Limitations & Future Work

1. **Gaussian Patch Assumption**: The core analysis relies on a block-diagonal Gaussian covariance assumption; non-Gaussian cases (e.g., texture-rich data) require more refined suppression field modeling
2. **PIFS Regularizer Not Experimentally Validated**: The $\mathcal{L}_{\mathrm{PIFS}}$ regularizer is proposed but not evaluated in end-to-end training experiments; its practical effect remains to be verified
3. **Limited to DDIM Deterministic Sampling**: The analysis primarily targets DDIM (probability-flow ODE); applicability to DDPM stochastic sampling is not discussed in detail
4. **Loose Attention Gradient Bound in Regime I**: The bound on $\|\nabla_x A_{k\ell}\|_{\mathrm{op}}$ in Thm 23 is loose in Regime I; a more precise characterization of query/key temperature is left for future work
5. **Effect of Skip Connections**: Encoder skip connections in the UNet architecture lie outside the scope of the attention bound; $\delta_t^{\mathrm{cross,skip}}$ may dominate in Regime II but is not precisely quantified

## Related Work & Insights

- **Fractal Image Compression** (Jacquin 1992, Barnsley 1988): PIFS and the Collage Theorem form the mathematical foundation of this paper; the authors reinterpret classical coding theory as an analytical tool for generative models
- **Two-Phase Behavior** (Raya & Ambrogioni 2023): Prior empirical observations of two-phase phenomena are given a structural proof (contraction vs. expansion) in this work
- **Information-Constant Schedules** (Kingma et al. 2021, Chen et al. 2023): This paper proves that IG uniformity is equivalent to KY dimension growth uniformity (Thm 32), endowing the information-theoretic criterion with geometric meaning
- **Align Your Steps** (Sabour et al. 2024): Optimizes step allocation via an upper bound on KL divergence; this paper provides a complementary derivation from the contraction margin perspective, and the two are consistent as both are governed by $\sqrt{v_t}$

## Rating

⭐⭐⭐⭐⭐ A work of exceptional theoretical depth that seamlessly unifies fractal geometry with diffusion models, providing first-principles explanations for multiple previously isolated empirical techniques. The mathematical derivations are rigorous and the experimental validation is comprehensive (CIFAR-10 + CelebA-HQ), offering paradigm-level insights for the understanding and design of diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Smoothing the Score Function for Generalization in Diffusion Models: An Optimization-based Explanation Framework](smoothing_score_function_generalization_diffusion_models.md)
- [\[CVPR 2026\] Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache](dpcache_denoising_path_planning_diffusion_accel.md)
- [\[ICLR 2026\] Diffusion Fine-Tuning via Reparameterized Policy Gradient of the Soft Q-Function](../../ICLR2026/image_generation/diffusion_fine-tuning_via_reparameterized_policy_gradient_of_the_soft_q-function.md)
- [\[NeurIPS 2025\] Energy Loss Functions for Physical Systems](../../NeurIPS2025/image_generation/energy_loss_functions_for_physical_systems.md)
- [\[ICLR 2026\] Asynchronous Denoising Diffusion Models for Aligning Text-to-Image Generation](../../ICLR2026/image_generation/asynchronous_denoising_diffusion_models_for_aligning_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
