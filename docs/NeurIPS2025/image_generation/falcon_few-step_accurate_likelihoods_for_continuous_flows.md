---
title: >-
  [Paper Note] FALCON: Few-step Accurate Likelihoods for Continuous Flows
description: >-
  [NeurIPS 2025][Image Generation][Continuous Normalizing Flows] This paper proposes FALCON, which employs a hybrid training objective (flow matching + mean velocity loss + invertibility regularization) to enable continuou…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Continuous Normalizing Flows"
  - "Boltzmann Generators"
  - "Flow Matching"
  - "Importance Sampling"
  - "Molecular Sampling"
date: 2026-05-08
content_hash: 017878a37e4ddfdc
---

# FALCON: Few-step Accurate Likelihoods for Continuous Flows

**Conference**: NeurIPS 2025
**arXiv**: [2512.09914](https://arxiv.org/abs/2512.09914)
**Code**: To be confirmed
**Area**: Image Generation
**Keywords**: Continuous Normalizing Flows, Boltzmann Generators, Flow Matching, Importance Sampling, Molecular Sampling

## TL;DR

This paper proposes FALCON, which employs a hybrid training objective (flow matching + mean velocity loss + invertibility regularization) to enable continuous normalizing flows to provide sufficiently accurate likelihood estimates under few-step sampling, achieving Boltzmann sampling two orders of magnitude faster than conventional CNFs.

## Background & Motivation

Sampling molecular configurations from the Boltzmann distribution $p(x) \propto \exp(-\mathcal{E}(x))$ is a central challenge in statistical physics. Traditional methods (molecular dynamics, MCMC) are prone to becoming trapped in local minima and produce highly correlated samples. Boltzmann Generators (BGs) address this by training generative models to learn an approximation $p_\theta(x)$ of the target distribution, then applying self-normalized importance sampling (SNIS) to correct toward the true distribution, enabling efficient and statistically consistent sampling.

State-of-the-art BGs are primarily based on continuous normalizing flows (CNFs) trained via flow matching. CNFs offer strong expressiveness and stable training, but **likelihood computation is extremely expensive**: it requires computing the full Jacobian trace at every integration step, and thousands of discretization steps are needed to control numerical error. This severely limits the applicability of CNFs to large-scale molecular sampling in practice.

Recent few-step generative models (consistency models, MeanFlow, etc.) can greatly accelerate sampling, but they cannot natively provide high-accuracy likelihood estimates—because the learned flow map does not guarantee invertibility when training has not fully converged, rendering the standard change-of-variables formula inapplicable. This makes them unsuitable for importance-sampling applications that require precise likelihoods.

## Core Problem

How can one design a continuous flow model that simultaneously supports **few-step efficient sampling** and provides **sufficiently accurate likelihood estimates** for importance sampling? Four conditions must be satisfied concurrently: invertibility, regression-loss training, few-step generation, and free-form architecture. No existing method satisfies all four.

## Method

### 1. Core Idea of FALCON

FALCON is built on a flow map model $u_\theta(x_s, s, t)$ that learns the mean velocity from time $s$ to $t$. By introducing **invertibility regularization**, the flow map is encouraged to remain invertible throughout training, enabling efficient likelihood computation via the change-of-variables formula.

### 2. Key Theoretical Insights

- **Proposition 1**: Under optimal conditions (where $u_\theta^*$ perfectly minimizes the mean velocity objective), the flow map $X_u(\cdot, s, t)$ is invertible everywhere and the change-of-variables formula holds.
- **Proposition 2**: A weaker condition—so long as the invertibility loss $\mathcal{L}_{\text{inv}}$ is minimized, the flow map is invertible and the change-of-variables formula holds. This does not require the flow map to perfectly match the continuous-time flow.

This implies that exact fitting of the continuous flow is unnecessary; guaranteeing invertibility of the mapping suffices for effective Boltzmann generation.

### 3. Hybrid Training Objective

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{cfm}}(\theta) + \lambda_{\text{avg}} \mathcal{L}_{\text{avg}}(\theta) + \lambda_r \mathcal{L}_{\text{inv}}(\theta)$$

Three components:

- **$\mathcal{L}_{\text{cfm}}$**: Standard flow matching loss, learning the instantaneous velocity field.
- **$\mathcal{L}_{\text{avg}}$**: Mean velocity loss (equivalent to MeanFlow), training the model to accurately predict target positions over large step sizes. Implemented efficiently via JVP (Jacobian-vector product) with a single forward automatic differentiation call.
- **$\mathcal{L}_{\text{inv}}$**: Invertibility regularization, implemented as a cycle-consistency loss:

$$\mathcal{L}_{\text{inv}}(\theta) = \mathbb{E}_{s,t,x_s} \|x_s - X_u(X_u(x_s, s, t), t, s)\|^2$$

This requires that applying the forward map followed by the reverse map recovers the original point, thereby encouraging invertibility.

### 4. Parameterization Trick

Since FALCON requires flow maps in both forward and backward directions, a discontinuity exists at $s = t$. This is resolved by parameterizing the model as $u_\theta(x_s, s, t) = \text{sign}(t - s) \cdot h_\theta(x_s, s, t)$, eliminating the discontinuity when the direction switches.

### 5. Scalable Architecture

Owing to the low inference cost of few-step sampling, FALCON is the first Boltzmann Generator to employ a **Diffusion Transformer (DiT)** architecture (with an additional time-embedding head), breaking the previous limitation of using only small equivariant networks with 2.3 million parameters. Soft SO(3) equivariance is achieved via data augmentation, and translational invariance is enforced by mean subtraction.

### 6. Likelihood Computation

For few-step (e.g., 4-step) invertible flow maps, likelihood is computed via the change-of-variables formula:

$$\log p_t(x_t) = \log p_s(x_s) - \log |\det \mathbf{J}_{X_u}(x_s)|$$

Each step requires only $d$ function evaluations to compute the Jacobian, and the determinant computation is relatively negligible. Because the number of steps is extremely small, the total computational cost is far lower than the thousands of integration steps required by conventional CNFs.

## Key Experimental Results

Evaluation is conducted on four molecular systems: alanine dipeptide (ALDP), tri-alanine (AL3), alanine tetrapeptide (AL4), and hexa-alanine (AL6).

### Comparison with Continuous Flows (ECNF++)

| System | Method | ESS ↑ | $\mathcal{E}$-$\mathcal{W}_2$ ↓ | $\mathbb{T}$-$\mathcal{W}_2$ ↓ |
|--------|--------|-------|----------------------------------|----------------------------------|
| AL3 | ECNF++ | 0.003 | 2.206 | 0.962 |
| AL3 | **FALCON** | **0.077** | **0.544** | **0.452** |
| AL4 | ECNF++ | 0.016 | 5.638 | 1.002 |
| AL4 | **FALCON** | **0.055** | **0.686** | **0.858** |
| AL6 | ECNF++ | 0.006 | 10.668 | 1.902 |
| AL6 | **FALCON** | **0.060** | **0.892** | **1.256** |

### Comparison with Discrete Flows (SBG)

- FALCON with 4-step sampling outperforms SBG (current state-of-the-art discrete NF) across all larger molecular systems.
- Even when SBG uses $5 \times 10^6$ samples (250× more than FALCON), its energy Wasserstein distance remains notably worse than FALCON's.

### Computational Efficiency

- FALCON inference is **two orders of magnitude faster** than CNFs of comparable performance.
- On hexa-alanine, conventional CNFs cannot generate $10^4$ samples within a reasonable time.
- Total training + inference time: FALCON is the fastest across all systems (AL6: 25.76 h vs. CNF 82.10 h vs. SBG 57.50 h).

## Highlights & Insights

1. **Theoretical rigor**: Proposition 2 is proposed and proved, establishing the sufficiency of invertibility regularization and providing a solid theoretical foundation for the method.
2. **Two orders of magnitude speedup**: Inference is more than 100× faster than conventional CNFs, making large-scale molecular sampling practical.
3. **Architectural freedom**: Few-step inference enables the first use of large-scale Transformers such as DiT in Boltzmann Generators.
4. **Efficient implementation**: JVP allows simultaneous computation of the forward pass and gradients in a single call; $\mathcal{L}_{\text{cfm}}$ and $\mathcal{L}_{\text{avg}}$ can be merged into a unified loss.
5. **Statistical consistency**: Combined with SNIS reweighting, generated samples theoretically converge to the true Boltzmann distribution.

## Limitations & Future Work

1. **Imperfect invertibility**: In practice, $\mathcal{L}_{\text{inv}}$ can only be approximately minimized; residual cycle-consistency error still affects the accuracy of likelihood estimates.
2. **Limited molecular system scale**: The largest test system is hexa-alanine (6 residues), leaving a substantial gap relative to real large-scale systems such as proteins.
3. **Dependence on training data**: The method relies on biased MD simulation data; data quality and degree of bias affect final performance.
4. **Soft equivariance constraint**: Soft SO(3) equivariance achieved through data augmentation is less strict than exact equivariant architectures.
5. **Inconsistent ESS performance**: ESS on ALDP is inferior to ECNF++, suggesting that invertibility regularization may introduce some accuracy loss on small molecules.

## Related Work & Insights

| Method | Invertible | Regression Loss | Few-step | Free Architecture | Likelihood Accuracy |
|--------|-----------|-----------------|----------|-------------------|---------------------|
| ECNF/ECNF++ | ✓ (continuous) | ✓ | ✗ (thousands of steps) | ✓ | High but slow |
| SBG (TARFlow) | ✓ | ✗ (MLE) | ✓ (1 step) | ✗ | Exact but limited expressiveness |
| RegFlow | ✓ | ✓ | ✓ | ✗ | Moderate |
| MeanFlow/ConsistencyFM | ✗ | ✓ | ✓ | ✓ | Not computable |
| **FALCON** | **✓** | **✓** | **✓** | **✓** | **Fast and accurate** |

FALCON is the first method to satisfy all four conditions simultaneously. The key distinction lies in the introduction of invertibility regularization, which bridges the gap in likelihood estimation capability inherent to few-step flow models.

The following broader insights emerge from this work:

1. **Generalization potential**: The idea of driving invertibility via cycle-consistency regularization can be extended to any flow model application requiring accurate likelihoods (image generation, anomaly detection, etc.).
2. **Decoupling architecture and invertibility**: Decomposing "good generation" and "invertibility" into separate loss terms is a transferable design principle—invertibility can be obtained without explicitly designing invertible architectures.
3. **Inference-time scaling**: The SNIS framework naturally supports inference-time scaling (more samples → more accurate estimates), aligning with current trends in AI inference scaling.
4. **Transformers for scientific computing**: This work demonstrates the feasibility of DiT in scientific computing (molecular sampling), potentially inspiring further transfer of vision/language Transformers to scientific tasks.

## Rating
- Novelty: 8/10 — The combination of invertibility regularization with few-step flows is novel and theoretically well-supported.
- Experimental Thoroughness: 8/10 — Four molecular systems, multiple baselines, and complete ablation studies; however, system scale remains limited.
- Writing Quality: 9/10 — Clear structure, well-motivated contributions, and tight integration of theory and experiments.
- Value: 8/10 — A two-orders-of-magnitude speedup offers significant practical value for scientific computing, though a gap to large-scale applications remains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Amortized Sampling with Transferable Normalizing Flows](amortized_sampling_with_transferable_normalizing_flows.md)
- [\[NeurIPS 2025\] Multimodal Generative Flows for LHC Jets](multimodal_generative_flows_for_lhc_jets.md)
- [\[NeurIPS 2025\] On the Relation between Rectified Flows and Optimal Transport](on_the_relation_between_rectified_flows_and_optimal_transport.md)
- [\[ICCV 2025\] SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation](../../ICCV2025/image_generation/sana-sprint_one-step_diffusion_with_continuous-time_consistency_distillation.md)
- [\[NeurIPS 2025\] BoltzNCE: Learning Likelihoods for Boltzmann Generation with Stochastic Interpolants](boltznce_learning_likelihoods_for_boltzmann_generation_with_stochastic_interpola.md)

</div>

<!-- RELATED:END -->
