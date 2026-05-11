---
title: >-
  [Paper Note] SoFlow: Solution Flow Models for One-Step Generative Modeling
description: >-
  [ICLR 2026][Image Generation][solution function] This paper proposes Solution Flow Models (SoFlow), which directly learn the solution function $f(x_t, t…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "solution function"
  - "flow matching"
  - "one-step generation"
  - "consistency loss"
  - "JVP-free"
date: 2026-05-08
content_hash: c3888d08d273ade0
---

# SoFlow: Solution Flow Models for One-Step Generative Modeling

**Conference**: ICLR 2026
**arXiv**: [2512.15657](https://arxiv.org/abs/2512.15657)
**Code**: [https://github.com/zlab-princeton/SoFlow](https://github.com/zlab-princeton/SoFlow)
**Area**: Diffusion Models / One-Step Generation
**Keywords**: solution function, flow matching, one-step generation, consistency loss, JVP-free

## TL;DR
This paper proposes Solution Flow Models (SoFlow), which directly learn the solution function $f(x_t, t, s)$ of the velocity ODE (mapping $x_t$ at time $t$ to the solution at time $s$). Trained from scratch via a Flow Matching loss combined with a JVP-free solution consistency loss, SoFlow achieves a 1-NFE FID of 2.96 on ImageNet 256 (XL/2), outperforming MeanFlow (3.43).

## Background & Motivation

**Background**: Consistency models (CM/iCT/ECT/sCT) and MeanFlow have enabled few-step/one-step generation, but MeanFlow's Flow Matching anchoring requires expensive Jacobian-vector product (JVP) computations (poorly optimized in PyTorch), and consistency models trained from scratch struggle to leverage CFG.

**Limitations of Prior Work**: (a) JVP is inefficient in deep learning frameworks, as it is neither a standard forward nor backward pass; (b) consistency training objectives are unstable due to stop-gradient pseudo-target drift; (c) one-step models trained from scratch do not support CFG during training.

**Key Challenge**: Learning to "jump to the endpoint in one step" currently requires either JVP (slow) or unstable objectives (poor quality).

**Goal**: Design a one-step generative framework that is JVP-free, supports CFG during training, and can be trained from scratch.

**Key Insight**: Rather than learning a velocity field and integrating it (Flow Matching), SoFlow directly learns the solution function $f(x_t, t, s)$ of the ODE. The solution function inherently satisfies two properties: (1) the initial condition $f(x_t, t, t) = x_t$, and (2) ODE solution consistency — the latter can be approximated with a consistency loss that requires no JVP.

**Core Idea**: Learn the ODE solution function instead of the velocity field, and replace JVP with a three-timepoint $(s, l, t)$ consistency constraint to enforce ODE consistency.

## Method

### Overall Architecture

The SoFlow model $f_\theta(x_t, t, s)$ takes three inputs — noisy data $x_t$, current time $t$, and target time $s$ — and outputs a prediction at time $s$. The training loss is a weighted combination: $\lambda$ × Flow Matching loss + $(1-\lambda)$ × solution consistency loss.

### Key Designs

1. **Solution Consistency Loss (JVP-Free)**:

    - **Function**: Enforces the transitivity property $f(x_t, t, s) = f(f(x_t, t, l), l, s)$ required by the ODE solution.
    - **Mechanism**: Three timepoints $s < l < t$ are sampled, and the loss is computed as $\|f_\theta(x_t, t, s) - f_{\theta^-}(x_t + (\alpha_t' x_0 + \beta_t' x_1)(l-t), l, s)\|^2$, where the intermediate point is obtained via a one-step Euler step from a teacher model (stop-gradient), requiring no JVP.
    - **Design Motivation**: MeanFlow's consistency loss requires JVP to compute the partial derivative of the velocity field with respect to time, whereas SoFlow defines consistency over the solution function, requiring only forward passes.

2. **Flow Matching Loss (Velocity Field + CFG)**:

    - **Function**: The behavior of the solution function near $s = t$ is equivalent to a velocity field, enabling simultaneous velocity prediction training.
    - **Mechanism**: Since $\partial_3 f(x_t, t, s)|_{s=t} = v(x_t, t)$, under the Euler parameterization $f_\theta(x_t, t, s) = x_t + (s-t) F_\theta(x_t, t, s)$, it follows that $F_\theta(x_t, t, t) = v_\theta(x_t, t)$.
    - **Design Motivation**: (a) Enables CFG — a guided velocity field can be used directly during training; (b) Stabilizes training by providing a well-defined FM objective.

3. **Training-Time CFG Integration**:

    - **Function**: Injects CFG signal during training rather than solely at inference.
    - **Mechanism**: The FM loss uses a guided velocity target $w(\alpha_t' x_0 + \beta_t' x_1) + (1-w) v_{\text{uncond}}$; the consistency loss replaces high-variance targets with the model's predicted guided velocity.
    - **Design Motivation**: With only one step at inference, CFG cannot be applied between intermediate states, so the model must internalize guidance during training.

### Loss & Training
- DiT architecture (B/4, L/2, XL/2) in latent space (SD-VAE).
- $\lambda$ controls the FM ratio: approximately 80% FM + 20% consistency.
- Time sampling follows a logit-normal distribution.
- Adaptive Huber loss ($p=0.5$ or $1$) is used for robustness against large-error samples.

## Key Experimental Results

### Main Results (ImageNet 256×256, 1-NFE)

| Model Size | MeanFlow FID | **SoFlow FID** | Gain |
|------------|-------------|----------------|------|
| B/2 | 6.17 | **4.85** | −1.32 |
| M/2 | 5.01 | **3.73** | −1.28 |
| L/2 | 3.84 | **3.20** | −0.64 |
| XL/2 | 3.43 | **2.96** | −0.47 |

### Ablation Study

| Configuration | FID (B/4) |
|---------------|-----------|
| 100% Consistency, 0% FM | 53.78 |
| 20% FM + 80% Consistency | 47.65 |
| 80% FM + 20% Consistency | **44.64** |
| MSE ($p=0$) | 62.93 |
| Huber ($p=0.5$) | **44.64** |
| Without CFG | 44.64 |
| With CFG ($w=1.0$) | **14.92** |

### Key Findings
- SoFlow outperforms MeanFlow across all model sizes under identical architectures and training steps.
- An 80% FM loss ratio is optimal — excessive consistency loss is detrimental, as the FM loss provides stable velocity field supervision.
- Huber loss substantially outperforms MSE (62.93 → 44.64), and robustness to large-error samples is critical.
- Training-time CFG yields dramatic gains (44.64 → 14.92), making it a key factor in one-step generation quality.

## Highlights & Insights
- **JVP-free** is the most significant practical advantage — MeanFlow requires JVP, but PyTorch's JVP implementation is inefficient (2–4× slower than backpropagation). SoFlow requires only forward passes, leading to simpler engineering.
- The conceptual shift from **solution function vs. velocity field** is illuminating — learning the "answer" (solution function) rather than the "direction" (velocity field) naturally sidesteps the integration process.
- **Training-time CFG** resolves the fundamental limitation of one-step models, which cannot apply guidance at inference due to the absence of intermediate states.

## Limitations & Future Work
- The XL/2 FID of 2.96 still lags behind multi-step methods such as SiT/DiT (~2.0 with 250 steps), indicating room for improvement in one-step generation quality.
- Validation is limited to ImageNet 256; experiments at higher resolutions (512/1024) and text-to-image settings are absent.
- The solution function requires an additional $s$ input (via positional embedding of $s-t$), introducing additional model design complexity.
- Direct comparisons with recent consistency-based methods such as sCT and IMM are lacking.

## Related Work & Insights
- **vs. MeanFlow**: The core distinction lies in the JVP-free formulation and solution function parameterization. SoFlow consistently outperforms MeanFlow across all model sizes (FID improvements of −0.47 to −1.32).
- **vs. Consistency Models (iCT/sCT)**: Both share the spirit of consistency-based training, but SoFlow's solution function consistency is more general (supporting arbitrary $(t, s)$ pairs) and naturally accommodates training-time CFG.
- **vs. Shortcut/IMM**: Both approaches learn mappings between arbitrary time pairs, but differ in theoretical motivation — SoFlow is grounded in the mathematical properties of ODE solution functions.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Solution function learning combined with a JVP-free consistency loss is a meaningful contribution, though not a fundamental paradigm shift.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive ablations and multi-scale model comparisons, but lacking high-resolution and text-to-image evaluations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Mathematical derivations are clear and the relationship between motivation and method is well-structured.
- **Value**: ⭐⭐⭐⭐ — Represents a substantive advance in one-step generation, with significant engineering value due to the JVP-free design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows](twinflow_realizing_one-step_generation_on_large_models_with_self-adversarial_flo.md)
- [\[ICLR 2026\] Laplacian Multi-scale Flow Matching for Generative Modeling](laplacian_multi-scale_flow_matching_for_generative_modeling.md)
- [\[ICLR 2026\] GenCP: Towards Generative Modeling Paradigm of Coupled Physics](gencp_towards_generative_modeling_paradigm_of_coupled_physics.md)
- [\[ICLR 2026\] DoFlow: Flow-based Generative Models for Interventional and Counterfactual Forecasting](doflow_flow-based_generative_models_for_interventional_and_counterfactual_foreca.md)
- [\[ICLR 2026\] SSCP: Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning](flow-based_single-step_completion_for_efficient_and_expressive_policy_learning.md)

</div>

<!-- RELATED:END -->
