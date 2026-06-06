---
title: >-
  [Paper Note] Flow Matching Neural Processes
description: >-
  [NeurIPS 2025][Image Generation][neural processes] This paper proposes FlowNP, which integrates flow matching into the neural process framework. By employing a transformer to predict velocity fields at target points…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "neural processes"
  - "flow matching"
  - "stochastic processes"
  - "conditional generation"
  - "transformer"
date: 2026-05-08
content_hash: e7b9d9837910e780
---

# Flow Matching Neural Processes

**Conference**: NeurIPS 2025
**arXiv**: [2512.23853](https://arxiv.org/abs/2512.23853)  
**Code**: [danrsm/flowNP](https://github.com/danrsm/flowNP)  
**Area**: Image Generation
**Keywords**: neural processes, flow matching, stochastic processes, conditional generation, transformer

## TL;DR

This paper proposes FlowNP, which integrates flow matching into the neural process framework. By employing a transformer to predict velocity fields at target points, FlowNP enables parallel sampling from conditional distributions, achieving state-of-the-art performance on three benchmarks spanning 1D Gaussian processes, image data, and meteorological data.

## Background & Motivation

Neural Processes (NPs) are a class of models that learn stochastic processes directly from data, providing predictions of conditional distributions over arbitrary target point sets. Existing approaches suffer from the following core issues:

**Latent variable models (CNP/NP/ANP)** tend to underfit training functions and struggle to capture complex global uncertainty structure. ANP in particular tends to explain uncertainty through local variance rather than generating globally coherent samples.

**Autoregressive models (TNP-A)**, while more expressive, require sequential point-by-point sampling whose cost scales linearly with the number of target points. Moreover, the first point in the autoregressive order can only model a Gaussian distribution, precluding multimodal structures.

**Diffusion-based models (NDP)** learn only the joint distribution; generating conditional samples requires auxiliary guidance methods, increasing complexity and computational overhead.

Flow matching, a next-generation generative paradigm closely related to diffusion models, has demonstrated strong performance across modalities including images and video. It defines continuous probability paths from a simple distribution to the data distribution, using ODE solvers for sampling and likelihood computation in a conceptually clean and flexible manner. The authors identify that directly incorporating the flow matching paradigm into the NP framework with amortized conditioning training can simultaneously address all three issues above.

## Method

### Overall Architecture

FlowNP employs a transformer architecture as the velocity predictor, with inputs consisting of two types of tokens:

- **Context tokens**: representing observed function values $(x^{ctx}, y^{ctx})$, always using true function values at fixed time $t=1$
- **Target tokens**: representing function points to be predicted $(x^{tgt}, y_t^{tgt})$, where $y_t^{tgt}$ is an intermediate value along the probability path

**Full self-attention** is applied across all tokens without positional encodings (which would impose ordering dependence), ensuring permutation invariance (exchangeability) with respect to both context and target sets. The model outputs velocity vectors $u_t$ for each target token to drive ODE integration.

### Key Designs

**Token construction**: Each target token concatenates location $x^{tgt}_i$, current time $t$, and intermediate value $y_t^{tgt_i}$, then maps to the latent space via an embedding layer:

$$\text{token}^{tgt_i} = \text{embed}([x^{tgt_i}, t, y_t^{tgt_i}])$$

Context tokens are constructed analogously using true values $y^{ctx}$ and $t=1$:

$$\text{token}^{ctx_i} = \text{embed}([x^{ctx_i}, 1, y^{ctx_i}])$$

**Amortized conditioning training**: Unlike NDP, which learns only the joint distribution, FlowNP incorporates context directly as independent tokens during training, allowing target predictions to be naturally conditioned on context through the full attention mechanism. This design enables direct conditional sample generation at inference without auxiliary guidance or replacement methods.

**Sampling procedure**: Initial noise is drawn as $y_0^{tgt} \sim \mathcal{N}(0, I)$, followed by Euler integration of the ODE over $t \in [0, 1]$, with target values updated at each step using the model-predicted velocity field: $y^{tgt} \leftarrow y^{tgt} + \delta \hat{u}_t$. All target points are updated **in parallel**, with no dependence on autoregressive ordering.

**Likelihood computation**: The reverse ODE (from $t=1$ to $t=0$) maps data back to the noise space; the Hutchinson trace estimator computes the Jacobian determinant to correct for volume changes, and likelihood is then evaluated under the standard normal.

**Stochastic sampling variant**: For lower-capacity models (e.g., CelebA experiments), injecting a small amount of noise and rescaling the velocity at each sampling step produces more coherent samples:

$$y^{tgt} \leftarrow y^{tgt} + \delta \alpha_t \hat{u}_t + \delta \sigma_t \nu, \quad \nu \sim \mathcal{N}(0, I)$$

where $\alpha_t = 1 + t(1-t)$ and $\sigma_t = 0.2 t^2(1-t)^2$.

### Loss & Training

The standard squared-error conditional flow matching objective is adopted:

$$\mathcal{L}(\theta) = \mathbb{E}\| u^\theta(y_t^{tgt}, t, x^{tgt}, x^{ctx}, y^{ctx}) - (y^{tgt} - y_0^{tgt}) \|^2$$

The expectation is taken over function samples $f \sim \mathcal{F}$, context/target set sampling, time $t \sim \mathcal{U}[0,1]$, and noise $y_0^{tgt} \sim \mathcal{N}(0, I)$. Intermediate values are constructed by linear interpolation: $y_t^{tgt} = t \cdot y^{tgt} + (1-t) \cdot y_0^{tgt}$ (conditional optimal-transport schedule).

## Key Experimental Results

### Main Results: 1D GP Benchmark (Log-likelihood, ↑ higher is better)

| Model | RBF | Matérn-5/2 | Periodic | Fixed-Noisy RBF | Fixed-Noisy Matérn |
|-------|-----|-----------|----------|----------------|--------------------|
| CNP | 0.31 | 0.12 | -0.63 | -1.00 | -1.09 |
| NP | 0.31 | 0.13 | -0.61 | -1.13 | -1.18 |
| ANP | 1.10 | 0.85 | -0.89 | -0.87 | -0.98 |
| TNP | 1.65 | 1.29 | -0.58 | 0.68 | 0.30 |
| NDP | 1.20 | 0.94 | -0.54 | 0.71 | 0.27 |
| **FlowNP** | **1.69** | **1.30** | **-0.50** | **0.71** | **0.30** |

### Main Results: Image and Meteorological Data (Log-likelihood, ↑ higher is better)

| Model | EMNIST 0-9 | EMNIST 10-46 | CelebA | ERA5 |
|-------|-----------|-------------|--------|------|
| CNP | 1.27 | 0.73 | 2.10 | 4.06 |
| TNP | 2.08 | 1.80 | 3.95 | 11.32 |
| NDP | 1.58 | 1.47 | 4.28 | 6.76 |
| **FlowNP** | **2.50** | **2.42** | **6.37** | **12.79** |

FlowNP achieves a score of 6.37 on CelebA, substantially outperforming NDP (4.28) and TNP (3.95). On the ERA5 meteorological dataset, it also surpasses the second-best method TNP (11.32) with a score of 12.79.

### Ablation Study: Key Design Factors in FlowNP vs. NDP

| Configuration | Network Output | Noise Schedule | Conditioning | RBF | EMNIST |
|--------------|---------------|----------------|--------------|-----|--------|
| NDP | clean $y_1$ | linear-vp | unconditional | 1.20 | 1.58 |
| diffusion:clean | clean $y_1$ | linear-vp | conditional | 1.38 | 1.64 |
| flow:lin-vp | velocity $y_1-y_0$ | linear-vp | conditional | 0.41 | 0.48 |
| flow:joint | velocity $y_1-y_0$ | cond-ot | **unconditional** | **1.73** | **2.54** |
| **FlowNP** | velocity $y_1-y_0$ | **cond-ot** | **conditional** | **1.69** | **2.50** |

The ablation reveals three key findings: (1) predicting flow velocity outperforms predicting clean data or noise; (2) the cond-OT schedule ($\alpha_t=t, \beta_t=1-t$) substantially outperforms the linear-vp schedule; (3) unconditional joint training yields slightly higher likelihood but requires auxiliary methods for sampling, whereas conditional training sacrifices marginal likelihood to enable direct sampling.

### Key Findings

- **Generation speed**: On GP tasks, FlowNP generates one sample in 0.2s vs. TNP at 0.8s and NDP at 0.5s; on EMNIST, FlowNP takes 4.6s vs. TNP at 72.6s and NDP at 10.4s, representing speedups of **15.8×** and **2.3×** respectively.
- **Multimodal distributions**: On the random step-function experiment, TNP cannot capture the bimodal distribution at discontinuities because each step is constrained to a Gaussian; FlowNP's marginal distributions clearly exhibit the bimodal structure.
- **ODE steps–accuracy trade-off**: Likelihood evaluation converges after approximately 60 steps, with sampling quality following a similar trend, providing flexible control over the accuracy–speed trade-off.

## Highlights & Insights

1. **Conceptual simplicity with strong effectiveness**: The entire model requires only a standard transformer and a flow matching loss, with no need for causal masking, latent variables, ELBOs, or auxiliary guidance—making it among the simplest NP methods to implement.
2. **Global coherence via parallel sampling**: TNP's point-by-point autoregressive sampling may yield discontinuous or inconsistent samples, whereas FlowNP updates all target points simultaneously, producing smoother and globally coherent function samples.
3. **Critical role of the noise schedule**: The ablation shows that combining flow matching with the linear-vp schedule leads to catastrophic failure (RBF score of only 0.41), revealing the decisive importance of the cond-OT schedule for the success of flow matching in the NP setting—a point insufficiently discussed in prior literature.
4. **Elegant representation of uncertainty**: FlowNP's conditional distributions can represent genuine multimodal uncertainty (e.g., at step function discontinuities), rather than collapsing to a single Gaussian approximation as in CNP/ANP.

## Limitations & Future Work

1. **Iterative sampling and likelihood computation**: Both sampling and likelihood evaluation require multi-step ODE solving (100 steps by default), each step involving a full forward pass. TNP requires only a single forward pass for likelihood computation.
2. **No theoretical consistency guarantees**: Although the training paradigm empirically promotes consistency (marginal and conditional), the model architecture provides no mathematical guarantees.
3. **Scalability to large-scale data**: Current experiments use small transformers (6 layers, 128 dimensions); CelebA experiments use a larger model but still operate at the pixel level, and scaling to high-resolution images poses computational challenges.
4. **Sensitivity to noise schedule**: The ablation demonstrates that an incorrect schedule choice (e.g., linear-vp) causes a dramatic performance drop, requiring careful tuning in practical deployment.

## Related Work & Insights

The core contribution of this paper is introducing recent advances in flow matching—a member of the continuous normalizing flow family—into the NP framework. Notable related directions include:

- **Neural Diffusion Process (NDP)**: Models the joint distribution with a diffusion model and achieves conditioning via guidance. FlowNP's ablation demonstrates the advantage of the flow matching objective and direct conditional training.
- **TNP-A**: The autoregressive variant of Transformer NPs, previously the strongest baseline. FlowNP demonstrates that its performance can be surpassed without autoregression.
- **Shortcut Models**: The authors note in the limitations that shortcut flow methods could reduce ODE steps, representing a promising direction for acceleration.
- **Function-space diffusion**: Works such as infinite-dimensional diffusion that perform diffusion in continuous spaces may find inspiration in FlowNP's conditioning approach.

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|------------|-------|
| Novelty | 4 | The combination of flow matching and NPs is natural yet effective; the conditional training design is elegant |
| Technical Depth | 4 | Solid theoretical analysis (exchangeability/consistency discussion); comprehensive and systematic ablations |
| Experimental Thoroughness | 4 | Covers three domains (synthetic/image/meteorological), five baselines, and detailed ablations |
| Writing Quality | 5 | Clear and fluent exposition; polished figures; logical progression from motivation to method to analysis |
| Value | 4 | Open-source implementation; faster than TNP/NDP; applicable to function modeling across multiple domains |
| **Overall** | **4.2** | A high-quality contribution introducing flow matching into the NP framework, with a clean method and thorough evaluation |

## Additional Details

### ODE Solver and Step Count Analysis

The paper compares different ODE solvers in the appendix:

- **Likelihood computation**: Uses the midpoint method (second-order), with 100 steps by default; likelihood values converge after approximately 60 steps, with diminishing returns beyond that.
- **Sampling**: Uses the Euler method (first-order), with 100 steps by default; fewer steps yield marginally lower quality but faster sampling.
- This provides a practical **accuracy–speed knob**: step counts can be chosen according to deployment requirements.

### Kolmogorov Extension Theorem Perspective

The paper provides a systematic analytical framework for consistency in NP models:

| Model | Exchangeability | Marginal Consistency | Conditional Consistency |
|-------|----------------|---------------------|------------------------|
| CNP/NP | ✓ | ✓ (independent predictions) | ✗ |
| TNP-A | ✗ (order-dependent) | ✗ | ✗ |
| NDP | ✓ | ✗ (coupled via full attention) | ✓ (via joint/marginal ratio) |
| FlowNP | ✓ | Approximate (training-induced) | Approximate (training-induced) |

Although FlowNP provides no formal guarantees, consistency between marginal distributions and conditional sampling is validated empirically on the step-function experiment.

### Wall-Clock Time Comparison

| Task | FlowNP | TNP | NDP | Notes |
|------|--------|-----|-----|-------|
| GP single-sample generation | 0.2s | 0.8s | 0.5s | FlowNP is 4× faster than TNP |
| EMNIST single-sample generation | 4.6s | 72.6s | 10.4s | FlowNP is 15.8× faster than TNP |

TNP's generation time scales linearly with the number of target points $N$ (autoregressive), whereas FlowNP's scales with the number of ODE steps but is independent of $N$.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Value Gradient Guidance for Flow Matching Alignment](value_gradient_guidance_for_flow_matching_alignment.md)
- [\[NeurIPS 2025\] Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)
- [\[CVPR 2026\] RenderFlow: Single-Step Neural Rendering via Flow Matching](../../CVPR2026/image_generation/renderflow_single-step_neural_rendering_via_flow_matching.md)
- [\[ICCV 2025\] Contrastive Flow Matching (ΔFM)](../../ICCV2025/image_generation/contrastive_flow_matching.md)
- [\[NeurIPS 2025\] Equivariant Flow Matching for Symmetry-Breaking Bifurcation Problems](equivariant_flow_matching_for_symmetry-breaking_bifurcation_problems.md)

</div>

<!-- RELATED:END -->
