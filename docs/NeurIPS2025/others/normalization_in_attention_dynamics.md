---
title: >-
  [Paper Note] Normalization in Attention Dynamics
description: >-
  [NeurIPS 2025][Layer Normalization] This paper unifies various normalization schemes (Post-LN, Pre-LN, Mix-LN, Peri-LN, nGPT, sqrt-scaling) under a single framework of velocity modulation in an interacting particle syste…
tags:
  - "NeurIPS 2025"
  - "Layer Normalization"
  - "Attention Dynamics"
  - "Representation Collapse"
  - "Interacting Particle Systems"
  - "Velocity Modulation"
date: 2026-05-08
content_hash: 8a4cb9434ff16690
---

# Normalization in Attention Dynamics

**Conference**: NeurIPS 2025
**arXiv**: [2510.22026](https://arxiv.org/abs/2510.22026)  
**Code**: None  
**Area**: Deep Learning Theory, Transformer Architecture
**Keywords**: Layer Normalization, Attention Dynamics, Representation Collapse, Interacting Particle Systems, Velocity Modulation

## TL;DR

This paper unifies various normalization schemes (Post-LN, Pre-LN, Mix-LN, Peri-LN, nGPT, sqrt-scaling) under a single framework of velocity modulation in an interacting particle system on the sphere. It theoretically characterizes how each scheme affects token clustering dynamics and representation collapse, identifying Peri-LN as the theoretically optimal choice.

## Background & Motivation

Layer normalization (LayerNorm) in Transformers is a critical component that governs training stability and representation quality in deep networks. Several normalization schemes have been proposed:

- **Post-LN** (original Transformer): normalization applied after the residual connection
- **Pre-LN** (default in GPT, LLaMA): normalization applied before the attention layer, yielding more stable training
- **Mix-LN**: Post-LN for early layers, Pre-LN for later layers
- **Peri-LN** (used in Gemma-3): normalization applied both before and after attention
- **nGPT**: introduces a learnable parameter $\alpha_t$ to control update magnitude
- **sqrt-scaling**: scales residuals by the square root of depth

In practice, the "curse of depth" causes deep layers to degenerate into near-identity transformations that can be pruned without performance loss. Simultaneously, "representation collapse" limits the scalability of model depth. This paper provides a unified theoretical analysis of how different normalization schemes contribute to these phenomena from a dynamical systems perspective.

## Method

### Overall Architecture

The core idea is to model token representations across Transformer layers as an interacting particle system on the sphere $\mathcal{S}^{d-1}$. Each token is decomposed as $x_k = r_k \cdot \theta_k$, where $\theta_k$ is the directional unit vector and $r_k$ is the magnitude. Since a normalization step typically precedes the final decoding layer, the paper focuses on the evolution of directions $\theta_k$.

The unified dynamical equation is formulated as Normalized Attention (NA) dynamics:

$$\dot{\theta}_j(t) = \frac{1}{s_j(t)} \mathbf{P}_{\theta_j(t)} A_j^t(\Theta(t))$$

where $s_j(t)$ is a velocity modulation factor determined by the normalization scheme, and $\mathbf{P}_\theta$ denotes projection onto the tangent space of the sphere.

### Key Designs

1. **Unified Velocity Modulation Perspective**: All normalization schemes can be characterized by different choices of $s_j(t)$ and $\dot{r}_j(t)$. Post-LN sets $s_j = 1$ (constant speed); Pre-LN sets $s_j = r_j(t)$ (deceleration as magnitude grows); Peri-LN sets $s_j = r_j(t)\|A_j^t\|$ (double deceleration); nGPT sets $s_j = \alpha_t^{-1}\|A_j^t\|$ (learnable control). This unified perspective constitutes the central contribution of the paper.

2. **Asymptotic Clustering Theorem (Theorem 3.1)**: Under the simplified setting $Q=K=V=I_d$, the paper proves that token directions under Post-LN, nGPT, and sqrt-scaling almost surely converge to a synchronized cluster, while Pre-LN, Mix-LN, and Peri-LN also cluster when magnitude growth ceases. Convergence is established via a generalization of the Łojasiewicz inequality.

3. **Initial and Terminal Velocity Analysis (Theorems 4.1–4.3)**:

    - **Initial velocity**: Peri-LN and nGPT produce $O(1)$ angular displacement in early layers, whereas Post-LN and Pre-LN yield only $O(\log n / d)$, a gap of $\Omega(\min(d/\log n, \sqrt{n/\log n}))$.
    - **Terminal velocity**: Post-LN exhibits exponential clustering decay $Ce^{-2t}$; Pre-LN, Peri-LN, and Mix-LN exhibit polynomial decay $C/t^3$; nGPT depends on the choice of $\alpha_t$.
    - Polynomial decay implies that tokens continue to evolve meaningfully in deep layers, enabling better utilization of intermediate representations and resistance to collapse.

### Theoretical Tools

- Riemannian gradient flow theory on the sphere
- Symmetric orthogonal initialization for simplified ODE analysis
- Local cone initialization for terminal behavior analysis
- Tracking decay rate of within-cluster variance $\text{Var}(t)$

## Key Experimental Results

### Cosine Similarity Evolution under Symmetric Initialization (Theoretical ODE, $\beta=5, n=256$)

| Normalization | Initial Velocity $\dot{\gamma}(0)$ | Terminal Velocity $\dot{\gamma}(\infty)$ | Clustering Rate |
|---|---|---|---|
| Post-LN | $\frac{2}{e^\beta + n - 1}$ | $Ce^{-2t}$ (exponential decay) | Fastest clustering → deep collapse risk |
| Pre-LN | $\frac{2}{r_0(e^\beta + n-1)}$ | $C/t^3$ (polynomial decay) | Slow clustering → collapse-resistant |
| Peri-LN | $\frac{2}{r_0\sqrt{e^{2\beta}+n-1}}$ | $C/t^3$ (polynomial decay) | Fast early + slow terminal |
| nGPT | $\frac{2\alpha_0}{\sqrt{e^{2\beta}+n-1}}$ | Depends on $\alpha_t$ | Controllable |
| sqrt-scaling | $\frac{2}{e^\beta+n-1}$ | $Ce^{-4\sqrt{t}}/\sqrt{t}$ | Between exponential and polynomial |

### Validation with Random Initialization ($d=512, n_{\text{heads}}=1, \beta=\sqrt{d}$)

| Normalization | Token Movement in Early Layers | Collapse Rate in Deep Layers | Overall Assessment |
|---|---|---|---|
| Post-LN | Moderate | Fast (exponential) | Deep layers nearly redundant |
| Pre-LN | Slow | Slow (polynomial) | Collapse-resistant but underutilizes early layers |
| Peri-LN | **Fast** | **Slow (polynomial)** | **Best: strong at both ends** |
| nGPT ($\alpha_t \equiv 1$) | Fast | Fast (exponential) | Requires careful $\alpha_t$ tuning |
| Mix-LN | Moderate | Slow (polynomial) | Transitional solution |

### Key Findings

- **Peri-LN is theoretically optimal**: it produces large angular displacements in early layers (effective utilization of shallow layers) while maintaining polynomial clustering decay in deep layers (resistance to representation collapse), achieving the best of both regimes.
- **The curse of depth in Post-LN has a theoretical root**: exponential clustering causes tokens to nearly cease moving in deep layers, consistent with empirical observations that deep layers are prunable.
- **Pre-LN's advantage stems from magnitude growth**: the linear growth $r_j(t) \sim t$ slows clustering from exponential to polynomial, but early layers remain underutilized.
- **nGPT offers fine-grained control**: the $\alpha_t$ parameter allows per-layer behavioral tuning, but requires careful adjustment.
- **Temperature $\beta$ exponentially suppresses initial velocity**: smaller QK magnitudes are recommended in early layers.

## Highlights & Insights

- Six normalization schemes are unified within a single interacting-particle ODE framework, distinguished elegantly by a velocity modulation factor.
- The paper provides both rigorous asymptotic convergence proofs and quantitative characterizations of initial and terminal velocities, combining theoretical depth with practical insight.
- Experiments using randomly initialized Kaiming weights validate the qualitative consistency of theoretical predictions.
- The theoretical superiority of Peri-LN resonates with its practical adoption in Gemma-3.

## Limitations & Future Work

- The theoretical analysis relies on strong assumptions such as $Q=K=V=I_d$, which do not reflect the parameter diversity encountered in actual training.
- The FFN layers are omitted; only pure attention dynamics are analyzed.
- End-to-end validation through training realistic models is absent (no concrete architecture is trained and compared).
- The theoretically predicted linear magnitude growth under Pre-LN is inconsistent with the empirically observed $\sqrt{t}$ growth, due to differences between weight-tied and randomly initialized settings.
- Gradient propagation analysis is deferred to future work.

## Related Work & Insights

- This work extends the interacting particle system modeling tradition for Transformer attention dynamics established by Geshkovski et al.
- It provides a theoretical explanation for the empirical findings on the "curse of depth" (Sun et al.) and "deep layer pruning" (Gromov et al.).
- Peri-LN (Kim et al.) and nGPT (Loshchilov et al.) represent the most recent schemes analyzed within this framework.
- The paper lays groundwork for future "gradient flow analysis" and "complete analysis including MLP layers."

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The unified velocity modulation perspective is elegant, with solid theoretical contributions.
- **Experimental Thoroughness**: ⭐⭐⭐ — Primarily theoretical, supplemented by simplified experimental validation; large-scale training comparisons are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Mathematical derivations are rigorous; tables and figures are clear; argumentation proceeds in well-structured layers.
- **Value**: ⭐⭐⭐⭐ — Provides systematic theoretical guidance for normalization scheme selection in Transformers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians](recurrent_self-attention_dynamics_an_energy-agnostic_perspective_from_jacobians.md)
- [\[NeurIPS 2025\] Learning Dynamics of RNNs in Closed-Loop Environments](learning_dynamics_of_rnns_in_closed-loop_environments.md)
- [\[NeurIPS 2025\] Inferring Stochastic Dynamics with Growth from Cross-Sectional Data](inferring_stochastic_dynamics_with_growth_from_cross-sectional_data.md)
- [\[NeurIPS 2025\] FlashMD: Long-Stride, Universal Prediction of Molecular Dynamics](flashmd_long-stride_universal_prediction_of_molecular_dynamics.md)
- [\[NeurIPS 2025\] Alias-Free ViT: Fractional Shift Invariance via Linear Attention](alias-free_vit_fractional_shift_invariance_via_linear_attention.md)

</div>

<!-- RELATED:END -->
