---
title: >-
  [Paper Note] LILAD: Learning In-context Lyapunov-stable Adaptive Dynamics Models
description: >-
  [AAAI 2026][Lyapunov stability] This paper proposes LILAD, a framework that leverages the in-context learning (ICL) capability of GPT-2 to jointly learn a dynamics model and a Lyapunov function…
tags:
  - "AAAI 2026"
  - "Lyapunov stability"
  - "in-context learning"
  - "adaptive system identification"
  - "non-stationary dynamics"
  - "GPT-2"
date: 2026-05-08
content_hash: 69ac00b2afb3aa35
---

# LILAD: Learning In-context Lyapunov-stable Adaptive Dynamics Models

**Conference**: AAAI 2026
**arXiv**: [2511.21846](https://arxiv.org/abs/2511.21846)
**Code**: [https://github.com/amitjena1992/LILAD](https://github.com/amitjena1992/LILAD)
**Area**: Other
**Keywords**: Lyapunov stability, in-context learning, adaptive system identification, non-stationary dynamics, GPT-2

## TL;DR

This paper proposes LILAD, a framework that leverages the in-context learning (ICL) capability of GPT-2 to jointly learn a dynamics model and a Lyapunov function, achieving adaptive identification of non-stationary parametric dynamical systems while guaranteeing global exponential stability. LILAD outperforms baselines such as ICL and MAML on multiple benchmark systems.

## Background & Motivation

**Background**: System identification aims to approximate dynamical systems from trajectory data. Neural networks are widely used due to their strong expressiveness, but they typically provide no guarantees on physical properties (e.g., stability) and assume stationary systems.

**Limitations of Prior Work**:
   - Stability-constrained methods (e.g., Lyapunov-constrained neural networks) assume fixed system parameters and cannot handle non-stationary dynamics.
   - Adaptive methods (e.g., meta-learning, ICL) optimize solely for prediction accuracy without guaranteeing stability of the learned model.
   - When system parameters change, retraining a stable model is prohibitively expensive for time-sensitive, safety-critical applications.

**Key Challenge**: Stability and adaptability have been studied independently, with no unified framework guaranteeing both simultaneously.

**Key Insight**: The prompt mechanism of ICL is exploited for zero-shot adaptation, while adversarial training jointly learns a Lyapunov function to enforce stability.

**Core Idea**: A GPT-2-based ICL framework jointly trains a dynamics model $G_\theta$ and a Lyapunov function $V_\phi$, and computes a state-dependent decay factor $\gamma(x)$ via bisection to strictly enforce stability.

## Method

### Overall Architecture

**Input**: A multi-task trajectory data pool comprising $M$ tasks, each corresponding to a dynamical system $x_{k+1} = f_{\vartheta_i}(x_k)$ with different sampled parameters. Two GPT-2-based models are trained: a dynamics model $G_\theta$ and a Lyapunov model $V_\phi$. At test time, only a short trajectory from the new system is required as a prompt for zero-shot inference.

### Key Designs

1. **Dual-model architecture under the ICL framework**:

    - Function: Simultaneously trains a dynamics predictor and a Lyapunov function using the same prompt structure.
    - Mechanism: A prompt $\mathscr{P}_j^i = \{x_{i,1}, f(x_{i,1}), \ldots, x_{i,j}, f(x_{i,j}), x_{i,j+1}\}$ is constructed; $G_\theta$ predicts the next state, and $V_\phi$ outputs the Lyapunov value. Both share the same architecture (linear → GPT-2 transformer → linear).
    - Design Motivation: ICL natively supports in-context adaptation, enabling generalization to new tasks without gradient updates.

2. **Output reparameterization to enforce positive semi-definiteness**:

    - Function: Forces the Lyapunov function to satisfy the positive definite condition $V(x) > 0,\ V(0) = 0$.
    - Mechanism: $V_\phi(x|\mathscr{C}) = \sigma(c \cdot \tanh(V^{\text{raw}}_\phi(x|\mathscr{C})) - c \cdot \tanh(V^{\text{raw}}_\phi(0|\mathscr{C}))) + \epsilon \|x\|^2$, where $\sigma$ is a smooth ReLU.
    - Design Motivation: The first two Lyapunov conditions are intrinsically satisfied by architectural design, so training only needs to address the third condition (exponential decay).

3. **State-dependent decay factor $\gamma(x)$**:

    - Function: Enforces stability at inference time for states that may violate the Lyapunov condition.
    - Mechanism: For violating states, solves $V_\phi(\gamma(x) \cdot G_\theta(x|\mathscr{C})|\mathscr{C}) - \beta V_\phi(x|\mathscr{C}) = 0$; the intermediate value theorem guarantees the existence of $\gamma \in [0,1]$, and bisection is used for efficient computation.
    - Design Motivation: Provides rigorous out-of-distribution stability guarantees without relying on convexity assumptions on the Lyapunov model.

### Loss & Training

Adversarial alternating training:
- **Freeze $G_\theta$, update $V_\phi$**: $\mathcal{L}^{\text{Lyap}} = \frac{1}{M(n+1)} \sum_{i,j} \max\{V_\phi(G_\theta(x|\mathscr{C})|\mathscr{C}) - \beta V_\phi(x|\mathscr{C}),\ 0\}$
- **Freeze $V_\phi$, update $G_\theta$**: $\mathcal{L}^{\text{Dyn}} = \text{MSE} + \lambda \cdot \text{Lyapunov violation penalty}$

## Key Experimental Results

### Main Results (Table 1: MAE Comparison)

| System | Dim | ICL | MAML | CVaR | Stable-Linear | LILAD |
|--------|-----|-----|------|------|---------------|-------|
| Simple Pendulum | 2 | 0.018 | 0.023 | 0.085 | 0.065 | **0.004** |
| Double Pendulum | 4 | 0.039 | 0.022 | 0.12 | 0.17 | **0.011** |
| Microgrid | 5 | **0.005** | 0.007 | 0.014 | 0.011 | **0.005** |
| SEIR | 8 | 0.022 | 0.032 | 0.077 | 1.049 | **0.017** |
| PDE-SM | 100 | 6.354 | – | – | – | **0.060** |

### Ablation Study: Stability Guarantee Comparison

| Method | Adaptive | Stability Guarantee | High-dim Scalable |
|--------|----------|--------------------|--------------------|
| ICL | ✅ | ❌ | ✅ |
| MAML | ✅ | ❌ | ❌ |
| CVaR | ❌ (robust) | ❌ | ❌ |
| Stable-Linear | ❌ | ✅ (linear) | ❌ |
| LILAD | ✅ | ✅ | ✅ |

### Key Findings
- On the high-dimensional PDE system (100-dim), LILAD achieves an MAE approximately **106× lower** than ICL (0.060 vs. 6.354).
- ICL trajectories fail to converge to the origin on certain test instances, whereas LILAD consistently guarantees convergence.
- On the Microgrid system, LILAD and ICL perform comparably, as the system is strongly damped and the stability constraint has limited effect.
- MAML and CVaR fail to scale to high-dimensional systems.

## Highlights & Insights
- **Cross-disciplinary innovation: ICL + Lyapunov**: This is the first work to unify in-context learning with Lyapunov stability theory. ICL provides adaptability while Lyapunov theory provides safety guarantees — the two are complementary.
- **State-dependent decay factor $\gamma(x)$**: Unlike prior work, no convexity assumption on the Lyapunov function is required. The intermediate value theorem elegantly guarantees the existence of a solution, which is then efficiently computed via bisection.
- **Adversarial training strategy**: The dynamics model and Lyapunov function are updated alternately, allowing both to converge toward a consistent, stable representation.

## Limitations & Future Work
- Applicable only to autonomous systems (no external control inputs); extension to controlled systems is not addressed.
- Assumes all parametric systems share the same equilibrium point (the origin), limiting the scope of applicability.
- Using GPT-2 as the backbone requires large models and extended training in high-dimensional settings (PDE-SM requires 2M epochs).
- Bisection-based computation of $\gamma(x)$ introduces additional inference-time overhead.
- The conservatism of the Lyapunov function is not discussed — excessive decay may degrade dynamics prediction accuracy.

## Related Work & Insights
- **vs. Kolter & Zico (2019)**: Their work jointly trains a neural network dynamics model and a Lyapunov function but assumes stationary systems; LILAD extends this to non-stationary settings via ICL.
- **vs. standard ICL (Forgione et al.)**: Standard ICL optimizes only for prediction accuracy; LILAD augments this with stability constraints.
- **vs. MAML**: MAML requires gradient updates to adapt to new tasks, whereas LILAD adapts zero-shot via prompting.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to unify adaptability and stability; the combination of ICL and Lyapunov theory is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five benchmark systems spanning varying dimensionalities; lacks validation on real physical systems.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous, though the paper is lengthy.
- Value: ⭐⭐⭐⭐ Significant implications for adaptive modeling in safety-critical systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] In-Context Algebra](../../ICLR2026/others/in-context_algebra.md)
- [\[AAAI 2026\] Beyond World Models: Rethinking Understanding in AI Models](beyond_world_models_rethinking_understanding_in_ai_models.md)
- [\[ICLR 2026\] LPWM: Latent Particle World Models for Object-Centric Stochastic Dynamics](../../ICLR2026/others/latent_particle_world_models_self-supervised_object-centric_stochastic_dynamics_.md)
- [\[AAAI 2026\] Deviation Dynamics in Cardinal Hedonic Games](deviation_dynamics_in_cardinal_hedonic_games.md)
- [\[AAAI 2026\] Verification-Guided Context Optimization for Tool Calling via Hierarchical LLMs-as-editors](verification-guided_context_optimization_for_tool_calling_via_hierarchical_llms-.md)

</div>

<!-- RELATED:END -->
