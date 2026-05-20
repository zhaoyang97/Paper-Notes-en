---
title: >-
  [Paper Note] Flow Sampling: Learning to Sample from Unnormalized Densities via Denoising Conditional Processes
description: >-
  [ICML 2026][Scientific Computing][Diffusion Sampling] This paper proposes Flow Sampling, which reverses flow matching/diffusion models from "data-driven" to "noise-driven"—constructing a denoising diffusion drift conditi…
tags:
  - "ICML 2026"
  - "Scientific Computing"
  - "Diffusion Sampling"
  - "Flow Matching"
  - "Amortized Sampling"
  - "Riemannian Manifold"
  - "Molecular Conformation"
date: 2026-05-08
content_hash: 1f52cdd18d9e136d
---

# Flow Sampling: Learning to Sample from Unnormalized Densities via Denoising Conditional Processes

**Conference**: ICML 2026  
**arXiv**: [2605.03984](https://arxiv.org/abs/2605.03984)  
**Code**: Not released  
**Area**: Diffusion Models / Sampling / Molecular Conformation Generation  
**Keywords**: Diffusion Sampling, Flow Matching, Amortized Sampling, Riemannian Manifold, Molecular Conformation

## TL;DR
This paper proposes Flow Sampling, which reverses flow matching/diffusion models from "data-driven" to "noise-driven"—constructing a denoising diffusion drift conditioned on source noise samples. On the interpolant, the detached model samples the energy gradient of $X_1$ as the regression target, enabling the learning of efficient diffusion samplers in the absence of data, and naturally generalizing to constant curvature Riemannian manifolds.

## Background & Motivation

**Background**: Many scientific computing problems (molecular dynamics, materials, chemical reaction pathways) require sampling from unnormalized densities $q(x)=\exp(r(x))/Z$, where $r(x)$ and $\nabla r(x)$ are known but samples are unavailable. MCMC/Langevin methods are asymptotically correct but sequential and mix slowly. Recently, diffusion samplers have emerged in two categories: (a) iDEM/PIS/DDS learn sampling dynamics via Monte Carlo correction (importance sampling, resampling); (b) SOC/Schrödinger bridge approaches (Adjoint Sampling, ASBS) optimize path measure divergence to learn diffusion dynamics.

**Limitations of Prior Work**: (a) Methods of type (a) require multiple energy evaluations per step to control variance, making computation expensive; (b) Methods of type (b) require modeling optimal control or bridges, with complex training often needing auxiliary networks. Both assume Euclidean space, and extending to manifolds (spherical, hyperbolic) requires redesign.

**Key Challenge**: Standard FM/diffusion model training objectives are "given a data point $x_1$, conditionally construct a noising process so the model regresses to the velocity field of this process"; but without $x_1$, this approach fails—one can only indirectly introduce target information via the score $\nabla r$.

**Goal**: To find a "dual perspective"—given a noise point $x_0$, conditionally construct a denoising process so that the marginal still matches the target distribution; and to design a training loop that efficiently reuses energy gradients, minimizing NFE.

**Key Insight**: The authors reverse the conditional direction of FM—FM is the push-forward of the source ($p_{t|1}(x|x_1)=\tfrac{1}{\sigma_t^d}p_0(\tfrac{x-\alpha_t x_1}{\sigma_t})$, conditioned at $t=1$); this work is the push-forward of the target ($p_{t|0}(x|x_0)=\tfrac{1}{\alpha_t^d}q(\tfrac{x-\sigma_t x_0}{\alpha_t})$, conditioned at $t=0$). Both have the same marginal (equal to $p_t$), but the former requires data samples, while the latter only needs noise samples—precisely what is needed in the data-free setting.

**Core Idea**: Condition on $X_0\sim p_0$ to construct a "denoising diffusion process", supervising drift along the interpolant $X_t=\sigma_t X_0+\alpha_t X_1$ can be written in closed form as $u_{t|0}(X_t|X_0)=\dot\alpha_t X_1+\dot\sigma_t X_0+\tfrac{g_t^2}{2\alpha_t}\nabla r(X_1)$. The detached current model samples $X_1^{\bar\theta}$ and caches $\nabla r(X_1^{\bar\theta})$ in a replay buffer for repeated use.

## Method

### Overall Architecture
Two-stage alternating loop (similar to Adjoint Sampling): ① **Exploration**—use the current detached model $u^{\bar\theta}$ to simulate from $X_0\sim\mathcal{N}(0,I)$ to $X_1^{\bar\theta}$ via Euler-Maruyama, evaluate $\nabla r(X_1^{\bar\theta})$, and push the pair $(X_1^{\bar\theta}, \nabla r)$ into the replay buffer. ② **Optimization**—sample batch $(X_1, \nabla r)$ from the buffer, sample $X_0\sim p_0$ and $t\sim\text{Unif}[0,1]$, compute the conditional drift target $u_{t|0}=X_1-X_0+\gamma t\nabla r(X_1)$ along the interpolant $X_t=(1-t)X_0+tX_1$, and perform MSE regression for the model $u^\theta(X_t)$. Alternate the two stages until convergence.

### Key Designs

1. **Denoising Supervising Drift under Noise Conditioning**:

    - **Function**: Constructs a stochastic process with target marginal $q$ in the absence of data samples, learnable via regression.
    - **Mechanism**: Defines the conditional path probability $p_{t|0}(x|x_0)=\tfrac{1}{\alpha_t^d}q(\tfrac{x-\sigma_t x_0}{\alpha_t})$ (push-forward of target), and identifies two generating processes: (i) Conditional velocity $v_{t|0}(x|x_0)=\tfrac{\dot\alpha_t}{\alpha_t}(x-\sigma_t x_0)+\dot\sigma_t x_0$, solved in closed form via the interpolant $X_t=\sigma_t X_0+\alpha_t X_1$; (ii) Conditional drift $u_{t|0}=v_{t|0}+\tfrac{g_t^2}{2}\nabla\log p_{t|0}$, usable for training diffusion samplers. Proposition 3.2 proves that on the interpolant, $u_{t|0}(X_t|x_0)=\dot\alpha_t X_1+\dot\sigma_t x_0+\tfrac{g_t^2}{2\alpha_t}\nabla r(X_1)$, meaning each $X_1$ only requires **one** evaluation of $\nabla r$, which can be reused for all $t\in[0,1]$ and any $X_0$.
    - **Design Motivation**: Energy evaluation is the main cost in molecular dynamics and similar scenarios; maximizing the reuse of $\nabla r$ over time and source samples is a key engineering optimization.

2. **Fixed-Point Iterative Training with Detached Model**:

    - **Function**: Since true $X_1\sim q$ is unavailable, use $X_1^{\bar\theta}$ generated by the current model as a substitute, forming a fixed point.
    - **Mechanism**: Define $X_1^{\bar\theta}\sim p_1^{\bar\theta}$ as the endpoint sample from the detached (no gradient for $\theta$) model, solved via Euler-Maruyama $X_{t+h}^{\bar\theta}=X_t^{\bar\theta}+h u_t^{\bar\theta}(X_t^{\bar\theta})+\sqrt{2\gamma th}Z_t$. Then minimize $\mathcal{L}_{FS}=\mathbb{E}\|u^\theta(X_t)-u_{t|0}(X_t|X_0)\|^2$, where $X_1$ comes from $X_1^{\bar\theta}$ rather than the true $q$.
    - **Design Motivation**: Fixed-point iteration is a common trick in EM-like training; theoretically, the optimal solution satisfies $p_1^\theta=q$, and empirically, convergence is stable with the replay buffer.

3. **Extension to Constant Curvature Riemannian Manifolds (Hypersphere/Hyperbolic Closed-Form Drift)**:

    - **Function**: Extends Flow Sampling to $\mathbb{S}^d$, hyperbolic spaces, and other constant curvature $\kappa$ manifolds.
    - **Mechanism**: Replace the affine interpolant with the geodesic interpolant $X_t=\exp_{X_1}[(1-t)\log_{X_1}(x_0)]$; replace Euclidean Brownian motion with $P_{X_t}^\perp\circ dB_t$ to ensure diffusion remains in the tangent space. Proposition 4.1 provides a closed-form rank-1 Jacobian for the geodesic, $J_t=t T_{X_1\to X_t}P_{\dot X_1}+c_t T_{X_1\to X_t}P_{\dot X_1}^\perp$, where $c_t=\sin(t\omega_1\sqrt\kappa)/\sin(\omega_1\sqrt\kappa)$ ($\kappa>0$) or the $\sinh$ version ($\kappa<0$); this enables closed-form computation of the conditional score and drift, avoiding numerical reverse-mode differentiation.
    - **Design Motivation**: Existing diffusion samplers assume Euclidean space; applying them to spheres/hyperbolic spaces (directional data, robot pose, graph embeddings) requires rewriting the entire pipeline. The closed-form formulas provided here make generalization nearly cost-free—this is the paper's most pioneering contribution.

### Loss & Training

Linear scheduler $\alpha_t=t,\sigma_t=1-t,g_t^2=2\gamma t$; $\gamma$ is adaptively set as $\gamma=c/\sqrt{\mathbb{E}_{x_1\sim\mathcal{B}}[\|\nabla r(x_1)\|^2]+\varepsilon}$ to suppress energy gradient scale. Buffer size is 10,000–60,000, with 100–300 gradient updates per round and 128–2048 new samples per round.

## Key Experimental Results

### Main Results
Synthetic energy benchmarks (DW-4/LJ-13/LJ-55) + peptides (Ala2/Ala4) + large-scale amortized molecular conformation generation (SPICE/GEOM-DRUGS) + spherical vMF mixtures.

| Dataset | Metric | Flow Sampling | ASBS (Prev. SOTA) | Gain |
|--------|------|------|----------|------|
| DW-4 | $\mathcal{W}_2$ ↓ | **0.36** | 0.43 | -16% |
| DW-4 | $E(\cdot)\mathcal{W}_2$ ↓ | **0.11** | 0.20 | -45% |
| LJ-13 | $E(\cdot)\mathcal{W}_2$ ↓ | **0.97** | 1.99 | -51% |
| LJ-55 | $E(\cdot)\mathcal{W}_2$ ↓ | **21.32** | 28.10 | -24% |
| Ala2 JSD (NFE=1024) | ↓ | **0.018** | 0.242 | -93% |
| Ala2 Energy $\mathcal{W}_2$ (NFE=128) | ↓ | **3.58** | $10^7$ | catastrophic |

### Ablation Study / Efficiency

| NFE (Train) | SPICE Recall Cov | SPICE Recall AMR |
|------|---------|------|
| 256 (Flow Sampling) | **91.89** | **0.86** |
| 128 (Flow Sampling) | 91.39 | 0.87 |
| 64 (Flow Sampling) | 90.13 | 0.87 |
| 256 (AS) | 88.60 | 0.87 |
| 128 (AS) | 77.97 | 0.98 |
| 64 (AS) | 29.13 (collapse) | 1.43 |
| 512 (ASBS) | 89.66 | 0.86 |

### Key Findings
- Under low NFE=64 budget, AS completely collapses (Cov 5.50), while Flow Sampling remains stable (Cov 71.14); this shows that FS's supervising target has much lower variance than AS's stochastic adjoint signal, enabling robustness at low NFE.
- Training cost can be reduced by 4–8×: On SPICE, NFE=64 achieves nearly the same performance as NFE=256, meaning the exploration phase cost can be directly reduced by 4×.
- The spherical vMF mixture experiment is the first demo of diffusion sampling on curved manifolds, proving the Riemannian extension is practical, not just theoretical.
- Appendix D rigorously proves that FS and Adjoint Sampling have the same conditional expectation on Brownian bridge paths (Theorem D.1), i.e., they are essentially equivalent as control variates.

## Highlights & Insights
- **The "reversed conditional direction" reframing is the soul of this work**: Data-driven FM/DM conditions on $X_1$, while data-free Flow Sampling conditions on $X_0$. This seemingly simple symmetry leads to a huge engineering difference—FS can cover the entire time axis and source sample space with a single energy gradient evaluation, minimizing expensive energy computations.
- **The closed-form supervising target $\dot\alpha_t X_1+\dot\sigma_t x_0+\tfrac{g_t^2}{2\alpha_t}\nabla r(X_1)$** is as elegant as score matching: minimal, interpretable, and analyzable. This formula is the key technical hub integrating "data-free + diffusion + replay buffer".
- **The rank-1 Jacobian on constant curvature manifolds** (Proposition 4.1) is a mathematically beautiful result, turning manifold sampling from a concept into an engineerable algorithm. The paper provides explicit closed forms for hypersphere and hyperbolic cases, opening up diffusion sampling for directional data, hyperbolic embeddings, and more.

## Limitations & Future Work
- Fixed-point iteration lacks global convergence guarantees; replay buffer dynamics may be unstable on pathological energy landscapes (strong multimodality, large energy gradients).
- The Riemannian extension is currently limited to constant curvature; for general curvature manifolds (e.g., non-Euclidean metrics in protein conformation space), numerical Jacobian inversion is required, greatly reducing efficiency.
- Compared to SOC/Schrödinger bridge methods, FS lacks optimal control guarantees and may not generate the most efficient paths for some tasks.
- Experiments are limited to molecules/spheres and do not cover popular RL scenarios such as image reward model fine-tuning.

## Related Work & Insights
- **vs iDEM**: iDEM estimates the target score along the noising path via MC, requiring multiple energy evaluations per step; FS only needs one energy evaluation per denoising path and reuses it, greatly reducing NFE.
- **vs Adjoint Sampling (AS) / ASBS**: AS uses the stochastic adjoint method to backpropagate SOC gradients; ASBS adds a Schrödinger bridge corrector. FS does not require a corrector, directly regresses on the closed-form target, and has a simpler training logic. Theorem D.1 also proves their essential equivalence (differing by a control variate).
- **vs Tilt Matching (concurrent)**: TM also adopts a data-free reward-tilted approach but uses an annealing path; FS directly fixes a linear scheduler, offering greater simplicity.
- **Insights**: The "reversed conditional direction" idea of FS can be generalized to any scenario where "one condition is known but another needs to be constructed", such as inverse problems (constructing prior samples from observations) or retrosynthesis (backtracking reactants from products).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Dual reversal from data-driven to noise-driven + closed-form drift on constant curvature manifolds, representing true conceptual innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ DW-4/LJ-13/LJ-55 + Ala2/Ala4 + SPICE/GEOM-DRUGS + spherical vMF, broad coverage; but lacks image/text reward fine-tuning experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ The "conditional direction duality" diagram in Figure 2 is very intuitive; Algorithm 1 provides pseudocode for reproducibility; Proposition 3.2 is the key formula, presented concisely and powerfully.
- Value: ⭐⭐⭐⭐⭐ Reduces training cost by 4–8× for molecular conformation generation + first realization of manifold diffusion sampling, with direct application value in computational chemistry and directional data fields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities](../../NeurIPS2025/image_generation/progressive_inference-time_annealing_of_diffusion_models_for_sampling_from_boltz.md)
- [\[NeurIPS 2025\] Flow Matching Neural Processes](../../NeurIPS2025/image_generation/flow_matching_neural_processes.md)
- [\[ICML 2026\] Information-Geometric Adaptive Sampling for Graph Diffusion](information-geometric_adaptive_sampling_for_graph_diffusion.md)
- [\[ICML 2026\] Mixture Prototype Flow Matching for Open-Set Supervised Anomaly Detection](mixture_prototype_flow_matching_for_open-set_supervised_anomaly_detection.md)
- [\[ICLR 2026\] FlowCast: Advancing Precipitation Nowcasting with Conditional Flow Matching](../../ICLR2026/image_generation/flowcast_advancing_precipitation_nowcasting_with_conditional_flow_matching.md)

</div>

<!-- RELATED:END -->
