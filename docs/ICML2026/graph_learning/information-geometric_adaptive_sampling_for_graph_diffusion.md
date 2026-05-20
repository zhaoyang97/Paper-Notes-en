---
title: >-
  [Paper Note] Information-Geometric Adaptive Sampling for Graph Diffusion
description: >-
  [ICML 2026][Graph Learning][Graph Diffusion] This work treats the sampling trajectory of the reverse SDE in graph diffusion as a parameterized curve on a Riemannian statistical manifold…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Graph Diffusion"
  - "Fisher-Rao Metric"
  - "Adaptive Step Size"
  - "Information Geometry"
  - "Molecular Generation"
date: 2026-05-08
content_hash: fec9e134c75bccad
---

# Information-Geometric Adaptive Sampling for Graph Diffusion

**Conference**: ICML 2026  
**arXiv**: [2605.00250](https://arxiv.org/abs/2605.00250)  
**Code**: None  
**Area**: Diffusion Models / Graph Generation / Adaptive Sampling  
**Keywords**: Graph Diffusion, Fisher-Rao Metric, Adaptive Step Size, Information Geometry, Molecular Generation

## TL;DR
This work treats the sampling trajectory of the reverse SDE in graph diffusion as a parameterized curve on a Riemannian statistical manifold, deriving a training-free Drift Variation Score (DVS) from the Fisher-Rao metric to measure the local "information curvature" of the trajectory. Step sizes are adaptively scaled so that each step advances an equal length on the information manifold, achieving higher FCD / MMD fidelity with fewer steps in molecular (QM9/ZINC250k) and graph (Planar/SBM/Ego) generation.

## Background & Motivation

**Background**: Graph diffusion models (e.g., GDSS, GruM) perform joint denoising on node features $\mathbf{X}$ and adjacency $\mathbf{A}$ using reverse SDEs. Mainstream samplers use fixed-step predictor-corrector frameworks such as Euler-Maruyama or Heun.

**Limitations of Prior Work**: (i) Fixed step size implicitly assumes "equal time intervals = equal distributional change," but the reverse SDE dynamics are highly non-uniform: drift is smooth in high-noise regimes and changes abruptly (stiff) in low-noise regimes; (ii) Heuristic quadratic schedules are statically preset and cannot adapt to data/model; (iii) Adaptive step sizes based on local truncation error estimate error in state space, ignoring the intrinsic geometry of probability paths; (iv) The asynchronous denoising of nodes vs. edges in graph data leads to misaligned stiff moments, making it hard for a single step size to suffice.

**Key Challenge**: To uniformly characterize the "rate of distributional evolution," one must abandon using time $t$ as arc length—time is extrinsic, while distributional distance is intrinsic geometry.

**Goal**: (i) Provide an "information geometric" adaptive step size criterion for graph diffusion reverse SDEs; (ii) Enable nodes and edges to have their own stiff detection signals and make joint decisions; (iii) Plug-and-play, no retraining required.

**Key Insight**: Regard the Gaussian transition kernel $p(x_{t+dt}|x_t; f_t)$ induced by the reverse SDE at each moment as a point on a statistical manifold parameterized by drift $f_t$, with the entire sampling trajectory forming a curve. Use the Fisher-Rao metric (the unique invariant metric per Chentsov's theorem) to measure the arc length of this curve—arc length thus represents the intrinsic distance of distributional change.

**Core Idea**: Each step $\Delta s^2 \approx$ constant ⇒ $\Delta t \propto 1/V_t$, where $V_t = \|d f_t\|^2 / g_t^2$ is the DVS.

## Method

### Overall Architecture
For the reverse SDE $dx_t = f_t dt + g_t d\bar{w}_t$, at each discrete time $t_k$, compute drift differences for nodes and edges to obtain $V_{\mathbf{X},k}$ and $V_{\mathbf{A},k}$, then smooth them with EMA to get $\bar{V}_{\mathbf{X}}, \bar{V}_{\mathbf{A}}$. Node and edge step sizes are computed via a power law $\Delta t \propto (\kappa_{\text{ref}}/\bar{V})^\beta$ (clipped to $[\Delta t_{\min}, \Delta t_{\max}]$), and the minimum of the two is used for this step. Update $(\mathbf{X}, \mathbf{A})$ with a single Euler or Heun solver step, and multiply $\bar{V}$ by $\gamma$ to feed back into the next step. The entire process does not modify the pretrained score network and is only inserted at sampling.

### Key Designs

1. **Fisher-Rao Line Element + Drift Variation Score (DVS)**:

    - **Function**: Quantifies "how fast the distribution changes on the statistical manifold" as a dimensionless, online-evaluable scalar.
    - **Mechanism**: Write the transition kernel as $p(x_{t+dt}|x_t; f_t) = \mathcal{N}(x_t + f_t dt, g_t^2 dt I)$, take the log-likelihood gradient with respect to $f_t$ to obtain the Fisher information matrix $\mathcal{I}(f_t) = \frac{dt}{g_t^2}I$, so the line element $ds^2 = \frac{dt}{g_t^2}\|df_t\|^2$; after time normalization, $V_t = ds^2/dt = \|df_t\|^2 / g_t^2$, i.e., DVS. In discrete solvers, $V_k = \|f(x_k, t_k) - f(x_{k-1}, t_{k-1})\|^2 / g_{t_k}^2$.
    - **Design Motivation**: DVS explicitly incorporates both "drift variation" and "noise scale," capturing the intuition that "a small drift change in low-noise regimes can cause failure"—when $g_t$ is small, $V_t$ is large, naturally prompting the sampler to reduce step size. The Fisher-Rao metric is uniquely determined by Chentsov's theorem as the only sufficient-statistic invariant metric on the statistical manifold, making it more principled than heuristics.

2. **Equal Arc Length Adaptive Step Size Rule**:

    - **Function**: Ensures each sampling step advances approximately equal length on the statistical manifold, evenly distributing "quality risk" and "step budget."
    - **Mechanism**: Require $\Delta s_k^2 = V_k\cdot\Delta t_k \approx \text{const}$, so $\Delta t_k = \text{clip}(\Delta t_{\text{base}}(\kappa_{\text{ref}}/\bar{V})^\beta, \Delta t_{\min}, \Delta t_{\max})$, with fixed $\beta=0.5$ for square-root damping to prevent oscillation; $\kappa_{\text{ref}}$ is the target curvature reference. High $V$ (stiff region) → step size shrinks; low $V$ (smooth region) → step size expands.
    - **Design Motivation**: The problem with fixed $\Delta t$ is most evident in Fig 3—$\Delta s^2$ is tiny in early/mid stages and explodes exponentially at the end; the equal $\Delta s^2$ strategy flattens the "information progress," avoiding wasted computation early and catastrophic failures at the end.

3. **Node-Edge Dual-Channel DVS + Bottleneck Step Size + EMA Smoothing**:

    - **Function**: Addresses asynchronous denoising of nodes and edges in graph diffusion and prevents SDE stochasticity from making DVS noisy.
    - **Mechanism**: Compute $V_{\mathbf{X},k}, V_{\mathbf{A},k}$ separately, each filtered by EMA $\bar{V}\leftarrow(1-\alpha)\bar{V} + \alpha V_k$ ($\alpha=0.2$) to suppress high-frequency jitter; obtain candidate step sizes $\Delta t_{\mathbf{X},k}, \Delta t_{\mathbf{A},k}$, and finally $\Delta t_k = \min(\cdot, \cdot)$, ensuring the stiffer branch is not skipped; after each step, $\bar{V}\leftarrow\gamma(\bar{V}_{\mathbf{X}} + \bar{V}_{\mathbf{A}})$ injects cross-modal feedback.
    - **Design Motivation**: Node (continuous features) and edge (discrete adjacency) denoising speeds differ greatly; a single-channel metric would neglect one. EMA suppresses high-frequency noise while tracking structural changes; bottleneck (min) ensures the weakest link is stable.

### Loss & Training
Completely training-free, with no learnable parameters. Four hyperparameters are introduced at sampling: $\kappa_{\text{ref}}$ (data-adaptive curvature reference), $\gamma$ (feedback gain, swept from 0.10–0.35 with 0.20 optimal on QM9), $\beta=0.5$ (fixed damping exponent), and $\alpha=0.2$ (fixed EMA coefficient). On some datasets, DVS is only enabled for part of the sampling trajectory (see appendix B.1), with fixed step size elsewhere for numerical stability.

## Key Experimental Results

### Main Results

| Dataset | Model | Method | Key Metric |
|---------|-------|--------|-----------|
| QM9 | GruM + Euler | Fixed-Step | FCD 0.107 |
| QM9 | GruM + Euler | Quadratic | FCD 0.107 |
| QM9 | GruM + Euler | DVS (Ours) | **FCD 0.095** |
| QM9 | GruM + Heun | DVS | **FCD 0.099 / SSIM overall best** |
| ZINC250k | GruM + Euler | DVS | FCD 2.092 vs 2.207 baseline |
| QM9 | GDSS + Euler | DVS | FCD 2.482 vs 2.551 |
| Planar | GruM + Heun | DVS | Spec MMD 0.0049 vs 0.0059 |
| SBM | GruM + Euler | DVS | Spec MMD 0.0030 vs 0.0051 |

### Ablation Study

| $\gamma$ | NFE (Steps) | Valid ↑ | FCD ↓ | Scaf. ↑ |
|----------|-------------|---------|-------|---------|
| Euler Baseline | 1000 | 0.9943 | 0.1065 | 0.9341 |
| 0.10 | 706 | 0.9937 | 0.1050 | 0.9370 |
| 0.20 | 745 | 0.9947 | **0.0976** | 0.9415 |
| 0.25 | 770 | 0.9956 | 0.1028 | **0.9455** |
| 0.35 | 836 | 0.9951 | 0.1043 | 0.9428 |

### Key Findings
- **25% Fewer Steps, Higher Quality**: On QM9, DVS achieves FCD 0.0976 in 745 steps, while Euler with 1000 steps only reaches 0.1065, indicating that "allocation" is more important than "quantity."
- **DVS-Euler often matches or surpasses Fixed-Step Heun**: Suggests that for graph data, "equal arc length on the manifold" is more effective than "higher-order solver local accuracy."
- **Equal Arc Length Visualization (Fig 3)**: For Euler, $\Delta s^2$ is near zero in early/mid stages and explodes at the end; DVS flattens the curve to near-constant, only rising slightly when hitting $\Delta t_{\min}$—an intuitive match to the "rush through stiff" issue described in the InfoLaw paper.
- **$\gamma$ Controls Conservativeness**: Larger $\gamma$ means stronger feedback, smaller step size, and higher NFE; FCD is optimal at 0.20, Scaf at 0.25, indicating different metrics prefer different granularity.
- Spectral/orbit MMD on general graphs (SBM, Planar, Ego-small) also outperforms quadratic, showing the information geometric metric effectively captures both "global topology + local motifs."

## Highlights & Insights
- **Elevating "when to take larger/smaller steps" to information geometry**: Compared to EDM-Karras and other heuristics, DVS is derived from the Fisher-Rao metric, making it principled; this "coordinate system shift for sampling scheduling" can be directly transferred to score-based image and video diffusion.
- **Training-free and plug-and-play**: Simply add 4 lines in the sampling loop to compute DVS, update EMA, and decide $\Delta t$; zero-intrusion for existing GruM/GDSS models, making it engineering-friendly.
- **Dual-channel + bottleneck**: Treats nodes and edges as two asynchronous components, with the final step size determined by the stiffer bottleneck—this idea can be extended to any multi-component coupled diffusion (text+image, 3D geometry+semantics).
- **Fig 3 Equal Arc Length Visualization**: The most pedagogically valuable figure in the paper—demonstrates to the community the awkwardness of "fixed step size = wasted early steps, catastrophic late steps."

## Limitations & Future Work
- Only validated on two types of graph diffusion models (GruM's OU bridge and GDSS's score SDE); not tested on discrete diffusion (e.g., DiGress).
- On some datasets, DVS is only enabled for part of the trajectory, with interval selection still empirically set and no unified rule provided.
- $\gamma, \kappa_{\text{ref}}$ are dataset-dependent hyperparameters and require resweeping for new datasets; lacks an automatic calibration method.
- DVS estimates gradients via drift differences between adjacent steps; for ultra-low NFE (e.g., 10 steps), noise estimation may be distorted.
- No comparison of inference cost—although NFE is reduced, each step adds an EMA update and difference computation; true end-to-end wall-clock speedup is unreported.

## Related Work & Insights
- **vs AYS (Sabour 2024)**: AYS reparameterizes time based on local truncation error in state space; DVS estimates Fisher-Rao arc length in distribution space, which is more intrinsic geometrically and independent of specific SDE forms.
- **vs Quadratic schedule (Song 2021a)**: Quadratic is a data-agnostic fixed power law; DVS is jointly adaptive to data and model, and the paper shows DVS outperforms quadratic in most settings.
- **vs Karras EDM**: EDM tunes $\sigma(t)$ via empirical design; DVS directly uses the Fisher metric in the reverse SDE, which is theoretically clearer but more limited in scope (requires analytically tractable Fisher for Gaussian transition kernels).
- **vs Song & Lai (Fisher Information for diffusion)**: They use Cramér-Rao reweighted scores; this work reallocates step sizes via the Fisher metric. The two are complementary and could be combined in future work.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing Fisher-Rao arc length into diffusion sampling scheduling is a rare perspective, with self-consistent derivations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 2 models (GruM/GDSS) × 2 tasks (molecule/general graph) × multiple solvers + $\gamma$ sweep; only lacks wall-clock time and ultra-low NFE evaluation.
- Writing Quality: ⭐⭐⭐⭐ Fig 1 concept diagram + Fig 3 equal arc length visualization clearly convey the core ideas, with robust formula segmentation.
- Value: ⭐⭐⭐⭐ Training-free, plug-and-play, and interpretable, highly attractive for deployment; extension to image/video diffusion would have even greater impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](../../CVPR2026/image_generation/adaptive_spectral_feature_forecasting_for_diffusion_sampling_acceleration.md)
- [\[ECCV 2024\] EchoScene: Indoor Scene Generation via Information Echo over Scene Graph Diffusion](../../ECCV2024/image_generation/echoscene_indoor_scene_generation_via_information_echo_over_scene_graph_diffusio.md)
- [\[NeurIPS 2025\] Graph Diffusion that can Insert and Delete](../../NeurIPS2025/image_generation/graph_diffusion_that_can_insert_and_delete.md)
- [\[ICML 2026\] Watch Your Step: Information Injection in Diffusion Models via Shadow Timestep Embedding](watch_your_step_information_injection_in_diffusion_models_via_shadow_timestep_em.md)
- [\[ICLR 2026\] The Spacetime of Diffusion Models: An Information Geometry Perspective](../../ICLR2026/image_generation/the_spacetime_of_diffusion_models_an_information_geometry_perspective.md)

</div>

<!-- RELATED:END -->
