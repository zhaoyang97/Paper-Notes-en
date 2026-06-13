---
title: >-
  [Paper Note] Information-Geometric Adaptive Sampling for Graph Diffusion
description: >-
  [ICML 2026][Image Generation][Graph Diffusion] This paper views the sampling trajectory of the reverse SDE in graph diffusion as a parametric curve on a Riemannian statistical manifold. Using the Fisher-Rao metric…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Graph Diffusion"
  - "Fisher-Rao Metric"
  - "Adaptive Step Size"
  - "Information Geometry"
  - "Molecule Generation"
date: 2026-05-08
content_hash: 5c0a4e583c49af4e
---

# Information-Geometric Adaptive Sampling for Graph Diffusion

**Conference**: ICML 2026  
**arXiv**: [2605.00250](https://arxiv.org/abs/2605.00250)  
**Code**: None  
**Area**: Diffusion Models / Graph Generation / Adaptive Sampling  
**Keywords**: Graph Diffusion, Fisher-Rao Metric, Adaptive Step Size, Information Geometry, Molecule Generation

## TL;DR
This paper views the sampling trajectory of the reverse SDE in graph diffusion as a parametric curve on a Riemannian statistical manifold. Using the Fisher-Rao metric, it derives a training-free Drift Variation Score (DVS) to measure the local "information curvature" of the trajectory. By adaptively scaling the step size such that each step covers an equal length on the information manifold, it achieves higher FCD/MMD fidelity with fewer steps in molecule (QM9/ZINC250k) and graph (Planar/SBM/Ego) generation.

## Background & Motivation

**Background**: Graph diffusion models (e.g., GDSS, GruM) utilize reverse SDEs to jointly denoise node features $\mathbf{X}$ and adjacency matrices $\mathbf{A}$. Prevailing samplers rely on fixed-step predictor-corrector frameworks like Euler-Maruyama or Heun.

**Limitations of Prior Work**: (i) Fixed step sizes implicitly assume "equal time intervals = equal distribution changes," yet reverse SDE dynamics are highly non-uniform: drifts are smooth in high-noise regions but change sharply (stiff) in low-noise regions; (ii) Heuristic quadratic schedules are static prescriptions and cannot adapt to specific data or models; (iii) Adaptive step sizes based on local truncation errors estimate errors in the state space, ignoring the intrinsic geometry of the probability path; (iv) The unique "node vs. edge asynchronous denoising" in graph data causes inconsistent "stiff" moments for nodes and edges, making a single step size difficult to balance.

**Key Challenge**: To uniformly characterize the "evolution rate of distributions," one must move away from using time $t$ as the arc length—time is an external parameter, while distributional distance is the intrinsic geometry.

**Goal**: (i) Provide an adaptive step size metric under "information geometry" semantics for graph diffusion reverse SDEs; (ii) Enable separate stiff detection signals for nodes and edges for joint decision-making; (iii) Ensure a plug-and-play mechanism without retraining.

**Key Insight**: Treat the Gaussian transition kernel $p(x_{t+dt}|x_t; f_t)$ induced by the reverse SDE at each moment as a point on a statistical manifold coordinated by the drift $f_t$, making the entire sampling process a curve. Use the Fisher-Rao metric (the unique invariant metric determined by Chentsov's theorem) to measure the curve's arc length—this length represents the "intrinsic distance of distributional change."

**Core Idea**: Each step $\Delta s^2 \approx$ constant $\implies \Delta t \propto 1/V_t$, where $V_t = \|d f_t\|^2 / g_t^2$ is the DVS.

## Method

### Overall Architecture
For the reverse SDE $dx_t = f_t dt + g_t d\bar{w}_t$, at each discrete step $t_k$, the drift difference is evaluated for nodes and edges to obtain $V_{\mathbf{X},k}$ and $V_{\mathbf{A},k}$, which are smoothed via EMA into $\bar{V}_{\mathbf{X}}, \bar{V}_{\mathbf{A}}$. A power law $\Delta t \propto (\kappa_{\text{ref}}/\bar{V})^\beta$ is used to calculate candidate step sizes for nodes and edges (clipped to $[\Delta t_{\min}, \Delta t_{\max}]$), and the minimum of the two is taken as the current step's progress. $(\mathbf{X}, \mathbf{A})$ are updated using a single Euler or Heun solver step, and $\bar{V}$ is fed back to the next step multiplied by $\gamma$. The entire process does not modify the pretrained score network and is only inserted during sampling.

### Key Designs

1.  **Fisher-Rao Line Element + Drift Variation Score (DVS)**:
    - **Function**: Quantifies "how fast the distribution changes on the statistical manifold" using a dimensionless, online-evaluable scalar.
    - **Mechanism**: The transition kernel is written as $p(x_{t+dt}|x_t; f_t) = \mathcal{N}(x_t + f_t dt, g_t^2 dt I)$. Taking the log-likelihood gradient with respect to $f_t$ yields the Fisher Information Matrix $\mathcal{I}(f_t) = \frac{dt}{g_t^2}I$. The line element is thus $ds^2 = \frac{dt}{g_t^2}\|df_t\|^2$. After time-normalization, $V_t = ds^2/dt = \|df_t\|^2 / g_t^2$, which is the DVS. In a discrete solver, $V_k = \|f(x_k, t_k) - f(x_{k-1}, t_{k-1})\|^2 / g_{t_k}^2$.
    - **Design Motivation**: DVS explicitly incorporates both "drift variation" and "noise scale," capturing the physical intuition that "in low-noise regions, a small drift shift causes failure." As $g_t$ decreases, $V_t$ increases, naturally prompting the sampler to reduce the step size. Fisher-Rao is the unique "sufficient-statistic invariant metric" on statistical manifolds, making it more principled than heuristics.

2.  **Constant Arc-Length Adaptive Step Rule**:
    - **Function**: Ensures each sampling step covers an approximately equal distance on the statistical manifold, spreading "quality risk" and "step budget" evenly.
    - **Mechanism**: Setting $\Delta s_k^2 = V_k\cdot\Delta t_k \approx \text{const}$ leads to $\Delta t_k = \text{clip}(\Delta t_{\text{base}}(\kappa_{\text{ref}}/\bar{V})^\beta, \Delta t_{\min}, \Delta t_{\max})$, where $\beta=0.5$ is a square-root damping factor used to prevent drastic fluctuations; $\kappa_{\text{ref}}$ is the target curvature reference. High $V$ (stiff regions) $\to$ step contraction; low $V$ (smooth regions) $\to$ step expansion.
    - **Design Motivation**: The issues with fixed $\Delta t$ are most visible in Fig 3—$\Delta s^2$ is tiny in the early/middle stages but explodes exponentially at the end. The equal $\Delta s^2$ strategy flattens "information progress," neither wasting computation early nor failing during late structural bond formation.

3.  **Dual-channel DVS + Bottleneck Step Size + EMA Smoothing**:
    - **Function**: Addresses the asynchronous denoising of nodes and edges in graph diffusion and prevents SDE stochasticity from making DVS noisy.
    - **Mechanism**: $V_{\mathbf{X},k}$ and $V_{\mathbf{A},k}$ are calculated separately, each filtered by EMA $\bar{V}\leftarrow(1-\alpha)\bar{V} + \alpha V_k$ ($\alpha=0.2$) to remove high-frequency jitter. Candidate steps $\Delta t_{\mathbf{X},k}, \Delta t_{\mathbf{A},k}$ are derived, and finally $\Delta t_k = \min(\cdot, \cdot)$ ensures the stiffer component is prioritized. After each step, $\bar{V}\leftarrow\gamma(\bar{V}_{\mathbf{X}} + \bar{V}_{\mathbf{A}})$ injects cross-modal coupling feedback.
    - **Design Motivation**: The denoising speeds of nodes (continuous features) and edges (discrete adjacency) differ vastly; a single-channel metric would overlook one. EMA suppresses high-frequency noise while tracking structural mutations; the bottleneck min ensures stability in the weakest link.

### Loss & Training
Completely training-free with no learnable parameters. Four hyperparameters are introduced during sampling: $\kappa_{\text{ref}}$ (reference curvature, data-adaptive), $\gamma$ (feedback gain, optimal at 0.20 in QM9 experiments), $\beta=0.5$ (damping exponent, fixed), and $\alpha=0.2$ (EMA coefficient, fixed). On some datasets, DVS is only enabled for specific segments of the sampling trajectory (Appendix B.1), with fixed steps maintained elsewhere for numerical stability.

## Key Experimental Results

### Main Results

| Dataset | Model | Method | Key Metrics |
| :--- | :--- | :--- | :--- |
| QM9 | GruM + Euler | Fixed-Step | FCD 0.107 |
| QM9 | GruM + Euler | Quadratic | FCD 0.107 |
| QM9 | GruM + Euler | DVS (Ours) | **FCD 0.095** |
| QM9 | GruM + Heun | DVS | **FCD 0.099 / Best SSIM etc.** |
| ZINC250k | GruM + Euler | DVS | FCD 2.092 vs 2.207 baseline |
| QM9 | GDSS + Euler | DVS | FCD 2.482 vs 2.551 |
| Planar | GruM + Heun | DVS | Spec MMD 0.0049 vs 0.0059 |
| SBM | GruM + Euler | DVS | Spec MMD 0.0030 vs 0.0051 |

### Ablation Study

| $\gamma$ | NFE (Steps) | Valid ↑ | FCD ↓ | Scaf. ↑ |
| :--- | :--- | :--- | :--- | :--- |
| Euler Baseline | 1000 | 0.9943 | 0.1065 | 0.9341 |
| 0.10 | 706 | 0.9937 | 0.1050 | 0.9370 |
| 0.20 | 745 | 0.9947 | **0.0976** | 0.9415 |
| 0.25 | 770 | 0.9956 | 0.1028 | **0.9455** |
| 0.35 | 836 | 0.9951 | 0.1043 | 0.9428 |

### Key Findings
- **25% Step Reduction, Increased Quality**: On QM9, DVS achieves FCD 0.0976 using 745 steps, whereas Euler requires 1000 steps to reach 0.1065, demonstrating that "allocation" is more important than "volume."
- **DVS-Euler often matches or exceeds Fixed-Step Heun**: This suggests that for graph data, "moving at equal arc lengths on the manifold" is more efficient than "higher-order solver local precision."
- **Arc Length Visualization (Fig 3)**: Euler's $\Delta s^2$ is near 0 in early/middle stages and explodes exponentially at the end; DVS flattens the entire curve to near-constant, only rising slightly when hitting $\Delta t_{\min}$—this is a direct response to the "rush through stiff" problem described in InfoLaw.
- **$\gamma$ Determines Conservativeness**: Larger $\gamma$ leads to stronger feedback, finer steps, and higher NFE. FCD peaks at 0.20 and Scaf at 0.25, indicating that different metrics prefer different "granularity."
- For general graphs (SBM, Planar, Ego-small), spectral/orbit MMD also outperforms quadratic, proving information-geometric metrics are effective for capturing both "global topology + local motifs."

## Highlights & Insights
- **Elevating "Step Control" to Information Geometry**: Unlike EDM-Karras, which uses empirical noise schedule tuning, DVS is derived from the Fisher-Rao metric, providing a strong principled basis. This "coordinate systems for sampling schedules" perspective can be directly transferred to score-based image or video diffusion.
- **Training-free and Plug-and-Play**: Adding just 4 lines to calculate DVS, update EMA, and decide $\Delta t$ within the sampling loop makes it zero-intrusion for existing models like GruM/GDSS, making it a deployment-friendly engineering enhancement.
- **Dual-channel + Bottleneck**: Treating nodes and edges as two asynchronous components where the step size is determined by the stiffest "bottleneck" is an idea scalable to any multi-component coupled diffusion (Text+Image, 3D Geometry+Semantics).
- **Fig 3 Equal Arc Length Visualization**: This is the most educational figure in the paper—it demonstrates the embarrassment of "fixed steps = early waste, late explosion" to the community.

## Limitations & Future Work
- Only validated on two types of graph diffusion models (GruM's OU bridge and GDSS's score SDE); not tested on discrete diffusion (e.g., DiGress).
- In some datasets, DVS is only enabled for partial trajectory intervals. The selection of these intervals remains empirical without a unified rule.
- $\gamma, \kappa_{\text{ref}}$ are dataset-dependent hyperparameters; moving to new datasets requires re-sweeping. An automatic calibration method is missing.
- DVS estimates the gradient using the drift difference between two adjacent steps, which may distort noise estimation under ultra-low NFE (e.g., 10 steps).
- The actual inference cost is not compared—while NFE decreases, each step adds an EMA update and a difference operation; whether there is true end-to-end wall-clock speedup is not reported.

## Related Work & Insights
- **vs AYS (Sabour 2024)**: AYS estimates local truncation error in state space for time reparameterization; DVS estimates Fisher-Rao arc length in distribution space, which is more geometrically intrinsic and independent of the specific SDE form.
- **vs Quadratic schedule (Song 2021a)**: Quadratic is a data-independent fixed power law; DVS is data-model joint adaptive. The paper proves DVS outperforms quadratic in most settings.
- **vs Karras EDM**: EDM tunes $\sigma(t)$ empirically; DVS uses the Fisher metric directly in the reverse SDE. The theory is clearer but narrower (relying on Gaussian transition kernel assumptions to calculate Fisher analytically).
- **vs Song & Lai (Fisher Information for diffusion)**: They use Cramér-Rao for reweighting scores; this paper uses the Fisher metric to reallocate step sizes. The two are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing Fisher-Rao arc length to diffusion sampling scheduling is a rare and self-consistent perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage of 2 models (GruM/GDSS) × 2 tasks (Molecule/General Graph) × multiple solvers + $\gamma$ sweep is thorough. Misses wall-clock time and ultra-low NFE evaluation.
- Writing Quality: ⭐⭐⭐⭐ The conceptual map in Fig 1 and the visualization in Fig 3 clearly explain the core idea.
- Value: ⭐⭐⭐⭐ Training-free + plug-and-play + interpretable makes it highly attractive for deployment; impact would be greater if extended to image/video diffusion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](../../CVPR2026/image_generation/adaptive_spectral_feature_forecasting_for_diffusion_sampling_acceleration.md)
- [\[ICML 2026\] Escaping Mode Collapse in LLM Generation via Geometric Regulation](escaping_mode_collapse_in_llm_generation_via_geometric_regulation.md)
- [\[ICLR 2026\] The Spacetime of Diffusion Models: An Information Geometry Perspective](../../ICLR2026/image_generation/the_spacetime_of_diffusion_models_an_information_geometry_perspective.md)
- [\[ICML 2026\] Watch Your Step: Information Injection in Diffusion Models via Shadow Timestep Embedding](watch_your_step_information_injection_in_diffusion_models_via_shadow_timestep_em.md)
- [\[ICML 2026\] Restoring Initial Noise Sensitivity in Text-to-Image Distillation via Geometric Alignment](restoring_initial_noise_sensitivity_in_text-to-image_distillation_via_geometric_.md)

</div>

<!-- RELATED:END -->
