---
title: >-
  [Paper Note] STEP: Warm-Started Visuomotor Policies with Spatiotemporal Consistency Prediction
description: >-
  [ICML 2026][Robotics][diffusion policy] STEP attaches a lightweight "previous action history + current observation → next action" Transformer predictor to a diffusion policy…
tags:
  - "ICML 2026"
  - "Robotics"
  - "diffusion policy"
  - "warm-start"
  - "spatiotemporal consistency"
  - "local contraction"
  - "velocity-aware perturbation"
date: 2026-05-08
content_hash: 05b064378492fb83
---

# STEP: Warm-Started Visuomotor Policies with Spatiotemporal Consistency Prediction

**Conference**: ICML 2026  
**arXiv**: [2602.08245](https://arxiv.org/abs/2602.08245)  
**Code**: <https://github.com/Kimho666/STEP>  
**Area**: Robotics / Embodied Intelligence / Diffusion Policy Acceleration  
**Keywords**: diffusion policy, warm-start, spatiotemporal consistency, local contraction, velocity-aware perturbation

## TL;DR
STEP attaches a lightweight "previous action history + current observation → next action" Transformer predictor to a diffusion policy, using its output as a denoising warm-start. This compresses 100 denoising steps to just 2, and introduces an execution deadlock defense: if the action change is too small, a bit of noise is injected. Across 9 simulation and 2 real-world tasks, STEP outperforms BRIDGER / DDIM by 21.6% / 27.5% average success rate.

## Background & Motivation

**Background**: Diffusion Policy (DP) is the de facto standard for visuomotor control: it models action sequences as a generative distribution, iteratively denoising from Gaussian noise over 100 steps to produce action blocks. It captures multimodality and long-range dependencies, achieving the highest success rates, but also incurs the highest latency.

**Limitations of Prior Work**: Existing DP acceleration methods fall into three categories: (1) Numerical solvers (DDIM, DPM-Solver series) can reduce 100 steps to 4–2, but performance collapses at 2 steps (Push-T only 0.29); (2) Distillation/direct prediction (CP, OneDP, BRIDGER) use small predictors to replace the denoising process, but lack expressiveness and degrade on complex tasks; (3) Action reuse (RTI-DP, RNR-DP, Falcon) uses the previous action as a warm-start, which fails when state changes rapidly. Each only partially addresses either "speed" or "accuracy".

**Key Challenge**: The crux of acceleration is "providing a good denoising starting point", which must satisfy **both**: **spatial consistency** (close to the target action manifold under current state) and **temporal consistency** (smooth transition from previous action). Existing methods satisfy at most one (BRIDGER: spatial only, Falcon: temporal only), which is insufficient.

**Goal**: (a) Design a warm-start that preserves DP expressiveness and achieves both spatial and temporal consistency; (b) Ensure stability even with only 2 denoising steps; (c) Prevent the robot from getting stuck at static friction zero points due to overly "smooth" warm-starts during real-world deployment.

**Key Insight**: Rather than replacing or distilling the original DP, **attach a lightweight predictor** that maps $(\mathbf o_t, \mathbf A_{t-H})$ to $\hat{\mathbf A}_t$ as the starting point, then inject a small amount of noise from an intermediate denoising step $K' < K$—thus combining the speed of warm-start with the multimodal generation of DP.

**Core Idea**: Use a conditional Transformer on "previous action block + current observation" to achieve initialization with both temporal (conditioned on previous actions) and spatial (conditioned on observation) consistency; add a velocity-aware perturbation mechanism to counteract real-world deadlocks; finally, use contraction-mapping theory to prove improved convergence of such starting points.

## Method

### Overall Architecture
Inference pipeline (Algorithm 1): (1) Observe $\mathbf o_t$; if the action cache is filled with $H$ steps, use the predictor to compute $\hat{\mathbf A}_t=f_\theta(\mathbf o_t,\mathbf A_{cache})$; (2) Construct warm-start $\tilde{\mathbf A}_{K'}=\sigma\hat{\mathbf A}_t+\sigma_t\boldsymbol\epsilon_t$, where $K'\ll K$ is the intermediate denoising step; (3) Run reverse diffusion from $K'$ to $0$ to obtain the final action $\mathbf A_t$ and execute; (4) Update cache with $\mathbf A_t$ for the next cycle. During training, the predictor and DP are trained separately: DP uses standard noise prediction (Eq. 5), the predictor uses MSE; at inference, they are cascaded.

### Key Designs

1. **Spatiotemporal Consistency Predictor**:

    - **Function**: A single forward pass yields an action starting point satisfying both temporal consistency ($\|\tilde a_t-a_{t-1}\|\le\epsilon_t$) and spatial consistency ($\mathrm{dist}(\tilde a_t,\mathcal M(s_t))\le\epsilon_s$).
    - **Mechanism**: $f_\theta:\mathcal O\times\mathcal A^H\to\mathcal A^H$, implemented as a **2-layer cross-attention Transformer** (actions as query, observations as key/value, 128-dim embedding), mapping $(\mathbf o_t,\mathbf A_{t-H})$ to $\hat{\mathbf A}_t$; training objective is $\mathcal L_{pred}=\mathbb E\|\hat{\mathbf A}_t-\mathbf A_t\|^2$, learning the conditional expectation $\mathbb E[\mathbf A_t\mid\mathbf o_t,\mathbf A_{t-H}]$.
    - **Design Motivation**: Temporal consistency is ensured by conditioning on $\mathbf A_{t-H}$, spatial consistency by conditioning on $\mathbf o_t$; no extra regularization needed, making the design extremely simple. Cross-attention is more suitable than self-attention for mixing two heterogeneous sequences. Experiments (Fig. 3) show 2 blocks suffice; more only increase latency without benefit, so 2 blocks are used—an example of engineering "sweet spot".

2. **Velocity-Aware Perturbation Injection**:

    - **Function**: When the predictor outputs minimal action change (robot about to enter "static friction deadlock"), automatically injects noise to help the actuator overcome dead zones; otherwise, keeps the original signal.
    - **Mechanism**: Compute action difference $\Delta\mathbf A_t=\mathbf A_{cache}-\mathbf A_{t-2H}$, use indicator $\mathbb I_t=\mathbb I(\|\Delta\mathbf A_t\|<\epsilon_a)$ to detect stalling; warm-start scale $\sigma$ and noise amplitude $\sigma_t$ switch between two modes per Eq. 14—normally $(\sigma,\sigma_t)=(1,0)$ fully trusts the predictor, in stalling $(\sigma,\sigma_t)=(\sigma_{scale},\sigma_{stall})$ shrinks amplitude and injects small Gaussian noise ($\epsilon_a=0.01$, simulation $\sigma_{stall}=0.1$, larger for real robots).
    - **Design Motivation**: Real robots face "control dead zones + static friction" issues; predictor-learned "correct actions" work perfectly in simulation but cause motors to "refuse to move" in reality. Vanilla DDPM's randomness helps cross dead zones—this observation led to "on-demand randomness injection" as a switch, not always-on noise.

3. **Convergence Proof via Local Contraction Mapping**:

    - **Function**: Provides theoretical explanation for why "good warm-start + few reverse steps" can stably converge to correct actions.
    - **Mechanism**: Unifies DDPM, DDIM, DPM-Solver as $\mathbf A_{k-1}=\mu_k(\mathbf A_k,\mathbf o_t)+\boldsymbol\xi_k$ (Eq. 15); assuming the denoising network $\epsilon_\theta$ is Lipschitz with constant $L$ in the data manifold neighborhood $\mathcal U$, the reverse mean $\mu_k$ is also Lipschitz with coefficient $c_k<1$ (Eq. 16); recursively, $\|\tilde{\mathbf A}_0-\mathbf A_0\|\le\prod_{k=1}^{K'}c_k\|\tilde{\mathbf A}_{K'}-\mathbf A_{K'}\|$ (Eq. 18), so as long as the predictor brings the starting point into $\mathcal U$, the error decays exponentially.
    - **Design Motivation**: Uses a unified contraction framework to explain "why starting denoising from an intermediate step is better than from pure noise", and proves it holds for DDIM / DPM-Solver, independent of the solver—key for elevating an engineering trick to a generalizable method.

### Loss & Training

- Predictor: $\mathcal L_{pred}=\mathbb E\|\hat{\mathbf A}_t-\mathbf A_t\|^2$, 100k steps.
- DP: Uses default codebase configuration, unchanged.
- Inference hyperparameters: $K'$ = starting denoising step (i.e., STEP = 2 / 4); $\sigma=1, \sigma_t=0.1$ default for simulation; real robot uses larger $\sigma_{stall}$ to overcome dead zones.

## Key Experimental Results

### Main Results

**State-based RoboMimic / Push-T (Table 2 excerpt)**: Score (higher is better) / Time (ms, lower is better).

| Method | Step | Push-T | Square | ToolHang |
|---|---|---|---|---|
| Vanilla DDPM | 100 | 0.94 | 0.94 | 0.68 |
| DDIM | 2 | 0.29 | 0.84 | 0.06 |
| DPM-Solver++ | 2 | 0.20 | 1.00 | 0 |
| BRIDGER | 2 | 0.37 | 0.84 | 0.08 |
| Falcon | 2 | 0.21 | 1.00 | 0 |
| **STEP (Ours)** | **2** | **0.49** | **0.96** | **0.64** |

**Image-based RoboMimic (Table 3 excerpt)**: For vision input, long-horizon tasks like ToolHang show especially clear differences.

| Method | Step | Square | ToolHang |
|---|---|---|---|
| DDIM | 2 | 0.74 | 0.5 |
| BRIDGER | 2 | 0.92 | 0.72 |
| **STEP (Ours)** | **2** | – (>BRIDGER, see paper) | – |

Core conclusion: STEP at 2 steps achieves +21.6% average over BRIDGER and +48.8% over Falcon (temporal-only) on RoboMimic; on real robots, +27.5% average success rate over DDIM; average episode execution time on real robots reduced by 59% due to velocity-aware perturbation.

### Ablation Study

| Configuration | Key Metric | Notes |
|------|---------|------|
| Full STEP (2 step) | Push-T 0.49 / Lift 1.0 / Square 0.96 | Spatiotemporal consistency + perturbation + intermediate step |
| No predictor (= DDIM) | Push-T 0.29 / Lift 0.80 / Square 0.84 | Starting point degrades to pure noise, performance collapses |
| Spatial only (BRIDGER) | Push-T 0.37 / Lift 1.0 / Square 0.84 | No temporal continuity, long-horizon tasks degrade |
| Temporal only (Falcon) | Push-T 0.21 / Square 1.00 / ToolHang 0 | No spatial consistency, ToolHang drops to 0 |
| Cross-attn block 1/2/4 | 2 is sweet spot | Fig 3, 4 blocks increase latency without gain |

### Key Findings
- **2-step inference is STEP's core selling point**: Other methods collapse at 2 steps (Falcon ToolHang=0), but STEP maintains success rates close to 100-step DDPM, pushing the Pareto frontier (latency vs. success) to the lower right.
- **Both spatial and temporal consistency are essential**: Table 1 shows that any method with only TC or SC drops at least one task to 0 at 2 steps; only STEP remains robust across all tasks.
- **Sim-to-real gap**: $\sigma_{stall}$ suffices at 0.1 in simulation, but needs to be larger on real robots, highlighting "friction/dead zone" as a neglected sim-to-real bottleneck.
- **Predictor is very small**: Only 128-dim, 2 cross-attention blocks, 100k training steps; extremely lightweight and easy to embed.

## Highlights & Insights
- **Clear conceptual contribution**: Explicitly formalizes "warm-start must be spatiotemporally consistent" (Eq. 7-8 + Table 1), providing a simple analytical dimension for DP acceleration.
- **Decoupled predictor + DP training**: Retains DP's multimodal generation (no distillation or replacement), yet enables lightweight acceleration—a practical design pattern applicable to any DP backbone.
- **Velocity-aware perturbation**: The "on-demand randomness injection" idea can transfer to any domain needing dynamic switching between deterministic prediction and exploration (e.g., imitation + RL hybrid training, model predictive control).
- **Contraction proof**: Simple yet strong—provides a unified theoretical explanation for all "intermediate-step warm-start" methods, not just anecdotal evidence.

## Limitations & Future Work
- Temporal consistency relies on conditioning on previous $\mathbf A_{t-H}$, which may introduce bias during abrupt state transitions (e.g., sudden obstacles); the perturbation mechanism compensates, but its trigger only considers action change, not observation change, making it coarse.
- The predictor is a single forward pass, without explicit multimodal modeling; if the current state corresponds to multiple equivalent action modes, the predictor may "average out" to meaningless actions (the paper acknowledges this issue for BRIDGER-type methods).
- Only validated in imitation learning; compatibility of warm-start with policy drift in true closed-loop RL remains untested.
- Real robot $\sigma_{stall}$ requires manual tuning, lacking adaptive strategies; a learned critic could be added to determine "whether perturbation is needed".

## Related Work & Insights
- **vs DDIM / DPM-Solver++**: These only modify the solver, do not introduce warm-start, and collapse at 2 steps; STEP is orthogonal and can be applied to any solver.
- **vs BRIDGER (spatial-only)**: BRIDGER uses a predictor for the starting point but only considers current state, not time; STEP adds just previous $\mathbf A_{t-H}$ and achieves 21.6% average improvement.
- **vs Falcon / RTI-DP (temporal-only)**: These assume smooth dynamics, but fail on tasks with rapid state changes (ToolHang / Push-T); STEP leverages observation conditioning to handle state transitions.
- **vs CP / OneDP (distillation)**: Distillation destroys DP's multimodal generation; STEP preserves DP, allowing "more denoising steps for complex tasks, just 2 for simple ones".

## Rating
- Novelty: ⭐⭐⭐⭐ Spatiotemporal consistency as a two-dimensional criterion + intermediate-step warm-start + velocity-aware perturbation, all simple yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 simulation + 2 real-world tasks × 8 baselines × state/image inputs, comprehensive ablation tables.
- Writing Quality: ⭐⭐⭐⭐ Conceptual diagrams (Fig 1) + consistency table (Table 1) help readers quickly grasp the framework; contraction proof is concise and elegant.
- Value: ⭐⭐⭐⭐⭐ Directly usable engineering solution, code open-sourced, a plug-and-play acceleration tool for all teams deploying diffusion policy in robotics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Latent Reasoning VLA: Latent Thinking and Prediction for Vision-Language-Action Models](latent_reasoning_vla_latent_thinking_and_prediction_for_vision-language-action_m.md)
- [\[CVPR 2026\] Action–Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation](../../CVPR2026/robotics/actiongeometry_prediction_with_3d_geometric_prior.md)
- [\[ICLR 2026\] When would Vision-Proprioception Policies Fail in Robotic Manipulation?](../../ICLR2026/robotics/when_would_vision-proprioception_policies_fail_in_robotic_manipulation.md)
- [\[NeurIPS 2025\] Towards Reliable Code-as-Policies: A Neuro-Symbolic Framework for Embodied Task Planning](../../NeurIPS2025/robotics/towards_reliable_code-as-policies_a_neuro-symbolic_framework_for_embodied_task_p.md)
- [\[ICCV 2025\] PacGDC: Label-Efficient Generalizable Depth Completion with Projection Ambiguity and Consistency](../../ICCV2025/robotics/pacgdc_label-efficient_generalizable_depth_completion_with_projection_ambiguity_.md)

</div>

<!-- RELATED:END -->
