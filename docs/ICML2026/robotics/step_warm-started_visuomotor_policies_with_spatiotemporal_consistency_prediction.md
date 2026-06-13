---
title: >-
  [Paper Note] STEP: Warm-Started Visuomotor Policies with Spatiotemporal Consistency Prediction
description: >-
  [ICML 2026][Robotics][diffusion policy] STEP attaches a lightweight Transformer predictor ("previous action history + current observation $\to$ next action") to the diffusion policy. Its output serves as the denoising st…
tags:
  - "ICML 2026"
  - "Robotics"
  - "diffusion policy"
  - "warm-start"
  - "spatiotemporal consistency"
  - "local contraction"
  - "velocity-aware perturbation"
date: 2026-05-08
content_hash: 1217727125d31d52
---

# STEP: Warm-Started Visuomotor Policies with Spatiotemporal Consistency Prediction

**Conference**: ICML 2026  
**arXiv**: [2602.08245](https://arxiv.org/abs/2602.08245)  
**Code**: <https://github.com/Kimho666/STEP>  
**Area**: Robotics / Embodied AI / Acceleration for Diffusion Policy  
**Keywords**: diffusion policy, warm-start, spatiotemporal consistency, local contraction, velocity-aware perturbation

## TL;DR
STEP attaches a lightweight Transformer predictor ("previous action history + current observation $\to$ next action") to the diffusion policy. Its output serves as the denoising starting point (warm-start), compressing 100 denoising steps to 2. It also includes a velocity-aware perturbation mechanism to prevent execution deadlocks when action changes are too small. On 9 simulation tasks and 2 real-world tasks, it outperforms BRIDGER/DDIM by an average of 21.6% / 27.5% in success rate.

## Background & Motivation

**Background**: Diffusion Policy (DP) is the current de facto standard for visuomotor control: it models action sequences as a generative distribution, iteratively denoising from Gaussian noise through 100 steps. While it captures multi-model, long-range dependencies with high success rates, it suffers from high latency.

**Limitations of Prior Work**: Existing DP acceleration methods fall into three categories: (1) Numerical solvers (DDIM, DPM-Solver series) can compress 100 steps to 4-2 steps, but performance collapses at 2 steps (Push-T drops to 0.29); (2) Distillation/Direct prediction (CP, OneDP, BRIDGER) replace the denoising process with smaller predictors but lack expressiveness, leading to failures in complex tasks; (3) Action recycling (RTI-DP, RNR-DP, Falcon) uses actions from the previous timestep as a warm-start, which ensures temporal continuity but fails when states change rapidly. All three types only achieve partial solutions for speed or accuracy.

**Key Challenge**: The key to acceleration is providing a "good starting point" for denoising. A good starting point must **simultaneously** satisfy two conditions: **Spatial Consistency** (closeness to the target action manifold given the current state) and **Temporal Consistency** (smooth transition from the previously executed action). Existing methods satisfy at most one (BRIDGER only spatial, Falcon only temporal).

**Goal**: (a) Design a warm-start that preserves the original DP's expressiveness while possessing both spatial and temporal consistency; (b) Ensure stability even with only 2 denoising steps; (c) Prevent robots from getting stuck at zero-velocity (static friction) due to overly "smooth" warm-starts during real-world deployment.

**Key Insight**: Instead of replacing or distilling the original DP, **attach a lightweight predictor** that maps $(\mathbf o_t, \mathbf A_{t-H})$ to $\hat{\mathbf A}_t$ as a starting point. Then, inject a small amount of noise at an intermediate denoising step $K'<K$ and proceed—this leverages the speed of warm-starts while retaining the multi-modal generation of DP.

**Core Idea**: Use a conditional Transformer predictor with "previous action block + current observation" to achieve initialization that is both temporally (via previous actions) and spatially (via observations) consistent. Additionally, implement a velocity-aware perturbation mechanism to combat real-world deadlocks. Finally, use contraction-mapping theory to prove that this starting point yields better convergence.

## Method

### Overall Architecture
Inference pipeline (Algorithm 1): (1) Observe $\mathbf o_t$; if the action cache is filled with $H$ steps, use the predictor to compute $\hat{\mathbf A}_t=f_\theta(\mathbf o_t,\mathbf A_{cache})$; (2) Construct the warm-start $\tilde{\mathbf A}_{K'}=\sigma\hat{\mathbf A}_t+\sigma_t\boldsymbol\epsilon_t$, where $K'\ll K$ denotes starting from an intermediate step; (3) Run the $K'\to 0$ reverse diffusion to obtain the final action $\mathbf A_t$ and execute it; (4) Update the cache with $\mathbf A_t$ for the next loop. During training, the predictor and DP are trained decoupled: the DP follows standard noise prediction training (Eq. 5), while the predictor is trained via MSE. They are cascaded during inference.

### Key Designs

1.  **Spatiotemporal Consistency Predictor**:
    - **Function**: Provides an action starting point in a single forward pass that satisfies both temporal consistency ($\|\tilde a_t-a_{t-1}\|\le\epsilon_t$) and spatial consistency ($\mathrm{dist}(\tilde a_t,\mathcal M(s_t))\le\epsilon_s$).
    - **Mechanism**: $f_\theta:\mathcal O\times\mathcal A^H\to\mathcal A^H$ utilizes a **2-layer cross-attention Transformer** (actions as queries, observations as keys/values, 128-dim embedding) to map $(\mathbf o_t,\mathbf A_{t-H})$ to $\hat{\mathbf A}_t$. The training objective is simply $\mathcal L_{pred}=\mathbb E\|\hat{\mathbf A}_t-\mathbf A_t\|^2$, learning the conditional expectation $\mathbb E[\mathbf A_t\mid\mathbf o_t,\mathbf A_{t-H}]$.
    - **Design Motivation**: Temporal consistency is derived from including $\mathbf A_{t-H}$ in the conditions, while spatial consistency comes from $\mathbf o_t$. No extra regularization is needed. Cross-attention is better suited for mixing heterogeneous sequences than self-attention. Experiments (Fig. 3) show performance saturates at 2 blocks; adding more only increases latency.

2.  **Velocity-Aware Perturbation Injection**:
    - **Function**: Automatically injects noise when the predictor's output change is extremely small (approaching static friction deadlock) to help the actuator cross the dead zone, otherwise maintaining the original signal.
    - **Mechanism**: Calculates the action difference $\Delta\mathbf A_t=\mathbf A_{cache}-\mathbf A_{t-2H}$ and detects stagnation using an indicator function $\mathbb I_t=\mathbb I(\|\Delta\mathbf A_t\|<\epsilon_a)$. The warm-start scale $\sigma$ and noise amplitude $\sigma_t$ switch between two levels according to Eq. 14: under normal conditions, $(\sigma,\sigma_t)=(1,0)$; during stagnation, $(\sigma,\sigma_t)=(\sigma_{scale},\sigma_{stall})$ to reduce amplitude and inject small Gaussian noise ($\epsilon_a=0.01$, simulation $\sigma_{stall}=0.1$).
    - **Design Motivation**: Real-world deployment faces "control dead zones + static friction" issues. Perfectly predicted actions in simulation may result in motor "laziness" in reality. Vanilla DDPM's inherent randomness can overcome these zones; this observation led to "on-demand stochasticity" as a toggle rather than constant noise.

3.  **Convergence Proof via Local Contraction Mapping**:
    - **Function**: Provides a theoretical explanation for why a "good warm-start + few reverse steps" converges stably to the correct action.
    - **Mechanism**: Formulates DDPM, DDIM, and DPM-Solver as $\mathbf A_{k-1}=\mu_k(\mathbf A_k,\mathbf o_t)+\boldsymbol\xi_k$ (Eq. 15). Assuming the denoising network $\epsilon_\theta$ has a Lipschitz constant $L$ within the neighborhood $\mathcal U$ of the data manifold, the reverse mean $\mu_k$ is also Lipschitz with coefficient $c_k<1$ (Eq. 16). Recursing through the steps yields $\|\tilde{\mathbf A}_0-\mathbf A_0\|\le\prod_{k=1}^{K'}c_k\|\tilde{\mathbf A}_{K'}-\mathbf A_{K'}\|$ (Eq. 18). Thus, if the predictor brings the starting point into $\mathcal U$, the error decays exponentially.
    - **Design Motivation**: Uses a unified contraction framework to explain why denoising from an intermediate step is superior to starting from pure noise, valid for DDIM/DPM-Solver regardless of the specific solver.

### Loss & Training
- Predictor: $\mathcal L_{pred}=\mathbb E\|\hat{\mathbf A}_t-\mathbf A_t\|^2$, 100k steps.
- DP: Follows original codebase default settings.
- Inference Hyperparameters: $K'$ = starting denoising step (i.e., STEP = 2 / 4); $\sigma=1, \sigma_t=0.1$ for simulation; real-world $\sigma_{stall}$ is larger to overcome dead zones.

## Key Experimental Results

### Main Results

**State-based RoboMimic / Push-T (Partial Table 2)**: Score (higher is better) / Time (ms, lower is better).

| Method | Step | Push-T | Square | ToolHang |
|---|---|---|---|---|
| Vanilla DDPM | 100 | 0.94 | 0.94 | 0.68 |
| DDIM | 2 | 0.29 | 0.84 | 0.06 |
| DPM-Solver++ | 2 | 0.20 | 1.00 | 0 |
| BRIDGER | 2 | 0.37 | 0.84 | 0.08 |
| Falcon | 2 | 0.21 | 1.00 | 0 |
| **STEP (Ours)** | **2** | **0.49** | **0.96** | **0.64** |

**Image-based RoboMimic (Partial Table 3)**: Long-range tasks like ToolHang show particularly significant improvement under visual input.

| Method | Step | Square | ToolHang |
|---|---|---|---|
| DDIM | 2 | 0.74 | 0.5 |
| BRIDGER | 2 | 0.92 | 0.72 |
| **STEP (Ours)** | **2** | – | – |

Core Conclusion: At 2 steps, STEP achieves an average success rate increase of 21.6% compared to BRIDGER and 48.8% compared to Falcon (temporal-only) on RoboMimic. In real-world tasks, it improves success rates by 27.5% over DDIM and reduces execution time by 59% via velocity-aware perturbations.

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Full STEP (2 step) | Push-T 0.49 / Lift 1.0 / Square 0.96 | Spatiotemporal consistency + perturbation + intermediate start. |
| No Predictor (= DDIM) | Push-T 0.29 / Lift 0.80 / Square 0.84 | Degenerates to pure noise start; performance collapses. |
| Spatial Only (BRIDGER) | Push-T 0.37 / Lift 1.0 / Square 0.84 | Lacks temporal continuity; fails on long-horizon tasks. |
| Temporal Only (Falcon) | Push-T 0.21 / Square 1.00 / ToolHang 0 | Lacks spatial consistency; ToolHang drops to zero. |
| Cross-attn block 1/2/4 | 2 is the sweet spot | 4 blocks increase latency without gain (Fig 3). |

### Key Findings
- **2-step inference is the core selling point**: While other methods fail almost completely at 2 steps (Falcon ToolHang=0), STEP maintains a success rate close to 100-step DDPM, pushing the Pareto frontier (latency vs success) significantly.
- **Spatiotemporal consistency is non-negotiable**: Any method with only TC or SC fails on at least one task at 2 steps; STEP remains robust across all.
- **Sim-to-Real Gap**: $\sigma_{stall}$ of 0.1 works in simulation, but larger values are needed for real robots, highlighting "friction/dead zones" as a major bottleneck.
- **Minimalist Predictor**: Uses only a 128-dim, 2-block cross-attention Transformer trained for 100k steps. It is extremely lightweight and easy to embed.

## Highlights & Insights
- **Clear Perspective**: Explicitly formalizes the requirement that "warm-starts must be spatiotemporally consistent" (Eq. 7-8 + Table 1), providing a simple analytical dimension for DP acceleration.
- **Decoupled Training Pattern**: This engineering route preserves the multi-modal generation of the original DP (no distillation or replacement) while providing lightweight acceleration—a practical design pattern applicable to any DP backbone.
- **Velocity-Aware Perturbation**: The idea of "on-demand stochasticity" can be migrated to any field needing dynamic switching between deterministic prediction and exploration (e.g., hybrid imitation/RL).
- **Contraction Proof**: Simple yet powerful, providing a unified theoretical explanation for "intermediate warm-starts" rather than just a narrative.

## Limitations & Future Work
- Temporal consistency relies on the previous $\mathbf A_{t-H}$, which may introduce bias during sudden state changes (e.g., encountering an obstacle). The current perturbation mechanism only monitors action magnitude, not observation changes.
- The predictor is a single forward pass and does not explicitly model multi-modality; it may "average out" meaningful actions if multiple modes exist (a common issue for BRIDGER-like methods).
- Only validated in imitation learning; compatibility between warm-starts and policy drift in closed-loop RL remains untested.
- Real-world $\sigma_{stall}$ requires manual tuning; a learned critic for adaptive perturbation could be a future improvement.

## Related Work & Insights
- **vs. DDIM / DPM-Solver++**: These only modify the solver without warm-starts, failing at 2 steps. STEP is orthogonal and can be applied to any solver.
- **vs. BRIDGER (Spatial-only)**: BRIDGER uses a predictor as a starting point but only considers the current state. STEP achieves a 21.6% average gain simply by adding $\mathbf A_{t-H}$.
- **vs. Falcon / RTI-DP (Temporal-only)**: These assume smooth dynamics and fail on tasks with rapid state changes like ToolHang/Push-T. STEP handles state mutations using observation conditions.
- **vs. CP / OneDP (Distillation)**: Distillation destroys multi-modal expressiveness; STEP preserves it, allowing more denoising steps for complex tasks and 2 steps for simple ones.

## Rating
- Novelty: ⭐⭐⭐⭐ Spatiotemporal二分 dimension + intermediate warm-start + velocity-aware perturbation; simple but effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 sim tasks + 2 real tasks × 8 baselines × state/image inputs; exhaustive ablation.
- Writing Quality: ⭐⭐⭐⭐ Conceptual diagrams (Fig 1) and consistency tables (Table 1) build the framework quickly; contraction proof is concise.
- Value: ⭐⭐⭐⭐⭐ A directly applicable engineering solution; open-source and easy to use for any robotics team deploying diffusion policies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Latent Reasoning VLA: Latent Thinking and Prediction for Vision-Language-Action Models](latent_reasoning_vla_latent_thinking_and_prediction_for_vision-language-action_m.md)
- [\[ICML 2026\] Lagrangian Perturbation Diffusion Steering: Latent Reinforcement Learning for Generative Policies](lagrangian_perturbation_diffusion_steering_latent_reinforcement_learning_for_gen.md)
- [\[ICML 2026\] RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies](robomme_benchmarking_and_understanding_memory_for_robotic_generalist_policies.md)
- [\[ICML 2026\] Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies](discrete_diffusion_vla_bringing_discrete_diffusion_to_action_decoding_in_vision-.md)
- [\[ICML 2026\] The Lie We Tell: Correcting the Euclidean Fallacy in Vision-Language-Action Policies via Score Matching on Tangent Space](the_lie_we_tell_correcting_the_euclidean_fallacy_in_vision_language_action_polic.md)

</div>

<!-- RELATED:END -->
