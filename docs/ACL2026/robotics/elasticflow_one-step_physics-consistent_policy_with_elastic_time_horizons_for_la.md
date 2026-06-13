---
title: >-
  [Paper Note] ElasticFlow: One-Step Physics-Consistent Policy with Elastic Time Horizons for Language-Guided Manipulation
description: >-
  [ACL 2026][Robotics][One-step diffusion] The authors propose ElasticFlow, which replaces instantaneous velocity field learning with a mean velocity field (MeanFlow) to learn language-conditioned robot actions. By explici…
tags:
  - "ACL 2026"
  - "Robotics"
  - "One-step diffusion"
  - "Mean velocity field"
  - "Elastic time"
  - "Flow matching"
  - "VLA"
date: 2026-05-08
content_hash: 5ae16e7541c93130
---

# ElasticFlow: One-Step Physics-Consistent Policy with Elastic Time Horizons for Language-Guided Manipulation

**Conference**: ACL 2026  
**arXiv**: [2605.08799](https://arxiv.org/abs/2605.08799)  
**Code**: To be confirmed  
**Area**: Embodied AI / Diffusion Policy / Flow Matching / Robot Manipulation  
**Keywords**: One-step diffusion, Mean velocity field, Elastic time, Flow matching, VLA

## TL;DR
The authors propose ElasticFlow, which replaces instantaneous velocity field learning with a mean velocity field (MeanFlow) to learn language-conditioned robot actions. By explicitly encoding control granularity via an "Elastic Time Horizon $\Delta t=t-r$," it achieves 1-NFE single-step inference (~71Hz) and outperforms OpenVLA and $\pi_0$ on long-horizon tasks such as LIBERO-Long and CALVIN ABC-D.

## Background & Motivation

**Background**: In Embodied AI, generalist policies that map visual observations and language instructions to continuous actions primarily follow two paths: Diffusion Policies (e.g., Diffusion Policy, $\pi_0$) have become mainstream due to their strong multimodal modeling capabilities; autoregressive VLAs (e.g., OpenVLA, RT-2) rely on token discretization for language-action alignment.

**Limitations of Prior Work**: Iterative denoising in diffusion policies requires dozens of NFEs (Network Function Evaluations), leading to latency > 100ms and control frequencies of only 8–12Hz, which cannot respond to rapidly changing physical environments (e.g., intercepting rolling objects). Existing acceleration schemes (Consistency Model, Progressive Distillation) require complex teacher-student pipelines and often sacrifice "physics consistency," leading to high-frequency jitter (jerk) or non-smooth paths. Autoregressive VLAs are even slower (~5Hz) due to token-by-token generation, and discretization introduces quantization errors.

**Key Challenge**: (1) There is a trade-off between inference speed and physics consistency; simply reducing steps causes trajectories to lose geometric smoothness. (2) The "temporal heterogeneity" of robot tasks is often ignored—short-range reactive control requires millisecond jitter suppression, while long-range tasks require second-level trajectory planning. Traditional fixed-horizon networks fall into Spectral Bias, failing to model high-frequency and low-frequency signals simultaneously.

**Goal**: (1) Achieve 1-NFE inference without distillation. (2) Ensure single-step predictions remain geometrically smooth and physically consistent. (3) Enable a single set of weights to perform both millisecond-level reactive control and long-range multi-stage planning.

**Key Insight**: MeanFlow, proposed by Geng et al. (2025) in generative modeling, directly learns the "mean velocity" over a time interval $[r,t]$ rather than the instantaneous velocity. A single forward pass yields the overall displacement from noise to data. The authors adapt this to robot action flows and expose $\Delta t=t-r$ to the network as a "control granularity knob."

**Core Idea**: The method utilizes a mean velocity field $u(z_t,r,t)$ instead of instantaneous velocity $v(z_t,t)$ for action generation, using $\Delta t$ as a "Spectral Zoom Lens" to unify short-range reaction and long-range planning.

## Method

### Overall Architecture
Observations $o$ are encoded via SigLIP, and language instructions $\ell$ are encoded via T5. These are injected into a DiT backbone (150M parameters) via cross-attention. The Elastic Time Horizon module encodes $(r,t,\Delta t)$ using Fourier features, which are injected via AdaLN modulation. The output is the mean velocity field prediction $u_\theta(z_t,r,t,o,\ell)$. Training is supervised using the MeanFlow Identity Loss with forward-mode AD. During inference, given $z_1\sim\mathcal{N}(0,I)$, a single forward pass calculates the action chunk $\hat{x}=z_1-u_\theta(z_1,0,1,o,\ell)$ without any iteration or distillation.

### Key Designs

1.  **Mean Velocity Field Modeling (MeanFlow Identity)**:
    - **Function**: Transforms action generation from "solving multi-step ODE integration" into "learning a one-step mapping," while implicitly incorporating trajectory curvature correction to ensure physical smoothness.
    - **Mechanism**: Define $u(z_t,r,t)\triangleq\frac{1}{t-r}\int_{r}^{t}v(z_\tau,\tau)d\tau$. From the fundamental theorem of calculus, the identity $u(z_t,r,t)=v(z_t,t)-(t-r)\frac{d}{dt}u(z_t,r,t)$ is derived, where $\frac{d}{dt}$ is the total derivative (including $v\cdot\nabla_z u$ and $\partial_t u$). The training objective aligns the network prediction with the right side of this identity. The instantaneous velocity ground truth is constructed using optimal transport paths $v(z_t,t)=x_{\text{target}}-x_{\text{noise}}$.
    - **Design Motivation**: Instantaneous velocity only describes local tangents, requiring ODE integration to recover global displacement, which makes multi-step inference a "necessary cost." By using mean velocity, a single step captures geometric information across the entire time interval, and the $(t-r)\frac{d}{dt}u$ term naturally acts as a "manifold curvature correction," suppressing high-frequency jitter at the formula level.

2.  **Elastic Time Horizon**:
    - **Function**: Uses a continuous parameter $\Delta t=t-r$ to allow the same network to switch seamlessly between "short-range high-frequency reaction" and "long-range low-frequency planning."
    - **Mechanism**: In addition to absolute time $t$, $\Delta t$ is fed into the network. Both are encoded via Gaussian Fourier features $\text{Emb}(r,t)=\text{MLP}([\text{FF}(t),\text{FF}(t-r)])$ and modulate the DiT via AdaLN. During inference, $\Delta t$ is selected based on task granularity: small $\Delta t$ focuses on local pose adjustment, while large $\Delta t$ performs long-range trajectory planning, switching dynamically within a single weight space. During execution, the continuous flow is discretized into $N$ steps based on the target control frequency, with a physical step size $\delta t=T/N$.
    - **Design Motivation**: Neural networks exhibit Spectral Bias, making it difficult to fit high-frequency signals. Explicitly injecting $\Delta t$ is equivalent to telling the network "which scale to observe," acting as a spectral zoom lens that covers both short-range transients and long-range structures under one set of weights. Mismatch tests (forcing an incorrect $\Delta t$) verify this physical significance.

3.  **Forward-mode AD + Stop-gradient + CFG Training**:
    - **Function**: Stably utilizes the MeanFlow Identity as a training objective while supporting language-conditioned Classifier-Free Guidance (CFG).
    - **Mechanism**: The loss is $\mathcal{L}(\theta)=\mathbb{E}_{t,r,x_1,\epsilon,c}[\|u_\theta(z_t,r,t,o,c)-\text{sg}(\mathcal{T}_{\text{target}})\|_2^2]$, where $\mathcal{T}_{\text{target}}=v(z_t,t)-(t-r)(v(z_t,t)\cdot\nabla_z u_\theta+\partial_t u_\theta)$. The Jacobian-vector product $\nabla_z u_\theta$ is computed efficiently using forward-mode automatic differentiation (AD) to avoid Hessian overhead. Conditions $c\in\{\ell,\emptyset\}$ are substituted with an empty set with probability $p_{\text{drop}}$ for joint training. Inference uses $\hat{x}=z_1-(u_\theta(\cdot,\emptyset)+w(u_\theta(\cdot,\ell)-u_\theta(\cdot,\emptyset)))$.
    - **Design Motivation**: Directly calculating gradients for bootstrapped targets can diverge; stop-gradient isolates self-referencing terms to stabilize optimization. Forward-mode AD addresses the computational bottleneck of second-order terms. CFG allows single-step inference to still adjust semantic alignment strength.

### Loss & Training
The model minimizes the Mean Square Error (MSE) towards the target $\mathcal{T}_{\text{target}}$ derived from the MeanFlow Identity. The stop-gradient $\text{sg}(\cdot)$ stabilizes bootstrap optimization. Using a CFG drop probability $p_{\text{drop}}$, the guidance scale $w$ remains stable within $[1.5, 2.5]$ during inference. The 150M DiT backbone is lighter than the 300M UNet used in Diffusion Policy.

## Key Experimental Results

### Main Results
**LIBERO Benchmarks + CALVIN ABC-D**:

| Benchmark / Metric | ElasticFlow | Prev. SOTA | Key Baseline |
|---|---|---|---|
| LIBERO-Spatial (SR ↑) | 98.4 | 98.8 (HiF-VLA) | $\pi_0$ 96.8 |
| LIBERO-Object (SR ↑) | 99.3 | 99.4 (HiF-VLA) | $\pi_0$ 98.8 |
| LIBERO-Goal (SR ↑) | **98.7** | 97.9 (OpenVLA-OFT) | $\pi_0$ 95.8 |
| LIBERO-Long (SR ↑) | **97.6** | 96.4 (HiF-VLA) | $\pi_0$ 85.2 |
| LIBERO Average | **98.5** | 98.0 (HiF-VLA) | Octo 75.1 |
| CALVIN ABC-D 3rd Person (Avg.Len. ↑) | **4.15** | 4.08 (HiF-VLA) | $\pi_0$ 3.65 |
| CALVIN ABC-D Multi-view (Avg.Len. ↑) | **4.37** | 4.35 (HiF-VLA) | $\pi_0$ 3.92 |
| RoboTwin Long & Extra Long (SR ↑) | **71.1** | 69.0 (SimpleVLA-RL) | $\pi_0$ 43.3 / RDT 27.8 |

The 12 percentage point jump in LIBERO-Long (from 85.2 for $\pi_0$ to 97.6) is the most significant result, as long-horizon tasks are where diffusion policies typically fail.

**Inference Latency (RTX 4090, batch=1)**:

| Method | NFE | Latency (ms) ↓ | Frequency (Hz) ↑ |
|------|-----|-----------|------------|
| OpenVLA (7B Transformer) | Auto-reg. | 200.0 | ∼5 |
| Diffusion Policy (300M UNet) | 16 (DDIM) | 120.0 | ∼8 |
| $\pi_0$ (300M DiT) | 10 (Euler) | 85.0 | ∼12 |
| Consistency Policy | 2 | 28.0 | ∼35 |
| **ElasticFlow (150M DiT)** | **1** | **14.0** | **71** |

This represents a 5× speedup compared to Diffusion Policy and a 14× speedup compared to OpenVLA.

### Ablation Study

| Configuration | Long Horizon SR | Short Horizon SR | Description |
|------|----------------|----------------|------|
| w/o Horizon Input | 52.7% | 61.5% | Removing $\Delta t$ loses 18.4 pts |
| Fixed $\Delta t=10$ | 58.2% | 94.5% | Strong short-range, weak long-range |
| Fixed $\Delta t=50$ | 62.1% | 55.4% | Slightly stronger long-range, short-range fails |
| Mismatch Force $\Delta t=10$ on Long | 45.3% | — | "Myopic": Only looks at immediate surroundings |
| Mismatch Force $\Delta t=50$ on Short | — | 55.7% | "Sluggish": Cannot react quickly |
| **ElasticFlow (Dynamic $\Delta t$)** | **71.1%** | **98.2%** | Full model |

| Training Objective | Step | Success Rate | Jerk ↓ | Latency |
|---------|------|-------|-------|------|
| Standard CFM ($v_t$) | 1-NFE | 12.4% | $8.5\times 10^{-2}$ | 14ms |
| Standard CFM ($v_t$) | 10-NFE | 68.5% | $3.2\times 10^{-3}$ | 140ms |
| **ElasticFlow ($u_t$)** | **1-NFE** | **71.1%** | $\mathbf{1.1\times 10^{-3}}$ | **14ms** |

### Key Findings
- The $\Delta t$ module is the primary contributor to ElasticFlow: removing it results in an 18.4 point drop. Mismatch tests (forcing the wrong $\Delta t$) caused success rates to fall to 45.3% for long-horizon and 55.7% for short-horizon tasks, disproving the feasibility of a "fixed horizon" approach.
- The primary benefit of mean velocity field modeling is maintaining high physics consistency at 1-NFE: the Jerk of single-step ElasticFlow ($1.1\times 10^{-3}$) is lower than that of 10-step CFM ($3.2\times 10^{-3}$), proving that curvature correction terms suppress jitter at the mathematical level.
- Standard Flow Matching fails in a single step (SR 12.4%), but ElasticFlow achieves 71.1%, indicating that speed is not simply a trade-off between model size and step count; the underlying modeling target must change.
- ElasticFlow maintains 83.6% / 72.7% success rates on the 4th and 5th instructions in CALVIN, demonstrating superior stability in long instruction chains compared to baselines that decay after initial tasks.
- The CFG guidance scale $w$ is stable within $[1.5, 2.5]$ and insensitive to hyperparameters.

## Highlights & Insights
- "Learning average velocity instead of instantaneous velocity" represents a paradigm shift: the ODE integration bottleneck is moved from "runtime computation" to "training-time representation," making 1-NFE a mathematical equivalence rather than an engineering compromise.
- The Elastic Time Horizon $\Delta t$ is an extremely economical design: by adding just one dimensional parameter + Fourier encoding, a single network handles both high-frequency reaction and long-range planning, eliminating the need for cascaded networks of different horizons.
- The "Spectral Zoom Lens" metaphor is more than just rhetorical—mismatch tests provide physical validation through counterfactual analysis, grounding the abstract concept of spectral bias in observable robot behavior.
- The distillation-free nature provides significant engineering value: unlike Consistency Policy or Progressive Distillation which require a teacher-student pipeline, ElasticFlow achieves 1-NFE via one-stage training, reducing deployment costs.

## Limitations & Future Work
- The data scale is limited to millions of interactions and has not been verified against scaling laws on billion-level cross-embodiment data like Open X-Embodiment.
- While forward-mode AD for Jacobian-vector products is cheaper than Hessian calculations, it still incurs significant overhead compared to standard training; a detailed wall-clock comparison was not provided.
- The selection of $\Delta t$ currently relies on task priors; ideally, $\Delta t$ should be adaptively selected by the policy based on context.
- Language conditions are only injected via cross-attention; semantic parsing of complex multi-stage instructions is still limited by the T5 encoder.
- High-frequency (71Hz) single-step inference leaves very little time for error recovery; integration with MPC or online RL for closed-loop correction is a natural progression.
- Sim-to-Real evaluation was limited to qualitative and partial quantitative tests on xArm6; reliability for cross-platform real-world deployment requires further validation.

## Related Work & Insights
- **vs. Diffusion Policy (Chi 2023) / $\pi_0$**: Both are flow/diffusion-based, but DP/$\pi_0$ learn instantaneous velocity requiring multi-step integration; ElasticFlow learns the mean velocity field for 1-step, smoother physics.
- **vs. Consistency Policy (Prasad 2024) / Progressive Distillation**: Those require teacher-student pipelines that are complex and often sacrifice diversity; ElasticFlow offers distillation-free one-stage training.
- **vs. OpenVLA / CogACT / OpenVLA-OFT**: Autoregressive VLAs use token discretization, resulting in low frequency and quantization errors; ElasticFlow generates in continuous action space with an order of magnitude higher frequency.
- **vs. MeanFlow (Geng 2025)**: The core idea stems from the image generation version of MeanFlow; this work implements it for robot actions, adds Elastic Time Horizons, and language-conditioned CFG.
- **vs. ACT (Zhao 2023)**: ACT utilizes action chunking and heuristic temporal ensembling to reduce inference pressure but is prone to over-smoothing; ElasticFlow enforces physics consistency at the formula level and allows $\Delta t$ to dynamically adjust chunk length.

## Rating
- Novelty: ⭐⭐⭐⭐ First application of MeanFlow in robot control combined with a clear Elastic Time Horizon mechanism.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive simulation (LIBERO, CALVIN, RoboTwin, RoboCasa) + xArm6 real-robot tests + latency scans; Mismatch tests are cleverly designed.
- Writing Quality: ⭐⭐⭐⭐ Clear derivations, intuitive analogies ("Spectral Zoom Lens"), and geometric explanations; dense but well-structured.
- Value: ⭐⭐⭐⭐⭐ Provides a viable 1-NFE paradigm for VLA/robot diffusion policies with both engineering and methodological merit.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Mixture of Horizons in Action Chunking](../../ICML2026/robotics/mixture_of_horizons_in_action_chunking.md)
- [\[ICLR 2026\] VLBiMan: Vision-Language Anchored One-Shot Demonstration Enables Generalizable Bimanual Robotic Manipulation](../../ICLR2026/robotics/vlbiman_vision-language_anchored_one-shot_demonstration_enables_generalizable_bi.md)
- [\[ICLR 2026\] ST4VLA: Spatially Guided Training for Vision-Language-Action Models](../../ICLR2026/robotics/st4vla_spatially_guided_training_for_vision-language-action_models.md)
- [\[ICLR 2026\] Real-Time Robot Execution with Masked Action Chunking](../../ICLR2026/robotics/real-time_robot_execution_with_masked_action_chunking.md)
- [\[ICLR 2026\] MoMaGen: Generating Demonstrations under Soft and Hard Constraints for Multi-Step Bimanual Mobile Manipulation](../../ICLR2026/robotics/momagen_generating_demonstrations_under_soft_and_hard_constraints_for_multi-step.md)

</div>

<!-- RELATED:END -->
