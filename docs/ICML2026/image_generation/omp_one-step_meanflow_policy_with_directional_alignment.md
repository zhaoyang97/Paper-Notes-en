---
title: >-
  [Paper Note] OMP: One-step Meanflow Policy with Directional Alignment
description: >-
  [ICML 2026][Image Generation][MeanFlow] This paper identifies three theoretical pathologies—spectral bias, gradient starvation in low-speed zones, and nested JVP memory explosion—when directly applying the MeanFlow paradigm to robot manipulation. It proposes OMP, which "locks" the direction of the predicted average velocity to the ground truth using a cosine
tags:
  - ICML 2026
  - Image Generation
  - MeanFlow
date: 2026-05-08
content_hash: aaf5651e690b9980
---
# OMP: One-step Meanflow Policy with Directional Alignment

**Conference**: ICML2026  
**arXiv**: [2512.19347](https://arxiv.org/abs/2512.19347)  
**Code**: To be confirmed  
**Area**: Robotics / Embodied AI / Generative Policy  
**Keywords**: MeanFlow, One-step Policy, Directional Alignment, JVP Finite Difference, Robot Manipulation

## TL;DR
This paper identifies three theoretical pathologies—spectral bias, gradient starvation in low-speed zones, and nested JVP memory explosion—when directly applying the MeanFlow paradigm to robot manipulation. It proposes OMP, which "locks" the direction of the predicted average velocity to the ground truth using a cosine-style directional alignment loss and decouples forward/backward passes using a Differential Derivation Equation (DDE) finite difference approximation. OMP achieves a 6.8ms inference latency (NFE=1) with success rates 3.4% higher than MP1 on average across Adroit/Meta-World, and 10.6% higher on Meta-World "Very Hard" tasks.

## Background & Motivation
**Background**: Generative robot policies currently model action generation as a probabilistic denoising process. Diffusion policies like Diffusion Policy and DP3 achieve high success rates through roughly 10 denoising iterations, but NFE=10 introduces inference latency that hinders high-frequency closed-loop control. To accelerate this, methods based on flow matching or consistency distillation (e.g., FlowPolicy, ManiFlow) compress inference to a single step. However, these rely on piecewise linear flows or explicit consistency constraints, where overly rigid architectural constraints may sacrifice generalization.

**Limitations of Prior Work**: MeanFlow (2025) theoretically provides a cleaner "one-step" path by directly learning the interval average velocity $u(z_t, r, t)$ to bypass ODE solvers. Its robotic implementation, MP1, reduced latency to 6.8ms. However, the authors find that directly migrating MeanFlow to robotics exposes three pathologies invisible in image generation scenarios.

**Key Challenge**: In image generation, the large pixel-level dynamic range and sufficient gradient signals mask the spectral and geometric defects of the MeanFlow objective. In contrast, robotics involves low-dimensional action spaces where the ground truth average velocity $\|v_0\|$ approaches zero in precision tasks. Three theoretical pathologies emerge: (1) **Spectral Bias**: Time integration acts as a low-pass filter (dividing by $i\omega$), causing the target PSD to decay by $1/\omega^2$, which suppresses high-frequency directional adjustments. (2) **Gradient Starvation**: The MSE loss gradient w.r.t. the angular error is $2\rho\rho^*\sin\alpha$, which is multiplicatively coupled with the target magnitude $\rho^*$. As $\rho^* \to 0$, the model tends to collapse its output to zero rather than aligning directions. (3) **Memory Complexity**: The total derivative in the MeanFlow Identity expands into a JVP $\nabla_z u \cdot dz/dt$. Computing $\nabla_\theta$ for this term is equivalent to nested Forward-AD and Reverse-AD, requiring the simultaneous storage of primal, tangent, and adjoint activations, which exceeds the memory capacity of large point cloud backbones.

**Goal**: (a) Decouple direction and magnitude in the loss function to prevent direction supervision from vanishing in low-speed zones. (b) Replace the JVP with an approximation that does not require symbolic differentiation, reducing training memory to standard backpropagation levels. (c) Maintain NFE=1 inference speed.

**Key Insight**: Since the root cause lies in the tight coupling of direction and magnitude in MSE and the analytical expansion of JVP, these can be bypassed directly. Use a cosine term to supervise direction as an independent loss and use central difference to approximate the time derivative.

**Core Idea**: Explicitly lock the "pointing" of the predicted average velocity to the ground truth $v_0$ using a directional alignment loss, and use an $O(\epsilon)$ central difference instead of JVP to decouple the forward and backward passes.

## Method

### Overall Architecture
OMP adopts the MeanFlow concept of "learning interval average velocity for one-step generation" for robot manipulation, specifically repairing the spectral bias, gradient starvation, and memory explosion issues seen in low-dimensional action spaces. The overall framework follows MP1: it takes 3D point cloud observations (downsampled to 512 or 1024 points) and 2 history steps to learn a model $u_\theta(z_t, r, t \mid c)$ representing the average velocity between times $r$ and $t$, obeying the MeanFlow Identity:

$$u(z_t,r,t|c)=v(z_t,t|c)-(t-r)\dfrac{d}{dt}u(z_t,r,t|c)$$

The right side serves as the target. During inference, a single forward pass goes from noise $z_T \sim \mathcal{N}(0, I)$ directly to action $z_0$, defining $v_0 \triangleq z_T - z_0$ as the ground truth average velocity. OMP adds two modifications over MP1’s $\mathcal{L}_{mse} + \lambda_{Disp}\mathcal{L}_{Disp}$: a directional alignment loss $\mathcal{L}_{DA}$ to treat geometric issues, and a central difference approximation for $\frac{d}{dt}u$ to treat memory issues. The latter results in two versions: OMP-JVP (analytical) and OMP-DDE (difference-based).

### Key Designs

**1. Directional Alignment Loss $\mathcal{L}_{DA}$: Decoupling Direction from Magnitude**

This addresses the failure of MSE in precision tasks. The authors show in §4.2.2 that the MSE gradient w.r.t. angle $\alpha$ is $\partial\mathcal{L}_{MSE}/\partial\alpha = 2\rho\rho^*\sin\alpha$. The angular gradient is suppressed by the target magnitude $\rho^*$. Since $\rho^* \approx 0$ during precision contact phases, MSE encourages the model to shrink its output ($\rho \to 0$) into a "static policy" rather than learning the correct direction. Furthermore, time integration acts as a $1/\omega^2$ low-pass filter. $\mathcal{L}_{DA}$ calculates cosine similarity $\cos\alpha = \dfrac{v_0 \cdot u}{\|v_0\| \cdot \|u\|}$ (with $\epsilon_{dir} \approx 10^{-6}$ for stability) and uses a log-form: $\mathcal{L}_{DA} = -\log\!\big(\frac{\cos\alpha+1}{2}\big)$. This loss depends solely on direction; thus, gradients do not collapse as $\|v_0\| \to 0$. The log-form provides the strongest penalty (gradient divergence) at $\cos\alpha = -1$ (opposite direction). In training, $\mathcal{L}_{mse}$ dominates the ballistic phase (large translations), while $\mathcal{L}_{DA}$ dominates the contact phase (direction), ensuring non-zero gradients throughout.

**2. Differential Derivation Equation (DDE): Using Central Difference to Replace Analytical Derivatives**

This addresses the memory cost of computing $\frac{d}{dt}u$. In §4.2.3, it is noted that the total derivative expansion involves a JVP $\nabla_z u_\theta \cdot v$. Computing $\nabla_\theta$ for this term results in second-order mixed partial derivatives $\partial^2 u/\partial\theta\partial z$, necessitating nested Forward-AD and Reverse-AD. This requires storing original activations $X$, tangents $\delta X$, and adjoints of tangents, which is infeasible for point cloud backbones on consumer GPUs. DDE approximates the time derivative with a central difference: $\dfrac{du_\theta(z_t,t,r|c)}{dt} \approx \dfrac{u_\theta(z_{t+\epsilon},t+\epsilon,r|c) - u_\theta(z_{t-\epsilon},t-\epsilon,r|c)}{2\epsilon}$. This reduces the training graph to two standard forward passes and one backward pass, requiring memory levels consistent with standard backpropagation. The $O(\epsilon^2)$ truncation error is a trade-off for decoupling the computation graph.

**3. Composite Loss and Dual Versions: Switchable Memory Optimization**

The final objective is $\mathcal{L} = \mathcal{L}_{mse} + \lambda_{Disp}\mathcal{L}_{Disp} + \lambda_{DA}\mathcal{L}_{DA}$, where $\mathcal{L}_{Disp}$ follows MP1's dispersive loss for feature separability. The three terms provide magnitude, discriminative features, and direction signals, respectively. Implementation of $\frac{d}{dt}u$ is split into two versions: OMP-JVP preserves the analytical JVP for maximum accuracy (academic baseline), while OMP-DDE uses the DDE approximation to save VRAM (for deployment), allowing users to switch based on task scale.

### Loss & Training
- **Loss**: $\mathcal{L} = \mathcal{L}_{mse} + \lambda_{Disp}\mathcal{L}_{Disp} + \lambda_{DA}\mathcal{L}_{DA}$; $\mathcal{L}_{DA} = -\log\!\big(\frac{\cos\alpha+1}{2}\big)$; DDE time step $\epsilon$ sensitivity is analyzed in §E.2.
- **Data**: 10 expert demonstrations per task; points sampled at 512 or 1024; images at 84×84; history=2, prediction horizon=4, execution horizon=3.
- **Training**: AdamW, lr=1e-4, batch=128; Adroit for 3000 epochs, Meta-World for 1000 epochs. Performance is the average of the top 5 evaluations across 3 seeds (0/10/20); trained on a single RTX 4090.

## Key Experimental Results

### Main Results: Adroit + Meta-World (37 Tasks Average)

| Method | NFE | Adroit Pen | MW Medium | MW Hard | MW Very Hard | Overall Avg |
|------|-----|------------|-----------|---------|--------------|--------|
| DP (RSS'23) | 10 | 13±2 | 11.0±2.5 | 5.25±2.5 | 22.0±5.0 | 35.2±5.3 |
| DP3 (RSS'24) | 10 | 46±10 | 44.5±8.7 | 32.7±7.7 | 39.4±9.0 | 68.7±4.7 |
| FlowPolicy (AAAI'25) | 1 | 54±4 | 58.2±7.9 | 40.2±4.5 | 52.2±5.0 | 71.6±3.5 |
| MP1 (AAAI'26) | 1 | 58±5 | 68.0±3.1 | 58.1±5.0 | 67.2±2.7 | 78.9±2.1 |
| **OMP-JVP (Ours)** | 1 | **60±4** | **77.4±2.2** | **62.5±3.1** | **77.8±3.0** | **82.3±1.6** |
| **OMP-DDE (Ours)** | 1 | 64±3 | 76.4±2.7 | 61.0±3.0 | 70.6±4.9 | 80.8±2.2 |

OMP-JVP outperforms MP1 by 3.4% and FlowPolicy by 10.7% in overall average. The Gain is more pronounced in harder tasks: +9.4% in Meta-World Medium and +10.6% in Very Hard.

### Real-Robot Experiments (3 Tasks, Success Rate %)

| Method | Place | Clean | Slip Ring |
|------|-------|-------|-----------|
| DP3 | 65 | 60 | 50 |
| FlowPolicy | 60 | 50 | 40 |
| MP1 | 70 | 65 | 55 |
| **OMP (Ours)** | **80** | **75** | **70** |

On the most difficult task (Slip Ring), OMP exceeds MP1 by 15%, verifying the benefit of directional alignment in real-world precision manipulation.

### Ablation Study

| Configuration | Overall Avg Success Rate | Description |
|------|--------------|------|
| OMP-JVP (Full) | 82.3 | Full model |
| − $\mathcal{L}_{Disp}$ | 81.2 | Remove dispersive loss (-1.1%) |
| − $\mathcal{L}_{DA}$ | 78.9 | Remove directional alignment (-3.4%, back to MP1) |
| − $\mathcal{L}_{Disp}$ − $\mathcal{L}_{DA}$ | 78.3 | Baseline MeanFlow |
| OMP-DDE (Full) | 80.8 | Difference approximation version |

**VRAM Comparison**:
| Task / Horizon | OMP-JVP VRAM | OMP-DDE VRAM |
|----------------|--------------|--------------|
| Place Bottle / H=16 | **26.71 GB** | **19.19 GB** |

### Key Findings
- **Directional Alignment is the Key**: Removing $\mathcal{L}_{DA}$ reduces success by 3.4–3.6%, whereas removing $\mathcal{L}_{Disp}$ only results in a ~1% drop, proving that the pathology lies in MSE's geometric coupling.
- **Gain Correlates with Task Difficulty**: MP1 nearly saturates Easy tasks (88%+), but OMP provides a +10.6% boost in Very Hard tasks, confirming that directional alignment specifically saves low-speed precision tasks.
- **JVP to DDE is an Accuracy-Memory Trade-off**: DDE drops average accuracy by 1.5% but reduces VRAM by 28% for large-scale tasks (e.g., $H=16$).
- **Stable Training**: Figure 5 shows OMP success curves have significantly lower variance compared to the oscillations in FlowPolicy/MP1.

## Highlights & Insights
- **Unified Narrative for Three Pathologies**: The author does not just "add a loss" but uses PSD frequency analysis, cosine-gradient derivations, and AD graph analysis to tell a cohesive story. The persuasiveness comes from the theoretical motivation rather than just the benchmarks.
- **Logarithmic Cosine Trick**: Using $-\log((\cos\alpha+1)/2)$ rather than $1-\cos\alpha$ is a reusable trick; the gradient diverges at $\cos\alpha=-1$, providing strong correction for completely wrong directions, whereas $1-\cos\alpha$ has its smallest gradient there.
- **Value of DDE for Graph Decoupling**: This trick is applicable to any scenario requiring backpropagation through $du_\theta/dt$ (e.g., higher-order score matching), allowing memory to remain within standard levels.
- **Task-Difficulty Bucketing**: Explicitly noting that MP1 is near saturation in Easy subsets and focusing on Hard/Very Hard gains clarifies the "applicability boundary" of the method.

## Limitations & Future Work
- **Small-Scale Real Robot Experiments**: 20 trials per task (10% granularity) lacks statistical strength compared to simulation.
- **Hyperparameter $\epsilon$**: DDE relies on a manually tuned $\epsilon$. Adaptive selection based on trajectory curvature could be a future direction.
- **Comparison with Distillation-based Methods**: Lacks a detailed comparison with well-distilled one-step methods like OneDP or ManiFlow.
- **Multimodal Action Distributions**: Does forcing a single direction with cosine loss sacrifice mode diversity in dexterous tasks where multiple paths exist?

## Related Work & Insights
- **vs MP1 (AAAI'26)**: MP1 introduced MeanFlow to robotics but struggled with directional gradients. OMP adds the "last missing piece" ($\mathcal{L}_{DA}$) to MP1.
- **vs DP3 (RSS'24)**: DP3 uses NFE=10 to reach 68.7%; OMP uses NFE=1 to reach 82.3%, suggesting that "one-step" and "high quality" are no longer mutually exclusive if geometric signals are handled.
- **vs Consistency Policy**: Consistency policies rely on multi-step teachers. OMP is trained from scratch but must handle the second-order costs of the MeanFlow Identity, which DDE resolves.

## Rating
- Novelty: ⭐⭐⭐⭐ While cosine losses and differences are not new math, combining them through spectral/AD analysis as a "MeanFlow fix kit" for robotics is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive simulation (37 tasks) and real-robot tests, though real-world stats are limited.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical motivation and structure.
- Value: ⭐⭐⭐⭐ Provides a template for fixing MeanFlow in low-dimensional action spaces.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation](../../AAAI2026/image_generation/mp1_meanflow_tames_policy_learning_in_1-step_for_robotic_manipulation.md)
- [\[CVPR 2026\] Temporal Equilibrium MeanFlow: Bridging the Scale Gap for One-Step Generation](../../CVPR2026/image_generation/temporal_equilibrium_meanflow_bridging_the_scale_gap_for_one-step_generation.md)
- [\[CVPR 2026\] MeanFlow Transformers with Representation Autoencoders](../../CVPR2026/image_generation/meanflow_transformers_with_representation_autoencoders.md)
- [\[CVPR 2026\] Understanding, Accelerating, and Improving MeanFlow Training](../../CVPR2026/image_generation/understanding_accelerating_and_improving_meanflow_training.md)
- [\[CVPR 2026\] Taming Preference Mode Collapse via Directional Decoupling Alignment in Diffusion Reinforcement Learning](../../CVPR2026/image_generation/taming_preference_mode_collapse_via_directional_decoupling_alignment_in_diffusio.md)

</div>

<!-- RELATED:END -->
