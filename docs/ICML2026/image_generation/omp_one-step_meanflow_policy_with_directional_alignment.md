---
title: >-
  [Paper Note] OMP: One-step Meanflow Policy with Directional Alignment
description: >-
  [ICML2026][Image Generation][MeanFlow] This paper addresses three theoretical "pathologies" exposed when applying the MeanFlow paradigm directly to robotic manipulation (spectral bias, gradient starvation in low-speed zones, and nested JVP memory explosion). It proposes OMP: a cosine-style directional alignment loss is used to "lock" the predicted mean velocity direction to the ground truth, and a Finite Difference DDE is utilized to approximate the Jacobian-Vector Product (J…
tags:
  - "ICML2026"
  - "Image Generation"
  - "MeanFlow"
  - "One-step Policy"
  - "Directional Alignment"
  - "JVP Finite Difference"
  - "Robotic Manipulation"
date: 2026-05-08
content_hash: 0e10dfc65e6a9bc2
---

# OMP: One-step Meanflow Policy with Directional Alignment

**Conference**: ICML2026  
**arXiv**: [2512.19347](https://arxiv.org/abs/2512.19347)  
**Code**: To be confirmed  
**Area**: Robotics / Embodied AI / Generative Policy  
**Keywords**: MeanFlow, One-step Policy, Directional Alignment, JVP Finite Difference, Robotic Manipulation

## TL;DR
This paper addresses three theoretical "pathologies" exposed when applying the MeanFlow paradigm directly to robotic manipulation (spectral bias, gradient starvation in low-speed zones, and nested JVP memory explosion). It proposes OMP: a cosine-style directional alignment loss is used to "lock" the predicted mean velocity direction to the ground truth, and a Finite Difference DDE is utilized to approximate the Jacobian-Vector Product (JVP), decoupling forward and backward passes. This allows the one-step (NFE=1) generative policy to achieve a 6.8ms latency while outperforming MP1 by an average of 3.4% on Adroit/Meta-World and by 10.6% on Meta-World "Very Hard" tasks.

## Background & Motivation
**Background**: Current mainstream generative robot policies model action generation as a probabilistic denoising process. Diffusion policies like Diffusion Policy / DP3 achieve high success rates through iterative denoising (approx. 10 steps), but the NFE=10 latency hinders high-frequency closed-loop control. To accelerate, methods based on flow matching or consistency distillation, such as FlowPolicy and ManiFlow, compress inference to a single step. however, training relies on piecewise linear flows or explicit consistency constraints, where overly strong architectural constraints sacrifice generalization.

**Limitations of Prior Work**: MeanFlow (2025) theoretically provides a cleaner "one-step" path by directly learning the interval mean velocity $u(z_t, r, t)$, bypassing ODE solvers. Its robotic implementation, MP1, reduced latency to 6.8 ms. However, the authors found that directly transplanting MeanFlow to robotics exposes three pathologies not visible in image generation scenarios.

**Key Challenge**: In image generation, the large pixel-level dynamic range and sufficient gradient signals mask the spectral and geometric defects of the MeanFlow objective. Conversely, robotic action spaces are low-dimensional, and the true mean velocity $\|v_0\|$ approaches 0 in fine-grained tasks, causing three theoretical pathologies to erupt: (1) **Spectral Bias**—time integration is equivalent to dividing by $i\omega$, causing the target PSD to decay by $1/\omega^2$, acting as a low-pass filter that suppresses high-frequency directional adjustments in fine manipulation; (2) **Gradient Starvation**—the gradient of the MSE loss with respect to the angular error is $2\rho\rho^*\sin\alpha$, which is multiplicatively coupled with the target magnitude $\rho^*$. As $\rho^*\to 0$, the model collapses its output to zero rather than aligning the direction; (3) **Memory Complexity**—the total derivative in the MeanFlow Identity contains the JVP $\nabla_z u\cdot dz/dt$. Computing $\nabla_\theta$ for this is equivalent to nested Forward-AD + Reverse-AD, requiring the simultaneous storage of primal, tangent, and adjoint activations, which large point cloud backbones cannot handle.

**Goal**: (a) Decouple the strong coupling of direction and magnitude in MSE so directional supervision does not vanish in low-speed zones; (b) replace the JVP with an approximation that does not require symbolic differentiation to reduce training memory to standard backprop levels; (c) maintain NFE=1 inference speed.

**Key Insight**: Since the root cause is the hard coupling of direction and magnitude by MSE and the memory explosion from the analytical expansion of JVP, the authors bypass them directly—using cosine to isolate direction as an independent loss term and using central difference to approximate the time derivative.

**Core Idea**: Lock the "heading" of the predicted mean velocity to the true mean velocity $v_0$ via directional alignment loss, combined with an $O(\epsilon)$ central difference to replace JVP and decouple forward/backward passes.

## Method

### Overall Architecture
OMP adopts the MeanFlow approach of "learning interval mean velocity for one-step generation" for robotic manipulation but fixes the three pathologies (spectral bias, gradient starvation, and memory explosion) exposed in low-dimensional action spaces. The overall framework follows MP1: it takes 3D point cloud observations (downsampled to 512 or 1024 points via FPS) plus 2 steps of observation history. The model $u_\theta(z_t, r, t \mid c)$ learns the mean velocity between times $r$ and $t$, following the MeanFlow Identity:

$$u(z_t,r,t|c)=v(z_t,t|c)-(t-r)\dfrac{d}{dt}u(z_t,r,t|c)$$

where the right side is the target. During inference, a single forward pass goes from noise $z_T\sim\mathcal{N}(0,I)$ directly to action $z_0$, defining $v_0 \triangleq z_T - z_0$ as the true mean velocity. The two modifications in OMP are layered on top of the $\mathcal{L}_{mse}+\lambda_{Disp}\mathcal{L}_{Disp}$ used in MP1: a directional alignment loss $\mathcal{L}_{DA}$ is added to treat geometric pathologies, and $\frac{d}{dt}u$ in the Identity is replaced with central difference to treat memory pathologies. The latter yields two versions: OMP-JVP (retaining analytical JVP) and OMP-DDE (using difference approximation).

### Key Designs

**1. Directional Alignment Loss $\mathcal{L}_{DA}$: Decoupling Direction from Magnitude to Solve Gradient Starvation**

This targets the root cause of MSE failure in fine-grained manipulation. In §4.2.2, the authors decompose the MSE gradient with respect to the angle using the Law of Cosines, obtaining $\partial\mathcal{L}_{MSE}/\partial\alpha = 2\rho\rho^*\sin\alpha$. The angular gradient is multiplicatively suppressed by the target magnitude $\rho^*$. Since $\rho^*\approx 0$ during fine-grained contact in robotics, MSE encourages the model to shrink the output $\rho\to 0$ to obtain a "static policy," failing to learn the direction. Additionally, the target being time-integrated acts as a $1/\omega^2$ low-pass filter. $\mathcal{L}_{DA}$ computes the cosine similarity $\cos\alpha = \dfrac{v_0\cdot u}{\|v_0\|\cdot\|u\|}$ (with $\epsilon_{dir}\approx 10^{-6}$ to prevent division by zero) and is formulated as a log-loss: $\mathcal{L}_{DA}=-\log\!\big(\frac{\cos\alpha+1}{2}\big)$. This loss depends only on direction, ensuring gradients do not collapse as $\|v_0\|\to 0$. The logarithmic form also makes the gradient diverge at $\cos\alpha=-1$ (complete opposite direction) for maximum penalty, while approaching zero at $\cos\alpha=1$. During training, the ballistic phase (large translations) is dominated by magnitude via $\mathcal{L}_{mse}$, while the contact phase is dominated by direction via $\mathcal{L}_{DA}$.

**2. Differential Derivation Equation (DDE): Approximating Analytical Time Derivatives via Central Difference**

This addresses the memory cost of implementing $\frac{d}{dt}u$ in MeanFlow Identity. §4.2.3 calculates that the expansion of this total derivative contains the JVP $\nabla_z u_\theta\cdot v$. Computing $\nabla_\theta$ for this involves second-order mixed partial derivatives $\partial^2 u/\partial\theta\partial z$, requiring Forward-AD to be nested within Reverse-AD. This necessitates storing three sets of computational graphs (primal activations $X$, tangent $\delta X$, and tangent adjoint), which point cloud backbones like PointNet++/Transformer cannot fit into a single RTX 4090. DDE approximates the time derivative using central difference: $\dfrac{du_\theta(z_t,t,r|c)}{dt}\approx\dfrac{u_\theta(z_{t+\epsilon},t+\epsilon,r|c)-u_\theta(z_{t-\epsilon},t-\epsilon,r|c)}{2\epsilon}$ ($\epsilon$ is a small perturbation constant; sensitivity scan in §E.2). The training graph then only requires two standard forward passes and one backward pass, eliminating tangent activation storage and returning memory usage to standard backprop levels. The trade-off is an $O(\epsilon^2)$ truncation error, but the true value lies in decoupling the forward and backward computation graphs.

**3. Combined Loss and Dual JVP/DDE Versions: Switchable Memory Optimization**

The final training objective is $\mathcal{L}=\mathcal{L}_{mse}+\lambda_{Disp}\mathcal{L}_{Disp}+\lambda_{DA}\mathcal{L}_{DA}$, where $\mathcal{L}_{Disp}$ follows MP1’s dispersive loss to make the feature space more separable. The three terms provide magnitude, feature discriminability, and directional signals, ensuring effective gradients in both ballistic and contact phases. The implementation of $\frac{d}{dt}u$ is split into two versions, leaving the "memory-accuracy" trade-off to the user: OMP-JVP retains analytical JVP for maximum accuracy (academic baseline), while OMP-DDE uses the DDE approximation to save VRAM (practical deployment).

### Loss & Training
- **Loss**: $\mathcal{L}=\mathcal{L}_{mse}+\lambda_{Disp}\mathcal{L}_{Disp}+\lambda_{DA}\mathcal{L}_{DA}$; $\mathcal{L}_{DA}=-\log\!\big(\frac{\cos\alpha+1}{2}\big)$; DDE time step $\epsilon$ sensitivity scan in §E.2.
- **Data**: 10 expert demonstrations per simulation task; point clouds at 512 or 1024 points; images at 84×84; observation history=2, prediction horizon=4, execution horizon=3.
- **Training**: AdamW, lr=1e-4, batch=128; Adroit for 3000 epochs, Meta-World for 1000 epochs; evaluation every 200 epochs; success rate averaged over top 5 and across seeds (0/10/20); hardware: single RTX 4090.

## Key Experimental Results

### Main Results: Adroit + Meta-World (Average of 37 Tasks)

| Method | NFE | Adroit Pen | MW Medium | MW Hard | MW Very Hard | Total Avg |
|------|-----|------------|-----------|---------|--------------|--------|
| DP (RSS'23) | 10 | 13±2 | 11.0±2.5 | 5.25±2.5 | 22.0±5.0 | 35.2±5.3 |
| DP3 (RSS'24) | 10 | 46±10 | 44.5±8.7 | 32.7±7.7 | 39.4±9.0 | 68.7±4.7 |
| FlowPolicy (AAAI'25) | 1 | 54±4 | 58.2±7.9 | 40.2±4.5 | 52.2±5.0 | 71.6±3.5 |
| MP1 (AAAI'26) | 1 | 58±5 | 68.0±3.1 | 58.1±5.0 | 67.2±2.7 | 78.9±2.1 |
| **OMP-JVP** | 1 | **60±4** | **77.4±2.2** | **62.5±3.1** | **77.8±3.0** | **82.3±1.6** |
| OMP-DDE | 1 | 64±3 | 76.4±2.7 | 61.0±3.0 | 70.6±4.9 | 80.8±2.2 |

OMP-JVP is 3.4% higher than MP1 and 10.7% higher than FlowPolicy in total average. The harder the task, the greater the OMP gain—Meta-World Medium +9.4%, Very Hard +10.6%. MP1 already approaches the ceiling (88%+) on Easy subsets (21/37 tasks), which lowers the absolute average Gain to 1.5%.

### Real-world Experiments (3 Tasks, Success Rate %)

| Method | Place | Clean | Slip Ring |
|------|-------|-------|-----------|
| DP3 | 65 | 60 | 50 |
| FlowPolicy | 60 | 50 | 40 |
| MP1 | 70 | 65 | 55 |
| **OMP** | **80** | **75** | **70** |

On the most difficult task, Slip Ring, OMP is 15% higher than MP1, validating the benefit of directional alignment in real-world low-speed fine manipulation.

### Ablation Study

| Configuration | Total Avg Success Rate | Description |
|------|--------------|------|
| OMP-JVP (Full) | 82.3 | Full model |
| − $\mathcal{L}_{Disp}$ | 81.2 | Removed dispersive loss, -1.1% (minor) |
| − $\mathcal{L}_{DA}$ | 78.9 | Removed directional alignment, -3.4%, back to MP1 level |
| − $\mathcal{L}_{Disp}$ − $\mathcal{L}_{DA}$ | 78.3 | Both removed, Adroit Pen 60→48 |
| OMP-DDE (Full) | 80.8 | Finite difference version |

| Task / Horizon | OMP-JVP VRAM | OMP-DDE VRAM |
|----------------|--------------|--------------|
| Adroit Hammer / H=4 | 6.60 GB | 5.35 GB |
| Place Bottle / H=4 | 23.49 GB | 18.33 GB |
| Adroit Hammer / H=16 | 7.69 GB | 6.12 GB |
| Place Bottle / H=16 | **26.71 GB** | **19.19 GB** |

### Key Findings
- **Directional Alignment is the Key**: Removing $\mathcal{L}_{DA}$ causes a 3.4–3.6% drop, whereas removing $\mathcal{L}_{Disp}$ only causes a 0.7–1.1% drop, proving the pathology root is geometric coupling in MSE rather than feature discriminability.
- **OMP Gain Correlates with Task Difficulty**: MP1 already hits the 88%+ ceiling on Easy tasks. The +10.6% on Very Hard tasks confirms that directional alignment primarily saves low-speed fine-grained tasks, consistent with theoretical expectations.
- **JVP→DDE is an Accuracy-Memory Trade-off**: DDE loses an average of 1.5% accuracy (more in Very Hard at 7.2%) but reduces VRAM by 28% (26.71 GB→19.19 GB) for Place Bottle/H=16. DDE becomes more cost-effective as point cloud size and horizon increase.
- **More Stable Training**: Figure 5 shows the success rate curve of OMP has much lower variance than the oscillations seen in FlowPolicy/MP1, suggesting directional alignment enhances training stability.

## Highlights & Insights
- **Thematic Theoretical Narrative**: The authors do not just "throw in" a cosine loss; they integrate PSD frequency analysis, the Law of Cosines for MSE angular gradients, and AD graph analysis of activations into a cohesive story of "why MeanFlow is unfit for robotics." This provides strong motivation for $\mathcal{L}_{DA}$ + DDE.
- **Log-cosine Loss Form**: Formulating the loss as $-\log((\cos\alpha+1)/2)$ instead of $1-\cos\alpha$ is a reusable trick; the logarithm makes gradients diverge near $\cos\alpha=-1$, heavily penalizing "wrong-way" movements, whereas $1-\cos\alpha$ has minimal gradients when reversed.
- **The True Value of DDE is Decoupling Computation Graphs**: This is applicable beyond MeanFlow—to any scenario requiring $du_\theta/dt$ while backpropagating through $\theta$ (e.g., second-order variants of score matching, NeuralODE training) to reduce memory to standard backprop levels.
- **Task Difficulty Bucketing**: By explicitly stating that "MP1 is near the ceiling on Easy subsets," and splitting gains across Easy/Medium/Hard/Very Hard, the authors clarify the "applicability boundary" of the method, a practice highly recommended for nearly-saturated benchmarks.

## Limitations & Future Work
- **Small-scale Real-world Experiments**: Only 3 tasks with 20 trials each (10% granularity) limit statistical strength and lack standard deviations. The +15% on Slip Ring needs larger sample sizes for support.
- **Hyperparameter $\epsilon$ in DDE**: $\epsilon$ requires manual tuning; an adaptive scheme based on trajectory curvature or second-order trust regions would be a natural next step.
- **Lack of Detailed Comparison with Distillation Methods**: Baselines are mostly contemporaries (MP1/FlowPolicy). A strong baseline like a well-distilled DP3 is missing.
- **Effect on Multimodal Action Distributions**: Directional alignment might force a single mode in tasks where multiple directions are valid. The impact on mode coverage in dexterous tasks remains undiscussed.
- **Scalability with Data**: Only 10 demos were used; whether the benefit of $\mathcal{L}_{DA}$ diminishes as demos increase (e.g., 100+ demos) was not scanned.

## Related Work & Insights
- **vs MP1 (AAAI'26)**: MP1 first brought MeanFlow to robotics with 6.8 ms one-step inference but used MSE + Dispersive objectives. OMP’s gains are built on top of MP1, proving $\mathcal{L}_{DA}$ is the "missing piece" for handling directional gradients in low-speed zones.
- **vs FlowPolicy (AAAI'25)**: FlowPolicy uses consistency flow matching but requires piecewise linear flows and explicit constraints. OMP avoids these, performing direction correction directly on the MeanFlow Identity, which is cleaner and yields +10.7% performance.
- **vs DP3 (RSS'24)**: DP3 uses NFE=10 to reach 68.7% average. OMP reaches 82.3% with NFE=1, showing that "one-step" and "high quality" are not mutually exclusive if geometric/spectral target signals are corrected.
- **vs Consistency Policy / OneDP**: These rely on distillation from a diffusion teacher. OMP is trained from scratch. The cost is handling second-order derivatives of the MeanFlow Identity, which DDE addresses.
- **Transferable Insights**: (a) Log-cosine loss is applicable to any regression where high-precision direction is needed despite near-zero target norms (e.g., SLAM pose estimation, haptic force feedback). (b) DDE can be generalized to higher-order score matching variants. (c) Bucketing performance by task difficulty should be standard for robot learning benchmarks to prevent saturation on Easy tasks from diluting real improvements.

## Rating
- Novelty: ⭐⭐⭐⭐ While cosine loss and central difference are not new, combining them through spectral/geometric/AD analysis into a "MeanFlow fix kit for robotics" is a genuine insight.
- Experimental Thoroughness: ⭐⭐⭐⭐ 37 simulation tasks × 3 seeds + 3 real-world tasks + full ablation + VRAM tables, though real-world stats are limited.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical analysis (§4.2) and well-motivated sections. Difficulty bucketing in the main results is a highlight.
- Value: ⭐⭐⭐⭐ Provides a template for fixing MeanFlow in low-dimensional action spaces; $\mathcal{L}_{DA}$ and DDE can be independently reused in other one-step generative frameworks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Riemannian MeanFlow for One-Step Generation on Manifolds](riemannian_meanflow_for_one-step_generation_on_manifolds.md)
- [\[AAAI 2026\] MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation](../../AAAI2026/image_generation/mp1_meanflow_tames_policy_learning_in_1-step_for_robotic_manipulation.md)
- [\[CVPR 2026\] Temporal Equilibrium MeanFlow: Bridging the Scale Gap for One-Step Generation](../../CVPR2026/image_generation/temporal_equilibrium_meanflow_bridging_the_scale_gap_for_one-step_generation.md)
- [\[CVPR 2026\] MeanFlow Transformers with Representation Autoencoders](../../CVPR2026/image_generation/meanflow_transformers_with_representation_autoencoders.md)
- [\[CVPR 2026\] Understanding, Accelerating, and Improving MeanFlow Training](../../CVPR2026/image_generation/understanding_accelerating_and_improving_meanflow_training.md)

</div>

<!-- RELATED:END -->
