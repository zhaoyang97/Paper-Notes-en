---
title: >-
  [Paper Note] OMP: One-step Meanflow Policy with Directional Alignment
description: >-
  [ICML2026][Image Generation][MeanFlow] Addressing three theoretical pathologies when applying the MeanFlow paradigm to robotics (spectral bias, gradient starvation in low-speed regions, and nested JVP memory explosion)…
tags:
  - "ICML2026"
  - "Image Generation"
  - "MeanFlow"
  - "One-step Policy"
  - "Directional Alignment"
  - "JVP Finite Difference"
  - "robot manipulation"
date: 2026-05-08
content_hash: b47278053ae907ed
---

# OMP: One-step Meanflow Policy with Directional Alignment

**Conference**: ICML2026  
**arXiv**: [2512.19347](https://arxiv.org/abs/2512.19347)  
**Code**: TBD  
**Area**: Robotics / Embodied AI / Generative Policy  
**Keywords**: MeanFlow, One-step Policy, Directional Alignment, JVP Finite Difference, robot manipulation

## TL;DR
Addressing three theoretical pathologies when applying the MeanFlow paradigm to robotics (spectral bias, gradient starvation in low-speed regions, and nested JVP memory explosion), this paper proposes OMP. It uses a cosine-style directional alignment loss to "lock" the predicted mean velocity direction with the ground truth and employs a finite difference DDE to approximate the Jacobian-Vector Product (JVP) for decoupling forward and backward passes. This allows the one-step (NFE=1) policy to achieve 6.8ms latency with an average success rate 3.4% higher than MP1 on Adroit/Meta-World, and 10.6% higher on Meta-World Very Hard tasks.

## Background & Motivation
**Background**: Current mainstream generative robot policies model action generation as a probabilistic denoising process. Diffusion policies like Diffusion Policy / DP3 achieve high success rates via iterative denoising (approx. 10 steps), but the NFE=10 latency hinders high-frequency closed-loop control. To accelerate, methods based on flow matching or consistency distillation (e.g., FlowPolicy, ManiFlow) compress inference to a single step, but rely on piecewise linear flows or explicit consistency constraints, where overly rigid architectural constraints sacrifice generalization.

**Limitations of Prior Work**: MeanFlow (2025) theoretically provides a cleaner "single-step" path by directly learning the interval mean velocity $u(z_t, r, t)$, bypassing ODE solvers. Its robotics implementation, MP1, reduced latency to 6.8 ms. However, the authors found that directly applying MeanFlow to robotics reveals three pathologies not apparent in image generation scenarios.

**Key Challenge**: In image generation, the large pixel-level dynamic range and sufficient gradient signals mask the spectral and geometric defects of the MeanFlow objective. Conversely, robot action spaces have low dimensionality, and in fine-grained tasks, the ground truth mean velocity $\|v_0\|$ approaches 0, leading to three theoretical issues: (1) **Spectral Bias**—time integration is equivalent to dividing by $i\omega$, causing the PSD target to decay by $1/\omega^2$, acting as a low-pass filter that suppresses high-frequency directional adjustments in fine manipulation; (2) **Gradient Starvation**—the MSE loss gradient with respect to the angular error is $2\rho\rho^*\sin\alpha$, which is multiplicatively coupled with the target magnitude $\rho^*$; when $\rho^*\to 0$, the model cenderung to shrink its output to 0 rather than aligning the direction; (3) **Memory Complexity**—the total derivative in the MeanFlow Identity involves a JVP $\nabla_z u\cdot dz/dt$, and computing $\nabla_\theta$ is equivalent to nested Forward-AD + Reverse-AD, requiring simultaneous storage of primal, tangent, and adjoint activations, making it untrainable for large point cloud backbones.

**Goal**: (a) Remove the strong coupling of direction and magnitude in MSE, ensuring directional supervision does not vanish in low-speed regions; (b) Replace JVP with an approximation that does not require symbolic differentiation, reducing training memory to standard backprop levels; (c) Maintain NFE=1 inference speed.

**Key Insight**: Since the root causes are the coupling of direction/magnitude in MSE and the analytical expansion of JVP exploding memory, the solution is to bypass them—using a cosine term to isolate direction as an independent loss and using central difference to approximate the time derivative.

**Core Idea**: Explicitly lock the "pointing" of the predicted mean velocity to the ground truth mean velocity $v_0$ using a directional alignment loss, combined with an $O(\epsilon)$ central difference to replace JVP and decouple forward/backward passes.

## Method

### Overall Architecture
OMP is built on the MeanFlow framework: taking 3D point cloud observations (downsampled to 512 or 1024 points via FPS) + 2-step observation history as input, it outputs an action sequence of length 4, executing 3 steps at a time. During training, the model $u_\theta(z_t, r, t \mid c)$ learns the mean velocity between times $r$ and $t$, following the MeanFlow Identity:

$$u(z_t,r,t|c)=v(z_t,t|c)-(t-r)\dfrac{d}{dt}u(z_t,r,t|c)$$

where the RHS is treated as the target $u_{tgt}$. For inference, a single forward pass goes directly from noise $z_T\sim\mathcal{N}(0,I)$ to action $z_0$, using $v_0 \triangleq z_T - z_0$ as the ground truth mean velocity. OMP adds a third term $\mathcal{L}_{DA}$ for directional alignment to MP1's $\mathcal{L}_{mse} + \lambda_{Disp}\mathcal{L}_{Disp}$ and replaces $\frac{d}{dt}u$ in the Identity with DDE central difference, resulting in two versions: OMP-JVP (preserving analytical JVP) and OMP-DDE (using difference approximation).

### Key Designs

1.  **Directional Alignment Loss $\mathcal{L}_{DA}$**:
    *   **Function**: Explicitly aligns the direction of the predicted mean velocity $u(z_t,r,t|c)$ with the ground truth mean velocity $v_0=z_T-z_0$, resolving gradient starvation and spectral bias of MSE in low $\rho^*$ regions.
    *   **Mechanism**: First calculates cosine similarity $\cos\alpha = \dfrac{v_0\cdot u}{\|v_0\|\cdot\|u\|}$, then uses the logarithmic form $\mathcal{L}_{DA}=-\log\!\big(\frac{\cos\alpha+1}{2}\big)$ as the loss. This form offers several benefits: it diverges at $\cos\alpha=-1$ for maximum penalty, approaches zero at $\cos\alpha=1$, and depends only on direction, not magnitude. Thus, even if $\|v_0\|\to 0$, the directional gradient does not collapse. A small $\epsilon_{dir}\approx 10^{-6}$ is added to the denominator to prevent division by zero.
    *   **Design Motivation**: In §4.2.2, the authors use the law of cosines to decompose MSE, obtaining $\partial\mathcal{L}_{MSE}/\partial\alpha = 2\rho\rho^*\sin\alpha$, proving the angular gradient is suppressed by $\rho^*$. In fine robot manipulation where $\rho^*\approx 0$, MSE encourages $\rho\to 0$, leading to a "static policy." $\mathcal{L}_{DA}$ decouples direction and magnitude; $\mathcal{L}_{mse}$ dominates during the ballistic phase (magnitude), while $\mathcal{L}_{DA}$ dominates during the contact phase (direction). Explicitly aligning $v_0$ also bypasses the $1/\omega^2$ low-pass filter as the target no longer undergoes time integration.

2.  **Differential Derivation Equation (DDE)**:
    *   **Function**: Replaces the analytical $du_\theta/dt$ in MeanFlow Identity with a central difference, avoiding the memory explosion caused by backpropagating through JVP (nested AD).
    *   **Mechanism**: Approximates the time derivative as $\dfrac{du_\theta(z_t,t,r|c)}{dt}\approx\dfrac{u_\theta(z_{t+\epsilon},t+\epsilon,r|c)-u_\theta(z_{t-\epsilon},t-\epsilon,r|c)}{2\epsilon}$, where $\epsilon$ is a small perturbation constant (see §E.2 for sensitivity analysis). This leaves only two standard forward passes and one backward pass in the training graph, eliminating the need to store tangent activations and returning to standard backprop memory scales.
    *   **Design Motivation**: §4.2.3 shows that computing $\nabla_\theta$ for the JVP $\nabla_z u_\theta\cdot v$ is equivalent to the second-order mixed partial derivative $\partial^2 u/\partial\theta\partial z$. In PyTorch/JAX, this requires nesting Forward-AD outside Reverse-AD, necessitating the storage of primal $X$, tangent $\delta X$, and tangent adjoint graphs—unfeasible for point cloud backbones like PointNet++/Transformer on a 4090. The difference approximation has an $O(\epsilon^2)$ truncation error, which has a controllable impact on success rates while providing significant memory benefits.

3.  **Combined Loss & JVP/DDE Versions**:
    *   **Function**: Merges direction, magnitude, and feature discriminability signals into one training objective while making memory optimization an toggleable option.
    *   **Mechanism**: The final training objective is $\mathcal{L}=\mathcal{L}_{mse}+\lambda_{Disp}\mathcal{L}_{Disp}+\lambda_{DA}\mathcal{L}_{DA}$, where $\mathcal{L}_{Disp}$ follows MP1's dispersive loss for feature space separability. Two implementations of $\frac{d}{dt}u$ are provided: OMP-JVP (preserving analytical JVP for peak accuracy) and OMP-DDE (using DDE for VRAM efficiency), allowing users to switch based on task scale (point cloud size, action horizon).
    *   **Design Motivation**: Ballistic (large translation) and fine contact (angular alignment) stages have different loss requirements; a weighted sum ensures non-zero gradients in both. The JVP/DDE dual version acknowledges the "memory-accuracy" trade-off, providing OMP-JVP for academic comparison and OMP-DDE for practical deployment.

### Loss & Training
*   **Loss**: $\mathcal{L}=\mathcal{L}_{mse}+\lambda_{Disp}\mathcal{L}_{Disp}+\lambda_{DA}\mathcal{L}_{DA}$; $\mathcal{L}_{DA}=-\log\!\big(\frac{\cos\alpha+1}{2}\big)$; DDE time step $\epsilon$ is swept in §E.2.
*   **Data**: 10 expert demonstrations per simulation task; point cloud FPS to 512 or 1024 points; images 84×84; observation history=2, prediction horizon=4, execution horizon=3.
*   **Training**: AdamW, lr=1e-4, batch=128; Adroit for 3000 epochs, Meta-World for 1000 epochs, evaluated every 200 epochs. Final success rate is the average of top 5 peaks, then averaged across seeds (0/10/20); hardware: single RTX 4090.

## Key Experimental Results

### Main Results: Adroit + Meta-World 37-Task Average

| Method | NFE | Adroit Pen | MW Medium | MW Hard | MW Very Hard | Total Avg. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DP (RSS'23) | 10 | 13±2 | 11.0±2.5 | 5.25±2.5 | 22.0±5.0 | 35.2±5.3 |
| DP3 (RSS'24) | 10 | 46±10 | 44.5±8.7 | 32.7±7.7 | 39.4±9.0 | 68.7±4.7 |
| FlowPolicy (AAAI'25) | 1 | 54±4 | 58.2±7.9 | 40.2±4.5 | 52.2±5.0 | 71.6±3.5 |
| MP1 (AAAI'26) | 1 | 58±5 | 68.0±3.1 | 58.1±5.0 | 67.2±2.7 | 78.9±2.1 |
| **OMP-JVP** | 1 | **60±4** | **77.4±2.2** | **62.5±3.1** | **77.8±3.0** | **82.3±1.6** |
| OMP-DDE | 1 | 64±3 | 76.4±2.7 | 61.0±3.0 | 70.6±4.9 | 80.8±2.2 |

OMP-JVP is 3.4% higher than MP1 and 10.7% higher than FlowPolicy in total average. Gains are larger for harder tasks—Meta-World Medium +9.4%, Very Hard +10.6%. MP1 already approaches the ceiling in the Easy subset (21/37 tasks, 88%+), which dilutes the absolute total gain to 1.5%.

### Real-world Experiments (3 Tasks, Success % )

| Method | Place | Clean | Slip Ring |
| :--- | :--- | :--- | :--- |
| DP3 | 65 | 60 | 50 |
| FlowPolicy | 60 | 50 | 40 |
| MP1 | 70 | 65 | 55 |
| **OMP** | **80** | **75** | **70** |

On the most difficult task, Slip Ring, OMP outperforms MP1 by 15%, validating the core benefit of directional alignment in "real-world low-speed fine manipulation."

### Ablation Study & Memory

| Configuration | Total Avg. Success | Note |
| :--- | :--- | :--- |
| OMP-JVP (Full) | 82.3 | Full model |
| − $\mathcal{L}_{Disp}$ | 81.2 | Removed dispersive, -1.1% (minor) |
| − $\mathcal{L}_{DA}$ | 78.9 | Removed dir. align, -3.4%, back to MP1 level |
| − $\mathcal{L}_{Disp}$ − $\mathcal{L}_{DA}$ | 78.3 | Both removed, Adroit Pen 60→48 |
| OMP-DDE (Full) | 80.8 | Finite difference approx version |
| − $\mathcal{L}_{DA}$ (DDE) | 77.2 | Validates dir. align is core |

| Task / Horizon | OMP-JVP VRAM | OMP-DDE VRAM |
| :--- | :--- | :--- |
| Adroit Hammer / H=4 | 6.60 GB | 5.35 GB |
| Place Bottle / H=4 | 23.49 GB | 18.33 GB |
| Adroit Hammer / H=16 | 7.69 GB | 6.12 GB |
| Place Bottle / H=16 | **26.71 GB** | **19.19 GB** |

### Key Findings
*   **Directional alignment is the main contributor**: Removing $\mathcal{L}_{DA}$ causes a 3.4–3.6% drop, while removing $\mathcal{L}_{Disp}$ only drops 0.7–1.1%, proving the issue lies in MSE's geometric coupling rather than feature discriminability.
*   **OMP gain correlates positively with task difficulty**: On Easy tasks, MP1 reaches the ceiling (88%+), so OMP shows little difference; the +10.6% on Very Hard tasks confirms that directional alignment primarily aids low-speed fine tasks, as theorized.
*   **JVP to DDE is an accuracy-memory trade-off**: DDE drops an average of 1.5% (with a larger 7.2% drop on Very Hard) in exchange for a 28% VRAM reduction (26.71 GB→19.19 GB) on Place Bottle/H=16. The larger the point cloud or horizon, the better the value proposition of DDE.
*   **Stabler training curves**: Figure 5 shows that OMP's success rate curve variance is much smaller than the violent oscillations seen in FlowPolicy/MP1, suggesting directional alignment enhances training stability.

## Highlights & Insights
*   **Unified theoretical narrative**: The authors do not just drop a cosine loss; they integrate PSD frequency analysis, law of cosines derivation for MSE angular gradients, and AD graph analysis of activations. This weaves three separate issues into one convincing story: "MeanFlow is unsuitable for robotics," giving strong motivation for $\mathcal{L}_{DA}$ + DDE. The paper's strength lies in these analyses rather than just the numbers.
*   **Logarithmic form of cosine loss**: Using $-\log((\cos\alpha+1)/2)$ instead of $1-\cos\alpha$ is a reusable trick—the logarithm makes gradients diverge near $\cos\alpha=-1$, heavily punishing "moving in the exact opposite direction," whereas $1-\cos\alpha$ has minimal gradients during reversal, potentially trapping the model in local optima.
*   **Value of DDE for decoupling computation graphs**: Reusable across tasks—any scenario requiring $du_\theta/dt$ backprop through $\theta$ (not just MeanFlow, but second-order score matching variants or NeuralODE training) can use this to reduce memory to standard backprop levels.
*   **Difficulty-binned task reporting**: By explicitly noting that "MP1 is near the ceiling in the Easy subset" and splitting gains across Easy/Medium/Hard/Very Hard, the applicability boundaries of the method are clear—a practice worth adopting by any work where benchmarks are nearing saturation.

## Limitations & Future Work
*   **Scale of real-world experiments**: Only 3 tasks with 20 trials each (10% granularity) provide limited statistical strength and no standard deviation. The +15% on Slip Ring needs larger sample support.
*   **Hyperparameter $\epsilon$ for DDE**: Sensitivity analysis is provided in the appendix, but no adaptive scheme. Different tasks may require retuning; an ideal next step is choosing $\epsilon$ via trajectory curvature or second-order trust regions.
*   **Lack of detailed comparison with recent distillation methods**: Baselines are mostly contemporaries (MP1/FlowPolicy). A strong baseline like a well-distilled DP3 (multi-step teacher + consistency distillation) is missing, which theoretically could match OMP's accuracy.
*   **Multimodality impact**: Directional alignment for multimodal distributions isn't discussed. Fine manipulation might have multiple equivalent directions (left/right hand); whether forcing a single direction via cosine loss loses mode diversity deserves study via mode coverage metrics in dexterous tasks.
*   **Lack of data scale scanning**: Only 10 expert demos are used. The impact of increasing demos (10→100) on the $\mathcal{L}_{DA}$ benefit is not provided.

## Related Work & Insights
*   **vs MP1 (AAAI'26)**: MP1 pioneered MeanFlow in robotics with 6.8ms single-step inference but used MSE + Dispersive objectives. OMP's gains are built on top of MP1, proving directional alignment is the "missing piece" for low-speed directional gradients.
*   **vs FlowPolicy (AAAI'25)**: FlowPolicy uses consistency flow matching but requires piecewise linear flows and explicit constraints. OMP avoids these, performing directional correction directly on MeanFlow Identity—cleaner engineering and +10.7% performance.
*   **vs DP3 (RSS'24)**: DP3 reaches 68.7% average via NFE=10 denoising; OMP reaches 82.3% via NFE=1. This shows "single-step" and "high quality" are no longer mutually exclusive if the geometric/spectral issues of the objective signal are resolved.
*   **vs Consistency Policy / OneDP**: These rely on consistency distillation from multi-step diffusion teachers. OMP is trained from scratch. The cost is handling the second-order derivatives of MeanFlow Identity, which DDE resolves.
*   **Transferable Insights**: (a) Log-cosine loss is applicable to any regression where the target norm is near-zero but direction remains critical (e.g., small displacement pose estimation in SLAM). (b) DDE can be generalized to higher-order score matching variants where Fisher trace training is memory-limited. (c) Binning performance by task difficulty should be standard to avoid saturation in Easy tasks diluting real improvements.

## Rating
*   Novelty: ⭐⭐⭐⭐ While cosine loss and central difference are standard math, the insight of packaging them as a "MeanFlow fix kit for robotics" via spectral/geometric/AD analysis is genuine.
*   Experimental Thoroughness: ⭐⭐⭐⭐ 37 simulation tasks × 3 seeds + 3 real-world tasks + full ablation + memory tables, though real-world stats are limited and distilled single-step baselines are missing.
*   Writing Quality: ⭐⭐⭐⭐ Theoretical analyses (§4.2) are well-structured and motivated. Presentation by task difficulty is a highlight.
*   Value: ⭐⭐⭐⭐ Provides a "MeanFlow repair template" for low-dimensional action spaces. Both the directional alignment loss and DDE can be independently integrated into other one-step generative frameworks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation](../../AAAI2026/image_generation/mp1_meanflow_tames_policy_learning_in_1-step_for_robotic_manipulation.md)
- [\[CVPR 2026\] Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation](../../CVPR2026/image_generation/emf_meanflow_text_to_image.md)
- [\[CVPR 2026\] Taming Preference Mode Collapse via Directional Decoupling Alignment in Diffusion Reinforcement Learning](../../CVPR2026/image_generation/taming_preference_mode_collapse_via_directional_decoupling_alignment_in_diffusio.md)
- [\[ICML 2026\] Principled RL for Flow Matching Emerges from the Chunk-level Policy Optimization](principled_rl_for_flow_matching_emerges_from_the_chunk-level_policy_optimization.md)
- [\[ICML 2026\] Discrete Diffusion Samplers and Bridges: Off-Policy Algorithms and Applications in Latent Spaces](discrete_diffusion_samplers_and_bridges_off-policy_algorithms_and_applications_i.md)

</div>

<!-- RELATED:END -->
