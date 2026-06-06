---
title: >-
  [Paper Note] SGMD: Score Gradient Matching Distillation for Few-Step Video Diffusion
description: >-
  [ICML 2026][Video Generation][Diffusion model distillation] SGMD introduces a **stable teacher stop-gradient Fisher objective** and a **dual-potential (NR/RC) mechanism**. It addresses the high cost of fake score trackin…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Diffusion model distillation"
  - "Score matching"
  - "few-step generation"
  - "motion preservation"
date: 2026-05-08
content_hash: 10f9cbc51d68d4e5
---

# SGMD: Score Gradient Matching Distillation for Few-Step Video Diffusion

**Conference**: ICML 2026  
**arXiv**: [2605.30116](https://arxiv.org/abs/2605.30116)  
**Code**: To be confirmed  
**Area**: Video Generation / Diffusion Model Distillation  
**Keywords**: Diffusion model distillation, Score matching, few-step generation, motion preservation

## TL;DR
SGMD introduces a **stable teacher stop-gradient Fisher objective** and a **dual-potential (NR/RC) mechanism**. It addresses the high cost of fake score tracking (5 updates per round in DMD2) and motion suppression issues in few-step video diffusion distillation. It achieves ~3× training speedup while improving motion quality from 0.65 to 0.78 (VideoAlign) under 4-step distillation.

## Background & Motivation

**Background**: Diffusion models exhibit superior performance in video generation but suffer from high inference costs (large parameters, high latent dimensions, multi-step sampling). The Distribution Matching Distillation (DMD) series is the mainstream few-step acceleration solution, compressing sampling steps by matching student and teacher generation distributions.

**Limitations of Prior Work**: DMD-style distillation faces two coupled challenges in aggressive few-step scenarios:
- **Tracking Cost**: The auxiliary score network (fake score) on the student side must track an evolving generator. Maintaining tracking consistency requires multiple fake score updates (DMD2 requires 5 per iteration).
- **Motion Suppression**: Reverse-KL style matching is **mode-seeking and conservative**, tending to avoid low-density regions in the target distribution. In few-step distillation, this leads to insufficient motion and loss of detail.

**Key Challenge**: How to maintain consistency for distribution matching while reducing fake score tracking overhead and preventing motion dynamic suppression? These two problems appear mutually restrictive.

**Goal**:
1. Identify a distribution matching objective that is consistent with DMD under ideal tracking conditions but more stable.
2. Design a lightweight tracking mechanism to reduce fake score update frequency.
3. Simultaneously maintain motion intensity and visual quality in few-step video distillation.

**Key Insight**: Rethink distillation from the fake-score perspective—instead of treating the fake score solely as a tracking tool, view it as the primary optimization target that should move towards the teacher. Meanwhile, let the generator act as a tracker to maintain score consistency. This role reversal breaks the cyclic dependency.

**Core Idea**: Use teacher stop-gradient Fisher instead of reverse-KL as the distribution matching objective for smoother gradient signals. Introduce dual-potential pairs (NR/RC) to decouple the tracking problem into outer correction and inner contraction, enabling two-step bilevel optimization with only 1 fake score update.

## Method

### Overall Architecture
Two alternating stages (see Algorithm 1):
1. **Generator Update Stage** (detach fake score): Optimize $\mathcal{L}_{\text{Fisher}}(\theta) + \lambda \mathcal{L}_{\text{NR}}(\theta)$—Fisher provides the primary gradient for distribution matching, while NR (negative residual) corrects the outer optimization direction.
2. **Fake-score Update Stage** (detach generator): Optimize $\lambda \mathcal{L}_{\text{RC}}(\psi)$—RC (residual contraction) contracts the tracking residual to maintain compatibility between the fake score and the generated distribution.

The overall approach forms a **lightweight two-step bilevel update**, with only 1 fake score backpropagation per round.

### Key Designs

1. **Teacher Stop-Gradient Fisher Objective**:
    - Function: Provides stable distribution matching signals and avoids the conservatism of reverse-KL.
    - Mechanism: Standard score matching backpropagates teacher gradients through generated samples (OOD states), leading to instability. SGMD freezes the teacher input gradient (stop-gradient), letting the fake score track the teacher score difference: $\mathcal{L}_{\text{Fisher}}(\theta, \psi) := \frac{1}{2} \|s_{\text{fake}}(x_t, t) - s_{\text{real}}(\text{sg}[x_t], t)\|^2 = \frac{1}{2} c(t) \|\Delta_t\|^2$, where $\Delta_t = \mu_\psi(x_t, t) - \mu_{\text{base}}(x_t, t)$ and $c(t) = \alpha_t^2 / \sigma_t^4$. Under ideal tracking (Proposition 3.1), this aligns with the reverse-KL distribution matching direction.
    - Design Motivation: Fisher provides smoother matching signals (unlike reverse-KL, which avoids low-density areas); the stop-gradient operation eliminates ineffective backpropagation of OOD gradients.

2. **Dual-Potential (NR / RC) Mechanism**:
    - Function: Decouples the tracking problem into collaborative alignment between the generator and fake score.
    - Mechanism: Define tracking residual $r(x_0, x_t) := \text{sg}[x_0] - x_{\text{fake}}$ to measure the gap between generator output and fake score prediction. Construct dual potentials with opposite signs: $\mathcal{L}_{\text{NR}}(\theta) := -\frac{1}{2} \|r\|^2$ (outer negative residual) and $\mathcal{L}_{\text{RC}}(\psi) := +\frac{1}{2} \|r\|^2$ (inner residual contraction). In $x_{\text{fake}}$ space, they induce opposite gradients $\nabla_{x_{\text{fake}}} \mathcal{L}_{\text{NR}} = r$ and $\nabla_{x_{\text{fake}}} \mathcal{L}_{\text{RC}} = -r$. Through the chain $x_0 \to x_t \to x_{\text{fake}}$, NR pushes the generator output towards the fake score prediction, while RC pulls the fake score towards the generator output.
    - Design Motivation: Methods like SIM use implicit gradients to handle tracking precisely but are computationally complex with fixed parameters. SGMD simplifies this with explicit dual potentials—the outer NR corrects the generator gradient direction (eliminating tracking lag curvature), and the inner RC stabilizes fake score updates.

3. **Lightweight Two-step Bilevel Update**:
    - Function: Achieves stable two-time-scale optimization with minimal computational cost.
    - Mechanism: Two backpropagations per iteration—first update generator $\theta \leftarrow \theta - \eta_\theta \nabla_\theta(\mathcal{L}_{\text{Fisher}} + \lambda \mathcal{L}_{\text{NR}})$ (with $\psi$ detached), then update fake score $\psi \leftarrow \psi - \eta_\psi \nabla_\psi(\lambda \mathcal{L}_{\text{RC}})$ (with $\theta$ detached). This forms an explicit single-step bilevel iteration, avoiding second-order implicit gradient calculations.
    - Design Motivation: Compared to 5 fake score updates per round in DMD2, SGMD uses only 1, reaching ~3× training speedup on 32 H100 GPUs. The explicit design is easier to debug and understand than SIM's implicit gradients. $\lambda = 0.1$ is experimentally optimal.

### Loss & Training
The PPO algorithm treats multi-turn trajectories as single token sequences; inter-turn discount $\gamma_{\text{turn}} = 0.95$, intra-turn $\gamma_{\text{token}} = 1.0$. AdamW with $\beta_1 = 0, \beta_2 = 0.999$, learning rate $1 \times 10^{-6}$; fixed 4-step timesteps $\{1000, 960, 889, 727\}$, Euler solver.

## Key Experimental Results

### Main Results (4-step distillation on Wan2.1-T2V-14B teacher)

| Method | NFE | Fake-R | FVD ↓ | OptFlow ↑ | VBench-Quality ↑ | VBench-Semantic ↑ | DynDeg ↑ |
|--------|-----|--------|------|---------|-----------|-----------|---------|
| Base Model (50 steps) | 100 | — | 0.0 | 9.41 | 86.67 | 84.44 | 94.26 |
| DMD2 | 4 | 5 | 115.1 | 4.51 | 85.05 | 77.46 | 80.56 |
| TSG-Fisher | 4 | 5 | 126.7 | 8.18 | 82.98 | 71.50 | 94.25 |
| TSG-SIM | 4 | 1 | 193.0 | 3.27 | 82.68 | 73.21 | 59.72 |
| **SGMD** | **4** | **1** | **100.3** | **9.29** | **84.77** | **75.64** | **93.06** |

SGMD achieves motion intensity close to the base model (OptFlow 9.29 vs 9.41) with only 1 fake score update; motion metrics significantly outperform DMD2 (OptFlow +106%, DynDeg +15.5%); VBench quality remains competitive.

### Ablation Study

| $\lambda$ | Total ↑ | Quality ↑ | Semantic ↑ | DynDeg ↑ |
|----------|---------|-------|-------|---------|
| 0.05 | 81.92 | 83.68 | 74.90 | 93.55 |
| **0.1** | **82.95** | **84.77** | **75.64** | **93.06** |
| 0.2 | 82.01 | 84.06 | 73.81 | 94.23 |
| 0.5 | 79.54 | 81.49 | 71.75 | 76.52 |

### Key Findings
- **Motion-Quality Trade-off**: Fisher tends to preserve motion details in low-density areas (smoother gradient signals), whereas reverse-KL (DMD2) conservatively concentrates on high-probability regions, leading to motion suppression. SGMD gains strong motion via Fisher and restores visual quality via stable NR/RC tracking.
- **Training Efficiency**: Fake score updates reduced from 5 to 1 per iteration, reducing gradient calculations by 80% per backpropagation; measured ~3× speedup on 32 H100 GPUs.
- **Human Evaluation Preferences**: Overall preference for SGMD 65% vs DMD2 13%; Motion quality 71% win rate for SGMD; text alignment and visual quality mostly rated as ties.
- **VideoAlign Evaluation**: Total score SGMD 19.36 > DMD2 18.86; Motion quality 4.99 > 4.15 (+20%); Visual quality 8.19 ≈ DMD2 8.47 (-2%).

## Highlights & Insights
- **Stable Distribution Matching Objective**: Teacher stop-gradient Fisher cleverly avoids OOD gradients while remaining equivalent to reverse-KL under ideal conditions (Proposition 3.1); applicable to all diffusion distillation scenarios.
- **Reshaping the Fake-score Perspective**: Redefines the fake score from a "tracker" to an "optimization target," and the generator becomes the "tracker"—this role reversal eliminates cyclic dependency and is transferable to other two-time-scale dual optimization problems.
- **Geometric Intuition of Dual Potentials**: NR and RC induce opposite gradients in $x_{\text{fake}}$ space to form a "push-pull" mechanism; simpler to implement and debug than SIM's implicit gradients.
- **Path to Efficient Distillation**: Achieves 3× speedup by reducing fake score updates (from 5 to 1) while maintaining or improving performance.

## Limitations & Future Work
- Tracking weight $\lambda$ is manually tuned; different teachers or distillation goals might require a new search. Adaptive $\lambda$ warrants investigation.
- Experiments focused on 4-step distillation; the effectiveness of SGMD in more aggressive 1-2 step scenarios needs further verification.
- Motion-Clarity Trade-off: While human evaluations show the trade-off is acceptable, clarity-sensitive applications (product videos, medical) may still require adjustments.
- Results are based on a single teacher (Wan2.1-T2V-14B); adaptability to other video models (e.g., I2V-Turbo) needs validation.

## Related Work & Insights
- **vs. DMD2**: Both use distribution matching, but DMD2 uses reverse-KL + frequent fake score updates; SGMD uses Fisher + dual potentials—avoiding reverse-KL conservatism while significantly reducing update frequency.
- **vs. SIM**: Both focus on the implicit term of tracking lag, but SIM uses implicit gradients (complex), whereas SGMD uses explicit dual potentials (concise); SIM relative strength is fixed, while SGMD allows tuning of $\lambda$.
- **vs. Flash-DMD**: The latter focuses on timestep-aware training and RL, which are orthogonal optimization directions that could be combined with SGMD.
- **Insights**: The fake-score perspective and dual-potential mechanism are transferable to image diffusion distillation and two-time-scale optimization in other generative models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  Innovation in fake-score perspective, elegant dual-potential design, deep theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐  Large-scale 14B model + multi-dimensional evaluation (VBench + FVD + OptFlow + Human + VideoAlign) + Ablations.
- Writing Quality: ⭐⭐⭐⭐  Clear logic, intuitive gradient analysis; length could be more compact.
- Value: ⭐⭐⭐⭐⭐  3× training speedup + significant motion improvement, directly reducing video distillation costs; points towards efficient distillation for future large models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis](../../ICCV2025/video_generation/adversarial_distribution_matching_for_diffusion_distillation_towards_efficient_i.md)
- [\[ICML 2026\] AAD-1: Asymmetric Adversarial Distillation for One-Step Autoregressive Video Generation](aad-1_asymmetric_adversarial_distillation_for_one-step_autoregressive_video_gene.md)
- [\[ICCV 2025\] DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization](../../ICCV2025/video_generation/dollar_fewstep_video_generation_via_distillation_and_latent.md)
- [\[CVPR 2026\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](../../CVPR2026/video_generation/flashmotion_fewstep_controllable_video_generation.md)
- [\[AAAI 2026\] Phased One-Step Adversarial Equilibrium for Video Diffusion Models](../../AAAI2026/video_generation/phased_one-step_adversarial_equilibrium_for_video_diffusion_models.md)

</div>

<!-- RELATED:END -->
