---
title: >-
  [Paper Note] Any-step Generation via N-th Order Recursive Consistent Velocity Field Estimation
description: >-
  [ICLR 2026][Image Generation][Consistency Models] This paper proposes RCGM, which unifies few-step generation methods such as consistency models, MeanFlow, and shortcuts as 1st-order special cases of "N-th Order Recursive Velocity Field Estimation." By extending to 2nd-order and higher, the high-order targets avoid expensive JVPs and remain compatible with aggressive EMA smoothing. This enables the stable expansion of few-step generation training to 20B large models…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Consistency Models"
  - "MeanFlow"
  - "Flow Matching"
  - "Few-step Sampling"
  - "High-order Recursion"
  - "EMA Stability"
  - "Diffusion Transformer"
date: 2026-05-08
content_hash: 30966d1c0b0b553f
---

# Any-step Generation via N-th Order Recursive Consistent Velocity Field Estimation

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=GnawtLKGkP](https://openreview.net/forum?id=GnawtLKGkP)  
**Code**: [https://github.com/LINs-lab/RCGM](https://github.com/LINs-lab/RCGM)  
**Area**: Image Generation / Few-step Generative Models  
**Keywords**: Consistency Models, MeanFlow, Flow Matching, Few-step Sampling, High-order Recursion, EMA Stability, Diffusion Transformer  

## TL;DR
This paper proposes RCGM, which unifies few-step generation methods such as consistency models, MeanFlow, and shortcuts as 1st-order special cases of "N-th Order Recursive Velocity Field Estimation." By extending to 2nd-order and higher, the high-order targets avoid expensive JVPs and remain compatible with aggressive EMA smoothing. This enables the stable expansion of few-step generation training to 20B large models, achieving 1.48 FID in 2 steps on ImageNet 256×256.

## Background & Motivation
**Background**: Few-step generative models (typically 1–8 steps), represented by Consistency Models (CM/sCM), shortcut, and MeanFlow, can generate high-fidelity samples with minimal sampling overhead, becoming the mainstream for deployment-friendly directions. Their shared approach is teaching the model to jump from any noise state to data endpoints in one step, compressing multi-step diffusion trajectories into one or several forwards via "self-supervised consistency."

**Limitations of Prior Work**: The authors point out that current SOTA few-step methods are hindered by three issues: (a) training requires expensive Jacobian-Vector Products (JVP), leading to massive VRAM and compute overhead while being incompatible with architectural optimizations like Flash-Attention; (b) multiple losses or auxiliary models (e.g., consistency loss plus adversarial loss, or extra fake image generators) must be stacked, breaking end-to-end simplicity; (c) theoretical fragmentation—highly related methods like CM, shortcut, and MeanFlow develop independently without a common foundation.

**Key Challenge**: The authors attribute these vulnerabilities to an overlooked essence—these methods are inherently "**1st-order recursive training objectives**." 1st-order recursion naturally conflicts with EMA (Exponential Moving Average): self-supervised learning should rely on high EMA decay rates (e.g., $\kappa=0.999$) to provide smooth, stable regression targets, but 1st-order recursion degrades severely under high $\kappa$ (e.g., MMD on the Moons dataset surges from 0.0066 to 0.2131; ImageNet FID collapses to 294), forming the "**EMA Incompatibility Paradox**": large $\kappa$ is stable but underperforms, while small $\kappa$ does not collapse but provides no gain. This makes few-step training prone to collapse or VRAM explosion when scaled to large models or high learning rates.

**Goal**: To build a unified and concise framework that incorporates existing few-step methods as special cases while breaking the 1st-order limitation. This aims to unlock stabilization techniques like EMA, discard JVPs and auxiliary models, and allow few-step generation to scale stably to large-scale models.

**Core Idea**: **[Elevating from 1st-order to N-th order recursion]** By performing **piecewise integration** on PF-ODE trajectories, the learning objective for instantaneous velocity is generalized from "one-step approximation + one future segment" to "one-step approximation + N future integral correction segments." The N-th order objective constructs stable training signals using more complete trajectory information, resulting in both stability and strength under high EMA without increasing VRAM.

## Method

### Overall Architecture
The core of RCGM is the unification of generative models as "N-th order recursive consistent velocity field estimation." Starting from the exact integral identity of the PF-ODE, the trajectory from $x_t$ to $x_{t_{N+1}}$ is segmented using $N$ intermediate points. The first segment uses a 1st-order Euler approximation, while the remaining segments are retained as integral correction terms. Thus, the instantaneous velocity $\mathrm{d}x_t/\mathrm{d}t$ has a target composed of the "total displacement minus N future displacement segments." Parametrizing this as a displacement function $f_\theta(x_t, r)$ (predicting displacement from $x_t$ to any future moment $r$), where future segments are estimated by an EMA target model with stop-gradients, yields a unified training objective requiring only "1 gradient forward + N non-gradient forwards." When $N=0$, it reduces to diffusion/flow matching; $N=1$ reduces to consistency/shortcut/MeanFlow; and $N \ge 2$ represents the high-order method proposed.

```mermaid
flowchart LR
    A[PF-ODE Trajectory<br/>x_t → x_tN+1] --> B[Piecewise Integration<br/>N intermediate points split into N+1 segments]
    B --> C[1st segment Euler approx.<br/>-v·Δt]
    B --> D[Remaining N segments kept as<br/>integral correction terms]
    C --> E[N-th order velocity target<br/>Total displacement - Σ Future segments]
    D --> E
    E --> F[Displacement Parameterization<br/>f_θ=F_θ·(t-r)]
    F --> G[EMA Target Model<br/>Provides future segments + stop-grad]
    G --> H[Variance Reduction Loss<br/>Decoupled time-step scale]
    H --> I[Any-step Generation<br/>N=0/1/≥2 Unified]
```

### Key Designs

**1. Deriving N-th order recursive objective via piecewise integration: Correcting one-step approximation with future integrals.** This is the theoretical fulcrum. PF-ODE gives the exact displacement between any two points $x_{t_{N+1}}-x_t=\sum_{i=0}^{N}\int_{t_i}^{t_{i+1}} v(x_\tau,\tau)\,\mathrm{d}\tau$. The authors approximate only the first segment (step size $\Delta t=t_0-t_1$ small enough) using 1st-order Taylor/Euler as $-v(x_{t_0},t_0)\Delta t$, keeping the other $N$ segments as is. Rearranging gives the "N-th order recursive estimate" of instantaneous velocity $\frac{\mathrm{d}x_t}{\mathrm{d}t}\approx\frac{1}{\Delta t}\big[f(x_t,t_{N+1})-\sum_{i=1}^{N} f(x_{t_i},t_{i+1})\big]$. It is termed "recursive" because the velocity at $t$ depends on the integral of the same velocity field at future times; "N-th order" refers to the $N$ integral correction terms, which provide more accuracy than simple one-step approximation. Intuitively, 1st-order focuses on one future segment, concentrating all error there; high-order spreads error across multiple segments, leading to smoother, noise-resistant targets.

**2. Unified training objective for any steps + EMA target model.** By defining the displacement function $-f_\theta(x_t,r):=x_r-x_t=\int_t^r v\,\mathrm{d}\tau$, the formula is rewritten as the loss $L(\theta)=\mathbb{E}\,d\big(\frac{\mathrm{d}x_t}{\mathrm{d}t},\ \frac{1}{\Delta t}[f_\theta(x_t,t_{N+1})-\sum_{i=1}^{N} f_{\theta^-}(x_{t_i},t_{i+1})]\big)$. The true velocity $\mathrm{d}x_t/\mathrm{d}t$ is known analytically via PF-ODE, and $N$ future displacements are calculated by the target model $f_{\theta^-}$ (EMA or periodic copy) with stop-gradients. Time points are sampled hierarchically ($t\sim U[0,T]$, $t_1\sim U[0,t)$, and so on). Crucially: regardless of $N$, **only 1 gradient forward + N non-gradient forwards** are required, unlike JVPs which double VRAM, enabling usage on large models. $N=0/1$ accurately restores Diffusion/Flow Matching and Consistency/Shortcut/MeanFlow, respectively, proving it is a true unified framework.

**3. High-order solving the EMA Incompatibility Paradox.** Empirical findings suggest 1st-order recursion and aggressive EMA are mutually exclusive: at $N=1$, $\kappa=0.999$ pushes FID to 294 (overly stable but non-convergent), while $\kappa=0.9$ yields 31.7 (insufficient stability). Contrastingly, $N=2$ under the same $\kappa=0.999$ maintains high sample quality (MMD stable at ~0.0035 on Moons). High-order allows the model to utilize the stability of strong EMA smoothing without sacrificing quality. Furthermore, increasing $N$ with fixed high $\kappa$ shows performance improves monotonically to $N=4$ before declining due to accumulated approximation error in high-order velocity estimates. Defaults are set to $N=2$ and $\kappa=0.999$.

**4. Linear transport parametrization + Variance reduction loss.** Implementation uses the linear path $\alpha(t)=t,\gamma(t)=1-t$ common in flow matching, where velocity is constant. Displacement is proportional to time difference, parametrized as $f_\theta(x_t,t,r)=F_\theta(x_t,t,r)\cdot(t-r)$, with network $F_\theta$ approximating the average displacement $(x_t-x_r)/(t-r)$. However, since $f_\theta$ scale grows linearly with $(t-r)$, large time steps dominate gradients, causing instability. The authors borrow the gradient identity from sCM $\nabla_\theta\mathbb{E}[F_\theta^\top y]=\frac12\nabla_\theta\mathbb{E}\|F_\theta-F_{\theta^-}+y\|_2^2$ to rewrite the target as a scale-decoupled regression $L(\theta)=\mathbb{E}\|F_\theta(x_t,t,t_{N+1})-F_{\theta^-}(x_t,t,t_{N+1})+\xi\|_2^2$, where $\xi$ bundles future displacements and true velocity differences. Combined with CFG-style guidance and time embeddings, this forms a stable and scalable training pipeline.

## Key Experimental Results

### Main Results (ImageNet-1K Class-conditional Few-step Generation, FID-50K)

| Resolution | Method | NFE | FID ↓ | #Params | #Epochs |
|---|---|---|---|---|---|
| 256×256 | MeanFlow-XL/2 | 1 | 3.43 | 676M | 240 |
| 256×256 | IMM-XL/2 (Optimal) | 8×2=16 | 1.99 | 675M | 3840 |
| 256×256 | **RCGM ⊕ VA-VAE** | **2** | **1.48** | 675M | 424 |
| 256×256 | RCGM ⊕ SD-VAE | 2 | 1.92 | 675M | 424 |
| 512×512 | sCD-L (Distillation) | 2 | 2.04 | 778M | 1434 |
| 512×512 | sCD-XXL (Distillation) | 2 | 1.88 | 1.5B | 921 |
| 512×512 | **RCGM ⊕ DC-AE** | **2** | **1.79** | 675M | 800 |
| 512×512 | RCGM ⊕ SD-VAE | 2 | 2.25 | 675M | 360 |

- On 256×256, RCGM achieves **1.48 FID with 2 NFE**, outperforming IMM's 1.99 FID with 16 NFE—an 8x reduction in sampling steps.
- On 512×512, RCGM with a **675M** model achieves 1.79 FID in 2 steps, surpassing the 1.5B sCD-XXL (1.88) with fewer parameters and training epochs.

### Ablation Study (256×256, 675M DiT, 1-NFE, EMA Decay Rate κ and Order N)

| Order N | κ=0.0 | κ=0.9 | κ=0.99 | κ=0.999 |
|---|---|---|---|---|
| 1st-order | Diverges | 31.70 | Stable but poor | **294.18 (Collapse)** |
| 2nd-order | Unstable | 14.94 | 29.13 | Stable & Superior |

| Order N (Fixed κ=0.999) | 1 | 2 | 3 | 4 | >4 |
|---|---|---|---|---|---|
| Trend | Worst | Improvement | Better | **Lowest FID** | Regression |

### Key Findings
- **EMA Paradox is verified**: 1st-order FD collapses to 294 under $\kappa=0.999$, while 2nd-order converges healthily under the same conditions—direct evidence of the value of high-order.
- **Order sweet spot**: Performance improves monotonically from $N=1$ to $N=4$ and then regresses due to accumulated high-order velocity estimation errors; $N=2$ is the default for compute-performance trade-off.
- **Scalable to ultra-large models**: RCGM stably supports full-parameter training of 20B unified multi-modal models, reaching 0.86 GenEval in 4 steps (8 steps 0.87), whereas 1st-order methods typically suffer from instability, collapse, or VRAM explosion.
- **Strong real-world performance**: 2-NFE on text-to-image tasks reaches 0.85 GenEval, exceeding the previous SOTA SANA-Sprint (0.77).

## Highlights & Insights
- **Strong Unification**: A single "N-th order recursive" formula cleanly incorporates Diffusion ($N=0$) and Consistency/Shortcut/MeanFlow ($N=1$) as special cases, extending naturally to $N \ge 2$. This unifies the fragmented few-step generation landscape into a hierarchical structure.
- **Identifying the Root Cause**: The work explicitly identifies that few-step training vulnerability stems from the "1st-order recursion vs. EMA incompatibility paradox," corroborated by both Moons toy experiments and large-scale ImageNet tests.
- **Engineering Friendliness**: High-order targets only increase non-gradient forwards, adding no VRAM overhead and completely discarding JVPs. This ensuring compatibility with Flash-Attention, enabling scaling to 20B models.

## Limitations & Future Work
- **Extreme 1-NFE remains a weakness**: High-fidelity synthesis at 1 step and high resolution remains unsolved, with 1-NFE FID generally trailing 2-NFE.
- **Lack of Adversarial Objectives**: The 1-NFE shortfall may partly stem from the absence of adversarial loss. Future plans include integrating adversarial training into RCGM to enhance perceptual quality.
- **High-order Error Accumulation**: Performance regression for $N > 4$ suggests that high-order is not always better; selecting the order requires parameter tuning. Theoretical bounds for hierarchical time sampling and error accumulation are not yet fully characterized.

## Related Work & Insights
- **Consistency Model Lineage**: CM, sCM (continuous-time consistency), shortcut, and MeanFlow are the direct targets of unification. By abstracting their "self-recursion + one-step approximation" as 1st-order recursion, this paper provides a unified coordinate system for few-step methods.
- **Unified Diffusion/Flow Matching View**: Built upon the framework of Sun et al. (2025, UCGM), reusing its velocity field/score function construction and variance reduction gradient identities.
- **Inspiration**: Treating the "order of the objective" as a tunable dimension is a highly transferable idea—any paradigm relying on self-supervised recursive goals and suffering from EMA stability issues (e.g., distillation, representation learning) could benefit from "elevating the order."

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Unifying few-step generation as N-th order recursion and extending it to high-order for the first time while diagnosing the EMA paradox is highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple ImageNet VAEs, $\kappa \times N$ ablations, text-to-image, and 20B multi-modal models; however, systematic 1-NFE comparisons and theoretical analysis of high-order error are slightly lacking.
- **Writing Quality**: ⭐⭐⭐⭐ The narrative from 0/1/N-th order hierarchy is clear. Formula density is high; initial reading requires consistency model background.
- **Value**: ⭐⭐⭐⭐⭐ Provides both a unified theory for few-step generation and a practical solution for stable 20B model training, significantly impacting efficient high-fidelity generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Any-Order Flexible Length Masked Diffusion](any-order_flexible_length_masked_diffusion.md)
- [\[ICLR 2026\] Half-order Fine-Tuning for Diffusion Model: A Recursive Likelihood Ratio Optimizer](half-order_fine-tuning_for_diffusion_model_a_recursive_likelihood_ratio_optimize.md)
- [\[CVPR 2026\] Self-Evaluation Unlocks Any-Step Text-to-Image Generation](../../CVPR2026/image_generation/self-evaluation_unlocks_any-step_text-to-image_generation.md)
- [\[ICLR 2026\] Terminal Velocity Matching](terminal_velocity_matching.md)
- [\[ICLR 2026\] PixNerd: Pixel Neural Field Diffusion](pixnerd_pixel_neural_field_diffusion.md)

</div>

<!-- RELATED:END -->
