---
title: >-
  [Paper Note] Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems
description: >-
  [CVPR 2025][Image Generation][Diffusion Models] Proves that the DDIM deterministic reverse chain is a Partitioned Iterated Function System (PIFS), deriving three computationally accessible geometric quantities (contraction threshold $L_t^*$, expansion function $f_t(\lambda)$, and global expansion threshold $\lambda^{**}$) that require no model evaluation. Based on this, it theoretically explains four existing empirical design choices (cosine offset, resolution logSNR shift…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Diffusion Models"
  - "Fractal Geometry"
  - "PIFS"
  - "DDIM"
  - "Noise Schedule"
  - "Theoretical Analysis"
date: 2026-05-08
content_hash: 704f4ceaf08421ca
---

# Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems

**Conference**: CVPR 2025  
**arXiv**: [2603.13069](https://arxiv.org/abs/2603.13069)  
**Code**: None  
**Area**: Image Generation / Diffusion Model Theory  
**Keywords**: Diffusion Models, Fractal Geometry, PIFS, DDIM, Noise Schedule, Theoretical Analysis

## TL;DR

Proves that the DDIM deterministic reverse chain is a Partitioned Iterated Function System (PIFS), deriving three computationally accessible geometric quantities (contraction threshold $L_t^*$, expansion function $f_t(\lambda)$, and global expansion threshold $\lambda^{**}$) that require no model evaluation. Based on this, it theoretically explains four existing empirical design choices (cosine offset, resolution logSNR shift, Min-SNR weighting, and Align Your Steps).

## Background & Motivation

**Background**: The theoretical foundation of diffusion models is established on SDEs/ODEs, providing global guarantees for distribution convergence. However, the continuous perspective treats the score network as a black box, failing to explain two core phenomena: (a) Why does the denoising chain assemble global spatial context at high noise levels and synthesize local details at low noise levels? (b) Why is self-attention so effective?

**Limitations of Prior Work**: Many design choices in diffusion models remain empirical—why is the offset=0.008 in the cosine schedule good? Why is Min-SNR weighting effective? A unified theoretical framework to understand and predict these designs is lacking.

**Key Challenge**: Elegant theory but lacking structural insights—SDE theory tells us "it converges" but not "how it converges".

**Goal**: To provide a unified design language for understanding and optimizing schedules, architectures, and training objectives of diffusion models.

**Key Insight**: In 1988, Barnsley proposed that natural images exhibit local self-similarity, which can be compressed using Partitioned Iterated Function Systems (PIFS). This work finds that the DDIM reverse chain is precisely a PIFS—each denoising step is a partitioned contractive mapping.

**Core Idea**: The denoising chain of a diffusion model is a PIFS, whose fractal geometry fully characterizes the two-stage structure of denoising dynamics.

## Method

### Overall Architecture

Treating the single-step DDIM operator $\Phi_t(x) = \frac{\sqrt{\bar\alpha_{t-1}}}{\sqrt{\bar\alpha_t}} x + b_t \hat\varepsilon_\theta(x, t)$ as a single step of a PIFS. The core is to analyze the contraction/expansion properties of its Jacobian—specifically, the interactions between diagonal blocks (intra-patch dynamics) and cross-patch blocks (attention coupling).

### Key Designs

1. **Contraction Structure (Section 3)**:

    - Derives two contraction conditions: (EC) Euclidean Contraction and (PC) Patchwise Maximum Norm Contraction.
    - Contraction threshold $L_t^* = (\sqrt{\bar\alpha_{t-1}/\bar\alpha_t} - 1) / |b_t|$—solely determined by the noise schedule, independent of the data or model.
    - Score-matching training is the diffusion analog of Barnsley's collage theorem error minimization.
    - $L_2$-$W_1$ bridge: the training loss controls the Wasserstein distance of the PIFS attractor.

2. **Two-Stage Structure (Section 4)**:

    - **Regime I (High Noise)**: Diffuse attention maintains strong cross-patch coupling (large $\delta_t^{cross}$), where the learned "directional inhibition field" $S_{k,t}$ holds diagonal blocks below the expansion threshold $\rightarrow$ global context assembly.
    - **Regime II (Low Noise)**: Attention localizes, and inhibition is released patch-by-patch in order of variance $\rightarrow$ local detail synthesis.
    - Why self-attention is effective: It precisely controls $\delta_t^{cross}$ (via upper-bounding softmax weights), making it a natural primitive for PIFS contraction.
    - The two-stage transition aligns with the spontaneous symmetry breaking reported by Raya & Ambrogioni (2023).

3. **Attractor Geometry (Section 5)**:

    - The Kaplan-Yorke dimension of the PIFS attractor is determined by the discrete Moran S-equation: $\prod_t f_t(\lambda^{**}) = 1$.
    - A patch direction contributes to sample diversity $\iff$ its leading variance exceeds $\lambda^{**}$.

4. **Three Design Guidelines (Section 6)**:

    - **Guideline 1**: Maximize the minimum contraction threshold of the weakest link $\min_t L_t^*$ (inject noise early, raising $v_1$).
    - **Guideline 2**: Balance the Lyapunov contribution of each step = minimize $\text{Var}_t(\Delta d_t)$ $\approx$ Information Capacity Criterion.
    - **Guideline 3**: Balance the workload of sampling steps—concentration of steps where $L_t^*$ is minimized.

### Theoretical Explanation of Four Empirical Designs

| Empirical Design | Corresponding Guideline | PIFS Explanation |
|---------|---------|----------|
| Cosine offset $s_{off}=0.008$ | Guideline 1 | Increases $L_1^*$ from $7.9 \times 10^{-4}$ to $3.2 \times 10^{-3}$ (4x), enhancing the contraction margin of the weakest step |
| Resolution logSNR shift | Prerequisite for Guideline 1 | The schedule must cover the logSNR range of detail patch transitions |
| Min-SNR weighting | Guideline 2 | Balances the information gain of each step, equivalent to balancing KY dimension growth |
| Align Your Steps | Guideline 3 | Concentrates sampling steps where the geometric contribution is maximized |

## Key Experimental Results

### Schedule Comparison

| Schedule | Steps | Average $L_t^*$ | CV($L_t^*$) | Finest Step $L_t^*$ |
|------|------|-------------|-------------|---------------|
| Linear (DDPM) | 1000 | 0.805 | **0.341** | 0.00500 |
| Cosine ($s_{off}=0$) | 1000 | 0.637 | 0.483 | 0.00079 |
| Cosine ($s_{off}=0.008$) | 1000 | 0.641 | 0.474 | **0.00321** |
| 50-step DDIM | 50 | 0.637 | 0.483 | 0.01571 |

### Information Gain Balancing

| Schedule | CV(IG_t) | CV(|Δd_t|) | Spearman ρ(IG, Δd) |
|------|---------|-----------|-------------------|
| Linear | 1.107 | 0.836 | 0.9999 |
| Cosine | **0.867** | **0.570** | 0.9998 |

### Key Findings

- **$L_t^*$ is smallest at $t=1$ (finest step)**: $L_t^* \approx \frac{1}{2}\sqrt{v_t}$, indicating that detail synthesis is the most constrained stage.
- **All $8 \times 8$ patches under CIFAR-10 are expansion-forced throughout the entire 1000-step chain**: The leading eigenvalue far exceeds $\lambda^*(t) \approx 1.002$.
- **IG and KY dimension growth are near-perfectly proportional**: Spearman $\rho > 0.999$, validating the tightness of the theoretical CS inequality.
- **Linear schedule has good $L_t^*$ balancing but poor IG balancing; Cosine is the opposite**: No single schedule is optimal for both.

## Highlights & Insights

- **Unifying fractal image compression from 1988 with diffusion models from 2020**: A deep mathematical connection where Barnsley's self-similar structure drives the success of diffusion models. Score-matching is collage theorem error minimization—not an analogy, but a mathematical identity.
- **Three model-evaluation-free geometric quantities form the "design language"**: $L_t^*$, $f_t(\lambda)$, and $\lambda^{**}$ are entirely determined by the schedule and data covariance spectrum. The behavior of schedules can be predicted before training any model.
- **Extremely clear PIFS explanation of the two stages**: Regime I's "inhibition field" maintains contraction $\rightarrow$ global structure assembly; Regime II's inhibition release $\rightarrow$ detail emergence. This is an inevitable mathematical consequence, not a post-hoc description.
- **Structural proof for the necessity of self-attention**: It controls the cross-patch coupling $\delta_t^{cross}$, serving as the natural contraction primitive required by PIFS.

## Limitations & Future Work

- **Analysis limited to DDIM (deterministic sampling)**: The PIFS structure under DDPM (stochastic sampling) remains an open problem.
- **Experiments primarily based on CIFAR-10 analysis**: Not yet validated on high-resolution datasets (ImageNet 256/512).
- **Gaussian Assumption**: The attractor dimension analysis relies on a block-diagonal Gaussian assumption.
- **Practical training effects of the PIFS regularizer are not fully validated**: Theoretically, it can widen the contraction margin, but when the extra computational cost becomes worthwhile remains unclear.

## Related Work & Insights

- **vs Raya & Ambrogioni (2023)**: They discovered two-stage behavior and symmetry breaking, whereas this work provides a precise geometric mechanistic explanation.
- **vs Kingma et al. (2021) Information Capacity Criterion**: This work derives equivalent conclusions from a completely different perspective (equalizing KY dimension growth), offering a deeper understanding.
- **Insights**: The research paradigm of "reversing from existing empirical designs to theoretical optimal solutions" is highly inspiring—seeking to understand why existing practices work rather than proposing new methods.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The unification of diffusion models and fractal geometry is a brand-new perspective, with profound mathematical contributions.
- Experimental Thoroughness: ⭐⭐⭐ Primarily theoretical, with empirical validation concentrated on CIFAR-10 analysis.
- Writing Quality: ⭐⭐⭐⭐ Rigorous mathematical derivations, but with a high entry barrier for non-theoretical readers.
- Value: ⭐⭐⭐⭐⭐ Provides a unified theoretical design language for diffusion models, explaining multiple empirical designs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Optimizing for the Shortest Path in Denoising Diffusion Model](optimizing_for_the_shortest_path_in_denoising_diffusion_model.md)
- [\[CVPR 2025\] AniDoc: Animation Creation Made Easier](anidoc_animation_creation_made_easier.md)
- [\[ACL 2025\] Planning with Diffusion Models for Target-Oriented Dialogue Systems](../../ACL2025/image_generation/difftod_diffusion_dialogue_planning.md)
- [\[ICLR 2026\] Diffusion Negative Preference Optimization Made Simple](../../ICLR2026/image_generation/diffusion_negative_preference_optimization_made_simple.md)
- [\[NeurIPS 2025\] AccuQuant: Simulating Multiple Denoising Steps for Quantizing Diffusion Models](../../NeurIPS2025/image_generation/accuquant_simulating_multiple_denoising_steps_for_quantizing.md)

</div>

<!-- RELATED:END -->
