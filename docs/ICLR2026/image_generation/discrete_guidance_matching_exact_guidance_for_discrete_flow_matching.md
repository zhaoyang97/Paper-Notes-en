---
title: >-
  [Paper Note] Discrete Guidance Matching: Exact Guidance for Discrete Flow Matching
description: >-
  [ICLR 2026][Image Generation][Paper Note] Given a pre-trained discrete flow matching/diffusion model and the density ratio between target and source distributions, this paper derives **exact** transition rate guidance formulas. This reduces the sampling overhead from multiple forward passes per step to a **single forward pass** and unifies energy guidance, cla
tags:
  - ICLR 2026
  - Image Generation
date: 2026-05-08
content_hash: ecd3680c7f8ba90a
---
# Discrete Guidance Matching: Exact Guidance for Discrete Flow Matching

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=N1RYhOg6ib](https://openreview.net/forum?id=N1RYhOg6ib)  
**Code**: [https://github.com/WanZhengyan/Discrete-Guidance-Matching](https://github.com/WanZhengyan/Discrete-Guidance-Matching)  
**Area**: image_generation (Discrete Flow Matching / Discrete Diffusion Guidance, Posterior Sampling, Preference Alignment)  
**Keywords**: Discrete Flow Matching, Guided Sampling, Continuous Time Markov Chain, Posterior Guidance, Preference Alignment, Text-to-Image  

## TL;DR
Given a pre-trained discrete flow matching/diffusion model and the density ratio between target and source distributions, this paper derives **exact** transition rate guidance formulas. This reduces the sampling overhead from multiple forward passes per step to a **single forward pass** and unifies energy guidance, classifier guidance, and RLHF preference alignment into one framework.

## Background & Motivation
- **Background**: Discrete diffusion models and Discrete Flow Matching (DFM) have emerged as powerful alternatives to Autoregressive (AR) models for generating discrete data (text, tokenized images). "Guidance" is a key mechanism for controlling these models toward a target distribution (conditional generation, energy weighting, preference alignment).
- **Limitations of Prior Work**: Discrete guidance inherently requires modifying a **transition rate/probability** matrix. Since transition rates involve all possible target states reachable from the current state, naive computation requires a forward pass for every candidate position, incurring massive overhead. To accelerate this, existing methods (Vignac 2023, Schiff 2025, Nisonoff 2025) treat discrete models as continuous functions and use **first-order Taylor approximations** of the log-density ratio.
- **Key Challenge**: First-order approximations are **fundamentally invalid in discrete state spaces**. The approximation quality depends on the relative Euclidean positions of $z$ and $x$, but discrete tokens lack meaningful Euclidean distances. This leads to significant approximation errors (e.g., distributions clearly deviate from the ground truth when guidance strength $\gamma=10, 20$). Furthermore, existing methods focus on specific cases like class-conditioning or energy weighting, lacking a unified perspective.
- **Goal**: Construct a discrete guidance framework that is **both exact (no approximation error) and efficient (single forward pass)** while being general enough to encompass energy, classifier, and preference guidance.
- **Core Idea**: **[Exact Transition Rate Rewriting]** Within the CTMC framework, if the source and target distributions share the same conditional probability path, the target posterior can be obtained by **reweighting** the source posterior with the density ratio $r(x)=q_1(x)/p_1(x)$. This reweighting term (conditional expectation of the density ratio) can be learned offline via **Bregman divergence**, enabling single-pass sampling.

## Method

### Overall Architecture
The method is built upon Continuous Time Markov Chains (CTMC). A pre-trained model provides the posterior $p_{1|t}$ sampled from the source distribution $p_1$. Given a known density ratio $r(x)=q_1(x)/p_1(x)$, the framework directly computes the target velocity field/transition rate required to generate the target distribution $q_1$. The pipeline consists of "Learning the conditional expectation of the density ratio → Reweighting the source posterior → Always-valid sampling." A guidance network $h_t$ is trained using Bregman divergence and element-wise multiplied with the pre-trained posterior during sampling.

```mermaid
flowchart LR
    A[Pre-trained Discrete Flow Model<br/>Source Posterior p_1|t] --> D[Target Posterior q_1|t]
    B[Density Ratio r=q_1/p_1<br/>from Energy/Classifier/Reward] --> C[Guidance Network h_t<br/>Bregman Divergence Training]
    C --> D
    D --> E[Always-valid Sampling<br/>Single Forward per Step]
    E --> F[Target Samples q_1]
```

### Key Designs

**1. Posterior-Based Guidance: Exact reweighting of the source posterior with density ratios in a single pass.** This is the theoretical cornerstone (Theorem 1). Under Assumption 1 (target distribution is absolutely continuous w.r.t. the source distribution), if the source and target share the same **conditional** probability path $p_{t|1}=q_{t|1}$, the target posterior has a closed-form:
$$q_{1|t}(z^d|x) = \frac{\mathbb{E}_{x_1^{\setminus d}\sim p(x_1^{\setminus d}|x_1^d=z^d,x_t=x)}[r(x_1)]}{\mathbb{E}_{x_1\sim p_{1|t}(x_1|x)}[r(x_1)]}\, p_{1|t}(z^d|x).$$
This is an **exact equality rather than an approximation**. Both numerator and denominator are conditional expectations of the density ratio over the posterior, requiring no Taylor expansion. When $q_1(x)=p_1(x|y)$ (class-conditional), the density ratio reduces to the classifier ratio $p(y|x_1^d=z^d,x_t=x)/p(y|x_t=x)$, recovering classical classifier guidance. Since it only requires one posterior forward pass for the current state $x$, each sampling step requires only one function evaluation.

**2. Unification with Rate-Based Guidance: Subsuming existing methods as special cases.** Theorem 2 provides the rate-based form: if the forward noise transition rates of the source and target are also identical (a stronger condition than Theorem 1), the reverse transition rate is $u_t^q(z,x)=\frac{\mathbb{E}_{x_1\sim p_{1|t}(x_1|z)}[r(x_1)]}{\mathbb{E}_{x_1\sim p_{1|t}(x_1|x)}[r(x_1)]}u_t^p(z,x)$, recovering the predictor guidance of Nisonoff et al. (2025). The paper categorizes guidance into a spectrum: posterior-based (Ours, exact, 1 forward), rate-based (exact, $D+1$ forwards), and first-order approximated (asymptotic error, 2 forwards). The error in the first-order approximation $u_t^q(z,x)=\exp\langle z-x,\nabla_x\log\mathbb{E}[r(x_1)]\rangle u_t^p(z,x)$ is highlighted: its value depends on the Euclidean positions of $z$ and $x$, which is irrational for discrete data.

**3. Bregman Divergence Training + Target Regularization: Learning conditional expectations of density ratios.** Implementing Theorem 1 requires estimating $h_t^d(z^d,x)=\mathbb{E}[r(x_1)\mid \cdot]$. Since density ratios are naturally positive, standard $\ell_2$ loss ($F(x)=\|x\|^2/2$) performs poorly. This work uses $F(x)=\langle x,\log x\rangle$, resulting in the objective:
$$\mathcal{L}_{h,p}(\theta)=\mathbb{E}\Big[\sum_{d=1}^{D} h_t^{d,\theta}(x_1^d,x_t)-r(x_1)\log h_t^{d,\theta}(x_1^d,x_t)\Big],$$
which **only requires source distribution data**. If target samples are available, a regularization term $\mathcal{L}_{h,q}$ (whose minimum is also the exact guidance $h_t$) can be added. The final objective is $\mathcal{L}_h=\mathcal{L}_{h,p}+\lambda\mathcal{L}_{h,q}$.

**4. Unifying Three Tasks (Energy Guidance / Classifier Guidance / RLHF Preference Alignment).** The framework's generality stems from the density ratio being derived from various sources. Energy guidance uses $p_1^{(\gamma)}(x)\propto p_1(x)e^{-\gamma E(x)}$, hence $r\propto p^\gamma(y=1|x)$. Classifier guidance uses the classifier ratio. Preference alignment leverages the closed-form optimal policy from RLHF: $\pi^*(o_1|c)\propto \pi_{\text{ref}}(o_1|c)\exp(R(c,o_1)/\tau)$, where $p_1=\pi_{\text{ref}}$ and $q_1=\pi^*$, such that the guidance network approximates $\exp(R(c,o_1)/\tau)$.

## Key Experimental Results

### Main Results (GenEval Text-to-Image, based on FUDOKI)

| Method | Single | Two | Counting | Colors | Position | Color Attri. | Overall ↑ |
|---|---|---|---|---|---|---|---|
| FUDOKI (Baseline) | 0.96 | 0.85 | 0.56 | 0.88 | 0.68 | 0.67 | 0.77 |
| **Ours (Exact Guidance)** | 0.94 | 0.86 | 0.53 | 0.89 | 0.70 | **0.77** | **0.78** |

Ours exceeds the baseline in four out of six sub-tasks, with Color Attribution showing the most significant gain (0.67 to 0.77).

### Ablation Study (Multimodal Understanding, 1.5B parameters)

| Model | POPE ↑ | MME-P ↑ | MMB ↑ | GQA ↑ | MMMU ↑ | MM-Vet ↑ |
|---|---|---|---|---|---|---|
| FUDOKI (Baseline) | 86.1 | 1485.4 | 73.9 | 57.6 | 34.3 | 38.0 |
| **Ours** | **86.8** | **1492.7** | **74.2** | **58.2** | **35.4** | **38.6** |

Guidance provides consistent improvements across all six understanding benchmarks. In 2-D energy guidance simulations, first-order predictor guidance deviates significantly from the ground truth at $\gamma=10, 20$, while the proposed posterior/rate-based guidance methods remain accurate.

### Key Findings
- **Sampling Efficiency**: Posterior-based guidance is approximately **1.6×** faster than rate-based guidance (the latter requires $D+1$ forwards per step, while ours requires only 1).
- **Precision**: First-order approximations suffer from severe distortion at high guidance strengths, whereas the proposed exact method stably approximates the target distribution.
- **Unification**: The framework is effective across energy guidance, text-to-image RLHF, and multimodal understanding, validating the universality of the density ratio perspective.

## Highlights & Insights
- **Precise Diagnosis**: The paper clearly explains why first-order approximations fail in discrete spaces—the approximation values depend on Euclidean positions of tokens, which lack meaningful geometric relationships.
- **Solving the Exactness-Efficiency Trade-off**: Theorem 1 utilizes the mild "shared conditional path" assumption to reduce guidance to a simple reweighting of the source posterior, achieving exactness with a single forward pass.
- **Theoretical Consolidation**: Theorem 2 frames previous rate-based methods as special cases under stronger assumptions, providing a clear method spectrum (precision vs. evaluation counts).
- **Bregman Divergence Selection**: Choosing $F=\langle x,\log x\rangle$ over $\ell_2$ to account for the positive nature of density ratios is a well-grounded engineering choice.

## Limitations & Future Work
- **Dependency on Density Ratios**: The framework assumes the density ratio $r(x)=q_1/p_1$ is known or learnable, which may limit applicability for target distributions where ratios are hard to define or estimate (including the absolute continuity requirement in Assumption 1).
- **Moderate Gains**: Absolute gains in GenEval (0.77 to 0.78) and multimodal understanding are relatively small, representing robust incremental improvements rather than a massive leap.
- **Path Consistency Assumption**: Theorem 1 requires shared conditional probability paths, which might restrict certain cross-model transfer scenarios.
- **Future Work**: Extending exact guidance to more complex reward/energy structures, larger-scale multimodal models, and exploring adaptive strategies for the $\lambda$ regularization term when target samples are scarce.

## Related Work & Insights
- **Continuous Guidance**: Classifier guidance (Dhariwal & Nichol 2021) and energy guidance (Lu 2023, Ouyang 2024) serve as the conceptual foundations that this work precisely translates to the discrete domain.
- **Discrete Guidance**: This work surpasses the first-order approximations of Vignac 2023, Schiff 2025, and Nisonoff 2025, explicitly recovering Nisonoff's predictor guidance in Theorem 2.
- **Discrete Flow Matching**: Theoretical foundations provided by Campbell 2024, Gat 2024, and others, with FUDOKI (Wang 2025) serving as the multimodal backbone.
- **Preference Alignment**: By utilizing optimal policies from DPO/RLHF (Rafailov 2023, Ouyang 2022), reward alignment is integrated into the unified guidance framework.
- **Insight**: In discrete generation control, rather than using geometric approximations on transition rates, it is superior to return to probabilistic reweighting of CTMC posteriors—precision and efficiency can be achieved simultaneously.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First general exact discrete guidance; clarifies the relationship between energy, classifier, and preference alignment while subsuming prior work.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 2-D simulations, GenEval, and 6 multimodal benchmarks with efficiency comparisons, though absolute downstream gains are moderate.
- **Writing Quality**: ⭐⭐⭐⭐ — Logical flow from theorems to framework to experiments; the method spectrum comparison in Table 1 and the analysis of first-order failure are very persuasive.
- **Value**: ⭐⭐⭐⭐ — Single-pass exact guidance is highly practical for controllable discrete diffusion/flow models, especially given its compatibility with masked diffusion and large multimodal models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Discrete Adjoint Matching](discrete_adjoint_matching.md)
- [\[ICLR 2026\] What Exactly Does Guidance Do in Masked Discrete Diffusion Models](what_exactly_does_guidance_do_in_masked_discrete_diffusion_models.md)
- [\[ICLR 2026\] Delay Flow Matching](delay_flow_matching.md)
- [\[ICLR 2026\] Flow Matching with Semidiscrete Couplings](flow_matching_with_semidiscrete_couplings.md)
- [\[ICLR 2026\] Carré du champ Flow Matching: 用几何感知噪声改善生成模型的质量-泛化权衡](carré_du_champ_flow_matching_better_quality-generalisation_tradeoff_in_generativ.md)

</div>

<!-- RELATED:END -->
