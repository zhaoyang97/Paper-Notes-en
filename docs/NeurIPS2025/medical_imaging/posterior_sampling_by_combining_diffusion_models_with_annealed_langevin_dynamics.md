---
title: >-
  [Paper Note] Posterior Sampling by Combining Diffusion Models with Annealed Langevin Dynamics
description: >-
  [NeurIPS 2025][Medical Imaging][posterior sampling] This paper proposes an algorithm combining diffusion models with annealed Langevin dynamics that requires only $L^4$-accurate score estimates to achieve polynomial-time posterior sampling under (locally) log-concave distributions, providing the first theoretical guarantees for warm-started inverse problem solving.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - posterior sampling
  - diffusion models
  - Langevin dynamics
  - inverse problems
  - compressed sensing
date: 2026-05-08
content_hash: e289165faf416f33
---

# Posterior Sampling by Combining Diffusion Models with Annealed Langevin Dynamics

**Conference**: NeurIPS 2025
**arXiv**: [2510.26324](https://arxiv.org/abs/2510.26324)
**Code**: N/A
**Area**: Medical Imaging
**Keywords**: posterior sampling, diffusion models, Langevin dynamics, inverse problems, compressed sensing

## TL;DR

This paper proposes an algorithm combining diffusion models with annealed Langevin dynamics that requires only $L^4$-accurate score estimates to achieve polynomial-time posterior sampling under (locally) log-concave distributions, providing the first theoretical guarantees for warm-started inverse problem solving.

## Background & Motivation

**State of the Field**: Diffusion models are the leading generative modeling approach, based on learning smoothed scores $s_{\sigma^2}(x)$ and sampling via SDE/ODE. Posterior sampling (drawing from $p(x|y)$ where $y = Ax + \xi$) encompasses applications such as inpainting, deblurring, and MRI reconstruction, and is a central use case for generative models as priors.

**Limitations of Prior Work**:
   - **Diffusion models excel at unconditional sampling but struggle with posterior sampling**: The smoothed conditional score $\nabla_x \log p(y|x_{\sigma_t^2})$ is intractable to compute exactly; methods such as DPS resort to heuristic approximations.
   - **Langevin dynamics can perform posterior sampling but lacks robustness**: It requires score errors to satisfy moment-generating function (MGF) bounds, which are far stronger than $L^2$ bounds.
   - **General impossibility results**: Gupta et al. proved that efficient and robust posterior sampling is computationally infeasible for general distributions.

**Root Cause**: Unconditional sampling with diffusion models requires only $L^2$-accurate scores, whereas posterior sampling demands MGF-level accuracy — a substantial gap between the two regimes.

**Paper Goals**: To achieve efficient posterior sampling under mild distributional assumptions (log-concavity) using score accuracy requirements significantly weaker than MGF bounds.

**Starting Point**: Diffusion models provide a warm initialization by projecting samples onto the data manifold, after which annealed Langevin dynamics progressively approaches the posterior — the two components are complementary.

**Core Idea**: Diffusion model initialization combined with annealed Langevin dynamics; only $L^4$ score accuracy is required to enable efficient posterior sampling under (locally) log-concave distributions.

## Method

### Overall Architecture

**Algorithm 1**: PosteriorSampler
1. Construct a decreasing noise schedule $\eta_1 > \eta_2 > \cdots > \eta_N = \eta$.
2. Generate auxiliary measurements $y_i$ (by adding noise) such that $y_i \sim Ax + \mathcal{N}(0, \eta_i^2 I_m)$.
3. Sample $X_1$ from $p(x)$ via diffusion SDE (initialization).
4. For $i = 1$ to $N-1$, run the Langevin SDE: $dx_t = \hat{s}_{i+1}(x_t^{(h)})dt + \sqrt{2}dB_t$.
5. Output $X_N$ as an approximate sample from $p(x|y)$.

### Key Designs

#### 1. Design Motivation for the Annealing Schedule

**Problem**: Running Langevin directly from $p(x)$ to $p(x|y)$ may cause intermediate distributions $x_t$ to drift off the data manifold, degrading score estimation accuracy.

**Core Insight**:
- At the start ($t=0$, $x_0 \sim p(x)$) and at convergence ($t=\infty$, $x_\infty \sim p(x|y)$), score estimates are distributionally accurate.
- However, at intermediate times, the marginal variance of $x_t$ may contract (as illustrated by the Gaussian example in Figure 3, where variance shrinks to $cI$, $c<1$), making $L^p$ accuracy insufficient in this regime.

**Solution**: Annealing decomposes a single long Langevin run into $N$ short mixing steps:
- Each step connects two statistically adjacent intermediate posteriors $p(x|y_i)$ and $p(x|y_{i+1})$.
- Short runs ensure $x_t$ does not stray far from its initial distribution.
- Frequent "checkpoints" re-anchor the process to regions where score estimation is accurate.

**Admissible schedule** requirements:
- $\eta_1$ sufficiently large so that $p(x|y_1) \approx p(x)$.
- Adjacent values $\eta_i$ and $\eta_{i+1}$ sufficiently close.

#### 2. Globally Log-Concave Case (Theorem 1.1)

**Assumption**: $p(x)$ is $\alpha$-strongly log-concave with an $L$-Lipschitz score.

**Result**: If the score error satisfies an $L^4$ bound ($\mathbb{E}[\|\hat{s} - s\|^4] \leq \varepsilon_{\text{score}}^4$) and $\varepsilon_{\text{score}} \leq \sqrt{\alpha}/K_1$, then Algorithm 1 outputs a sample with TV distance $\leq \varepsilon$ from the posterior within $K_2 = \text{poly}(d, m, \|A\|/(\eta\sqrt{\alpha}), 1/\varepsilon, L/\alpha)$ steps.

#### 3. Locally Log-Concave Case (Theorem 1.2)

**Motivation**: In settings such as MRI reconstruction, the distribution concentrates near a low-dimensional manifold and is globally non-log-concave but locally log-concave.

**Key Idea**: Given a coarse estimate $x_0$ (e.g., from LASSO), $p(x)$ may be log-concave in a neighborhood of $x_0$. Using $x_0$ as a Gaussian measurement $x_0 = x + \mathcal{N}(0, \sigma^2 I)$, the algorithm targets $p(x|x_0, y)$.

**Result**: As long as the radius $R$ of the locally log-concave region is sufficiently large (polynomially larger than $\sigma$), efficient sampling is achievable.

#### 4. Competitive Compressed Sensing (Corollary 1.3)

If a "naive algorithm" (e.g., LASSO) provides an initial estimate with error $\leq R$, and $p(x)$ is locally log-concave within a $R \cdot \text{poly}$ ball, then:
- Whenever any exponential-time algorithm achieves error $r$, the proposed algorithm achieves error $2r$ in polynomial time.

### Loss & Training

This is a theoretical paper and does not involve model training. The central assumption is the availability of a pretrained diffusion model providing score estimates satisfying an $L^4$ accuracy bound.

## Key Experimental Results

### Main Results: FFHQ-256 Inverse Problems

Validation on three inverse problems (inpainting, 4× super-resolution, Gaussian deblurring):

| Task | Metric | DPS (baseline) | Ours |
|------|--------|---------------|------|
| Inpainting | L2 / FID | Baseline | Both metrics improve over DPS when step size is sufficiently small |
| Super-resolution | L2 / FID | Baseline | L2 decreases as annealing time increases |
| Gaussian deblur | L2 / FID | Baseline | L2 decreases as annealing time increases |

### Key Findings

- Experiments use 1k FFHQ validation images and a DPS pretrained diffusion model.
- Initial reconstructions are obtained via DPS; annealed Langevin dynamics further refines them.
- **Increasing annealing time → lower L2 but higher FID** (trade-off).
- **On the inpainting task, with sufficiently small step size, the proposed method outperforms DPS on both L2 and FID.**
- Reconstructions better preserve ground-truth attributes (qualitative comparisons in Figures 5 and 6).
- Experiments are run on 4× NVIDIA A100 GPUs, approximately 2 hours per task.

## Highlights & Insights

- **Strong theoretical contribution**: For the first time, it is proven that posterior sampling under log-concave distributions requires only $L^4$ score accuracy, substantially relaxing the MGF condition required by Langevin dynamics.
- **Elegant intuition for the annealing scheme**: The distribution transitions gradually, with each step executed in a region where score estimation is accurate.
- **Circumventing impossibility results**: Through warm starting and localization conditions, the method sidesteps general computational lower bounds.
- **Practical utility of local log-concavity**: Combined with coarse estimates from traditional methods such as LASSO, the approach handles distributions supported near manifolds.
- **Refined analysis of hard instances**: The paper demonstrates how lower-bound constructions based on one-way functions become tractable when a warm start is available.

## Limitations & Future Work

1. **Log-concavity assumption remains restrictive**: Real image distributions are typically multimodal and non-log-concave.
2. **$L^4$ requirement is stronger than $L^2$**: It still exceeds the error bounds sufficient for unconditional diffusion sampling.
3. **Limited experimental scale**: Empirical validation is restricted to FFHQ-256.
4. **Computational overhead**: Multi-step annealed Langevin is slower than a single diffusion sampling pass.
5. **Dependence on warm-start quality**: In the locally log-concave regime, algorithm performance depends on the accuracy of the initial estimate.

## Related Work & Insights

- Compared to diffusion-based posterior sampling methods such as DPS, DDNM, and DDRM, this work provides theoretical correctness guarantees.
- Compared to methods using annealed Langevin such as DAPS (Zhang et al. 2025), this is the first work to provide principled design criteria for the annealing schedule along with formal correctness guarantees.
- **Core insight**: The difficulty of posterior sampling lies in "localization" — once the search space is localized via a coarse estimate, the computational hardness is substantially reduced.

## Rating

⭐⭐⭐⭐ (4/5)

**Rationale**: The theoretical contributions are rigorous and elegant, with deep core insights — annealing bridges the gap between $L^2$ and MGF accuracy, and warm starting circumvents impossibility results. Although the empirical evaluation is limited in scale, it validates the central ideas. As a theoretical work, the paper makes an important conceptual contribution.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)
- [\[NeurIPS 2025\] Atomic Diffusion Models for Small Molecule Structure Elucidation from NMR Spectra](atomic_diffusion_models_for_small_molecule_structure_elucidation_from_nmr_spectr.md)
- [\[NeurIPS 2025\] Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology](semantic_and_visual_crop-guided_diffusion_models_for_heterogeneous_tissue_synthe.md)
- [\[NeurIPS 2025\] Fractional Diffusion Bridge Models](fractional_diffusion_bridge_models.md)
- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)

<!-- RELATED:END -->
