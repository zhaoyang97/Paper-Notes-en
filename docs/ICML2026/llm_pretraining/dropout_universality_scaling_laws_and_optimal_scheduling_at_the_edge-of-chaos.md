---
title: >-
  [Paper Note] Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos
description: >-
  [ICML 2026][LLM Pretraining][Mean Field Theory] The authors interpret dropout as an "external field" $h$ that breaks the $c^*=1$ perfect alignment fixed point in mean field signal propagation theory. They derive the Landau equation, two-parameter scaling collapse, and identify two distinct universality classes for smooth and kinked activations. This leads to a "zero-overhead" practical conclusion: a **front-loaded schedule** reduces test loss by 18–35% in MLPs and ViTs compar…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "Mean Field Theory"
  - "edge-of-chaos"
  - "dropout scheduling"
  - "universality classes"
  - "scaling laws"
date: 2026-05-08
content_hash: bb870c76e281be44
---

# Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos

**Conference**: ICML 2026  
**arXiv**: [2605.21648](https://arxiv.org/abs/2605.21648)  
**Code**: Available (dropout-universality-experiments, commit-pinned repository in paper)  
**Area**: Training Theory / Dropout / Mean Field / Signal Propagation / Scheduling  
**Keywords**: Mean Field Theory, edge-of-chaos, dropout scheduling, universality classes, scaling laws

## TL;DR
The authors interpret dropout as an "external field" $h$ that breaks the $c^*=1$ perfect alignment fixed point in mean field signal propagation theory. They derive the Landau equation, two-parameter scaling collapse, and identify two distinct universality classes for smooth and kinked activations. This leads to a "zero-overhead" practical conclusion: a **front-loaded schedule** reduces test loss by 18–35% in MLPs and ViTs compared to constant dropout under the same budget.

## Background & Motivation

**Background**: Mean Field Theory (MFT) for randomly initialized deep networks (Poole et al. 2016; Schoenholz et al. 2017) categorizes networks into ordered, chaotic, and critical phases. At the "edge-of-chaos," the correlation length $\xi_c$ diverges, allowing signals to propagate deeply. He initialization ($\sigma_w^2=2$) is essentially the criticality condition for ReLU.

**Limitations of Prior Work**: Dropout is the default regularization in industry, but in MFT, it is merely treated as something that "destroys the $c^*=1$ fixed point," without providing usable scaling laws. Regarding scheduling, the industry relies on heuristics like constant dropout, stochastic depth, or curriculum dropout, lacking a first-principles explanation for **why** certain schedules are superior.

**Key Challenge**: Dropout provides regularization (reducing overfitting) while simultaneously **severing** the correlation of signal propagation along depth. These are adjusted independently at each layer, but current theory cannot guide how to distribute dropout across depth for a given budget, nor whether smooth vs. kinked activations require different strategies.

**Goal**: (i) Embed dropout into MFT to provide scaling law descriptions; (ii) distinguish universality classes for smooth/kinked activations; (iii) translate theory into executable scheduling rules.

**Key Insight**: By viewing dropout as the "external field" $h$ from statistical mechanics and the de-alignment $m\equiv 1-c^*$ as the "order parameter," the problem transforms into the standard paradigm of Landau critical phenomena—allowing the application of RG, scaling collapse, and universality tools.

**Core Idea**: Dropout adds a **constant offset** to the correlation map at $c=1$, making $c^*<1$ a fixed point with finite correlation length; this offset is the "external field" $h$. Maximizing $\xi_{\rm eff}$ under a budget $\sum_\ell h_\ell = L\bar{h}$ is a concave optimization, where **saturated step solutions** are optimal. "Regularization reach" further selects the "front-loaded" branch.

## Method

### Overall Architecture

The paper addresses a question often ignored: Given a fixed total dropout budget, how should it be distributed across network depth? The logic extends MFT for deep networks—where without dropout, forward correlation follows $c^l = F(c^{l-1})$, and criticality is defined by $\chi_\perp \equiv F'(1) = 1$. With inverted dropout (keep-probability $\rho$), the map becomes $\bar{F}_\rho(c)$, and $c=1$ is no longer a fixed point. The authors identify this shift as the "field" $h$ and $m\equiv 1-c^*$ as the "order parameter," applying Landau theory to define actionable dropout schedules. Experiments validate this rule on MLPs and ViTs (CIFAR-10/100) by comparing constant, front-loaded, back-loaded, and linear schedules under a fixed budget $\bar{h}$.

### Key Designs

**1. Identifying Dropout as a Field $h$ Breaking Alignment Symmetry**

Prior work (Schoenholz et al. 2017) noted dropout destroys the $c=1$ fixed point but stopped there, leaving correlation length undefined. The author's key step is evaluating the correlation recursion $\bar{F}_\rho$ after independent masking at $c=1$: $\bar{F}_\rho(1) = 1 - \frac{1-\rho}{\rho \bar{q}^*}\sigma_w^2 \int Dz\,\phi^2(\sqrt{\bar{q}^*}z) < 1$. This defines the field $h \equiv 1-\bar{F}_\rho(1)$ (where $h \approx a(1-\rho)$ for weak dropout) and the order parameter $m\equiv 1-c^*$. Taylor expanding $\bar{F}_\rho(1-m)$ around $m=0$ yields the standard Landau equation:

$$h = \tfrac{g_\rho}{2}m^2 - tm,\qquad m(t,h) = \frac{t+\sqrt{t^2+2g_\rho h}}{g_\rho},$$

where $t\equiv \chi_\rho - 1$ is the reduced temperature. This proves that the deformed recursion **still has a fixed point $c^*<1$**, allowing correlation lengths and scaling laws to be defined.

**2. Smooth vs. Kinked Universality Classes + Two-Parameter Scaling Collapse**

The authors show that the difference in critical behavior between smooth (tanh, GELU) and kinked (ReLU) activations is determined by the **analytical structure** of the correlation map near $c=1$. Smooth activations follow Price’s Theorem and allow a smooth Taylor expansion where the $g_\rho m^2$ term dominates, giving $m\sim\sqrt{h}$ ($\delta=2$) and $\xi\sim h^{-1/2}$. Kinked activations (ReLU) involve a branch point at $c=1$ due to $\delta$-functions in $\phi''$, resulting in $h = \kappa m^{3/2} - tm$, giving $m\sim h^{2/3}$ ($\delta=3/2$) and $\xi\sim h^{-1/3}$. Critical exponents ($\nu_t, \beta, \theta_{\rm rel}, \gamma, \delta, \nu_\rho, \alpha$) are provided for both. $(t,h)$ curves within a class collapse to a universal function; e.g., for smooth activations, defining $\tilde{m}\equiv m\sqrt{g_\rho/(2h)}$ and $\tilde{t}\equiv -t/\sqrt{2g_\rho h}$ results in $\tilde{m} = \sqrt{1+\tilde{t}^2}-\tilde{t}$.

**3. Front-Loaded Dropout Schedule: Concave Budgeting and Regularization Reach**

With correlation length scaling laws, dropout placement becomes an optimization problem. Let keep probability vary by layer $\ell$. The effective inverse correlation length is $\xi_{\rm eff}^{-1} \approx \frac{1}{L}\sum_\ell \sqrt{t^2+2g_\rho h_\ell}$. At criticality ($t=0$), this simplifies to $\xi_{\rm eff}^{-1} \propto \frac{1}{L}\sum_\ell h_\ell^{1/2}$ subject to $\sum_\ell h_\ell = L\bar{h}$ and $h_\ell \leq h_{\max}$. Since $h^{1/2}$ is concave, Jensen's inequality dictates the **step** solution (budget allocated to $\{0, h_{\max}\}$) is optimal. The gain over constant dropout is $\xi_{\rm step}/\xi_{\rm const} = \sqrt{h_{\max}/\bar{h}}$. To break the degeneracy of step placement, "downstream exposure" $\mathcal{D}_\ell \approx h_\ell \xi_c(1-e^{-(L-\ell)/\xi_c})$ is introduced—masks in early layers are "seen" by more downstream layers, selecting the **front-loaded schedule**.

### Loss & Training

The training objective remains unchanged; only the distribution of dropout across layers is modified. For a fixed average budget $\bar{h}$, the authors compare constant, linear-decreasing, early-step, and late-step schedules. Theoretical values for $\bar{F}_\rho, \chi_\rho, g_\rho$ are computed via Gaussian measure integration of MFT recursions.

## Key Experimental Results

### Main Results

| Experimental Setup | Schedule | Loss Reduction | Δacc (pp) | Relative Gain |
|---------|------|---------|-----------|----------|
| MLP Overfitting (Fig.6) | Step (early) | +17.9% | +0.83 | +2.0% |
| MLP Budget Control (Fig.7) | Big step (1/3) | +22.6% | +1.08 | +2.6% |
| ReLU $\bar{h}=0.1$ sweep | Big step (1/3) | **+35.4%** | +2.04 | +5.0% |
| GELU $\bar{h}=0.1$ sweep | Big step (1/3) | +29.8% | +0.62 | +1.5% |
| ViT CIFAR-100 | Linear (decreasing) | +4.2% | +0.66 | +1.4% |
| ViT CIFAR-10 ablation | Both blocks, step (early) | +6.3% | +0.52 | +0.7% |

On ViT CIFAR-100, linear-decreasing reached 49.38% vs. constant 48.69% ($p<0.05$).

### Ablation Study

| Configuration | Key Phenomenon | Explanation |
|------|---------|------|
| Smooth (tanh/GELU) MFT | $m\sim\sqrt{h}$ ($\delta=2$), $\xi\sim h^{-1/2}$ | Consistent with Landau $m^2$ term |
| Kinked (ReLU) MFT | $m\sim h^{2/3}$ ($\delta=3/2$), $\xi\sim h^{-1/3}$ | Branch point $m^{3/2}$ term dominates |
| Scaling Collapse (Fig.2) | All $(t,h)$ curves collapse | Smooth class: $\tilde{m}=\sqrt{1+\tilde{t}^2}-\tilde{t}$ |
| Width $\gg$ Depth | Front-loading advantage stable | Holds in the MFT regime |
| High Dropout / Narrow Nets | Advantage diminishes | Occurs where theory breaks down |

### Key Findings

- ReLU MLPs achieved the **largest** gains (+35.4% loss reduction), confirming that the kinked class's lower-order nonlinearity allows more aggressive budget reallocation.
- Smooth class (GELU) gains (+29.8%) were also significant, showing the conclusion is universal across activations.
- Gains on ViTs were smaller (4–6%), consistent with theory: attention and skip connections alter global depth dynamics but preserve local Gaussian kernels; the "order" (early layers favored) remains valid but magnitude decreases.
- The disappearance of advantages in narrow or high-dropout networks provides **negative evidence** supporting the theory's boundaries.

## Highlights & Insights

- Mapping dropout to an "external field" $h$ and de-alignment to an "order parameter" $m$ provides an elegant "problem alignment," creating a template for treating any hyperparameter as a "field" in statistical mechanics.
- The classification of universality classes is based on the **analytical structure** of the activation (Taylor expansion vs. branch point), not just scale-invariance. This explains the behavioral split between ReLU and tanh-like families.
- The **two-step optimization** (concave budget allocation $\to$ step function; monotonicity weights $\to$ front-loading) is insightful, as MFT objectives are often permutation invariant.

## Limitations & Future Work

- **Forward MFT Only**: There is asymmetry in backward gradient covariance due to mask independence; the authors offer recursion (18) but lack a full backward critical theory. Finite-width susceptibilities and training dynamics are unmodeled.
- **Architecture Constraints**: The dropout-deformed MFT for CNNs/ResNets is discussed but lacks extensive experiments. ViT attention in the large-width limit needs more granular treatment.
- **Initialization Theory**: Conclusions are at **initialization time**; the feedback from representation learning during training on the schedule is not captured.
- **Mask Correlation**: If masks are shared within a batch, the $c=1$ fixed point is restored, weakening regularization and requiring new analysis.

## Related Work & Insights

- **vs. Schoenholz et al. (2017)**: They noted dropout destroys the $c=1$ fixed point but declared "criticality disappears." This paper proves $c^*<1$ is still a fixed point, enabling RG/scaling tools—a crucial "half-step forward."
- **vs. Hayou et al. (2019)**: Observed behavior differences between smooth/ReLU at the edge-of-chaos; this paper formalizes the **critical exponents** and universality classes.
- **vs. LayerDrop / Stochastic Depth**: These are temporal or layer-wise schedules; this paper focuses on **spatial depth-wise** dropout intensity scheduling.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Fully imports statistical physics universality tools to dropout scheduling; defines distinct smooth/kinked exponents.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive $\bar{h}$-sweeps and ablations on MLP/ViT, though CNN/ResNet experiments are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear derivation from Landau equations to scaling collapse and scheduling rules.
- Value: ⭐⭐⭐⭐ Practical "zero-overhead" rule; theoretical framework foundational for treating other hyperparameters as "fields."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Explaining Data Mixing Scaling Laws](explaining_data_mixing_scaling_laws.md)
- [\[ICLR 2026\] Pretraining Scaling Laws for Generative Evaluations of Language Models](../../ICLR2026/llm_pretraining/pretraining_scaling_laws_for_generative_evaluations_of_language_models.md)
- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)
- [\[ICLR 2026\] Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining](../../ICLR2026/llm_pretraining/scaling_laws_revisited_modeling_the_role_of_data_quality_in_language_model_pretr.md)
- [\[NeurIPS 2025\] Gemstones: A Model Suite for Multi-Faceted Scaling Laws](../../NeurIPS2025/llm_pretraining/gemstones_a_model_suite_for_multi-faceted_scaling_laws.md)

</div>

<!-- RELATED:END -->
