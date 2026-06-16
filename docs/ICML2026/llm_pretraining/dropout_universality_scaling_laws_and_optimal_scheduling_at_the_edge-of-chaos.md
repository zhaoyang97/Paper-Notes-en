---
title: >-
  [Paper Note] Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos
description: >-
  [ICML 2026][Pretraining][edge-of-chaos] The authors interpret dropout as an "external field" $h$ that breaks the perfect alignment fixed point $c^*=1$ in Mean Field Theory (MFT) of signal propagation. They derive Landau equations, a two-parameter scaling collapse, and two distinct universality classes for smooth vs. kinked activations. Consequently, they pro
tags:
  - ICML 2026
  - Pretraining
  - edge-of-chaos
date: 2026-05-08
content_hash: a2c0c108936e0e38
---
# Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos

**Conference**: ICML 2026  
**arXiv**: [2605.21648](https://arxiv.org/abs/2605.21648)  
**Code**: Available (dropout-universality-experiments, commit-pinned repository in paper)  
**Area**: Training Theory / Dropout / Mean Field / Signal Propagation / Scheduling  
**Keywords**: Mean field theory, edge-of-chaos, dropout scheduling, universality class, scaling laws

## TL;DR
The authors interpret dropout as an "external field" $h$ that breaks the perfect alignment fixed point $c^*=1$ in Mean Field Theory (MFT) of signal propagation. They derive Landau equations, a two-parameter scaling collapse, and two distinct universality classes for smooth vs. kinked activations. Consequently, they propose a "zero-overhead" practical conclusion: the **front-loaded schedule** reduces test loss by 18–35% on MLPs and ViTs compared to constant dropout under the same budget.

## Background & Motivation

**Background**: MFT for randomly initialized deep networks (Poole et al. 2016; Schoenholz et al. 2017) categorizes networks into ordered, chaotic, and critical phases. At the "edge-of-chaos," the correlation length $\xi_c$ diverges, allowing signals to propagate most deeply. He initialization ($\sigma_w^2=2$) is essentially the critical condition for ReLU.

**Limitations of Prior Work**: Dropout is the default industrial regularization term, but its treatment in MFT has been limited to stating that "it destroys the $c^*=1$ fixed point," without providing actionable scaling laws. Regarding scheduling, the industry relies on heuristics (constant dropout, stochastic depth, or curriculum dropout). There is a lack of first-principle explanations for **why** certain schedules perform better.

**Key Challenge**: Dropout simultaneously provides regularization (reducing overfitting) and **severs** the propagation of signal correlation along depth. While these are adjusted independently at each layer, current theory fails to explain how to allocate dropout given a budget or whether smooth vs. kinked activations necessitate different strategies.

**Goal**: (i) Embed dropout into MFT to provide descriptions at the scaling law level; (ii) distinguish universality classes for smooth/kinked activations; (iii) translate theory into executable scheduling rules.

**Key Insight**: By viewing dropout as the "external field" $h$ in statistical mechanics and the de-alignment $m \equiv 1-c^*$ as the "order parameter," the problem transforms into the standard paradigm for Landau critical phenomena. This allows the application of established tools like Renormalization Group (RG), scaling collapse, and universality classes.

**Core Idea**: Dropout adds a **constant offset** to the correlation map at $c=1$, such that $c^*<1$ remains a fixed point but with a finite correlation length. This offset is the "external field" $h$. Maximizing $\xi_{\rm eff}$ under the budget $\sum_\ell h_\ell = L\bar{h}$ is a concave optimization problem where **saturated step solutions** are optimal. "Regularization reach" further identifies the "front-loaded" branch as superior.

## Method

### Overall Architecture

The paper addresses a fundamental question: given a fixed total dropout budget, how should it be distributed along the network's depth? The logic extends MFT for randomly initialized deep nets. Without dropout, forward correlation follows the recursion $c^l = F(c^{l-1})$, where the edge-of-chaos is defined by $\chi_\perp \equiv F'(1) = 1$, making $c=1$ a perfectly aligned fixed point. Introducing inverted dropout with keep-probability $\rho$ modifies the map to $\bar{F}_\rho(c)$, where $c=1$ is no longer a fixed point. The authors identify this shift as the "external field" $h$ and $m\equiv 1-c^*$ as the "order parameter." Landau critical phenomena tools are then used to derive an executable dropout scheduling rule. Experiments validate this rule on MLPs and ViTs (CIFAR-10/100) by comparing constant, front-loaded, back-loaded, and linear schedules under a fixed budget $\bar{h}$.

### Key Designs

**1. Identifying Dropout as an External Field $h$ Breaking Alignment Symmetry**

Prior work noted that dropout destroys the $c=1$ fixed point but stopped there, leaving correlation length undefined and RG analysis impossible. The authors evaluate the correlation recursion $\bar{F}_\rho$ under independent masks at $c=1$, obtaining $\bar{F}_\rho(1) = 1 - \frac{1-\rho}{\rho \bar{q}^*}\sigma_w^2 \int Dz\,\phi^2(\sqrt{\bar{q}^*}z) < 1$. They define the external field as $h \equiv 1-\bar{F}_\rho(1)$ (where $h \approx a(1-\rho)$ for weak dropout) and the order parameter as $m\equiv 1-c^*$. Taylor expanding $\bar{F}_\rho(1-m)$ near $m=0$ and substituting into the fixed-point condition $1-m = \bar{F}_\rho(1-m)$ yields a standard Landau equation:

$$h = \tfrac{g_\rho}{2}m^2 - tm,\qquad m(t,h) = \frac{t+\sqrt{t^2+2g_\rho h}}{g_\rho},$$

where $t\equiv \chi_\rho - 1$ acts as the reduced temperature. This proves that the dropout-deformed recursion **still possesses a fixed point $c^*<1$**, allowing correlation lengths and scaling laws to be well-defined.

**2. Smooth vs. Kinked Universality Classes + Two-Parameter Scaling Collapse**

The authors explain why tanh and ReLU exhibit different critical behaviors. The answer lies in the **analytic structure** of the correlation map near $c=1$. Smooth activations (tanh, GELU) satisfy Price's Theorem and allow smooth Taylor expansion at $c=1$, where the second-order term $g_\rho m^2$ dominates, yielding $m\sim\sqrt{h}$ ($\delta=2$) and $\xi\sim h^{-1/2}$. Kinked activations (ReLU), whose $\phi''$ contains a Dirac delta function, have a branch point at $c=1$, leading to $h = \kappa m^{3/2} - tm$, which gives $m\sim h^{2/3}$ ($\delta=3/2$) and $\xi\sim h^{-1/3}$. Critical exponents ($\nu_t, \beta, \theta_{\rm rel}, \gamma, \delta, \nu_\rho, \alpha$) are provided for both. Curves for $(t,h)$ within the same class collapse onto a single universal function. This incorporates engineering choices (tanh vs. ReLU) into statistical mechanics: details within a class are irrelevant, but crossing classes requires different scaling laws.

**3. Front-loaded Dropout Schedule: Saturated Steps and Regularization Reach**

With the scaling law for correlation length, dropout placement becomes an optimization problem. Let keep-probability vary by layer $\ell$. The effective inverse correlation length is $\xi_{\rm eff}^{-1} \approx \frac{1}{L}\sum_\ell \sqrt{t^2+2g_\rho h_\ell}$. At criticality ($t=0$), this simplifies to $\xi_{\rm eff}^{-1} \propto \frac{1}{L}\sum_\ell h_\ell^{1/2}$ subject to $\sum_\ell h_\ell = L\bar{h}$ and $h_\ell \leq h_{\max}$. Since $h^{1/2}$ is concave, Jensen's inequality implies that a **step function** solution—concentrating budget at the boundaries $\{0, h_{\max}\}$—is optimal. To break the degeneracy between placing the step at the front or back, the authors introduce "downstream exposure" $\mathcal{D}_\ell \approx h_\ell \xi_c\big(1-e^{-(L-\ell)/\xi_c}\big)$. Early layer masks are "seen" by more downstream layers, making the weight monotonically decreasing with $\ell$. Thus, the optimal solution is to **fill early layers first**, resulting in the front-loaded schedule.

### Loss & Training

The training objective remains unchanged; only the distribution of dropout across layers is modified. Under a fixed average budget $\bar{h}$, various schedules (constant, linear-decreasing, early-step, late-step) are compared. Theoretical MFT recursions for $\bar{F}_\rho, \chi_\rho, g_\rho$ are solved numerically via Gaussian measures as a baseline for comparison.

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

On ViT CIFAR-100, the linear-decreasing schedule achieved 49.38% vs. 48.69% for constant dropout ($p<0.05$).

### Ablation Study

| Configuration | Key Phenomenon | Description |
|------|---------|------|
| Smooth (tanh/GELU) MFT | $m\sim\sqrt{h}$ ($\delta=2$), $\xi\sim h^{-1/2}$ | Consistent with Landau $m^2$ term |
| Kinked (ReLU) MFT | $m\sim h^{2/3}$ ($\delta=3/2$), $\xi\sim h^{-1/3}$ | Dominated by branch point $m^{3/2}$ term |
| Scaling Collapse (Fig.2) | All $(t,h)$ curves collapse | Smooth class closed-form: $\tilde{m}=\sqrt{1+\tilde{t}^2}-\tilde{t}$ |
| Width $\gg$ Depth | Stable front-loading advantage | Holds within MFT validity domain |
| High dropout / Narrow net | Diminishing advantage | Occurs precisely where theory breaks down |

### Key Findings

- **ReLU MLPs** see the largest gains (+35.4% loss reduction), confirming that the kinked class's lower-order nonlinearity allows for more aggressive budget reallocation.
- Smooth classes (GELU) also show significant gains (+29.8%), proving the conclusion is universal across activation types.
- Gains on **ViT** are smaller (4–6%), which is consistent with theory: attention and skip connections alter global depth dynamics but preserve local Gaussian kernels; the "early layer" priority remains valid but with reduced magnitude.
- The advantage vanishes when dropout is increased to the point of theoretical failure (high $\bar{h}$ or narrow networks), providing **inverse evidence** supporting the theory.

## Highlights & Insights

- Mapping dropout to an "external field" $h$ and de-alignment to an "order parameter" $m$ allows the elegant application of statistical physics tools. This provides a template for treating other hyperparameters as fields.
- Universality classes are determined by the **analytic structure** of the activation (Taylor expansion vs. branch points) rather than simple scale-invariance. This explains the behavioral split between ReLU and tanh families.
- The **two-step optimization** (concave budget allocation $\to$ breaking degeneracy with monotonic weights) is highly instructive, as MFT objectives are often invariant to layer permutations.

## Limitations & Future Work

- **Forward-only MFT**: While backward gradient covariance also exhibits asymmetry due to mask independence, a full backward critical theory is not developed. Finite-width susceptibilities and representations changes during training are unmodeled.
- **Architectural Constraints**: Dropout-deformed MFT for CNNs/ResNets is discussed but lacks extensive experimentation. ViT attention in the large-width limit offers more space for refinement.
- **Initialization-time Theory**: Conclusions are based on the **initial state**; the back-reaction of representation learning on the schedule during training is not characterized.
- **Mask Correlation**: Shared masks within a batch restore the $c=1$ fixed point and weaken regularization, requiring new analysis.

## Related Work & Insights

- **vs. Schoenholz et al. (2017)**: They noted dropout destroys the $c=1$ fixed point but stopped at "vanishing criticality." This paper proves $c^*<1$ is still a fixed point, enabling scaling law analysis.
- **vs. Hayou et al. (2019)**: While they observed differences between smooth and ReLU at the edge-of-chaos, this work formalizes **critical exponents** and universality class criteria.
- **vs. Stochastic Depth / Curriculum Dropout**: These address temporal or whole-layer scheduling. This work focuses on **spatial depth-wise** dropout intensity, which is orthogonal and stackable.
- **vs. Roberts et al. (2022)**: The smooth/kinked classification here is based on analytic structure rather than scale-invariance, providing a more precise fit for dropout scaling laws.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Fully imports statistical physics universality tools to dropout scheduling and defines smooth/kinked critical exponents.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive $\bar{h}$-sweeps and ablations on MLPs/ViTs, though lacking CNN/ResNet experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear three-part derivation (Landau $\to$ Collapse $\to$ Schedule) with precise theoretical-experimental alignment.
- Value: ⭐⭐⭐⭐ The "zero-overhead" front-loaded schedule is immediately applicable; the framework lays the groundwork for studying other hyperparameters as "fields."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)
- [\[NeurIPS 2025\] Gemstones: A Model Suite for Multi-Faceted Scaling Laws](../../NeurIPS2025/llm_pretraining/gemstones_a_model_suite_for_multi-faceted_scaling_laws.md)
- [\[ICML 2026\] XTransfer: Modality-Agnostic Few-Shot Model Transfer for Human Sensing at the Edge](xtransfer_modality-agnostic_few-shot_model_transfer_for_human_sensing_at_the_edg.md)
- [\[ICML 2026\] Scaling Depth Capacity via Zero/One-Layer Model Expansion](scaling_depth_capacity_via_zeroone-layer_model_expansion.md)
- [\[NeurIPS 2025\] Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training](../../NeurIPS2025/llm_pretraining/power_lines_scaling_laws_for_weight_decay_and_batch_size_in_llm_pre-training.md)

</div>

<!-- RELATED:END -->
