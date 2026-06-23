---
title: >-
  [Paper Note] Homeostatic Adaptation of Optimal Population Codes under Metabolic Stress
description: >-
  [ICLR 2026][Others][Paper Note] This paper supplements the classic "Optimal Population Coding" theory with two overlooked biological constraints—firing rate homeostasis and a direct ATP-linked energy budget. It is the first to mathematically predict the "flattening of tuning curves" observed in the mouse visual cortex under metabolic stress while uni
tags:
  - ICLR 2026
  - Others
date: 2026-05-08
content_hash: 0a9523f6180a872c
---
# Homeostatic Adaptation of Optimal Population Codes under Metabolic Stress

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=LMvlwGkXpX](https://openreview.net/forum?id=LMvlwGkXpX)  
**Code**: To be confirmed  
**Area**: Computational Neuroscience / Optimal Population Coding  
**Keywords**: Efficient coding, Population coding, Fisher information, Energy constraints, Firing rate homeostasis, Biophysical simulation  

## TL;DR
This paper supplements the classic "Optimal Population Coding" theory with two overlooked biological constraints—firing rate homeostasis and a direct ATP-linked energy budget. It is the first to mathematically predict the "flattening of tuning curves" observed in the mouse visual cortex under metabolic stress while unifying previously contradictory models as special cases of its own framework.

## Background & Motivation
**Background**: The efficient coding hypothesis (Barlow 1961) posits that sensory neuron populations are optimized to maximize information about the environment under resource constraints. Over the past half-century, this idea has developed into a suite of optimal population coding models based on Fisher Information (FI). The gain/density parametrization + tiling assumption by Ganguli & Simoncelli (2010) serves as the standard tool: using a gain function $g(s)$ and a density function $d(s)$ to modulate a basis tuning curve to compactly describe heterogeneous populations.

**Limitations of Prior Work**: Existing models characterize "energy constraints" too simplistically—either constraining the average firing rate (Ganguli & Simoncelli 2010) or the maximum firing rate (Wang et al. 2016a). However, energy expenditure in real neurons is not dominated by spikes: action potentials account for only 22% of ATP consumption, while postsynaptic receptor currents account for 50%, and reversing sodium leaks via the Na⁺/K⁺ pump accounts for 20%. More critically, experimental findings by Padamsey et al. (2022) revealed that cortical neurons in mice under chronic caloric restriction enter a "low-power mode": saving energy while **maintaining nearly constant spike counts**, at the cost of increased noise and flattened tuning curves.

**Key Challenge**: This experimental fact challenges existing theories. Models constraining average firing rates predict that tuning curves will "shorten," while models constraining maximum firing rates predict they will "widen." Real data shows both widening and shortening (flattening) simultaneously, with firing rate conservation. No existing optimal coding model can replicate both phenomena.

**Goal**: To establish a population coding framework that is both analytically solvable and directly linked to ATP consumption, providing the first accurate mathematical description of how neurons optimally adapt under metabolic stress.

**Core Idea**: The authors argue that the theory can be fixed by adding two "surprisingly simple" constraints: **① An approximation of firing rate homeostasis** (forcing every neuron's expected firing rate to be conserved), and **② An energy budget linked to noise levels and calibrated by biophysical simulations**. These constraints do not increase analytical complexity; instead, they serve as a natural generalization of existing models.

## Method

### Overall Architecture
The framework consists of two layers: The upper layer is an **analytical population coding optimization problem**—maximizing the expected Fisher Information under the constraints of firing rate homeostasis and an energy budget. Utilizing the tiling approximation of Ganguli & Simoncelli, this is simplified into a solvable optimization for $g(s)$ and $d(s)$, with closed-form solutions provided via the Lagrangian method. The lower layer is a **biophysical simulation** (stochastic Hodgkin-Huxley single-compartment neurons using the NEURON simulator). It does not introduce new equations but provides numerical calibration for the abstract parameters—the energy exponent $\alpha$ and the noise dispersion factor $\eta_\kappa(E)$—anchoring the "abstract energy budget $E$" to the physical quantity of "ATP/second."

```mermaid
flowchart TD
    A[生物物理仿真<br/>随机HH单房室神经元] -->|扫描 vrest/gleak/gsyn| B[噪声σ²与能量εtotal]
    B -->|沿等能量线取最低噪声| C[最优衰减路径 δopt]
    C -->|拟合| D["标定 α=1<br/>离散因子 ηκ(E)=η0+c1/(E−c2)"]
    D --> E[解析群体编码优化]
    E -->|稳态约束 + 能量约束 + tiling| F[闭式解 g(s),d(s),FI(s)]
    F --> G[预测调谐曲线变平<br/>复现 Padamsey 2022]
```

### Key Designs

**1. Energy-dependent discrete Poisson noise: Writing noise as a tunable multiple of the firing rate.** Standard optimal coding assumes Poisson noise (variance = mean), but metabolic stress systematically amplifies noise. Starting from simulations, the authors found that spike count variance follows a parabolic relationship with the mean: $\sigma_F^2=\eta_\kappa\,\mu_F(1-\mu_F)$ (Bernoulli shape), where the scaling factor $\eta_\kappa$ increases with metabolic stress. They thus generalized population noise to "discrete Poisson": $\mathrm{Var}(h_n(s);E)=\eta_\kappa(E)\,h_n(s)$, where $\eta_\kappa(E)$ is a dispersion factor regulated by the energy budget. This term allows Fisher Information $\mathrm{FI}(s;E)=\sum_n h_n'(s)^2/\mathrm{Var}(h_n(s);E)$ to degrade with energy changes, bridging "energy saving" and "increased noise."

**2. Approximate firing rate homeostasis constraint: Elevating individual conservation to population continuity via tiling.** Strict homeostasis is $\int p(s)h_n(s)\,ds=R_n$. Under the gain/density parametrization $h_n(s)=g(s)\hat h(D(s)-D(s_n))$, the authors approximate each Gaussian tuning curve as a rectangle of height $g(s_n)$ and width $1/d(s_n)$. Assuming $p(s)$ varies more slowly than any tuning curve, the homeostasis constraint is reduced to an elegant continuous equality: $p(s)g(s)=R(s)d(s)$. The physical meaning is straightforward: to save energy, the gain $g$ must decrease (corresponding to reduced synaptic conductance), forcing the density $d$ (tuning width) to undergo compensatory changes—translating the mechanism of "reduced synaptic conductance leading to energy savings and wider tuning" in Padamsey's data.

**3. Generalized energy constraint + analytical closed-form solution: A single exponent $\alpha$ unifying prior models.** The optimization above is unbounded without a third constraint. The authors propose a generalized energy constraint $\int p(s)g(s)^\alpha\,ds=E$ ($\alpha\ge 1$). Substituting the homeostasis constraint and using the Lagrangian method for the infomax case ($f=\log x$) yields a remarkably simple solution: $g(s)=E^{1/\alpha}$, $d(s)=E^{1/\alpha}p(s)/R(s)$, and $\mathrm{FI}\propto\eta_\kappa(E)^{-1}E^{3/\alpha}$. This $\alpha$ is the key to unification: setting $\alpha=1$ and redefining $E$ as the average firing rate reduces the constraint to Ganguli & Simoncelli (2010); setting $\alpha=3/2$ reduces it to Wang et al. (2012/2016a). Proving that previously contradictory sets of conclusions are slices of the same framework at different $\alpha$ values.

**4. Noise-optimal decay path: Using simulation to fix $\alpha$ and $\eta_\kappa$ as measurable physical quantities.** The authors scanned the three variable cellular parameters identified by Padamsey (resting potential $v_{rest}$, leak conductance $g_{leak}$, and synaptic conductance $g_{syn}$) in simulations. They first sliced the "fixed spike count" plane using homeostasis and then hypothesized that **for each energy budget, the cell degrades along a path that minimizes noise**: $\delta_{opt}(\epsilon;\kappa)=\arg\min_{v_{rest},g_{leak}}\eta_\kappa\ \text{s.t.}\ \epsilon_{total}=\epsilon$. Along this optimal path, they found an affine relationship between optimal density and energy, **analytically deriving $\alpha=1$**. The noise dispersion factor was fitted as $\eta_\kappa(E)=\eta_0+\frac{c_1}{E-c_2}$, providing an explicit trade-off between energy reduction and noise increase. This step assigns units of ATP/second to energy budget $E$, making the theory a testable prediction for patch-clamp experiments.

## Key Experimental Results

This is a theoretical and simulation-based work; "experiments" refer to consistency checks against mouse data from Padamsey et al. (2022) and comparisons with two types of prior models.

### Main Results: Consistency with Mouse Cortical Data

| Method | Tuning Width Change | Firing Rate Change (Homeostasis Violation) | ATP Consumption Change | Matches data |
|------|------------|--------------------|------------|------------|
| Mouse L2/3 data (Padamsey 2022) | +32% (widen) | ≈0% (Not statistically significant) | −29% | — |
| **Ours (infomax, α=1)** | **+32%** | **0% (Exact homeostasis)** | **−29%** | ✅ Replicates flattening and energy saving |
| Ganguli & Simoncelli (2010) | Fixed width (shorten) | −24% | — | ❌ Violates homeostasis |
| Wang et al. (2016a) | +32% (widen) | +32% | — | ❌ Violates homeostasis |

Under infomax + uniform prior, where the homeostasis approximation is exact, the framework yields a **perfect** tuning curve match.

### Key Findings
- **Unity**: Proven by $\alpha=1$ (reducing to Ganguli & Simoncelli) and $\alpha=3/2$ (reducing to Wang et al.’s FI capacity constraint), demonstrating the framework is a generalized superset of existing models.
- **Scaling Laws**: Analytically derived that density $d$ and gain $g$ both scale by $E^{1/\alpha}$, while Fisher Information scales by $\eta_\kappa(E)^{-1}E^{3/\alpha}$.
- **Homeostasis as a signature of optimality**: The authors proved that even if Ganguli & Simoncelli (2010) did not explicitly require homeostasis, it emerges naturally as a byproduct of their model's optimal solution.
- **Robustness**: The analytical solution is robust to "differential correlations" and holds even when noise correlations change significantly across different metabolic states.

## Highlights & Insights
- **Simplicity over complexity**: The core contribution is identifying that "two simple constraints" are sufficient to repair the theory. These biological constraints, rather than complicating analysis, become the key to unifying prior work.
- **Bidirectional anchoring**: The abstract parameters $\alpha$ and $\eta_\kappa$ are grounded in biophysical simulations of ATP/second and electrical properties, allowing the theoretical framework to generate predictions falsifiable by patch-clamp experiments.
- **Unified Narrative**: By explaining "shortening vs widening" as special cases of the same framework with different $\alpha$ values, the model provides a unified mathematical account for the partial successes of previous models.
- **ATP-level energy budget**: For the first time, non-spike factors like resting leak currents and postsynaptic currents—which dominate ATP overhead—are included in the energy constraints of optimal coding, correcting the long-standing simplification that "spikes dominate energy consumption."

## Limitations & Future Work
- **Dependency on a single data source**: The calibration and validation are heavily tied to the mouse data from Padamsey et al. (2022); generalization requires testing against more metabolic stress datasets.
- **Boundaries of tiling + Gaussian approximation**: The homeostasis approximation is exact only for Gaussian, tiling, and slowly varying $p(s)$; solvable approximations for other families (sigmoid, Gabor, etc.) are currently absent.
- **Simplification of single-compartment simulation**: The biophysical simulation uses a single-compartment HH model, failing to capture high-fidelity cellular behaviors like dendritic processing.
- **Time-scale issues**: Changes in Padamsey's study occur over "weeks," but some metabolic stress responses might be faster. This opens a new problem space for defining optimality based on long-term survival strategies rather than instantaneous snapshots.

## Related Work & Insights
- **Efficient Coding Lineage**: Directly builds on the lineage of Barlow (1961) → Ganguli & Simoncelli (2010, 2014) → Wang et al. (2012, 2016a/b), generalizing Fisher Information-based optimal coding from "single metabolic state" to "variable metabolic states."
- **Constraint Classification**: Previous constraints were categorized as energy (mean/max rate), coding capacity ($\sum \sqrt{\mathrm{FI}}$), or population size; the energy constraint here unifies the first two for specific $\alpha$ values.
- **Biological Basis**: The model is grounded in Harris et al. (2012) for ATP accounting, Padamsey et al. (2022) for low-power modes, and Hengen et al. (2013) for firing rate homeostasis.
- **Insight**: This dual-layer paradigm of "abstract optimization + biophysical calibration" can be transferred to any scenario seeking to link information-theoretic optimality to measurable physiological quantities. In NeuroAI, it suggests re-evaluating old data as slices of different energy budgets rather than disparate strategies.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First to replicate "flattening + firing rate homeostasis" using two simple constraints while unifying prior models.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Solid simulation calibration and strong controls, though verification relies heavily on a single experimental source.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — exceptionally clear narrative; the transition between unification arguments and biophysical calibration is seamless.
- **Value**: ⭐⭐⭐⭐⭐ — Fills a long-standing gap in metabolic dynamics within optimal population coding and offers falsifiable predictions for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Optimal Welfare in Noncooperative Network Formation under Attack](../../AAAI2026/others/optimal_welfare_in_noncooperative_network_formation_under_attack.md)
- [\[ICLR 2026\] An Information-Theoretic Framework For Optimizing Experimental Design To Distinguish Probabilistic Neural Codes](an_information-theoretic_framework_for_optimizing_experimental_design_to_disting.md)
- [\[ICLR 2026\] Prior-Free Tabular Test-Time Adaptation](prior-free_tabular_test-time_adaptation.md)
- [\[ICLR 2026\] Adaptive Conformal Guidance for Learning under Uncertainty](adaptive_conformal_guidance_for_learning_under_uncertainty.md)
- [\[ICLR 2026\] PriorGuide: Test-Time Prior Adaptation for Simulation-Based Inference](priorguide_test-time_prior_adaptation_for_simulation-based_inference.md)

</div>

<!-- RELATED:END -->
