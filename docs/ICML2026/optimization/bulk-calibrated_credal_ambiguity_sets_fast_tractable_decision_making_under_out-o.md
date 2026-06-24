---
title: >-
  [Paper Note] Bulk-Calibrated Credal Ambiguity Sets: Fast, Tractable Decision Making under Out-of-Sample Contamination
description: >-
  [ICML2026][Optimization][Distributionally Robust Optimization] To address the long-standing issue where placing a "Huber (linear-vacuous) contamination set" in an unbounded space causes the worst-case risk to become $+\infty$ and the DRO objective to fail, this paper proposes **bulk-calibrated credal ambiguity sets**. By learning a high-probability "bulk set" $\Xi_0$ from data, restricting the contamination budget solely within $\Xi_0$, and using moment conditions to separate…
tags:
  - "ICML2026"
  - "Optimization"
  - "Distributionally Robust Optimization"
  - "Huber Contamination"
  - "Imprecise Probabilities"
  - "credal set"
  - "Decision Robustness"
date: 2026-05-08
content_hash: cdd8cfcfba4946bc
---

# Bulk-Calibrated Credal Ambiguity Sets: Fast, Tractable Decision Making under Out-of-Sample Contamination

**Conference**: ICML2026  
**arXiv**: [2601.21324](https://arxiv.org/abs/2601.21324)  
**Code**: https://github.com/MengqiChenMC/credal-ambiguity-sets-code-repo  
**Area**: Distributionally Robust Optimization / Imprecise Probabilities / Robust Statistics  
**Keywords**: Distributionally Robust Optimization, Huber Contamination, Imprecise Probabilities, credal set, Decision Robustness

## TL;DR
To address the long-standing issue where placing a "Huber (linear-vacuous) contamination set" in an unbounded space causes the worst-case risk to become $+\infty$ and the DRO objective to fail, this paper proposes **bulk-calibrated credal ambiguity sets**. By learning a high-probability "bulk set" $\Xi_0$ from data, restricting the contamination budget solely within $\Xi_0$, and using moment conditions to separately control the tail, the authors derive a **closed-form $\text{mean}+\sup$** robust objective. This objective is fast, finite, and can be solved via common LP/SOCP for standard losses.

## Background & Motivation

**Background**: Both Imprecise Probabilities (IP) and Distributionally Robust Optimization (DRO) perform "worst-case decision making"—instead of betting on a single data distribution, they protect decisions against the worst-case member within a "set of credible distributions." When the IP credal set $\mathcal M$ equals the DRO ambiguity set $\mathcal A$, their robust objectives (worst-case expected loss) coincide, i.e., the IP **upper expectation** $\overline{\mathbb E}_{Q\in\mathcal M}[f_x(\xi)]=\sup_{Q\in\mathcal M}\mathbb E_{\xi\sim Q}[f_x(\xi)]$. Modern ML DRO typically focuses on $f$-divergence balls and Wasserstein neighborhoods due to their smooth objectives and clean duality.

**Limitations of Prior Work**: The most classic contamination model in robust statistics is **Huber $\varepsilon$-contamination**: $\tilde{\mathbb P}=(1-\varepsilon)\mathbb P^\star+\varepsilon\tilde R$, where a "clean distribution" is perturbed by an $\varepsilon$ proportion of **arbitrary vacuous (uninformative, arbitrarily placed)** distributions. However, if Huber contamination is applied to an **unbounded space $\Xi$ with an unbounded loss $f_x$**, the $\sup_{\xi}f_x(\xi)$ in the worst-case risk becomes $+\infty$, rendering the DRO objective vacuous. Existing DRO methods "with contamination" only avoid this by **forcefully assuming $\Xi$ is bounded** or restricting the function class (e.g., kernel-DRO).

**Key Challenge**: The Huber contamination's "minimal assumptions and protection against arbitrary perturbations" is its most valuable property, but this property directly conflicts with the requirement for "finite tractability in unbounded spaces"—an adversary can push the $\varepsilon$ mass to infinity to drive the loss to infinity. Duchi & Namkoong (2021) previously called for clarifying the link between "divergence ball DRO" and "classical Huber robustness"; this paper responds to that call.

**Goal**: To **translate Huber $\varepsilon$-contamination into a DRO objective that remains well-defined and computationally tractable** in unbounded continuous spaces, while preserving its IP interpretation (interpretable tolerance $\varepsilon$).

**Key Insight**: The authors observe that contamination explodes because the adversary can place mass anywhere. Thus, they **first learn a bounded "bulk set" $\Xi_0$ from data that contains the vast majority of the mass of $\mathbb P^\star$ with high probability**, restrict the contamination budget within $\Xi_0$ (so the $\sup$ is taken over a bounded set and remains finite), and then use a moment condition to separately bound the tail contributions outside $\Xi_0$.

**Core Idea**: By using a triad of "bulk set + explicit mass certificate + tail moment control," the divergent Huber-DRO is transformed into a closed-form finite objective of the form **$(1-\varepsilon)\,\text{mean}+\varepsilon\,\sup$**, where the $\sup$ over $\Xi_0$ has closed-form LP/SOCP representations for common losses.

## Method

### Overall Architecture

The overall approach is a two-layer pipeline: "define a tractable robust objective $\rightarrow$ calibrate it with data for statistical guarantees." Given a bounded bulk set $\Xi_0$, a central distribution $\mathbb P_c$ (which could be a Bayesian posterior predictive, frequentist plug-in, or empirical distribution), and a tolerance $\varepsilon$, the authors define a **support-restricted LV set** (Huber contamination set on $\Xi_0$). They prove that its worst-case risk has a closed-form $\text{mean}+\sup$ solution (Thm 2.1). Subsequently, they replace the fixed $\Xi_0$ with a **data-driven** version—parameterizing $\Xi_0$ as a level set $\{s(\xi)\leq t\}$ using a scalar scoring function $s(\cdot)$, and selecting the threshold $t$ via the DKW inequality. This provides a **finite-sample mass certificate** (Lemma 3.2) ensuring "$\mathbb P^\star(\Xi_0)\geq 1-\gamma$ holds with probability $\geq 1-\delta$," paired with a **high-probability risk certificate** (Thm 3.4) that decouples "bulk robustness" from "tail control." Finally, they discuss selecting $\varepsilon$ using a validation set when no deployment samples are available.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["i.i.d. Training Data D~P*<br/>Unbounded Space + Loss"] --> B["Bulk Set Calibration Ξ0<br/>Score s(ξ)≤t + DKW Threshold"]
    A --> C["Select Central Dist Pc<br/>Bayesian/Freq/Empirical"]
    B --> D["bulk-restricted LV Set<br/>Contamination within Ξ0"]
    C --> D
    D -->|Thm 2.1 Closed-form| E["mean+sup Robust Objective<br/>(1−ε)E[f]+ε·sup_Ξ0 f"]
    E -->|sup closed-form on Ξ0| F["LP / SOCP Decision x̂"]
    B -.Lemma 3.2 Mass Certificate<br/>+ Thm 3.4 Risk Certificate.-> E
```

### Key Designs

**1. Bulk-restricted LV set: Confining contamination to a bounded bulk to force a closed-form mean+sup objective**

This is the foundation of the paper, directly addressing the "unbounded space $\rightarrow$ infinite risk" pain point. The authors define the ambiguity set as a **support-restricted linear-vacuous (LV) set** that only allows Huber contamination on $\Xi_0$:

$$\mathcal A^{\operatorname{LV}}_{\varepsilon,\Xi_0}(\mathbb P_{c,\Xi_0}):=\{(1-\varepsilon)\mathbb P_{c,\Xi_0}+\varepsilon R:R\in\mathcal P(\Xi_0)\},$$

where $\mathbb P_{c,\Xi_0}$ is the normalized truncation of the central distribution on $\Xi_0$. Because the adversary's $\varepsilon$ mass must stay within the **bounded** $\Xi_0$, $\sup_{\xi\in\Xi_0}f_x(\xi)$ remains finite. Thm 2.1 yields a clean closed-form worst-case risk:

$$\mathcal R(f_x)=(1-\varepsilon)\,\mathbb E_{\xi\sim\mathbb P_{c,\Xi_0}}[f_x(\xi)]+\varepsilon\,\sup_{\xi\in\Xi_0}f_x(\xi),$$

which corresponds to the upper expectation of the $\varepsilon$-contamination model in IP. Cor 2.2 further notes that the worst-case distribution places all $\varepsilon$ mass on the point in $\Xi_0$ that maximizes the loss. This $\text{mean}+\sup$ form preserves the "worst-case arbitrary perturbation" spirit of Huber while remaining naturally tractable.

**2. Unifying the LV set into divergence balls and contamination neighborhoods**

To satisfy the DRO community, the authors prove this contamination set **is a forward-LV divergence ball**: Prop 2.3 shows $\mathcal A^{\operatorname{LV}}_{\varepsilon,\Xi_0}$ is exactly $\{Q:\operatorname{LV}(Q,\mathbb P_{c,\Xi_0})\leq\varepsilon\}$, where the LV distortion is $\operatorname{LV}(Q,\mathbb P):=\sup_{A:\mathbb P(A)>0}\frac{\mathbb P(A)-Q(A)}{\mathbb P(A)}$. Based on this, they link three types of contamination neighborhoods: **forward-LV** corresponds to "adding $\varepsilon$ mass to the worst state," with risk $(1-\varepsilon)\mathbb E[f]+\varepsilon\sup f$; **reverse-LV** corresponds to "removing $\varepsilon$ low-loss mass from $\mathbb P$," with risk exactly $\mathrm{CVaR}^{\mathbb P}_{1-\varepsilon}(f)$ (explaining outlier-robust DRO); the **symmetric TV ball** involves both, $\mathcal R=(1-\varepsilon)\mathrm{CVaR}^{\mathbb P}_{1-\varepsilon}(f)+\varepsilon\sup f$ (Prop 2.4). The authors provide a new proof for the TV objective using the similarity decomposition of total variation, clarifying the intrinsic links between these neighborhoods. The value of this design is that it puts "adding adversarial mass (forward)" and "removing low-loss mass (reverse)" on the same coordinate system, giving the "IP credal set $\leftrightarrow$ DRO objective" translation an interpretable tolerance scale.

**3. Data-driven bulk calibration: Score level sets + DKW certificates for tractability and finite-sample guarantees**

While $\Xi_0$ is assumed given above, in practice, it must be learned reliably. Naive "three-sigma truncation" confidence sets may provide a bounded $\Xi_0$ but **cannot prove** $\mathbbmathbb P^\star(\Xi_0)\geq 1-\gamma$, leaving the tail mass uncontrolled. The authors restrict $\Xi_0$ to a **level set of a scalar score** $\Xi_0(t)=\{\xi:s(\xi)\leq t\}$ (e.g., Mahalanobis ellipsoid $\|\Sigma_{\rm fit}^{-1/2}(\xi-\mu_{\rm fit})\|_2$ or box $\max_i|\xi_i-\mu_{\rm fit,i}|/w_i$), splitting samples into $\mathcal D_{\rm fit}$ (to fit $s$) and $\mathcal D_{\rm select}$ (to select the threshold). Using the **Dvoretzky–Kiefer–Wolfowitz (DKW)** inequality on the empirical CDF $F_m$ of the selection scores, they construct a one-sided lower envelope $L^{\rm DKW}(t)=[F_m(t)-r_{m,\delta}]_+$ (where $r_{m,\delta}=\sqrt{\frac{1}{2m}\log\frac{2}{\delta}}$). Selecting the minimum threshold that clears the target mass, $\hat t_{\rm DKW}=\inf\{t:L^{\rm DKW}(t)\geq 1-\gamma\}$, ensures (Lemma 3.2):

$$\Pr\{\mathbb P^\star(\Xi_0(\hat t_{\rm DKW}))\geq 1-\gamma\}\geq 1-\delta.$$

Calibration only takes $O(m\log m)$. This level-set approach is a win-win: it preserves closed-form $\sup_{\xi\in\Xi_0}f_x(\xi)$ (see table below) and provides finite-sample mass certificates (similar to conformal prediction, but for high-probability bulk mass). Thm 3.4 finally decomposes the true risk under deployment contamination $\tilde{\mathbb P}=(1-\varepsilon^\star)\mathbb P^\star+\varepsilon^\star\tilde R$ into a **"bulk LV objective + tail moment term"**: $\mathbb E_{\tilde{\mathbb P}}[f_x]\leq(1-\varepsilon_{\rm eff})\mathbb E_{\mathbb P_{c,\Xi_0}}[f_x]+\varepsilon_{\rm eff}\sup_{\Xi_0}f_x+M_p(x)\cdot(\cdots)^{1/q}$, where the effective tolerance $\varepsilon_{\rm eff}$ combines "central mismatch $\varepsilon_c$" and "deployment contamination within the bulk $\varepsilon^\star\rho_{\Xi_0}$," with the tail controlled by the $p$-th moment $M_p(x)$.

### Loss & Training: sup Closed-forms and Tractable Forms

For ellipsoid/box bulk geometries, $\sup_{\xi\in\Xi_0(t)}f_x(\xi)$ has closed-forms for common losses (derived from support functions in convex optimization), allowing the $\text{mean}+\sup$ objective to be written as LP/SOCP:

| Loss $f_x(\xi)$ | $\sup$ over Ellipsoid $\Xi_0^{\rm ellip}(t)$ | $\sup$ over Box $\Xi_0^{\rm box}(t)$ |
|------|------|------|
| Linear $a_x^\top\xi+b_x$ | $C_x+t\,m_2(a_x)$ | $C_x+t\,m_1(a_x)$ |
| ReLU $\max\{0,a_x^\top\xi+b_x\}$ | $\max\{0,C_x+t\,m_2(a_x)\}$ | $\max\{0,C_x+t\,m_1(a_x)\}$ |
| Abs $|a_x^\top\xi+b_x|$ | $|C_x|+t\,m_2(a_x)$ | $|C_x|+t\,m_1(a_x)$ |
| Piecewise Linear $\max_j\{a_{x,j}^\top\xi+b_{x,j}\}$ | $\max_j\{a_{x,j}^\top\mu_{\rm fit}+b_{x,j}+t\,m_2(a_{x,j})\}$ | Same as left, replace $m_2$ with $m_1$ |

Where $C_x=a_x^\top\mu_{\rm fit}+b_x$, $m_2(a_x)=\|\Sigma_{\rm fit}^{1/2}a_x\|_2$, and $m_1(a_x)=\sum_i w_i|a_{x,i}|$. The decision is $\hat x\in\arg\min_x\mathcal R(f_x)$.

## Key Experimental Results

Three experiments using **Bayesian, frequentist, and empirical** central distributions verify the robust-accuracy trade-offs and solving speeds of LV under contamination.

### Main Results

| Experiment | Center / Scenario | Key Results |
|------|-------------|----------|
| Heavy-tailed Newsvendor (Synthetic, Student-$t$, $\nu=3$, $d=5$) | Bayesian Posterior Predictive | Under contamination ($\varepsilon_{\rm cont}=0.1,0.2$), LV achieves the strongest mean–variance frontier and lowest MSD; runtime **1.3s** vs KL-BDRO 2.4s, OR-WDRO 23.5s. |
| California Housing Regression (East$\rightarrow$West drift, 30% gap) | Freq. Gaussian Copula | **Ours** outperforms in all four metrics. Compared to the strongest baseline, Wasserstein: MAE $\downarrow$**13%**, $p_{98}$ and $\mathrm{CVaR}_{2\%}$ both $\downarrow \sim$**10%**; total time 1.44s vs Wass 6.16s. |
| CivilComments classification (WILDS subgroup drift, 16 slices) | Empirical (LV-Group extension) | Avg accuracy **0.828** / Worst-group **0.516**, both higher than GroupDRO's 0.770 / 0.456. |

### Key Findings

| Configuration | Phenomenon | Explanation |
|------|------|------|
| LV vs KL-BDRO | KL **saturates** at large $\varepsilon_{\rm KL}$ (points for 5, 10 overlap) | KL is limited by finite scenarios; LV continues to trade OOS mean for variance. |
| LV Sample Efficiency | Performance stabilizes with only **50** truncated samples | KL-based Bayesian methods require much larger SAA budgets to stabilize; this is why LV is the fastest. |
| Clean Scenarios | LV is conservative when $\varepsilon_{\rm LV}$ is large | OR-WDRO/KL-BDRO have lower MSD here, but LV's optimal MSD remains close to the best baseline. |
| Tail Budget $\gamma$ | Performance stable within reasonable ranges | Recommended to pick slightly above the min certifiable value, leaving $\sim 10$ scores outside the bulk ($\gamma\approx r_{m,\delta}+10/m$). |

- **Highlights**: LV is not only the most robust under contamination but also the **fastest to solve** (approx. 18x faster than OR-WDRO in the newsvendor task) due to it being a sample-efficient truncated SAA.
- **Bulk Geometry Matters**: In regression where $Y\mid X$ changes, using a $\Xi_{0,X}\times\Xi_{0,Y}$ block-decoupling of covariate geometry and outcome volatility works best; joint ellipsoids hardcode training-phase dependencies, while boxes are too conservative. Rule of thumb: default to Mahalanobis, use blocks for heterogeneous dimensions, and boxes only for coordinate-independent adversarial shifts.

## Highlights & Insights
- **"Bulk set + explicit mass certificate" is the key to taming divergent contamination objectives**: Instead of assuming $\Xi$ is bounded, **learn a high-probability bounded subset that captures most mass**, confine contamination within it, and control the tail separately. This "bulk-tail separation" is generalizable to any robust problem where an adversary can exploit infinity.
- **Unifying forward-LV / reverse-LV / TV neighborhoods**: Mapping "adding adversarial mass = sup" and "removing low-loss mass = CVaR" into a single framework provides a common language for Huber robustness and outlier-robust DRO, offering insights into the "geometric origins" of DRO objectives.
- **DKW calibration similar to conformal prediction but for bulk-mass**: Using score level sets and DKW lower envelopes for thresholding allows $O(m\log m)$ finite-sample mass certificates. It also supports block-calibration ($\sum\gamma_i\leq\gamma,\sum\delta_i\leq\delta$), making it plug-and-play for engineering.

## Limitations & Future Work
- **Unidentifiability of $\varepsilon$**: The true deployment contamination rate $\varepsilon^\star$ cannot be identified from training data; **ours** treats $\varepsilon$ as a **robustness budget** tuned via validation (geo-block CV / minimax validation). Without deployment samples, Thm 3.4 is a structural decomposition where $\varepsilon^\star$ and $M_p$ are unobserved.
- **Dependence on Bulk Geometry**: The choice of $\Xi_0$ (ellipsoid/box/blocks) is a critical modeling decision; incorrect choices (e.g., joint ellipsoids for regression) can lead to failure. Currently relies on rules of thumb rather than automated selection.
- **Moment Conditions**: Risk certificates require finite $p > 1$ order moments ($M_p(x)<\infty$), which is unsuitable for extremely heavy-tailed or Cauchy distributions.
- **Potential Over-conservativeness**: At large $\varepsilon$, LV can be overly conservative, requiring careful parameter tuning.

## Related Work & Insights
- **vs KL-BDRO / KL-BAS (Shapiro 2023; Dellaporta 2025)**: These use KL divergence balls. Their objectives are smooth but **saturate** at large radii and require large SAA budgets. **Ours** uses LV (Huber) contamination, providing a closed-form $\text{mean}+\sup$ objective that is sample-efficient and allows continuous mean-variance trade-offs.
- **vs OR-WDRO (Nietert et al. 2023, outlier-robust Wasserstein)**: This corresponds to the reverse-LV (CVaR) line of removing low-loss mass. **Ours** uses forward-LV to add adversarial mass and solves an order of magnitude faster (18x). This paper unifies forward/reverse/TV into one picture.
- **vs GroupDRO (Sagawa 2020)**: This paper proposes an **LV-Group** extension—layering $\varepsilon$-contamination (IP discounting) over the GroupDRO mixture credal set. This protects against both group proportion shifts and uncaptured intra-group contamination, outperforming GroupDRO on CivilComments in both average and worst-group accuracy.
- **vs TRO (Tsang & Shehadeh 2025)**: TRO also resembles $\text{mean}+\sup$ but is based on empirical risk and treats the mixing weight as a conservatism parameter; the $\varepsilon$ in **ours** has a clear Huber contamination/IP unreliability interpretation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Uses "data-learned bulk set + explicit mass certificates" to tame the divergent Huber-DRO, unifying three contamination neighborhoods and addressing a major open challenge in DRO.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three types of centers $\times$ synthetic and two real-world tasks, including runtime and sample efficiency analysis. Good coverage, though limited to linear predictors.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical progression from definitions to certificates; the IP $\leftrightarrow$ DRO correspondence is explained clearly with operational rules of thumb.
- Value: ⭐⭐⭐⭐ Provides a fast, guaranteed landing solution for "DRO with Huber contamination," particularly useful for robust decision-making and sub-group fairness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Gen-DFL: Decision-Focused Generative Learning for Robust Decision Making](../../ICLR2026/optimization/gen-dfl_decision-focused_generative_learning_for_robust_decision_making.md)
- [\[ICML 2026\] ePC: Fast and Deep Predictive Coding in Digital Simulation](epc_fast_and_deep_predictive_coding_in_digital_simulation.md)
- [\[ICLR 2026\] Distributionally Robust Optimization via Generative Ambiguity Modeling](../../ICLR2026/optimization/distributionally_robust_optimization_via_generative_ambiguity_modeling.md)
- [\[ICML 2026\] Probing Neural TSP Representations for Prescriptive Decision Support](probing_neural_tsp_representations_for_prescriptive_decision_support.md)
- [\[ICML 2026\] Full-Batch Gradient Descent Outperforms One-Pass SGD: Sample Complexity Separation in Single-Index Learning](full-batch_gradient_descent_outperforms_one-pass_sgd_sample_complexity_separatio.md)

</div>

<!-- RELATED:END -->
