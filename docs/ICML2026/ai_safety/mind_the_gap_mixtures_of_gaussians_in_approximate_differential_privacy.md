---
title: >-
  [Paper Note] Mind the Gap: Mixtures of Gaussians in Approximate Differential Privacy
description: >-
  [ICML 2026][AI Safety][Paper Note] This paper designs a class of Gaussian mixture additive noise mechanisms (multi-Gaussian mixture and hyperparameter-free quasi-Gaussian mixture) for $(\varepsilon,\delta)$-DP. These mechanisms close the optimality gap of the analytic Gaussian mechanism by up to 99% in low-to-medium privacy regimes while preserving the
tags:
  - ICML 2026
  - AI Safety
date: 2026-05-08
content_hash: a87a37b13955755f
---
# Mind the Gap: Mixtures of Gaussians in Approximate Differential Privacy

**Conference**: ICML 2026  
**arXiv**: [2605.28078](https://arxiv.org/abs/2605.28078)  
**Code**: https://github.com/selvi-aras/MindTheGap  
**Area**: AI Security / Differential Privacy  
**Keywords**: Approximate Differential Privacy, Gaussian Mixture Mechanism, Additive Noise, Low-to-Medium Privacy Regime, zCDP Composition

## TL;DR
This paper designs a class of Gaussian mixture additive noise mechanisms (multi-Gaussian mixture and hyperparameter-free quasi-Gaussian mixture) for $(\varepsilon,\delta)$-DP. These mechanisms close the optimality gap of the analytic Gaussian mechanism by up to 99% in low-to-medium privacy regimes while preserving the tight zCDP composition properties of Gaussians.

## Background & Motivation

**Background**: Approximate differential privacy $(\varepsilon,\delta)$-DP is the de facto industrial standard (used in the 2020 US Census, Google VaultGemma LLM, Opacus, etc.). Common implementations include the Dwork-Roth Gaussian mechanism $\sigma=\sqrt{2\log(1.25/\delta)}(\Delta/\varepsilon)$ and the "analytic Gaussian mechanism" numerically tightened via binary search by Balle-Wang 2018. Gaussians are widely used because they have unbounded support (no distinguishing events), follow an approximate 3σ empirical rule, and can be integrated into the zCDP framework for lossless composition.

**Limitations of Prior Work**: DP theoretical analysis almost exclusively targets the asymptotic high-privacy regime where $\varepsilon,\delta \downarrow 0$. However, real-world deployments often fall into the low-to-medium privacy regime where $\varepsilon \geq 1$ (e.g., VaultGemma $\varepsilon=2$, Opacus tutorials $\varepsilon=50$, industry $\varepsilon$ is typically large, while $\delta$ must be cryptographically small). The numerical optimization framework of Selvi et al. 2025 proved that in these regimes, the expected loss of the analytic Gaussian can be up to 700% suboptimal.

**Key Challenge**: The "unimodal + unbounded support" property of the Gaussian mechanism makes it easy to compose and analyze. However, the numerical optimal results from Selvi et al. 2025 show that the truly optimal noise distribution is **not unimodal**; instead, it features peaks at every interval of length $\Delta$, with a density ratio between adjacent peaks of approximately $e^{\varepsilon}$. Furthermore, Rinberg et al. 2025 proved that no **unimodal** generalized Gaussian can outperform the standard Gaussian, effectively locking the path for improvement to "multimodal" distributions.

**Goal**: To construct a **multimodal** Gaussian-like mechanism that recovers the optimality gap of the analytic Gaussian in the $\varepsilon \geq 1$ range, while maintaining Gaussian tails and compatibility with tight zCDP composition.

**Key Insight**: Encode the two empirical laws from numerical optimal results—"$\Delta$-periodic peaks + $e^{-\varepsilon}$ proportional decay"—directly into the distribution structure. Using the analytic Gaussian as a backbone, convolve it with multiple Gaussian components of the same variance, centered at $k\Delta$ and weighted by $e^{-|k|\varepsilon}$.

**Core Idea**: Replace a single Gaussian with a convex combination of "zero-mean Gaussian + several Gaussians shifted by $\pm k\Delta$" as DP noise to approximate the optimal multimodal density shape. Using identical variances ensures that the zCDP composition constant remains **exactly the same** as a single Gaussian; multimodality is used solely to reduce expected loss without increasing composition costs.

## Method

### Overall Architecture

The problem to be solved is: given the sensitivity $\Delta$ of a query function $q:\mathcal{D}\to\mathbb{R}$ and a privacy budget $(\varepsilon,\delta)$, find the smallest possible $\sigma$ such that the additive noise mechanism $\mathcal{A}(D)=q(D)+\tilde{X}$ satisfies $(\varepsilon,\delta)$-DP with minimal expected noise loss. The paper shifts the focus from "tuning $\sigma$ in a single Gaussian" to "designing a noise distribution $\tilde{X}$ within a family of **multimodal Gaussian mixtures**." Building on the analytic Gaussian backbone, the distribution is augmented with shifted Gaussian components following two geometric laws from numerical optima ($\Delta$-periodic multiple peaks and adjacent peak density ratios of approximately $e^{\varepsilon}$). Along this line, the paper introduces two classes of mechanisms (the lowest-loss but hyperparameter-dependent multi-Gaussian and the hyperparameter-free, lightweight quasi-Gaussian) and proves that both fit into the zCDP framework for lossless tight composition.

### Key Designs

**1. Multi-Gaussian mixture: Encoding optimal geometry into closed-form distributions**

This addresses the pain point that the numerically optimal noise distributions found by Selvi et al. 2025, while low in loss, lack closed forms, cannot be sampled, and lack analytically tractable moments, making them unusable in engineering. This paper fits those geometric features using a convex combination of $2K+1$ **homoscedastic** Gaussians with density $f_{\mathrm{m}}(x;\sigma,K)=\frac{1}{c_K}\sum_{k=-K}^{K}e^{-|k|\varepsilon}\phi(x;k\Delta,\sigma)$. The $k$-th component center is placed at $k\Delta$ (replicating $\Delta$-periodic peaks), and the weights are $\propto e^{-|k|\varepsilon}$ (replicating geometric decay). The difficulty lies in verifying $(\varepsilon,\delta)$-DP for the uncountable family of neighbor shifts $\varphi\in[0,\Delta]$; Theorem 3.2 introduces a discretization parameter $\eta\in(0,1)$ to relax constraints to a finite grid $\{0,\beta,2\beta,\ldots,\Delta\}$ (step size $\beta\leq\sqrt{2\pi}\eta\sigma\delta$) while compressing the right-hand $\delta$ to $(1-\eta)\delta$ as compensation for leakage. This transforms the "infinite condition" into a computable certificate ($\eta$ approaching zero recovers the original definition). Lemma 3.4 proves this sufficient condition is monotonic in $\sigma$, allowing Algorithm 1 to use the $\sigma_g$ of the analytic Gaussian as a right bound for binary search, returning the tightest $\sigma$ under this relaxation in $\mathcal{O}\!\left(\frac{K^2}{\eta\delta}(\log(1+1/\varepsilon)+\log(1+\log 1/\delta))\right)$ time. This is effective because the multimodal geometry aligns the density shape with the true optimum, while the closed-form Gaussian mixture retains the convenience of sampling, moment calculation, and analysis.

**2. Quasi-Gaussian mixture: Eliminating hyperparameters and reducing $1/\delta$ to $\log 1/\delta$**

While the multi-Gaussian achieves minimal loss, Algorithm 1's complexity involves $K^2/(\eta\delta)$, which becomes expensive as $\delta$ decreases, making it unsuitable for repeated budget scans. The Quasi-Gaussian compresses multimodality into a single expression using one zero-mean Gaussian (weight $e^{\varepsilon}$) plus a "pseudo-Gaussian" where $|x|$ replaces $x$ (providing two peaks at $\pm\Delta$ via the absolute value, weight $1$): $f_{\mathrm{q}}(x;\sigma)=\frac{e^{\varepsilon}}{c}\exp(-x^2/(2\sigma^2))+\frac{1}{c}\exp(-(|x|-\Delta)^2/(2\sigma^2))$, requiring neither $K$ nor $\eta$. Theorem 4.2 analytically decomposes the DP conditions into two paths and takes $\sigma=\max(\sigma_1,\sigma_2)$: $\sigma_1$ handles $\delta$ leakage via a closed-form inequality $h_1(\sigma)+h_2(\sigma)\geq 0$ (involving $\Phi$ functions and $e^{2\varepsilon},e^{\varepsilon}$ terms); $\sigma_2$ handles pointwise amplification via the constraint $\max_{x\in[0,\Delta]}f_{\mathrm{q}}/\min_{x\in[0,\Delta]}f_{\mathrm{q}}\leq e^{\varepsilon}$. Lemmas 4.3–4.5 prove monotonicity for both paths and provide search bounds ($\sigma_1\leq\sqrt{2(\varepsilon-\log\delta)}\Delta/\varepsilon$, $\sigma_2\leq\sqrt{\Delta^2/(2\varepsilon)}$). Lemma 4.4 simplifies the $\max/\min$ search to two unimodal sub-intervals, enabling golden section search. Consequently, Algorithm 3 uses a double binary search nested with Algorithm 4's golden section search, reducing complexity to $\mathcal{O}(\log(1+1/\varepsilon)+\log(1+\log 1/\delta))$, which only has logarithmic coupling with $\delta$, making it ideal for online budget scanning.

**3. zCDP equivalent composition: Multimodality reduces loss without increasing costs**

The biggest engineering concern for DP mechanisms is "performing well in one step but collapsing after composition"—mechanisms like truncated Laplace that lack Gaussian tails have poor composition constants. This paper's solution is to constrain all components to be **homoscedastic**. Since the multi-Gaussian is a convex combination of homoscedastic Gaussians, leveraging the quasi-convexity of $\alpha$-Rényi divergence from Bun-Steinke 2016 Lemma 15, Corollary 3.7 proves it satisfies the **exact same** $\rho=\Delta^2/(2\sigma^2)$-zCDP as a single Gaussian. Thus, $T$ compositions degrade directly to $\varepsilon_{\mathrm{tot}}=\rho_{\mathrm{tot}}+2\sqrt{\rho_{\mathrm{tot}}\log(1/\delta_{\mathrm{tot}})}$ (Corollary 3.8), providing the tight composition of Gaussians for free. Complementarily, Propositions 3.3 and 4.7 prove that for any $\delta\in(0,1/2)$, there exists $\varepsilon_0$ such that for $\varepsilon\geq\varepsilon_0$, the $l_2$-loss of the multi-/quasi-Gaussian is **strictly** smaller than that of the analytic Gaussian, elevating numerical advantages to analytical guarantees. Homoscedasticity is a trade-off—it sacrifices the freedom to assign different variances to different components, but in exchange, composition costs remain unchanged, which is the prerequisite for plugging multimodal Gaussians directly into iterative algorithms like DP-SGD or proximal methods.

### Loss & Training

This paper does not train models; the goal is to minimize the closed-form expected loss $\mathbb{E}|\tilde X|$ ($l_1$-loss / noise amplitude) and $\mathbb{E}\tilde X^2$ ($l_2$-loss / noise power). Algorithms 1 and 3 use binary search to find the minimum $\sigma$ satisfying DP conditions. Empirical hyperparameter values are $K\in[20]$ and $\eta=0.01$. Numerical integration is performed using the Julia QuadGK package, root finding via Roots, and unimodal search via Optim.

## Key Experimental Results

Experiments fix $\Delta=1$, scanning $\varepsilon\in\{0.1, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 10\}$ and $\delta$ in 15 levels from $5\times 10^{-7}$ to $0.25$, totaling 150 $(\varepsilon,\delta)$ grid points. Reported metrics use $100\cdot(a-m)/\max(a,m)\,\%$, where $a$ is the baseline loss and $m$ is the best loss from this work.

### Main Results

**Table 1 (vs. unimodal analytic Gaussian, $l_1$-loss Improvement %) — multi-Gaussian with best $K\in\{1,\dots,20\}$, $\eta=0.01$:**

| Configuration | Key Metrics | Description |
|------|---------|------|
| Mean over All 150 Grid Points | 53.73 % (sd 34.86) | Multimodality reduces expected Gaussian noise amplitude by over half on average |
| Median over All 150 Grid Points | 61.86 % | Median improvement is larger than mean; long tail in high-privacy regime |
| Strictly better than unimodal | 142 / 150 | Only a few extreme high-privacy points remain equal |
| $\varepsilon=1, \delta=10^{-5}$ | 67.80 % | Typical medium-privacy operating point |
| $\varepsilon=2, \delta=10^{-5}$ | 79.16 % | $\varepsilon$ scale similar to VaultGemma |
| $\varepsilon=5, \delta=10^{-5}$ | 94.68 % | Low-privacy regime almost entirely closes the Gaussian gap |
| $\varepsilon=10, \delta=10^{-6}$ | 88.08 % | Stable and significant lead in extremely low-privacy regime |
| Max Gap Closure Rate | Up to 99 % | Compared against Selvi et al. 2025 numerical lower bound |

**Table 2 (vs. non-Gaussian asymptotically optimal family, $l_1$-loss Improvement %) — Baseline is the best among truncated Laplace / Tulap / staircase / cactus / flipped Huber:**

| Range | Conclusion | Description |
|------|------|------|
| $\varepsilon\geq 1$ | Strictly better than all non-Gaussian baselines | Multimodal Gaussians overtake truncated Laplace and other "asymptotically high-privacy optimal" mechanisms |
| $\varepsilon<1$ | Equal / Slightly inferior | This range is the fundamental limit of Gaussian types; this work claims no improvement here |
| Any $\delta$ | Improvement is nearly independent of $\delta$ | $\delta$ is usually cryptographically small; $\varepsilon$ is the actual tunable dimension |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Full multi-Gaussian ($K^*$ best) + $\eta=0.01$ | Minimum expected $l_1$-loss | Full version, average improvement of 53.73 % |
| $K=0$ Degradation | Equivalent to analytic Gaussian | Degenerates to Balle-Wang 2018 baseline |
| Quasi-Gaussian (No Hyperparams) | Slightly worse than multi-Gaussian, significantly better than analytic Gaussian | Calculation of $\sigma$ is only $\mathcal{O}(\log 1/\delta)$, suitable for repeated calls |
| Homoscedasticity Constraint | zCDP constant $\rho=\Delta^2/(2\sigma^2)$ identical to single Gaussian | Key to "lossless" compositionality; varying variances would destroy zCDP equivalence |

### Key Findings
- **Multimodality is the Key**: Rinberg et al. 2025 proved that unimodal generalized Gaussians do not outperform standard Gaussians. By using the same Gaussian tails but changing "unimodal" to "$2K+1$ peaks decaying at $e^{-\varepsilon}$," this work achieves 53–99% loss reduction—multimodal structures are "non-trivial geometric features" of numerically optimal distributions that unimodal families cannot reach.
- **Improvement scales monotonically with $\varepsilon$**: Improvement is only 2–15% at $\varepsilon=0.25$, but approaches 95% at $\varepsilon=5$ and 99% at $\varepsilon=10$. This means the more common the low-to-medium privacy regime in industrial deployment, the greater the gain; conversely, gains are smaller in the asymptotic high-privacy regime favored in papers.
- **Improvement is nearly insensitive to $\delta$**: Changes across the table in the $\delta$ direction (from $5\times 10^{-7}$ to $0.25$) are mild; the key variable is $\varepsilon$. This perfectly matches reality where $\delta$ must be cryptographically small and $\varepsilon$ is the only tunable dimension.
- **Quasi-Gaussian reduces $1/\delta$ to $\log 1/\delta$**: Algorithm 1 for multi-Gaussian has complexity proportional to $1/\delta$ due to $\eta\delta$ discretization, which is expensive for $\delta=10^{-7}$ scales. By analytically splitting DP conditions into $\sigma_1, \sigma_2$ constraints, quasi-Gaussian reduces this to $\log 1/\delta$, making it an engineering standard for budget scanning.

## Highlights & Insights
- **"Translating geometric features of numerical optima into closed-form distributions" is a reusable paradigm**: Start with numerical optimization (Selvi et al. 2025) to find geometric features (e.g., "$\Delta$-periodic peaks + $e^{-\varepsilon}$ ratio"), then use a closed-form parametric family (Gaussian mixtures) to fit these features. This bypasses the issues of numerical solutions lacking closed forms or being non-sampleable, serving as a bridge between "numerical optimal bounds" and "deployable mechanisms."
- **Homoscedasticity is a non-trivial choice for zCDP equivalence**: Intuitively, allowing "different variances for different components" should offer more freedom and lower loss, but it would break $\rho$-zCDP equivalence (quasi-convexity of $\alpha$-Rényi divergence requires members in a convex combination to be of the same type). Sacrificing this freedom allows "noise replacement in DP-SGD with unchanged composition constants," a design philosophy of "conceding for the downstream."
- **$\eta$ discretization + compressing $\delta$ to $(1-\eta)\delta$ is a general trick for turning uncountable DP constraints into computable certificates**: DP definitions must hold over all neighbor sets and all measurable sets, which is rarely verified directly in engineering. The "grid + $\delta$ concession" template provided in Theorem 3.2 is transferable to any mechanism design involving "supremum over continuous neighbor parameters $\varphi$."
- **The shift in DP research focus from asymptotic to low-to-medium privacy is a true signal**: Opacus uses $\varepsilon=50$, VaultGemma $\varepsilon=2$, and the Census operates at $\varepsilon\geq 1$. This paper sounds the clarion for "low-to-medium privacy mechanism design" at ML venues like ICML—many "solved" DP problems, including basic noise selection, are far from optimal in real deployment regimes.

## Limitations & Future Work
- **Limited to 1D scalar queries**: All DP conditions, $\sigma$ derivation, and optimality proofs assume $q:\mathcal{D}\to\mathbb{R}$. Whether multimodality still outperforms unimodality in high-dimensional queries is unknown; some mechanisms (like Flipped Huber) are near-optimal in high dimensions but inferior to truncated Laplace in 1D, suggesting 1D and high-dimensional certificates should be established separately.
- **"Asymptotically optimal" only proven for $\varepsilon\geq\varepsilon_0$ without explicit $\varepsilon_0$**: Propositions 3.3 / 4.7 only guarantee the existence of some $\varepsilon_0$ without providing values; analytical optimality for medium $\varepsilon$ (e.g., $\varepsilon\in[0.5,1]$) still relies on numerical tables.
- **$\eta$ discretization introduces conservative $\sigma$**: Algorithm 1 returns the "tightest $\sigma$ under the relaxation," not necessarily the absolute minimum $\sigma$ satisfying $(\varepsilon,\delta)$-DP. The paper uses "conservative rounding + Selvi numerical bounds" to verify the gap is small, but lacks an analytical tightness proof.
- **Missing end-to-end empirical evidence on downstream algorithms like DP-SGD**: The paper only measures expected noise loss and does not report model accuracy changes after replacing noise in DP-SGD for LLM training or classifiers. The most natural next step is to reproduce experiments from Abadi 2016 or Sinha 2025 to see if downstream utility yields synchronized gains.
- **Generalization directions**: The "multimodal + decay weights" template could be applied to Laplace (multimodal Laplace mixtures), Cauchy (heavy-tail mixtures), or discrete counting queries to replicate similar low-to-medium privacy regime improvements over their respective asymptotically optimal mechanisms.

## Related Work & Insights
- **vs. Balle & Wang 2018 (analytic Gaussian)**: They pushed $\sigma$ to the limit within the single Gaussian family; this paper proves "the room for improvement within the single Gaussian family is far less than switching to a Gaussian mixture family"—elevating the dimension of improvement from "parameter tuning" to "family expansion."
- **vs. Selvi et al. 2025 (numerically optimal DP mechanisms)**: They used cutting-plane methods to find "numerical approximations of optimal distributions" (no closed form, non-sampleable). This work does the reverse: it uses closed-form Gaussian mixtures to **approximate** their geometric features, balancing engineering usability and optimality.
- **vs. Rinberg et al. 2025 (generalized Gaussians not better than Gaussian)**: Their negative result only holds for unimodal generalized Gaussians. This work provides a **complementary** positive result using multimodal constructions—directing research away from "trying broader unimodal families" toward "testing multimodal families."
- **vs. Geng et al. 2020 (truncated Laplace) / Awan & Slavkovic 2020 (Tulap) / Soria-Comas 2013 (staircase)**: These mechanisms are near-optimal in the $\varepsilon\downarrow 0$ asymptotic high-privacy regime but have poor composition due to lack of Gaussian tails and are outperformed by this work in the industrial $\varepsilon\geq 1$ range. This paper argues "asymptotically optimal $\neq$ engineering optimal" and uses zCDP tight composition as a differentiator.
- **vs. Bun & Steinke 2016 (zCDP)**: Corollary 3.7 extends zCDP applicability from single Gaussians to "convex combinations of homoscedastic Gaussians"—a non-trivial new member of the zCDP family with methodological value for designing other Gaussian-like variants.
- **vs. Abadi et al. 2016 (DP-SGD) / Sinha et al. 2025 (VaultGemma)**: Both DP-SGD mainlines use Gaussian noise. This paper provides a **plug-and-play superior noise source**—same zCDP constants, same composition analysis, but smaller expected loss. Theoretically, it can be replaced directly, representing the most natural follow-up work.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Translating geometric features of numerical optima into closed-form Gaussian mixtures is non-obvious and complements Rinberg 2025's negative result for unimodal cases.
- Experimental Thoroughness: ⭐⭐⭐⭐ 150 grid points + multiple baselines (5 non-Gaussian + analytic Gaussian) + comparison with rigorous numerical lower bounds is sufficient, though downstream DP-SGD validation is missing.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain—motivation, numerical evidence, closed-form construction, algorithm, complexity, compositionality, and analytical optimality proofs—is very smooth, well-balancing theory and utility.
- Value: ⭐⭐⭐⭐⭐ A "free upgrade" that can actually be swapped in for low-to-medium privacy regimes, directly impacting industrial DP deployment and establishing a paradigm for "how numerical optima can guide closed-form family design."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] GEM-FI: Gated Evidential Mixtures with Fisher Modulation](gem-fi_gated_evidential_mixtures_with_fisher_modulation.md)
- [\[NeurIPS 2025\] Sequentially Auditing Differential Privacy](../../NeurIPS2025/ai_safety/sequentially_auditing_differential_privacy.md)
- [\[CVPR 2025\] Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis](../../CVPR2025/ai_safety/mind_the_gap_detecting_black-box_adversarial_attacks_in_the_making_through_query.md)
- [\[ICML 2026\] Persuasive Privacy](persuasive_privacy.md)
- [\[CVPR 2026\] LDP-Slicing: Local Differential Privacy for Images via Randomized Bit-Plane Slicing](../../CVPR2026/ai_safety/ldp-slicing_local_differential_privacy_for_images_via_randomized_bit-plane_slici.md)

</div>

<!-- RELATED:END -->
