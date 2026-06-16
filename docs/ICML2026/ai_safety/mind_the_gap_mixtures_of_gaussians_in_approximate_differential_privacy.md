---
title: >-
  [Paper Note] Mind the Gap: Mixtures of Gaussians in Approximate Differential Privacy
description: >-
  [ICML 2026][AI Safety][Paper Note] This paper designs a class of Gaussian mixture additive noise mechanisms (hyperparameter-dependent multi-Gaussian mixture and hyperparameter-free quasi-Gaussian mixture) for $(\varepsilon,\delta)$-DP. It closes the suboptimality gap of the analytic Gaussian mechanism by up to 99% in the low-to-medium privacy regime whi
tags:
  - ICML 2026
  - AI Safety
date: 2026-05-08
content_hash: 5b979b06cf9da4af
---
# Mind the Gap: Mixtures of Gaussians in Approximate Differential Privacy

**Conference**: ICML 2026  
**arXiv**: [2605.28078](https://arxiv.org/abs/2605.28078)  
**Code**: https://github.com/selvi-aras/MindTheGap  
**Area**: AI Safety / Differential Privacy  
**Keywords**: Approximate Differential Privacy, Gaussian Mixture Mechanism, Additive Noise, Low-to-Medium Privacy Regime, zCDP Composition

## TL;DR
This paper designs a class of Gaussian mixture additive noise mechanisms (hyperparameter-dependent multi-Gaussian mixture and hyperparameter-free quasi-Gaussian mixture) for $(\varepsilon,\delta)$-DP. It closes the suboptimality gap of the analytic Gaussian mechanism by up to 99% in the low-to-medium privacy regime while preserving the tight zCDP composition properties of Gaussians.

## Background & Motivation

**Background**: Approximate differential privacy $(\varepsilon,\delta)$-DP is the industry de facto standard (used in the US 2020 Census, Google VaultGemma LLM, Opacus, etc.). Common implementations include the Dwork-Roth Gaussian mechanism $\sigma=\sqrt{2\log(1.25/\delta)}(\Delta/\varepsilon)$ and the "analytic Gaussian mechanism" numerically tightened by Balle-Wang 2018 via binary search. Gaussians are widely used due to their unbounded support (no distinguishing events), approximate $3\sigma$ empirical rule, and seamless fit into the zCDP framework for lossless composition.

**Limitations of Prior Work**: DP theoretical analysis is almost exclusively focused on the asymptotic high-privacy regime where $\varepsilon,\delta \downarrow 0$. However, real-world deployments often fall into the low-to-medium privacy regime where $\varepsilon \geq 1$ (VaultGemma uses $\varepsilon=2$, Opacus tutorials suggest $\varepsilon=50$, industry $\varepsilon$ is typically large, while $\delta$ must be cryptographically small). The numerical optimization framework of Selvi et al. 2025 proved that in these regimes, the expected loss suboptimality of the analytic Gaussian can reach 700%.

**Key Challenge**: The "unimodal + unbounded support" property of the Gaussian mechanism makes it easy to compose and analyze. However, numerical results for the optimal distribution from Selvi et al. 2025 show that the true optimal noise distribution **is not unimodal**. Instead, it features a peak on every interval of length $\Delta$, with a density ratio between adjacent peaks of approximately $e^{\varepsilon}$. Rinberg et al. 2025 further proved that any **unimodal** generalized Gaussian cannot outperform the standard Gaussian, locking the direction of improvement toward "multi-modality."

**Goal**: To construct a **multi-modal** Gaussian-like mechanism that compensates for the suboptimality gap of the analytic Gaussian in the $\varepsilon \geq 1$ regime, while maintaining Gaussian tails and compatibility with tight zCDP composition.

**Key Insight**: The two empirical laws from numerical optimal results—"$\Delta$-periodic peaks + $e^{-\varepsilon}$ proportional decay"—are directly encoded into the distribution structure. Using the analytic Gaussian as a backbone, several equal-variance Gaussian components centered at $k\Delta$ and weighted by $e^{-|k|\varepsilon}$ are convolved.

**Core Idea**: A convex combination of a "zero-mean Gaussian + several shifted Gaussians at $\pm k\Delta$" replaces a single Gaussian as the DP noise, making the density shape approximate the optimal multi-modal distribution. Equal variance ensures that the zCDP composition constants are **identical** to those of a single Gaussian; multi-modality is used solely to suppress expected loss without increasing composition costs.

## Method

### Overall Architecture

The problem to be solved is: given a sensitivity $\Delta$ of a query function $q:\mathcal{D}\to\mathbb{R}$ and a privacy budget $(\varepsilon,\delta)$, find a $\sigma$ as small as possible such that the noise mechanism $\mathcal{A}(D)=q(D)+\tilde{X}$ satisfies $(\varepsilon,\delta)$-DP with minimal expected noise loss. This paper shifts from "tuning $\sigma$ within a single Gaussian" to "designing a noise distribution $\tilde{X}$ within a family of **multi-modal Gaussian mixtures**." Based on the analytic Gaussian skeleton, shifted Gaussian components are added following two geometric laws of numerical optimal solutions ($\Delta$-periodic multi-modality and an adjacent peak density ratio of approximately $e^{\varepsilon}$). Following this line, the paper presents two mechanisms (multi-Gaussian with the lowest loss but requiring hyperparameters, and a lightweight, hyperparameter-free quasi-Gaussian), proving both can be integrated into the zCDP framework for lossless tight composition.

### Key Designs

**1. Multi-Gaussian mixture: Writing the geometry of the optimal solution into a closed-form distribution**

The limitation is that the optimal noise distribution numerically solved by Selvi et al. 2025 lacks a closed form, cannot be sampled, and has no computable moments, making it unusable in engineering. This paper uses a convex combination of $2K+1$ **equal-variance** Gaussians to fit its geometric characteristics, with the density $f_{\mathrm{m}}(x;\sigma,K)=\frac{1}{c_K}\sum_{k=-K}^{K}e^{-|k|\varepsilon}\phi(x;k\Delta,\sigma)$. The $k$-th component center is placed at $k\Delta$ (replicating $\Delta$-periodic peaks), and its weight is $\propto e^{-|k|\varepsilon}$ (replicating geometric decay). The difficulty lies in the fact that $(\varepsilon,\delta)$-DP must hold for the uncountable family of all neighbor shifts $\varphi\in[0,\Delta]$, which cannot be directly verified. Theorem 3.2 introduces a discretization parameter $\eta\in(0,1)$, relaxing the constraint to a finite grid $\{0,\beta,2\beta,\ldots,\Delta\}$ (step size $\beta\leq\sqrt{2\pi}\eta\sigma\delta$) while compressing the right-hand $\delta$ to $(1-\eta)\delta$ as compensation for missed detections, transforming the "infinite condition" into a computable certificate ($\sigma$ approaches the original definition as $\eta$ decreases). Lemma 3.4 proves this sufficient condition is monotonic in $\sigma$, thus Algorithm 1 performs a binary search using the analytic Gaussian's $\sigma_g$ as the right bound, returning the tightest $\sigma$ under this relaxation framework in $\mathcal{O}\!\left(\frac{K^2}{\eta\delta}(\log(1+1/\varepsilon)+\log(1+\log 1/\delta))\right)$ time. This is effective because the multi-modal geometry aligns the density shape with the true optimal, while the closed-form Gaussian mixture retains the convenience of sampling, moment calculation, and analysis.

**2. Quasi-Gaussian mixture: Eliminating hyperparameters and reducing $1/\delta$ to $\log 1/\delta$**

Multi-Gaussian achieves the lowest loss, but Algorithm 1's complexity involves $K^2/(\eta\delta)$, becoming expensive as $\delta$ decreases, which is unsuitable for budget scanning with repeated calls. Quasi-Gaussian uses a zero-mean Gaussian (weight $e^{\varepsilon}$) plus a "pseudo-Gaussian" where $x$ is replaced by $|x|$ (intrinsic $\pm\Delta$ peaks through the absolute value, weight $1$) to compress multi-modality into a single expression: $f_{\mathrm{q}}(x;\sigma)=\frac{e^{\varepsilon}}{c}\exp(-x^2/(2\sigma^2))+\frac{1}{c}\exp(-(|x|-\Delta)^2/(2\sigma^2))$, which has neither $K$ nor $\eta$. Theorem 4.2 analytically decomposes the DP conditions into two paths and takes $\sigma=\max(\sigma_1,\sigma_2)$. $\sigma_1$ regulates the $\delta$ leakage via a closed-form inequality $h_1(\sigma)+h_2(\sigma)\geq 0$ (including $\Phi$ functions and $e^{2\varepsilon},e^{\varepsilon}$ terms); $\sigma_2$ regulates the pointwise amplification factor via the constraint $\max_{x\in[0,\Delta]}f_{\mathrm{q}}/\min_{x\in[0,\Delta]}f_{\mathrm{q}}\leq e^{\varepsilon}$. Lemmas 4.3–4.5 prove monotonicity for both paths and provide search upper bounds ($\sigma_1\leq\sqrt{2(\varepsilon-\log\delta)}\Delta/\varepsilon$, $\sigma_2\leq\sqrt{\Delta^2/(2\varepsilon)}$). Lemma 4.4 simplifies $\max/\min$ to two unimodal sub-intervals enabling golden section search. Finally, Algorithm 3 uses binary search nested with Algorithm 4's golden section search, reducing complexity to $\mathcal{O}(\log(1+1/\varepsilon)+\log(1+\log 1/\delta))$, which only has logarithmic coupling with $\delta$ and fits as a standard for online budget scanning.

**3. zCDP equivalent composition: Multi-modality suppresses loss without increasing composition cost**

The biggest engineering concern for DP mechanisms is "performing well in a single step but failing after composition"—mechanisms like truncated Laplace lack Gaussian tails and have poor composition constants. The solution here is to constrain all components to have **equal variance**. Since the multi-Gaussian is a convex combination of equal-variance Gaussians, and utilizing the quasi-convexity of $\alpha$-Rényi divergence from Bun-Steinke 2016 Lemma 15, Corollary 3.7 proves it satisfies the **exact same** $\rho=\Delta^2/(2\sigma^2)$-zCDP as a single Gaussian. Consequently, $T$ compositions degenerate directly to $\varepsilon_{\mathrm{tot}}=\rho_{\mathrm{tot}}+2\sqrt{\rho_{\mathrm{tot}}\log(1/\delta_{\mathrm{tot}})}$ (Corollary 3.8), gaining the tight composition of Gaussians for free. Correspondingly, Propositions 3.3 and 4.7 prove that for any $\delta\in(0,1/2)$, there exists an $\varepsilon_0$ such that for $\varepsilon\geq\varepsilon_0$, the $l_2$-loss of multi-/quasi-Gaussian is **strictly** less than that of the analytic Gaussian, upgrading the advantage from numerical experiments to an analytical guarantee. Equal variance has a cost—it sacrifices the freedom to assign different variances to different components—but in return, the composition cost remains unchanged, which is the prerequisite for plugging multi-modal Gaussians directly into iterative algorithms like DP-SGD or proximal methods as noise sources.

### Loss & Training

This paper does not train models; the objective is to minimize the closed-form expected loss $\mathbb{E}|\tilde X|$ ($l_1$-loss / noise amplitude) and $\mathbb{E}\tilde X^2$ ($l_2$-loss / noise power). Algorithms 1 and 3 use binary search to find the minimum $\sigma$ satisfying the DP conditions. Empirical hyperparameter values are $K\in[20]$ and $\eta=0.01$. Numerical integration uses the QuadGK package in Julia, root finding uses Roots, and unimodal search uses Optim.

## Key Experimental Results

Experiments fix $\Delta=1$ while scanning $\varepsilon\in\{0.1,0.25,0.5,0.75,1,2,3,4,5,10\}$ and 15 levels of $\delta$ from $5\times 10^{-7}$ to $0.25$, totaling 150 $(\varepsilon,\delta)$ grid points. Reported metrics are $100\cdot(a-m)/\max(a,m)\,\%$, where $a$ is the baseline loss and $m$ is the best loss in this work.

### Main Results

**Table 1 (vs. unimodal analytic Gaussian, $l_1$-loss Improvement %) — multi-Gaussian with optimal $K\in\{1,\dots,20\}$, $\eta=0.01$:**

| Configuration | Key Metrics | Description |
|------|---------|------|
| Average over all 150 points | 53.73 % (sd 34.86) | Multi-modal on average halves the expected amplitude of single-peak Gaussians |
| Median over all 150 points | 61.86 % | Median improvement is larger than the mean; long tail is in high-privacy regions |
| Strictly superior to unimodal | 142 / 150 | Exceptional cases occur only at a few extreme high-privacy points |
| $\varepsilon=1, \delta=10^{-5}$ | 67.80 % | Typical operating point for medium privacy |
| $\varepsilon=2, \delta=10^{-5}$ | 79.16 % | $\varepsilon$ scale similar to VaultGemma |
| $\varepsilon=5, \delta=10^{-5}$ | 94.68 % | Almost entire Gaussian gap is closed in low-privacy regimes |
| $\varepsilon=10, \delta=10^{-6}$ | 88.08 % | Stable and significant lead in extremely low-privacy regimes |
| Best gap closure rate | Up to 99 % | Compared against the numerical optimal lower bound from Selvi et al. 2025 |

**Table 2 (vs. non-Gaussian asymptotically optimal families, $l_1$-loss Improvement %) — baseline is the best among truncated Laplace / Tulap / staircase / cactus / flipped Huber:**

| Range | Conclusion | Description |
|------|------|------|
| $\varepsilon\geq 1$ | Strictly superior to all non-Gaussian baselines | Multi-modal Gaussian overtakes "asymptotically high-privacy optimal" mechanisms like truncated Laplace |
| $\varepsilon<1$ | Comparable / slightly inferior | This range represents the fundamental limit of Gaussian-like mechanisms; no improvement claimed here |
| Arbitrary $\delta$ | Improvement mostly independent of $\delta$ | $\delta$ is usually cryptographically small; $\varepsilon$ is the actual tunable dimension |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Full multi-Gaussian (Optimal $K^*$) + $\eta=0.01$ | Minimum expected $l_1$-loss | Full version, average improvement of 53.73 % |
| $K=0$ Degeneration | Equivalent to analytic Gaussian | Degenerates to the Balle-Wang 2018 baseline |
| Quasi-Gaussian (Hyperparameter-free) | Slightly worse than multi-Gaussian, significantly better than analytic Gaussian | Computing $\sigma$ takes only $\mathcal{O}(\log 1/\delta)$; suitable for repeated calls |
| Equal variance constraint | zCDP constant $\rho=\Delta^2/(2\sigma^2)$ matches single Gaussian | Key to "lossless" compositionality; varying component variances would break zCDP equivalence |

### Key Findings
- **Multi-modality is the key**: Rinberg et al. 2025 proved that unimodal generalized Gaussians do not outperform the standard Gaussian. By using the same Gaussian tail but replacing "unimodal" with "$2K+1$ peaks decaying at $e^{-\varepsilon}$," this work achieves a 53–99% loss reduction—multi-modal structure is a "non-trivial geometric feature" of the numerical optimal distribution that absolute unimodal classes cannot reach.
- **Improvement scales monotonically with $\varepsilon$**: At $\varepsilon=0.25$, improvement is only 2–15%; at $\varepsilon=5$, it approaches 95%; at $\varepsilon=10$, it reaches nearly 99%. This means the gains are larger in the low-to-medium privacy regimes common in industrial deployments, while gains are smaller in the asymptotic high-privacy regimes favoured in theoretical papers.
- **Improvement is sensitized primarily by $\varepsilon$, not $\delta$**: From $\delta=5\times 10^{-7}$ to $\delta=0.25$, the table columns show mild changes; $\varepsilon$ is the critical variable. This matches the reality of deployment where $\delta$ must be small and the only tunable dimension is $\varepsilon$.
- **Quasi-Gaussian reduces $1/\delta$ to $\log 1/\delta$**: Algorithm 1 for multi-Gaussian involves $1/\delta$ due to $\eta\delta$ discretization, making it overhead-heavy for $\delta=10^{-7}$. Resolving DP conditions into $\sigma_1, \sigma_2$ dual constraints in quasi-Gaussian reduces this to $\log 1/\delta$, which is engineering-critical for making the method a standard for budget scanning.

## Highlights & Insights
- **"Translating numerical optimal geometric features into closed-form distributions" is a reusable paradigm**: First, use numerical optimization (Selvi et al. 2025) to identify the geometric features of the optimal solution (here, "$\Delta$-periodic peaks + $e^{-\varepsilon}$ ratio"), then use a closed-form parametric family (Gaussian mixture) to fit these features. This bypasses the issues of numerical solutions being non-closed, non-sampleable, and non-calculable, acting as a bridge from "numerical optimal bounds" to "deployable mechanisms."
- **Equal variance is a non-trivial choice for zCDP equivalence**: Intuitively, allowing "different variances for different components" should offer more freedom and smaller loss, but that would break $\rho$-zCDP equivalence (quasi-convexity of $\alpha$-Rényi divergence requires members in the convex combination to be of the same type). Sacrificing this freedom allows "iterative algorithms like DP-SGD to replace noise directly while keeping composition constants unchanged"—a design philosophy of "conceding one step for the downstream" that is worth emulating.
- **Discretization $\eta$ + compressing $\delta$ to $(1-\eta)\delta$ is a general trick for turning uncountable DP constraints into computable certificates**: The DP definition itself must hold for all neighbors and all measurable sets, which is almost never directly verified in engineering. The template of "gridding + $\delta$ concession" in Theorem 3.2 (approaching the original definition as $\eta \downarrow 0$) can be migrated to any mechanism design involving "supremum over continuous neighbor parameters $\varphi$."
- **The shift in DP research from asymptotic to low-to-medium privacy is a genuine signal**: Opacus uses $\varepsilon=50$, VaultGemma $\varepsilon=2$, and the Census operates at $\varepsilon\geq 1$. This paper sounds a clarion call for "mechanism design for low-to-medium privacy" at ML venues like ICML—many seemingly "solved" DP problems, including basic noise selection, are far from optimal in real-world deployment regimes.

## Limitations & Future Work
- **Limited to 1D scalar queries**: All DP conditions, $\sigma$ solutions, and optimality proofs assume $q:\mathcal{D}\to\mathbb{R}$. Whether multi-modal superiority holds for multi-dimensional queries is unknown; mechanisms like Flipped Huber are near-optimal in high dimensions but lose to truncated Laplace in 1D, suggesting 1D and high-dimensional certificates must be established separately.
- **"Asymptotic optimality" only proven for $\varepsilon\geq\varepsilon_0$ without explicit $\varepsilon_0$**: Propositions 3.3 / 4.7 only guarantee the existence of some $\varepsilon_0$ without providing the actual value; analytical optimality for intermediate $\varepsilon$ (e.g., $\varepsilon\in[0.5,1]$) still relies on numerical tables.
- **$\eta$ discretization introduces conservative $\sigma$**: Algorithm 1 returns the "tightest $\sigma$ under this relaxation framework," not the true minimum $\sigma$ satisfying $(\varepsilon,\delta)$-DP. While "conservative rounding + Selvi numerical bounds" suggest the gap is small, there is no analytical proof of tightness.
- **Lack of end-to-end empirical verification in downstream algorithms (e.g., DP-SGD)**: The paper only measures expected noise loss without integrating the multi-modal Gaussian into DP-SGD to train LLMs/classifiers or reporting model accuracy changes. The most natural next step is to replicate experiments from Abadi 2016 / Sinha 2025 to see if downstream utility benefits proportionally.
- **Generalization directions**: The "multi-modal + decaying weights" template could be applied to Laplace families (multi-modal Laplace mixtures), Cauchy families (heavy-tail mixtures), or discrete counting queries, potentially replicating the same mid-privacy regime overtake on their respective asymptotically optimal mechanisms.

## Related Work & Insights
- **vs. Balle & Wang 2018 (analytic Gaussian)**: They pushed $\sigma$ to the limit within the single Gaussian class. This work proves that "the room for tuning within the single Gaussian class is far less than the room gained by switching to a multi-Gaussian mixture class"—elevating the improvement dimension from "parameter tuning" to "family expansion."
- **vs. Selvi et al. 2025 (numerically optimal DP mechanisms)**: They used cutting-plane methods for "numerical approximations of the optimal distribution" (no closed form, no sampling). This paper does the reverse, using closed-form Gaussian mixtures to **approximate** their geometric features, balancing engineering usability and optimality—a standard translation from "numerical optimal bounds" to "engineering-ready families."
- **vs. Rinberg et al. 2025 (generalized Gaussians not better than Gaussian)**: Their negative results only hold for unimodal generalized Gaussians. This work provides a **complementary** positive result using multi-modal constructions—directing research from "testing broader unimodal families" to "testing multi-modal families."
- **vs. Geng et al. 2020 (truncated Laplace) / Awan & Slavkovic 2020 (Tulap) / Soria-Comas 2013 (staircase)**: These mechanisms are near-optimal in high-privacy $(\varepsilon\downarrow 0)$ regimes but suffer from poor composition due to lack of Gaussian tails and are overtaken by this work in the industrial $\varepsilon\geq 1$ regime. This paper argues "asymptotic optimality $\neq$ engineering optimality" and positions tight zCDP composition as the competitive barrier for Gaussian families.
- **vs. Bun & Steinke 2016 (zCDP)**: Corollary 3.7 extends the applicability of zCDP from single Gaussians to "convex combinations of equal-variance Gaussians"—a non-trivial new member for the zCDP framework, offering methodological value for designing other Gaussian variants (e.g., multi-modal Laplace + Gaussian tail mixtures).
- **vs. Abadi et al. 2016 (DP-SGD) / Sinha et al. 2025 (VaultGemma)**: Both DP-SGD lines use Gaussian noise. This work provides a **plug-and-play superior noise source**—same zCDP constants, same composition analysis, lower expected loss. Theoretically, it can be replaced directly, though the lack of end-to-end experiments makes this the most natural follow-up work.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Translating the geometric features of numerical optimal distributions into closed-form Gaussian mixtures is non-obvious and complements the unimodal negative results of Rinberg 2025.
- Experimental Thoroughness: ⭐⭐⭐⭐ The use of 150 grid points, multiple baselines (5 non-Gaussian + analytic Gaussian), and rigorous numerical lower bound comparisons is sufficient, though it lacks DP-SGD end-to-end downstream validation.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from motivation, numerical evidence, closed-form construction, algorithm design, complexity analysis, compositionality, to analytical optimality proof is very smooth, balancing theory and practical value well.
- Value: ⭐⭐⭐⭐⭐ It offers a "free-lunch upgrade" that can be genuinely replaced in low-to-medium privacy regimes, directly impacting industrial DP deployment and establishing a paradigm for "how to use numerical optimality to guide closed-form family design."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] GEM-FI: Gated Evidential Mixtures with Fisher Modulation](gem-fi_gated_evidential_mixtures_with_fisher_modulation.md)
- [\[NeurIPS 2025\] Sequentially Auditing Differential Privacy](../../NeurIPS2025/ai_safety/sequentially_auditing_differential_privacy.md)
- [\[CVPR 2025\] Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis](../../CVPR2025/ai_safety/mind_the_gap_detecting_black-box_adversarial_attacks_in_the_making_through_query.md)
- [\[ICML 2026\] Persuasive Privacy](persuasive_privacy.md)
- [\[NeurIPS 2025\] Mitigating Privacy-Utility Trade-off in Decentralized Federated Learning via f-Differential Privacy](../../NeurIPS2025/ai_safety/mitigating_privacy-utility_trade-off_in_decentralized_federated_learning_via_f-d.md)

</div>

<!-- RELATED:END -->
