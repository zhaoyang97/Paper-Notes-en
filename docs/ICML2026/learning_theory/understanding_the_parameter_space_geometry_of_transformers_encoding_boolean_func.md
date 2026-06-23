---
title: >-
  [Paper Note] Understanding the Parameter Space Geometry of Transformers Encoding Boolean Functions
description: >-
  [ICML 2026][learning_theory][Parity] This paper explains why Transformers fail to learn "sensitive" Boolean functions like Parity from the perspective of **parameter space geometry**. It proves that a randomly initialized Transformer almost surely computes functions containing a large number of strings with zero sensitivity. The parameters corresponding t
tags:
  - ICML 2026
  - learning_theory
  - Parity
  - layer norm
date: 2026-05-08
content_hash: 24f51755895750ff
---
# Understanding the Parameter Space Geometry of Transformers Encoding Boolean Functions

**Conference**: ICML2026  
**arXiv**: [2606.08768](https://arxiv.org/abs/2606.08768)  
**Code**: To be confirmed  
**Area**: Learning Theory / Transformer Expressivity and Learnability / Formal Languages  
**Keywords**: Sensitivity profile, parameter space geometry, measure theory, learnability, Parity, layer norm

## TL;DR
This paper explains why Transformers fail to learn "sensitive" Boolean functions like Parity from the perspective of **parameter space geometry**. It proves that a randomly initialized Transformer almost surely computes functions containing a large number of strings with zero sensitivity. The parameters corresponding to functions like Parity or First, which lack zero-sensitivity strings, constitute only a **Lebesgue measure zero** subset of the parameter space. Consequently, random initialization almost certainly misses these functions, rendering them provably unlearnable.

## Background & Motivation

**Background**: The mainstream tool for understanding Transformer capability boundaries is **expressivity**—whether a set of parameters exists that allows a Transformer to compute a given function. Significant work has characterized the upper and lower bounds of Transformer expressivity.

**Limitations of Prior Work**: Expressivity does not equate to **learnability**. Even if "correct parameter settings exist," training may not reach them. A classic counterexample is Parity (determining the parity of the number of 1s in the input): while Transformers with layer norm can theoretically represent it, they fail to learn it in practice. A general empirical and theoretical finding is that Transformers fail to learn **sensitive** functions—those where flipping a single input bit is highly likely to change the output.

**Key Challenge**: It is known that Transformers prefer functions with **low average sensitivity**, but the precise mechanism linking "average sensitivity" to "what optimization can actually find" has been missing. Furthermore, average sensitivity is too coarse: the First function (a dictator function looking only at the first bit) has an average sensitivity of only 1, yet it is similarly unlearnable. Thus, average sensitivity alone cannot explain these phenomena.

**Goal**: To characterize the "volume" occupied by different classes of Boolean functions within the parameter space using **measure theory**, translating "learnability" into whether the corresponding parameter region has a non-zero Lebesgue measure.

**Key Insight**: The authors upgrade the perspective from "average sensitivity" to the **sensitivity profile**—the distribution of sensitivity values across all input strings—which carries significantly finer information than a single mean. The core physical quantity is the "**blowup**" induced by layer norm: the stability constant $\epsilon$ in the denominator determines how much a single parameter perturbation or bit flip can be amplified.

**Core Idea**: To prove that the sensitivity profile of a randomly initialized Transformer **necessarily has a non-empty low-end tail** (at least a polynomial number of strings with very low sensitivity). Equivalently, the parameters of "functions encoding very few zero-sensitivity strings" form a measure zero set, which random initialization almost surely avoids. Thus, functions lacking zero-sensitivity strings are provably unlearnable.

## Method

### Overall Architecture
The paper treats the Transformer as a **recognizer**: it classifies a binary string $\mathbf{x}\in\{0,1\}^N$ (appended with an eos token) as belonging or not belonging to a language, using a recognition definition with a margin $\xi$—the output logit $T(\mathbf{x})=\boldsymbol{w}^\top\boldsymbol{y}_{N+1}^L+\omega$ must satisfy $\ge\xi$ or $\le-\xi$. The critical "lifting" operation is viewing the Transformer as a **bivariate function of parameters and inputs** $T:\Theta\times\{0,1\}^*\to\mathbb{R}$, where the parameter space $\Theta\subset\mathbb{R}^M$ is a compact set, allowing for discussions on the "volume" of parameter subsets.

The analysis diverges into two paths based on the layer norm stability constant $\epsilon$: $\epsilon>0$ (warm-up, denominator bounded below, network is uniformly Lipschitz w.r.t. parameters) and $\epsilon=0$ (main focus, denominator can be arbitrarily small, blowup is no longer uniformly bounded). Both cases lead to the counterintuitive conclusion that Transformer recognizers are **remarkably weak in expressivity**. Four key designs follow: measurement tools (sensitivity profile + blowup), strong inexpressibility conclusions for $\epsilon>0$, the measure-theoretic main theorem for $\epsilon=0$, and the resulting unlearnability list.

### Key Designs

**1. Sensitivity Profile + Blowup: Translating Learnability into Calculable Geometry**

Since average sensitivity is too coarse to account for counterexamples like First, the authors introduce two metrics. First is the **sensitivity profile**: for a function $f$ at length $N$, it records $K_n(N)$—the number of strings with sensitivity exactly $n$, where single-string sensitivity $s_N(\mathbf{x},f)=\sum_{n=1}^N|f(\mathbf{x})-f(\mathbf{x}^{\oplus n})|^2$ counts neighbors where flipping a bit changes the output. This is more granular than average sensitivity $\mathrm{as}_N(f)=2^{-N}\sum_\mathbf{x}s_N(\mathbf{x},f)$, focusing specifically on the **number of zero-sensitivity strings** $K_0(N)$. Second is the **blowup factor**: single-layer blowup $\tau^\ell=\max_n(1+z_n^\ell)$ quantifies how much layer norm output amplifies input changes (where $z_n^\ell=1/\sqrt{\sigma^2(\boldsymbol{d}_n^\ell)+\epsilon}$ is the normalization coefficient), with cumulative blowup $\beta^L=\prod_\ell\tau^\ell$. Lemma 2.9 bounds the influence of flipping the $n$-th bit by the product of blowup factors $\mathrm{I}_n\le C_{\text{infl}}\beta^L(\boldsymbol{\theta},\mathbf{x})\beta^L(\boldsymbol{\theta},\mathbf{x}^{\oplus n})(\frac{1}{N}+\delta_{m,n})$. This leverages blowup to determine output sensitivity.

**2. The Case $\epsilon>0$: Uniform Lipschitz Directly Forces "Asymptotically Constant" Functions**

When $\epsilon>0$, the denominator is bounded $z_n^\ell\le1/\sqrt{\epsilon}$, so cumulative blowup is strictly controlled $\beta^L\le(1+1/\sqrt{\epsilon})^L$ (Lemma 3.2), independent of input length $N$. Substituting into Lemma 2.9, the influence of flipping any non-eos bit on the final eos activation is only $\mathcal{O}(1/N)$ (Cor. 3.3). The consequence is strong: if a function flips its output for any bit of any string (i.e., it is non-constant, $\mathrm{maxs}_N\ge1$), this $\mathcal{O}(1/N)$ change cannot overcome the margin $\xi$ as $N \to \infty$. Thus, **any non-constant Boolean function becomes unrecognizable** (Cor. 3.4), i.e., $|\mathcal{F}_N|\to2$. Furthermore, since $T$ is uniformly Lipschitz w.r.t. parameters (Lemma 3.1), a packing argument shows the number of recognizable functions is capped by a constant **independent of $N$**: $|\mathcal{F}_N|\le(1+C_{\text{Lip}}\mathrm{diam}(\Theta)/\xi)^M$ (Prop. 3.5), whereas there are $2^{2^N}$ Boolean functions in total. Conclusion: Transformers with $\epsilon>0$ are profoundly inexpressive as computational models, prompting the shift to $\epsilon=0$.

**3. Main Framework for $\epsilon=0$: Random Transformers Almost Surely Have Zero-Sensitivity Strings**

At $\epsilon=0$, blowup can theoretically be arbitrarily large, lacking the deterministic bound of Lemma 3.2—this is the source of potential expressivity and the primary analytical challenge. The authors replace "deterministic bounds" with "**high-probability bounds**" in three steps. First (Lemma 4.2): for a fixed string $\mathbf{x}$, applying a small uniform perturbation $\boldsymbol{\Delta}\sim\mathrm{Unif}(B_\infty^M(\rho))$ to parameters ensures, **with high probability**, the existence of a bit set $S$ of size $k$ such that blowup on $\mathbf{x}$ and its $S$-Hamming neighborhood is bounded by $\beta^L=\mathcal{O}(N^{\zeta/2})$. The intuition is that "danger zones" (parameters triggering large blowup) occupy little volume. Second (Cor. 4.3): bounded blowup implies bit flips in $S$ have $\mathcal{O}(N^{\zeta-1})$ influence, failing to cross the margin. Third (Lemma 4.4) uses measure theory to extend local neighborhood results to the entire $\Theta$: by covering the parameter space with small cubes and applying the previous step, it is shown that for $\boldsymbol{\theta}\sim\mathrm{Unif}(\Theta)$, $s_N(\mathbf{x},T(\boldsymbol{\theta},\cdot))\le N-k$ holds with high probability. Finally, aggregating for all sufficiently large $N$ via the first Borel–Cantelli lemma yields **Main Theorem 4.5**: with probability 1, a random Transformer has at least $N^{\frac{D-1}{2L}-5}$ zero-sensitivity strings for all large $N$. In other words, parameters for functions with very few zero-sensitivity strings form a measure zero set.

**4. From Sensitivity Profile to Unlearnability: Parity/First are Provably Unlearnable, Majority Escapes**

Corollary 4.6 provides a clean criterion: if a function family satisfies $K_0(N)<N^{\frac{D-1}{2L}-5}$, Transformers fail to recognize it with margin $\xi$ for almost all parameters as $N \to \infty$. This yields several unlearnability conclusions: **Parity** has sensitivity $N$ for every input ($K_0=0$) and is almost surely unlearnable (Cor. 5.1); $m$-Sparse Parity is similar (5.2); **Dictator/First** functions (minimum sensitivity 1) are also unlearnable (5.3, 5.4). Notably, average sensitivity cannot explain First, but the sensitivity profile can. Conversely, **Majority** has exponentially many zero-sensitivity strings (sensitive only for "near-balanced" inputs), easily exceeding the polynomial threshold $N^{\frac{D-1}{2L}-5}$—the theorem **does not apply** to it. This aligns with empirical evidence that Transformers learn Majority well. The authors propose Conjecture 5.5: the parameter subset corresponding to Majority has positive Lebesgue measure.

## Key Experimental Results

Experiments aim to **validate theoretical predictions**: whether the sensitivity profile of Transformers (randomly initialized and after training) is heavily biased toward zero.

### Main Results: Sensitivity Profile Heavily Biased Toward Zero at Initialization
Sensitivity profiles were calculated for models sampled using four initialization schemes (Uniform, Gaussian, Xavier Uniform, Xavier Gaussian).

| Setting | Observed Sensitivity Profile | Relation to Theory |
|---------|-----------------------------|-------------------|
| Random Init (4 schemes) | Consistently and heavily biased toward 0, stronger than Thm. 4.5 | Confirms sensitive functions occupy tiny regions |
| After Training | Low-sensitivity bias persists | Suggests inductive bias from initialization constrains optimization |
| Uniform Random Boolean (Control) | Probability of zero-sensitivity string ≈ $1-1/e\approx0.63$ | Random Transformer prob. is 1, confirming intrinsic bias |

### Training Experiments: Which Functions are Learnable
Transformers were trained on Parity, Majority, First, and $m$-Sparse variants.

| Target Function | Learning Result | Sensitivity Profile Behavior | Corresponding Theory |
|-----------------|-----------------|------------------------------|----------------------|
| Parity | Failed | Failed to reach correct (all $N$) profile | Confirms Cor. 5.1 Unlearnable |
| First | Learned at medium lengths | Sensitivity correctly clustered at 1 | Consistent: Unlearnability is asymptotic |
| Majority | Reliably learned | Sensitivity clustered at 0 and $N/2$ | Supports Conj. 5.5 |
| $m$-Sparse Variants | Mirrors respective base versions | Consistent with parent function patterns | Consistent with corollaries |

### Key Findings
- **Low-sensitivity bias persists after training**; it is not a transient artifact of initialization. This suggests the inductive bias substantially constrains generalization (argued via PAC-Bayesian and Bayesian perspectives: the posterior is absolutely continuous w.r.t. the prior and inherits its measure-zero sets).
- **Asymptotic $\neq$ Practical Length**: Learning First at medium lengths does not contradict "asymptotic unlearnability"—failure cases exist but may appear at lengths exceeding practical applications.
- **The bias is intrinsic**: Compared to the 0.63 probability of zero-sensitivity strings in random Boolean functions versus 1.0 in Transformers, it is proven to be a structural bias rather than a statistical artifact of function distribution.

## Highlights & Insights
- **Upgrading from "average sensitivity" to "sensitivity profile" is the key move**: It captures cases like First that "low average sensitivity" frameworks miss, demonstrating the power of looking at distributions rather than means.
- **Geometricization/Measurization of Learnability**: Defining "almost surely unlearnable" via Lebesgue measure zero sets provides a rigorous mathematical characterization of the "expressivity vs. learnability" gap.
- **Dual Path Analysis ($\epsilon>0$ vs. $\epsilon=0$)**: Using uniform Lipschitz for a strong conclusion at $\epsilon>0$, then applying a measure-theoretic toolkit (cube covering, Borel-Cantelli) at $\epsilon=0$, creates a robust logical structure.
- **Symmetric Viewpoint**: Beyond proving Parity/First are unlearnable (measure zero), the conjecture that Majority occupies positive measure prevents a purely pessimistic narrative.

## Limitations & Future Work
- **Asymptotic Conclusions**: Unlearnability is only guaranteed as $N \to \infty$. For practical lengths, First remains learnable. The threshold for zero-sensitivity strings may exceed realistic input lengths.
- **Bounds may be loose**: The theory provides a **polynomial** lower bound for zero-sensitivity strings, while experiments suggest **exponential** growth; proving exponential scaling remains an open challenge.
- **Majority is only a conjecture**: Conj. 5.5 has empirical support but no proof; the converse (whether sufficient zero-sensitivity strings guarantee learnability) is also unresolved.
- **Idealized Recognizer Model**: Analysis is based on recognizers with margins and fixed positional encodings. While authors argue engineering variations (learnable encodings, different LN placements) don't change the conclusion, they remain simplifications.
- The link between "random init measure zero" and "unreachable after training" relies on Bayesian/PAC-Bayes arguments, though training is not strictly Bayesian inference.

## Related Work & Insights
- **vs. Hahn & Rofin (2024)**: Uses their blowup framework but generalizes from **average sensitivity** to the **entire sensitivity profile**, covering functions like First.
- **vs. Bhattamishra et al. (2023), Abbe et al. (2023)**: They observed difficulty in learning sensitive/sparse Boolean functions; this paper explains "why" by attributing it to the geometric fact that sensitive functions occupy measure zero space.
- **vs. Chiang & Cholak (2022)**: They proved Transformers with layer norm can express Parity; this work fills the gap by explaining why they can express but cannot learn it.
- **vs. Buzaglo et al. (2024), Dziugaite & Roy (2025)**: They link generalization to "compatible parameter volume"; this paper uses that logic to push "measure zero initialization" toward "failure to generalize."

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Provides a provable characterization of the "expressivity vs. learnability" gap using parameter space measure geometry and sensitivity profiles.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Systematically validates the bias across initializations and functions, though primarily verificatory in scale.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure; honest treatment of limitations regarding asymptotic vs. practical length and conjectures.
- **Value**: ⭐⭐⭐⭐ Provides a solid theoretical foundation for Transformer inductive bias; however, asymptotic and idealized assumptions limit direct engineering guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Generalizing Analogical Inference from Boolean to Continuous Domains](../../AAAI2026/learning_theory/generalizing_analogical_inference_from_boolean_to_continuous_domains.md)
- [\[ICML 2026\] Task-Restricted Symmetries in Recurrent Weight Space](task-restricted_symmetries_in_recurrent_weight_space.md)
- [\[ICLR 2026\] A Faster Parameter-Free Regret Matching Algorithm](../../ICLR2026/learning_theory/a_faster_parameter-free_regret_matching_algorithm.md)
- [\[ICLR 2026\] Barriers for Learning in an Evolving World: Mathematical Understanding of Loss of Plasticity](../../ICLR2026/learning_theory/barriers_for_learning_in_an_evolving_world_mathematical_understanding_of_loss_of.md)
- [\[ICML 2026\] Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function](provably_data-driven_multiple_hyper-parameter_tuning_with_structured_loss_functi.md)

</div>

<!-- RELATED:END -->
