---
title: >-
  [Paper Note] Smooth Calibration Error: Uniform Convergence and Functional Gradient Analysis
description: >-
  [ICLR 2026][Learning Theory][Calibration error] This paper establishes a **finite-sample** theory for smooth calibration error (smooth CE). It first proves that the population smooth CE can be controlled by "training smooth CE + generalization gap" via uniform convergence. It then demonstrates that the training smooth CE is upper-bounded by the **functional gradient norm** of the loss. Consequently, it provides the first provable guarantees for "calibration + accuracy" for gr…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Probability Calibration"
  - "Calibration error"
  - "smooth CE"
  - "uniform convergence"
  - "functional gradient"
  - "gradient boosting"
date: 2026-05-08
content_hash: c8a515129ae9b27f
---

# Smooth Calibration Error: Uniform Convergence and Functional Gradient Analysis

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=qXVmmj8J0T](https://openreview.net/forum?id=qXVmmj8J0T)  
**Code**: Experimental source code provided in appendix (primarily theoretical)  
**Area**: Learning Theory / Probability Calibration  
**Keywords**: Calibration error, smooth CE, uniform convergence, functional gradient, gradient boosting  

## TL;DR
This paper establishes a **finite-sample** theory for smooth calibration error (smooth CE). It first proves that the population smooth CE can be controlled by "training smooth CE + generalization gap" via uniform convergence. It then demonstrates that the training smooth CE is upper-bounded by the **functional gradient norm** of the loss. Consequently, it provides the first provable guarantees for "calibration + accuracy" for gradient boosted trees, kernel boosting, and two-layer neural networks simultaneously.

## Background & Motivation
**Background**: Probability prediction is crucial in high-risk scenarios such as healthcare, meteorology, and language modeling. **Calibration** (where predicted probabilities align with true label frequencies) is a core requirement for measuring its reliability. Existing methods to improve calibration fall into two categories: adding regularization during training and post-processing recalibration.

**Limitations of Prior Work**: Most of these methods **only have empirical validation without theoretical guarantees**, or they create a trade-off between calibration and sharpness (predictive discriminative power)—recalibration often sacrifices accuracy. Thus, the question of "which learning algorithms can train well-calibrated models without losing accuracy" remained unanswered.

**Key Challenge**: Recently, Błasiok et al. (2023) proposed using the **post-processing gap** to characterize calibration from an optimization perspective and defined smooth CE with favorable theoretical properties. However, their analysis is **built on population risk (infinite data)** and cannot be directly applied to real algorithms trained on finite samples. Furthermore, few results explicitly link calibration error to **specific learning algorithms**.

**Goal**: (1) Extend the analysis of smooth CE from the population level to the **finite training sample** level; (2) Provide an "optimization-based calibration" criterion applicable to real-world algorithms; (3) Use a unified framework to explain why GBT/Kernel Boosting/two-layer NN can achieve both accuracy and calibration.

**Key Insight**: The authors seize a critical algebraic fact of smooth CE—it is essentially the supremum of the inner product between the **functional gradient** of the loss and a family of Lipschitz test functions. Since mainstream boosting-style algorithms themselves descend along functional gradients, "minimizing the gradient" naturally leads to better calibration.

**Core Idea**: Decompose smooth CE into two segments: "Uniform Convergence (Training ↔ Population)" + "Functional Gradient (Upper Bound of Training smooth CE)". Then, substitute three algorithms that iterate along functional gradients into the framework to obtain provable joint $\epsilon$-calibration + $\epsilon$-misclassification guarantees.

## Method

### Overall Architecture
The entire paper is a **two-stage theoretical proof chain** aimed at answering "whether population smooth CE can be guaranteed to be small using finite data." The first stage decomposes the target: via uniform convergence, the problem of controlling **population** smooth CE is transformed into controlling **training** smooth CE plus a generalization gap that decays with sample size. The second stage tackles training smooth CE: proving it is upper-bounded by the **functional gradient norm** of the loss. Thus, algorithms that "iterate along functional gradients to minimize the gradient" naturally suppress training calibration error. Finally, this framework is instantiated for three representative algorithms (GBT, Kernel Boosting, and two-layer NN) to derive the sample size and iteration count required to achieve $\epsilon$-level smooth CE and misclassification rates under margin assumptions.

The definition of smooth CE is the core starting point: given a predictor $f$ and distribution $D$,
$$\mathrm{smCE}(f,D):=\sup_{h\in\mathrm{Lip}_1([0,1],[-1,1])}\mathbb{E}\left[h(f(X))\cdot(Y-f(X))\right]$$
which is the supremum over all 1-Lipschitz test functions $h$. Compared to the widely used ECE, it is **continuous, efficiently estimable**, and can upper/lower bound binning ECE, making it a reliable proxy for calibration analysis. For models using sigmoid to map logit $g$ to probability $f=\sigma(g)$, the authors also define a **dual smooth CE** $\mathrm{smCE}_\sigma(g,D)$ on the logits, satisfying $\mathrm{smCE}(f,D)\le\mathrm{smCE}_\sigma(g,D)$—thus, suppressing the dual version is sufficient.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Goal: population smooth CE<br/>smCE(f, D)"] --> B["1. Uniform convergence decomposition<br/>≤ training smCE + generalization gap"]
    B -->|generalization gap = O(1/√n)| C["2. Functional gradient control<br/>training smCE ≤ ‖functional gradient‖"]
    C -->|substitute gradient-descent iteration algorithm| D["3. Three algorithm instantiations<br/>GBT / kernel boosting / two-layer NN"]
    D -->|margin assumption + hyperparameter selection| E["Joint guarantee<br/>ε-smooth CE + ε-misclassification rate"]
```

### Key Designs

**1. Uniform Convergence Bound for smooth CE: Downscaling Population Calibration to Finite Training Sets**

The first stage addresses the pain point that Błasiok's calibration-optimization relationship is entirely based on population expectation $\mathbb{E}_D$, which is inapplicable to finite-data training. The authors aim to prove that $|\mathrm{smCE}(f,D)-\mathrm{smCE}(f,S_{\mathrm{tr}})|$ is controllable. Since it is known on the test set side that $|\mathrm{smCE}(f,D)-\mathrm{smCE}(f,S_{\mathrm{te}})|=O_p(1/\sqrt{n})$, they only need to bridge the "Training ↔ Test" generalization gap of smooth CE. Theorem 1 utilizes a **covering number chaining argument**:
$$\sup_{f\in\mathcal{F}}|\mathrm{smCE}(f,S_{\mathrm{te}})-\mathrm{smCE}(f,S_{\mathrm{tr}})|\le\inf_{\epsilon\ge0}\Big(8\epsilon+24\int_{\epsilon}^{1}\sqrt{\tfrac{\ln N(\epsilon',\mathcal{F},\|\cdot\|_\infty)}{n}}\,d\epsilon'\Big)+2\sqrt{\tfrac{\log\delta^{-1}}{n}}$$
A key subtlety is that smooth CE contains a composite function $h(f(X))$. A naive approach would incur the complexity cost of the composite class $\mathrm{Lip}_1\circ\mathcal{F}$, and the standard contraction lemma cannot shrink $R(\mathrm{Lip}_1\circ\mathcal{F})$ back to $R(\mathcal{F})$. By leveraging the smoothness of smooth CE itself, authors ensure the final bound **does not contain the complexity of the Lipschitz function class**, only that of the hypothesis class $\mathcal{F}$. Combined with Theorem 2, a more interpretable Rademacher complexity version is provided: $\sup_f|\mathrm{smCE}(f,D)-\mathrm{smCE}(f,S_{\mathrm{tr}})|\le \tfrac{C_2}{\sqrt n}+4R_{D,n}(\mathcal{F})+\cdots$. Conclusion: **By simultaneously controlling hypothesis class complexity and minimizing training smooth CE, population smooth CE is guaranteed to be small**.

**2. Functional Gradient Control for Training smooth CE: A Practical Criterion for "Optimization-based Calibration"**

The second stage answers: how to minimize training smooth CE? The authors identify a key algebraic identity—on the training set, smooth CE is precisely the supremum of the inner product between the test function $h$ and the **functional gradient of the loss**. For squared loss $\nabla_f\ell_{\mathrm{sq}}=f-y$, and for cross-entropy $\nabla_g\ell_{\mathrm{ent}}=\sigma(g)-y$, we have:
$$\mathrm{smCE}_\sigma(g,S_{\mathrm{tr}})=\sup_{h\in\mathrm{Lip}_{1/4}(\mathbb{R},[-1,1])}\langle h(g(X)),-\nabla_g L_n(g)\rangle_{L_2(S_n)}\le\|\nabla_g\ell_{\mathrm{ent}}(g(X),Y)\|_{L_1(S_n)}$$
This means training smooth CE is upper-bounded by the $L_1$ norm of the functional gradient (taking $h=\mathrm{sgn}$). The significance of this step is translating the abstract concept of "calibration quality" into "whether the gradient is minimized"—the latter being exactly what all algorithms **iterating along functional gradients** (Boosting, Kernel Boosting, NN under the NTK perspective) naturally do. It also explains the long-standing empirical observation: gradient boosting often works well for calibration out-of-the-box because it inherently minimizes the functional gradient.

**3. Three Algorithm Instantiations + Margin Assumption: Joint Guarantees for Calibration and Accuracy**

With the first two stages established, the remaining part is substituting the three "functional gradient-characterized" algorithms and adding a standard **margin assumption** (existence of $\gamma>0$ such that for any weighted sample distribution, a base learner with non-trivial correlation to labels can be found) to drive the functional gradient norm to convergence. Each algorithm provides a decay rate for training smooth CE, overlaid with their respective Rademacher complexities (which grow at $O(\sqrt{wT})$ or $O(wT)$ with iterations, revealing the trade-off between "reducing training CE ↔ increasing model complexity"). After hyperparameter selection, a unified $\epsilon$-guarantee is obtained:

- **GBT**: The average predictor satisfies $\mathrm{smCE}_\sigma(\bar g^{(T)},S_n)\le\frac{L_n(g^{(0)})}{\gamma BwT}+\frac{wB}{8\gamma}$. Setting $w=O(1/\sqrt T)$ yields convergence at $O(1/\sqrt T)$. Choosing $T=\Omega(\gamma^{-2}\epsilon^{-2})$ and $n=\tilde\Omega(\gamma^{-2}\epsilon^{-4})$ provides $\epsilon$-smooth CE and $\epsilon$-misclassification (**first analysis to simultaneously control GBT accuracy and smooth CE**).
- **Kernel Boosting**: Utilizing kernel operators to approximate functional gradients in RKHS, $\mathrm{smCE}_\sigma(\bar g^{(T)},S_{\mathrm{tr}})\le\frac{1}{\gamma}\sqrt{L_n(g^{(0)})/(wT)}$, similarly reaching the goal in $O(1/\epsilon^2)$ iterations.
- **Two-layer NN**: Via NTK, NN training is equated to kernel boosting, where $\frac1T\sum_t\|\nabla_g\ell\|^2_{L_1}$ is controlled. This yields bounds similar to kernel boosting. Specifically, the $\beta=0$ regime can reach the target with **fewer iterations** $T=\Theta(\gamma^{-2}\epsilon^{-1}\log^2(1/\epsilon))$ by increasing the number of hidden units $m$, offering better complexity than kernel boosting.

All three share the same "functional gradient → smooth CE" backbone, with differences only in regularization forms (implicit regularization in trees / RKHS norm / NTK), demonstrating the unity of the framework.

### Loss & Training
This paper does not introduce new losses or algorithms but utilizes standard squared loss $\ell_{\mathrm{sq}}$ and cross-entropy $\ell_{\mathrm{ent}}$ (both proper losses). The objects of analysis are the **existing** functional gradient iteration processes of these algorithms. Key strategic quantities are the constant step size $w$ and iteration count $T$. The authors prove that taking $w=O(1/\sqrt T)$ ensures training smooth CE convergence, while $T$ controls the trade-off between convergence and model complexity.

## Key Experimental Results

> This paper is primarily theoretical. The main results are the sample/iteration complexities required for each algorithm to achieve $\epsilon$-level smooth CE and misclassification rates. Numerical experiments (Appendix K) verify the trends of smooth CE and accuracy relative to iteration $T$ and sample size $n$. Source code is provided with the supplementary materials.

### Comparison of Theoretical Guarantees for the Three Algorithms

| Algorithm | Training smooth CE Upper Bound | Iterations $T$ for $\epsilon$ | Samples $n$ required | Remarks |
|------|---------------------|---------------------------|--------------|------|
| Gradient Boosted Trees (GBT) | $\frac{L_n(g^{(0)})}{\gamma BwT}+\frac{wB}{8\gamma}$ | $\Omega(\gamma^{-2}\epsilon^{-2})$ | $\tilde\Omega(\gamma^{-2}\epsilon^{-4})$ | First joint acc+smCE analysis |
| Kernel Boosting (RKHS) | $\frac1\gamma\sqrt{L_n(g^{(0)})/(wT)}$ | $\Omega(\gamma^{-2}\epsilon^{-2})$ | $\tilde\Omega(\gamma^{-2}\epsilon^{-4})$ | Complexity grows at $O(\sqrt{wT})$ |
| Two-layer NN ($\beta=0$) | NTK approx. Kernel Boosting | $\Theta(\gamma^{-2}\epsilon^{-1}\log^2\tfrac1\epsilon)$ | $\tilde\Omega(\epsilon^{-2})$ | More $m$ for fewer iterations |

### Ablation Study: Breakdown of the Boundary Sources

| Controlling Factor | Corresponding Term in Bound | Description |
|--------|-------------|------|
| Uniform Convergence (Training ↔ Population) | $C/\sqrt n + R_{D,n}(\mathcal{F})$ | Only includes $\mathcal{F}$ complexity, no Lipschitz class |
| Training smooth CE | $\le\|\nabla_g\ell\|_{L_1}$ | Upper-bounded by functional gradient norm |
| Complexity-Optimization Trade-off | $O(wT)$ / $O(\sqrt{wT})$ | Increasing $T$ reduces training CE but raises model complexity |

### Key Findings
- **Calibration quality reduces to "whether the gradient is minimized"**: Training smooth CE is upper-bounded by the $L_1$ norm of the functional gradient. This explains why gradient boosting algorithms often exhibit good calibration empirically.
- **Trade-off between training CE reduction and complexity control**: A larger iteration count $T$ leads to smaller training smooth CE, but Rademacher complexity grows accordingly, necessitating a balance via appropriate hyperparameters.
- **The $\beta=0$ regime for NN is most iteration-efficient**: By increasing the number of hidden units $m$, two-layer NNs can reach the $\epsilon$ target with fewer iterations than kernel boosting, demonstrating superior complexity.

## Highlights & Insights
- **Covering Number Chaining Argument Bypassing Composite Class Complexity**: smooth CE involves the composite structure $h(f(X))$. Naive analysis would pay the cost of $\mathrm{Lip}_1\circ\mathcal{F}$. The authors use the smoothness of smooth CE to ensure the final bound depends only on $\mathcal{F}$ complexity. This is a key "clean-up" technique for analysis involving Lipschitz test functions.
- **Perspective of "Calibration = Supremum of Functional Gradient Inner Product"**: Translating a statistical reliability concept into an optimization quantity turns the question of "which algorithm calibrates well" into "which algorithm minimizes the functional gradient," unifying GBT, kernel boosting, and NN.
- **Theoretical Explanation of Empirical Phenomena**: Rather than inventing new algorithms, the paper proves why **existing** gradient boosting is naturally calibrated. This "explaining existing success" work is arguably more valuable than creating a new method for guidance.

## Limitations & Future Work
- **Strong Margin Assumption**: The analysis relies on a margin assumption where data is well-separated. Since calibration is a much weaker requirement than accuracy, requiring strong data separability is somewhat restrictive; relaxing this to more realistic conditions is a vital direction.
- **Uniform Bounds for $h$ vs Lipschitz Class**: The bounds are obtained by taking a uniform upper bound over post-processing functions $h$, which might be loose. More refined analysis tailored to actual performance would be valuable.
- **Binary Classification Limit**: smooth CE is currently designed for binary classification; multi-class extension remains unresolved.
- **Constant Step Size Only**: Step size selection is critical for performance in boosting. Extending calibration analysis to variable step sizes is an open problem.

## Related Work & Insights
- **vs Błasiok et al. (2023)**: They proposed smooth CE and established the calibration-post-processing relationship but stayed at the population level. This work scales it down to finite samples using uniform convergence and functional gradients.
- **vs Futami & Fujisawa (2024)**: They provided algorithm-dependent generalization bounds for binning ECE using information theory with a convergence rate of $O(\log n/n^{1/3})$, focusing only on the generalization gap. This paper provides a faster, more general uniform convergence bound for smooth CE and characterizes when training CE decreases.
- **vs Traditional Boosting Analysis (Nitanda & Suzuki 2018, etc.)**: Previous functional gradient analyses focused solely on accuracy. This paper is the first to use functional gradients to characterize **calibration**, providing a new perspective on boosting algorithms.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First unified framework of uniform convergence + functional gradient for smooth CE applied to three mainstream algorithms.
- Experimental Thoroughness: ⭐⭐⭐ Primarily theoretical, with only appendix experiments verifying trends rather than large-scale empirical studies.
- Writing Quality: ⭐⭐⭐⭐ Clear proof chain with well-organized two-stage framework and case studies.
- Value: ⭐⭐⭐⭐ Provides theoretical criteria for designing provably calibrated models and explains the empirical calibration advantages of gradient boosting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Practical Estimation of the Optimal Classification Error with Soft Labels and Calibration](practical_estimation_of_the_optimal_classification_error_with_soft_labels_and_ca.md)
- [\[ICLR 2026\] Finite-Time Convergence Analysis of ODE-based Generative Models for Stochastic Interpolants](finite-time_convergence_analysis_of_ode-based_generative_models_for_stochastic_i.md)
- [\[ICLR 2026\] Slicing Wasserstein over Wasserstein via Functional Optimal Transport](slicing_wasserstein_over_wasserstein_via_functional_optimal_transport.md)
- [\[ICLR 2026\] Stable Coresets: Unleashing the Power of Uniform Sampling](stable_coresets_unleashing_the_power_of_uniform_sampling.md)
- [\[ICLR 2026\] A Sharp KL Convergence Analysis for Diffusion Models under Minimal Assumptions](a_sharp_kl_convergence_analysis_for_diffusion_models_under_minimal_assumptions.md)

</div>

<!-- RELATED:END -->
