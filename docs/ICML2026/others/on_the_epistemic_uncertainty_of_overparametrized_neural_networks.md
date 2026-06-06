---
title: >-
  [Paper Note] On the Epistemic Uncertainty of Overparametrized Neural Networks
description: >-
  [ICML 2026][Epistemic Uncertainty] This paper points out that the "epistemic uncertainty" of overparameterized neural networks does not vanish as the data scale increases: because the parameters are unidentifiable (due t…
tags:
  - "ICML 2026"
  - "Epistemic Uncertainty"
  - "Overparameterization"
  - "Unidentifiability"
  - "ReLU Network Posterior"
  - "Dirichlet Splitting"
date: 2026-05-08
content_hash: f918e149f25ba0a4
---

# On the Epistemic Uncertainty of Overparametrized Neural Networks

**Conference**: ICML 2026  
**arXiv**: [2605.25234](https://arxiv.org/abs/2605.25234)  
**Code**: None  
**Area**: Bayesian Neural Networks / Uncertainty Quantification / Learning Theory  
**Keywords**: Epistemic Uncertainty, Overparameterization, Unidentifiability, ReLU Network Posterior, Dirichlet Splitting

## TL;DR
This paper points out that the "epistemic uncertainty" of overparameterized neural networks does not vanish as the data scale increases: because the parameters are unidentifiable (due to permutation and neuron splitting), the parameter space posterior retains continuous uncertainty on the splitting manifold even when the function is fully identified. The authors provide a precise posterior description (Dirichlet on simplex) using single-hidden layer ReLU networks as an example and validate it empirically.

## Background & Motivation

**Background**: The uncertainty quantification (UQ) community conventionally defines epistemic uncertainty (EU) as "uncertainty that decays as the sample size increases," often measured via function-space variance or mutual information. Theoretically, the Bernstein–von Mises (BvM) theorem guarantees that the posterior contracts to the true value at a rate of $n^{-1/2}$.

**Limitations of Prior Work**: The premise of BvM is that the Fisher information is positive definite, meaning the parameters are locally identifiable. However, deep networks naturally possess permutation and rescaling symmetries. More severely, during overparameterization, a large number of "redundant neurons" make parameters completely unidentifiable, where multiple sets of parameters represent the same function. Existing studies on permutation/rescaling symmetries mainly discuss optimization and mode connectivity, while few have systematically discussed their impact on the "posterior-uncertainty" metric itself.

**Key Challenge**: While function-space EU indeed vanishes as $n\to\infty$ (since the function is identified), the weight-space posterior covariance does not vanish and may even diverge. If downstream tasks (e.g., Fisher importance in continual learning, sampling diagnostics, compression, interpretability) directly consume the weight-space posterior, they will be misled.

**Goal**: (i) Formalize why "uncertainty caused by unidentifiability" cannot be captured by predictive variance; (ii) Provide the exact posterior geometry of overparameterized ReLU networks (permutation patterns + continuous splitting manifolds); (iii) Explain how practical samplers (NUTS/SGLD) traverse these manifolds and what this implies for practice.

**Key Insight**: Starting from the known result that "two-layer ReLU networks under $L_2$ regularization are functionally equivalent $\iff$ parameter equivalent up to permutation + positive rescaling," the authors construct a set of surjective assignment mappings $\varsigma:[M]\to[M^\star]$. This explicitly parameterizes how "true neurons are replicated by multiple model neurons," thereby geometrizing unidentifiability into a Dirichlet simplex.

**Core Idea**: The unidentifiability is decomposed into a two-layer structure: an "assignment mapping $\varsigma$" and "splitting coefficients $c_m$." The former provides discrete permutation patterns, while the latter yields continuous simplex manifolds $\mathcal{M}_\varsigma\cong \prod_{m'} \Delta^{k_{m'}-1}$. The induced distribution of the posterior on these manifolds is exactly a symmetric Dirichlet, allowing the non-vanishing phenomenon of EU to be written in a computable closed form.

## Method

### Overall Architecture
The analysis focuses on a single-hidden layer ReLU network $f_\mathbf{w}(x)=\mathbf{w}_2^\top \phi(\mathbf{W}_1 x)$ with width $M$, where the true function $f^\star$ is implemented by a similar network of width $M^\star\le M$. The posterior $p(\mathbf{w}\mid \mathcal{D}_n)\propto \exp(-\sum_i \ell(y_i,f_\mathbf{w}(x_i)))\cdot \mathcal{N}(\mathbf{0},(2\lambda)^{-1}\mathbf{I})$ is analyzed in three parts: (1) Using a variance-based EU metric $\mathrm{EU}=\mathrm{tr}(\mathrm{Cov}(\mathbf{w}\mid \mathcal{D}_n))$ to explain why function-space metrics fail to see residual uncertainty in weight-space; (2) The $M=M^\star$ case—exact permutation unidentifiability, where the posterior in the large-$n$ limit is a uniform mixture of Dirac measures over $M!$ permutations; (3) The $M>M^\star$ case—incorporating continuous splitting, where the posterior concentrates on $\bigcup_\varsigma \mathcal{M}_\varsigma$ and follows a symmetric Dirichlet along the splitting coordinates within each $\mathcal{M}_\varsigma$.

### Key Designs

1.  **Variance-based EU Definition + Non-zero Residual under Unidentifiability**:
    - **Function**: Replace the classical $\mathrm{Var}_\mathcal{P}(f_\mathbf{w}(x))$ with $\mathrm{EU}(y,\mathbf{w}\mid x,\mathcal{D}_n) = \mathrm{tr}(\mathrm{Cov}(\mathbf{w}\mid \mathcal{D}_n))$ to incorporate weight-space unidentifiability into EU.
    - **Mechanism**: In a linear deep network example $\mathcal{N}(\mathbf{w}_L^\top \mathbf{W}_{L-1}\cdots \mathbf{w}_1 x, \sigma^2)$, it is explicitly calculated that while function-space EU vanishes, the weight-space EU $=d\tau^2$ (where $\tau^2$ is prior variance). This constructs a minimal counterexample where the function is fully identified but parameter EU does not vanish.
    - **Design Motivation**: Traditional information-theoretic EU ($I(\mathbf{w};y)$) is naturally invisible to permutation equivalence classes. Using the trace of covariance is compatible with Bayesian definitions and captures non-identifiable directions, providing a metric basis for the precise characterization of ReLU networks.

2.  **Assignment-Splitting Decomposition for Overparameterization**:
    - **Function**: Map each "model neuron" to a "true neuron" and explicitly express how redundancy "splits" the contribution of one true neuron into multiple model neurons.
    - **Mechanism**: For each surjection $\varsigma:[M]\to[M^\star]$, groups $G_{m'}=\varsigma^{-1}(m')$ and splitting coefficients $(c_m)_{m\in G_{m'}}\in\Delta^{k_{m'}-1}$ are defined. It is proven that all solutions must satisfy $\mathbf{w}_{1,m}=\sqrt{c_m}\mathbf{w}_{1,\varsigma(m)}^\star,\ w_{2,m}=\sqrt{c_m}w_{2,\varsigma(m)}^\star$. Thus, the function-equivalence class geometrically corresponds to a product of simplices $\mathcal{M}_\varsigma \cong \prod_{m'=1}^{M^\star}\Delta^{k_{m'}-1}$.
    - **Design Motivation**: This decomposition makes the "continuous degrees of freedom introduced by overparameterization" fully explicit, reducing the study of the posterior to the study of induced distributions on simplices. This is the geometric prerequisite for proving that the interiors of manifolds for different $\varsigma$ do not intersect (Lemma 3).

3.  **Dirichlet Posterior Closed-form + Balanced Scaling Theorem**:
    - **Function**: Provide exact posterior moments for splitting coefficients and neuron parameter blocks, answering "how much EU overparameterization introduces."
    - **Mechanism**: Using an $\varepsilon$-tube induced conditional distribution to define the posterior $\mathbb{P}_n^\varsigma$ on the manifold $\mathcal{M}_\varsigma$, it is proven that $(c_m)_{m\in G_{m'}}\sim \mathrm{Dir}(\alpha,\ldots,\alpha)$ with $\alpha=(p+1)/2$. This yields $\mathbb{E}[c_m]=k_{m'}^{-1}$ and $\mathrm{Cov}(c_m,c_{\tilde m})=-1/\kappa$, leading to parameter block moments $\mathbb{E}[\boldsymbol{\omega}_m]=\mu_{k,\alpha}\,\boldsymbol{\omega}_{m'}^\star$ and $\mathbb{E}[\boldsymbol{\omega}_m\boldsymbol{\omega}_m^\top]=k_{m'}^{-1}\boldsymbol{\omega}_{m'}^\star \boldsymbol{\omega}_{m'}^{\star\top}$. The balanced limit $k_{m'}\asymp M/M^\star, M\to\infty$ gives $\mathbb{E}[\boldsymbol{\omega}_m]=\Theta(M^{-1/2})$ and $\mathrm{Cov}=\Theta(M^{-1})$.
    - **Design Motivation**: Closed-form moments allow for rigorous comparison in experiments (empirical mean/second-order moments vs. $\mu_{k,\alpha}$, $1/k$). It clarifies that in the infinite-width limit, while individual neuron contributions shrink, the total group contribution remains constant, and the splitting degrees of freedom grow into a high-dimensional simplex.

### Loss & Training
The corresponding Bayesian model uses the loss $\mathcal{L}(\mathbf{w})=\sum_i \ell(y_i,f_\mathbf{w}(x_i))+\lambda\|\mathbf{w}\|_2^2$, which is equivalent to a Gaussian likelihood + Gaussian prior $\mathbf{w}\sim\mathcal{N}(\mathbf{0},(2\lambda)^{-1}\mathbf{I})$. In experiments, the posterior is obtained via a two-stage sampling: initialization with multi-chain Adam (equivalent to a deep ensemble of MAP), followed by sampling using SGLD or NUTS, resulting in a Bayesian Deep Ensemble (BDE).

## Key Experimental Results

### Main Results
All experiments were conducted on synthetic data: the true function is generated by a ReLU network with $M^\star=5$ hidden units, $p=5$, $y=f^\star(x)+\varrho$, where $\varrho\sim\mathcal{N}(0,1)$. Sample size $n\in\{2^6,\ldots,2^{14}\}$, and model width $M\in\{M^\star,2M^\star,4M^\star,8M^\star\}$. The table below summarizes the expectations and observations of the four core experiments.

| Experiment | Expectation (Theory) | Observation |
|------|------|------|
| Convergence (RMSE / LPPD, DE vs BDE) | DE and BDE are close at large $n$; BDE is better at small $n$; BDE captures continuous EU during overparameterization | All three observations align |
| function-space EU vs. weight-space trace EU | $\mathrm{Var}(f_\mathbf{w}(x))\downarrow 0$; $\mathrm{tr}(\mathrm{Cov}(\mathbf{w}))$ is constant | Fully aligned |
| Single-chain NUTS permutation switching rate | Should rarely cross under Lipschitz/continuous dynamics | Switch rate $<1$ chamber/chain; trends with $n, M$ match predictions |
| Splitting coefficient distribution | Marginal $c_m\sim\mathrm{Beta}(\alpha,(k-1)\alpha)$, $\alpha=(p+1)/2$ | Empirical histograms closely align with theoretical Beta |

### Ablation Study
The following table compares the differences in four downstream properties with and without explicit handling of unidentifiability.

| Configuration | function-space EU ($n\to\infty$) | weight trace EU ($n\to\infty$) | $\mathrm{Cov}(\boldsymbol{\omega}_m,\boldsymbol{\omega}_{m'})$ | splitting distribution |
|------|------|------|------|------|
| $M=M^\star$ Single chain/chamber | $0$ | $\sum_i \upsilon_i^2 \ne 0$ | $0$ (within single chamber) | N/A |
| $M=M^\star$ Full permutation mix | $0$ | $\sum_i \upsilon_i^2$ | $O(M^{-1})$ cross-block covariance | N/A |
| $M>M^\star$ Fixed $\varsigma$ | $0$ | Non-degenerate | Determined by Dirichlet | $\mathrm{Dir}(\alpha,\ldots,\alpha)$ |
| $M>M^\star$ Balanced limit $M\to\infty$ | $0$ | Indiv. $\boldsymbol{\omega}_m$ scales to $\Theta(M^{-1})$; group total constant | $\Theta(M^{-1})\boldsymbol{\omega}_{m'}^\star \boldsymbol{\omega}_{m'}^{\star\top}$ | $\alpha$ constant, dim$\uparrow$ |

### Key Findings
- As $n\to\infty$, function-space EU vanishes while weight-space EU does not; the former arises from function identification, the latter from parameter unidentifiability, and they are fundamentally different.
- Single-chain NUTS / SGLD rarely cross permutation chambers under reasonable learning rates, implying that a multi-chain ensemble is necessary; otherwise, EU estimates are naturally biased.
- In overparameterized settings, a single chain can freely wander within $\mathcal{M}_\varsigma^\circ$ but will not cross different $\varsigma$; this means empirical EU estimates are simultaneously affected by "single-chain manifold wandering" and "multi-chain chamber coverage" biases.
- The influence of the prior is subtle: it has no effect in the tangential direction of $\mathcal{M}_\varsigma$, but shapes near-Gaussian fluctuations in the normal direction. This explains why empirical weight marginals often look Gaussian.
- In continual learning, Fisher importance is "amplified" by non-identifiability, leading to excessively restricted updates; correction can improve adaptation performance.

## Highlights & Insights
- The difference between deep ensembles (DE) and BNNs in the "overparameterization + large $n$" limit is clarified: DE cannot capture continuous splitting degrees of freedom, while BDE can. This provides a theoretical basis for BDE's superiority over DE.
- Using a Dirichlet simplex to provide a precise distribution for "splitting" is an elegant tool for geometrizing unidentifiability. The idea can be extended to biases, multi-layer MLPs, or even redundant dimension analysis in LoRA/low-rank adaptation.
- The assertion that "single-chain samplers rarely cross permutation chambers" provides clear meaning for sampling diagnostics: traditional metrics like R̂ will overestimate convergence if chamber switching is ignored. The authors propose a diagnostic that subtracts variance in non-identifiable directions.

## Limitations & Future Work
- The theory strictly holds only under single-hidden layer ReLU + Gaussian prior + corresponding regularization. Mainstream architectures like multi-layer nets, non-ReLU activations, and Batch Norm/skip connections would break the clean splitting geometry of Corollary 2.
- Experiments were entirely conducted on synthetic data, without validating how much the EU estimation bias actually affects decisions (e.g., OOD detection, active learning) in real image/NLP tasks.
- The proposed splitting distribution relies on the premise "$L_2$ regularization + functional equivalence $\implies$ parameter equivalence up to permutation/scaling." It is unclear if this applies to implicit regularization (e.g., SGD bias) beyond dropout/weight decay.
- No specific correction algorithms were provided (except for two cases in §B.5); converting "knowing EU is overestimated" into a practical UQ pipeline remains for future work.

## Related Work & Insights
- **vs Hüllermeier & Waegeman (Information-theoretic EU decomposition)**: They define EU using mutual information, which is invisible to permutation equivalence classes; this paper identifies this as a source of systematic underestimation in overparameterized scenarios and uses a variance-based metric instead.
- **vs Simsek et al. (2021) (Symmetry in overparameterized landscapes)**: They characterize connected manifolds on the optimization landscape; this paper brings the "manifold" concept to the Bayesian posterior, proving it leads to non-vanishing posterior mass.
- **vs Kobialka et al. (2026) (Splitting geometry)**: This paper adopts their surjective assignment tools but is the first to link them to the Bayesian posterior, providing a closed-form Dirichlet posterior.
- **vs Deep Ensembles (Lakshminarayanan et al.)**: This paper proves that for $M=M^\star$ and large $n$, DE exactly matches the true posterior in permutation patterns—a rare theoretical support for DE—but DE is insufficient to express continuous splitting for $M>M^\star$, necessitating BDE.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Geometrizing continuous unidentifiability into a Dirichlet simplex and providing closed-form moments is a rare precise characterization in BNN theory.
- Experimental Thoroughness: ⭐⭐⭐ Validation on synthetic data is thorough, but lacks evaluation of downstream impacts on real tasks.
- Writing Quality: ⭐⭐⭐⭐ Concepts are clear; Figure 1/2 provides intuitive visualizations of unidentifiable manifolds; the theorem-corollary chain is compact.
- Value: ⭐⭐⭐⭐ Directly relevant to the practice of BNNs, deep ensembles, continual learning, and sampling diagnostics, and changes the common perception that "epistemic uncertainty vanishes with data."

## Rating
- Novelty: To be rated
- Experimental Thoroughness: To be rated
- Writing Quality: To be rated
- Value: To be rated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation](mmd-balls_as_credal_sets_a_pac-bayesian_framework_for_epistemic_uncertainty_in_t.md)
- [\[ICML 2026\] Singular Bayesian Neural Networks](singular_bayesian_neural_networks.md)
- [\[ICML 2026\] Possibilistic Predictive Uncertainty for Deep Learning](possibilistic_predictive_uncertainty_for_deep_learning.md)
- [\[ICML 2026\] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](the_implicit_bias_of_adam_and_muon_on_smooth_homogeneous_neural_networks.md)
- [\[ICML 2026\] Bullet Trains: Parallelizing Training of Temporally Precise Spiking Neural Networks](bullet_trains_parallelizing_training_of_temporally_precise_spiking_neural_networ.md)

</div>

<!-- RELATED:END -->
