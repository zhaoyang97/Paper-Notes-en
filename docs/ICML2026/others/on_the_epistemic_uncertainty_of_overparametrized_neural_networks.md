---
title: >-
  [Paper Note] On the Epistemic Uncertainty of Overparametrized Neural Networks
description: >-
  [ICML 2026][Epistemic Uncertainty] This paper points out that the "epistemic uncertainty" of overparameterized neural networks does not vanish as the data volume increases. Due to parameter unidentifiability (permutation + neuron splitting), even if the function is fully identified, the parameter space posterior still retains continuous uncertainty on the splitting manifold. Using single-hidden-layer ReLU networks as an example, the authors provide a precise posterior descrip…
tags:
  - "ICML 2026"
  - "Epistemic Uncertainty"
  - "Overparameterization"
  - "Unidentifiability"
  - "ReLU Network Posterior"
  - "Dirichlet Splitting"
date: 2026-05-08
content_hash: b991241de417a0fc
---

# On the Epistemic Uncertainty of Overparametrized Neural Networks

**Conference**: ICML 2026  
**arXiv**: [2605.25234](https://arxiv.org/abs/2605.25234)  
**Code**: None  
**Area**: Bayesian Neural Networks / Uncertainty Quantification / Learning Theory  
**Keywords**: Epistemic Uncertainty, Overparameterization, Unidentifiability, ReLU Network Posterior, Dirichlet Splitting

## TL;DR
This paper points out that the "epistemic uncertainty" of overparameterized neural networks does not vanish as the data volume increases. Due to parameter unidentifiability (permutation + neuron splitting), even if the function is fully identified, the parameter space posterior still retains continuous uncertainty on the splitting manifold. Using single-hidden-layer ReLU networks as an example, the authors provide a precise posterior description (Dirichlet on simplex) and empirical validation.

## Background & Motivation

**Background**: The UQ community typically defines epistemic uncertainty (EU) as "uncertainty that decays as the sample size increases," often measured through function-space variance or mutual information. Theoretically, the Bernstein–von Mises (BvM) theorem guarantees that the posterior contracts to the ground truth at a rate of $n^{-1/2}$.

**Limitations of Prior Work**: The premise of BvM is that the Fisher information is positive definite, implying local parameter identifiability. However, deep networks naturally possess permutation and rescaling symmetries. Worse, in overparameterized settings, a large number of "redundant neurons" make parameters completely unidentifiable, with multiple sets of parameters expressing the same function. Existing research on permutation/rescaling symmetries mostly focuses on optimization and mode connectivity, while systematic discussion of their impact on the "posterior-uncertainty" metric itself is lacking.

**Key Challenge**: Function-space EU indeed vanishes as $n\to\infty$ (since the function is identified), but the weight-space posterior covariance does not vanish and may even diverge. If downstream tasks (such as Fisher importance in continual learning, sampling diagnostics, compression, or interpretability) directly consume the weight-space posterior, they will be misled.

**Goal**: (i) Formalize why "uncertainty caused by unidentifiability" cannot be captured by predictive variance; (ii) Provide the precise posterior geometry of overparameterized ReLU networks (permutation modes + continuous splitting manifolds); (iii) Explain how practical samplers (NUTS/SGLD) traverse these manifolds and what this implies for practice.

**Key Insight**: Starting from the known result that "two-layer ReLU networks under $L_2$ regularization are functionally equivalent $\iff$ parameters are equivalent up to permutation + positive rescaling," a set of surjective assignment mappings $\varsigma:[M]\to[M^\star]$ is constructed. This explicitly parameterizes how "true neurons are replicated by several model neurons," thereby geometricizing unidentifiability into Dirichlet simplices.

**Core Idea**: Unidentifiability is decomposed into a two-level structure: an "assignment mapping $\varsigma$" and "splitting coefficients $c_m$." The former provides discrete permutation patterns, while the latter yields continuous simplex manifolds $\mathcal{M}_\varsigma\cong \prod_{m'} \Delta^{k_{m'}-1}$. The induced distribution of the posterior on these manifolds is exactly a symmetric Dirichlet, allowing the non-vanishing phenomenon of EU to be expressed in a computable closed form.

## Method

### Overall Architecture
The analysis focuses on single-hidden-layer ReLU networks $f_\mathbf{w}(x)=\mathbf{w}_2^\top \phi(\mathbf{W}_1 x)$ with width $M$, where the true function $f^\star$ is realized by a similar network of width $M^\star\le M$. The posterior $p(\mathbf{w}\mid \mathcal{D}_n)\propto \exp(-\sum_i \ell(y_i,f_\mathbf{w}(x_i)))\cdot \mathcal{N}(\mathbf{0},(2\lambda)^{-1}\mathbf{I})$ is analyzed in three parts: (1) Using variance-based EU to measure $\mathrm{EU}=\mathrm{tr}(\mathrm{Cov}(\mathbf{w}\mid \mathcal{D}_n))$ to explain why function-space metrics miss residual uncertainty in weight-space; (2) The $M=M^\star$ case—precise permutation unidentifiability, where the posterior in the large $n$ limit is a uniform mixture of Diracs over $M!$ permutations; (3) The $M>M^\star$ case—introducing continuous splitting, where the posterior concentrates on $\bigcup_\varsigma \mathcal{M}_\varsigma$ and follows a symmetric Dirichlet along splitting coordinates within each $\mathcal{M}_\varsigma$.

### Key Designs

**1. Variance-based EU Definition + Non-zero Residual under Unidentifiability: Making EU sensitive to weight-space residual uncertainty**

Information-theoretic EU metrics (such as $I(\mathbf{w};y)$ or function-space variance) commonly used in the UQ community are naturally blind to permutation equivalence classes—they vanish once the function is identified, failing to see the remaining uncertainty in parameter space. The authors adopt $\mathrm{EU}(y,\mathbf{w}\mid x,\mathcal{D}_n)=\mathrm{tr}(\mathrm{Cov}(\mathbf{w}\mid\mathcal{D}_n))$, the trace of the weight-space posterior covariance, to incorporate unidentifiability into the metric. A simple counterexample illustrates this: in a linear deep network $\mathcal{N}(\mathbf{w}_L^\top\mathbf{W}_{L-1}\cdots\mathbf{w}_1 x,\sigma^2)$, one can explicitly calculate that function-space EU vanishes while weight-space EU $=d\tau^2$ ($\tau^2$ is the prior variance)—the function is fully identified, yet parameter EU does not disappear. This metric is compatible with Bayesian definitions while capturing unidentifiable directions, laying the foundation for the precise characterization of ReLU networks.

**2. Overparameterized Assignment-Splitting Decomposition: Explicitly parameterizing the continuous degrees of freedom of redundant neurons**

To answer "how much non-vanishing uncertainty overparameterization introduces," the mechanism of redundancy must be clearly articulated. For each surjection $\varsigma:[M]\to[M^\star]$ (model neurons $\to$ true neurons), the authors define groups $G_{m'}=\varsigma^{-1}(m')$ and splitting coefficients $(c_m)_{m\in G_{m'}}\in\Delta^{k_{m'}-1}$, proving that all solutions must satisfy:

$$\mathbf{w}_{1,m}=\sqrt{c_m}\,\mathbf{w}_{1,\varsigma(m)}^\star,\qquad w_{2,m}=\sqrt{c_m}\,w_{2,\varsigma(m)}^\star,$$

Thus, function equivalence classes geometrically correspond to products of simplices $\mathcal{M}_\varsigma\cong\prod_{m'=1}^{M^\star}\Delta^{k_{m'}-1}$. This decomposition explicitly characterizes how "a true neuron is replicated by several model neurons with contributions shared via $c_m$," reducing posterior analysis to the study of induced distributions on simplices. It is also the geometric prerequisite for Lemma 3, which proves that the open interiors of different $\mathcal{M}_\varsigma$ manifolds are almost disjoint (continuous dynamical systems rarely cross them).

**3. Dirichlet Posterior Closed-form + Balanced Scaling Theorem: Providing a precise characterization of "EU Redistribution rather than Elimination"**

With the manifold structure established, the final step provides precise posterior moments for splitting coefficients and parameter blocks. Defining the posterior on $\mathcal{M}_\varsigma$ using an $\varepsilon$-tube induced conditional distribution, the authors prove $(c_m)_{m\in G_{m'}}\sim\mathrm{Dir}(\alpha,\dots,\alpha)$ where $\alpha=(p+1)/2$. Consequently, $\mathbb{E}[c_m]=k_{m'}^{-1}$, $\mathrm{Cov}(c_m,c_{\tilde m})=-1/\kappa$, leading to $\mathbb{E}[\boldsymbol{\omega}_m]=\mu_{k,\alpha}\boldsymbol{\omega}_{m'}^\star$ and $\mathbb{E}[\boldsymbol{\omega}_m\boldsymbol{\omega}_m^\top]=k_{m'}^{-1}\boldsymbol{\omega}_{m'}^\star\boldsymbol{\omega}_{m'}^{\star\top}$. In the balanced limit $k_{m'}\asymp M/M^\star,\ M\to\infty$, we obtain $\mathbb{E}[\boldsymbol{\omega}_m]=\Theta(M^{-1/2})$ and $\mathrm{Cov}=\Theta(M^{-1})$. These closed-form moments allow for rigorous experimental comparison (measured mean/second-order moments vs. $\mu_{k,\alpha}$, $1/k$) and highlight the core conclusion: in the infinite-width limit, individual neuron contributions shrink, but the total contribution of the group remains constant, and the splitting degrees of freedom grow into high-dimensional simplices—under overparameterization, EU is redistributed rather than eliminated.

### Loss & Training
The corresponding Bayesian model uses the loss: $\mathcal{L}(\mathbf{w})=\sum_i \ell(y_i,f_\mathbf{w}(x_i))+\lambda\|\mathbf{w}\|_2^2$, equivalent to Gaussian likelihood + Gaussian prior $\mathbf{w}\sim\mathcal{N}(\mathbf{0},(2\lambda)^{-1}\mathbf{I})$. In experiments, the posterior is obtained via a two-stage sampling: initialization with multi-chain Adam (equivalent to a deep ensemble of MAP), followed by SGLD or NUTS sampling to obtain a Bayesian Deep Ensemble (BDE).

## Key Experimental Results

### Main Results
All experiments were conducted on synthetic data: the true function is generated by a ReLU net with $M^\star=5$, $p=5$, $y=f^\star(x)+\varrho$, where $\varrho\sim\mathcal{N}(0,1)$, sample size $n\in\{2^6,\ldots,2^{14}\}$, and model width $M\in\{M^\star,2M^\star,4M^\star,8M^\star\}$. The table below summarizes expectations and observations from four core experiments.

| Experiment | Expectation (from Theory) | Observation |
|------|------|------|
| Convergence (RMSE / LPPD, DE vs BDE) | DE/BDE converge at large $n$; BDE superior at small $n$; BDE captures additional continuous EU when overparameterized | All three align with measurements |
| function-space EU vs. weight-space trace EU | $\mathrm{Var}(f_\mathbf{w}(x))\downarrow 0$; $\mathrm{tr}(\mathrm{Cov}(\mathbf{w}))$ constant | Fully aligned |
| Single-chain NUTS cross-permutation switch rate | Should rarely cross under Lipschitz/continuous dynamics | switch rate $<1$ chamber/chain; trends with $n,M$ match predictions |
| Splitting coefficient distribution | Marginal $c_m\sim\mathrm{Beta}(\alpha,(k-1)\alpha)$, $\alpha=(p+1)/2$ | Empirical histograms closely align with theoretical Beta |

### Ablation Study
The table below compares the impact of "treating vs. not treating unidentifiability explicitly" on four downstream properties, corresponding to Lemma 1, Corollary 1, and Theorem 1 in the original paper.

| Configuration | function-space EU ($n\to\infty$) | weight trace EU ($n\to\infty$) | $\mathrm{Cov}(\boldsymbol{\omega}_m,\boldsymbol{\omega}_{m'})$ | Splitting distribution |
|------|------|------|------|------|
| $M=M^\star$ single chain/single chamber | $0$ | $\sum_i \upsilon_i^2 \ne 0$ | $0$ (within single chamber) | N/A |
| $M=M^\star$ full permutation mixture | $0$ | $\sum_i \upsilon_i^2$ | $O(M^{-1})$ inter-block covariance | N/A |
| $M>M^\star$ fixed $\varsigma$ | $0$ | Non-degenerate | Determined by Dirichlet | $\mathrm{Dir}(\alpha,\ldots,\alpha)$ |
| $M>M^\star$ balanced limit $M\to\infty$ | $0$ | Individual $\boldsymbol{\omega}_m$ shrinks to $\Theta(M^{-1})$; group sum invariant | $\Theta(M^{-1})\boldsymbol{\omega}_{m'}^\star \boldsymbol{\omega}_{m'}^{\star\top}$ | $\alpha$ invariant, dim$\uparrow$ |

### Key Findings
- After $n\to\infty$, function-space EU vanishes while weight-space EU does not. The former stems from function identification, the latter from parameter unidentifiability; they are fundamentally different.
- Single-chain NUTS / SGLD rarely cross permutation chambers under reasonable learning rates, implying that multi-chain ensembles are necessary, otherwise EU estimation is naturally biased.
- Under overparameterization, a single chain can freely traverse within $\mathcal{M}_\varsigma^\circ$ but will not cross different $\varsigma$; this means empirical EU estimates are simultaneously influenced by "single-chain manifold traversal" and "multi-chain chamber coverage" biases.
- The prior's effect is subtle: it has no effect in directions tangential to $\mathcal{M}_\varsigma$ (since $\sum_m c_m\|\boldsymbol{\omega}_{m'}^\star\|^2$ is constant tangentially), but it shapes near-Gaussian fluctuations in the normal direction. This explains why weight marginals often appear Gaussian in practice.
- In continual learning, Fisher importance is "amplified" by non-identifiability, leading to overly restricted updates; correction can improve adaptation performance (Appendix §B.5).

## Highlights & Insights
- Clarifies the difference between deep ensembles and BNNs in the "overparameterization + large $n$" limit: DE cannot capture continuous splitting degrees of freedom, while BDE can. This provides a theoretical basis for BDE's superiority over DE beyond empirical claims.
- Using Dirichlet simplices to provide the exact distribution of "splitting" is an elegant tool for geometricizing unidentifiability. The approach can be generalized to biases, multi-layer MLPs, and even redundant dimension analysis in LoRA/low-rank adaptation.
- The assertion that "single-chain samplers rarely cross permutation chambers" provides clear meaning for sampling diagnostics: traditional metrics like $\hat{R}$ will overestimate convergence if chamber switching is ignored. The authors propose a diagnostic that subtracts variance from unidentifiable directions (§B.5).

## Limitations & Future Work
- The theory is strictly valid only for single-hidden-layer ReLU + Gaussian prior + corresponding regularization. Multi-layer networks, non-ReLU activations, Batch Norm, and skip connections all break the clean splitting geometry of Corollary 2, which the authors acknowledge as a limitation of the main scope.
- Experiments are conducted entirely on synthetic data, lacking validation on real image/NLP tasks to see how much EU estimation bias actually affects decisions (e.g., OOD detection, active learning); a common limitation for theoretical papers.
- The proposed splitting distribution relies on the premise "$L_2$ regularization + functional equivalence $\implies$ parameter equivalence up to permutation/scaling." It is unclear if this applies to implicit regularization (e.g., SGD implicit bias) beyond dropout/weight decay.
- No specific correction algorithm is provided (except for two cases in §B.5). Transforming the knowledge that "EU is overestimated" into a practical UQ pipeline under the premise of underestimation remains for future work.

## Related Work & Insights
- **vs. Hüllermeier & Waegeman's info-theoretic EU decomposition**: They define EU using mutual information, which is blind to permutation equivalence classes. This paper points out this as a source of systematic underestimation in overparameterized scenarios and uses a variance-based metric instead.
- **vs. Simsek et al. (2021) on overparameterized network loss landscape symmetry**: They characterize connected manifolds in the optimization landscape; this paper brings the "manifold" concept to the Bayesian posterior, proving it yields non-vanishing parameter posterior mass.
- **vs. Kobialka et al. (2026) splitting geometry**: This paper borrows the surjective assignment tool but is the first to link it to the Bayesian posterior, providing a closed-form Dirichlet posterior.
- **vs. Deep Ensembles (Lakshminarayanan et al.)**: This paper proves that for $M=M^\star$ + large $n$, DE exactly matches the true posterior in permutation modes—a rare theoretical support for DE. However, for $M>M^\star$, DE is insufficient to express continuous splitting, necessitating BDE.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Geometricizing overparameterized continuous unidentifiability as Dirichlet simplices and providing closed-form posterior moments is a rare precise characterization in BNN theory.
- Experimental Thoroughness: ⭐⭐⭐ Validation on synthetic data is sound, but it lacks impact assessments on downstream real-world tasks.
- Writing Quality: ⭐⭐⭐⭐ Concepts are clear; Figures 1 and 2 visualize unidentifiable manifolds intuitively; the Theorem-Corollary chain is compact.
- Value: ⭐⭐⭐⭐ Has direct implications for BNNs, deep ensembles, continual learning, and sampling diagnostics, and changes the common perception that "epistemic uncertainty vanishes with data."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Rethinking Aleatoric and Epistemic Uncertainty](../../ICML2025/others/rethinking_aleatoric_and_epistemic_uncertainty.md)
- [\[ICML 2025\] Improved Exploration in GFlowNets via Enhanced Epistemic Neural Networks](../../ICML2025/others/improved_exploration_in_gflownets_via_enhanced_epistemic_neural_networks.md)
- [\[ICML 2026\] Possibilistic Predictive Uncertainty for Deep Learning](possibilistic_predictive_uncertainty_for_deep_learning.md)
- [\[ICML 2026\] Bullet Trains: Parallelizing Training of Temporally Precise Spiking Neural Networks](bullet_trains_parallelizing_training_of_temporally_precise_spiking_neural_networ.md)
- [\[CVPR 2025\] Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach](../../CVPR2025/others/rethinking_epistemic_and_aleatoric_uncertainty_for_active_open-set_annotation_an.md)

</div>

<!-- RELATED:END -->
