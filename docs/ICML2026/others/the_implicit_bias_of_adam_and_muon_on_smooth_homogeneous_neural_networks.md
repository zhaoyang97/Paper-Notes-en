---
title: >-
  [Paper Note] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks
description: >-
  [ICML 2026][Implicit Bias] This paper proves that under the setting of smooth $L$-homogeneous models + exponential tail loss + learning rate decay, Muon (including Muon-Signum and Muon-Adam)…
tags:
  - "ICML 2026"
  - "Implicit Bias"
  - "Adam"
  - "Muon"
  - "Homogeneous Networks"
  - "Max-margin"
  - "Spectral Norm"
date: 2026-05-08
content_hash: f695e21d9b6448cb
---

# The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks

**Conference**: ICML 2026  
**arXiv**: [2602.16340](https://arxiv.org/abs/2602.16340)  
**Code**: None  
**Area**: Optimization Theory / Implicit Bias / LLM Training  
**Keywords**: Implicit Bias, Adam, Muon, Homogeneous Networks, Max-margin, Spectral Norm  

## TL;DR
This paper proves that under the setting of smooth $L$-homogeneous models + exponential tail loss + learning rate decay, Muon (including Muon-Signum and Muon-Adam), acting as momentum-based "normalized steepest descent," converges to the KKT points of the corresponding norm max-margin problem. Meanwhile, Adam (without the stability constant) converges to the KKT points of the $\ell_\infty$ max-margin problem, thereby extending implicit bias conclusions previously only valid for linear models to all smooth homogeneous networks.

## Background & Motivation

**Background**: The mystery of generalization in over-parameterized neural networks is widely attributed to the "implicit bias" of optimizers; even without explicit regularization, gradient-based algorithms tend to converge to some maximum margin solution. Starting from the proof by Soudry et al. (2017) that GD maximizes the $\ell_2$-margin on linear models, to Lyu & Li (2019) extending this to the KKT form for any $L$-homogeneous (deep ReLU) network, the landscape of implicit bias for gradient descent has become largely complete.

**Limitations of Prior Work**: However, industry training for LLMs and ViTs no longer employs GD, but rather adaptive optimizers with momentum like Adam and Muon. Their implicit biases have only been characterized for linear predictors (Zhang et al. 2024, Fan et al. 2025), lacking theory for truly nonlinear networks. Furthermore, earlier analyses of Adam (Wang et al. 2021) retained the stability constant $\varepsilon$, causing $\sqrt{v_t}$ to be asymptotically dominated by $\varepsilon$ and essentially degenerating Adam into GD—a serious disconnect from the practical norm where $\varepsilon$ is negligible.

**Key Challenge**: The update direction of momentum algorithms (MSD = momentum steepest descent) no longer strictly satisfies the algebraic condition of "steepest descent" ($\langle\dot{\boldsymbol{\theta}}/\|\dot{\boldsymbol{\theta}}\|, -\boldsymbol{g}/\|\boldsymbol{g}\|_\star\rangle=1$). However, the overshoot introduced by momentum is only of order $o(1)$, requiring new tools to characterize this "approximate steepest descent." Adam is even more complex, being the ratio of two momentums with different decay rates ($\hat{\boldsymbol{m}}_t/\sqrt{\hat{\boldsymbol{v}}_t}$), which does not fall under any standard MSD paradigm.

**Goal**: (i) Provide KKT convergence proofs for Muon/MSD on smooth homogeneous models; (ii) offer similar results for $\varepsilon$-free Adam; (iii) derive the implicit norms for hybrid algorithms (Muon-Signum, Muon-Adam); (iv) explain why momentum does not destroy the max-margin bias using a unified framework ("Approximate Steepest Descent").

**Key Insight**: By analyzing in continuous time (gradient flow limit), the asymptotic relationship between momentum $\boldsymbol{m}_t=\int_0^t c_1 e^{-c_1(t-s)}\boldsymbol{g}_s\,ds$ and the instantaneous gradient is characterized as $\boldsymbol{m}_t[j]=\boldsymbol{g}_t[j](1\pm o(1))$ (for "instantaneously significant" coordinates $j$), such that momentum becomes an asymptotic first-order approximation of the gradient.

**Core Idea**: Define "Approximate Steepest Descent"—as long as the infimum of the alignment between the update direction and the negative gradient is asymptotically $\ge 1$, and the parameter norm is controlled by a certain integral upper bound, it suffices to derive the max-margin KKT points, bypassing the need for an exact characterization of momentum.

## Method

### Overall Architecture
The analysis focuses on continuous-time flows with the following unified forms:

- **(Normalized) MSD**: $d\boldsymbol{\theta}_t/dt\in\eta(t)\arg\min_{\|\boldsymbol{u}\|=1}\langle\boldsymbol{u},\boldsymbol{m}_t\rangle$, where momentum $\boldsymbol{m}_t$ evolves as $d\boldsymbol{m}_t/dt=c_1(\boldsymbol{g}_t-\boldsymbol{m}_t)$ ($c_1\sim -\log\beta_1$).
- **Muon = Normalized MSD under the matrix spectral norm**: $\|\cdot\|=\|\cdot\|_{\mathrm{sp}}$ for single layers, $\|\cdot\|_{\mathrm{msp}}=\max_k\|W_k\|_{\mathrm{sp}}$ for multiple layers.
- **Adam** (without $\varepsilon$): $d\boldsymbol{\theta}_t/dt=-\eta(t)\,\hat{\boldsymbol{m}}_t/\sqrt{\hat{\boldsymbol{v}}_t}$, where $\boldsymbol{v}_t$ evolves as $d\boldsymbol{v}_t/dt=c_2(\boldsymbol{g}_t^2-\boldsymbol{v}_t)$.

Model assumptions include (M1) $f\in C^1$, (M2) $L$-homogeneity $f(\boldsymbol{x};\alpha\boldsymbol{\theta})=\alpha^L f(\boldsymbol{x};\boldsymbol{\theta})$. The loss $\mathcal{L}(\boldsymbol{\theta})=\sum_i e^{-\varphi(y_i f(\boldsymbol{x}_i;\boldsymbol{\theta}))}$ covers both exponential and logistic types. Trajectory assumptions include (T1) non-vanishing norm, (T2) directional convergence with positive margin, and (A1) non-zero Adam initial effective gradient. Learning rate decay (LR-MSD/LR-Adam): $\int\eta=\infty$ and $\eta(t)\le o(t^{1/L-1})$ (satisfied by $\eta(t)=1/t$ for $L>1$).

### Key Designs

1. **Approximate Steepest Descent Framework (Definition 5.1 + Theorem C.17)**:
    - **Function**: Standardizes the max-margin bias of a wide range of optimizers into whether the "trajectory aligns with the negative gradient," without relying on exact momentum analytical solutions.
    - **Mechanism**: If for a trajectory $\boldsymbol{\theta}_t$, there exist $\nu(t), R_{\max}$ such that $N(t)=\int_0^t\nu\to\infty$, $\limsup\|\boldsymbol{\theta}_t\|/N(t)\le R_{\max}$, and the alignment infimum $\operatorname{ess\,liminf} r(t)\ge 1$ (where $r(t)=\sup_{\boldsymbol{g}_t}\langle\nu^{-1}\dot{\boldsymbol{\theta}}_t,-\boldsymbol{g}_t/\|\boldsymbol{g}_t\|_\star\rangle$), then under (T2) and $R_{\max}\le 1$, $\bar{\boldsymbol{\theta}}=\lim\boldsymbol{\theta}_t/\|\boldsymbol{\theta}_t\|$ must be a KKT point of Problem (11).
    - **Design Motivation**: Setting $\nu=\|\dot{\boldsymbol{\theta}}\|$ reduces MSD to classical steepest descent; Adam is not an MSD but fits this framework if $\nu=\eta(t)$ is chosen appropriately—this is the fundamental reason the paper can handle Adam and Muon together.

2. **Asymptotic Consistency of Momentum and Gradient (Lemma C.19)**:
    - **Function**: Proves that under LR decay $\|\dot{\boldsymbol{\theta}_t}\|\le o(t^{1/L-1})$, the momentum $\boldsymbol{m}_t$ equals the instantaneous gradient multiplied by $(1\pm o(1))$ across all "instantaneously significant" coordinates $J_\varepsilon(t)=\{j:|\boldsymbol{g}_t[j]|/\|\boldsymbol{g}_t\|_\star>\varepsilon\}$.
    - **Mechanism**: Corollary B.8 in Appendix B provides a key scalar fact—as long as $d\log g/dt$ converges, the scalar momentum ratio $m(t)/g(t)$ converges to a well-defined limit; LR decay ensures this convergence condition is automatically met along the training trajectory. This implies $\boldsymbol{m}_t/\|\boldsymbol{m}_t\|_\star-\boldsymbol{g}_t/\|\boldsymbol{g}_t\|_\star\to 0$, aligning the MSD update direction with the negative gradient.
    - **Design Motivation**: Satisfies the third alignment requirement of Definition 5.1 for momentum algorithms; simultaneously prepares the key approximation "$\hat{\boldsymbol{m}}_t[j]/\sqrt{\hat{\boldsymbol{v}}_t[j]}=\mathrm{sign}(\boldsymbol{g}_t[j])(1\pm o(1))$" for Adam—the latter being the bridge from Adam to sign gradient descent to $\ell_\infty$ max-margin.

3. **Composite Algorithms = Max-Norm MSD (Appendix C.6)**:
    - **Function**: Translates practical recipes like Muon-Signum and Muon-Adam, which use different optimizers for different parameter blocks, into a unified (approximate) MSD under a single norm.
    - **Mechanism**: If independent normalized (momentum) steepest descents are run in parallel on parameter blocks $\boldsymbol{\theta}=(W_1,\dots,W_K,\boldsymbol{u})$ with a shared $\eta(t)$, the ensemble is equivalent to a normalized MSD under the norm $\|\boldsymbol{\theta}\|=\max\{\|(W_1,\dots,W_K)\|_{\mathrm{msp}},\|\boldsymbol{u}\|_\infty\}$ (Corollary 3.4). Muon-Adam further allows different base learning rates $\eta_0^M, \eta_0^A$, resulting in a norm of $\max\{(\eta_0^A/\eta_0^M)\|(W_1,\dots,W_K)\|_{\mathrm{msp}},\|\boldsymbol{u}\|_\infty\}$ (Theorem 3.6).
    - **Design Motivation**: In practice, Muon often switches to Adam for non-matrix parameters (e.g., LayerNorm, bias) (Jordan et al. 2024, Liu et al. 2025); recently, Scion (Pethick et al. 2025) assembled Muon with sign GD. This "max-norm equivalence" is the first time the implicit margin objective of such hybrid training has been written analytically.

### Loss & Training
The study targets the optimizers themselves and does not rewrite the loss; all results hold under exponential/logistic loss, requiring $\varphi$ to be $C^2$, strictly monotonic, convex, with bounded derivatives (Appendix C.1). The model class includes deep linear networks and nonlinear networks with smooth homogeneous activations (e.g., $\mathrm{ReLU}^q\; (q>1)$, quadratic activation), but strictly speaking does not cover standard ReLU (only holding under the additional subgradient directional convergence assumption (T3)).

## Key Experimental Results

### Main Results
A 2-layer (1 hidden layer) homogeneous network with $m=2048$ was used for MNIST odd/even classification with logistic loss, trained to a loss of $10^{-8}$. Learning rate $\eta(t)=\eta_0 t^{-0.8}$ (satisfying both LR-MSD and LR-Adam); Adam stability constant $\varepsilon=10^{-20}$ was negligible relative to gradient scale.

| Optimizer | Implicit Bias Prediction | Norm for Empirical Max Margin |
| :--- | :--- | :--- |
| Normalized GD (±momentum) | $\ell_2$ | $\ell_2$ |
| Signum | $\ell_\infty$ | $\ell_\infty$ (slightly better than Adam) |
| Adam (without $\varepsilon$) | $\ell_\infty$ | $\ell_\infty$ |
| Muon | $\|\cdot\|_{\mathrm{msp}}$ | $\|\cdot\|_{\mathrm{msp}}$ |
| Muon-Adam | $\max\{(\eta_0^A/\eta_0^M)\|\cdot\|_{\mathrm{msp}},\|\cdot\|_\infty\}$ | Consistent with hybrid norm (Figure 2) |

### Ablation Study
| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| squared-ReLU activation | Satisfies (M1)+(M2); margin growth matches theory | Valid testbed for main claims |
| Standard ReLU activation | Empirically matches predicted norms | Suggests theory might relax to non-smooth cases (beyond (T3)) |
| Disable momentum (NGD vs MSD, etc.) | Negligible difference in margin behavior | Validates KKT equivalence of "normalized vs unnormalized, with or without momentum" in Theorem 3.2 |
| Directional convergence (Figure 1b) | $\langle\boldsymbol{\theta}_t,\boldsymbol{\theta}_{\text{last}}\rangle/\|\cdot\|\|\cdot\|>0.99$ after training | Confirms (T2) is satisfied across all algorithms |
| 4-layer network + CIFAR-10 (Appx D) | Trends match MNIST | Validates that conclusions extrapolate to deeper networks |

### Key Findings
- The mapping from algorithm to implicit norm is strictly confirmed experimentally: Muon always maximizes the $\|\cdot\|_{\mathrm{msp}}$ margin, while Adam and Signum always maximize the $\ell_\infty$ margin.
- Signum is slightly superior to Adam in $\ell_\infty$ margin—consistent with the theoretical explanation that "Adam is merely an approximation of sign gradient descent."
- NGD is sub-optimal on $\|\cdot\|_{\mathrm{msp}}$—because if the last layer is a single-row matrix, its spectral norm is exactly the $\ell_2$ norm, letting NGD partially benefit.

## Highlights & Insights
- **"Approximate Steepest Descent" is an elegant abstraction**: It absorbs complex technical details such as momentum overshoot and Adam’s dual-momentum ratios into the alignment condition $r(t)\ge 1$, providing a template for future analysis of "impure" optimizers like Lion, Shampoo, or Scion.
- **The Muon-Adam hybrid norm formula is pragmatic**: In practice, the learning rate ratio $\eta_0^A/\eta_0^M$ "re-weights" the two norms—meaning adjusting this ratio allows an explicit trade-off between matrix layer margins and bias layer margins, which may directly guide LLM training stability.
- **Adam without $\varepsilon$ is the "True Adam"**: The paper bridges the gap between practice and theory—the stability constant is negligible in measurement and must be removed in theory; otherwise, the conclusion would degenerate into GD due to $\varepsilon$.

## Limitations & Future Work
- The key assumption (T2) (directional convergence) is observed experimentally but not proven theoretically for Adam/Muon; for GD, this took years (Ji & Telgarsky 2020), and it likely remains a long-term open problem here.
- Covers only smooth homogeneous models; extending to ReLU requires (T3) (subgradient directional convergence), yet experiments on 2-layer ReLU MNIST suggest (T3) might not hold—implying the implicit bias of Adam/Muon on ReLU networks might be fundamentally different from spectral/sign margins.
- Binary classification setting; whether conclusions hold for multiclass (especially next-token prediction in LLMs) requires further proof; Fan et al. 2025 addressed linear multiclass but not homogeneous networks.
- Does not address AdamW: This model does not include explicit weight decay. The max-margin properties of AdamW involve the "constrained $\ell_\infty$ norm boundedness" results from Xie et al. 2024, which still differs formally from this work.

## Related Work & Insights
- **vs Lyu & Li (2019)**: They proved GD on homogeneous networks converges to $\ell_2$ max-margin KKT points; this paper uses approximate steepest descent to replicate this conclusion for Muon, Signum, and Adam, substituting the norm from $\ell_2$ to the respective algorithm's norm.
- **vs Tsilivis et al. (2025)**: They generalized Lyu-Li to any steepest descent; Theorem 3.2 here redid this for MSD and went a step further to Adam (which is not an MSD paradigm).
- **vs Zhang et al. (2024) / Fan et al. (2025)**: They proved Adam maximizes $\ell_\infty$ and Muon maximizes spectral norm on linear models; this paper upgrades this restriction to smooth homogeneous networks, representing a true "nonlinear extension."
- **vs Xie et al. (2024)** (AdamW): That work proved the trajectory limit of AdamW is KKT under $\|\boldsymbol{\theta}\|_\infty$ bounded constraints; this paper does not analyze weight decay, but the role of the $\ell_\infty$ norm is consistent, suggesting the universality of the $\ell_\infty$ bias in Adam-based algorithms.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to push the implicit bias of Muon and $\varepsilon$-free Adam to smooth homogeneous networks, proposing a reusable "Approximate Steepest Descent" framework.
- Experimental Thoroughness: ⭐⭐⭐ MNIST plus a CIFAR-10 appendix experiment; sufficient for a theory paper but not extensive.
- Writing Quality: ⭐⭐⭐⭐ Clearly assembles the machinery of flow limits, momentum integration, and KKT; Appendices B/C provide intuitive explanations for every Lemma.
- Value: ⭐⭐⭐⭐⭐ Provides a clear theoretical guide for optimizer selection in LLM/Muon practice ("which margin do you want?"); the most systematic work in the 2024-2026 wave of optimizer analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Implicit Bias of Per-sample Adam on Separable Data: Departure from the Full-batch Regime](../../ICLR2026/others/implicit_bias_of_per-sample_adam_on_separable_data_departure_from_the_full-batch.md)
- [\[ICML 2026\] How the Optimizer Shapes Learned Solutions in Equivariant Neural Networks](how_the_optimizer_shapes_learned_solutions_in_equivariant_neural_networks.md)
- [\[ICML 2026\] Singular Bayesian Neural Networks](singular_bayesian_neural_networks.md)
- [\[ICML 2026\] On the Epistemic Uncertainty of Overparametrized Neural Networks](on_the_epistemic_uncertainty_of_overparametrized_neural_networks.md)
- [\[ICML 2026\] Adaptive Preconditioners Trigger Loss Spikes in Adam](adaptive_preconditioners_trigger_loss_spikes_in_adam.md)

</div>

<!-- RELATED:END -->
