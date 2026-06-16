---
title: >-
  [Paper Note] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks
description: >-
  [ICML 2026][Optimization & Theory][Adam] This paper proves that under the setting of smooth $L$-homogeneous models, exponential-tail loss, and learning rate decay, Muon (including Muon-Signum and Muon-Adam) as a "normalized steepest descent" with momentum converges to the KKT points of the corresponding norm max-margin problem. Adam (without the stability con
tags:
  - ICML 2026
  - Optimization & Theory
  - Adam
  - Muon
date: 2026-05-08
content_hash: 4de2818a02cc03f3
---
# The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks

**Conference**: ICML 2026  
**arXiv**: [2602.16340](https://arxiv.org/abs/2602.16340)  
**Code**: None  
**Area**: Optimization Theory / Implicit Bias / LLM Training  
**Keywords**: Implicit Bias, Adam, Muon, Homogeneous Networks, Max-margin, Spectral Norm  

## TL;DR
This paper proves that under the setting of smooth $L$-homogeneous models, exponential-tail loss, and learning rate decay, Muon (including Muon-Signum and Muon-Adam) as a "normalized steepest descent" with momentum converges to the KKT points of the corresponding norm max-margin problem. Adam (without the stability constant) converges to the KKT points of the $\ell_\infty$ max-margin problem. These results extend prior implicit bias conclusions, previously only valid for linear models, to all smooth homogeneous networks.

## Background & Motivation

**Background**: The generalization puzzle of over-parameterized neural networks is commonly attributed to the "implicit bias" of optimizers—even without explicit regularization, gradient-based algorithms tend to converge to certain max-margin solutions. Starting from the proof by Soudry et al. (2017) that GD maximizes the $\ell_2$-margin on linear models, to the extension by Lyu & Li (2019) to the KKT form for any $L$-homogeneous (deep ReLU) networks, the implicit bias landscape of gradient descent is largely complete.

**Limitations of Prior Work**: However, the industry does not use GD for training LLMs or ViTs; instead, adaptive optimizers with momentum like Adam and Muon are the standard. Their implicit bias results exist only for linear predictors (Zhang et al. 2024, Fan et al. 2025) and lack theory for true nonlinear networks. Furthermore, early analyses of Adam (Wang et al. 2021) kept the stability constant $\varepsilon$, causing $\sqrt{v_t}$ to be asymptotically dominated by $\varepsilon$ and essentially reducing Adam to GD—a significant disconnect from the practice where $\varepsilon$ is negligible.

**Key Challenge**: The update direction of momentum algorithms (MSD = momentum steepest descent) no longer strictly satisfies the algebraic condition of "steepest descent" ($\langle\dot{\boldsymbol{\theta}}/\|\dot{\boldsymbol{\theta}}\|, -\boldsymbol{g}/\|\boldsymbol{g}\|_\star\rangle=1$). Moreover, the overshoot caused by momentum is only of order $o(1)$, requiring new tools to characterize this "approximate steepest descent." Adam is even more complex, being the ratio of two momentums with different decay rates ($\hat{\boldsymbol{m}}_t/\sqrt{\hat{\boldsymbol{v}}_t}$), which does not fall into any standard MSD paradigm.

**Goal**: (i) Provide KKT convergence proofs for Muon/MSD on smooth homogeneous models; (ii) Provide similar results for Adam without $\varepsilon$; (iii) Derive the implicit norms for hybrid algorithms (Muon-Signum, Muon-Adam); (iv) Explain why momentum does not destroy max-margin bias using a unified framework ("Approximate Steepest Descent").

**Key Insight**: By analyzing in continuous time (flow limit), the relationship between momentum $\boldsymbol{m}_t=\int_0^t c_1 e^{-c_1(t-s)}\boldsymbol{g}_s\,ds$ and the instantaneous gradient is characterized as $\boldsymbol{m}_t[j]=\boldsymbol{g}_t[j](1\pm o(1))$ (for "instantaneously significant" coordinates $j$), meaning momentum serves as an asymptotic first-order approximation of the gradient.

**Core Idea**: Define "Approximate Steepest Descent"—as long as the infimum of the alignment between the update direction and the negative gradient is asymptotically $\ge 1$ and the parameter norm is controlled by a certain integral upper bound, it suffices to derive max-margin KKT points, bypassing the need for an exact characterization of momentum.

## Method

### Overall Architecture
The analysis focuses on continuous-time flows with the following unified forms:

- **(Normalized) MSD**: $d\boldsymbol{\theta}_t/dt\in\eta(t)\arg\min_{\|\boldsymbol{u}\|=1}\langle\boldsymbol{u},\boldsymbol{m}_t\rangle$, where momentum $\boldsymbol{m}_t$ evolves as $d\boldsymbol{m}_t/dt=c_1(\boldsymbol{g}_t-\boldsymbol{m}_t)$ ($c_1\sim -\log\beta_1$).
- **Muon = Normalized MSD under the matrix spectral norm**: $\|\cdot\|=\|\cdot\|_{\mathrm{sp}}$ for single layers, and $\|\cdot\|_{\mathrm{msp}}=\max_k\|W_k\|_{\mathrm{sp}}$ for multiple layers.
- **Adam (without $\varepsilon$)**: $d\boldsymbol{\theta}_t/dt=-\eta(t)\,\hat{\boldsymbol{m}}_t/\sqrt{\hat{\boldsymbol{v}}_t}$, where $\boldsymbol{v}_t$ evolves as $d\boldsymbol{v}_t/dt=c_2(\boldsymbol{g}_t^2-\boldsymbol{v}_t)$.

Model assumptions: (M1) $f\in C^1$, (M2) $L$-homogeneity $f(\boldsymbol{x};\alpha\boldsymbol{\theta})=\alpha^L f(\boldsymbol{x};\boldsymbol{\theta})$. Loss: $\mathcal{L}(\boldsymbol{\theta})=\sum_i e^{-\varphi(y_i f(\boldsymbol{x}_i;\boldsymbol{\theta}))}$, covering both exponential and logistic losses. Trajectory assumptions: (T1) Norm does not vanish, (T2) Direction converges and margin is positive, (A1) Initial effective gradient in Adam is non-zero. Learning rate decay (LR-MSD/LR-Adam): $\int\eta=\infty$ and $\eta(t)\le o(t^{1/L-1})$ (satisfied by $\eta(t)=1/t$ for $L>1$).

### Key Designs

**1. Approximate Steepest Descent Framework: Capturing max-margin bias for a broad class of optimizers via "trajectory alignment"**

The update direction of momentum algorithms no longer strictly satisfies the algebraic conditions of steepest descent, and Adam is a ratio of two different momentum decay rates, not belonging to the MSD paradigm at all. The authors solve this by abstracting "Approximate Steepest Descent": if for a trajectory $\boldsymbol{\theta}_t$ there exist $\nu(t), R_{\max}$ such that $N(t)=\int_0^t\nu\to\infty$, $\limsup\|\boldsymbol{\theta}_t\|/N(t)\le R_{\max}$, and the alignment infimum $\operatorname{ess\,liminf} r(t)\ge 1$ (where $r(t)=\sup_{\boldsymbol{g}_t}\langle\nu^{-1}\dot{\boldsymbol{\theta}}_t,-\boldsymbol{g}_t/\|\boldsymbol{g}_t\|_\star\rangle$), then under (T2) and $R_{\max}\le 1$, $\bar{\boldsymbol{\theta}}=\lim\boldsymbol{\theta}_t/\|\boldsymbol{\theta}_t\|$ must be a KKT point of the corresponding max-margin problem. The power of this framework lies in its "interface"—setting $\nu=\|\dot{\boldsymbol{\theta}}\|$ recovers classical steepest descent, and while Adam is not MSD, choosing $\nu=\eta(t)$ allows it to fit the framework. Technical details like momentum overshoot are absorbed into the $r(t)\ge 1$ alignment requirement.

**2. Asymptotic Consistency of Momentum and Gradient: Proving momentum is a first-order approximation under decaying learning rates**

To satisfy the alignment condition, the momentum direction must asymptotically equal the gradient direction. In continuous time, momentum is represented as $\boldsymbol{m}_t=\int_0^t c_1 e^{-c_1(t-s)}\boldsymbol{g}_s\,ds$. Under the learning rate decay $\|\dot{\boldsymbol{\theta}}_t\|\le o(t^{1/L-1})$, it is proven that for all "instantaneously significant" coordinates $J_\varepsilon(t)=\{j:|\boldsymbol{g}_t[j]|/\|\boldsymbol{g}_t\|_\star>\varepsilon\}$, the relation $\boldsymbol{m}_t[j]=\boldsymbol{g}_t[j](1\pm o(1))$ holds. The core is a scalar fact in Appendix B: as long as $d\log g/dt$ converges, the scalar momentum ratio $m(t)/g(t)$ converges to a well-defined limit. Learning rate decay ensures this condition is met along the trajectory, thus $\boldsymbol{m}_t/\|\boldsymbol{m}_t\|_\star-\boldsymbol{g}_t/\|\boldsymbol{g}_t\|_\star\to 0$. This step also prepares the key approximation for Adam: $\hat{\boldsymbol{m}}_t[j]/\sqrt{\hat{\boldsymbol{v}}_t[j]}=\mathrm{sign}(\boldsymbol{g}_t[j])(1\pm o(1))$, which bridges Adam to sign gradient descent and $\ell_\infty$ max-margin.

**3. Composite Algorithms = Max-norm MSD: Translating hybrid training recipes into single-norm (approximate) MSD**

In practice, Muon is often used for matrix parameters while Adam or sign GD is used for LayerNorm/bias. The joint implicit margin objective for such hybrid training was previously unknown. The authors prove that if normalized (momentum) steepest descent is run in parallel on parameter blocks $\boldsymbol{\theta}=(W_1,\dots,W_K,\boldsymbol{u})$ using the same $\eta(t)$, it is equivalent to a normalized MSD under the norm:

$$\|\boldsymbol{\theta}\|=\max\{\|(W_1,\dots,W_K)\|_{\mathrm{msp}},\ \|\boldsymbol{u}\|_\infty\}$$

(Corollary 3.4). Muon-Adam allows for different base learning rates, resulting in the norm $\max\{(\eta_0^A/\eta_0^M)\|(W_1,\dots,W_K)\|_{\mathrm{msp}},\|\boldsymbol{u}\|_\infty\}$ (Theorem 3.6). This "max-norm equivalence" provides an analytical expression for the hybrid implicit margin and implies a practical trade-off between matrix layer margins and bias layer margins by tuning $\eta_0^A/\eta_0^M$.

### Loss & Training
The analysis targets the optimizers and does not redefine the loss. Results hold for exponential and logistic losses, requiring $\varphi$ to be twice continuously differentiable, strictly monotonically convex, with bounded first and second derivatives (Appendix C.1). The model class includes deep linear networks and nonlinear networks with smooth homogeneous activations (e.g., $\mathrm{ReLU}^q\;(q>1)$, quadratic activation), but strictly speaking, it does not cover standard ReLU (which requires the additional (T3) subgradient direction convergence assumption).

## Key Experimental Results

### Main Results
A 2-layer (1 hidden layer) homogeneous network with $m=2048$ was used for MNIST odd/even classification with logistic loss, trained until loss reached $10^{-8}$. Learning rate $\eta(t)=\eta_0 t^{-0.8}$ (satisfying both LR-MSD and LR-Adam). Adam’s stability constant was set to $\varepsilon=10^{-20}$ to be negligible relative to the gradient scale.

| Optimizer | Predicted Implicit Bias | Norm with Empirical Max Margin |
|-----------|-------------------------|-------------------------------|
| Normalized GD (±momentum) | $\ell_2$ | $\ell_2$ |
| Signum | $\ell_\infty$ | $\ell_\infty$ (Slightly better than Adam) |
| Adam (without $\varepsilon$) | $\ell_\infty$ | $\ell_\infty$ |
| Muon | $\|\cdot\|_{\mathrm{msp}}$ | $\|\cdot\|_{\mathrm{msp}}$ |
| Muon-Adam | $\max\{(\eta_0^A/\eta_0^M)\|\cdot\|_{\mathrm{msp}},\|\cdot\|_\infty\}$ | Consistent with hybrid norm (Figure 2) |

### Ablation Study
| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| squared-ReLU activation | Satisfies (M1)+(M2); margin growth matches theory | Valid testbed for main claims |
| Standard ReLU activation | Empirically matches predicted norms | Suggests theory might relax to non-smooth cases beyond (T3) |
| No momentum (NGD vs MSD, etc.) | Margin behavior almost identical | Confirms KKT equivalence between normalized vs unnormalized, with/without momentum |
| Directional convergence (Fig 1b) | $\langle\boldsymbol{\theta}_t,\boldsymbol{\theta}_{\text{last}}\rangle/\|\cdot\|\|\cdot\|>0.99$ at end of training | Empirically validates (T2) for all algorithms |
| 4-layer network + CIFAR-10 | Trend consistent with MNIST | Validates extrapolation to deeper networks |

### Key Findings
- The mapping from algorithm to implicit norm is strictly validated experimentally: Muon always maximizes the $\|\cdot\|_{\mathrm{msp}}$ margin, while Adam and Signum always maximize the $\ell_\infty$ margin.
- Signum slightly outperforms Adam on $\ell_\infty$ margin, consistent with the theoretical explanation that Adam is an approximation of sign gradient descent.
- NGD is suboptimal on $\|\cdot\|_{\mathrm{msp}}$ because if the last layer is a single-row matrix, its spectral norm is exactly the $\ell_2$ norm, which inherently benefits NGD.

## Highlights & Insights
- **"Approximate Steepest Descent" is a versatile abstraction**: Technical complexities like momentum overshoot and the ratio of momentums in Adam are handled by the $r(t)\ge 1$ alignment condition, providing a template for analyzing "non-pure" optimizers like Lion, Shampoo, or Scion.
- **Utility of Muon-Adam's hybrid norm formula**: In practice, the ratio $\eta_0^A/\eta_0^M$ "re-weights" the two norms. This means adjusting this ratio allows for an explicit trade-off between matrix layer margins and bias layer margins, which may directly inform LLM training stability.
- **Adam without $\varepsilon$ is the "True Adam"**: The paper bridges the gap between practice and theory. Since $\varepsilon$ is negligible in practice, it must be removed in theory, otherwise the conclusion degenerates to GD due to $\varepsilon$.

## Limitations & Future Work
- The directional convergence assumption (T2) is only observed experimentally and not proven theoretically for Adam/Muon; this likely remains a long-term open problem, similar to how it took years to prove for GD (Ji & Telgarsky, 2020).
- Limited to smooth homogeneous models. Extension to ReLU requires (T3) (subgradient direction convergence). Experiments on 2-layer ReLU MNIST suggest (T3) might not hold, implying Adam/Muon's implicit bias on ReLU networks might fundamentally differ from spectral/sign margins.
- Binary classification setting. Whether conclusions hold for multi-classification (especially next-token prediction in LLMs) needs separate proof; Fan et al. (2025) addressed linear multi-classification but not homogeneous networks.
- AdamW was not addressed. This paper lacks explicit weight decay. AdamW's max-margin nature involves constrained $\ell_\infty$ norms (Xie et al. 2024), which differs from the form here.

## Related Work & Insights
- **vs Lyu & Li (2019)**: They proved GD in homogeneous networks converges to $\ell_2$ max-margin KKT. This paper uses approximate steepest descent to replicate that result for Muon, Signum, and Adam, merely changing the norm.
- **vs Tsilivis et al. (2025)**: They extended Lyu-Li to any steepest descent. This paper re-proves Theorem 3.2 for MSD and goes further by including Adam (which is not an MSD paradigm).
- **vs Zhang et al. (2024) / Fan et al. (2025)**: They proved Adam maximizes $\ell_\infty$ and Muon maximizes spectral norm in linear models. This paper upgrades the restriction to smooth homogeneous networks, providing a true "nonlinear extension."
- **vs Xie et al. (2024)** (AdamW): They proved AdamW's trajectory limit is a KKT point under bounded $\|\boldsymbol{\theta}\|_\infty$ constraints. While this paper does not analyze weight decay, the role of the $\ell_\infty$ norm is consistent, suggesting the universality of the $\ell_\infty$ bias in Adam-based algorithms.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to push Muon and $\varepsilon$-free Adam's implicit bias to smooth homogeneous networks with a reusable "Approximate Steepest Descent" framework.
- Experimental Thoroughness: ⭐⭐⭐ MNIST plus a CIFAR-10 appendix; sufficient for a theory paper but not exhaustive.
- Writing Quality: ⭐⭐⭐⭐ Clearly assembles machines like flow limits, momentum integrals, and KKT points; Appendix B/C provides intuitive explanations for lemmas.
- Value: ⭐⭐⭐⭐⭐ Provides a clear theoretical guide ("which margin do you want?") for optimizer selection in LLM/Muon practice, representing the most systematic analysis in the 2024-2026 wave of optimizer research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Rich and the Simple: On the Implicit Bias of Adam and SGD](../../NeurIPS2025/optimization/the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)
- [\[ICLR 2026\] Implicit Bias of Per-sample Adam on Separable Data: Departure from the Full-batch Regime](../../ICLR2026/optimization/implicit_bias_of_per-sample_adam_on_separable_data_departure_from_the_full-batch.md)
- [\[NeurIPS 2025\] Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data](../../NeurIPS2025/optimization/implicit_bias_of_spectral_descent_and_muon_on_multiclass_separable_data.md)
- [\[ICML 2026\] LiMuon: Light and Fast Muon Optimizer for Large Models](limuon_light_and_fast_muon_optimizer_for_large_models.md)
- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](../../NeurIPS2025/optimization/understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)

</div>

<!-- RELATED:END -->
