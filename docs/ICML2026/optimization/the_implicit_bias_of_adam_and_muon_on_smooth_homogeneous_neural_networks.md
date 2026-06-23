---
title: >-
  [Paper Note] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks
description: >-
  [ICML 2026][Optimization & Theory][Adam] This paper proves that under the setting of smooth $L$-homogeneous models + exponential-tail loss + learning rate decay, Muon (including Muon-Signum, Muon-Adam) as a momentum-based "normalized steepest descent" converges to the KKT points of the corresponding norm max-margin problem; Adam (without the stability constan
tags:
  - ICML 2026
  - Optimization & Theory
  - Adam
  - Muon
date: 2026-05-08
content_hash: 657d934152c888d9
---
# The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks

**Conference**: ICML 2026  
**arXiv**: [2602.16340](https://arxiv.org/abs/2602.16340)  
**Code**: None  
**Area**: Optimization Theory / Implicit Bias / LLM Training  
**Keywords**: Implicit Bias, Adam, Muon, Homogeneous Networks, Max-margin, Spectral Norm  

## TL;DR
This paper proves that under the setting of smooth $L$-homogeneous models + exponential-tail loss + learning rate decay, Muon (including Muon-Signum, Muon-Adam) as a momentum-based "normalized steepest descent" converges to the KKT points of the corresponding norm max-margin problem; Adam (without the stability constant $\varepsilon$) converges to the KKT points of the $\ell_\infty$ max-margin problem. This elevates implicit bias conclusions, previously only valid for linear models, to all smooth homogeneous networks.

## Background & Motivation

**Background**: The mystery of generalization in overparameterized neural networks is commonly attributed to the "implicit bias" of optimizers—even without explicit regularization, gradient-based algorithms tend to converge to some maximum margin solution. Starting from Soudry et al. (2017) proving that GD maximizes $\ell_2$-margin on linear models, to Lyu & Li (2019) extending this to KKT forms of arbitrary $L$-homogeneous (deep ReLU) networks, the implicit bias landscape of gradient descent is largely complete.

**Limitations of Prior Work**: However, the industry uses adaptive optimizers with momentum like Adam and Muon to train LLMs/ViTs, not GD. Their implicit bias results exist only for linear predictors (Zhang et al. 2024, Fan et al. 2025) and lack theory for true non-linear networks. Furthermore, early analyses of Adam (Wang et al. 2021) retained the stability constant $\varepsilon$, causing $\sqrt{v_t}$ to be asymptotically dominated by $\varepsilon$, essentially degenerating Adam into GD—a serious disconnect from the practice where $\varepsilon$ is negligible.

**Key Challenge**: The update direction of momentum algorithms (MSD = momentum steepest descent) no longer satisfies the algebraic condition of "steepest descent" ($\langle\dot{\boldsymbol{\theta}}/\|\dot{\boldsymbol{\theta}}\|, -\boldsymbol{g}/\|\boldsymbol{g}\|_\star\rangle=1$) strictly. While the overshoot from momentum is only $o(1)$, new tools are needed to characterize this "approximate steepest descent." Adam is even more complex: it is the ratio of two momenta with different decay rates ($\hat{\boldsymbol{m}}_t/\sqrt{\hat{\boldsymbol{v}}_t}$), which does not fit any standard MSD paradigm.

**Goal**: (i) Provide KKT convergence proofs for Muon/MSD on smooth homogeneous models; (ii) Provide similar results for Adam without $\varepsilon$; (iii) Formalize the implicit norms of hybrid algorithms (Muon-Signum, Muon-Adam); (iv) Use a unified framework ("Approximate Steepest Descent") to explain why momentum does not destroy max-margin bias.

**Key Insight**: Analyze in continuous time (flow limit), characterizing the asymptotic relationship between momentum $\boldsymbol{m}_t=\int_0^t c_1 e^{-c_1(t-s)}\boldsymbol{g}_s\,ds$ and the instantaneous gradient as $\boldsymbol{m}_t[j]=\boldsymbol{g}_t[j](1\pm o(1))$ for "instantaneously significant" coordinates $j$. Thus, momentum becomes an asymptotic first-order approximation of the gradient.

**Core Idea**: Define "Approximate Steepest Descent"—as long as the infimum of the alignment between the update direction and the negative gradient is asymptotically $\ge 1$, and the parameter norm is controlled by a certain integral upper bound, it suffices to derive the max-margin KKT points, bypassing the need for an exact characterization of momentum.

## Method

### Overall Architecture
The analysis focuses on continuous time flows with the following unified forms:

- **(Normalized) MSD**: $d\boldsymbol{\theta}_t/dt\in\eta(t)\arg\min_{\|\boldsymbol{u}\|=1}\langle\boldsymbol{u},\boldsymbol{m}_t\rangle$, where momentum $\boldsymbol{m}_t$ evolves as $d\boldsymbol{m}_t/dt=c_1(\boldsymbol{g}_t-\boldsymbol{m}_t)$ ($c_1\sim -\log\beta_1$).
- **Muon = Normalized MSD under matrix spectral norm**: $\|\cdot\|=\|\cdot\|_{\mathrm{sp}}$ for single layers, $\|\cdot\|_{\mathrm{msp}}=\max_k\|W_k\|_{\mathrm{sp}}$ for multiple layers.
- **Adam** (without $\varepsilon$): $d\boldsymbol{\theta}_t/dt=-\eta(t)\,\hat{\boldsymbol{m}}_t/\sqrt{\hat{\boldsymbol{v}}_t}$, where $\boldsymbol{v}_t$ evolves as $d\boldsymbol{v}_t/dt=c_2(\boldsymbol{g}_t^2-\boldsymbol{v}_t)$.

Model assumptions: (M1) $f\in C^1$, (M2) $L$-homogeneity $f(\boldsymbol{x};\alpha\boldsymbol{\theta})=\alpha^L f(\boldsymbol{x};\boldsymbol{\theta})$. Loss: $\mathcal{L}(\boldsymbol{\theta})=\sum_i e^{-\varphi(y_i f(\boldsymbol{x}_i;\boldsymbol{\theta}))}$, covering exponential and logistic losses. Trajectory assumptions: (T1) norm does not vanish, (T2) direction converges with positive margin, (A1) initial effective gradient for Adam is non-zero. Learning rate decay (LR-MSD/LR-Adam): $\int\eta=\infty$ and $\eta(t)\le o(t^{1/L-1})$ (satisfied by $\eta(t)=1/t$ for $L>1$).

### Key Designs

**1. Approximate Steepest Descent Framework: Capturing max-margin bias via directional alignment**

The update direction of momentum algorithms no longer strictly satisfies the steepest descent algebraic conditions, and Adam, being a ratio of two momenta with different decay rates, does not fit the MSD paradigm at all. The authors resolve this through "Approximate Steepest Descent": if for a trajectory $\boldsymbol{\theta}_t$ there exist $\nu(t), R_{\max}$ such that $N(t)=\int_0^t\nu\to\infty$, $\limsup\|\boldsymbol{\theta}_t\|/N(t)\le R_{\max}$, and the alignment infimum $\operatorname{ess\,liminf} r(t)\ge 1$ (where $r(t)=\sup_{\boldsymbol{g}_t}\langle\nu^{-1}\dot{\boldsymbol{\theta}}_t,-\boldsymbol{g}_t/\|\boldsymbol{g}_t\|_\star\rangle$), then under (T2) and $R_{\max}\le 1$, $\bar{\boldsymbol{\theta}}=\lim\boldsymbol{\theta}_t/\|\boldsymbol{\theta}_t\|$ must be a KKT point of the corresponding max-margin problem. This framework handles Muon and Adam together; technical details like momentum overshoot or ratios are subsumed into the $r(t)\ge 1$ alignment requirement.

**2. Asymptotic Consistency of Momentum and Gradient: Proving momentum is a first-order gradient approximation**

To satisfy the alignment condition, the momentum direction must asymptotically match the gradient direction. In continuous time, momentum is written as $\boldsymbol{m}_t=\int_0^t c_1 e^{-c_1(t-s)}\boldsymbol{g}_s\,ds$. Ours proves that under LR decay $\|\dot{\boldsymbol{\theta}}_t\|\le o(t^{1/L-1})$, for all "instantaneously significant" coordinates $J_\varepsilon(t)=\{j:|\boldsymbol{g}_t[j]|/\|\boldsymbol{g}_t\|_\star>\varepsilon\}$, the relation $\boldsymbol{m}_t[j]=\boldsymbol{g}_t[j](1\pm o(1))$ holds. The core is a scalar property: as long as $d\log g/dt$ converges, the scalar momentum $m(t)/g(t)$ converges to a well-defined limit. LR decay ensures this condition is met along the trajectory, leading to $\boldsymbol{m}_t/\|\boldsymbol{m}_t\|_\star-\boldsymbol{g}_t/\|\boldsymbol{g}_t\|_\star\to 0$. This step also prepares the key approximation for Adam: $\hat{\boldsymbol{m}}_t[j]/\sqrt{\hat{\boldsymbol{v}}_t[j]}=\mathrm{sign}(\boldsymbol{g}_t[j])(1\pm o(1))$, which bridges Adam to SignGD and $\ell_\infty$ max-margin.

**3. Compound Algorithms = Max-norm MSD: Mapping hybrid training to a single norm**

In practice, Muon often handles matrix parameters while Adam or SignGD is used for LayerNorm/biases. No previous work has derived the implicit margin objective for such hybrid setups. Ours proves that if normalized (momentum) steepest descent is run in parallel on parameter blocks $\boldsymbol{\theta}=(W_1,\dots,W_K,\boldsymbol{u})$ with a shared $\eta(t)$, it is equivalent to a normalized MSD under the norm:

$$\|\boldsymbol{\theta}\|=\max\{\|(W_1,\dots,W_K)\|_{\mathrm{msp}},\ \|\boldsymbol{u}\|_\infty\}$$

(Corollary 3.4). Muon-Adam further allows different base learning rates, changing the norm to $\max\{(\eta_0^A/\eta_0^M)\|(W_1,\dots,W_K)\|_{\mathrm{msp}},\|\boldsymbol{u}\|_\infty\}$ (Theorem 3.6). This "max-norm equivalence" provides the first analytical formula for the implicit margin of hybrid training and suggests that tuning the ratio $\eta_0^A/\eta_0^M$ allows an explicit trade-off between matrix layer margin and bias layer margin.

### Loss & Training
The optimizer is the object of analysis; no new loss is proposed. Results hold for exponential/logistic losses where $\varphi$ is $C^2$, strictly monotonic convex, and has bounded derivatives (Appendix C.1). The model class includes deep linear networks and non-linear networks with smooth homogeneous activations (e.g., $\mathrm{ReLU}^q\;(q>1)$, squared activations), but strictly speaking does not cover standard ReLU (valid only under the additional subgradient direction convergence assumption (T3)).

## Key Experimental Results

### Main Results
2-layer ($m=2048$) homogeneous network on MNIST parity classification, logistic loss, trained to loss $10^{-8}$; LR $\eta(t)=\eta_0 t^{-0.8}$ (satisfying LR-MSD and LR-Adam); Adam stability constant $\varepsilon=10^{-20}$.

| Optimizer | Implicit Bias Prediction | Empirically Maximized Norm-Margin |
|-----------|--------------------------|-----------------------------------|
| Normalized GD (±momentum) | $\ell_2$ | $\ell_2$ |
| Signum | $\ell_\infty$ | $\ell_\infty$ (slightly better than Adam) |
| Adam (without $\varepsilon$) | $\ell_\infty$ | $\ell_\infty$ |
| Muon | $\|\cdot\|_{\mathrm{msp}}$ | $\|\cdot\|_{\mathrm{msp}}$ |
| Muon-Adam | $\max\{(\eta_0^A/\eta_0^M)\|\cdot\|_{\mathrm{msp}},\|\cdot\|_\infty\}$ | Consistent with hybrid norm (Figure 2) |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| squared-ReLU activation | Satisfies (M1)+(M2), margin growth matches theory | A valid testbed for main claims |
| Standard ReLU activation | Empirically matches predicted norms | Suggests theory may relax to non-smooth cases |
| No momentum (NGD vs MSD) | Negligible difference in margin behavior | Validates KKT equivalence between momentum and non-momentum |
| Directional convergence (Fig 1b) | $\langle\boldsymbol{\theta}_t,\boldsymbol{\theta}_{\text{last}}\rangle/\|\cdot\|\|\cdot\|>0.99$ late in training | Empirically validates (T2) for all algorithms |
| 4-layer + CIFAR-10 (Appx D) | Trends consistent with MNIST | Validates extrapolation to deeper networks |

### Key Findings
- The mapping from algorithm to implicit norm is strictly confirmed: Muon maximizes $\|\cdot\|_{\mathrm{msp}}$ margin, while Adam/Signum maximize $\ell_\infty$ margin.
- Signum is slightly superior to Adam in $\ell_\infty$ margin—consistent with the theoretical interpretation of Adam being an approximation of SignGD.
- NGD is sub-optimal on $\|\cdot\|_{\mathrm{msp}}$ because if the last layer is a single-row matrix, its spectral norm is the $\ell_2$ norm, partially benefiting NGD.

## Highlights & Insights
- **"Approximate Steepest Descent" is a versatile abstraction**: It subsumes technical complexities like momentum overshoot and dual-momentum ratios into a simple alignment condition $r(t) \ge 1$, providing a template for analyzing "impure" optimizers like Lion, Shampoo, or Scion.
- **The Muon-Adam hybrid norm formula is practical**: The learning rate ratio $\eta_0^A/\eta_0^M$ re-weights the norms, meaning adjusting this ratio explicitly trades off margins between matrix and bias layers, potentially guiding LLM training stability.
- **Adam without $\varepsilon$ is the "True Adam"**: The paper bridges the gap between theory and practice—since $\varepsilon$ is negligible in empirical results, it must be removed theoretically to avoid degenerating the conclusion into GD.

## Limitations & Future Work
- The key assumption (T2) (directional convergence) is observed empirically but not proven theoretically for Adam/Muon; for GD, this took years to resolve.
- Only covers smooth homogeneous models; extending to ReLU requires (T3) (subgradient directional convergence), which experiments suggest might not hold for 2-layer ReLU MNIST, implying Adam/Muon bias on ReLU might differ from spectral/sign margins.
- Binary classification setting; extension to multiclass (especially next-token prediction in LLMs) needs further proof.
- Does not cover AdamW: Ours lacks explicit weight decay. AdamW's max-margin nature involves constrained $\ell_\infty$ norms (Xie et al. 2024), which differs formally from the current results.

## Related Work & Insights
- **vs Lyu & Li (2019)**: They proved GD in homogeneous networks converges to $\ell_2$ max-margin KKT. Ours reproduces this for Muon, Signum, and Adam by replacing $\ell_2$ with the respective norms.
- **vs Tsilivis et al. (2025)**: They extended Lyu-Li to arbitrary steepest descent. Ours re-derives this for MSD and goes further to include Adam, which is not an MSD paradigm.
- **vs Zhang et al. (2024) / Fan et al. (2025)**: They proved Adam maximizes $\ell_\infty$ and Muon maximizes spectral norm in linear models. Ours generalizes this to smooth homogeneous networks, providing a true non-linear extension.
- **vs Xie et al. (2024)** (AdamW): They proved AdamW's trajectory limit is KKT under $\ell_\infty$ constraints. While Ours does not analyze weight decay, the consistent role of the $\ell_\infty$ norm suggests a universal bias for Adam-family algorithms.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to push implicit bias of Muon and epsilon-free Adam to smooth homogeneous networks with a reusable framework.
- Experimental Thoroughness: ⭐⭐⭐ Sufficient for theory; includes MNIST and a CIFAR-10 appendix.
- Writing Quality: ⭐⭐⭐⭐ Clearly assembles flow limits, momentum integrals, and KKT machinery; provides intuitive explanations in the appendix.
- Value: ⭐⭐⭐⭐⭐ Provides a clear theoretical guide for optimizer selection in LLM/Muon training based on desired margin types.

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
