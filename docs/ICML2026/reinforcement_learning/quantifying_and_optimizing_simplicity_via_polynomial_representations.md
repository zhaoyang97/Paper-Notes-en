---
title: >-
  [Paper Note] Quantifying and Optimizing Simplicity via Polynomial Representations
description: >-
  [ICML2026][Reinforcement Learning][simplicity measure] The authors propose fitting Chebyshev polynomials along data interpolation paths as a low-dimensional function space proxy for neural networks. They define "Effectiv…
tags:
  - "ICML2026"
  - "Reinforcement Learning"
  - "simplicity measure"
  - "function space"
  - "Chebyshev polynomials"
  - "generalization proxy"
  - "differentiable regularization"
date: 2026-05-08
content_hash: 6a0ee00763bab194
---

# Quantifying and Optimizing Simplicity via Polynomial Representations

**Conference**: ICML2026  
**arXiv**: [2605.29823](https://arxiv.org/abs/2605.29823)  
**Code**: https://github.com/xinzaixinzai/Effective-Degree  
**Area**: interpretability  
**Keywords**: simplicity measure, function space, Chebyshev polynomials, generalization proxy, differentiable regularization  

## TL;DR
The authors propose fitting Chebyshev polynomials along data interpolation paths as a low-dimensional function space proxy for neural networks. They define "Effective Degree" (ED)—calculated by multiplying the absolute values of coefficients by the polynomial orders—as a scalar measure of "how simple a function is." ED predicts generalization gaps more accurately than known proxies like sharpness or parameter $L_2$ norms on CIFAR-10, ImageNet, and CLIP. Furthermore, the entire estimation pipeline is differentiable, allowing it to serve as a "simplicity regularizer" during training, which consistently yields performance gains across image, text, CLIP fine-tuning, and RL tasks.

## Background & Motivation

**Background**: Deep networks are over-parameterized yet generalize well, a phenomenon largely explained by "simplicity bias"—the tendency of optimization dynamics to select simple solutions. Several "simplicity/generalization proxies" have been proposed, such as max-margin, minimum-norm, information-theoretic description length, PAC-Bayes, the number of linear regions in ReLU networks, parameter $L_2$ norms, sharpness, and adaptive sharpness.

**Limitations of Prior Work**: A robust simplicity measure must simultaneously satisfy three criteria: (i) generalizability across tasks/architectures; (ii) computational feasibility at scale; and (iii) differentiability for optimization. Existing measures typically meet only one or two:

- Max-margin / Minimum-norm: Theoretically sound but only hold for linear/homogeneous models; difficult to extrapolate to deep non-linear models.
- Description length / PAC-Bayes: Universal but difficult to estimate stably and hard to use directly as a training objective.
- Number of linear regions: Aligned with expressivity but strongly architecture-dependent and uncomputable at scale.
- Parameter space measures (norms, Jacobian, sharpness): Sensitive to re-parameterization, unstable across architectures, and sharpness often anti-correlates with generalization under recipes like mixup.

**Key Challenge**: Simplicity should be an inherent property of the "learned function itself," yet most current proxies reside in parameter space or rely on architectural assumptions. Moreover, most definitions are non-differentiable, preventing their use as direct regularization terms.

**Goal**: (1) Provide a simplicity measure defined directly in the function space; (2) Ensure it is estimable on large-scale trained models; (3) Make it end-to-end differentiable for injection into training as a regularizer.

**Key Insight**: Directly expanding polynomials in a $d$-dimensional input space leads to a combinatorial explosion of basis functions $\binom{d+K}{K}$. The authors slice the network into a one-dimensional function $g_{\bm{x}_1,\bm{x}_2}(\alpha)=f(\alpha\bm{x}_1+(1-\alpha)\bm{x}_2)$ along the interpolation path between two data points and perform polynomial fitting. They prove that random paths preserve the "degree order" of multivariate polynomials almost everywhere, making the 1D proxy sufficient to reflect the non-linearity of the original network.

**Core Idea**: Restricting the network to 1D interpolation paths $\to$ performing Chebyshev polynomial fitting $\to$ taking the "coefficient $L_1$-weighted degree" as the simplicity scalar ED, and estimating the ED of the entire network via path averaging. The entire process is closed-form differentiable, allowing it to serve as both a post-hoc measure and a training-time regularizer.

## Method

### Overall Architecture

Given a predictive network $f:\mathbb{R}^d\to\mathbb{R}^{m'}$ and a data distribution $\mathcal{D}$, the pipeline follows five steps:

1. **Sample Interpolation Paths**: Draw a pair $\bm{x}_1,\bm{x}_2$ from $\mathcal{D}$ and define $\bm{x}(\alpha)=\alpha\bm{x}_1+(1-\alpha)\bm{x}_2$ for $\alpha\in[0,1]$.
2. **Node Sampling**: Select $r$ "random cosine nodes" $\alpha_i=\tfrac{1}{2}(1-\cos\theta_i)$ where $\theta_i\sim U[(i-1)\pi/r,i\pi/r]$, representing stratified randomization on the Chebyshev measure.
3. **Output Dimensionality Reduction**: Perform path-specific PCA on the outputs $\{f(\bm{x}(\alpha_i))\}$ along the path, retaining the top $m$ dimensions (typically $m=2,3$) to simplify multi-output fitting into low-dimensional scalar sequences.
4. **Chebyshev Least Squares Fitting**: For each PCA dimension, fit $P(\alpha)=\sum_{k=0}^K c_k T_k(2\alpha-1)$, solving the damped normal equations $(\bm{T}^\top\bm{T}+\epsilon\bm{I})\bm{c}_\epsilon=\bm{T}^\top\bm{y}$ for numerical stability.
5. **ED Calculation & Averaging**: Compute $\mathrm{ED}(P)=\sum_k|c_k|\cdot k$ and average across outputs. The final estimate is $\widehat{\mathrm{ED}}(f)=\mathbb{E}_{\bm{x}_1,\bm{x}_2\sim\mathcal{D}}[\mathrm{ED}(P_{\bm{x}_1,\bm{x}_2})]$, using the empirical mean across $n_p$ path pairs within a minibatch during training.

### Key Designs

1. **Interpolation Path + Degree Order Preservation Theorem**:
    - **Function**: Reduces high-dimensional polynomial simplicity measurement to 1D path fitting, avoiding the $\binom{d+K}{K}$ combinatorial explosion.
    - **Mechanism**: Defines complexity by limiting $f$ to a 1D path $g_{\bm{x}_1,\bm{x}_2}(\alpha)=f(\bm{x}(\alpha))$. Theoretically, one might fear information loss or degree reduction upon projection. The authors prove Theorem 3.1: for any two non-zero polynomials $P_1, P_2$ with degrees $D_1 > D_2$, the empirical mean of path degrees $\widehat{d}_n(P_i)$ obtained from i.i.d. sampled path pairs $(\bm{x}_1^{(j)},\bm{x}_2^{(j)})$ satisfies $\widehat{d}_n(P_1) > \widehat{d}_n(P_2)$ almost surely as $n \to \infty$. The proof utilizes the lemma that the Lebesgue measure of the zero set of a non-zero polynomial is zero: random directions almost never hit the zero set that would cause a degree drop.
    - **Design Motivation**: Expanding basis functions in $d$-dimensional space is inherently unscalable. Interpolation paths provide "1D slices near the data manifold," preserving distribution correlations while compressing the estimation into a 1D least-squares problem that is interpretable, computable, and scalable.

2. **Effective Degree (ED) + Differentiable Implementation with Closed-form Gradients**:
    - **Function**: Compresses the set of polynomial coefficients into a single scalar ED to reflect "how non-linear a function is," ensuring differentiability with respect to network parameters.
    - **Mechanism**: Arithmetic degree $\deg(P)$ is discrete and sensitive to small perturbations. Instead, $\mathrm{ED}(P)=\sum_k|c_k|\cdot k$ is used, which is essentially a weight-weighted degree using coefficient magnitudes. This corresponds to an $\ell_1$-style constraint on coefficients, aligning with Rademacher complexity capacity control where low-dimensional representations are tighter. A normalized version $\mathrm{ED}_{\text{norm}}=\sum|c_k|k/\sum|c_k|$ removes scale. Proposition 5.1 provides the analytical gradient with respect to sampled outputs $\bm{y}$: $\partial \mathrm{ED}/\partial\bm{y}=\bm{T}(\bm{T}^\top\bm{T})^{-1}(\mathrm{sign}(\bm{c})\odot\bm{d})$, where $\bm{d}=[0,\dots,K]^\top$ and $\bm{T}_{i,k}=T_k(2\alpha_i-1)$. In practice, the damped system $\bm{c}_\epsilon=\texttt{LinearSolve}(\bm{T}^\top\bm{T}+\epsilon\bm{I},\bm{T}^\top\bm{y})$ is solved using PyTorch's LU solver for stability.
    - **Design Motivation**: (i) Ensures Lipschitz continuity of the measure relative to coefficient perturbations, preventing domination by high-order noise; (ii) The $\ell_1$ weighting makes the gradient naturally scale-invariant to coefficient magnitude; (iii) Using a damped linear system solver allows stable backpropagation even in ill-conditioned cases with high-order polynomials and small batches.

3. **Label-anchored ED + Path-specific PCA Output Compression**:
    - **Function**: Prevents conflict between the ED regularizer and the cross-entropy objective while compressing high-dimensional outputs (e.g., 1000-class logits).
    - **Mechanism**: In classification, cross-entropy encourages predictions to move away from uniform distributions, while ED encourages "smooth prediction changes along the path." These signals conflict early in training. "Label-anchored ED" replaces the predictions at the path endpoints with the corresponding ground-truth one-hot labels (fixing $\theta_1=0, \theta_r=\pi$ and sampling only $r-2$ intermediate nodes). This forces the polynomial to pass through "true endpoints," punishing only the *additional* non-linearity within the path. High-dimensional outputs are further processed via path-specific PCA: each path calculates its own PCA projection to $m=2,3$ dimensions before fitting.
    - **Design Motivation**: (i) In classification, "correctly classifying endpoints" is a real task constraint that should not be flattened by simplicity penalties; anchoring allows ED to punish only redundant non-linearity. (ii) Path-specific PCA fits the current path distribution better than global PCA, allowing stable fitting even with statistical noise. Ablations show PCA is not the primary source of gains but significantly reduces overhead.

### Loss & Training

The total objective is $\mathcal{L}(\theta;\mathcal{B})=\mathcal{L}_{\text{task}}(\theta;\mathcal{B})+\lambda\,\widehat{\mathrm{ED}}_{\mathcal{B}}$. Key hyperparameters include the number of paths $n_p$, number of nodes $r$, polynomial degree $K$, damping $\epsilon$, and regularization strength $\lambda$. For text tasks, where linear interpolation on tokens is impossible, paths are constructed in the embedding space. For CLIP fine-tuning, interpolation occurs at the image input. In RL, only the actor network is penalized.

## Key Experimental Results

### Main Results

| Task / Model | Baseline | SAM | ASAM | Jacobian | Mixup | **+ ED (Ours)** |
|------|-------|-----|------|----------|-------|---------|
| CIFAR-10, ViT-Tiny (Top-1 %, 3 seeds) | 87.80 ± 1.17 | 87.85 ± 1.27 | 87.85 ± 1.24 | 87.81 ± 0.17 | 88.83 ± 1.48 | **90.82 ± 0.11** |
| ImageNet, ViT-S/16 (Original ViT recipe) | 71.37 ± 0.17 | — | — | — | — | **72.76 ± 0.16** |
| ImageNet, ViT-S/16 (Strong recipe) | 74.42 ± 0.13 | — | — | — | — | **75.01 ± 0.11** |
| CLIP ViT-B/32 ImageNet ID | 76.20 ± 0.02 | — | — | — | — | **77.14 ± 0.05** |
| CLIP ViT-B/32 OOD Average (5 shifts) | 44.04 ± 0.08 | — | — | — | — | **45.31 ± 0.08** |
| CLIP ViT-B/16 ImageNet ID | 81.35 ± 0.11 | — | — | — | — | **82.19 ± 0.03** |
| CLIP ViT-B/16 OOD Average (5 shifts) | 53.69 ± 0.04 | — | — | — | — | **55.29 ± 0.14** |

Consistently positive across modalities/tasks: ED outperforms SAM/ASAM/Jacobian/Mixup on CIFAR-10 (+3.0 points over baseline). In CLIP fine-tuning, both ID and all 5 OOD shifts (ImageNetV2/R/A/Sketch/ObjectNet) improve simultaneously. BERT-base results on RTE/MRPC/CoLA outperform mixup baselines. In Procgen reinforcement learning, adding ED to PPO improves generalization across Four environments (Dodgeball, Fruitbot, Jumper, StarPilot) on unseen levels.

### Ablation Study

| Design Option | Effect / Conclusion |
|-------|------|
| Replacing interpolation paths with random noise | Correlation between ED and generalization weakens; regularization effect drops, showing the "near data manifold" aspect is critical. |
| Chebyshev vs. Legendre bases | Similar performance; ED is insensitive to the choice of orthogonal basis. |
| Random cosine sampling vs. Fixed Chebyshev vs. Uniform | Uniform sampling is significantly unstable for large $K$; random cosine is most stable. |
| PCA compression to 2/3 dims vs. Full dimension | Works without PCA; PCA is primarily for efficiency rather than accuracy. |
| ED w/o Label Anchoring (LA) | Slightly lower than ED with LA (90.00 vs 90.82 on ViT-Tiny/CIFAR-10) but still superior to other regularizations. |
| Correlation of ED with Sharpness/$L_2$ | ED has the strongest Pearson correlation with generalization gaps; sharpness anti-correlates under mixup, and $L_2$ correlation is weak. |

### Key Findings

- **ED is the most stable generalization proxy currently**: Across ResNet18 and ViT-Tiny, and 27 hyperparameter sets, ED's Pearson correlation with generalization gaps is significantly stronger than sharpness, adaptive-sharpness, or $L_2$ norms. In grokking experiments, only ED exhibits a distinct peak followed by a decline near the point where validation loss drops; sharpness and $L_2$ remain monotonic or erratic.
- **Regularization gains are rooted in measurement accuracy**: That ED predicts well and can be optimized further confirms the link between "function space simplicity $\to$ generalization." Sharpness neither predicts accurately nor guarantees improvement when minimized.
- **Cross-modal universality**: Effective across image, text, vision-language, and RL, suggesting that "penalizing high-order non-linearity" is a relatively model-agnostic inductive bias.
- **Failure Modes**: ED may fail in scenarios where "simple features are easily exploited but detrimental to robust generalization" (e.g., certain shortcut learning cases), where ED might further reinforce the shortcut.

## Highlights & Insights

- **"Function Space Measure + Closed-form Differentiability" is the true key combination**: Previous function space measures were either uncomputable (PAC-Bayes) or non-optimizable (linear regions). By using 1D interpolation paths and Chebyshev bases, this work transforms estimation into a small-matrix linear solve, bridging measurement and regularization.
- **"Path Anchoring" for data correlation is an overlooked degree of freedom**: Unlike sharpness, which uses data-agnostic random perturbations in parameter space, ED's interpolation paths bind the measure to the data distribution. This maintains consistency across different recipes (e.g., mixup), which is crucial in settings like CLIP fine-tuning.
- **Label-anchored ED is a clever engineering compromise**: By explicitly acknowledging the task constraint that endpoints must be far from uniform, it restricts simplicity penalties to "internal path curvature." This distinction between "necessary vs. redundant non-linearity" could be transferred to other tasks like contrastive learning.

## Limitations & Future Work

- **Stated Limitations**: Theoretical work is still needed to identify exactly which class of function space simplicity the path-based polynomial proxy captures (beyond Theorem 3.1). Regularization can fail in shortcut learning scenarios.
- **Inferred Limitations**: (i) Computational overhead involves $n_p$ paths $\times$ $r$ nodes of forward passes plus matrix solves per minibatch; (ii) Requires continuous space for interpolation; discrete inputs (graphs, tokens) require embedding; (iii) Coupling between $K, r, m$ lacks theoretical guidance and currently relies on "robust defaults."
- **Future Directions**: (1) Adaptive path lengths expanding beyond $[0,1]$ to capture "out-of-boundary" non-linearity; (2) Coupling ED with mixup data augmentation; (3) Orthogonal combination with sharpness-aware training.

## Related Work & Insights

- **vs. SAM / ASAM (Foret 2021; Kwon 2021)**: SAM uses worst-case perturbations in parameter space (sensitive to re-parameterization); ED measures function space (stable across architectures/recipes). SAM anti-correlates with generalization under mixup, whereas ED remains positively correlated.
- **vs. Jacobian regularization (Hoffman 2019)**: Jacobian reg only controls local first-order sensitivity (slope); ED captures high-order non-linearity along the whole path, including curvature information Jacobian misses.
- **vs. Mixup (Zhang 2018)**: Mixup enforces "interpolated input $\to$ interpolated label," which is too rigid for language. ED estimates complexity without imposing synthetic labels, performing better on BERT/GLUE where mixup often fails.
- **vs. PAC-Bayes / Compression (Dziugaite-Roy 2017; Arora 2018)**: These provide theoretical bounds that are often vacuous and cannot be directly optimized; ED lacks formal bounds but excels empirically and is optimizable.

## Rating
- Novelty: ⭐⭐⭐⭐ The "path-based polynomial proxy + closed-form gradient ED" is a clean and practical framework that bridges theory and engineering.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Rare coverage across four major task types, ID/OOD shifts, grokking tracking, and failure mode analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear derivation chain (preservation theorem, differentiability, damped implementation, label anchoring); some hyperparameter details are in the appendix.
- Value: ⭐⭐⭐⭐⭐ Provides a stable generalization proxy that outperforms sharpness and serves as a near-maintenance-free universal regularizer for improving simplicity bias understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Can Large Language Models Generalize Procedures Across Representations?](can_large_language_models_generalize_procedures_across_representations.md)
- [\[ICML 2026\] Laplacian Representations for Decision-Time Planning](laplacian_representations_for_decision-time_planning.md)
- [\[NeurIPS 2025\] Quantifying Generalisation in Imitation Learning](../../NeurIPS2025/reinforcement_learning/quantifying_generalisation_in_imitation_learning.md)
- [\[ICLR 2026\] Dual Goal Representations](../../ICLR2026/reinforcement_learning/dual_goal_representations.md)
- [\[ICML 2026\] DR.Q: Debiased Model-based Representations for Sample-efficient Continuous Control](debiased_model-based_representations_for_sample-efficient_continuous_control.md)

</div>

<!-- RELATED:END -->
