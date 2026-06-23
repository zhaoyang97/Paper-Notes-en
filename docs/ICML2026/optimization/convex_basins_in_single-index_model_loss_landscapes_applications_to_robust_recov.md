---
title: >-
  [Paper Note] Convex Basins in Single-Index Model Loss Landscapes: Applications to Robust Recovery under Strong Adversarial Corruption
description: >-
  [ICML 2026][Optimization & Theory][Paper Note] Under heavy-tailed noise and constant-proportion strong adversarial corruption, the authors prove that a dimension-independent, constant-radius convex basin exists in the squared loss of Gaussian Single-Index Models for a wide class of non-monotonic link functions (GeLU, Swish, Tanh, Probit, Logistic, Phase Retrieval..
tags:
  - ICML 2026
  - Optimization & Theory
date: 2026-05-08
content_hash: 562987375fb7b2dd
---
# Convex Basins in Single-Index Model Loss Landscapes: Applications to Robust Recovery under Strong Adversarial Corruption

**Conference**: ICML2026  
**arXiv**: [2605.29497](https://arxiv.org/abs/2605.29497)  
**Code**: None  
**Area**: Optimization / Robust Statistics / Single-Index Models  
**Keywords**: Single-Index Model, Robust Regression, Heavy-tailed Noise, Strong Adversarial Corruption, Convex Basin, Spectral Initialization

## TL;DR
Under heavy-tailed noise and constant-proportion strong adversarial corruption, the authors prove that a dimension-independent, constant-radius convex basin exists in the squared loss of Gaussian Single-Index Models for a wide class of non-monotonic link functions (GeLU, Swish, Tanh, Probit, Logistic, Phase Retrieval...). Based on this, they design a robust recovery algorithm with $\tilde{O}(nd)$ time and $\tilde{O}(d)$ sample complexity, achieving a final estimation error of $O(\sigma\sqrt{\epsilon})$.

## Background & Motivation
**Background**: Single-Index Models (SIM) $Y=f(X^\top\beta^\star)+\zeta$ unify linear regression, Logistic regression, phase retrieval, and GLMs into a semi-parametric family. Modern gated neural networks also utilize GeLU/Swish as non-monotonic scalar primitives. Existing robust recovery theories only cover three narrow settings: linear ($f(x)=x$), strictly monotonic links (GLMs like Logistic), and phase retrieval ($f(z)=z^2$), as investigated by Pensia et al. (JASA 2024), Awasthi et al. (NeurIPS 2022), and Buna and Rebeschini (AISTATS 2025).

**Limitations of Prior Work**: Extending these proofs to general "non-monotonic + asymmetric" link functions (e.g., GeLU, Swish) fails immediately. First-order proof techniques (e.g., Arous et al.) rely on martingale-drift decomposition, requiring zero-mean random bias. Strong adversarial corruption can arbitrarily pollute an $\epsilon$ fraction of samples, destroying this property. Furthermore, the symmetric structure of phase retrieval (quadratic link) prevents migration to asymmetric cases.

**Key Challenge**: To perform robust recovery in high dimensions, two structural conditions must hold: (i) the squared loss must possess a **dimension-independent** constant-radius convex basin around $\beta^\star$ to enable second-order convergence proofs; (ii) this basin must be **efficiently reachable** from random initialization. These conditions were previously only simultaneous in phase retrieval, with no known results for broader non-monotonic links.

**Goal**: Identify a set of mild sufficient conditions for the link function $f$ such that (i) and (ii) hold simultaneously, providing a near-linear time, sample-optimal robust recovery algorithm.

**Key Insight**: The existence of a convex basin is translated into a pure 1D integral condition regarding Gaussian expectations of $f$ (Assumption 2.1). Reachability is translated into a second-moment criterion $\mathrm{ESC}(\beta,f):=\mathbb{E}[(f'(X^\top\beta))^2 + f(X^\top\beta)f''(X^\top\beta)]>0$ (Assumption 2.2). This reduces the burden of high-dimensional proofs to the 1D properties of the link function itself.

**Core Idea**: "Local Lipschitz complexity + ESC" characterize "basin existence + reachability." The algorithm employs spectral initialization to enter the basin followed by Robust Gradient Descent (RGD) to refine the estimate, generalizing robust recovery from phase retrieval to the entire class of SIMs with an index $\le 2$.

## Method

### Overall Architecture
Input: A dataset $\{(x_i,y_i)\}_{i=1}^N$ corrupted by an $\epsilon$ fraction of strong adversarial noise, $x_i\sim\mathcal{N}(0,\mathbf I_d)$, and an unknown index vector $\beta^\star$ with $\|\beta^\star\|_2=1$. Output: A unit vector $\hat\beta$ satisfying $\|\hat\beta-\beta^\star\|_2=O(\sigma\sqrt\epsilon)$. Algorithm 1 splits samples into $P+1$ equal batches: it uses LRSI for spectral initialization ($\beta_0\leftarrow\text{LRSI}(N_1,\epsilon)$) to reach the basin, then refined with LRGD ($\beta_P\leftarrow\text{LRGD}(N_{2..P+1},\beta_0,\epsilon,\alpha,\gamma)$) to reduce error from $O(\epsilon^{1/4})$ to $O(\sqrt\epsilon)$, finally outputting the normalized vector.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Corrupted Dataset<br/>Heavy-tailed noise + ε-strong adversarial corruption"] --> B["Random Batching<br/>(Sample Splitting)"]
    C["Convex Basin Existence<br/>1D Gaussian criteria → Dimension-independent,<br/>constant radius basin near β*"]
    C -. Guarantees convergence & strong convexity .-> D
    B --> D["LRSI Spectral Init<br/>YX Second Moment + Robust 1-ePCA<br/>→ β0 enters basin"]
    D -->|Spectral error capped at O(ε^1/4)| E["LRGD Robust Gradient Descent<br/>Robust Mean Estimation + Independent Batches<br/>P=O(1) iterations"]
    E --> F["Normalized Output βP/‖βP‖<br/>Final Error O(σ√ε)"]
```

### Key Designs

**1. Convex Basin Existence (Assumption 2.1 + Theorem 3.1): Collapsing High-Dimensional Basins to 1D Gaussian Integrals**

Robust recovery requires a dimension-independent constant-radius convex basin for the squared loss $\mathcal L(\beta)=\frac12\mathbb E[(f(X^\top\beta)-Y)^2]$ around $\beta^\star$. The authors collapse the high-dimensional Hessian at $\beta^\star$ using Gaussian symmetry:

$$H(\beta^\star)=\mathbb E[(f'(Z))^2]\,\mathbf I_d+\big(\mathbb E[Z^2(f'(Z))^2]-\mathbb E[(f'(Z))^2]\big)\beta^\star\beta^{\star\top},$$

leading to $\lambda_{\min}(H(\beta^\star))=\mu:=\min\{\mathbb E[f'(Z)^2],\mathbb E[Z^2 f'(Z)^2]\}$. The operator norm of $H(\beta)-H(\beta^\star)$ is bounded by $C_{\text{lip}}(R)\cdot\|\beta-\beta^\star\|$ via the mean value theorem, where $C_{\text{lip}}(R)$ involves only 1D Gaussian integrals of $f$ and its derivatives up to order 3. Thus, GeLU, Swish, Tanh, Probit, Logistic, and phase retrieval all fit within this framework.

**2. ESC Condition + LRSI Spectral Initialization (Assumption 2.2 + Theorem 4.2): Entering the Basin via Robust Spectral Methods**

To reach the basin efficiently, the authors define $\tilde Y:=YX$. Utilizing the Stein second-order identity, $\beta^\star$ is proven to be the top eigenvector of $\mathbb E[\tilde Y\tilde Y^\top]$ if and only if $\mathrm{ESC}(\beta;f):=\mathbb E[(f'(X^\top\beta))^2+f(X^\top\beta)f''(X^\top\beta)]>0$. This "high-order monotonicity" allows the signal direction to be identified by second-moment methods even if $f$ itself is not monotonic. By proving $\tilde Y$ is $(4,C_4)$ hypercontractive, the algorithm utilizes a near-linear time robust 1-ePCA subroutine to obtain $\beta_0$.

**3. LRGD Robust Gradient Descent (Theorem 4.1): Refining $\epsilon^{1/4}$ Error to Optimal $\sigma\sqrt\epsilon$**

Spectral initialization is limited by the $\epsilon^{1/4}$ statistical lower bound. Within the convex basin, $\mathcal L$ is $\gamma$-strongly convex and $\alpha$-smooth. The gradient is expressed as $\nabla\mathcal L(\beta)=\mathbb E[(f(X^\top\beta)-Y)f'(X^\top\beta)X]$. By applying robust mean estimation to this expectation, the algorithm achieves an error of $O(\sigma\sqrt\epsilon)$, which is information-theoretically optimal.

### Loss & Training
The target is the squared loss $\mathcal L(\beta)=\frac12\mathbb E[(f(X^\top\beta)-Y)^2]$. Total sample complexity is $n=\tilde O(m+P\tilde m)$, where $m$ is used for spectral initialization and $\tilde m$ for each round of robust gradient estimation. Total runtime is $\tilde O(nd)$.

## Key Experimental Results

This is a theoretical paper. The following tables summarize theoretical metrics across different link functions and settings.

### Main Results Comparison Table

| Link / Task | Noise + Corruption | Error Rate | Time | Samples | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Linear $f(x)=x$ | Heavy-tailed + Robust | $O(\sigma\sqrt\epsilon)$ | $\tilde O(nd)$ | $\tilde O(d)$ | Cherapanamjeri et al. 2020 |
| Logistic (GLM) | Gauss + Robust | $O(\sigma\epsilon\log\frac1\epsilon)$ | N/A | $\tilde O(d)$ | Awasthi et al. 2022 |
| Phase Retrieval $f(z)=z^2$ | Heavy-tailed + Robust | $O(\sigma\sqrt\epsilon)$ | Poly | $\tilde O(d)$ | Das & Batra 2026 |
| **GeLU / Swish / Tanh / Logistic / etc.** | Heavy-tailed + Robust | $O(\sigma\sqrt\epsilon)$ | $\tilde O(nd)$ | $\tilde O(d)$ | **Ours (Thm 4.1)** |

### Key Findings
- **Basin existence is fully characterized by 1D conditions**: $C_{\text{lip}}(R)$ depends only on 1D Gaussian integrals of $f$. If $f$ grows at most polynomially, the basin radius is dimension-independent.
- **Spectral methods hit the $\epsilon^{1/4}$ limit**: Refinement via robust GD is necessary to reach $\sqrt\epsilon$.
- **ESC serves as "High-order Monotonicity"**: It determines if the top eigenvector of the second-moment matrix aligns with $\beta^\star$, enabling signal recovery even for non-monotonic links.

## Highlights & Insights
- **1D Reduction Paradigm**: All high-dimensional structures (basin radius, Hessian spectrum, reachability) are reduced to 1D properties of the link function using Stein identities.
- **Modern Activations in SIM Theory**: GeLU and Swish are formally proven to be "robustly recoverable," suggesting that gated Transformer layers may have broader theoretically analyzable boundaries.
- **Transferable Techniques**: Using $(4,C_4)$ hypercontractivity to integrate robust PCA subroutines into SIM and the two-stage "Spectral + RGD" framework are modularized.

## Limitations & Future Work
- **Requirement $\|\beta^\star\|_2=1$**: This is a standard constraint but fixed; joint estimation of magnitude and direction remains an open question.
- **Non-optimal Error Rate**: The $O(\sigma\sqrt\epsilon)$ rate is suboptimal compared to the $O(\sigma\epsilon\log\frac1\epsilon)$ achievable in monotonic cases.
- **Indices $>2$**: When the signal disappears in the second-moment (e.g., certain cubic links), the spectral framework fails.
- **Known Link Function**: In semi-parametric statistics, $f$ is often unknown; joint estimation under corruption is a natural extension.

## Related Work & Insights
- **vs. Buna & Rebeschini (2025)**: They focus on phase retrieval using symmetric structures. Ours generalizes to asymmetric non-monotonic links using ESC and reduces PCA complexity to near-linear.
- **vs. Diakonikolas et al. (2022)**: Their squared loss sample complexity for Logistic reaches $\tilde O(d^2)$. Ours maintains the optimal $\tilde O(d)$ for a broader class.
- **Insight**: This "Basin = 1D Integral + Reachability = ESC" decomposition can be applied to other non-linear structural recovery problems, such as low-rank phase retrieval or sparse SIM.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Robust Estimation Under Heterogeneous Corruption Rates](../../NeurIPS2025/optimization/robust_estimation_under_heterogeneous_corruption_rates.md)
- [\[NeurIPS 2025\] Learning Single-Index Models via Harmonic Decomposition](../../NeurIPS2025/optimization/learning_single-index_models_via_harmonic_decomposition.md)
- [\[ICML 2026\] Automatic Unsupervised Ensemble Outlier Model Selection–Extended Version](automatic_unsupervised_ensemble_outlier_model_selection--extended_version.md)
- [\[AAAI 2026\] Convex Clustering Redefined: Robust Learning with the Median of Means Estimator](../../AAAI2026/optimization/convex_clustering_redefined_robust_learning_with_higher_order_norms_and_beyond.md)
- [\[ICML 2026\] Full-Batch Gradient Descent Outperforms One-Pass SGD: Sample Complexity Separation in Single-Index Learning](full-batch_gradient_descent_outperforms_one-pass_sgd_sample_complexity_separatio.md)

</div>

<!-- RELATED:END -->
