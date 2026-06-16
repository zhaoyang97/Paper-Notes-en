---
title: >-
  [Paper Note] Convex Basins in Single-Index Model Loss Landscapes: Applications to Robust Recovery under Strong Adversarial Corruption
description: >-
  [ICML 2026][Optimization & Theory][Paper Note] Under heavy-tailed noise and a constant fraction of strong adversarial corruption, the authors prove that the squared loss of Gaussian Single-Index Models (SIMs) for a broad class of non-monotonic link functions (GeLU, Swish, Tanh, Probit, Logistic, Phase Retrieval...) possesses a dimension-independent convex basin of
tags:
  - ICML 2026
  - Optimization & Theory
date: 2026-05-08
content_hash: ac46780cb35f9f38
---
# Convex Basins in Single-Index Model Loss Landscapes: Applications to Robust Recovery under Strong Adversarial Corruption

**Conference**: ICML2026  
**arXiv**: [2605.29497](https://arxiv.org/abs/2605.29497)  
**Code**: None  
**Area**: Optimization / Robust Statistics / Single-Index Models  
**Keywords**: Single-Index Models, Robust Regression, Heavy-tailed Noise, Strong Adversarial Corruption, Convex Basins, Spectral Initialization

## TL;DR
Under heavy-tailed noise and a constant fraction of strong adversarial corruption, the authors prove that the squared loss of Gaussian Single-Index Models (SIMs) for a broad class of non-monotonic link functions (GeLU, Swish, Tanh, Probit, Logistic, Phase Retrieval...) possesses a dimension-independent convex basin of constant radius. Based on this, they design a robust recovery algorithm with $\tilde{O}(nd)$ time and $\tilde{O}(d)$ samples, achieving an optimal estimation error of $O(\sigma\sqrt{\epsilon})$.

## Background & Motivation
**Background**: Single-Index Models (SIM) $Y=f(X^\top\beta^\star)+\zeta$ unify linear regression, logistic regression, phase retrieval, and generalized linear models (GLMs) into a semi-parametric family. Modern gated neural network primitives like GeLU/Swish are naturally non-monotonic scalar functions. Existing robust recovery theories only cover three narrow settings: linear ($f(x)=x$), strictly monotonic links (Logistic and other GLMs), and phase retrieval ($f(z)=z^2$), as investigated by Pensia et al. (JASA 2024), Awasthi et al. (NeurIPS 2022), and Buna and Rebeschini (AISTATS 2025).

**Limitations of Prior Work**: Extending these proofs to general "non-monotonic + asymmetric" link functions (e.g., GeLU, Swish) results in immediate failure. First-order proof strategies (Arous et al.) rely on martingale-drift decomposition, which requires zero-mean random bias. Strong adversarial corruption destroys this property by arbitrarily polluting an $\epsilon$ fraction of samples. Furthermore, the symmetric structure of phase retrieval (quadratic link) prevents existing proofs from being transferred to asymmetric cases.

**Key Challenge**: To perform robust recovery in high dimensions, at least two structural conditions must be met: (i) the squared loss must have a **dimension-independent** convex basin of constant radius near $\beta^\star$ to permit second-order convergence proofs; (ii) this basin must be **efficiently reachable** from random initialization. These conditions were previously only known to hold simultaneously for phase retrieval; it was unknown whether broader non-monotonic links possess both properties.

**Goal**: To identify a set of mild sufficient conditions on the link function $f$ such that (i) and (ii) hold simultaneously, and to provide a robust recovery algorithm with near-linear time and optimal sample complexity.

**Key Insight**: The paper translates the "existence of a convex basin" into a purely 1D integral condition regarding the Gaussian expectation of $f$ (Assumption 2.1), and "reachability of the basin" into a second-moment criterion $\mathrm{ESC}(\beta,f):=\mathbb{E}[(f'(X^\top\beta))^2 + f(X^\top\beta)f''(X^\top\beta)]>0$ (Assumption 2.2). This effectively collapses the burden of high-dimensional proofs onto the 1D properties of the link function itself.

**Core Idea**: The authors characterize "basin existence + reachability" using two 1D conditions: "Local Lipschitz Complexity + ESC". They then use spectral initialization to enter the basin and Robust Gradient Descent to refine the estimate, extending robust recovery from phase retrieval to the entire class of SIMs with a generative index $\le 2$.

## Method

### Overall Architecture
Input: A sample set $\{(x_i,y_i)\}_{i=1}^N$ corrupted by an $\epsilon$ fraction of strong adversarial noise, where $x_i\sim\mathcal{N}(0,\mathbf I_d)$, and an unknown index vector $\beta^\star$ with $\|\beta^\star\|_2=1$. Output: A unit vector satisfying $\|\hat\beta-\beta^\star\|_2=O(\sigma\sqrt\epsilon)$. Algorithm 1 splits samples into $P+1$ equal-sized buckets. It first uses LRSI for spectral initialization ($\beta_0\leftarrow\text{LRSI}(N_1,\epsilon)$) to fall into the convex basin, then employs LRGD for robust gradient descent ($\beta_P\leftarrow\text{LRGD}(N_{2..P+1},\beta_0,\epsilon,\alpha,\gamma)$) to refine the error from $O(\epsilon^{1/4})$ to $O(\sqrt\epsilon)$, before finally normalizing the output. The algorithmic framework is supported by the structural guarantee of the convex basin:

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Corrupted Sample Set<br/>Heavy-tailed noise + ε strong adversarial corruption"] --> B["Random Split into P+1 Buckets<br/>(sample splitting)"]
    C["Convex Basin Existence<br/>1D Gaussian integral criterion → Near β*<br/>Dimension-independent, constant radius basin"]
    C -. Guarantees convergence & strong convexity .-> D
    B --> D["LRSI Spectral Initialization<br/>YX second moment + Robust 1-ePCA<br/>→ β0 enters the basin"]
    D -->|Spectral error capped at O(ε^1/4)| E["LRGD Robust Gradient Descent<br/>Robust mean estimation of gradients + Independent buckets<br/>P=O(1) iterations"]
    E --> F["Normalized Output βP/‖βP‖<br/>Final error O(σ√ε)"]
```

### Key Designs

**1. Existence of Convex Basin (Assumption 2.1 + Theorem 3.1): Collapsing High-Dimensional Basins into 1D Gaussian Integrals**

For robust recovery to utilize second-order convergence proofs, the squared loss $\mathcal L(\beta)=\frac12\mathbb E[(f(X^\top\beta)-Y)^2]$ must have a dimension-independent constant-radius convex basin near $\beta^\star$. This had not been proven for non-monotonic links. The authors simplify the high-dimensional structure into 1D: they use Gaussian symmetry at $\beta^\star$ to simplify the Hessian to:

$$H(\beta^\star)=\mathbb E[(f'(Z))^2]\,\mathbf I_d+\big(\mathbb E[Z^2(f'(Z))^2]-\mathbb E[(f'(Z))^2]\big)\beta^\star\beta^{\star\top},$$

yielding a minimum eigenvalue $\lambda_{\min}(H(\beta^\star))=\mu:=\min\{\mathbb E[f'(Z)^2],\mathbb E[Z^2 f'(Z)^2]\}$. They then use the mean value theorem to bound the operator norm of $H(\beta)-H(\beta^\star)$ by $C_{\text{lip}}(R)\cdot\|\beta-\beta^\star\|$, where $C_{\text{lip}}(R)=\sup_{\|\beta-\beta^\star\|\le R}\sqrt{\mathbb E_{z\sim\mathcal N(0,\|\beta\|^2)}[18f'(z)^2 f''(z)^2+2f'''(z)^2 f(z)^2]}$. Crucially, $C_{\text{lip}}(R)$ only involves 1D Gaussian integrals of $f$ and its first three derivatives, independent of dimension $d$. Choosing $R\le\mu/(2(315)^{1/4}C_{\text{lip}}(R))$ guarantees $\frac{\mu}{2}\mathbf I_d\preceq H(\beta)\preceq(\frac{\mu}{2}+\mu_1)\mathbf I_d$ over the ball $\mathcal B(\beta^\star,R)$. "Basin existence" is thus reduced to the link function $f$ having finite fourth moments (polynomial growth).

**2. ESC Condition + LRSI Spectral Initialization (Assumption 2.2 + Theorem 4.2): Using 2nd Moment Spectral Methods under Corruption**

Possessing a basin is insufficient; one must reach it efficiently. Previous initialization methods for phase retrieval relied on the symmetry of $f$, which does not apply to GeLU/Swish. The authors define $\tilde Y:=YX$ and use Stein's second-order identity to prove that $\beta^\star$ is the top eigenvector of $\mathbb E[\tilde Y\tilde Y^\top]$ if and only if the criterion $\mathrm{ESC}(\beta;f):=\mathbb E[(f'(X^\top\beta))^2+f(X^\top\beta)f''(X^\top\beta)]>0$. Since $\mathrm{ESC}(\beta,f)=\mathbb E[(f^2(X^\top\beta))'']$, it represents a form of "higher-order monotonicity"—even if $f$ is not monotonic, if $f^2$ is sufficiently convex, the signal can be identified via second-moment spectral methods. They prove $\tilde Y$ is $(4,C_4)$-hypercontractive, allowing the use of the near-linear time robust 1-ePCA subroutine from Jambulapati et al. (2024), yielding $\beta_0$ with $\text{dist}(\beta_0,\beta^\star)=O(C_4\epsilon^{1/4}\sqrt{\sigma^2+\mathbb E[f^2]+c}/\sqrt c)$.

**3. LRGD Robust Gradient Descent (Theorem 4.1): Refining $\epsilon^{1/4}$ Error to the Information-Theoretic Optimal $\sigma\sqrt\epsilon$**

Spectral initialization is naturally limited by an $\epsilon^{1/4}$ statistical lower bound. To reach $\sigma\sqrt\epsilon$, the authors express the gradient as $\nabla\mathcal L(\beta)=\mathbb E[(f(X^\top\beta)-Y)f'(X^\top\beta)X]$ and replace this expectation with a robust mean estimation (Diakonikolas et al., 2022). This yields a robust gradient $\hat g$ satisfying $\|\hat g-\nabla\mathcal L(\beta)\|=O(\sigma'\sqrt\epsilon)$. By using independent sample buckets (sample splitting) for each step to avoid trajectory-dependent correlations, the error is reduced to the optimal magnitude within $P=O(1)$ iterations while maintaining $\tilde O(nd)$ total time.

### Loss & Training
The objective is the squared loss $\mathcal L(\beta)=\frac12\mathbb E[(f(X^\top\beta)-Y)^2]$. Total sample complexity is $n=\tilde O(m+P\tilde m)$, where $m=\Theta(C_4^2(d\log d+\log(1/\delta))/\epsilon^{3/2})$ is for spectral initialization and $\tilde m=\tilde O(d/\epsilon)$ is for each round of robust gradient estimation. Total time is $\tilde O(nd)$. The tolerated corruption fraction $\epsilon$ is regulated by the links' hypercontractivity and the basin radius.

## Key Experimental Results

This is a theoretical paper. The following tables summarize theoretical metrics across different settings.

### Main Results Comparison (Robust Recovery Metrics)

| Link / Task | Noise + Corruption | Error Rate | Time | Samples | Source |
|----------------|-------------|--------|------|------|------|
| Linear $f(x)=x$ | Heavy-tail + Adversarial | $O(\sigma\sqrt\epsilon)$ (Opt.) | $\tilde O(nd)$ | $\tilde O(d)$ | Cherapanamjeri et al. 2020 |
| Logistic (Monotone GLM) | Gauss + Adversarial | $O(\sigma\epsilon\log\frac1\epsilon)$ (Opt.) | Not explicit | $\tilde O(d)$ | Awasthi et al. 2022 |
| Logistic (Squared loss) | Heavy-tail + Adversarial | $O(\sigma\sqrt\epsilon)$ | $\tilde O(nd)$ Streaming | $\tilde O(d^2)$ | Diakonikolas et al. 2022 |
| Phase Retrieval $f(z)=z^2$ | Heavy-tail + Adversarial | $O(\sigma\sqrt\epsilon)$ | Poly | $\tilde O(d)$ | Das & Batra 2026 |
| **GeLU / Swish / Tanh / SIMs** | Heavy-tail + Adversarial | $O(\sigma\sqrt\epsilon)$ | $\tilde O(nd)$ | $\tilde O(d)$ | **Ours (Thm 4.1)** |

### Structural Comparison Table

| Condition / Conclusion | Phase Retrieval | Monotone GLM | **Ours** |
|-------------|------|------|------|
| Basin Existence | Quadratic only | Monotone only | Any $f$ satisfying Assump 2.1 |
| Basin Radius $R$ | Dim-independent constant | Unclear | Dim-independent (1D integral) |
| Reachability Tool | Symmetry + Vanilla PCA | Tensor PCA | Stein + Robust 1-ePCA |
| Asymmetric & Non-monotone | No | No | **Yes** |
| Init Complexity | Poly | Poly | $\tilde O(nd)$ Near-linear |

### Key Findings
- **Convex basin existence is fully characterized by 1D conditions**: $C_{\text{lip}}(R)$ depends only on the Gaussian integrals of $f$ and its first three derivatives. Thus, as long as $f$ grows at most polynomially, the basin radius is automatically dimension-independent.
- **Spectral methods are limited to $\epsilon^{1/4}$**, requiring Robust GD for $\sqrt\epsilon$ refinement. This replicates the "Spectral Init + GD" structure of Wirtinger Flow but generalizes reachability using ESC.
- **ESC is the true "High-order Monotonicity"**: Since $\mathrm{ESC}(\beta,f)=\mathbb E[(f^2)'']$, it determines if the signal direction aligns with the top eigenvector of the second-moment matrix, allowing recovery even when $f$ itself is not monotonic.

## Highlights & Insights
- **1D-Reduction Proof Paradigm**: All high-dimensional structures (basin radius, Hessian spectrum, reachability) are collapsed into 1D Gaussian integrals side-stepping $d$-dependence.
- **Integrating Modern Activations into SIM Theory**: GeLU and Swish are now theoretically "robustly recoverable," suggesting that the boundary of gated Transformer layers is more analytically accessible than previously thought.
- **Transferable Techniques**: (1) Linearizing non-linear signal identification via $YX$ second moments and Stein identities; (2) Connecting robust PCA complexity directly to SIM via hypercontractivity.

## Limitations & Future Work
- **Constraint on $\|\beta^\star\|_2=1$**: This is a standard assumption, but estimating both magnitude and direction simultaneously remains an open question.
- **Sub-optimal Error Rate**: This work achieves $O(\sigma\sqrt\epsilon)$, while Awasthi et al. (2022) achieve $O(\sigma\epsilon\log\frac1\epsilon)$ for monotonic links. Improving the rate to near-linear in $\epsilon$ for non-monotonic links is unsolved.
- **Generative Index $>2$**: If signals vanish in the second moment (e.g., specific terms in $f(z)=z^3$), the spectral framework fails.
- **Known Link Function**: In many semi-parametric settings, $f$ is unknown. Joint estimation of $f$ and $\beta$ under corruption is a natural extension.

## Related Work & Insights
- **vs Buna & Rebeschini (AISTATS 2025)**: They target phase retrieval using symmetric links and poly-time PCA; Ours extends this to non-monotonic asymmetric links in near-linear time using ESC.
- **vs Diakonikolas et al. (2022)**: Their squared-loss logistic recovery requires $\tilde O(d^2)$ samples; Ours maintains the optimal $\tilde O(d)$ while covering Logistic as a sub-case.
- **vs Awasthi et al. (2022)**: They provide an optimal rate for monotone GLMs but lack runtime guarantees; Ours trades rate for near-linear time and non-monotonicity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐ (Theory only)
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Robust Estimation Under Heterogeneous Corruption Rates](../../NeurIPS2025/optimization/robust_estimation_under_heterogeneous_corruption_rates.md)
- [\[NeurIPS 2025\] Learning Single-Index Models via Harmonic Decomposition](../../NeurIPS2025/optimization/learning_single-index_models_via_harmonic_decomposition.md)
- [\[AAAI 2026\] Convex Clustering Redefined: Robust Learning with the Median of Means Estimator](../../AAAI2026/optimization/convex_clustering_redefined_robust_learning_with_higher_order_norms_and_beyond.md)
- [\[ICML 2026\] Automatic Unsupervised Ensemble Outlier Model Selection–Extended Version](automatic_unsupervised_ensemble_outlier_model_selection--extended_version.md)
- [\[ICLR 2026\] Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate](../../ICLR2026/optimization/convex_dominance_in_deep_learning_i_a_scaling_law_of_loss_and_learning_rate.md)

</div>

<!-- RELATED:END -->
