---
title: >-
  [Paper Note] Differentially Private Two-Stage Gradient Descent for Instrumental Variable Regression
description: >-
  [ICLR2026][AI Safety][Differential Privacy] This paper proposes DP-2S-GD—the first differentially private algorithm for Instrumental Variable Regression (IVaR). It rewrites the classic Two-Stage Least Squares (2SLS) as a two-stage gradient descent process, performing per-sample clipping and injecting calibrated Gaussian noise in each gradient update step to satisfy $\rho$-zCDP. The authors provide finite-sample convergence rates that explicitly characterize the optimization-p…
tags:
  - "ICLR2026"
  - "AI Safety"
  - "Differential Privacy"
  - "Instrumental Variable Regression"
  - "Two-Stage Least Squares"
  - "Gradient Perturbation"
  - "Finite-Sample Convergence"
date: 2026-05-08
content_hash: aadd525ba630c46f
---

# Differentially Private Two-Stage Gradient Descent for Instrumental Variable Regression

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=XQDy4obYLZ](https://openreview.net/forum?id=XQDy4obYLZ)  
**Code**: To be confirmed  
**Area**: Differential Privacy / Causal Inference / Learning Theory  
**Keywords**: Differential Privacy, Instrumental Variable Regression, Two-Stage Least Squares, Gradient Perturbation, Finite-Sample Convergence

## TL;DR
This paper proposes DP-2S-GD—the first differentially private algorithm for Instrumental Variable Regression (IVaR). It rewrites the classic Two-Stage Least Squares (2SLS) as a two-stage gradient descent process, performing per-sample clipping and injecting calibrated Gaussian noise in each gradient update step to satisfy $\rho$-zCDP. The authors provide finite-sample convergence rates that explicitly characterize the optimization-privacy-sampling trade-off.

## Background & Motivation
**Background**: Instrumental Variable Regression (IVaR) is a fundamental tool for causal inference. When an explanatory variable $x$ is influenced by unobserved confounders $u$, violating the exogeneity assumption, Ordinary Least Squares (OLS) yields biased and inconsistent estimates. IVaR introduces an instrumental variable $z$ that is correlated with $x$ but affects the outcome $y$ only through $x$, thereby recovering consistent estimates of the structural parameter $\beta$. It is increasingly important in machine learning scenarios such as recommendation systems (where exposure is confounded by historical preferences) and reinforcement learning (where actions and rewards are influenced by unobserved contexts). The classic approach is Two-Stage Least Squares (2SLS), which has a closed-form solution.

**Limitations of Prior Work**: Many applications of IVaR involve sensitive data—personal health records, financial transactions, or user behavior. In such scenarios, even releasing model estimates or intermediate statistics may leak information about a single individual. Differential Privacy (DP) provides a rigorous mathematical framework to ensure that algorithm outputs do not expose any individual data point. However, to the authors' knowledge, **no prior work has addressed the problem of "performing IVaR under differential privacy constraints."**

**Key Challenge**: Mature solutions for DP exist for OLS, where gradient perturbation (clipping + Gaussian noise injection) combined with modern privacy accounting yields the sharpest statistical rates. However, IVaR is not a single moment condition like OLS—its closed-form 2SLS estimator relies on **nested matrix multiplication and inversion**, making the sensitivity extremely difficult to characterize directly. Additionally, the ill-posedness under weak instruments means that sufficient statistic perturbation and consensus-type mechanisms are difficult to apply directly. Simply put: the closed-form structure makes it unclear where and how much noise should be injected.

**Goal**: Can a differentially private algorithm be designed for linear IVaR models that simultaneously achieves statistically efficient convergence rates? This requires solving three sub-problems: (1) finding an algorithmic structure that allows "noise injection during iteration"; (2) strictly controlling per-step sensitivity and calibrating noise; and (3) providing non-asymptotic utility guarantees.

**Key Insight**: Instead of tackling the closed-form 2SLS directly, the authors replace it with an iterative **Two-Stage Gradient Descent** (2S-GD) process. The advantage of the gradient method is that noise can be injected into the gradient updates in a controlled manner during each iteration. Sensitivity is directly bounded by the clipping threshold, and the overall privacy loss over multiple steps can be cleanly accumulated using the additive composition rules of zCDP.

**Core Idea**: By rewriting 2SLS as an iterative two-stage gradient descent and applying per-sample clipping plus Gaussian noise to the gradients of both stages, the authors use zCDP accounting to sum the privacy loss, obtaining both privacy guarantees and provable convergence rates for IVaR for the first time.

## Method

### Overall Architecture
The linear model for IVaR is:
$$y = \beta^\top x + \epsilon_1, \qquad x = \Theta^\top z + \epsilon_2,$$
where $\epsilon_1, \epsilon_2$ are correlated due to a common confounder $u$. Given data $(Z,X,Y)=\{(z_i,x_i,y_i)\}_{i=1}^n$, IVaR solves a bi-level optimization problem: the outer layer $\hat\beta=\arg\min_\beta \frac1n\sum_i (y_i-\beta^\top\hat\Theta^\top z_i)^2$, and the inner layer $\hat\Theta=\arg\min_\Theta \frac1n\sum_j \|x_j-\Theta^\top z_j\|^2$. Classic 2SLS provides a closed-form solution: the first stage $\hat\Theta=(Z^\top Z)^{-1}Z^\top X$ (regressing $x$ on $z$), and the second stage $\hat\beta_{\text{2SLS}}=(\hat\Theta^\top Z^\top Z \hat\Theta)^{-1}\hat\Theta^\top Z^\top Y$ (regressing $y$ on the predicted $\hat X=Z\hat\Theta$).

The overall workflow of this paper is: **first, replace the closed-form 2SLS with a non-private Two-Stage Gradient Descent (2S-GD) baseline**—alternately updating the first-stage projection matrix $\Theta^{(t)}$ and the second-stage regression parameter $\beta^{(t)}$ in each iteration, which serves as a "gradient version" of 2SLS. **Then, privatize it into DP-2S-GD** by adding two modifications: (i) per-sample clipping for gradients in both stages to bound sensitivity; (ii) injecting Gaussian perturbation into each update step for $\Theta$ and $\beta$, with noise scales calibrated according to target privacy budgets $\rho_1, \rho_2$. Finally, zCDP composition is used to provide the overall privacy guarantee, accompanied by a finite-sample error bound characterizing the trade-offs. The inputs are sensitive data $(Z,X,Y)$ and privacy budgets $(\rho_1, \rho_2)$; the output is the private parameter trajectory $\{\Theta^{(t)}\}, \{\beta^{(t)}\}$.

### Key Designs

**1. Replacing closed-form 2SLS with Two-Stage Gradient Descent: Making noise injection solvable**

The trouble with closed-form 2SLS is that $\hat\beta$ depends on nested $Z^\top Z$ inversions and matrix chains, making sensitivity characterization impossible. Injecting noise directly into the Gram matrix $Z^\top Z$ would amplify errors due to ill-posedness (weak instruments). The authors switch to an iterative 2S-GD algorithm: alternating between two steps of gradient descent. The first step updates $\Theta^{(t)}$ using gradient $z_i(z_i^\top\Theta^{(t)}-x_i^\top)$, and the second step updates $\beta^{(t)}$ using gradient $\Theta^{(t)\top}z_i(z_i^\top\Theta^{(t)}\beta^{(t)}-y_i)$. The value of this rewrite is not faster computation, but that it **shifts the privacy injection point from the "unmanageable closed-form" to "clear gradients at each step"**. Gradients have a per-sample summation structure where the influence of a single sample is obvious, providing a clear focus for clipping and noise injection. Furthermore, gradient methods naturally support regularization, minibatch, streaming data, and early stopping, making them more modular and practical than closed-form solutions.

**2. Per-sample gradient clipping to bound sensitivity**

Differential privacy requires the influence of any single data point on the output to be bounded. However, in IVaR gradients, $z_i, x_i, y_i$ are sub-Gaussian random variables, and single-sample gradient norms are theoretically unbounded. The authors apply $\mathrm{CLIP}$ with thresholds $\gamma_1, \gamma_2$ to per-sample gradients in the two stages: $z_i(z_i^\top\Theta^{(t)}-x_i^\top)$ during $\Theta$ updates, and $\Theta^{(t)\top}z_i(z_i^\top\Theta^{(t)}\beta^{(t)}-y_i)$ during $\beta$ updates. Clipping hard-caps the sensitivity of each update to the order of $2\gamma_1/n$ and $2\gamma_2/n$, determining the Gaussian noise scale required for target privacy. A key technical detail: $\gamma_1, \gamma_2$ are chosen on the order of $c_0(\sqrt q+\sqrt{\tau+\log(nT)})^2$, so that **clipping does not actually change the gradient with high probability** (see Lemma D.1). This means clipping only serves to bound worst-case sensitivity without introducing extra bias under normal conditions, allowing the private convergence rate to align with the non-private 2S-GD.

**3. Two-stage Gaussian noise injection + zCDP budget allocation: Cleanly summing multi-step privacy loss**

After clipping, DP-2S-GD injects Gaussian noise into each stage: $\Theta^{(t+1)}=\Theta^{(t)}-\frac\eta n\sum_i \mathrm{CLIP}_{\gamma_1}(\cdot)+\eta\Xi^{(t)}$ and $\beta^{(t+1)}=\beta^{(t)}-\frac\alpha n\sum_i\mathrm{CLIP}_{\gamma_2}(\cdot)+\alpha\nu^{(t)}$, where $\mathrm{vec}(\Xi^{(t)})\sim N(0,\lambda_1^2 I)$ and $\nu^{(t)}\sim N(0,\lambda_2^2 I)$. The privacy analysis treats the two stages as two independent Gaussian mechanisms with sensitivity controlled by $\gamma_1, \gamma_2$. The authors **specifically choose Zero-Concentrated Differential Privacy (zCDP) instead of $(\epsilon,\delta)$-DP**. Since the algorithm composes many identical Gaussian mechanisms over $T$ iterations across two stages, $(\epsilon,\delta)$-DP composition would lead to linear accumulation of $\epsilon, \delta$ and cumbersome formulas. zCDP uses Rényi divergence to characterize privacy loss, allows direct addition of parameters, and results in tighter, cleaner composition. Specifically (Proposition 3.1), by setting
$$\lambda_1=\frac{2\gamma_1}{n}\sqrt{\frac{T}{\rho_1}},\qquad \lambda_2=\frac{2\gamma_2}{n}\sqrt{\frac{T}{\rho_2}},$$
the algorithm satisfies $\rho$-zCDP, where $\rho=\rho_1+\rho_2=\frac{2T}{n^2}\big(\frac{\gamma_1^2}{\lambda_1^2}+\frac{\gamma_2^2}{\lambda_2^2}\big)$. Here $\rho_1$ protects the first-stage trajectory $\{\Theta^{(t)}\}$ and $\rho_2$ protects the second-stage trajectory $\{\beta^{(t)}\}$. They can be allocated as needed (e.g., if $\Theta$ is non-sensitive, let $\rho_1=\infty$ for no noise), and the overall $(\rho_1+\rho_2)$-zCDP is controllable end-to-end.

**4. Finite-sample error bound for optimization-privacy-sampling trade-off (Theoretical Core)**

The most substantial contribution is the derivation of the non-asymptotic error bound for this private iterative process (Theorem 3.1). Under Assumption 2 ($z$ and errors are sub-Gaussian), with high probability:
$$\|\beta^{(T)}-\hat\beta\|\lesssim \underbrace{\kappa(\tau)^{T/2}}_{\text{Optimization}}+\underbrace{\frac{\sqrt p(\sqrt q+\sqrt\tau)^3}{n\sqrt{\min\{\rho_1,\rho_2\}}}\sqrt T}_{\text{Privacy Noise}}+\underbrace{\frac{\sqrt{pq}(\tau+\log(pq))}{\sqrt n}}_{\text{Statistical Error}},$$
where $0<\kappa(\tau)<1$ is the contraction rate. The three terms correspond to: **Optimization** (exponential decay with $T$), **Privacy Noise** (grows with $\sqrt T$ because noise scale $\lambda$ contains $\sqrt{T}$), and **Statistical Error** (inherent error of non-private GD that decreases with $n$). This decomposition is elegant as it **directly provides the optimal range for the number of iterations $T$**—more iterations reduce the optimization error but amplify privacy noise. Thus, the optimal $T$ is "sub-linear yet super-logarithmic" relative to $n$, i.e., $T\lesssim \rho_1 n^{2-\epsilon}/[p(\sqrt q+\sqrt\tau)^6]$. The technical difficulty lies in controlling the interaction between "privacy noise" and "gradient dynamic contraction" over multiple steps. Corollary 3.1 shows that when $\rho_1=\rho_2=\infty$ (no noise), the bound reduces to the rate of non-private 2S-GD, ensuring consistency.

### Loss & Training
No additional training loss: the algorithm simply performs two-stage gradient descent on the IVaR bi-level objective with clipping and noise injection. Key hyperparameters include step sizes $\eta, \alpha$ (which must satisfy stability conditions in Eq. 3; $\eta=\frac1{(1+\delta(\tau))^2}$ and $\alpha=\frac2{2\bar\gamma(\tau)+\gamma(\tau)}$ in experiments), clipping thresholds $\gamma_1, \gamma_2$, noise scales $\lambda_1, \lambda_2$ (derived from $\rho_1, \rho_2, T$), and iteration number $T$. Remark 3.3 provides approximately optimal step sizes and notes that the error is insensitive in the neighborhood of the optimal step size.

## Key Experimental Results

### Main Results
The authors validate the theory on synthetic and real data. Privacy intensity levels: $\rho=0.1$ for strong privacy, $\rho=1$ for medium, and $\rho=10$ for weak (corresponding to $\epsilon \approx 2.25 / 7.79 / 31.47$ when $\delta=10^{-5}$). Real data uses the classic Angrist dataset studying the "causal effect of fertility on mothers' labor supply," where the instrumental variable is "whether the first two children have the same gender."

| Experiment | Setting | Key Results |
|--------|------|------|
| Synthetic vs $n$ (Fig 3a) | $p=q=r=5$, $T=20$, $\rho=10$ | Points fall into the plateau of the error bound; error decreases at $1/\sqrt n$, matching theory |
| Synthetic vs $n$ (Fig 3b) | $p=q=r=50$, $T=20$ | $T=20$ violates $T$ conditions in Eq (5); error is significantly larger than low-dim case, confirming condition necessity |
| Angrist Real Data (Fig 5) | $n=8065$, $T=20$, $\rho_1=\rho_2=1$ | $\beta^{(T)}$ concentrates around $-4.3$ (one extra child reduces annual supply by ~4.3 weeks), consistent with 2SLS; converges in ~15 iterations |

### Ablation Study

| Configuration | Phenomenon | Explanation |
|------|---------|------|
| Varying $T$, small $\rho_1$ (Fig 4a) | Error rises sharply after a critical $T$ | Matches Eq (5) upper bound for $T$: strong privacy on $\{\Theta^{(t)}\}$ requires moderate $T$ |
| Varying $T$, noise only on $\beta$ ($\rho_2$ small, Fig 4b) | Error trend fits theory curve Fig 2 | Protecting only $\beta$ ($\rho_1=\infty$) avoids $T$ constraints in Eq (5) |
| Privacy budget $\rho_1, \rho_2$ size (Fig 5b) | Larger budget leads to closer fit to 2SLS baseline | Direct manifestation of the privacy-utility trade-off |
| $\rho=\infty$ (Corollary 3.1) | Reduces to 2S-GD convergence rate | Still slower than closed-form 2SLS by a factor of $\sqrt p$ (Remark 3.8) |

### Key Findings
- **Optimal $T$ is sub-linear and super-logarithmic with respect to $n$**: Too few iterations prevent optimization convergence, while too many lead to privacy noise explosion. A "sweet spot" exists as characterized by Eq (5), and experiments confirm its location.
- **Privacy budget allocation across stages is critical**: With finite samples and strong privacy on the first stage $\Theta$ (small $\rho_1$), error worsens sharply after a certain $T$. If only $\beta$ is protected, this restriction is absent.
- **Gradient methods have an inherent $\sqrt p$ loss compared to closed-form 2SLS** (Remark 3.8): Even without noise, the rate at which 2S-GD approaches $\hat\beta$ is slower than closed-form 2SLS by a factor of $\sqrt p$, rooted in the fact that gradient iteration only approximates the second-stage moment condition.

## Highlights & Insights
- **"Algorithm structure modification for noise injection" is key**: Closed-form 2SLS makes sensitivity characterization impossible. Instead of forcing it, the authors step back and rewrite it as an iterative gradient method—shifting noise from an "unmanageable closed-form" to "per-sample visible gradients." This strategy of "modifying algorithm structure for analyzability" is transferable to other closed-form estimators.
- **Three-term error decomposition provides operational guidance**: Decomposing error into "Optimization / Privacy Noise / Statistics" and deriving that optimal $T$ is sub-linear/super-logarithmic is not just post-hoc explanation but a quantitative guide for practice.
- **Pragmatic choice of zCDP over $(\epsilon,\delta)$-DP**: For scenarios involving many identical Gaussian mechanisms across stages and iterations, zCDP yields cleaner formulas and tighter composition, making it the right tool for multi-step iteration analysis.
- **Clear positioning as a foundational work**: This first work on IVaR + DP fills a clear gap, providing both privacy guarantees and provable rates while honestly reporting the inherent $\sqrt p$ gap compared to 2SLS.

## Limitations & Future Work
- **$\sqrt p$ gap vs closed-form 2SLS**: The authors acknowledge (Remark 3.8, Conclusion) that 2S-GD is slower than closed-form 2SLS by a $\sqrt p$ factor regardless of privacy constraints. Narrowing this gap through algorithm modification remains an open question.
- **Lack of privacy-utility lower bounds**: The paper only provides upper bounds without proving matching lower bounds for IVaR under DP, so it is unclear if the current rates are optimal. This is listed as a future direction.
- **Limited to linear IVaR**: The method and analysis are tailored to linear structural models and specific distribution assumptions (sub-Gaussian, isotropic, instrument validity). Nonlinear IV (e.g., KernelIV, DeepIV) under DP is not covered.
- **Dependence on condition number**: Like most private first-order methods, convergence is highly dependent on the problem's condition number, which can be poor under weak instruments.

## Related Work & Insights
- **Comparison to DP-SGD and OLS**: Prior works show that gradient perturbation + clipping in OLS provides the sharpest statistical rates. This paper follows that path but addresses the unique difficulty of IVaR's nested two-stage moment conditions by using iterative 2S-GD to bypass closed-form sensitivity issues.
- **Comparison to Sufficient Statistic Perturbation / Consensus Mechanisms**: These are usable for OLS but difficult for IVaR due to ill-posedness and high sensitivity of moment equations. Purely sufficient statistic pipelines also require larger samples ($n$) in high dimensions. Gradient perturbation avoids spectral dependence amplification from Gram matrix noise.
- **Comparison to Optimization-based IVaR (2S-GD, bi-level GD)**: Recent works have framed IVaR as scalable/online stochastic optimization but assume unlimited data access without end-to-end DP. This paper is the **first to add DP to such optimization frameworks and provide a unified finite-sample analysis**.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work on IVaR + DP, fills a clear gap, "rewriting algorithm for noise injection" is a clean and powerful idea.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic + Angrist data confirm theoretical findings, though scale and diversity are limited (primary focus is validation).
- Writing Quality: ⭐⭐⭐⭐⭐ Logical flow from motivation to algorithm, privacy, and utility analysis. The three-term error decomposition is very clear.
- Value: ⭐⭐⭐⭐ Provides a theoretically grounded private tool for causal analysis on sensitive data, with direct practical guidance (optimal $T$, trade-offs).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Missing Mass for Differentially Private Domain Discovery](differentially_private_domain_discovery.md)
- [\[ICLR 2026\] PE-SGD: Differentially Private Deep Learning via Evolution of Gradient Subspace for Text](pe-sgd_differentially_private_deep_learning_via_evolution_of_gradient_subspace_f.md)
- [\[NeurIPS 2025\] Differentially Private High-dimensional Variable Selection via Integer Programming](../../NeurIPS2025/ai_safety/differentially_private_high-dimensional_variable_selection_via_integer_programmi.md)
- [\[ICLR 2026\] On Optimal Hyperparameters for Differentially Private Deep Transfer Learning](on_optimal_hyperparameters_for_differentially_private_deep_transfer_learning.md)
- [\[ICLR 2026\] Optimizing Canaries for Privacy Auditing with Metagradient Descent](optimizing_canaries_for_privacy_auditing_with_metagradient_descent.md)

</div>

<!-- RELATED:END -->
