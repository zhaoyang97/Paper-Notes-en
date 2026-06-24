---
title: >-
  [Paper Note] Scaling Laws and Spectra of Shallow Neural Networks in the Feature Learning Regime
description: >-
  [ICLR 2026][Learning Theory][Scaling Laws] This paper **precisely maps** the Empirical Risk Minimization (ERM) problem of two-layer neural networks (diagonal and quadratic) under weight decay training to LASSO and low-rank matrix compressed sensing. This mapping allows for the first analytical characterization of a complete excess risk phase diagram (8 phases, including benign/harmful overfitting and interpolation peaks) in a regime where "true feature learning occurs." Furth…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Neural Scaling Laws"
  - "Feature Learning"
  - "Scaling Laws"
  - "Weight Spectra"
  - "Compressed Sensing"
  - "Approximate Message Passing"
date: 2026-05-08
content_hash: 363eb3164cd96f56
---

# Scaling Laws and Spectra of Shallow Neural Networks in the Feature Learning Regime

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Q3yLIIkt7z](https://openreview.net/forum?id=Q3yLIIkt7z)  
**Code**: None  
**Area**: Learning Theory / Neural Scaling Laws / Feature Learning  
**Keywords**: Scaling Laws, Feature Learning, Weight Spectra, Compressed Sensing, Approximate Message Passing

## TL;DR
This paper **precisely maps** the Empirical Risk Minimization (ERM) problem of two-layer neural networks (diagonal and quadratic) under weight decay training to LASSO and low-rank matrix compressed sensing. This mapping allows for the first analytical characterization of a complete excess risk phase diagram (8 phases, including benign/harmful overfitting and interpolation peaks) in a regime where "true feature learning occurs." Furthermore, it establishes a first-principles correspondence between each scaling law phase and the post-training weight spectrum (bulk / spike / heavy-tail), explaining the empirical observation that "heavy-tailed weight spectra $\leftrightarrow$ better generalization."

## Background & Motivation
**Background**: Neural scaling laws (the power-law decay of test loss with respect to sample size $n$, parameter count, and compute) are a pillar of modern deep learning. However, current **theoretical understanding is almost entirely limited to "lazy / random feature" regimes**, where features are fixed, the network degrades into a kernel method, and scaling behavior is determined by classic source & capacity conditions (Caponnetto-De Vito, Cui, etc.).

**Limitations of Prior Work**: The most critical defect of the lazy regime is that **features are not learned**—the first-layer weights barely move. In reality, the power of deep networks stems from feature learning: weight spectra evolve with rich phenomena such as heavy tails, outliers, and rank collapse/bleed-out (empirical observations by Martin-Mahoney). These phenomena are inexplicable in lazy theory; thus, existing scaling law theories fail to explain both "how feature learning changes scaling exponents" and "why weight spectra relate to generalization."

**Key Challenge**: To analyze feature learning, one must confront the **non-convex, high-dimensional** ERM problem of two-layer networks, which typically lacks an analytical end-to-end characterization. To maintain tractability, previous work often retreated to the lazy regime. There exists a tension between tractability and "true feature learning."

**Goal**: In a teacher-student setup, for two-layer network ERM with weight decay $\lambda$ and label noise $\Delta$ (Eq. 1): (1) Provide a complete scaling phase diagram of excess risk $R$ relative to effective sample size and regularization strength; (2) Characterize the post-training weight spectrum; (3) Link the two to explain the spectrum-generalization relationship.

**Key Insight**: The authors identify two classes of networks that allow for feature learning—**diagonal linear networks** and **quadratic activation networks**—whose ERM problems exhibit **exact mathematical equivalences**: diagonal networks $\equiv$ LASSO, and quadratic networks $\equiv$ low-rank matrix estimation with nuclear norm regularization (matrix compressed sensing). This allows the direct application of the powerful toolbox accumulated in signal processing for LASSO / compressed sensing (particularly Approximate Message Passing, AMP, and its State Evolution, SE) to obtain precise end-to-end predictions.

**Core Idea**: Replace "hard-solving non-convex optimization" with "equivalent sparse / low-rank estimation" to obtain analytical characterizations of excess risk scaling exponents and weight spectra while preserving true feature learning.

## Method

### Overall Architecture
The entire paper is an **analytical theory** without a trainable algorithmic pipeline. The main line of reasoning involves three steps: **Establishing the equivalent mapping → Deriving risk and spectra using AMP/SE → Mapping risk decomposition terms to spectral features**.

The research object is the ERM of a two-layer network $f(x;W,a)=a^\top\sigma(Wx+b)$ under $\ell_2$ weight decay (Eq. 1):

$$\min_{W,a}\sum_{\mu=1}^{n}\big(y_\mu-f(x_\mu;W,a)\big)^2+\lambda\big(\|W\|_F^2+\|a\|_2^2\big)$$

Data $x_\mu\sim\mathcal N(0,I_d)$, and labels are generated by a teacher of the same architecture with Gaussian noise of variance $\Delta$ added (Eq. 2). The quantity of interest is the excess risk $R(W,a)=\mathbb E_x[(f(x;W^\star,a^\star)-f(x;W,a))^2]$.

The authors consider two classes of networks, assuming the teacher's spectrum follows a **power-law / quasi-sparse** decay (coefficients decay as $i^{-\gamma}$ with $\gamma>1/2$). Both classes are unified into a single set of results via a **unified effective sample size** $n_{\rm eff}$:

$$n_{\rm eff}=\begin{cases}n & \text{Diagonal network}\\ n/d & \text{Quadratic network}\end{cases}$$

This simple definition allows the seemingly different diagonal and quadratic models to exhibit striking **universality** (sharing the same phase diagram and risk expressions).

### Key Designs

**1. Precise Equivalence to Sparse / Low-Rank Estimation: Replacing Non-convex Training with Mature Convex Problems**

This is the foundation of the paper, addressing the non-convexity of two-layer network ERM. For a **diagonal linear network** $f(x;W,a)=a^\top(w\odot x)/\sqrt d$ (where $W=\mathrm{diag}(w)$, linear activation, no bias), let the effective weights be $\theta_i=a_iw_i/\sqrt d$. The ERM with $\ell_2$ weight decay is exactly equivalent to a **LASSO** problem:

$$\hat\theta=\arg\min_{\theta}\tfrac12\sum_{\mu=1}^{n}\big(y_\mu-\theta^\top x_\mu\big)^2+\lambda\|\theta\|_1$$

While the expressivity of this network is the same as a linear model, the $a_iw_i$ **reparameterization generates implicit $\ell_1$ regularization**, leading to feature selection. For a **quadratic network** $f(x;W,a)=\mathrm{Tr}[S(xx^\top-I_d)/\sqrt d]$ (with $S=W^\top W/\sqrt{pd}$, quadratic activation, and $a$ fixed to all ones), the ERM is exactly equivalent to **low-rank matrix estimation / matrix compressed sensing with nuclear norm regularization**:

$$\hat S=\arg\min_{S\succeq0}\sum_{\mu=1}^{n}\big(y_\mu-\mathrm{Tr}[SZ_\mu]\big)^2+\lambda\|S\|_*,\quad Z_\mu=\tfrac{x_\mu x_\mu^\top-I_d}{\sqrt d}$$

With these equivalences, **AMP and its State Evolution (SE)** from LASSO / compressed sensing can be directly applied to provide precise predictions for excess risk and spectra—this is the fundamental reason Ours can perform end-to-end analysis where lazy theory cannot.

**2. Complete Excess Risk Phase Diagram: A Unified Expression Covering 8 Scaling Phases**

Addressing how feature learning changes scaling exponents, the authors provide piecewise precise rates for excess risk regarding $(n_{\rm eff},\lambda)$ when $n,d\gg1$, $p\ge d$, and noise $\Delta>0$ (Result 1, Eq. 11), visualized in a phase diagram (Figure 1). Representative phases include:

- **Fast Decay / Minimax Phase (Phase IV)**: Small regularization, $1\ll n_{\rm eff}\ll d$; $R=\Theta(n_{\rm eff}^{-1+1/(2\gamma)})$, matching the $\ell_q$-ball minimax rates of Raskutti et al.
- **Harmful Overfitting Phase (Phase V)**: As $n_{\rm eff}\to d$ under under-regularization, the estimator begins fitting noise. Risk is dominated by a non-universal scale $\rho(n_{\rm eff}/d)$ (where $\rho(t)=-1/\log t$ for diagonal and $\rho(t)=t^{2/5}$ for quadratic networks).
- **Interpolation Peak**: Risk reaches a maximum $R\sim\lambda^{-2/3}$ near $n_{\rm eff}\sim d$. This non-monotonicity at interpolation is a manifestation of **double descent**, which this work generalizes from linear to non-linear models.
- **Secondary Fast Decay (Phase VIa/VIb)**: For $n_{\rm eff}\gg d$, $R\propto d/n_{\rm eff}$.

The phase diagram contains several **rate-discontinuous phase boundaries** (red lines), such as the harmful overfitting $\leftrightarrow$ fast decay transition at $n_{\rm eff}=\Theta(d)$. As a corollary, the optimal regularization $\lambda_{\rm opt}=\tilde\Theta(\sqrt{n_{\rm eff}/d})$ avoids the harmful overfitting phase and achieves the **Bayes optimal rate** (Corollary 1). The key universality is that diagonal and quadratic models share the same phase diagram in terms of $n_{\rm eff}$.

**3. Characterization of Weight Spectra and "Soft-thresholding" Structure: Post-training Weights are Noisy Soft-thresholded Teacher Weights**

To address why weight spectra are distributed as they are, the authors use SE to give analytical forms of the post-training weight spectrum (Result 2). The core conclusion is that learned weights are **noisy + soft-thresholded** versions of teacher weights. For diagonal networks:

$$\hat\theta_i\sim\sigma_d(\theta_i^\star+\delta z_i;\,\epsilon),\quad z_i\sim\mathcal N(0,1)$$

where $\sigma_d(x;a)=\max(x-a,0)-\max(-x-a,0)$ is the soft-thresholding function. The constants have clear physical meanings: $\delta$ quantifies **noise intensity** from label noise and finite-sample estimation, and $\lambda\epsilon$ provides a **truncation threshold**—singular values below this are suppressed to zero by regularization. The quadratic network spectrum (Eq. 16) consists of a Dirac mass $\delta_0$ at zero, a bulk near zero, and a few outliers aligned with the teacher's top eigenvectors. This structure explains experienced phenomena like rank collapse (entire spectrum at zero), heavy-tails (perturbed versions of teacher weights), and bleed-out (smallest outliers merging into the bulk boundary).

**4. "Universal" Error Decomposition: Mapping Excess Risk to Underfitting / Overfitting / Approximation Terms via Spectral Features**

This step links risk and spectra explicitly (Result 3). Taking the under-regularized quadratic network as an example, the excess risk decomposes into (Eq. 17):

$$R_{n,d}=\underbrace{\delta^2\!\int(\cdots)\mu_{sc}+\tfrac1d\delta K'(\delta)(2\delta-\lambda\epsilon)^2}_{\text{Overfitting (Learned Noise)}}+\underbrace{\tfrac1d\!\!\sum_{i=K(\delta)+1}^{d}\!\!s_i^2}_{\text{Underfitting (Unlearned Features)}}+\underbrace{\tfrac1d\!\sum_{i=1}^{K(\delta)}(\cdots)}_{\text{Approximation Error of Learned Features}}$$

Each term has a spectral interpretation: **Overfitting = second moment of the bulk** (learned noise power, corresponding to the Wigner semicircle law $\mu_{sc}$); **Underfitting = power of spikes hidden below the bulk** (truncation point $K(\delta)$ determined by noise and regularization); **Approximation error = average error on outliers above the bulk**, depending on the signal-to-noise ratio $s_i/\delta$ and effective regularization $\lambda\epsilon$. This decomposition is called "universal" because it **does not depend on the teacher spectrum, sample size, or regularization**, holding for all $\Delta\ge0$ across all spectral phases.

This provides a first-principles explanation: **Bulk = learned noise, Spikes hidden below bulk = unlearned features, Outliers = learned features**. Optimal regularization aims to truncate the bulk (suppressing the overfitting term) while minimizing impact on the other two. Crucially, in the $d\ll n_{\rm eff}\ll d^2$ range, optimal performance occurs in Phase II ($\lambda=\sqrt{n_{\rm eff}/d}$), where the spectrum transitions from "outlier-dominated" to "heavy-tailed," theoretically **supporting the empirical claim by Martin et al. that "heavy-tailed spectra correlate with better generalization."**

### Loss & Training
No specialized loss was designed; the objective is the squared error ERM with $\ell_2$ weight decay described in Eq. (1). In numerical experiments, networks are trained to global minima using PyTorch + LBFGS (Appendix F). On the theoretical side, State Evolution equations from AMP are solved. A key technical claim of the paper is that while SE equations strictly hold only in the proportional asymptotic limit ($n_{\rm eff}/d$ fixed, $\lambda$ fixed), the authors **heuristically extrapolate them to arbitrary scales of $n,d,\lambda$**, and verify through extensive simulations that they remain accurate to within a constant factor far beyond the proven range.

## Key Experimental Results
This is a theoretical paper; "experiments" consist of numerical simulations to verify analytical predictions.

### Main Results: Risk Scaling Laws vs. State Evolution
Figure 3 compares actual excess risk from diagonal/quadratic network training (PyTorch+LBFGS, $d=100,200,400,800$) against non-asymptotic SE predictions (solid lines):

| Setting | Regularization $\lambda$ | Comparison | Conclusion |
|------|------|----------|------|
| Quadratic | $1/d,\,1,\,\sqrt d$ | Sims vs. SE Line | Excellent fit across entire $n_{\rm eff}$ range |
| Diagonal | $1/d,\,1,\,\sqrt d$ | Sims vs. SE Line | Similar agreement, matching the black rate lines in Result 1 |

Despite SE being strictly valid only in the asymptotic limit $n_{\rm eff}/d=\Theta(1)$, simulations show it remains accurate to within a constant factor well beyond that range.

### Spectral Verification
Figure 2 compares eigenvalue histograms (blue) of learned weights with theoretical bulk (purple) / spike (orange) predictions across phases like Ia / IV / VIa ($\lambda=1/d$) and Ib / II / III ($\lambda=\sqrt d$) for $d=800$ (Phase III at $d=400$). The theoretical curves accurately capture bulk shapes and outlier positions in each phase.

### Key Findings
- **Universality**: Diagonal and quadratic networks share the same phase diagram under the $n_{\rm eff}$ framework, including the same transition from benign to harmful overfitting.
- **Non-monotonic risk emerges from spectral evolution**: As samples increase, the bulk contracts and spikes "emerge," but the second moment of the bulk increases near interpolation, leading to non-monotonic risk (a spectral explanation for double descent).
- **Causal link between spectrum and generalization established**: Heavy-tailed spectra correlate with good generalization because they correspond to the transition from outlier-dominance to the heavy-tail regime found in the optimal regularization Phase II.
- **Robust extrapolation of SE**: AMP/SE has predictive power far beyond its proven asymptotic assumptions. The authors propose a broader conjecture that spin-glass theory tools may remain valid outside standard asymptotic limits.

## Highlights & Insights
- **The exact equivalence of "Network Training = Sparse/Low-rank Estimation"** is a sophisticated maneuver: it is an identity mapping, not an approximation, allowing thirty years of tools from LASSO / compressed sensing (AMP, SE, minimax bounds) to be applied to neural scaling laws.
- The **unified effective sample size $n_{\rm eff}=n$ or $n/d$** collapses two distinct architectures into one set of formulas, suggesting that scaling laws may be determined by the sparse/low-rank structure of the problem rather than specific architecture details.
- The **Spectral Trisection (bulk = noise / hidden spikes = unlearned features / outliers = learned features)** is highly transferable: it provides a first-principles basis for using weight spectra to diagnose network health, potentially theorizing spectrum diagnostics for large models.
- It **unifies disparate empirical phenomena**—double descent, benign/harmful overfitting, and heavy-tailed spectra—within a single phase diagram.

## Limitations & Future Work
- Only analyzes **global minima**, omitting GD/SGD training dynamics and compute scaling laws (explicitly noted as a future direction).
- Limited to **two-layer networks + quadratic/linear activation + isotropic Gaussian data**; deeper networks, general activations, and non-trivial covariance structures are not yet covered.
- Core results depend on the **heuristic extrapolation of SE** beyond proportional asymptotics. While numerically supported, it lacks rigorous proof; the authors list "rigorizing the SE conjecture" as future work.
- The teacher-student + power-law spectrum setup is idealized; there remains a gap between this and real-world data scaling laws. Whether universality extends beyond these two models remains an open question.

## Related Work & Insights
- **vs. Random Feature / Kernel Scaling Laws (Bahri, Maloney, Atanasov, Bordelon, etc.)**: These analyze scaling laws in the lazy regime with fixed features, where problems reduce to kernel methods. Ours moves beyond lazy regimes to feature learning, explaining weight spectrum evolution and the spectrum-gen relationship.
- **vs. Ben Arous et al. (2025)**: Also study quadratic networks but focus on specific SGD dynamics without noise or regularization, recovering only one of the many scaling exponents provided here. Ours systematically covers the entire phase diagram for any $\lambda>0, \Delta\ge0$.
- **vs. Ren et al. (2025)**: Study activations with large information exponents; their direction is orthogonal to this work.
- **vs. Classic LASSO/Matrix Sensing (Raskutti & Wainwright, Negahban & Wainwright)**: Ours not only reproduces their minimax rates at specific $\lambda$ but extends conclusions to all regularization strengths and data scales, revealing the transition between minimax rates and faster $\Theta(d/n_{\rm eff})$ rates.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to provide a complete two-layer scaling law phase diagram in the feature learning regime and unify risk with weight spectra from first principles.
- Experimental Thoroughness: ⭐⭐⭐⭐ Theoretical paper; numerical simulations cover multiple phases, dimensions, risk, and spectra with high accuracy, though limited to two toy architectures.
- Writing Quality: ⭐⭐⭐⭐ Clear logical chain (Equivalence → Phase Diagram → Spectra → Decomposition), though phase diagrams and formulas are dense with a high barrier to entry.
- Value: ⭐⭐⭐⭐⭐ Provides a rare analytical foundation for neural scaling laws and the "heavy-tail $\leftrightarrow$ generalization" link, informing both theory and spectral diagnostic practices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Implicit bias produces neural scaling laws in learning curves, from perceptrons to deep networks](implicit_bias_produces_neural_scaling_laws_in_learning_curves_from_perceptrons_t.md)
- [\[ICLR 2026\] Mitigating the Curse of Detail: Scaling Arguments for Feature Learning and Sample Complexity](mitigating_the_curse_of_detail_scaling_arguments_for_feature_learning_and_sample.md)
- [\[ICLR 2026\] Transfer Learning in Infinite Width Feature Learning Networks](transfer_learning_in_infinite_width_feature_learning_networks.md)
- [\[ICLR 2026\] Theory of Scaling Laws for In-Context Regression: Depth, Width, Context and Time](theory_of_scaling_laws_for_in-context_regression_depth_width_context_and_time.md)
- [\[ICLR 2026\] Feature Compression is the Root Cause of Adversarial Fragility in Neural Networks](feature_compression_is_the_root_cause_of_adversarial_fragility_in_neural_network.md)

</div>

<!-- RELATED:END -->
