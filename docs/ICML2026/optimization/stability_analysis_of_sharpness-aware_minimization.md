---
title: >-
  [Paper Note] Stability Analysis of Sharpness-Aware Minimization
description: >-
  [ICML 2026][Optimization][SAM] This paper analyzes the convergence instability of SAM near saddle points from a dynamical systems perspective. It first proves under deterministic gradient flow that as long as the neighbo…
tags:
  - "ICML 2026"
  - "Optimization"
  - "SAM"
  - "Saddle Point Escape"
  - "Dynamical Systems"
  - "Diffusion Equations"
  - "momentum and batch size"
date: 2026-05-08
content_hash: b155e0ac8ecd88dc
---

# Stability Analysis of Sharpness-Aware Minimization

**Conference**: ICML 2026  
**arXiv**: [2301.06308](https://arxiv.org/abs/2301.06308)  
**Code**: None  
**Area**: Optimization / Training Dynamics  
**Keywords**: SAM, Saddle Point Escape, Dynamical Systems, Diffusion Equations, momentum and batch size

## TL;DR
This paper analyzes the convergence instability of SAM near saddle points from a dynamical systems perspective. It first proves under deterministic gradient flow that as long as the neighborhood radius $\rho > -1/\lambda_1$, a saddle point becomes an attractor for SAM. Subsequently, within a stochastic diffusion framework, it demonstrates that the mean square displacement for SAM's saddle point escape is $2\eta t^2|\lambda_j|^3\rho/B$ smaller than that of SGD. Finally, the SAM diffusion formula is utilized to explain why momentum and batch size are the true underlying contributors to SAM’s SOTA generalization performance.

## Background & Motivation
**Background**: The "flat minima" training framework, represented by Sharpness-Aware Minimization (SAM) proposed by Foret et al. in 2020, has become a standard configuration for improving generalization in various fields such as CIFAR, ImageNet, ViT, and NLP. The core of SAM is to first take a small step along the current gradient direction to obtain an adversarial perturbation weight $\bm{w}^p = \bm{w} + \rho \nabla \ell(\bm{w})$, and then use the gradient at the perturbed point to update the original parameters $\bm{w}_{t+1} = \bm{w}_t - \eta \nabla \ell(\bm{w}_t^p)$. This forces the optimizer to find solutions where the worst-case loss within a neighborhood is small, thereby tending towards flatter minima.

**Limitations of Prior Work**: The authors ran both SAM and vanilla GD on the Beale function (a classic optimization test function); GD successfully converged to the global minimum, while SAM became stuck at a saddle point. Kaddour et al. (2022) also reported abnormal performance of SAM under certain settings. This "saddle point trapping" phenomenon is more costly in the highly nonlinear loss landscapes of deep learning—where the number of saddle points far exceeds minima. If SAM is truly captured by them, its effectiveness in large-scale experiments requires a new explanation.

**Key Challenge**: The geometric motivation of sharpness encourages SAM to move toward directions where the "loss is small throughout the neighborhood," but this precisely conflicts with the need to "rapidly escape saddle points along the unstable manifold." Near a saddle point, the loss fluctuations within the neighborhood are small and the worst-case loss is not large, so SAM perceives it as "flat" and stops; whereas GD, looking only at the current gradient, is carried away by tiny perturbations around the saddle point.

**Goal**: (1) Find the precise conditions under which SAM treats saddle points as attractors under deterministic dynamics; (2) Quantitatively compare the saddle point escape speeds of SAM and SGD under stochastic diffusion; (3) Explain why momentum and batch size can mitigate this instability, thus becoming hidden keys to SAM's success.

**Key Insight**: Using the qualitative theory of dynamical systems, the trajectories of SAM near saddle points are decomposed into three categories (Case-I/II/III), and the Lambda Lemma is used to argue that Case-III inevitably oscillates between two basins of attraction. Subsequently, switching to the Fokker-Planck framework, the diffusion tensor of SAM is derived using a Fisher Information Matrix approximation, turning the saddle point escape problem into a mean square displacement problem.

**Core Idea**: The perturbation $\bm{w}^p$ may fall into the basin of attraction of an adjacent minimum, causing SAM's update direction to reverse repeatedly near the saddle point. When $\rho|\lambda_j|$ is sufficiently large, the original escape force $\lambda_j$ along the negative Hessian eigenvalue direction is counteracted by $\rho\lambda_j^2$, and the saddle point geometrically transforms from a "hyperbolic unstable point" into a "stable attractor."

## Method

### Overall Architecture
This paper does not propose a new algorithm but provides a comprehensive "stability pathology report" for SAM. The paper progresses in four steps: (a) Using Lambda Lemma + three-case geometric analysis to explain why gradient oscillations occur near saddle points; (b) Providing analytical conditions for saddle points becoming attractors via Hessian eigenvalues (Theorem 1); (c) Deriving the SAM diffusion formula using the Fokker-Planck equation and proving slower escape (Theorem 2 + Corollary 1); (d) Incorporating momentum and batch size into the diffusion formula to provide precise quantification of why increasing momentum and decreasing batch size "rescue" SAM (Theorem 3). Finally, experiments on CIFAR-10/100 serve as empirical closure for this theory.

### Key Designs

1.  **Lambda Lemma Three-Case Geometric Analysis + Saddle Point Attractor Condition (Theorem 1)**:

    - **Function**: Provides the geometric mechanism and closed-form conditions for SAM being trapped by saddle points under deterministic gradient flow.
    - **Mechanism**: Consider an index-1 saddle point $\bm{d}$ between gradient flow $d\bm{w}/dt = -\nabla \ell(\bm{w})$ and two adjacent minima $\bm{s}_1, \bm{s}_2$. When $\bm{w}_t \in A(\bm{s}_1)$, three cases are distinguished by distance to $\bm{d}$: Case-I is far from $\bm{d}$ and the stable manifold $W^s(\bm{d})$, where $-\nabla\ell(\bm{w}_t^p) \sim -\nabla\ell(\bm{w}_t)$, making SAM behave like GD; Case-II is near $\bm{d}$ but within $A(\bm{s}_1)$, following the Lambda Lemma along the unstable manifold $W^u(\bm{d})$; Case-III falls into the $\rho$-neighborhood of $\bm{d}$, where the perturbation $\bm{w}_t^p$ may cross the boundary into $A(\bm{s}_2)$, thus $-\nabla\ell(\bm{w}_t^p)$ points to $\bm{s}_2$, causing $\bm{w}_{t+1}$ to return to $A(\bm{s}_2)$ and then back to $A(\bm{s}_1)$ in the next step, forming oscillations along $W^u(\bm{d})$. Theorem 1 gives the closed-form condition: for an index-1 saddle point $\bm{d}$ with a negative Hessian eigenvalue $\lambda_1<0$, as long as $\rho > -1/\lambda_1$ (equivalently $\lambda_1 + \rho \lambda_1^2 > 0$), the saddle point is upgraded to an attractor under SAM dynamics.
    - **Design Motivation**: Traditional stability analysis classifies saddle points with negative Hessian eigenvalues as "unstable." This paper rewrites the dynamical system with SAM's perturbation $\rho \nabla\ell(\bm{w})$ and finds that the Hessian term is modified to $\Lambda + \rho \Lambda^2$. This quadratic term $\rho \lambda^2$ is always positive and can reverse the original negative eigenvalue, geometrically turning the saddle point from a "hyperbolic point" into a "sink." Numerical experiments on the Beale function and $f(x,y)=x^2-y^2$ confirm the simultaneous occurrence of Case-III and Theorem 1.

2.  **Fokker-Planck Derivation of SAM Diffusion and Escape Speed Comparison (Theorem 2 + Corollary 1)**:

    - **Function**: Mathematizes the saddle point escape problem to prove that SAM escapes saddle points slower than SGD under mini-batch noise, and larger $\rho$ exacerbates this delay.
    - **Mechanism**: Write SAM as a stochastic differential equation $d\bm{w} = -\nabla\ell(\bm{w}^p)dt + [\eta C(\bm{w}^p)]^{1/2} dW_t$, approximate the noise covariance $C(\bm{w})$ as $\frac{1}{B}[H(\bm{w})]^+$ using the Fisher Information Matrix, and perform a second-order Taylor expansion to rewrite the loss as a local quadratic form. Solving the Fokker-Planck equation yields $\bm{w} \sim \mathcal{N}(\bm{d}, Q\,\mathrm{diag}(\bm{\sigma}^2(t))Q^T)$, with variance in each eigen-direction $\sigma_j^2(t) = \frac{\eta|\lambda_j|}{2B \lambda_j (1+\rho\lambda_j)^2}\left[1 - \exp(-2\lambda_j(1+\rho\lambda_j)^2 t)\right]$. Subtracting SGD (where $\rho=0$) from SAM and expanding under a small time window $|\lambda_j|t \ll 1$ yields $\Delta_{SGD} - \Delta_{SAM} = 2\eta t^2 |\lambda_j|^3 \rho / B + \mathcal{O}(B^{-1}\eta t^3 \lambda_j^4)$, where the difference is strictly positive and grows linearly with $\rho$.
    - **Design Motivation**: Deterministic analysis alone cannot explain why SAM works in large-scale deep learning, as training is inherently noisy. The paper clarifies how SAM's perturbation interacts with mini-batch noise—the conclusion is that $\rho$ suppresses stochastic diffusion by enlarging the $(1+\rho\lambda_j)^2$ term in the denominator. The term $\rho|\lambda_j|^3/B$ is a clean dimensional formula: larger $\rho$, sharper Hessian (large $|\lambda_j|$), and larger batch (large $B$) all widen the escape gap between SAM and SGD.

3.  **SAM Diffusion with Momentum + Batch Size Formula (Theorem 3)**:

    - **Function**: Simultaneously incorporates momentum $\gamma$ and batch size $B$ into the SAM diffusion formula, quantifying how they rescue saddle point escape.
    - **Mechanism**: Based on the previous model and adding momentum, the mean square displacement becomes $\Delta_{SAM} = C_1 \frac{(1-e^{-C_2(1-\gamma)})^2}{(1-\gamma)^3 B} + C_3 \frac{(1-e^{-C_4/(1-\gamma)})}{(1-\gamma)B}$, where $C_1=\eta^2|\lambda_j|/2$, $C_2=\eta/t$, $C_3=\eta|\lambda_j|/[2\lambda_j(1+\rho\lambda_j)^2]$, and $C_4=2\lambda_j(1+\rho\lambda_j)^2 t$. From the formula, as $\gamma\to 1^-$, the power of $(1-\gamma)$ in the denominator is highest (reaching 3), causing $\Delta_{SAM}$ to increase; since $B$ appears in the first power in the denominator, reducing batch size also accelerates escape. More importantly, $\rho$ causes the $C_3$ term to shrink, so to maintain the same diffusion as SGD, a larger $\gamma$ must be used to compensate.
    - **Design Motivation**: In practice, momentum and batch size are often treated as "engineering hyperparameters." This paper argues they are actually hidden pillars for SAM's success. On CIFAR-10 with ResNet-18 (with BN/augmentation disabled), tuning only $\gamma$ and $B$ reveals: (a) At $B=512$, SAM training loss sticks above 1 and test acc < 60%; (b) $\gamma=0.9$ lifts SAM's acc by 20%+, while SGD only gains 5%. Table 1 further shows $\rho=0.1, \gamma=0.95$ is the optimal combination, validating the prediction that larger $\rho$ requires larger $\gamma$.

### Loss & Training
This paper does not introduce new loss functions; the experimental losses are standard Cross-Entropy (CIFAR-10/100) or Mean Squared Error (toy neural networks). All theoretical derivations are based on (1) second-order Taylor expansion around the saddle point $\bm{d}$, and (2) the Fisher Information Matrix approximation $\frac{1}{N}\sum_i \nabla\ell_i \nabla\ell_i^T \approx [H]^+$. Complete proofs are provided in Appendix A.

## Key Experimental Results

### Main Results (Toy and Neural Network Validation)

| Experiment | Setting | GD/SGD Results | SAM Results | Interpretation |
|------|------|-------------|----------|------|
| Beale Function | $\eta=10^{-4}$, saddle point at $(0,1)$ | Converges to global minimum | Stuck at saddle point $(0,1)$ | Directly validates Case-III and Theorem 1 |
| $f(x,y)=x^2-y^2$ | Initial point $(-3,-\epsilon), \epsilon=0.01$ | Crosses saddle and converges | Attracted to saddle point | Hessian eigenvalues $\{2,-2\}$ satisfy $\lambda+\rho\lambda^2 > 0$ |
| Toy NN (Ziyin et al.) | Two layers, one neuron, $\varphi(x)=x^2$ | Most seeds converge to global minimum | Most seeds remain in saddle region | Validates Theorem 2 under stochastic diffusion |
| Increasing $\rho$ | Same toy NN | — | Average loss increases monotonically | Consistent with Corollary 1: larger $\rho$ makes escape harder |

### Ablation Study (CIFAR-10/100, ResNet-18, BN + data augmentation disabled)

| Configuration | CIFAR-10 Test Results | Description |
|------|-------------------|------|
| SAM, $B=512$, $\gamma=0$ | train loss > 1, test acc < 60% | Failure with large batch + no momentum |
| SAM, decrease $B$ | Train loss decreases, test acc significantly rises | Consistent with $\Delta_{SAM} \propto 1/B$ in Theorem 3 |
| SAM, increase $\gamma$ | Train loss decreases, test acc significantly rises | Consistent with $\Delta_{SAM}$ increasing with the reciprocal of $(1-\gamma)$ in Theorem 3 |
| SGD vs SAM, $\gamma=0 \to 0.9$ | SGD +5%, SAM +20%+ | Marginal gain from momentum is much larger for SAM than SGD |
| SAM, $\rho=0.1, \gamma=0.95$ (BN+aug enabled) | 95.08% (Best) | $\rho=0.5, \gamma=0$ is only 86.20%, Max-Min gap 5.79 |

### Key Findings
- SAM saddle point trapping is not a edge case: It is reproducible in Beale, $x^2-y^2$, toy NNs, and CIFAR, and this paper provides a precise condition $\rho > -1/\lambda_1$ for its occurrence.
- The saddle point escape gap $\Delta_{SGD}-\Delta_{SAM} \propto \rho|\lambda_j|^3/B$ is a clean dimensional formula, explaining why the combination of large $\rho$ and large $B$ is most likely to fail for SAM.
- In practice, momentum is almost a necessary condition for SAM to work: $\gamma=0.9$ causes a 20%+ jump in acc for SAM on CIFAR-10, providing the first theoretical explanation for why "empirically tuning momentum is effective."
- Larger $\rho$ requires larger $\gamma$ to maintain the same diffusion as SGD, providing a clear direction for joint hyperparameter search.
- On the Beale function, $\cos(\nabla\ell(\bm{w}_t), \nabla\ell(\bm{w}_t^p))$ continues to oscillate between $-1$ and $1$ after converging to the saddle point, providing direct visual evidence for the geometric description of Case-III-(ii).
- In Table 1, the Max-Min gap reaches 5.79% at $\rho=0.5$, indicating that more aggressive SAM is more sensitive to momentum selection, a $\rho$-$\gamma$ coupling that perfectly aligns with the predictions of Theorem 3.

## Highlights & Insights
- The paper uses Lambda Lemma and three cases to move the intuition of "perturbation crossing boundaries" into a geometrically rigorous Case-III. Coupled with the oscillation of $\cos$ on the Beale function, the argument is very clean.
- Theorem 1's $\rho > -1/\lambda_1$ is an unexpectedly simple critical condition, implying that choosing a large $\rho$ will make all saddle points with $|\lambda_1| > 1/\rho$ attractive, providing direct targets for improvements like "adaptive/layer-normalized $\rho$."
- Incorporating momentum into the diffusion formula and yielding a strong dependency like $(1-\gamma)^{-3}$ provides the first theoretical explanation for why momentum has consistently defaulted to 0.9 in SAM papers.
- The closed-form difference $\Delta_{SGD}-\Delta_{SAM} = 2\eta t^2|\lambda_j|^3\rho/B$ is transferable and can be applied directly to any SAM variant featuring "neighborhood perturbation + single gradient" for stability analysis.

## Limitations & Future Work
- The entire theory is built on second-order Taylor expansion around the saddle point $\bm{d}$ and Fisher Information approximations; global dynamics outside the local neighborhood can only be extrapolated through experiments.
- It is assumed that $\bm{w}^p$ is computed using an exact first-order approximation, without handling normalization or second-order corrections in variants like ASAM, GSAM, or Lookahead-SAM.
- Experiments were primarily conducted on medium scales such as CIFAR-10/100 + ResNet-18, lacking validation on ImageNet or LLM training. Comparisons with complementary theories like Bartlett et al. 2023 ("bouncing across ravines") or Chen et al. 2023 ("transient saddle attraction is beneficial") were only briefly mentioned in Limitations.
- While the paper proves SAM escapes saddle points slowly, it does not provide a falsifiable boundary for settings where SAM actually improves generalization *because* of its stay near saddle points—the perspective of Chen et al. 2023.
- The critical condition $\rho > -1/\lambda_1$ depends on accurate local Hessian information; accurately estimating the Hessian spectrum in practical deep networks is another challenge, so "how to determine online that a dangerous $\rho$ zone has been entered" remains open.
- The article does not provide an automated $\rho$/$\gamma$/$B$ collaborative scheduling algorithm, leaving the engineering gap between "diagnosis" and "treatment" for future work.

## Related Work & Insights
- **vs Compagnoni et al. 2023**: They treat noise as an implicit smoothing term using Lipschitz assumptions and small learning rates; this paper treats perturbations as random excitations driving trajectories away from unstable attractors using local dynamical analysis, which is closer to the "large Hessian" context of edge-of-stability.
- **vs Andriushchenko & Flammarion 2022**: They empirically found that small batch sizes significantly boost SAM performance; this paper provides the analytical form $\Delta_{SAM} \propto 1/B$ in Theorem 3, elevating empirical observation to mechanistic explanation.
- **vs Kaddour et al. 2022**: They were the first to report abnormal behavior of SAM near saddle points; this paper provides the first systematic theoretical characterization of this phenomenon (Theorem 1 + Lambda Lemma geometry).
- **vs Bartlett et al. 2023 / Chen et al. 2023**: The former characterizes SAM's bouncing behavior along ravines from the perspective of sharpness, while the latter argues that transient saddle point stay helps generalization; this work complements them by pointing out that "permanent entrapment in saddle points" is a failure mode requiring momentum/batch size to balance exploration and escape.
- **vs Long & Bartlett 2024 (edge-of-stability)**: EoS focuses on the behavior of the maximum Hessian eigenvalue drifting near $2/\eta$ during training; the critical condition $\rho > -1/\lambda_1$ in Theorem 1 is structurally complementary to EoS and can be seen as the SAM version of a "critical edge."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Adaptive Sharpness-Aware Minimization with a Polyak-type Step size: A Theory-Grounded Scheduler](adaptive_sharpness-aware_minimization_with_a_polyak-type_step_size_a_theory-grou.md)
- [\[ICLR 2026\] Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization](../../ICLR2026/optimization/minor_first_major_last_a_depth-induced_implicit_bias_of_sharpness-aware_minimiza.md)
- [\[NeurIPS 2025\] A Unified Stability Analysis of SAM vs SGD: Role of Data Coherence and Emergence of Simplicity Bias](../../NeurIPS2025/optimization/a_unified_stability_analysis_of_sam_vs_sgd_role_of_data_cohe.md)
- [\[NeurIPS 2025\] Unveiling the Power of Multiple Gossip Steps: A Stability-Based Generalization Analysis in Decentralized Training](../../NeurIPS2025/optimization/unveiling_the_power_of_multiple_gossip_steps_a_stability-based_generalization_an.md)
- [\[NeurIPS 2025\] Unveiling m-Sharpness Through the Structure of Stochastic Gradient Noise](../../NeurIPS2025/optimization/unveiling_m-sharpness_through_the_structure_of_stochastic_gradient_noise.md)

</div>

<!-- RELATED:END -->
