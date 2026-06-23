---
title: >-
  [Paper Note] Stability Analysis of Sharpness-Aware Minimization
description: >-
  [ICML 2026][Optimization & Theory][SAM] This paper analyzes the convergence instability of SAM near saddle points from a dynamical systems perspective. It first proves under deterministic gradient flow that a saddle point becomes an attractor for SAM as long as the neighborhood radius $\rho > -1/\lambda_1$. Subsequently, within a stochastic diffusion framewo
tags:
  - ICML 2026
  - Optimization & Theory
  - SAM
date: 2026-05-08
content_hash: 4fa29ec69a1efa2e
---
# Stability Analysis of Sharpness-Aware Minimization

**Conference**: ICML 2026  
**arXiv**: [2301.06308](https://arxiv.org/abs/2301.06308)  
**Code**: None  
**Area**: Optimization / Training Dynamics  
**Keywords**: SAM, saddle point escape, dynamical systems, diffusion equations, momentum and batch size

## TL;DR
This paper analyzes the convergence instability of SAM near saddle points from a dynamical systems perspective. It first proves under deterministic gradient flow that a saddle point becomes an attractor for SAM as long as the neighborhood radius $\rho > -1/\lambda_1$. Subsequently, within a stochastic diffusion framework, it demonstrates that the mean square displacement for saddle point escape in SAM is smaller than that of SGD by $2\eta t^2|\lambda_j|^3\rho/B$. Finally, the SAM diffusion formula is utilized to explain why momentum and batch size are the true hidden drivers behind SAM achieving SOTA generalization performance.

## Background & Motivation
**Background**: The "flat minima" training framework, represented by Sharpness-Aware Minimization (SAM) proposed by Foret et al. in 2020, has become a standard for improving generalization performance across various fields such as CIFAR, ImageNet, ViT, and NLP. The core of SAM involves taking a small step along the current gradient direction to obtain an adversarial perturbation weight $\bm{w}^p = \bm{w} + \rho \nabla \ell(\bm{w})$, and then updating the original parameters using the gradient at the perturbed point $\bm{w}_{t+1} = \bm{w}_t - \eta \nabla \ell(\bm{w}_t^p)$. This forces the optimizer to find a solution where the worst-case loss within a neighborhood is also small, thereby tending towards flat minima.

**Limitations of Prior Work**: The authors applied SAM and vanilla GD simultaneously to the Beale function (a classic optimization test function), where GD successfully converged to the global minimum, while SAM became stuck at a saddle point. Kaddour et al. (2022) also reported anomalous SAM behavior in certain settings. This "saddle point trapping" phenomenon is more costly in the highly nonlinear loss landscapes of deep learning—the number of saddle points in network training far exceeds that of minima. If SAM is truly captured by them, its continued success in large-scale experiments requires a new explanation.

**Key Challenge**: The geometric motivation of sharpness encourages SAM to move towards directions where the "loss is small throughout the neighborhood," but this precisely conflicts with the requirement to "rapidly escape saddle points along the unstable manifold." Near a saddle point, the loss fluctuations within the neighborhood are small and the worst-case loss is not large, leading SAM to perceive the region as "flat" and stop; conversely, GD, looking only at the current point gradient, is carried out by the tiny perturbations around the saddle point.

**Goal**: (1) Identify the precise conditions under which SAM treats a saddle point as an attractor under deterministic dynamics; (2) Quantitatively compare the escape speeds of SAM and SGD from saddle points under stochastic diffusion; (3) Explain why momentum and batch size can alleviate this instability, thereby becoming the hidden keys to SAM's success.

**Key Insight**: Using the qualitative theory of dynamical systems, the trajectory of SAM near a saddle point is decomposed into three categories: Case-I/II/III. Then, the Lambda Lemma is used to demonstrate that Case-III inevitably oscillates between two basins of attraction. Subsequently, switching to the Fokker-Planck framework, the SAM diffusion tensor is derived using the Fisher information matrix approximation, transforming the saddle point escape problem into a mean square displacement problem.

**Core Idea**: The perturbation $\bm{w}^p$ might fall into the basin of attraction of an adjacent minimum, causing the SAM update direction to repeatedly flip near the saddle point. When $\rho|\lambda_j|$ is sufficiently large, the original escape force along the negative Hessian eigenvalue direction $\lambda_j$ is overridden by $\rho\lambda_j^2$ in the opposite direction, and the saddle point geometrically transforms from a "hyperbolic unstable point" into a "stable attractor."

## Method

### Overall Architecture
This paper does not propose a new algorithm but rather provides a comprehensive "pathology report" on SAM stability. The research progresses in four steps: (a) Using Lambda Lemma + three-case geometric analysis to explain why gradient oscillations occur around saddle points in SAM; (b) Providing analytical conditions for a saddle point becoming an attractor using Hessian eigenvalues (Theorem 1); (c) Deriving the SAM diffusion formula using the Fokker-Planck equation to prove slower escape (Theorem 2 + Corollary 1); (d) Incorporating momentum and batch size into the diffusion formula to provide precise quantification of why "increasing momentum and decreasing batch size" can rescue SAM (Theorem 3). Finally, experiments on CIFAR-10/100 serve as empirical validation of this theory.

### Key Designs

**1. Lambda Lemma Three-Case Geometric Analysis + Saddle Point Attractor Condition (Theorem 1): Providing the closed-form critical value for SAM saddle point trapping**

To explain why SAM remains stationary at saddle points, its trajectory near the saddle point must first be clarified. Consider an index-1 saddle point $\bm{d}$ between gradient flow $d\bm{w}/dt = -\nabla \ell(\bm{w})$ and two adjacent minima $\bm{s}_1, \bm{s}_2$. Trajectories are classified into three types based on the distance of $\bm{w}_t$ from $\bm{d}$: Case-I is far from $\bm{d}$ and the stable manifold $W^s(\bm{d})$, where $-\nabla\ell(\bm{w}_t^p) \sim -\nabla\ell(\bm{w}_t)$ and SAM behaves like GD; Case-II is near $\bm{d}$ but remains within $A(\bm{s}_1)$, where the trajectory follows the unstable manifold $W^u(\bm{d})$ according to the Lambda Lemma; Case-III is the pathology—once falling into the $\rho$-neighborhood of $\bm{d}$, the perturbation $\bm{w}_t^p$ may cross the attraction basin boundary into $A(\bm{s}_2)$, causing $-\nabla\ell(\bm{w}_t^p)$ to point toward $\bm{s}_2$ and then be pulled back to $A(\bm{s}_1)$ in the next step, forming oscillations along $W^u(\bm{d})$. The key to quantifying this oscillation is that the SAM perturbation $\rho\nabla\ell$ rewrites the Hessian term of the dynamical system from $\Lambda$ to $\Lambda + \rho\Lambda^2$. Since the quadratic term $\rho\lambda^2$ is always positive, it can reverse the original negative eigenvalue. Thus, Theorem 1 provides a clean critical condition: for an index-1 saddle point with a negative eigenvalue $\lambda_1<0$, as long as $\rho > -1/\lambda_1$ (equivalent to $\lambda_1 + \rho\lambda_1^2 > 0$), the saddle point geometrically upgrades from a "hyperbolic unstable point" to an attractor under SAM dynamics. Numerical experiments on the Beale function and $f(x,y)=x^2-y^2$ replicate both Case-III and this condition.

**2. Fokker-Planck Derivation of SAM Diffusion and Escape Speed Comparison (Theorem 2 + Corollary 1): Proving SAM escapes saddle points slower than SGD**

Deterministic analysis cannot explain why SAM still works in real (noisy SGD) training, so the "interaction between perturbation and mini-batch noise" must be incorporated into a stochastic framework. Expressing SAM as an SDE $d\bm{w} = -\nabla\ell(\bm{w}^p)dt + [\eta C(\bm{w}^p)]^{1/2} dW_t$, the noise covariance is approximated using the Fisher information matrix as $C(\bm{w})\approx \frac{1}{B}[H(\bm{w})]^+$ and the loss is expanded via second-order Taylor series into a local quadratic form. Substituting into the Fokker-Planck equation yields $\bm{w} \sim \mathcal{N}(\bm{d}, Q\,\mathrm{diag}(\bm{\sigma}^2(t))Q^T)$, where the variance in each eigen-direction is:

$$\sigma_j^2(t) = \frac{\eta|\lambda_j|}{2B \lambda_j (1+\rho\lambda_j)^2}\Big[1 - \exp\big(-2\lambda_j(1+\rho\lambda_j)^2 t\big)\Big]$$

Subtracting the mean square displacement of SAM from that of SGD (i.e., $\rho=0$) and expanding under a small time window $|\lambda_j|t \ll 1$ gives $\Delta_{SGD} - \Delta_{SAM} = 2\eta t^2 |\lambda_j|^3 \rho / B + \mathcal{O}(B^{-1}\eta t^3 \lambda_j^4)$. This difference is always positive and grows linearly with $\rho$. The mechanism is clear: the perturbation $\rho$ increases the $(1+\rho\lambda_j)^2$ term in the denominator, suppressing stochastic diffusion and thus weakening the escape capability of the noise itself. The dimension formula $\rho|\lambda_j|^3/B$ cleanly illustrates that a larger $\rho$, a sharper Hessian, and a larger batch size will all widen the escape gap between SAM and SGD.

**3. SAM Diffusion Formula with Momentum + Batch Size (Theorem 3): Quantifying why two "engineering hyperparameters" are hidden pillars**

Finally, momentum $\gamma$ and batch size $B$ are incorporated into the diffusion formula to observe how they rescue saddle point escape. With the inclusion of the momentum term, the mean square displacement becomes:

$$\Delta_{SAM} = C_1 \frac{(1-e^{-C_2(1-\gamma)})^2}{(1-\gamma)^3 B} + C_3 \frac{(1-e^{-C_4/(1-\gamma)})}{(1-\gamma)B}$$

where $C_1=\eta^2|\lambda_j|/2$, $C_2=\eta/t$, $C_3=\eta|\lambda_j|/[2\lambda_j(1+\rho\lambda_j)^2]$, and $C_4=2\lambda_j(1+\rho\lambda_j)^2 t$. Since $(1-\gamma)$ appears in the denominator up to the 3rd power, $\gamma\to 1^-$ significantly increases $\Delta_{SAM}$ (accelerating escape). As $B$ is only in the first power of the denominator, reducing the batch size also accelerates escape. More importantly, $\rho$ reduces the $C_3$ term, meaning that the larger the $\rho$ used by SAM, the larger the $\gamma$ required to compensate for the diffusion to match SGD levels. This repositions momentum and batch size from "default or roughly tuned engineering hyperparameters" to hidden pillars of SAM's success. Verification on CIFAR-10 + ResNet-18 (with BN/aug disabled) showed: at $B=512$, SAM training loss stayed above 1 and test acc < 60%; $\gamma=0.9$ raised SAM acc by 20%+, whereas SGD only increased by 5%, and $\rho=0.1, \gamma=0.95$ was the optimal combination, confirming the prediction that "larger $\rho$ requires more $\gamma$ compensation."

### Loss & Training
This paper does not introduce a new loss function; the experimental losses used are standard Cross-Entropy (CIFAR-10/100) or Mean Squared Error (toy neural networks). All theoretical derivations are based on two approximations: (1) Second-order Taylor expansion around the saddle point $\bm{d}$, and (2) Approximation of the Fisher information matrix as $\frac{1}{N}\sum_i \nabla\ell_i \nabla\ell_i^T \approx [H]^+$. Complete proofs are provided in Appendix A.

## Key Experimental Results

### Main Results (Toy and Neural Network Validation)

| Experiment | Setting | GD/SGD Results | SAM Results | Interpretation |
|------|------|-------------|----------|------|
| Beale Function | $\eta=10^{-4}$, single saddle point at $(0,1)$ | Converged to global minimum | Stuck at saddle point $(0,1)$ | Direct validation of Case-III and Theorem 1 |
| $f(x,y)=x^2-y^2$ | Initial point $(-3,-\epsilon), \epsilon=0.01$ | Escaped saddle point and converged | Attracted to saddle point | Hessian eigenvalues $\{2,-2\}$ satisfy $\lambda+\rho\lambda^2 > 0$ |
| Toy NN (Ziyin et al.) | Two layers, one neuron, $\varphi(x)=x^2$ | Most seeds converged to global minimum | Most seeds stalled in saddle region | Validates Theorem 2 under stochastic diffusion |
| Increasing $\rho$ | Same toy NN as above | — | Average loss increased monotonically | Consistent with Corollary 1: larger $\rho$ makes escape harder |

### Ablation Study (CIFAR-10/100, ResNet-18, BN + data augmentation disabled)

| Configuration | CIFAR-10 Test Results | Description |
|------|-------------------|------|
| SAM, $B=512$, $\gamma=0$ | train loss > 1, test acc < 60% | Large batch + no momentum leads to failure |
| SAM, decreasing $B$ | Training loss decreased, test acc increased significantly | Consistent with $\Delta_{SAM} \propto 1/B$ in Theorem 3 |
| SAM, increasing $\gamma$ | Training loss decreased, test acc increased significantly | Consistent with $\Delta_{SAM}$ rising with reciprocal of $(1-\gamma)$ in Theorem 3 |
| SGD vs SAM, $\gamma=0 \to 0.9$ | SGD +5%, SAM +20%+ | Marginal gain from momentum is much larger for SAM than SGD |
| SAM, $\rho=0.1, \gamma=0.95$ (BN+aug enabled) | 95.08% (Best) | $\rho=0.5, \gamma=0$ was only 86.20%, Max-Min gap 5.79 |

### Key Findings
- SAM trapping at saddle points is not an edge case: it can be replicated in Beale, $x^2-y^2$, toy NN, and CIFAR; this paper provides the precise condition $\rho > -1/\lambda_1$.
- The saddle point escape gap $\Delta_{SGD}-\Delta_{SAM} \propto \rho|\lambda_j|^3/B$ is a clean dimension formula, explaining why the combination of large $\rho$ and large $B$ is most likely to fail in SAM.
- In practice, momentum is almost a necessary condition for SAM to work: $\gamma=0.9$ causes a 20%+ jump in SAM accuracy on CIFAR-10, providing the first theoretical explanation for the empirical effectiveness of tuning momentum.
- Larger $\rho$ requires larger $\gamma$ to maintain diffusion equivalent to SGD, providing a clear direction for joint hyperparameter search.
- On the Beale function, $\cos(\nabla\ell(\bm{w}_t), \nabla\ell(\bm{w}_t^p))$ oscillates between $-1$ and $1$ after converging to the saddle point, providing direct visual evidence for the geometric description of Case-III-(ii).
- In Table 1, the Max-Min gap reaches 5.79% when $\rho=0.5$, indicating that more aggressive SAM is more sensitive to momentum selection, a $\rho$-$\gamma$ coupling that perfectly aligns with Theorem 3 predictions.

## Highlights & Insights
- Using Lambda Lemma + three cases, the paper moves "why SAM gets stuck at saddle points" from a vague "perturbation crossing boundaries" intuition to a geometrically rigorous Case-III description, supported by direct visualizations of cosine oscillations.
- The condition $\rho > -1/\lambda_1$ in Theorem 1 is surprisingly simple, implying that choosing a large $\rho$ will turn all saddle points with $|\lambda_1|>1/\rho$ into attractors, providing a direct design goal for improvement methods like adaptive or layer-normalized $\rho$.
- Incorporating momentum into the diffusion formula yields a strong $(1-\gamma)^{-3}$ dependence, offering the first theoretical explanation for why the momentum in SAM papers has consistently defaulted to 0.9; such "post-hoc explanations of engineering hyperparameters" are rare in deep learning.
- The closed-form difference $\Delta_{SGD}-\Delta_{SAM} = 2\eta t^2|\lambda_j|^3\rho/B$ is transferable and can be directly applied to analyze the stability of any SAM variant involving "neighborhood perturbation + single gradient."

## Limitations & Future Work
- All theory is established on second-order Taylor expansion near the saddle point $\bm{d}$ and Fisher information approximation; global dynamics beyond local neighborhoods rely on experimental extrapolation.
- It assumes $\bm{w}^p$ is a perturbation precisely calculated using a first-order approximation, without addressing normalization or second-order corrections in variants like ASAM, GSAM, or Lookahead-SAM.
- Experiments were mainly conducted on medium-scale setups (CIFAR-10/100 + ResNet-18), lacking validation on ImageNet or LLM training. Comparison with complementary theories like Bartlett et al. (2023) "bouncing across ravines" or Chen et al. (2023) "transient saddle attraction is beneficial" is only briefly mentioned.
- While the paper proves SAM escapes saddle points slowly, it does not provide a falsifiable boundary for "when SAM benefits from stagnation near saddle points"—which is precisely the perspective of Chen et al. (2023).
- The critical condition $\rho > -1/\lambda_1$ depends on accurate local Hessian information; since precisely estimating the Hessian spectrum in actual deep networks is difficult, "how to online determine entry into the dangerous $\rho$ zone" remains open.
- The paper does not provide an automated $\rho$/$\gamma$/$B$ synergistic scheduling algorithm based on its theory, leaving the engineering bridge between "diagnosis" and "treatment" for future work.

## Related Work & Insights
- **vs Compagnoni et al. 2023**: They treat noise as an implicit smoothing term using Lipschitz assumptions and small learning rates; this paper uses local dynamical analysis to treat perturbation as a stochastic excitation driving trajectories away from unstable attractors. The two are complementary, with this paper being closer to "edge-of-stability" large Hessian contexts.
- **vs Andriushchenko & Flammarion 2022**: They empirically found that small batch sizes significantly boost SAM performance; this paper provides the analytical formula $\Delta_{SAM} \propto 1/B$ in Theorem 3, elevating empirical observations to mechanistic explanations.
- **vs Kaddour et al. 2022**: They were the first to report anomalous behavior of SAM near saddle points; this paper provides the first systematic theoretical characterization of this phenomenon (Theorem 1 + Lambda Lemma geometry).
- **vs Bartlett et al. 2023 / Chen et al. 2023**: The former characterizes SAM's bouncing behavior along ravines from the perspective of sharpness, while the latter argues that transient saddle point stagnation helps generalization; this paper is complementary, pointing out that "permanent trapping" is a failure mode requiring momentum/batch size to balance exploration and escape.
- **vs Long & Bartlett 2024 (edge-of-stability)**: EoS focuses on the behavior when the maximum Hessian eigenvalue drifts near $2/\eta$ during training; the critical condition $\rho > -1/\lambda_1$ in Theorem 1 is structurally complementary to EoS and can be viewed as the SAM version of a "critical edge."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Adaptive Sharpness-Aware Minimization with a Polyak-type Step size: A Theory-Grounded Scheduler](adaptive_sharpness-aware_minimization_with_a_polyak-type_step_size_a_theory-grou.md)
- [\[ICML 2025\] Tilted Sharpness-Aware Minimization](../../ICML2025/optimization/tilted_sharpness-aware_minimization.md)
- [\[ICLR 2026\] Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization](../../ICLR2026/optimization/minor_first_major_last_a_depth-induced_implicit_bias_of_sharpness-aware_minimiza.md)
- [\[ICML 2026\] Conflicting Biases at the Edge of Stability: Norm versus Sharpness Regularization](conflicting_biases_at_the_edge_of_stability_norm_versus_sharpness_regularization.md)
- [\[ICML 2026\] Cost-Aware Stopping for Bayesian Optimization](cost-aware_stopping_for_bayesian_optimization.md)

</div>

<!-- RELATED:END -->
