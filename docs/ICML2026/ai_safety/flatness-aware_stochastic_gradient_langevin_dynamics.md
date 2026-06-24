---
title: >-
  [Paper Note] Flatness-Aware Stochastic Gradient Langevin Dynamics
description: >-
  [ICML 2026][AI Safety][SGLD] This paper proposes fSGLD: it replaces the parameter $\theta$ at the gradient step in standard SGLD with a Gaussian-perturbed $\theta+\epsilon$, and strictly couples the perturbation scale $\sigma$ with the inverse temperature $\beta$ via $\sigma=\beta^{-(1+\eta)/4}$. Without adding any gradient or memory overhead, the algorithm's invariant measure approximates the Gibbs distribution corresponding to the Hessian-trace regularized objective $v(\the…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "SGLD"
  - "Flat Minima"
  - "Hessian-trace Regularization"
  - "Gibbs Distribution"
  - "Random Weight Perturbation"
date: 2026-05-08
content_hash: be8187e73a456a36
---

# Flatness-Aware Stochastic Gradient Langevin Dynamics

**Conference**: ICML 2026  
**arXiv**: [2510.02174](https://arxiv.org/abs/2510.02174)  
**Code**: https://github.com/youngsikhwang/Flatness-aware-SGLD (Available)  
**Area**: Optimization / Bayesian Sampling / Flat Minima  
**Keywords**: SGLD, Flat Minima, Hessian-trace Regularization, Gibbs Distribution, Random Weight Perturbation

## TL;DR
This paper proposes fSGLD: it replaces the parameter $\theta$ at the gradient step in standard SGLD with a Gaussian-perturbed $\theta+\epsilon$, and strictly couples the perturbation scale $\sigma$ with the inverse temperature $\beta$ via $\sigma=\beta^{-(1+\eta)/4}$. Without adding any gradient or memory overhead, the algorithm's invariant measure approximates the Gibbs distribution corresponding to the Hessian-trace regularized objective $v(\theta)=u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))$. The authors provide non-asymptotic bounds for Wasserstein-1 distance and excess risk, achieving performance comparable to or better than SAM/ASAM on CIFAR/WebVision/ViT with nearly halved training time.

## Background & Motivation
**Background**: The generalization of deep networks is highly correlated with the "flatness" of the loss surface. Mainstream approaches include the SAM series (min-max inner perturbation + double gradients) and Entropy-SGD/Entropy-MCMC (introducing auxiliary variables for local entropy smoothing). These methods push training toward low-curvature basins but come with significant costs: SAM requires two gradients per step, and the Entropy series doubles the memory.

**Limitations of Prior Work**: These methods are inherently "local"—they only use geometric information within a small neighborhood of the current point, making it difficult to escape sharp basins on multi-modal, highly non-convex loss surfaces; theoretical guarantees are also mostly limited to local convergence. Another line of work is Langevin-type global sampling (SGLD), which theoretically concentrates on global minima at sufficiently low temperatures. However, its invariant measure $\pi_\beta^{\text{SGLD}}\propto\exp(-\beta u)$ is entirely determined by the objective function and is agnostic to the surface geometry, meaning it finds "any" global minimum rather than a "flat" one.

**Key Challenge**: There is currently no algorithm in the literature that simultaneously possesses (a) global exploration capability, (b) inductive bias toward low-curvature regions, and (c) computational/memory costs equivalent to SGD. Entropy-MCMC is the closest work but requires auxiliary variables, doubles memory, and its theory only holds under strong convexity.

**Goal**: To design a first-order Langevin algorithm with no extra gradient or memory overhead, such that its invariant measure concentrates on the global minima of the "Hessian-trace regularized objective" $v(\theta)=u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))$ (i.e., "global flat minima"), and to provide non-asymptotic Wasserstein and excess risk bounds in non-convex settings.

**Key Insight**: The authors observe that replacing the gradient $\nabla U(\theta,X)$ in SGLD with the perturbed gradient $\nabla U(\theta+\epsilon,X)$ evaluated at $\theta+\epsilon$ yields an expectation that is exactly the gradient of the randomized smoothing proxy $g_\epsilon(\theta)=\mathbb{E}[u(\theta+\epsilon)]$. The second-order Taylor expansion of $g_\epsilon$ is equal to $u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))$ plus a higher-order residual. In other words, "perturbed gradient + Langevin noise" naturally embeds Hessian-trace regularization—as long as the higher-order residual can be controlled.

**Core Idea**: Use a $\sigma$–$\beta$ coupling formula $\sigma=\beta^{-(1+\eta)/4}$ (with $\eta$ fixed at 0.1) to bridge the sampling temperature and perturbation scale. This ensures that as $\beta$ increases, the residual vanishes at a controlled rate, allowing the invariant measure of fSGLD to strictly approximate the "flatness-biased Gibbs distribution" $\pi^\star_{\beta,\sigma}\propto\exp(-\beta v(\theta))$.

## Method

### Overall Architecture
fSGLD aims to solve the trilemma of "requiring global exploration, favoring flat basins, and not being more expensive than SGD." Its approach is remarkably simple: the only difference from standard SGLD is "where the gradient is evaluated"—shifting the gradient from the current parameter $\theta_k$ to a Gaussian-perturbed point $\theta_k+\epsilon_{k+1}$, combined with an analytical formula that ties the perturbation scale $\sigma$ to the sampling temperature $\beta$. Given initial parameters $\theta_0$ and a data distribution, it outputs a parameter chain $\{\theta_k\}$, which can serve as a Bayesian predictor via posterior averaging or as a normal optimizer using the final state.

### Key Designs

**1. Perturbed Gradient: Injecting Second-Order Curvature Info at Zero Extra Cost**

Addressing the pain point that SAM requires two gradients and Hessian-penalty requires approximating Hessian-vector products, fSGLD recognizes that second-order information is usually expensive. It replaces $\nabla_\theta U(\theta_k,X_{k+1})$ in the SGLD update directly with $\nabla_\theta U(\theta_k+\epsilon_{k+1},X_{k+1})$, where $\epsilon_{k+1}\sim\mathcal{N}(0,\sigma^2 I_d)$. Combined with standard Langevin noise $\xi_{k+1}\sim\mathcal{N}(0,I_d)$, the full step is: $\theta_{k+1}=\theta_k-\lambda\,\nabla_\theta U(\theta_k+\epsilon_{k+1},X_{k+1})+\sqrt{2\lambda\beta^{-1}}\,\xi_{k+1}$.

This step appears to be just "adding noise to weights," but its expectation hides curvature: the expectation of the perturbed gradient is exactly the gradient of the randomized smoothing proxy $\mathbb{E}_{\epsilon,X}[\nabla_\theta U(\theta+\epsilon,X)]=\nabla g_\epsilon(\theta)$. The second-order Taylor expansion of $g_\epsilon$ under Gaussian expectation is $g_\epsilon(\theta)=u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))+\mathbb{E}[\mathcal{R}(\theta,\epsilon)]$. Thus, a single Gaussian perturbation implicitly injects the Hessian-trace into the optimization objective, avoiding explicit ascent gradients and Hessian approximations, while maintaining the single gradient and $O(d)$ memory of SGLD.

**2. $\sigma$–$\beta$ Coupling Formula: Synchronized Decay of Approximation Error and Flatness Bias**

Design 1 leaves a potential issue: the Taylor expansion has a residual $\mathbb{E}[\mathcal{R}(\theta,\epsilon)]=O(\sigma^4 d^2)$. If the perturbation scale $\sigma$ is treated as an independent hyperparameter, either the residual explodes (destroying the Hessian-trace bias) or the perturbation is too small (regressing to standard SGLD). fSGLD's solution is to bind $\sigma$ to the sampling temperature via the analytical relation: $\sigma=\beta^{-(1+\eta)/4}$, with $\eta$ fixed at $0.1$.

This formula was derived theoretically rather than tuned empirically. Proposition 3.4 proves that when $\eta\in(0,1)$, $W_2(\pi^{\text{fSGLD}}_\beta,\pi^\star_{\beta,\sigma})=O(\beta^{-\eta/4}\sqrt d+\beta^{-\eta/2}d+\beta^{-(1+\eta)/2}d^2)$, meaning the approximation error can be suppressed by increasing $\beta$. Simultaneously, $\sigma=\beta^{-(1+\eta)/4}$ ensures the flatness bias does not vanish too quickly as $\beta\to\infty$, resulting in a "sweet spot" at finite $\beta$. In summary, the coupling collapses the "approximation accuracy vs. flatness bias strength" trade-off into a single-parameter curve—leaving the user with the same hyperparameters as SGLD ($\beta$ and step size $\lambda$).

**3. Flatness-biased Gibbs Distribution: Proven Sampling Goal**

The first two designs answer "how," while this point answers "what is achieved." fSGLD formalizes the heuristic "finding flat basins" into a probability measure $\pi^\star_{\beta,\sigma}\propto\exp(-\beta v(\theta))$, where $v(\theta)=u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))$ is the Hessian-trace regularized objective, and provides non-asymptotic guarantees around it.

Under standard SGLD assumptions (fourth-order differentiability + data-dependent Lipschitz + dissipativity), Theorem 3.5 gives $W_1(\mathcal{L}(\theta_k^{\text{fSGLD}}),\pi^\star_{\beta,\sigma})\le D_1 e^{-\dot c\lambda k/2}+(D_2+D_3)\sqrt\lambda+\underline{D}$, representing exponential mixing of overdamped Langevin, $O(\lambda^{1/2})$ discretization error of Euler–Maruyama, and the bias of the invariant measure. Theorem 3.8 further translates this into an excess risk bound $\mathbb{E}[v(\theta_k)]-\inf v\le D_1^\diamond e^{-\dot c\lambda k/4}+D_2^\diamond\lambda^{1/4}+D_3^\diamond$. This signifies that while previous Langevin global convergence theories targeted the minima of $u$, this is the first to target the flat objective $v$, proving the algorithm's bias is a characterized "global sampling of flat minima," with discretization rates matching the best standard SGLD analysis (Zhang et al., 2023).

### Loss & Training
The authors do not explicitly modify the loss function—the "effective objective" $v(\theta)=u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))$ is implicitly defined by the algorithm's dynamics. Implementation simply requires adding Gaussian noise to the parameters at the gradient evaluation point. $\eta=0.1$ is fixed throughout, while $\beta$ and step size $\lambda$ follow standard SGLD schedules for each benchmark. Theoretically, $\beta$, $\lambda$, and the number of iterations $k$ must satisfy lower/upper bounds in equations (63)–(65) to ensure the $W_1$ error $\le\bar\delta$.

## Key Experimental Results

### Main Results

Bayesian image classification on ResNet-18 (Bayesian Model Averaging, results are mean±std of 3 seeds; other baselines except fSGLD and ASAM are cited from the Entropy-MCMC paper):

| Dataset | Metric | fSGLD | Prev. SOTA | Gain |
|--------|------|-------|-----------|------|
| CIFAR-10 | ACC % ↑ | **95.73** | Entropy-MCMC 95.69 | +0.04 |
| CIFAR-10 | NLL ↓ | **0.144** | ASAM 0.150 | -0.006 (≈ 4% relative) |
| CIFAR-100 | ACC % ↑ | 78.53 | Entropy-MCMC **79.16** | -0.63 (3rd) |
| CIFAR-100 | NLL ↓ | **0.810** | ASAM 0.814 | -0.004 |
| CIFAR-10→SVHN OOD | AUROC % | **98.91** | Entropy-SGD 98.71 | +0.20 |
| CIFAR-100→SVHN OOD | AUPR % | **88.01** | ASAM 87.93 | +0.08 |

Training from scratch on ResNet-34/50 with noisy labels (CIFAR-N and WebVision, mean of 5 seeds; s/epoch measured on CIFAR-10N):

| Model | Optimizer | CIFAR-10N | CIFAR-100N | WV-1 | WV-5 | s/epoch |
|-------|-----------|-----------|------------|------|------|---------|
| ResNet-34 | SGD | 89.31 | 58.47 | 71.87 | 89.33 | 22.0 |
| ResNet-34 | SAM | 91.53 | 59.18 | 73.49 | **90.32** | 41.3 |
| ResNet-34 | ASAM | **91.73** | 60.79 | 73.46 | 90.14 | 41.4 |
| ResNet-34 | **fSGLD** | 91.37 | **61.51** | **73.95** | 90.03 | **23.7** |
| ResNet-50 | SAM | 90.88 | 59.01 | 72.52 | 89.53 | 60.7 |
| ResNet-50 | ASAM | **91.25** | 60.47 | 71.92 | 88.48 | 60.9 |
| ResNet-50 | **fSGLD** | 90.86 | **61.26** | **73.54** | **90.34** | **34.1** |

ViT-B/16 Fine-tuning: fSGLD achieves 75.67 on CIFAR-100N, surpassing ASAM's 74.86, with a single epoch time of 345.8s (SAM 656.7s, ASAM 662.5s), nearly halved.

### Ablation Study

| Configuration | Key Metric | Description |
|------|----------|------|
| Coupled $\sigma=\beta^{-(1+\eta)/4}$, $\eta\in(0,1)$ | Performance stable at peak | Recommended $\eta=0.1$ |
| Fixed $\beta=10^8$, sweep $\sigma$ | Significant drop when $\eta\notin(0,1)$ | Validates that perturbation scale cannot be set independently of temperature |
| Fixed $\sigma=10^{-3}$, sweep $\beta$ | Same as above | Inverse validation: Temperature cannot be set independently of perturbation |
| Hessian Spectrum (ResNet-34 / CIFAR-10N) | $\lambda_{\text{top}}$ and $\mathrm{tr}(H)$ significantly smaller than SGD/SGLD | Directly confirms fSGLD converges to flatter minima |

### Key Findings
- Compared to SAM/ASAM: fSGLD outperforms on "harder" tasks (high noise + many classes) like CIFAR-100N and WebVision Top-1, **while training time is approximately half of SAM/ASAM**—proving that "perturbed gradients" are a cost-effective alternative to explicit second-order methods.
- Compared to Entropy-MCMC: fSGLD requires no auxiliary variables (halving memory), outperforms on CIFAR-10, and is slightly lower on CIFAR-100 ACC (0.6%) but better on NLL.
- Sensitivity: Performance is robust to $\eta$ within $(0,1)$, suggesting the coupling formula is both necessary and stable; in practice, tuning $\beta$ is sufficient.
- Hessian Spectrum: Validates the "mechanism → geometry" loop. While theory predicts implicit $\mathrm{tr}(H)$ regularization, experiments confirm $\mathrm{tr}(H)$ actually decreases.

## Highlights & Insights
- **"Randomized Smoothing = Implicit Hessian-trace Regularization"**: This equivalence is utilized cleanly. The authors avoid auxiliary variables, Hessian-vector products, or double gradients, packing SAM/Hessian-penalty benefits into a single SGLD perturbation.
- **Theory-derived Hyperparameter Coupling**: $\sigma=\beta^{-(1+\eta)/4}$ is derived from optimal coupling rates in Wasserstein bounds rather than empirical tuning. Consequently, fSGLD exposes no more hyperparameters to the user than standard SGLD.
- **Transferable Design**: Any Langevin/diffusion-based optimizer (e.g., training diffusion models, Bayesian fine-tuning) can adopt "perturbed gradient evaluation + temperature coupling" to obtain flatness bias with zero extra cost.
- **Theoretical Paradigm Shift**: Moving from "Wasserstein convergence to objective $u$" to "Wasserstein convergence to flat objective $v$." This provides the first global non-asymptotic result for "sampling toward flat minima"—previously, such paths only had local PAC-Bayes bounds.

## Limitations & Future Work
- Admitted Limitations: Constants $D_1, D_3$ depend exponentially on dimension $d$ and temperature $\beta$ (inherited from Eberle’s coupling arguments), which is the current "ceiling" of SGLD theory. Also, analysis requires global Lipschitz $u$; semi-convex scenarios are left for future work.
- Observed Limitations: Theoretical selections for $\beta, \lambda, k$ involve constants of order $d^2$, making them difficult to use directly in engineering; empirical tuning of $\beta$ is still required. Experiments are restricted to ResNet/ViT image classification; scaling to modern LLMs/diffusion models remains an open question.
- Future Directions: (i) Implementing an $\eta$ schedule (larger early, smaller later) for better exploration/accuracy; (ii) Combining with preconditioned/replica-exchange SGLD to mitigate exponential constants; (iii) Validating on diffusion models to test if "flatter → higher quality/diversity."

## Related Work & Insights
- **vs. SAM/ASAM**: SAM uses min-max to find the worst point in a neighborhood for gradients, requiring double gradients. fSGLD uses Gaussian expectation for a neighborhood average, requiring single gradients, and naturally possesses global sampling properties (Langevin noise) to escape local sharp valleys.
- **vs. Entropy-SGD / Entropy-MCMC**: Both introduce auxiliary variables to approximate local entropy, doubling memory, and Entropy-MCMC's theory only holds under strong convexity. fSGLD has no auxiliary variables and provides non-asymptotic bounds under general non-convexity.
- **vs. Standard SGLD**: Standard SGLD's Gibbs measure is geometry-agnostic, concentrating only on $u$. fSGLD changes the target to $v=u+\tfrac{\sigma^2}{2}\mathrm{tr}(H)$, providing the first non-asymptotic global result for "flat minima sampling."
- **vs. Random Weight Perturbation (RWP)**: RWP usually treats perturbation scale as an independent hyperparameter and lacks global convergence guarantees. fSGLD can be viewed as "SGLD + forced coupling RWP," bringing the geometric role of RWP into the Langevin framework.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to formalize "perturbed gradient + temperature coupling" as a provably correct flatness-biased SGLD.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers Bayesian classification, uncertainty, OOD, noisy labels, and ViT fine-tuning, with $\beta$-$\sigma$ ablation and Hessian visualization; lacks NLP or generative tasks.
- Writing Quality: ⭐⭐⭐⭐ Concepts progress clearly (Motivation → Smoothing → Coupling → Bounds → Experiments); formulas are dense but well-explained.
- Value: ⭐⭐⭐⭐⭐ High-performance flatness bias with first-order, single-gradient, zero extra memory cost. Halves SAM/ASAM training time, serving as a high-value SGD replacement for Bayesian workflows.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Multilingual Unlearning in LLMs: Transfer, Dynamics, and Reversibility](multilingual_unlearning_in_llms_transfer_dynamics_and_reversibility.md)
- [\[CVPR 2026\] FlowHijack: A Dynamics-Aware Backdoor Attack on Flow-Matching Vision-Language-Action Models](../../CVPR2026/ai_safety/flowhijack_a_dynamics-aware_backdoor_attack_on_flow-matching_vision-language-act.md)
- [\[ICML 2026\] Gradient Transformer: Learning to Generate Updates for LLMs](gradient_transformer_learning_to_generate_updates_for_llms.md)
- [\[ICLR 2026\] When Flatness Does (Not) Guarantee Adversarial Robustness](../../ICLR2026/ai_safety/when_flatness_does_not_guarantee_adversarial_robustness.md)
- [\[ICML 2026\] FedHPro: Federated Hyper-Prototype Learning via Gradient Matching](fedhpro_federated_hyper-prototype_learning_via_gradient_matching.md)

</div>

<!-- RELATED:END -->
