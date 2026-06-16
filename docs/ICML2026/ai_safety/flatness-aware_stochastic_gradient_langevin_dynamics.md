---
title: >-
  [Paper Note] Flatness-Aware Stochastic Gradient Langevin Dynamics
description: >-
  [ICML 2026][AI Safety][SGLD] This paper proposes fSGLD: it replaces the parameter $\theta$ at the gradient evaluation in standard SGLD with a Gaussian-perturbed $\theta+\epsilon$, and strictly couples the perturbation scale $\sigma$ with the inverse temperature $\beta$ via $\sigma=\beta^{-(1+\eta)/4}$. Without adding any extra gradient or memory o
tags:
  - ICML 2026
  - AI Safety
  - SGLD
date: 2026-05-08
content_hash: ac7c61f1b0c39ce4
---
# Flatness-Aware Stochastic Gradient Langevin Dynamics

**Conference**: ICML 2026  
**arXiv**: [2510.02174](https://arxiv.org/abs/2510.02174)  
**Code**: https://github.com/youngsikhwang/Flatness-aware-SGLD (Available)  
**Area**: Optimization / Bayesian Sampling / Flat Minima  
**Keywords**: SGLD, Flat Minima, Hessian-trace Regularization, Gibbs Distribution, Random Weight Perturbation

## TL;DR
This paper proposes fSGLD: it replaces the parameter $\theta$ at the gradient evaluation in standard SGLD with a Gaussian-perturbed $\theta+\epsilon$, and strictly couples the perturbation scale $\sigma$ with the inverse temperature $\beta$ via $\sigma=\beta^{-(1+\eta)/4}$. Without adding any extra gradient or memory overhead, the algorithm's invariant measure approximates the Gibbs distribution corresponding to the Hessian-trace regularized objective $v(\theta)=u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))$. The authors provide non-asymptotic bounds for Wasserstein-1 distance and excess risk, achieving performance comparable to or better than SAM/ASAM on CIFAR/WebVision/ViT while nearly halving the training time.

## Background & Motivation
**Background**: Generalization in deep networks is highly correlated with the "flatness" of the loss surface. Dominant approaches include the SAM series (min-max inner perturbation + double gradients) and Entropy-SGD/Entropy-MCMC (introducing auxiliary variables for local entropy smoothing). Both push training toward low-curvature basins but come with significant costs: SAM requires two gradients per step, and the Entropy series doubles memory usage.

**Limitations of Prior Work**: These methods are inherently "local"—they only utilize geometric information within a small neighborhood of the current point, making it difficult to escape sharp basins on multimodal, highly non-convex loss surfaces. Theoretical guarantees are mostly limited to local convergence. Another line of work is Langevin-type global sampling (SGLD), which theoretically concentrates on the global minimum at sufficiently low temperatures. However, its invariant measure $\pi_\beta^{\text{SGLD}}\propto\exp(-\beta u)$ is determined entirely by the objective function and is agnostic to the surface geometry; thus, it finds "any" global minimum rather than a "flat" one.

**Key Challenge**: No existing algorithm simultaneously possesses (a) global exploration capability, (b) an inductive bias toward low-curvature regions, and (c) the same computational/memory cost as SGD. Entropy-MCMC is the closest work but requires auxiliary variables, doubles memory, and is theoretically limited to strongly convex settings.

**Goal**: Design a first-order Langevin algorithm without extra gradient or memory overhead, such that its invariant measure concentrates on the global minimum of the "Hessian-trace regularized objective" $v(\theta)=u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))$ (i.e., the "global flat minimum"), and provide non-asymptotic Wasserstein and excess risk bounds in non-convex settings.

**Key Insight**: The authors observe that replacing the gradient $\nabla U(\theta,X)$ in SGLD with the perturbed gradient $\nabla U(\theta+\epsilon,X)$ evaluated at $\theta+\epsilon$ yields an expectation that is exactly the gradient of the randomized smoothing surrogate $g_\epsilon(\theta)=\mathbb{E}[u(\theta+\epsilon)]$. The second-order Taylor expansion of $g_\epsilon$ equals $u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))$ plus a higher-order residual. In other words, "perturbed gradients + Langevin noise" naturally embeds Hessian-trace regularization—provided the higher-order residual can be controlled.

**Core Idea**: Use a "$\sigma$–$\beta$ coupling formula" $\sigma=\beta^{-(1+\eta)/4}$ (with $\eta$ fixed at 0.1) to bridge the sampling temperature and perturbation scale. This ensures that as $\beta$ increases, the residual vanishes at a controlled rate, allowing the invariant measure of fSGLD to strictly approximate the "flat-biased Gibbs distribution" $\pi^\star_{\beta,\sigma}\propto\exp(-\beta v(\theta))$.

## Method

### Overall Architecture
fSGLD addresses the trilemma of "requiring global exploration, favoring flat basins, and not being more expensive than SGD." Its approach is remarkably simple: the only difference from standard SGLD is "where the gradient is evaluated"—shifting the gradient from the current parameter $\theta_k$ to a Gaussian-perturbed point $\theta_k+\epsilon_{k+1}$, paired with an analytical formula that ties the perturbation scale $\sigma$ to the sampling temperature $\beta$. Given initial parameters $\theta_0$ and the data distribution, it outputs a parameter chain $\{\theta_k\}$, which can either perform posterior averaging for Bayesian prediction or be used as a standard optimizer taking the final state.

### Key Designs

**1. Perturbed Gradient: Injecting Second-Order Curvature Info at Zero Extra Cost**

Addressing the pain point that SAM requires two gradients and Hessian-penalty requires approximating Hessian-vector products, fSGLD directly replaces $\nabla_\theta U(\theta_k,X_{k+1})$ in SGLD with $\nabla_\theta U(\theta_k+\epsilon_{k+1},X_{k+1})$, where $\epsilon_{k+1}\sim\mathcal{N}(0,\sigma^2 I_d)$. Combined with standard Langevin noise $\xi_{k+1}\sim\mathcal{N}(0,I_d)$, the step is defined as $\theta_{k+1}=\theta_k-\lambda\,\nabla_\theta U(\theta_k+\epsilon_{k+1},X_{k+1})+\sqrt{2\lambda\beta^{-1}}\,\xi_{k+1}$.

While this appears to be "adding noise to weights," its expectation encodes curvature: the expected perturbed gradient is the gradient of the randomized smoothing surrogate $\mathbb{E}_{\epsilon,X}[\nabla_\theta U(\theta+\epsilon,X)]=\nabla g_\epsilon(\theta)$. The second-order Taylor expansion of $g_\epsilon$ under Gaussian expectation is $g_\epsilon(\theta)=u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))+\mathbb{E}[\mathcal{R}(\theta,\epsilon)]$. This effectively embeds the Hessian-trace into the objective with a single perturbation, bypassing explicit ascent steps and Hessian approximations while maintaining SGLD's single-gradient and $O(d)$ memory complexity.

**2. $\sigma$–$\beta$ Coupling Formula: Synchronizing Approximation Error and Flatness Bias**

Design 1 leaves a potential issue: the Taylor expansion has a residual $\mathbb{E}[\mathcal{R}(\theta,\epsilon)]=O(\sigma^4 d^2)$. If $\sigma$ is treated as an independent hyperparameter, the residual could explode (destroying the Hessian-trace bias) or the perturbation could be too small (degenerating to standard SGLD). fSGLD solves this by tying $\sigma$ to the temperature: $\sigma=\beta^{-(1+\eta)/4}$, with $\eta$ fixed at $0.1$.

This formula is derived, not just tuned. Proposition 3.4 proves that for $\eta\in(0,1)$, $W_2(\pi^{\text{fSGLD}}_\beta,\pi^\star_{\beta,\sigma})=O(\beta^{-\eta/4}\sqrt d+\beta^{-\eta/2}d+\beta^{-(1+\eta)/2}d^2)$, meaning the error can be minimized by increasing $\beta$. Simultaneously, $\sigma=\beta^{-(1+\eta)/4}$ ensures the flatness bias does not vanish too quickly as $\beta\to\infty$, yielding a "sweet spot" at finite $\beta$. This coupling collapses the "accuracy vs. flatness bias" trade-off into a single-parameter curve, exposing only $\beta$ and step size $\lambda$ to the user.

**3. Flatness-Biased Gibbs Distribution: Provable Target for Flat Minima Sampling**

fSGLD formalizes the heuristic "search for flat basins" as a probability measure $\pi^\star_{\beta,\sigma}\propto\exp(-\beta v(\theta))$, where $v(\theta)=u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))$ is the Hessian-trace regularized objective. 

Under standard SGLD assumptions (fourth-order differentiability + data-dependent Lipschitz + dissipativity), Theorem 3.5 provides $W_1(\mathcal{L}(\theta_k^{\text{fSGLD}}),\pi^\star_{\beta,\sigma})\le D_1 e^{-\dot c\lambda k/2}+(D_2+D_3)\sqrt\lambda+\underline{D}$. These terms represent exponential mixing of overdamped Langevin, $O(\lambda^{1/2})$ discretization error of Euler–Maruyama, and the invariant measure bias. Theorem 3.8 further translates this into an excess risk bound $\mathbb{E}[v(\theta_k)]-\inf v\le D_1^\diamond e^{-\dot c\lambda k/4}+D_2^\diamond\lambda^{1/4}+D_3^\diamond$. This is significant because it is the first to characterize Langevin global convergence toward the flat objective $v$ rather than the original $u$, with discretization rates matching current SOTA SGLD analysis (Zhang et al., 2023).

### Loss & Training
The authors do not explicitly change the loss function; the "effective objective" $v(\theta)$ is implicitly defined by the algorithm dynamics. Implementation only involves adding Gaussian noise to the parameters *before* evaluating the gradient. $\eta=0.1$ is fixed throughout, and $\beta$ and step size $\lambda$ follow standard SGLD schedules. Theoretically, $\beta$, $\lambda$, and iterations $k$ must satisfy specific bounds (Equations 63–65) to ensure the $W_1$ error remains $\le\bar\delta$.

## Key Experimental Results

### Main Results

Bayesian Image Classification on ResNet-18 (Bayesian Model Averaging, results averaged over 3 seeds; baselines except fSGLD and ASAM are cited from Entropy-MCMC):

| Dataset | Metric | fSGLD | Prev. SOTA | Gain |
|--------|------|-------|-----------|------|
| CIFAR-10 | ACC % ↑ | **95.73** | Entropy-MCMC 95.69 | +0.04 |
| CIFAR-10 | NLL ↓ | **0.144** | ASAM 0.150 | -0.006 (≈ 4% relative) |
| CIFAR-100 | ACC % ↑ | 78.53 | Entropy-MCMC **79.16** | -0.63 (3rd) |
| CIFAR-100 | NLL ↓ | **0.810** | ASAM 0.814 | -0.004 |
| CIFAR-10→SVHN OOD | AUROC % | **98.91** | Entropy-SGD 98.71 | +0.20 |
| CIFAR-100→SVHN OOD | AUPR % | **88.01** | ASAM 87.93 | +0.08 |

Training from scratch on CIFAR-N (noisy labels) and WebVision using ResNet-34/50 (5 seeds; s/epoch measured on CIFAR-10N):

| Model | Optimizer | CIFAR-10N | CIFAR-100N | WV-1 | WV-5 | s/epoch |
|-------|-----------|-----------|------------|------|------|---------|
| ResNet-34 | SGD | 89.31 | 58.47 | 71.87 | 89.33 | 22.0 |
| ResNet-34 | SAM | 91.53 | 59.18 | 73.49 | **90.32** | 41.3 |
| ResNet-34 | ASAM | **91.73** | 60.79 | 73.46 | 90.14 | 41.4 |
| ResNet-34 | **fSGLD** | 91.37 | **61.51** | **73.95** | 90.03 | **23.7** |
| ResNet-50 | SAM | 90.88 | 59.01 | 72.52 | 89.53 | 60.7 |
| ResNet-50 | ASAM | **91.25** | 60.47 | 71.92 | 88.48 | 60.9 |
| ResNet-50 | **fSGLD** | 90.86 | **61.26** | **73.54** | **90.34** | **34.1** |

ViT-B/16 Fine-tuning: fSGLD reaches 75.67 on CIFAR-100N, exceeding ASAM's 74.86, with an epoch time of 345.8s (vs. SAM's 656.7s and ASAM's 662.5s), nearly halving the cost.

### Ablation Study

| Configuration | Key Metric | Description |
|------|----------|------|
| Coupling $\sigma=\beta^{-(1+\eta)/4}$, $\eta\in(0,1)$ | Stable peak performance | $\eta=0.1$ is recommended |
| Fixed $\beta=10^8$, sweep $\sigma$ | Performance drops when $\eta\notin(0,1)$ | Confirms $\sigma$ cannot be set independently of temperature |
| Fixed $\sigma=10^{-3}$, sweep $\beta$ | Same as above | Confirms temperature cannot be set independently of perturbation |
| Hessian Spectrum (ResNet-34 / CIFAR-10N) | fSGLD $\lambda_{\text{top}}$ and $\mathrm{tr}(H)$ significantly lower than SGD/SGLD | Verifies convergence to flatter minima |

### Key Findings
- **Computational Efficiency**: Compared to SAM/ASAM, fSGLD outperforms on difficult tasks (noisy labels, many classes) while training in approximately half the time, proving perturbed gradients effectively substitute explicit second-order steps.
- **Memory Efficiency**: Compared to Entropy-MCMC, fSGLD requires no auxiliary variables and halves the memory usage, with superior NLL on CIFAR-100.
- **Robustness**: Performance is insensitive to $\eta$ within $(0,1)$, indicating the coupling formula is robust.
- **Geometric Feedback**: Hessian spectrum experiments provide direct evidence that the algorithms theoretically targeted $\mathrm{tr}(H)$ regularization results in measurably smaller curvature in practice.

## Highlights & Insights
- **"Randomized Smoothing = Implicit Hessian-trace"**: The authors elegantly utilize this equivalence to embed the benefits of SAM or Hessian-penalty into a single perturbation of SGLD.
- **Coupling as a Theoretical Derivation**: The formula $\sigma=\beta^{-(1+\eta)/4}$ is not a heuristic trick but the optimal coupling rate derived from Wasserstein bounds, reducing the user-facing hyperparameter search.
- **Transferable Design**: The "perturbed gradient evaluation + temperature coupling" strategy can be applied to any Langevin/diffusion-based optimizer to gain flatness bias without overhead.
- **Theoretical Advancement**: This shifts the Langevin global convergence paradigm from targeting $u$ to targeting a flatness-regularized $v$, providing the first non-asymptotic global results for sampling flat minima.

## Limitations & Future Work
- **Theoretical Constants**: Constants $D_1, D_3$ have exponential dependence on dimension $d$ and temperature $\beta$ (inherited from Eberle’s coupling), a common ceiling in current SGLD theory.
- **Assumption Scope**: The analysis requires global Lipschitz continuity; handling semi-convex scenarios remains for future work.
- **Empirical Scope**: Experiments are limited to vision classification. Scaling to modern LLMs or diffusion generative models is an open challenge.
- **Future Directions**: Implementing an $\eta$ schedule or combining with replica-exchange SGLD to mitigate exponential constants in high dimensions.

## Related Work & Insights
- **vs. SAM/ASAM**: SAM uses min-max to find the worst point in a neighborhood (requiring double gradients); fSGLD uses Gaussian expectation for a neighborhood average (single gradient) and includes global sampling noise. fSGLD is faster and more robust to noise.
- **vs. Entropy-SGD / Entropy-MCMC**: These methods double memory via auxiliary variables and lack non-convex non-asymptotic theory. fSGLD maintains SGLD's memory footprint and provides general non-convex Wasserstein bounds.
- **vs. Standard SGLD**: Standard SGLD is geometry-agnostic. fSGLD shifts the target to $v=u+\tfrac{\sigma^2}{2}\mathrm{tr}(H)$ without sacrificing discretization rates.
- **vs. Random Weight Perturbation (RWP)**: RWP typically lacks global convergence guarantees. fSGLD can be seen as "SGLD + theoretically coupled RWP."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Elegant integration of randomized smoothing and temperature coupling into a provable Langevin framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive coverage across classification and OOD; lacks generative and NLP tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear progression from motivation to theory to validation.
- Value: ⭐⭐⭐⭐⭐ Extremely high utility as a generic SGD/SGLD replacement that offers "SAM-like" flatness benefits at zero extra gradient or memory cost.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FlowHijack: A Dynamics-Aware Backdoor Attack on Flow-Matching Vision-Language-Action Models](../../CVPR2026/ai_safety/flowhijack_a_dynamics-aware_backdoor_attack_on_flow-matching_vision-language-act.md)
- [\[ICML 2026\] FedHPro: Federated Hyper-Prototype Learning via Gradient Matching](fedhpro_federated_hyper-prototype_learning_via_gradient_matching.md)
- [\[ICML 2026\] Hidden in Plain Tokens: Simply Robust, Gradient-Free Watermark for Synthetic Audio](hidden_in_plain_tokens_simply_robust_gradient-free_watermark_for_synthetic_audio.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](../../AAAI2026/ai_safety/robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[ICML 2026\] How Hard Can It Be? Hardness-Aware Multi-Objective Unlearning](how_hard_can_it_be_hardness-aware_multi-objective_unlearning.md)

</div>

<!-- RELATED:END -->
