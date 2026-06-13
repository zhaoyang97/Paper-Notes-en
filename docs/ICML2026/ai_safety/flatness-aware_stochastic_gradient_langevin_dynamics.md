---
title: >-
  [Paper Note] Flatness-Aware Stochastic Gradient Langevin Dynamics
description: >-
  [ICML 2026][AI Safety][SGLD] This paper proposes fSGLD: replacing parameter $\theta$ with Gaussian-perturbed $\theta+\epsilon$ in standard SGLD updates and strictly coupling perturbation scale $\sigma$ with inverse tempe…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "SGLD"
  - "Flat Minima"
  - "Hessian-trace Regularization"
  - "Gibbs Distribution"
  - "Random Weight Perturbation"
date: 2026-05-08
content_hash: 3600e4d2ff834a8e
---

# Flatness-Aware Stochastic Gradient Langevin Dynamics

**Conference**: ICML 2026  
**arXiv**: [2510.02174](https://arxiv.org/abs/2510.02174)  
**Code**: https://github.com/youngsikhwang/Flatness-aware-SGLD (Available)  
**Area**: Optimization / Bayesian Sampling / Flat Minima  
**Keywords**: SGLD, Flat Minima, Hessian-trace Regularization, Gibbs Distribution, Random Weight Perturbation

## TL;DR
This paper proposes fSGLD: replacing parameter $\theta$ with Gaussian-perturbed $\theta+\epsilon$ in standard SGLD updates and strictly coupling perturbation scale $\sigma$ with inverse temperature $\beta$ via $\sigma=\beta^{-(1+\eta)/4}$. This allows the invariant measure to approximate the Gibbs distribution of the Hessian-trace regularized objective $v(\theta)=u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))$ without additional gradient or memory overhead. Non-asymptotic Wasserstein-1 and excess risk bounds are provided, and experiments on CIFAR/WebVision/ViT achieve performance comparable to or better than SAM/ASAM with nearly halved training time.

## Background & Motivation
**Background**: Generalization in deep networks is highly correlated with the "flatness" of the loss surface. Dominant approaches include the SAM series (min-max inner perturbation + double gradients) and Entropy-SGD/Entropy-MCMC (local entropy smoothing via auxiliary variables). These guide training toward low-curvature basins but incur high costs: SAM requires two gradients per step, and the Entropy series doubles memory usage.

**Limitations of Prior Work**: These methods are essentially "local"—they only utilize geometric information within a small neighborhood of the current point, making it difficult to escape sharp basins on multi-modal, highly non-convex loss surfaces. Theoretical guarantees are mostly limited to local convergence. Another line of work is Langevin-style global sampling (SGLD), which theoretically concentrates on global minima at sufficiently low temperatures. However, its invariant measure $\pi_\beta^{\text{SGLD}}\propto\exp(-\beta u)$ is solely determined by the objective function and is agnostic to surface geometry; thus, it finds "any" global minimum rather than a "flat" one.

**Key Challenge**: No existing algorithm simultaneously possesses (a) global exploration capability, (b) inductive bias toward low-curvature regions, and (c) computational/memory costs equivalent to standard SGD. Entropy-MCMC is the closest work but requires auxiliary variables, doubles memory, and its theory holds primarily under strong convexity.

**Goal**: Design a first-order Langevin algorithm with no extra gradient or memory overhead, such that its invariant measure concentrates on the global minima of the "Hessian-trace regularized objective" $v(\theta)=u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))$ (i.e., "global flat minima"), and provide non-asymptotic Wasserstein and excess risk bounds in non-convex settings.

**Key Insight**: The authors observe that replacing the gradient $\nabla U(\theta,X)$ in SGLD with a perturbed gradient $\nabla U(\theta+\epsilon,X)$ calculated at $\theta+\epsilon$ results in an expectation that is exactly the gradient of the randomized smoothing surrogate $g_\epsilon(\theta)=\mathbb{E}[u(\theta+\epsilon)]$. The second-order Taylor expansion of $g_\epsilon$ equals $u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))$ plus a higher-order residual. In other words, "perturbed gradients + Langevin noise" naturally embeds Hessian-trace regularization, provided the higher-order residual can be controlled.

**Core Idea**: A "$\sigma$–$\beta$ coupling formula" $\sigma=\beta^{-(1+\eta)/4}$ (with $\eta$ fixed at 0.1) serves as the bridge between sampling temperature and perturbation scale. This ensures that as $\beta$ increases, the residual vanishes at a controlled rate, allowing the invariant measure of fSGLD to strictly approximate the "flat-biased Gibbs distribution" $\pi^\star_{\beta,\sigma}\propto\exp(-\beta v(\theta))$.

## Method

### Overall Architecture
fSGLD is nearly identical to SGLD, with the only difference being "where the gradient is calculated." At each step:

1. Sample a Gaussian perturbation $\epsilon_{k+1}\sim\mathcal{N}(0,\sigma^2 I_d)$ and a Langevin noise $\xi_{k+1}\sim\mathcal{N}(0,I_d)$.
2. Compute the gradient $\nabla_\theta U(\theta_k+\epsilon_{k+1},X_{k+1})$ at the **perturbed point** $\theta_k+\epsilon_{k+1}$ for a mini-batch $X_{k+1}$.
3. Standard SGLD-style update:

    $$\theta_{k+1}=\theta_k-\lambda\,\nabla_\theta U(\theta_k+\epsilon_{k+1},X_{k+1})+\sqrt{2\lambda\beta^{-1}}\,\xi_{k+1}$$

4. Key constraint: $\sigma$ is not an independent hyperparameter but is determined by $\beta$ via $\sigma=\beta^{-(1+\eta)/4}$ with $\eta=0.1$. Consequently, the **hyperparameters exposed to the user are identical to SGLD** (only $\beta$ and $\lambda$ need tuning).

Inputs are model parameters $\theta_0$ and the data distribution; the output is the parameter chain $\{\theta_k\}$, which can be used for Bayesian posterior averaging or as a standard optimizer taking the final state.

### Key Designs

1. **Perturbed Gradient as an Implicit Hessian-trace Estimator**:
    - **Function**: Replaces $\nabla_\theta U(\theta,X)$ in SGLD with $\nabla_\theta U(\theta+\epsilon,X)$ to inject second-order curvature information at zero extra gradient cost.
    - **Mechanism**: The expectation of the perturbed gradient is $\mathbb{E}_{\epsilon,X}[\nabla_\theta U(\theta+\epsilon,X)]=\nabla g_\epsilon(\theta)$. Taylor expansion under Gaussian expectation gives $g_\epsilon(\theta)=u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))+\mathbb{E}[\mathcal{R}(\theta,\epsilon)]$. Thus, adding simple weight noise implicitly embeds the Hessian-trace into the optimization objective.
    - **Design Motivation**: SAM explicitly calculates curvature using an extra "ascent gradient," and Hessian-penalty methods approximate Hessian-vector products. fSGLD bypasses both through Gaussian randomization, maintaining $O(d)$ memory and single-gradient cost.

2. **$\sigma$–$\beta$ Coupling Formula $\sigma=\beta^{-(1+\eta)/4}$**:
    - **Function**: Links "perturbation scale" to "sampling temperature" through an analytical relationship, ensuring a precise balance between the Taylor residual $\mathbb{E}[\mathcal{R}(\theta,\epsilon)]=O(\sigma^4 d^2)$ and the "temperature sensitivity" $\beta$ of the Gibbs measure.
    - **Mechanism**: In Proposition 3.4, the authors prove that when $\eta\in(0,1)$, $W_2(\pi^{\text{fSGLD}}_\beta,\pi^\star_{\beta,\sigma})=O(\beta^{-\eta/4}\sqrt d+\beta^{-\eta/2}d+\beta^{-(1+\eta)/2}d^2)$, which converges to 0 as $\beta$ increases. Simultaneously, $\sigma=\beta^{-(1+\eta)/4}$ ensures that the strength of the flatness bias does not vanish too quickly as $\beta\to\infty$, creating a "sweet spot" for finite $\beta$.
    - **Design Motivation**: If $\sigma$ were an independent hyperparameter, the Taylor residual and $\beta$ would be decoupled, leading to either residual explosion (destroying the Hessian-trace bias) or insufficient perturbation (degrading to standard SGLD). The coupling formula synchronizes their decay, collapsing the "approximation accuracy vs. flatness bias strength" trade-off onto a single curve and keeping the hyperparameter count equal to SGLD.

3. **Flat-biased Gibbs Distribution as the Theoretical Target**:
    - **Function**: Upgrades "finding flat minima" from a heuristic goal to a well-defined probability measure $\pi^\star_{\beta,\sigma}\propto\exp(-\beta v(\theta))$, providing non-asymptotic bounds for Wasserstein-1 and excess risk.
    - **Mechanism**: Under standard SGLD assumptions (fourth-order differentiability + data-dependent Lipschitz + dissipativity), Theorem 3.5 proves $W_1(\mathcal{L}(\theta_k^{\text{fSGLD}}),\pi^\star_{\beta,\sigma})\le D_1 e^{-\dot c\lambda k/2}+(D_2+D_3)\sqrt\lambda+\underline{D}$. These terms correspond to the exponential mixing of overdamped Langevin, Euler-Maruyama discretization error $O(\lambda^{1/2})$, and invariant measure deviation, respectively. Theorem 3.8 further provides an excess risk bound of $\mathbb{E}[v(\theta_k)]-\inf v\le D_1^\diamond e^{-\dot c\lambda k/4}+D_2^\diamond\lambda^{1/4}+D_3^\diamond$.
    - **Design Motivation**: Previous global convergence theories for Langevin targeted the minima of the original objective $u$. This work is the first to target $v$, demonstrating that the algorithm's bias is not just "empirically observed" but is theoretically characterized as "global sampling of flat minima." The discretization error rate aligns with the best standard SGLD analyses (Zhang et al., 2023), showing no loss in convergence speed.

### Loss & Training
The authors do not explicitly modify the loss function; the "effective objective" prioritized by the dynamics is implicitly defined as $v(\theta)=u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))$. During training, one only needs to introduce a Gaussian perturbation to the parameters where the gradient is computed. $\eta=0.1$ is fixed throughout, while $\beta$ and step size $\lambda$ follow standard SGLD schedules for each benchmark. Theoretically, $\beta, \lambda,$ and iteration count $k$ must satisfy lower/upper bounds given in (63)–(65) to ensure $W_1$ error $\le\bar\delta$.

## Key Experimental Results

### Main Results

Bayesian image classification on ResNet-18 (Bayesian Model Averaging, results are mean±std of 3 random seeds; other baselines are cited from Entropy-MCMC original text, excluding fSGLD and ASAM):

| Dataset | Metric | fSGLD | Prev. SOTA | Gain |
|--------|------|-------|-----------|------|
| CIFAR-10 | ACC % ↑ | **95.73** | Entropy-MCMC 95.69 | +0.04 |
| CIFAR-10 | NLL ↓ | **0.144** | ASAM 0.150 | -0.006 (≈ 4% rel.) |
| CIFAR-100 | ACC % ↑ | 78.53 | Entropy-MCMC **79.16** | -0.63 (3rd) |
| CIFAR-100 | NLL ↓ | **0.810** | ASAM 0.814 | -0.004 |
| CIFAR-10→SVHN OOD | AUROC % | **98.91** | Entropy-SGD 98.71 | +0.20 |
| CIFAR-100→SVHN OOD | AUPR % | **88.01** | ASAM 87.93 | +0.08 |

ResNet-34/50 training from scratch on noisy labels (CIFAR-N) and WebVision (mean of 5 seeds; s/epoch measured on CIFAR-10N):

| Model | Optimizer | CIFAR-10N | CIFAR-100N | WV-1 | WV-5 | s/epoch |
|-------|-----------|-----------|------------|------|------|---------|
| ResNet-34 | SGD | 89.31 | 58.47 | 71.87 | 89.33 | 22.0 |
| ResNet-34 | SAM | 91.53 | 59.18 | 73.49 | **90.32** | 41.3 |
| ResNet-34 | ASAM | **91.73** | 60.79 | 73.46 | 90.14 | 41.4 |
| ResNet-34 | **Ours** | 91.37 | **61.51** | **73.95** | 90.03 | **23.7** |
| ResNet-50 | SAM | 90.88 | 59.01 | 72.52 | 89.53 | 60.7 |
| ResNet-50 | ASAM | **91.25** | 60.47 | 71.92 | 88.48 | 60.9 |
| ResNet-50 | **Ours** | 90.86 | **61.26** | **73.54** | **90.34** | **34.1** |

ViT-B/16 Fine-tuning: fSGLD achieves 75.67 on CIFAR-100N, surpassing ASAM's 74.86, with a per-epoch time of 345.8s (vs. SAM's 656.7s and ASAM's 662.5s), nearly a 50% reduction.

### Ablation Study

| Configuration | Key Metric | Description |
|------|----------|------|
| Coupled $\sigma=\beta^{-(1+\eta)/4}$, $\eta\in(0,1)$ | Performance stable at peak | $\eta=0.1$ recommended |
| Fixed $\beta=10^8$ and sweep $\sigma$ | Performance drops significantly when $\eta\notin(0,1)$ | Confirms perturbation scale cannot be set independently of temperature |
| Fixed $\sigma=10^{-3}$ and sweep $\beta$ | Same as above | Confirms temperature cannot be set independently of perturbation |
| Hessian Spectrum Comparison (ResNet-34 / CIFAR-10N) | $\lambda_{\text{top}}$ and $\mathrm{tr}(H)$ significantly lower for fSGLD | Directly confirms fSGLD converges to flatter minima |

### Key Findings
- **Comparison with SAM/ASAM**: fSGLD outperforms in "difficult" tasks (high noise + many classes) like CIFAR-100N and WebVision Top-1, **while requiring roughly half the training time**—proving that replacing explicit second-order calculation with perturbed gradients is both efficient and effective.
- **Comparison with Entropy-MCMC**: fSGLD requires no auxiliary variables, halves memory usage, and outperforms in CIFAR-10 accuracy; it is slightly lower in CIFAR-100 accuracy (0.6%) but superior in NLL.
- **Robustness**: Performance is insensitive to $\eta$ within $(0,1)$, suggesting the coupling formula is robust and only $\beta$ needs tuning in practice.
- **Validation**: Hessian spectrum experiments provide a closed loop from mechanism to geometric result: theory predicts regularized $\mathrm{tr}(H)$, and experiments demonstrate reduced $\mathrm{tr}(H)$.

## Highlights & Insights
- **"Randomized Smoothing = Implicit Hessian-trace Regularization"**: This equivalence is utilized elegantly. The authors avoid auxiliary variables, Hessian-vector products, or double gradients, packing all desired SAM/Hessian-penalty properties into a single perturbation within SGLD.
- **Theoretical Hyperparameter Coupling**: $\sigma=\beta^{-(1+\eta)/4}$ is derived from the optimal coupling rate required to balance Wasserstein bounds and Taylor residuals, not empirical trial and error. This allows fSGLD to match SGLD's hyperparameter count.
- **Transferable Design**: The strategy of "calculating gradients at perturbed points + temperature-coupled perturbation" can be applied to any Langevin/diffusion-based optimizer (e.g., training diffusion models or Bayesian fine-tuning) to gain a flatness bias without overhead.
- **Theoretical Paradigm Shift**: By shifting focus from Wasserstein convergence toward the original target $u$ to the flat-biased target $v$, this work provides the first non-asymptotic global result for "sampling from flat minima." Discretization error rates match the state-of-the-art standard SGLD analyses.

## Limitations & Future Work
- **Theoretical Constants**: Constants $D_1, D_3$ depend exponentially on dimension $d$ and inverse temperature $\beta$, reflecting the current limits of SGLD theory (inherited from Eberle et al.).
- **Assumptions**: Analysis currently requires global Lipschitz continuity; extending this to semi-convex cases is future work.
- **Scope**: Experiments are focused on ResNet/ViT for image classification. Scaling and validating on modern LLMs or large-scale diffusion models remain open challenges.
- **Future Directions**: (i) Implementing an $\eta$ schedule (large initially, small later) to balance exploration and precision; (ii) Combining with preconditioned/replica-exchange SGLD; (iii) Extending to training diffusion models to verify if "flatter = higher quality/diverse samples."

## Related Work & Insights
- **vs. SAM/ASAM**: SAM uses min-max to find the worst point in a neighborhood, requiring two gradients. fSGLD uses Gaussian expectation for a neighborhood average, requiring only one gradient, and possesses global sampling properties via Langevin noise.
- **vs. Entropy-SGD / Entropy-MCMC**: These require auxiliary variables and double memory; Entropy-MCMC theory is restricted to strong convexity. fSGLD preserves standard memory usage and provides non-asymptotic Wasserstein bounds under general non-convexity.
- **vs. Standard SGLD**: Standard SGLD targets minima of $u$ and is geometry-agnostic. fSGLD targets $v=u+\tfrac{\sigma^2}{2}\mathrm{tr}(H)$, providing the first global non-asymptotic result for flat-minima sampling without sacrificing convergence rate.
- **vs. Random Weight Perturbation (RWP)**: RWP typically treats perturbation scale as an independent hyperparameter and lacks global convergence guarantees. fSGLD effectively integrates RWP into the Langevin framework with a theoretically-grounded coupling rate.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to establish a theoretically sound flat-biased SGLD via "perturbed gradients + coupled temperature."
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage across Bayesian classification, OOD, noisy labels, and ViT fine-tuning, including Hessian spectrum visualization. Limited to vision tasks.
- **Writing Quality**: ⭐⭐⭐⭐ Logical progression with clear intuitions provided for dense mathematical proofs.
- **Value**: ⭐⭐⭐⭐⭐ High-performance alternative to SAM/ASAM with half the computational cost and no extra memory, suitable as a general-purpose replacement for Bayesian workflows.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] FedHPro: Federated Hyper-Prototype Learning via Gradient Matching](fedhpro_federated_hyper-prototype_learning_via_gradient_matching.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](../../AAAI2026/ai_safety/robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[ICML 2026\] Hidden in Plain Tokens: Simply Robust, Gradient-Free Watermark for Synthetic Audio](hidden_in_plain_tokens_simply_robust_gradient-free_watermark_for_synthetic_audio.md)
- [\[CVPR 2026\] Monte Carlo Stochastic Depth for Uncertainty Estimation in Deep Learning](../../CVPR2026/ai_safety/mcsd_uncertainty_estimation.md)
- [\[ICML 2026\] How Hard Can It Be? Hardness-Aware Multi-Objective Unlearning](how_hard_can_it_be_hardness-aware_multi-objective_unlearning.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] FedHPro: Federated Hyper-Prototype Learning via Gradient Matching](fedhpro_federated_hyper-prototype_learning_via_gradient_matching.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](../../AAAI2026/ai_safety/robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[ICML 2026\] Hidden in Plain Tokens: Simply Robust, Gradient-Free Watermark for Synthetic Audio](hidden_in_plain_tokens_simply_robust_gradient-free_watermark_for_synthetic_audio.md)
- [\[ICML 2026\] How Hard Can It Be? Hardness-Aware Multi-Objective Unlearning](how_hard_can_it_be_hardness-aware_multi-objective_unlearning.md)
- [\[CVPR 2026\] Monte Carlo Stochastic Depth for Uncertainty Estimation in Deep Learning](../../CVPR2026/ai_safety/mcsd_uncertainty_estimation.md)

</div>

<!-- RELATED:END -->
