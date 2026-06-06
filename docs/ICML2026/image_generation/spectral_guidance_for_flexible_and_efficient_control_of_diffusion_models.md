---
title: >-
  [Paper Note] Spectral Guidance for Flexible and Efficient Control of Diffusion Models
description: >-
  [ICML 2026][Image Generation][Spectral Guidance] This paper proposes Spectral Guidance: a method that learns the left singular functions of the diffusion process's conditional expectation operator via self-supervised lea…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Spectral Guidance"
  - "Training-free Guidance"
  - "Conditional Expectation Operator"
  - "Singular Value Decomposition"
  - "Self-Supervised Learning"
date: 2026-05-08
content_hash: b8bc71b7f52eee6d
---

# Spectral Guidance for Flexible and Efficient Control of Diffusion Models

**Conference**: ICML 2026  
**arXiv**: [2605.28900](https://arxiv.org/abs/2605.28900)  
**Code**: https://github.com/gabmoreira/spectralguidance  
**Area**: Diffusion Models / Image Generation / Controllable Generation  
**Keywords**: Spectral Guidance, Training-free Guidance, Conditional Expectation Operator, Singular Value Decomposition, Self-Supervised Learning

## TL;DR
This paper proposes Spectral Guidance: a method that learns the left singular functions of the diffusion process's conditional expectation operator via self-supervised learning. By projecting arbitrary guidance signals (labels / CLIP / mask) onto these spectral bases aligned with diffusion dynamics, it bypasses denoiser backpropagation. On CIFAR-10, it improves accuracy by 37 percentage points over the strongest training-free baseline while being 4x faster in sampling.

## Background & Motivation

**Background**: Controllable generation in diffusion models primarily follows two paths. First, classifier guidance / classifier-free guidance, which binds the model to a fixed set of conditions during training. Second, training-free guidance (DPS / LGD / FreeDoM / TFG), which pulls arbitrary clean-data losses $p(y\mid x_0)$ back to the $x_t$ space during sampling using the denoiser's point estimate $\hat{x}_0(x_t)$.

**Limitations of Prior Work**: The first category lacks flexibility, requiring retraining for new conditions. The second category is flexible but costly: it requires backpropagation through the denoiser at every sampling step, which is computationally expensive and prone to vanishing gradients. Furthermore, the approximation $p(y\mid x_0)\approx p(y\mid \hat{x}_0(x_t))$ holds strictly only when $p(y\mid x_0)$ is an affine function of $x_0$; at high noise levels, the posterior mean often drifts off the data manifold, leading to incorrect guidance gradients.

**Key Challenge**: Training-free guidance aims to use arbitrary clean-data signals but is forced to pass through the denoiser for point estimation, creating an inherent conflict between flexibility and stability/efficiency.

**Goal**: Construct an intermediate representation independent of specific guidance signals such that "calculating $p_t(y\mid x_t)$" reduces to a linear projection, decoupled from the denoiser.

**Key Insight**: View the conditional expectation $p_t(y\mid x_t)=\mathbb{E}_{X_0\sim p_t(\cdot\mid x_t)}[p(y\mid X_0)]$ as a linear operator $T_t$ mapping from the clean space $\mathcal{H}_0$ to the noisy space $\mathcal{H}_t$. As noise increases and erases information, $T_t$ becomes low-rank almost everywhere, leaving only a few "noise-resistant" directions. These directions are the left singular functions $\{\phi_{t,k}\}$ of $T_t$, which form a set of time-varying, low-dimensional coordinates aligned with the diffusion dynamics.

**Core Idea**: Perform a spectral expansion of any guidance signal on this set of left singular bases: $\mathbb{E}[h(X_0)\mid x_t]=\sum_k c_{t,k}\phi_{t,k}(x_t)$. Truncating to the first $K+1$ terms yields a stable and inexpensive guidance estimate. The $\phi_{t,k}$ themselves can be learned offline using a VICReg-style SSL objective, removing reliance on denoiser gradients.

## Method

### Overall Architecture
The method consists of two phases. **Offline Phase**: Use a lightweight time-conditioned ResNet $f_\phi:\mathcal{X}\times\mathbb{R}_{>0}\to\mathbb{R}^K$ to learn the first $K$ non-trivial left singular functions of the diffusion operator $T_t$. After training, precompute the whitening transform $(\boldsymbol{\mu}_t, \mathbf{W}_t)$ and the reference feature matrix $\boldsymbol{\Phi}_t\in\mathbb{R}^{M\times(K+1)}$ on a reference set $\mathcal{D}_\text{ref}=\{x_0^{(i)}\}_{i=1}^M$ for each timestep $t$ and cache them. **Online Phase**: For any new guidance signal $h(x_0)$ (label probabilities / CLIP embedding / segmentation mask), first estimate the spectral coefficients using Monte Carlo as $\hat{\mathbf{c}}_t=\boldsymbol{\Phi}_t^\top \mathbf{H}/M$. Then, at each DDIM step, approximate $\mathbb{E}[h(X_0)\mid x_t]$ using $\hat{\mathbf{c}}_t^\top f_\phi^w(x_t,t)$. The guidance vector $g$ is obtained by taking the gradient with respect to $x_t$ and injected into the sampling trajectory as $x\leftarrow x+\kappa\sqrt{1-\bar\alpha_t}\,g$. The same set of $\{\boldsymbol{\Phi}_t\}$ is reused across different tasks (label / CLIP / mask) by simply changing $h$.

### Key Designs

1.  **Low-rank Spectral Decomposition of the Conditional Expectation Operator**:
    *   **Function**: Re-expresses the "posterior expectation on $x_t$" as an expansion of orthogonal bases tied to the diffusion process, retaining only a few noise-resistant modes after truncation.
    *   **Mechanism**: Define $T_t:\mathcal{H}_0\to\mathcal{H}_t$ as $(T_tf)(x_t):=\mathbb{E}[f(X_0)\mid x_t]$, where the adjoint $T_t^\ast$ corresponds to forward diffusion. The covariance operator $T_tT_t^\ast$ is compact and self-adjoint, allowing for spectral decomposition $T_tf=\sum_k \sigma_{t,k}\phi_{t,k}(x_t)\mathbb{E}_{p_0}[f\psi_{t,k}]$, where $\sigma_{t,1}=1$ corresponds to the constant mode. Proposition 4.1 gives the expansion for any $h\in\mathcal{H}_0$ as $\mathbb{E}[h(X_0)\mid x_t]=\sum_k c_{t,k}\phi_{t,k}(x_t)$ with $c_{t,k}=\mathbb{E}[h(X_0)\phi_{t,k}(X_t)]$. The $L^2(p_t)$ error for truncation at $K$ is bounded by $\sigma_{t,K+1}^2\|h\|_{p_0}^2$. Proposition 4.7 proves $\sigma_{t,k}^2\le \mathbb{E}_{p_0}[\chi^2(p_t(\cdot\mid X_0)\|p_t)]$ ($k\ge2$), which tends to zero as $\bar\alpha_t\to 0$. Thus, low-rank approximation is strictly reliable at high noise.
    *   **Design Motivation**: Transforms "calculating posterior expectation" from a point estimate dependent on $h$ and the denoiser into a fixed linear projection dependent only on the diffusion process itself, providing an intrinsic "upper bound on information dimension" for guidance.

2.  **VICReg-style SSL for Learning Spectral Bases**:
    *   **Function**: Learns the first $K$ left singular functions of $T_t$ from the diffusion process itself without accessing the denoiser.
    *   **Mechanism**: Theorem 4.2 proves that for any $f=(f_1,\dots,f_K)^\top$ with $\mathbb{E}_{p_t}[f]=0$, $\max_f \operatorname{Tr}(\mathbf{C}_t(f)\boldsymbol{\Sigma}_t(f)^{-1})=\sum_{k=2}^{K+1}\sigma_{t,k}^2$, where the maximizers span $\{\phi_{t,k}\}$. This is the Rayleigh–Ritz form, equivalent to Kernel PCA with kernel $\zeta(x_t,\tilde x_t):=\int p_t(x_t\mid x_0)p_t(\tilde x_t\mid x_0)p_0(x_0)\,dx_0$. Implementation: For each $x_0^{(i)}$, sample two independent noises to get $(x_t,\tilde x_t)$ as natural "augmentations." Pass them through $f_\phi$ to get $\mathbf{Z},\tilde{\mathbf{Z}}\in\mathbb{R}^{B\times K}$. Construct the whitening matrix $\mathbf{W}=\mathbf{V}(\boldsymbol{\Lambda}+\xi\mathbf{I})^{-1/2}$ using the eigendecomposition of the batch covariance $\hat{\boldsymbol{\Sigma}}=\mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$. The loss $L=-\operatorname{Tr}((\mathbf{Z}^w)^\top\tilde{\mathbf{Z}}^w)/(K(B-1))$ uses stop-gradient on one side for stable training.
    *   **Design Motivation**: Replaces manual augmentations (e.g., cropping/color jitter) in VICReg with independent noise samples from the diffusion process—this corresponds exactly to paired sampling of the covariance operator $T_tT_t^\ast$, ensuring the SSL objective matches the spectral decomposition. The whitening term $\boldsymbol{\Sigma}_t(f)^{-1}$ prevents collapse.

3.  **Unified Spectral Projection Guidance Algorithm**:
    *   **Function**: Compresses label, CLIP, and mask guidance into a linear projection on cached features plus shallow gradients, removing denoiser backpropagation.
    *   **Mechanism**: After training $f_\phi$, precompute $\boldsymbol{\Phi}_t=[\mathbf{1}\;(\mathbf{Z}_t-\boldsymbol{\mu}_t)\mathbf{W}_t]$ on $\mathcal{D}_\text{ref}$ for every $t\in\mathcal{T}$. For new tasks, compute $\hat{\mathbf{c}}_t=\boldsymbol{\Phi}_t^\top\mathbf{H}/M$ once. In the sampling phase (Algorithm 2), each step performs standard DDIM denoising, calculates guidance $g=\nabla_{x}\mathcal{L}(\hat{\mathbf{c}}_t^\top f_\phi^w(x,t))$, and updates $x\leftarrow x+\kappa\sqrt{1-\bar\alpha_t}\,g$. Differences between tasks lie only in $\mathcal{L}$: labels use log-likelihood in $\nabla z/z$ form (as truncation may locally violate positivity, ratios replace $\log$); CLIP uses $\mathcal{L}(\mathbf{z})=\mathbf{z}^\top \mathbf{e}_\text{text}/\|\mathbf{z}\|$; masks use $-\|\mathbf{z}-\mathbf{z}_\text{target}\|^2$. Computationally, each step only passes through $f_\phi$ (16M parameters) vs. the denoiser (114M), without backpropagating through the latter.
    *   **Design Motivation**: Preloads the "heavy lifting" to a one-time offline phase, leaving only gradients from a shallow network for the online phase. A single set of $\{\boldsymbol{\Phi}_t\}$ is reused for all downstream tasks.

### Loss & Training
Training optimizes a single objective $L=-\operatorname{Tr}((\mathbf{Z}^w)^\top\tilde{\mathbf{Z}}^w)/(K(B-1))$ plus a small ridge term $\xi$ for whitening. Timesteps are sampled uniformly from $\mathcal{T}$. $\boldsymbol{\mu},\mathbf{W}$ are recalculated within batches, and stop-gradient is applied to one side. $K=512$ is used for CIFAR-10 / CelebA-HQ, and $K=2000$ for ImageNet. Training $f_\phi$ on CelebA-HQ takes ~10 GPU·h, while precomputing $\{\boldsymbol{\Phi}_t\}$ takes only 0.8 GPU·h.

## Key Experimental Results

### Main Results
Evaluated on CIFAR-10 / CelebA-HQ / ImageNet against DPS / LGD / FreeDoM / MPGD / UGD / TFG, covering label, attribute composition, CLIP, and mask guidance using a shared unconditional DDPM U-Net.

| Dataset / Task | Metric | Uncond. | Strongest Baseline | Ours | Gain |
|---|---|---|---|---|---|
| CIFAR-10 / Labels | Acc↑ | 10.0 | 52.0 (TFG) | **89.4** | **+37.4** |
| CIFAR-10 / Labels | FID↓ | 98.1 | 88.3 (MPGD) | **70.7** | −17.6 |
| CelebA-HQ / Gender+Age | Acc↑ | 25.0 | 75.2 (TFG) | **91.5** | +16.3 |
| CelebA-HQ / Gender+Hair | Acc↑ | 22.4 | 76.0 (TFG) | **88.3** | +12.3 |
| ImageNet / Labels | Acc↑ | 0.0 | 40.9 (TFG) | **41.6** | +0.7 |
| CelebA-HQ / Mask | IoU↑ | 0.38 | 0.78 (TFG, FreeDoM) | **0.80** | +0.02 |
| CelebA-HQ / CLIP | VQAScore↑ | 0.34 | 0.62 (TFG) | **0.64** | +0.02 |

Efficiency Comparison (CelebA-HQ, 100 DDIM steps, batch=1):

| Phase | Metric | Uncond. | TFG | Ours |
|---|---|---|---|---|
| Offline | Train $f_\phi$ / GPU·h | – | – | 10.0 |
| Offline | Precompute $\{\Phi_t\}$ / GPU·h | – | – | 0.8 |
| Online | Latency per step / ms | 19.2 | 81.2 | **21.7** |
| Online | Throughput (1 image) / s | 1.9 | 8.1 | **2.2** |
| Online | Peak VRAM / GB | 1.1 | 2.8 | 3.6 |
| End-to-End | 10k images total / h | 5.3 | 22.5 | **16.9** |

### Ablation Study
| Configuration | Key Metrics | Note |
|---|---|---|
| Full ($K=512$, sweep $\kappa$) | Acc-FID Frontier | Significantly outperforms training-free baselines, approaching CG with noisy classifiers. |
| Rank $K\in\{8,\dots,512\}$ | Acc | Sharp increase from $K=8\to 128$ then saturates; confirms Prop. A.11. |
| High $\kappa$ | FID degradation | Guidance dominates scores and pushes trajectories off-manifold; typical diversity-fidelity trade-off. |
| Sliding window $[\tau-100,\tau+100]$ | Acc(τ) correlation | Highly correlated with normalized trace of $T_tT_t^\ast$. Optimal windows at $\tau\approx 400$ (CIFAR-10) and $\tau\approx 700$ (CelebA-HQ). |

### Key Findings
- The +37% gain on CIFAR-10 stems from the spectral bases themselves—the same $\{\boldsymbol{\Phi}_t\}$ supports label, CLIP, and mask tasks, proving these coordinates are "task-agnostic internal structures of diffusion."
- $K$ is not just a dimension; beyond saturation, it acts like a "guidance intensity knob": adding modes increases effective scale at fixed $\kappa$, reducing intra-class diversity.
- Spectral phase transitions exist in the midpoint of $T_tT_t^\ast$ (CIFAR-10 ~400, CelebA-HQ ~700). These regions are the most effective guidance windows, providing an interpretable criterion for "when to guide."
- Dense pixel-level constraints (e.g., inpainting half of 256×256×3) exceed the capacity of a $K$-dimensional subspace; thus, Spectral Guidance is complementary to, not a replacement for, DPS-style methods.

## Highlights & Insights
- **Redefining "Training-free Guidance" as "Training-free Spectral Projection"**: Previous routes were stuck on the difficulty of calculating $p_t(y\mid x_t)$. This work algebraicizes it as operator SVD learned via SSL, unifying all guidance signals on a single basis.
- **Natural Coupling of VICReg and Diffusion**: Double noise sampling is exactly the paired sampling of the covariance operator $T_tT_t^\ast$, giving the heuristic "augmentation-invariance" a rigorous spectral interpretation.
- **Spectral Phase Transition as a Physical Pointer**: The decision of which steps to apply guidance is shifted from empirical hyperparameter tuning to a choice determined by $\sigma_{t,k}$ decay curves.
- **Offline-Online Amortization**: Completely removes denoiser backpropagation from the online path, leaving only gradients for a 16M network, which is key for scaling up plug-and-play guidance.

## Limitations & Future Work
- Validated on pixel-level, medium-scale DDPMs; not yet evaluated on latent diffusion or large-scale T2I foundation models. However, $T_t$ and its SVD should extend naturally to latent spaces.
- Estimating $\hat{\mathbf{c}}_t$ requires a reference set $\mathcal{D}_\text{ref}$ with $h$ annotations, whereas other training-free baselines only need a loss or pretrained model. This is a cost if the target domain lacks labels, though $\mathcal{D}_\text{ref}$ can be small or self-sampled.
- Low-rank subspace representation is insufficient for dense pixel-level inverse problems (inpainting/super-resolution).

## Related Work & Insights
- **vs CG / CFG**: CG/CFG bake conditions into training; this method uses an unconditional model + spectral bases for one-time training and universal condition reuse.
- **vs DPS / LGD / MPGD**: These rely on $\hat{x}_0(x_t)$ and denoiser gradients; this method explicitly models the posterior expectation via SVD and limits gradients to a shallow network $f_\phi$, improving both accuracy and speed.
- **vs UGD / FreeDoM / TFG**: These use "time-travel" and adaptive schedules; this method reads the guidance window directly from spectral decay, making schedule design theoretically grounded.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Rewriting training-free guidance as operator SVD + SSL learned spectral bases is a genuine paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets / four tasks / seven baselines, though lacks latent diffusion and large T2I validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear mathematical derivations (Prop 4.1 / Thm 4.2 / Prop A.11) well-integrated with algorithms and experiments.
- Value: ⭐⭐⭐⭐⭐ Delivers gains in controllability, efficiency, and interpretability simultaneously.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance](../../CVPR2026/image_generation/cfg-ctrl_control-based_classifier-free_diffusion_guidance.md)
- [\[ICML 2026\] Caracal: Causal Architecture via Spectral Mixing](caracal_causal_architecture_via_spectral_mixing.md)
- [\[AAAI 2026\] RelaCtrl: Relevance-Guided Efficient Control for Diffusion Transformers](../../AAAI2026/image_generation/relactrl_relevance-guided_efficient_control_for_diffusion_transformers.md)
- [\[ICML 2026\] GuidedBridge: Training-freely Improving Bridge Models with Prior Guidance](guidedbridge_training-freely_improving_bridge_models_with_prior_guidance.md)
- [\[ICML 2026\] Local Hessian Spectral Filtering for Robust Intrinsic Dimension Estimation](local_hessian_spectral_filtering_for_robust_intrinsic_dimension_estimation.md)

</div>

<!-- RELATED:END -->
