---
title: >-
  [Paper Note] ReBaPL: Repulsive Bayesian Prompt Learning
description: >-
  [CVPR 2026][Multimodal VLM][SGHMC] ReBaPL transforms CLIP prompt learning from searching for a "single optimal solution" to "sampling a diverse set of high-quality prompts from the posterior using cyclical SGHMC." By introducing a "repulsive force" in the representation space via MMD/Wasserstein metrics to prevent sampling collapse into a single mode, i
tags:
  - CVPR 2026
  - Multimodal VLM
  - SGHMC
date: 2026-05-08
content_hash: 39864e9b01663d6e
---
# ReBaPL: Repulsive Bayesian Prompt Learning

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Bendou_ReBaPL_Repulsive_Bayesian_Prompt_Learning_CVPR_2026_paper.html)  
**Code**: https://github.com/SigmaNova/ReBaPL  
**Area**: Multimodal VLM  
**Keywords**: prompt learning, Bayesian inference, SGHMC, repulsive force, posterior sampling, generalization

## TL;DR
ReBaPL transforms CLIP prompt learning from searching for a "single optimal solution" to "sampling a diverse set of high-quality prompts from the posterior using cyclical SGHMC." By introducing a "repulsive force" in the representation space via MMD/Wasserstein metrics to prevent sampling collapse into a single mode, it serves as a plug-and-play Bayesian extension for any MLE prompt learning method (e.g., MaPLe, MMRL), significantly improving base-to-novel, cross-dataset, and domain generalization.

## Background & Motivation
**Background**: Vision-Language Models like CLIP utilize prompt learning for few-shot adaptation—tuning only continuous prompt vectors instead of the entire model. Performance has consistently improved from text-only methods (CoOp/CoCoOp) to multimodal prompt learning (MaPLe, MMRL, VaMP) that inserts learnable tokens across multiple layers of both image and text branches with coupling functions.

**Limitations of Prior Work**: Standard prompt learning relies on Maximum Likelihood Estimation (MLE), which is **highly prone to overfitting** on training classes, leading to poor generalization on out-of-distribution (OOD) samples and unseen classes. CoOp is particularly susceptible; while regularization methods (PromptSRC, ProDA) provide relief, they still steer learning toward a **single optimal manifold**, failing to capture the potential multimodal structure within the prompt posterior.

**Key Challenge**: The prompt loss landscape contains **many solutions with comparable training loss but significantly different generalization capabilities**. Focusing on a single point estimate (even with regularization) misses "other modes" that might generalize better.

**Goal**: Instead of seeking a point solution, this work aims to **characterize the entire prompt posterior distribution** $p(\omega\mid\mathcal{D})$, covering as many high-density modes as possible to enhance generalization to novel classes without overfitting base classes.

**Key Insight**: Existing Bayesian prompt learning (often using unimodal Gaussian variational approximations, like VaMP) or deterministic particle methods (like APP using SVGD) are limited—either by unimodality or expensive particle interactions. The authors propose using **sampling-based MCMC** (SGHMC) to represent the posterior, augmented with cyclical scheduling and repulsive forces to actively explore multimodality.

**Core Idea**: Cyclical Stochastic Gradient Hamiltonian Monte Carlo (rcSGHMC) = Hamiltonian dynamics + cyclical learning rates (alternating exploration/sampling) + representation-space repulsion. This serves as a **plug-and-play Bayesian extension for any MLE-based prompt learning method**.

## Method

### Overall Architecture
ReBaPL does not modify the underlying network architecture of a prompt learning method but replaces its "optimization process." Instead of treating multimodal prompt learning as a MAP estimation $\omega^*_{\text{MAP}}=\arg\max_\omega \log p(\omega)+\sum_i p(y_i\mid u_i,\omega)$, ReBaPL **collects a set of samples** $\{\omega^{(c)}_{k,T}\}$ from the posterior $p(\omega\mid\mathcal{D})\propto p(\mathcal{D}\mid\omega)p(\omega)$ (where $\omega$ includes all learnable prompts and coupling parameters). Final predictions are made via an ensemble $p(y\mid x)=\sum_{c,k}\gamma_{c,k}\,p(y\mid x,\omega^{(c)}_{k,T})$ with uniform weights $\gamma_{c,k}=(CK)^{-1}$. The sampling utilizes rcSGHMC (Algorithm 1): step sizes are updated via a cosine cyclical schedule. Each cycle consists of an **exploration phase** (no noise injection, relying on Hamiltonian dynamics to traverse the loss landscape for new modes) and a **sampling phase** (noise injection to sample high-quality points near the mode). Samples in the current cycle are **repelled** by samples from the previous cycle to prevent mode collapse. As this is an algorithmic improvement to optimization/sampling dynamics rather than a modular architectural change, no pipeline diagram is provided.

### Key Designs

**1. Cyclical SGHMC: Alternating "Mode Exploration" and "Intra-mode Sampling"**

To address the issue of converging to a single manifold, the authors replace standard gradient optimization with SGHMC (a stochastic gradient sampler with momentum $r$ and friction $\alpha$) combined with a cyclical schedule. The step size $\eta_t$ follows a cosine schedule; within each cycle $c$, the process is split by a balance parameter $\beta$. In the exploration phase ($\frac{t}{T}\le\beta$), no noise is added, allowing momentum to drive samples across the landscape to discover new modes. In the sampling phase ($\frac{t}{T}>\beta$), noise $\sqrt{2(\alpha-\hat\gamma)\eta_t}\,\xi_t$ is injected to sample near the discovered mode. The update follows:

$$r^{(c)}_{k,t+1}=(1-\alpha)r^{(c)}_{k,t}-\eta_t\nabla\tilde U(\omega^{(c)}_{k,t})+\mathbb{I}_{t/T>\beta}\sqrt{2(\alpha-\hat\gamma)\eta_t}\,\xi_t,$$

where $\tilde U$ is the potential energy (negative log-likelihood) estimated on a mini-batch. This maintains the computational efficiency of mini-batches while leveraging momentum for rapid landscape exploration.

**2. Repulsion in Representation Space: Pushing Apart "Functionally Similar" Prompts**

Cyclical sampling alone might still revisit similar modes. Therefore, the authors introduce an inter-cycle repulsive force: current samples $\{\omega^{(c)}_{k,t}\}$ are pushed away by samples from the previous cycle $\{\omega^{(c-1)}_{\ell,T}\}$. This is derived from a potential function $V(\omega,\omega')=\frac{1}{d_\Pi(\omega,\omega')^2+\epsilon}$, where the force $F(\omega,\omega')=-\nabla_\omega V$ increases with parameter similarity. Crucially, instead of comparing parameters directly in weight space (which suffers from permutation invariance and data scarcity), they compare the distance between "representation distributions induced by the parameters": $d_\Pi(\omega,\omega')=d_{\mathcal{P}(U)}(U_\omega,U_{\omega'})$, where $U_\omega=\{u_{\omega,i}\}$ is the set of image representations for a mini-batch under given prompts. $d_{\mathcal{P}(U)}$ is chosen as MMD or Wasserstein distance. This encourages exploring **functionally different** modes. Since calculations are per mini-batch (approx. 32 samples), the $O(n^2)$/$O(n^3)$ overhead is negligible.

**3. Plug-and-play Bayesian Extension + Ensemble Inference**

Unlike previous Bayesian prompt methods (e.g., VaMP with unimodal Gaussian variational or APP with deterministic SVGD), ReBaPL is not tied to a specific network. It acts as a **training algorithm to replace MLE optimizers**, compatible with any MLE-based prompt learning method. Evaluations were conducted using MaPLe and MMRL as bases. After training, $C\times K$ parameter samples are obtained for uniform weighted ensemble inference; these $C\times K$ forward passes are "embarrassingly parallel," incurring low additional latency.

> ⚠️ Note: Some Greek letters/symbols in the cached text (e.g., friction terms, noise estimates) may have OCR noise; refer to Algorithm 1 and Eq.(16) in the original paper for details.

## Key Experimental Results

Evaluated under a unified 16-shot setting across three protocols: base-to-novel class generalization, cross-dataset transfer, and domain generalization. ReBaPL was applied to MaPLe and MMRL bases and compared against CLIP, CoOp, CoCoOp, APP, PromptSRC, etc.

### Main Results
Base-to-Novel (Average across 11 datasets; Base/Novel/HM accuracy, where HM is Harmonic Mean):

| Method | Base | Novel | HM |
|------|------|-------|----|
| CLIP | 69.34 | 74.22 | 71.70 |
| PromptSRC* | 84.93 | 74.49 | 78.61 |
| MaPLe* | 82.03 | 75.03 | 78.37 |
| MaPLe* + ReBaPL | 83.28 | 76.08 | **79.52** (+1.15) |
| MMRL* | 85.54 | 76.52 | 80.59 |
| MMRL* + ReBaPL | 85.74 | 77.44 | **81.38** (+0.79) |

Both bases show improvements in Base and Novel accuracy with ReBaPL, with HM gains of +1.15 and +0.79, respectively. Significant gains on Novel classes were observed in challenging datasets like FGVCAircraft, EuroSAT, and DTD (e.g., MMRL+ReBaPL yielded +6.43 on EuroSAT Novel).

Cross-dataset Transfer (ImageNet trained → 10 target sets): MMRL+ReBaPL achieved the highest average of **67.62%** (+0.75), and MaPLe+ReBaPL reached 66.77% (+1.14). Domain Generalization (ImageNet → V2/Sketch/A/R): ReBaPL consistently improved performance on OOD variants.

### Ablation Study
Choice of Repulsion and Probabilistic Metric (11-dataset avg, Base/Novel/HM, Tab.4):

| Configuration | Base | Novel | HM |
|------|------|-------|----|
| MaPLe (Base) | 82.03 | 75.03 | 78.37 |
| + ReBaPL (No Repulsion) | 83.39 | 75.47 | 78.93 |
| + ReBaPL (Wasserstein) | 83.39 | 75.86 | 79.44 |
| + ReBaPL (MMD) | 83.28 | 76.08 | **79.52** |

### Key Findings
- **Sampling provides baseline gains; Repulsion targets Novel classes**: Even without repulsion (pure cyclical SGHMC), HM is higher than MaPLe (78.93 vs 78.37), showing the benefit of "posterior sampling over point estimation." Repulsion further elevates **Novel accuracy** (75.47→76.08), supporting the theory that thorough landscape exploration yields better generalization.
- **Robustness to metric choice**: The HM difference between Wasserstein and MMD is only 0.08%, much smaller than the overall ~1% gain over MaPLe, indicating the method is not sensitive to the specific probability metric.
- **Greatest gains on difficult/OOD datasets**: Improvements are most pronounced in scenarios with large distribution shifts (EuroSAT, FGVCAircraft, ImageNet-A/Sketch), consistent with expectations for Bayesian posterior characterization.

## Highlights & Insights
- **Repulsion in representation space rather than weight space**: Avoids weight space permutation invariance and data scarcity issues by using distance between prompt-induced representation distributions. This idea of moving diversity constraints to function space is highly transferable.
- **Plug-and-play Bayesian wrapper**: Framing "posterior exploration" as an optimizer replacement allows any MLE-based method to gain Bayesian benefits without modifying network architecture, making it highly engineering-friendly.
- **Explicit decoupling of "Exploration vs. Sampling"**: The cyclical phases (momentum-driven traversal vs. noise-injected sampling) are key to stably discovering multiple modes, proving more effective than unimodal variational approximations.

## Limitations & Future Work
- **Memory/Storage Overhead**: Obtaining $C\times K$ samples increases costs linearly; the paper lacks a direct resource comparison against single-model methods.
- **Metrics Selection**: Currently limited to MMD and Wasserstein; future work could explore Sinkhorn divergence or information-theoretic metrics.
- **Hyperparameter Sensitivity**: Parameters like cycle count $C$, exploration ratio $\beta$, and repulsion strength $\zeta$ require manual tuning; adaptive adjustment mechanisms are suggested for future research.
- ⚠️ Lacks a direct comparison with VaMP (due to unavailable code), making the landscape of multimodal Bayesian prompt comparisons somewhat incomplete.

## Related Work & Insights
- **vs. APP (SVGD Particle Method)**: APP uses deterministic Stein Variational Gradient Descent for particle interaction; ReBaPL uses efficient MCMC (SGHMC) and representation-space repulsion for diverse mode coverage.
- **vs. VaMP (Variational Multimodal)**: VaMP uses unimodal Gaussian variational inference for instance-level uncertainty; ReBaPL is not limited by unimodality and covers multiple modes via sampling + repulsion.
- **vs. PromptSRC / ProDA (Regularized MLE)**: These remain point estimates on a single manifold; ReBaPL characterizes the full posterior to solve the issue of missing generalization-optimized modes.
- **vs. MaPLe / MMRL (Bases)**: These are the extension targets—ReBaPL adds Bayesian gains to their established multimodal coupling structures.

## Rating
- Novelty: ⭐⭐⭐⭐ Combination of cyclical SGHMC + representation repulsion for plug-and-play prompt posterior sampling is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive evaluation across three protocols and 11 datasets; however, lacks resource overhead analysis and direct VaMP comparison.
- Writing Quality: ⭐⭐⭐⭐ Strong foundation in background topics; some dense mathematical notation.
- Value: ⭐⭐⭐⭐ Provides a universal, stackable Bayesian enhancement paradigm for prompt learning with low barrier to adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] BiomedCCPL: Causal Conditional Prompt Learning for Biomedical Vision-Language Models](biomedccpl_causal_conditional_prompt_learning_for_biomedical_vision-language_mod.md)
- [\[CVPR 2026\] Controllable Federated Prompt Learning at Test Time](controllable_federated_prompt_learning_at_test_time.md)
- [\[CVPR 2026\] Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment](noise-aware_few-shot_learning_through_bi-directional_multi-view_prompt_alignment.md)
- [\[CVPR 2026\] STAR: Test-Time Adaptation Can Enhance Universal Prompt Learning for Vision-Language Models](star_test-time_adaptation_can_enhance_universal_prompt_learning_for_vision-langu.md)
- [\[ICCV 2025\] Calibrating MLLM-as-a-Judge via Multimodal Bayesian Prompt Ensembles](../../ICCV2025/multimodal_vlm/calibrating_mllm-as-a-judge_via_multimodal_bayesian_prompt_ensembles.md)

</div>

<!-- RELATED:END -->
