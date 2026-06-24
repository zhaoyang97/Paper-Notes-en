---
title: >-
  [Paper Note] Unifying Continuous and Discrete Text Diffusion with Non-simultaneous Diffusion Processes
description: >-
  [ACL 2025][Text diffusion models] This paper proposes NeoDiff, which unifies the theoretical framework of discrete and continuous text diffusion models. By introducing a dual-time framework consisting of "extrinsic time" (sentence-level diffusion progress) and "intrinsic time" (token-level diffusion progress), NeoDiff utilizes a Poisson process to independently allocate fine-grained noise levels to each token and adaptively adjusts denoising progress with a context-aware time…
tags:
  - "ACL 2025"
  - "Text diffusion models"
  - "Poisson diffusion process"
  - "non-simultaneous denoising"
  - "time predictor"
  - "continuous-discrete unification"
date: 2026-05-08
content_hash: 15912019482a6f69
---

# Unifying Continuous and Discrete Text Diffusion with Non-simultaneous Diffusion Processes

**Conference**: ACL 2025  
**arXiv**: [2505.22165](https://arxiv.org/abs/2505.22165)  
**Code**: None  
**Area**: Others  
**Keywords**: Text diffusion models, Poisson diffusion process, non-simultaneous denoising, time predictor, continuous-discrete unification

## TL;DR
This paper proposes NeoDiff, which unifies the theoretical framework of discrete and continuous text diffusion models. By introducing a dual-time framework consisting of "extrinsic time" (sentence-level diffusion progress) and "intrinsic time" (token-level diffusion progress), NeoDiff utilizes a Poisson process to independently allocate fine-grained noise levels to each token and adaptively adjusts denoising progress with a context-aware time predictor. NeoDiff outperforms existing diffusion baselines across multiple tasks, including machine translation, paraphrasing, and text simplification.

## Background & Motivation

**Background**: Text diffusion models are categorized into two major paradigms:
   - **Discrete Diffusion** (e.g., D3PM, Absorbing Diffusion): Perform state transitions independently for each token on categorical distributions. Different tokens can have different diffusion progress, but the noise control is coarse-grained (tokens are either preserved or completely corrupted).
   - **Continuous Diffusion** (e.g., DiffuSeq, Difformer): Map tokens into continuous spaces and inject Gaussian noise. The noise control is fine-grained, but all tokens share a uniform noise level.

**Limitations of Prior Work**:
   - The coarseness of discrete diffusion limits the benefits of multi-step generation—tokens only have "uncorrupted" and "corrupted" states, lacking fine-grained intermediate transitions.
   - The uniform noise in continuous diffusion forces all tokens to stay at the same noise level, which prevents low-noise tokens from acting as context to help restore high-noise tokens.
   - Existing improvements (such as mask blending in DiffuSeq-V2 or monotonic noise in AR-Diffusion) have not fully achieved token-level, fine-grained noise control.

**Key Challenge**: There is a need for a unified framework that enjoys both the fine-grained noise control of continuous diffusion and the varying token-level diffusion progress of discrete diffusion.

**Goal**
   - Achieving token-level non-simultaneous diffusion in the continuous space.
   - Adaptively adjusting the denoising rate of each token using contextual information during the reverse denoising process.
   - Optimizing the inference time schedule to improve generation quality.

**Key Insight**: Generalize the diffusion time variable into two dimensions: extrinsic time $t$ (global progress) and intrinsic time $\tau$ (token-level progress). By randomizing the intrinsic time via a Poisson process, token-level non-simultaneous diffusion is naturally achieved.

**Core Idea**: Unify discrete and continuous diffusion using a dual-time framework (extrinsic + intrinsic time). The forward process assigns token-level noise using a Poisson process, while the reverse process adaptively regulates denoising via a time predictor based on semantic contexts.

## Method

### Overall Architecture
Input text $\to$ **Word embedding mapping** to continuous space $\mathbf{z}_0$ $\to$ **Poisson Forward Process**: Each token independently samples an intrinsic time $\tau_t$ and is corrupted with Gaussian noise of corresponding intensity based on $\tau_t$ $\to$ **Reverse Denoising**: Encoder-Decoder Transformer predicts $\hat{\mathbf{z}}_0$ + a time predictor predicts $\tau_{t'}$ of each token for the next step $\to$ **Rounding** back to discrete tokens $\to$ Output text. The entire process conducts inference using an optimized extrinsic time schedule.

### Key Designs

1. **Dual-Time Unified Framework**:

    - **Function**: Integrate existing discrete and continuous diffusion models into a unified theory.
    - **Mechanism**:
        - **Extrinsic time** $t \in [0,1]$: Global diffusion progress for the entire sentence.
        - **Intrinsic time** $\tau_t \in [0,1]$: Independent diffusion progress for each token.
        - Discrete diffusion = $\tau_t \in \{0, 1\}$ (a binary random function).
        - Continuous diffusion = $\tau_t = t$ (a deterministic function, synchronized across all tokens).
        - DiffuSeq-V2 = $\tau_t = \max(t + \tau_{\text{mask}}(t), 1)$ (blending).
        - **NeoDiff** = $\tau_t \in [0,1]$, modeled as a continuous random function of $t$.
    - **Design Motivation**: By generalizing the time variables, a unified theory covering all existing methods is established, with NeoDiff being the most general instantiation.

2. **Poisson Forward Diffusion Process**:

    - **Function**: Independently sample fine-grained diffusion progress for each token.
    - **Mechanism**:
        - Introduce a discrete state function $s_t \in \{0, 1, ..., s_{\max}\}$, corresponding to the range from clean to maximum noise.
        - The evolution of $s_t$ follows a Poisson process: $s_t \sim \text{Poisson}(\lambda(t))$, where $\lambda(t) = ks_{\max}t$.
        - Normalize and clip: $\tau_t = \text{Clip}(s_t / s_{\max}, 1)$.
    - **Key variance control issue**: When $s_{\max}$ is very large, $\text{CV} = 1/\sqrt{s_{\max}} \to 0$, causing $\tau$ values of all tokens to converge and degenerate into continuous diffusion.
    - **Solution**: Variance rescaling transformation:
    $$\tau_t = \text{Clip}\left(\text{Round}\left(\frac{s_t - \lambda(t)}{\sqrt{\lambda(t)}} \sigma(t) + \lambda(t)\right), s_{\max}\right) / s_{\max}$$
    Setting $\sigma(t) = \lambda(t)$ guarantees that discrete characteristics are independent of $s_{\max}$.
    - **Design Motivation**: The Poisson process is naturally suited for modeling "jump-like" state transitions. It has a single parameter ($\lambda$) and an analytical distribution, serving as a natural bridge between discrete jumps and continuous gradients.

3. **Context-Aware Time Predictor**:

    - **Function**: Adaptively adjust the denoising rate of each token based on semantic context during the reverse process.
    - **Mechanism**:
        - Instead of rigidly mirroring the forward process ($p_\theta(\tau_{t'}|\mathbf{z}_t, \tau_t) = q(\tau_{t'})$), explicitly model $p_\theta(\tau_{t'}|\mathbf{z}_t, \tau_t)$.
        - Input: predicted generation $\hat{\mathbf{z}}_0$ (instead of $\mathbf{z}_t$!), target time $t'$, conditional sentence embedding $\mathbf{x}$.
        - Train with cross-entropy, framing the prediction of $\tau$ as a discrete classification problem.
    - **Pseudo-label strategy**: Instead of directly using $\tau_{t'}$ as labels (which introduces bias), pseudo-labels are obtained by mapping the reconstruction loss ranking of $\hat{\mathbf{z}}_0$ through the inverse Poisson CDF. Tokens with larger loss are assigned higher $\tau$ (more noise requires slower denoising).
    - **Design Motivation**: Dynamically link the token-level denoising speed to generation quality—well-generated tokens complete denoising first, while difficult tokens retain more noise and recover gradually.

4. **Bayes-Optimized Extrinsic Time Schedule**:

    - **Function**: Optimize the global timestep sequence during inference.
    - **Mechanism**: Treat $\{t_1, t_2, ..., t_K\}$ as continuous variables and search for the optimal schedule on the validation set using Bayesian optimization.
    - **Design Motivation**: Different tasks require different time allocation strategies.

### Loss & Training
- Total loss:
$$\mathcal{L} = \mathcal{L}_z + \mathcal{L}_\tau + \mathcal{L}_{\text{anchor}}$$
    - $\mathcal{L}_z = \|\hat{\mathbf{z}}_0 - \mathbf{z}_0\|^2$ (prediction loss)
    - $\mathcal{L}_\tau = \text{KL}(q(\tau_{t'}) \| p_\theta(\tau_{t'}|\mathbf{z}_t, \tau_t))$ (time prediction loss)
    - $\mathcal{L}_{\text{anchor}} = -\log p_\theta(y|\hat{\mathbf{z}}_0)$ (anchor loss, preventing embedding space collapse)

## Key Experimental Results

### Main Results (Machine Translation BLEU)

| Model | Type | Beam | IWSLT14 | WMT14 | WMT16 |
|------|------|------|---------|-------|-------|
| Absorbing | Discrete | 5 | 28.32 | 21.62 | 30.41 |
| SeqDiffuSeq | Continuous | 10 | 30.03 | 24.24 | 26.17 |
| Difformer | Continuous | 10 | 32.09 | 23.80 | 30.93 |
| DiNoiSer | Continuous | 50 | 31.61 | 24.26 | 31.08 |
| **NeoDiff** | **Hybrid** | **10** | **33.14** | **25.28** | **32.31** |

NeoDiff outperforms all baselines across all translation tasks, exceeding the maximum beam size of baselines while using a smaller beam size.

### Other Tasks (BLEU)

| Model | QQP (Paraphrase) | Quasar-T (Question Gen) | Wiki-Auto (Text Simplification) |
|------|-----------|-------------------|---------------------|
| Difformer (b=10) | 30.43 | 16.66 | 40.77 |
| **NeoDiff (b=10)** | **30.87** | **18.35** | **41.33** |

### Ablation Study

| Configuration | IWSLT BLEU | WMT14 BLEU |
|------|-----------|-----------|
| NeoDiff (Full) | 33.14 | 25.28 |
| w/o Poisson (Unified $\tau=t$) | 32.09 (-1.05) | 24.24 (-1.04) |
| w/o Time Predictor | 32.22 (-0.92) | 24.41 (-0.87) |
| w/o Bayesian Scheduler | 32.31 (-0.83) | 24.74 (-0.54) |
| w/o Variance Rescaling | 32.58 (-0.56) | — |

### Key Findings
- **Poisson diffusion is the most critical component**: removing it degrades the model to standard continuous diffusion (Difformer), causing a ~1.0 BLEU drop.
- **The contribution of the time predictor is second only to the Poisson process**: adaptive denoising rates are more effective than rigidly mirroring the forward process.
- **The necessity of variance rescaling**: it prevents degeneration into continuous diffusion when $s_{\max} \to \infty$.
- **Bayes-optimized time scheduling consistently yields additional gains**, showing that the optimal time allocation is task-specific.
- LLM evaluation (DeepSeek-V3) also confirms NeoDiff's comprehensive leading performance in accuracy, fluency, and creativity.

## Highlights & Insights
- **Theoretical elegance of the dual-time framework**: By generalizing the temporal variables, it unifies discrete, continuous, and hybrid diffusion models, providing a consolidated theoretical foundation for future research. This "parameterized unification" approach can be transferred to other domains requiring the reconciliation of distinct core methodologies.
- **Ingenious design of Poisson + variance rescaling**: The Poisson process naturally captures "jumps" (discreteness), while variance rescaling decouples the resolution from $s_{\max}$, resolving the tension between fine-grained control and diversity.
- **The pseudo-label strategy is a key technical innovation**: by mapping generation quality (reconstruction loss ranking) $\to$ inverse Poisson CDF $\to$ temporal labels, it translates "which tokens are well-generated" into "which tokens should finish denoising first", achieving generation quality-guided adaptive denoising.

## Limitations & Future Work
- The experiments are small-scale (at the Transformer-base level) and not validated on large models (such as LLaMA-based text diffusion).
- The generation quality still lags behind autoregressive models (Transformer with $b=5$ achieves 30.83 on QQP vs. NeoDiff with $b=10$ achieving 30.87).
- Bayes-optimized time scheduling requires extra evaluation overhead on the validation set.
- The time predictor increases model complexity and training costs.
- The synergy with more advanced continuous diffusion techniques (such as flow matching) remains unexplored.

## Related Work & Insights
- **vs. Difformer**: NeoDiff introduces token-level non-simultaneous diffusion on top of Difformer. It represents a fundamental improvement rather than an incremental optimization, boosting BLEU by 1-2 points.
- **vs. DiffuSeq-V2**: DiffuSeq-V2 blends discrete and continuous noise using [MASK] tokens, but falls short of true fine-grained token-level control. NeoDiff's Poisson process is more natural and theoretically sound.
- **vs. AR-Diffusion**: AR-Diffusion achieves semi-autoregressive generation via monotonically increasing noise but artificially restricts noise pattern design. NeoDiff's Poisson allocation is much more flexible.
- **vs. D3PM (Discrete Diffusion)**: Discrete diffusion lacks fine-grained transitions. NeoDiff preserves the discrete behavior of state jumps within a continuous space.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The dual-time framework is a highly original theoretical contribution that unifies the formalization of discrete and continuous diffusion.
- Experimental Thoroughness: ⭐⭐⭐⭐ Tested on 6 tasks with multiple baselines and detailed ablations, though model scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐ Mathematically rigorous with clear framework diagrams, though high notation density may affect readability.
- Value: ⭐⭐⭐⭐ Provides a unified theoretical framework and practical directions for improvement in the text diffusion domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rethinking Losses for Diffusion Bridge Samplers](../../NeurIPS2025/others/rethinking_losses_for_diffusion_bridge_samplers.md)
- [\[ICML 2025\] Optimal Sensor Scheduling and Selection for Continuous-Discrete Kalman Filtering with Auxiliary Dynamics](../../ICML2025/others/optimal_sensor_scheduling_and_selection_for_continuous-discrete_kalman_filtering.md)
- [\[ICCV 2025\] LayerTracer: Cognitive-Aligned Layered SVG Synthesis via Diffusion Transformer](../../ICCV2025/others/layertracer_cognitive-aligned_layered_svg_synthesis_via_diffusion_transformer.md)
- [\[ACL 2025\] Towards Text-Image Interleaved Retrieval](towards_text-image_interleaved_retrieval.md)
- [\[ICCV 2025\] SyncDiff: Synchronized Motion Diffusion for Multi-Body Human-Object Interaction Synthesis](../../ICCV2025/others/syncdiff_synchronized_motion_diffusion_for_multi-body_human-object_interaction_s.md)

</div>

<!-- RELATED:END -->
