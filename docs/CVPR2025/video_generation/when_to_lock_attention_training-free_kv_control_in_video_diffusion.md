---
title: >-
  [Paper Note] When to Lock Attention: Training-Free KV Control in Video Diffusion
description: >-
  [CVPR2025][Video Generation][video editing] Proposes KV-Lock, a training-free video editing framework based on diffusion hallucination detection. It dynamically schedules the KV cache fusion ratio and the CFG guidance scale to preserve background consistency while enhancing foreground generation quality.
tags:
  - "CVPR2025"
  - "Video Generation"
  - "video editing"
  - "KV cache"
  - "training-free"
  - "diffusion hallucination detection"
  - "classifier-free guidance"
  - "DiT"
date: 2026-05-08
content_hash: ff8434a2b96979cc
---

# When to Lock Attention: Training-Free KV Control in Video Diffusion

**Conference**: CVPR2025  
**arXiv**: [2603.09657](https://arxiv.org/abs/2603.09657)  
**Code**: To be confirmed  
**Area**: Video Generation  
**Keywords**: video editing, KV cache, training-free, diffusion hallucination detection, classifier-free guidance, DiT

## TL;DR
Proposes KV-Lock, a training-free video editing framework based on diffusion hallucination detection. It dynamically schedules the KV cache fusion ratio and the CFG guidance scale to preserve background consistency while enhancing foreground generation quality.

## Background & Motivation
**Key Challenge in Video Editing**: There is a fundamental conflict between maintaining background consistency and enhancing foreground quality—injecting global information leads to background artifacts, while rigidly locking the background restricts the foreground generation capability.

**Limitations of Training-based Methods**: Training-based methods require massive computational resources and time to adapt to new data distributions, hindering flexible deployment.

**Limitations of Prior Training-free Methods**: Inversion-based methods usually only provide coarse-grained control, and editing effects easily leak into the background areas.

**Defects of Fixed KV Fusion**: Completely locking the KV cache or using fixed fusion weights significantly degrades the quality of foreground generation.

**The Question of "When to Lock Attention"**: A deeper but unexplored problem—when cached KVs should be locked, and when the model should be allowed to recompute attention patterns.

**Intrinsic Connection between CFG and Hallucination**: The CFG guidance scale in diffusion models regulates generation diversity, which is naturally correlated with the variance metric in hallucination detection.

## Method

### Overall Architecture
KV-Lock comprises three synergistic components: (1) Token-level KV cache locking, (2) hallucination detection-driven foreground CFG optimization, and (3) a dynamic scheduler. Workflow: Encoding $\rightarrow$ Inversion (caching source KV) $\rightarrow$ Denoising (hallucination detection-driven dynamic fusion + CFG adjustment) $\rightarrow$ Decoding.

### Key Designs

**1. Token-level KV Cache Extraction and Locking**
- Project the binary mask from pixel space to latent space and then to token space, utilizing 3D max-pooling to align with patchification operations.
- For each denoising step $t_k$ of the source video, run forward propagation and cache all KV pairs of all $L=24$ layers.
- During editing, use the token-level mask to distinguish the foreground (using new KVs) and background (fusing cached KVs), achieving precise region-level control.

**2. Hallucination Detection-Driven Dynamic KV Fusion**
- Track the local variance of the predicted clean latent $\hat{x}_0$ during denoising as a hallucination metric.
- Calculate variance using a sliding window $W=10$: $\sigma^2_{\hat{x}_0^{(k)}}$.
- Dynamic fusion rate: $\alpha_k = \text{clamp}(\sigma^2 / \tau, 0, 1)$, $\tau=0.01$.
- Strengthen KV locking when the risk of hallucination is high, and loosen it when the risk is low—translating heuristic hyperparameter tuning into principled, variance-driven decisions.

**3. Foreground CFG Optimization**
- Introduce an optimizable scaling factor $s \in \mathbb{R}_{>0}$ to calibrate the unconditional noise prediction.
- Closed-form analytical solution: $s^* = \frac{\langle \epsilon_\theta(x_t,t|y), \epsilon_\theta(x_t,t|\emptyset) \rangle}{\|\epsilon_\theta(x_t,t|\emptyset)\|_2^2 + \varepsilon}$
- Hallucination-aware dynamic CFG: $\omega = \omega_0 \cdot \text{clamp}(\sigma^2/\tau, 0, b)$, $b=2$.
- Core Insight: CFG regulates generation diversity $\leftrightarrow$ variance quantifies hallucination risk, presenting a natural correspondence.

### Loss & Training
- Training-free method; no additional training loss is required.
- The derivation of the scaling factor $s^*$ is based on minimizing the upper bound between the CFG-guided noise and the ground-truth noise.

## Key Experimental Results

### Main Results

| Method | VBench Ave.↑ | BG SSIM↑ | BG PSNR↑ | User Study Ave.↑ |
|------|-------------|----------|----------|-----------------|
| FateZero | 77.23% | 0.7151 | 17.57 | 1.74 |
| TokenFlow | 83.03% | 0.8050 | 20.07 | 2.51 |
| CFG-Zero* | 84.16% | 0.9107 | 26.65 | 4.01 |
| ProEdit | 84.52% | 0.9116 | 27.57 | 4.06 |
| VACE | 84.13% | 0.9218 | **31.20** | 4.10 |
| **KV-Lock** | **84.87%** | **0.9309** | 31.04 | **4.21** |

KV-Lock comprehensively leads in VBench average (84.87%), background fidelity (SSIM 0.9309), and user studies (4.21).

### Ablation Study

| Configuration | VBench Ave.↑ | BG SSIM↑ | BG PSNR↑ |
|------|-------------|----------|----------|
| Variance-only KV Scheduling | 83.69% | 0.9129 | 31.01 |
| CFG $\omega$-only Scheduling | 83.46% | 0.9217 | 29.84 |
| Fixed Fusion $\alpha=0.5$ | 82.58% | 0.9175 | 30.90 |
| Global Hallucination Detection | 84.05% | 0.9254 | 30.96 |
| **Full Model** | **84.87%** | **0.9309** | **31.04** |

Key Findings:
- The synergy of the three components (KV scheduling + CFG $\omega$ + CFG $s^*$) yields the best performance.
- **Local hallucination detection** (foreground mask region) significantly outperforms global detection.
- Fixed fusion weight ($\alpha=0.5$) severely restricts performance.

### Key Findings
- The variance signal of hallucination detection is most informative during the late stages of denoising—variance is high for all samples in the early stages.
- Performing dynamic scheduling only in the final $\kappa=20$ sampling steps is effective.
- Inference time is 7.39s, comparable with ProEdit (7.20s), but far superior to TokenFlow (11.92s).

## Highlights & Insights
1. **Solid Theoretical Foundation**: Converts "when to lock attention" from heuristic tuning into a principled decision based on variance, supported by a clear theoretical rationale.
2. **Connection between CFG and Hallucination**: Discovers the natural correspondence between the CFG guidance scale controlling generation diversity and the variance metric of diffusion hallucination, elegantly unifying two seemingly independent mechanisms.
3. **Closed-form Optimization**: The scaling factor $s^*$ has an analytical solution (orthogonal projection), requiring no additional training or iterative optimization.
4. **Plug-and-Play**: The training-free framework can be integrated into any DiT video model, offering high practicality.

## Limitations & Future Work
1. Requires masks to specify editing regions, which limits applicability in fully automated editing scenarios.
2. The hallucination threshold $\tau=0.01$ and upper bound $b=2$ are manually set hyperparameters that may need adjustment across different tasks/models.
3. Validated only on DiT models like CogVideoX, with unknown applicability to UNet architectures.
4. Background PSNR is slightly lower than VACE (31.04 vs 31.20), indicating that dynamic fusion might introduce minor deviations under extreme conditions.

## Related Work & Insights
- **vs ProEdit**: ProEdit simply decouples foreground/background attention and caches KVs, but lacks a dynamic scheduling mechanism.
- **vs CFG-Zero\***: CFG-Zero corrects unmatched initial values of noise prediction, while KV-Lock dynamically adjusts CFG intensity.
- **vs VACE**: VACE shows strong background fidelity but is slightly weaker in foreground quality and user preferences.
- **Insight**: Diffusion hallucination detection can be used not only for quality evaluation but also as a dynamic control signal—this concept can be extended to image editing, 3D generation, and other scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ — The design of hallucination detection-driven dynamic KV/CFG scheduling is novel, with an elegant theoretical connection.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers VBench + user studies + detailed ablations, balancing both quantitative and qualitative aspects.
- Writing Quality: ⭐⭐⭐⭐ — Rigorous theoretical derivation, though numerous formulas make the methodology section somewhat long.
- Value: ⭐⭐⭐⭐ — Highly practical training-free plug-and-play framework, and the concept of hallucination-driven scheduling has broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models](../../ICLR2026/video_generation/frame_guidance_training-free_guidance_for_frame-level_control_in_video_diffusion.md)
- [\[CVPR 2025\] LongDiff: Training-Free Long Video Generation in One Go](longdiff_training-free_long_video_generation_in_one_go.md)
- [\[CVPR 2026\] SwitchCraft: Training-Free Multi-Event Video Generation with Attention Controls](../../CVPR2026/video_generation/switchcraft_training-free_multi-event_video_generation_with_attention_controls.md)
- [\[ICML 2026\] Attention Sparsity is Input-Stable: Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering](../../ICML2026/video_generation/attention_sparsity_is_input-stable_training-free_sparse_attention_for_video_gene.md)
- [\[ICLR 2026\] LightCtrl: Training-free Controllable Video Relighting](../../ICLR2026/video_generation/lightctrl_training-free_controllable_video_relighting.md)

</div>

<!-- RELATED:END -->
