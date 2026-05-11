---
title: >-
  [Paper Note] When to Lock Attention: Training-Free KV Control in Video Diffusion
description: >-
  [Video Generation] This paper proposes KV-Lock, a training-free framework that dynamically schedules background KV cache fusion ratios and CFG guidance strength based on diffusion hallucination detection…
tags:
  - "Video Generation"
date: 2026-05-08
content_hash: df986a5241c048fa
---

# When to Lock Attention: Training-Free KV Control in Video Diffusion

## Basic Information

- **Conference**: CVPR 2026
- **arXiv**: [2603.09657](https://arxiv.org/abs/2603.09657)
- **Code**: Not released
- **Area**: Image Generation / Video Editing
- **Keywords**: Training-Free Video Editing, KV Cache, Classifier-Free Guidance, Diffusion Hallucination Detection, DiT

## TL;DR

This paper proposes KV-Lock, a training-free framework that dynamically schedules background KV cache fusion ratios and CFG guidance strength based on diffusion hallucination detection, simultaneously ensuring background consistency and foreground generation quality in video editing.

## Background & Motivation

The core challenge in video editing lies in editing foreground targets while preserving the high fidelity of background scenes. Existing methods fall into two extremes:

**Full-frame information injection** (e.g., cross-attention manipulation, latent space interpolation): Edit effects tend to leak into background regions, causing background artifacts—particularly localized hallucinations in attributes such as color and pose.

**Rigid background locking** (fixed KV cache weights): Overly constrains the model's expressive capacity, degrading foreground generation quality.

Recent works (ProEdit, Follow-Your-Shape) leverage KV caches in DiT architectures to preserve backgrounds, but adopt fixed fusion weights or simple heuristic schedules, failing to adaptively balance foreground quality and background consistency. This raises a central question: **When should attention be locked to cached KVs, and when should the model be allowed to recompute attention patterns?**

The core insight of KV-Lock is that the hallucination detection metric of diffusion models (variance of the $\hat{x}_0$ trajectory) naturally corresponds to the diversity modulation function of CFG guidance scale—variance can thus serve as a unified scheduling signal, transforming heuristic hyperparameter tuning into principled variance-based decisions.

## Method

### Overall Architecture

KV-Lock is a plug-and-play training-free framework applicable to any pretrained DiT model. The overall pipeline consists of three stages:

1. **Encoding Stage**: A 3D VAE encodes the source video into a latent representation, and the editing mask is mapped into token space.
2. **Inversion Stage**: The source video undergoes forward diffusion; KV pairs from all Transformer layers are cached at each timestep.
3. **Denoising Stage**: A hallucination-aware scheduler dynamically fuses newly generated KVs with cached KVs (for background preservation), while dynamically adjusting CFG guidance strength (for foreground quality).

### Key Design 1: Token-Level KV Cache Locking

#### Latent Space Mask Encoding

The input video $\mathcal{V}_{\text{src}} \in \mathbb{R}^{3 \times F \times H \times W}$ is encoded by a 3D VAE (compression ratio $s = (4, 8, 8)$). The editing mask $\mathcal{M}$ is aligned with the VAE's temporal compression via temporal max-pooling:

$$m_0^{\text{latent},t} = \begin{cases} \max(\mathcal{M}_0), & t=0 \\ \max(\mathcal{M}_{[1+(t-1)s_t : 1+ts_t]}), & t \geq 1 \end{cases}$$

Max-pooling ensures that whenever any frame within a temporal window requires editing, the corresponding latent mask is marked as 1.

#### Token Space Projection

DiT patchifies the latent with patch size $p = (1, 2, 2)$, producing $N = T \cdot (h/p_h) \cdot (w/p_w)$ tokens. The mask is aligned to token space via 3D MaxPool:

$$m_{\text{token}} = \text{Flatten}(\text{MaxPool3D}(m_0^{\text{latent}}, \text{kernel}=p, \text{stride}=p)) \in \{0,1\}^N$$

This ensures that any token $i$ whose receptive field covers any masked pixel is labeled as a foreground token.

#### KV Cache Extraction

At each denoising timestep $t_k$, the noisy source latent is constructed as:

$$z_{t_k}^{\text{src}} = \sqrt{\bar{\alpha}_{t_k}} \mathcal{E}(\mathcal{V}_{\text{src}}) + \sqrt{1 - \bar{\alpha}_{t_k}} \epsilon$$

KV pairs are extracted from all $L=24$ Transformer layers via a forward pass:

$$\mathcal{K}_k^\ell = W_K^{(\ell)} h_{t_k}^{(\ell)}, \quad \mathcal{V}_k^\ell = W_V^{(\ell)} h_{t_k}^{(\ell)}, \quad \forall \ell \in \{1, \ldots, L\}$$

These cached KVs serve as "content anchors." The attention mechanism can be understood as differentiable retrieval: query $q_i$ computes similarity with all keys and aggregates values by weighting. When the KVs of background tokens are replaced with cached source video KVs, attention outputs are constrained to the manifold of source content, providing a deterministic reconstruction mechanism.

### Key Design 2: Hallucination-Aware Dynamic KV Fusion

Fully locking background KVs constrains the model's foreground generation capacity. A dynamic fusion rate $\alpha_k \in [0,1]$ is introduced to modulate KV locking strength according to denoising variance:

$$\alpha_k = \text{clamp}\left(\frac{\sigma_{\hat{x}_0^{(k)}}^2}{\tau}, 0, 1\right)$$

where $\tau = 0.01$ is the hallucination threshold. In the final $\kappa = 20$ sampling steps, weighted interpolation is applied to background tokens:

$$K_k^{\text{mix}} = m_{\text{token}} \odot K_k^{\text{new}} + (1 - m_{\text{token}}) \odot (\alpha_k \cdot \tilde{\mathcal{K}}_k^\ell + (1 - \alpha_k) \cdot K_k^{\text{new}})$$

$$V_k^{\text{mix}} = m_{\text{token}} \odot V_k^{\text{new}} + (1 - m_{\text{token}}) \odot (\alpha_k \cdot \tilde{\mathcal{V}}_k^\ell + (1 - \alpha_k) \cdot V_k^{\text{new}})$$

- Foreground tokens ($m_{\text{token}} = 1$): use newly generated KVs, retaining full degrees of freedom.
- Background tokens ($m_{\text{token}} = 0$): interpolate between cached and new KVs, with larger $\alpha_k$ imposing stronger locking.

Design motivation: high variance = model uncertainty in current region → stronger background constraints are needed to prevent hallucinations from spreading to the background.

### Key Design 3: Foreground Generation Guidance (CFG Optimization)

#### Adaptive Scaling Factor $s^*$

Standard CFG uses a fixed guidance strength $\omega$ to linearly interpolate conditional and unconditional noise predictions, but cannot compensate for noise estimation bias caused by model underfitting (especially in early denoising stages). An optimizable scaling factor $s$ is introduced:

$$\tilde{\epsilon}_\theta(x_t, t | y) = (1 - \omega) \cdot s \cdot \epsilon_\theta(x_t, t | \emptyset) + \omega \cdot \epsilon_\theta(x_t, t | y)$$

Objective: minimize $\|\tilde{\epsilon}_\theta - \epsilon_t\|_2^2$. Since the true noise $\epsilon_t$ is unobservable, an upper bound is derived via the triangle inequality; after eliminating $\epsilon_t$, a closed-form solution is obtained:

$$s^* = \frac{\langle \epsilon_\theta(x_t, t | y), \epsilon_\theta(x_t, t | \emptyset) \rangle}{\|\epsilon_\theta(x_t, t | \emptyset)\|_2^2 + \varepsilon}$$

Geometric interpretation: $s^*$ is the orthogonal projection of the conditional noise prediction vector onto the direction of the unconditional noise prediction, aligning the two noise estimates to reduce bias introduced by model underfitting. The computational overhead consists of only one inner product and one norm operation.

#### Hallucination-Aware Dynamic CFG Guidance

When hallucination risk is detected, the guidance strength is dynamically increased within a window $W = 10$:

$$\omega = \omega_0 \cdot \text{clamp}\left(\frac{\sigma_{\hat{x}_0^{(k)}}^2}{\tau}, 0, b\right)$$

where $b = 2$ is the clamp upper bound. Core insight: the CFG guidance strength $\omega$ itself modulates the diversity of generated samples, which naturally corresponds to the variance metric of hallucination detection—increasing $\omega$ when variance is high (high hallucination risk) constrains sample diversity, enforces conditional alignment, and stabilizes the diffusion process. Since all samples exhibit high variance in early diffusion stages, dynamic scheduling is only activated in the final $\kappa = 20$ steps.

### Key Design 4: Local Hallucination Detection

A sliding window is used to track the $\hat{x}_0$ variance in the foreground region as a hallucination proxy metric:

$$\hat{x}_0^{\text{masked},(k)} = \frac{1}{B} \sum_{b=1}^{B} \text{Flatten}(\hat{x}_0^{(k,b)} \odot m_0^{\text{latent}})$$

$$\sigma_{\hat{x}_0^{(k)}}^2 = \frac{1}{W-1} \sum_{i=t_k-W+1}^{t_k} (\hat{x}_0^{\text{masked},(i)} - \bar{\hat{x}}_0^{\text{masked}})^2$$

If variance exceeds threshold $\tau = 0.01$, hallucination risk is flagged. Key improvement: compared to global variance computation, **local variance** (mask region only) captures hallucination signals more sensitively—in the ablation study, global detection achieves Ave. 84.05% vs. local detection's 84.87%.

Theoretical basis: $\hat{x}_0$ of in-support samples converges to consistent representations (low variance) in late denoising stages; hallucinated samples exhibit persistent fluctuation (high variance) due to mode interpolation uncertainty.

## Experiments

### Experimental Setup

- **Base model**: Wan 2.1 (for CFG-Zero*, APG, ProEdit, KV-Lock), SD 2.1 (for FateZero, FLATTEN, TokenFlow)
- **Test data**: 52 samples (22 VACE-Benchmark + 30 web videos), 80–210 frames, resolution 480×832
- **Hardware**: A100 80GB GPU
- **Evaluation metrics**: VBench 5 dimensions (SC/BC/MS/AQ/IQ), background metrics (SSIM/PSNR), user study across 3 dimensions (PF/FC/VQ, 54 valid questionnaires)

### Main Results

| Method | SC↑ | BC↑ | AQ↑ | IQ↑ | Ave.↑ | SSIM↑ | PSNR↑ | User↑ | Time(s)↓ |
|------|------|------|------|------|-------|-------|-------|-------|---------|
| FateZero | 87.17 | 92.89 | 53.84 | 57.53 | 77.23 | 0.715 | 17.57 | 1.74 | 3.98 |
| FLATTEN | 92.90 | 95.54 | 53.24 | 59.41 | 79.71 | 0.772 | 19.30 | 2.60 | **1.14** |
| TokenFlow | 93.64 | 96.17 | 57.22 | 69.67 | 83.03 | 0.805 | 20.07 | 2.51 | 11.92 |
| CFG-Zero* | 93.80 | 95.99 | 61.22 | 71.04 | 84.16 | 0.911 | 26.65 | 4.01 | 5.58 |
| APG | 93.39 | 96.25 | 60.09 | 71.53 | 84.02 | 0.921 | 26.04 | 3.95 | 5.80 |
| ProEdit | 93.96 | 96.23 | 61.62 | **72.23** | 84.52 | 0.912 | 27.57 | 4.06 | 7.20 |
| VACE | 93.82 | 95.85 | 61.25 | 71.01 | 84.13 | 0.922 | **31.20** | 4.10 | 5.25 |
| **KV-Lock** | **94.56** | **96.92** | **62.15** | 72.18 | **84.87** | **0.931** | 31.04 | **4.21** | 7.39 |

### Ablation Study

| Configuration | SC↑ | BC↑ | MS↑ | Ave.↑ | SSIM↑ | PSNR↑ |
|------|------|------|------|-------|-------|-------|
| Variance KV scheduling only | 93.01 | 95.89 | 98.10 | 83.69 | 0.913 | 31.01 |
| CFG ω scheduling only | 93.32 | 93.89 | 97.72 | 83.46 | 0.922 | 29.84 |
| CFG s* scheduling only | 91.76 | 92.18 | 96.92 | 82.24 | 0.914 | 29.59 |
| CFG s* + ω scheduling | 93.28 | 95.71 | **98.63** | 84.05 | 0.913 | 30.55 |
| Fixed fusion α=0.5 | 90.33 | 93.97 | 97.51 | 82.58 | 0.918 | 30.90 |
| Global hallucination detection | 93.14 | 95.85 | 98.28 | 84.05 | 0.925 | 30.96 |
| **Full model** | **94.56** | **96.92** | 98.57 | **84.87** | **0.931** | **31.04** |

### Key Findings

1. **All three modules are essential**: The combination of KV scheduling, CFG ω scheduling, and CFG s* optimization is required to achieve optimal performance; using any component individually yields a notable gap (Ave. 82.24–83.69 vs. 84.87).
2. **Dynamic scheduling substantially outperforms fixed strategies**: Fixed α=0.5 achieves only 90.33% SC, far below dynamic scheduling's 94.56% (↓4.23%), demonstrating the core value of adaptive scheduling.
3. **Local hallucination detection outperforms global**: Global detection dilutes signals and causes missed detections; SSIM improves from 0.925 to 0.931 with local detection.
4. **Surpasses the training-based method VACE**: KV-Lock outperforms VACE on both VBench Ave. (84.87 vs. 84.13) and user study (4.21 vs. 4.10).
5. **Inference time cost**: 7.39s per iteration, with primary overhead from KV caching and sliding window computation, plus approximately 10GB additional GPU memory.

## Highlights & Insights

- **Theory-driven unified scheduling**: Variance → hallucination risk → simultaneously drives KV fusion rate and CFG strength; a single signal addresses two problems with an elegant and concise design.
- **Closed-form CFG scaling factor $s^*$**: Unobservable true noise is eliminated via an upper bound derivation, yielding an analytic solution as an orthogonal projection without iterative optimization.
- **Plug-and-play**: Training-free; seamlessly integrates into any pretrained DiT model (validated on Wan 2.1).
- **Comprehensive evaluation**: 52 samples × 5 VBench metrics + 2 background metrics + 3 user study dimensions + 54 valid questionnaires + detailed ablation study.

## Limitations & Future Work

- Inference speed is relatively slow (7.39s/iter); KV caching requires a full forward pass over the source video.
- Approximately 10GB additional GPU memory overhead.
- Relies on external mask input to separate foreground and background; automatic segmentation is not supported.
- The definition of diffusion hallucination is ambiguous; variance-based detection may miss non-variance-type hallucinations.
- Some baselines (FateZero/FLATTEN/TokenFlow) use SD 2.1 rather than Wan 2.1, introducing backbone discrepancy.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty** ⭐⭐⭐⭐: The hallucination detection-driven dynamic scheduling approach is novel; the theoretical connections among variance, CFG, and KV are well-argued.
- **Experimental Thoroughness** ⭐⭐⭐⭐: Metrics are comprehensive and ablations are detailed, but 52 samples is a relatively small test set and some baselines use inconsistent backbones.
- **Writing Quality** ⭐⭐⭐⭐: Mathematical derivations are rigorous, the framework diagram is intuitive, and the motivation is clearly articulated.
- **Value** ⭐⭐⭐: Training-free plug-and-play nature is a clear advantage, but slow inference and dependency on external masks limit practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models](../../ICCV2025/video_generation/leanvae_an_ultra-efficient_reconstruction_vae_for_video_diffusion_models.md)
- [\[ICCV 2025\] V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](../../ICCV2025/video_generation/vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)
- [\[ICCV 2025\] Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM](../../ICCV2025/video_generation/prompt-a-video_prompt_your_video_diffusion_model_via_preference-aligned_llm.md)
- [\[ICCV 2025\] EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models](../../ICCV2025/video_generation/efficientmt_efficient_temporal_adaptation_for_motion_transfer_in_text-to-video_d.md)
- [\[ICCV 2025\] DIVE: Taming DINO for Subject-Driven Video Editing](../../ICCV2025/video_generation/dive_taming_dino_for_subject-driven_video_editing.md)

</div>

<!-- RELATED:END -->
