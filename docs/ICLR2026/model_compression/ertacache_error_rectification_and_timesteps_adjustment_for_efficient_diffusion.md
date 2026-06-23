---
title: >-
  [Paper Note] ERTACache: Error Rectification and Timesteps Adjustment for Efficient Diffusion
description: >-
  [ICLR 2026][Model Compression][Diffusion Model] ERTACache formalizes "quality degradation caused by cache acceleration" into two categories: **feature shift error** and **step amplification error**. It employs a "three-piece set" consisting of offline strategy calibration, trajectory-aware step adjustment, and closed-form residual linearization correction to suppres
tags:
  - ICLR 2026
  - Model Compression
  - Diffusion Model
  - Inference Acceleration
  - Video Generation
  - Flow Matching
date: 2026-05-08
content_hash: e69fc2b6b5b6f4f8
---
# ERTACache: Error Rectification and Timesteps Adjustment for Efficient Diffusion

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=InvyBiYcK5](https://openreview.net/forum?id=InvyBiYcK5)  
**Code**: TBD  
**Area**: Diffusion Model Acceleration / Feature Caching  
**Keywords**: Diffusion models, feature caching, inference acceleration, video generation, error rectification, Flow Matching  

## TL;DR
ERTACache formalizes "quality degradation caused by cache acceleration" into two categories: **feature shift error** and **step amplification error**. It employs a "three-piece set" consisting of offline strategy calibration, trajectory-aware step adjustment, and closed-form residual linearization correction to suppress these errors simultaneously, achieving over 2× acceleration with near-lossless quality on video and image diffusion models.

## Background & Motivation

**Background**: Inference for diffusion models (especially DiT-based video generation) is extremely slow—generating an 81-frame 480p video with Wan2.1 takes about 200 seconds on an A800, primarily because huge Transformers must be executed for every denoising step. Among various acceleration methods, **feature caching** is most practical: it exploits the high similarity of features between adjacent timesteps to reuse intermediate outputs across steps, thereby skipping entire forward passes without **requiring retraining** or modification of the diffusion backbone.

**Limitations of Prior Work**: Existing caching methods have distinct drawbacks. Methods that cache internal Transformer states (AdaCache, BAC, DiTFastAttn) suffer from memory growth quadratic to granularity, leading to OOM in long videos. "Online heuristic" methods like TeaCache rely on input-dependent thresholds to dynamically predict whether to reuse, but there is a **systematic deviation between predicted reuse quality and true reconstruction error**. Experimental measurements (Fig 2a) show that the $\ell_1$ distance predicted by TeaCache deviates significantly from the true value in late timesteps, making reuse decisions unreliable.

**Key Challenge**: More aggressive caching (reusing more steps) leads to greater acceleration, but errors **accumulate and amplify along the ODE trajectory**, causing collapsed image quality. How to suppress cumulative errors under aggressive reuse is the core difficulty of cache acceleration.

**Core Idea**: **Understand the error first, then apply targeted solutions**. The paper makes a key observation: although diffusion trajectories are input-dependent, the **cache reuse patterns and errors remain highly consistent across different prompts** (the blue line in Fig 2a varies little across steps). This implies that a general, offline-optimized caching strategy is sufficient to approach the optimum without needing online prediction. Based on this, the authors formalize cache error into two categories and design three complementary components to correct them.

## Method

### Overall Architecture
ERTACache operates within the Flow-Matching framework: the forward pass is linear interpolation $x_t=(1-t)x_0+t\cdot x_T$, and sampling integrates along the learned ODE using the Euler method: $x_{i-1}=x_i+\Delta t_i\cdot v_\theta(x_i,t)$. The cached component is the **residual** $\tilde r=v_\theta(x_{i+1},t)-x_{i+1}$. If step $i$ is skipped, the output is reconstructed as $\tilde v_i=x_i+\tilde r$, saving a forward pass. The workflow consists of three stages: **(a) Offline strategy calibration** uses a small batch of calibration samples to find the cache set $S$ of "reusable steps"; **(b) Timestep adjustment** dynamically scales the integration step size on reused steps to correct trajectory drift; **(c) Inference-time error rectification** uses pre-calculated closed-form coefficients to compensate for additive errors introduced by caching.

```mermaid
flowchart LR
    A[Offline Full Inference<br/>Record GT Residual r_gt] --> B[Threshold Search<br/>ℓ1rel < λ?]
    B --> C[Obtain Reusable Step Set S]
    C --> D[Inference: Reuse r̃ if t∈S<br/>Otherwise Recompute and Refresh]
    D --> E[Trajectory-Aware Step Adjustment<br/>Δt_i = Δt_c · φ_i]
    E --> F[Closed-form Error Correction<br/>v̂_i = ṽ_i - ε_i]
    F --> G[High-Fidelity Accelerated Sampling Output]
```

### Key Designs

**1. Formal Decomposition of Cache Error: Splitting "Quality Degradation" into Two Correctable Sources.** This is the theoretical foundation. Let the cache output be $\tilde v_i=v_i+\varepsilon_i$, where $\varepsilon_i$ is the additive error. Substituting $\tilde v_{i+1}$ into the Euler update, the single-step trajectory deviation is $\delta_i=\tilde x_i-x_i=\Delta t_{i+1}\cdot\varepsilon_{i+1}$. If $m$ steps are continuously cached from step $i$, the deviation accumulates as $\delta_{i-m}=\sum_{k=0}^{m-1}\Delta t_{i-k}\cdot\varepsilon_{i-k}$. This formula clearly identifies two error sources: $\varepsilon_{i-k}$ is the **Feature Shift Error**—the difference between cached and true features; the weight $\Delta t_{i-k}$ **linearly amplifies** the error, forming the **Step Amplification Error**. The subsequent three components are designed to reduce $\varepsilon$ and control the scale of $\Delta t$.

**2. Offline Strategy Calibration: Replacing "Online Guessing" with "Offline Searching" via Residual Profiling.** Since error patterns are consistent across prompts, online heuristics are unnecessary. The authors run full inference on a small calibration set and record the ground truth residuals $r_{gt}(x_i,t)$. For a range of candidate thresholds $\lambda$, the relative $\ell_1$ error $\ell_{1rel}(x_i,t)=\frac{\|\tilde r_{cali}-r_{cali}(x_i,t)\|_1}{\|r_{gt}(x_i,t)\|_1}$ is used to determine if the cached residual is sufficiently close to ground truth. If $\ell_{1rel}<\lambda$, the step is reused; otherwise, it is recomputed. Scanning different $\lambda$ yields a globally effective reusable step set $S=\{s_0,\dots,s_c\}$. Smaller $\lambda$ values result in fewer reuses and higher fidelity, while larger $\lambda$ values increase speed at the cost of potential quality loss. This stage directly reduces Feature Shift Error caused by unnecessary reuse.

**3. Trajectory-Aware Timestep Adjustment: Correcting Integration Steps During Cache Reuse.** Naive reuse assumes strict adherence to the original ODE trajectory, but fixed-step reuse causes significant trajectory drift (Fig 2b). The authors introduce a correction coefficient $\phi_i\in[0,1]$ to dynamically scale step sizes: $\Delta t_i=\Delta t_c$ for non-reused steps and $\Delta t_i=\Delta t_c\cdot\phi_i$ for reused steps, where $\phi_i=\mathrm{clip}\!\left(1-\frac{\|\tilde v_i-v_i\|_1}{\|v_i-v_{i+1}\|_1},0,1\right)$. Essentially, the more a cache output deviates from ground truth, the more its integration step size is compressed to mitigate error amplification. After each update, the remaining budget is reallocated as $\Delta t_c=\frac{1-\sum_{j=0}^{i}\Delta t_j}{1-i/T}$ to ensure the trajectory still aligns with the target path within $[0,1]$. This component targets Step Amplification Error.

**4. Closed-form Residual Linearization Correction: Compensating Each Cache Step with a Pre-calculated Error Term.** Since the additive error $\varepsilon_i=\tilde v_i-v_i$ is difficult to predict directly due to prompt structure, the authors approximate it using a lightweight linearization model $\varepsilon_i=\sigma(K_i\cdot\tilde v_i+B_i)$ (where $\sigma$ is sigmoid), fitting $K_i,B_i$ under MSE loss. A key trick is applying a first-order Taylor expansion to the sigmoid $\sigma(K_i\tilde v_i+B_i)\approx\frac14(K_i\tilde v_i+B_i)+\frac12$, allowing for a **closed-form solution** to the least squares problem: $K_i\approx 4\cdot\frac{S_{v_i\varepsilon_i}}{S_{v_iv_i}}$ and $B_i\approx 4(\bar\varepsilon_i-\frac12)-4K_i\bar v_i$ (where $S$ denotes variance/covariance terms). These coefficients are **pre-calculated** on a small dataset and used during inference to correct cache outputs with <0.5% overhead.

## Key Experimental Results

### Main Results (Video Generation, VBench)

| Model / Method | Speedup↑ | VBench↑ | LPIPS↓ | SSIM↑ | PSNR↑ |
|---|---|---|---|---|---|
| **Open-Sora 1.2** (T=30) | 1× | 79.22% | - | - | - |
| TeaCache-slow | 1.55× | 79.28% | 0.1316 | 0.8415 | 23.62 |
| **ERTACache-slow** (λ=0.1) | 1.55× | **79.36%** | **0.1006** | **0.8706** | **25.45** |
| TeaCache-fast | 2.25× | 78.48% | 0.2511 | 0.7477 | 19.10 |
| **ERTACache-fast** (λ=0.18) | **2.47×** | **78.64%** | **0.1659** | **0.8170** | **22.34** |
| **CogVideoX** (T=50) | 1× | 80.18% | - | - | - |
| TeaCache | 2.92× | 79.00% | 0.2057 | 0.7614 | 20.97 |
| **ERTACache-fast** (λ=0.3) | **2.93×** | 78.79% | **0.1012** | **0.8702** | **26.44** |
| **Wan2.1-1.3B** (T=50) | 1× | 81.30% | - | - | - |
| TeaCache | 2.00× | 76.04% | 0.2913 | 0.5685 | 16.17 |
| ProfilingDiT | 2.01× | 76.15% | 0.1256 | 0.7899 | 22.02 |
| **ERTACache** (λ=0.08) | **2.17×** | **80.73%** | **0.1095** | **0.8200** | **23.77** |

For image generation (Flux-dev 1.0, T=30): ERTACache (λ=0.6) achieves 1.86× acceleration with CLIP 0.9534 / LPIPS 0.3029 / SSIM 0.8962 / PSNR 20.51, significantly outperforming TeaCache at a similar ratio (0.9065 / 0.4427 / 0.7445 / 16.48).

### Ablation Study (Component Accumulation)

| Wan2.1-1.3B | VBench↑ | LPIPS↓ | SSIM↑ | PSNR↑ |
|---|---|---|---|---|
| Uniform Cache | 79.35% | 0.5041 | 0.4058 | 13.76 |
| + Offline Policy | 80.59% | 0.1477 | 0.7738 | 22.09 |
| + Time Adjustment | **80.89%** | 0.1267 | 0.7988 | 22.94 |
| + Error Rectification | 80.73% | **0.1095** | **0.8200** | **23.77** |

### Key Findings
- **Complementary Components**: Offline strategy provides the foundational recovery (LPIPS drops from 0.50 to 0.15), timestep adjustment optimizes structure and temporal consistency (PSNR/SSIM gain of +0.95/+0.023 on Flux), and error rectification maximizes detail preservation (PSNR +0.83).
- **Efficiency of Offline Strategy**: On Wan2.1, the offline policy improves VBench by 1.24% over uniform caching, suggesting that "searching for the right steps to reuse" significantly reduces information loss.
- **Superiority on Wan2.1**: At similar or higher speedup ratios compared to TeaCache, VBench score is nearly preserved (80.73% vs. base 81.30%), whereas TeaCache drops to 76.04%.
- **Negligible Overhead**: Components such as error rectification are pre-calculated, adding **<0.5%** to inference time.

## Highlights & Insights
- **"Explain then Solve" Paradigm**: By strictly decomposing cache error into two interpretable categories and mapping components to them, the method is more transparent and explainable than purely heuristic threshold-tuning.
- **Insight: Offline > Online**: The discovery that cache errors are highly consistent across prompts allows for replacing "per-inference online prediction" with "single-time offline global calibration," improving both stability and efficiency.
- **Engineering Elegance of Closed-form Solutions**: Using a first-order Taylor expansion for sigmoid reduces error fitting to a least-squares closed-form solution, enabling error correction during inference with almost zero cost.
- **Strong Generalization**: The method effectively supports four backbones (Open-Sora, CogVideoX, Wan2.1, Flux-dev) across both video and image tasks.

## Limitations & Future Work
- **Dependency on Offline Calibration Set**: Both the strategy $S$ and correction coefficients are fitted on a small calibration set. If the deployment distribution shifts significantly from the calibration distribution, the "consistency across prompts" assumption may weaken.
- **Precision Ceiling of Linearization**: Modeling errors with first-order Taylor expansions of sigmoid may fail to fit highly non-linear error structures; the paper lacks an in-depth analysis of this approximation error.
- **Hyperparameter $\lambda$ Tuning**: Different models and speed targets require different $\lambda$ values (ranging from 0.08 to 0.6), and an automated selection mechanism is missing.
- **Orthogonality**: While potentially orthogonal to token-level methods (FastCache, TokenCache), the paper suggests they can be combined but does not provide extensive joint experiments.

## Related Work & Insights
- **vs. TeaCache**: TeaCache uses online heuristic prediction, where prediction error diverges in later steps; ERTACache uses offline calibration + explicit error modeling, proving more reliable.
- **vs. AdaCache/BAC/DiTFastAttn**: These methods cache internal states or attention maps, causing quadratic memory growth and OOM risks; ERTACache only caches residuals, making it memory-friendly.
- **vs. LazyDiT/ICC**: LazyDiT requires additional training loss for regularization (training overhead), and ICC uses low-rank calibration; ERTACache explicitly models residual errors without retraining while balancing precision and memory.
- **Insight**: Formalizing "acceleration-induced error" into categories and designing targeted solutions is more robust than single heuristics; the offline calibration + closed-form correction approach can be transferred to other training-free acceleration scenarios like quantization or distillation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The combination of two-category error decomposition and closed-form residual linearization correction is novel, moving cache acceleration from "threshold heuristics" to "interpretable targeted correction."
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers four backbones, video + image tasks, multiple baselines, and comprehensive component ablations. However, it lacks deeper analysis of automated λ selection and robustness to distribution shift.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear derivation of error decomposition, effective illustrations, and a logical narrative that maps formulas directly to motivations.
- **Value**: ⭐⭐⭐⭐ — Training-free, plug-and-play, memory-friendly, and 2× acceleration with near-lossless quality provides immediate practical value for deploying video diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Plug-and-Play Fidelity Optimization for Diffusion Transformer Acceleration via Cumulative Error Minimization](plug-and-play_fidelity_optimization_for_diffusion_transformer_acceleration_via_c.md)
- [\[ICML 2026\] FedSDR: Federated Self-Distillation with Rectification](../../ICML2026/model_compression/fedsdr_federated_self-distillation_with_rectification.md)
- [\[CVPR 2026\] Otil: Accelerating Diffusion Model Inference via Communication-Efficient Multi-GPU Parallelism](../../CVPR2026/model_compression/otil_accelerating_diffusion_model_inference_via_communication-efficient_multi-gp.md)
- [\[ICLR 2026\] ES-dLLM: Efficient Inference for Diffusion Large Language Models by Early-Skipping](es-dllm_efficient_inference_for_diffusion_large_language_models_by_early-skippin.md)
- [\[ICLR 2026\] RAPID$^3$: Tri-Level Reinforced Acceleration Policies for Diffusion Transformer](rapid3_tri-level_reinforced_acceleration_policies_for_diffusion_transformer.md)

</div>

<!-- RELATED:END -->
