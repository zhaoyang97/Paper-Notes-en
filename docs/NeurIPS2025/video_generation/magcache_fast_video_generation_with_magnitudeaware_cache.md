---
title: >-
  [Paper Note] MagCache: Fast Video Generation with Magnitude-Aware Cache
description: >-
  [NeurIPS 2025][Video Generation][Video diffusion model acceleration] This paper discovers that the magnitude ratio of residual outputs between adjacent timesteps in video diffusion models follows a universally monotonica…
tags:
  - "NeurIPS 2025"
  - "Video Generation"
  - "Video diffusion model acceleration"
  - "cache reuse"
  - "residual magnitude law"
  - "timestep skipping"
  - "inference acceleration"
date: 2026-05-08
content_hash: d48f7b2bb2dc1ae3
---

# MagCache: Fast Video Generation with Magnitude-Aware Cache

**Conference**: NeurIPS 2025
**arXiv**: [2506.09045](https://arxiv.org/abs/2506.09045)  
**Code**: [https://github.com/Zehong-Ma/MagCache](https://github.com/Zehong-Ma/MagCache)  
**Area**: Image/Video Generation / Model Compression
**Keywords**: Video diffusion model acceleration, cache reuse, residual magnitude law, timestep skipping, inference acceleration

## TL;DR

This paper discovers that the magnitude ratio of residual outputs between adjacent timesteps in video diffusion models follows a universally monotonically decreasing pattern across models and prompts — termed the "Unified Magnitude Law" — and proposes MagCache: a method that accurately models skip-step error accumulation via magnitude ratios, adaptively skips redundant timesteps and reuses cached outputs with only a single calibration sample, achieving 2.10–2.68× speedup on Open-Sora, CogVideoX, Wan 2.1, and HunyuanVideo while outperforming TeaCache and other existing methods across all three metrics of LPIPS, SSIM, and PSNR.

## Background & Motivation

**Background**: Diffusion models have achieved remarkable progress in video generation, evolving from early U-Net architectures (Stable Diffusion, ModelScope) to Transformer-based DiT architectures (Open-Sora, CogVideoX, Wan 2.1, HunyuanVideo), with continuously improving generation quality and temporal consistency. However, the core bottleneck remains inference speed — the generation process is inherently a serialized multi-step denoising procedure, and the larger the model, the higher the resolution, and the longer the video, the more prohibitive the latency. For example, generating a 5-second 480P video with Wan 2.1 on a single A800 GPU takes several minutes.

**Limitations of Prior Work**: Existing approaches for accelerating video diffusion models each have notable drawbacks. Distillation methods (e.g., VideLCM) require extensive retraining and additional data, making them costly and non-generalizable. Quantization methods (e.g., PTQ4DM) require careful calibration and suffer severe quality degradation at extremely low bit-widths. By contrast, **cache-based acceleration methods** (e.g., DeepCache, Δ-DiT, PAB) require no retraining and reduce computation by reusing intermediate features across adjacent timesteps — a much more lightweight approach. However, existing cache methods suffer from three key issues: (1) Methods such as DeepCache and PAB adopt **uniform strategies** (e.g., reusing every $N$ steps), completely ignoring the dynamic variation between different timesteps — some steps change minimally and can be safely skipped, while others change dramatically and must be fully computed; (2) TeaCache attempts an adaptive strategy by constructing a skip-step function based on time embedding differences or modulation input differences, but requires **70 carefully selected prompts for polynomial fitting**, incurring high calibration costs and risking overfitting to the calibration set, potentially failing on different prompts; (3) Methods such as FasterCache, DuCa, and TaylorSeer, while also adaptive, require caching large volumes of intermediate states, resulting in **enormous additional memory overhead** (TaylorSeer requires 40GB extra memory on Wan 2.1), making practical deployment difficult.

**Key Challenge**: The fundamental problem is the lack of a metric that is both **accurate** and **stable** (invariant across models and prompts) for measuring the difference between residual outputs of adjacent timesteps. If such a metric existed, one could precisely determine when it is safe to skip a timestep and when recomputation is necessary, thereby achieving an optimal balance between efficiency and quality.

**Goal**: This paper decomposes the above problem into three specific sub-problems: (1) identifying a residual variation metric that holds universally across models and prompts; (2) constructing an accurate skip-step error model based on this metric that supports cumulative error estimation over consecutive skipped steps; and (3) designing an adaptive caching strategy that maximizes the number of skipped steps while keeping error under control.

**Key Insight**: The authors begin from a simple but profound empirical observation — by analyzing the residual outputs (model-predicted velocity minus input) of diffusion models at different timesteps, they find that the **magnitude ratio** of residuals between adjacent timesteps (i.e., $\gamma_t = \text{mean}(\|\mathbf{r}_t\|_2 / \|\mathbf{r}_{t-1}\|_2)$) exhibits a highly regular monotonically decreasing trend. More critically, this trend is nearly identical across different models (Wan 2.1, Open-Sora) and different prompts, suggesting that the magnitude ratio is a natural, calibration-free indicator of redundancy.

**Core Idea**: Replace TeaCache's polynomial fitting — which requires extensive calibration — with the magnitude ratio, a stable indicator invariant across models and prompts, enabling adaptive cache-based acceleration with only a single calibration sample.

## Method

### Overall Architecture

The overall pipeline of MagCache is clear and intuitive. During inference of a video diffusion model, at each timestep $t$, the current residual $\mathbf{r}_t = \mathbf{v}_\theta(\mathbf{x}_t, t) - \mathbf{x}_t$ (model-predicted velocity minus input) is computed. The cumulative skip-step error $\mathcal{E}_t$ from the last cache refresh point $\hat{t}$ to the current step $t$ is then estimated using the pre-calibrated magnitude ratio curve $\{\gamma_i\}$. If the cumulative error is below threshold $\delta$ and the number of skipped steps does not exceed upper bound $K$, the cached residual $\mathbf{r}_{\hat{t}}$ is reused to skip the current step; otherwise, the residual is recomputed, the cache is updated, and the error counter is reset. The entire process requires no training and serves as a plug-and-play inference acceleration framework.

The input is a noisy video latent $\mathbf{x}_T$ and a text prompt; the output is the denoised video latent $\mathbf{x}_0$. The denoising process nominally requires $T$ full steps (e.g., 50 steps). MagCache reduces the number of actual computation steps to $T/2$ or fewer by skipping large numbers of redundant steps, achieving over 2× speedup.

### Key Designs

1. **Discovery and Verification of the Unified Magnitude Law**:

    - **Function**: Reveals an empirical law that holds universally across different video diffusion models and prompts — the magnitude ratio $\gamma_t$ of residuals between adjacent timesteps exhibits a monotonically decreasing trend.
    - **Mechanism**: The per-step magnitude ratio is defined as $\gamma_t = \text{mean}(\|\mathbf{r}_t\|_2 / \|\mathbf{r}_{t-1}\|_2)$, where $\mathbf{r}_t = \mathbf{v}_\theta(\mathbf{x}_t, t) - \mathbf{x}_t$ is the residual at step $t$. Through experiments, three key phenomena are observed: **First**, during the first 80% of timesteps, $\gamma_t$ decreases slowly and steadily from near 1, while the token-level cosine distance remains minimal (close to 0), indicating that differences between adjacent-step residuals arise primarily from **magnitude scaling** rather than **directional change**. This implies $\|\mathbf{r}_t - \mathbf{r}_{t-1}\| \approx |\|\mathbf{r}_t\| - \|\mathbf{r}_{t-1}\||$, meaning the magnitude ratio alone accurately characterizes residual differences. **Second**, during the final 20% of timesteps, $\gamma_t$ drops sharply and cosine distance increases significantly, indicating that the final generation phase involves rapid change and is unsuitable for skipping. **Third**, and most critically, the $\gamma_t$ curves produced by different prompts are nearly identical, with extremely low standard deviation, and the trend is fully consistent across models (Wan 2.1 vs. Open-Sora). This is the meaning of "unified": the law is model-agnostic and prompt-agnostic.
    - **Design Motivation**: This finding directly addresses TeaCache's core limitation — TeaCache requires 70 carefully selected prompts for polynomial fitting to predict residual differences, whereas MagCache, leveraging this unified law, requires only a single random prompt with one forward pass to obtain a magnitude ratio curve applicable to all scenarios. This reduces calibration cost from "70 prompts × full inference" to "1 prompt × 1 inference" and, more importantly, eliminates the risk of overfitting, since the law itself is prompt-invariant.

2. **Magnitude-Ratio-Based Precise Error Modeling**:

    - **Function**: Accurately estimates the cumulative error introduced by skipping consecutive timesteps, providing a reliable basis for adaptive decisions.
    - **Mechanism**: Let $\hat{t}$ denote the last cache-refresh timestep. After skipping steps $\hat{t}+1, \ldots, t$, the skip-step error is $\varepsilon_{\text{skip}}(\hat{t}, t) = 1 - \text{mean}(\|\mathbf{r}_t\|_2 / \|\mathbf{r}_{\hat{t}}\|_2) \approx 1 - \prod_{i=\hat{t}+1}^{t} \gamma_i$. This formula is elegant: due to the multiplicative chain structure of magnitude ratios, the error of skipping multiple consecutive steps can be expressed directly as the product of per-step magnitude ratios, without requiring independent predictions at each step as TeaCache does. A running cumulative error $\mathcal{E}_t = \mathcal{E}_{t-1} + \varepsilon_{\text{skip}}(\hat{t}, t)$ is maintained, initialized as $\mathcal{E}_{\hat{t}} = 0$. Since $\gamma_i$ exhibits extremely low standard deviation during the early and middle phases (Figure 1(b)), this cumulative estimate is tight and reliable.
    - **Design Motivation**: TeaCache performs poorly when skipping multiple consecutive steps because its polynomial fitting is essentially a step-by-step prediction, causing prediction errors to accumulate rapidly under consecutive skipping. MagCache's error model naturally supports multi-step skipping — the multiplicative structure of magnitude ratios ensures that even when skipping 3–4 steps, the error estimate remains accurate. This is the mathematical foundation for MagCache's significant advantage over TeaCache at the same speedup ratio.

3. **Adaptive Caching Strategy**:

    - **Function**: Based on the error model's estimates, adaptively determines whether to skip (reuse cache) or recompute at each timestep.
    - **Mechanism**: At each timestep $t$, MagCache checks two conditions: (1) cumulative error $\mathcal{E}_t \leq \delta$ (a user-defined total error threshold); (2) $t - \hat{t} \leq K$ (the number of skipped steps since the last refresh does not exceed the maximum skip length $K$). Only when both conditions are satisfied is the current step skipped and the cache $\mathbf{r}_{\hat{t}}$ reused; if either condition is violated, a reset is triggered: $\hat{t} \leftarrow t$, $\mathcal{E}_t \leftarrow 0$, $\mathbf{r}_t$ is recomputed, and the cache is updated. This constitutes a dual-safeguard mechanism. Following the practice of prior work, the first 20% of denoising steps are kept intact (not skipped), as these initial steps are critical for overall generation quality and empirically exhibit larger magnitude ratio variations.
    - **Design Motivation**: The constraint on maximum skip length $K$ may appear conservative, but is in fact a subtle engineering design — while the magnitude ratio error model is statistically accurate, it remains an approximation, and small modeling errors can still accumulate over long consecutive skip sequences. $K$ provides a "periodic correction" mechanism ensuring the model does not deviate too far from the true residual trajectory. In practice, $K=2$ corresponds to slow mode (quality-prioritized) and $K=4$ to fast mode (speed-prioritized), and users can cover the vast majority of speed-quality trade-off requirements through these two intuitive hyperparameters $(K, \delta)$.

### Loss & Training

MagCache is a completely **training-free** inference acceleration framework with no training involved. The only "calibration" step is a single forward pass of the diffusion model using one random prompt, recording the per-step magnitude ratios $\{\gamma_t\}_{t=1}^{T}$ — equivalent in time to generating a single video. The resulting magnitude ratio curve can be reused for all subsequent inferences on the same model without recalibration.

In terms of implementation, MagCache only needs to store one cached residual (~0.5GB extra memory), compared to TaylorSeer's 40GB extra memory and FasterCache/DuCa's tens of gigabytes. MagCache's memory overhead is minimal, making it a truly plug-and-play solution.

## Key Experimental Results

### Main Results

Experiments cover five mainstream generative models: Open-Sora 1.2 (video), Wan 2.1 1.3B (video), HunyuanVideo (video), CogVideoX 2B (video), and Flux (image). Comparison methods include PAB, T-GATE, Δ-DiT, FasterCache, DuCa, TeaCache, and TaylorSeer. Efficiency metrics use FLOPs and latency; quality metrics use LPIPS (lower is better), SSIM (higher is better), and PSNR (higher is better).

| Model / Method | FLOPs(P) | Speedup | Latency(s) | LPIPS↓ | SSIM↑ | PSNR↑ |
|---|---|---|---|---|---|---|
| **Open-Sora 1.2 (51 frames, 480P)** | | | | | | |
| Baseline (T=30) | 3.15 | 1× | 44.56 | - | - | - |
| TeaCache-slow | 2.40 | 1.40× | 31.69 | 0.1303 | 0.8405 | 23.67 |
| TeaCache-fast | 1.64 | 2.05× | 21.67 | 0.2527 | 0.7435 | 18.98 |
| **MagCache-slow** | 2.40 | 1.41× | 31.48 | **0.0827** | **0.8859** | **26.93** |
| **MagCache-fast** | 1.64 | **2.10×** | 21.21 | 0.1522 | 0.8266 | 23.37 |
| **Wan 2.1 1.3B (81 frames, 480P)** | | | | | | |
| Baseline (T=50) | 8.21 | 1× | 187.21 | - | - | - |
| TeaCache-slow | 5.25 | 1.59× | 117.20 | 0.1258 | 0.8033 | 23.35 |
| TeaCache-fast | 3.94 | 2.14× | 87.55 | 0.2412 | 0.6571 | 18.14 |
| **MagCache-slow** | 3.94 | **2.14×** | 87.27 | **0.1206** | **0.8133** | **23.42** |
| **MagCache-fast** | 3.11 | **2.68×** | 69.75 | 0.1748 | 0.7490 | 21.54 |
| **HunyuanVideo (129 frames, 540P)** | | | | | | |
| Baseline (T=50) | 45.93 | 1× | 1163 | - | - | - |
| TeaCache-slow | 27.56 | 1.63× | 712 | 0.1832 | 0.7876 | 23.87 |
| TeaCache-fast | 20.21 | 2.26× | 514 | 0.1971 | 0.7744 | 23.38 |
| **MagCache-slow** | 20.21 | **2.25×** | 516 | **0.0377** | **0.9459** | **34.51** |
| **MagCache-fast** | 18.37 | **2.63×** | 441 | **0.0626** | **0.9206** | **31.77** |
| **CogVideoX 2B (49 frames, 480P)** | | | | | | |
| Baseline (T=50) | 2.36 | 1× | 74.10 | - | - | - |
| TeaCache | 1.03 | 2.30× | 32.20 | 0.1221 | 0.8815 | 27.08 |
| **MagCache** | 0.99 | **2.37×** | 31.15 | **0.0787** | **0.9210** | **30.44** |

Several particularly noteworthy results: on HunyuanVideo, MagCache-slow achieves an LPIPS of only 0.0377 (vs. 0.1832 for TeaCache-slow, a 4.9× gap) and a PSNR of 34.51 (vs. 23.87 for TeaCache-slow, a gap exceeding 10dB), indicating that MagCache's error control advantage becomes more pronounced on larger models. On Wan 2.1, MagCache-fast achieves 2.68× speedup while outperforming TeaCache-fast across all quality metrics (LPIPS 0.1748 vs. 0.2412); moreover, MagCache-slow matches TeaCache-fast's speedup ratio (2.14×) while delivering quality even better than TeaCache-slow. MagCache-fast achieves a PSNR of 23.37 on Open-Sora, far exceeding TeaCache-fast's 18.98 — when PSNR falls below 20, visual distortion becomes highly apparent, indicating that TeaCache-fast is nearly unusable at high speedup ratios, whereas MagCache still maintains acceptable quality.

### Ablation Study

| Mode | K | δ | Speedup | LPIPS↓ | SSIM↑ | PSNR↑ |
|---|---|---|---|---|---|---|
| **MagCache-slow (Wan 2.1)** | | | | | | |
| slow | 2 | 0.06 | 2.0× | 0.0940 | 0.8383 | 24.57 |
| slow | 2 | 0.12 | 2.1× | 0.1053 | 0.8275 | 24.32 |
| slow | 2 | 0.03 | 1.9× | 0.0888 | 0.8427 | 24.68 |
| **MagCache-fast (Wan 2.1)** | | | | | | |
| fast | 4 | 0.06 | 2.4× | 0.1375 | 0.7749 | 22.34 |
| fast | 4 | 0.12 | 2.7× | 0.1625 | 0.7571 | 22.25 |
| fast | 4 | 0.03 | 2.0× | 0.1263 | 0.7828 | 22.51 |

Calibration prompt robustness experiment (Wan 2.1, slow mode):

| Calibration Strategy | Speedup | LPIPS↓ | SSIM↑ | PSNR↑ |
|---|---|---|---|---|
| Single random prompt (default) | 2.14× | 0.1206 | 0.8133 | 23.42 |
| Average over all 944 prompts | 2.14× | 0.1162 | 0.8163 | 23.52 |
| Most outlying prompt | 2.21× | 0.1209 | 0.8103 | 23.36 |

### Key Findings

- **Maximum skip length $K$ is the primary knob for speed-quality trade-off**: Increasing $K$ from 2 to 4 raises the speedup ratio from 2.0× to 2.4× (Wan 2.1, $\delta=0.06$), while LPIPS increases from 0.0940 to 0.1375. $K$ determines the "speed gear" (slow vs. fast), with $\delta$ serving as a fine-grained adjustment within the same gear.
- **Error threshold $\delta$ has monotonic and controllable influence**: Under the same $K$, decreasing $\delta$ consistently improves quality at the cost of speed, with a smooth and predictable relationship. For example, with $K=2$, reducing $\delta$ from 0.12 to 0.03 improves LPIPS from 0.1053 to 0.0888 while reducing speedup from 2.1× to 1.9×. A satisfactory configuration can typically be found in 1–2 adjustments.
- **Calibration is extremely robust**: Calibrating with a single random prompt, the average of 944 prompts, or even the most outlying prompt yields nearly identical performance (LPIPS difference < 0.005, PSNR difference < 0.2dB). This conclusively confirms the prompt-invariance of the Unified Magnitude Law — TeaCache's 70-prompt calibration is effectively unnecessary.
- **Dominant memory efficiency advantage**: MagCache requires only ~0.5GB extra memory, whereas TaylorSeer requires 40GB on Wan 2.1, and FasterCache and DuCa are noted as "not memory-efficient." This makes MagCache the only high-speedup solution that can realistically be deployed on consumer-grade GPUs.
- **Advantage is more pronounced on larger models**: On the larger HunyuanVideo, MagCache's quality advantage is even more substantial (LPIPS 0.0377 vs. 0.1832, PSNR 34.51 vs. 23.87), indicating that the Unified Magnitude Law holds — and is even more precise — on deeper and more complex models, which has important implications for deploying future large-scale models.

## Highlights & Insights

- **Paradigm shift from "fitting" to "discovering laws"**: TeaCache's approach is to "fit a prediction function from data to determine when to skip steps," which is essentially using statistical methods to approximate an unknown underlying regularity. MagCache directly identifies this underlying regularity (the Unified Magnitude Law), replacing data-driven fitting with first-principles reasoning. This is analogous to the difference between Newton discovering the law of universal gravitation and predicting celestial orbits via lookup tables — the former is not only more accurate and robust, but far more parsimonious.
- **Multiplicative chain error modeling as a key mathematical contribution**: The skip-step error formula $\varepsilon_{\text{skip}} \approx 1 - \prod \gamma_i$ is deceptively simple, yet it elegantly solves the problem of estimating error when skipping multiple consecutive steps. TeaCache's step-by-step predictions suffer from error explosion under consecutive skipping, whereas MagCache's multiplicative structure naturally ensures accuracy even with multi-step skipping, making the $K=4$ fast mode feasible.
- **"Direction-invariant, magnitude-decreasing" physical intuition is transferable**: This paper reveals the internal structure of residuals in the denoising process of diffusion models — during the first 80% of steps, residual directions are nearly fixed while only magnitudes decrease. This finding is not limited to cache acceleration; it also hints that the "denoising trajectory" learned by diffusion models possesses a certain low-dimensional structure, which could inform other acceleration directions such as model compression and step distillation in future work.
- **The $(K, \delta)$ dual-parameter design provides an intuitive control interface**: $K$ selects the "gear" (slow/fast) while $\delta$ provides fine-tuning — this hierarchical parameter design allows users to find a satisfactory configuration in 1–2 experiments, making it particularly well-suited for integration into interactive tools such as ComfyUI.

## Limitations & Future Work

- **Validation limited to video/image generation tasks**: The authors acknowledge in the conclusion that the Unified Magnitude Law and MagCache's effectiveness have so far only been validated on video and image generation models, with no extension to other diffusion model tasks (e.g., 3D generation, audio generation, image editing). Whether the magnitude law holds in non-generative tasks (e.g., diffusion models used for discriminative tasks) remains to be explored.
- **First 20% of timesteps cannot be accelerated**: Following prior work, MagCache keeps the first 20% of denoising steps intact. While there are empirical justifications (these steps are critical for overall quality and exhibit larger magnitude ratio variations), this imposes a hard upper limit on the theoretical speedup. Identifying safe skip-step methods for the initial phase would enable further gains.
- **The Unified Magnitude Law is empirical, not theoretically proven**: The monotonic decrease of magnitude ratios and prompt-invariance are currently empirical observations without theoretical explanation. Why do the residual magnitudes of diffusion models exhibit this pattern? Is it an intrinsic property of the flow matching training objective, or a consequence of the model architecture? If future models with entirely different training paradigms violate this law, MagCache may fail.
- **Fixed calibration curves do not adapt to dynamic scenarios**: MagCache uses a fixed magnitude ratio curve to make decisions for all inferences. While robustness across different prompts is demonstrated experimentally, the introduction of additional control mechanisms during inference (e.g., ControlNet, IP-Adapter) may alter residual dynamics, potentially rendering the fixed curve suboptimal. A promising direction is lightweight online adaptation — computing magnitude ratios in real time during inference rather than relying on offline cached curves.
- **Joint use with step-reduction methods is underexplored**: Compatibility with distillation and low-precision computation is preliminarily verified in the appendix, but joint use with step-reduction methods such as consistency models and progressive distillation has not been explored. If the diffusion model itself requires only 4–8 steps (e.g., LCM), the skip-step gains from MagCache may diminish substantially.

## Related Work & Insights

- **vs. TeaCache**: TeaCache is MagCache's most direct competitor and primary target for improvement. TeaCache fits a polynomial using 70 curated prompts to predict the skip-step function, while MagCache, after discovering the Unified Magnitude Law, requires only a single calibration sample. The core distinction is "fitting vs. discovering a law" — TeaCache approximates a function, while MagCache exploits an intrinsic regularity. This yields comprehensive advantages across three dimensions: calibration cost (1 vs. 70 prompts), robustness (prompt-invariant vs. potentially overfitted), and multi-step skip accuracy (multiplicative chain vs. step-by-step prediction). On HunyuanVideo, MagCache-slow's LPIPS is only 1/5 of TeaCache-slow's (0.0377 vs. 0.1832), a striking gap.
- **vs. DeepCache / PAB**: These represent uniform caching strategies. They reuse caches at fixed intervals (e.g., every $N$ steps), without accounting for the dynamic variation between different timesteps. On Open-Sora, PAB-slow achieves an LPIPS of 0.1471 versus MagCache-slow's 0.0827, demonstrating that adaptive strategies offer a fundamental advantage over uniform ones. These methods are conceptually simple but have a clear performance ceiling — quality degrades severely as the speedup ratio increases.
- **vs. FasterCache / DuCa / TaylorSeer**: Although these methods also implement adaptive caching, they require storing large volumes of intermediate features, leading to prohibitive memory overhead (TaylorSeer requires 40GB). MagCache caches only a single residual (~0.5GB), offering an order-of-magnitude advantage in memory efficiency — a critical factor for consumer-GPU deployment.
- **Connection to the broader direction of diffusion model acceleration**: MagCache belongs to the category of "training-free inference-time acceleration," complementing rather than replacing distillation (requires retraining and data) and quantization (requires calibration and incurs precision loss). The appendix preliminarily validates compatibility with low-precision computation, and these techniques can be stacked in future work to achieve further efficiency gains.

## Rating

- Novelty: ⭐⭐⭐⭐ The Unified Magnitude Law is a concise and elegant discovery, though the idea of "using residual variation metrics to decide skip steps" was already pioneered by TeaCache; MagCache's contribution is primarily identifying a better metric.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 5 models (4 video + 1 image) and 7+ comparison methods, with comprehensive ablations over $K$, $\delta$, and calibration strategies, demonstrating consistent advantages on every model.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from empirical discovery → theoretical modeling → method design → experimental validation is exceptionally clear and fluent; Figure 1's visualization intuitively and powerfully supports the central claim.
- Value: ⭐⭐⭐⭐⭐ Extremely high practical value — plug-and-play, training-free, minimal memory overhead, 2×+ speedup, controllable quality degradation; open-sourced on GitHub with support for the latest models including Wan 2.2 and FramePack.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Generating, Fast and Slow: Scalable Parallel Video Generation with Video Interface Networks](../../ICCV2025/video_generation/generating_fast_and_slow_scalable_parallel_video_generation_with_video_interface.md)
- [\[CVPR 2026\] FastLightGen: Fast and Light Video Generation with Fewer Steps and Parameters](../../CVPR2026/video_generation/fastlightgen_fast_and_light_video_generation_with_fewer_steps_and_parameters.md)
- [\[ICLR 2026\] Geometry-aware 4D Video Generation for Robot Manipulation](../../ICLR2026/video_generation/geometry-aware_4d_video_generation_for_robot_manipulation.md)
- [\[AAAI 2026\] FilmWeaver: Weaving Consistent Multi-Shot Videos with Cache-Guided Autoregressive Diffusion](../../AAAI2026/video_generation/filmweaver_weaving_consistent_multi-shot_videos_with_cache-guided_autoregressive.md)
- [\[ICLR 2026\] Target-Aware Video Diffusion Models](../../ICLR2026/video_generation/target-aware_video_diffusion_models.md)

</div>

<!-- RELATED:END -->
