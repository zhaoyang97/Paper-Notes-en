---
title: >-
  [Paper Note] Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization
description: >-
  [ICML 2026][Video Generation][Autoregressive Video Diffusion] QVG is a training- and finetuning-free KV-cache quantization framework for autoregressive video diffusion. It performs token smoothing via semantic-aware clus…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Autoregressive Video Diffusion"
  - "KV-Cache"
  - "2-bit Quantization"
  - "Spatiotemporal Redundancy"
  - "Residual Quantization"
date: 2026-05-08
content_hash: 2d614312d5adf73a
---

# Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization

**Conference**: ICML 2026  
**arXiv**: [2602.02958](https://arxiv.org/abs/2602.02958)  
**Code**: Available (paper notes Website + GitHub)  
**Area**: Video Generation / KV-Cache Quantization / Model Compression  
**Keywords**: Autoregressive Video Diffusion, KV-Cache, 2-bit Quantization, Spatiotemporal Redundancy, Residual Quantization

## TL;DR
QVG is a training- and finetuning-free KV-cache quantization framework for autoregressive video diffusion. It performs token smoothing via semantic-aware clustering and progressively compresses residuals in multiple stages. On LongCat-Video/HY-WorldPlay/Self-Forcing, it reduces KV memory to 1/7 of the original, with end-to-end latency overhead <4%. At 2 bits, it significantly outperforms LLM quantization baselines such as KIVI/QuaRot in quality.

## Background & Motivation

**Background**: Video diffusion models are shifting from "bidirectional attention + short-clip denoising" to a **chunk-by-chunk autoregressive + causal attention + KV-cache** generation paradigm (e.g., CausVid, Self-Forcing, HY-WorldPlay), aiming to support long-term, streaming, and interactive video generation. The key dependency brought by autoregression is the KV-cache: early-frame K/V must reside in GPU memory to avoid recomputation.

**Limitations of Prior Work**: KV-cache memory grows almost linearly with the number of frames, quickly saturating the GPU. For example, LongCat-Video generating a 5-second 480p video requires about 38K latent tokens, corresponding to 34 GB KV-cache, exceeding a single RTX 5090. HY-WorldPlay-8B cannot run on a 4090. Worse, **short context is not only an efficiency bottleneck but also a capability bottleneck**—KV length directly determines long-term consistency of identity/layout/motion, and even top-tier long video systems can only sustain about 60 seconds.

**Key Challenge**: LLM-style KV quantization (KIVI / KVQuant / QuaRot / RotateKV) **fails outright** for video: video KV exhibits highly heterogeneous value distributions across both token and channel dimensions—max|K|~$10^2$, max|V|~$10^3$, and outlier channels vary by token. Symmetric per-group quantization $X_\text{INT}=\lfloor X/S\rceil$, with scale $S=\max(|X|)/(2^{b-1}-1)$, directly absorbs the token's maximum value, causing quantization error $\mathbb E[|x-\hat x|]\propto S$ to explode.

**Goal**: (i) Upgrade KV-cache quantization from LLM-style "generic smoothing" to handle the heterogeneous distributions in video; (ii) Maintain video quality even at extremely low bit-widths like 2-bit; (iii) Require no training or finetuning.

**Key Insight**: The authors observe that video KV exhibits **strong spatiotemporal redundancy**—neighboring frames at the same spatial patch, and neighboring patches within the same frame, have high latent token cosine similarity. Moreover, video content naturally supports **progressive encoding** (coarse-to-fine), similar to SVC streaming codecs. These two properties suggest two opportunities: similar tokens can share a centroid (subtracting it flattens the heterogeneous distribution before quantization), and residuals can be further refined in multiple stages.

**Core Idea**: Use **k-means to cluster similar tokens and subtract the centroid** to obtain low-magnitude, quantization-friendly residuals (Semantic-Aware Smoothing), then apply **Progressive Residual Quantization** to compress residuals in multiple coarse-to-fine stages, replacing the LLM-style outlier handling paradigm with a video-style redundancy exploitation paradigm.

## Method

### Overall Architecture
QVG is integrated into the KV-cache write path of any autoregressive video diffusion model without training or finetuning: it processes KV chunk-by-chunk, and for each chunk, (1) runs k-means to cluster $N$ tokens into $C$ groups, computing centroid $C_i$ for each; (2) subtracts the centroid from each token to obtain residual $R_i$; (3) applies standard per-group symmetric quantization (INT2 or INT4) to the residuals; (4) to further reduce error, recursively applies "residual smoothing + quantization" for several rounds (Pro version). During dequantization, $S_X\cdot X_{\text{INT}}+C_i$ is added back to approximate K/V. All centroids are stored in BF16 (very small). Algorithm and system are co-optimized on GPU to keep latency <4%.

### Key Designs

1. **Semantic-Aware Smoothing**:

    - **Function**: Transforms the "channel-token chaotic large-magnitude" distribution of video KV into a "low-magnitude near-zero" distribution, naturally reducing rounding error in low-bit quantization.
    - **Mechanism**: For a chunk with $N=HWT_c$ tokens, k-means clusters them into $C$ groups $\{\mathcal G_i\}$, each with centroid $C_i\in\mathbb R^d$. Each token subtracts its centroid: $\mathbf R_i=\mathbf X_{\mathcal G_i}-C_i$, and the residuals undergo symmetric per-group quantization. Since tokens in the same group have similar hidden representations, "channels that are all outliers" are absorbed by the centroid, greatly reducing the maximum value in the residuals. Experiments show Key cache quantization error drops by ~6.9×, Value cache by ~2.6×.
    - **Design Motivation**: LLM quantization (KIVI's per-token, QuaRot's rotation) assumes "channel outliers are consistent across all tokens," which does not hold for video—video tokens correspond to different spatial regions and motion patterns, and outliers drift by token. Clustering achieves **local homogenization** by leveraging spatiotemporal redundancy, aligning better with data characteristics than forced rotation.

2. **Progressive Residual Quantization**:

    - **Function**: Further compresses residuals in multiple stages on top of Semantic-Aware Smoothing, diluting quantization error and enabling flexible quality-memory tradeoff.
    - **Mechanism**: The first stage smooths and quantizes the original KV to obtain $\hat X_1$; the dequantized residual $\Delta_1=X-\hat X_1$ undergoes Semantic-Aware Smoothing + quantization again to get $\hat\Delta_1$, repeated for $L$ stages, yielding $\hat X=\hat X_1+\hat\Delta_1+\cdots+\hat\Delta_{L-1}$. Each additional stage further refines the residual, akin to multi-resolution encoding in SVC. The number of stages controls the "quality vs. compression ratio" curve.
    - **Design Motivation**: Single-stage quantization has a hard lower bound of $S_X/2$ for rounding error; multi-stage reduces error geometrically, leveraging the natural hierarchy of "coarse structure + high-frequency residuals" in video. In the paper, QVG-Pro (multi-stage) achieves PSNR 30.4 at INT2, QVG (single-stage) achieves 28.7, both far surpassing baselines.

3. **Algorithm-System Co-Design**:

    - **Function**: Implements the above steps efficiently on GPU, ensuring training-free integration into the autoregressive inference pipeline with controllable latency.
    - **Mechanism**: k-means is performed at chunk granularity, centroids stored in BF16; quantization/dequantization is fused with the attention kernel; 2-bit uses packed INT representation. The paper reports end-to-end latency overhead <4%.
    - **Design Motivation**: If KV quantization slows down inference, it loses its purpose; preserving intra-chunk parallelism and minimizing dequantization overhead are engineering keys for deployment on consumer GPUs like RTX 4090/5090.

### Loss & Training
Completely training-free, with no gradient updates; the only hyperparameters are: number of clusters $C$, number of residual stages $L$, and quantization bit-width $b$.

## Key Experimental Results

### Main Results
On LongCat-Video-13B, HY-WorldPlay-8B, and Self-Forcing, using BF16 full precision as reference, compared with RTN/KIVI/QuaRot.

| Model | Setting | Method | Compression Ratio | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|
| LongCat-Video | INT2 480p | RTN | 6.40× | 20.87 | 0.719 | 0.203 |
| LongCat-Video | INT2 480p | KIVI | 6.40× | 20.32 | 0.719 | 0.208 |
| LongCat-Video | INT2 480p | QuaRot | 6.40× | 21.57 | 0.759 | 0.171 |
| LongCat-Video | INT2 480p | **QVG-Pro** | 4.97× | **30.38** | **0.935** | **0.048** |
| LongCat-Video | INT2 480p | **QVG** | 6.94× | **28.72** | **0.909** | **0.065** |
| LongCat-Video | INT4 480p | QuaRot | 3.55× | 33.74 | 0.960 | 0.033 |
| LongCat-Video | INT4 480p | **QVG-Pro** | 3.05× | **37.10** | **0.977** | **0.024** |
| HY-WorldPlay | INT2 480p | QuaRot | 6.40× | 25.21 | 0.738 | 0.205 |
| HY-WorldPlay | INT2 480p | **QVG-Pro** | < 6.40× | 29+ | High | Low |

At the extreme INT2 bit-width, all LLM baselines have PSNR ≤ 25, while QVG achieves 28–30; at INT4, QVG-Pro even surpasses BF16's near-lossless performance on some metrics (>37 PSNR).

### Ablation Study

| Configuration | Explanation | Effect |
|---|---|---|
| Full QVG-Pro | k-means smoothing + multi-stage residual | Optimal |
| Semantic-Aware Smoothing only | Single-stage, single centroid subtraction | Moderate gain |
| Progressive Residual only | No clustering, direct residual recursion | Cannot handle channel outliers, fails at 2-bit |
| Naive per-group quantization (RTN) | No smoothing or residual | Fails at 2-bit |

Key/Value cache quantization errors are reduced by ~6.9× / ~2.6×, respectively.

### Key Findings
- **First practical 2-bit video KV quantization**: Previous best LLM quantization achieves PSNR 20-25 at INT2 for video, QVG reaches 28-30.
- **HY-WorldPlay-8B runs on single RTX 4090 for the first time**: Previously, KV memory exceeded capacity, making deployment impossible.
- **Self-Forcing uses longer context within fixed memory**: Quality even surpasses original BF16 default KV budget, turning "quantization saves memory" into "quantization enables longer context and thus better quality."

## Highlights & Insights
- **Diagnoses "video KV specificity" down to token-channel dimension**: Rather than settling for the vague conclusion that "LLM quantization performs poorly," the authors pinpoint the quantization "root causes"—max|K|~$10^2$, max|V|~$10^3$, token-dependent outlier drift—and design targeted smoothing. This "diagnose before remedy" research paradigm is highly reusable.
- **k-means + centroid subtraction = data-driven local outlier absorption**: Compared to QuaRot's fixed Hadamard rotation for global distribution smoothing, content-based clustering for local homogenization naturally fits video spatiotemporal redundancy, marking a leap from "generic" to "domain-aware" methods.
- **"Memory compression → longer context → quality improvement" flywheel**: In LLMs, quantization is usually about "compression-accuracy Pareto"; QVG reveals that in video generation, quantization unlocks longer KV, boosting long-term consistency—a capability metric—thus injecting new freedom into long video research from the memory dimension.

## Limitations & Future Work
- k-means clustering is sensitive to the number of tokens per chunk, and the number of clusters $C$ is a manually tuned hyperparameter; adaptive clustering strategies remain to be explored.
- Increasing the number of residual stages $L$ sacrifices compression ratio; dynamically selecting stages for the "quality-memory" Pareto curve still requires empirical tuning.
- The paper focuses on 480p and chunk-level autoregressive models; pixel-level autoregressive video generation (e.g., token-by-token) has not yet been validated.
- Evaluation mainly uses reference-based metrics such as PSNR/SSIM/LPIPS, with limited discussion on the impact on "generation diversity."

## Related Work & Insights
- **vs KIVI / KVQuant**: Effective for LLMs, but token-channel heterogeneity in video renders outlier handling ineffective; QVG localizes and resolves heterogeneity via clustering.
- **vs QuaRot / RotateKV**: Rotational transforms smooth global distributions but cannot handle token-dependent outlier drift; QVG uses data-driven centroids instead of fixed rotations.
- **vs Vector Quantization (PQCache, CommVQ)**: Uses codebooks to represent tokens; QVG adopts "subtract centroid + quantize residual," with centroids as anchors rather than full codebooks, making it lighter and training-free.
- **vs StreamingT2V / WorldMem / FramePack**: These design memory mechanisms at the algorithmic level; QVG expands existing KV budgets from the system side, making them complementary and stackable.

## Rating
- Novelty: ⭐⭐⭐⭐ First to apply "semantic clustering and centroid subtraction" for video KV smoothing, combined with progressive residuals, with clear originality in the combination.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three SOTA autoregressive video models, INT2/INT4 multi-bit, multiple baseline comparisons, consumer GPU deployment validation, and comprehensive quantization error and quality curves.
- Writing Quality: ⭐⭐⭐⭐⭐ Smooth logical flow from "system-algorithm coupled KV bottleneck" to problem diagnosis, opportunity identification, method design, and experimental validation.
- Value: ⭐⭐⭐⭐⭐ Directly enables long video generation, consumer GPU deployment, and context expansion, with immediate engineering impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] When to Lock Attention: Training-Free KV Control in Video Diffusion](../../CVPR2026/video_generation/when_to_lock_attention_training-free_kv_control_in_video_diffusion.md)
- [\[NeurIPS 2025\] MagCache: Fast Video Generation with Magnitude-Aware Cache](../../NeurIPS2025/video_generation/magcache_fast_video_generation_with_magnitudeaware_cache.md)
- [\[ICLR 2026\] QuantSparse: Comprehensively Compressing Video Diffusion Transformer with Model Quantization and Attention Sparsification](../../ICLR2026/video_generation/quantsparse_comprehensively_compressing_video_diffusion_transformer_with_model_q.md)
- [\[ICCV 2025\] Long Context Tuning for Video Generation](../../ICCV2025/video_generation/long_context_tuning_for_video_generation.md)
- [\[CVPR 2026\] Free-Lunch Long Video Generation via Layer-Adaptive O.O.D Correction](../../CVPR2026/video_generation/free-lunch_long_video_generation_via_layer-adaptive_ood_correction.md)

</div>

<!-- RELATED:END -->
