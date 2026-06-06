---
title: >-
  [Paper Note] Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization
description: >-
  [ICML 2026][Video Generation][Autoregressive Video Diffusion] QVG is a training-free KV-cache quantization framework for autoregressive video diffusion. By employing semantic-aware clustering for token smoothing and prog…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Autoregressive Video Diffusion"
  - "KV-Cache"
  - "2-bit Quantization"
  - "Spatiotemporal Redundancy"
  - "Residual Quantization"
date: 2026-05-08
content_hash: 5adbbbaff7b6fe25
---

# Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization

**Conference**: ICML 2026  
**arXiv**: [2602.02958](https://arxiv.org/abs/2602.02958)  
**Code**: Available (Website + GitHub as marked in the paper)  
**Area**: Video Generation / KV-Cache Quantization / Model Compression  
**Keywords**: Autoregressive Video Diffusion, KV-Cache, 2-bit Quantization, Spatiotemporal Redundancy, Residual Quantization

## TL;DR
QVG is a training-free KV-cache quantization framework for autoregressive video diffusion. By employing semantic-aware clustering for token smoothing and progressive residual multi-stage compression, it reduces KV memory consumption to 1/7 on LongCat-Video, HY-WorldPlay, and Self-Forcing. It maintains an end-to-end latency overhead of <4% and significantly outperforms LLM quantization baselines like KIVI and QuaRot in 2-bit scenarios.

## Background & Motivation

**Background**: Video diffusion models are shifting from "bidirectional attention + short clip denoising" to a **chunk-by-chunk generation paradigm** based on **autoregressive + causal attention + KV-cache** (e.g., CausVid, Self-Forcing, HY-WorldPlay). This aims to support long-duration, streaming, and interactive video generation. The key dependency introduced by the autoregressive nature is the KV-cache: K/V tensors from early frames must reside in GPU memory to avoid recomputation.

**Limitations of Prior Work**: KV-cache memory consumption grows nearly linearly with the number of frames, quickly exhausting GPU resources. For instance, generating a 5-second 480p video with LongCat-Video requires approximately 38K latent tokens, corresponding to 34 GB of KV-cache, which exceeds the capacity of a single RTX 5090. HY-WorldPlay-8B cannot even run on an RTX 4090. Worse, **short context is not only an efficiency bottleneck but also a capability bottleneck**—the KV length directly determines long-term consistency in identity, layout, and motion. Even top-tier industrial long-video systems can currently only support up to about 60 seconds.

**Key Challenge**: Standard LLM KV quantization methods (KIVI, KVQuant, QuaRot, RotateKV) **fail catastrophically** when applied to video. Video KV tensors exhibit highly heterogeneous numerical distributions across both token and channel dimensions—$\max|K|\sim 10^2$ and $\max|V|\sim 10^3$, with outlier channels varying per token. Symmetric per-group quantization $X_{\text{INT}}=\lfloor X/S\rceil$, where the scale $S=\max(|X|)/(2^{b-1}-1)$, is forced to accommodate the extreme maximum values of the entire token, causing the quantization error $\mathbb E[|x-\hat x|]\propto S$ to explode.

**Goal**: (i) Upgrade KV-cache quantization from LLM-style "general smoothing" to handle the heterogeneous distributions of video; (ii) maintain video quality even under extreme 2-bit low-bitrate conditions; (iii) achieve this without requiring training or fine-tuning.

**Key Insight**: The authors observe that video KV pairs possess **strong spatiotemporal redundancy**—neighboring spatial patches in adjacent frames and neighboring patches within the same frame show high cosine similarity in latent tokens. Furthermore, video content naturally supports **progressive encoding** (coarse-to-fine), similar to SVC streaming. These properties present two opportunities: similar tokens can share a centroid (subtracting it flattens the heterogeneous distribution), and residuals can be further refined through multiple stages.

**Core Idea**: Utilize **k-means to group similar tokens and subtract centroids** to obtain low-amplitude, quantization-friendly residuals (Semantic-Aware Smoothing), followed by **Progressive Residual Quantization** (PRQ) to compress residuals in a multi-stage coarse-to-fine manner. This replaces the LLM-style outlier processing paradigm with a video-style redundancy utilization paradigm.

## Method

### Overall Architecture
QVG integrates into the KV-cache write path of any autoregressive video diffusion model without fine-tuning. It processes KV pairs chunk-by-chunk. For each chunk: (1) k-means clustering partitions $N$ tokens into $C$ groups and calculates centroids $C_i$; (2) tokens subtract their respective centroids to obtain residuals $R_i$; (3) residuals undergo standard per-group symmetric quantization (INT2 or INT4); (4) to further reduce error, the "smoothing + quantization" process is applied recursively for several rounds (Pro version). During dequantization, $S_X\cdot X_{\text{INT}}+C_i$ is calculated to approximate K/V. Centroids are stored in BF16 (minimal overhead), and the algorithm is co-optimized with the system on the GPU to maintain <4% latency.

### Key Designs

1.  **Semantic-Aware Smoothing**:
    - **Function**: Transforms the high-amplitude "channel-token chaotic" distribution of video KV into a "low-amplitude, near-zero" residual distribution, naturally reducing rounding errors in low-bit quantization.
    - **Mechanism**: For $N=HWT_c$ tokens in a chunk, k-means produces $C$ groups $\{\mathcal G_i\}$ and centroids $C_i\in\mathbb R^d$. Tokens are adjusted: $\mathbf R_i=\mathbf X_{\mathcal G_i}-C_i$. Residuals then enter symmetric per-group quantization. Since tokens in the same group have similar latent representations, the "channels that happen to be outliers" are absorbed by the centroid, drastically shrinking the maximum values in the residuals. Experiments show Key cache quantization error drops by $\sim$6.9× and Value cache by $\sim$2.6×.
    - **Design Motivation**: LLM quantization (e.g., per-token in KIVI, rotation in QuaRot) assumes "channel outliers are consistent across all tokens," which is false for video. Video tokens correspond to different spatial regions and motion patterns, causing outliers to shift per token. **Local homogenization** via clustering is a direct byproduct of exploiting spatio-temporal redundancy, making it more suited to the data's nature than forced distribution rotation.

2.  **Progressive Residual Quantization**:
    - **Function**: Compresses residuals across multiple stages on top of Semantic-Aware Smoothing, further thinning quantization errors and supporting flexible quality-memory tradeoffs.
    - **Mechanism**: The first round yields $\hat X_1$ via smoothed quantization. The dequantization residual $\Delta_1=X-\hat X_1$ is then processed via another round of Semantic-Aware Smoothing and quantization to get $\hat\Delta_1$. This repeats for $L$ rounds. The final approximation is $\hat X=\hat X_1+\hat\Delta_1+\dots+\hat\Delta_{L-1}$. Each stage refines the residual, similar to multi-resolution encoding in SVC.
    - **Design Motivation**: Scaling single-stage quantization is limited by a hard lower bound of rounding error ($S_X/2$). Multi-stage quantization allows errors to decay geometrically by utilizing the natural hierarchy of "coarse structure + high-frequency residuals" in video. Internally, QVG-Pro (multi-stage) achieves a PSNR of 30.4 at INT2, far exceeding baselines.

3.  **Algorithm-System Co-design**:
    - **Function**: Implements the above steps on GPUs to ensure training-free embedding into autoregressive inference pipelines with controllable latency.
    - **Mechanism**: k-means is performed at the chunk level with BF16 centroids. Quantization/dequantization is fused with attention kernels. 2-bit representations use packed INT. The paper reports <4% end-to-end latency overhead.
    - **Design Motivation**: KV quantization is meaningless if it slows down inference. Maintaining intra-chunk parallelism and minimizing dequantization overhead are critical for deploying this method on consumer GPUs like the RTX 4090/5090.

### Loss & Training
The method is completely training-free with no gradient updates. Hyperparameters include the number of clusters $C$, number of residual stages $L$, and quantization bits $b$.

## Key Experimental Results

### Main Results
Evaluated on LongCat-Video-13B, HY-WorldPlay-8B, and Self-Forcing, using BF16 full precision as a reference against RTN, KIVI, and QuaRot.

| Model | Setting | Method | Compression | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|---|
| LongCat-Video | INT2 480p | RTN | 6.40× | 20.87 | 0.719 | 0.203 |
| LongCat-Video | INT2 480p | KIVI | 6.40× | 20.32 | 0.719 | 0.208 |
| LongCat-Video | INT2 480p | QuaRot | 6.40× | 21.57 | 0.759 | 0.171 |
| LongCat-Video | INT2 480p | **QVG-Pro** | 4.97× | **30.38** | **0.935** | **0.048** |
| LongCat-Video | INT2 480p | **QVG** | 6.94× | **28.72** | **0.909** | **0.065** |
| LongCat-Video | INT4 480p | QuaRot | 3.55× | 33.74 | 0.960 | 0.033 |
| LongCat-Video | INT4 480p | **QVG-Pro** | 3.05× | **37.10** | **0.977** | **0.024** |

At the extreme INT2 bit-depth, LLM baselines all result in PSNR $\le 25$, while QVG reaches 28–30. At INT4, QVG-Pro even achieves near-lossless performance (>37 PSNR), occasionally exceeding BF16 on certain metrics.

### Ablation Study

| Configuration | Explanation | Effect |
|---|---|---|
| Full QVG-Pro | k-means smoothing + multi-stage residual | Optimal |
| Semantic-Aware Smoothing only | Single stage, subtraction of centroid | Moderate gain |
| Progressive Residual only | No clustering, recursive residuals | Fails to resolve channel outliers; collapses at 2-bit |
| Naive per-group (RTN) | No smoothing or residuals | Collapses at 2-bit |

Key and Value cache quantization errors are reduced by $\sim$6.9× and $\sim$2.6× respectively.

### Key Findings
- **First viable 2-bit video KV quantization**: Previous LLM-based quantization reached only PSNR 20-25; QVG elevates this to 28-30.
- **First deployment of HY-WorldPlay-8B on a single RTX 4090**: Previously undeployable due to KV memory overflow.
- **Context expansion in Self-Forcing**: Given a fixed memory budget, QVG allows for a longer context, which actually results in higher quality than the default BF16 with more limited context.

## Highlights & Insights
- **Precise Diagnosis at the Token-Channel Dimension**: The authors did not stop at the conclusion that "LLM quantization performs poorly." They identified the root causes—$\max|K|\sim 10^2$, $\max|V|\sim 10^3$, and token-dependent outlier drift—and designed smoothing specifically for them. This "diagnosis before treatment" approach is highly replicable.
- **K-means + Centroid Subtraction = Data-driven Local Outlier Absorption**: Unlike QuaRot's fixed Hadamard rotation for global distribution smoothing, clustering by content to achieve local homogenization naturally fits video redundancy. It represents a successful transition from "general methods" to "domain-aware methods."
- **The "Memory-Context-Quality" Flywheel**: In LLMs, quantization is usually just a "compression-accuracy" trade-off. QVG reveals that in video generation, quantization unlocks longer KV-caches, which improves **capability metrics** like long-term consistency. This provides a new degree of freedom in memory for long video research.

## Limitations & Future Work
- k-means clustering is sensitive to the number of tokens in a chunk; the cluster count $C$ is currently a manually tuned hyperparameter. Adaptive clustering strategies remain unexplored.
- Increasing the number of residual stages $L$ sacrifices the compression ratio; dynamic selection of the "quality-memory" Pareto curve needs empirical tuning.
- The paper focuses on 480p and chunk-level autoregressive models; it has not yet validated pixel-level autoregressive generation (e.g., token-by-token).
- Evaluation primarily relies on reference-based metrics (PSNR/SSIM/LPIPS); the impact on "generation diversity" is not deeply discussed.

## Related Work & Insights
- **vs KIVI / KVQuant**: Effective for LLMs, but token-channel heterogeneity in video renders outlier handling ineffective. QVG "localizes" and resolves heterogeneity via clustering.
- **vs QuaRot / RotateKV**: Rotational transforms smooth global distributions but cannot handle token-dependent outlier drift. QVG replaces fixed rotations with data-driven centroids.
- **vs Vector Quantization (PQCache, CommVQ)**: Uses codebooks to represent tokens. QVG uses "centroid subtraction + residual quantization," where centroids act as anchors rather than a full codebook, making it more lightweight and training-free.
- **vs StreamingT2V / WorldMem / FramePack**: These design memory mechanisms at the algorithmic level; QVG expands existing KV budgets at the system level. The two are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ High originality in combining semantic clustering for smoothing with progressive residuals for video KV.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers three SOTA autoregressive video models, multi-bit (INT2/INT4) scenarios, multiple baselines, consumer GPU validation, and comprehensive error analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear progression from system-algorithm bottleneck diagnosis to opportunity discovery and design verification.
- Value: ⭐⭐⭐⭐⭐ Immediate impact on long video generation, consumer GPU deployment, and context expansion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Generation](quantized_keys_steal_attention_bias_correction_for_kv-cache_compression_in_video.md)
- [\[ICML 2026\] LocoT2V-Bench: Benchmarking Long-form and Complex Text-to-Video Generation](locot2v-bench_benchmarking_long-form_and_complex_text-to-video_generation.md)
- [\[ICML 2026\] Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos](enhancing_train-free_infinite-frame_generation_for_consistent_long_videos.md)
- [\[ICML 2026\] Explainable Forensics of Manipulated Segments in Untrimmed Long Videos](explainable_forensics_of_manipulated_segments_in_untrimmed_long_videos.md)
- [\[CVPR 2026\] When to Lock Attention: Training-Free KV Control in Video Diffusion](../../CVPR2026/video_generation/when_to_lock_attention_training-free_kv_control_in_video_diffusion.md)

</div>

<!-- RELATED:END -->
