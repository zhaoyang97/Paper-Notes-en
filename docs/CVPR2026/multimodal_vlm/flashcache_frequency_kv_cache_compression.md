---
title: >-
  [Paper Note] FlashCache: Frequency-Domain-Guided Outlier-KV-Aware Multimodal KV Cache Compression
description: >-
  [CVPR 2026][Multimodal VLM][KV Cache Compression] This paper proposes FlashCache, the first method to analyze the importance distribution of multimodal KV Cache from a frequency-domain perspective. It discovers that KV pairs deviating from low-frequency principal components—termed "outlier KVs"—encode features critical for inference. By identifying outlier KVs via DCT low-pass filtering and prioritizing their retention alongside dynamic per-layer budget allocation, FlashCache achieves 1.69× decoding speedup under 80% KV memory compression with negligible task performance degradation, while being natively compatible with FlashAttention.
tags:
  - CVPR 2026
  - Multimodal VLM
  - KV Cache Compression
  - Frequency-Domain Analysis
  - Outlier KV
  - Dynamic Budget Allocation
  - FlashAttention Compatibility
date: 2026-05-08
content_hash: ce8e8fa2fee4598b
---

# FlashCache: Frequency-Domain-Guided Outlier-KV-Aware Multimodal KV Cache Compression

**Conference**: CVPR 2026
**arXiv**: [2511.16786](https://arxiv.org/abs/2511.16786)
**Code**: None
**Area**: Multimodal VLM / Model Compression
**Keywords**: KV Cache Compression, Frequency-Domain Analysis, Outlier KV, Dynamic Budget Allocation, FlashAttention Compatibility

## TL;DR
This paper proposes FlashCache, the first method to analyze the importance distribution of multimodal KV Cache from a frequency-domain perspective. It discovers that KV pairs deviating from low-frequency principal components—termed "outlier KVs"—encode features critical for inference. By identifying outlier KVs via DCT low-pass filtering and prioritizing their retention alongside dynamic per-layer budget allocation, FlashCache achieves 1.69× decoding speedup under 80% KV memory compression with negligible task performance degradation, while being natively compatible with FlashAttention.

## Background & Motivation

**Background**: Multimodal large language models (MLLMs) excel at visual understanding and reasoning, but their KV Cache grows linearly with visual input length during inference. In long-context scenarios such as multi-image, high-resolution, and video inputs, the GPU memory footprint of the KV Cache grows dramatically, accompanied by significant increases in decoding latency—constituting a core deployment bottleneck.

**Limitations of Prior Work**: Existing multimodal KV Cache compression methods (e.g., LOOK-M, MEDA, H2O, SnapKV) almost universally rely on attention scores to decide which KV pairs to retain. This introduces two critical issues: (1) efficient attention kernels such as FlashAttention do not explicitly output full attention matrices, and recomputing attention scores incurs additional overhead that contradicts the goal of efficient inference; (2) attention scores are determined solely by Query-Key dot products, ignoring the actual contribution of Value vectors to the final attention output, resulting in an incomplete importance signal.

**Key Challenge**: Achieving high-ratio KV Cache compression requires accurately identifying which KV pairs are most important for inference; however, the mechanism by which existing methods obtain importance signals (attention scores) is both insufficiently accurate and incompatible with efficient inference frameworks.

**Goal**: To design a compression method that does not rely on attention scores, instead deriving importance directly from the distributional characteristics of the KV matrices themselves—enabling efficient identification of inference-critical KV pairs while remaining natively compatible with FlashAttention and similar efficient attention implementations.

**Key Insight**: Drawing inspiration from frequency-domain analysis in signal processing, the authors transform KV matrices into the frequency domain to observe their energy distribution. They find that the frequency-domain energy of KV matrices is highly concentrated in low frequencies—low-frequency components correspond to smooth, redundant principal-component patterns—while KV pairs that deviate from these principal components ("outlier KVs") are more likely to encode features critical for inference, and preferentially removing them leads to significant performance degradation.

**Core Idea**: Low-pass filtering in the frequency domain is used to extract the principal components of KV matrices (Base KV). KV pairs with the largest deviation from these principal components are defined as "outlier KVs" and prioritized for retention, while KV Cache budgets are dynamically allocated per layer.

## Method

### Overall Architecture
FlashCache performs one-shot compression of the multimodal KV Cache after the prefilling stage. The overall pipeline comprises two core modules: (1) the **Outlier KV Recognition Module**, which applies DCT frequency-domain transformation to the KV matrices of each layer, obtains smooth Base KVs via low-pass filtering, computes the MSE deviation of each KV pair from the Base KV, and preferentially retains the KV pairs with the largest deviations (outlier KVs); and (2) the **Dynamic Budget Allocation Module**, which analyzes the ratio of outlier-information energy to total energy in the frequency domain of each layer's KV matrices, normalizes these ratios, and dynamically allocates different KV Cache retention quotas to each layer under a global budget constraint. The entire process requires no training, depends on no attention scores, and is natively compatible with FlashAttention. DCT operations are accelerated using the NVIDIA CuPy operator library.

### Key Designs

1. **Outlier KV Recognition Module**:

    - **Function**: Identify and preferentially retain the KV pairs most critical for inference in each layer.
    - **Mechanism**: For the Key/Value tensors $K^l, V^l$ of layer $l$, a Discrete Cosine Transform (DCT) is first applied to obtain frequency-domain coefficients $C_k^l[m], C_v^l[m]$. A low-pass filter with cutoff factor $\gamma$ is then applied, retaining the leading $\omega = \gamma \cdot N$ low-frequency coefficients; an inverse DCT is applied to recover smooth Base KVs $K_{base}^l, V_{base}^l$ in the time domain. The deviation of each KV pair from the Base KV is computed as $Dev[x] = \text{MSE}(K^l[x], K_{base}^l[x]) + \text{MSE}(V^l[x], V_{base}^l[x])$. KV pairs are sorted in descending order of deviation, and those with the largest deviations (outlier KVs) are retained.
    - **Design Motivation**: Low-frequency principal components represent smooth, redundant KV patterns. Experiments verify that preferentially removing high-deviation KV pairs causes a sharp performance drop—faster than random removal or removal of low-deviation KVs—confirming that outlier KVs encode features critical for inference. This is analogous to outlier value protection in model quantization, where a similar outlier phenomenon exists.

2. **Dynamic Budget Allocation Module**:

    - **Function**: Adaptively allocate different KV Cache retention quotas to different layers of the model.
    - **Mechanism**: Via Parseval's theorem, the power spectrum $P_k^l[m] = |C_k^l[m]|^2$ of each layer's KV matrices is computed directly in the frequency domain. The ratio of outlier information energy to total energy per layer is computed as $R^l = R_k^l + R_v^l$, where $R_k^l = \sum_{\ell=\omega+1}^{N-1}P_k^l[\ell] / \sum_{\ell=0}^{N-1}P_k^l[\ell]$. These per-layer ratios are normalized into weights, and each layer's retention quota is allocated under a global budget constraint.
    - **Design Motivation**: The degree of low-frequency concentration varies substantially across Transformer layers—some layers' KV matrices are almost entirely concentrated in low-frequency principal components, while others carry considerably more high-frequency outlier information. Uniform compression ratios over-compress the latter, causing unnecessary performance loss.

3. **Native FlashAttention Compatibility**:

    - **Function**: Ensure that the compression method can be used directly with efficient attention kernels.
    - **Mechanism**: All importance-judgment operations are based solely on the frequency-domain characteristics of the KV matrices themselves, requiring neither computation nor access to the attention score matrix; the FlashAttention computation pipeline therefore requires no modification.
    - **Design Motivation**: Attention-score-based methods (H2O, SnapKV, LOOK-M, MEDA) require explicit output of the attention matrix or recomputation of attention scores, which conflicts with FlashAttention's IO-aware design and introduces additional computational overhead.

### Loss & Training
FlashCache is an entirely training-free inference-time compression scheme. One-shot compression is performed after the prefilling stage, involving no parameter updates or backpropagation. Key hyperparameters include the low-pass filter cutoff factor $\gamma$ (optimal range: 0.1–0.2) and the global KV Cache retention ratio $\rho$. DCT/IDCT operations are accelerated via the NVIDIA CuPy operator library; at 8K token inputs, the additional latency is only 6.77 ms—significantly lower than attention-score-based methods (27–84 ms).

## Key Experimental Results

### Main Results
MileBench multi-image understanding benchmark (KV retention ratio $\rho=0.2$):

| Method | Task T (Qwen-7B) | Task S (Qwen-7B) | NH (Qwen-7B) | IR (Qwen-7B) |
|--------|-------------------|-------------------|---------------|---------------|
| Full Cache | 55.59 | 69.17 | 27.35 | 14.17 |
| StreamingLLM | 55.59 | 67.51 | 9.69 | 14.00 |
| H2O | 55.59 | 68.60 | 12.66 | 14.67 |
| SnapKV | 55.59 | 68.27 | 13.59 | 15.33 |
| LOOK-M | 55.55 | 67.50 | 11.88 | 11.83 |
| MEDA | 55.59 | 68.13 | 9.07 | 14.50 |
| **FlashCache** | **55.59** | **68.85** | **26.72** | **15.50** |

High-resolution benchmarks (Qwen2.5-VL-7B, $\rho=0.05$):

| Method | V* | HR-Bench |
|--------|----|----------|
| Full Cache | 80.23 | 70.75 |
| SnapKV | 78.89 | 71.12 |
| LOOK-M | 77.78 | 70.25 |
| **FlashCache** | **79.66** | **72.38** |

Additional latency comparison (ms):

| Method | 2K tokens | 4K tokens | 8K tokens |
|--------|-----------|-----------|-----------|
| H2O | 3.83 | 10.29 | 27.62 |
| SnapKV | 2.53 | 4.95 | 9.57 |
| LOOK-M | 6.93 | 18.66 | 53.97 |
| MEDA | 16.6 | 38.39 | 83.75 |
| **FlashCache** | **1.66** | **3.86** | **6.77** |

### Ablation Study
Ablation on low-pass filter cutoff factor $\gamma$ (Qwen2.5-VL-7B, $\rho=0.2$):

| $\gamma$ | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 0.9 |
|----------|------|------|------|------|------|------|
| INIAH | 29.06 | 29.69 | 25.0 | 22.5 | 22.81 | 20.08 |
| GPR1200 | 15.0 | 15.5 | 15.17 | 15.17 | 14.83 | 13.05 |

Ablation on the Dynamic Budget Allocation (DBA) module:

| Configuration | INIAH | GPR1200 | ALFRED | CLEVR-Change |
|---------------|-------|---------|--------|--------------|
| w/o DBA | 24.69 | 14.67 | 34.32 | 35.85 |
| w/ DBA | 29.69 | 15.50 | 34.39 | 41.04 |

### Key Findings
- FlashCache's advantage is most pronounced on the Needle-in-a-Haystack (NH) task: at $\rho=0.2$ it retains a score of 26.72 (close to Full Cache's 27.35), while H2O reaches only 12.66 and SnapKV only 13.59, demonstrating that frequency-domain outlier detection significantly outperforms attention-score-based methods on long-range information retrieval.
- The low-pass filter cutoff factor $\gamma$ is optimal in the range 0.1–0.2; excessively large $\gamma$ (>0.3) incorporates too many frequencies into the "principal components," preventing effective identification of outlier KVs and causing significant performance degradation.
- The DBA module contributes most on the CLEVR-Change dataset (+5.19 points), indicating that dynamic allocation is particularly important for visual reasoning tasks.
- FlashCache maintains performance close to Full Cache even at an extremely low retention ratio of $\rho=0.05$ (V* task: 79.66 vs. 80.23), exhibiting robustness superior to all baselines.
- Method latency is extremely low (only 6.77 ms at 8K tokens)—one-quarter of H2O and one-eighth of LOOK-M—because frequency-domain operations do not depend on attention computation.

## Highlights & Insights
- **Frequency-domain perspective as a novel KV importance signal**: This is the first work to introduce frequency-domain analysis from signal processing into KV Cache compression, uncovering the distributional regularity that "low frequency = redundant principal components; high deviation = critical outliers," providing an alternative importance signal to attention scores.
- **Analogy between outlier KVs and outlier value protection in quantization**: The outlier phenomenon in KV Cache compression shares a structural similarity with the outlier value problem in model quantization—in both cases, a small number of elements deviating from the principal components carry a disproportionate share of critical information.
- **First training-free, attention-score-free multimodal KV compression framework**: Native FlashAttention compatibility enables direct deployment in production environments without modifying the inference framework.
- **Dynamic per-layer budget allocation**: The frequency-domain energy distribution of KV matrices varies substantially across layers, making uniform compression ratios a suboptimal strategy.

## Limitations & Future Work
- Although DCT operations are accelerated via CuPy, memory and computational overhead at extremely long sequences (64K+ tokens) requires further optimization.
- The low-pass filter cutoff factor $\gamma$ requires manual tuning (optimal: 0.1–0.2), lacking an adaptive selection mechanism.
- The current approach performs one-shot compression only after the prefilling stage; incremental KV management strategies during decoding have not been explored.
- The theoretical explanation for frequency-domain analysis remains insufficiently deep—why does the energy of KV matrices concentrate in low frequencies, and what semantic information do outlier KVs encode?
- Validation is limited to multimodal scenarios; systematic evaluation on pure-text long-context LLMs (e.g., 128K-window models) has not been conducted.

## Related Work & Insights
- **vs. H2O / SnapKV**: Both methods evict low-importance KV pairs based on attention scores, relying on explicit attention matrix output and thus incompatible with FlashAttention. SnapKV uses partial attention computation to reduce overhead, yet still falls far short of FlashCache on the NH task (13.59 vs. 26.72).
- **vs. LOOK-M**: Filters and merges low-attention-score visual KV pairs during prefilling, but KV merging introduces 53.97 ms (at 8K) of additional latency—eight times that of FlashCache.
- **vs. MEDA**: Allocates per-layer KV budgets using cross-modal attention entropy, conceptually similar to FlashCache's DBA module, but still depends on attention computation (83.75 ms latency) and encounters OOM on MUIRBench.
- **Insights**: The idea of using frequency-domain analysis to assess KV importance can be generalized to KV compression in pure-text LLMs. The concept of outlier KVs may have an intrinsic connection to the attention sink phenomenon. Dynamic budget allocation could be combined with mixed-precision KV quantization to form a multi-level compression scheme of "retain important KVs + quantize redundant KVs at low precision."

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Frequency-domain analysis of KV importance is an entirely novel perspective; the discovery of outlier KVs is insightful, and the analogy to outlier protection in quantization is inspirational.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 3 MLLMs and 6 benchmarks (multi-image / high-resolution / video) across multiple retention ratio settings, with complete ablations and efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ The motivation chain is clear (attention scores unavailable → frequency-domain alternative → outlier KV discovery → retention strategy), well supported by informative figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ The first training-free multimodal KV compression scheme compatible with FlashAttention; the practical value of 80% memory savings with 1.69× speedup is substantial.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Revisiting Multimodal KV Cache Compression: A Frequency-Domain-Guided Outlier-KV-Aware Approach](revisiting_multimodal_kv_cache_compression_a_frequency-domain-guided_outlier-kv-.md)
- [\[CVPR 2026\] ApET: Approximation-Error Guided Token Compression for Efficient VLMs](apet_approximation_error_token_compression.md)
- [\[ICLR 2026\] Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models](../../ICLR2026/multimodal_vlm/mixing_importance_with_diversity_joint_optimization_for_kv_cache_compression_in_.md)
- [\[CVPR 2026\] V2Drop: Variation-aware Vision Token Dropping for Faster Large Vision-Language Models](v2drop_variation_aware_token_dropping.md)
- [\[CVPR 2026\] Variation-Aware Vision Token Dropping for Faster Large Vision-Language Models](variation-aware_vision_token_dropping_for_faster_large_vision-language_models.md)

</div>

<!-- RELATED:END -->
