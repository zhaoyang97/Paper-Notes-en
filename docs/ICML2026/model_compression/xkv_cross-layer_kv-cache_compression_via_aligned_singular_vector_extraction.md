---
title: >-
  [Paper Note] xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction
description: >-
  [ICML 2026][Model Compression][KV-cache compression] xKV discovers that while the token-wise cosine similarity of KV-caches across different LLM layers is low…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "KV-cache compression"
  - "long-context inference"
  - "cross-layer low-rank decomposition"
  - "selective reconstruction"
  - "CKA"
date: 2026-05-08
content_hash: 9edcd7a2b145a95c
---

# xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction

**Conference**: ICML 2026  
**arXiv**: [2503.18893](https://arxiv.org/abs/2503.18893)  
**Code**: https://github.com/abdelfattah-lab/xKV  
**Area**: Model Compression / LLM Inference Acceleration  
**Keywords**: KV-cache compression, long-context inference, cross-layer low-rank decomposition, selective reconstruction, CKA  

## TL;DR
xKV discovers that while the token-wise cosine similarity of KV-caches across different LLM layers is low, their principal singular vectors are highly aligned. Consequently, it uses cross-layer shared low-rank bases to simultaneously compress multi-layer KV-caches. Combined with selective reconstruction, it achieves up to 8x compression and a 4.23x increase in end-to-end throughput for long-context inference.

## Background & Motivation
**Background**: The primary bottleneck for long-context LLMs has shifted from parameter storage to the ever-expanding KV-cache during inference. Existing compression approaches include low-bit quantization, token eviction, single-layer low-rank decomposition, dynamic token selection, and CPU offloading; these methods reduce memory footprint but mostly treat the cache of each layer as an independent object.

**Limitations of Prior Work**: Single-layer methods only exploit intra-layer redundancy and are prone to losing fine-grained information needed for long-range retrieval, variable tracking, and multi-round dialogues when compression rates increase. Cross-layer KV sharing methods also face two types of issues: architectural modifications like CLA or YOCO require re-pretraining, while post-processing methods like MiniCache rely on the cosine similarity of token representations in adjacent layers. Evaluations in the paper show that this token-wise similarity is unstable, causing significant performance drops even at 1.3x compression.

**Key Challenge**: The redundancy in KV-cache is not simply "the same token looks very similar in adjacent layers," but rather "the token spaces of different layers are spanned by similar principal directions." Looking only at token-level cosine similarity underestimates the space for cross-layer sharing; to explicitly utilize this shared space, one must also avoid the computational overhead of reconstructing the full long sequence for every decoding step.

**Goal**: The authors aim to compress the KV-cache of existing long-context LLMs without architectural modifications or fine-tuning, while maintaining accuracy on tasks like RULER, LongBench, and multi-round NIAH, and effectively converting memory savings into throughput gains rather than having them offset by reconstruction computation or PCIe transmission.

**Key Insight**: The paper uses CKA to examine the overall geometric structure of KV-caches across layers instead of comparing individual token vector directions. Results show a clear alignment of principal singular vectors across layers in Llama, Qwen, and even hybrid attention architectures. When horizontally concatenating multiple adjacent layers, the relative rank required to preserve 95% of spectral energy decreases as the window size grows.

**Core Idea**: Use cross-layer SVD to extract a set of shared token bases, allowing multi-layer KV-caches within the same layer group to share a principal space, then use layer-specific small reconstruction matrices to recover tokens required for attention.

## Method

### Overall Architecture
xKV targets long-context inference for decoder-only LLMs. The input is the cache of each layer generated during the prefilling stage for a long prompt; the output is a compressed cache representation rather than new model weights. It groups model layers into contiguous windows, performs joint low-rank decomposition on multi-layer caches within each window, and stores shared bases and layer-specific reconstruction matrices. During decoding, the model can choose to reconstruct the full layer cache densely or only reconstruct a few tokens most likely to be attended to by the current query, achieving a better balance between memory and computation.

More specifically, let the key or value cache of a layer be $X_\ell \in \mathbb{R}^{L \times d}$, where $L$ is the context length and $d$ is the KV hidden size. xKV concatenates $W$ layers within a window $\mathcal{W}_k$ horizontally as $X_k^{cat}=[X_{kW},\ldots,X_{kW+W-1}]$, then performs low-rank approximation $X_k^{cat}\approx A_k[B_{kW},\ldots,B_{kW+W-1}]$. Here $A_k\in\mathbb{R}^{L\times r}$ is the token basis shared by this layer group, and $B_\ell\in\mathbb{R}^{r\times d}$ is the layer-specific reconstruction matrix.

### Key Designs
1. **Finding cross-layer shared bases via CKA instead of token-level similarity**:

	- **Function**: Provides a reliable basis for cross-layer compression and explains why MiniCache-style adjacent layer token merging fails.
	- **Mechanism**: The paper first calculates token-wise cosine similarity and finds that adjacent layer token vectors are not similar enough; it subsequently calculates CKA between centered Gram matrices, observing highly consistent geometric structures across many layer pairs. High CKA implies alignment of principal left singular vectors, allowing multiple layer caches to share the same low-dimensional token basis.
	- **Design Motivation**: Truly compressible redundancy in KV-cache is hidden in the subspace structure rather than one-to-one correspondences between individual token representations. This observation shifts cross-layer compression from "hard-merging similar tokens" to "sharing principal directions," preserving information at higher compression rates.

2. **Cross-Layer Factorization for unified compression of key/value caches in a layer group**:

	- **Function**: Compresses $O(WLd)$ cache storage into $O(Lr+Wrd)$ shared bases plus layer-specific coefficients.
	- **Mechanism**: Performs online SVD on multi-layer caches within the same window during the prefill phase. Basis $A_k$ is stored only once, with inter-layer differences represented by each layer's $B_\ell$; both keys and values are processed similarly, with a default window size of 4 and a key/value rank ratio of 1:1.5.
	- **Design Motivation**: Single-layer SVD repeatedly stores highly similar principal directions for each layer, causing accuracy to drop rapidly at high compression rates. Cross-layer factorization merges these redundant bases, preserving more effective information at the same compression rate.

3. **Selective Reconstruction shifts reconstruction costs from full sequences to query-related tokens**:

	- **Function**: Avoids reconstructing the full cache of $L$ tokens at every decoding step, enabling the compressed representation to deliver end-to-end acceleration.
	- **Mechanism**: For each decoding step, head, and layer, a subset of tokens $\mathcal{S}_{t,\ell,g}$ is first selected using approximate attention, and only $\hat{X}_{\ell,g}[\mathcal{S}_{t,\ell,g},:]=A_k[\mathcal{S}_{t,\ell,g},:]B_{\ell,g}$ is computed. Since the number of selected tokens is fixed and much smaller than the context length, the reconstruction overhead no longer grows linearly with $L$.
	- **Design Motivation**: While dense reconstruction saves memory, it becomes a computational bottleneck; ShadowKV also used selective reconstruction but often offloaded values to the CPU due to insufficient fidelity in single-layer SVD. xKV-SR offers higher compression quality, keeping both keys and values in GPU HBM to avoid PCIe bottlenecks.

### Loss & Training
xKV does not introduce training loss or require fine-tuning. It is a post-training, plug-and-play inference-time compression method: it performs online low-rank decomposition on the cache generated by real prompts after prefilling and uses selective reconstruction during decoding based on the query. The authors also implemented custom randomized SVD kernels using 16-bit GEMM and shifted Cholesky QR to reduce online decomposition overhead; for context lengths of 64k to 256k on Llama-3.1-8B, SVD time is approximately 5.72% to 1.24% of the prefill time when the window size is 4.

## Key Experimental Results

### Main Results
RULER 64K is the core experiment. The following table retains average scores and compression rates, showing the differences between xKV and single-layer, token eviction, quantization, and cross-layer merging baselines.

| Model | Method | Type | KV Compression | RULER Avg. | Description |
|------|------|------|-----------|------------|------|
| Llama-3.1-8B-Instruct | Full Attention | No Compr. | 1.00x | 91.89 | Original KV-cache |
| Llama-3.1-8B-Instruct | KIVI-2 | Intra-layer Quant. | 7.10x | 86.87 | 2-bit KV Quantization |
| Llama-3.1-8B-Instruct | SnapKV | Intra-layer Eviction | 8.00x | 89.68 | First-step attention driven eviction |
| Llama-3.1-8B-Instruct | Single SVD | Intra-layer Low-rank | 8.40x | 45.71 | Independent SVD per layer, severe drop |
| Llama-3.1-8B-Instruct | MiniCache | Cross-layer Merging | 1.30x | 45.04 | Unstable cosine similarity assumption |
| Llama-3.1-8B-Instruct | xKV | Cross-layer Low-rank | 8.03x | 88.50 | Close to strong token eviction baseline |
| Qwen2.5-14B-Instruct-1M | Full Attention | No Compr. | 1.00x | 93.36 | Original KV-cache |
| Qwen2.5-14B-Instruct-1M | SnapKV | Intra-layer Eviction | 6.00x | 91.66 | Qwen has fewer KV heads, harder to compress |
| Qwen2.5-14B-Instruct-1M | Single SVD | Intra-layer Low-rank | 6.35x | 71.79 | Single-layer low-rank still loses info |
| Qwen2.5-14B-Instruct-1M | MiniCache | Cross-layer Merging | 1.30x | 13.78 | Nearly unusable on this model |
| Qwen2.5-14B-Instruct-1M | xKV-4 | Cross-layer Low-rank | 6.21x | 90.19 | ~3.17 points gap with full attention |

### Ablation Study
Selective reconstruction and window size are the most critical analyses. The first set of results shows that xKV-SR maintains accuracy while keeping compressed caches on the GPU; the second set shows that "cross-layer" itself significantly contributes to performance.

| Configuration | Compr./Mem. Effect | RULER Avg. | Description |
|------|---------------|------------|------|
| Full Attention | 1.00x | 91.89 | Llama-3.1-8B-Instruct baseline |
| Quest | 1.00x KV, ~8.00x GPU memory reduction | 84.87 | Dynamic token loading only, no cache compression |
| ShadowKV | 1.64x KV, ~9.08x GPU memory reduction | 87.17 | Single-layer SVD + value offloading |
| xK-SR | 1.63x KV, ~8.90x GPU memory reduction | 89.70 | Replace ShadowKV's single-layer key compr. with cross-layer |
| ShadowKV‡ | 5.51x KV | 70.94 | Accuracy drops sharply after further value compression |
| xKV-SR | 5.35x KV | 89.69 | Both K/V cross-layer compressed and kept on GPU |

| Window Size | xKV Avg. | xK-SR Avg. | xKV-SR Avg. | Conclusion |
|----------|----------|-------------|--------------|------|
| 1 | 45.71 | 87.17 | 72.27 | Insufficient fidelity when degraded to intra-layer |
| 2 | 75.15 | 88.43 | 86.06 | Significant improvement with 2 layers |
| 4 | 88.50 | 89.70 | 89.69 | Main experiment default, balanced precision/overhead |
| 8 | 88.91 | 89.74 | 89.72 | Gains saturate as window increases further |

### Key Findings
- The core advantage of xKV is not just "low-rank," but shifting low-rank from being layer-independent to cross-layer shared. As window size increases from 1 to 4, dense xKV average scores jump from 45.71 to 88.50, directly proving that cross-layer bases are the primary source of gain.
- MiniCache achieves only 45.04 at 1.3x compression on Llama and 13.78 on Qwen, indicating that token-level cosine similarity is insufficient to support cross-layer cache merging; this reinforces the decision to use CKA over cosine for motivational analysis.
- The system value of xKV-SR is evident: it maintains an 89.69 average score at 5.35x KV compression and achieves a 4.23x end-to-end throughput gain at 122k context. Without selective reconstruction, memory bottlenecks would ease, but reconstruction FLOPs would reduce speed.
- In multi-round NIAH, SnapKV/PyramidKV drop significantly in later rounds because they prune tokens based on the first round's query attention; xKV's global compressed information is more stable across rounds.

## Highlights & Insights
- The most significant insight is reframing "cross-layer similarity" from token representation similarity to subspace similarity. This perspective explains why existing cross-layer merging methods are fragile and provides a new application for SVD in the KV-cache context.
- xKV discusses algorithmic compression and system throughput within a closed loop. While many methods only report memory or accuracy, this work explicitly notes that dense reconstruction becomes a computational bottleneck and that speed gains necessitate selective reconstruction and GPU residency.
- The method is training-free, making it highly compatible with existing long-context models. It does not require re-pretraining models like CLA/YOCO or assuming access to massive training resources.
- The window size ablation is highly explanatory. The jump from 1 to 4 shows that cross-layer redundancy is substantial, while the saturation at 8 suggests that deployments need not indefinitely expand layer groups, avoiding excessive prefill buffers and SVD overhead.

## Limitations & Future Work
- The paper focus primarily on long-prefill scenarios where the initial long context is compressed, while newly generated tokens are not. For long generation or test-time scaling, the continuously growing KV-cache remains a potential bottleneck.
- Strategies for rank, window size, and key/value compression ratios are mostly fixed. The appendix shows that different tasks tolerate key/value compression differently; future work could explore task-aware or context-aware dynamic rank allocation.
- The method relies on online SVD after prefilling, which, while relatively low-overhead in long contexts, still requires evaluation for Implementation complexity and overhead in short-context, high-QPS, or small-GPU scenarios.
- Selective reconstruction requires prior token selection, and its effectiveness is bounded by approximate attention quality. For tasks requiring highly dispersed evidence or with frequently changing query patterns, a fixed token budget might be insufficient.

## Related Work & Insights
- **vs KIVI / KV Quantization**: Quantization reduces bits per element, while xKV reduces the storage dimension of the subspace; these are orthogonal, and appendix results show xKV can be stacked with 4-bit/3-bit quantization for further compression.
- **vs SnapKV / PyramidKV**: Token eviction explicitly discards historical tokens, suitable for focused attention and stable single-round queries; xKV preserves a low-rank approximation of the global cache, making it more stable for multi-round retrieval and re-attending to old information.
- **vs MiniCache**: MiniCache uses adjacent layer token cosine similarity for merging, while xKV uses CKA to find shared principal singular vectors. The former makes too strong a compression assumption, while the latter is more robust to inter-layer representation rotations and token-level differences.
- **vs ShadowKV**: ShadowKV's selective reconstruction is valuable but based on single-layer SVD, leading to drops when compressing values; xKV's cross-layer factorization allows both keys and values to stay on the GPU.
- **Insights**: Inference acceleration research should perhaps look beyond element-wise or token-wise similarity and instead examine representation subspaces, Gram structures, and spectral energy distributions; this has potential transfer value for MoE activation caches, vision token caches, and diffusion model feature caches.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Using CKA to derive cross-layer shared singular vectors is highly distinctive; SVD is traditional but the combination is new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers RULER, LongBench, multi-round NIAH, window sizes, K/V compression ratios, quantization stacking, and throughput measurements.
- Writing Quality: ⭐⭐⭐⭐☆ Clear motivational chain and solid system experiments; some formula and table layouts are slightly crowded in text-based views.
- Value: ⭐⭐⭐⭐⭐ Extremely practical for long-context LLM deployment, especially for training-free compression and GPU batch expansion of existing models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] The Pitfalls of KV Cache Compression](../../ACL2026/model_compression/the_pitfalls_of_kv_cache_compression.md)
- [\[ICML 2026\] Semantic Integrity Matters: Benchmarking and Preserving High-Density Reasoning in KV Cache Compression](semantic_integrity_matters_benchmarking_and_preserving_high-density_reasoning_in.md)
- [\[ICML 2026\] FlattenGPT: Depth Compression for Transformer with Layer Flattening](flattengpt_depth_compression_for_transformer_with_layer_flattening.md)
- [\[ICML 2026\] EpiCache: Episodic KV Cache Management for Long-Term Conversation on Resource-Constrained Environments](epicache_episodic_kv_cache_management_for_long-term_conversation_on_resource-con.md)
- [\[ICML 2026\] A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints](a_queueing-theoretic_framework_for_stability_analysis_of_llm_inference_with_kv_c.md)

</div>

<!-- RELATED:END -->
