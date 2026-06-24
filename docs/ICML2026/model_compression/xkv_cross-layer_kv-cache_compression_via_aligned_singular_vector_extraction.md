---
title: >-
  [Paper Note] xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction
description: >-
  [ICML 2026][Model Compression][KV-cache Compression] xKV discovers that while the per-token cosine similarity of KV-caches across different LLM layers is low, their principal singular vectors are highly aligned. Consequently, xKV utilizes a cross-layer shared low-rank basis to compress multiple layers of KV-cache simultaneously. Combined with selective reconstruction, it achieves up to 8x compression and a 4.23x increase in end-to-end throughput for long-context inference.
tags:
  - "ICML 2026"
  - "Model Compression"
  - "KV-cache Compression"
  - "Long-Context Inference"
  - "Cross-Layer Low-Rank Decomposition"
  - "Selective Reconstruction"
  - "CKA"
date: 2026-05-08
content_hash: 3cc444d84db15fab
---

# xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction

**Conference**: ICML 2026  
**arXiv**: [2503.18893](https://arxiv.org/abs/2503.18893)  
**Code**: https://github.com/abdelfattah-lab/xKV  
**Area**: Model Compression / LLM Inference Acceleration  
**Keywords**: KV-cache Compression, Long-Context Inference, Cross-Layer Low-Rank Decomposition, Selective Reconstruction, CKA  

## TL;DR
xKV discovers that while the per-token cosine similarity of KV-caches across different LLM layers is low, their principal singular vectors are highly aligned. Consequently, xKV utilizes a cross-layer shared low-rank basis to compress multiple layers of KV-cache simultaneously. Combined with selective reconstruction, it achieves up to 8x compression and a 4.23x increase in end-to-end throughput for long-context inference.

## Background & Motivation
**Background**: The primary bottleneck for long-context LLMs has shifted from parameter storage to the continuously expanding KV-cache during inference. Existing compression approaches include low-bit quantization, token eviction, single-layer low-rank decomposition, dynamic token selection, and CPU offloading; these methods reduce memory footprint but mostly treat the cache of each layer as an independent object.

**Limitations of Prior Work**: Single-layer methods only exploit intra-layer redundancy, often losing fine-grained information necessary for long-range retrieval, variable tracking, and multi-turn dialogues when compression rates increase. Cross-layer KV sharing methods also face issues: architectural modifications like CLA or YOCO require re-pretraining, while post-processing methods like MiniCache rely on the cosine similarity of token representations between adjacent layers. Empirical tests in this paper show that such per-token similarity is unstable, causing significant performance drops even at 1.3x compression.

**Key Challenge**: The redundancy in KV-cache is not simply that "the same token looks similar in adjacent layers," but rather that "token spaces of different layers are spanned by similar principal directions." Relying solely on token-level cosine similarity underestimates the potential for cross-layer sharing. Explicitly utilizing this shared space must also avoid the massive computational overhead of reconstructing full long sequences for every decoding step.

**Goal**: The authors aim to compress the KV-cache of existing long-context LLMs without structural changes or fine-tuning, while maintaining accuracy on tasks like RULER, LongBench, and multi-turn NIAH. The objective is to translate memory savings into actual throughput gains without being offset by reconstruction computation or PCIe transfer overhead.

**Key Insight**: The paper uses CKA to inspect the overall geometric structure of KV-caches across different layers instead of comparing individual token vector directions. Results indicate that principal singular vectors of KV-caches in Llama, Qwen, and even hybrid-attention architectures exhibit significant alignment. When multiple adjacent layers are concatenated horizontally, the relative rank required to preserve 95% of the spectral energy decreases as the window size grows.

**Core Idea**: Use cross-layer SVD to extract a set of shared token bases, allowing multiple layers within a group to share a principal space, then use layer-specific small reconstruction matrices to recover tokens required for attention.

## Method

### Overall Architecture
xKV is designed for long-context inference in decoder-only LLMs. The input is the KV-cache generated during the prefill phase for a long prompt across all layers; the output is a compressed cache representation. It groups model layers into contiguous windows, performs joint low-rank decomposition for each window, and stores shared bases and layer-specific reconstruction matrices. During decoding, the model can choose between dense reconstruction of the entire cache or selective reconstruction of only the tokens most likely to be attended to by the current query, achieving a better balance between memory and computation.

Specifically, let the key or value cache of a layer be $X_\ell \in \mathbb{R}^{L \times d}$, where $L$ is the context length and $d$ is the KV hidden size. xKV concatenates $W$ layers within a window $\mathcal{W}_k$ horizontally as $X_k^{cat}=[X_{kW},\ldots,X_{kW+W-1}]$, then applies a low-rank approximation $X_k^{cat}\approx A_k[B_{kW},\ldots,B_{kW+W-1}]$. Here $A_k\in\mathbb{R}^{L\times r}$ is the shared token basis for the group, and $B_\ell\in\mathbb{R}^{r\times d}$ is the layer-specific reconstruction matrix. The pipeline is summarized as follows:

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Long Prompt Prefill<br/>Generates K/V cache for each layer"] --> B["Cross-Layer Alignment (CKA) Analysis<br/>Principal singular vectors align → Group adjacent W layers (default W=4)"]
    B --> C["Cross-Layer Factorization (CLF)<br/>Horizontal concatenation → Cross-layer SVD<br/>Yields shared basis A_k and reconstruction matrices B_ℓ"]
    C --> D{Decoding Reconstruction}
    D -->|Dense Reconstruction: All L tokens<br/>O(L) computational bottleneck| E["X̂_ℓ = A_k · B_ℓ"]
    D -->|Selective Reconstruction (SR)| F["Approximate attention selects query-related token set S<br/>Only reconstruct rows in S (|S| ≪ L)"]
    E --> G["Attention Operation → Long-context inference acceleration"]
    F --> G
```

### Key Designs

**1. Discovering Cross-Layer Shared Singular Vectors via CKA: Shifting "Cross-Layer Similarity" from Token-Level to Subspace-Level**

The premise of xKV starts by answering whether KV-caches of different layers are truly similar. Methods like MiniCache compare per-token cosine similarity between adjacent layers and conclude they are "similar enough to merge," but tests on Llama show performance drops to 45 at 1.3x compression. xKV switches to Centered Kernel Alignment (CKA), which compares the centralized Gram matrices of token embeddings between two layers to characterize geometric consistency. It reveals that while per-token cosine similarity is low, many layer pairs exhibit high CKA. Mathematically, high CKA is equivalent to high alignment of principal left singular vectors (proven in the appendix). In other words, the compressible redundancy lies in shared principal directions across layer token spaces. Ignoring this leads to an underestimation of compression potential, explaining the fragility of MiniCache and justifying xKV's "shared principal direction" approach.

**2. Cross-Layer Factorization (CLF): Sharing Token Bases Across a Group of Layers**

Given the alignment of principal singular vectors, xKV groups adjacent $W$ layers (default $W=4$) into a window and performs joint low-rank decomposition during prefill. By concatenating caches horizontally into $X_k^{cat}$ and performing cross-layer SVD, it yields a shared token basis $A_k\in \mathbb{R}^{L\times r}$ and small reconstruction matrices $B_\ell\in\mathbb{R}^{r\times d}$. This reduces the storage from $O(WLd)$ to $O(Lr+Wrd)$ ("one shared basis + $W$ specific coefficients"). Both keys and values follow this process, with rank ratios typically set to 1:1.5. Compared to single-layer SVD—which stores similar principal directions for every layer and drops in accuracy quickly (45.71 at 8.4x on Llama)—cross-layer factorization merges these redundant bases, preserving significantly more effective information at the same compression rate.

**3. Selective Reconstruction (SR): Reducing Reconstruction Cost from Full Sequence to Query-Related Tokens**

While compression saves memory, reconstructing the full cache of $L$ tokens ($\hat{X}_\ell=A_k B_\ell$) during every decoding step introduces reconstruction FLOPs that grow linearly with $L$. This can create a bottleneck—tests showed dense reconstruction slowing kernels down to 0.4x at 122k context. Selective Reconstruction leverages the intrinsic sparsity of LLM attention: for each decoding step, head, and layer, approximate attention (via landmark-guided chunk selectors) identifies the token set $\mathcal{S}_{t,\ell,g}$ most relevant to the current query. Only these rows are reconstructed: $\hat{X}_{\ell,g}[\mathcal{S}_{t,\ell,g},:]=A_k[\mathcal{S}_{t,\ell,g},:]B_{\ell,g}$. Since $|\mathcal{S}|$ is fixed and much smaller than $L$, overhead is decoupled from context length. Unlike ShadowKV, which uses single-layer SVD and must offload values to CPU due to fidelity loss, xKV-SR maintains high fidelity via cross-layer factorization, keeping both keys and values in GPU HBM to avoid PCIe bottlenecks, resulting in a 4.23x end-to-end throughput gain at 122k context.

### Loss & Training
xKV introduces no training loss and requires no fine-tuning. It is a post-training, plug-and-play inference-time compression method. After prefill, it performs online low-rank decomposition on the generated cache. The authors implemented a custom randomized SVD kernel using 16-bit GEMM and shifted Cholesky QR to minimize online decomposition overhead; for Llama-3.1-8B at 64k to 256k context, SVD time ranges from 5.72% down to 1.24% of prefill time.

## Key Experimental Results

### Main Results
RULER 64K is the core experiment. The following table shows average scores and compression rates, highlighting the differences between xKV and baselines.

| Model | Method | Type | KV Compression | RULER Avg. | Description |
|------|------|------|-----------|------------|------|
| Llama-3.1-8B-Instruct | Full Attention | No Comp. | 1.00x | 91.89 | Original KV-cache |
| Llama-3.1-8B-Instruct | KIVI-2 | Intra-layer Quant. | 7.10x | 86.87 | 2-bit KV Quantization |
| Llama-3.1-8B-Instruct | SnapKV | Intra-layer Eviction | 8.00x | 89.68 | First-round attention driven |
| Llama-3.1-8B-Instruct | Single SVD | Intra-layer Low-rank | 8.40x | 45.71 | Independent SVD, massive drop |
| Llama-3.1-8B-Instruct | MiniCache | Cross-layer Merging | 1.30x | 45.04 | Unstable cosine assumption |
| Llama-3.1-8B-Instruct | xKV | Cross-layer Low-rank | 8.03x | 88.50 | Compares to strong eviction baselines |
| Qwen2.5-14B-Instruct-1M | Full Attention | No Comp. | 1.00x | 93.36 | Original KV-cache |
| Qwen2.5-14B-Instruct-1M | SnapKV | Intra-layer Eviction | 6.00x | 91.66 | Qwen has fewer KV heads |
| Qwen2.5-14B-Instruct-1M | Single SVD | Intra-layer Low-rank | 6.35x | 71.79 | Single-layer loss still significant |
| Qwen2.5-14B-Instruct-1M | MiniCache | Cross-layer Merging | 1.30x | 13.78 | Nearly unusable on this model |
| Qwen2.5-14B-Instruct-1M | xKV-4 | Cross-layer Low-rank | 6.21x | 90.19 | ~3.17 point gap from full attn |

### Ablation Study
Selective reconstruction and window size are the critical factors. The first set shows xKV-SR maintains accuracy while keeping the cache on the GPU; the second confirms the contribution of the "cross-layer" aspect.

| Configuration | Comp./Memory Effect | RULER Avg. | Description |
|------|---------------|------------|------|
| Full Attention | 1.00x | 91.89 | Llama-3.1-8B-Instruct baseline |
| Quest | 1.00x KV, ~8.00x GPU memory red. | 84.87 | Dynamic token loading only |
| ShadowKV | 1.64x KV, ~9.08x GPU memory red. | 87.17 | Single-layer SVD + value offloading |
| xK-SR | 1.63x KV, ~8.90x GPU memory red. | 89.70 | Replacing ShadowKV key comp. with xKV |
| ShadowKV‡ | 5.51x KV | 70.94 | Accuracy drops if value is further compressed |
| xKV-SR | 5.35x KV | 89.69 | Both K/V cross-layer compressed and on GPU |

| Window Size | xKV Avg. | xK-SR Avg. | xKV-SR Avg. | Conclusion |
|----------|----------|-------------|--------------|------|
| 1 | 45.71 | 87.17 | 72.27 | Low fidelity when degraded to single-layer |
| 2 | 75.15 | 88.43 | 86.06 | Significant gain with just 2 layers |
| 4 | 88.50 | 89.70 | 89.69 | Optimal balance between accuracy and overhead |
| 8 | 88.91 | 89.74 | 89.72 | Marginal gains; saturation reached |

### Key Findings
- The core advantage of xKV is not "low-rank" per se, but shifting low-rank from intra-layer to cross-layer. When the window size increases from 1 to 4, dense xKV score jumps from 45.71 to 88.50, proving cross-layer bases are the primary source of gain.
- MiniCache's poor performance (45.04 on Llama, 13.78 on Qwen at 1.3x) suggests token-level cosine similarity is insufficient for cross-layer merging, supporting the authors' use of CKA.
- xKV-SR demonstrates clear system value: it maintains an 89.69 average score at 5.35x KV compression and yields a 4.23x throughput boost at 122k context.
- In multi-turn NIAH, methods like SnapKV/PyramidKV drop in performance in later turns because they prune tokens based on the first query. xKV preserves a low-rank approximation of the global information, making it more stable across turns.

## Highlights & Insights
- The most significant insight is redefining "cross-layer similarity" as subspace similarity rather than token representation similarity. This explains why previous merging methods were fragile and gives a modern utility to SVD in the KV-cache context.
- xKV bridges the gap between algorithmic compression and system throughput. While many methods report only memory or accuracy, xKV explicitly addresses how dense reconstruction can become a bottleneck and solves it via selective reconstruction.
- The method is training-free and plug-and-play, making it highly compatible with existing long-context models without requiring large-scale retraining resources.
- The window size ablation proves cross-layer redundancy is real. The saturation at $W=8$ suggests that deployments do not need excessively large layer groups, thus avoiding excessive prefill buffering and SVD costs.

## Limitations & Future Work
- The study primarily focuses on long-prefill scenarios where the initial context is compressed, but subsequent generated tokens are not. For long generation or test-time scaling, the growth of new KV-cache might still pose a bottleneck.
- Ranks, window sizes, and K/V ratios are mostly fixed through strategies. The appendix suggests different tasks have varying tolerances for K/V compression, indicating potential for task-aware dynamic rank allocation.
- The method relies on online SVD after prefill. While the relative overhead is low in long-context scenarios, the cost and implementation complexity for short contexts or high-QPS scenarios still require further evaluation.
- Selective reconstruction relies on approximate attention to select tokens; its robustness depends on the quality of these selectors. If a task requires extremely fragmented evidence, a fixed token budget might be insufficient.

## Related Work & Insights
- **vs KIVI / KV Quantization**: Quantization reduces bits per element, while xKV reduces subspace dimensionality. They are orthogonal; the appendix shows xKV can be combined with 4-bit/3-bit quantization for further compression.
- **vs SnapKV / PyramidKV**: Token eviction discards history, which is risky for multi-turn or non-concentrated attention patterns. xKV preserves a global low-rank approximation, offering more stability.
- **vs MiniCache**: MiniCache assumes adjacent layers are identical at the token level. xKV is more robust by discovering shared principal singular vectors even when token-level differences exist.
- **vs ShadowKV**: While xKV adopts the selective reconstruction idea from ShadowKV, its cross-layer factorization provides higher fidelity, allowing the value cache to be compressed and kept on the GPU rather than offloaded.
- **Insight**: Inference acceleration problems should perhaps move beyond element-wise similarity and focus on representation subspaces, Gram structures, and spectral energy distributions. This perspective could benefit MoE activation caches, visual token caches, and diffusion feature caches.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Using CKA to motivate cross-layer shared singular vectors is a distinctive and insightful application of SVD.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage includes RULER, LongBench, NIAH, window sizes, K/V ratios, quantization tiling, and throughput.
- Writing Quality: ⭐⭐⭐⭐☆ Clear logic and solid systems-level experiments.
- Value: ⭐⭐⭐⭐⭐ Highly practical for long-context LLM deployment, especially for training-free compression and GPU batch scaling.

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
