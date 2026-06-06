---
title: >-
  [Paper Note] Embedding Compression via Spherical Coordinates
description: >-
  [ICLR 2026][Model Compression][embedding compression] This paper proposes an embedding compression method based on spherical coordinate transformation. By exploiting the mathematical property that angular coordinates of…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "embedding compression"
  - "spherical coordinates"
  - "IEEE 754"
  - "lossless compression"
  - "unit vectors"
date: 2026-05-08
content_hash: 9601e6b5ec55e6d2
---

# Embedding Compression via Spherical Coordinates

**Conference**: ICLR 2026
**arXiv**: [2602.00079](https://arxiv.org/abs/2602.00079)  
**Code**: None (algorithm fully specified in the paper)  
**Area**: Model Compression / Embedding Storage
**Keywords**: embedding compression, spherical coordinates, IEEE 754, lossless compression, unit vectors

## TL;DR

This paper proposes an embedding compression method based on spherical coordinate transformation. By exploiting the mathematical property that angular coordinates of high-dimensional unit vectors concentrate near $\pi/2$, the method substantially reduces the entropy of the exponent bits and high-order mantissa bits in IEEE 754 floating-point representations, achieving a 1.5× compression ratio — a 25% improvement over the best lossless methods — with reconstruction error below float32 machine precision.

## Background & Motivation

Embedding vectors are fundamental to RAG, retrieval, and multimodal systems. A typical 1024-dimensional float32 vector requires 4 KB of storage, and 100 million such vectors require 400 GB. For multi-vector representations such as ColBERT, with approximately 100 vectors per document, storage demand increases by a factor of 100. Existing lossless compression methods (e.g., ZipNN) achieve only ~1.2× compression on float32 embeddings, because the entropy of float32 mantissa bits approaches the maximum (~7.3 bits/byte), leaving a theoretical upper bound of only 1.33× even with perfect exponent compression.

**Core insight**: Most embedding models output unit-norm vectors ($\|x\|_2=1$) for cosine similarity computation, meaning the vectors lie on the high-dimensional hypersphere $S^{d-1}$. Yet all existing lossless methods ignore this geometric structure. A unit vector can be equivalently represented by $d-1$ spherical coordinate angles, and in high dimensions these angles are mathematically known to concentrate near $\pi/2 \approx 1.57$ — a well-established result in probability theory.

**Key Insight**: Spherical coordinate transformation is used as an entropy-reduction preprocessing step, making both exponents and mantissas more predictable at the IEEE 754 level before applying standard byte shuffling and entropy coding.

## Method

### Overall Architecture

Compression pipeline: Cartesian coordinates → spherical coordinate transformation → transposition (aggregating same-position angles) → byte shuffling (separating exponent/mantissa bytes) → zstd compression. Decompression proceeds in reverse.

### Key Designs

1. **Entropy reduction via spherical coordinate transformation**: Cartesian coordinate values fall in the range $[\pm 0.001, \pm 0.3]$, requiring 22–40 distinct IEEE 754 exponent values. The first $d-2$ spherical angles lie in $[0, \pi]$ and concentrate near $\pi/2 \approx 1.57$. Validation on jina-embeddings-v4 (2048-dim) shows that Cartesian coordinates require 23 distinct exponent values, whereas 99.7% of spherical angles have exponent 127. Exponent entropy drops from 2.6 bits/byte to 0.03 bits/byte.

2. **Additional gains from mantissa bytes**: Exponent compression alone theoretically contributes only ~0.1× additional compression. The critical extra gain comes from the high-order mantissa bytes: when angles cluster near $\pi/2 \approx 1.5708$, the high-order bytes encoding the fractional part of the IEEE 754 mantissa also become predictable. Empirically, high-order mantissa byte entropy decreases from 8.0 to 4.5 bits/byte, contributing an additional ~11% compression saving.

3. **Implicit gain from the dimension reduction**: A $d$-dimensional unit vector requires only $d-1$ angles (the radius is fixed at 1), directly reducing data volume by $1/d$.

4. **Direct similarity computation in spherical coordinates**: Cosine similarity can be computed directly in spherical coordinate space without reconstructing Cartesian vectors. Through backward recursion in $O(d)$ time: $R \leftarrow \cos\theta_k\cos\phi_k + \sin\theta_k\sin\phi_k \cdot R$. This supports streaming decompression with early termination for top-$k$ retrieval.

### Loss & Training

- **No training required**: the method is a purely mathematical transformation with no learnable parameters.
- The spherical coordinate transformation is exactly invertible, but floating-point transcendental functions introduce bounded errors.
- Intermediate computations use double precision to keep reconstruction error below 1e-7, which is below the float32 machine epsilon of 1.19e-7.
- Transformation complexity is $O(nd)$; a C implementation achieves throughput >1 GB/s, with an end-to-end pipeline speed of 487 MB/s encoding at zstd level 1.

## Key Experimental Results

### Main Results

**Table 1: Baseline comparison (jina-embeddings-v4, 2048-dim, 7,600 vectors)**

| Method | Size (MB) | Ratio | Max Error | Cos Max Error |
|--------|-----------|-------|-----------|---------------|
| Raw float32 | 59.38 | 1.00× | 0 | 0 |
| ZipNN (baseline) | 49.57 | 1.20× | 0 | 0 |
| Truncate 6 bits | 40.30 | 1.55× | 2e-6 | 5e-6 |
| **Spherical (Ours)** | **37.59** | **1.58×** | **9e-8** | **2e-7** |

**Table 2: Compression results across 26 embedding configurations (selected)**

| Model | Dim | Ratio | Gain over Baseline |
|-------|-----|-------|--------------------|
| MiniLM | 384 | 1.50× | +26.0% |
| BGE-base | 768 | 1.52× | +27.3% |
| GTE-large | 1024 | 1.58× | +29.0% |
| jina-embeddings-v4 | 2048 | 1.59× | +31.8% |
| jina-colbert-v2 (multi-vector) | 1024 | 1.52× | +26.5% |
| jina-clip-v2 (image) | 1024 | 1.50× | +24.9% |

### Ablation Study

- Compression ratios range from 1.47× to 1.59× consistently across all 26 configurations, indicating that the gains stem from the unit-norm constraint rather than modality-specific properties.
- Higher dimensionality yields better compression (384-dim: 1.50× → 2048-dim: 1.59×), consistent with high-dimensional spherical concentration theory.
- At the same compression ratio, the spherical coordinate method achieves reconstruction error 10× lower than mantissa truncation.

### Key Findings

- Lossless compression of float32 embeddings has a theoretical ceiling of 1.33× (from exponent compression alone); the spherical coordinate method surpasses this by simultaneously reducing mantissa entropy.
- In ColBERT settings, a 1-million-document index shrinks from 240 GB to 160 GB, which is practically significant.
- Reconstruction error below float32 machine precision has no impact on any retrieval quality metric.

## Highlights & Insights

- The approach is remarkably elegant: a mathematical fact (high-dimensional spherical angle concentration) is leveraged to solve an engineering problem (floating-point compression).
- The analysis operates at the bit level of IEEE 754, precisely bridging geometric properties and floating-point representation.
- The method requires no training and no codebook, and is applicable to any embedding model that produces unit vectors.
- Direct similarity computation in spherical coordinates opens up the possibility of streaming retrieval.

## Limitations & Future Work

- Applicable only to float32 embeddings; BF16 (8-bit mantissa) or INT8 require different strategies.
- Although surpassing lossless methods, the 1.5× compression ratio remains orders of magnitude below lossy quantization (4–32×).
- Vectors must be strictly unit-norm; non-normalized vectors are not supported.
- The last angle $\theta_{d-1} \in [-\pi, \pi]$ does not concentrate and represents a bottleneck for compression.

## Related Work & Insights

- **ZipNN** (Hershcovitch et al., 2025): byte shuffling + entropy coding lossless baseline; this work adds spherical coordinate preprocessing on top.
- **PolarQuant** (Han et al., 2025): polar coordinates for lossy KV cache quantization; this work instead achieves near-lossless compression of embeddings.
- **ECF8/DFloat11**: exploit natural exponent concentration in model weights; this work creates concentration through a deterministic geometric transformation.
- **Insight**: The strategy of exploiting geometric constraints in data to reduce information entropy generalizes to other floating-point data with geometric structure.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Highly original perspective connecting high-dimensional geometry with floating-point representation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage across 26 configurations, though end-to-end retrieval performance validation is absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Exceptionally clear and concise, with complete algorithmic descriptions and naturally motivated derivations.
- Value: ⭐⭐⭐⭐ A plug-and-play compression solution with practical value for large-scale vector database deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] ArcVQ-VAE: A Spherical Vector Quantization Framework with ArcCosine Additive Margin](../../ICML2026/model_compression/arcvq-vae_a_spherical_vector_quantization_framework_with_arccosine_additive_marg.md)
- [\[ICML 2026\] Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models](../../ICML2026/model_compression/dispersion_loss_counteracts_embedding_condensation_and_improves_generalization_i.md)
- [\[ICLR 2026\] A universal compression theory for lottery ticket hypothesis and neural scaling laws](a_universal_compression_theory_for_lottery_ticket_hypothesis_and_neural_scaling_.md)
- [\[ICLR 2026\] FreqKV: Key-Value Compression in Frequency Domain for Context Window Extension](freqkv_key-value_compression_in_frequency_domain_for_context_window_extension.md)
- [\[ICLR 2026\] Cut Less, Fold More: Model Compression through the Lens of Projection Geometry](cut_less_fold_more_model_compression_through_the_lens_of_projection_geometry.md)

</div>

<!-- RELATED:END -->
