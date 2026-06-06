---
title: >-
  [Paper Note] Intermediate N-Gramming: Deterministic and Fast N-Grams For Large N and Large Datasets
description: >-
  [AAAI 2026][n-gram computation] This paper proposes Intergrams, a multi-pass scanning algorithm that recursively uses shorter n-grams as prefixes to filter candidates for longer n-grams…
tags:
  - "AAAI 2026"
  - "n-gram computation"
  - "hardware-aware algorithms"
  - "cache-friendly"
  - "Zipf distribution"
  - "multi-pass scanning"
date: 2026-05-08
content_hash: b3d99823b3042f73
---

# Intermediate N-Gramming: Deterministic and Fast N-Grams For Large N and Large Datasets

**Conference**: AAAI 2026
**arXiv**: [2511.14955](https://arxiv.org/abs/2511.14955)  
**Code**: [GitHub](https://github.com/rcurtin/Intergrams)  
**Area**: High-Performance Computing / Data Mining
**Keywords**: n-gram computation, hardware-aware algorithms, cache-friendly, Zipf distribution, multi-pass scanning

## TL;DR

This paper proposes Intergrams, a multi-pass scanning algorithm that recursively uses shorter n-grams as prefixes to filter candidates for longer n-grams, fully exploiting the processor cache hierarchy to achieve cache-friendly memory access patterns. On TB-scale datasets, Intergrams achieves 6–33× speedup over the previously fastest hash-gramming method while recovering nearly all top-k n-grams with high accuracy.

## Background & Motivation

N-gram features are widely used across multiple domains: byte-sequence features are used to train classifiers for malware detection; character-level n-grams in NLP are competitive with embedding models on certain tasks; and k-mers in genomics are used for species classification and phylogenetic analysis. A common pipeline in practice is to first compute the top-k most frequent n-grams in a dataset and then transform each sample into a sparse feature vector for training.

However, the number of possible n-grams grows exponentially with $n$ (for byte data, 8-grams yield $256^8 \approx 1.8 \times 10^{19}$ possibilities), making the computation of top-k n-grams the most time-consuming bottleneck in the entire pipeline. Among existing methods, scikit-learn's CountVectorizer requires maintaining a full dictionary and does not scale for non-trivial $n$ and $k$. Streaming algorithms (e.g., Space-Saving) require far more than $k$ counters and are ill-suited for the "count each sequence only once" setting. While the current fastest hash-gramming approach is substantially faster than naive methods, its throughput of only 10–20 MB/s falls far short of disk bandwidth (~600 MB/s).

The **root cause** is that existing algorithms ignore the hardware memory hierarchy. Hash-gramming performs random accesses over a bucket array of size $2^{31}$, which far exceeds processor cache capacity, resulting in massive cache misses, TLB misses, and page faults that severely degrade throughput. The authors' core insight is: **if intermediate data structures are kept within cache capacity and large arrays are accessed sequentially, n-gram computation can approach the disk I/O speed limit.**

## Method

### Overall Architecture

Intergrams employs a "recursive multi-pass scanning" strategy. It first computes the top-$zk$ 3-grams (where $z$ is an oversampling factor) using a highly optimized dedicated algorithm, then uses these 3-grams as prefix filters to identify candidate 4-grams, computes the top-$zk$ 4-grams, then uses 4-gram prefixes to filter candidate 5-grams, and so on up to the target length $n$. Each round requires only a single pass over the entire dataset. Because prefix filtering dramatically reduces the candidate set, intermediate data structures remain cache-friendly throughout.

### Key Designs

1. **Fast 3-gram Algorithm (Base Step)**:

    - **Function**: Exploits the enumerability of 3-grams ($256^3 = 16\text{M}$ possibilities) by counting directly with a bit vector.
    - **Mechanism**: For each sequence $s_i$, a 2 MB bit vector $C_i$ ($16\text{M}$ bits) is maintained to mark which 3-grams appear; it is then flushed sequentially into a global count array $C$ ($16\text{M} \times 4$ bytes = 64 MB). The bit vector involves random writes but is only 2 MB, fitting comfortably in L2 cache; the global array is 64 MB but is accessed sequentially, enabling hardware prefetching for high bandwidth.
    - **Design Motivation**: The 2 MB bit vector fits in L2 cache; the 64 MB global array fits in L3 cache; sequential access patterns avoid cache misses. AVX2 SIMD instructions further accelerate the flush operation (5 SIMD instructions can increment 8 elements of a bit vector into $C$), achieving near-peak disk bandwidth of ~5 GB/s on capable hardware.

2. **Recursive Prefix Filtering (Core of Intergrams)**:

    - **Function**: Extends from 3-grams to n-grams of arbitrary length, retaining only $zk$ candidate prefixes at each step.
    - **Mechanism**: At step $j$ ($j \geq 4$), only $j$-grams whose $(j-1)$-gram prefix belongs to the previous top-$zk$ set $P^{(j-1)}$ are counted. The candidate count drops from $256^j$ to $256 \times zk$, and the bit vector size shrinks to $32zk$ bytes, typically well within cache capacity.
    - **Design Motivation**: This exploits the empirical observation that n-gram frequencies in real data follow a Zipf distribution—prefix frequency is highly correlated with the frequency of subsequent n-grams, so a small set of high-frequency prefixes covers the vast majority of high-frequency longer n-grams.

3. **Trie Data Structure Optimization**:

    - **Function**: Efficiently queries whether the prefix of an n-gram belongs to the candidate set $P$.
    - **Mechanism**: A compact trie stores candidate prefixes; each node occupies only 2 bytes (value + child count). Nodes with fewer than 4 children use a list (12 bytes); nodes with 4 or more children use a 256-element lookup table (1 KB). Children are laid out in memory in frequency order so that high-frequency trie paths are spatially contiguous.
    - **Design Motivation**: Trie lookup is in the innermost loop of the algorithm and must be as small as possible to fit in cache; spatial locality along high-frequency paths reduces cache misses. This optimization reduces overall runtime by approximately 10–15%.

4. **Parallelism and I/O Optimization**:

    - **Function**: Fully utilizes multi-core parallelism and disk I/O bandwidth.
    - **Mechanism**: A thread-pair strategy is employed—one thread is dedicated to disk I/O reads while another processes already-buffered data. Since the bottleneck of Intergrams lies in disk I/O rather than computation, the dedicated I/O thread ensures maximum disk throughput.
    - **Design Motivation**: Hash-gramming cannot saturate disk bandwidth because its computational bottleneck prevents overlapping I/O and computation, whereas each sequence in Intergrams is handled by a thread pair that simultaneously saturates both.

### Theoretical Analysis

Assuming n-gram frequencies follow a Zipf distribution with parameter $a$ ($p_i \propto 1/i^a$), the authors prove:
- **Lemma 1**: If the top-$k'$ n-grams account for fraction $\beta$ of total occurrences, then the $(n+1)$-grams prefixed by them account for at least fraction $\beta' = \beta - m/(N-m)$ of all $(n+1)$-gram occurrences.
- **Theorem 1**: The fraction of top-$k$ $(n+1)$-grams recalled by Intergrams is at least $1 - \frac{(|\mathbf{D}_{n+1}|^{1-a} - a)(1-\beta') - a}{(k+1)^{1-a} - 1}$.
- **Theorem 2**: Accounting for sampling noise, with probability $1-\delta$, the same bound holds with $\beta'$ replaced by $\beta'' = \beta' - \Delta(\delta)$, where $\Delta(\delta) = 4\sqrt{k^2 \ln(2|\mathbf{D}_n|/\delta) / (2N)}$.

When $a > 1$ (as is typical in real data), the recall rate asymptotically approaches $1 - O((k/|D_{n+1}|)^{a-1})$, converging rapidly to 1 as $k$ grows.

## Key Experimental Results

### Datasets

| Dataset | Type | Size | # Sequences |
|---------|------|------|-------------|
| EMBER | Bytes (executables) | 1009 GB | 800k |
| C4 | Text | 751 GB | 6.22M |
| 1000gp | Genomics | 1.4 TB | 1.58M |

### Main Results (Runtime and Speedup, $n=6$, varying $k$)

| Algorithm | EMBER ($k$=100k) | C4 ($k$=10k) | 1000gp ($k$=10k) |
|-----------|-----------------|--------------|-----------------|
| hg-vanilla | 9078.5s (1×) | 39787.1s (1×) | 10042.0s (1×) |
| hg-fast (all optimizations) | 7162.6s (1.27×) | 30811.2s (1.29×) | 7061.5s (1.42×) |
| Intergrams, $z$=1 | 1413.7s (6.42×), Jaccard=0.71 | 1183.9s (33.6×), Jaccard=0.91 | 1215.7s (8.26×), Jaccard=1.0 |
| Intergrams, $z$=1.5 | 1458.8s (6.22×), Jaccard=0.91 | 1373.7s (29.0×), Jaccard=1.0 | 1152.9s (8.71×), Jaccard=1.0 |
| Intergrams, $z$=2 | 1771.4s (5.13×), Jaccard=0.97 | 1501.9s (26.5×), Jaccard=1.0 | 1144.6s (8.77×), Jaccard=1.0 |

### Ablation Study (Per-Step Runtime Breakdown, EMBER, $k$=10k, $z$=1.5)

| Step | Runtime | Throughput |
|------|---------|------------|
| 3-gram scan | 228.2s | 4.42 GB/s |
| Top-$zk$ 3-gram / trie construction | 3.25s | — |
| 4-gram scan | 205.95s | 4.90 GB/s |
| 5-gram scan | 189.37s | 5.33 GB/s |
| 6-gram scan | 190.57s | 5.29 GB/s |

### Key Findings
- Intergrams consistently and substantially outperforms hash-gramming (up to 33× speedup) across all three TB-scale datasets from different domains.
- Even with oversampling factor $z=1$, Intergrams perfectly recovers all top-k n-grams on C4 and 1000gp (Jaccard=1.0).
- Per-step throughput approaches or reaches peak disk bandwidth (~5 GB/s), validating the design goal of making disk I/O the sole bottleneck.
- Later passes run faster than earlier ones, as trie filtering eliminates more candidates and reduces the number of actual updates.
- The various optimizations applied to hash-gramming (huge pages, prefetching, cuckoo hashing, trie) yield limited speedup (at most 1.42×), because the fundamental issue—random memory access patterns—remains unaddressed.

## Highlights & Insights
- An exemplary case of hardware-algorithm co-design: rather than fine-tuning at the algorithmic level, the entire computational flow is redesigned from the perspective of the hardware memory hierarchy, achieving order-of-magnitude speedups.
- Theory and empirics are in perfect agreement: the theoretical recall guarantees derived under the Zipf distribution assumption are validated experimentally.
- The recursive extension from 3-grams to arbitrary n-grams is elegant—an exponentially large search space is resolved through a series of linear-cost steps.
- Highly practical: the algorithm is deterministic, parallelizable, and open-sourced, making it directly applicable to malware detection, NLP, and genomics pipelines.
- The oversampling factor $z$ provides a clean and intuitive accuracy-speed trade-off knob.

## Limitations & Future Work
- The current implementation is optimized for byte sequences (alphabet size = 256); adapting to larger alphabets (e.g., Unicode) would require modifications to the trie structure and the 3-gram base step.
- The theoretical analysis assumes a constant Zipf parameter across different values of $n$; in practice, longer n-grams tend to exhibit larger Zipf parameters (acknowledged by the authors as a pessimistic assumption).
- For very small $n$ (e.g., $n=3$), Intergrams offers no advantage over direct enumeration.
- Distributed (multi-node) scenarios are not discussed; the current implementation is limited to single-machine multi-core settings.

## Related Work & Insights
- Hash-gramming (Raff 2018) was the prior fastest method but ignores hardware constraints; the central contribution of this paper is to demonstrate that algorithm design must be hardware-aware.
- The spirit of cache-oblivious algorithms (Frigo 1999) is well instantiated in this work.
- The recursive filtering approach may inspire analogous optimizations in other frequent pattern mining scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ (The core ideas of hardware-awareness and recursive prefix filtering are clean, though not an entirely new paradigm)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (TB-scale datasets across three domains, with detailed ablation and per-step breakdown)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear exposition, accessible hardware analysis, tight integration of theory and experiments)
- Value: ⭐⭐⭐⭐ (High practical value, though the application scope is relatively specific)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Torus Graphs for Large-Scale Neural Phase Analysis](../../ICML2026/others/torus_graphs_for_large_scale_neural_phase_analysis.md)
- [\[ICML 2026\] HASTE: Hardware-Aware Dynamic Sparse Training for Large Output Spaces](../../ICML2026/others/haste_hardware-aware_dynamic_sparse_training_for_large_output_spaces.md)
- [\[ICCV 2025\] Kaputt: A Large-Scale Dataset for Visual Defect Detection](../../ICCV2025/others/kaputt_a_large-scale_dataset_for_visual_defect_detection.md)
- [\[ICML 2026\] AMDP: Asynchronous Multi-Directional Pipeline Parallelism for Large-Scale Models Training](../../ICML2026/others/amdp_asynchronous_multi-directional_pipeline_parallelism_for_large-scale_models_.md)
- [\[CVPR 2026\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](../../CVPR2026/others/sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)

</div>

<!-- RELATED:END -->
