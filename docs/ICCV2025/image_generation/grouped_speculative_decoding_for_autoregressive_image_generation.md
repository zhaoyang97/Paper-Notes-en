---
title: >-
  [Paper Note] Grouped Speculative Decoding for Autoregressive Image Generation
description: >-
  [ICCV 2025][Image Generation][autoregressive image generation] This paper proposes Grouped Speculative Decoding (GSD), a training-free acceleration method for autoregressive image generation that performs speculative verification at the level of semantically valid token clusters rather than the single most probable token, achieving an average speedup of 3.7× without degrading image quality.
tags:
  - ICCV 2025
  - Image Generation
  - autoregressive image generation
  - speculative decoding
  - inference acceleration
  - training-free
  - visual token redundancy
date: 2026-05-08
content_hash: 5c8b5f368ca45c8c
---

# Grouped Speculative Decoding for Autoregressive Image Generation

**Conference**: ICCV 2025
**arXiv**: [2508.07747](https://arxiv.org/abs/2508.07747)
**Code**: [GitHub](https://github.com/junhyukso/GSD)
**Area**: Image Generation
**Keywords**: autoregressive image generation, speculative decoding, inference acceleration, training-free, visual token redundancy

## TL;DR

This paper proposes Grouped Speculative Decoding (GSD), a training-free acceleration method for autoregressive image generation that performs speculative verification at the level of semantically valid token clusters rather than the single most probable token, achieving an average speedup of 3.7× without degrading image quality.

## Background & Motivation

Autoregressive (AR) image models (e.g., Lumina-mGPT, LlamaGen) have demonstrated impressive generative capabilities, yet face a fundamental challenge: **sequential token-by-token generation**. Producing high-resolution images requires generating thousands of tokens, and meaningful output cannot be rendered until the image is fully decoded.

**The dilemma of Speculative Decoding (SD)**: While SD achieves significant speedups in LLMs (acceptance rate ~70%), its acceptance rate in image AR models is only ~40%, yielding limited gains. Existing image SD methods either provide modest acceleration (SJD: 2.1×) or require additional training (LANTERN: 1.6×).

**Core observation — the fundamental difference of image tokens**:

**Redundancy**: Visual tokens originate from vector-quantized continuous latent spaces and retain low-frequency redundancy; many tokens differ only in subtle high-frequency details yet produce visually similar outputs.

**Diversity**: Without grammatical constraints, multiple valid visual patterns are acceptable as the next token. For 50–95% of tokens, the top-1 probability falls below 5%.

**Key experimental evidence** (Fig. 4): Randomly replacing 50% of tokens with Top-100 candidates leaves overall image quality nearly unaffected (CLIP_score = 32.089), confirming that multiple tokens are simultaneously valid.

## Method

### Problem Analysis: Why SD Fails in Image AR

Using Total Variation (TV) analysis:

$$\alpha_{p,q} = 1 - TV(p,q) = 1 - \frac{1}{2}\sum_x |p(x) - q(x)|$$

Even when both $p(x)$ and $q(x)$ assign non-negligible probability to multiple tokens (low top-1 probability), accumulated small differences between the two distributions result in high TV. The key mechanism: with a vocabulary of ~20,000 tokens, this accumulation effect is amplified — as illustrated by the toy example in Fig. 6, a uniform distribution actually yields higher TV than a concentrated one.

### Core Idea of GSD

The vocabulary $\mathcal{X}$ is partitioned into disjoint clusters: $\mathcal{C} = \{C_1, C_2, \ldots, C_K\}$

Grouped probability mass is defined as:

$$p'(C_i) = \sum_{x \in C_i} p(x), \quad q'(C_i) = \sum_{x \in C_i} q(x)$$

The acceptance criterion is then applied at the cluster level:

$$\min\left(1, \frac{p'(C(x))}{q'(C(x))}\right) \geq r, \quad r \sim \mathcal{U}[0,1]$$

### Theoretical Guarantee (Theorem 1)

**The acceptance rate of GSD is strictly no lower than that of standard SD**:

$$\alpha_{GSD} \geq \alpha_{SD}$$

The proof follows from the triangle inequality: for any cluster $C_i$,

$$\left|\sum_{x \in C_i}(p(x)-q(x))\right| \leq \sum_{x \in C_i}|p(x)-q(x)|$$

Summing over all clusters yields $TV(p', q') \leq TV(p, q)$, hence $\alpha_{GSD} = 1 - TV(p', q') \geq 1 - TV(p, q) = \alpha_{SD}$.

### Context-Aware Dynamic Clustering

**Why static clustering fails**:
1. **Uniformity of the codebook embedding space**: t-SNE visualization (Fig. 3) shows that embeddings are nearly uniformly distributed, making it difficult to form meaningful clusters.
2. **Context-dependence of tokens**: The same token is decoded into different RGB values at different decoding steps (Fig. 8).

**Dynamic GSD strategy**:
- At each step $t$, tokens are sorted by their probability $p(x)$.
- The $G$ tokens with probabilities nearest to that of the token being verified are grouped into one cluster.
- Filtering condition: tokens whose embedding distance exceeds $d$ or whose probability difference exceeds $\delta$ are excluded.

Key insight: The model's predicted probability already encodes contextual information and reflects token similarity more faithfully than static embedding distances.

### Algorithm (Algorithm 3, VERIFY_GSD)

1. For each token $\hat{X}_k$ to be verified:
    - Sort by $p_k$; locate the position of $\hat{X}_k$ in the sorted order.
    - Take the $G/2$ nearest tokens on each side to form cluster $C_{idxs}$.
    - Filter out tokens exceeding the embedding distance or probability difference thresholds.
    - Compute cluster probabilities $p'_C = \sum p_k[C_{idxs}]$ and $q'_C = \sum q_k[C_{idxs}]$.
    - Accept with probability $\min(1, p'_C/q'_C)$.
2. Upon rejection, resample from $[p_k - q_k]_+$.

## Key Experimental Results

### Main Results: Parti-Prompt and MS-COCO (Table 1)

**Parti-Prompt (1,600 prompts)**:

| Configuration | Latency | NFE↓ | Speedup↑ | CLIP Score↑ |
|------|------|------|---------|-----------|
| Vanilla AR | 112.29s | 2392 | 1.00× | 32.091 |
| SJD | 52.34s | 1035 | 2.15× | 32.090 |
| Amplify (k=3) | 31.31s | 586 | 3.59× | 31.774 |
| **Ours (G=50)** | **24.13s** | **637** | **4.65×** | **32.075** |

**MS-COCO (5,000 prompts)**:

| Configuration | Latency | FID↓ | CLIP Score↑ |
|------|------|------|-----------|
| Vanilla AR | 122.45s | 30.79 | 31.308 |
| SJD | 55.26s | 30.78 | 31.308 |
| Amplify (k=4) | 32.29s | 40.05 | 30.99 |
| **Ours (G=25)** | **32.52s** | **33.55** | **31.24** |

Key findings:
- At G=3, the CLIP Score (32.125) actually exceeds that of Vanilla AR (32.091), as cluster-level probability mass smooths out prediction noise.
- Naïve lossy methods (Amplify, Addition) achieve speedups but substantially degrade quality (FID 40.05 vs. 30.79).
- GSD incurs only minimal quality loss at a 3.7× speedup.

### Ablation: Clustering Strategies (Table 2, Parti-Prompt)

| Clustering Method | NFE | CLIP Score |
|----------|-----|-----------|
| Baseline (SJD) | 1035 | 32.090 |
| (A) Embedding distance | 913.80 | 32.081 |
| (B) Draft q(x) | 685.31 | 31.791 |
| (C) Expert p(x) | 636.24 | 32.056 |
| **(D) Ours (dynamic)** | **636.75** | **32.075** |

Static embedding-distance clustering (A) is nearly ineffective; clustering by draft q(x) (B) leads to notable quality degradation; clustering by expert p(x) with filtering (D) achieves the best overall performance.

### Pareto Frontier Analysis (Fig. 10)

GSD strictly dominates naïve lossy methods on the NFE vs. CLIP Score Pareto frontier, and through the smoothing effect even outperforms lossless SJD at certain NFE values.

## Highlights & Insights

1. **Identifying the root cause of SD failure in image AR**: Through TV analysis, the paper establishes the causal chain "low top-1 probability + large vocabulary = high TV," a connection insufficiently recognized in prior work.
2. **Elegance of the theoretical guarantee**: The speedup guarantee of GSD relies solely on the triangle inequality and is independent of the specific clustering strategy — any clustering scheme is guaranteed not to slow down decoding.
3. **"Free lunch" at G=3**: A small group size not only accelerates decoding but also improves quality, as cluster-level probability mass smooths prediction noise — a highly practical finding.
4. **Necessity of dynamic clustering**: Experiments clearly demonstrate the ineffectiveness of static embedding-distance clustering (due to the uniformly distributed codebook space) and the effectiveness of using $p(x)$ as a context-aware similarity measure.

## Limitations & Future Work

1. Validation is limited to Lumina-mGPT (7B); additional AR image models have not been tested.
2. Hyperparameters $G, d, \delta$ require tuning for different models and resolutions.
3. No combination experiments are conducted with orthogonal acceleration methods such as continuous speculative decoding or parallelized AR.
4. The additional computational overhead introduced by the clustering operation may become non-negligible at very large values of $G$.

## Related Work & Insights

- Compared to LANTERN (Jang et al., 2024), GSD requires no training and avoids bias and explosion issues by simultaneously increasing both numerator and denominator.
- GSD is in principle orthogonal to tree-based SD, multi-draft SD, and similar methods, and can be further combined with them.
- The finding of image token redundancy may inspire VQ codebook design — e.g., whether redundant codewords can be reduced at training time.
- The dynamic clustering idea is potentially transferable to other high-entropy token prediction settings, such as protein sequence generation.

## Rating ⭐⭐⭐⭐

Novelty ★★★★☆: The core idea is concise and elegant, with rigorous theoretical analysis.
Experimental Thoroughness ★★★★☆: Quantitative and qualitative evaluations are comprehensive; the Pareto frontier analysis is convincing.
Writing Quality ★★★★★: Problem motivation is clearly developed; the logical chain from observation to theory to method is complete.
Value ★★★★★: Training-free, plug-and-play, with publicly available code — low barrier to adoption.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Annealed Relaxation of Speculative Decoding for Faster Autoregressive Image Generation](../../AAAI2026/image_generation/annealed_relaxation_of_speculative_decoding_for_faster_autor.md)
- [\[CVPR 2026\] SJD-PAC: Accelerating Speculative Jacobi Decoding via Proactive Drafting and Adaptive Continuation](../../CVPR2026/image_generation/sjd-pac_accelerating_speculative_jacobi_decoding_via_proactive_drafting_and_adap.md)
- [\[ICCV 2025\] DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer](dc-ar_efficient_masked_autoregressive_image_generation_with_deep_compression_hyb.md)
- [\[ICCV 2025\] Holistic Tokenizer for Autoregressive Image Generation](holistic_tokenizer_for_autoregressive_image_generation.md)
- [\[ICLR 2026\] Autoregressive Image Generation with Randomized Parallel Decoding](../../ICLR2026/image_generation/autoregressive_image_generation_with_randomized_parallel_decoding.md)

<!-- RELATED:END -->
