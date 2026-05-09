---
title: >-
  [Paper Note] Robustifying Learning-Augmented Caching Efficiently without Compromising 1-Consistency
description: >-
  [NeurIPS 2025][Model Compression][Online caching] This paper proposes Guard, a lightweight robustification framework that improves the robustness of a broad class of learning-augmented caching algorithms to $2H_{k-1}+2$ while preserving 1-consistency and incurring only O(1) additional overhead per request.
tags:
  - NeurIPS 2025
  - Model Compression
  - Online caching
  - learning-augmented algorithms
  - robustification
  - 1-consistency
  - competitive ratio
date: 2026-05-08
content_hash: bb2249001bf6ef7b
---

# Robustifying Learning-Augmented Caching Efficiently without Compromising 1-Consistency

**Conference**: NeurIPS 2025
**arXiv**: [2507.16242](https://arxiv.org/abs/2507.16242)
**Code**: Available
**Area**: Algorithms / Learning-Augmented Algorithms
**Keywords**: Online caching, learning-augmented algorithms, robustification, 1-consistency, competitive ratio

## TL;DR

This paper proposes Guard, a lightweight robustification framework that improves the robustness of a broad class of learning-augmented caching algorithms to $2H_{k-1}+2$ while preserving 1-consistency and incurring only O(1) additional overhead per request.

## Background & Motivation

### State of the Field

**Background**: The online caching problem requires minimizing cache misses under a cache of size $k$. Learning-augmented caching algorithms leverage ML predictions to guide eviction decisions, facing an inherent consistency–robustness tradeoff:

### Limitations of Prior Work

**Limitations of Prior Work**: **1-consistency**: achieves optimality (competitive ratio of 1) when predictions are perfect.

### Root Cause

**Key Challenge**: **Robustness**: performance does not degrade arbitrarily when predictions are severely inaccurate.

Challenges with existing approaches:

**Naïve methods** (e.g., BlindOracle): 1-consistent but without robustness guarantees — performance can be arbitrarily bad under mispredictions.

**Marking methods** (e.g., PredictiveMarker): $4H_k$-robust but not 1-consistent.

**Switching methods** (e.g., F&R): $O(\log k)$-robust and 1-consistent, but computationally expensive at $O(n^2 \log k)$, requiring recomputation of the optimal solution.

Core problem: **Can robustness be efficiently enhanced while preserving 1-consistency?**

## Method

### Overall Architecture

Guard is a general robustification framework applicable to any **RB-following** (relaxed-Belady-following) algorithm. Its design is grounded in three key observations:

1. **Immediate protection harms consistency**: Marking methods protect pages after every request, which precludes 1-consistency.
2. **Error detection must balance precision and efficiency**: The optimal solution recomputation in switching methods is prohibitively expensive.
3. **Structural properties of RB-compliant algorithms**: Optimal eviction behavior exhibits exploitable causal chains.

### Key Designs

**Definition of RB-following algorithms**:
- Under accurate predictions, pages labeled as 1-pages (Belady label 1) are evicted before 0-pages.
- This class includes a broad range of algorithms such as BlindOracle, LRB, and Parrot.

**Phase mechanism of Guard**:
- Execution is divided into phases.
- At the start of each phase, all cached pages are marked as "old pages" and added to set $\mathcal{U}$.
- A new phase begins when $\mathcal{U}$ becomes empty.

**Prediction error detection**:
- A misprediction is detected when a cache miss occurs for a page that has already been evicted in the current phase.
- In this case, a random unprotected page from $\mathcal{U}$ is evicted, and the requested page is *guarded*.
- A guarded page will not be evicted again within the current phase.

**Lightweight design**:
- Only $\mathcal{U}$ and guard flags need to be maintained.
- Implemented with hash tables for O(1) per-request overhead.
- No recomputation of the optimal solution is required.

### Loss & Training

Theoretical analysis framework:
- **1-consistency proof** (Proposition 4.1): Under perfect predictions, no pages are guarded, and Guard&A behaves identically to A.
- **Robustness proof** (Theorem 4.6): By analyzing the number of old and new page evictions per phase, an upper bound of $2H_{k-1}+2$ is established.

## Key Experimental Results

### Main Results

Average cost ratio on SPEC CPU2006 with PLECO/POPU predictors:

| Algorithm | Prediction | Avg. Cost Ratio | 1-Consistent? | Robust? |
|-----------|-----------|----------------|--------------|--------|
| LRU | None | 1.478 | N/A | N/A |
| Marker | None | 1.404 | No | Yes |
| BlindOracle | NRT | 1.335 | Yes | No |
| PredictiveMarker | NRT | 1.346 | No | Yes |
| LMarker | NRT | 1.335 | No | Yes |
| B.O.&M.D | NRT | 1.294 | No | Yes |
| F&R | NRT | 1.360 | Yes | Yes |
| **Guard&B.O.** | **NRT** | **1.226** | **Yes** | **Yes** |

Results using the LRB predictor:

| Algorithm | Predictor | Avg. Cost Ratio |
|-----------|----------|----------------|
| LRU | - | 1.478 |
| Marker | - | 1.394 |
| LRB | LRB | 1.281 |
| MARK0 | LRB | 1.268 |
| **Guard&LRB** | **LRB** | **1.171** |

### Ablation Study

| Dataset / Condition | Guard Effect | Analysis |
|--------------------|-------------|----------|
| BrightKite (synthetic NRT noise) | Low cost ratio maintained as noise increases | Robustness validation |
| BrightKite (binary label flipping) | Outperforms baselines even at high flip rates | Misprediction tolerance |
| SPEC CPU (multiple datasets) | Best or near-best across 13 datasets | Generalizability |
| Varying cache sizes | Effective at both $k=10$ and $k=100$ | Scalability |

### Key Findings

1. **Guard achieves the state-of-the-art consistency–robustness tradeoff**: $(1, 2H_{k-1}+2)$.
2. **Significant efficiency advantage**: O(1) overhead vs. $O(n^2 \log k)$ for F&R.
3. **Comprehensive empirical superiority**: Lowest cost ratio under both synthetic and real predictors.
4. **Strong generality**: Successfully applied to BlindOracle, LRB, and Parrot — three distinct prediction paradigms.

## Highlights & Insights

1. **Elegant phase mechanism**: Prediction errors are detected at low cost by tracking whether old pages are re-requested.
2. **Theory–practice alignment**: Theoretical guarantees are thoroughly validated by experiments.
3. **Plug-and-play design**: Can directly augment any RB-following algorithm without modifying its core logic.
4. **O(1) overhead**: Robustness is achieved without increasing asymptotic complexity.

## Limitations & Future Work

1. Whether the robustness bound $2H_{k-1}+2$ can be further tightened (exploration of theoretical lower bounds).
2. Guard is currently limited to RB-following algorithms; extension to a broader algorithm class may be valuable.
3. Smoothness optimization via ExGuard shows progress but leaves room for further improvement.
4. Real-system performance under very large caches and high concurrency remains to be validated.
5. Integration with distributed caching settings has not been explored.

## Related Work & Insights

- **BlindOracle** (Lykouris & Vassilvtiskii 2018): The first learning-augmented caching algorithm.
- **LMarker** (Rohatgi 2020): An improvement over marking-based methods.
- **F&R** (Sadek & Elias 2024): Achieves 1-consistency and robustness via switching, but is computationally expensive.
- **LRB** (Song et al. 2020): A practical learning-based method for CDN caching.
- Insight: Simple and principled mechanism design often offers greater practical value than complex optimization.

## Rating

- Novelty: ⭐⭐⭐⭐ (phase-based error detection mechanism is novel)
- Technical Depth: ⭐⭐⭐⭐⭐ (rigorous and complete theoretical proofs)
- Experimental Thoroughness: ⭐⭐⭐⭐ (validated across multiple datasets and predictors)
- Writing Quality: ⭐⭐⭐⭐⭐ (directly integrable into caching systems)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Representation Consistency for Accurate and Coherent LLM Answer Aggregation](representation_consistency_for_accurate_and_coherent_llm_answer_aggregation.md)
- [\[NeurIPS 2025\] Elastic ViTs from Pretrained Models without Retraining](elastic_vits_from_pretrained_models_without_retraining.md)
- [\[NeurIPS 2025\] AutoJudge: Judge Decoding Without Manual Annotation](autojudge_judge_decoding_without_manual_annotation.md)
- [\[ICLR 2026\] LightMem: Lightweight and Efficient Memory-Augmented Generation](../../ICLR2026/model_compression/lightmem_lightweight_and_efficient_memory-augmented_generation.md)
- [\[AAAI 2026\] Prototype-Based Semantic Consistency Alignment for Domain Adaptive Retrieval](../../AAAI2026/model_compression/prototype-based_semantic_consistency_alignment_for_domain_adaptive_retrieval.md)

<!-- RELATED:END -->
