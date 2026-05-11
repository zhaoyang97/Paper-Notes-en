---
title: >-
  [Paper Note] FASA: Frequency-Aware Sparse Attention
description: >-
  [ICLR 2026][Signal & Communication][KV cache compression] This paper identifies functional sparsity at the frequency component (FC) level within RoPE—a small subset of "dominant FCs" can effectively predict token importa…
tags:
  - "ICLR 2026"
  - "Signal & Communication"
  - "KV cache compression"
  - "sparse attention"
  - "RoPE frequency analysis"
  - "token pruning"
  - "long-context inference"
date: 2026-05-08
content_hash: 76580a8edc7c7c0a
---

# FASA: Frequency-Aware Sparse Attention

**Conference**: ICLR 2026
**arXiv**: [2602.03152](https://arxiv.org/abs/2602.03152)
**Code**: [GitHub](https://github.com/AMAP-ML/FASA-ICLR2026)
**Area**: Signal Communication
**Keywords**: KV cache compression, sparse attention, RoPE frequency analysis, token pruning, long-context inference

## TL;DR
This paper identifies functional sparsity at the frequency component (FC) level within RoPE—a small subset of "dominant FCs" can effectively predict token importance. Based on this finding, the paper proposes the FASA framework, which achieves training-free KV cache compression via two stages: dominant-FC-based token importance prediction and focused attention computation. On LongBench, retaining only 256 tokens approaches 100% of full-KV performance; on AIME24, FASA achieves a 2.56× speedup using only 18.9% of the KV cache.

## Background & Motivation

1. **Background**: LLM long-context processing faces a memory bottleneck due to the linear growth of KV caches. Mainstream compression directions include token pruning (StreamingLLM, SnapKV), low-rank compression, quantization, KV merging, and budget allocation.
2. **Limitations of Prior Work**: (1) Static strategies (StreamingLLM) retain fixed leading/trailing tokens, causing irreversible information loss; (2) adaptive strategies (SnapKV, H2O) use heuristic ranking that fails to fully capture the query-dependent nature of token importance; (3) learning-based strategies require training token predictors that generalize poorly across datasets.
3. **Key Challenge**: Token importance is inherently query-dependent, yet existing methods either apply static, query-agnostic rules or evaluate importance at a cost equivalent to computing full attention. The key question is whether query-aware importance prediction can be achieved more cheaply.
4. **Goal**: To achieve query-aware token importance prediction at minimal computational cost without any training.
5. **Key Insight**: RoPE decomposes attention computation into independent contributions from $d/2$ 2D frequency components (FCs). Different FCs serve different roles due to their distinct rotation frequencies: high-frequency FCs encode positional patterns, while low-frequency FCs carry semantic information. A small subset of "dominant FCs" suffices to approximately reconstruct the full-head attention pattern.
6. **Core Idea**: Exploit the intrinsic FC-level functional sparsity in RoPE, replacing full-dimensional attention with lightweight computation over a small number of dominant FCs to predict token importance.

## Method

### Overall Architecture
FASA consists of two stages: (1) Token Importance Prediction (TIP)—using a pre-calibrated dominant FC set $\mathcal{I}_{dom}$ to efficiently estimate per-token importance scores; and (2) Focused Attention Computation (FAC)—performing full-dimensional attention exclusively over the selected critical token subset. Dominant FC identification is a one-time offline calibration process.

### Key Designs

1. **Functional Sparsity Discovery in RoPE Frequency Components**:
    - Function: Provides the theoretical and empirical foundation for token importance prediction.
    - Mechanism: In RoPE, a $d$-dimensional vector is divided into $d/2$ 2D frequency components, where the rotation frequency of the $i$-th FC is $\theta_i = B^{-2(i-1)/d}$. A Contextual Agreement (CA) metric is proposed: $\text{CA}_\mathcal{K}^{l,h,i} = |\text{TopK-I}(\alpha_{l,h}, \mathcal{K}) \cap \text{TopK-I}(\alpha_{l,h}^{(i)}, \mathcal{K})| / \mathcal{K}$, which measures the top-K token set overlap between a single-FC attention pattern and the full-head attention. Empirically, dominant FCs are sparse (fewer than 1% of FCs contribute >90% of contextual agreement), universal across tasks (dominant FC overlap across different calibration datasets exceeds 70%), and consistent across models.
    - Design Motivation: FC-level sparsity is an intrinsic property of the RoPE structure rather than a task-specific artifact. High-frequency FCs primarily encode positional patterns (recency bias) rather than semantic information and can therefore be safely ignored.

2. **TIP: Token Importance Predictor**:
    - Function: Predicts the importance ranking of all tokens at minimal computational cost.
    - Mechanism: Offline calibration selects $N_{tip}$ FCs that maximize the sum of expected CA scores: $\mathcal{I}_{dom}^{l,h} = \text{TopK-I}(\{\overline{\text{CA}}_\mathcal{K}^{l,h,i}\}, N_{tip})$. During online inference, scores are aggregated over only the dominant FCs: $S_t^{l,h} = \sum_{i \in \mathcal{I}_{dom}} \alpha^{l,h,i}(q_t, K_{1:t})$, and the top-$N_{fac}$ tokens are selected as $\mathcal{T}_t = \text{TopK-I}(S_t^{l,h}, N_{fac})$. TIP complexity is $O(2tN_{tip})$, far lower than full attention at $O(td)$.
    - Design Motivation: Dominant FCs account for only 1/4 to 1/8 of the full dimensionality yet accurately reconstruct context selection behavior. A single calibration generalizes across tasks.

3. **FAC: Focused Attention Computation**:
    - Function: Executes full-precision attention over the filtered critical tokens.
    - Mechanism: Key and Value entries corresponding to $\mathcal{T}_t$ are retrieved from the full KV cache via a Gather operation: $K_{\mathcal{T}_t} = \text{Gather}(K_{1:t}, \mathcal{T}_t)$, $V_{\mathcal{T}_t} = \text{Gather}(V_{1:t}, \mathcal{T}_t)$. Original absolute positions of retained tokens are preserved to maintain the integrity of RoPE positional encodings. Complexity is $O(N_{fac}d)$. Two variants are provided: FASA-M (memory-optimized, offloads KV cache to CPU) and FASA-C (compute-optimized, retains the full cache on GPU but accesses only sparse Keys).
    - Design Motivation: With critical tokens already identified by TIP, FAC performs full-fidelity attention on the reduced set to ensure generation quality. Position preservation avoids performance degradation caused by positional encoding distortion.

### Loss & Training
FASA is a fully **training-free** framework. Dominant FC identification requires only a small number of calibration samples in a one-time offline process. It is orthogonal to and composable with layer-wise budget allocation methods (e.g., PyramidKV). Theoretical speedup is $\text{Speedup} = d / N_{tip}$ (when $N_{fac} \ll t$).

## Key Experimental Results

### Main Results

| Task / Method | Stream | SnapKV | Quest | FASA | Full KV | Oracle |
|--------|------|------|-------|------|---------|--------|
| LongBench (K=256) | ~80% | ~92% | ~90% | **~99%** | 100% | 100% |
| AIME24 Speedup | — | — | — | **2.56×** | 1× | — |
| KV Cache Usage | — | — | — | **18.9%** | 100% | — |

Cross-model validation: consistently effective on Llama-3.1-8B, Mistral-7B, Qwen2-7B, and others.

### Ablation Study

| FC Count (F) / KV Budget (K) | K=64 | K=256 | K=512 | K=1024 |
|------|------|------|------|------|
| Random FC | 2.0 | 3.6 | 6.4 | 25.5 |
| Stream | 34.4 | 26.8 | 24.4 | 30.7 |
| SnapKV | 37.9 | 40.9 | 41.9 | 49.5 |
| F=8 (1/8) | **43.0** | **49.4** | **54.3** | **62.6** |
| F=16 (1/4) | 55.3 | 59.7 | 62.8 | 70.1 |

### Key Findings
- Only 1/8 of FCs suffice to outperform SnapKV by 10.3% in composite CA score across all budget levels.
- FC functional sparsity is an intrinsic model property: highly consistent across architectures, scales, and tasks.
- FASA-C achieves a 2.56× speedup on the AIME24 long-CoT reasoning task with less than 0.7% performance degradation.
- Dominant FCs constitute fewer than 1% of all FCs yet contribute the majority of contextual information.
- Retaining only 256 tokens on LongBench approaches 100% of full-KV performance.

## Highlights & Insights
- A novel theoretical perspective on RoPE: FC-level functional sparsity reveals an elegant division of labor between high-frequency FCs (positional encoding) and low-frequency FCs (semantic content).
- Completely training-free with one-time calibration for lifelong use: the task-agnostic nature of dominant FCs makes calibration highly efficient.
- Orthogonal to existing methods: seamlessly composable with quantization, layer-wise budget allocation, and other techniques.
- Granularity innovation from token-level to frequency-component-level: finer-grained than page-level (Quest) or token-level (SnapKV) approaches.

## Limitations & Future Work
- Whether the 1/4 dominant FC selection ratio remains optimal under extremely long contexts (100K+) requires further investigation.
- The current implementation targets decoder-only architectures; adaptation is needed for encoder-decoder models and non-RoPE architectures.
- CPU–GPU data transfer latency in FASA-M may become a bottleneck in high-throughput scenarios.
- Integration with speculative decoding remains unexplored.

## Related Work & Insights
- **vs StreamingLLM**: Static retention strategy discards tokens in intermediate positions that may be critical.
- **vs SnapKV**: One-shot filtering at the prefill stage cannot adapt to changes in token importance during generation.
- **vs Quest**: Page-level granularity is too coarse, retrieving entire pages even when only a few tokens are needed.
- **vs SparQ/LoKi**: Low-rank methods require auxiliary memory to store projection matrices.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ FC-level functional sparsity is a novel theoretical discovery about RoPE with broad implications.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across three paradigms: long-context benchmarks, sequence modeling, and long-CoT reasoning.
- Writing Quality: ⭐⭐⭐⭐ A complete logical chain from observation to hypothesis to validation to method.
- Value: ⭐⭐⭐⭐⭐ A training-free, efficient, and general KV cache compression solution with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection](../../CVPR2026/signal_comm/faar_efficient_frequency-aware_multi-task_fine-tuning_via_automatic_rank_selecti.md)
- [\[ICCV 2025\] Rectifying Magnitude Neglect in Linear Attention](../../ICCV2025/signal_comm/rectifying_magnitude_neglect_in_linear_attention.md)
- [\[AAAI 2026\] Task Aware Modulation Using Representation Learning for Upscaling of Terrestrial Carbon Fluxes](../../AAAI2026/signal_comm/task_aware_modulation_using_representation_learning_for_upsaling_of_terrestrial_.md)
- [\[NeurIPS 2025\] ConTextTab: A Semantics-Aware Tabular In-Context Learner](../../NeurIPS2025/signal_comm/contexttab_a_semantics-aware_tabular_in-context_learner.md)
- [\[NeurIPS 2025\] Feature-aware Modulation for Learning from Temporal Tabular Data](../../NeurIPS2025/signal_comm/feature-aware_modulation_for_learning_from_temporal_tabular_data.md)

</div>

<!-- RELATED:END -->
