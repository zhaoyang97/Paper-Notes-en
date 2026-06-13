---
title: >-
  [Paper Note] DSAS: A Universal Plug-and-Play Framework for Attention Optimization in Multi-Document Question Answering
description: >-
  [NeurIPS 2025][LLM Evaluation][Multi-doc QA] This paper proposes Dual-Stage Adaptive Sharpening (DSAS), a training-free plug-and-play attention optimization framework. It employs Contextual Gate Weighting (CGW) to enhanc…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "Multi-doc QA"
  - "attention sharpening"
  - "lost-in-the-middle"
  - "plug-and-play"
  - "information flow"
date: 2026-05-08
content_hash: 6983a2c04cd6d102
---

# DSAS: A Universal Plug-and-Play Framework for Attention Optimization in Multi-Document Question Answering

**Conference**: NeurIPS 2025
**arXiv**: [2510.12251](https://arxiv.org/abs/2510.12251)  
**Code**: None  
**Area**: Video Understanding
**Keywords**: Multi-doc QA, attention sharpening, lost-in-the-middle, plug-and-play, information flow

## TL;DR
This paper proposes Dual-Stage Adaptive Sharpening (DSAS), a training-free plug-and-play attention optimization framework. It employs Contextual Gate Weighting (CGW) to enhance attention from key passages toward the question and target positions, and Reciprocal Attention Suppression (RAS) to suppress information exchange between key and irrelevant passages, achieving an average F1 improvement of 4.2% on multi-document QA benchmarks.

## Background & Motivation

**Background**: The context windows of Transformer-based LLMs have been extended to 128K or even 1M tokens, theoretically enabling multi-document QA. However, naively concatenating multiple documents causes attention dilution, where critical cross-document dependencies are overwhelmed by irrelevant tokens.

**Limitations of Prior Work**: (a) Insufficient long-range dependency modeling — the RULER benchmark reveals that LLMs perform poorly on compositional reasoning tasks; StreamingLLM truncates global interactions, sacrificing cross-document reasoning. (b) "Lost-in-the-middle" — LLMs exhibit degraded ability to process information in the middle of long inputs; methods such as LongAlign require additional training.

**Key Challenge**: A general, training-free solution that requires no architectural modifications is lacking.

**Goal**: Design a training-free attention score adjustment mechanism that automatically identifies key passages, strengthens their information flow, and suppresses interference.

**Key Insight**: The authors first conduct an information flow analysis, identifying two findings: (a) inter-layer information flow exhibits a two-stage pattern, and (b) information flow from key passages is significantly higher in correctly answered instances.

**Core Idea**: Quantify passage importance via layer-wise attention score analysis and dynamically adjust the attention matrix to enhance multi-document QA without any training.

## Method

### Overall Architecture
DSAS inserts two stages into the attention computation during LLM inference. CGW computes per-passage contextual gate weights $w_m$ (combining content relevance and position awareness) and adjusts attention scores at question/target positions. RAS partitions passages into key and irrelevant groups based on $w_m$ and suppresses cross-group attention interactions. Only the attention score matrix is modified; no additional parameters are introduced.

### Key Designs

1. **Information Flow Analysis (Preliminary Finding)**:

    - Defines $\mathcal{I}_{p^m,q}$ and $\mathcal{I}_{p^m,t}$ to measure Top-K aggregated attention values from each passage to the question and to the target position, respectively.
    - Finding: differences are minimal in shallow layers, but in deeper layers the information flow of supporting passages diverges clearly from that of negative passages.

2. **Contextual Gate Weighting (CGW)**:

    - Extracts attention sub-matrices $M$ (concatenating attention from $q$ and $t$ toward each passage) and computes aggregate information flow via column-wise Top-K summation.
    - Z-normalization followed by sigmoid scaling: $v_m = 0.5 \cdot \sigma((\mathcal{I}_{Comb^m} - \mu_I)/\sigma_I) + 0.5$
    - Position-aware weighting: a Gaussian PDF corrects the U-shaped attention bias, applied only to the top-50% passages to avoid over-boosting irrelevant middle passages.
    - Final weight $w_m = v_m \cdot g_m^\alpha$, min-max normalized with a lower bound of $\beta=0.7$.
    - Attention scores are directly scaled, acting only on rows corresponding to question and target positions.

3. **Reciprocal Attention Suppression (RAS)**:

    - Passages are partitioned into $P_{key}$ and $P_{irr}$ using the mean of $w_m$ as the threshold.
    - Cross-group attention suppression: $A^S_{h,l}(i,j) = \min(w_{m_1}, w_{m_2}) \cdot A^S_{h,l}(i,j)$
    - **Design Motivation**: Bidirectional suppression severs the pathway through which irrelevant passages inject noise into key passages.

### Hyperparameter Settings
- $K=10$, $\alpha=1$, $\beta=0.7$; these values are shared across all datasets and models.
- DSAS is applied to the last 50% of layers of all LLMs.

## Key Experimental Results

### Main Results (F1-score %)

| Model | Method | HotpotQA | 2WikiMQA | MuSiQue | L-HotpotQA | L-2WikiMQA | L-MuSiQue | Avg. |
|------|------|----------|----------|---------|------------|------------|-----------|------|
| Llama-3.1-8B | Vanilla | 43.6 | 47.3 | 34.6 | 53.3 | 42.6 | 25.4 | 41.1 |
| | **DSAS** | **47.1** | **50.8** | **39.2** | **56.5** | **47.3** | **32.0** | **45.5 (+4.2)** |
| Qwen2.5-14B | Vanilla | 48.2 | 55.3 | 38.0 | 57.6 | 53.1 | 32.8 | 47.5 |
| | **DSAS** | **51.8** | **58.2** | **43.8** | **60.9** | **56.1** | **39.3** | **51.7 (+4.2)** |
| Qwen2.5-32B | Vanilla | 48.8 | 60.7 | 42.3 | 58.6 | 48.2 | 35.0 | 48.9 |
| | **DSAS** | **50.8** | **62.2** | **45.4** | **59.5** | **50.5** | **39.9** | **51.4 (+2.5)** |

### Ablation Study (Qwen2.5-7B)

| Configuration | HotpotQA | 2WikiMQA | MuSiQue | Avg. | Notes |
|------|----------|----------|---------|------|------|
| DSAS (full) | 46.1 | 49.9 | 35.0 | 45.8 | Full model |
| w/o CGW | 45.5 | 48.2 | 33.8 | 44.3 | −1.5 without CGW |
| w/o RAS | 45.4 | 47.8 | 32.6 | 44.3 | −1.5 without RAS |
| w/o position weighting | 44.7 | 50.5 | 33.2 | — | Position weighting contributes most |

### Key Findings
- Both CGW and RAS contribute independently; removing either causes a performance drop. Position-aware weighting yields the largest individual contribution.
- Medium-scale models (7B–14B) benefit the most; larger models (32B) show smaller margins.
- The largest gains are observed on the complex multi-hop reasoning task MuSiQue.
- Consistent positive improvements are observed across models from 3B to 32B, with hyperparameters shared universally.

## Highlights & Insights
- **Information-flow-driven passage importance estimation**: The method leverages the LLM's own attention to assess passage relevance, which is transferable to document re-ranking in RAG pipelines.
- **Position-aware Gaussian correction**: A Gaussian PDF elegantly corrects the U-shaped attention bias, applied only to the top-50% passages to prevent over-amplification of irrelevant middle passages.
- **Causal severance via bidirectional suppression**: RAS performs blockage from an information-flow completeness perspective, suppressing not only attention from $q$ to irrelevant passages but also the reverse path from irrelevant to key passages.

## Limitations & Future Work
- Passage boundaries must be known in advance, limiting applicability to unstructured text.
- The restriction to the last 50% of layers is fixed; automatic layer selection per model has not been explored.
- Computing information flow metrics per layer may increase inference latency (wall-clock overhead is not reported).
- Potential improvements: (a) adaptive layer selection, and (b) integration with retrieval augmentation to process only the retrieved document subset.
- Applicability to open-domain QA (non-multi-document settings) has not been validated.
- The impact on practical inference throughput requires more detailed benchmarking.

## Related Work & Insights
- **vs. PINE**: PINE mitigates the lost-in-the-middle effect by reordering passage positions based on attention pattern analysis, but yields unstable improvements (negative effects on some models). DSAS dynamically adjusts attention scores directly, achieving consistent positive gains across all seven models.
- **vs. StreamingLLM**: StreamingLLM truncates global dependencies for efficiency, sacrificing cross-document reasoning. DSAS retains the full context and only modulates attention weights without discarding global information.
- **vs. LongAlign**: LongAlign requires fine-tuning with long instruction datasets, limiting generality and adaptability to new models. DSAS is entirely training-free and plug-and-play.
- **vs. Selective Self-Attention**: Selective attention lacks generality and requires task-specific tuning. DSAS uses unified hyperparameters across all models and tasks.
- The information flow analysis methodology proposed in this work can serve as a general tool for interpretability research on LLM reasoning processes.
- The method is compatible with RAG pipelines, where DSAS can enhance attention toward key documents after retrieved documents are concatenated.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The information-flow-driven attention adjustment paradigm is novel, and the two-stage design is well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 7 models, 4 datasets, and comprehensive ablations with broad coverage.
- **Writing Quality**: ⭐⭐⭐⭐ — The information flow analysis is clearly presented, though the heavy use of notation makes some sections dense.
- **Value**: ⭐⭐⭐⭐ — The training-free plug-and-play nature is highly practical for industrial deployment, though applicability is currently limited to multi-document QA scenarios.
<!-- NeurIPS 2025 | video_understanding -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering](../../ACL2026/llm_evaluation/recoqa_a_benchmark_for_tool-augmented_and_multi-step_reasoning_in_real_estate_qu.md)
- [\[NeurIPS 2025\] PaTH Attention: Position Encoding via Accumulating Householder Transformations](path_attention_position_encoding_via_accumulating_householder_transformations.md)
- [\[NeurIPS 2025\] Leveraging Robust Optimization for LLM Alignment under Distribution Shifts](leveraging_robust_optimization_for_llm_alignment_under_distribution_shifts.md)
- [\[NeurIPS 2025\] OptiTree: Hierarchical Thoughts Generation with Tree Search for LLM Optimization Modeling](optitree_hierarchical_thoughts_generation_with_tree_search_for_llm_optimization_.md)
- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](../../ACL2026/llm_evaluation/retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)

</div>

<!-- RELATED:END -->
