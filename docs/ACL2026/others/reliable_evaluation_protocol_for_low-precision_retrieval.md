---
title: >-
  [Paper Note] Reliable Evaluation Protocol for Low-Precision Retrieval
description: >-
  [ACL 2026][low-precision retrieval] This paper identifies that low-precision retrieval systems (e.g., binarized or quantized embeddings) suffer from a large number of spurious ties due to reduced score granularity, leading to highly unstable evaluation results. Two complementary strategies are proposed—High-Precision Scoring (HPS) and Tie-aware Retrieval Metrics (TRM)—to enable more reliable and consistent evaluation of low-precision retrieval systems.
tags:
  - ACL 2026
  - low-precision retrieval
  - spurious ties
  - evaluation protocol
  - high-precision scoring
  - tie-aware metrics
date: 2026-05-08
content_hash: f7d8364e9542aef5
---

# Reliable Evaluation Protocol for Low-Precision Retrieval

**Conference**: ACL 2026
**arXiv**: [2508.03306](https://arxiv.org/abs/2508.03306)
**Code**: None
**Area**: Other
**Keywords**: low-precision retrieval, spurious ties, evaluation protocol, high-precision scoring, tie-aware metrics

## TL;DR

This paper identifies that low-precision retrieval systems (e.g., binarized or quantized embeddings) suffer from a large number of spurious ties due to reduced score granularity, leading to highly unstable evaluation results. Two complementary strategies are proposed—High-Precision Scoring (HPS) and Tie-aware Retrieval Metrics (TRM)—to enable more reliable and consistent evaluation of low-precision retrieval systems.

## Background & Motivation

**Background**: Reducing the numerical precision of model parameters and computations (e.g., FP16, INT8, binarization) is a mainstream approach for improving retrieval system efficiency. Low-precision representations can significantly reduce storage requirements and accelerate similarity computation, making them critical for large-scale retrieval scenarios.

**Limitations of Prior Work**: When computing query-document relevance scores using low-precision arithmetic, the coarser numerical granularity causes many originally distinct documents to receive identical scores, producing "spurious ties." For example, in binarized embeddings, Hamming distances take only a limited set of discrete values, causing many documents to share the same distance. The ordering of tied documents depends on arbitrary tie-breaking rules (e.g., document ID order), leading to large random fluctuations in evaluation metrics such as nDCG and MRR.

**Key Challenge**: The efficiency gains of low-precision retrieval are genuine, but reliably evaluating retrieval quality under such conditions is infeasible—the same model can yield substantially different evaluation scores under different tie-breaking strategies, making model comparison and directional improvement judgments unreliable.

**Goal**: Design an evaluation protocol that produces stable, reproducible, and meaningful results under the constraints of low-precision retrieval.

**Key Insight**: The root cause of the problem is "low scoring precision leading to ties," which naturally suggests two remedies: (1) increase scoring precision to eliminate ties; (2) make the metric computation tie-aware and report uncertainty.

**Core Idea**: Elevate the final scoring step to high precision (HPS) to eliminate ties at minimal computational cost, while designing tie-aware retrieval metrics (TRM) that report expected values and uncertainty ranges.

## Method

### Overall Architecture

The evaluation protocol consists of two independent but complementary components. The input is a ranked candidate list with scores returned by a low-precision retrieval system; the output is stable and reliable evaluation metrics. The pipeline first applies HPS to re-score tied candidates for deterministic ranking, then applies TRM to compute expected metric values and confidence intervals that account for tie-induced uncertainty.

### Key Designs

1. **High-Precision Scoring (HPS)**:

    - Function: At the final stage of retrieval, re-compute scores for tied candidates at higher precision to eliminate spurious ties.
    - Mechanism: The retrieval process still operates in low precision (preserving efficiency), but for the final top-$K$ candidates, their embeddings are upcast to higher precision (e.g., FP32) and similarity scores are recomputed. Since only a small number of candidates are processed, the additional computational overhead is negligible (typically $<1\%$), yet spurious ties within the top-$K$ are completely eliminated.
    - Design Motivation: Ties disproportionately affect top-ranked documents (since evaluation metrics are more sensitive to top positions), and the number of top-$K$ documents is small. Increasing precision only at this final step incurs minimal cost while yielding substantial gains.

2. **Tie-aware Retrieval Metrics (TRM)**:

    - Function: In the presence of ties, report the expected metric value, range, and bias to quantify ranking uncertainty.
    - Mechanism: For each group of tied documents, all possible orderings are enumerated and the metric value is computed for each; the expected value $E[M]$, maximum $M_{max}$, minimum $M_{min}$, and bias $M_{bias} = E[M] - M_{default}$ are reported. In practice, closed-form analytical expressions are used for efficient computation without exhaustive enumeration.
    - Design Motivation: Even with HPS, residual ties may persist in extreme low-precision scenarios. TRM provides an "honest reporting" mechanism—rather than ignoring ties, it explicitly communicates the uncertainty range of the evaluation.

3. **Unified Treatment of Three Scoring Functions**:

    - Function: Ensure the protocol generalizes across different similarity computation methods.
    - Mechanism: The paper analyzes the mechanisms and frequency of tie generation for three common low-precision scoring functions—dot product, cosine similarity, and Hamming distance—and verifies that both HPS and TRM are effective under each.
    - Design Motivation: Different low-precision quantization schemes correspond to different scoring functions; the protocol must be sufficiently general to achieve broad adoption.

## Key Experimental Results

### Main Results

| Scoring Function | Precision | Tie Rate (w/o HPS) | Tie Rate (w/ HPS) | nDCG@10 Coefficient of Variation |
|---|---|---|---|---|
| Dot product | INT8 | Moderate | ~0% | Substantially reduced |
| Cosine similarity | BF16 | Low | ~0% | Reduced to negligible |
| Hamming distance | 1-bit | Very high (>50%) | Significantly reduced | Substantially reduced |
| Dot product | 4-bit | High | ~0% | Fluctuation eliminated |

### Ablation Study

| Configuration | Metric Stability | Notes |
|---|---|---|
| Raw low-precision evaluation | High variance | Large metric differences across random seeds |
| HPS only | High stability | Deterministic ranking after tie elimination |
| TRM only | Moderate | Reports range but does not address root cause |
| HPS + TRM | Optimal | Eliminates most ties + honestly reports residual uncertainty |

### Key Findings

- Hamming distance (1-bit embeddings) exhibits the most severe tie problem—over 50% of top-100 candidates may be tied with others, causing nDCG@10 fluctuations exceeding 15%.
- HPS incurs negligible computational overhead yet yields significant improvements: re-scoring only the top-1000 candidates eliminates nearly all spurious ties.
- TRM reveals an important finding: the default tie-breaking strategy (ordering by document ID) typically introduces systematic bias, causing reported metrics to be consistently over- or under-estimated.
- These conclusions are consistently validated across multiple models on two retrieval benchmarks (MS MARCO and BEIR).

## Highlights & Insights

- **A simple problem that has been widely overlooked**: Papers on low-precision retrieval typically do not discuss the impact of ties on evaluation, yet this issue can render experimental conclusions entirely unreliable. The central contribution of this work is raising community awareness of the problem.
- **Exceptional cost-effectiveness of HPS**: A near-zero-cost modification resolves a serious problem; this "minimal intervention" design philosophy is instructive.
- **The "honest reporting" philosophy of TRM** generalizes to other evaluation scenarios with inherent uncertainty—such as position-bias-induced evaluation uncertainty in recommendation systems, or metric variance from stochastic sampling in generation tasks.

## Limitations & Future Work

- The paper focuses on ties in retrieval evaluation; analogous issues may arise during the training phase of learning-to-rank models but are not explored.
- HPS requires access to the original high-precision embeddings or the ability to recompute them; it is inapplicable when original embeddings are unavailable.
- The analytical computation in TRM may become expensive in extreme tie scenarios (e.g., hundreds of documents sharing the same score).
- Future work could investigate how different quantization schemes affect tie frequency, informing the design of better quantization strategies.

## Related Work & Insights

- **vs. Standard retrieval evaluation**: Standard evaluation frameworks (e.g., TREC eval) assume continuous scores and do not account for ties; the proposed protocol is a necessary complement to these standards.
- **vs. Embedding quantization research**: Existing quantization work focuses on the precision-efficiency trade-off but overlooks the reliability of the evaluation itself; this paper calls for more careful evaluation practices in quantization research.
- **vs. Tie handling in learning-to-rank (LTR)**: A small body of LTR work has discussed ties, but no systematic solution has been proposed specifically for the pervasive tie problem in low-precision retrieval scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ The problem is clearly and originally defined; while the technical solutions are relatively straightforward, the insights are valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic experiments spanning multiple scoring functions, precision levels, and datasets are well-designed.
- Writing Quality: ⭐⭐⭐⭐⭐ The problem is articulated very clearly and the proposed solutions are concise and elegant.
- Value: ⭐⭐⭐⭐ Provides important evaluation infrastructure for the low-precision retrieval community with tangible practical impact.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Consistent Low-Rank Approximation](../../ICLR2026/others/consistent_low-rank_approximation.md)
- [\[AAAI 2026\] Whispering Agents: An Event-Driven Covert Communication Protocol for the Internet of Agents](../../AAAI2026/others/whispering_agents_an_event-driven_covert_communication_protocol_for_the_internet.md)
- [\[ICLR 2026\] Exchangeability of GNN Representations with Applications to Graph Retrieval](../../ICLR2026/others/exchangeability_gnn_representations.md)
- [\[AAAI 2026\] ASAG: Toward the Frontiers of Reliable Diffusion Sampling via Adversarial Sinkhorn Attention Guidance](../../AAAI2026/others/toward_the_frontiers_of_reliable_diffusion_sampling_via_adversarial_sinkhorn_att.md)
- [\[AAAI 2026\] Forest vs Tree: The (N, K) Trade-off in Reproducible ML Evaluation](../../AAAI2026/others/forest_vs_tree_the_n_k_trade-off_in_reproducible_ml_evaluation.md)

<!-- RELATED:END -->
