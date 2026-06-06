---
title: >-
  [Paper Note] Reliable Evaluation Protocol for Low-Precision Retrieval
description: >-
  [ACL 2026][Information Retrieval & RAG][Low-precision retrieval] This paper reveals that low-precision retrieval systems (e.g., binarized or quantized embeddings) suffer from numerous "spurious ties" during evaluation be…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Low-precision retrieval"
  - "spurious ties"
  - "evaluation protocol"
  - "high-precision scoring"
  - "tie-aware metrics"
date: 2026-05-08
content_hash: 3105a80a682cd84a
---

# Reliable Evaluation Protocol for Low-Precision Retrieval

**Conference**: ACL 2026  
**arXiv**: [2508.03306](https://arxiv.org/abs/2508.03306)  
**Code**: None  
**Area**: Others  
**Keywords**: Low-precision retrieval, spurious ties, evaluation protocol, high-precision scoring, tie-aware metrics

## TL;DR

This paper reveals that low-precision retrieval systems (e.g., binarized or quantized embeddings) suffer from numerous "spurious ties" during evaluation because of reduced score granularity, leading to highly unstable evaluation results. It proposes two complementary strategies, HPS (High-Precision Scoring) and TRM (Tie-aware Retrieval Metrics), to ensure more reliable and consistent evaluation for low-precision retrieval.

## Background & Motivation

**Background**: Reducing the numerical precision of model parameters and computations (e.g., FP16, INT8, binarization) is a mainstream method for enhancing retrieval system efficiency. Low-precision representations significantly reduce storage requirements and accelerate similarity calculations, which is critical in large-scale retrieval scenarios.

**Limitations of Prior Work**: When calculating relevance scores between queries and documents using low-precision values, the coarse numerical granularity causes many documents that were originally distinct to receive identical scores, creating "spurious ties." For example, in binarized embeddings, Hamming distance has a limited set of discrete values, leading to many documents sharing the same distance. The ranking of these tied documents depends on arbitrary tie-breaking rules (like document ID order), resulting in highly random fluctuations in evaluation metrics such as nDCG and MRR.

**Key Challenge**: While the efficiency gains of low-precision retrieval are real, its retrieval quality cannot be reliably evaluated—the same model can yield vastly different scores under different tie-breaking strategies. This makes model comparisons and the identification of improvement directions unreliable.

**Goal**: To design an evaluation protocol that provides stable, reproducible, and meaningful evaluation results under the constraints of low-precision retrieval.

**Key Insight**: The root cause is that "low scoring precision leads to ties." The solution naturally follows two paths: (1) increasing precision during the scoring phase to eliminate ties; (2) making metrics tie-aware to report the associated uncertainty during the evaluation phase.

**Core Idea**: Upscale the final scoring step to high precision (HPS) to eliminate ties at a low computational cost, while simultaneously designing tie-aware retrieval metrics (TRM) to report expected values and uncertainty ranges.

## Method

### Overall Architecture

The evaluation protocol consists of two independent but complementary components. The inputs are candidate document lists and scores returned by a low-precision retrieval system, and the outputs are stable and reliable evaluation metrics. The process involves using HPS to rescore tied candidates for deterministic ranking, followed by using TRM to calculate expected metric values and confidence ranges that account for tie-related uncertainty.

### Key Designs

1.  **High-Precision Scoring (HPS)**:
    - **Function**: Increases the precision of score calculations for tied candidates during the final retrieval stage to eliminate spurious ties.
    - **Mechanism**: The retrieval process is still performed in low precision to maintain efficiency. However, for the final top-K candidate documents, their embeddings are cast to a higher precision (e.g., FP32) to recalculate similarity scores. Since this only involves a small number of candidates, the computational overhead is minimal (typically <1% additional computation) while completely eliminating spurious ties within the top-K.
    - **Design Motivation**: Ties primarily affect top-ranked documents because evaluation metrics are most sensitive to top positions, and the number of top-K documents is limited. Increasing precision for only this small step provides significant benefits for minimal investment.

2.  **Tie-aware Retrieval Metrics (TRM)**:
    - **Function**: Reports the expected value, range, and bias of metrics when ties are present, quantifying ranking uncertainty.
    - **Mechanism**: For groups of tied documents, the system considers all possible ranking permutations. It reports the expected value $E[M]$, the maximum value $M_{max}$, the minimum value $M_{min}$, and the bias $M_{bias} = E[M] - M_{default}$. These are calculated efficiently via analytical formulas rather than explicit enumeration of all permutations.
    - **Design Motivation**: Even with HPS, some extreme low-precision scenarios may still harbor ties. TRM provides a way to "report honestly"—acknowledging evaluation uncertainty rather than pretending ties do not exist.

3.  **Unified Processing of Three Scoring Functions**:
    - **Function**: Ensures the protocol is applicable to various similarity calculation methods.
    - **Mechanism**: The paper analyzes the tie generation mechanisms and frequencies for three common low-precision scoring functions—dot product, cosine similarity, and Hamming distance—and verifies that HPS and TRM are effective for each.
    - **Design Motivation**: Different quantization schemes rely on different scoring functions; the protocol must be sufficiently general for wide adoption.

## Key Experimental Results

### Main Results

| Scoring Function | Precision | Tie Rate (w/o HPS) | Tie Rate (w/ HPS) | nDCG@10 CV |
| :--- | :--- | :--- | :--- | :--- |
| Dot Product | INT8 | Medium | ~0% | Substantially Reduced |
| Cosine Similarity | BF16 | Low | ~0% | Reduced to Negligible |
| Hamming Distance | 1-bit | Very High (>50%) | Significantly Reduced | Substantially Reduced |
| Dot Product | 4-bit | High | ~0% | Eliminated Fluctuations |

### Ablation Study

| Configuration | Metric Stability | Description |
| :--- | :--- | :--- |
| Original Low-Precision Eval. | High Variation | Large metric differences across different random seeds |
| HPS Only | High Stability | Deterministic ranking after eliminating ties |
| TRM Only | Medium | Reports uncertainty ranges but doesn't eliminate the root cause |
| HPS + TRM | Optimal | Eliminates most ties + honestly reports residual uncertainty |

### Key Findings

- The tie problem is most severe for Hamming distance (1-bit embeddings)—over 50% of the top-100 candidates may be tied with others, causing fluctuations in nDCG@10 of up to 15%+.
- HPS has minimal computational overhead but is highly effective: rescoring only the top-1000 candidates eliminates almost all ties.
- TRM reveals an important discovery: default tie-breaking strategies (such as sorting by document ID) often introduce systematic bias, causing reported metrics to be artificially high or low.
- These conclusions were consistently validated across multiple models on two retrieval datasets (MS MARCO, BEIR).

## Highlights & Insights

- **Simple yet previously overlooked problem**: Papers on low-precision retrieval often ignore the impact of ties on evaluation, yet this issue can render experimental conclusions completely unreliable. The core contribution is bringing this to the community's attention.
- **High cost-benefit ratio of HPS**: A near-zero-cost change solves a serious problem. This "minimal intervention" design philosophy is highly effective.
- **The "Honest Reporting" concept of TRM** can be transferred to other evaluation scenarios involving uncertainty, such as evaluation uncertainty caused by position bias in recommendation systems or metric fluctuations due to random sampling in generation tasks.

## Limitations & Future Work

- The paper focuses on the retrieval evaluation phase, but similar tie issues might exist during the training phase of Learning to Rank (LTR), which was not explored.
- HPS requires keeping the original high-precision embeddings or the ability to re-calculate them; if original embeddings are unavailable, it cannot be used.
- The analytical calculation of TRM may become computationally complex under conditions of extreme ties (e.g., hundreds of documents with the same score).
- Future research could investigate how different quantization schemes affect ties to guide the design of better quantization strategies.

## Related Work & Insights

- **vs. Standard Retrieval Evaluation**: Standard evaluations (e.g., TREC eval) assume scores are continuous and do not account for ties; the protocol in this paper is a necessary supplement to these standards.
- **vs. Embedding Quantization Research**: Existing quantization research focuses on the precision-efficiency trade-off but neglects the reliability of the evaluation itself; this paper serves as a reminder that quantization research requires more rigorous evaluation.
- **vs. Tie Handling in LTR**: Some work in the LTR field has discussed tie issues, but they have not specialized in the systematic ties found in low-precision scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ The problem definition is clear and novel; though the technical approach is relatively simple, the insights are valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive experimental design across multiple scoring functions, precisions, and datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ The problem statement is exceptionally clear, and the solution is concise and elegant.
- Value: ⭐⭐⭐⭐ Provides essential evaluation infrastructure for the low-precision retrieval community with practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] The Dilemma of Low-Resource Languages in Multilingual Retrieval: Evidence from Amharic](the_multilingual_curse_at_the_retrieval_layer_evidence_from_amharic.md)
- [\[ACL 2026\] AuthorityBench: Benchmarking LLM Authority Perception for Reliable Retrieval-Augmented Generation](authoritybench_benchmarking_llm_authority_perception_for_reliable_retrieval-augm.md)
- [\[ACL 2026\] RARE: Redundancy-Aware Retrieval Evaluation Framework for High-Similarity Corpora](rare_redundancy-aware_retrieval_evaluation_framework_for_high-similarity_corpora.md)
- [\[ACL 2026\] FLARE: Task-Agnostic Embedding Model Evaluation via Normalizing Flows](flare_task-agnostic_embedding_model_evaluation_through_a_normalization_process.md)
- [\[NeurIPS 2025\] Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations](../../NeurIPS2025/information_retrieval/retrieval-augmented_generation_for_reliable_interpretation_of_radio_regulations.md)

</div>

<!-- RELATED:END -->
