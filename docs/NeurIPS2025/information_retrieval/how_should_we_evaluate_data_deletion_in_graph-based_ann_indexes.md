---
title: >-
  [Paper Note] How Should We Evaluate Data Deletion in Graph-Based ANN Indexes?
description: >-
  [NeurIPS 2025 (Workshop on ML for Systems)][Information Retrieval & RAG][ANNS] To address the lack of a unified evaluation methodology for data deletion in graph-based ANN indexes…
tags:
  - "NeurIPS 2025 (Workshop on ML for Systems)"
  - "Information Retrieval & RAG"
  - "ANNS"
  - "HNSW"
  - "data deletion"
  - "vector database"
  - "evaluation framework"
date: 2026-05-08
content_hash: 119f68c56e971caa
---

# How Should We Evaluate Data Deletion in Graph-Based ANN Indexes?

**Conference**: NeurIPS 2025 (Workshop on ML for Systems)  
**arXiv**: [2512.06200](https://arxiv.org/abs/2512.06200)  
**Code**: None  
**Area**: Information Retrieval / Approximate Nearest Neighbor Search  
**Keywords**: ANNS, HNSW, data deletion, vector database, evaluation framework

## TL;DR

To address the lack of a unified evaluation methodology for data deletion in graph-based ANN indexes, this paper formally defines three baseline approaches—lazy deletion, eager deletion, and reconstruction—proposes a deployment-oriented evaluation framework and metric suite, and introduces the Deletion Control algorithm, which dynamically switches deletion strategies under accuracy constraints based on empirical analysis.

## Background & Motivation

**Background**: Approximate nearest neighbor search (ANNS) is a core building block for applications such as RAG and recommendation systems. In these settings, data changes frequently—new items are added, expired items are removed, and documents are updated—requiring ANNS algorithms to support efficient deletion. Graph-based methods such as HNSW are among the most widely adopted ANNS algorithms.

**Limitations of Prior Work**: Existing research on data deletion in ANNS suffers from three key issues. First, experimental setups are unrealistic—studies such as FreshDiskANN and MN-RU reinsert previously deleted data, a scenario that rarely occurs in practice. Second, evaluation metrics are insufficiently comprehensive—most work focuses solely on post-deletion search accuracy, while practical deployment also requires consideration of deletion latency, insertion throughput, and memory footprint. Third, no unified and formal comparison framework exists for different deletion approaches.

**Key Challenge**: There is a fundamental trade-off among deletion speed, post-deletion search accuracy, and memory efficiency. Lazy deletion is fastest but leads to accuracy degradation; eager deletion is intermediate but disrupts graph connectivity; reconstruction restores accuracy but incurs prohibitive cost. Theoretical guidance for choosing the appropriate strategy in practical deployments is lacking.

**Goal**: (1) Establish a unified formal framework for data deletion in graph-based ANNS; (2) Design evaluation protocols and a metric suite oriented toward real-world use cases; (3) Propose an algorithm that adaptively selects deletion strategies based on accuracy requirements.

**Key Insight**: The three deletion approaches are formalized using a unified mathematical language, compared fairly under identical experimental conditions, and empirical regularities observed from experiments—specifically, that eager deletion accuracy converges to a stable value while lazy deletion accuracy degrades linearly—are exploited to design a dynamic switching algorithm.

**Core Idea**: Formal definitions and standardized evaluation reveal the characteristics of each deletion approach, which are then leveraged to design an accuracy-aware adaptive deletion control strategy.

## Method

### Overall Architecture

The work consists of three components: (1) formalizing the three deletion approaches with pseudocode and mathematical definitions; (2) designing a multi-dimensional evaluation framework; (3) proposing the Deletion Control algorithm based on experimental findings. A graph-based ANN index is represented by a node set $\mathcal{P}$ and neighbor set $\mathcal{N}$, and the deletion operation is defined as $DELETE: (\mathcal{D}, \mathcal{P}, \mathcal{N}) \mapsto (\mathcal{P}', \mathcal{N}')$.

### Key Designs

1. **Formalization of Three Deletion Approaches**:

    - Function: Establish a unified formal framework for data deletion in graph-based ANNS.
    - Mechanism: (a) Lazy deletion—graph structure is not modified; only a deletion marker set $\mathcal{F}$ is maintained, and marked nodes are excluded from search results $\mathcal{R} \leftarrow SEARCH(\mathbf{q}, \mathcal{P}, \mathcal{N}) \setminus \mathcal{F}$; (b) Eager deletion—nodes are physically removed from the graph, all incident edges are deleted, neighbor lists of adjacent nodes are updated $\mathcal{N}_i \leftarrow \mathcal{N}_i \setminus \mathcal{D}$, and memory is released; (c) Reconstruction—the entire graph is rebuilt from the remaining data after removing deleted entries $\mathcal{N} \leftarrow CONSTRUCT(\mathcal{P})$.
    - Design Motivation: Prior work uses different implementations and definitions, precluding meaningful comparison. Formalization makes the distinctions between approaches immediately apparent—lazy deletion preserves graph structure but wastes memory; eager deletion reclaims memory but disrupts connectivity; reconstruction restores an optimal graph but at high cost.

2. **Deployment-Oriented Evaluation Metric Suite**:

    - Function: Comprehensively assess the impact of deletion operations on ANNS systems.
    - Mechanism: Four-dimensional metrics are used—(a) search accuracy: 1-Recall@10; (b) search throughput: QPS-search; (c) deletion throughput: QPS-delete; (d) insertion throughput: QPS-add. QPS-Recall curves jointly capture the speed-accuracy trade-off. The experimental protocol consists of repeatedly executing insertions and deletions with the same batch size, recomputing ground truth after each round.
    - Design Motivation: Existing evaluations consider only post-deletion search accuracy, neglecting the efficiency of deletion itself and its effect on subsequent operations.

3. **Deletion Control Algorithm**:

    - Function: Automatically select the optimal deletion strategy under accuracy constraints.
    - Mechanism: Two key parameters are introduced—$\theta$ (the lower bound to which accuracy converges after multiple rounds of eager deletion) and $\pi$ (the maximum number of steps before lazy deletion reduces accuracy below target $\alpha$). Estimation: $\theta$ is taken as the minimum accuracy observed in eager deletion experiments; $\pi$ exploits the linear degradation of lazy deletion accuracy, estimated as $\pi \approx (\alpha - R_0) / \Delta$. Strategy selection depends on the relationship between $\alpha$ and $\theta$: if $\alpha < \theta$, only eager deletion is used (accuracy will not fall below the requirement); if $\alpha \geq \theta$, $\pi$ steps of lazy deletion alternate with one reconstruction (periodically restoring accuracy).
    - Design Motivation: Pure eager deletion has an accuracy floor; pure lazy deletion causes continuous accuracy degradation; pure reconstruction is too slow. The hybrid strategy exploits the speed advantage of lazy deletion and the accuracy recovery of reconstruction.

## Key Experimental Results

### Main Results

Comparison of search performance of the three deletion approaches on SIFT1M (after step 5):

| Deletion Method | Search Accuracy (1-Recall@10) | QPS-delete (b=10⁵) | Memory Efficiency |
|----------------|-------------------------------|---------------------|-------------------|
| Reconstruction | Highest | Lowest | Medium |
| Eager Deletion | Medium | Medium | Best |
| Lazy Deletion | Lowest | **Highest** | Worst |

Deletion throughput on SIFT1B under different batch sizes:

| Batch Size | Lazy Deletion QPS | Eager Deletion QPS | Reconstruction QPS |
|-----------|-------------------|--------------------|--------------------|
| b=10⁵ | Fastest | Medium | Slowest |
| b=10³ | Fastest | Medium | Relatively slower than at b=10⁵ (relative to eager deletion) |

### Ablation Study

Deletion Control validation (SIFT1B, b=10⁵, α=0.84):

| Strategy | Accuracy Maintenance | Total Deletion Time |
|----------|---------------------|---------------------|
| Eager deletion only | θ=0.816 < α=0.84, not satisfied | — |
| Lazy deletion only | Rapid degradation, not satisfied | — |
| Deletion Control (lazy + reconstruction alternating) | ≈ satisfies α=0.84 requirement | **Minimum** among all strategies satisfying the accuracy constraint |

### Key Findings

- Accuracy degradation under eager deletion is not unbounded—after multiple rounds of deletion and insertion, it converges to a stable value $\theta$, which is an important empirical finding.
- Accuracy degradation under lazy deletion is approximately linear, and can be accurately predicted using $\Delta = (R_S - R_0)/S$.
- At high-frequency small-batch deletion (b=10³), the relative overhead of reconstruction is larger, making eager deletion more advantageous.
- Deletion Control requires only 10% of queries as a calibration set to accurately estimate $\theta$ and $\pi$.

## Highlights & Insights

- **Practical Value of the Evaluation Framework**: This work establishes the first unified and realistic evaluation methodology for deletion in graph-based ANNS. The experimental setups of prior work (reinserting deleted data) obscure the problems that arise in real scenarios; the iterative insert-delete protocol proposed here more closely reflects practical conditions.
- **The Convergence of Eager Deletion Accuracy Is a Noteworthy Finding**: Intuitively, eager deletion continuously degrades graph connectivity and should cause monotonically decreasing accuracy; however, experiments reveal that accuracy converges to a stable value. This implies that in scenarios with moderate accuracy requirements, eager deletion is an efficient and memory-friendly choice.
- **Deletion Control Is Elegantly Simple and Effective**: Only two estimable parameters ($\theta$, $\pi$) are required to adaptively select strategies, making the algorithm straightforward to deploy in production systems.

## Limitations & Future Work

- As a Workshop paper, the experimental scope is relatively limited—validation is conducted primarily on HNSW without testing other graph-based indexes such as DiskANN and NSG.
- Deletion Control assumes linear accuracy degradation under lazy deletion, which may not hold under extreme data distributions.
- More sophisticated deletion methods (e.g., edge rewiring in FreshDiskANN, neighbor recomputation in IPGM) are not evaluated; only the three basic approaches are compared.
- Quantitative analysis of memory consumption is insufficiently detailed.
- Performance differences related to hardware characteristics (e.g., SSD vs. RAM) are not explored.
- The performance impact of concurrent deletion and querying is not discussed.

## Related Work & Insights

- **vs. FreshDiskANN**: FreshDiskANN rewires neighbor edges during eager deletion to preserve search quality, but its evaluation protocol of reinserting deleted data is unrealistic. The evaluation framework proposed in this paper is better suited for measuring the true effectiveness of such advanced methods.
- **vs. MN-RU (HNSW-update)**: MN-RU addresses the unreachable node problem by maintaining a backup graph at the cost of additional memory. The finding that baseline eager deletion accuracy converges to a stable value raises the question: under what accuracy requirements does the additional overhead of MN-RU become worthwhile?
- **vs. Ada-IVF/SPFresh**: These IVF-based methods handle updates by reassigning vectors to clusters, which is fundamentally different from the deletion mechanisms of graph-based methods; nevertheless, the conceptual approach to evaluation design is transferable.

## Rating

- Novelty: ⭐⭐⭐ The formalization and evaluation framework are valuable, but technical innovation is limited, and the Deletion Control algorithm is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐ Multi-dataset validation, but testing is confined to HNSW; scope is constrained by Workshop format.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear; formalization is rigorous and concise.
- Value: ⭐⭐⭐⭐ Fills a gap in ANNS deletion evaluation and provides direct practical guidance for vector database engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] How Retrieved Context Shapes Internal Representations in RAG](../../ACL2026/information_retrieval/how_retrieved_context_shapes_internal_representations_in_rag.md)
- [\[NeurIPS 2025\] MuRating: A High Quality Data Selecting Approach to Multilingual Large Language Model Pretraining](murating_a_high_quality_data_selecting_approach_to_multilingual_large_language_m.md)
- [\[ICLR 2026\] G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge](../../ICLR2026/information_retrieval/g-reasoner_foundation_models_for_unified_reasoning_over_graph-structured_knowled.md)
- [\[AAAI 2026\] N2N-GQA: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using LLMs](../../AAAI2026/information_retrieval/n2n-gqa_noise-to-narrative_for_graph-based_table-text_question_answering_using_l.md)
- [\[ACL 2026\] Domain-Specific Data Generation Framework for RAG Adaptation](../../ACL2026/information_retrieval/domain-specific_data_generation_framework_for_rag_adaptation.md)

</div>

<!-- RELATED:END -->
