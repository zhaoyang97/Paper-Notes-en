---
title: >-
  [Paper Note] Is Complex Query Answering Really Complex?
description: >-
  [ICML 2025][Graph Learning][Complex Query Answering] This paper reveals that up to 98% of "complex" queries in existing benchmarks for complex query answering (CQA) over knowledge graphs can actually be simplified to simple single-link prediction problems, leading to a severe overestimation of research progress. The authors propose new benchmarks with balanced sampling (FB15k237+H, NELL995+H, ICEWS18+H) and introduce a hybrid solver, CQD-Hybrid…
tags:
  - "ICML 2025"
  - "Graph Learning"
  - "Complex Query Answering"
  - "Knowledge Graphs"
  - "Multi-hop Reasoning"
  - "Benchmark Evaluation"
  - "Data Leakage"
date: 2026-05-08
content_hash: f7bdb6a3b0ece898
---

# Is Complex Query Answering Really Complex?

**Conference**: ICML 2025  
**arXiv**: [2410.12537](https://arxiv.org/abs/2410.12537)  
**Code**: Not mentioned  
**Area**: Graph Learning / Knowledge Graph Reasoning  
**Keywords**: Complex Query Answering, Knowledge Graphs, Multi-hop Reasoning, Benchmark Evaluation, Data Leakage

## TL;DR

This paper reveals that up to 98% of "complex" queries in existing benchmarks for complex query answering (CQA) over knowledge graphs can actually be simplified to simple single-link prediction problems, leading to a severe overestimation of research progress. The authors propose new benchmarks with balanced sampling (FB15k237+H, NELL995+H, ICEWS18+H) and introduce a hybrid solver, CQD-Hybrid, to validate this finding. On these new benchmarks, the MRR of all SOTA methods drops significantly (by up to more than 30 percentage points).

## Background & Motivation

**Background**: Complex Query Answering (CQA) is a core reasoning task on Knowledge Graphs (KGs) that requires models to retrieve answers over incomplete KGs based on logical queries (including operations like conjunction, disjunction, and existential quantification). In recent years, numerous neural methods (such as BetaE, ConE, GNN-QE, CQD, QTO, etc.) have been proposed, achieving impressive performance gains on standard benchmarks like FB15k237 and NELL995.

**Limitations of Prior Work**: However, the construction of these benchmarks hides a critical issue: the training set already contains a large portion of the edges (links) in the reasoning trees of the test queries. This means a complex query labeled as a "2i1p" pattern may actually only require predicting a single missing link, while all other links have already appeared in the training data.

**Key Challenge**: Existing benchmarks treat all query-answer (QA) pairs of the same query type as having equal difficulty, whereas there actually exists a difficulty spectrum ranging from "only 1 link needs to be predicted" (partial-inference, the simplest) to "all links are missing" (full-inference, the hardest). Since the vast majority of QA pairs in these benchmarks belong to the simplest partial-inference category, the MRR metrics are severely inflated, distorting the research community's perception of actual progress in the CQA field.

**Goal**
   - Quantify the proportion of "easy" QA pairs in existing benchmarks
   - Reveal the actual performance of SOTA methods on genuinely difficult full-inference queries
   - Construct new benchmarks that are fairer and more challenging

**Key Insight**: The authors adopt the perspective of a "reasoning tree" to analyze how many links in the reasoning tree of each QA pair originate from the training set, thereby decoupling the "formal complexity" of a query from its "actual difficulty."

**Core Idea**: By analyzing the extent of training link leakage in the reasoning trees, this work exposes the false complexity of existing CQA benchmarks and constructs new benchmarks with balanced difficulty to realistically measure the multi-hop reasoning capabilities of models.

## Method

### Overall Architecture

This paper does not propose a new CQA model, but rather presents a systematic analysis and reconstruction of benchmarks. The overall workflow consists of three phases:
1. **Diagnostics Phase**: Define the reasoning tree and a hardness taxonomy of QA pairs (trivial / partial-inference / full-inference) to quantify the distribution of various QA pairs in existing benchmarks.
2. **Validation Phase**: Propose the CQD-Hybrid solver to validate the hypothesis that "simply memorizing training links" can explain SOTA performance; conduct a stratified performance evaluation on multiple SOTA methods.
3. **Reconstruction Phase**: Construct three new benchmarks (FB15k237+H, NELL995+H, ICEWS18+H) to ensure uniform sampling of QA pairs across different difficulty levels, and introduce two more complex query types: 4p and 4i.

### Key Designs

1. **Reasoning Tree & Hardness Taxonomy**

    - **Function**: Define a reasoning tree for each QA pair $(q, t)$—a directed acyclic graph from anchor entities to target entities, with a structure matching the query graph.
    - **Mechanism**: Check if each edge in the reasoning tree belongs to $G_{\text{train}}$. If all edges are in the training set, it is categorized as trivial (filtered out); if some edges are in the training set, it is partial-inference (which can be simplified to simpler query types); if all edges are in the test set, it is full-inference (genuinely requiring multi-hop reasoning). When multiple reasoning paths exist, the path with the fewest missing links and the fewest hops is selected.
    - **Design Motivation**: Existing evaluations treat all QA pairs of the same query type as having equal difficulty. However, a 2i1p query might actually require predicting only 1 link (equivalent to a 1p link prediction), which does not require multi-hop reasoning capabilities at all. This taxonomy allows for the precise quantification of "false difficulty" within benchmarks.

2. **CQD-Hybrid Solver**

    - **Function**: Assign the maximum score to known links in the training set based on CQD, rather than relying on the output of the neural link predictor.
    - **Mechanism**: CQD decomposes a complex query into multiple 1p sub-queries, with each 1p query scored by a neural link predictor, and combines the scores using fuzzy logic operators. The only modification in CQD-Hybrid is that if an edge $(s, p, o) \in G_{\text{train}}$, it is directly assigned the maximum score instead of using the predictor's output. Consequently, the greedy search process prioritizes reasoning along known links.
    - **Design Motivation**: To validate the hypothesis that if simply utilizing memorized training links can significantly improve performance, then the "good results" on existing benchmarks primarily stem from memory capacity rather than reasoning capabilities. CQD-Hybrid is simpler than other hybrid methods such as QTO and FIT (as it does not require score calibration or forward/backward updates) but is already sufficient to reach or even exceed SOTA.

3. **New Benchmark Construction Strategy**

    - **Function**: Construct three new benchmarks: FB15k237+H, NELL995+H, and ICEWS18+H.
    - **Mechanism**: Uniformly sample QA pairs of various difficulty levels for each query type. For example, for a 3p query, 30,000 QA pairs are sampled, including 10,000 that can be simplified to 1p, 10,000 that can be simplified to 2p, and 10,000 that are full-inference. Additionally, QA pairs in union queries whose reasoning trees contain "non-existent links" (edges that are neither in the training set nor in the test set) are filtered out. Two more challenging query types, 4p (four-hop path) and 4i (four-way intersection), are introduced.
    - **Design Motivation**: Eliminate the extreme bias toward simple QA pairs in existing benchmarks (where 1p simplicity can reach up to 98%), allowing the MRR metric to realistically reflect the reasoning capabilities of models across different difficulty levels.

4. **ICEWS18 Temporal Splitting Strategy**

    - **Function**: Construct a more realistic training/test split based on the temporal knowledge graph ICEWS18.
    - **Mechanism**: Traditional benchmarks split the training and test data randomly (assuming triple independence), which is unrealistic. ICEWS18+H sorts triples of facts chronologically: the first 80% is used for the training set, the middle 10% for validation, and the final 10% for testing, with only the earliest timestamp preserved for identical facts.
    - **Design Motivation**: Simulate the real-world scenario of "predicting the future using past knowledge," making even simple 1p queries significantly more challenging.

### Discovery of False Difficulty in Union Queries

The authors found that in FB15k237 and NELL995, the reasoning trees of 2u and 2u1p query types contain "non-existent links"—edges that belong to neither $G_{\text{train}}$ nor $G_{\text{test}}$. These edges violate the definition of inference-based QA pairs. After filtering out these anomalous QA pairs, the MRR of 2u queries recovered from being far below 1p to a level comparable to 1p, validating the theoretical expectation that 2u and 1p should share a similar level of difficulty.

## Key Experimental Results

### Main Results: Hardness Distribution of QA Pairs in Existing Benchmarks

| Query Type | Can be simplified to 1p (FB15k237) | Full-inference (FB15k237) | Can be simplified to 1p (NELL995) | Full-inference (NELL995) |
|----------|----------------------|--------------------------|---------------------|-------------------------|
| 2p | 98.1% | 1.9% | 97.6% | 2.4% |
| 3p | 97.2% | 0.1% | 95.6% | 0.1% |
| 2i | 96.0% | 4.0% | 94.0% | 6.0% |
| 3i | 91.6% | 0.2% | 87.4% | 0.5% |
| 1p2i | 86.8% | 0.2% | 49.5% | 0.9% |
| 2i1p | 96.7% | 0.1% | 96.2% | 0.2% |
| 2u1p | 98.3% | 0.1% | 98.5% | 0.1% |

### MRR Comparison of SOTA Methods on the New Benchmarks

| Method | 1p | 2p | 3p | 2i | 3i | 2i1p | 4p | 4i |
|------|-----|-----|-----|-----|-----|------|-----|-----|
| **FB15k237+H** | | | | | | | | |
| GNN-QE | 42.8 | 5.2 | 4.0 | 6.0 | 8.8 | 9.9 | 4.3 | 20.0 |
| CQD | 46.7 | 4.4 | 2.4 | 11.3 | 12.8 | 11.5 | 1.1 | 23.8 |
| CQD-Hybrid | 46.7 | 4.8 | 3.1 | 6.0 | 8.6 | 12.9 | 2.4 | 17.4 |
| QTO | 46.7 | 4.9 | 3.7 | 8.7 | 10.1 | 13.5 | 3.9 | 20.2 |
| CLMPT | 45.3 | 5.3 | 4.7 | 10.2 | 12.2 | 14.9 | 4.5 | 24.0 |
| **NELL995+H** | | | | | | | | |
| CQD | 60.4 | 9.6 | 4.2 | 18.5 | 19.6 | 22.6 | 2.0 | 24.8 |
| QTO | 60.3 | 9.8 | 8.0 | 14.6 | 15.8 | 21.1 | 7.0 | 20.9 |
| CLMPT | 58.1 | 10.1 | 7.8 | 22.7 | 25.0 | 24.4 | 7.2 | 29.1 |

### Key Findings

- **Staggering Performance Inflation**: On the old FB15k237 benchmark, QTO achieved an MRR of 54.6 on 3i queries, but on the new FB15k237+H benchmark, it scored only 10.1, a drop of over 44 points.
- **Memorization Equals SOTA**: Simply by adding training link memory onto CQD, CQD-Hybrid outperforms or matches more complex non-hybrid SOTA methods across 9 out of 14 metrics on the old benchmarks (Table 3).
- **No Single King of SOTA**: On the new benchmarks, no single method consistently claims the SOTA position across the majority of query types, indicating that existing methods might have overfit the simple QA pair distribution of old benchmarks.
- **False Difficulty of 2u Queries**: After filtering out non-existent links, the MRR of 2u queries recovers from ~15 to ~33-41 (FB15k237), close to the performance on 1p queries, confirming that the "difficulty" of 2u is merely an artifact of dataset construction.
- **Temporal Splitting is More Challenging**: ICEWS18+H is far more difficult than FB15k237+H and NELL995+H even on 1p queries (e.g., CQD's 1p MRR is only 16.6 vs 46.7 and 60.4, respectively), demonstrating that the dataset splitting method has a huge impact on benchmark difficulty.

## Highlights & Insights

- **The reasoning tree difficulty classification taxonomy is a concise yet profound analytical framework.** It formalizes the intuition that "query type $\neq$ query difficulty" into quantifiable concepts (trivial / partial-inference / full-inference), providing a theoretical foundation for the construction of all future CQA benchmarks. This idea can be transferred to any graph reasoning tasks plagued by "training information leakage."
- **The minimalist design of CQD-Hybrid reveals an embarrassing reality**: simply adding a single line of logic to an existing method—"assign the highest score if the edge exists in the training set"—yields results close to, or even exceeding, elaborately designed SOTA methods. This "minimal intervention experiment" serves as an excellent methodological exemplar for hypothesis validation.
- **The discovery of "non-existent links" in Union queries** is a pure data quality issue, yet it has been overlooked for a long time, causing the community to systematically misjudge the difficulty of 2u/2u1p query types. This serves as a reminder to always validate semantic consistency when constructing benchmarks.
- **The design methodology of new benchmarks using balanced sampling + temporal splitting** can be extended to the evaluation of other tasks requiring multi-step reasoning (such as multi-hop QA, program synthesis, etc.).

## Limitations & Future Work

- **Only single-target variable queries considered**: This analysis covers only queries that bind a single target variable, whereas many real-world queries require multi-variable binding (tuple answers). This needs to be extended to multi-variable settings in future work.
- **Analysis scope limited to the closed-world assumption**: The new benchmarks still assume that test links are the only "missing" links, failing to account for unknown entities and relations that may exist in an open world.
- **Lack of coverage for inductive settings**: The authors acknowledge that they have not analyzed inductive scenarios where unseen entities and relations appear during testing, which is an important research direction for CQA.
- **The simplicity of CQD-Hybrid is both an advantage and a limitation**: It only validates the "memorization hypothesis" but does not propose any new methods that can actually improve full-inference performance on the new benchmarks.
- **The difficulty of the new benchmarks may be excessively uniform**: Highly uniform distributions of query difficulty are unlikely in real-world applications. Whether the sampling proportion should be determined based on the statistical properties of real KGs is worth exploring.

## Related Work & Insights

- **vs CQD (Arakelyan et al., 2021)**: CQD decomposes complex queries into 1p sub-queries and scores them using a neural link predictor. Building upon this, the proposed CQD-Hybrid beats most methods simply by adding training link memory, which suggests that although CQD's framework itself is valid, its evaluation methodology had issues.
- **vs QTO (Bai et al., 2023)**: QTO employs more complex forward/backward updates and score calibration to leverage training links. This paper shows that even without these complex techniques, a simple maximum score assignment is already sufficiently effective.
- **vs GNN-QE (Zhu et al., 2022)**: GNN-QE decomposes complex queries into projection operations over fuzzy sets. Its multi-hop query performance drops drastically on the new benchmarks, indicating that its design relies more heavily on memorization than actual compositional reasoning.
- **vs ULTRAQ (Galkin et al., 2024)**: As a zero-shot CQA foundation model, ULTRAQ also performs poorly on the new benchmarks, illustrating that even highly generalizable foundation models have not truly mastered multi-hop reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐ The core finding (false complexity of benchmarks) is of high value, but the analysis is somewhat intuitive and technical novelty is moderate.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely systematic analysis, covering two legacy benchmarks and one new temporal benchmark, evaluating 6 SOTA methods across various query types (both with and without negation).
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is logically clear, has highly intuitive examples (e.g., Figure 1), and progresses smoothly step-by-step from diagnosis to validation and finally to the new benchmark construction.
- Value: ⭐⭐⭐⭐⭐ Has a fundamental impact on the benchmark evaluation paradigm of the CQA community; the new benchmarks will undoubtedly become standard testing grounds for future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Extending Complex Logical Queries on Uncertain Knowledge Graphs](../../ACL2025/graph_learning/extending_complex_logical_queries_uncertain_knowledge_graphs.md)
- [\[ACL 2025\] Can LLMs Evaluate Complex Attribution in QA? Automatic Benchmarking using Knowledge Graphs](../../ACL2025/graph_learning/paper_2401_14640.md)
- [\[ICLR 2026\] Latent Geometry-Driven Network Automata for Complex Network Dismantling](../../ICLR2026/graph_learning/latent_geometry-driven_network_automata_for_complex_network_dismantling.md)
- [\[ACL 2026\] LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs](../../ACL2026/graph_learning/llms_underperform_graph-based_parsers_on_supervised_relation_extraction_for_comp.md)
- [\[AAAI 2026\] Human Cognition Inspired RAG with Knowledge Graph for Complex Problem Solving](../../AAAI2026/graph_learning/human_cognition_inspired_rag_with_knowledge_graph_for_complex_problem_solving.md)

</div>

<!-- RELATED:END -->
