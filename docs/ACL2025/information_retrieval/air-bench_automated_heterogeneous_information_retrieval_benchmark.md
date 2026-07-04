---
title: >-
  [Paper Note] AIR-Bench: Automated Heterogeneous Information Retrieval Benchmark
description: >-
  [ACL 2025][Information Retrieval & RAG][Information Retrieval] This paper proposes AIR-Bench, the first heterogeneous IR benchmark that leverages LLMs to automatically generate test data. It covers 2 tasks (QA/Long-Doc), 9 domains, and 13 languages across 69 datasets. A three-stage quality control pipeline ensures that the generated data is highly consistent with human annotations, addressing the limitations of narrow domain coverage and high update costs in traditional IR be…
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Information Retrieval"
  - "Automated Benchmark"
  - "LLM Data Generation"
  - "Multilingual Retrieval"
  - "RAG Evaluation"
date: 2026-05-08
content_hash: 0840076cbcdcd330
---

# AIR-Bench: Automated Heterogeneous Information Retrieval Benchmark

**Conference**: ACL 2025  
**arXiv**: [2412.13102](https://arxiv.org/abs/2412.13102)  
**Code**: [GitHub](https://github.com/AIR-Bench/AIR-Bench)  
**Area**: Information Retrieval / Benchmark Evaluation  
**Keywords**: Information Retrieval, Automated Benchmark, LLM Data Generation, Multilingual Retrieval, RAG Evaluation

## TL;DR

This paper proposes AIR-Bench, the first heterogeneous IR benchmark that leverages LLMs to automatically generate test data. It covers 2 tasks (QA/Long-Doc), 9 domains, and 13 languages across 69 datasets. A three-stage quality control pipeline ensures that the generated data is highly consistent with human annotations, addressing the limitations of narrow domain coverage and high update costs in traditional IR benchmarks.

## Background & Motivation

**Background**: IR evaluation has evolved from monolingual QA (MS MARCO, Natural Questions) to multilingual QA (Mr.TyDi, MIRACL), and then to general cross-domain retrieval (BEIR, MTEB). These benchmarks have driven the rapid development of retrieval models.

**Limitations of Prior Work**: (1) Limited domain coverage: Existing benchmarks are constrained by predefined domains and human annotation, making it difficult to efficiently scale to emerging domains (e.g., sector-specific legal or financial IR); (2) Risk of data leakage: Test data of popular benchmarks may have already been crawled or included in the training sets of retrieval models, leading to distorted evaluations; (3) High update costs: Human annotation of new domain data requires substantial time and domain expertise, failing to keep pace with rapid domain evolution.

**Key Challenge**: IR models are becoming increasingly general and powerful, but the coverage and update frequency of evaluation benchmarks lag far behind. This leads to saturation on many popular benchmarks (e.g., intensive in-domain fine-tuning on MTEB/C-MTEB has pushed scores close to their upper limits).

**Goal**: Build an automated, heterogeneous, and dynamic IR benchmark that can continuously scale to new domains and languages at low cost while ensuring that the quality of the generated test data is highly consistent with human annotations.

**Key Insight**: Leverage LLMs (GPT-4) to automatically generate test data. Instead of simply prompting LLMs to fabricate query-passage pairs out of thin air, a multi-stage pipeline of "Persona $\to$ Scenario $\to$ Query $\to$ Rewrite $\to$ Hard Negatives" is designed based on real-world corpora to generate high-quality and diverse test data.

**Core Idea**: Automate the construction of IR benchmarks using LLMs, enabling low-cost coverage of arbitrary new domains while ensuring consistency with human-annotated benchmarks through rigorous quality control.

## Method

### Overall Architecture

The data generation pipeline of AIR-Bench consists of three stages: (1) **Corpus Preparation**: Collecting and preprocessing real-world multi-domain and multilingual corpora; (2) **Candidate Generation**: Iteratively generating queries, positives, and hard negatives using LLM based on the corpora; (3) **Quality Control**: Filtering low-quality queries and rectifying incorrect label annotations. The final generated dataset contains a corpus $\mathcal{D}$, a query set $\mathcal{Q}$, and a relevance label set $\mathcal{R}$.

### Key Designs

1. **Multi-Stage Query Generation Pipeline**:

    - **Function**: Automatically generate diverse and high-quality retrieval queries starting from a real-world corpus.
    - **Mechanism**: A six-step iterative loop: (1) Sample a positive document $d_i^+$ from the corpus; (2) Use LLM to generate a "character" (persona) who might need this document; (3) Generate a "scenario" where this character would use the document; (4) Generate an original query $ori\_q_i$ based on the character + scenario while controlling query length, type, information need type, and expression style; (5) Use LLM to rewrite the query to remove overlapping vocabulary with the source document, yielding the final $q_i$; (6) Generate hard negative documents $\{d_i^{-}(j)\}$ based on the query and positive document.
    - **Design Motivation**: The intermediate steps of character + scenario are more transparent and controllable than direct query generation, significantly increasing diversity. Query rewriting increases retrieval difficulty, and hard negative generation makes the evaluation more discriminative.

2. **Two-Level Quality Control Mechanism**:

    - **Function**: Filter low-quality LLM-generated data and rectify incorrect relevance labels.
    - **Mechanism**: Two components: (a) **Low-Quality Query Filtering**: Use LLM to predict the relevance between $q_i$ and $d_i^+$, discarding the query if predicted as negative; (b) **Label Error Rectification**: A three-step pipeline: retrieve top-1000 documents using an embedding model $\to$ pre-label with multiple reranking models (majority voting) $\to$ final LLM judgment. Different strategies are applied to three categories of documents: original positives (retained), generated hard negatives judged as positive (discarded), and missed positive documents in the corpus (added to $\mathcal{D}_+$).
    - **Design Motivation**: LLM generation inevitably introduces low-quality queries and incorrect labels. Combining multi-model voting with LLM verification guarantees data quality to the greatest extent.

3. **Dual-Task Heterogeneous Evaluation Design**:

    - **Function**: Cover two core application scenarios: traditional retrieval and modern RAG.
    - **Mechanism**: (a) **QA Task**: Classic question-answering retrieval evaluated on large-scale document collections, with nDCG@10 as the primary metric; (b) **Long-Doc Task**: Chunk-level retrieval of long documents, closely resembling RAG scenarios, with Recall@10 as the primary metric (as recall of positives is more critical than ranking in RAG). These tasks cover 9 major domains (News, Web, Wiki, Science, Finance, Medicine, Law, ArXiv, Books) and 13 languages.
    - **Design Motivation**: Distinguish the evaluation requirements of different retrieval scenarios: traditional QA focuses on ranking quality, while RAG focuses on recall completeness.

### Design Considerations for Data Generation

- **Reliance on Real Corpus**: Generated based on real-world corpora, ensuring the evaluation is close to practical scenarios while keeping costs manageable.
- **Query Rewriting**: Changes query forms while preserving semantics, increasing retrieval difficulty.
- **Hard Negative Generation**: Increases the distinguishability of the evaluation.
- **GPT-4 temperature=1.0**: Encourages greater diversity.

## Key Experimental Results

### Consistency Validation: Generated vs. Human-Annotated Data

Taking MS MARCO as an example, comparing the original human annotations (R-MSMARCO) and the AIR-Bench generated version (G-MSMARCO):

| Dataset | Corpus Size | Query Count | Positive Labels |
|--------|---------|--------|-----------|
| R-MSMARCO (Human) | 8,841,823 | 6,980 | 7,437 |
| G-MSMARCO (Generated + QC) | 8,872,840 | 6,319 | 31,447 |
| G-MSMARCO (W/O QC) | 8,878,865 | 7,429 | 7,429 |

### Model Ranking Consistency

| Model | Parameters | R-MSMARCO Rank | G-MSMARCO Rank | Rank Consistency |
|------|--------|-------------|-------------|-----------|
| repllama-v1-7b | 6.74B | 1 | 1 | ✅ |
| e5-large-v2 | 335M | 2 | 4 | ≈ |
| multilingual-e5-large | 560M | 3 | 5 | ≈ |
| bge-large-en-v1.5 | 335M | 5 | 2 | ≈ |

Crucial role of quality control: Without QC, the model ranking is severely inconsistent with human annotation (e.g., repllama rank drops from 1st to 2nd). After QC, the rankings become highly consistent.

### Ablation Study

| Ablation Item | Impact | Description |
|--------|------|------|
| Remove QC | Model ranking consistency drops significantly | Proves the QC module is indispensable |
| Remove Query Rewriting | Retrieval difficulty decreases | Rewriting increases vocabulary diversity |
| Remove Hard Negatives | Evaluation discriminativeness drops | Hard negatives make the benchmark more challenging |
| Query Type Distribution | "what" is the most frequent (30-34%), followed by "claim" (22-26%) | Covers multiple query types |

### Key Findings

- **LLM-generated test data is highly consistent with human annotations**: The key prerequisite is quality control; without QC, both data quality and model ranking consistency drop significantly.
- **AIR-Bench remains highly discriminative for existing models**: It avoids the saturation issue observed in benchmarks like MTEB, because the generated data is highly unlikely to be covered by the training sets.
- **High query diversity**: It covers various query types such as how/what/when/where/which/who/why/Yes-No/claim with a relatively balanced distribution.
- **Dynamic updates are a core advantage**: The transition from version 24.04 to 24.05 expanded to 69 datasets, with continuous scaling in progress.

## Highlights & Insights

- **The three-step "Persona $\to$ Scenario $\to$ Query" generation is more controllable and diverse than direct generation**: The design of this intermediate step is inspired by the fact that different users have diverse information needs in reality, preventing the generated queries from falling into the "default mode" of LLMs.
- **Quality control uses a three-tier "Embedding Retrieval + Multi-model Voting + LLM Judgment" verification**: It neither solely relies on LLMs (which may share generation biases) nor entirely on embedding models (which may have retrieval blind spots), utilizing complementary advantages. This pipeline design can be migrated to the construction of other automated evaluation benchmarks.
- **Dynamic benchmark addresses the "arms race" in AI evaluation**: Traditional static benchmarks are increasingly prone to overfitting. The dynamic generation + periodic update model keeps the benchmark continuously challenging.

## Limitations & Future Work

- **By-product of GPT-4 capacity boundaries**: The generation quality might degrade in domains where GPT-4 has weaker comprehension (e.g., highly specialized medical/legal terminology).
- **Binary relevance annotation only**: It only determines relevant/irrelevant, lacking multi-level relevance labels (such as 1-5 stars in ACORD), which limits the evaluation precision of ranking quality.
- **Chunking strategies in the Long-Doc task are not fully discussed**: While different chunking methods significantly impact retrieval performance, this variable is not analyzed in the paper.
- **Risk of circular bias in "LLMs evaluating LLMs"**: Evaluating with GPT-4-generated data may favor GPT-4-associated embedding models.
- **Underrepresentation of low-resource languages among the 13 covered languages**: It lacks representation for low-resource languages from regions such as Africa and Southeast Asia.

## Related Work & Insights

- **vs BEIR (Thakur et al., 2021)**: BEIR aggregates existing human-annotated datasets for cross-domain evaluation, but its domains are fixed; whereas AIR-Bench can automatically scale to any new domain.
- **vs MTEB (Muennighoff et al., 2023)**: MTEB is currently the most popular multi-task embedding benchmark, but faces saturation and data leakage issues; AIR-Bench's dynamic nature inherently resists overfitting.
- **vs MIRACL (Zhang et al., 2023)**: MIRACL focuses on multilingual retrieval but relies on human annotation (covering 18 languages, which is extremely expensive); AIR-Bench's automated pipeline covers 13 languages at a fraction of the cost with sustainable scalability.

## Rating

- Novelty: ⭐⭐⭐⭐ The first fully automated heterogeneous IR benchmark, addressing the scalability and timeliness limitations of traditional benchmarks.
- Experimental Thoroughness: ⭐⭐⭐⭐ Thorough consistency validation between generated and human-annotated data, along with a complete ablation analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, highly detailed descriptions of pipeline designs, and standard mathematical notations.
- Value: ⭐⭐⭐⭐ Highly valuable, providing the IR community with a continuously evolving evaluation platform.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Atomic LLM: A Fine-Grained Information Retrieval Evaluation Benchmark for Language Models](atomic_llm_a_fine-grained_information_retrieval_evaluation_benchmark_for_languag.md)
- [\[ACL 2025\] CoIR: A Comprehensive Benchmark for Code Information Retrieval Models](coir_a_comprehensive_benchmark_for_code_information_retrieval_models.md)
- [\[ACL 2025\] HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation](hoh_a_dynamic_benchmark_for_evaluating_the_impact_of_outdated_information_on_ret.md)
- [\[ACL 2025\] Logical Consistency is Vital: Neural-Symbolic Information Retrieval for Negative-Constraint Queries](logical_consistency_is_vital_neural-symbolic_information_retrieval_for_negative-.md)
- [\[ACL 2025\] Any Information Is Just Worth One Single Screenshot: Unifying Search With Visualized Information Retrieval](any_information_is_just_worth_one_single_screenshot_unifying_search_with_visuali.md)

</div>

<!-- RELATED:END -->
