---
title: >-
  [Paper Note] Query-driven Document-level Scientific Evidence Extraction from Biomedical Studies
description: >-
  [ACL 2025][Medical LLM][evidence extraction] This paper proposes the URCA (Uniform Retrieval Clustered Augmentation) framework, which automatically extracts scientific evidence and conclusions related to clinical questions from the full texts of RCT studies using a RAG pipeline of uniform retrieval, clustering, and knowledge extraction. It achieves an 8.81% F1 improvement over the best baseline on the newly constructed CochraneForest dataset.
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "evidence extraction"
  - "systematic review"
  - "RAG"
  - "biomedical NLP"
  - "forest plot"
date: 2026-05-08
content_hash: cfa3db842854924b
---

# Query-driven Document-level Scientific Evidence Extraction from Biomedical Studies

**Conference**: ACL 2025  
**arXiv**: [2505.06186](https://arxiv.org/abs/2505.06186)  
**Code**: To be released  
**Area**: Biomedical NLP  
**Keywords**: evidence extraction, systematic review, RAG, biomedical NLP, forest plot  

## TL;DR
This paper proposes the URCA (Uniform Retrieval Clustered Augmentation) framework, which automatically extracts scientific evidence and conclusions related to clinical questions from the full texts of RCT studies using a RAG pipeline of uniform retrieval, clustering, and knowledge extraction. It achieves an 8.81% F1 improvement over the best baseline on the newly constructed CochraneForest dataset.

## Background & Motivation

**Background**: Systematic reviews are the gold standard of evidence-based medicine, but producing one takes 1-2 years on average and costs over $141,000. One of the core steps is extracting evidence conclusions (e.g., "favours intervention", "favours placebo", "no difference") related to the research question from multiple RCT papers and visualizing them in a forest plot.

**Limitations of Prior Work**: (1) Existing automation studies mainly focus on retrieving and screening papers, with insufficient attention paid to document-level evidence extraction; (2) An RCT study may involve multiple papers (published at different times), requiring cross-document information synthesis; (3) Standard RAG systems are prone to biasing towards certain sources in multi-source scenarios, leading to uneven retrieval.

**Key Challenge**: In multi-paper scenarios, standard RAG tends to favor documents with the highest superficial similarity to the query, ignoring key evidence in other sources. However, clinical evidence requires balanced consideration of information across all relevant papers.

**Goal**: (1) Formally define the "document-level scientific evidence extraction" task; (2) Build a benchmark dataset containing conflicting conclusions; (3) Propose a RAG framework suitable for multi-source evidence synthesis.

**Key Insight**: Leveraging forest plots from Cochrane systematic reviews as the source of annotation—the forest plot itself contains the annotated conclusion of each study on a specific question.

**Core Idea**: Solve the information skewness issue in multi-source RAG by uniformly allocating retrieval quotas to each source paper, clustering the retrieval results, and extracting knowledge cluster-by-cluster.

## Method

### Overall Architecture
Given a clinical research question $q$ and multiple papers of a study $S = \{p_1, ..., p_n\}$ $\rightarrow$ URCA's three-step pipeline: (1) Uniform Retrieval: uniformly retrieve $k_s$ paragraphs from each paper; (2) Clustering + Knowledge Extraction: cluster the retrieved paragraphs and use an LLM to extract information related to $q$ from each cluster; (3) Answer Finalization: generate the final conclusion based on the extracted knowledge (favours left/right intervention, no difference).

### Key Designs

1. **Uniform Retrieval**:

    - **Function**: Allocate the retrieval quota $k$ uniformly across all papers, rather than performing global top-k retrieval.
    - **Mechanism**: Each source is allocated $k_s = \lceil \min(k + \beta \cdot \log(S), N_{\max}) / S \rceil$ retrieved paragraphs, where $\beta$ controls the logarithmic adjustment for the number of sources.
    - **Design Motivation**: The global top-k of standard RAG tends to favor papers with high surface similarity to the query, overlooking key information in other papers (such as secondary outcomes or results from different periods). Uniform retrieval ensures that every paper has representation.
    - Ablation studies demonstrate that adding only uniform retrieval provides stable improvements, serving as the foundation of the entire framework.

2. **Clustering & Knowledge Extraction**:

    - **Function**: Perform UMAP dimension reduction and GMM clustering on retrieved paragraphs, and then use an LLM to extract knowledge related to $q$ from each cluster.
    - **Mechanism**: Borrowing from the recursive clustering method of RAPTOR, but instead of recursive summarization, an LLM is used to extract query-related information snippets $D_i = \mathcal{M}_\theta(p_{\text{extr}}, q, c_i)$ from each cluster individually.
    - **Design Motivation**: Directly concatenating all retrieved paragraphs for the LLM introduces substantial noise. Extracting cluster-by-cluster after clustering allows for more accurate filtering of irrelevant information, while preserving complementary information from different sources.

3. **Answer Finalization**:

    - **Function**: Use the knowledge extracted from all clusters $\langle D_1, ..., D_n \rangle$ as context for the LLM to generate the final conclusion.
    - The conclusion is a three-way child classification: flavours left intervention / favours right intervention / no difference.

### CochraneForest Dataset Construction
- **Source**: Cochrane CDSR database (9,301 systematic reviews, 220,000+ studies)
- **Filtering Pipeline**: Exclude retracted reviews $\rightarrow$ Keep the latest version $\rightarrow$ At least 2 studies $\rightarrow$ Full text available for all studies $\rightarrow$ Include forest plots with conflicting conclusions
- **Final Scale**: 202 annotated forest plots, originating from 48 systematic reviews, 263 unique studies, and 923 records
- **Annotation Content**: (1) Verification/editorial review of research questions; (2) Conclusion annotation for each study (automatically pre-annotated based on CI); (3) Correction of intervention names
- **Inter-annotator Agreement**: Semantic cosine similarity of 0.95 (Task 1) and 0.90 (Task 3), indicating high quality

## Key Experimental Results

### Main Results

| Method | Llama-3.1-70B | Mistral-Large | Granite-8B | GPT-3.5 | GPT-4 |
|------|-------------|-------------|-----------|---------|-------|
| No RAG | 49.06 | 46.09 | 36.15 | 24.07 | 47.46 |
| Abstracts | 60.71 | 62.58 | 57.65 | 56.04 | 61.04 |
| RAG | 62.09 | 60.87 | 56.11 | 59.06 | 61.56 |
| + Uniform | 63.42 | 63.27 | 58.71 | 61.83 | 61.99 |
| RAPTOR | 60.60 | 61.70 | 54.70 | 53.61 | 60.07 |
| InstructRAG | 60.92 | 62.57 | 51.46 | 57.42 | 61.63 |
| **URCA** | **66.11** | **67.26** | **59.53** | **62.42** | **65.72** |

### Ablation Study

| Configuration | F1 (Mistral-Large) | Description |
|------|-------------------|------|
| Full URCA | 67.26 | Full model |
| w/o Uniform (Standard Retrieval) | ~61% | Remove uniform retrieval, degrading to RAPTOR |
| w/o Clustering (Direct Concatenation) | ~63% | Remove clustering, degrading to Uniform RAG |
| RAG + Uniform only | 63.27 | Only uniform retrieval, no clustering |

### Key Findings
- **URCA consistently outperforms all baselines**: Achieving the best performance across 5 LLMs, with a maximum gain of 8.81% F1 (on GPT-3.5 vs. RAPTOR).
- **Uniform retrieval is a crucial foundation**: Simply adding uniform retrieval stably improves standard RAG by 1-3% F1.
- **RAPTOR performs unexpectedly poorly**: Recursive summarization loses fine-grained evidence when aggregating information.
- **Using only abstracts is highly competitive**: The "Abstracts" baseline is close to or even exceeds standard RAG on multiple LLMs, showing that abstracts contain a large amount of crucial information.
- **No RAG performs poorly**: Internal knowledge of LLMs is insufficient to determine the conclusions of specific RCTs, making external retrieval indispensable.

## Highlights & Insights
- **Precise task definition**: Leveraging forest plots as "natural annotations" cleverly solves the cost issue of manual labeling. Since forest plots already contain the annotated conclusions of each study on specific questions, transforming this into an NLP task is highly sophisticated.
- **Generality of uniform retrieval**: This approach can be transferred to any multi-document question answering scenario—as long as the answer needs to balance references across multiple sources, uniformly allocating retrieval quotas is more rational than a global top-k approach.
- **Two-stage design of clustering and extraction**: Clustering similar paragraphs first and then extracting cluster-by-cluster strikes a better balance between signal-to-noise ratio and information coverage compared to direct concatenation or recursive summarization.

## Limitations & Future Work
- The dataset scale is limited (202 forest plots, 923 records), which may be insufficient to train end-to-end models.
- Only three-way classification conclusions (favours left/right/no difference) are considered; finer-grained information such as effect size is not modeled.
- The clustering and knowledge extraction steps introduce additional LLM call costs, which might be expensive in practical deployment.
- Special retrievers for the biomedical domain (such as PubMedBERT-based retriever) have not been explored.
- The task assumes that source papers are pre-screened; an automated paper screening step is still required in the full systematic review pipeline.

## Related Work & Insights
- **vs. RAPTOR**: RAPTOR uses recursive clustering + summarization to build a document tree but underperforms URCA on this task, as recursive summarization loses fine-grained details specifically related to the query.
- **vs. InstructRAG**: InstructRAG requires the model to generate a reasoning chain connecting the answer in the evidence, but the quality of the reasoning chain is unstable in the BioRE scenario.
- **vs. vanilla RAG**: The global top-k of standard RAG suffers from skewness in multi-source scenarios. URCA's uniform retrieval directly addresses this core bottleneck.

## Rating
- Novelty: ⭐⭐⭐⭐ The task definition and dataset construction method are novel; the RAG process of uniform retrieval + clustering extraction is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison across 5 LLMs and 6 methods, with thorough ablation.
- Writing Quality: ⭐⭐⭐⭐ The problem formalization is clear, and the method description is complete.
- Value: ⭐⭐⭐⭐ Holds practical significance for evidence synthesis automation, and the dataset fills an important gap.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Improving Automatic Evaluation of LLMs in Biomedical Relation Extraction via LLMs-as-the-Judge](biore_llm_judge_evaluation.md)
- [\[NeurIPS 2025\] Demo: Guide-RAG: Evidence-Driven Corpus Curation for Retrieval-Augmented Generation in Long COVID](../../NeurIPS2025/medical_nlp/demo_guide-rag_evidence-driven_corpus_curation_for_retrieval-augmented_generatio.md)
- [\[ACL 2026\] Ryze: Evidence-Enriched Data Synthesis from Biomedical Papers](../../ACL2026/medical_nlp/ryze_evidence-enriched_data_synthesis_from_biomedical_papers.md)
- [\[ACL 2025\] ANGEL: Learning from Negative Samples in Biomedical Generative Entity Linking](learning_from_negative_samples_in_biomedical_generative_entity_linking.md)
- [\[ACL 2025\] SECRET: Semi-supervised Clinical Trial Document Similarity Search](secret_semi-supervised_clinical_trial_document_similarity_search.md)

</div>

<!-- RELATED:END -->
