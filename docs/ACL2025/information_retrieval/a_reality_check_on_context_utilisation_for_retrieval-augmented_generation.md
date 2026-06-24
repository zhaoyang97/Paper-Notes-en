---
title: >-
  [Paper Note] A Reality Check on Context Utilisation for Retrieval-Augmented Generation
description: >-
  [ACL 2025][Information Retrieval & RAG][RAG] This paper proposes the DRUID real-world fact verification dataset and the ACU evaluation metric, revealing that synthetic datasets (CounterFact, ConflictQA) exaggerate the impact of context features, leading to overly optimistic assessments of LLM context utilization capabilities, and calling for the study of RAG using real-world retrieved data.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "RAG"
  - "Context Utilisation"
  - "Fact Verification"
  - "Synthetic vs. Real Data"
  - "Knowledge Conflict"
date: 2026-05-08
content_hash: 721aae717c282962
---

# A Reality Check on Context Utilisation for Retrieval-Augmented Generation

**Conference**: ACL 2025  
**arXiv**: [2412.17031](https://arxiv.org/abs/2412.17031)  
**Code**: None (DRUID dataset and evaluation framework are provided)  
**Area**: RAG / Fact Verification / Context Utilisation  
**Keywords**: RAG, Context Utilisation, Fact Verification, Synthetic vs. Real Data, Knowledge Conflict

## TL;DR

This paper proposes the DRUID real-world fact verification dataset and the ACU evaluation metric, revealing that synthetic datasets (CounterFact, ConflictQA) exaggerate the impact of context features, leading to overly optimistic assessments of LLM context utilization capabilities, and calling for the study of RAG using real-world retrieved data.

## Background & Motivation

Retrieval-Augmented Generation (RAG) is widely used to compensate for the deficiency in LLMs' parametric knowledge. For RAG to work effectively, two conditions must be met: (1) the retrieval module must find useful information, and (2) the generator model must correctly utilize the retrieved information. Prior research has focused heavily on the second issue—how LLMs utilize contextual information—but almost all of these studies rely on **synthetic datasets**.

The **Limitations of Prior Work** lie in the fact that synthetic datasets (e.g., CounterFact creating knowledge conflicts through template substitution, ConflictQA generating counterfactual evidence via LLMs) are overly simplistic and artificial, failing to represent the **complexity and diversity** of retrieved contexts in real RAG scenarios. In the real world, retrieved evidence can be insufficient, irrelevant, unreliable, implicit, or hard to comprehend—characteristics that are almost nonexistent in synthetic datasets.

**Key Challenge**: Can context utilization conclusions derived from synthetic data transfer to real-world scenarios? If not, guidance from existing research on optimizing RAG systems could be misleading.

The **Key Insight** of this paper is to build DRUID, a dataset based on real-world claims and automatically retrieved evidence, to conduct a systematic comparison with synthetic datasets and propose a new context utilization metric, ACU, for fair evaluation.

## Method

### Overall Architecture

The research pipeline consists of three phases: (1) building the DRUID dataset (real-world claims + automatically retrieved evidence + human annotation), (2) analyzing the context feature differences between DRUID and synthetic datasets, and (3) using the ACU metric to compare LLM context utilization behaviors across different datasets.

### Key Designs

1. **DRUID Dataset Construction**:

    - **Function**: Collect real-world fact-checking claims, retrieve evidence using automated retrieval, and conduct human annotation
    - **Mechanism**: Collect claims from 7 different sources of fact-checking websites, retrieve relevant web pages using Google and Bing search engines, and extract evidence snippets re-ranked by the Cohere Rerank model. DRUID contains 1,329 claims and 5,490 <claim, evidence, label> triplets; the extended version DRUID+ contains 48,517 samples
    - **Annotation Scheme**: Each piece of evidence is annotated by two annotators for **relevance** (Relevant/Irrelevant) and **stance** (Support / Insufficient-Support / Insufficient-Neutral / Insufficient-Contradict / Insufficient-Refute / Refute), forming a six-level annotation. The Krippendorff's alpha is approximately 0.71
    - **Design Motivation**: Unlike synthetic datasets, real-world retrieved evidence naturally contains characteristics such as insufficiency, irrelevance, and leakage. 50% of the automatically retrieved evidence lacks a clear stance or has insufficient information—a phenomenon completely absent in synthetic datasets

2. **ACU (Accumulated Context Usage) Evaluation Metric**:

    - **Function**: Uniformly measure the degree of context utilization of different LLMs on different datasets
    - **Mechanism**: Given a claim $C$ and evidence $E$, compute the softmax probability difference of the model for target tokens $t \in \{True, None, False\}$ with and without evidence, scaled by normalization:

    $\Delta P_M(t|C,E) = \begin{cases} \frac{P_M(t|C,E) - P_M(t|C)}{1 - P_M(t|C)} & \text{if } P_M(t|C,E) \geq P_M(t|C) \\ \frac{P_M(t|C,E) - P_M(t|C)}{P_M(t|C)} & \text{otherwise} \end{cases}$

      ACU aggregates change across all tokens: $ACU(C,E,S_E,M) = \frac{1}{|T|} \sum_{t \in T} D(t, S_E) \cdot \Delta P_M(t|C,E)$

      where $D(t, S_E)$ is the expected direction of change (+1 or -1), and ACU ranges from [-1, 1]

    - **Design Motivation**: Existing metrics either inspect only overall output distribution changes (which does not guarantee changes are context-related) or use unnormalized logit differences (rendering different models incomparable). ACU addresses both issues simultaneously

3. **Context Feature Detection and Analysis**:

    - **Function**: Detect and compare multiple context features in synthetic and real-world datasets
    - **Mechanism**: Detect the following features: claim-evidence similarity (Jaccard similarity), incomprehensibility (Flesch readability score, perplexity), implicitness (named entity overlap), citation of external sources, unreliable sources, and uncertainty markers (hedge word detection). A hybrid of rule-based and LLM-prompted approaches is used for detection
    - **Design Motivation**: Quantify the gap between synthetic and real-world data through comprehensive feature comparison

### Evaluation Settings

- Models: Pythia 6.9B and Llama 3.1 8B
- Datasets: CounterFact, ConflictQA, DRUID
- Prompts: Manually-tuned 3-shot prompts (optimized on 390 DRUID samples)
- All data unified into a claim verification format

## Key Experimental Results

### Main Results: ACU Comparison

| Dataset | Model | ACU of Supporting Evidence | ACU of Refuting Evidence | Description |
|--------|------|-------------|-------------|------|
| CounterFact | Llama 3-shot | High (~0.74 avg) | Negative value (context-repulsion) | Synthetic data exaggerates context utilization |
| ConflictQA | Llama 3-shot | Highest (~0.71 avg) | Significantly negative | Generated evidence leads to stronger bias |
| DRUID | Llama 3-shot | Moderate (~0.84 avg) | Fewer negative values | Milder utilization patterns in real-world scenarios |
| DRUID | Pythia 3-shot | Low (~0.25 avg) | - | Huge differences across different models |

### Ablation Study: Correlation Between Context Features and ACU

| Context Feature | CounterFact/ConflictQA Correlation | DRUID Correlation | Description |
|-----------|---------------------------|------------|------|
| Claim-evidence overlap | Moderately high & significant | Low & insignificant | Similarity is a strong predictor in synthetic data, but not in real-world data |
| Perplexity | High (CF) | Low | Artificial construction of CounterFact leads to high perplexity |
| Evidence source (fact-checking site) | N/A | Significantly positively correlated | Source type is more important than single features in real-world data |
| Citation of external sources | Low | Low | Consistent with findings of prior studies |
| Evidence length | Insignificant in synthetic data | Negatively correlated | LLMs are less faithful to long refuting evidence |

### Key Findings

- **Synthetic datasets severely exaggerate "context-repulsion"**: Context repulsion rarely occurs for models on DRUID, whereas it frequently occurs on synthetic data.
- **Knowledge conflict is less frequent in real-world scenarios**: Only 58% of supporting evidence in DRUID involves knowledge conflict (Llama), compared to up to 97% in CounterFact.
- **No single context feature can predict RAG failure in real-world scenarios**: Features highly correlated in synthetic data show surprisingly low correlation in DRUID.
- **Aggregate features of evidence sources are more important than single features**: Evidence from fact-checking sources exhibits higher ACU scores, likely because these sources integrate multiple beneficial features.
- Llama is generally more faithful to context than Pythia.

## Highlights & Insights

- **Significant methodological contribution**: The ACU metric addresses comparability across different models and datasets, making it worthy of wide adoption in the RAG evaluation field.
- **Six-level stance annotation of DRUID** innovatively introduces fine-grained categories of insufficient evidence, which is closer to real-world scenarios than the traditional support/neutral/refute tri-classification.
- **Core Insight**: When studying RAG context utilization, conclusions derived using synthetic data may not transfer—posing questions to a large body of mechanistic interpretability research based on CounterFact.
- **Transferable design**: The method of unifying all data into a claim-verification format can be generalized to other RAG tasks.

## Limitations & Future Work

- **Limited task scope**: Based only on fact-verification tasks, and has not validated whether conclusions transfer to other RAG tasks such as open-domain QA.
- **Insufficient model coverage**: Only two open-source models (Pythia 6.9B, Llama 3.1 8B) were tested, omitting larger scale or commercial models.
- **Correlation analysis only without causal analysis**: Spearman correlation cannot reveal causal relationships between features and context utilization.
- **Single retrieval method**: Relies solely on a Google/Bing + Cohere Rerank retrieval pipeline; different retrieval methods may yield different distributions of context features.
- **English language constraint**: Only covers English claims and evidence.

## Related Work & Insights

- **vs CounterFact (Ortu et al., 2024)**: CounterFact is based on WikiData template substitution, with highly simplified and artificial contexts; DRUID uses real retrieved evidence, which is more representative but harder to control variables.
- **vs ConflictQA (Xie et al., 2024)**: ConflictQA uses LLMs to generate evidence, which is more natural than CounterFact, yet still suffers from generation biases (highly coherent and persuasive texts are uncommon in real retrieval).
- **vs RAG Evaluation Survey (Gao et al., 2024)**: This work fills the vacancy of "real-world retrieved context" in RAG evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐ The DRUID dataset and ACU metric display clear innovation, but the core idea (synthetic vs. real comparison) is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The experiments are highly comprehensive: 3 datasets, 2 models, multiple context features, and various prompting strategies.
- Writing Quality: ⭐⭐⭐⭐⭐ The structure is clear, analysis is deep, and the appendix is extremely detailed.
- Value: ⭐⭐⭐⭐⭐ An important reflection on the RAG evaluation paradigm, with DRUID and ACU holding broad practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation](faithfulrag_fact_level_conflict.md)
- [\[ACL 2025\] Hierarchical Document Refinement for Long-context Retrieval-augmented Generation](hierarchical_document_refinement_for_long-context_retrieval-augmented_generation.md)
- [\[ACL 2025\] EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation](exit_context-aware_extractive_compression_for_enhancing_retrieval-augmented_gene.md)
- [\[ACL 2025\] Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs](gor_rag_long_context_summary.md)
- [\[ACL 2025\] Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation](towards_adaptive_memory-based_optimization_for_enhanced_retrieval-augmented_gene.md)

</div>

<!-- RELATED:END -->
