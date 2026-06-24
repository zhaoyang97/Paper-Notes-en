---
title: >-
  [Paper Note] Multilingual Retrieval Augmented Generation for Culturally-Sensitive Tasks: A Benchmark for Cross-lingual Robustness
description: >-
  [ACL 2025][Information Retrieval & RAG][Multilingual RAG] The BordIRLines benchmark dataset was constructed, containing territorial dispute queries in 49 languages with paired retrieved Wikipedia documents. Through a systematic evaluation of cross-lingual robustness in multilingual RAG environments, it was found that **retrieving multilingual documents improves response consistency and reduces geopolitical bias better than retrieving only same-language documents**.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Multilingual RAG"
  - "Cross-lingual Robustness"
  - "Territorial Disputes"
  - "Geopolitical Bias"
  - "Culturally-Sensitive Tasks"
date: 2026-05-08
content_hash: 92b43cd43966cd5d
---

# Multilingual Retrieval Augmented Generation for Culturally-Sensitive Tasks: A Benchmark for Cross-lingual Robustness

**Conference**: ACL 2025  
**arXiv**: [2410.01171](https://arxiv.org/abs/2410.01171)  
**Code**: Yes (Both dataset and code are released)  
**Area**: Information Retrieval  
**Keywords**: Multilingual RAG, Cross-lingual Robustness, Territorial Disputes, Geopolitical Bias, Culturally-Sensitive Tasks

## TL;DR

The BordIRLines benchmark dataset was constructed, containing territorial dispute queries in 49 languages with paired retrieved Wikipedia documents. Through a systematic evaluation of cross-lingual robustness in multilingual RAG environments, it was found that **retrieving multilingual documents improves response consistency and reduces geopolitical bias better than retrieving only same-language documents**.

## Background & Motivation

RAG (Retrieval-Augmented Generation) can mitigate LLM hallucinations but also introduces biases contained in the retrieved documents. These biases are amplified in multilingual and culturally-sensitive scenarios:

**Language influences stance**: Li et al. (2024b) found that LLM responses to territorial disputes vary with the query language—asking "To which country does Ceuta belong?" in Spanish yields "Spain", while asking in Arabic yields "Morocco".

**Document selection influences answers**: The answers of RAG systems are highly dependent on the retrieved documents, while Wikipedia documents in different languages may present different perspectives.

**Core Problem**:
   - How does the language composition of documents affect responses?
   - Does gathering information from different languages increase or decrease consistency?
   - Is multilingual retrieval superior to monolingual retrieval?

Limitations of Prior Work:
- Existing multilingual RAG research only considers a few high-resource languages and uses synthetic documents.
- Open-retrieval multilingual QA mainly focuses on simple factual questions where LLMs might have already memorized the answers.
- There is a lack of large-scale benchmarks to systematically evaluate cross-lingual RAG.

## Method

### Overall Architecture

The construction and evaluation pipeline of the BordIRLines benchmark:
1. **Data Source**: Based on 720 queries (251 disputed territories) from the BorderLines dataset, expanded to 49 languages.
2. **Retrieval Phase**: Retrieve relevant documents from Wikipedia using a multilingual IR system.
3. **Generation Phase**: Combine the query and retrieved documents into a prompt for the LLM to generate responses.
4. **Evaluation Phase**: Evaluate response quality through cross-lingual robustness metrics.

### Key Designs

1. **Five IR Modes (Information Retrieval Modes)**:

    - Function: Defines five different strategy combinations of languages for document retrieval.
    - **qlang**: Retrieves documents only in the query language (monolingual IR).
    - **rel_langs**: Retrieves documents in all relevant languages (query language + English + other relevant languages) (multilingual IR).
    - **qlang+en**: Retrieves documents in the query language and English.
    - **en_only**: Retrieves documents only in English (cross-lingual IR).
    - **swap_docs**: Adversarially selects documents in non-query languages.
    - Design Motivation: Each mode reflects different real-world information-seeking needs.

2. **Cross-lingual Robustness Metrics**:

    - **Factuality (KB CS ↑)**: Consistency of English response with the knowledge base ground truth.
    - **Consistency (Cst CS ↑)**: Consistency of responses to the same question across different languages.
    - **Geopolitical Bias (Δ CS ↓)**: The difference in responses between controlled languages (languages of the disputing parties) and non-controlled languages.
    - Based on the Concurrence Score (CS) metric—binary precision (1 if two strings are equivalent, 0 otherwise).

3. **Document Content Annotation**:

    - Function: Annotates relevance and territorial stance for each query-document pair.
    - Two-stage approach: Small-scale human annotation (5 languages, 543 pairs) $\rightarrow$ large-scale GPT-4o annotation (19k pairs, 49 languages).
    - Two dimensions (relevance and stance), achieving an agreement rate of 76% F1 between human and LLM annotations on relevance.

4. **Citation Analysis**:

    - Function: Analyzes how LLMs use the provided documents under citation formats.
    - **Inclusion rate**: The proportion of documents in a specific language within the prompt.
    - **Citation rate**: The proportion of documents in a specific language cited by the LLM.
    - A query language bias exists when the citation rate is much higher than the inclusion rate (citation rate $\gg$ inclusion rate).

### Loss & Training

BordIRLines is an **evaluation benchmark** and does not involve training. Key configurations:
- IR System: OpenAI embeddings (text-embedding-3-large) + cosine similarity, and the open-source M3-Embedding.
- LLMs: GPT-4o, GPT-4o-mini, Llama 3 (1B/3B/8B), Command-R (7B/35B).
- 10 runs per setting (10 fixed random seeds, temperature=0.5), reporting average scores and 95% confidence intervals.
- Two response formats: Direct format (multiple-choice answer) and Citation format (answer + explanation + cited document IDs).

## Key Experimental Results

### Main Results of Cross-lingual Robustness (Table)

**Factuality (KB CS ↑, English queries only):**

| Model | no_ir | qlang | rel_langs |
|------|-------|-------|-----------|
| Llama-3-8B | ~55 | ~62 | ~60 |
| Command-R-35B | ~65 | ~70 | ~68 |
| GPT-4o-mini | ~72 | ~76 | ~74 |
| GPT-4o | ~68 | ~76 | ~73 |

**Consistency (Cst CS ↑, Multilingual queries):**

| Model | no_ir | qlang | rel_langs |
|------|-------|-------|-----------|
| Command-R-35B | 64.2 | 74.3 | **78.7** |
| GPT-4o-mini | 78.6 | 71.7 | ~77 |
| GPT-4o | 79.9 | 77.2 | ~80 |

**Geopolitical Bias (Δ CS ↓, lower is better):**

| Model | no_ir | qlang | rel_langs |
|------|-------|-------|-----------|
| Command-R-35B | 28.7 | 12.2 | **5.9** |
| GPT-4o-mini | 23.6 | 71.9 | **0.9** |

### Key Findings of Citation Analysis (Table)

| Language Resource Level | Citation Rate Variance | Query Language Bias |
|-------------|-----------|-------------|
| High-resource languages | Low | Moderate |
| Low-resource languages | **High (much larger variance)** | Unstable |

### Key Findings

1. **Multilingual retrieval outperforms monolingual retrieval**: `rel_langs` comprehensively outperforms `qlang` across consistency and geopolitical bias metrics. This is the most crucial finding of the paper—gathering information from multiple languages actually improves consistency.
2. **`qlang` negatively affects consistency**: For GPT-4o-mini ($78.6 \rightarrow 71.7$) and GPT-4o ($79.9 \rightarrow 77.2$), retrieving documents only in the query language decreases consistency.
3. **RAG generally reduces geopolitical bias**: All IR modes reduce $\Delta \text{CS}$, with `rel_langs` performing the best. The bias of Command-R drops from 28.7 to 5.9.
4. **Models vary in sensitivity to RAG**: Command-R (specifically trained for RAG) is the most affected, while Llama is the least affected.
5. **Unstable citation in low-resource languages**: The variance in the document citation rate of low-resource languages is much larger than that of high-resource languages, suggesting that LLM RAG behavior on low-resource languages is more unpredictable.
6. **LLMs selectively interpret documents**: Analysis shows that LLMs sometimes selectively extract information from the same document to support their own bias.

## Highlights & Insights

1. **Counter-intuitive finding**: Previous research and concerns suggested that multilingual documents might introduce knowledge conflicts. However, this study finds that on reliable sources like Wikipedia, multilingual RAG actually improves consistency, dispelling the concern that "multilingual = more noise".
2. **Comprehensive evaluation**: Evaluating the cross-lingual robustness of RAG from three dimensions (factuality, consistency, and geopolitical bias) provides deeper insights than simple accuracy metrics.
3. **Scale of 49 languages**: Covering a large number of low-resource languages reveals unique challenges faced by low-resource languages in RAG.
4. **Added value of document annotations**: Relevance and stance annotations make BordIRLines not only a QA benchmark but also useful for evaluating IR quality.

## Limitations & Future Work

1. **Task specificity**: Territorial disputes represent a highly specific task. Whether the findings generalize to other culturally-sensitive tasks (e.g., religion, historical events) remains to be validated.
2. **Wikipedia bias**: Despite Wikipedia's neutrality guidelines, editor groups of different language versions may exhibit systematic biases.
3. **Limited IR coverage**: Only articles related to territories and disputing parties are indexed, rather than the entire Wikipedia, which artificially restricts the retrieval space.
4. **Difficulty in stance annotation**: The agreement rate between human annotations and LLM annotations on the stance dimension is relatively low, indicating that stance judgment is inherently subjective.
5. **Experimental cost**: The combination of multiple languages $\times$ multiple IR modes $\times$ multiple LLMs $\times$ 10 runs incurs extremely high API costs.

## Related Work & Insights

- BorderLines provides the query set for territorial disputes; BordIRLines extends this to the retrieval and document dimensions.
- Factual robustness studies such as CRAG (Chen et al., 2024b) focus on single interactions, whereas this study focuses on consistency across multiple cross-lingual interactions.
- Open-retrieval multilingual QA such as MKQA (Clark et al., 2020) mainly focuses on simple facts, while this study focuses on questions where answers are stance-dependent.
- Insights for RAG system design: In culturally-sensitive tasks, multilingual retrieval should be prioritized over monolingual retrieval.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — A systematic evaluation of multilingual RAG in culturally-sensitive tasks was previously missing; the definition of five IR modes and the cross-lingual robustness evaluation framework are novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Highly comprehensive, involving 49 languages, 7 LLMs, 5 IR modes, 2 IR systems, 2 response formats, and a two-stage annotation.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear formalization of task definitions, intuitive illustrations, and in-depth analysis of results.
- **Value**: ⭐⭐⭐⭐⭐ — High value as a benchmark dataset; the finding that "multilingual retrieval improves consistency" offers direct guidance for the design of RAG systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cross-Lingual Relevance Transfer for Document Retrieval](cross-lingual_relevance_transfer_for_document_retrieval.md)
- [\[ACL 2025\] MEMERAG: A Multilingual End-to-End Meta-Evaluation Benchmark for Retrieval Augmented Generation](memerag_a_multilingual_end-to-end_meta-evaluation_benchmark_for_retrieval_augmen.md)
- [\[ACL 2025\] Investigating the Robustness of Retrieval-Augmented Generation at the Query Level](investigating_the_robustness_of_retrieval-augmented_generation_at_the_query_leve.md)
- [\[ACL 2025\] Automatic Benchmark Generation from Scientific Papers via Retrieval-Augmented LLMs](automatic_benchmark_generation_from_scientific_papers_via_retrieval-augmented_ll.md)
- [\[ACL 2025\] HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation](hoh_a_dynamic_benchmark_for_evaluating_the_impact_of_outdated_information_on_ret.md)

</div>

<!-- RELATED:END -->
