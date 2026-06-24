---
title: >-
  [Paper Note] mRAKL: Multilingual Retrieval-Augmented Knowledge Graph Construction for Low-Resourced Languages
description: >-
  [ACL 2025][Graph Learning][Multilingual Knowledge Graph Construction] Reframe multilingual knowledge graph construction (mKGC) as a question answering (QA) task, and propose mRAKL, a RAG-based system that leverages unstructured monolingual data as a retrieval source to overcome the scarcity of structured data in low-resource languages. The method significantly outperforms existing approaches on two low-resource languages, Tigrinya and Amharic.
tags:
  - "ACL 2025"
  - "Graph Learning"
  - "Multilingual Knowledge Graph Construction"
  - "Retrieval-Augmented Generation"
  - "Low-Resource Languages"
  - "Cross-Lingual Transfer"
  - "Knowledge Graph Completion"
date: 2026-05-08
content_hash: f67671e3da0c8527
---

# mRAKL: Multilingual Retrieval-Augmented Knowledge Graph Construction for Low-Resourced Languages

**Conference**: ACL 2025  
**arXiv**: [2507.16011](https://arxiv.org/abs/2507.16011)  
**Code**: Coming soon  
**Area**: Graph Learning  
**Keywords**: Multilingual Knowledge Graph Construction, Retrieval-Augmented Generation, Low-Resource Languages, Cross-Lingual Transfer, Knowledge Graph Completion

## TL;DR

Reframe multilingual knowledge graph construction (mKGC) as a question answering (QA) task, and propose mRAKL, a RAG-based system that leverages unstructured monolingual data as a retrieval source to overcome the scarcity of structured data in low-resource languages. The method significantly outperforms existing approaches on two low-resource languages, Tigrinya and Amharic.

## Background & Motivation

**Background**: Knowledge graphs (KGs) are crucial for downstream applications such as question answering, information retrieval, and language model enhancement. However, most KGs are incomplete, and missing information is even more severe in low-resource languages. For instance, only 0.2% of entities in Wikidata have Amharic labels.

**Limitations of Prior Work**: 
   - Existing mKGC methods (such as KGT5) rely on massive amounts of structured training data (e.g., 52M triples), which low-resource languages completely lack.
   - Cross-lingual methods based on KG embeddings assume a closed-world setup, failing to leverage open-domain natural language knowledge.
   - Pre-trained language models exhibit an extreme scarcity of parametric knowledge for low-resource languages (e.g., GPT-4 achieves only 5.83% zero-shot H@1 on Amharic).

**Key Challenge**: Low-resource languages lack structured annotated data but possess relatively more unstructured monolingual texts (such as Wikipedia articles). The key challenge lies in utilizing these unstructured resources to construct KGs.

**Goal**: Construct and complete KGs for extremely low-resource languages such as Tigrinya (3.5k triples) and Amharic (34k triples).

**Key Insight**: Convert KG triples into QA pairs (head + relation → question, tail → answer) and employ a RAG approach to retrieve relevant passages from Wikipedia to assist generation.

**Core Idea**: Translate unstructured monolingual data into KG completion capability through RAG + cross-lingual QA, mitigating the insufficiency of structured data in low-resource languages.

## Method

### Overall Architecture

mRAKL consists of two core components:
- **Retriever**: Retrieves query-related sentences from monolingual Wikipedia to serve as context.
- **Generator**: Generates the tail entity as the answer based on the retrieved context and the templated question.

### Key Designs

1. **KG-to-QA Conversion**: 

    - Manually construct question templates in four languages for 120 relations.
    - For each triple $(h, r, t)$, instantiate the head entity into the relation template to form a question, and use the tail entity as the answer.
    - Example: Triple (Surafel Dagnachew, place of birth, Ethiopia) → Question "What is Surafel Dagnachew's place of birth?"

2. **Cross-Lingual Entity Alignment**: 

    - Use language markers `[C-LAN]`, `[Q-LAN]`, and `[A-LAN]` in the input sequence to indicate the languages of the context, question, and answer, respectively.
    - Support cross-lingual link prediction: given a head + relation in one language, predict the tail in another language.
    - Format: `[C-LANt]C | [Q-LANt]Q? [A-LANt']` (where context/question language $t$ and answer language $t'$ can differ).

3. **Retriever Design**: 

    - **BM25**: Builds monolingual Wikipedia indexes for the four languages individually.
    - **LaBSE**: A multilingual sentence embedding model fine-tuned using contrastive loss (noting that LaBSE does not natively support Tigrinya).
    - **(Im)perfect Retriever**: Upper bound experiments where sentences containing the tail entity are directly searched from the Wikipedia article of the head entity.

4. **Generator Training**: 

    - Base model: AfriTeVa-base (a pre-trained T5 model containing Tigrinya and Amharic).
    - Fine-tuned with LoRA using cross-entropy loss, with beam search decoding (beam size = 10).
    - Four training setups: No-Context / Monolingual Self-Context / Multilingual Self-Context / Cross-Lingual Context.

### Loss & Training

- The generator utilizes the standard cross-entropy loss.
- No explicit negative sampling is employed.
- The LaBSE retriever is fine-tuned using contrastive loss.

## Key Experimental Results

### Parameterized Knowledge Probing (Zero-shot H@1)

| Model | Tigrinya | Amharic |
|------|----------|---------|
| mT5 | - | 0.49 |
| AfriTeVa | 0.22 | 0.61 |
| Aya | 0.67 | 1.52 |
| GPT-4 | 2.23 | 5.83 |
| AfriTeVa (finetuned) | 5.13 | 29.15 |

### Main Results: Monolingual Link Prediction

| Method | Tigrinya H@1 | Tigrinya H@10 | Amharic H@1 | Amharic H@10 |
|------|-------------|---------------|-------------|--------------|
| KGT5-No-Context | 6.91 | 28.57 | 32.58 | 52.57 |
| KGT5-Description | 5.80 | 23.44 | 32.91 | 43.32 |
| KGT5-One-Hop | 4.46 | 24.33 | 28.83 | 48.17 |
| mRAKL No-Context | 5.13 | 26.11 | 29.15 | 54.81 |
| **mRAKL Self-Context** | **11.83** | **34.59** | **41.37** | **61.87** |

### Cross-Lingual Link Prediction (BM25, H@1)

| Target Language | Amharic Context | Arabic Context | English Context | Average |
|----------|---------------|--------------|---------------|------|
| Tigrinya | 15.75 | 12.30 | 14.73 | 14.15 |
| Amharic | 38.52 | 33.58 | 38.22 | 35.27 |

### Key Findings

- **RAG Significantly Improves KGC for Low-Resource Languages**: Compared to KGT5 No-Context, mRAKL Self-Context improves Tigrinya H@1 by 4.92 percentage points and Amharic by 8.79 percentage points.
- **Structured Context is Unsuitable for Low-Resource Languages**: KGT5-Description and KGT5-One-Hop degrade performance instead, as entity descriptions and one-hop connections are intrinsically lacking in low-resource settings.
- **Cross-Lingual Transfer is Effective**: BM25 outperforms the LaBSE retriever in all settings and exceeds the no-context baselines.
- **Cognate/Related Languages are More Beneficial**: When Amharic is used as the context language for Tigrinya, H@1 reaches the highest score (15.75), which is partially due to the fact that 35.88% of tail entities share the same spelling across both languages.
- **Multilingual Training Boosts Low-Resource Languages**: Multilingual Self-Context improves Tigrinya performance by 4.69 percentage points (from 11.83 to 15.18).
- **Cultural/Geographical Relevance**: Arabic context performs better on Middle Eastern/Asian queries, while English is more advantageous for Western-related topics.

## Highlights & Insights

- **Paradigm Innovation**: Reframes KGC as a QA + RAG task, elegantly utilizing unstructured data to compensate for the lack of structured data in low-resource languages.
- **Cross-Lingual Entity Linking**: Implicitly achieves cross-lingual entity alignment by specifying different language tags in the answer position, bypassing the need for explicit alignment models.
- **Modular Design**: The retriever and generator can be optimized independently, making it easy to leverage readily available monolingual data.
- **Realistic Low-Resource Scenarios**: The 3.5k triple scale of the Tigrinya KG represents a genuinely low-resource setting, offering highly practical benchmarking value.
- **Insights on Transfer Language Selection**: Both language family-level relations and cultural/geographical relevance influence transfer effectiveness.

## Limitations & Future Work

- The data scale for Tigrinya is extremely small (3.5k triples, 272 entities), which may lead to unstable results.
- There is still considerable room for improvement in retriever (BM25/LaBSE) performance; future work could investigate retrievers specifically trained for low-resource languages.
- Question templates must be manually crafted for each language, which incurs high overhead when scaling to more languages.
- Entity coverage bias: KGs of source transfer languages lack entities specific to the target language (e.g., Eritrea-related entities for Tigrinya).
- Wikipedia data inherently contains social bias, and discrepancies across different language versions may propagate to the resulting KG.

## Related Work & Insights

- **KGT5** (Saxena et al., 2022): Pioneer work modeling KGC as a sequence-to-sequence task.
- **RAG** (Lewis et al., 2020): An effective paradigm for injecting external knowledge into LLMs.
- **AfriTeVa** (Ogundepo et al., 2022): A multilingual T5 model tailored for African languages.
- This work has direct reference value for KG construction in other low-resource languages, such as minority languages or dialects.
- Inspiration: In data-scarce scenarios, leveraging unstructured data combined with cross-lingual transfer is a highly viable path.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (The combination of KGC + RAG + QA is novel, especially for realistically low-resource languages.)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Comprehensive evaluations including various retriever comparisons, cross-lingual settings, ablation studies, and qualitative analysis.)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear motivation and well-organized experimental setup.)
- **Value**: ⭐⭐⭐⭐ (Provides practical baselines and dataset contributions for low-resource KG construction.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SimGRAG: Leveraging Similar Subgraphs for Knowledge Graphs Driven Retrieval-Augmented Generation](simgrag_leveraging_similar_subgraphs_for_knowledge_graphs_driven_retrieval-augme.md)
- [\[ACL 2025\] Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation (K-RagRec)](kg_rag_recommendation.md)
- [\[ACL 2026\] MegaRAG: Multimodal Knowledge Graph-Based Retrieval Augmented Generation](../../ACL2026/graph_learning/megarag_multimodal_knowledge_graph-based_retrieval_augmented_generation.md)
- [\[CVPR 2026\] M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation](../../CVPR2026/graph_learning/m3kg_rag_multi_hop_multimodal_knowledge_graph_enhanced_retrieval_augmented_genera.md)
- [\[NeurIPS 2025\] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](../../NeurIPS2025/graph_learning/gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)

</div>

<!-- RELATED:END -->
