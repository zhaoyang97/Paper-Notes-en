---
title: >-
  [Paper Note] Any Information Is Just Worth One Single Screenshot: Unifying Search With Visualized Information Retrieval
description: >-
  [ACL 2025][Information Retrieval & RAG][Visualized Information Retrieval] This paper formally defines the Visualized Information Retrieval (Vis-IR) paradigm, which uniformly renders multimodal information into screenshots for retrieval. It constructs the VIRA dataset containing 13 million screenshots, the UniSE retrieval model family, and the MVRB benchmark, laying the foundation for unified search engines.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Visualized Information Retrieval"
  - "Screenshot Representation"
  - "Multimodal Retrieval"
  - "Unified Search"
  - "Cross-modal Embeddings"
date: 2026-05-08
content_hash: b5fabd754025eeb6
---

# Any Information Is Just Worth One Single Screenshot: Unifying Search With Visualized Information Retrieval

**Conference**: ACL 2025  
**arXiv**: [2502.11431](https://arxiv.org/abs/2502.11431)  
**Area**: Multimodal VLM / Information Retrieval  
**Keywords**: Visualized Information Retrieval, Screenshot Representation, Multimodal Retrieval, Unified Search, Cross-modal Embeddings

## TL;DR
This paper formally defines the Visualized Information Retrieval (Vis-IR) paradigm, which uniformly renders multimodal information into screenshots for retrieval. It constructs the VIRA dataset containing 13 million screenshots, the UniSE retrieval model family, and the MVRB benchmark, laying the foundation for unified search engines.

## Background & Motivation

**Background**: Significant progress has been made in text and multimodal information retrieval. With the development of vision-language models, information is increasingly presented in visual formats—webpage screenshots contain a mixture of text, images, tables, and charts. Users have started using new interaction modes such as "Circle to Search" (e.g., Google Circle to Search).

**Limitations of Prior Work**: (1) Existing retrieval systems handle either pure text or images, making it difficult to uniformly process complex documents containing mixed modalities. (2) Retrieving rendered webpage/PDF screenshots (e.g., ColPali) represents only preliminary attempts, lacking a systematic problem definition, large-scale datasets, and comprehensive benchmarks. (3) Formats like text, charts, and layouts within screenshots must be understood synergistically rather than being processed separately.

**Key Challenge**: Different types of information (text, images, tables, code) have their own optimal representation formats, but users often face mixed content in real-world searches. Utilizing visual representations (i.e., screenshots) to uniformly represent all information is conceptually highly elegant, yet it introduces the technical challenge of enabling models to comprehend the rich semantics embedded in screenshots.

**Goal**: Formally define the Vis-IR problem and construct a complete data-model-evaluation ecosystem to advance the development of this new paradigm.

**Key Insight**: Any information can be rendered into a single screenshot—text is a screenshot, webpages are screenshots, papers are screenshots, and product listings are screenshots. By adopting "screenshots" as a unified information carrier, all retrieval tasks are transformed into matching either between screenshots or between queries and screenshots.

**Core Idea**: Train general-purpose screenshot embedding models using a large-scale screenshot-annotation dataset, enabling unified support for various retrieval modes, such as "screenshot-to-screenshot," "text-to-screenshot," and "conditional search."

## Method

### Overall Architecture
Three major contributions: (1) VIRA: A large-scale training dataset containing 13 million screenshots and 20 million data samples; (2) UniSE: A general-purpose screenshot embedding model based on both CLIP and MLLM architectures; (3) MVRB: A visualized retrieval benchmark covering diverse tasks and domains.

### Key Designs

1. **VIRA Dataset Construction**:

    - **Function**: Provides large-scale, diverse training data for screenshot retrieval.
    - **Mechanism**: 13 million screenshots are collected from seven categories of sources (news websites, e-commerce platforms, Wikipedia, GitHub, arXiv papers, PDF documents, and charts). Fine-grained captions are generated for each screenshot (via metadata extraction or OCR tools), and LLMs are utilized to construct two types of Q&A data: q2s (query-to-screenshot) tuples and sq2s (screenshot + conditional query-to-screenshot) triplets. Hard negatives are also mined through text/visual similarity.
    - **Design Motivation**: The diversity of screenshot sources ensures model generalization, the granularity of captions guarantees the quality of training signals, and hard negatives boost the discriminative capability of the retrieval model.

2. **UniSE Model Family**:

    - **Function**: Maps screenshots and text queries into a unified embedding space.
    - **Mechanism**: Two architecture variants are provided. UniSE-CLIP, based on CLIP, processes screenshots through a vision encoder and text via a text encoder, combining queries through embedding addition; this offers high efficiency but limited expressive power. UniSE-MLLM, based on Qwen2-VL-2B, encodes screenshot visual tokens and query text tokens via a multimodal LLM, using the output of the [EOS] token as the embedding; this yields stronger representation capacity but higher computational overhead.
    - **Design Motivation**: The two architectures target different application scenarios—the CLIP version is suitable for large-scale online retrieval, while the MLLM version is tailored for scenarios demanding deep comprehension.

3. **Two-Stage Training Pipeline**:

    - **Function**: Incrementally improves model capability through phased training.
    - **Mechanism**: The first stage conducts contrastive pre-training on screenshot-caption pairs to enable the model to capture fine-grained screenshot semantics. The second stage fine-tunes the model on q2s and sq2s data to learn retrieval-specific matching capabilities. Both stages employ a bidirectional contrastive loss $\mathcal{L} = \mathcal{L}_{con}(e_s, e_c) + \mathcal{L}_{con}(e_c, e_s)$.
    - **Design Motivation**: Comprehending the screenshot first before learning physical retrieval matching makes progressive training more stable than training directly on retrieval task data.

### Loss & Training
A bidirectional InfoNCE contrastive loss is used, with a learnable temperature parameter $\tau$. In-batch negatives and mined hard negatives are utilized. Training is based on DeepSpeed ZeRO-2, with UniSE-CLIP trained on 64 GPUs and UniSE-MLLM trained on 32 GPUs.

## Key Experimental Results

### Main Results

| Model | MVRB Average (nDCG@10) | q2s Task | s2s Task | sq2s Task |
|------|-------------------|---------|---------|----------|
| CLIP-Large (zero-shot) | 38.2 | 42.1 | 31.5 | 35.8 |
| ColPali | 45.6 | 51.3 | 38.2 | 41.7 |
| E5-V | 43.1 | 48.5 | 36.8 | 39.4 |
| UniSE-CLIP | 58.7 | 63.2 | 52.4 | 55.1 |
| **UniSE-MLLM** | **62.3** | **67.5** | **55.8** | **58.9** |

### Ablation Study

| Configuration | MVRB Average | Description |
|------|---------|------|
| UniSE-MLLM (Full) | 62.3 | Full model |
| Stage 1 training only | 52.8 | Pre-trained but not fine-tuned on retrieval data |
| Without Hard Negatives | 58.5 | A decrease in discriminative ability due to lack of hard negatives |
| Trained on news domain only | 54.2 | Poor generalization due to single-domain training |
| Without sq2s data | 59.1 | Performance drops by several points on conditional retrieval tasks |
| CLIP architecture vs MLLM | 58.7 vs 62.3 | MLLM is stronger but 3x slower |

### Key Findings
- Existing multimodal retrievers perform poorly on Vis-IR tasks (e.g., CLIP scores only 38.2%), indicating that screenshot retrieval is far from solved.
- UniSE-MLLM beats UniSE-CLIP by 3.6 percentage points, showing a larger advantage on conditional retrieval (sq2s), because compositional queries demand deep semantic comprehension.
- Data diversity is critical—models trained only on a single domain generalize poorly.
- Hard negatives contribute a gain of approximately 4 points, proving crucial for retrieval models.

## Highlights & Insights
- The unified paradigm of "everything is a screenshot" is highly elegant and is poised to become the core concept of next-generation search engines. This aligns perfectly with the product direction of Google Circle to Search.
- The pipeline for constructing the VIRA dataset is exemplary: combining automated caption generation, LLM-assisted QA data construction, and hard negative mining creates an extensible data flywheel.
- Providing two architectural choices (CLIP vs. MLLM) reflects strong engineering design, allowing users to choose according to their latency and accuracy requirements.

## Limitations & Future Work
- Utilizing screenshots as a unified representation can lose precise text details in text documents, making OCR quality a bottleneck.
- The MVRB benchmark is currently dominated by English; multilingual screenshot retrieval is an important future direction.
- Video content has not yet been integrated into the Vis-IR framework.
- Screenshot resolution and the representation of multi-page screenshots for long documents remain key challenges in practical deployment.

## Related Work & Insights
- **vs ColPali**: ColPali only processes document screenshots, whereas Vis-IR extends the scope to all screenshot types (webpages, products, charts, etc.).
- **vs DSE (Document Screenshot Embedding)**: DSE focuses on Wikipedia documents, while VIRA far exceeds it in data diversity and scale.
- **vs CLIP/SigLIP**: General vision-language models perform poorly on information-dense visual content like screenshots, necessitating specialized training.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Formally defines the Vis-IR paradigm, constructs a complete data-model-evaluation ecosystem, and is highly pioneering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 13 million data points, two model architectures, and comprehensive benchmarks.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with three highly-structured contributions.
- **Value**: ⭐⭐⭐⭐⭐ Provides the technical foundation for next-generation search engines, offering broad industrial application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AIR-Bench: Automated Heterogeneous Information Retrieval Benchmark](air-bench_automated_heterogeneous_information_retrieval_benchmark.md)
- [\[ACL 2025\] CoIR: A Comprehensive Benchmark for Code Information Retrieval Models](coir_a_comprehensive_benchmark_for_code_information_retrieval_models.md)
- [\[ACL 2025\] Atomic LLM: A Fine-Grained Information Retrieval Evaluation Benchmark for Language Models](atomic_llm_a_fine-grained_information_retrieval_evaluation_benchmark_for_languag.md)
- [\[ACL 2025\] Logical Consistency is Vital: Neural-Symbolic Information Retrieval for Negative-Constraint Queries](logical_consistency_is_vital_neural-symbolic_information_retrieval_for_negative-.md)
- [\[ACL 2025\] HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation](hoh_a_dynamic_benchmark_for_evaluating_the_impact_of_outdated_information_on_ret.md)

</div>

<!-- RELATED:END -->
