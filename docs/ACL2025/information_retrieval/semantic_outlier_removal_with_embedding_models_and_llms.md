---
title: >-
  [Paper Note] Semantic Outlier Removal with Embedding Models and LLMs
description: >-
  [ACL 2025 (Industry Track)][Information Retrieval & RAG][Semantic Outlier Detection] Proposes SORE (Semantic Outlier Removal), a text cleaning method based on multilingual sentence embeddings and approximate nearest neighbor (ANN) search. It identifies core content via metadata embeddings and flags text segments that match predefined outlier categories or significantly deviate from the core content. SORE achieves extremely low computational costs while approaching LLM-level p…
tags:
  - "ACL 2025 (Industry Track)"
  - "Information Retrieval & RAG"
  - "Semantic Outlier Detection"
  - "Text Cleaning"
  - "Sentence Embeddings"
  - "Approximate Nearest Neighbor Search"
  - "Multilingual Processing"
date: 2026-05-08
content_hash: ff0dda29b3d1817b
---

# Semantic Outlier Removal with Embedding Models and LLMs

**Conference**: ACL 2025 (Industry Track)  
**arXiv**: [2506.16644](https://arxiv.org/abs/2506.16644)  
**Code**: Yes (provided in paper)  
**Area**: Information Retrieval  
**Keywords**: Semantic Outlier Detection, Text Cleaning, Sentence Embeddings, Approximate Nearest Neighbor Search, Multilingual Processing  

## TL;DR

Proposes SORE (Semantic Outlier Removal), a text cleaning method based on multilingual sentence embeddings and approximate nearest neighbor (ANN) search. It identifies core content via metadata embeddings and flags text segments that match predefined outlier categories or significantly deviate from the core content. SORE achieves extremely low computational costs while approaching LLM-level precision, and has been deployed in production to process millions of documents daily.

## Background & Motivation

**Background**: Modern text processing pipelines (such as search engines, knowledge base construction, and RAG systems) require content extraction and cleaning of webpages and documents—removing irrelevant content like navigation bars, advertisements, copyright notices, and cookie pop-ups, while retaining only the core body text. Traditional methods include template-based extraction utilizing HTML structures (e.g., readability, trafilatura) and keyword-matching filters.

**Limitations of Prior Work**: HTML structure-based methods perform poorly on non-standard pages, single-page applications (SPAs), and dynamically loaded content. Keyword filtering requires maintaining a large number of rules for each language, which scales poorly in multilingual scenarios. While LLMs excel at content understanding and extraction, their computational cost is prohibitively high—making it impractical to process every document with an LLM in a production pipeline handling millions of documents daily.

**Key Challenge**: High-quality text cleaning requires semantic understanding capabilities (distinguishing "core content" from "auxiliary information"), but LLMs with semantic understanding are too expensive. Traditional methods are inexpensive but lack semantic understanding.

**Goal**: Design a text cleaning method that is cost-efficient, multilingual-supportive, and achieves cleaning precision close to LLMs, capable of being deployed in large-scale production environments.

**Key Insight**: The authors observe that sentence embedding models (such as multilingual-e5, LaBSE, etc.) can compute semantic representations of text at extremely low costs, and "outlier content" (like ads and copyright notices) typically exhibits distinct distance differences from the "core content" in the embedding space. Exploiting the geometric properties of the embedding space for anomaly detection allows balancing semantic understanding and computational efficiency.

**Core Idea**: Anchor the core content using metadata embeddings, utilize ANN search to rapidly identify segments matching known outlier categories, and apply distance thresholds to detect unknown outliers that deviate from the core content.

## Method

### Overall Architecture

The input to SORE is a document (such as the text content of a webpage), and the output is the cleaned core content. The processing pipeline is: (1) segment the document into sentence-level spans and compute vector representations for each span using a multilingual sentence embedding model; (2) utilize embeddings of document metadata (title, description, etc.) to locate the embedding center of the core content; (3) flag outlier spans via two mechanisms—a) ANN matching against a predefined group of outlier embeddings (ads, copyright, navigation, etc.), and b) identifying spans whose distance from the core content embedding center exceeds a threshold; (4) remove the flagged outlier spans and output the cleaned text.

### Key Designs

1. **Metadata-based Core Identification**:

    - **Function**: Determine the location of the document's core content in the embedding space.
    - **Mechanism**: Most documents contain metadata (such as HTML HTML tags like title, meta description, Open Graph tags, etc.), which typically provides an accurate summary of the core content. After embedding the metadata text, the resulting vector serves as the anchor point for the "core content." Segments closer to this anchor in the embedding space are more likely to be core content, whereas those further away are more likely to be irrelevant. This method purely relies on semantic similarity without analyzing document structure.
    - **Design Motivation**: Metadata is a readily available, "free" content summary. Using it to anchor core content is both simple and reliable. Compared to letting the model heuristically determine what the core content is, metadata provides a more deterministic signal.

2. **Predefined Outlier Groups with ANN Search**:

    - **Function**: Rapidly identify text segments belonging to known outlier categories.
    - **Mechanism**: Pre-collect common types of outlier text (e.g., "Subscribe to our newsletter", "Accept cookies", "All rights reserved"), embed these template texts, and build an outlier embedding index. For each segment in a document, approximate nearest neighbor (ANN) search (such as FAISS) is used to find the nearest neighbor in the outlier index database. If the similarity exceeds a set threshold, the segment is flagged as an outlier. Leveraging multilingual embedding models, outlier templates defined in English can automatically match semantically similar expressions in other languages.
    - **Design Motivation**: Many outlier contents are highly formatted (advertisements, copyright disclaimers); although the phrasing may vary, the semantics remain highly consistent. Using ANN search in the embedding space enables ultra-fast (sub-millisecond) matching and automatically handles synonymous expressions and multilingual variants.

3. **Distance-based Unknown Outlier Detection**:

    - **Function**: Detect segments that do not fall into known outlier categories but still do not belong to the core content.
    - **Mechanism**: Predefined outlier groups cannot cover all scenarios. For segments not matched by the ANN search, SORE calculates their embedding distance to the core content anchor. If the distance is significantly larger than that of other segments in the document (i.e., an "outlier" relative to the core content), it is flagged as an outlier. Specifically, the distance threshold can be adaptively calculated based on the distribution of distances of segments within the document, such as the mean plus $k$ times the standard deviation.
    - **Design Motivation**: Relying solely on predefined categories inevitably misses some cases. Distance detection serves as a fallback mechanism to capture unknown types of outlier content. The adaptive threshold avoids the need for different configurations across different document types.

### Loss & Training

SORE is a training-free method. The sentence embedding model utilizes an existing pre-trained multilingual model (such as multilingual-e5), and the outlier index database is constructed using manually defined outlier template texts. The entire system requires no annotated data or model training.

## Key Experimental Results

### Main Results

Evaluated on HTML content extraction datasets, comparing SORE against traditional structured methods and LLM-based approaches.

| Method | Precision | Recall | F1 | Cost (Relative) |
|------|--------|-------|-----|------------|
| Readability | Medium | High | Medium | 1x |
| Trafilatura | Medium-High | Medium | Medium | 1x |
| Keyword Filtering | Low | Medium | Low | 1x |
| LLM (GPT-4) | **Highest** | **Highest** | **Highest** | 100x+ |
| **SORE** | Close to LLM | High | Close to LLM | **~2x** |

### Ablation Study

| Configuration | Precision | F1 | Description |
|------|--------|-----|------|
| Full SORE | Optimal | Optimal | Complete system |
| w/o Metadata Anchoring | Decreased | Significantly Decreased | Lacks core content reference |
| w/o ANN Outlier Matching | Medium | Moderately Decreased | High miss rate relying only on distance detection |
| w/o Distance Detection | High Precision | Decreased (High miss rate) | Can only detect known outlier types |
| Monolingual Embedding Model | Good for English, poor for others | Average Decreased | Multilingual embedding is key |
| Different Embedding Models | Minor difference | Stable | SORE is not very sensitive to embedding models |

### Key Findings

- **SORE approaches LLM performance in accuracy, but at less than 1/50th of the cost**: This is the most crucial result, demonstrating that semantic embeddings + ANN can approximate the content understanding capabilities of LLMs.
- **Multilingual capability is obtained "for free"**: Because a multilingual embedding model is used, outlier templates defined in English seamlessly match content in French, German, Japanese, etc., without requiring additional language-specific rules.
- **The metadata anchoring and ANN matching components complement each other**: The former excels at identifying unknown outliers, while the latter is proficient at rapid and precise matching of known patterns. Removing either component significantly degrades performance.
- **The system is already running in production**: Processing millions of multilingual documents daily, truly validating the reliability and efficiency of the method.

## Highlights & Insights

- **"Anomaly detection in embedding space" is a general paradigm**: SORE essentially reformulates the text cleaning problem as an anomaly detection task in embedding space. This paradigm can be extended to other text quality control tasks, such as training data deduplication, low-quality response filtering, and spam detection.
- **The design of predefined outlier groups is highly practical**: By maintaining an "outlier template library", detection capabilities can be incrementally scaled at a very low cost—simply adding a new template when a new outlier pattern is discovered. This is far simpler than retraining models.
- **Engineering value from an industrial deployment perspective**: This is a solid Industry Track paper. Instead of pursuing theoretical novelty, it identifies a highly cost-effective solution within the constraints of large-scale deployment. Real-world validation on millions of documents daily is more convincing than academic benchmarks.

## Limitations & Future Work

- It relies on document metadata to anchor the core content; its performance may be limited for documents with missing or low-quality metadata.
- The predefined outlier groups require manual maintenance; while low in cost, they may struggle to keep up with the continuous evolution of outlier content.
- The granularity of sentence segmentation impacts performance—too coarse segments mix core and outlier content, while too fine segments increase computation.
- For content with ambiguous semantic boundaries (such as "author bio"—which may be core content in some tasks but not in others), task-specific customization is required.
- Future work can explore fine-tuning embedding models with small groups of annotated data to improve cleaning effects in specific domains.

## Related Work & Insights

- **vs Trafilatura / Readability**: These methods rely on the HTML DOM structure to extract body text, which fails for non-HTML content or poorly structured pages. SORE is purely semantic-based and independent of document structure.
- **vs LLM-based Content Extraction**: LLMs achieve the best performance but at the highest cost. SORE demonstrates that in most scenarios, embeddings + ANN can achieve 90%+ of the performance of LLMs while reducing costs by more than 50x.
- **vs Traditional Anomaly Detection (such as Isolation Forest)**: Traditional methods operate on the raw feature space, whereas SORE maps the problem into a semantic space using pre-trained embedding models, yielding higher-quality feature representations.

## Rating

- **Novelty**: ⭐⭐⭐ The technical components (sentence embeddings, ANN, distance-based anomaly detection) are not new; the innovation lies in combining them cleverly to solve a real-world problem.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes comparative experiments, ablation studies, multilingual testing, and production-environment validation.
- **Writing Quality**: ⭐⭐⭐⭐ Distinct problem formulation, concise description of methods, meeting the standards of an Industry Track paper.
- **Value**: ⭐⭐⭐⭐ Directly applicable engineering value for large-scale text-processing pipelines; cost-effective solutions are in high demand in the industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Uncovering Visual-Semantic Psycholinguistic Properties from the Distributional Structure of Text Embedding Space](psycholinguistic_visual_semantic.md)
- [\[ACL 2025\] Length-Induced Embedding Collapse in PLM-based Models](length-induced_embedding_collapse_in_plm-based_models.md)
- [\[ACL 2025\] Sticking to the Mean: Detecting Sticky Tokens in Text Embedding Models](sticking_to_the_mean_detecting_sticky_tokens_in_text_embedding_models.md)
- [\[ACL 2025\] Optimized Text Embedding Models and Benchmarks for Amharic Passage Retrieval](optimized_text_embedding_models_and_benchmarks_for_amharic_passage_retrieval.md)
- [\[ACL 2025\] A Text is Worth Several Tokens: Text Embedding from LLMs Secretly Aligns Well with The Key Tokens](a_text_is_worth_several_tokens_text_embedding_from_llms_secretly_aligns_well_wit.md)

</div>

<!-- RELATED:END -->
