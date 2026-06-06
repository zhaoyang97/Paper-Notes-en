---
title: >-
  [Paper Note] Knowledge Completes the Vision: A Multimodal Entity-aware Retrieval-Augmented Generation Framework for News Image Captioning
description: >-
  [AAAI 2026][Information Retrieval & RAG][News Image Captioning] This paper proposes MERGE, the first multimodal entity-aware RAG framework for news image captioning. Through three core components — an Entity-centric Mult…
tags:
  - "AAAI 2026"
  - "Information Retrieval & RAG"
  - "News Image Captioning"
  - "RAG"
  - "Entity Knowledge Base"
  - "Cross-modal Alignment"
  - "Multimodal Large Language Model"
date: 2026-05-08
content_hash: 209b14e493959b71
---

# Knowledge Completes the Vision: A Multimodal Entity-aware Retrieval-Augmented Generation Framework for News Image Captioning

**Conference**: AAAI 2026
**arXiv**: [2511.21002](https://arxiv.org/abs/2511.21002)  
**Code**: [https://github.com/youxiaoxing/MERGE](https://github.com/youxiaoxing/MERGE)  
**Area**: Information Retrieval
**Keywords**: News Image Captioning, RAG, Entity Knowledge Base, Cross-modal Alignment, Multimodal Large Language Model

## TL;DR

This paper proposes MERGE, the first multimodal entity-aware RAG framework for news image captioning. Through three core components — an Entity-centric Multimodal Knowledge Base (EMKB), Hypothetical Caption-guided Multimodal Alignment (HCMA), and Retrieval-driven Multimodal Knowledge Integration (RMKI) — MERGE achieves CIDEr +6.84 and F1 +4.14 on GoodNews, and demonstrates strong generalization with CIDEr +20.17 on the unseen Visual News benchmark.

## Background & Motivation

### State of the Field
News Image Captioning requires models to combine visual content with contextual information from news articles to generate informative descriptions. Unlike generic image captioning, which only describes visible content, this task demands precise entity recognition (persons, locations, events) and integration of deep background knowledge. Editors must analyze key elements and tailor captions to varying news contexts.

### Limitations of Prior Work
Despite advances across multiple paradigms (template filling → Transformer → CLIP-guided → MLLM), three core challenges persist:

**Incomplete Information Coverage**: Generating accurate captions often requires referencing entities not mentioned in the article. For example, Ruth Wilson may appear in an image but go unmentioned in the text, making identification impossible for existing methods. Current approaches lack effective knowledge retrieval and integration capabilities.

**Insufficient Cross-modal Alignment**: Existing methods either focus on describing visual scenes or extracting entity-dense sentences, struggling to comprehensively align visual objects with numerical details — such as associating a Toyota Tacoma with its 2011 release date.

**Imprecise Visual-Entity Association**: Mapping visual cues to correct named entities remains difficult, especially in images containing multiple persons or objects. Implicit matching methods offer limited control, and fine-tuned models generalize poorly to unseen entities. Existing RAG approaches (e.g., ConceptNet) still lack robust visual-textual integration.

### Core Idea
Construct an entity-centric knowledge base that integrates textual, visual, and structured knowledge; achieve fine-grained cross-modal alignment through multi-stage CoT reasoning; and support precise visual-entity association via dynamic retrieval.

## Method

### Overall Architecture
MERGE comprises three core components:
1. **EMKB**: Entity-centric Multimodal Knowledge Base, integrating named entities, images, and structured background knowledge
2. **HCMA**: Hypothetical Caption-guided Multimodal Alignment, employing three-stage CoT reasoning for fine-grained sentence-level alignment
3. **RMKI**: Retrieval-driven Multimodal Knowledge Integration, matching visual cues to entities and dynamically constructing a background knowledge graph

The final generation uses InstructBLIP with a 4-layer GAT to encode the knowledge graph, integrating all multimodal inputs for caption generation.

### Key Designs

#### 1. **Entity-centric Multimodal Knowledge Base (EMKB)**

EMKB construction pipeline:
- **Entity Extraction**: spaCy is used to extract named entities (celebrities, locations, artworks, landmarks, etc.) from GoodNews and NYTimes800k; the entity set is further expanded via LLMs
- **Image Collection**: Wikipedia images + up to 5 Google Search images per entity + 4 public face datasets (IMDb-WIKI, VGGFace2, CACD, IMDb-Face), capped at 5 images per entity
- **Background Knowledge Acquisition**: Entity background information is extracted from Wikipedia and IMDb, then structured into knowledge subgraphs by an LLM
- Final scale: 489,085 entities, 2,186,557 images

$$\mathbf{B} = \{(\mathbf{e}_i, \{\mathbf{I}_j\}, \mathbf{b}_i, \mathbf{G}_{sub}^i)\}_{i=1}^{N}$$

**Design Motivation**: Existing methods cannot handle entities absent from articles. EMKB provides external knowledge supplementation. Moreover, knowledge subgraphs are retrieved dynamically at inference time rather than statically appended, adapting to news-specific knowledge demands.

#### 2. **Hypothetical Caption-guided Multimodal Alignment (HCMA)**

Three-stage Chain-of-Thought reasoning:

- **Stage 1 — Hypothetical Caption Generation**: The MLLM first extracts key sentences from image $\mathbf{I}$ and article $\mathbf{T}$, then generates a hypothetical caption $\hat{\mathbf{h}}$ of no more than 30 words, serving as an anchor for subsequent alignment
- **Stage 2 — Relevant Sentence Selection**: Using the hypothetical caption and image as anchors, up to 5 relevant sentences $\mathbf{S}$ are selected from the article, balancing informativeness and efficiency
- **Stage 3 — Global Summary Generation**: A global summary $\mathbf{U}$ of no more than 100 words is generated from the full article, capturing broader cross-paragraph connections that local sentence selection may miss

**Design Motivation**: Single-stage CLIP retrieval tends to capture only local relevance. The three-stage progressive reasoning first establishes a global hypothesis, then selects local evidence, and finally supplements with a global perspective — forming a bidirectional "coarse-to-fine + fine-to-coarse" alignment strategy.

#### 3. **Retrieval-driven Multimodal Knowledge Integration (RMKI)**

RMKI operates on EMKB via two retrieval augmentation strategies:

**RAS 1: Entity Matching**
- Face images: InsightFace extracts feature vectors; cosine similarity is used to match faces in EMKB: $j^* = \arg\max_j \cos(\mathbf{x}_j, \mathbf{y})$
- Non-face images: CLIP visual encoder generates embeddings; cosine similarity retrieves the nearest image

**RAS 2: Background Knowledge Graph Construction**
1. NER: spaCy extracts named entities $\mathbf{E}_{sen}$ from relevant sentences $\mathbf{S}$
2. Relation Extraction: An LLM extracts inter-entity relations to construct a base relation graph $\mathbf{G}_{base}$
3. Subgraph Retrieval: Knowledge subgraphs are retrieved from EMKB for each entity
4. Graph Integration: Subgraphs are merged into the base graph with deduplication of nodes and edges

### Loss & Training

InstructBLIP is trained with standard cross-entropy loss:

$$\mathcal{L}_{CE} = -\sum_{i=1}^{|\mathbf{c}|} \log P(c_i | c_{<i}, \mathbf{X})$$

where $\mathbf{X} = \{\mathbf{I}, \hat{\mathbf{h}}, \mathbf{S}, \mathbf{U}, \mathbf{E}, \mathbf{G}\}$ integrates all multimodal inputs. The knowledge graph is encoded by a 4-layer GAT before being injected into the MLLM.

## Key Experimental Results

### Main Results

**GoodNews Dataset**:

| Method | BLEU-4 | CIDEr | F1-score |
|--------|--------|-------|----------|
| Tell | 6.05 | 53.80 | 20.30 |
| EAMA (MLLM) | 10.04 | 87.70 | 28.23 |
| xu2024cross | 8.49 | 83.52 | 28.26 |
| **MERGE** | **10.19** | **94.54** | **32.40** |

**NYTimes800k Dataset**:

| Method | BLEU-4 | CIDEr | F1-score |
|--------|--------|-------|----------|
| EAMA | 11.03 | 87.00 | 30.97 |
| **MERGE** | **11.47** | **88.16** | **33.83** |

**Visual News (Generalization Test, Unseen Dataset)**:

| Method | CIDEr | F1-score |
|--------|-------|----------|
| zhou2022focus | 107.60 | 23.44 |
| **MERGE** | **127.77** | **29.66** |

On Visual News, which was not used in EMKB construction, MERGE still achieves a substantial lead with CIDEr +20.17, demonstrating the generalization capability of the knowledge base.

### Ablation Study

**Component Ablation on GoodNews**:

| Configuration | CIDEr | F1-score | Notes |
|---------------|-------|----------|-------|
| InstructBLIP (w/o FT) | 24.42 | 15.17 | Zero-shot; large domain gap |
| InstructBLIP (w/ FT) | 84.80 | 29.76 | Fine-tuned baseline |
| + HCMA (3 Stage) | 86.08 | 30.02 | +1.28 CIDEr |
| + RMKI (RAS 1) | 91.52 | 32.29 | +6.72 CIDEr; entity matching is critical |
| + RMKI (RAS 1+2) | 91.36 | 32.29 | Knowledge graph is complementary |
| **MERGE (All)** | **94.54** | **32.40** | All components synergize |

Key findings:
- RAS 1 (entity matching) yields the largest gain, indicating that visual-entity alignment is the core bottleneck
- Each stage of HCMA contributes incrementally, though its standalone effect is smaller than RMKI
- Full component integration outperforms any subset, confirming synergistic interaction among components

### Key Findings

1. EMKB remains effective even on datasets not involved in its construction, demonstrating the generality of the constructed entity knowledge
2. Case studies show that MERGE correctly identifies persons not mentioned in the article (e.g., Clint Eastwood), precisely aligns numerical details (e.g., "11,232 units," "80 acres"), and distinguishes multiple individuals in the same image
3. InstructBLIP performs poorly in the zero-shot news captioning setting (CIDEr 24.42), but achieves a substantial performance jump after fine-tuning, underscoring the importance of domain adaptation

## Highlights & Insights

1. **First complete RAG framework for news image captioning** — extends retrieval-augmented generation from text-based QA to vision-language tasks, simultaneously retrieving images and knowledge graphs
2. **Large-scale EMKB construction**: 489K entities + 2.18M images + structured knowledge graphs represent a valuable resource in their own right
3. **Elegant three-stage CoT design**: hypothesis → selection → global supplement, ensuring both local precision and global contextual coverage
4. **Dual-channel entity matching (face + non-face)**: InsightFace for faces and CLIP for non-face entities — a practically strong design choice
5. The generalization results on Visual News are impressive, demonstrating that the framework does not overfit to training data

## Limitations & Future Work

- EMKB construction incurs high cost (requiring crawling from Wikipedia, IMDb, and Google Images) and may suffer from knowledge staleness
- Three-stage CoT reasoning introduces inference latency (multiple MLLM forward passes), making it unsuitable for real-time applications
- EMKB currently focuses on English-language news; multilingual extension is a natural future direction
- InsightFace-based face recognition may be unstable under cross-age or cross-appearance scenarios
- The paper does not provide detailed analysis of inference speed or computational resource consumption

## Related Work & Insights

- RAG has been widely adopted in the text domain but remains relatively novel in multimodal tasks — MERGE demonstrates how to simultaneously retrieve images and structured knowledge to enhance vision-language generation
- The entity-centric knowledge base design is generalizable to other knowledge-intensive V+L tasks (e.g., VQA, visual dialogue)
- The application of CoT in multimodal settings is an emerging trend; MERGE's three-stage design serves as a strong reference

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RobustVisRAG: Causality-Aware Vision-Based Retrieval-Augmented Generation under Visual Degradations](../../CVPR2026/information_retrieval/robustvisrag_causality-aware_vision-based_retrieval-augmented_generation_under_v.md)
- [\[ACL 2026\] Disco-RAG: Discourse-Aware Retrieval-Augmented Generation](../../ACL2026/information_retrieval/disco-rag_discourse-aware_retrieval-augmented_generation.md)
- [\[AAAI 2026\] RAGFort: Dual-Path Defense Against Proprietary Knowledge Base Extraction in Retrieval-Augmented Generation](ragfort_dual-path_defense_against_proprietary_knowledge_base_extraction_in_retri.md)
- [\[ACL 2026\] VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval](../../ACL2026/information_retrieval/visret_visualization_improves_knowledge-intensive_text-to-image_retrieval.md)
- [\[AAAI 2026\] Cog-RAG: Cognitive-Inspired Dual-Hypergraph with Theme Alignment Retrieval-Augmented Generation](cog-rag_cognitive-inspired_dual-hypergraph_with_theme_alignment_retrieval-augmen.md)

</div>

<!-- RELATED:END -->
