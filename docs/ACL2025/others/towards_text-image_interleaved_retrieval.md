---
title: >-
  [Paper Note] Towards Text-Image Interleaved Retrieval
description: >-
  [ACL 2025][interleaved retrieval] This paper defines the new task of Text-Image Interleaved Retrieval (TIIR), constructs the first TIIR benchmark dataset based on wikiHow (155K documents, 7,654 test pairs), and proposes the Matryoshka Multimodal Embedder (MME). MME addresses the efficiency issue and semantic bias caused by excessive visual tokens in MLLMs via multi-granularity visual token compression, significantly improving retrieval performance.
tags:
  - "ACL 2025"
  - "interleaved retrieval"
  - "multimodal retrieval"
  - "visual token compression"
  - "Matryoshka embedding"
  - "MLLM"
date: 2026-05-08
content_hash: b4bff3d700e1bcd1
---

# Towards Text-Image Interleaved Retrieval

**Conference**: ACL 2025  
**arXiv**: [2502.12799](https://arxiv.org/abs/2502.12799)  
**Code**: [https://github.com/vec-ai/wikiHow-TIIR](https://github.com/vec-ai/wikiHow-TIIR)  
**Area**: Others  
**Keywords**: interleaved retrieval, multimodal retrieval, visual token compression, Matryoshka embedding, MLLM

## TL;DR
This paper defines the new task of Text-Image Interleaved Retrieval (TIIR), constructs the first TIIR benchmark dataset based on wikiHow (155K documents, 7,654 test pairs), and proposes the Matryoshka Multimodal Embedder (MME). MME addresses the efficiency issue and semantic bias caused by excessive visual tokens in MLLMs via multi-granularity visual token compression, significantly improving retrieval performance.

## Background & Motivation

**Background**: Existing multimodal information retrieval mainly targets single-image input (at most one image in the query or document), which fails to satisfy real-world scenarios like tutorials and manuals where multiple images are interleaved.

**Limitations of Prior Work**: Single-image retrieval models cannot handle interleaved content; directly stitching multiple images into a single image loses interleaved contextual information; replacing images with text captions discards visual semantics.

**Key Challenge**: Although MLLMs naturally support interleaved inputs, each image generates hundreds of visual tokens (e.g., 576 tokens). In multi-image scenarios, excessively long sequences lead to: (1) low computational efficiency; (2) visual information dominating the embedding space; (3) key semantics being lost due to truncation after exceeding the context length limit.

**Goal**: Define the TIIR task, build a benchmark dataset, and propose an effective TIIR model.

**Key Insight**: Leverage the idea of Matryoshka nested representation learning to perform multi-granularity compression on visual tokens.

**Core Idea**: Build an efficient interleaved multimodal retriever using Matryoshka-style visual token compression.

## Method

### Overall Architecture
1. **Task Definition**: Both the query and the document are text-image interleaved sequences $X = [x_1, ..., x_n]$ (where $x_i$ can be a text block or an image). The objective is to retrieve the documents most relevant to the query from the corpus.
2. **Benchmark Construction**: Build a corpus of 155K interleaved documents based on wikiHow tutorial articles + automate the interleaved query generation pipeline + manual annotation of the test set.
3. **Model**: Utilize DeepSeek-VL as the backbone to construct a DPR baseline -> propose MME visual token compression on top of it.

### Key Designs

1. **wikiHow-TIIR Dataset Construction**

    - **Document Construction**: Extract goals, step titles, and corresponding images from wikiHow tutorials to construct interleaved documents.
    - **Query Generation Pipeline** (three stages):
        - (a) Use Idefics3-8B to generate captions for document images -> Qwen2.5-72B generates text queries based on text and captions.
        - (b) Use BM25 to find the most informative sentences -> LLM selects entities/actions to transform into image captions -> rewrite query text to remove information already captured by images.
        - (c) Use FLUX.1-dev text-to-image tool -> merge generated images with rewritten text to form interleaved queries.
    - **Test Set Annotation**: Manually audit 10K pairs, filtering out invalid content, illogical contents, image-text mismatches, etc., leaving 7,654 high-quality query-document pairs.

2. **DPR Baseline**

    - Use DeepSeek-VL-1.3B as the backbone, which natively supports interleaved inputs.
    - Use [EOS] state as the sequence-level embedding.
    - InfoNCE contrastive learning loss is used for training.

3. **Matryoshka Multimodal Embedder (MME)**

    - Add an average pooling layer after the visual projection of the MLLM to compress $24 \times 24 = 576$ visual tokens into $N \times N$ tokens.
    - $N \in \{1, 2, 3, 4, 6, 8, 12, 24\}$, allowing flexible selection during inference.
    - **Three Training Strategies**:
        - Random (Rand): Randomly sample one N for each micro-batch.
        - Matryoshka Learning (MRL): Jointly train all M granularities and compute a weighted sum loss.
        - Mean Learning (Mean): Compute the average loss across all $M \times M$ combinations of query-document sizes.

### Loss & Training

- InfoNCE Contrastive Loss: $$\mathcal{L} = -\log \frac{\exp(s(X^Q, X_+^D)/\tau)}{\sum_{i=1}^N \exp(s(X^Q, X_i^D)/\tau)}$$
- Temperature $\tau = 0.05$, batch size 32, learning rate $5 \times 10^{-5}$.
- In-batch negatives + 1 random hard negative.
- Train for 3 epochs with linear warmup.

## Key Experimental Results

### Main Results

| Model | Type | Recall@5 | MRR@10 | nDCG@10 |
|------|------|----------|--------|---------|
| VISTA | Single-image Stitching | 45.06 | 33.73 | 35.22 |
| GME-Qwen2-VL-2B | Single-image Stitching | 65.85 | 51.65 | 54.06 |
| MM-Embed | Single-image Stitching | 68.73 | 53.67 | 56.37 |
| BGE-v1.5 | Text + Caption | 39.66 | 29.14 | 30.56 |
| GTE-Qwen2-7B | Text + Caption | 47.24 | 35.28 | 36.85 |
| Fine-tuned CLIP | Vector Fusion | 69.41 | 54.73 | 57.15 |
| DPR baseline | Native Interleaved | 74.79 | 60.87 | 63.28 |
| **MME (N=3)** | **Native Interleaved** | **77.40** | **63.40** | **65.91** |

- MME compared to DPR baseline: Recall@5 +2.61, MRR@10 +2.53, nDCG@10 +2.63.
- All non-interleaved models perform worse than native interleaved models, showing the necessity of modeling interleaved context.

### Ablation Study

1. **Validation of Interleaved Context Effectiveness**: Shuffling image order, shuffling image positions, or shuffling both lead to significant performance drops (Figure 4), proving that interleaved context is effectively modeled.
2. **Adaptation Strategy Effectiveness**: After adapting existing models to TIIR, they perform worse than text-only retrieval in most cases (Table 3), demonstrating that baseline adaptation strategies (stitching, caption replacement) introduce noise.
3. **Visual Document Adaptation**: GME's screenshot mode is effective (outperforming text-only) because it preserves the structural layout of interleaved information.

### Key Findings

1. **Optimal Visual Token Count**: Performance exhibits an inverted U-curve, peaking at N=3 (9 tokens per image). Too few tokens lose semantics, while too many tokens allow visual information to dominate the embedding space.
2. **Training Strategy**: Mean Learning > MRL > Random. Mean training calculated across various granularity combinations provides better generalization.
3. **Visual Dominance Analysis**: Visual and textual information are most balanced and symmetric in distribution when N=3.
4. **Text Caption Replacement**: Replacing images with text captions improves text retriever performance (BGE rises from 29.14 to 44.55), indicating that captions maintain textual retrieval capability even though they lose visual semantics. Conversely, the degradation of adapted multimodal retrievers highlights that image-stitching strategies introduce noise.

## Highlights & Insights

1. **New Task Definition**: TIIR is the first retrieval task targeting queries and documents that both contain interleaved text-image content, aligning closer with real-world RAG scenarios.
2. **Meticulous Data Construction**: Three-stage automatic generation pipeline combined with human verification balances scale and quality.
3. **Matryoshka Compression**: Ingeniously applies the idea of Matryoshka representation learning to visual token compression, offering flexible granularity adjustment during inference.
4. **In-depth Analysis**: Five research questions (RQs) are validated through extensive experiments (validity of interleaved context, adaptation strategies, visual modalities, token quantity, training strategies).

## Limitations & Future Work

1. Currently, only DeepSeek-VL-1.3B is used as the backbone; the effectiveness of larger MLLMs (e.g., 7B+) is not verified.
2. All training data is derived from wikiHow tutorials, representing a single domain; generalization to other interleaved scenarios (e.g., papers, product pages) remains to be validated.
3. Each image still needs to be encoded by ViT, resulting in relatively high computational overhead in multi-image scenarios.
4. The query generation pipeline relies on a text-to-image model (FLUX.1), whose generated image quality and realism are limited.

## Related Work & Insights

- **Matryoshka Representation Learning**: Inspires multi-granularity visual token compression.
- **ColPali / Visual Document Retrieval**: A new paradigm of document screenshot retrieval, proven effective in retaining interleaved structures (Table 3).
- **DeepSeek-VL**: An MLLM that natively supports interleaved input, serving as an ideal backbone for building the TIIR model.
- **Insights**: As RAG scenarios become increasingly complex, retrievers must transition from single-image/single-modality to multimodal interleaved retrieval, with visual token compression being a key challenge.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| **Overall Rating** | **4.2** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Learning Visual Hierarchies in Hyperbolic Space for Image Retrieval](../../ICCV2025/others/learning_visual_hierarchies_in_hyperbolic_space_for_image_retrieval.md)
- [\[ACL 2025\] ACORD: An Expert-Annotated Retrieval Dataset for Legal Contract Clause Retrieval](acord_an_expert-annotated_retrieval_dataset_for_legal_contract_drafting.md)
- [\[ACL 2025\] FRACTAL: Fine-Grained Scoring from Aggregate Text Labels](fractal_fine-grained_scoring_from_aggregate_text_labels.md)
- [\[ACL 2025\] MIR: Methodology Inspiration Retrieval for Scientific Research Problems](mir_methodology_inspiration_retrieval_for_scientific_research_problems.md)
- [\[ACL 2025\] Map&Make: Schema Guided Text to Table Generation](mapmake_schema_guided_text_to_table_generation.md)

</div>

<!-- RELATED:END -->
