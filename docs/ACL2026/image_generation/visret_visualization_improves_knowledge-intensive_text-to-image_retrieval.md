---
title: >-
  [Paper Note] VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval
description: >-
  [ACL 2026][Image Generation][Text-to-image retrieval] This paper proposes Visualize-then-Retrieve (VisRet), a novel retrieval paradigm that first visualizes a text query into images via a T2I generative model and then performs retrieval within the image modality. VisRet achieves an average nDCG@30 improvement of 0.125 (CLIP) and 0.121 (E5-V) across four benchmarks, and improves downstream VQA accuracy by 15.7% on Visual-RAG-ME.
tags:
  - ACL 2026
  - Image Generation
  - Text-to-image retrieval
  - query visualization
  - cross-modal alignment
  - retrieval-augmented generation
  - modality projection
date: 2026-05-08
content_hash: 365d36b9e80eef9d
---

# VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval

**Conference**: ACL 2026
**arXiv**: [2505.20291](https://arxiv.org/abs/2505.20291)
**Code**: [GitHub](https://github.com/xiaowu0162/Visualize-then-Retrieve)
**Area**: Image Generation
**Keywords**: Text-to-image retrieval, query visualization, cross-modal alignment, retrieval-augmented generation, modality projection

## TL;DR

This paper proposes Visualize-then-Retrieve (VisRet), a novel retrieval paradigm that first visualizes a text query into images via a T2I generative model and then performs retrieval within the image modality. VisRet achieves an average nDCG@30 improvement of 0.125 (CLIP) and 0.121 (E5-V) across four benchmarks, and improves downstream VQA accuracy by 15.7% on Visual-RAG-ME.

## Background & Motivation

**Background**: Text-to-image (T2I) retrieval is a critical component in knowledge-intensive applications. Common approaches embed text queries and candidate images into a shared representation space and rank by similarity. Despite continuous advances in cross-modal embedding models (e.g., CLIP, E5-V), cross-modal similarity alignment remains a fundamental challenge.

**Limitations of Prior Work**: Cross-modal embeddings often behave as "bags of concepts," failing to capture structured visual relationships such as pose, viewpoint, and spatial layout. For instance, when querying "a bar-headed goose with wings spread," embedding models can match the species but fail to recognize subtle visual features such as wing posture and the upward-looking angle. Existing improvements (query rewriting, multi-stage reranking) remain constrained by the inherent difficulty of cross-modal similarity alignment.

**Key Challenge**: Text is inherently insufficient for exhaustively describing complex visual-spatial relationships, and cross-modal retrievers exhibit an intrinsic weakness in recognizing fine-grained visual-spatial features. Encoding all visual requirements into a text query may actually degrade retrieval performance due to embedding quality limitations.

**Goal**: To propose a retrieval paradigm that bypasses the weaknesses of cross-modal similarity matching by projecting text queries into the image modality, thereby leveraging the stronger intra-modal retrieval capability of existing retrievers.

**Key Insight**: Visualization provides a more intuitive and expressive medium than text for conveying compositional concepts (entity + pose + spatial relationship). Performing retrieval within the image modality avoids the weaknesses of cross-modal retrievers and exploits their stronger intra-modal capabilities.

**Core Idea**: Decompose T2I retrieval into two stages—"text → image modality projection" and "image → image intra-modal retrieval"—by visualizing text queries via a T2I generative model and subsequently performing image-to-image retrieval using the generated images.

## Method

### Overall Architecture

VisRet consists of two stages: (1) **Modality Projection**—an LLM transforms the original text query into a T2I instruction, which a T2I generative model uses to produce $m$ visualization images $\{v_1,\ldots,v_m\} \equiv \mathcal{T}(q)$; (2) **Intra-modal Retrieval**—each generated image is used independently for retrieval, and the resulting ranked lists are aggregated via Reciprocal Rank Fusion (RRF) to produce the final result.

### Key Designs

1. **Modality Projection**:

    - **Function**: Transforms a text query into an image query, making key visual-spatial requirements explicit.
    - **Mechanism**: Given the original query $q$, an LLM first drafts a T2I instruction $q'$ in text space, describing an image that likely satisfies the implicit visual feature requirements of $q$. A T2I generation method (e.g., Stable Diffusion) then projects $q'$ into $m$ images $\{v_1,\ldots,v_m\}$. Diversity is introduced through multiple sampling.
    - **Design Motivation**: A visualized query can simultaneously depict the required entity, pose, and viewpoint—information that, when encoded solely through text, is limited by the quality of cross-modal matching.

2. **Intra-modal Retrieval and RRF Aggregation**:

    - **Function**: Performs retrieval within the image modality and aggregates results from multiple visualizations.
    - **Mechanism**: Each generated image $v_i$ is used to independently retrieve a ranked list $\mathcal{R}(v_i, \mathcal{I})$, and the $m$ lists are fused via RRF: $\text{score}_{\text{RRF}}(r) = \sum_{i=1}^{m} \frac{1}{\lambda + \text{rank}_i(r)}$, where $\lambda$ controls the influence of low-ranked items. The top-$k$ results with the highest scores are returned.
    - **Design Motivation**: Operating entirely within the image modality avoids the weaknesses of cross-modal retrievers and leverages their stronger intra-modal capabilities. Multi-image aggregation increases query diversity.

3. **Visual-RAG-ME Benchmark Construction**:

    - **Function**: Provides a retrieval evaluation benchmark for comparing visual features across multiple entities.
    - **Mechanism**: Visual-RAG is extended with questions comparing visual features of two biologically similar entities (e.g., which has a lighter color or smoother surface). Candidate entities are identified via BM25, comparison questions are manually constructed, and retrieval labels are annotated from iNaturalist, yielding 50 high-quality queries.
    - **Design Motivation**: Existing benchmarks primarily evaluate single-entity retrieval and lack scenarios requiring cross-entity visual feature reasoning, which represents an important challenge for T2I retrieval.

### Loss & Training

VisRet is a training-free, plug-and-play method that requires no modification to retrievers or pre-computed image embedding indices. It requires only a one-time use of an LLM to generate T2I instructions and a T2I model to produce visualization images.

## Key Experimental Results

### Main Results

**nDCG@30 across four benchmarks (CLIP retriever)**

| Method | Visual-RAG | Visual-RAG-ME | INQUIRE-Rerank-Hard | COCO-Hard |
|--------|-----------|---------------|---------------------|-----------|
| Original Query | 0.385 | 0.435 | 0.412 | 0.042 |
| LLM Rewriting | 0.395 | 0.572 | 0.407 | 0.093 |
| Corpus Captioning (BLIP) | 0.271 | 0.371 | 0.401 | 0.153 |
| VISA Reranking | 0.388 | 0.457 | 0.000 | 0.000 |
| **VisRet** | **0.438** | **0.605** | **0.455** | **0.108** |

**nDCG@30 across four benchmarks (E5-V retriever)**

| Method | Visual-RAG | Visual-RAG-ME | INQUIRE-Rerank-Hard | COCO-Hard |
|--------|-----------|---------------|---------------------|-----------|
| Original Query | 0.407 | 0.486 | 0.407 | 0.178 |
| LLM Rewriting | 0.391 | 0.566 | 0.412 | 0.182 |
| **VisRet** | **0.461** | **0.622** | **0.425** | **0.205** |

### Ablation Study

**Effect of T2I generation model on Visual-RAG-ME (CLIP retriever)**

| T2I Model | N@1 | N@10 | N@30 |
|-----------|-----|------|------|
| Stable Diffusion 3.5 | 0.270 | 0.467 | 0.484 |
| FLUX.1-dev | 0.320 | 0.501 | 0.494 |
| DALL-E 3 | 0.346 | 0.554 | 0.553 |
| gpt-image-1 (high quality) | **0.460** | **0.632** | **0.605** |

**Multi-image aggregation vs. single image (CLIP retriever)**

| Benchmark | 3-image N@30 | 1-image N@30 |
|-----------|-------------|-------------|
| Visual-RAG | 0.438 | 0.425 |
| Visual-RAG-ME | 0.605 | 0.602 |

### Key Findings

- VisRet achieves an average nDCG@10 improvement of 0.109 (38%↑) with the CLIP retriever and 0.078 (23%↑) with E5-V.
- Downstream VQA accuracy improves by 3.8% on Visual-RAG (top-1 retrieval) and 15.7% on Visual-RAG-ME.
- T2I generation model quality is the critical performance bottleneck: gpt-image-1 substantially outperforms Stable Diffusion 3.5; three failure modes are identified—lack of focus, factual inaccuracies, and poor instruction following.
- Using a single visualization incurs only a marginal performance drop; the gains from multi-image aggregation stem from increased query diversity.
- Visualized queries improve retrieval but cannot substitute real images as an independent knowledge source.

## Highlights & Insights

- The perspective is novel and practical: by adopting a "visualize-then-retrieve" paradigm, the paper elegantly circumvents the fundamental difficulty of cross-modal alignment.
- The training-free, plug-and-play design requires no retraining of retrievers or modification of existing infrastructure, enabling direct use of precomputed image embedding indices.
- The Visual-RAG-ME benchmark addresses the gap in retrieval evaluation for multi-entity visual feature comparison.
- VisRet incurs lower practical latency than VISA reranking (approximately 5× faster), since VISA requires an LVLM to process top-$k$ candidates.

## Limitations & Future Work

- Performance is strongly dependent on T2I generation model quality; weaker models (e.g., Stable Diffusion) yield limited gains.
- Generated images may contain factual errors (e.g., inaccurate species appearances), degrading retrieval quality.
- Evaluation is currently focused on natural species; effectiveness in other knowledge-intensive domains (e.g., medicine, architecture) remains to be validated.
- The computational cost of T2I generation is higher than that of simple query rewriting.

## Related Work & Insights

- **vs. LLM Query Rewriting**: Query rewriting still operates in the text–image cross-modal space; VisRet fully transitions into the image modality, avoiding cross-modal weaknesses.
- **vs. VISA Reranking**: VISA relies on an LVLM to process top-$k$ candidates, with cost scaling linearly in $k$ and quality bounded by the initial retrieval; VisRet fundamentally changes the query modality.
- **vs. Corpus Captioning**: Converting images to text loses information, particularly in knowledge-intensive scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Reframes T2I retrieval as "visualization + image-to-image retrieval," offering a distinctive and elegant perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers four benchmarks, two retrievers, extensive ablation analyses, and downstream VQA evaluation.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated, the method is concise, and figures are intuitive.
- Value: ⭐⭐⭐⭐ Introduces a new paradigm for knowledge-intensive T2I retrieval; the training-free plug-and-play nature enhances practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TruthfulRAG: Resolving Factual-level Conflicts in Retrieval-Augmented Generation with Knowledge Graphs](../../AAAI2026/image_generation/truthfulrag_resolving_factual-level_conflicts_in_retrieval-augmented_generation_.md)
- [\[NeurIPS 2025\] Can Knowledge-Graph-based Retrieval Augmented Generation Really Retrieve What You Need?](../../NeurIPS2025/image_generation/can_knowledge-graph-based_retrieval_augmented_generation_really_retrieve_what_yo.md)
- [\[ICLR 2026\] Neon: Negative Extrapolation From Self-Training Improves Image Generation](../../ICLR2026/image_generation/neon_negative_extrapolation_image_generation.md)
- [\[AAAI 2026\] Hyperbolic Hierarchical Alignment Reasoning Network for Text-3D Retrieval](../../AAAI2026/image_generation/hyperbolic_hierarchical_alignment_reasoning_network_for_text-3d_retrieval.md)
- [\[CVPR 2026\] ViStoryBench: Comprehensive Benchmark Suite for Story Visualization](../../CVPR2026/image_generation/vistorybench_comprehensive_benchmark_suite_for_story_visualization.md)

</div>

<!-- RELATED:END -->
