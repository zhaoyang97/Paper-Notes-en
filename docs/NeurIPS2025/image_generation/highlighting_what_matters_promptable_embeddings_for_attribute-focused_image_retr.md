---
title: >-
  [Paper Note] Highlighting What Matters: Promptable Embeddings for Attribute-Focused Image Retrieval
description: >-
  [NeurIPS 2025][Image Generation][attribute-focused retrieval] This paper proposes Promptable Embeddings, a method that highlights target visual attributes at retrieval time to improve attribute-focused text-to-image retrieval, and introduces the COCO-Facet benchmark dataset.
tags:
  - NeurIPS 2025
  - Image Generation
  - attribute-focused retrieval
  - promptable image embeddings
  - CLIP
  - multimodal large language models
  - text-to-image retrieval
date: 2026-05-08
content_hash: 3877eee8d00c8b66
---

# Highlighting What Matters: Promptable Embeddings for Attribute-Focused Image Retrieval

**Conference**: NeurIPS 2025
**arXiv**: [2505.15877](https://arxiv.org/abs/2505.15877)
**Authors**: Siting Li, Xiang Gao, Simon Shaolei Du
**Code**: N/A
**Area**: Computer Vision / Image Retrieval
**Keywords**: attribute-focused retrieval, promptable image embeddings, CLIP, multimodal large language models, text-to-image retrieval

## TL;DR

This paper proposes Promptable Embeddings, a method that highlights target visual attributes at retrieval time to improve attribute-focused text-to-image retrieval, and introduces the COCO-Facet benchmark dataset.

## Background & Motivation

In text-to-image (T2I) retrieval, an ideal retrieval system should attend to specific visual attributes—such as color, material, shape, and action—as dictated by the query. However, existing approaches suffer from notable shortcomings:

**Limitations of CLIP-based retrievers**: Widely adopted for their efficiency and zero-shot capability, CLIP image embeddings emphasize global semantics and salient objects, leading to poor and uneven performance on attribute-focused queries.

**Limitations of MLLM-based retrievers**: Even stronger retrievers built on multimodal large language models (MLLMs), despite operating in higher-dimensional embedding spaces, still struggle with attribute-focused queries.

**Key Challenge**: Using a single general-purpose image embedding for attribute-focused retrieval is inherently suboptimal—a fixed embedding cannot simultaneously serve all possible attribute queries optimally.

Core hypothesis: Dynamically conditioning image embeddings on the query to "highlight" relevant attributes can substantially improve retrieval performance.

## Method

### Overall Architecture

The method consists of three core components:

1. **COCO-Facet Benchmark Construction**: An attribute-focused retrieval evaluation benchmark built upon the COCO dataset.
2. **Promptable Embedding Generation Pipeline**: An MLLM-based pipeline for generating attribute-sensitive image embeddings.
3. **Acceleration Strategies**: Two acceleration schemes targeting practical deployment requirements.

### Key Designs

#### 1. COCO-Facet Benchmark Dataset

Built on top of COCO, the benchmark contains 9,112 queries covering diverse visual attributes, including:
- Color and appearance
- Material and texture
- Shape and size
- Spatial relations
- Action and state
- Quantity and counting

Each query targets a specific visual attribute, requiring the retrieval system to focus on fine-grained image details rather than global semantics.

#### 2. Promptable Image Embeddings

**Core Idea**: Rather than relying on a generic image embedding, the method generates attribute-specific prompts conditioned on the query type, guiding the MLLM retriever to produce embeddings that emphasize the relevant attribute.

**Mechanism**:
1. Given a query text, identify the attribute type it focuses on.
2. Construct an attribute prompt to direct the image encoder's attention toward that attribute.
3. Inject the prompt into the MLLM retriever to produce the promptable embedding.
4. Compute similarity between the promptable embedding and the query embedding for retrieval.

The pipeline generalizes well across:
- Different query types
- Different image galleries
- Different base retriever architectures

#### 3. Acceleration Strategies

Two acceleration schemes are provided to enhance practical usability:

**Strategy 1: Pre-computed Promptable Embeddings**
- **Applicable when**: prompts are predefined (i.e., a finite set of attribute types is known in advance).
- **Approach**: Offline pre-computation of all image embeddings under each attribute prompt.
- **Effect**: +15% Recall@5.
- **Trade-off**: Storage cost scales linearly with the number of attribute types.

**Strategy 2: Linear Approximation**
- **Applicable when**: prompts are only available at inference time.
- **Approach**: A learned linear transformation that approximates the conversion from generic embeddings to promptable embeddings.
- **Effect**: +8% Recall@5.
- **Advantage**: No need to re-run the MLLM; negligible computational overhead.

### Loss & Training

Training of the linear approximation module:
- Existing promptable embeddings serve as training targets for the linear mapping.
- Loss is defined as MSE or cosine similarity loss in the embedding space.
- Generalizes well and can be trained on a small number of samples.

## Key Experimental Results

### Main Results

#### Evaluation of Existing Retrievers on COCO-Facet

| Retriever Type | Model | Recall@5 | Attribute Balance | Dimensionality |
|---|---|---|---|---|
| CLIP-based | CLIP ViT-B/32 | Baseline | Poor, uneven | 512 |
| CLIP-based | CLIP ViT-L/14 | Slightly above baseline | Poor, uneven | 768 |
| MLLM-based | Multimodal LLM retriever | Relatively better | Still uneven | Higher |
| Ours | Pre-computed promptable embeddings | **Baseline +15%** | Significantly improved | Same as MLLM |
| Ours | Linear approximation | **Baseline +8%** | Significantly improved | Same as MLLM |

Key finding: CLIP-based models exhibit large performance disparities across attribute types—performing reasonably well on subject/object queries, but poorly on color, texture, and spatial relation queries.

#### Per-Attribute Performance Breakdown

| Attribute Type | CLIP Baseline | Promptable Embeddings | Improvement |
|---|---|---|---|
| Subject / Object | High | Slight gain | Small |
| Color / Appearance | Low | Substantial gain | Large |
| Material / Texture | Low | Substantial gain | Large |
| Spatial Relations | Lowest | Noticeable gain | Medium |
| Action / State | Medium | Noticeable gain | Medium |

### Ablation Study

#### Trade-off Analysis of Acceleration Strategies

| Strategy | Inference Speed | Recall@5 Gain | Storage Cost | Applicable Scenario |
|---|---|---|---|---|
| Full promptable embeddings | Slowest | Highest | Low | Research / prototyping |
| Pre-computed embeddings | Fast | +15% | High ($N$ copies) | Predefined attributes |
| Linear approximation | Fastest | +8% | Low | Online inference |

#### Cross-Architecture Generalization

The promptable embedding method proves effective across different base retriever architectures, validating the generality of the approach.

### Key Findings

1. General-purpose image embeddings exhibit systematic deficiencies in attribute-focused retrieval.
2. Even stronger MLLM-based retrievers are constrained by their reliance on a single generic embedding.
3. Promptable embeddings effectively address this limitation and generalize across models.
4. Reasonable performance–efficiency trade-offs are achievable in practice.

## Highlights & Insights

1. **Precise problem formulation**: The authors accurately identify the core bottleneck in attribute-focused retrieval—inadequate representation of fine-grained attributes in generic embeddings.
2. **High-quality benchmark**: COCO-Facet provides 9,112 queries spanning diverse attribute types, offering lasting value to the research community.
3. **Simple yet effective method**: Significant performance gains are achieved through prompt engineering alone, without retraining the underlying model.
4. **Deployment-ready acceleration**: The pre-computation and linear approximation strategies make the method practically viable.
5. **Cross-architecture generalization**: The method is not tied to any specific model, ensuring broad applicability.

## Limitations & Future Work

1. **MLLM dependency**: Generating promptable embeddings requires MLLM-scale models, incurring higher computational cost than pure CLIP-based approaches.
2. **Predefined attribute types**: The pre-computation acceleration strategy requires a fixed attribute vocabulary, making it less suited for fully open-ended queries.
3. **Performance gap of linear approximation**: A ~7% gap remains relative to the full promptable embeddings; stronger approximation methods deserve exploration.
4. **Single-dataset evaluation**: Validation is primarily conducted on COCO-Facet; broader evaluation across additional datasets is needed.
5. **Composite attribute queries**: Handling queries that simultaneously specify multiple attributes (e.g., "red and metallic") remains underexplored.
6. **Scope**: The 27-page paper includes 6 figures; while experimental detail is thorough, it also reflects the large number of design choices involved.

## Related Work & Insights

- **CLIP** (Radford et al., 2021): The foundational text–image alignment paradigm; this paper exposes its limitations in attribute representation.
- **MLLM-based retrievers**: Demonstrate that even larger models face the same attribute representation bottleneck when relying on a single generic embedding.
- **Prompt engineering**: This work innovatively applies prompting concepts to the retrieval setting.
- **Fine-grained retrieval**: Complements prior work in fine-grained image retrieval while addressing a broader range of attribute types.

## Rating

- **Novelty**: ★★★★☆ — The concept of promptable embeddings is original; COCO-Facet is a valuable contribution.
- **Theoretical Depth**: ★★★☆☆ — Primarily empirical; theoretical analysis is limited.
- **Experimental Thoroughness**: ★★★★☆ — 27 pages with 6 figures; comprehensive experiments, though validation is confined to COCO.
- **Value**: ★★★★☆ — Acceleration strategies make the method practically deployable.
- **Writing Quality**: ★★★★☆ — Problem motivation is clear; method description is complete.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Can Knowledge-Graph-based Retrieval Augmented Generation Really Retrieve What You Need?](can_knowledge-graph-based_retrieval_augmented_generation_really_retrieve_what_yo.md)
- [\[NeurIPS 2025\] Instance-Level Composed Image Retrieval](instance-level_composed_image_retrieval.md)
- [\[NeurIPS 2025\] GenIR: Generative Visual Feedback for Mental Image Retrieval](genir_generative_visual_feedback_for_mental_image_retrieval.md)
- [\[NeurIPS 2025\] On the Emergence of Linear Analogies in Word Embeddings](on_the_emergence_of_linear_analogies_in_word_embeddings.md)
- [\[NeurIPS 2025\] What We Don't C: Manifold Disentanglement for Structured Discovery](what_we_dont_c_manifold_disentanglement_for_structured_discovery.md)

<!-- RELATED:END -->
