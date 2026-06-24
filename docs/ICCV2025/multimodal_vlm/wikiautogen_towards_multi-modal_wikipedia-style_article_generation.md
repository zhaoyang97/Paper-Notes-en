---
title: >-
  [Paper Note] WikiAutoGen: Towards Multi-Modal Wikipedia-Style Article Generation
description: >-
  [ICCV2025][Multimodal VLM][Multimodal Article Generation] This paper proposes WikiAutoGen, a multi-agent framework that automatically generates high-quality multimodal Wikipedia-style articles by integrating multimodal (text and image) retrieval with a multi-perspective self-reflection mechanism, achieving an 8%–29% improvement over existing methods on the self-constructed benchmark WikiSeek.
tags:
  - "ICCV2025"
  - "Multimodal VLM"
  - "Multimodal Article Generation"
  - "Multi-Agent Framework"
  - "Self-Reflection Mechanism"
  - "Wikipedia"
  - "Knowledge Retrieval"
date: 2026-05-08
content_hash: 2c11add79b51d1e5
---

# WikiAutoGen: Towards Multi-Modal Wikipedia-Style Article Generation

**Conference**: ICCV2025  
**arXiv**: [2503.19065](https://arxiv.org/abs/2503.19065)  
**Code**: [wikiautogen.github.io](https://wikiautogen.github.io/)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Article Generation, Multi-Agent Framework, Self-Reflection Mechanism, Wikipedia, Knowledge Retrieval  

## TL;DR

This paper proposes WikiAutoGen, a multi-agent framework that automatically generates high-quality multimodal Wikipedia-style articles by integrating multimodal (text and image) retrieval with a multi-perspective self-reflection mechanism, achieving an 8%–29% improvement over existing methods on the self-constructed benchmark WikiSeek.

## Background & Motivation

Knowledge discovery and content generation are crucial components of information organization and dissemination, but traditional methods rely heavily on manual collection, structuring, and validation. With the development of LLMs, automatic Wikipedia-style article generation has emerged, with representative methods such as Storm and Co-Storm. However, existing methods suffer from two core limitations:

**Only supporting text-only generation**: They cannot retrieve and integrate multimodal content such as images, reducing the information richness and readability of the articles.

**Lack of breadth, depth, and reliability in generated content**: The information volume and credibility of the articles are insufficient to meet the demands of high-quality knowledge generation.

In addition, existing evaluation benchmarks (such as Surfer100, FreshWiki) mainly focus on text-only generation, lacking evaluation standards for multimodal knowledge generation, and mostly cover simple topics, which do not fully test the model's capabilities on obscure or difficult topics.

## Method

### WikiAutoGen Overall Architecture

WikiAutoGen is a multi-agent collaborative framework containing four core modules that work together in a pipeline:

#### 1. Outline Proposal Module

- For **textual topics**: The LLM analyzes the input, identifies subtopics, and generates a structured outline.
- For **visual topics**: Metadata is obtained using Google Vision Search, and NER is applied to extract the Top-10 high-frequency entities as query keywords, which are combined with the original topic to generate the outline.
- For **mixed text-image topics**: Subtopics extracted from both modalities are fused, and the LLM integrates them to generate a unified outline.

#### 2. Textual Article Writing Module

This module consists of three sub-components:

- **Persona Generator**: Based on the outline draft, the LLM generates $n$ different personas related to the topic. Each persona acts as an independent agent with access to external search tools.
- **Multi-agent Knowledge Exploration**: A fixed "questioner" agent traverses the outline to pose questions, and the $n$ persona agents search the internet to retrieve information, share discoveries, and discuss. During this process, they interact with the multi-perspective self-reflection module to obtain feedback on reliability, consistency, etc., from the "writer" perspective.
- **Article Generation**: The LLM writing agent summarizes the gathered knowledge and generates an initial draft of the article. Subsequently, each paragraph is fed into the self-reflection module for iterative refinement.

#### 3. Multi-Perspective Self-Reflection Module

This is one of the core innovations of this work. It assesses article quality from **seven key dimensions** (reliability, engagement, information richness, coherence, readability, consistency, and usefulness) and designs **four evaluation perspectives**:

- **Supervisor Perspective**: Evaluates whether the content fully addresses the questions, examines the depth, breadth, and coherence of the article, and assesses the effectiveness of the multi-agent discussion.
- **Writer Perspective**: Focuses on the knowledge exploration and article generation phases, assessing coherence, engagement, factual accuracy, and logic consistency while providing specific suggestions for improvement (such as reordering sentences, adding transition words, and simplifying complex concepts).
- **Reader Perspective**: Evaluates the readability, engagement, and usefulness of the image placements and descriptions, ensuring that visual content is effectively integrated into the reading experience.
- **Editor Perspective**: Examines the consistency and alignment between images and text descriptions, offering suggestions to adjust image captions, placements, or textual explanations.

#### 4. Multimodal Article Writing Module

After text generation is completed, visual content is integrated through the following steps:

- **Image Placement Proposal**: An LLM agent proposes image placement schemes and corresponding descriptions at appropriate locations within the article, which are then evaluated by the self-reflection module from the reader's perspective.
- **Image Retrieval**: Image searches are conducted based on multiple sources (general search engines, Wikipedia, and websites within references).
- **Image Selection**: The CLIP model is first used to rank retrieved images based on semantic similarity to select the Top-3 candidates, and then a multimodal model is employed to further evaluate and select the most appropriate image.
- **Article Polishing**: After integrating images into the article, a multimodal model is utilized to revise the entire text to enhance cross-modal coherence, followed by further refinement based on self-reflection feedback from the editor's perspective.

### WikiSeek Benchmark

WikiSeek is the multimodal evaluation benchmark proposed in this work, featuring the following characteristics:

- Topics are selected from the WikiWeb2M dataset (containing approximately 2 million English Wikipedia articles).
- Focuses on obscure topics that are under-covered on Wikipedia, categorized into three difficulty levels based on the character count of the main content: hard (300–500 characters), very hard (100–300 characters), and extremely hard (<100 characters).
- Consists of 300 topics, with 100 in each of the three difficulty levels, supporting three input formats: text-only, image-only, and mixed text-image.
- Underwent rigorous manual filtering to exclude topics that are too generic (e.g., "1997 in Japan") or sematically ambiguous.

### Implementation Details

- Built with the DSPy framework + GPT-4o / GPT-4o-mini / GPT-o3-mini.
- The self-reflection module uses GPT-o3-mini (for strong reasoning capabilities), multimodal knowledge exploration uses GPT-4o, and other tasks utilize GPT-4o-mini.
- Real-time web retrieval is performed via the Serper API, returning up to 5 web pages per query.
- Temperature 1.0, top_p 0.9.

## Key Experimental Results

### Text Quality Evaluation (9-Dimensional Evaluation)

| Method | Textual Topics | Visual Topics | Mixed Topics |
|------|---------|---------|---------|
| oRAG | 60.71 | 48.38 | 58.76 |
| Storm | 65.26 | 43.57 | 61.12 |
| Co-Storm | 69.96 | 44.39 | 63.77 |
| OmniThink | 63.98 | 41.39 | 58.67 |
| **WikiAutoGen** | **78.73** | **77.49** | **78.82** |

- Textual topics improvement: +8.8 (vs Co-Storm)
- Visual topics improvement: +29.1 (vs oRAG), representing the most significant advantage
- Mixed topics improvement: +15.1 (vs Co-Storm)

### Image Quality Evaluation (4-Dimensional Evaluation)

| Method | Textual Topics | Visual Topics | Mixed Topics |
|------|---------|---------|---------|
| oRAG | 57.28 | 54.32 | 58.85 |
| WikiAutoGen | **68.99** | **68.78** | **70.98** |

Image quality is improved by 11%–14% across all input modalities.

### Ablation Study (Visual Topic Input)

| Configuration | Average Score |
|------|-------|
| W/o any module | 48.78 |
| + Multi-agent | 66.56 (+17.78) |
| + Outline Proposal | 72.88 (+24.10) |
| + Self-Reflection | 71.60 (+22.82) |
| All modules | **77.49** |

The outline proposal contributes the most to content structuring (+24.10), while self-reflection is crucial for quality refinement (+22.82), with all three components being complementary.

### Comparison with Commercial Deep Research

| Method | Average Score | Time Consumed |
|------|-------|------|
| OpenAI | 93.06 | ~30 min |
| Grok | 85.06 | ~10 min |
| Google | 81.91 | ~12 min |
| **WikiAutoGen** | **89.01** | **~8 min** |

WikiAutoGen approaches OpenAI's performance, outperforms Grok and Google, and is the fastest (3.75× faster than OpenAI).

### Human Evaluation

A pairwise comparison was conducted on 100 textual topics via AMT:
- 97.7% of the participants agreed that adding images enhances the understanding of the topics.
- WikiAutoGen consistently outperforms Storm and OmniThink in readability, engagement, information richness, and overall preference.

## Highlights & Insights

1. **The first multimodal Wikipedia article generation system**: It breaks through the limitation of existing text-only methods, truly realizing automated knowledge generation with rich illustrations.
2. **Exquisite multi-perspective self-reflection design**: A matrix-style reflection of four roles (Supervisor/Writer/Reader/Editor) × seven evaluation dimensions mimics the real collaborative writing process of humans, with each role intervening at different stages.
3. **The design of the difficulty-tiered benchmark is worth emulating**: WikiSeek defines difficulty based on the level of Wikipedia coverage, using "obscure topics" as the true challenge to prevent models from simply copying existing high-quality Wikipedia entries.
4. **Significant advantage in visual-topic scenarios** (+29.1 points) indicates the immense value of multimodal retrieval when inputs are images only—whereas baseline methods barely know how to generate articles starting from an image.
5. **Outstanding practical efficiency**: It takes 8 minutes to generate an article, which is 3.75× faster than OpenAI Deep Research while maintaining comparable quality.

## Limitations & Future Work

1. **Reliance on closed-source models**: Core components rely on GPT-4o/o3-mini, which is costly and hinders reproducibility; the performance of open-source LLMs has not been explored.
2. **Evaluations are predominantly GPT-4o-based**: The bias representing the "LLM-as-a-judge" approach is not fully discussed. Although the appendix supplements evaluations with Gemini and Prometheus2, the scale of human evaluation remains limited.
3. **Limitations in image selection process**: The two-stage selection of CLIP → multimodal model might filter out images that are semantically more relevant but mismatch visually; image quality (resolution, copyright) is not fully considered.
4. **Questionable scalability**: The complexity of the multi-agent + multi-round self-reflection pipeline is high, requiring a large number of API calls per article, leading to considerable costs for batch generation.
5. **Absence of factual validation**: Self-reflection focuses heavily on text quality dimensions, lacking an external validation mechanism for factual accuracy (such as cross-checking with knowledge graphs).
6. **Small scale of the WikiSeek benchmark** (300 topics), which only covers English Wikipedia.

## Related Work & Insights

- **Storm / Co-Storm** (Shao et al., 2024): Pioneers in automated Wikipedia article generation, upon which WikiAutoGen builds by introducing multimodal capabilities and finer-grained self-reflection.
- **OmniThink** (Xi et al., 2025): Enhances article quality through iterative expansion and reflection, simulating human slow-thinking.
- **Self-RAG** (Asai et al., 2023): A representative of self-reflective retrieval-augmented generation, which inspired the oRAG baseline.
- **DSPy** (Khattab et al., 2024): A modular programming framework, which WikiAutoGen utilizes to construct multi-agent pipelines.
- The design concept of multi-perspective self-reflection can be extended to other long-text generation tasks (such as industry research reports, review papers, and technical documentation).

## Rating

- Novelty: ⭐⭐⭐⭐ — The first multimodal Wikipedia article generation system, featuring an innovative multi-perspective self-reflection mechanism.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive in dimensions including text, image, ablation, human evaluation, and commercial comparison.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-defined modules, and rich diagrams.
- Value: ⭐⭐⭐⭐ — Multimodal knowledge generation is an important direction, and the framework design offers good engineering reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[CVPR 2025\] MARTEN: Visual Question Answering with Mask Generation for Multi-Modal Document Understanding](../../CVPR2025/multimodal_vlm/marten_visual_question_answering_with_mask_generation_for_multi-modal_document_u.md)
- [\[CVPR 2026\] Wan-Weaver: Interleaved Multi-modal Generation via Decoupled Training](../../CVPR2026/multimodal_vlm/wan-weaver_interleaved_multi-modal_generation_via_decoupled_training.md)
- [\[NeurIPS 2025\] mmWalk: Towards Multi-modal Multi-view Walking Assistance](../../NeurIPS2025/multimodal_vlm/mmwalk_towards_multi-modal_multi-view_walking_assistance.md)
- [\[ICLR 2026\] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](../../ICLR2026/multimodal_vlm/multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)

</div>

<!-- RELATED:END -->
