---
title: >-
  [Paper Note] MAVIS: A Benchmark for Multimodal Source Attribution in Long-form Visual Question Answering
description: >-
  [AAAI 2026][source attribution] MAVIS is the first benchmark for evaluating multimodal source attribution systems, comprising 157K visual QA instances with fact-level citations to multimodal documents per answer, along with automatic evaluation metrics across three dimensions: informativeness, groundedness, and fluency.
tags:
  - AAAI 2026
  - source attribution
  - multimodal RAG
  - visual question answering
  - citation generation
  - reliability evaluation
date: 2026-05-08
content_hash: ad397e2f429b3602
---

# MAVIS: A Benchmark for Multimodal Source Attribution in Long-form Visual Question Answering

**Conference**: AAAI 2026
**arXiv**: [2511.12142](https://arxiv.org/abs/2511.12142)
**Code**: Available
**Area**: Information Retrieval
**Keywords**: source attribution, multimodal RAG, visual question answering, citation generation, reliability evaluation

## TL;DR
MAVIS is the first benchmark for evaluating multimodal source attribution systems, comprising 157K visual QA instances with fact-level citations to multimodal documents per answer, along with automatic evaluation metrics across three dimensions: informativeness, groundedness, and fluency.

## Background & Motivation

**Background**: Source attribution enhances the verifiability of AI-generated answers by attaching reference sources. Existing work focuses predominantly on text-only settings, such as WebGPT and ALCE.

**Limitations of Prior Work**: (1) Existing source attribution research almost entirely neglects multimodality — real-world evidence sources include both text and images; (2) LVLMs relying on multimodal RAG exhibit noticeably weaker groundedness for image documents compared to text documents; (3) No standardized benchmark or evaluation metrics exist for multimodal source attribution.

**Key Challenge**: While multimodal information improves answer informativeness and fluency, citation accuracy for image sources is paradoxically worse — the multimodal setting amplifies this gap.

**Goal**: Construct the first multimodal source attribution benchmark, providing evaluation metrics and baseline analyses.

**Key Insight**: Starting from users' visual questions, the work evaluates whether systems can understand visual intent, retrieve multimodal evidence, and generate long-form answers with accurate citations.

**Core Idea**: Construct a large-scale multimodal VQA dataset in which each answer is annotated with fact-level citations pointing to multimodal documents, and design automatic evaluation metrics across three dimensions.

## Method

### Overall Architecture
The MAVIS evaluation pipeline: given a visual question → retrieve relevant text/image documents from a multimodal document corpus → an LVLM generates a long-form answer with citations → evaluate across informativeness, groundedness, and fluency.

### Key Designs

1. **MAVIS Dataset Construction**:

    - **Function**: Provide large-scale annotated data for evaluating multimodal source attribution.
    - **Mechanism**: 157K visual QA instances are collected, with each answer annotated at the fact level — specifying which document (text or image) each sentence cites. Multi-round review is applied to ensure annotation quality. The dataset covers diverse visual question types and difficulty levels.
    - **Design Motivation**: Systematic evaluation of multimodal source attribution is impossible without annotated data.

2. **Three-Dimensional Automatic Evaluation Metrics**:

    - **Function**: Automatically assess the output quality of source attribution systems.
    - **Mechanism**: (1) *Informativeness*: whether the answer contains sufficient information relevant to the question; (2) *Groundedness*: whether each factual claim is genuinely supported by the cited document; (3) *Fluency*: the linguistic quality of the answer and the naturalness of citation integration. Each metric is computed independently and validated against human judgments with strong correlation.
    - **Design Motivation**: A single metric cannot comprehensively reflect source attribution quality — high informativeness with low reliability is actively harmful.

3. **Multimodal RAG Baseline Analysis**:

    - **Function**: Provide systematic baseline comparisons.
    - **Mechanism**: Single-modal and multimodal RAG are compared across LVLMs; the informativeness–groundedness trade-off under different prompting strategies is analyzed. The importance of mitigating contextual bias in interpreting image documents is highlighted.
    - **Design Motivation**: Understanding the capabilities and limitations of existing methods guides future research directions.

### Loss & Training
MAVIS is a benchmark rather than a training methodology. Baseline models follow standard LVLM + RAG pipelines.

## Key Experimental Results

### Main Results

| Setting | Informativeness↑ | Groundedness↑ | Fluency↑ | Notes |
|---|---|---|---|---|
| Multimodal RAG | Highest | Lower | Highest | Rich information but weak image citation |
| Text-only RAG | Lower | Higher | Lower | Text citations more reliable |
| No RAG | Lowest | — | Moderate | Model knowledge only |

### Ablation Study

| Configuration | Informativeness | Groundedness | Notes |
|---|---|---|---|
| Image + Text documents | Highest | Lowest | Multimodal setting amplifies the gap |
| Text documents only | Lower | Higher | Text citations more accurate |
| Different prompting | Trade-off present | Trade-off present | Informativeness↑ leads to Groundedness↓ |

### Key Findings
- Multimodal RAG produces richer and more fluent answers, but groundedness for image documents is substantially weaker than for text documents.
- A trade-off exists between informativeness and groundedness — different prompting strategies improve one dimension at the expense of the other.
- Contextual bias is the primary cause of unreliable image document citations — models tend to exploit surface-level visual features rather than deep semantic content.

## Highlights & Insights
- **First multimodal source attribution benchmark**: Fills a critical gap in the field and provides a standardized evaluation platform for future research.
- **Three-dimensional metric design**: Decomposing evaluation into independent dimensions of informativeness, groundedness, and fluency offers greater diagnostic value than a single quality score.
- **Informativeness–groundedness trade-off**: This finding directly informs RAG system design — application scenarios must explicitly prioritize one dimension over the other.

## Limitations & Future Work
- Dataset construction relies on human annotation, making large-scale expansion costly.
- Although automatic metrics correlate strongly with human judgments, evaluation of complex reasoning chains remains insufficient.
- Only a limited number of LVLMs are evaluated; more comprehensive model comparisons remain to be conducted.
- Fine-tuning strategies for specifically improving image source attribution are not explored.

## Related Work & Insights
- **vs. ALCE (text source attribution)**: ALCE considers only text documents; MAVIS extends attribution to multimodal documents, posing greater challenges.
- **vs. WebGPT**: WebGPT allows the model to autonomously search and cite; MAVIS provides retrieved documents via a RAG framework.
- **vs. MMQA**: Multimodal QA datasets that do not require source attribution; MAVIS mandates that every factual claim be accompanied by a citation.

## Rating
- Novelty: ⭐⭐⭐⭐ First multimodal source attribution benchmark
- Experimental Thoroughness: ⭐⭐⭐⭐ 157K samples + multiple baselines + detailed analysis
- Writing Quality: ⭐⭐⭐⭐ Clear problem formulation and experimental analysis
- Value: ⭐⭐⭐⭐⭐ Significant contribution to reliable AI and multimodal research

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RAVENEA: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding](../../ICLR2026/information_retrieval/ravenea_a_benchmark_for_multimodal_retrieval-augmented_visual_culture_understand.md)
- [\[AAAI 2026\] N2N-GQA: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using LLMs](n2n-gqa_noise-to-narrative_for_graph-based_table-text_question_answering_using_l.md)
- [\[AAAI 2026\] REAP: Enhancing RAG with Recursive Evaluation and Adaptive Planning for Multi-Hop Question Answering](reap_enhancing_rag_with_recursive_evaluation_and_adaptive_planning_for_multi-hop.md)
- [\[CVPR 2026\] CC-VQA: Conflict- and Correlation-Aware Method for Mitigating Knowledge Conflict in Knowledge-Based Visual Question Answering](../../CVPR2026/information_retrieval/cc-vqa_conflict-_and_correlation-aware_method_for_mitigating_knowledge_conflict_.md)
- [\[NeurIPS 2025\] Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering](../../NeurIPS2025/information_retrieval/benchmarking_retrievalaugmented_multimodal_generation_for_do.md)

</div>

<!-- RELATED:END -->
