---
title: >-
  [Paper Note] M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation
description: >-
  [CVPR 2026][Graph Learning][Multimodal Knowledge Graph] This paper proposes M3KG-RAG, which constructs a multi-hop multimodal knowledge graph (M3KG) via a lightweight multi-agent pipeline and introduces the GRASP mechanism for entity grounding and selective pruning. By retaining only query-relevant and answer-useful knowledge, the approach substantially improves audio-visual reasoning capabilities of MLLMs.
tags:
  - CVPR 2026
  - Graph Learning
  - Multimodal Knowledge Graph
  - Retrieval-Augmented Generation
  - Audio-Visual Reasoning
  - Graph Pruning
  - Multi-hop Reasoning
date: 2026-05-08
content_hash: bbda231e8caaca9b
---

# M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation

**Conference**: CVPR 2026
**arXiv**: [2512.20136](https://arxiv.org/abs/2512.20136)
**Code**: [Project Page](https://kuai-lab.github.io/cvpr2026m3kgrag/)
**Area**: Graph Learning
**Keywords**: Multimodal Knowledge Graph, Retrieval-Augmented Generation, Audio-Visual Reasoning, Graph Pruning, Multi-hop Reasoning

## TL;DR

This paper proposes M3KG-RAG, which constructs a multi-hop multimodal knowledge graph (M3KG) via a lightweight multi-agent pipeline and introduces the GRASP mechanism for entity grounding and selective pruning. By retaining only query-relevant and answer-useful knowledge, the approach substantially improves audio-visual reasoning capabilities of MLLMs.

## Background & Motivation

Existing multimodal RAG approaches suffer from two key bottlenecks: (1) current MMKGs primarily cover image-text modalities with limited audio-visual coverage, and most are single-hop graphs lacking multi-hop connections that capture temporal or causal dependencies; (2) similarity-based retrieval over shared embedding spaces suffers from modality gaps, failing to filter off-topic or redundant knowledge and potentially injecting noise even when relevant context is retrieved.

The core innovations of M3KG-RAG are: constructing a multi-hop knowledge graph spanning audio and visual modalities + modality-wise retrieval to bypass modality gaps + GRASP for precise retention of answer-useful subgraphs.

## Method

### Overall Architecture

Raw multimodal corpus → Three-stage agent pipeline to construct M3KG (context-enhanced triple extraction → knowledge anchoring → context-aware description refinement with self-reflection loop) → Modality-wise retrieval of candidate subgraphs → GRASP grounding + pruning → Graph-augmented MLLM generation.

### Key Designs

1. **Multi-Agent M3KG Construction Pipeline**:
    - Function: Constructs a multi-hop, cross-modal knowledge graph from raw multimodal corpora.
    - Mechanism: Rewriter enhances captions → Extractor extracts triples → Normalizer standardizes entities → Searcher queries knowledge bases for descriptions → Selector chooses context-relevant descriptions → Refiner adapts to original expressions → Inspector performs self-reflection to ensure quality.
    - Design Motivation: The pipeline requires only lightweight LLMs such as Qwen3-8B, and the self-reflection loop prevents hallucinated descriptions.

2. **GRASP (Grounded Retrieval And Selective Pruning)**:
    - Function: Ensures retrieved knowledge is both query-relevant and answer-useful.
    - Mechanism: Visual grounding (GroundingDINO detects entity presence in video frames → mask IoU threshold filtering) + Audio grounding (TAG model evaluates triple-query audio alignment) + Lightweight LLM binary-mask pruning of uninformative triples.
    - Design Motivation: Similarity-based retrieval only captures broad semantics; GRASP provides fine-grained filtering through grounding and pruning.

3. **Modality-Wise Retrieval**:
    - Function: Bypasses the modality gap in cross-modal embedding spaces.
    - Mechanism: Video queries use InternVL2 to match visual items; audio queries use CLAP to match audio items; results are then lifted to the triple level via graph links.
    - Design Motivation: Matching video queries against text-based knowledge bases in a shared embedding space frequently fails.

### Loss & Training

No model training is involved; the approach is a pure pipeline solution. M3KG construction is performed on the training splits of evaluation benchmarks using a single H100 GPU.

## Key Experimental Results

### Main Results (Model-as-Judge Scoring)

| MLLM | Method | Audio QA | Video QA | AV QA |
|------|--------|----------|----------|-------|
| Qwen2.5-Omni | None | 49.00 | 42.21 | 32.42 |
| Qwen2.5-Omni | VAT-KG | 51.30 | 43.50 | 35.44 |
| Qwen2.5-Omni | M3KG-RAG | 60.77 | 44.35 | 44.67 |

### Win-Rate Comparison (vs. VAT-KG)

| Benchmark | VAT-KG Win Rate | M3KG-RAG Win Rate |
|-----------|-----------------|-------------------|
| AudioCaps-QA | 25.6% | 74.4% |
| VCGPT | 47.6% | 52.4% |
| VALOR | 41.8% | 58.2% |

### Key Findings

- Text KGs with simple RAG frequently degrade performance (Wikidata performs worse than no retrieval in multiple settings).
- Single-hop MMKGs (VAT-KG) yield limited improvements; the multi-hop structure is critical.
- Even GPT-4o benefits from M3KG-RAG, demonstrating that external knowledge remains valuable for large-scale models.
- Each component of GRASP (grounding + pruning) independently contributes to performance gains.

## Highlights & Insights

- An end-to-end multimodal knowledge graph construction and retrieval framework covering audio, visual, and textual modalities.
- The two-stage "grounding → pruning" design of GRASP is both intuitively clean and effective.
- High-quality knowledge graphs can be constructed using only the lightweight Qwen3-8B, keeping computational cost manageable.

## Limitations & Future Work

- The modality-wise retrieval threshold $\tau$ and GRASP threshold $\eta$ require manual tuning per dataset.
- Knowledge graph construction depends on training sets; generalization to new domains requires rebuilding the graph.
- Grounding models used in GRASP (GroundingDINO/TAG) may themselves introduce errors.
- Evaluation is limited to open-ended QA and does not cover other multimodal tasks.

## Related Work & Insights

- **vs. VAT-KG**: Single-hop concept graph with simple retrieval; M3KG-RAG uses multi-hop graphs with precise GRASP filtering.
- **vs. GraphRAG/LightRAG**: Text-only graph RAG; M3KG-RAG extends to audio-visual multimodal settings.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of multi-hop multimodal knowledge graphs and GRASP is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three benchmarks, multiple MLLMs, and dual evaluation via win-rate and Model-as-Judge.
- Writing Quality: ⭐⭐⭐⭐ Architecture diagrams are clear and pipeline steps are described in detail.
- Value: ⭐⭐⭐⭐ Provides a practical knowledge graph-enhanced solution for multimodal RAG.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](../../NeurIPS2025/graph_learning/gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)
- [\[CVPR 2026\] Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs](graph2eval_multimodal_task_generation_agents.md)
- [\[ICLR 2026\] RAS: Retrieval-And-Structuring for Knowledge-Intensive LLM Generation](../../ICLR2026/graph_learning/ras_retrieval-and-structuring_for_knowledge-intensive_llm_generation.md)
- [\[CVPR 2026\] Graph-to-Frame RAG: Visual-Space Knowledge Fusion for Training-Free and Auditable Video Reasoning](graph-to-frame_rag_visual-space_knowledge_fusion_for_training-free_and_auditable.md)
- [\[CVPR 2026\] Mario: Multimodal Graph Reasoning with Large Language Models](mario_multimodal_graph_reasoning_with_large_language_models.md)

</div>

<!-- RELATED:END -->
