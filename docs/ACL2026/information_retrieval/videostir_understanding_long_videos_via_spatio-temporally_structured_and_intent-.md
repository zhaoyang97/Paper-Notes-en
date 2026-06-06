---
title: >-
  [Paper Note] VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG
description: >-
  [ACL 2026][Information Retrieval & RAG][long video understanding] VideoStir proposes a structured and intent-aware RAG framework for long video understanding. By modeling videos as spatio-temporal graphs for multi-hop cl…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "long video understanding"
  - "retrieval-augmented generation"
  - "spatio-temporal graph structure"
  - "intent-aware retrieval"
  - "multi-hop reasoning"
date: 2026-05-08
content_hash: 15cecffae76bbe66
---

# VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG

**Conference**: ACL 2026
**arXiv**: [2604.05418](https://arxiv.org/abs/2604.05418)  
**Code**: [https://github.com/RomGai/VideoStir](https://github.com/RomGai/VideoStir)  
**Area**: Information Retrieval
**Keywords**: long video understanding, retrieval-augmented generation, spatio-temporal graph structure, intent-aware retrieval, multi-hop reasoning

## TL;DR

VideoStir proposes a structured and intent-aware RAG framework for long video understanding. By modeling videos as spatio-temporal graphs for multi-hop clip retrieval and training an intent relevance scorer for frame-level filtering, the framework achieves performance comparable to state-of-the-art long video RAG methods without relying on any auxiliary text tools.

## Background & Motivation

**Background**: Long video understanding is a core frontier task in multimodal intelligence. Current methods either extend context windows with uniform sampling—risking the omission of critical details or being overwhelmed by redundant information—or apply RAG to retrieve key segments and compress the context.

**Limitations of Prior Work**:
- **Spatio-temporal structure decoupling**: Existing RAG methods flatten videos into independent segments, destroying the inherent spatio-temporal structure and preventing the association of contextually related events dispersed across different time points.
- **Insufficient intent modeling**: Mainstream methods rely on contrastive embeddings such as CLIP to compute semantic similarity, which can only match content that "looks similar" rather than content that is "truly important for answering the query's intent" (e.g., for the query "What does the recorder do with the printer?", semantic retrieval selects frames showing the printer rather than scenes relevant to the actual purpose).

**Key Challenge**: Flattened retrieval decouples spatio-temporal context that should remain connected, causing contextually associated evidence to be missed; semantic matching favors surface-level similarity, overlooking intent-relevant but semantically non-overlapping key cues.

**Goal**: To improve long video RAG along two dimensions — (1) from flat to structured: reconstructing the spatio-temporal topology of videos; (2) from semantic to intent-aware: moving beyond surface semantic matching to model the alignment between query intent and visual cues.

**Key Insight**: Analogous to human episodic memory — first coarsely locating relevant episodes (clip retrieval), then finely examining details (frame retrieval). Graph structure is used at the clip level to preserve spatio-temporal associations, and MLLM-based reasoning is applied at the frame level to assess intent relevance.

**Core Idea**: Videos are modeled as spatio-temporal graphs (nodes = semantically coherent clips, edges = temporal proximity / spatial similarity). Multi-hop traversal aggregates structured evidence, followed by a distillation-trained intent relevance scorer for fine-grained frame-level filtering.

## Method

### Overall Architecture

VideoStir operates in three stages: (1) **Spatio-temporal topology modeling** — an event boundary detector segments the video into clips, from which a spatio-temporal graph is constructed; (2) **Graph-structured clip retrieval** — query embeddings match anchor nodes, and multi-hop traversal expands the spatio-temporal neighborhood; (3) **Intent-aware frame retrieval** — an intent relevance scorer ranks candidate frames, and intent-aligned keyframes are selected and passed to the downstream MLLM.

### Key Designs

1. **Spatio-Temporal Topology Modeling**:
    - Function: Models the long video as a graph $\mathcal{G}=(\mathcal{V}, \mathcal{E})$ that preserves spatio-temporal structure.
    - Mechanism: An event boundary detector (PELT change-point detection on frame embeddings) adaptively segments the video into semantically coherent clip nodes. Temporal edges connect adjacent clips to maintain narrative continuity; spatial edges connect semantically related but temporally distant clips based on cosine similarity of clip embeddings.
    - Design Motivation: Flattened retrieval decouples spatio-temporal contexts that should remain linked. The graph structure re-entangles these relationships, enabling multi-hop retrieval to aggregate evidence along both the temporal axis and the semantic space.

2. **Graph-Structured Multi-Hop Clip Retrieval**:
    - Function: Starting from anchor nodes matched to the query, collects contextually relevant clips by traversing the spatio-temporal graph.
    - Mechanism: The top-N (default: 3) clips most similar to the query are selected as anchors; L-hop (default: 2) traversal is then performed on the graph, with weak connections filtered by an edge weight threshold η (default: 0.4) to collect the spatio-temporal neighborhood.
    - Design Motivation: A query may involve only a small portion of an event, yet complete reasoning requires temporally adjacent and semantically related context. Multi-hop traversal leverages intrinsic inter-clip associations to supplement evidence missed by direct query matching.

3. **Intent-Aware Frame Retrieval + IR-600K Dataset**:
    - Function: Distinguishes "intent-relevant" from "merely semantically similar" visual cues at the frame level.
    - Mechanism: Qwen2.5-VL-72B serves as the teacher model to annotate intent relevance scores (1–5) for 605K query-frame pairs, distilling into a Qwen2.5-VL-3B student scorer (LoRA, only 3.7M parameters). At inference time, a probability-weighted expected score is computed for each candidate frame, and frames exceeding threshold $\kappa_s$ are retained.
    - Design Motivation: Contrastive models such as CLIP are optimized for semantic alignment rather than intent alignment, frequently selecting frames that "look relevant" but contribute nothing to answering the query. MLLMs possess the reasoning capability to assess a frame's contribution to query intent, but full-scale inference is prohibitively slow; hence the capability is distilled into a lightweight scorer.

### Loss & Training

Scorer training uses cross-entropy loss: $\mathcal{L}_{CE} = -\sum_{\ell=1}^{5} \mathbf{1}[\ell=y_t] \log P_\theta(\ell|q, x_t, \mathcal{P}_{intent})$, optimizing LoRA parameters. AdamW (lr=5e-5) with cosine schedule, 1 epoch, batch size 128.

## Key Experimental Results

### Main Results

| Base MLLM | Method | LV-Bench | MLVU | Video-MME-Long |
|-----------|--------|----------|------|----------------|
| LLaVA-Video 7B | Native | 56.6 | 70.8 | - |
| LLaVA-Video 7B | +Video-RAG | 58.7 (+3.7%) | 72.4 (+2.3%) | - |
| LLaVA-Video 7B | **+VideoStir** | **60.3 (+6.5%)** | **73.1 (+3.2%)** | - |
| LLaVA-Video 72B | Native | 61.9 | 73.1 | 61.5 |
| LLaVA-Video 72B | +Video-RAG | 65.4 (+5.7%) | 73.8 (+1.0%) | 62.3 (+1.3%) |
| LLaVA-Video 72B | **+VideoStir** | **66.0 (+6.6%)** | **74.1 (+1.4%)** | 62.1 (+1.0%) |

### Ablation Study

| Configuration | Overall↑ | Retrieval Acc.↑ | Note |
|---------------|---------|-----------------|------|
| Full | **64.5** | **92.2** | Complete model |
| w/o intent scorer (w/ PE) | 58.1 | 79.8 | Semantic matching insufficient for intent capture |
| w/o probability-weighted expectation | 54.2 | 71.6 | Discrete scores inferior to distributional scoring |
| w/o spatio-temporal graph | 56.4 | 74.8 | Flattened retrieval loses structural information |
| w/o spatial edges | 57.2 | 79.3 | Semantically related distant clips are missed |
| w/o temporal edges | 59.8 | 83.4 | Narrative continuity is disrupted |

### Key Findings
- VideoStir relies solely on native visual input without any auxiliary text tools (OCR, caption generation, etc.), yet achieves state-of-the-art performance.
- The intent scorer outperforms the strongest semantic matching baseline (PE) by 6.4%/12.4% on Overall/Retrieval Acc., underscoring the critical importance of intent modeling.
- LoRA fine-tuning (3.7M parameters) nearly matches full-parameter fine-tuning (3.0B parameters), demonstrating the efficiency of the distillation strategy.
- Both spatial and temporal edges contribute to graph-structured retrieval, but removing spatial edges causes a larger performance drop, indicating that long-range semantic associations are particularly critical.

## Highlights & Insights
- The paradigm shift from "semantic matching to intent-aware retrieval" is precisely motivated — semantic similarity does not equal utility for answering queries, an insight broadly applicable to all RAG systems.
- The spatio-temporal graph with multi-hop retrieval is an elegant design: it reconstructs the intrinsic topological structure of videos rather than resorting to brute-force search, analogous to human episodic memory retrieval.
- The IR-600K dataset is itself a contribution: the first dataset targeting intent-level frame–query alignment, reusable for future research.
- The scorer distillation strategy is practically efficient: from a 72B teacher to a 3B student with only 3.7M LoRA parameters, balancing quality and deployability.

## Limitations & Future Work
- Spatio-temporal graph construction and multi-hop retrieval introduce additional system latency; end-to-end latency optimization is an important direction.
- The quality of the event boundary detector directly affects the graph structure and may lack robustness for complex, interleaved narratives.
- On Video-MME-Long, VideoStir's gains are less pronounced than those of Video-RAG on certain MLLMs, suggesting that auxiliary textual information retains value in some scenarios.
- Evaluation is currently limited to QA tasks; applicability to other tasks such as video summarization and temporal grounding remains to be verified.

## Related Work & Insights
- **vs. Video-RAG**: Video-RAG enhances retrieval with auxiliary text tools; VideoStir relies solely on native visual input, achieving comparable performance with a simpler pipeline.
- **vs. DrVideo/Vgent (agent-based methods)**: Agent-based methods incur high reasoning overhead; VideoStir is more efficient through graph structure combined with a lightweight scorer.
- **vs. AKS (keyframe selection)**: AKS optimizes semantic similarity and temporal uniformity; VideoStir introduces intent-level frame filtering.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of spatio-temporal graph and intent scorer addresses two core pain points in long video RAG.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark evaluation, multiple MLLM backbones, detailed ablations, and scorer training strategy analysis.
- Writing Quality: ⭐⭐⭐⭐ The narrative structure — problem analysis → two gaps → two paradigm shifts — is clear and compelling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] All Languages Matter: Understanding and Mitigating Language Bias in Multilingual RAG](all_languages_matter_understanding_and_mitigating_language_bias_in_multilingual_.md)
- [\[ICLR 2026\] Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding](../../ICLR2026/information_retrieval/beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn.md)
- [\[ACL 2026\] Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches](is_agentic_rag_worth_it_an_experimental_comparison_of_rag_approaches.md)
- [\[ACL 2026\] How Retrieved Context Shapes Internal Representations in RAG](how_retrieved_context_shapes_internal_representations_in_rag.md)
- [\[ICLR 2026\] Q-RAG: Long Context Multi-Step Retrieval via Value-Based Embedder Training](../../ICLR2026/information_retrieval/q_rag_long_context_multi_step_retrieval.md)

</div>

<!-- RELATED:END -->
