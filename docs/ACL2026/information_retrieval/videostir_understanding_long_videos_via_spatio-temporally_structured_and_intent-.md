---
title: >-
  [Paper Note] VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG
description: >-
  [ACL 2026][Information Retrieval & RAG][Long Video Understanding] VideoStir proposes a structured and intent-aware long video RAG framework. It models videos as spatio-temporal graphs for multi-hop clip retrieval and tra…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Long Video Understanding"
  - "Retrieval-Augmented Generation"
  - "Spatio-Temporal Graph Structure"
  - "Intent-Aware Retrieval"
  - "Multi-hop Reasoning"
date: 2026-05-08
content_hash: 4fc6f02b4bce1f8e
---

# VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG

**Conference**: ACL 2026  
**arXiv**: [2604.05418](https://arxiv.org/abs/2604.05418)  
**Code**: [https://github.com/RomGai/VideoStir](https://github.com/RomGai/VideoStir)  
**Area**: Information Retrieval  
**Keywords**: Long Video Understanding, Retrieval-Augmented Generation, Spatio-Temporal Graph Structure, Intent-Aware Retrieval, Multi-hop Reasoning

## TL;DR

VideoStir proposes a structured and intent-aware long video RAG framework. It models videos as spatio-temporal graphs for multi-hop clip retrieval and trains an intent relevance scorer for frame-level filtering. Without relying on auxiliary text tools, it achieves performance comparable to SOTA long video RAG methods.

## Background & Motivation

**Background**: Long video understanding is a core frontier task in multimodal intelligence. Current methods either extend the context window for uniform sampling (prone to missing critical details or being overwhelmed by redundant information) or use RAG to retrieve key segments to compress the context.

**Limitations of Prior Work**:
- **Spatio-temporal Structural Decoupling**: Existing RAG methods flatten videos into independent segments, breaking the inherent spatio-temporal structure. This prevents the associated retrieval of events that are contextually related but scattered across different timestamps.
- **Insufficient Intent Modeling**: Mainstream methods rely on contrastive embeddings like CLIP to calculate semantic similarity, which only match content that "looks similar" rather than content that is "actually important for answering the query intent" (e.g., for the query "What does the recorder do with the printer?", semantic retrieval might select printer shots rather than scenes related to the actual purpose).

**Key Challenge**: Flattened retrieval loses structure $\rightarrow$ missing contextually related evidence; semantic matching biases towards surface similarity $\rightarrow$ missing critical clues that are intent-relevant but lack semantic overlap.

**Goal**: Improve long video RAG from two dimensions: (1) From flattened to structured: Reconstructing the video spatio-temporal topology; (2) From semantic to intent: Moving beyond surface semantic matching to model the alignment between query intent and visual clues.

**Key Insight**: Analogous to human episodic memory—first coarsely locate relevant episodes (clip retrieval), then finely examine details (frame retrieval). Use graph structures at the clip level to maintain spatio-temporal associations and MLLM reasoning at the frame level for intent relevance.

**Core Idea**: Model the video as a spatio-temporal graph (nodes = semantically consistent clips, edges = temporal proximity/spatial similarity). Aggregate structured evidence through multi-hop traversal, followed by fine-grained filtering at the frame level using an intent relevance scorer trained via distillation.

## Method

### Overall Architecture

VideoStir consists of three stages: (1) Spatio-temporal topology modeling—an event boundary detector segments the video into clips to build a spatio-temporal graph; (2) Graph-structured clip retrieval—query embeddings match anchor nodes, and multi-hop traversal expands the spatio-temporal neighborhood; (3) Intent-aware frame retrieval—an intent relevance scorer ranks candidate frames, filtering intent-aligned keyframes to be sent to the downstream MLLM.

### Key Designs

1. **Spatio-Temporal Topology Modeling**:
    - **Function**: Models long videos as a graph $\mathcal{G}=(\mathcal{V}, \mathcal{E})$ that preserves spatio-temporal structure.
    - **Mechanism**: An event boundary detector (PELT change-point detection on frame embeddings) adaptively segments the video into semantically consistent clip nodes. Temporal edges connect adjacent clips to maintain narrative continuity, while spatial edges connect clips that are semantically related but temporally distant based on cosine similarity of clip embeddings.
    - **Design Motivation**: Flattened retrieval decouples spatio-temporal contexts that should remain connected. The graph structure re-entangles these relationships, allowing multi-hop retrieval to aggregate evidence along both the timeline and semantic space.

2. **Graph-Structured Multi-hop Clip Retrieval**:
    - **Function**: Starts from query-matched anchor nodes to collect contextually relevant clips along the spatio-temporal graph.
    - **Mechanism**: Select the top-N (default 3) anchor clips most similar to the query, then perform an L-hop (default 2) traversal on the graph. Filter weak connections using an edge weight threshold $\eta$ (default 0.4) to collect the spatio-temporal neighborhood.
    - **Design Motivation**: A query might only involve a small part of an event, but complete reasoning requires temporally adjacent and semantically related context. Multi-hop traversal utilizes the internal associations between clips to supplement evidence missed by direct query matching.

3. **Intent-Aware Frame Retrieval + IR-600K Dataset**:
    - **Function**: Distinguishes "intent-relevant" visual clues from those that are "only semantically similar" at the frame level.
    - **Mechanism**: Uses Qwen2.5-VL-72B as a teacher model to label the intent relevance (levels 1-5) of 605,000 query-frame pairs. A Qwen2.5-VL-3B student scorer is trained via distillation (LoRA, only 3.7M parameters). During inference, a probability-weighted expected score is calculated for candidate frames, retaining those exceeding a threshold $\kappa_s$.
    - **Design Motivation**: Contrastive models like CLIP optimize for semantic alignment rather than intent alignment, often selecting frames that "look relevant" but are useless for answering. MLLMs possess reasoning capabilities to judge a frame's contribution to the query intent, but large model inference is too slow; hence, they are distilled into a lightweight scorer.

### Loss & Training

Scorer Training: Cross-entropy loss $\mathcal{L}_{CE} = -\sum_{\ell=1}^{5} \mathbf{1}[\ell=y_t] \log P_\theta(\ell|q, x_t, \mathcal{P}_{intent})$ is used to optimize LoRA parameters. Training uses AdamW (lr=5e-5), a cosine schedule, 1 epoch, and a batch size of 128.

## Key Experimental Results

### Main Results

| Base MLLM | Method | LV-Bench | MLVU | Video-MME-Long |
|---------|------|----------|------|------------|
| LLaVA-Video 7B | Native | 56.6 | 70.8 | - |
| LLaVA-Video 7B | +Video-RAG | 58.7 (+3.7%) | 72.4 (+2.3%) | - |
| LLaVA-Video 7B | **+VideoStir** | **60.3 (+6.5%)** | **73.1 (+3.2%)** | - |
| LLaVA-Video 72B | Native | 61.9 | 73.1 | 61.5 |
| LLaVA-Video 72B | +Video-RAG | 65.4 (+5.7%) | 73.8 (+1.0%) | 62.3 (+1.3%) |
| LLaVA-Video 72B | **+VideoStir** | **66.0 (+6.6%)** | **74.1 (+1.4%)** | 62.1 (+1.0%) |

### Ablation Study

| Configuration | Overall↑ | Retrieval Acc.↑ | Description |
|------|---------|----------------|------|
| Full | **64.5** | **92.2** | Complete model |
| w/o Intent Scorer (using PE) | 58.1 | 79.8 | Semantic matching is insufficient to capture intent |
| w/o Prob-weighted Expectation | 54.2 | 71.6 | Discrete scores are inferior to distributed scoring |
| w/o Spatio-temporal Graph | 56.4 | 74.8 | Flattened retrieval loses structural information |
| w/o Spatial Edges | 57.2 | 79.3 | Semantically related long-distance clips are missed |
| w/o Temporal Edges | 59.8 | 83.4 | Narrative continuity is broken |

### Key Findings
- VideoStir achieves SOTA without using any auxiliary text tools (OCR, captions, etc.), relying solely on native visual input.
- The intent scorer provides a 6.4%/12.4% boost (Overall/Retrieval Acc.) over the strongest semantic matching (PE), highlighting the criticality of intent modeling.
- LoRA fine-tuning (3.7M parameters) nearly matches the performance of full-parameter fine-tuning (3.0B parameters), demonstrating the efficiency of the distillation strategy.
- Both spatial and temporal edges in the graph contribute, but removing spatial edges has a larger impact, indicating that long-distance semantic associations are crucial.

## Highlights & Insights
- The paradigm shift from "semantic matching to intent awareness" is accurately positioned—semantic similarity $\neq$ utility for answering. This insight is enlightening for all RAG systems.
- The design of the spatio-temporal graph + multi-hop retrieval is elegant: it reconstructs the video's intrinsic topology rather than performing a brute-force search, analogous to the human episodic memory recall process.
- The IR-600K dataset is a contribution in itself: it is the first dataset oriented towards "intent-level frame-query alignment" and can be reused in future research.
- The scorer distillation strategy is practical: moving from a 72B teacher to a 3B student using LoRA with only 3.7M parameters maintains quality while being suitable for deployment.

## Limitations & Future Work
- The construction of the spatio-temporal graph and multi-hop retrieval introduces additional system latency; end-to-end latency optimization is an important direction.
- The quality of the event boundary detector directly affects the graph structure and may not be robust enough for complex, interleaved narratives.
- On Video-MME-Long, VideoStir's improvement is less significant than Video-RAG (on some MLLMs), suggesting that auxiliary text information remains valuable in certain scenarios.
- Currently evaluated only on QA tasks; the applicability to other tasks like video summarization and temporal localization needs verification.

## Related Work & Insights
- **vs Video-RAG**: Video-RAG enhances retrieval with auxiliary text tools. VideoStir relies solely on native visual input, being more concise while achieving comparable performance.
- **vs DrVideo/Vgent (Agent methods)**: Agent methods involve high reasoning overhead. VideoStir is more efficient through the graph structure + lightweight scorer.
- **vs AKS (Keyframe Selection)**: AKS optimizes for semantic similarity + temporal uniformity. VideoStir introduces intent-level frame filtering.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of spatio-temporal graphs + intent scorers addresses two core pain points of long video RAG.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple benchmarks, multiple MLLM backbones, detailed ablations, and analysis of scorer training strategies.
- Writing Quality: ⭐⭐⭐⭐ Clear and powerful narrative structure: problem analysis $\rightarrow$ two gaps $\rightarrow$ two shifts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] All Languages Matter: Understanding and Mitigating Language Bias in Multilingual RAG](all_languages_matter_understanding_and_mitigating_language_bias_in_multilingual_.md)
- [\[ACL 2026\] Disco-RAG: Discourse-Aware Retrieval-Augmented Generation](disco-rag_discourse-aware_retrieval-augmented_generation.md)
- [\[ICLR 2026\] Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding](../../ICLR2026/information_retrieval/beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn.md)
- [\[ACL 2026\] S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA](s2g-rag_structured_sufficiency_and_gap_judging_for_iterative_retrieval-augmented.md)
- [\[ACL 2026\] Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches](is_agentic_rag_worth_it_an_experimental_comparison_of_rag_approaches.md)

</div>

<!-- RELATED:END -->
