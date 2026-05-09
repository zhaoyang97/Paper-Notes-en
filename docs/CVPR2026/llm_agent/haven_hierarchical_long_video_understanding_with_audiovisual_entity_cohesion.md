---
title: >-
  [Paper Note] HAVEN: Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search
description: >-
  [CVPR 2026][LLM Agent][long video understanding] HAVEN proposes a unified framework combining audiovisual entity cohesion, hierarchical indexing, and agentic search. By leveraging speaker identity as a cross-modal coherence signal, it constructs a four-level hierarchical database (global → scene → clip → entity), achieving state-of-the-art 84.1% overall accuracy on LVBench.
tags:
  - CVPR 2026
  - LLM Agent
  - long video understanding
  - hierarchical indexing
  - entity cohesion
  - agentic search
  - audiovisual fusion
date: 2026-05-08
content_hash: 0dfef1e66c930652
---

# HAVEN: Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search

**Conference**: CVPR 2026
**arXiv**: [2601.13719](https://arxiv.org/abs/2601.13719)
**Code**: None
**Area**: LLM Agent
**Keywords**: long video understanding, hierarchical indexing, entity cohesion, agentic search, audiovisual fusion

## TL;DR
HAVEN proposes a unified framework combining audiovisual entity cohesion, hierarchical indexing, and agentic search. By leveraging speaker identity as a cross-modal coherence signal, it constructs a four-level hierarchical database (global → scene → clip → entity), achieving state-of-the-art 84.1% overall accuracy on LVBench.

## Background & Motivation
1. **Background**: Long video understanding is a major challenge for VLMs; existing approaches (RAG, agent frameworks) remain severely limited when handling hour-long videos.
2. **Limitations of Prior Work**: (i) Naïve chunk-based RAG causes information fragmentation and loss of global coherence; (ii) the absence of hierarchical video representations forces agents to perform inefficient multi-round retrieval to recover cross-clip continuity.
3. **Key Challenge**: Events in long videos span extended time horizons and evolve across multiple scenes; local clip descriptions cannot capture global narrative structure or long-range entity associations.
4. **Goal**: Transition from fragmented retrieval to coherent structured understanding—via offline hierarchical database construction and online adaptive agentic search.
5. **Key Insight**: Speaker identity is exploited as a long-range cross-modal coherence signal that remains effective even when visual cues are unreliable, enabling robust entity representation.
6. **Core Idea**: Audiovisual entity cohesion (integrating fragmented observations via speaker identity) + four-level hierarchical database + goal-driven multi-granularity agentic search.

## Method

### Overall Architecture
**Offline**: Construct a four-level hierarchical database $\mathcal{D} = \{\tilde{\mathcal{C}}, \tilde{\mathcal{E}}, \tilde{\mathcal{S}}, \tilde{\mathcal{G}}\}$ (clip → entity → scene → global). **Online**: The agent initializes with the global summary and adaptively searches and reasons within the hierarchical database through a think-act-observe loop.

### Key Designs

1. **Audiovisual Entity Cohesion**:
    - **Function**: Integrates fragmented entity observations across time and modalities into coherent canonical entities.
    - **Mechanism**: For each clip, audio annotations (WhisperX speaker diarization + ASR) and visual descriptions (VLM-generated) are extracted to construct clip representations $C_i^t = [P_i'; T_i; V_i]$. Entity consolidation proceeds in two steps—(1) embedding clustering: entity descriptions are encoded and clustered into candidate groups; (2) LLM canonicalization: each cluster is validated and either confirmed as a canonical entity or split. Crucially, when multiple clips share the same speaker label, the corresponding character entities are preferentially merged, even when visual descriptions differ due to occlusion or viewpoint changes.
    - **Design Motivation**: Speaker identity is a more stable long-range cue than visual appearance—occlusion, shot changes, and lighting variations do not affect vocal identity. This constitutes a powerful yet largely overlooked coherence signal.

2. **Four-Level Hierarchical Database**:
    - **Function**: Organizes video content at multiple granularities to support queries at different levels of abstraction.
    - **Mechanism**: (1) Clip level $\tilde{\mathcal{C}}$: one clip per 30 seconds, containing textual and visual embeddings; (2) Entity level $\tilde{\mathcal{E}}$: canonical entities along with focused re-descriptions in each associated clip; (3) Scene level $\tilde{\mathcal{S}}$: semantically continuous clips are adaptively grouped by an LLM and summarized; (4) Global level $\tilde{\mathcal{G}}$: an overall overview generated from scene summaries.
    - **Design Motivation**: Different query types require different granularities—"What is this video about?" requires the global level; "What happened at the 12-minute mark?" requires the clip level; "How does Sarah's expression change?" requires the entity level.

3. **Multi-Granularity Agentic Search**:
    - **Function**: Goal-driven navigation and reasoning within the hierarchical database.
    - **Mechanism**: The agent is equipped with five tool types—global scene browsing $T_{\text{scene}}$, clip caption search $T_{\text{caption}}$, clip visual search $T_{\text{visual}}$, entity search $T_{\text{entity}}$, and targeted inspection $T_{\text{inspect}}$ (both textual and visual modes). Initialized with the global summary, the agent iterates through a think-act-observe loop: selecting tools → executing queries → collecting evidence → reasoning → producing answers.
    - **Design Motivation**: Different queries call for entry at different levels. The agent autonomously determines the most efficient search path, such as coarse-to-fine navigation or direct entity localization.

### Loss & Training
The offline database construction requires no training. Agentic search employs a pretrained reasoning LLM with no additional training.

## Key Experimental Results

### Main Results

| Method | LVBench Overall | LVBench Reasoning | EgoSchema | Notes |
|--------|----------------|-------------------|-----------|-------|
| HAVEN (2fps) | **84.1** | **80.1** | — | SOTA |
| DVD w. subtitle | 76.0 | 68.7 | — | Prev. SOTA agent |
| OpenAI o3 | 57.1 | 50.8 | 63.2 | Closed-source |
| GPT-4o | 48.9 | 50.3 | 70.4 | Closed-source |

### Ablation Study

| Configuration | Overall | Notes |
|---------------|---------|-------|
| Full HAVEN | 84.1 | Complete framework |
| w/o speaker identity | Drops | Degraded entity consolidation quality |
| w/o hierarchical indexing | Significant drop | Degenerates to flat RAG |
| w/o multi-granularity tools | Drops | Reduced search efficiency |

### Key Findings
- HAVEN performs particularly strongly on the reasoning category (80.1%), indicating that hierarchical structure is especially beneficial for complex reasoning.
- Speaker identity is an indispensable cue for entity consolidation in long videos.
- Compared to DVD, HAVEN achieves improvements across all subcategories while requiring fewer search iterations.

## Highlights & Insights
- **Speaker identity as the "glue" for entity cohesion** is a largely overlooked yet highly effective innovation.
- **The four-level hierarchical architecture** aligns with the cognitive pattern humans employ when understanding long videos—global context before fine-grained detail.
- The offline construction + online search design eliminates the need to reprocess the video for repeated queries.

## Limitations & Future Work
- Offline hierarchical database construction incurs non-trivial computational cost due to multiple LLM calls.
- The approach depends on the quality of WhisperX speaker diarization, limiting effectiveness on non-dialogue-heavy videos.
- Fixed clip length (30 seconds) may not be optimal for all video types.

## Related Work & Insights
- **vs. DVD**: DVD relies on simple clip descriptions and global entity registration without hierarchical structure. HAVEN's four-level hierarchy enables significantly more efficient navigation.
- **vs. VideoRAG**: VideoRAG retrieves based on fragmented clips and lacks global coherence. HAVEN preserves narrative structure through hierarchical indexing.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Both audiovisual entity cohesion and four-level hierarchical indexing represent innovative designs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — LVBench SOTA with multi-benchmark validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear framework presentation and systematic method description.
- **Value**: ⭐⭐⭐⭐⭐ — A milestone contribution to the long video understanding field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search](haven_hierarchical_long_video_understanding_audiovisual_entity.md)
- [\[CVPR 2026\] Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding](think_then_verify_a_hypothesis-verification_multi-agent_framework_for_long_video.md)
- [\[CVPR 2026\] WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning](worldmm_dynamic_multimodal_memory_agent_for_long_video_reasoning.md)
- [\[NeurIPS 2025\] Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding](../../NeurIPS2025/llm_agent/deep_video_discovery_agentic_search_with_tool_use_for_longfo.md)
- [\[CVPR 2026\] ARGOS: Who, Where, and When in Agentic Multi-Camera Person Search](argos_agentic_multi_camera_person_search.md)

</div>

<!-- RELATED:END -->
