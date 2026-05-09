---
title: >-
  [Paper Note] Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search
description: >-
  [CVPR 2026][LLM Agent][Long video understanding] This paper proposes the HAVEN framework, which achieves 84.1% accuracy on LVBench through audiovisual entity cohesion and a four-level hierarchical video index (global–scene–clip–entity), coupled with an agentic search mechanism, attaining 80.1% on the reasoning category.
tags:
  - CVPR 2026
  - LLM Agent
  - Long video understanding
  - audiovisual entity cohesion
  - hierarchical indexing
  - agentic search
  - speaker identification
date: 2026-05-08
content_hash: 429356235812b046
---

# Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search

**Conference**: CVPR 2026
**arXiv**: [2601.13719](https://arxiv.org/abs/2601.13719)
**Code**: N/A
**Area**: LLM Agent
**Keywords**: Long video understanding, audiovisual entity cohesion, hierarchical indexing, agentic search, speaker identification

## TL;DR

This paper proposes the HAVEN framework, which achieves 84.1% accuracy on LVBench through audiovisual entity cohesion and a four-level hierarchical video index (global–scene–clip–entity), coupled with an agentic search mechanism, attaining 80.1% on the reasoning category.

## Background & Motivation

1. **State of the Field**: Long video understanding faces the challenge of extremely long context windows. RAG-based methods and agentic frameworks have made progress, but suffer from information fragmentation and loss of global coherence.
2. **Limitations of Prior Work**: Existing retrieval-driven methods retrieve based on isolated signals (clip-level captions), and fragmented or redundant evidence severely undermines global narrative coherence. The absence of hierarchical video representations leaves agents without the structured context required for multi-level reasoning.
3. **Root Cause**: Flat databases (frames, captions, entities) require extensive iterative retrieval to recover cross-clip continuity, introducing unnecessary complexity and computational cost.
4. **Paper Goals**: To shift from fragmented retrieval toward coherent, structured understanding.
5. **Starting Point**: Leveraging speaker identification as a strong signal for entity cohesion—speaker identity remains informative when visual cues degrade (occlusion, viewpoint change, etc.).
6. **Core Idea**: Audiovisual entity cohesion + four-level hierarchical indexing + goal-driven agentic search.

## Method

### Overall Architecture

An offline phase constructs a four-level database $\mathcal{D} = \{\tilde{\mathcal{C}}, \tilde{\mathcal{E}}, \tilde{\mathcal{S}}, \tilde{\mathcal{G}}\}$ (clip, entity, scene, global). At inference time, the agent navigates across levels via a think-act-observe loop to retrieve and reason.

### Key Designs

1. **Audiovisual Entity Cohesion**:

   - **Function**: Maintains semantic consistency of entities across time and modalities.
   - **Mechanism**: WhisperX is used for ASR and speaker diarization to obtain timestamped transcriptions and consistent speaker labels. After entity extraction, a two-stage merging process is applied: (1) embedding clustering to form candidate groups; (2) LLM review of each cluster for normalization or splitting. When multiple clips share the same speaker label, the corresponding character entities are preferentially merged.
   - **Design Motivation**: Speaker identity remains reliable when visual cues degrade (occlusion, shot changes, appearance variation) and serves as a "glue" for cross-clip entity association.

2. **Four-Level Hierarchical Database**:

   - **Function**: Supports flexible retrieval at multiple granularities.
   - **Mechanism**: (1) Clip level: fixed 30-second windows with textual descriptions and visual embeddings; (2) Entity level: normalized entities with re-descriptions of their associated clips; (3) Scene level: LLM adaptively aggregates semantically related clips into scene summaries; (4) Global level: a global summary generated from the set of scenes.
   - **Design Motivation**: Different query types require information at different granularities—"what is the video about" requires the global level, while "what happened at 12:00" requires clip-level detail.

3. **Agentic Search with Multi-Granularity Toolset**:

   - **Function**: Query-driven adaptive multi-level retrieval and reasoning.
   - **Mechanism**: Five tools are defined: global scene browsing $T_{scene}$, clip caption search $T_{caption}$, clip visual search $T_{visual}$, entity search $T_{entity}$, and an inspection tool $T_{inspect}$ (with both text and visual modes). The agent is initialized with the global summary and dynamically selects tools across multiple iterations.
   - **Design Motivation**: Low-cost text retrieval is prioritized; high-cost visual inspection is invoked only when necessary.

### Loss & Training

HAVEN is a purely inference-based framework with no training. The database is constructed offline, and the agent performs online search at inference time.

## Key Experimental Results

### Main Results

| Method | LVBench Overall | Reasoning Category |
|---|---|---|
| HAVEN (2fps) | 84.1 | 80.1 |
| DVD w/ subtitle | 76.0 | 68.7 |
| Seed1.5-VL-200B | 64.6 | 63.7 |
| OpenAI o3 | 57.1 | 50.8 |

### Ablation Study

| Configuration | LVBench Overall | Note |
|---|---|---|
| Full HAVEN | 84.1 | Best |
| w/o audio entity cohesion | Drops | Entity fragmentation |
| w/o hierarchical indexing | Drops | Lower retrieval efficiency |

### Key Findings

- HAVEN achieves 80.1% on the most challenging reasoning category, substantially outperforming DVD (68.7%).
- Speaker identity is a critical factor—Figure 3 demonstrates that characters with drastic appearance changes are correctly associated via speaker labels.
- Performance improves from 81.0 to 84.1 at 2fps, as denser sampling provides more visual evidence.

## Highlights & Insights

- **Speaker Identity as a Cross-Modal Glue**: The framework elegantly exploits speaker consistency in the audio signal, a signal previously underutilized.
- **Offline–Online Decoupling**: The hierarchical database is constructed offline, and inference requires only lightweight tool calls.
- **Strong Practicality**: The approach is particularly effective for dialogue-intensive content such as documentaries, TV series, and vlogs.

## Limitations & Future Work

- The framework depends on the accuracy of ASR and speaker diarization, and may degrade in noisy audio environments.
- Fixed 30-second clip segmentation may not be suitable for all video types.
- Cached content is limited; further experimental details require consulting the full paper.
- The computational cost and storage overhead of database construction are not analyzed in detail.

## Related Work & Insights

- **vs. DVD**: DVD relies on a flat database requiring extensive iterations, whereas HAVEN reduces the number of iterations through hierarchical database design.
- **vs. VideoRAG**: VideoRAG depends on graph-based semantic retrieval but lacks hierarchical structure.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Audiovisual entity cohesion and the use of speaker identity constitute novel contributions; the hierarchical indexing design is systematic.
- **Experimental Thoroughness**: ⭐⭐⭐ Limited cached details; LVBench results are strong but results on other benchmarks are incomplete.
- **Writing Quality**: ⭐⭐⭐⭐ Architecture diagrams are clear, case studies are intuitive, and method descriptions are well-organized.
- **Value**: ⭐⭐⭐⭐ A practical framework for long video understanding; the speaker identity utilization strategy is transferable to other multimodal scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] HAVEN: Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search](haven_hierarchical_long_video_understanding_with_audiovisual_entity_cohesion.md)
- [\[CVPR 2026\] Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding](think_then_verify_a_hypothesis-verification_multi-agent_framework_for_long_video.md)
- [\[CVPR 2026\] WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning](worldmm_dynamic_multimodal_memory_agent_for_long_video_reasoning.md)
- [\[NeurIPS 2025\] Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding](../../NeurIPS2025/llm_agent/deep_video_discovery_agentic_search_with_tool_use_for_longfo.md)
- [\[CVPR 2026\] ARGOS: Who, Where, and When in Agentic Multi-Camera Person Search](argos_agentic_multi_camera_person_search.md)

<!-- RELATED:END -->
