---
title: >-
  [Paper Note] WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning
description: >-
  [CVPR 2026][LLM Agent][Multimodal Memory] This paper proposes WorldMM, a video reasoning agent based on multimodal memory, which constructs three complementary memory types: episodic memory (multi-temporal-scale textual knowledge graphs), semantic memory (continuously updated relational knowledge graphs), and visual memory (frame-level retrieval stores). An adaptive multi-round retrieval agent dynamically selects the most relevant memory source and temporal granularity, achieving an average improvement of 8.4% over the previous state of the art across five long video QA benchmarks.
tags:
  - CVPR 2026
  - LLM Agent
  - Multimodal Memory
  - Long Video Understanding
  - Adaptive Retrieval
  - Knowledge Graph
  - Multi-Temporal Scale
date: 2026-05-08
content_hash: d1c96965319106a6
---

# WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning

**Conference**: CVPR 2026
**arXiv**: [2512.02425](https://arxiv.org/abs/2512.02425)
**Code**: [https://worldmm.github.io](https://worldmm.github.io)
**Area**: Video Understanding / LLM Agent / Long Video Reasoning
**Keywords**: Multimodal Memory, Long Video Understanding, Adaptive Retrieval, Knowledge Graph, Multi-Temporal Scale

## TL;DR
This paper proposes WorldMM, a video reasoning agent based on multimodal memory, which constructs three complementary memory types: episodic memory (multi-temporal-scale textual knowledge graphs), semantic memory (continuously updated relational knowledge graphs), and visual memory (frame-level retrieval stores). An adaptive multi-round retrieval agent dynamically selects the most relevant memory source and temporal granularity, achieving an average improvement of 8.4% over the previous state of the art across five long video QA benchmarks.

## Background & Motivation

1. **Background**: Video LLMs have demonstrated strong capabilities on short video understanding, yet extending them to long videos spanning hours or even days remains highly challenging. Existing memory-augmented methods (e.g., EgoRAG, M3-Agent) alleviate context capacity limitations by constructing textual summaries of video segments for external memory retrieval.
2. **Limitations of Prior Work**: Two core limitations exist: (a) over-reliance on textual representations—almost all existing methods convert events into text descriptions for retrieval and reasoning, discarding critical visual details required for attribute recognition, spatial reasoning, and similar tasks. Even M3-Agent, which uses visual input during memory construction, relies primarily on text at inference time. (b) Fixed temporal-scale retrieval—"where did I put my glasses" may require only a few seconds of video, while "what happened in the second half" demands a much longer temporal window, yet existing methods retrieve segments of a predetermined length (e.g., three 30-second clips) and cannot flexibly adapt to different queries.
3. **Key Challenge**: The multimodal nature of information in long videos (text cannot fully express visual details) and its multi-scale temporal structure (different events span different time ranges) demand that memory and retrieval systems be adaptive in both modality and scale, whereas existing methods are fixed in both dimensions.
4. **Goal**: (1) How can both textual and visual memory be leveraged simultaneously to support reasoning? (2) How can information be retrieved across multiple temporal scales? (3) How can the model autonomously decide when to consult text, when to consult images, and when to stop?
5. **Key Insight**: By analogy with the human memory system—episodic memory stores specific events, semantic memory stores abstract knowledge, and visual memory retains sensory details—three complementary memory types are constructed and dynamically combined through an iterative retrieval agent.
6. **Core Idea**: Three complementary multimodal memories (episodic + semantic + visual) combined with an adaptive multi-round retrieval agent enable on-demand selection of information modality and temporal granularity for reasoning over long videos.

## Method

### Overall Architecture
WorldMM operates in three stages: (1) multimodal memory construction—building episodic, semantic, and visual memories from the video stream; (2) adaptive memory retrieval—a retrieval agent iteratively selects memory sources and queries until sufficient information is gathered; (3) response generation—the retrieval history and original question are passed to a response agent to produce the final answer.

### Key Designs

1. **Episodic Memory**:

    - **Function**: Indexes factual events at multiple temporal resolutions.
    - **Mechanism**: The video is first segmented at the finest temporal scale $t_0$, and captions are generated using a Video LLM. A set of multi-scale temporal resolutions $\mathcal{T} = \{t_0, t_1, ..., t_N\}$ (e.g., 30s / 3min / 10min / 1h) is defined; at each scale $t_i$, captions are converted into factual triples (entity–action–entity) to construct a knowledge graph $G_{t_i}$. The episodic memory is thus a set of multi-scale knowledge graphs $\mathcal{M}_e = \{G_{t_0}, ..., G_{t_N}\}$. Retrieval follows a coarse-to-fine strategy: Personalized PageRank (PPR) first retrieves top-$k$ candidates from graphs at each scale, after which an LLM serves as a cross-scale re-ranker to select the most relevant temporal range and content.
    - **Design Motivation**: A fixed single temporal scale cannot capture events at different granularities ranging from seconds to hours. Multi-scale graphs ensure that both fine-grained event details and long-range narratives can be captured.

2. **Semantic Memory**:

    - **Function**: Continuously updated high-level conceptual knowledge (relationships, habits, etc.).
    - **Mechanism**: The video is segmented at a coarse temporal scale $t_s$, and semantic triples (focusing on conceptual knowledge rather than specific events) are generated for each segment. A consolidation process incrementally merges new knowledge into an evolving semantic graph: embedding similarity is first used to identify overlapping or conflicting triples, and an LLM then determines which outdated triples $T_{remove}$ should be deleted and which $T_{update}$ should be updated or added: $Consolidate(G_{t_s}^k, T_{t_s}^{k+1}) = (G_{t_s}^k \setminus T_{remove}) \cup T_{update}$.
    - **Design Motivation**: Episodic memory is composed of independent events and cannot maintain continuity across scenes or capture high-level knowledge (e.g., "the user habitually uses kitchen wipes"). Semantic memory addresses this gap through continuous consolidation.

3. **Visual Memory**:

    - **Function**: Preserves spatial and appearance details that text cannot fully express.
    - **Mechanism**: Two retrieval modes are supported. (a) Feature retrieval: the video is divided into short clips, each encoded into visual features $\mathcal{M}_v^f = \{f_v^1, ..., f_v^L\}$ using a multimodal encoder (VLM2Vec-V2), and matched against text queries via cosine similarity. (b) Timestamp retrieval: each frame is stored paired with its timestamp $\mathcal{M}_v^I = \{(t_i, I_i)\}$, enabling direct frame access once episodic retrieval has identified the relevant temporal segment.
    - **Design Motivation**: When verifying visual details such as the type of baked goods or the color of an object, textual descriptions are insufficiently precise. The dual-mode design covers both "search by semantics" and "retrieve by timestamp" use cases.

4. **Adaptive Multi-Round Retrieval Agent**:

    - **Function**: Dynamically decides at each round which memory to query and what to query, and when to stop.
    - **Mechanism**: The retrieval agent $\mathcal{R}$ takes the user question $q$ and prior retrieval history $r_{<i}$ as input, and at each round outputs a (memory source $m_i$, query $q_i$) pair or a STOP signal. The process iterates for at most $N$ rounds. Each round's retrieval result is appended to the history before the next round begins, until the agent determines that sufficient information has been gathered or the maximum number of rounds is reached.
    - **Design Motivation**: Different questions require different types and amounts of information, and a fixed strategy cannot satisfy all cases. Iterative retrieval allows the model to revise its retrieval strategy when the first round is unsatisfactory, progressively refining results.

### Loss & Training
WorldMM is an inference-time framework and requires no additional training. Memory construction uses GPT-5-mini; the retrieval and response agents use GPT-5 or Qwen3-VL-8B respectively.

## Key Experimental Results

### Main Results

| Model | EgoLifeQA | Ego-R1 Bench | HippoVlog | LVBench | Video-MME(L) | Avg. |
|---|---|---|---|---|---|---|
| GPT-5 (base) | 48.6 | 46.3 | 75.7 | 60.4 | 74.3 | 61.1 |
| HippoRAG | 59.6 | 56.0 | 63.2 | 54.0 | 52.1 | 57.0 |
| M3-Agent | 53.5 | 52.0 | 65.5 | 49.3 | 55.3 | 55.1 |
| HippoMM | 54.6 | 53.0 | 71.9 | 38.2 | 41.6 | 51.8 |
| WorldMM-8B | 56.4 | 52.0 | 69.7 | 55.4 | 66.0 | 59.9 |
| **WorldMM-GPT** | **65.6** | **65.3** | **78.3** | **61.9** | **76.6** | **69.5** |

### Ablation Study
Effect of different memory combinations:

| Configuration | EgoLifeQA | Ego-R1 | HippoVlog | LVBench | Video-MME | Avg. |
|---|---|---|---|---|---|---|
| E only | 62.6 | 57.0 | 73.6 | 60.6 | 72.7 | 64.9 |
| V only | 37.2 | 34.2 | 51.3 | 47.4 | 64.2 | 44.9 |
| E+S | 63.4 | 61.0 | 73.8 | 58.8 | 74.1 | 66.8 |
| E+V | 63.3 | 63.0 | 75.2 | 59.8 | 76.0 | 66.9 |
| **E+S+V** | **65.6** | **65.3** | **78.3** | **61.9** | **76.6** | **69.5** |

### Key Findings
- **Complementarity of multimodal memories**: Each memory type contributes distinctively—episodic memory forms the foundation (standalone 64.9 vs. visual-only 44.9); visual memory substantially improves EntityLog/EventRecall question types; semantic memory substantially improves HabitInsight (+23%) and RelationMap question types.
- **Necessity of adaptive retrieval**: Significant variation in memory utilization patterns across question categories is observed—EntityLog questions rely more heavily on visual memory, while HabitInsight questions rely more heavily on semantic memory, demonstrating that the model does dynamically select the most relevant memory source.
- **Clear benefit of multi-round retrieval**: Allowing up to 5 retrieval rounds improves performance on EgoLifeQA by 9.3% compared to single-round retrieval, as the model can revise its strategy when the first round is suboptimal.
- **Temporal localization accuracy far exceeds baselines**: WorldMM achieves a tIoU of approximately 10%, substantially outperforming the 2–4% of other methods, demonstrating that multi-scale retrieval significantly improves temporal segment localization.
- **Efficiency advantage**: Through adaptive termination and selective retrieval, WorldMM achieves a superior latency–accuracy trade-off compared to all baselines.

## Highlights & Insights
- **Analogy-driven design based on the human memory system**: The division into episodic, semantic, and visual memory directly corresponds to memory classifications in cognitive psychology. This design possesses theoretical elegance, and experiments confirm the unique contribution of each memory type. The framework is transferable to any AI agent system requiring long-term memory management.
- **Elegant design of multi-scale episodic memory**: Knowledge graphs at different scales provide event information at different granularities, and the coarse-to-fine retrieval strategy naturally supports information acquisition from macro to micro levels. This idea is transferable to document understanding (paragraph-level / sentence-level / word-level retrieval).
- **Incremental consolidation mechanism for semantic memory**: Continuous knowledge updating via embedding matching followed by LLM arbitration provides a concise and effective solution to the long-term knowledge maintenance problem.

## Limitations & Future Work
- Visual memory alone performs poorly (Avg. 44.9), indicating that current visual indexing and retrieval techniques remain a bottleneck.
- The memory construction stage relies on GPT-5-mini, incurring high cost and latency.
- For questions requiring fine-grained temporal reasoning, the tIoU of multi-scale memory remains at approximately 10%, leaving substantial room for improvement.
- The quality of semantic memory consolidation depends on the accuracy of LLM judgments, which may introduce error propagation.
- The incorporation of RL or feedback mechanisms during memory construction to improve memory quality remains unexplored.

## Related Work & Insights
- **vs. EgoRAG**: EgoRAG uses only hierarchical textual memory, whereas WorldMM adds visual memory and semantic memory and introduces adaptive retrieval, improving EgoLifeQA performance from 52.0 to 65.6.
- **vs. M3-Agent**: M3-Agent constructs entity-centric long-term memory and supports iterative reasoning, but relies solely on textual representations. WorldMM achieves substantial gains under equivalent conditions through multimodal memory (55.1→69.5).
- **vs. HippoMM**: HippoMM proposes dual-process memory (semantic summaries + multimodal cues), but its visual utilization is limited. WorldMM's adaptive retrieval strategy is more flexible, yielding significantly stronger performance (51.8→69.5).

## Rating
- **Novelty**: ⭐⭐⭐⭐ The multimodal memory framework design is innovative, though the RAG + LLM agent paradigm is already mature.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five benchmarks spanning hourly to weekly video lengths, extensive ablation studies, memory utilization analysis, and efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and intuitive conceptual diagrams, though notation is somewhat dense.
- **Value**: ⭐⭐⭐⭐⭐ Provides an effective framework paradigm for long video understanding and AI agent memory management.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding](think_then_verify_a_hypothesis-verification_multi-agent_framework_for_long_video.md)
- [\[CVPR 2026\] Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search](haven_hierarchical_long_video_understanding_audiovisual_entity.md)
- [\[CVPR 2026\] HAVEN: Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search](haven_hierarchical_long_video_understanding_with_audiovisual_entity_cohesion.md)
- [\[ICLR 2026\] MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains](../../ICLR2026/llm_agent/mc-search_evaluating_and_enhancing_multimodal_agentic_search_with_structured_lon.md)
- [\[AAAI 2026\] LLandMark: A Multi-Agent Framework for Landmark-Aware Multimodal Interactive Video Retrieval](../../AAAI2026/llm_agent/llandmark_a_multi-agent_framework_for_landmark-aware_multimodal_interactive_vide.md)

</div>

<!-- RELATED:END -->
