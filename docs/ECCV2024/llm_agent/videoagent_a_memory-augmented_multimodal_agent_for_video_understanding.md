---
title: >-
  [Paper Note] VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding
description: >-
  [ECCV 2024][LLM Agent][Video understanding Agent] This paper proposes VideoAgent, a memory-augmented multimodal Agent. By constructing structured memory (temporal memory storing event descriptions and object memory storing object tracking states) and utilizing four tools to interact with the memory, it performs zero-shot long video QA tasks. It achieves an average gain of +6.6% on NExT-QA and +26.0% on EgoSchema, approaching the performance of Gemini 1.5 Pro.
tags:
  - "ECCV 2024"
  - "LLM Agent"
  - "Video understanding Agent"
  - "structured memory"
  - "multimodal tool-use"
  - "long video understanding"
date: 2026-05-08
content_hash: ebb81064dbf06640
---

# VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding

**Conference**: ECCV 2024  
**arXiv**: [2403.11481](https://arxiv.org/abs/2403.11481)  
**Code**: [https://github.com/YueFan1014/VideoAgent](https://github.com/YueFan1014/VideoAgent)  
**Area**: LLM Agent / Video Understanding  
**Keywords**: Video understanding Agent, structured memory, multimodal tool-use, long video understanding, LLM Agent

## TL;DR

This paper proposes VideoAgent, a memory-augmented multimodal Agent. By constructing structured memory (temporal memory storing event descriptions and object memory storing object tracking states) and utilizing four tools to interact with the memory, it performs zero-shot long video QA tasks. It achieves an average gain of +6.6% on NExT-QA and +26.0% on EgoSchema, approaching the performance of Gemini 1.5 Pro.

## Background & Motivation

**Background**: Video understanding, especially long video question answering, is a highly challenging task. Existing methods primarily fall into two categories: (a) end-to-end video LLMs (such as Video-ChatGPT, VideoChat), which directly feed video frames into multimodal models; (b) agent-based methods, which leverage LLM reasoning and tool-calling capabilities to understand videos.

**Limitations of Prior Work**:
   - **End-to-end methods** have limited context windows and struggle to handle long videos (ranging from minutes to tens of minutes). They are usually restricted to uniformly sampling a small number of frames, losing a significant amount of temporal information.
   - **Existing agent methods** lack structured representations of videos, requiring reprocessing of video content for each query, which is inefficient and prone to overlooking global temporal relationships.
   - Long-term temporal relationships in videos (such as causal reasoning and event sequences) and object tracking across frames are core challenges, yet prior methods fail to address both aspects effectively.

**Key Challenge**: Long videos contain a massive volume of information that cannot directly fit into the context window of LLMs. How can key information in the video be efficiently retrieved and utilized by LLMs on demand while maintaining information integrity?

**Goal**:
   - How to construct a compact yet information-complete structured representation for long videos?
   - How to enable LLM Agents to retrieve and utilize this information on demand to answer complex questions?

**Key Insight**: This work draws inspiration from how humans comprehend videos: first forming a general impression (event stream) while remembering the occurrence and state changes of key objects (object memory), and then tracing back as needed when answering questions. This cognitive pattern is formalized into an Agent framework of "structured memory + tool-calling".

**Core Idea**: Preprocess the video using structured memory (event descriptions + object state database), and then leverage LLMs to interact with the memory using four specialized tools to answer questions.

## Method

### Overall Architecture

VideoAgent adopts a two-stage pipeline:

**Stage 1: Memory Construction**
- Input: Raw video
- Processing Steps: Segment the video into 2-second clips $\rightarrow$ Construct temporal memory and object memory separately
- Output: Structured memory storage

**Stage 2: Inference**
- Input: User question + structured memory
- Processing Steps: LLM selects tools based on the question $\rightarrow$ Multi-step interaction $\rightarrow$ Synthesize information to generate answers
- Output: Textual answer

This "understand first, QA later" design ensures that memory construction only needs to be executed once, after which different questions can be answered repeatedly, significantly improving efficiency.

### Key Designs

1. **Temporal Memory**:

    - **Function**: Stores the temporal event stream of the video, including the description, textual features, and visual features of each clip.
    - **Mechanism**: The video is segmented into a clip sequence $\{s_1, s_2, ..., s_n\}$ at 2-second intervals. For each clip, a video captioning model is used to generate an event description text $c_i$. Simultaneously, textual and visual features of each clip are computed (for subsequent similarity-based retrieval). All information is stored in the temporal memory: $M_{temp} = \{(c_i, f_i^{text}, f_i^{visual})\}_{i=1}^{n}$.
    - **Design Motivation**: Event descriptions capture "what happened" in textual form, which is easy for LLMs to read and understand directly. Visual and textual features are used for query-based segment localization. The 2-second granularity balances information density and processing overhead.

2. **Object Memory**:

    - **Function**: Tracks and stores information such as categories, visual features, and appearing intervals of all objects in the video.
    - **Mechanism**: An object detector is employed to detect objects in each frame, and a novel re-identification (re-ID) method is used for cross-segment object tracking. The key innovation lies in the re-ID method, which calculates object similarities in different frames using CLIP visual features to maintain a global object ID table. Object information is stored in an SQL database, where each object contains fields: category, CLIP features, and appearing segments.
    - **Representation**: $M_{obj} = \text{SQL\_DB}\{(\text{id}, \text{category}, \text{clip\_feat}, \text{segments})\}$
    - **Design Motivation**: Many video QA tasks involve specific people or objects (e.g., "What did the person in red do?", "Where was the cup placed at the end?"), requiring object-level tracking information. The SQL database format allows LLMs to retrieve precise information through structured queries.

3. **Tool Set**:
   VideoAgent designs four tools to interact with the memory, which are invoked by the LLM utilizing its zero-shot tool-use capabilities:

   **(a) Caption Retrieval**  
    - Input: Start segment $s_i$ and end segment $s_j$  
    - **Function**: Extracts all event descriptions (up to 15) between $[s_i, s_j]$ from the temporal memory.  
    - Purpose: Understand what events occurred within a specific time range.  

   **(b) Segment Localization**  
    - Input: Text query $q$  
    - **Function**: Localizes the most relevant video segments based on the similarity between query features and clip features stored in the temporal memory.  
    - Purpose: Identify the exact moment when key actions take place.  

   **(c) Visual Question Answering**  
    - Input: Question + target video segment  
    - **Function**: Uses a video LLM to describe and answer questions about a specified short video segment.  
    - Purpose: Perform in-depth visual analysis on a specific segment.  

   **(d) Object Memory Querying**  
    - Input: Questions regarding a specific object or person  
    - **Function**: Queries the SQL database to retrieve information such as object category, appearance times, and features.  
    - Purpose: Answer questions related to objects or people (e.g., "Who appeared in the second scene?").  

4. **Multi-step Reasoning Process**:

    - **Function**: The LLM serves as a central controller, performing multi-step reasoning based on the question and invoking one tool per step.
    - **Mechanism**: Each step contains three parts:
        - **Chain of Thought**: The LLM analyzes the information currently acquired and what is still missing.
        - **Action**: Selecting which tool to invoke and its parameters.
        - **Observation**: The results returned by the tool.
    - The loop continues until the LLM deems the information sufficient, then outputs the final answer.
    - **Design Motivation**: Multi-step reasoning allows the Agent to handle complex problems—first localizing relevant segments, then reading descriptions, delving deeper into specific frames if necessary, and finally synthesizing the reasoning to derive the answer. This is much more flexible than processing the entire video all at once.

### Loss & Training

VideoAgent is a **zero-shot framework** requiring no task-specific training:
- Each component utilizes off-the-shelf pre-trained models: the video captioning model generates captions, CLIP computes features and performs object re-identification, and the video LLM conducts visual QA.
- Tool-use by the LLM is achieved via prompt engineering (in-context learning), without fine-tuning.
- The only critical design decisions are the prompt construction and the writing of tool descriptions.

## Key Experimental Results

### Main Results

Comparison on long video QA benchmarks (NExT-QA and EgoSchema):

| Method | Type | NExT-QA Acc (%) | EgoSchema Acc (%) |
|------|------|----------------|-------------------|
| InternVideo | End-to-end | 60.0 | 32.1 |
| Video-ChatGPT | End-to-end | 54.4 | 36.0 |
| SeViLA | End-to-end | 73.4 | 25.7 |
| LLoVi (GPT-3.5) | Agent | 67.7 | 50.3 |
| Gemini 1.5 Pro | Closed-source End-to-end | — | 63.2 |
| **VideoAgent** | **Agent** | **~71.3** | **~60.2** |
| vs Best Open-source Baseline Gain | — | **+6.6 (avg)** | **+26.0** |

Key Findings:
- VideoAgent achieves a massive gain of +26.0% on EgoSchema (egocentric videos lasting several minutes), demonstrating that structured memory is highly beneficial for long video understanding.
- It approaches the performance of the closed-source Gemini 1.5 Pro (EgoSchema 60.2 vs 63.2) while relying entirely on open-source models.
- The improvement on NExT-QA is relatively modest (+6.6%), as NExT-QA contains shorter videos where the advantages of structured memory are not fully realized.

### Ablation Study

Ablation analysis of memory components and tools:

| Configuration | NExT-QA | EgoSchema | Description |
|------|---------|-----------|------|
| Full VideoAgent | **Highest** | **Highest** | All components |
| w/o Object Memory | Decline | Decline | Inability to answer object-related questions |
| w/o Segment Localization | Decline | Significant decline | Localization ability is critical in long videos |
| w/o Caption Retrieval | Significant decline | Significant decline | Event descriptions are the core information source |
| VQA Tool Only | Lowest | Lowest | Degenerates into frame-by-frame QA |

### Key Findings

- **Caption Retrieval is the most critical tool**: Performance drops the most without it, indicating that pre-generated event descriptions are the primary information source for the Agent to understand the video.
- **Object Memory is crucial for person/object-related questions**: Its impact is particularly prominent in EgoSchema, which involves a high volume of hand actions and object interactions.
- **Segment Localization is indispensable in long videos**: It helps the Agent quickly locate relevant temporal windows, avoiding searching for a needle in a haystack across the entire video.
- **2-second clip granularity**: This is discovered to be the optimal balance point in experiments—shorter segments increase the number of captions leading to redundancy, while longer segments lose fine-grained event information.
- **Multi-step reasoning is significantly better than single-step**: The Agent requires 2-4 tool calls on average to answer a question, indicating that questions indeed demand multi-step information aggregation.

## Highlights & Insights

- **Structured memory is a key differentiator for video agents**: Unlike directly feeding video frames to LLMs, VideoAgent first translates the video into structured textual representations (event descriptions + object database), enabling the LLM to retrieve information efficiently using natural language and SQL queries. This "understand-then-retrieve" paradigm is more suitable for long video scenarios than "directly watching the video".
- **The re-identification method is simple yet effective**: Utilizing CLIP features for cross-frame object matching avoids training a dedicated tracker. Maintaining an object ID table in an SQL database makes querying and updating object memory highly standardized.
- **Practical significance of a zero-shot framework**: It requires no training data and directly integrates off-the-shelf models, resulting in low deployment costs. This plug-and-play design allows the framework to naturally improve as the underlying component models upgrade.

## Limitations & Future Work

- **Memory construction quality depends on foundation models**: The accuracy of event descriptions depends on the video captioning model, and object detection/tracking relies on detector performance. Errors from foundation models accumulate in memory and cannot be corrected.
- **Fixed 2-second granularity lacks flexibility**: Different video content densities vary significantly (e.g., action scenes vs. static dialogues), making static granularity ineffective. Adaptive segmentation strategies could be considered.
- **Lack of spatial relationship modeling**: Object memory only records object categories and appearing times without spatial coordinates (e.g., "the person on the left", "the cup on the table"), limiting spatial reasoning capabilities.
- **Tool invocation efficiency issues**: Each step of multi-step reasoning requires calling the LLM, leading to high latency. This is less practical for real-time or near-real-time applications.
- **Limited to QA tasks**: The current framework is designed for video QA and does not cover other video understanding tasks such as video generation, editing, or summarization.

## Related Work & Insights

- **vs LLoVi**: LLoVi also uses LLMs to process video descriptions but lacks object-level memory and interactive tool usage, essentially boiling down to "answering QA after reading captions". VideoAgent's structured memory and multi-tool interaction enable it to handle more complex reasoning.
- **vs Video-ChatGPT / VideoChat**: End-to-end video LLMs directly process video frames and are limited by their context windows. VideoAgent bypasses this limitation via memory buffering, though this introduces additional overhead for memory construction.
- **vs Gemini 1.5 Pro**: Gemini has an ultra-long context window capable of directly processing entire videos, but it is closed-source. VideoAgent, as an open-source alternative, achieves comparable performance, demonstrating the potential of the Agent paradigm.
- **Takeaways**: The concept of memory-augmented Agents can be generalized to other long-sequence understanding tasks (long-document QA, audio understanding, etc.). The core idea of "structure first, retrieve on-demand" essentially extends the effective context of LLMs using external storage.

## Rating

- Novelty: ⭐⭐⭐⭐ The dual-memory design of structured memory (temporal + object) is intuitive and effective, with innovative aspects in the re-ID method.
- Experimental Thoroughness: ⭐⭐⭐⭐ Verified on two major benchmarks, NExT-QA and EgoSchema, with ablation studies and case analyses.
- Writing Quality: ⭐⭐⭐⭐ The framework diagram is clear, and the case studies are straightforward and easy to understand.
- Value: ⭐⭐⭐⭐⭐ The Agent + memory paradigm has a profound impact on long video understanding; the +26% improvement on EgoSchema is highly significant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Agent3D-Zero: An Agent for Zero-shot 3D Understanding](agent3d-zero_an_agent_for_zero-shot_3d_understanding.md)
- [\[CVPR 2026\] Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding](../../CVPR2026/llm_agent/think_then_verify_a_hypothesis-verification_multi-agent_framework_for_long_video.md)
- [\[CVPR 2026\] Symphony: A Cognitively-Inspired Multi-Agent System for Long-Video Understanding](../../CVPR2026/llm_agent/symphony_a_cognitively-inspired_multi-agent_system_for_long-video_understanding.md)
- [\[CVPR 2026\] HAVEN: Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search](../../CVPR2026/llm_agent/haven_hierarchical_long_video_understanding_with_audiovisual_entity_cohesion.md)
- [\[CVPR 2026\] SciEducator: Scientific Video Understanding and Educating via Deming-Cycle Multi-Agent System](../../CVPR2026/llm_agent/scieducator_scientific_video_understanding_and_educating_via_deming-cycle_multi-.md)

</div>

<!-- RELATED:END -->
