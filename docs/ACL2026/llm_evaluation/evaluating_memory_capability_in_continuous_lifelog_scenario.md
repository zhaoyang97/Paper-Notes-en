---
title: >-
  [Paper Note] Evaluating Memory Capability in Continuous Lifelog Scenario
description: >-
  [ACL 2026][LLM Evaluation][Lifelog Memory] This paper proposes LifeDialBench, a benchmark for evaluating memory capabilities in continuous lifelog scenarios (comprising EgoMem with 7 days of real data and LifeMem with 1…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Lifelog Memory"
  - "Online Evaluation"
  - "Wearable Devices"
  - "RAG Baseline"
  - "Long-term Dialogue"
date: 2026-05-08
content_hash: 594539b66456ec50
---

# Evaluating Memory Capability in Continuous Lifelog Scenario

**Conference**: ACL 2026  
**arXiv**: [2604.11182](https://arxiv.org/abs/2604.11182)  
**Code**: [https://github.com/RayNeo-AI-2025/LifeDialBench](https://github.com/RayNeo-AI-2025/LifeDialBench)  
**Area**: LLM Evaluation  
**Keywords**: Lifelog Memory, Online Evaluation, Wearable Devices, RAG Baseline, Long-term Dialogue

## TL;DR
This paper proposes LifeDialBench, a benchmark for evaluating memory capabilities in continuous lifelog scenarios (comprising EgoMem with 7 days of real data and LifeMem with 1 year of simulation). It introduces an online evaluation protocol to ensure temporal causality and counterintuitively finds that simple RAG baselines consistently outperform complex memory systems.

## Background & Motivation

**Background**: Wearable devices (e.g., Ray-Ban Meta, Xiaomi AI glasses) can now support "always-on" microphones to continuously record ambient conversations, creating significant opportunities for memory system applications. LLM memory systems typically include memory managers, summarization agents, and retrievers.

**Limitations of Prior Work**: Existing memory benchmarks primarily focus on one-on-one chats or human-AI interactions, ignoring the unique requirements of continuous lifelogs—multi-party interactions, casual and sequential event timelines, and simulated social networks. More critically, traditional offline evaluation protocols suffer from "temporal leakage," allowing systems to access the complete dataset before answering any question, which systematically overestimates real-world performance.

**Key Challenge**: Existing complex memory systems (e.g., graph-based, hierarchical) introduce lossy compression (summarization, entity extraction, etc.) that may discard detailed information critical in lifelog scenarios. However, due to the lack of strict online evaluation protocols, this information loss is masked by the temporal leakage in offline evaluations.

**Goal**: (1) Construct a memory evaluation benchmark matching the characteristics of continuous lifelogs; (2) Propose an online evaluation protocol that follows temporal causality; (3) Reveal the true capabilities of existing memory systems.

**Key Insight**: The authors utilize the EgoLife real-world first-person video dataset (6 people, 7 days) to construct real-scene data while simulating one year of life with LLMs to extend the temporal span. They introduce strict online evaluation—where information flows in linearly over time, and the system can only answer using information available "before the current time point."

**Core Idea**: Evaluating memory systems under strict temporal causality reveals a counterintuitive finding—simple RAG baselines outperform all complex dedicated memory systems because raw text preservation is more vital than lossy compression.

## Method

### Overall Architecture
LifeDialBench contains two complementary subsets: (1) EgoMem—based on the real EgoLife dataset, constructed through bottom-up hierarchical summarization; (2) LifeMem—simulated via LLM for one year of life, constructed through top-down hierarchical expansion. Both use multi-level event summarization to generate QA pairs and support the online evaluation protocol.

### Key Designs

1.  **Hierarchical Life Simulation Framework**:
    - **Function**: Generate long-span, scene-rich multi-party continuous dialogue logs.
    - **Mechanism**: EgoMem adopts a bottom-up approach—from second-level video segments $\rightarrow$ minute-level summaries $\rightarrow$ hourly $\rightarrow$ daily $\rightarrow$ weekly summaries. LifeMem uses a top-down approach—designing an annual outline via LLM $\rightarrow$ monthly plans $\rightarrow$ daily events $\rightarrow$ specific dialogues, simulating a full year of life with multi-party interactions. Qwen3-235B-Instruct is used for all dialogue and summary generation.
    - **Design Motivation**: EgoMem provides real-world grounding, while LifeMem provides long temporal spans and scene diversity; the two are complementary.

2.  **Online Evaluation Protocol**:
    - **Function**: Eliminate temporal leakage in offline evaluation and ensure the evaluation reflects real-world conditions.
    - **Mechanism**: Strictly follows temporal linearity—the system starts from an empty state and receives dialogue data progressively in chronological order. Upon reaching an evaluation point with a query timestamp, the system only uses information stored before that time point to answer. Information is updated incrementally, and evaluations occur intermittently during storage.
    - **Design Motivation**: Traditional offline evaluation grants the system a "God's eye view," allowing it to reference future information. Online evaluation eliminates this unfair advantage to simulate real deployment.

3.  **Multi-dimensional Query Design**:
    - **Function**: Comprehensively probe memory retrieval capabilities at different granularities.
    - **Mechanism**: Designs three types of queries—(a) Temporal Localization: determining when events occurred; (b) Fact Retrieval: recalling specific details; (c) Compositional Reasoning: association across events. QA pairs are generated from multi-level summaries to cover various temporal granularities.
    - **Design Motivation**: Lifelog queries require integrated capabilities in temporal reasoning, cross-event association, and detail recall beyond simple fact retrieval.

### Loss & Training
As a benchmark paper, no model training is involved. Four representative memory systems are evaluated: simple RAG baselines, summarization-compression methods, graph-structured methods, and hierarchical memory methods.

## Key Experimental Results

### Main Results

| Memory System | EgoMem | LifeMem | Description |
| :--- | :--- | :--- | :--- |
| Simple RAG | **Highest** | **Highest** | Simple retrieval of raw text |
| Summarization | Lower than RAG | Lower than RAG | Lossy compression loses details |
| Graph-based | Lower than RAG | Lower than RAG | Over-engineering is detrimental |
| Hierarchical | Lower than RAG | Lower than RAG | Complex but ineffective structure |

### Ablation Study

| Evaluation Mode | Effect Difference | Description |
| :--- | :--- | :--- |
| Online Evaluation | Scores drop for all systems | Performance decreases after eliminating temporal leakage |
| Offline Evaluation | Generally higher | Significant temporal leakage present |
| Rank Changes | Ranking reversals exist | Offline evaluation may misjudge system quality |

### Key Findings
- Counterintuitive conclusion: Simple RAG baselines consistently outperform all complex memory systems, including advanced graph-structured and hierarchical methods.
- Lossy compression (summarization, entity extraction) is more harmful than beneficial in lifelog scenarios—detail preservation is more important than structural abstraction.
- Temporal retrieval is a universal bottleneck—"when it happened" is significantly harder than "what happened."
- Online evaluation reveals gaps masked by offline tests; systems performing well offline may degrade significantly in online scenarios.
- Current design directions may have fundamental misjudgments; high-fidelity context preservation is currently more important than intelligent compression.

## Highlights & Insights
- **Importance of Online Evaluation**: Revealed temporal leakage in offline evaluation, which offers broad inspiration for all time-series AI assessments.
- **Simple yet Effective**: Elaborately designed complex memory systems were inferior to simple RAG, indicating that data fidelity is currently more critical than structural abstraction.
- **Wearable Scenario Foresight**: As smart glasses proliferate, continuous lifelogs will become a major AI application; this benchmark provides the necessary evaluation infrastructure.

## Limitations & Future Work
- LifeMem dialogues are LLM-synthesized and may not fully reflect the chaos of real-world interactions.
- EgoMem is limited to 7 days and 6 participants, lacking long-term demographic diversity.
- Simple RAG may face efficiency issues when data volume spans several years.
- Multimodal memory (incorporating visual data) has not yet been evaluated.

## Related Work & Insights
- **vs LoCoMo**: Focuses on human-human dialogue but lacks continuous recording and online evaluation. LifeDialBench is closer to real scenarios.
- **vs LongMemEval**: Human-AI interaction scenario with many sessions but lacks multi-party and continuous characteristics.
- **vs MemBank**: Only covers 10 days of human-AI interaction with a single scenario. LifeDialBench covers 1-year multi-party scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] StratMem-Bench: Evaluating Strategic Memory Use in Virtual Character Conversation Beyond Factual Recall](stratmem-bench_evaluating_strategic_memory_use_in_virtual_character_conversation.md)
- [\[ACL 2026\] Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language](exploring_the_capability_boundaries_of_llms_in_mastering_of_chinese_chouxiang_la.md)
- [\[ACL 2026\] SCAN: Structured Capability Assessment and Navigation for LLMs](scan_structured_capability_assessment_and_navigation_for_llms.md)
- [\[ACL 2026\] HoWToBench: Holistic Evaluation for LLM's Capability in Human-level Writing using Tree of Writing](howtobench_holistic_evaluation_for_llms_capability_in_human-level_writing_using_.md)
- [\[NeurIPS 2025\] MEMTRACK: Evaluating Long-Term Memory and State Tracking in Multi-Platform Dynamic Agent Environments](../../NeurIPS2025/llm_evaluation/memtrack_evaluating_long-term_memory_and_state_tracking_in_multi-platform_dynami.md)

</div>

<!-- RELATED:END -->
