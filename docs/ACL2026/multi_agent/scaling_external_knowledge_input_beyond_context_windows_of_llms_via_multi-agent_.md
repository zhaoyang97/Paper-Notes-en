---
title: >-
  [Paper Note] Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration
description: >-
  [ACL 2026][Multi-Agent][Context Window Extension] The ExtAgents multi-agent framework is proposed to address the performance bottleneck where existing multi-agent methods degrade when scaling external knowledge beyond th…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "Context Window Extension"
  - "Multi-Agent Collaboration"
  - "External Knowledge Scaling"
  - "Multi-hop QA"
  - "Knowledge Synchronization"
date: 2026-05-08
content_hash: 42d8eede1bb483b1
---

# Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration

**Conference**: ACL 2026  
**arXiv**: [2505.21471](https://arxiv.org/abs/2505.21471)  
**Code**: [GitHub](https://github.com/THUNLP-MT/ExtAgents)  
**Area**: LLM Agent  
**Keywords**: Context Window Extension, Multi-Agent Collaboration, External Knowledge Scaling, Multi-hop QA, Knowledge Synchronization

## TL;DR

The ExtAgents multi-agent framework is proposed to address the performance bottleneck where existing multi-agent methods degrade when scaling external knowledge beyond the context window. It achieves significant improvements in multi-hop QA and long survey generation through two mechanisms: Global Knowledge Synchronization (information exchange among all Seeking Agents) and Accumulative Reasoning (gradually injecting filtered knowledge into the Reasoning Agent).

## Background & Motivation

**Background**: With advancements in post-training inference and information retrieval (IR), LLMs can integrate more retrieved knowledge within their context windows to solve complex tasks, where more knowledge typically leads to better performance.

**Limitations of Prior Work**: When external knowledge exceeds the context window, direct truncation leads to information loss; RAG is limited by ranking errors and may miss critical evidence; context compression discards subtle cues. Multi-agent distributed approaches (e.g., LLM×MapReduce) represent a new paradigm, but experiments show their performance decreases as knowledge volume increases.

**Key Challenge**: Existing multi-agent orchestration suffers from two bottlenecks: (1) Small knowledge synchronization bandwidth, where each agent can only access 2 neighbor messages, requiring multiple rounds for global synchronization; (2) Reasoning context redundancy, where cramming all messages into the reasoning agent leads to information overload.

**Goal**: Design a scalable multi-agent framework that allows task performance to continuously improve as external knowledge input scales, even beyond the context window.

**Key Insight**: Simplify agent roles into two categories (Seeking + Reasoning) and design global synchronization and accumulative reasoning mechanisms specifically for the two bottlenecks.

**Core Idea**: Seeking Agents exchange information globally and score chunk relevance (Bandwidth = $N$). The Reasoning Agent performs accumulative reasoning by gradually increasing top-k knowledge (top-$2^s$) across rounds to avoid one-time information overload.

## Method

### Overall Architecture

ExtAgents divides the input into $N$ chunks, each assigned to a Seeking Agent. Process: (1) Global Knowledge Synchronization—all Seeking Agents share messages and score the relevance of their chunks to the query; (2) Knowledge Accumulative Reasoning—the Reasoning Agent starts with the most relevant chunks and doubles the amount of knowledge each round (top-$2^s$), accumulating information until the question can be answered or knowledge is exhausted. Highly parallelizable.

### Key Designs

1.  **Global Knowledge Synchronization (Bandwidth = $N$)**:
    - **Function**: Allows each agent to access information from all other agents.
    - **Mechanism**: Messages from all Seeking Agents are shared globally. Each agent digests its local chunk and evaluates its relevance to the query. Compared to Chain of Agents (Bandwidth = 2) and LLM×MapReduce (Bandwidth = $O(L/|m|)$), the bandwidth equals the number of agents $N$, achieving true global information exchange. Irrelevant chunks can optionally be excluded.
    - **Design Motivation**: Small bandwidth requires multiple synchronization rounds, causing information degradation during transfer; global access completes synchronization in one step.

2.  **Knowledge Accumulative Reasoning**:
    - **Function**: Avoids one-time information overload by progressively integrating knowledge.
    - **Mechanism**: The Reasoning Agent receives chunks sorted by relevance. In round $s$, it receives messages from the top-$2^s$ chunks. After each round, it determines if the question can be answered; if not, it expands the knowledge volume and continues reasoning. It ultimately outputs an answer or "Unable to answer."
    - **Design Motivation**: Injecting all messages at once (like LLM×MapReduce) causes redundant information to drown out critical evidence; progressive injection allows the reasoning agent to focus on the most relevant information in each round.

3.  **$\infty$Bench+ Enhanced Benchmark**:
    - **Function**: Eliminates bias in existing long-context benchmarks.
    - **Mechanism**: Filters out samples that can be answered by scanning only an 8k token window, retaining multi-hop questions that truly require cross-document information aggregation. Results: En.QA reduced from 351 samples to 157 + large document samples = 294; Zh.QA reduced from 189 to 56 + large document samples = 184.
    - **Design Motivation**: It was discovered that a large number of questions in the original $\infty$Bench could be answered by simply truncating the context, failing to truly test long-context capabilities.

## Key Experimental Results

### Main Results ($\infty$Bench+ En.QA, gpt-4o-mini)

| Method | 8k input | 32k input | 128k input | 256k+ input |
| :--- | :--- | :--- | :--- | :--- |
| Truncation | ~30 | ~35 | ~38 | N/A |
| LLM×MapReduce | ~32 | ~33 | ~34 | ~32 |
| ExtAgents | ~33 | ~38 | ~43 | **~46** |

### Key Findings
- ExtAgents is the only method where performance continuously improves as knowledge volume increases, even beyond the 128k context window.
- LLM×MapReduce performance drops below truncation after exceeding the context window, exposing its bottleneck.
- ExtAgents is equally effective on HotpotQA (multi-hop QA with large knowledge bases), verifying its generalization.
- It also demonstrates advantages in long survey generation tasks.
- High parallelism ensures efficiency—Seeking Agents are fully parallelizable.

## Highlights & Insights
- **Valuable Problem Definition**: For the first time, the problem of "scaling external knowledge input beyond the context window" is explicitly defined and an evaluation framework is constructed.
- **Precise Bottleneck Analysis**: Attributes the failure of existing methods to two specific bottlenecks: synchronization bandwidth and reasoning redundancy.
- **Simple and Effective Design**: Only two types of agents and two mechanisms make it easy to understand and implement.
- **Independent Value of $\infty$Bench+**: Eliminates measurement bias in existing long-context benchmarks.

## Limitations & Future Work
- **Dependency on LLM APIs**: Requires multiple LLM calls, leading to higher costs.
- **Simple Chunking Strategy**: Uses basic splitting; more intelligent semantic chunking has not been explored.
- **Limited Evaluation Coverage**: Primarily verified on QA and survey generation; other long-context tasks remain untested.
- Future Directions: Intelligent chunking strategies, combination with RAG, and post-training agent collaboration capabilities.

## Related Work & Insights
- **vs LLM×MapReduce**: SOTA multi-agent method but performance fails to scale with knowledge; ExtAgents overcomes this via global synchronization and accumulative reasoning.
- **vs Chain of Agents**: A sequential method with Bandwidth = 2, possessing poor scalability.
- **vs RAG**: Limited by retrieval ranking errors, which cannot guarantee the selection of critical evidence.

## Rating
- Novelty: ⭐⭐⭐⭐ Valuable problem definition; the dual design of global synchronization and accumulative reasoning is highly targeted.
- Experimental Thoroughness: ⭐⭐⭐⭐ Verified across multiple tasks and models; includes $\infty$Bench+ construction and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear formal definitions and systematic bottleneck analysis.
- Value: ⭐⭐⭐⭐ Provides a practical training-free solution for LLM ultra-long context reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs](topology_matters_measuring_memory_leakage_in_multi-agent_llms.md)
- [\[ACL 2026\] Multi-Agent Reasoning Improves Compute Efficiency: Pareto-Optimal Test-Time Scaling](multi-agent_reasoning_improves_compute_efficiency_pareto-optimal_test-time_scali.md)
- [\[ACL 2026\] ConSensus: Multi-Agent Collaboration for Multimodal Sensing](consensus_multi-agent_collaboration_for_multimodal_sensing.md)
- [\[ACL 2026\] PosterForest: Hierarchical Multi-Agent Collaboration for Scientific Poster Generation](posterforest_hierarchical_multi-agent_collaboration_for_scientific_poster_genera.md)
- [\[ACL 2026\] LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey](llm-based_human-agent_collaboration_and_interaction_systems_a_survey.md)

</div>

<!-- RELATED:END -->
