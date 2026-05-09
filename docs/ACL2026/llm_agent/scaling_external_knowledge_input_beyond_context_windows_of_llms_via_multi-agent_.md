---
title: >-
  [Paper Note] Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration
description: >-
  [ACL 2026][LLM Agent][Context window extension] This paper proposes ExtAgents, a multi-agent framework that addresses the performance degradation observed in existing multi-agent methods when scaling external knowledge input beyond the context window. It introduces two mechanisms—global knowledge synchronization (information exchange across all Seeking Agents) and knowledge-accumulative reasoning (progressively injecting filtered knowledge into the Reasoning Agent)—achieving significant improvements on multi-hop QA and long survey generation tasks.
tags:
  - ACL 2026
  - LLM Agent
  - Context window extension
  - multi-agent collaboration
  - external knowledge scaling
  - multi-hop QA
  - knowledge synchronization
date: 2026-05-08
content_hash: 6a1679fc655d7479
---

# Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration

**Conference**: ACL 2026
**arXiv**: [2505.21471](https://arxiv.org/abs/2505.21471)
**Code**: [GitHub](https://github.com/THUNLP-MT/ExtAgents)
**Area**: LLM Agent
**Keywords**: Context window extension, multi-agent collaboration, external knowledge scaling, multi-hop QA, knowledge synchronization

## TL;DR

This paper proposes ExtAgents, a multi-agent framework that addresses the performance degradation observed in existing multi-agent methods when scaling external knowledge input beyond the context window. It introduces two mechanisms—global knowledge synchronization (information exchange across all Seeking Agents) and knowledge-accumulative reasoning (progressively injecting filtered knowledge into the Reasoning Agent)—achieving significant improvements on multi-hop QA and long survey generation tasks.

## Background & Motivation

**State of the Field**: With advances in post-training reasoning and information retrieval, LLMs can integrate increasingly large amounts of retrieved knowledge within their context windows to address complex tasks, where more knowledge generally yields better performance.

**Limitations of Prior Work**: When the volume of external knowledge exceeds the context window, direct truncation causes information loss; RAG suffers from ranking errors that may omit critical evidence; context compression discards subtle cues. Multi-agent distributed methods (e.g., LLM×MapReduce) represent an emerging paradigm, but experiments reveal that their performance degrades rather than improves as knowledge volume increases.

**Root Cause**: Existing multi-agent orchestration suffers from two bottlenecks: (1) limited knowledge synchronization bandwidth, where each agent can only access messages from two neighbors, requiring multiple rounds to propagate global information; and (2) redundant reasoning context, where injecting all messages into the Reasoning Agent causes information overload.

**Paper Goals**: To design a scalable multi-agent framework in which task performance continues to improve as external knowledge input grows, even when it exceeds the context window.

**Starting Point**: Simplifying agent roles into two types (Seeking and Reasoning), and designing global synchronization and accumulative reasoning mechanisms to address each bottleneck separately.

**Core Idea**: Seeking Agents globally exchange and score chunk relevance (bandwidth = $N$); the Reasoning Agent performs accumulative reasoning by progressively incorporating top-$k$ knowledge across multiple rounds, avoiding one-shot information overload.

## Method

### Overall Architecture

ExtAgents partitions the input into $N$ chunks, each assigned to a Seeking Agent. The pipeline consists of: (1) **global knowledge synchronization**—all Seeking Agents share messages and score chunk relevance to the query; and (2) **knowledge-accumulative reasoning**—the Reasoning Agent starts from the most relevant chunks and doubles the knowledge quantity (top-$2^s$) each round, accumulating progressively until the question is answerable or knowledge is exhausted. The framework is highly parallelizable.

### Key Designs

1. **Global Knowledge Synchronization (bandwidth = $N$)**:

    - **Function**: Enables each agent to access information from all other agents.
    - **Mechanism**: Messages from all Seeking Agents are shared globally; each agent digests its local chunk and assesses its relevance to the query. Compared to Chain of Agents (bandwidth = 2) and LLM×MapReduce (bandwidth = $O(L/|m|)$), the bandwidth equals the number of agents $N$ directly, achieving true global information exchange. Irrelevant chunks may optionally be excluded.
    - **Design Motivation**: Low bandwidth requires multiple synchronization rounds, during which information degrades in transmission; global access completes synchronization in a single pass.

2. **Knowledge-Accumulative Reasoning**:

    - **Function**: Avoids one-shot information overload by progressively integrating knowledge.
    - **Mechanism**: The Reasoning Agent receives chunks sorted by relevance. At round $s$, it receives messages from the top-$2^s$ chunks. After each round of reasoning, it determines whether the question is answerable; if not, it expands the knowledge set and continues. The process terminates with a final answer or an "unanswerable" response.
    - **Design Motivation**: Injecting all messages at once (as in LLM×MapReduce) causes redundant information to overwhelm key evidence; progressive injection allows the Reasoning Agent to focus on the most relevant information at each round.

3. **∞Bench+ Enhanced Benchmark**:

    - **Function**: Eliminates bias in existing long-context benchmarks.
    - **Mechanism**: Samples answerable by scanning only an 8k token window are filtered out, retaining only multi-hop questions that genuinely require cross-document information aggregation. As a result, En.QA is reduced from 351 to 157 samples plus large-document samples (294 total), and Zh.QA from 189 to 56 plus large-document samples (184 total).
    - **Design Motivation**: A large proportion of questions in the original ∞Bench can be answered by simply truncating the context, and thus fail to genuinely test long-context capabilities.

## Key Experimental Results

### Main Results (∞Bench+ En.QA, gpt-4o-mini)

| Method | 8k input | 32k input | 128k input | 256k+ input |
|------|---------|----------|-----------|------------|
| Truncation | ~30 | ~35 | ~38 | N/A |
| LLM×MapReduce | ~32 | ~33 | ~34 | ~32 |
| ExtAgents | ~33 | ~38 | ~43 | **~46** |

### Key Findings
- ExtAgents is the only method whose performance continues to improve as knowledge volume increases, even beyond the 128k context window.
- LLM×MapReduce performs worse than truncation after exceeding the context window, exposing its bottlenecks.
- ExtAgents generalizes effectively to HotpotQA (multi-hop QA with large knowledge bases).
- Advantages are also demonstrated on long survey generation tasks.
- High parallelism ensures efficiency—Seeking Agents operate fully in parallel.

## Highlights & Insights
- **Clear and Valuable Problem Formulation**: This work is the first to explicitly define the problem of scaling external knowledge input beyond the context window and to construct a corresponding evaluation framework.
- **Precise Bottleneck Analysis**: The failures of existing methods are attributed to two concrete bottlenecks: synchronization bandwidth and reasoning redundancy.
- **Simple yet Effective Design**: Only two agent types and two mechanisms are used, making the framework easy to understand and implement.
- **Independent Value of ∞Bench+**: The construction of ∞Bench+ addresses measurement bias in existing long-context benchmarks.

## Limitations & Future Work
- **Reliance on LLM APIs**: Multiple LLM calls are required, leading to relatively high costs.
- **Simple Chunking Strategy**: Only basic splitting is employed; more intelligent semantic chunking strategies are not explored.
- **Limited Evaluation Coverage**: Validation is primarily conducted on QA and survey generation; other long-context tasks remain untested.
- Future directions include smarter chunking strategies, integration with RAG, and post-training of agent collaboration capabilities.

## Related Work & Insights
- **vs. LLM×MapReduce**: The prior SOTA multi-agent method whose performance degrades when scaling knowledge; ExtAgents overcomes this via global synchronization and accumulative reasoning.
- **vs. Chain of Agents**: A sequential method with bandwidth = 2 that scales poorly.
- **vs. RAG**: Constrained by retrieval ranking errors, with no guarantee that critical evidence is selected.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Valuable problem formulation; the dual design of global synchronization and accumulative reasoning is well-targeted.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-task and multi-model validation, with ∞Bench+ construction and efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Formal definitions are clear; bottleneck analysis is systematic.
- **Value**: ⭐⭐⭐⭐ — Provides a practical, training-free solution for LLM reasoning over extremely long contexts.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Creating ConLangs to Probe the Metalinguistic Grammatical Knowledge of LLMs](creating_conlangs_to_probe_the_metalinguistic_grammatical_knowledge_of_llms.md)
- [\[AAAI 2026\] KDR-Agent: A Multi-Agent LLM Framework for Multi-Domain Low-Resource In-Context NER via Knowledge Retrieval](../../AAAI2026/llm_agent/a_multi-agent_llm_framework_for_multi-domain_low-resource_in-context_ner_via_kno.md)
- [\[ACL 2026\] SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](silo-bench_a_scalable_environment_for_evaluating_distributed_coordination_in_mul.md)
- [\[ACL 2026\] StructMem: Structured Memory for Long-Horizon Behavior in LLMs](structmem_structured_memory_for_long-horizon_behavior_in_llms.md)
- [\[ACL 2026\] MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools](mcp-flow_facilitating_llm_agents_to_master_real-world_diverse_and_scaling_mcp_to.md)

<!-- RELATED:END -->
