---
title: >-
  [Paper Note] Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools
description: >-
  [ACL 2025][LLM Agent][agentic reasoning] Agentic Reasoning proposes a framework that integrates three agent tools—Web search, code execution, and knowledge-graph-based memory (Mind-Map)—into the LLM reasoning process. It improves the accuracy of DeepSeek-R1 on Humanity's Last Exam from 9.4% to 23.8% (+14.4%) and GPQA from 71.5% to 81.2%, approaching the performance level of OpenAI Deep Research.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "agentic reasoning"
  - "tool-use"
  - "mind-map"
  - "knowledge graph"
  - "web search"
  - "DeepSeek-R1"
date: 2026-05-08
content_hash: 10c62eb88e86216c
---

# Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools

**Conference**: ACL 2025  
**arXiv**: [2502.04644](https://arxiv.org/abs/2502.04644)  
**Code**: [https://github.com/theworldofagents/Agentic-Reasoning](https://github.com/theworldofagents/Agentic-Reasoning)  
**Area**: LLM Agent / LLM Reasoning  
**Keywords**: agentic reasoning, tool-use, mind-map, knowledge graph, web search, DeepSeek-R1

## TL;DR
Agentic Reasoning proposes a framework that integrates three agent tools—Web search, code execution, and knowledge-graph-based memory (Mind-Map)—into the LLM reasoning process. It improves the accuracy of DeepSeek-R1 on Humanity's Last Exam from 9.4% to 23.8% (+14.4%) and GPQA from 71.5% to 81.2%, approaching the performance level of OpenAI Deep Research.

## Background & Motivation

**Background**: Reasoning models such as DeepSeek-R1 and OpenAI o1 achieve long-chain reasoning through reinforcement learning, performing exceptionally well on verifiable tasks like mathematics and coding, but they remain limited by their internal knowledge on knowledge-intensive, open-ended questions.

**Limitations of Prior Work**:
   - Reasoning models perform poorly in domains requiring external knowledge, such as social sciences, medicine, and finance.
   - Existing search-in-reasoning methods (e.g., SearchO1) employ insufficiently fine-grained search strategies, resulting in unstable quality of retrieved information.
   - Long-chain reasoning is prone to "forgetting" key information from previous steps, making it difficult to maintain reasoning consistency.

**Key Challenge**: Strong reasoning capability but limited knowledge vs. requirement for external knowledge but tool integration potentially disrupting reasoning coherence.

**Goal**: Enable reasoning models to seamlessly call external tools (search, code, memory) during the reasoning process without interrupting the reasoning chain.

**Key Insight**: Humans rely on external tools (search engines, calculators, mind maps) to solve complex problems, and LLM reasoning has similar needs.

**Core Idea**: Three Agent tools—Web-Search (knowledge acquisition), Code (computational analysis), and Mind-Map (structured memory)—are dynamically invoked during reasoning, with Mind-Map maintaining the reasoning context via a knowledge graph.

## Method

### Overall Architecture
The reasoning LLM inserts special tokens (`<web_search>`/`<code>`/`<mind_map>`) into the reasoning sequence → reasoning is paused upon token detection → the query is extracted and sent to the corresponding Agent → the Agent returns results which are inserted back into the reasoning chain → reasoning resumes → iterate until the final answer is obtained.

### Key Designs

1. **Mind-Map Agent (Knowledge Graph Memory)**:

    - **Function**: Builds a structured knowledge graph in real-time from the reasoning process to serve as the "external memory" of reasoning.
    - **Mechanism**: Uses a graph-construction LLM to extract entities and semantic relations from the reasoning chain → constructs a knowledge graph → performs community clustering → generates summaries for each cluster.
    - **Two Functions**: (1) Provides reasoning context (synthesis of cluster summaries) to other Agents to make searching and coding more precise. (2) When the reasoning model loses its way in a long chain, it queries the Mind-Map to retrieve previous reasoning results.
    - **Design Motivation**: Resolves the "forgetting" issue in long-chain reasoning—as the reasoning chain grows, information from earlier steps becomes increasingly susceptible to being lost.

2. **Web-Search Agent (Intelligent Search)**:

    - **Function**: A four-step pipeline—query decomposition → search → reranking → RAG.
    - **Mechanism**: (1) Original query + Mind-Map context → the LLM decomposes it into multiple search-engine-friendly sub-queries. (2) Bing retrieves top-20 pages. (3) Cohere Rerank 3.5 reranks results; if the average relevance is <0.7, the query is iteratively optimized. (4) RAG is performed on highly relevant pages to extract info. (5) The LLM synthesizes several sub-query results into a natural language snippet.
    - **Design Motivation**: Directly using the reasoning model's raw queries for search yields poor results; context-aware query decomposition and quality control are essential.

3. **Code Agent (Code Execution)**:

    - **Function**: Delegates computational tasks to a specialized coding LLM (Claude 3.5 Sonnet) to prevent the reasoning model from writing code itself.
    - **Mechanism**: The reasoning model sends a task description + Mind-Map context → the coding LLM generates and executes the code → returns natural language results.
    - **Design Motivation**: The reasoning model's attention should center on reasoning; decoupling coding tasks helps maintain longer and more coherent reasoning chains.

## Key Experimental Results

### Main Results

| Benchmark | DeepSeek-R1 | + Agentic Reasoning | Gain |
|------|------------|-------------------|------|
| Humanity's Last Exam | 9.4% | **23.8%** | +14.4% |
| GPQA (All) | 71.5% | **81.2%** | +9.7% |
| GAIA (Avg) | - | **66.13%** | - |
| OpenAI Deep Research (HLE) | 26.6% | - | Only 2.8% diff |

### GPQA by Subject

| Subject | DeepSeek-R1 | Agentic Reasoning | o3-mini-high |
|------|------------|-------------------|-------------|
| Physics | 86.8 | **94.5** | - |
| Chemistry | 56.1 | **73.7** | - |
| Biology | 63.8 | **80.5** | - |
| All | 71.5 | **81.2** | 79.7 |

### Ablation Study

| Configuration | HLE Accuracy |
|------|-------------|
| Full (Search + Code + Mind-Map) | **23.8%** |
| w/o Mind-Map | 19.2% (-4.6%) |
| w/o Code | 21.5% (-2.3%) |
| w/o Search | 12.1% (-11.7%) |
| Base R1 (no tools) | 9.4% |

### Key Findings
- **Web-Search contributes the most** (+11.7%): Search is the most critical external tool in knowledge-intensive questions.
- **Mind-Map contributes significantly** (+4.6%): Structured memory is vital for maintaining consistency in long-chain reasoning.
- **Surpassing o3-mini-high on GPQA**: An open-source model outperforms OpenAI's strongest reasoning model on PhD-level QA for the first time.
- **Most significant gains in Chemistry and Biology**: These two subjects have the highest demand for external knowledge retrieval and fact verification.
- **Approaching OpenAI Deep Research on GAIA Level 3 tasks**: Trailing by only 2.14%.

## Highlights & Insights
- **Mind-Map is the primary technical innovation of this work**: Using a knowledge graph as the "structured working memory" of the reasoning process is more effective than simple dialogue histories because the knowledge graph preserves the relational structure between entities. This can be transferred to any scenario requiring long-chain reasoning.
- **Search strategy with query decomposition + quality control**: Instead of a simple single-pass search, the pipeline performs query decomposition → search → reranking → quality thresholding → iterative optimization. This workflow is directly reusable.
- **Decoupled "reasoning for reasoning, coding for coding" design**: Avoids distracting the reasoning model with coding tasks, maintaining reasoning coherence. This design philosophy is highly practical.

## Limitations & Future Work
- **High computational cost**: Involving multiple calls to DeepSeek-V3 (search) + Claude 3.5 (coding) + DeepSeek-R1 (reasoning); reasoning through a single question can require numerous LLM calls.
- **Mind-Map knowledge graph construction quality depends on the auxiliary LLM**: If the graph construction is inaccurate, the memory can be counterproductive.
- **Tested only on English benchmarks**: Multilingual capabilities remain unverified.
- **Pass@1 results**: Multiple sampling runs might further improve performance.

## Related Work & Insights
- **vs. SearchO1 (Li et al., 2025)**: SearchO1 only integrates search, whereas Agentic Reasoning additionally incorporates Mind-Map and Code, performing approximately 6.6% higher on GPQA.
- **vs. OpenAI Deep Research**: This work approaches the performance of the commercial system using open-source models, trailing by only 2.8% on HLE.
- **vs. GraphRAG**: GraphRAG constructs a knowledge graph during the indexing phase for retrieval, whereas the Mind-Map in this work is constructed dynamically during the reasoning process for memory—differing in purpose and timing.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of Mind-Map knowledge graph memory is novel, and the three-Agent joint design is well-rationalized.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage of HLE + GPQA + GAIA + deep research tasks, with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear, and the case studies are vivid.
- Value: ⭐⭐⭐⭐⭐ Open-source solution approaching the level of OpenAI Deep Research, offering extremely high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] OctoTools: An Agentic Framework with Extensible Tools for Complex Reasoning](../../ACL2026/llm_agent/octotools_an_agentic_framework_with_extensible_tools_for_complex_reasoning.md)
- [\[ACL 2025\] Table-Critic: A Multi-Agent Framework for Collaborative Criticism and Refinement in Table Reasoning](table_critic_multi_agent.md)
- [\[ACL 2025\] Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models](theorem-of-thought_a_multi-agent_framework_for_abductive_deductive_and_inductive.md)
- [\[ACL 2025\] LLM Agents Making Agent Tools](llm_agents_making_agent_tools.md)
- [\[ACL 2025\] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?](repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_research_claims.md)

</div>

<!-- RELATED:END -->
