---
title: >-
  [Paper Note] Select, Read, and Write: A Multi-Agent Framework of Full-Text-based Related Work Generation
description: >-
  [ACL 2025][LLM Agent][Related Work Generation] A collaborative three-agent framework, Select-Read-Write, is proposed. By employing graph-aware reading order decision-making and a shared working memory mechanism, it achieves automatic Related Work generation based on the full text of papers (rather than just abstracts). Consistent improvements are demonstrated across three base models (Llama3-8B, Claude-3-Haiku, and GPT-4o), with the Citation Graph strategy achieving the best…
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "Related Work Generation"
  - "Multi-Agent Framework"
  - "Full-Text Reading"
  - "Graph-Aware Selection"
  - "Working Memory"
date: 2026-05-08
content_hash: 74f0fae36d15f514
---

# Select, Read, and Write: A Multi-Agent Framework of Full-Text-based Related Work Generation

**Conference**: ACL 2025  
**arXiv**: [2505.19647](https://arxiv.org/abs/2505.19647)  
**Code**: [https://github.com/1190200817/Full_Text_RWG](https://github.com/1190200817/Full_Text_RWG)  
**Area**: LLM Agent / Academic Writing  
**Keywords**: Related Work Generation, Multi-Agent Framework, Full-Text Reading, Graph-Aware Selection, Working Memory

## TL;DR
A collaborative three-agent framework, Select-Read-Write, is proposed. By employing graph-aware reading order decision-making and a shared working memory mechanism, it achieves automatic Related Work generation based on the full text of papers (rather than just abstracts). Consistent improvements are demonstrated across three base models (Llama3-8B, Claude-3-Haiku, and GPT-4o), with the Citation Graph strategy achieving the best performance.

## Background & Motivation

**Background**: Automatic Related Work Generation (RWG) is an important task in academic writing assistance. Due to context window constraints, existing methods mostly rely on abstracts, Introduction/Conclusion, or retrieved snippets as input.

**Limitations of Prior Work**:
   - **Hallucinations due to Insufficient Information**: Confining the input to abstracts fails to capture methodological details and deep-level relationships between papers, leading to frequent misunderstandings and hallucinations by the model.
   - **Lack of Inter-paper Relationships**: Existing methods describe each reference independently, lacking comparison and correlation analysis between papers, which results in loosely structured Related Work Sections (RWS).
   - **Long-Text Processing Bottleneck**: The total tokens of multiple full-text papers easily exceed 70K+. Direct full-text insertion into LLMs deteriorates performance, even with a 128K window.

**Key Challenge**: Full-text level deep understanding is required to avoid hallucinations, yet complete full-text inputs exceed the effective processing limits of models; modeling inter-paper relationships is necessary, but existing implicit graph-based methods demonstrate limited effectiveness.

**Key Insight**: Decoupling the reading process into three independent agents: Select, Read, and Write solves context window limits via iterative reading and shared memory; utilizing explicit graph structures to constrain the reading order captures inter-paper relationships.

**Core Idea**: Replacing one-shot full-text input with a graph-constrained, multi-agent iterative reading framework preserves full-text information and inter-paper relationships while controlling the memory size.

## Method

### Overall Architecture
The input consists of a citing paper $C$ and a set of references $\mathcal{R} = \{R_1, ..., R_N\}$, where each paper contains complete sections $R_i = \{s_{i,1}, ..., s_{i,L_i}\}$. The framework consists of three agents:

- **Selector**: Based on the abstracts of all papers, the current working memory $M_{t-1}$, and the reading history $H_{t-1}$, it decides which section of which paper $(R_t, s_t)$ to read next.
- **Reader**: Digests the selected section content and updates the shared working memory $M_t$ (in JSON format, capped at 4096 tokens).
- **Writer**: After the iteration ends, generates the Related Work Section based on the final memory $M_T$ and the reading history $H_T$.

The Selector and Reader take turns iterating until the Selector outputs the termination signal `<End>`. The reading history $H$ records the sequence of read (Paper ID, Section Name) to prevent duplicate reading.

### Key Designs

1. **Working Memory Mechanism**:

    - **Function**: Maintains a shared memory in JSON format, storing key information extracted during the reading process.
    - **Mechanism**: Imposes an explicit size constraint (4096 tokens). At each step, the Reader reorganizes the content while updating the memory, discarding irrelevant information and retaining task-related key knowledge.
    - **Design Motivation**: Solves the issue where the total length of multiple full-text papers far exceeds the context window, transforming "one-shot full-text input" into "iterative reading + memory compression", mimicking human note-taking while reading.

2. **Graph-Aware Selector**:

    - **Function**: Constrains the reading order using relation graphs of papers, enabling the model to hop between related papers.
    - **Mechanism**: Constructs two types of graphs—**Co-occurrence Graph** ($G_{co}$: an edge connects two papers if they are co-cited in the RWS of a reference) and **Citation Graph** ($G_{ci}$: an edge connects papers with direct citation relationships). The Selector makes decisions within the one-hop subgraph $G_{t-1}$ of the current paper, either continuing to read the current paper or navigating to an adjacent paper in the graph.
    - **Design Motivation**: The graph structure explicitly encodes relationships between papers. Constraining the reading path enables the model to naturally establish comparisons and connections when processing related papers, rather than understanding each paper in isolation. The Citation Graph outperforms the Co-occurrence Graph because the latter exhibits over-connectivity, leading to insufficient constraint.

3. **Five Selector Strategy Comparison Design**:

    - **Function**: Provides 5 reading strategies from simple to complex—Sequential Reading (SR), Random Reading (RR), Vanilla LLM-Based, Graph-Co, and Graph-Ci.
    - **Mechanism**: SR reads chapter-by-chapter in order of Paper ID; RR shuffles the order to reduce positional bias; Vanilla relies on the LLM to autonomously decide the reading order; Graph-Co/Ci make decisions under graph constraints.
    - **Design Motivation**: Systematically compares the effects of different reading strategies to verify the necessity of graph constraints. Experiments demonstrate the performance ranking as SR < RR < Vanilla < Graph-Co < Graph-Ci.

### Loss & Training
No training required; this is a purely prompt-based multi-agent framework that directly uses off-the-shelf LLMs as the base models for the three agents. The Writer's generation quality is enhanced through prompt guidance (e.g., avoiding isolated descriptions, explaining relationships between papers, and grouping similar research) combined with in-context learning examples.

## Key Experimental Results

### Main Results
Evaluated on the OARelatedWork dataset (open domain, average testing set input of 70K tokens). Evaluation leverages graph metrics (number of edges/degree/clustering coefficient) and LLM-based evaluation (Coverage/Logic/Relevance, on a 5-point scale).

| Model Configuration | Avg. Edges | Clustering Coeff. | Overall (LLM) | Gain |
|----------|-----------|-------------------|---------------|------------|
| GPT-4o (Abstract) | 1.180 | 0.057 | 3.69 | baseline |
| GPT-4o + GO (Retrieval) | 1.611 | 0.096 | 3.75 | +1.6% |
| GPT-4o (Full, 128K) | 1.244 | 0.136 | 3.68 | -0.3% |
| **GPT-4o Graph-Ci (Ours)** | **2.125** | **0.128** | **3.87*** | **+3.2%** |
| Claude-3-Haiku + GO | 2.308 | 0.100 | 3.52 | baseline |
| **Claude-3-Haiku Graph-Ci** | **3.240** | **0.231** | **3.61*** | **+2.6%** |
| Llama3-8B + GO | 1.511 | 0.054 | 3.29 | baseline |
| **Llama3-8B Graph-Ci** | **1.410** | **0.154** | **3.44*** | **+4.6%** |

### Ablation Study

| Strategy | Performance Trend | Explanation |
|------|---------|------|
| Sequential Reading (SR) | Lowest | Fixed order introduces positional bias |
| Random Reading (RR) | > SR | Shuffled order mitigates bias |
| Vanilla LLM | > RR | Autonomous LLM decision-making is smarter |
| Graph-Co | > Vanilla | Co-occurrence graph constraints capture implicit relationships |
| Graph-Ci | **Optimal** | Citation graph constraints are the most direct and effective |

### Key Findings
- **Full Text > Abstract**: Models with full-text inputs outperform abstract-only models across all base models and metrics, validating the necessity of full-text RWG.
- **Direct Full-Text Insertion is Suboptimal**: Providing Claude-3-Haiku with the direct full text results in a 9.3% drop in the Overall score, indicating that effectively utilizing long-context LLMs remains challenging.
- **Graph Constraints are Consistently Effective**: Graph-Ci achieves statistically significant improvements ($p<0.05$) across the three base models of varying capability levels.
- **Human Evaluation Validation**: In pairwise evaluations conducted by 5 AI graduate students, Graph-Ci achieves the highest win rate (61-68%) in the Logic dimension, aligning with automatic metrics.
- **Weak Models are Insensitive to Strategies**: The margins among the five strategies on Llama3-8B are minimal, potentially due to its limited fundamental capability, making it insensitive to reading orders.

## Highlights & Insights
- **Modeling Human Reading Behavior as a Multi-Agent System**: The Select-Read-Write pipeline matches the workflow of "selecting papers, intensive reading, and writing surveys" in academic reading. The working memory mimics human note-taking, making it highly intuitive.
- **Graph Constraints as the Key Differentiator**: Instead of simple multi-agent chatting, the framework guides the agent to "walk" through related papers via paper relationship graphs, naturally generating structurally cohesive RWS that compare different papers.
- **Rigorous Experimental Design**: The pure prompt-based framework combined with the open-source Llama3 ensures reproducibility. Ablation of the 5 strategies plus human evaluation validates each design choice.

## Limitations & Future Work
- **Significant Gap with Human Level**: Human RWS exhibits an average of 9.48 edges vs. 3.24 for the best model; the models still tend to over-expand on minor references.
- **Requires Prerequisite Reference Sets**: Users must manually retrieve and filter papers, as the framework does not include an automatic paper retrieval module.
- **Limited Memory Capacity**: A working memory of 4096 tokens might lose key information when processing a large number of references, and the information loss during compression remains unquantified.
- **Base Model Capability Determines the Upper Bound**: Weak models (Llama3-8B) fail to outperform the baseline of medium-sized models even with the optimal strategy, indicating high dependency on base model capabilities.

## Related Work & Insights
- **vs. Li & Ouyang (2025)**: Both are RWG methods in the LLM era, but Li & Ouyang still rely on abstracts and prompt engineering, whereas this paper achieves deeper inter-paper relationship modeling via full-text reading and graph constraints.
- **vs. RAG-based Methods**: Retrieval methods (e.g., Greedy Oracle) select key sentences as input, potentially missing contextual information; the iterative reading and memory updating strategy in this work is more flexible.
- **vs. Long-Context LLMs**: Directly feeding the entire full text into 128K/200K windows worsens performance, indicating that effective long-text processing requires a structured reading strategy rather than blindly expanding window sizes.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of graph constraints, multi-agent coordination, and working memory is novel in the context of the RWG task, though individual components are not entirely brand new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Very comprehensive, covering three base models × five strategies × multiple input configurations, supplemented by human evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear architecture diagrams, standardized formulations, and intuitive case studies.
- Value: ⭐⭐⭐⭐ Holds practical significance for computer-assisted academic writing, validating the necessity of full-text RWG and graph structures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling](metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)
- [\[ACL 2025\] Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines](bel_esprit_multi-agent_framework_for_building_ai_model_pipelines.md)
- [\[ACL 2025\] Table-Critic: A Multi-Agent Framework for Collaborative Criticism and Refinement in Table Reasoning](table_critic_multi_agent.md)
- [\[ACL 2025\] MIND: A Multi-agent Framework for Zero-shot Harmful Meme Detection](mind_a_multi-agent_framework_for_zero-shot_harmful_meme_detection.md)
- [\[CVPR 2025\] ATA: Adaptive Transformation Agent for Text-Guided Subject-Position Variable Background Generation](../../CVPR2025/llm_agent/ata_adaptive_transformation_agent_for_text-guided_subject-position_variable_back.md)

</div>

<!-- RELATED:END -->
