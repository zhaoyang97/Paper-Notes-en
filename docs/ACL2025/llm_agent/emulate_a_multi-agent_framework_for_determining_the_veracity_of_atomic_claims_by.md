---
title: >-
  [Paper Note] EMULATE: A Multi-Agent Framework for Determining the Veracity of Atomic Claims by Emulating Human Actions
description: >-
  [ACL 2025][LLM Agent][Multi-Agent] The EMULATE multi-agent fact-checking framework is proposed, which simulates the complete human action chain for verifying claims (search → ranking → content evaluation → evidence sufficiency judgment → classification) through 7 specialized LLM agents, outperforming existing methods in both Macro-F1 and Weighted-F1 on three fact-checking benchmarks.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "Multi-Agent"
  - "Fact-checking"
  - "Claim Verification"
  - "Iterative Retrieval"
  - "Web Search"
date: 2026-05-08
content_hash: eb02f8aa21dabfee
---

# EMULATE: A Multi-Agent Framework for Determining the Veracity of Atomic Claims by Emulating Human Actions

**Conference**: ACL 2025  
**arXiv**: [2505.16576](https://arxiv.org/abs/2505.16576)  
**Code**: [https://github.com/qqqube/EMULATE](https://github.com/qqqube/EMULATE)  
**Area**: LLM Agent  
**Keywords**: Multi-Agent, Fact-checking, Claim Verification, Iterative Retrieval, Web Search

## TL;DR
The EMULATE multi-agent fact-checking framework is proposed, which simulates the complete human action chain for verifying claims (search → ranking → content evaluation → evidence sufficiency judgment → classification) through 7 specialized LLM agents, outperforming existing methods in both Macro-F1 and Weighted-F1 on three fact-checking benchmarks.

## Background & Motivation

**Background**: Automatic fact-checking systems usually adopt the "deconstruct-then-verify" paradigm: deconstructing text into atomic claims and verifying them one by one. The verification process generally follows a two-step pipeline of "evidence retrieval → LLM classification".

**Limitations of Prior Work**: (a) Traditional methods retrieve all evidence at once before classification, which is inconsistent with the iterative verification behavior of humans; (b) Search results are not filtered by credibility/relevance, leading to high noise; (c) Web content may not be self-contained (requiring additional context to understand), which can mislead classification when directly used as evidence.

**Key Challenge**: The single-round retrieval + direct classification paradigm misses key steps in the human verification process—evaluating source credibility, judging content completeness, and deciding whether further searching is needed.

**Goal**: Design a multi-agent framework closer to human behavior, where each agent is responsible for a small step in the verification workflow.

**Key Insight**: Analyze the complete human action chain of verifying claims (searching → selecting links → reading and judging → deciding the next step) and design a specialized LLM agent for each step.

**Core Idea**: Utilize 7 specialized LLM agents to respectively handle subtasks such as searching, ranking, completeness judgment, usefulness judgment, sufficiency judgment, and classification, gradually building a high-quality evidence set for fact-checking.

## Method

### Overall Architecture
The input is an atomic claim, and the output is a True/False label. Workflow: (1) InitialQueryGen generates the initial search queries; (2) The search engine returns results, and SearchRank ranks them based on credibility and relevance; (3) Trait-by-trait traversal of search results, where SelfContainedCheck determines whether the content is understandable and DetHelpful determines whether it provides new information; (4) The process iterates, and SufficientEvidence judges whether the evidence is sufficient; (5) If insufficient, AdditionalQueryGen generates new queries to continue searching; (6) Ultimately, the Classifier performs classification based on the collected evidence set.

### Key Designs

1. **Design of 7 Specialized Agents**:

    - **InitialQueryGen**: Generates an initial list of search queries based on the claim.
    - **SearchRank**: Ranks search results based on URL credibility and headline/snippet relevance, ensuring high-quality sources are processed first.
    - **SelfContainedCheck**: Determines if web content is self-contained (independently understandable or understandable when combined with existing evidence). Non-self-contained results are temporarily shelved and processed once the evidence set is enriched.
    - **DetHelpful**: Determines whether understandable search results provide new information not yet included in the evidence set.
    - **SufficientEvidence**: Judges whether the current evidence set is sufficient for the final classification.
    - **AdditionalQueryGen**: Generates new search queries when evidence is insufficient.
    - **Classifier**: Performs True/False classification based on the claim and the collected evidence set.
    - **Design Motivation**: Each agent only needs to complete a simple subtask, which reduces the LLM's cognitive load and makes system behavior more controllable and interpretable.

2. **Iterative Evidence Collection**:

    - **Function**: Instead of retrieving all at once, dynamically judge evidence sufficiency and continue searching as needed.
    - **Mechanism**: Set MAX_SEARCH_QUERIES (default 4) and MAX_SEARCH_RESULTS_PER_QUERY (default 2) as upper limits, and continue iterating until the limits are reached.
    - **Design Motivation**: Human verification is naturally iterative—if one source is insufficient, searching continues until there is enough confidence to make a judgment.

3. **Delayed Processing of Non-Self-Contained Documents**:

    - **Function**: When web content is not self-contained, temporarily shelve it instead of immediately discarding it or performing supplementary retrieval.
    - **Mechanism**: As the evidence set grows, previously non-self-contained content may become understandable; these shelved contents are processed in a unified manner before the end of the main loop.
    - **Design Motivation**: Prioritizing self-contained evidence can reduce unnecessary search query overhead.

### Loss & Training
- Pure inference-time method without training.
- All agents use GPT-4.1 with zero-shot prompting.
- The search API uses serper.dev.

## Key Experimental Results

### Main Results (GPT-4.1 as the base LLM)

| Dataset | Method | True F1 | False F1 | Macro-F1 | Weighted-F1 |
|--------|------|---------|----------|----------|-------------|
| BingCheck | FIRE | 0.89 | 0.63 | 0.76 | 0.84 |
| | **EMULATE** | **0.93** | **0.69** | **0.81** | **0.88** |
| FacTool-KBQA | FIRE | 0.89 | 0.66 | 0.78 | 0.83 |
| | **EMULATE** | **0.91** | **0.68** | **0.80** | **0.85** |
| Factcheck-Bench | FIRE | 0.87 | 0.68 | 0.78 | 0.82 |
| | **EMULATE** | **0.90** | **0.71** | **0.80** | **0.85** |

EMULATE achieves the best results in 6 out of 8 metrics across all three datasets.

### Ablation Study

| Configuration | FacTool-KBQA W-F1 | Factcheck-Bench W-F1 |
|------|-------------------|---------------------|
| EMULATE (full) | **0.85** | **0.85** |
| w/o SearchRank | 0.80 | 0.83 |
| w/o SelfContainedCheck | 0.84 | 0.82 |

### Key Findings
- **SearchRank contributes the most (-5pp on FacTool-KBQA)**: The quality ranking of search results is critical to the final accuracy, indicating that source filtering is a core step in the human action chain.
- **SelfContainedCheck is more important on Factcheck-Bench (-3pp)**: Claims in this dataset are more complex, and more web search results are non-self-contained.
- **Weaker models are also viable**: EMULATE on GPT-4.1-mini performs close to GPT-4.1, indicating that the framework design reduces dependency on single-model capabilities.

## Highlights & Insights
- **"Human-emulating" framework design philosophy**: Deconstructing tasks based on the human action chain of verifying information with a corresponding agent for each step is a generalizable multi-agent task-decomposition strategy. It can be transferred to other scenarios requiring a judge-retrieve-rejudge loop (such as academic literature reviews or competitive analysis).
- **Value of SearchRank**: Adding a search result ranking/filtering step in the RAG pipeline is an easily overlooked but highly effective improvement.
- **Iterative + delayed processing strategy**: Shelving non-self-contained documents and processing them after the evidence is enriched reduces unnecessary search API calls.

## Limitations & Future Work
- **Small dataset scale**: All three datasets have <1000 samples and exhibit class imbalance (significantly fewer "False" than "True").
- **Binary classification only**: In reality, claims may have fine-grained labels such as "partially true" or "misleading".
- **Strong dependency on Search API**: The quality of search results directly determines the final performance, which may degrade significantly in low-quality search scenarios.
- **High API call overhead of 7 agents**: Multiple LLM calls are required per claim, resulting in high cost and latency.
- Potential improvements: (a) Training lightweight models to replace some agents (e.g., using a small model for SearchRank); (b) Supporting compound claims that require multi-hop reasoning.

## Related Work & Insights
- **vs FIRE (Xie et al., 2025)**: FIRE also features iterative retrieval but only uses 3 components. EMULATE decomposes the verification workflow more finely (7 agents), specifically adding SearchRank and SelfContainedCheck.
- **vs FacTool (Chern et al., 2023)**: FacTool is a naive single-round retrieval + classification pipeline, lacking search result filtering and evidence sufficiency judgment.
- **vs SAFE (Wei et al., 2024)**: SAFE focuses on iterative query generation, but does not perform quality-based ranking of search results.

## Rating
- Novelty: ⭐⭐⭐⭐ The design idea of multi-agent human emulation is clear, and the decomposition into 7 specialized agents is creative.
- Experimental Thoroughness: ⭐⭐⭐ 3 datasets but with small scales, and the ablation study only evaluated 2 agents.
- Writing Quality: ⭐⭐⭐⭐ The framework diagram is clear and the action analysis is highly logical.
- Value: ⭐⭐⭐⭐ Highly valuable for both fact-checking and multi-agent system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MIND: A Multi-agent Framework for Zero-shot Harmful Meme Detection](mind_a_multi-agent_framework_for_zero-shot_harmful_meme_detection.md)
- [\[ACL 2025\] Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines](bel_esprit_multi-agent_framework_for_building_ai_model_pipelines.md)
- [\[ACL 2025\] Leveraging Dual Process Theory in Language Agent Framework for Real-time Simultaneous Human-AI Collaboration](dpt_agent_dual_process.md)
- [\[ACL 2025\] METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling](metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)
- [\[ACL 2025\] Table-Critic: A Multi-Agent Framework for Collaborative Criticism and Refinement in Table Reasoning](table_critic_multi_agent.md)

</div>

<!-- RELATED:END -->
