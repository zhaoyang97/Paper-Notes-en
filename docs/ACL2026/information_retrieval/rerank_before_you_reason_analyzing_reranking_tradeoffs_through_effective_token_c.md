---
title: >-
  [Paper Note] Rerank Before You Reason: Analyzing Reranking Tradeoffs through Effective Token Cost in Deep Search Agents
description: >-
  [ACL 2026][Information Retrieval & RAG][Paper Note] This paper systematically investigates the efficiency-effectiveness trade-offs of listwise reranking in deep search agents. By introducing the Effective Token Cost (ETC) metric, the study finds that moderate-depth reranking is generally more cost-effective than increasing search-time reasoning budgets, achieving compar
tags:
  - ACL 2026
  - Information Retrieval & RAG
date: 2026-05-08
content_hash: 0a6de2075fdfa187
---
# Rerank Before You Reason: Analyzing Reranking Tradeoffs through Effective Token Cost in Deep Search Agents

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.14224](https://arxiv.org/abs/2601.14224)  
**Code**: [https://github.com/sahel-sh/DeepHone](https://github.com/sahel-sh/DeepHone)  
**Area**: Information Retrieval  
**Keywords**: Deep Search Agents, Reranking, Reasoning Budget Allocation, Effective Token Cost, Retrieval-Augmented Reasoning

## TL;DR
This paper systematically investigates the efficiency-effectiveness trade-offs of listwise reranking in deep search agents. By introducing the Effective Token Cost (ETC) metric, the study finds that moderate-depth reranking is generally more cost-effective than increasing search-time reasoning budgets, achieving comparable or higher end-to-end accuracy with lower token overhead.

## Background & Motivation
**Background**: Deep research agents answer complex multi-hop queries through iterative retrieval and reasoning. They excel in reasoning-intensive benchmarks but face significant efficiency issues due to high test-time compute.

**Limitations of Prior Work**: Existing evaluations often rely on opaque Web search APIs, conflating retrieval quality with external service behavior. The specific role and cost-benefit ratio of reranking within deep search pipelines have not been systematically studied.

**Key Challenge**: Increasing the reasoning budget (more "thought" tokens) improves accuracy but escalates costs sharply. Enhancing retrieval quality through reranking might be a more efficient alternative, but a unified metric to compare these two types of "investments" is lacking.

**Goal**: Quantify the efficiency-effectiveness trade-offs between reranking and reasoning budgets in deep search agents to identify optimal computational allocation strategies.

**Key Insight**: Controlled experiments are conducted on the BrowseComp-Plus benchmark (using a fixed human-verified corpus), introducing the ETC metric to unify cost measurement across different configurations.

**Core Idea**: Shifting computational budget from search reasoning to retrieval reranking can yield better end-to-end results at a lower cost.

## Method

### Overall Architecture
Instead of training new models, this paper builds a controlled experimental pipeline to address an engineering question: Is it more cost-effective to spend compute on "making the agent think more" or "optimizing retrieval"? The pipeline consists of: search agents (gpt-oss-20b/120b) iteratively generating queries, qwen3-embedding-8b retrieving top-$d$ candidates, and listwise reranking (oss-20b/120b or lightweight qwen3-reranker-0.6b) feeding the top-5 results back to the agent. This cycle continues until an answer is found. All configurations are evaluated on BrowseComp-Plus using the ETC metric to normalize real costs, allowing a direct comparison between reranking and reasoning investments.

### Key Designs
**1. Effective Token Cost (ETC): A unified metric for different token costs**

Simple token counts do not reflect actual overhead—non-cached inputs, cached inputs, and reasoning-heavy outputs differ significantly in terms of hardware throughput and API pricing. ETC is calculated as a weighted sum: $\text{ETC} = \text{Input}_{nc} + \alpha \cdot \text{Input}_c + \beta \cdot \text{Output}_t$, where $\alpha$ (0.1–0.5) models the discount from cache reuse and $\beta$ (3–7) models the high cost of output decoding. ETC allows reranking tokens and reasoning tokens to be compared fairly.

**2. Multi-dimensional Reranking Analysis Framework: Controlled variables to isolate marginal gains**

Deep search involves multiple adjustable parameters. This paper fixes the retriever (qwen3-embedding-8b) and systematically varies reranking depth $d \in \{10, 20, 50\}$, reranking model scale (oss-20b/120b), and reasoning levels (low/med/high). Extensive ablations across 830 queries allow the marginal gains of deepening reranking versus increasing reasoning budget to be clearly quantified.

**3. Heterogeneous Settings and Generalized ETC: Incorporating lightweight cross-encoders**

Since listwise reranking itself consumes tokens, the study examines whether cheaper cross-encoders can serve as alternatives. Using qwen3-reranker-0.6b as a cross-encoder (outputting yes/no logits), ETC is extended to $\text{ETC} = \text{ETC}_{agent} + \gamma \cdot \text{ETC}_{reranker}$, where $\gamma = 0.32$ reflects the difference in FLOPs. While the lightweight reranker has lower peak performance, $\gamma \ll 1$ makes it highly attractive for efficiency-prioritized scenarios.

Fixed experimental settings: Reranking is based on the RankLLM framework (window size 20), and end-to-end accuracy is measured using LLM-as-a-judge (oss-120b) averaged over 5 runs.

## Key Experimental Results

### One-shot Reranking Performance (NDCG@5)

| Configuration | d=0 | d=10 | d=20 | d=50 |
|---------------|-----|------|------|------|
| No Reranking  | 19.72 | — | — | — |
| oss-20b-low   | — | 27.30 | 32.28 | 35.89 |
| oss-20b-med   | — | 28.34 | 34.37 | 39.86 |
| oss-120b-low  | — | 29.64 | 35.69 | 44.10 |
| oss-120b-med  | — | 29.78 | 36.63 | **46.05** |

### End-to-End Deep Search Accuracy (Accuracy %)

| Search Agent  | d=0 | d=10 | d=20 | d=50 |
|---------------|-----|------|------|------|
| oss-20b-low   | 17.33 | 19.64 | 22.96 | 25.73 |
| oss-20b-med   | 35.59 | 39.04 | 43.39 | 49.11 |
| oss-20b-high  | 41.38 | 48.31 | 52.70 | 55.97 |
| oss-120b-low  | 28.17 | 31.76 | 36.92 | 42.65 |
| oss-120b-med  | 40.43 | 48.65 | 53.96 | 57.97 |
| oss-120b-high | 51.20 | 54.65 | 58.99 | **63.35** |

### Key Findings
- Reranking depth offers high marginal returns: d=10→20 improves NDCG@5 by 5-7 points, and d=20→50 adds another 3.5-10 points.
- oss-120b-med + d=20 (53.96% accuracy) matches oss-120b-high + d=0 (51.20%) but with significantly lower ETC.
- The number of search calls only decreases by $\le 12\%$ with reranking, indicating gains stem primarily from retrieval quality rather than behavior changes.
- Heterogeneous settings (qwen3-reranker-0.6b) retain most end-to-end gains (52.91% vs 55.97% at oss-20b-high, d=50) while drastically reducing ETC ($\gamma=0.32$).
- Reranking reduces latency in high-reasoning settings (150.7s vs 184.7s at d=50 vs d=0) due to fewer search iterations.

## Highlights & Insights
- The ETC metric is cleverly designed to adapt to both API pricing and self-hosted scenarios, providing a unified framework for RAG system cost optimization.
- The "Rerank before reason" insight is simple yet effective: providing better documents is often superior to increasing the agent's internal thought process.
- The extension of heterogeneous ETC enables cost comparisons across different model scales.
- Latency analysis reveals a counter-intuitive result: reranking can speed up high-reasoning scenarios.

## Limitations & Future Work
- Only gpt-oss series models were used; generalization to commercial or open-source reasoning models remains to be verified.
- BrowseComp-Plus uses a fixed corpus rather than a live Web search, requiring caution when applying conclusions to production environments.
- Reranking uses a fixed top-$d$ selection; adaptive document subset selection might further improve efficiency.
- Full interaction history is retained (auto-truncated to 128K tokens); explicit compression or summarization strategies were not explored.

## Related Work & Insights
- **BrowseComp-Plus**: A controlled deep search benchmark that fixes the retrieval phase to isolate variables.
- **RankLLM**: The listwise reranking framework used as the experimental infrastructure.
- **AgentDiet**: Compares to this work by reducing redundant context in agent trajectories; these methods are complementary.
- Insight: Budget allocation in RAG/Agent systems deserves more attention—improving retrieval quality is often more cost-effective than increasing reasoning overhead.

## Rating
- Novelty: ⭐⭐⭐⭐ The ETC metric is practical and novel; the empirical insights on "rerank before reason" are valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 830 queries, multi-configuration ablations, heterogeneous settings, and latency analysis make it very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich visualizations, and detailed experimental descriptions.
- Value: ⭐⭐⭐⭐⭐ Directly applicable to engineering practices for deep search systems.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] RRRA: Resampling and Reranking through a Retriever Adapter](../../AAAI2026/information_retrieval/rrra_resampling_and_reranking_through_a_retriever_adapter.md)
- [\[ACL 2026\] Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization](enhancing_llm-based_search_agents_via_contribution_weighted_group_relative_polic.md)
- [\[ICLR 2026\] Hybrid Deep Searcher: Scalable Parallel and Sequential Search Reasoning](../../ICLR2026/information_retrieval/hybrid_deep_searcher_scalable_parallel_and_sequential_search_reasoning.md)
- [\[ICML 2026\] ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards](../../ICML2026/information_retrieval/reseek_a_self-correcting_framework_for_search_agents_with_instructive_rewards.md)
- [\[ACL 2026\] Can Compact Language Models Search Like Agents? Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities](can_compact_language_models_search_like_agents_distillation-guided_policy_optimi.md)

</div>

<!-- RELATED:END -->
