---
title: >-
  [Paper Note] Rerank Before You Reason: Analyzing Reranking Tradeoffs through Effective Token Cost in Deep Search Agents
description: >-
  [ACL 2026][Information Retrieval & RAG][Deep search agents] This paper systematically investigates the efficiency-effectiveness trade-offs of listwise reranking in deep search agents…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Deep search agents"
  - "reranking"
  - "inference budget allocation"
  - "effective token cost"
  - "retrieval-augmented reasoning"
date: 2026-05-08
content_hash: 77c253c160a1a7d0
---

<!-- Generated automatically from src/gen_stubs.py -->
# Rerank Before You Reason: Analyzing Reranking Tradeoffs through Effective Token Cost in Deep Search Agents

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.14224](https://arxiv.org/abs/2601.14224)  
**Code**: [https://github.com/sahel-sh/DeepHone](https://github.com/sahel-sh/DeepHone)  
**Area**: information_retrieval  
**Keywords**: Deep search agents, reranking, inference budget allocation, effective token cost, retrieval-augmented reasoning

## TL;DR
This paper systematically investigates the efficiency-effectiveness trade-offs of listwise reranking in deep search agents, proposing the Effective Token Cost (ETC) metric. The study finds that mid-depth reranking is often more cost-effective than increasing search-time inference budgets, achieving comparable or superior end-to-end accuracy with lower token overhead.

## Background & Motivation
**Background**: Deep research agents perform excellently on inference-intensive benchmarks by iterating through retrieval and reasoning to answer complex multi-hop queries, yet significant efficiency issues arise due to high test-time compute.

**Limitations of Prior Work**: Current evaluations often rely on opaque Web search APIs, obscuring the relationship between retrieval quality and external service behavior; the role and cost-effectiveness of reranking in deep search pipelines have not been systematically studied.

**Key Challenge**: Increasing the inference budget (more reasoning tokens) improves accuracy but escalates costs sharply; improving retrieval quality (via reranking) may be a more efficient alternative, but a unified metric to compare these two investments is lacking.

**Goal**: Quantify the efficiency-effectiveness trade-off between reranking and inference budgets in deep search agents to identify optimal compute allocation strategies.

**Key Insight**: Conducting controlled variable experiments on the BrowseComp-Plus benchmark (using a fixed human-verified corpus) and introducing the ETC metric to uniformly measure the costs of different configurations.

**Core Idea**: Shifting the compute budget from search inference to retrieval reranking can yield better end-to-end performance at a lower cost.

## Method

### Overall Architecture
The experimental pipeline involves search agents (gpt-oss-20b/120b) iteratively generating queries → qwen3-embedding-8b retrieving top-d candidates → listwise reranking (oss-20b/120b or qwen3-reranker-0.6b) → passing top-5 to the agent → iterating until an answer is found. Different configurations are compared using the ETC metric.

### Key Designs
1. **Effective Token Cost (ETC) Metric**:

    - **Function**: Uniformly quantify the real computational cost of different token types.
    - **Mechanism**: $\text{ETC} = \text{Input}_{nc} + \alpha \cdot \text{Input}_c + \beta \cdot \text{Output}_t$, where $\alpha$ models cache reuse efficiency (0.1-0.5) and $\beta$ models the high decoding cost of output tokens (3-7).
    - **Design Motivation**: Hardware throughput and API pricing differ significantly across token types (non-cached input, cached input, output including reasoning tokens); simple token counting fails to reflect real costs.

2. **Multi-dimensional Reranking Analysis Framework**:

    - **Function**: Systematically evaluate the interactive effects of model scale × inference budget × reranking depth.
    - **Mechanism**: Fix the retriever (qwen3-embedding-8b) while varying reranking depth $d \in \{10, 20, 50\}$, model (oss-20b/120b), and inference level (low/med/high), performing comprehensive ablations across 830 queries.
    - **Design Motivation**: Deep search involves multiple adjustable parameters; controlled experiments are necessary to isolate the marginal gains of each factor.

3. **Heterogeneous Settings and Generalized ETC**:

    - **Function**: Evaluate the feasibility of using lightweight cross-encoders as alternatives to listwise reranking.
    - **Mechanism**: Use qwen3-reranker-0.6b as a cross-encoder (outputting only yes/no logits) and extend ETC to $\text{ETC} = \text{ETC}_{agent} + \gamma \cdot \text{ETC}_{reranker}$, where $\gamma = 0.32$ reflects the difference in FLOPs.
    - **Design Motivation**: While lightweight rerankers may have slightly lower peak performance, the condition $\gamma \ll 1$ makes them more attractive in efficiency-priority scenarios.

### Loss & Training
This study is a pure inference-stage analysis with no training. Retrieval uses qwen3-embedding-8b, reranking uses the RankLLM framework (window size 20), and evaluation employs LLM-as-a-judge (oss-120b) with an average of 5 scores.

## Key Experimental Results

### One-shot Reranking Performance (NDCG@5)
| Configuration | d=0 | d=10 | d=20 | d=50 |
|------|-----|------|------|------|
| No Reranking | 19.72 | — | — | — |
| oss-20b-low | — | 27.30 | 32.28 | 35.89 |
| oss-20b-med | — | 28.34 | 34.37 | 39.86 |
| oss-120b-low | — | 29.64 | 35.69 | 44.10 |
| oss-120b-med | — | 29.78 | 36.63 | **46.05** |

### End-to-end Deep Search Accuracy (Accuracy %)
| Search Agent | d=0 | d=10 | d=20 | d=50 |
|------------|-----|------|------|------|
| oss-20b-low | 17.33 | 19.64 | 22.96 | 25.73 |
| oss-20b-med | 35.59 | 39.04 | 43.39 | 49.11 |
| oss-20b-high | 41.38 | 48.31 | 52.70 | 55.97 |
| oss-120b-low | 28.17 | 31.76 | 36.92 | 42.65 |
| oss-120b-med | 40.43 | 48.65 | 53.96 | 57.97 |
| oss-120b-high | 51.20 | 54.65 | 58.99 | **63.35** |

### Key Findings
- Marginal gains from reranking depth are most significant: d=10→20 improves NDCG@5 by 5-7 points, and d=20→50 adds another 3.5-10 points.
- oss-120b-med + d=20 (53.96% accuracy) matches oss-120b-high + d=0 (51.20%) with significantly lower ETC.
- Search calls decrease by only ≤12% when reranking is enabled, suggesting gains primarily stem from retrieval quality rather than behavioral changes.
- Heterogeneous settings (qwen3-reranker-0.6b) retain most end-to-end gains (52.91% vs 55.97% at oss-20b-high, d=50) with drastically reduced ETC ($\gamma=0.32$).
- Reranking actually reduces latency in high-inference settings (150.7s vs 184.7s at d=50 vs d=0) due to fewer search calls.

## Highlights & Insights
- The ETC metric is cleverly designed and adaptable to both API pricing and self-hosted scenarios, providing a unified framework for cost optimization in RAG systems.
- The core insight of "rerank before you reason" is simple but effective: rather than forcing the agent to think more, it is better to provide higher-quality documents first.
- The extension to heterogeneous ETC enables cost comparisons across different model scales.
- Latency analysis reveals a counter-intuitive result: reranking can accelerate high-inference scenarios.

## Limitations & Future Work
- Only gpt-oss series models were used; generalization to commercial models or other open-source reasoning models remains unverified.
- BrowseComp-Plus uses a fixed corpus rather than real Web search; caution is needed when transferring conclusions to production environments.
- Reranking uses a fixed top-d selection; adaptive document subset selection might further enhance efficiency.
- Full interaction history is preserved (auto-truncated to 128K tokens); explicit compression or summarization strategies were not explored.

## Related Work & Insights
- **BrowseComp-Plus**: A controlled benchmark for deep search that fixes the retrieval stage to isolate variables.
- **RankLLM**: A listwise reranking framework serving as the experimental infrastructure.
- **AgentDiet**: Focuses on reducing redundant context in agent trajectories, complementing the retrieval-side optimization in this work.
- Insight: In RAG/Agent systems, the problem of compute budget allocation deserves more attention—improvements in retrieval quality are often more cost-effective than increasing the volume of inference.

## Rating
- Novelty: ⭐⭐⭐⭐ The ETC metric is novel and practical; the empirical insights on "rerank before you reason" are valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage with 830 queries, multi-configuration ablations, heterogeneous settings, and latency analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich diagrams, and detailed description of experimental design.
- Value: ⭐⭐⭐⭐⭐ Provides direct guidance for the engineering practices of deep search systems.

## Rating
- Novelty: To be rated
- Experimental Thoroughness: To be rated
- Writing Quality: To be rated
- Value: To be rated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] RRRA: Resampling and Reranking through a Retriever Adapter](../../AAAI2026/information_retrieval/rrra_resampling_and_reranking_through_a_retriever_adapter.md)
- [\[ACL 2026\] Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization](enhancing_llm-based_search_agents_via_contribution_weighted_group_relative_polic.md)
- [\[ICLR 2026\] Hybrid Deep Searcher: Scalable Parallel and Sequential Search Reasoning](../../ICLR2026/information_retrieval/hybrid_deep_searcher_scalable_parallel_and_sequential_search_reasoning.md)
- [\[ACL 2026\] Can Compact Language Models Search Like Agents? Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities](can_compact_language_models_search_like_agents_distillation-guided_policy_optimi.md)
- [\[ICML 2026\] ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards](../../ICML2026/information_retrieval/reseek_a_self-correcting_framework_for_search_agents_with_instructive_rewards.md)

</div>

<!-- RELATED:END -->
