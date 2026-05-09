---
title: >-
  [Paper Note] LiveNewsBench: Evaluating LLM Web Search Capabilities with Freshly Curated News
description: >-
  [ICLR 2026][LLM Agent][LLM Web Search] This paper proposes LiveNewsBench, a periodically updated benchmark that automatically generates QA pairs from fresh news events to evaluate LLM agentic web search capabilities, effectively isolating model internal memory from genuine search ability.
tags:
  - ICLR 2026
  - LLM Agent
  - LLM Web Search
  - Benchmark
  - Agentic Search
  - News QA
  - Multi-hop Search
date: 2026-05-08
content_hash: 23919914cbe6b589
---

# LiveNewsBench: Evaluating LLM Web Search Capabilities with Freshly Curated News

**Conference**: ICLR 2026
**arXiv**: [2602.13543](https://arxiv.org/abs/2602.13543)
**Code**: [livenewsbench.com](https://livenewsbench.com)
**Area**: LLM Agent
**Keywords**: LLM Web Search, Benchmark, Agentic Search, News QA, Multi-hop Search

## TL;DR

This paper proposes LiveNewsBench, a periodically updated benchmark that automatically generates QA pairs from fresh news events to evaluate LLM agentic web search capabilities, effectively isolating model internal memory from genuine search ability.

## Background & Motivation

A core challenge in evaluating LLM search capabilities is **disentangling the contribution of external search from the model's internal memorized world knowledge**. Since state-of-the-art LLMs are pretrained on massive corpora, they already encode substantial factual knowledge. When benchmarks rely on static questions, models may answer correctly from memory alone rather than through genuine search behavior.

Three major limitations of existing benchmarks:

1. **Academic reasoning benchmarks** (e.g., HLE): primarily measure domain knowledge and reasoning ability rather than search behavior per se. For example, enabling search for GPT-5 on HLE only improves accuracy from 24.8% to 30.7%.
2. **Static factual QA benchmarks** (e.g., SimpleQA, BrowseComp, TriviaQA): models can achieve high accuracy without search (SimpleQA 62.5%, TriviaQA 82.9%), making these ineffective for evaluating search capability.
3. **Time-sensitive QA benchmarks** (e.g., FreshQA, SealQA): although answers change over time, the questions remain fixed, and strong models can often answer through reasoning or partial memorization.
4. **Deep Research benchmarks**: employ subjective evaluation criteria (completeness, insight, etc.) and lack verifiable factual answers.

## Method

### Overall Architecture

The automated data construction pipeline of LiveNewsBench consists of two main components:

1. **News article retrieval**: seeds news events from Wikipedia's Current Events Archive and retrieves related articles via a search engine.
2. **QA pair generation**: generates question-answer pairs from article clusters, followed by automated and human verification.

### Key Designs

**News retrieval pipeline**:

- Collects globally significant news events from the Wikipedia Current Events Archive as seeds.
- Uses GPT-4.1 to rewrite event summaries into search queries.
- Retrieves relevant news URLs via the Brave Search API, restricted to a whitelist of approximately 100 reputable media outlets.
- Applies a 14-day time window (3 days before to 11 days after the event).
- Downloads archived versions via archive.today to ensure stability and reproducibility.
- Uses GPT-4.1 to verify article relevance to the event; after filtering, each event yields an average of 5.3 articles.

**QA pair generation and validation**:

- Uses GPT-5.1 Thinking to generate QA pairs from article clusters, requiring citation of multiple articles (multi-hop).
- **Self-consistency filtering**: the question and articles are independently provided to GPT-5.1 to derive an answer; only QA pairs where both runs agree are retained.
- **Guideline compliance verification**: GPT-5.1 is prompted again with paraphrased guidelines to verify that each QA pair satisfies all criteria.
- **Human validation subset**: the original authors review QA pairs and reject approximately 15% of automatically validated samples; independent NLP researchers achieve a 92% agreement rate.

**Data splits**: partitioned by news event date — the most recent two months form the test set (340 samples), the third month forms the validation set (170 samples), and earlier data constitute the training set (600+). The human-validated subset contains 200 samples.

### Evaluation Framework

A custom ReAct-style agent framework is employed, supporting three actions:

- **Search**: issues a search query and returns the top-10 results (title, URL, snippet).
- **Visit**: accesses a webpage from the search results and returns the full text.
- **Finish & Answer**: produces the final answer.

Standard configuration: up to 5 search queries and 5 webpage visits.

## Key Experimental Results

### Main Results: Comparison Without Internet Access

| Model | LiveNewsBench (%) | FreshQA (%) | SealQA-Hard (%) |
|------|:-:|:-:|:-:|
| GPT-5.2 | 21.5 | 72.2 | 31.9 |
| Gemini 3 Pro | 20.5 | 74.3 | 46.5 |
| GPT-4.1 | 14.0 | 65.7 | 26.8 |
| Claude 4.5 Sonnet | 13.0 | 70.8 | 23.2 |
| DeepSeek V3.2 Thinking | 10.0 | 61.0 | 31.5 |

State-of-the-art models achieve 60–74% accuracy on FreshQA/SealQA without search, yet only 10–21.5% on LiveNewsBench, demonstrating the benchmark's effectiveness in resisting memorization.

### Search Agent Evaluation (Human-Validated Test Set)

| Method | Reasoning Model | # Searches | # Visits | Accuracy |
|------|:-:|:-:|:-:|:-:|
| DeepSeek V3.2 Thinking | ✓ | 3.3±1.3 | 2.6±1.4 | **84.5%** |
| DeepSeek V3.2 (No Think) | ✗ | 3.4±1.2 | 2.6±1.3 | 83.0% |
| Claude Sonnet 4.5 | ✗ | 2.9±1.1 | 1.3±1.3 | 82.0% |
| GPT-5.2 | ✓ | 2.9±1.1 | 1.8±1.3 | 74.0% |
| GPT-5.2 Official API | ✓ | N/A | N/A | **90.0%** |
| Llama 3.1 8B | ✗ | 3.9±1.1 | 0.3±1.0 | 11.0% |

### Ablation Study: Effect of Search Budget

| Model | Budget=1 | Budget=3 | Budget=5 | Budget=7 | Gain (1→7) |
|------|:-:|:-:|:-:|:-:|:-:|
| DeepSeek V3.2 Thinking | 48.5% | 80.5% | 84.5% | 84.5% | +36.0% |
| DeepSeek V3.2 (No Think) | 20.0% | 79.0% | 83.0% | 84.5% | +64.5% |
| Claude Sonnet 4.5 | 53.5% | 79.0% | 82.0% | 67.0% | +13.5% |
| GPT-5.2 | 62.5% | 72.5% | 74.0% | 74.5% | +12.0% |

### Key Findings

1. **Limited memorization**: disabling search reduces accuracy by 17%–74.5% (absolute), confirming the benchmark's validity for measuring search capability.
2. **Clear multi-hop characteristics**: increasing the search budget from 1 to 7 yields substantial gains across all models, with open-source models benefiting more.
3. **Tool-calling reliability matters significantly**: 44% of Kimi K2 Thinking samples fail to invoke the search tool correctly, compared to only 0.5% for Claude and DeepSeek.
4. **Models rely more on search snippets than full pages**: all models issue more search queries than webpage visits.
5. **Search implementation affects performance**: GPT-5.2 via the official API outperforms the local framework by 16%, whereas the Claude API underperforms by 42%.

## Highlights & Insights

- **Scalability of the automated pipeline**: the full construction cost is approximately $700, with a commitment to quarterly updates, alleviating the bottleneck of manual annotation in traditional benchmarks.
- **Decoupling memory from search**: by grounding questions in news events published after training cutoff dates, the benchmark effectively separates model-internal knowledge from search behavior.
- **Strong discriminative power**: accuracy ranges from 11% to 90%, providing strong differentiation across models and search frameworks.
- **Training data potential**: a large-scale open-source training set is provided, suitable for training agentic search models via RLVR.

## Limitations & Future Work

1. Even with fresh news, state-of-the-art models can still answer some questions through world knowledge and reasoning (3.5%–21.5%), making complete elimination of memorization difficult.
2. The benchmark focuses solely on short factual answers and does not cover scenarios requiring long-form research reports.
3. Reliance on the Wikipedia Current Events Archive as a seed source may omit non-prominent but evaluatively valuable events.
4. Evaluation costs remain high (requiring search API and LLM calls), limiting large-scale evaluation.
5. Future work plans to extend to time-sensitive questions whose answers evolve as events unfold.

## Related Work & Insights

- **Distinction from RealTimeQA**: RealTimeQA also updates periodically but provides only approximately 10 samples per update and relies on manual curation.
- **Complementarity with BrowseComp**: BrowseComp is static but tests browsing ability, while LiveNewsBench is dynamic and tests search ability.
- **Implications for RLVR training**: the large-scale training set provided can be used for reinforcement learning of agentic search models.
- **Insights for search agent design**: tool-calling reliability varies substantially across models and represents a critical performance bottleneck for search agents.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first search benchmark that simultaneously satisfies periodic updates, automated generation, memorization resistance, and objective evaluation.
- **Technical Depth**: ⭐⭐⭐ — Pipeline design is well-crafted, but technical contributions are primarily engineering-oriented.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 13 models, multiple search budgets, and comprehensive ablation analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with detailed experimental presentation.
- **Practical Value**: ⭐⭐⭐⭐⭐ — Fills an important gap in search evaluation with direct value to the community.
- **Overall Rating**: ⭐⭐⭐⭐ (8/10)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains](mc-search_evaluating_and_enhancing_multimodal_agentic_search_with_structured_lon.md)
- [\[ICLR 2026\] ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents](st-webagentbench_a_benchmark_for_evaluating_safety_and_trustworthiness_in_web_ag.md)
- [\[ICLR 2026\] OpenAgentSafety: A Comprehensive Framework for Evaluating Real-World AI Agent Safety](openagentsafety_a_comprehensive_framework_for_evaluating_real-world_ai_agent_saf.md)
- [\[ICLR 2026\] FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents](fingertip_20k_a_benchmark_for_proactive_and_personalized_mobile_llm_agents.md)
- [\[ICLR 2026\] Judge Reliability Harness: Stress Testing the Reliability of LLM Judges](judge_reliability_harness_stress_testing_the_reliability_of_llm_judges.md)

</div>

<!-- RELATED:END -->
