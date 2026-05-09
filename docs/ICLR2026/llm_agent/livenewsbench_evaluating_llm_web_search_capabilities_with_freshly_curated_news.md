---
title: >-
  [Paper Note] LiveNewsBench: Evaluating LLM Web Search Capabilities with Freshly Curated News
description: >-
  [ICLR 2026][LLM Agent][LLM Web Search] This paper proposes LiveNewsBench, an automatically generated and periodically refreshed benchmark derived from recent news articles. It evaluates LLMs' agentic web search capabilities through multi-hop, factual question answering, effectively decoupling models' parametric knowledge from their retrieval ability. Model performance ranges from 11% to 90%, demonstrating strong discriminative power.
tags:
  - ICLR 2026
  - LLM Agent
  - LLM Web Search
  - Agentic Search
  - benchmark
  - News QA
  - Multi-hop Retrieval
date: 2026-05-08
content_hash: fd23e0ab1b4863b4
---

# LiveNewsBench: Evaluating LLM Web Search Capabilities with Freshly Curated News

**Conference**: ICLR 2026
**arXiv**: [2602.13543](https://arxiv.org/abs/2602.13543)
**Code**: [https://livenewsbench.com](https://livenewsbench.com)
**Area**: LLM Agent
**Keywords**: LLM Web Search, Agentic Search, benchmark, News QA, Multi-hop Retrieval

## TL;DR
This paper proposes LiveNewsBench, an automatically generated and periodically refreshed benchmark derived from recent news articles. It evaluates LLMs' agentic web search capabilities through multi-hop, factual question answering, effectively decoupling models' parametric knowledge from their retrieval ability. Model performance ranges from 11% to 90%, demonstrating strong discriminative power.

## Background & Motivation
**State of the Field**: LLMs equipped with agentic web search (e.g., GPT-5.2, DeepSeek V3.2) excel at tasks requiring real-time information, yet evaluating such systems poses fundamental challenges.

**Limitations of Prior Work**: Existing benchmarks fall into three categories: (a) academic reasoning benchmarks (e.g., HLE) that effectively measure domain knowledge rather than search ability; (b) factual QA benchmarks (e.g., SimpleQA, BrowseComp) that use static question–answer pairs, allowing models to answer from memorization (GPT-5.2 achieves 62.5% on SimpleQA without web access); and (c) Deep Research benchmarks that rely on subjective evaluation metrics (completeness, insight), which cannot verify factual correctness.

**Root Cause**: As LLM training data grows increasingly comprehensive, static benchmarks struggle to distinguish whether a model answers correctly from memorization or from genuine retrieval. Time-sensitive benchmarks (FreshQA, SealQA) update answers over time but keep questions fixed and simple; frontier models still achieve 74.3% on FreshQA without internet access.

**Paper Goals**: To design a continuously updatable, memorization-resistant evaluation benchmark for LLM web search that requires multi-hop retrieval and provides objectively verifiable factual answers.

**Starting Point**: The pipeline seeds data collection from Wikipedia's Current Events Archive, automatically crawls recent news articles, and uses an LLM to generate multi-hop QA pairs spanning multiple articles. Answers emerge only after models' training cutoff dates, fundamentally limiting memorization.

**Core Idea**: A fully automated, periodically refreshed, multi-hop news QA pipeline combined with a human-verified subset yields a sustainably effective benchmark for agentic search evaluation.

## Method

### Overall Architecture
LiveNewsBench data construction proceeds in two stages: (1) retrieving news articles from the web and clustering them by event; (2) generating QA pairs from article clusters, followed by multiple rounds of automatic validation and human inspection. At evaluation time, a custom ReAct-style agentic search framework controls the search budget to enable fair comparison.

### Key Designs

1. **News Article Retrieval and Clustering**:

    - Function: Obtains summaries of major global news events from the Wikipedia Current Events Archive, rewrites them into search queries using GPT-4.1, and retrieves relevant news articles via the Brave Search API.
    - Mechanism: Applies a whitelist of approximately 100 reliable news sources and a temporal window spanning three days before to eleven days after the event date; GPT-4.1 then verifies article relevance to each event. Each event is linked to an average of 5.3 articles.
    - Design Motivation: Event-based clustering naturally enables multi-hop questions that draw on multiple articles; the whitelist and temporal window ensure data quality.

2. **Multi-hop QA Generation and Validation**:

    - Function: Uses GPT-5.1 Thinking to generate candidate QA pairs from article clusters.
    - Mechanism: Detailed guidelines ensure that questions require information from multiple articles (enforcing multi-hop reasoning) and that answers are factual, objective, and concise. Generated pairs undergo dual automatic validation: **self-consistency filtering** (the same model answers independently twice; only pairs with consistent answers are retained) and **guideline-compliance verification** (re-evaluated using a paraphrased version of the guidelines).
    - Design Motivation: Self-consistency filtering outperforms filtering with a weaker model and avoids discarding genuinely difficult questions; paraphrased guidelines increase robustness. The human rejection rate is approximately 15%, with 92% inter-annotator agreement.

3. **Data Splits and Update Strategy**:

    - Function: Partitions data into train/val/test splits based on news event timestamps.
    - Mechanism: Events from the most recent two months constitute the test set (minimizing memorization risk), the third most recent month serves as the validation set, and earlier events form the training set. The current version contains 600+ training, 170 validation, and 340 test instances (including a 200-instance human-verified subset).
    - Design Motivation: Chronological splitting ensures the test set reflects genuinely recent events for all evaluated models; the benchmark commits to quarterly updates at an estimated construction cost of approximately $700 per version.

4. **Agentic Web Search Evaluation Framework**:

    - Function: A custom ReAct-style framework in which each step may execute Search (issue a search query), Visit (retrieve the full text of a webpage), or Finish&Answer.
    - Mechanism: The standard configuration allows up to 5 searches and 5 page visits, using the Tavily Search API. GPT-4.1 serves as the judge using a SimpleQA-style prompt.
    - Design Motivation: General-purpose open-source frameworks (e.g., LangChain) are designed for deep research reports rather than factual QA and consume excessive tokens; controlling the search budget enables fair comparison across models.

### Loss & Training
This is a benchmark paper and does not involve model training. However, 600+ open-source training instances are provided to support RLVR training of agentic search models.

## Key Experimental Results

### Main Results

| Model | Reasoning? | Open-source? | Avg. Searches | Avg. Visits | Accuracy (%) |
|-------|-----------|-------------|--------------|------------|-------------|
| GPT-5.2 Official API | ✓ | ✗ | N/A | N/A | **90.0** |
| DeepSeek V3.2 Thinking | ✓ | ✓ | 3.3 | 2.6 | 84.5 |
| DeepSeek V3.2 (No Think) | ✗ | ✓ | 3.4 | 2.6 | 83.0 |
| Claude Sonnet 4.5 | ✗ | ✗ | 2.9 | 1.3 | 82.0 |
| Grok 4 | ✓ | ✗ | 2.7 | 1.7 | 82.0 |
| GPT-5.2 | ✓ | ✗ | 2.9 | 1.8 | 74.0 |
| GPT-4.1 | ✗ | ✗ | 1.7 | 0.6 | 72.5 |
| Gemini 3 Pro | ✓ | ✗ | 3.4 | 0.6 | 60.5 |
| Kimi K2 Thinking | ✓ | ✓ | 2.9 | 1.1 | 48.0 |
| Llama 3.1 8B | ✗ | ✓ | 3.9 | 0.3 | 11.0 |

### Ablation Study (Effect of Search Budget)

| Model | Budget=1 | Budget=3 | Budget=5 | Budget=7 | Gain (1→7) |
|-------|---------|---------|---------|---------|-----------|
| DeepSeek V3.2 Thinking | 48.5 | 80.5 | 84.5 | 84.5 | +36.0 |
| DeepSeek V3.2 (No Think) | 20.0 | 79.0 | 83.0 | 84.5 | +64.5 |
| Claude Sonnet 4.5 | 53.5 | 79.0 | 82.0 | 67.0 | +13.5 |
| GPT-5.2 | 62.5 | 72.5 | 74.0 | 74.5 | +12.0 |
| Kimi K2 Thinking | 7.5 | 47.0 | 48.0 | 52.0 | +44.5 |

### Key Findings
- **Minimal memorization contamination**: Without web access, the strongest model (GPT-5.2) achieves only 21.5% accuracy, compared to 74.0% with search—a gap of 52.5 percentage points. In contrast, FreshQA and SealQA-Hard yield 72.2% and 47.4% without web access, respectively, confirming that LiveNewsBench effectively mitigates memorization.
- **Search budget critically impacts performance**: Increasing budget from 1 to 7 yields substantial gains across all models (4%–64.5%), corroborating the multi-hop nature of the questions.
- **Large variance in tool-use capability**: Kimi K2 Thinking fails to correctly execute search actions on 44% of instances, whereas Claude Sonnet 4.5 fails on only 0.5%.
- **Models prefer search snippets over full-page reading**: Search counts consistently exceed page visit counts across all models.
- **Official APIs do not always outperform the custom framework**: GPT-5.2 improves by 16% with the official API, but Claude's performance drops by 42% under the same condition.

## Highlights & Insights
- **The fully automated update pipeline is the paper's most significant contribution**: From news collection and QA generation to validation, the entire process is automated at a cost of approximately $700 per version, enabling sustainable updates. This represents a substantial leap in scalability over manually curated benchmarks such as FreshQA.
- **Multi-layer validation ensures quality**: The pipeline applies a three-stage funnel—self-consistency filtering → guideline-compliance verification → human inspection. The strong rank correlation between the human-verified subset and the full set confirms the reliability of the automated pipeline.
- **Large-scale training data also provided**: The 600+ open-source training instances address the scarcity of training data for agentic search, and are suitable for RLVR-based training.

## Limitations & Future Work
- Despite the memorization-resistant design, frontier models can still infer some answers through reasoning without web access (e.g., GPT-5.2 achieves 21.5% without search); fully eliminating the influence of parametric knowledge remains difficult.
- Questions are sourced exclusively from English-language news and Western media outlets (approximately 100 whitelisted sources), which may introduce geographic and linguistic biases.
- GPT-4.1 is used as the judge, and its inherent limitations may affect evaluation accuracy.
- Time-sensitive scenarios in which answers evolve as events develop are not examined; the paper explicitly identifies this as future work.
- The quality of the search engine API (Tavily) influences results, yet the paper does not ablate the effect of different search engines.

## Related Work & Insights
- **vs. FreshQA/SealQA**: These benchmarks update answers over time while keeping questions fixed, meaning the questions themselves may already be memorized by models. LiveNewsBench updates both questions and answers simultaneously, providing more thorough memorization resistance.
- **vs. BrowseComp**: BrowseComp uses static, non-updating questions and primarily measures search depth rather than temporal recency. LiveNewsBench emphasizes the combination of news timeliness and multi-hop reasoning.
- **vs. RealTimeQA**: Both benchmarks update periodically, but RealTimeQA produces only approximately 10 questions per update from manually curated trivia on news websites. LiveNewsBench's automated pipeline operates at a substantially larger scale (1,000+ instances per version).
- **vs. Deep Research Bench**: The former evaluates subjective quality of long-form reports, while LiveNewsBench evaluates objective correctness of short factual answers; the two are complementary rather than substitutable.

## Rating
- Novelty: ⭐⭐⭐⭐ The design of a fully automated, periodically refreshed, memorization-resistant search benchmark is genuinely novel, though the core intuition (using news articles to resist memorization) is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluates 13 LLMs and 2 official APIs with multiple ablations (search budget, memorization test, full set vs. human-verified subset); the analysis is highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is logically structured with well-motivated contributions; the comparison table against related work is particularly clear and informative.
- Value: ⭐⭐⭐⭐ Fills an important gap in agentic search evaluation and provides long-term value to the community through its sustainable update mechanism.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains](mc-search_evaluating_and_enhancing_multimodal_agentic_search_with_structured_lon.md)
- [\[ICLR 2026\] ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents](st-webagentbench_a_benchmark_for_evaluating_safety_and_trustworthiness_in_web_ag.md)
- [\[ICLR 2026\] OpenAgentSafety: A Comprehensive Framework for Evaluating Real-World AI Agent Safety](openagentsafety_a_comprehensive_framework_for_evaluating_real-world_ai_agent_saf.md)
- [\[ICLR 2026\] Judge Reliability Harness: Stress Testing the Reliability of LLM Judges](judge_reliability_harness_stress_testing_the_reliability_of_llm_judges.md)
- [\[ICLR 2026\] NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)

<!-- RELATED:END -->
