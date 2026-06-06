---
title: >-
  [Paper Note] SoMe: A Realistic Benchmark for LLM-based Social Media Agents
description: >-
  [AAAI 2026][LLM Agent][social media agent] This paper introduces SoMe, the first comprehensive benchmark for social media agents, comprising 8 tasks, over 9 million real-world posts, and 17…
tags:
  - "AAAI 2026"
  - "LLM Agent"
  - "social media agent"
  - "LLM benchmark"
  - "tool-use"
  - "agent evaluation"
  - "social media analysis"
date: 2026-05-08
content_hash: 94f28599af591587
---

# SoMe: A Realistic Benchmark for LLM-based Social Media Agents

**Conference**: AAAI 2026  
**arXiv**: [2512.14720](https://arxiv.org/abs/2512.14720)  
**Code**: [https://github.com/LivXue/SoMe](https://github.com/LivXue/SoMe)  
**Area**: LLM Agent  
**Keywords**: social media agent, LLM benchmark, tool-use, agent evaluation, social media analysis

## TL;DR

This paper introduces SoMe, the first comprehensive benchmark for social media agents, comprising 8 tasks, over 9 million real-world posts, and 17,869 annotated queries. It evaluates 13 mainstream LLMs on social media agent capabilities and reveals substantial performance gaps on complex social tasks.

## Background & Motivation

LLM-driven agents are increasingly deployed on social media platforms for tasks such as event analysis, content recommendation, and user behavior simulation. However, existing evaluation work has notable shortcomings:

1. **Single-task limitations**: Existing benchmarks (e.g., BotSim, TrendSim) focus on a single task (e.g., user simulation) and cannot comprehensively assess agent capabilities.
2. **Insufficient data**: Existing evaluation datasets are limited in scale and lack reliable ground truth (e.g., TrendSim relies solely on LLM-based plausibility evaluation).
3. **Key Challenge**: Social media environments are characterized by high noise, temporal dynamics, and diversity, imposing extreme demands on agents for multi-round data processing and long-context reasoning.

The core idea of SoMe is to construct a comprehensive platform with real-world data, multi-task coverage, and tool interaction, enabling thorough evaluation of LLM agents in realistic social media environments.

## Method

### Overall Architecture

SoMe consists of three components: (1) definitions of 8 social media tasks, (2) 8 agent tool platforms built on the MCP protocol, and (3) over 9 million real-world data points from 32 social platforms. Given a task query, an agent invokes tools to retrieve and analyze data, performs multi-step reasoning, and outputs an answer, which is then evaluated by an LLM-based scorer.

### Key Designs

**Design 1: Hierarchical Evaluation Framework with Three Categories and Eight Tasks**

Tasks are organized into three categories:
- **Post-centric tasks**: Real-time Event Detection (RED), Streaming Event Summarization (SES), Misinformation Detection (MID) — requiring multi-round data processing and external knowledge.
- **User-centric tasks**: User Behavior Prediction (UBP), User Emotion Analysis (UEA), User Comment Simulation (UCS) — requiring understanding of user preferences and behavioral patterns.
- **Comprehensive tasks**: Multi-source Content Recommendation (MCR), Social Media Question Answering (SMQ) — requiring simultaneous analysis of large volumes of posts and users.

This design covers a broad capability spectrum spanning data analysis, user understanding, and knowledge reasoning.

**Design 2: MCP-based Tool Interaction Platform**

Eight tools are provided for data retrieval, management, and analysis:

| Tool Name | Function |
|-----------|----------|
| DataFolder | Outputs data from a specified folder |
| SearchPost | Searches posts by location and time |
| SearchTopic | Searches posts by topic |
| SearchUser | Searches specific users and their posts |
| RetrievePost | Retrieves relevant posts within a data folder |
| RetrieveKnowledge | Retrieves relevant reports from a knowledge base |
| PostClustering | Clusters posts |
| PostSummarization | Summarizes post clusters |

All tools conform to the MCP protocol, ensuring compatibility with mainstream LLMs.

**Design 3: Semi-Automatic Annotation Pipeline**

- For UBP/UCS/MCR: queries and ground truth are generated automatically via templates, requiring no human intervention.
- For RED/SES/UEA/SMQ: multi-round human–LLM interactive annotation is employed (based on Qwen3-32B), verified by 10 professional annotators.
- For MID: the LIAR-RAW and RAWFC open-source datasets are merged.

### Data Statistics

| Task | # Queries | Data Size | Data Type |
|------|-----------|-----------|-----------|
| RED | 568 | 476,611 | Posts |
| SES | 154 | 7,898,959 | Posts |
| MID | 1,451 | 27,137 | Posts & Knowledge |
| UBP | 3,000 | 840,200 | Posts & Users |
| UEA | 2,696 | 840,200 | Posts & Users |
| UCS | 4,000 | 840,200 | Posts & Users |
| MCR | 4,000 | 840,200 | Posts & Users |
| SMQ | 2,000 | 8,651,759 | Posts & Users |
| **Total** | **17,869** | **9,242,907** | All |

## Key Experimental Results

### Main Results

Performance of 13 mainstream LLMs across 8 tasks (scores ranging from 0 to 100):

| Model | Size | RED | SES | MID | UBP | UEA | UCS | MCR | SMQ | Avg. |
|-------|------|-----|-----|-----|-----|-----|-----|-----|-----|------|
| Gemini-2.5-Flash | N/A | 54.92 | **44.87** | 45.62 | 57.50 | 41.94 | 56.00 | 62.75 | 71.01 | **54.33** |
| GPT-4o | N/A | 47.59 | 36.17 | 50.24 | 55.17 | 31.53 | 52.48 | 61.63 | 64.21 | 49.88 |
| Qwen3-32B | 32B | 44.25 | 41.04 | 47.42 | **67.03** | 33.53 | 54.98 | **63.28** | **80.27** | 53.98 |
| Qwen3-8B | 8B | 40.38 | 36.69 | 45.21 | 61.73 | 33.03 | 53.33 | 60.55 | 76.18 | 50.89 |
| DeepSeek-R1-Qwen3-8B | 8B | 17.71 | 28.83 | 26.46 | 43.53 | 21.18 | 31.10 | 34.33 | 51.84 | 31.87 |
| Llama-3.1-8B | 8B | 3.37 | 20.78 | 40.45 | 34.23 | 37.11 | 33.98 | 47.40 | 31.05 | 31.65 |

### Task Completion Rate (TCR) Analysis

| Model | Size | Avg. TCR |
|-------|------|----------|
| DeepSeek-V3 | 671B | 98.60% |
| Qwen3-32B | 32B | 98.45% |
| Gemini-2.5-Flash | N/A | 96.61% |
| Llama-3.1-8B | 8B | 67.20% |
| DeepSeek-R1-Qwen3-8B | 8B | 69.73% |

### Key Findings

- **Universally suboptimal performance**: Most tasks score below 70; open-ended tasks such as RED, SES, and MID fall below 50.
- **Reasoning ability ≠ agent ability**: Despite strong reasoning performance, DeepSeek-R1-Qwen3-8B consistently underperforms Qwen3-8B across all 8 agent tasks (with drops of 21%–56% per task).
- **Model scale effect**: Qwen3-32B > Qwen3-14B (+2.2%) > Qwen3-8B (+3.8%).
- **Severe tool hallucination**: DeepSeek-R1 and Devstral exhibit tool invocation hallucination rates of 29% and 28%, respectively.
- **Widespread tool response hallucination**: Even Kimi-K2 at the 1T scale shows a 7% tool response hallucination rate.

## Highlights & Insights

- The first comprehensive social media agent benchmark covering 8 tasks and over 9 million real-world data points.
- Reveals the important finding that strong reasoning ability does not imply strong agent ability, offering guidance for future LLM training.
- The analysis of tool hallucination (response hallucination vs. invocation format hallucination) opens a new avenue for agent reliability research.
- The MCP-protocol-based tool platform design ensures broad compatibility.

## Limitations & Future Work

- Evaluation relies on LLM-based scorers for RED/SES/SMQ, which may introduce scoring bias.
- Coverage is currently limited to English and Chinese content, lacking multilingual evaluation.
- Task difficulty is unevenly distributed; some tasks (e.g., SMQ) are relatively straightforward, limiting discriminative power.
- Multi-agent collaboration scenarios are not evaluated.

## Related Work & Insights

- **vs. BotSim**: BotSim evaluates only user behavior simulation, whereas SoMe covers 8 tasks with a far larger dataset (9M vs. tens of thousands).
- **vs. TrendSim**: TrendSim lacks ground truth and relies solely on LLM-based plausibility assessment; SoMe provides annotations verified by 10 human annotators.
- **vs. OSWorld/WebArena**: These benchmarks assess computer/web navigation tasks, whereas social media environments are considerably noisier and more open-ended.

## Rating

- Novelty: ⭐⭐⭐⭐ The first comprehensive social media agent benchmark, achieving new heights in both task design and data scale.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation of 13 models across 8 tasks, including in-depth analysis of tool hallucination.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, with thorough analysis and rich figures and tables.
- Value: ⭐⭐⭐⭐ Significantly advances research on social media agents and reveals a critical distinction between reasoning and agent capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents](../../ICLR2026/llm_agent/fingertip_20k_a_benchmark_for_proactive_and_personalized_mobile_llm_agents.md)
- [\[ICLR 2026\] SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents](../../ICLR2026/llm_agent/simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_llm_agents.md)
- [\[ICML 2026\] Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use](../../ICML2026/llm_agent/reward_hacking_benchmark_measuring_exploits_in_llm_agents_with_tool_use.md)
- [\[ICLR 2026\] A Benchmark for Deep Information Synthesis (DeepSynth)](../../ICLR2026/llm_agent/a_benchmark_for_deep_information_synthesis.md)
- [\[ICLR 2026\] The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution](../../ICLR2026/llm_agent/the_tool_decathlon_benchmarking_language_agents_for_diverse_realistic_and_long-h.md)

</div>

<!-- RELATED:END -->
