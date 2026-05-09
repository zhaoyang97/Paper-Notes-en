---
title: >-
  [Paper Note] Promoting Sustainable Web Agents: Benchmarking and Estimating Energy Consumption Through Empirical and Theoretical Analysis
description: >-
  [AAAI 2026][LLM Agent][Web Agent] This paper presents the first systematic quantification of energy consumption and carbon emissions of Web Agents from both empirical benchmarking and theoretical estimation perspectives, finding that higher energy consumption does not equate to better performance, and advocating for the inclusion of energy efficiency metrics in evaluation protocols.
tags:
  - AAAI 2026
  - LLM Agent
  - Web Agent
  - Energy Consumption Benchmarking
  - Carbon Emission Estimation
  - Green AI
  - Sustainable Deployment
date: 2026-05-08
content_hash: 7703bf8057776b7b
---

# Promoting Sustainable Web Agents: Benchmarking and Estimating Energy Consumption Through Empirical and Theoretical Analysis

**Conference**: AAAI 2026
**arXiv**: [2511.04481](https://arxiv.org/abs/2511.04481)
**Code**: [GitHub](https://github.com/DFKIEI/WebAgentEnergy)
**Area**: LLM Agent / Sustainable AI
**Keywords**: Web Agent, Energy Consumption Benchmarking, Carbon Emission Estimation, Green AI, Sustainable Deployment

## TL;DR

This paper presents the first systematic quantification of energy consumption and carbon emissions of Web Agents from both empirical benchmarking and theoretical estimation perspectives, finding that higher energy consumption does not equate to better performance, and advocating for the inclusion of energy efficiency metrics in evaluation protocols.

## Background & Motivation

**Background**: Web Agents (e.g., OpenAI Operator, Google Project Mariner) are rapidly advancing, capable of autonomously browsing the web, filling forms, and comparing prices, representing a critical frontier in LLM applications.

**Limitations of Prior Work**: Sustainability concerns are almost entirely overlooked in current Web Agent research — existing benchmarks focus exclusively on task completion rates (e.g., Step Success Rate) with no energy consumption metrics. End users are presented with a simple input interface and remain completely unaware of the substantial computational energy expenditure behind it.

**Key Challenge**: Differing design philosophies across Web Agents lead to energy consumption gaps exceeding 10×, yet this disparity is entirely opaque to end users. High energy consumption does not necessarily yield better performance.

**Goal**: To quantify energy consumption differences across Web Agents, raise awareness of this issue's urgency within the research community and among users, and promote the integration of energy efficiency dimensions into evaluation standards.

**Key Insight**: The analysis proceeds along two complementary axes — **empirical measurement** (directly benchmarking open-source Agents) and **theoretical estimation** (targeting Agents built on proprietary LLMs).

**Core Idea**: Establish a dual-track evaluation framework — directly measuring energy consumption on real GPUs via carbontracker for open-source Agents, and theoretically estimating energy based on model parameter scale and token counts for closed-source Agents — thereby providing a comprehensive picture of Web Agent energy consumption.

## Method

### Overall Architecture

A dual-track evaluation framework is proposed: (1) **Empirical Benchmarking** — directly measuring energy consumption of 5 open-source Web Agents across 8 GPU types; (2) **Theoretical Estimation** — estimating energy consumption of Agents built on proprietary LLMs based on information from the literature. The two approaches complement each other to cover both open-source and closed-source Agents.

### Key Designs

**Module 1: Empirical Benchmarking**

- **Function**: Five open-source Web Agents (AutoWebGLM, MindAct, MultiUI, Synapse, Synatra) are executed on the Mind2Web benchmark, with GPU energy consumption measured directly using the carbontracker library.
- **Mechanism**: The original Agent code is modified by inserting carbontracker markers at the start and end of execution to capture actual GPU energy usage. Each configuration is run 5 times on 8 NVIDIA GPU types (A100, RTX 3090, H100, H200, L40S, etc.) and averaged.
- **Design Motivation**: Direct measurement is the most accurate approach, contingent on both the Agent and the LLM being open-source. Multiple runs across multiple GPU types ensure result stability and reliability.

**Module 2: Theoretical Energy Estimation**

- **Function**: Estimates energy consumption for Agents using proprietary LLMs (e.g., GPT-4). The core formula is $E_{action} = \bar{N} \cdot e_{token}$, where $\bar{N}$ is the average number of tokens per action and $e_{token}$ is the energy consumption per token.
- **Mechanism**: Agent papers and open-source code are analyzed to identify internal pipelines (input modalities, preprocessing steps, number of LLM calls), after which token counts and per-token energy consumption are estimated for each LLM component. For GPT-4, based on the leaked 1.8T-parameter MoE architecture, FLOPs are derived and mapped to H100 GPU throughput.
- **Design Motivation**: Closed-source Agents cannot be directly measured, yet some basis for comparison is still needed. Using MindAct for both direct measurement and theoretical estimation enables an assessment of the estimation method's accuracy.

**Module 3: Carbon Emission Conversion and Visualization**

- **Function**: Energy consumption is multiplied by the carbon intensity factors of different countries (Norway: 20 g/kWh, USA: 453 g/kWh, Australia: 800 g/kWh) to derive CO₂ emissions, which are further converted into equivalent driving distances.
- **Mechanism**: Renders the environmental cost of different Agents intuitively tangible.
- **Design Motivation**: Raw energy figures (kWh) lack intuitive meaning for most people, whereas expressing impact as "equivalent to driving X kilometers" makes it readily comprehensible.

### Loss & Training

This paper does not involve model training. The evaluation metric framework consists of: (1) **Total Energy Consumption** (kWh); (2) **Energy per Token** (kWh/token); (3) **Energy–Performance Ratio** (energy consumption vs. average Step Success Rate); (4) **CO₂ Emissions** (g CO₂e).

## Key Experimental Results

### Main Results

Comprehensive comparison on an Nvidia H100-NVL GPU:

| Agent | Avg. SSR (%) | Total Energy (kWh) | Runtime (min) |
|-------|-------------|-------------|---------------|
| AutoWebGLM | **53.53** | **0.33** | **57.0** |
| MindAct | 43.50 | 1.22 | 296.0 |
| MultiUI | 34.70 | 0.82 | 130.0 |
| Synapse | 21.67 | 1.74 | 356.0 |
| Synatra | 15.85 | 3.31 | 426.0 |

Theoretical estimation comparison (full Mind2Web):

| Agent | Method | Energy (kWh) |
|-------|------|-----------|
| MindAct | Benchmarking | 1.22 |
| MindAct | Theoretical Estimation | 8.5 |
| LASER (GPT-4) | Theoretical Estimation | 99.21 |

### Ablation Study

- **GPU Variation**: Among the 8 GPU types, the H100-NVL is the most energy-efficient; energy consumption varies substantially across GPUs, but the relative ranking of Agents remains consistent.
- **Estimation vs. Measurement**: The theoretical estimate for MindAct (8.5 kWh) is approximately 7× the measured value (1.22 kWh), indicating that theoretical estimation can only provide order-of-magnitude guidance.
- **Per-Token Energy**: Primarily governed by LLM scale, whereas **total energy consumption** is dominated by total token count — effective preprocessing (e.g., MindAct's HTML pruning) is therefore the key lever for reducing overall energy usage.

### Key Findings

1. AutoWebGLM, the most energy-efficient Agent, is simultaneously the best-performing — **more energy does not equal better results**.
2. The estimated energy consumption of LASER (GPT-4) is more than **10×** that of MindAct.
3. Under the U.S. electricity grid, the carbon emissions from a single complete Mind2Web run of LASER are equivalent to driving 181 kilometers.
4. For fully closed-source Agents (e.g., Operator, Mariner), even theoretical estimation is infeasible.

## Highlights & Insights

- **First systematic quantification of Web Agent energy consumption**: Fills a critical gap in the field and establishes baseline data.
- **Dual-track method design**: Using the same Agent (MindAct) for both direct measurement and theoretical estimation effectively validates the limitations of the estimation approach.
- **Insight on preprocessing as an energy-saving strategy**: The key to Web Agent energy efficiency lies not in model size but in whether clever preprocessing can reduce the total number of tokens that need to be processed.
- **Carbon emission conversion** is intuitive and compelling — translating abstract kWh figures into equivalent driving distances.

## Limitations & Future Work

1. **Limited accuracy of theoretical estimation**: A 7× overestimate indicates that the current method can only provide rough order-of-magnitude references.
2. **Fully closed-source Agents cannot be evaluated**: Agents such as OpenAI Operator and Google Mariner have no publicly available technical details, making even estimation impossible.
3. **Inference energy only**: Training energy costs associated with fine-tuning some Agents are not considered.
4. **Mind2Web benchmark limitations**: Offline benchmarks may not accurately reflect energy consumption patterns in real-world deployment.
5. **No solutions proposed**: The work is primarily diagnostic and measurement-oriented, without proposing concrete technical approaches to reduce energy consumption.

## Related Work & Insights

- **LLM carbon emission research**: GPT-3 training produces approximately 550 tonnes of CO₂, while BERT produces approximately 0.754 tonnes — energy consumption at both training and inference stages warrants attention.
- **Inference energy evaluation**: The energy-per-token metric proposed by Samsi et al. constitutes a useful evaluation indicator.
- **Web Agent diversity**: Design philosophy differences spanning input modalities (HTML/accessibility tree/screenshot) to model selection (open-source/closed-source) directly impact energy consumption.
- **Insight**: Future Agent evaluations should report both performance and energy consumption jointly, analogously to how MLPerf reports accuracy alongside throughput.

## Rating

⭐⭐⭐

The practical value is notable — this paper establishes the first energy consumption benchmark for the Web Agent field, with detailed data and sound experimental design. However, as a research contribution it remains primarily at the level of measurement and advocacy, lacking technical solutions for reducing energy consumption. The accuracy of the theoretical estimation method also has considerable room for improvement.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] LLMTM: Benchmarking and Optimizing LLMs for Temporal Motif Analysis in Dynamic Graphs](llmtm_benchmarking_and_optimizing_llms_for_temporal_motif_analysis_in_dynamic_gr.md)
- [\[AAAI 2026\] CausalTrace: A Neurosymbolic Causal Analysis Agent for Smart Manufacturing](causaltrace_a_neurosymbolic_causal_analysis_agent_for_smart_manufacturing.md)
- [\[AAAI 2026\] COACH: Collaborative Agents for Contextual Highlighting -- A Multi-Agent Framework for Sports Video Analysis](coach_collaborative_agents_for_contextual_highlighting_--_a_multi-agent_framewor.md)
- [\[AAAI 2026\] Prune4Web: DOM Tree Pruning Programming for Web Agent](prune4web_dom_tree_pruning_programming_for_web_agent.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)

<!-- RELATED:END -->
