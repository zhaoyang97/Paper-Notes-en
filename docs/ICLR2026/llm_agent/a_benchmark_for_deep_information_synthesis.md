---
title: >-
  [Paper Note] A Benchmark for Deep Information Synthesis (DeepSynth)
description: >-
  [ICLR 2026][LLM Agent][benchmark] This paper proposes DeepSynth, a benchmark comprising 120 real-world information synthesis tasks spanning 7 domains and 67 countries (averaging 5.5 hours of expert annotation per task). The benchmark requires agents to collect information from multiple web sources and perform structured reasoning. The strongest current agent (o3-deep-research) achieves only 8.97 F1 / 17.5% LLM-Judge, exposing a critical gap in LLM agents' information synthesis capabilities.
tags:
  - ICLR 2026
  - LLM Agent
  - benchmark
  - information synthesis
  - deep research
  - multi-source reasoning
  - agent evaluation
date: 2026-05-08
content_hash: 015728550e7d525e
---

# A Benchmark for Deep Information Synthesis (DeepSynth)

**Conference**: ICLR 2026
**arXiv**: [2602.21143](https://arxiv.org/abs/2602.21143)
**Code**: Available (public data and code)
**Area**: Agent
**Keywords**: benchmark, information synthesis, deep research, multi-source reasoning, agent evaluation

## TL;DR
This paper proposes DeepSynth, a benchmark comprising 120 real-world information synthesis tasks spanning 7 domains and 67 countries (averaging 5.5 hours of expert annotation per task). The benchmark requires agents to collect information from multiple web sources and perform structured reasoning. The strongest current agent (o3-deep-research) achieves only 8.97 F1 / 17.5% LLM-Judge, exposing a critical gap in LLM agents' information synthesis capabilities.

## Background & Motivation
**Background**: LLM agents have advanced rapidly in tool use (web browsing, code execution, data analysis), yet existing benchmarks primarily evaluate shallow fact retrieval or single-source information lookup.

**Limitations of Prior Work**: Existing benchmarks suffer from three issues: (1) most focus on shallow retrieval tasks (e.g., GAIA) that do not require cross-source synthesis; (2) most rely on English and well-known single sources such as Wikipedia; (3) they fail to cover globally diverse information sources and languages.

**Key Challenge**: Real-world information synthesis tasks require collecting structured and unstructured data from multiple sources and performing complex analyses (trend detection, correlation analysis, anomaly detection, etc.)—capabilities that existing benchmarks cannot assess.

**Goal**: Construct a benchmark for evaluating agents' deep information synthesis capabilities, where task answers cannot be retrieved directly but must be derived through multi-step reasoning and cross-source synthesis.

**Key Insight**: Grounded in realistic scenarios (16 domain experts, averaging 5.5 hours per task), the construction pipeline proceeds from source selection → hypothesis formulation → validation and analysis → question generation, ensuring answers are non-memorizable and require genuine synthesis reasoning.

**Core Idea**: Build a realistic benchmark requiring "deep research" capabilities that reveals the substantial gap in current agents' information synthesis performance.

## Method

### Overall Architecture
DeepSynth is a benchmark paper rather than a methodology paper. Its core contributions are the design of 120 tasks, the annotation pipeline, and comprehensive evaluation. Each task includes: a question (avg. 78.5 tokens), gold-standard intermediate reasoning steps (avg. 7.54 steps), supporting evidence URLs, and a JSON-formatted answer.

### Key Designs

1. **Four-Stage Data Construction Pipeline**:

    - **Function**: Proceeds from data source identification → hypothesis generation → hypothesis validation → task formulation, ensuring task authenticity and non-memorability.
    - **Mechanism**: Sixteen domain experts first propose 223 data sources across 7 domains, then formulate verifiable hypotheses for each source, conduct analysis to derive insights, and finally reverse-engineer the analytical process into questions. Crucially, answers cannot be obtained through direct search.
    - **Design Motivation**: Conventional approaches start with answers and then compose questions; this paper reverses the order—starting with analysis and then generating questions—ensuring that tasks require genuine multi-step reasoning rather than recall.

2. **Multi-Dimensional Evaluation Metrics**:

    - **Function**: Combines exact match, F1 (at the key-value pair level), and LLM-Judge across three evaluation tiers.
    - **Mechanism**: EM is the most stringent (all key-value pairs must be correct); F1 measures partial correctness; LLM-Judge permits semantic equivalence and small numerical deviations (1–5.5% margin).
    - **Design Motivation**: Task outputs are in JSON format for automatic verification, while multi-granularity metrics comprehensively reflect model capabilities.

3. **Task Diversity Design**:

    - Covers 7 domains (socioeconomic, finance, environment, science, education, transportation, politics).
    - Spans 67 countries to prevent bias toward English or Western data sources.
    - Encompasses multiple analytical operations: trend detection (21%), ranking (20%), counting and comparison (34%), correlation analysis (7%), and anomaly detection (7%).

### Loss & Training
N/A (benchmark paper)

## Key Experimental Results

### Main Results

| Model/Agent | F1 | EM | LLM-Judge |
|---|---|---|---|
| GPT-4.1 | 3.46 | 0.0 | 0.0 |
| GPT-5.1 | 3.83 | 0.0 | 0.0 |
| GPT-5.2-Pro | 8.70 | 6.25 | 6.67 |
| Gemini-2.5-Pro | 6.25 | 0.0 | 5.0 |
| DeepSeek-R1 | 3.23 | 1.67 | 2.5 |
| **o3-deep-research** | **8.97** | 2.50 | **17.5** |
| Smolagent (GPT-5) | 6.42 | 1.67 | 2.5 |
| OWL (GPT-4.1) | 5.41 | 1.67 | 12.5 |

### Ablation Study (OWL Tool Ablation)

| Configuration | F1 | Notes |
|------|-----|------|
| Full | 5.41 | Complete toolchain |
| − Search | 3.60 | Search is the most critical capability; removal causes a 1.81 drop |
| − Web Browsing | 4.80 | Browsing capability also significant |
| − Doc Processing | 4.90 | Document processing has a smaller impact |
| − Code Execution | 4.82 | Code execution also contributes |

### Key Findings
- **All models score near 0 on EM**: No model can perfectly solve any single task, demonstrating the benchmark's extreme difficulty.
- The F1 gap between reasoning models (o3, R1) and general LLMs (GPT-4.1) is small, indicating that **the bottleneck lies in information acquisition rather than reasoning itself**.
- Tool augmentation helps but is far from sufficient: o3-deep-research outperforms base o3 by 5.68 F1, yet still achieves only ~9 points.
- Best-of-5 improves LLM-Judge to 25%, whereas Self-Consistency@5 reaches only 5%—agent outputs exhibit extremely high variance, occasionally correct but not reliably so.
- Performance on Africa-related tasks drops significantly, exposing model weaknesses on under-represented data sources.

## Highlights & Insights
- **Reveals an important blind spot**: Current "deep research" agents are far from practically capable at information synthesis; among 120 tasks, the best agent can reliably solve only 3.
- **The data construction methodology is highly instructive**: The analysis-first-then-question approach, dual-person verification, and meticulous 5.5-hour-per-task annotation ensure high benchmark quality and resistance to contamination.
- **Bottleneck diagnosis is valuable**: By comparing agent performance with and without tools, the paper clearly identifies information acquisition—rather than reasoning—as the primary bottleneck.

## Limitations & Future Work
- The scale of 120 tasks is relatively small and may not cover all information synthesis scenarios.
- Evaluation relies primarily on JSON exact matching, limiting assessment of open-ended responses.
- Annotation depends on 16 domain-specific experts, which may introduce annotator bias.
- The benchmark does not evaluate agents' use of search engine APIs (focusing primarily on web browsing).

## Related Work & Insights
- **vs. GAIA**: GAIA targets general AI assistant evaluation, whereas DeepSynth focuses on deep reasoning for information synthesis, more closely approximating real-world deep research scenarios.
- **vs. BrowseComp**: BrowseComp emphasizes retrieval difficulty, while DeepSynth places greater emphasis on cross-source synthesis and analysis.
- **vs. FRAMES**: FRAMES addresses fact verification and multi-hop retrieval; DeepSynth additionally requires analysis and structured output generation.

## Supplementary Discussion

### Distinction Between Deep Information Synthesis and RAG
RAG primarily concerns information retrieval and combination, whereas Deep Information Synthesis requires the model to perform multi-step reasoning, cross-source verification, and data integration. This distinction is important—existing RAG benchmarks cannot evaluate an agent's "deep synthesis" capabilities.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First benchmark to systematically evaluate deep information synthesis
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 11 models/agents, multi-dimensional metrics, tool ablation, Best-of-N analysis
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with detailed description of the data construction process
- **Value**: ⭐⭐⭐⭐ Identifies both the direction and the performance gap for deep research agent development

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents](st-webagentbench_a_benchmark_for_evaluating_safety_and_trustworthiness_in_web_ag.md)
- [\[ICLR 2026\] FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents](fingertip_20k_a_benchmark_for_proactive_and_personalized_mobile_llm_agents.md)
- [\[ICLR 2026\] SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents](simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_llm_agents.md)
- [\[AAAI 2026\] ProBench: Benchmarking GUI Agents with Accurate Process Information](../../AAAI2026/llm_agent/probench_benchmarking_gui_agents_with_accurate_process_infor.md)
- [\[CVPR 2026\] HATS: Hardness-Aware Trajectory Synthesis for GUI Agents](../../CVPR2026/llm_agent/hats_hardness-aware_trajectory_synthesis_for_gui_agents.md)

</div>

<!-- RELATED:END -->
