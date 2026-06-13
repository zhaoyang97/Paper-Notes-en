---
title: >-
  [Paper Note] MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools
description: >-
  [ACL 2026][LLM Agent][Model Context Protocol] MCP-Flow proposes a Web Agent-based automated pipeline to collect tool information from 1,166 real-world MCP servers and synthesize 68…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Model Context Protocol"
  - "Tool Use"
  - "Automated Data Construction"
  - "Large-scale Benchmark"
date: 2026-05-08
content_hash: 8fecdfaadd55760f
---

# MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools

**Conference**: ACL 2026  
**arXiv**: [2510.24284](https://arxiv.org/abs/2510.24284)  
**Code**: [https://github.com/wwh0411/MCP-Flow](https://github.com/wwh0411/MCP-Flow)  
**Area**: LLM Agent / Tool Use  
**Keywords**: Model Context Protocol, Tool Use, Automated Data Construction, Large-scale Benchmark, LLM Agent

## TL;DR

MCP-Flow proposes a Web Agent-based automated pipeline to collect tool information from 1,166 real-world MCP servers and synthesize 68,733 high-quality training data points. This enables small-scale fine-tuned models (0.6B-8B) to outperform SOTA large models such as GPT-4o in MCP tool utilization.

## Background & Motivation

**Background**: The Model Context Protocol (MCP) is rapidly evolving as a unified framework for interactions between LLMs and external tools, with a surge of MCP servers and tools appearing in the community. Existing research has begun building benchmarks to evaluate LLMs' MCP capabilities, but significant limitations remain.

**Limitations of Prior Work**: Three critical issues exist: (1) existing MCP studies cover only a small number of servers ($\le 20$), which is far below the scale and diversity of the real MCP ecosystem; (2) server collection relies heavily on manual curation, failing to keep pace with rapid growth; (3) no existing framework provides training support, serving only as evaluation platforms without translating results into model capability enhancements.

**Key Challenge**: There is a substantial gap between the complexity, diversity, and rapid growth of the real-world MCP ecosystem and the limited MCP usage capabilities of current LLMs—even SOTA models like Claude-3.5-Sonnet perform poorly in simple settings.

**Goal**: (1) Automate large-scale server discovery and data collection; (2) synthesize high-quality and diverse training data; (3) validate dataset value through training, retrieval augmentation, and complex task evaluation.

**Key Insight**: Utilize Web Agents (Playwright) to automate navigation through MCP marketplace platforms as a replacement for manual scraping; combine few-shot generation, slot filling, and WizardLM evolution to synthesize diverse data.

**Core Idea**: Construct a large-scale real-world MCP training dataset using an automated pipeline (server discovery + data synthesis + rigorous filtering), allowing small models to reach or even exceed the MCP tool-use capabilities of large models through fine-tuning.

## Method

### Overall Architecture

MCP-Flow consists of two core phases: (1) Automated Server and Tool Collection—crawling server information from six MCP marketplace platforms via Web Agents; (2) Scalable Data Synthesis—including data generation (few-shot generation + slot filling + WizardLM evolution + function call generation + tool response collection) and data filtering (embedding similarity + tool call verification + quality scoring). The final output comprises 68,733 instruction-function call pairs and 6,439 trajectories covering 1,166 servers and 11,536 tools.

### Key Designs

1.  **Web Agent Automated Server Crawling**:
    *   **Function**: Automates the discovery and collection of MCP server configuration information.
    *   **Mechanism**: A Playwright Web Agent autonomously navigates to target server pages within predefined workflows, capturing JSON configuration files via page snapshots. It supports platforms including Smithery, Glama, MCP.so, MCPHub, PipeDream, and PulseMCP. A critical component is server de-duplication—using tool description lists as the criterion (rather than names or providers) since the same server may use different names and interfaces across platforms. Local deployment handles stdio servers via npm/uvx, while SSE servers connect via URLs.
    *   **Design Motivation**: Traditional crawlers require custom parsing logic for each website's HTML structure. Web Agents operate with high-level instructions and possess cross-platform generalization capabilities. After an initial large-scale crawl, only incremental updates for newly released servers are required.

2.  **Scalable Data Synthesis Pipeline**:
    *   **Function**: Synthesizes high-quality instruction-function call pairs starting from tool information.
    *   **Mechanism**: A four-step process: (a) Few-shot generation: generates 5 instructions starting from the tool to ensure inherent ground-truth labels; (b) Slot filling: treats required tool parameters as slots, automatically generating missing values and revising instructions to include them; (c) WizardLM evolution: randomly selects evolution directions (e.g., concretization, reasoning) with a depth of 2 to increase complexity and diversity; (d) Function call generation: generates formatted function calls using GPT-4o based on ground-truth tools and input schemas.
    *   **Design Motivation**: Starting from the tool perspective ensures the correspondence between instructions and tools. Slot filling guarantees parameter integrity, and WizardLM evolution balances generation costs with diversity.

3.  **Strict Data Filtering**:
    *   **Function**: Ensures the quality of training data.
    *   **Mechanism**: Three layers of filtering: (a) Embedding similarity filtering: calculates similarity between instructions and tool descriptions, discarding those exceeding a 0.8 threshold (to avoid trivial selection); (b) Tool call verification: requires GPT-4o and DeepSeek-V3 to select the correct tool from the labeled tool and two random candidates, discarding the sample if both fail; (c) Quality scoring: uses DeepSeek-V3 for scoring, discarding samples rated below 6/10.
    *   **Design Motivation**: Multi-layer filtering ensures data quality from different angles—avoiding trivial instructions, ensuring label correctness, and guaranteeing overall quality.

### Loss & Training

Standard instruction fine-tuning is employed to train Qwen2-0.5B/7B (noted as 0.6B/8B benchmarks) and Llama3.1-8B on synthetic data. Additionally, a retrieval-augmented scheme is provided—retrieving similar examples from the dataset to enhance the MCP capabilities of closed-source models.

## Key Experimental Results

### Main Results

**MCP Tool Selection and Formatting (10-Tool Setting)**

| Model | Seen Tool | Unseen Tool | Unseen Server |
| :--- | :--- | :--- | :--- |
| GPT-4o (Tool/Param/AST) | 88.6/68.2/58.8 | 85.0/71.4/62.0 | 81.4/55.6/50.8 |
| Claude-3.5-Sonnet | 85.8/68.6/56.6 | 83.0/74.4/63.6 | 72.6/56.0/48.4 |
| MCP-Flow-Qwen-0.6B | **96.8/87.2/75.4** | **98.2/86.8/75.2** | **98.4/70.6/58.0** |
| MCP-Flow-Qwen-4B | **99.2/91.8/81.2** | **98.6/91.4/78.2** | **98.4/72.2/59.8** |
| MCP-Flow-Llama-8B | **98.6/91.0/81.6** | **99.0/91.2/77.6** | **99.4/77.0/65.2** |

**100-Tool Large-Scale Setting**

| Model | Tool | Param | AST |
| :--- | :--- | :--- | :--- |
| GPT-4o | 72.3 | 66.9 | 53.8 |
| MCP-Flow-Qwen-0.6B | 64.7 | 63.4 | 51.6 |
| MCP-Flow-Qwen-4B | **81.7** | **82.1** | **67.0** |

### Ablation Study

| Configuration | Key Metrics | Note |
| :--- | :--- | :--- |
| Full MCP-Flow Data | — | Baseline |
| Remove WizardLM Evolution | Decrease | Reduced instruction complexity and diversity |
| Remove Slot Filling | Decrease | Incomplete parameter coverage |
| Remove Quality Filtering | Decrease | Noise introduced by low-quality samples |

### Key Findings

*   **0.6B models can outperform GPT-4o**: MCP-Flow-Qwen-0.6B achieved 96.8% Tool accuracy in the 10-tool setting, far exceeding GPT-4o's 88.6%, proving the value of specialized training data.
*   Performance for all models declines as the number of tools increases, but MCP-Flow models decline more slowly; the 4B model still outperforms GPT-4o in the 100-tool setting.
*   Retrieval augmentation improved GPT-4o's Tool accuracy on Seen Test from 88.6% to 91.2% (+2.6%) and Unseen Tool from 85.0% to 87.8% (+2.8%).
*   On complex GAIA agent tasks, replacing initial tool calls with MCP-Flow improves agent performance while reducing inference costs.

## Highlights & Insights

*   The Web Agent-driven automated crawling pipeline is highly practical—it eliminates the need for platform-specific parsing code and supports incremental updates, combining engineering wisdom with research contribution.
*   The insight to use tool descriptions rather than names for de-duplication reflects a deep understanding of the MCP ecosystem.
*   The tool-centric data synthesis approach (determining the target tool before generating instructions) ensures label accuracy, which is more reliable than the reverse process (generating instructions before labeling tools).

## Limitations & Future Work

*   Servers requiring API keys or proprietary software were excluded, which may contain important production-grade tools.
*   Data synthesis depends on GPT-4o, introducing dependency on a specific model.
*   Evaluation focused primarily on tool selection and formatting, with limited assessment of multi-step tool-chain reasoning.
*   Server quality is inconsistent; responses returned by some servers may be unreliable.

## Related Work & Insights

*   **vs ToolBench**: ToolBench uses RapidAPI’s REST APIs, which are unstable and lack standardization; MCP-Flow leverages the unified MCP protocol for more reliable tool interaction.
*   **vs MCPBench/MCP-Zero**: These benchmarks cover only 10-300 servers and do not provide training support; MCP-Flow covers 1,166 servers and provides a complete training dataset.

## Rating

*   Novelty: ⭐⭐⭐⭐ The automated pipeline and large-scale dataset construction provide practical contributions, though the core methods (few-shot + evolution + filtering) are not entirely new.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, involving multi-model comparisons, three test splits, varying tool counts, retrieval augmentation, and Agent task evaluations.
*   Writing Quality: ⭐⭐⭐⭐ Clear structure, though certain details (e.g., selection of filtering thresholds) could be discussed more fully.
*   Value: ⭐⭐⭐⭐⭐ Fills the gap in training data for the MCP field and significantly advances research into LLM Agent tool use.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MCP-Persona: Evaluating LLM Agent Capabilities in Real Personalized Applications via Environment Simulation](../../ICML2026/llm_agent/mcp-persona_benchmarking_llm_agents_on_real-world_personal_applications_via_envi.md)
- [\[ACL 2026\] AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts](agencybench_benchmarking_the_frontiers_of_autonomous_agents_in_1m-token_real-wor.md)
- [\[ACL 2026\] Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models](don39t_adapt_small_language_models_for_tools_adapt_tool_schemas_to_the_models.md)
- [\[ACL 2026\] Shopping Companion: A Memory-Augmented LLM Agent for Real-World E-Commerce Tasks](shopping_companion_a_memory-augmented_llm_agent_for_real-world_e-commerce_tasks.md)
- [\[ACL 2026\] OctoTools: An Agentic Framework with Extensible Tools for Complex Reasoning](octotools_an_agentic_framework_with_extensible_tools_for_complex_reasoning.md)

</div>

<!-- RELATED:END -->
