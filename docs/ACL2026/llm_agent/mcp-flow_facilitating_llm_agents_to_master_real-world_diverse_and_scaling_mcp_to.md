---
title: >-
  [Paper Note] MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools
description: >-
  [ACL 2026][LLM Agent][Model Context Protocol] MCP-Flow proposes a Web Agent-based automated pipeline that collects tool information from 1,166 real-world MCP servers and synthesizes 68,733 high-quality training samples, enabling small fine-tuned models (0.6B–8B) to surpass SOTA large models such as GPT-4o on MCP tool use.
tags:
  - ACL 2026
  - LLM Agent
  - Model Context Protocol
  - tool use
  - automated data construction
  - large-scale benchmark
date: 2026-05-08
content_hash: 68c5f85d0f9028b6
---

# MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools

**Conference**: ACL 2026
**arXiv**: [2510.24284](https://arxiv.org/abs/2510.24284)
**Code**: [https://github.com/wwh0411/MCP-Flow](https://github.com/wwh0411/MCP-Flow)
**Area**: LLM Agent / Tool Use
**Keywords**: Model Context Protocol, tool use, automated data construction, large-scale benchmark, LLM Agent

## TL;DR

MCP-Flow proposes a Web Agent-based automated pipeline that collects tool information from 1,166 real-world MCP servers and synthesizes 68,733 high-quality training samples, enabling small fine-tuned models (0.6B–8B) to surpass SOTA large models such as GPT-4o on MCP tool use.

## Background & Motivation

**Background**: The Model Context Protocol (MCP) is rapidly emerging as a unified framework for LLM–tool interaction, with a growing number of MCP servers and tools appearing in the community. Existing studies have begun constructing benchmarks to evaluate models' MCP usage capabilities, but significant limitations remain.

**Limitations of Prior Work**: Three critical issues: (1) existing MCP research covers only a small number of servers (≤20), far below the scale and diversity of the real MCP ecosystem; (2) server collection relies heavily on manual curation and cannot keep pace with the rapid growth of MCP servers; (3) no existing framework provides training support—they serve only as evaluation platforms and cannot translate evaluation results into improved model capabilities.

**Key Challenge**: A gap exists between the complexity, diversity, and rapid growth of the real MCP ecosystem and the limited MCP usage capabilities of current LLMs—even SOTA models (e.g., Claude-4-Sonnet) perform poorly under simple settings.

**Goal**: (1) Automate large-scale server discovery and data collection; (2) synthesize high-quality and diverse training data; (3) validate the dataset's value across three dimensions: fine-tuning, retrieval augmentation, and complex task evaluation.

**Key Insight**: Leverage a Web Agent (Playwright) to automate navigation of MCP marketplace platforms, replacing manual crawling; combine few-shot generation, slot filling, and WizardLM evolution to synthesize diverse data.

**Core Idea**: Build a large-scale real-world MCP training dataset via an automated pipeline (server discovery + data synthesis + rigorous filtering), enabling small models to match or surpass large models in MCP tool use through fine-tuning.

## Method

### Overall Architecture

MCP-Flow comprises two core stages: (1) automated server and tool collection—crawling server information from six MCP marketplace platforms via a Web Agent; (2) scalable data synthesis—including data generation (few-shot generation + slot filling + WizardLM evolution + function call generation + tool response collection) and data filtering (embedding similarity + tool call validation + quality scoring). The final dataset covers 1,166 servers and 11,536 tools, yielding 68,733 instruction–function call pairs and 6,439 trajectories.

### Key Designs

1. **Web Agent-Based Automated Server Crawling**:

    - Function: Automated discovery and collection of MCP server configuration information.
    - Mechanism: A Playwright Web Agent autonomously navigates to target server pages within predefined workflows, capturing configuration files in JSON format via page snapshots. Six platforms are supported: Smithery, Glama, MCP.so, MCPHub, PipeDream, and PulseMCP. A key contribution is server deduplication—using the list of tool descriptions as the identity criterion (rather than name or provider), since the same server may appear under different names and interfaces across platforms. Local deployment handles stdio servers via npm/uvx; SSE servers connect via URL.
    - Design Motivation: Traditional crawlers require custom HTML parsing logic for each website, whereas a Web Agent operates with high-level instructions and generalizes across platforms. After an initial large-scale crawl, only incremental updates for newly released servers are needed.

2. **Scalable Data Synthesis Pipeline**:

    - Function: Synthesize high-quality instruction–function call pairs from tool information.
    - Mechanism: A four-step process—(a) Few-shot generation: generate 5 instructions per tool, ensuring intrinsic ground-truth labels; (b) Slot filling: treat required tool parameters as slots, automatically generate missing values, and revise instructions to include these parameters; (c) WizardLM evolution: randomly select an evolution direction (e.g., concretization, reasoning) with depth 2 to increase instruction complexity and diversity; (d) Function call generation: use GPT-4o to generate formatted function calls based on the ground-truth tool and input schema.
    - Design Motivation: Generating instructions from the tool perspective ensures correct instruction–tool correspondence; slot filling guarantees parameter completeness; WizardLM evolution balances generation cost and diversity.

3. **Rigorous Data Filtering**:

    - Function: Ensure training data quality.
    - Mechanism: Three-layer filtering—(a) Embedding similarity filtering: compute embedding similarity between instructions and tool descriptions, discarding instructions above a threshold of 0.8 (excessively close similarity trivializes tool selection); (b) Tool call validation: GPT-4o and DeepSeek-V3 each select the correct tool from the annotated tool and two random candidates; samples where both fail are discarded; (c) Quality scoring: DeepSeek-V3 assigns a score, and samples below 6/10 are discarded.
    - Design Motivation: Multi-layer filtering ensures data quality from different angles—avoiding trivial instructions, ensuring annotation correctness, and maintaining overall quality.

### Loss & Training

Standard instruction fine-tuning is applied on the synthesized data to train Qwen3-0.6B/4B and Llama3.1-8B. A retrieval augmentation scheme is also provided, retrieving similar examples from the dataset to enhance closed-source models' MCP usage capabilities.

## Key Experimental Results

### Main Results

**MCP Tool Selection and Formatting (10-Tool Setting)**

| Model | Seen Tool | Unseen Tool | Unseen Server |
|-------|-----------|-------------|---------------|
| GPT-4o (Tool/Param/AST) | 88.6/68.2/58.8 | 85.0/71.4/62.0 | 81.4/55.6/50.8 |
| Claude-4-Sonnet | 85.8/68.6/56.6 | 83.0/74.4/63.6 | 72.6/56.0/48.4 |
| MCP-Flow-Qwen-0.6B | **96.8/87.2/75.4** | **98.2/86.8/75.2** | **98.4/70.6/58.0** |
| MCP-Flow-Qwen-4B | **99.2/91.8/81.2** | **98.6/91.4/78.2** | **98.4/72.2/59.8** |
| MCP-Flow-Llama-8B | **98.6/91.0/81.6** | **99.0/91.2/77.6** | **99.4/77.0/65.2** |

**Large-Scale 100-Tool Setting**

| Model | Tool | Param | AST |
|-------|------|-------|-----|
| GPT-4o | 72.3 | 66.9 | 53.8 |
| MCP-Flow-Qwen-0.6B | 64.7 | 63.4 | 51.6 |
| MCP-Flow-Qwen-4B | **81.7** | **82.1** | **67.0** |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|------------|------|
| Full MCP-Flow data | — | Baseline |
| Remove WizardLM evolution | Decrease | Reduced instruction complexity and diversity |
| Remove slot filling | Decrease | Incomplete parameter coverage |
| Remove quality filtering | Decrease | Low-quality samples introduce noise |

### Key Findings

- **A 0.6B model can surpass GPT-4o**: MCP-Flow-Qwen-0.6B achieves a Tool accuracy of 96.8% in the 10-tool setting, far exceeding GPT-4o's 88.6%, demonstrating the value of specialized training data.
- Performance degrades for all models as the number of tools increases, but MCP-Flow models degrade more slowly; in the 100-tool setting, the 4B model still outperforms GPT-4o.
- Retrieval augmentation improves GPT-4o's Tool accuracy on Seen Test from 88.6% to 91.2% (+2.6%) and on Unseen Tool from 85.0% to 87.8% (+2.8%).
- On complex GAIA Agent tasks, replacing initial tool calls with MCP-Flow improves agent performance while reducing inference cost.

## Highlights & Insights

- The Web Agent-driven automated crawling pipeline is highly practical—it requires no platform-specific parsing code and supports incremental updates, representing a synthesis of engineering ingenuity and research contribution.
- The insight of using tool descriptions rather than names for deduplication reflects a deep understanding of the MCP ecosystem.
- The tool-centric data synthesis approach (determining the target tool before generating instructions) ensures label accuracy and is more reliable than the reverse pipeline (generating instructions first and then annotating tools).

## Limitations & Future Work

- Servers requiring API keys or proprietary software are excluded, potentially omitting important production-grade tools.
- Data synthesis relies on GPT-4o, introducing a dependency on a specific proprietary model.
- Evaluation is limited to tool selection and formatting; assessment of multi-step tool-chain reasoning remains insufficient.
- Server quality varies considerably, and some servers may return unreliable responses.

## Related Work & Insights

- **vs. ToolBench**: ToolBench uses RapidAPI REST APIs, which are unstable and lack standardization; MCP-Flow leverages the MCP unified protocol for more reliable tool interaction.
- **vs. MCPBench/MCP-Zero**: These benchmarks cover only 10–300 servers and provide no training support; MCP-Flow covers 1,166 servers and provides a complete training dataset.

## Rating

- Novelty: ⭐⭐⭐⭐ The automated pipeline and large-scale dataset construction offer practical contributions, though the core methods (few-shot + evolution + filtering) are not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-model comparisons, three test splits, varying tool-count settings, retrieval augmentation, and agent task evaluation are comprehensively conducted.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear, though certain details (e.g., the rationale for filtering thresholds) could be more thoroughly discussed.
- Value: ⭐⭐⭐⭐⭐ Fills a critical gap in MCP-domain training data and significantly advances research on LLM Agent tool use.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts](agencybench_benchmarking_the_frontiers_of_autonomous_agents_in_1m-token_real-wor.md)
- [\[ICLR 2026\] OpenAgentSafety: A Comprehensive Framework for Evaluating Real-World AI Agent Safety](../../ICLR2026/llm_agent/openagentsafety_a_comprehensive_framework_for_evaluating_real-world_ai_agent_saf.md)
- [\[AAAI 2026\] D-GARA: A Dynamic Benchmarking Framework for GUI Agent Robustness in Real-World Anomalies](../../AAAI2026/llm_agent/d-gara_a_dynamic_benchmarking_framework_for_gui_agent_robust.md)
- [\[ACL 2026\] Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration](scaling_external_knowledge_input_beyond_context_windows_of_llms_via_multi-agent_.md)
- [\[ACL 2026\] HeLa-Mem: Hebbian Learning and Associative Memory for LLM Agents](hela-mem_hebbian_learning_and_associative_memory_for_llm_agents.md)

</div>

<!-- RELATED:END -->
