---
title: >-
  [Paper Note] Note 4: WebThinker — Empowering Reasoning Models with Deep Research Capabilities
description: >-
  [NeurIPS 2025][Deep Research] WebThinker equips large reasoning models (LRMs) with autonomous web search and navigation capabilities. Through a Think-Search-Draft strategy, it seamlessly interleaves reasoning…
tags:
  - "NeurIPS 2025"
  - "Deep Research"
  - "Web Navigation"
  - "Interactive Search"
  - "DPO Training"
  - "Multi-step Reasoning"
date: 2026-05-08
content_hash: a3c2a504db0f12bc
---

# Note 4: WebThinker — Empowering Reasoning Models with Deep Research Capabilities

**Conference**: NeurIPS 2025  
**arXiv**: [2504.21776](https://arxiv.org/abs/2504.21776)  
**Code**: [GitHub](https://github.com/RUC-NLPIR/WebThinker)  
**Area**: Other  
**Keywords**: Deep Research, Web Navigation, Interactive Search, DPO Training, Multi-step Reasoning

## TL;DR
WebThinker equips large reasoning models (LRMs) with autonomous web search and navigation capabilities. Through a Think-Search-Draft strategy, it seamlessly interleaves reasoning, information gathering, and report generation. After reinforcement learning optimization, it surpasses o1 and Gemini on complex reasoning and scientific report generation tasks.

## Background & Motivation
**Knowledge Silos in LRMs**: Reasoning models such as o1 and DeepSeek-R1 rely on static parametric knowledge, making them ill-suited for dynamic, knowledge-intensive tasks and incapable of generating comprehensive research reports.

**Limitations of RAG**: Standard RAG pipelines are statically predefined, lacking tight interaction between LRMs and search engines, which severely constrains decision-making capability.

**Open-Source Gap**: Deep research systems from OpenAI, Google, and xAI are largely closed-source, leaving the academic community without reproducible open frameworks.

**Core Requirement**: For complex real-world reasoning, models must dynamically detect knowledge gaps, autonomously retrieve information, and continuously update their reasoning state.

## Method

### Overall Architecture
WebThinker adopts a dual-mode design:
1. **Question-Answering Mode**: Equipped with a deep web browser that triggers web search whenever a knowledge gap is encountered during reasoning.
2. **Report Generation Mode**: Integrates the Think-Search-Draft strategy to simultaneously search, reason, and compose reports.

### Key Designs
**Deep Web Browser Component** $\mathcal{T}_{exp}$:
- **Search Tool** $\mathcal{T}_s$: Retrieves relevant web pages given a query $q_s$.
- **Navigation Tool** $\mathcal{T}_n$: Clicks links or buttons to interact with pages, supporting multi-hop navigation.
- **Recursive Reasoning**: The browser generates its own reasoning chain $\mathcal{R}_e$ to decide whether to navigate deeper or initiate a new search.

The generation process is modeled as:
$$P(\mathcal{R}_e,\mathcal{O}_{exp}|q_s,\mathcal{D},I_e) = \prod_{t=1}^{T_e}P(\mathcal{R}_{e,t}|\mathcal{R}_{e,<t},q_s,\mathcal{D}_t,I_e)·P(\mathcal{O}_{exp}|\mathcal{R}_e,q_s,\mathcal{D},I_e)$$

**Think-Search-Draft Strategy**:
A division of labor between a primary LRM and an assistant LRM:
- **Primary LRM**: Orchestrates overall reasoning, deciding when and what to search.
- **Assistant LRM** $\mathcal{T}_{write}=\{\mathcal{T}_{draft}, \mathcal{T}_{check}, \mathcal{T}_{edit}\}$: Executes text operations.
- **Document Memory** $\mathcal{M}$: Accumulates all browsed pages and provides context retrieval for the report-writing assistant.

**Online DPO Reinforcement Learning**:
Preference data construction follows a three-level priority scheme:
1. **Correctness First**: Correct answers or high-quality reports are preferred over incorrect or low-quality ones.
2. **Tool Efficiency**: Given equal correctness, fewer tool calls are preferred over more.
3. **Conciseness**: Given equal tool calls, more concise outputs are preferred over verbose ones (beyond threshold $\gamma=1.5$).

Preference pairs $(\mathcal{R}_w, \mathcal{R}_l)$ are constructed and iteratively optimized using the standard DPO loss:
$$\mathcal{L}_{DPO} = -\mathbb{E}[\log\sigma(\beta\log\frac{\pi_\theta(\mathcal{R}_w|I,q)}{\pi_{ref}(\mathcal{R}_w|I,q)} - \beta\log\frac{\pi_\theta(\mathcal{R}_l|I,q)}{\pi_{ref}(\mathcal{R}_l|I,q)})]$$

## Key Experimental Results

### Complex Reasoning Tasks — Pass@1 Accuracy

| Model | GPQA (Avg.) | GAIA (Avg.) | WebWalkerQA (Avg.) | HLE (Avg.) |
|:---:|:---:|:---:|:---:|:---:|
| **Baselines** | | | | |
| Qwen2.5-32B | 43.4% | 13.6% | 3.1% | 6.2% |
| DeepSeek-R1-32B | 62.6% | 17.5% | 3.8% | 8.5% |
| QwQ-32B | 64.1% | 22.3% | 4.3% | 12.1% |
| **WebThinker Results** | | | | |
| WebThinker (32B) | **71.8%** | **39.2%** | **18.6%** | **28.4%** |
| Relative Gain | +14.5% | +76.0% | **+333%** | +135% |

### Report Generation Task (Glaive Dataset)

| Method | Auto Eval (GPT-Judge) | Human Eval: Content Accuracy | Human Eval: Completeness |
|:---:|:---:|:---:|:---:|
| Qwen2.5-32B-RAG | 52.0% | 58% | 71% |
| DeepSeek-R1 (Reasoning Only) | 56.3% | 62% | 68% |
| WebThinker | **68.7%** | **79%** | **92%** |
| Grok-3 (Closed-Source Baseline) | 64.2% | 76% | 88% |

### Key Findings
1. **76% Gain on GAIA**: WebThinker achieves a 76% relative improvement over the strongest baseline (QwQ), underscoring the critical role of web interaction.
2. **Breakthrough on HLE**: Surpasses Gemini-2.0 on the most challenging frontier math tasks, demonstrating a threshold effect of search-augmented reasoning.
3. **Comprehensive Report Quality Superiority**: Outperforms closed-source systems on both content accuracy and completeness, validating the feasibility of open-source deep research solutions.

## Highlights & Insights
1. **Architectural Innovation**: First work to deeply integrate LRMs with web search, breaking the knowledge silo barrier.
2. **DPO Optimization**: The multi-level preference design (correctness → efficiency → conciseness) automatically induces desirable tool-use patterns.
3. **Open-Source Contribution**: Code and data are publicly released, providing the academic community with a reproducible deep research framework.
4. **Practical Utility**: Report generation surpasses closed-source systems, validating the multiplicative effect of sequential reasoning combined with web browsing.

## Limitations & Future Work
1. The search environment relies on Wikipedia and live web page accuracy; performance across different environments remains untested.
2. Multi-stage report evaluation still requires human validation, and automatic evaluation metrics are limited.
3. The trade-off between reasoning length and search steps is not thoroughly analyzed, and optimal allocation strategies remain undefined.

## Related Work & Insights
- Large reasoning models (o1 / DeepSeek-R1 / QwQ) and test-time compute scaling
- Retrieval-augmented generation (RAG) and multi-step reasoning
- Reinforcement learning for LLM alignment and tool use

## Rating
⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Note 5: ReSearch — Learning to Reason with Search](research_learning_to_reason_with_search_for_llms_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] Note 7: Value-Guided Search - Efficient Chain-of-Thought Reasoning](polymath_evaluating_mathematical_reasoning_in_multilingual_contexts.md)
- [\[NeurIPS 2025\] Deep Continuous-Time State-Space Models for Marked Event Sequences](deep_continuous-time_state-space_models_for_marked_event_sequences.md)
- [\[NeurIPS 2025\] Look-Ahead Reasoning on Learning Platforms](look-ahead_reasoning_on_learning_platforms.md)
- [\[NeurIPS 2025\] Deep Legendre Transform](deep_legendre_transform.md)

</div>

<!-- RELATED:END -->
