---
title: >-
  [Paper Note] T1: A Tool-Oriented Conversational Dataset for Multi-Turn Agentic Planning
description: >-
  [NeurIPS 2025][LLM Agent][tool-use] This paper introduces T1, a dataset of 13.5K multi-turn dialogues spanning 9 domains (4 single-domain + 5 cross-domain) and 14 tools…
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "tool-use"
  - "multi-turn dialogue"
  - "agentic planning"
  - "benchmark"
  - "inter-tool dependency"
date: 2026-05-08
content_hash: 0951b19611bdc81f
---

# T1: A Tool-Oriented Conversational Dataset for Multi-Turn Agentic Planning

**Conference**: NeurIPS 2025
**arXiv**: [2505.16986](https://arxiv.org/abs/2505.16986)  
**Code**: [https://github.com/CapitalOne-Research/T1](https://github.com/CapitalOne-Research/T1)  
**Area**: LLM Agent
**Keywords**: tool-use, multi-turn dialogue, agentic planning, benchmark, inter-tool dependency

## TL;DR
This paper introduces T1, a dataset of 13.5K multi-turn dialogues spanning 9 domains (4 single-domain + 5 cross-domain) and 14 tools, with a focus on inter-tool dependencies and dynamic replanning. A baseline system, T1-Agent (code generation + caching mechanism), is proposed for systematic evaluation. Experiments show that SFT-tuned Llama 8B achieves 87.17% Tool Call F1, surpassing untuned 70B models, yet still trailing closed-source models such as GPT-5 and o3.

## Background & Motivation
**Background**: Tool-call evaluation for LLM agents has attracted increasing attention, with benchmarks such as APIBank, ToolBench, TravelPlanner, GAIA, and GTA already proposed. However, these benchmarks primarily focus on atomic tool calls in single-turn interactions, treating tool use as isolated operations.

**Limitations of Prior Work**: (a) Inter-tool output dependencies—where the result of tool A serves as input to tool B (e.g., using flight arrival time to derive hotel check-in date)—are insufficiently covered in existing benchmarks; (b) In realistic scenarios, user requests evolve progressively across dialogue turns (e.g., searching flights, then adding hotels, then filtering by price), requiring agents to dynamically adjust plans; (c) No existing benchmark evaluates the efficiency of intermediate result caching and reuse, nor whether an agent redundantly invokes the same API.

**Key Challenge**: Existing benchmarks can assess whether a model correctly invokes a single tool, but cannot evaluate whether it can coordinate multiple tools across multi-turn dialogues to complete complex cross-domain tasks.

**Goal**: Construct a large-scale conversational evaluation benchmark supporting multi-tool dependencies, multi-turn interaction, cross-domain planning, and dynamic replanning.

**Key Insight**: A travel assistant scenario is adopted, covering four domains—flights, hotels, restaurants, and attractions—along with five cross-domain combinations, enabling systematic construction of tool dependency chains.

**Core Idea**: A knowledge base, templates, and human annotation are combined to construct a multi-turn dialogue dataset with inter-tool dependency graphs. A code generation and caching agent baseline is introduced to systematically evaluate LLMs' complex tool coordination capabilities.

## Method

### Overall Architecture
The T1 construction pipeline proceeds as follows: (1) Define ontologies for four domains—comprising 106 attributes in total, such as airline, cabin class, and number of stops for flights, and star rating, review score, and amenities for hotels; (2) Generate 60 dialogue templates × 9 domain combinations using Llama 3.3 70B, with human quality annotation; (3) Populate template placeholders using knowledge bases (e.g., 480K flights, 47K hotels) to generate 13.5K complete dialogues; (4) Each dialogue includes ground-truth code and the resulting cache state after execution. At inference time, T1-Agent receives the dialogue context and a cache summary, then generates executable Python code to invoke tools.

### Key Designs

1. **Inter-Tool Dependency Modeling**:

    - Function: 14 tools are categorized into domain-specific tools (search_flights/hotels/restaurants/attractions), cross-domain dependency tools (e.g., adjust_date, which derives hotel check-in date from flight arrival time), and general-purpose tools (filter/sort/cache operations).
    - Mechanism: Cross-domain dependencies are introduced naturally within dialogues—for instance, a user first searches for flights, whose arrival time determines the hotel check-in date, and the hotel location then constrains the restaurant search area. The agent must understand these implicit dependencies and correctly sequence tool invocations.
    - Design Motivation: Although benchmarks such as TravelPlanner involve travel planning, they do not model chained dependencies in which the output of one tool serves as the input to another.

2. **Multi-Turn Dynamic Replanning**:

    - Function: As user requirements evolve across dialogue turns, the agent must revise its plan based on previously obtained results.
    - Mechanism: For example, if the first turn searches for NYC→SFO flights and a later turn requests filtering by a specific airline, the agent should reuse cached results from the first turn rather than issuing a new search.
    - Design Motivation: This tests whether an agent can efficiently exploit existing information rather than restarting from scratch at each turn.

3. **T1-Agent Caching Mechanism**:

    - Function: After each tool execution, the agent stores results in a cache (save_to_cache). In subsequent turns, the agent consults a rule-based cache summary to decide whether to reuse results (get_results_from_cache).
    - Mechanism: Rather than placing the full cache in the prompt (which would be excessively long), only a concise cache summary is included, substantially reducing token overhead. When generating code, the agent may retrieve data from the cache and refine it using filter tools.
    - Design Motivation: Cache management is elevated from the tool-execution layer to the agent's planning layer, allowing the agent to autonomously decide when to reuse cached results and when to issue a fresh query.

4. **Data Quality Assurance**:

    - Five human annotators (with M.S. degrees in CS and Python proficiency) reviewed each instance through dual-stage annotation and QA checks.
    - Cities are partitioned by split (train/val/test) to prevent data leakage.
    - All ground-truth code is execution-verified to ensure the absence of syntactic or logical errors.

### Loss & Training
- T1-Agent SFT: Llama 3.1 8B Instruct is fine-tuned with LoRA on 8×A100 GPUs for 1 epoch using standard next-token prediction with cross-entropy loss on (prompt, completion) pairs.
- At inference, few-shot in-context learning ($k=13$) is used, with code executed in a sandboxed environment.

## Key Experimental Results

### Main Results

| Model | Tool Call F1 | Param Match F1 | Code Exec Acc | Cache EM |
|-------|-------------|---------------|--------------|---------|
| Llama 3.1 8B | 52.11 | 31.79 | 41.83 | 29.56 |
| Llama 3.1 8B **SFT** | **87.17** | **75.76** | 84.29 | **63.95** |
| Llama 3.3 70B | 79.72 | 67.74 | **91.43** | 57.83 |
| Phi-4-reasoning-plus | 68.58 | 51.31 | 86.37 | 40.59 |
| s1.1 32B | 83.38 | 70.95 | 60.76 | 54.29 |

Closed-source models (small dataset):

| Model | Tool Call F1 | Param Match F1 | Code Exec Acc | Cache EM |
|-------|-------------|---------------|--------------|---------|
| Gemini 2.5 Pro | 94.28 | 84.63 | 94.46 | 76.11 |
| GPT 5 | 93.14 | 86.51 | 94.45 | 76.25 |
| OpenAI o3 | 91.91 | 85.64 | 93.51 | 73.53 |
| GPT 4.1 | 92.32 | 85.53 | 91.20 | 73.80 |

### Ablation Study

| Configuration | Tool Call F1 (flights) | Note |
|---------------|----------------------|------|
| 0-shot | Very low | No context; model cannot understand tool format |
| 5-shot | Significant improvement | A small number of examples yields large gains |
| 13-shot | Marginal improvement | Diminishing returns beyond 5-shot |

**Per-domain analysis** (Llama 3.1 8B):

| Domain | Tool Call F1 | Code Exec Acc |
|--------|-------------|--------------|
| Restaurants (single-domain) | 57.55 | 56.99 |
| Hotels (single-domain) | 60.00 | 48.13 |
| F-H-R (3-domain) | 45.00 | 36.79 |
| F-H-A (3-domain) | 43.45 | 28.76 |

### Key Findings
- **SFT yields dramatic gains for small models**: The 8B SFT model achieves 87.17% Tool Call F1, greatly outperforming the untuned 8B (52.11%) and even surpassing the 70B model (79.72%), demonstrating that task-specific fine-tuning can compensate for differences in model scale.
- **Substantial gap between closed-source and open-source models**: GPT-5 and Gemini 2.5 Pro reach 93–94% Tool Call F1, while the best open-source 8B SFT model achieves 87%, indicating that complex tool planning still benefits from stronger base models.
- **Multi-domain settings are substantially harder than single-domain**: The 3-domain scenario (F-H-A) yields a Code Exec Acc of only 28.76%, far below single-domain performance (~50%), identifying cross-domain tool coordination as the core bottleneck.
- **Large variance in cache utilization**: The SFT model achieves a Cache EM of 63.95%, far exceeding the untuned model's 29.56%, suggesting that efficient cache reuse requires dedicated training.
- **Code generation paradigm is critical**: Expressing tool calls as Python code rather than JSON naturally supports variable passing and conditional logic, making it more suitable for representing complex inter-tool dependencies.

## Highlights & Insights
- **Systematic evaluation of inter-tool dependencies**: T1 is the first benchmark to treat input-output dependency chains between tools as a core evaluation dimension, filling a gap in agent benchmarking. This underscores that agent evaluation should assess not only whether a model can call tools, but whether it can coordinate them.
- **Design philosophy of the caching mechanism**: Cache management is elevated from the tool layer to the agent's planning layer—the agent autonomously decides, upon viewing the cache summary, whether to reuse results or issue a fresh query. This is more flexible than conventional automatic caching strategies (e.g., LRU) and more closely mirrors the reasoning process of a human assistant.
- **Insights from small-model SFT**: The finding that 8B SFT outperforms 70B zero-shot suggests that for structured tool-call tasks, domain adaptation is more important than model scale.

## Limitations & Future Work
- Dialogues are generated by populating templates with knowledge base entries; although human-reviewed, naturalness and diversity may be lower than in real conversations.
- Coverage is limited to travel scenarios (flights, hotels, restaurants, attractions); while 9 domain combinations are included, all revolve around travel, and generalization to other domains (e.g., programming, data analysis, scientific research) remains unknown.
- Only English is considered; multilingual settings are not addressed.
- Cache summaries are generated using rule-based methods; LLM-generated summaries could provide greater flexibility.
- The benchmark does not evaluate error recovery—specifically, how an agent replans when a tool call fails.

## Related Work & Insights
- **vs. TravelPlanner**: Both adopt a travel scenario, but TravelPlanner involves single-turn planning without multi-turn interaction or inter-tool dependencies; T1's multi-turn and caching design is more representative of real-world usage.
- **vs. APIBank/ToolBench**: These benchmarks cover large numbers of APIs but predominantly involve single-turn, single-tool calls and do not test tool coordination.
- **vs. GTA**: GTA includes multi-tool chains but lacks the caching and dynamic replanning dimensions.
- **vs. Tau-Bench**: Tau-Bench focuses on constraint tracking within dialogues, whereas T1 emphasizes inter-tool dependencies.

## Rating
- Novelty: ⭐⭐⭐⭐ — The inter-tool dependency and cache reuse evaluation dimensions are novel, though the scenario is limited to travel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 13.5K dialogues, comparisons across open- and closed-source models, and few-shot/SFT analyses provide comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ — The data construction process is transparent and tables are informative, though some details are deferred to the appendix.
- Value: ⭐⭐⭐⭐ — The work meaningfully advances the evaluation of agent tool coordination, and the cache design is reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Enhancing Demand-Oriented Regionalization with Agentic AI and Local Heterogeneous Data for Adaptation Planning](enhancing_demand-oriented_regionalization_with_agentic_ai_and_local_heterogeneou.md)
- [\[NeurIPS 2025\] AgentChangeBench: A Multi-Dimensional Evaluation Framework for Goal-Shift Robustness](agentchangebench_a_multi-dimensional_evaluation_framework_for_goal-shift_robustn.md)
- [\[NeurIPS 2025\] Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding](deep_video_discovery_agentic_search_with_tool_use_for_longfo.md)
- [\[AAAI 2026\] Towards Trustworthy Multi-Turn LLM Agents via Behavioral Guidance](../../AAAI2026/llm_agent/towards_trustworthy_multi-turn_llm_agents_via_behavioral_guidance.md)
- [\[NeurIPS 2025\] Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning](zero-shot_large_language_model_agents_for_fully_automated_radiotherapy_treatment.md)

</div>

<!-- RELATED:END -->
