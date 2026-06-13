---
title: >-
  [Paper Note] OctoTools: An Agentic Framework with Extensible Tools for Complex Reasoning
description: >-
  [ACL 2026][LLM Agent][Agentic framework] OctoTools is a training-free, user-friendly, and highly extensible multi-agent framework. By encapsulating heterogeneous tools via standardized **Tool Cards**…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Agentic framework"
  - "Tool-augmented reasoning"
  - "Multi-step planning"
  - "Tool cards"
  - "Extensibility"
date: 2026-05-08
content_hash: dd8ebf58262ce694
---

# OctoTools: An Agentic Framework with Extensible Tools for Complex Reasoning

**Conference**: ACL 2026  
**arXiv**: [2502.11271](https://arxiv.org/abs/2502.11271)  
**Code**: [https://github.com/octotools/octotools](https://github.com/octotools/octotools)  
**Area**: LLM Reasoning  
**Keywords**: Agentic framework, Tool-augmented reasoning, Multi-step planning, Tool cards, Extensibility

## TL;DR

OctoTools is a training-free, user-friendly, and highly extensible multi-agent framework. By encapsulating heterogeneous tools via standardized **Tool Cards**, utilizing a **Planner-Executor** separation paradigm, and employing a **task-specific toolset optimization** algorithm, it achieves an average accuracy improvement of +9.3% over GPT-4o and up to +10.6% over frameworks like AutoGen and LangChain across 16 diverse benchmarks.

## Background & Motivation

**Background**: LLMs have made rapid progress in tasks like summarization, translation, and code generation, but complex reasoning tasks involving multi-step logic decomposition or specialized domain knowledge remain challenging. Tool-augmented LLMs are a promising direction, enhancing LLM capabilities by offloading specialized sub-tasks to external tools (search engines, calculators, domain models, etc.).

**Limitations of Prior Work**: (1) Many methods require extensive training data and fine-tuning, limiting adaptability to new domains; (2) Some methods are only applicable to specific domains (chemistry, vision, medicine, etc.) and lack generality; (3) Existing general frameworks (AutoGen, LangChain, GPT-Functions) focus more on high-level abstraction or multi-agent collaboration, with insufficient quantitative evaluation on complex reasoning; (4) Merging planning and code execution in a single model leads to overhead and errors.

**Key Challenge**: How to build an agentic framework that is both general (cross-domain) and efficient (multi-step reasoning + tool calling) without requiring additional training.

**Goal**: Propose a training-free, modular, and extensible agentic framework that consistently improves performance across diverse complex reasoning tasks.

**Key Insight**: Standardize tool encapsulation (Tool Cards), decouple strategic planning from command execution (Planner vs. Executor), and automate tool selection (Toolset Optimization Algorithm).

**Core Idea**: Construct a modular multi-step reasoning pipeline through standardized tool card interfaces + a hierarchical Planner-Executor architecture + greedy toolset optimization, where each component focuses on its specific role.

## Method

### Overall Architecture

Given a user query $q$ and a pre-trained model $\text{LLM}_\theta$, OctoTools operates through an iterative process: the Planner analyzes the query and generates a high-level plan $\rightarrow$ in each step, the Planner selects tools and sets sub-goals $\rightarrow$ the Executor converts text actions into executable commands $\rightarrow$ tools are executed to obtain results $\rightarrow$ the context is updated $\rightarrow$ the process repeats until the problem is solved or the step limit is reached. Finally, the Solution Summarizer integrates the trajectory to generate the answer.

### Key Designs

**1. Standardized Tool Cards**

- **Function**: Encapsulates heterogeneous tools into a uniform interface.
- **Mechanism**: Each tool card contains metadata like tool name, input/output types, usage constraints, best practices, and examples, and implements two standard functions: `execute()` and `get_metadata()`. New tools can be integrated by creating a compliant tool card without modifying the framework code.
- **Design Motivation**: To avoid extensive engineering work required for every new tool integration in previous frameworks, achieving true plug-and-play capability.

**2. Planner-Executor Separation Architecture**

- **Function**: Decouples strategic decision-making from code generation, allowing each component to focus on its expertise.
- **Mechanism**: The Planner handles high-level planning (query analysis, skill identification, tool strategy) and low-level action prediction (tool selection, sub-goal setting). The Executor translates text actions into executable Python code and runs it. A Context Verifier checks if the problem is solved, and the Solution Summarizer generates the final answer.
- **Design Motivation**: Assigning both planning and execution to the same model leads to overhead and errors; separation improves reliability and debuggability.

**3. Task-specific Toolset Optimization Algorithm**

- **Function**: Automatically selects the most beneficial tool subset for a specific task.
- **Mechanism**: A three-stage greedy search—Stage 1: Establish baseline toolset performance $\rightarrow$ Stage 2: Evaluate the marginal gain of each candidate tool $\Delta_{d_i} = \text{Acc}(\mathcal{D}_i) - \text{Acc}(\mathcal{D}_{\text{base}})$ $\rightarrow$ Stage 3: Aggregate all tools with positive gain to form the optimal toolset $\mathcal{D}^* = \mathcal{D}_{\text{base}} \cup \{d_i | \Delta_{d_i} > 0\}$.
- **Design Motivation**: Enabling all tools may introduce noise or decrease performance; greedy search reduces complexity from $O(2^n)$ to $O(n)$.

## Key Experimental Results

### Main Results

| Method | Avg Accuracy (16 Tasks) | vs 0-shot | vs CoT |
|---|---|---|---|
| GPT-4o 0-shot | 49.2% | — | — |
| GPT-4o CoT | 50.8% | — | — |
| OctoTools_base (Basic tools only) | 53.4% | +4.2% | +2.6% |
| **OctoTools** (Optimized toolset) | **58.5%** | **+9.3%** | **+7.7%** |

| Method | Avg Accuracy | vs OctoTools |
|---|---|---|
| AutoGen | 47.9% | -10.6% |
| GPT-Functions | 51.0% | -7.5% |
| LangChain | 51.2% | -7.3% |
| **OctoTools** | **58.5%** | — |

### Ablation Study

| Ablation Dimension | Result |
|---|---|
| Basic tools only vs All tools vs Optimized toolset | 53.9% vs 57.4% vs 58.9% |
| Max Steps (1 $\rightarrow$ 10 steps) | Performance generally improves with more steps |
| Weak LLM (GPT-4o-mini) | Still achieves an average gain of +7.1% |

### Key Findings

1. **Significant differences in tool usage**: OctoTools has an external tool usage rate of 67.8%, while AutoGen is only 10.6% and LangChain is 10.7%, indicating that competing frameworks fail to effectively utilize external tools.
2. **Contributions from both decomposition and tools**: Tasks are categorized into three types—those benefiting primarily from multi-step decomposition (e.g., Hallusion-VD), those benefiting from tool calling (e.g., PathCLS +22.2%), and those benefiting from both (e.g., Game of 24).
3. **Largest gains in medical domain**: PathCLS +22.2%, PathVQA +21.4%, demonstrating the high value of specialized domain tools (e.g., CONCH pathology classifier).
4. **Significant improvement in agentic tasks (GAIA-Text)**: +9.7%, requiring collaboration between 5 different tools, showcasing the framework's advantages in complex multi-step tasks.

## Highlights & Insights

1. **Tool Cards are the core design highlight**: Standardized metadata descriptions allow LLMs to autonomously understand and select tools, enabling truly open-ended tool extension.
2. **88-page paper contains extremely detailed analysis**: 16 benchmarks, full configurations for 11 tools, and extensive visualizations and case studies provide high reference value for future research.
3. **Clear Planner-Executor separation**: Prevents LLM overload from simultaneous planning and code generation; each component has dedicated prompt templates.
4. **Greedy toolset optimization is simple yet effective**: While not guaranteeing a global optimum, experiments prove it outperforms full toolset configurations in most tasks.

## Limitations & Future Work

1. **Reliance on GPT-4o**: All experiments are based on GPT-4o; performance on open-source models is unknown (only GPT-4o-mini was tested).
2. **Toolset optimization requires a validation set**: Greedy search requires 100 validation samples, making it unavailable during cold starts.
3. **Single-Agent Architecture**: Multi-agent collaboration scenarios were not explored; complex tasks might benefit from discussion and correction between agents.
4. **Sequential execution**: Only one tool is called per step; lacks support for parallel tool calling, which may affect efficiency.
5. **Tool card design requires manual intervention**: Although simpler than before, it still requires manual metadata and example writing.

## Related Work & Insights

1. **Chameleon (Lu et al., 2023)**: A previous plug-and-play compositional reasoning framework, but with limited tool support and no multi-step iteration.
2. **Visual Sketchpad (Hu et al., 2024)**: A visual reasoning agent limited to the vision domain.
3. **AutoGen/LangChain**: General agent frameworks that underperform OctoTools quantitatively on complex reasoning benchmarks.
4. **ReAct (Yao et al., 2022)**: The classic paradigm of interleaved reasoning and action, upon which OctoTools builds with tool cards and hierarchical planning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Tool card standardization and the Planner-Executor separation paradigm are elegantly designed, and the toolset optimization algorithm is practical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 16 benchmarks, 4 comparative frameworks, and extensive ablation and case analyses; the 88-page paper is highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, rich visualizations, and intuitive case presentations.
- **Value**: ⭐⭐⭐⭐ — Provides a practical open-source agent framework template with high reference value for agent developers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] GOAT: A Training Framework for Goal-Oriented Agent with Tools](goat_a_training_framework_for_goal-oriented_agent_with_tools.md)
- [\[ICLR 2026\] FeatureBench: Benchmarking Agentic Coding for Complex Feature Development](../../ICLR2026/llm_agent/membership_privacy_risks_of_sharpness_aware_minimization.md)
- [\[ACL 2026\] FAMA: Failure-Aware Meta-Agentic Framework for Open-Source LLMs in Interactive Tool Use Environments](fama_failure-aware_meta-agentic_framework_for_open-source_llms_in_interactive_to.md)
- [\[ACL 2026\] Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models](don39t_adapt_small_language_models_for_tools_adapt_tool_schemas_to_the_models.md)
- [\[ACL 2026\] MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools](mcp-flow_facilitating_llm_agents_to_master_real-world_diverse_and_scaling_mcp_to.md)

</div>

<!-- RELATED:END -->
