---
title: >-
  [Paper Note] A Self-Improving Coding Agent
description: >-
  [NeurIPS 2025][Code Intelligence][self-improving agent] This paper proposes SICA (Self-Improving Coding Agent), a coding agent capable of autonomously editing its own codebase to improve performance. By eliminating the d…
tags:
  - "NeurIPS 2025"
  - "Code Intelligence"
  - "self-improving agent"
  - "meta-agent"
  - "coding agent"
  - "automated agent design"
  - "self-referential"
date: 2026-05-08
content_hash: 759cad923a03788a
---

# A Self-Improving Coding Agent

**Conference**: NeurIPS 2025
**arXiv**: [2504.15228](https://arxiv.org/abs/2504.15228)  
**Code**: [https://github.com/MaximeRobeyns/self_improving_coding_agent](https://github.com/MaximeRobeyns/self_improving_coding_agent)  
**Area**: Code Intelligence
**Keywords**: self-improving agent, meta-agent, coding agent, automated agent design, self-referential

## TL;DR
This paper proposes SICA (Self-Improving Coding Agent), a coding agent capable of autonomously editing its own codebase to improve performance. By eliminating the distinction between meta-agent and target-agent, SICA achieves iterative self-improvement, advancing from 17% to 53% on a subset of SWE-Bench Verified.

## Background & Motivation
The design of LLM agent systems—covering prompting strategies, tool calling, multi-agent collaboration, and related techniques—remains heavily reliant on manual effort. Researchers must hand-craft various prompting schemes (CoT, ToT, GoT, Self-Refine, etc.), and this trial-and-error process can only explore a small fraction of the design space. ADAS (Automated Design of Agentic Systems) attempts to use a meta-agent to optimize a target-agent, but these two agents are separate—the meta-agent itself never improves. A natural follow-up question is: can an agent improve itself? Coding agents are particularly suited for this, since agents are themselves implemented as code, and a capable coding agent inherently possesses the ability to modify its own source.

## Core Problem
How to build a truly **self-referential self-improving system** that enables an agent to continuously improve its performance on coding benchmarks by modifying its own codebase—without human intervention—rather than merely adjusting prompts or individual functions. The core challenges include: (1) ensuring that the improvement mechanism itself benefits from prior improvements (compound effect); (2) guaranteeing safety and observability; and (3) balancing performance, cost, and latency.

## Method

### Overall Architecture
SICA's core loop resembles ADAS but differs in a fundamental way:
1. An **agent archive** is maintained, recording the agent code and benchmark results from each iteration.
2. At each iteration, the best-performing agent in the archive is selected as the **meta-agent** (notably, this meta-agent is itself a prior target-agent—there is no separation).
3. The meta-agent consults the archive, analyzes historical performance, proposes improvements, and implements code modifications.
4. The new agent is evaluated on the benchmark set, and results are stored in the archive.
5. The loop repeats.

The key distinction from ADAS: ADAS keeps the meta-agent fixed and only modifies the target-agent's single forward function; SICA is fully self-referential—the meta-agent is drawn from the best agent in the archive and may modify the entire Python codebase.

### Key Designs

1. **Utility Function Design**: Performance is measured not only by benchmark accuracy but also by cost and latency: $U = w_{score} \cdot p_{score} + w_{cost} \cdot (1 - \min(1, p_{cost}/\$10)) + w_{time} \cdot (1 - \min(1, p_{time}/300s))$, with weights 0.5, 0.25, and 0.25 respectively. A timeout penalty ($\tau=0.5$) is additionally applied. This drives the agent to improve not only accuracy but also cost-efficiency and speed.

2. **Asynchronous Overseer Mechanism**: An independent LLM runs in a parallel thread every 30 seconds, monitoring whether the agent has entered a loop, deviated from the task, or exhibited pathological behavior. The overseer can send notifications or forcibly cancel agent execution—a critical safeguard for safe autonomous operation over long runs (approximately one hour per improvement iteration).

3. **Sub-agent Architecture and Context Management**: The initial agent comprises three sub-agents: a coding agent, a problem-solver agent, and a reasoning agent. Sub-agent invocations resemble tool calls and return string values. This design enables task decomposition (benchmark tasks run fully autonomously) while managing context length—once a sub-agent returns, its context is naturally discarded, retaining only a summary of the return value.

4. **KV Cache-Friendly Context Layout**: The LLM context is ordered as: system prompt (agent definition + tool documentation) → core prompt (problem + open files + directory tree) → assistant messages (CoT reasoning + tool calls). Content is append-only rather than rewritten, preserving the KV cache to reduce latency and cost. File edits are appended as diffs and periodically consolidated.

5. **XML-Format Tool Calling**: Unconstrained XML-based structured generation is used instead of JSON, because XML does not require escaping string literals—avoiding extensive escaping issues when editing code.

6. **Synthetic Benchmarks**: In addition to SWE-Bench and LiveCodeBench, two automatically constructed synthetic benchmarks are used: a file-editing benchmark (constructed from git commit pairs) and a symbol-localization benchmark (locating symbol definitions in a codebase). This demonstrates the system's capacity to generate new evaluations for itself.

## Key Experimental Results

| Dataset | Metric | Initial (Iter 0) | Final (Iter 15) | Gain |
|--------|------|------|----------|------|
| SWE-Bench Verified (50-problem subset) | Accuracy | 17% | 53% | +36pp |
| LiveCodeBench (50 problems) | Accuracy | Marginal improvement | — | Slight |
| File Editing (synthetic) | Accuracy | — | — | Significant ↑ |
| Symbol Localization (synthetic) | Accuracy | — | — | Significant ↑ (after Iter 9 AST tool) |
| AIME 2024 + GPQA Diamond | Average Accuracy | ~76% | ~80% | Marginal |

- Primary model: Sonnet 3.5 v2; reasoning sub-agent: o3-mini
- Total API cost for 15 iterations: approximately $7,000
- Key agent-discovered improvements (in iteration order): Smart Edit tool → Code Context Summarizer → File Edit Verification → AST Symbol Locator → Hybrid Symbol Locator
- Improvements on reasoning tasks (AIME/GPQA) are limited—o3-mini alone is already strong (AIME 87%, GPQA 79%), and agent scaffolding may interfere with the reasoning model's chain of thought.

### Ablation Study
- The **Smart Edit tool** (diff- and range-based editing replacing full-file overwrite) yields the largest early gain, directly reducing token consumption and editing time.
- The **AST Symbol Locator** (Iter 9) produces significant cross-benchmark improvements by exploiting the Python-dominant nature of the benchmarks.
- The **Hybrid Symbol Locator** (ripgrep candidates + AST verification) further improves symbol localization precision.
- On reasoning tasks, iterations introducing coarse reasoning components (e.g., iterations 4 and 6) actually **degrade** reasoning model performance—scaffolding interferes with the built-in chain of thought.
- Cache hit rate gradually improves from 32.5% to approximately 40%, indicating that the agent learns to optimize context reuse.

## Highlights & Insights
- **True self-reference**: Eliminating the meta/target separation means that the improved agent itself becomes the next improver, allowing improvements to compound.
- The agent modifies the **entire codebase** rather than a single function—enabling it to create new tools, new sub-agents, revise prompts, and even alter context management logic.
- Agent-discovered improvements are highly practical (Smart Edit, AST-based localization, etc.) and align with the intuitions of human engineers.
- The utility function balancing accuracy, cost, and time is pragmatic, avoiding single-minded optimization of accuracy at the expense of efficiency.
- The asynchronous Overseer is an elegant solution for ensuring safe long-horizon autonomous operation.
- The work demonstrates both the potential and the ceiling of scaffolding-only improvements (without updating model weights).

## Limitations & Future Work
- **Strong path dependence**: Poor improvement ideas in early iterations negatively affect the quality of ideas in all subsequent iterations, leading to high variance.
- **High cost**: 15 iterations cost approximately $7,000, and each meta-improvement round takes about one hour.
- **Limited gains on reasoning tasks**: For already strong reasoning models (o3-mini), the marginal benefit of agent scaffolding is small or even negative.
- **5-minute timeout constraint**: This results in artificially low initial benchmark scores (not reflective of true model capability), and many early "improvements" focus on acceleration rather than quality.
- **No weight updates**: Pure scaffolding improvements have a ceiling; the paper itself acknowledges that jointly updating model weights alongside scaffolding is an important future direction.
- **Fixed benchmarks**: Static benchmark sets may saturate quickly, motivating the need for self-curatable benchmark mechanisms.
- **Novelty generation challenge**: The agent struggles to produce genuinely novel and feasible improvement ideas—a core open challenge in open-ended learning.

## Related Work & Insights
- **vs. ADAS**: ADAS employs a fixed meta-agent and a restricted DSL (modifying only the forward function); SICA is fully self-referential and operates on the complete Python codebase. However, SICA requires stronger base models (Sonnet 3.5 + o3-mini) and incurs higher costs.
- **vs. Gödel Agent**: Gödel Agent provides specific tools to modify small portions of agent logic; it is not a general-purpose coding agent and has not been evaluated on coding benchmarks.
- **vs. STOP (Zelikman et al.)**: STOP is a recursively self-improving code generator but is limited to algorithmic tasks (parity, 3-SAT, etc.) and is not a general-purpose agent for arbitrary software engineering tasks.
- **vs. AlphaEvolve**: AlphaEvolve relies more heavily on structured evolutionary search and focuses on scientific discovery and computational infrastructure optimization; SICA takes a more free-form approach and focuses on improving the agent system itself.

The self-referential self-improvement paradigm has broad implications for agent system design: if agents can autonomously create new tools and optimize prompts, the role of human engineers shifts toward defining utility functions and safety constraints. The finding that scaffolding improvements have a ceiling is significant—suggesting that future breakthroughs will require combining code-level improvements with model weight updates. The utility function design (accuracy + cost + time) is broadly applicable to other agent optimization settings. The asynchronous Overseer mechanism can be directly reused in any agent system requiring long-horizon autonomous operation.

## Rating
- Novelty: ⭐⭐⭐⭐ Genuinely self-referential coding agent self-improvement, first validated on coding benchmarks, though the core loop is similar to ADAS.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple benchmarks, 15 iterations, reasoning task comparisons, and detailed improvement logs; however, SWE-Bench evaluation uses only a 50-problem subset.
- Writing Quality: ⭐⭐⭐⭐⭐ Exceptionally clear writing with well-structured motivation, method, experiments, and safety discussion; the agent trace in the appendix is highly valuable as a reference.
- Value: ⭐⭐⭐⭐ Validates the feasibility of self-improving agents with open-source code, though practical reproducibility is limited by high cost ($7k/run).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](../../ICLR2026/code_intelligence/reasoningbank_scaling_agent_self-evolving_with_reasoning_memory.md)
- [\[NeurIPS 2025\] VeriMaAS: Automated Multi-Agent Workflows for RTL Design](automated_multi-agent_workflows_for_rtl_design.md)
- [\[ICLR 2026\] KV Cache Transform Coding for Compact Storage in LLM Inference](../../ICLR2026/code_intelligence/kv_cache_transform_coding_for_compact_storage_in_llm_inference.md)
- [\[ICLR 2026\] Improving Code Localization with Repository Memory](../../ICLR2026/code_intelligence/improving_code_localization_with_repository_memory.md)
- [\[AAAI 2026\] MoSE: Hierarchical Self-Distillation Enhances Early Layer Embeddings](../../AAAI2026/code_intelligence/mose_hierarchical_self-distillation_enhances_early_layer_embeddings.md)

</div>

<!-- RELATED:END -->
