---
title: >-
  [Paper Note] Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models
description: >-
  [ICLR 2026][LLM Agent][context engineering] This paper proposes ACE (Agentic Context Engineering), a framework that treats context as a continuously evolving playbook. Through a Generator–Reflector–Curator role decomposition and incremental delta updates, ACE accumulates and refines strategies over time, addressing brevity bias and context collapse in existing prompt optimization methods. ACE achieves an average improvement of 10.6% on agent benchmarks and 8.6% on financial tasks, while reducing adaptation latency by 86.9%.
tags:
  - ICLR 2026
  - LLM Agent
  - context engineering
  - self-improving agent
  - prompt optimization
  - evolving memory
  - playbook
date: 2026-05-08
content_hash: 47c2afc19cb7fc43
---

# Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models

**Conference**: ICLR 2026
**arXiv**: [2510.04618](https://arxiv.org/abs/2510.04618)
**Code**: [https://github.com/ace-agent/ace](https://github.com/ace-agent/ace)
**Area**: Agent
**Keywords**: context engineering, self-improving agent, prompt optimization, evolving memory, playbook

## TL;DR
This paper proposes ACE (Agentic Context Engineering), a framework that treats context as a continuously evolving playbook. Through a Generator–Reflector–Curator role decomposition and incremental delta updates, ACE accumulates and refines strategies over time, addressing brevity bias and context collapse in existing prompt optimization methods. ACE achieves an average improvement of 10.6% on agent benchmarks and 8.6% on financial tasks, while reducing adaptation latency by 86.9%.

## Background & Motivation
**Background**: Context adaptation—improving model performance by modifying LLM inputs rather than weights—has become a core paradigm for building scalable AI systems. Existing approaches include prompt optimization methods (GEPA, MIPROv2) and test-time memory methods (Dynamic Cheatsheet).

**Limitations of Prior Work**: (1) **Brevity bias**: Most prompt optimizers favor concise, general instructions, discarding domain-specific strategies, tool-use guidelines, and common failure patterns. (2) **Context collapse**: Monolithic rewriting approaches progressively degrade into shorter, information-poorer summaries during iterative updates—experiments observed context shrinking abruptly from 18,282 tokens to 122 tokens, with a corresponding sharp drop in performance.

**Key Challenge**: Agent and knowledge-intensive applications require **comprehensive and detailed** domain knowledge, yet existing methods compress it. Unlike humans, who benefit from concise summaries, LLMs perform better under detailed context.

**Goal**: How to design a context adaptation approach that continuously accumulates knowledge without collapsing or degrading?

**Key Insight**: Treat context as an "evolving playbook" rather than an "optimized prompt," replacing monolithic rewrites with structured incremental updates.

**Core Idea**: Context should be a continuously growing and refined strategy playbook, not a compressed set of concise instructions.

## Method

### Overall Architecture
ACE consists of three roles: Generator (produces reasoning trajectories) → Reflector (extracts lessons and insights from trajectories) → Curator (integrates lessons into structured delta updates and merges them into the existing context). ACE supports both offline (system prompt optimization) and online (test-time memory) modes.

### Key Designs

1. **Three-Role Decomposition (Generator → Reflector → Curator)**:

    - **Function**: Decouple distinct responsibilities of context construction into specialized roles.
    - **Mechanism**: The Generator solves new problems using the current context, producing execution trajectories. The Reflector analyzes trajectories to extract concrete successful strategies and failure lessons (with optional multi-round iterative refinement). The Curator converts insights into structured bullets and merges them into the context.
    - **Design Motivation**: Avoid bottlenecks caused by assigning all responsibilities to a single model. Ablation studies confirm that the dedicated Reflector role is the primary source of performance gains.

2. **Incremental Delta Updates (replacing monolithic rewrites)**:

    - **Function**: Replace full context rewrites with localized bullet-level additions, deletions, and modifications.
    - **Mechanism**: The context is represented as a collection of bullets, each with a unique ID, helpfulness/harmfulness counters, and content. Each adaptation generates a small delta (new bullets or modifications to existing ones), which is deterministically merged via lightweight non-LLM logic and supports parallel processing.
    - **Design Motivation**: Fundamentally eliminates context collapse—since full rewrites are never performed, knowledge can only be added or locally modified, never accidentally compressed away.

3. **Grow-and-Refine Mechanism**:

    - **Function**: Balance continuous context growth with redundancy control.
    - **Mechanism**: New bullets are appended to the context; existing bullets are updated in place (e.g., counter increments). Semantic embedding comparison is used for de-duplication, which can be applied eagerly after each delta or lazily when the context window is exceeded.
    - **Design Motivation**: Keep context size manageable and prevent unbounded growth.

### Loss & Training
No model weight training is required. ACE is a purely context-adaptation method. In offline mode, context is iteratively built over multiple epochs on the training set; in online mode, the context is updated sample-by-sample at test time. Key hyperparameters: maximum Reflector refinement rounds = 5, maximum offline epochs = 5, batch size = 1. Notably, ACE can operate without annotations by leveraging execution feedback (e.g., code execution success/failure) as a natural learning signal.

## Key Experimental Results

### Main Results (AppWorld Agent Benchmark)

| Method | Requires Labels | Test-Normal TGC | Test-Challenge TGC | Average |
|--------|----------------|----------------|-------------------|---------|
| ReAct baseline | - | 63.7 | 41.5 | 42.4 |
| + ICL | ✓ | 64.3 | 46.0 | 46.0 |
| + GEPA | ✓ | 64.9 | 46.0 | 46.4 |
| **+ ACE (w/ labels)** | ✓ | **76.2** | **57.3** | **59.4** |
| + ACE (w/o labels) | ✗ | 75.0 | 54.4 | 57.2 |
| + DC (online) | ✗ | 65.5 | 52.3 | 51.9 |
| **+ ACE (online)** | ✗ | **69.6** | **66.0** | **59.5** |

### Ablation Study (Financial Benchmark)

| Method | FiNER Acc | Formula Acc | Average |
|--------|-----------|-------------|---------|
| Base LLM | 70.7 | 67.5 | 69.1 |
| GEPA | 73.5 | 71.5 | 72.5 |
| **ACE** | **78.3** | **85.5** | **81.9** |

### Key Findings
- ACE achieves an average improvement of 17% on AppWorld (offline with labels). Using the open-source model DeepSeek-V3.1, ACE matches the average performance of IBM CUGA (the top-ranked system on the leaderboard, powered by GPT-4.1) and surpasses it on the harder test-challenge split.
- **Strong without annotations**: ACE improves performance by 14.8% in the label-free setting, demonstrating effective self-improvement using only execution feedback.
- On financial tasks, ACE outperforms GEPA by 9.4% (72.5→81.9), indicating that aggressively accumulating domain knowledge yields clear advantages on knowledge-intensive tasks.
- Adaptation latency is reduced by 86.9%: incremental delta updates are substantially faster than monolithic rewrites.
- Ablation studies confirm that both the Reflector role and multi-epoch refinement each contribute significant and independent performance gains.

## Highlights & Insights
- **Paradigm shift from "prompt" to "playbook"**: Context should be continuously enriched rather than compressed. This aligns with trends in RAG and long-context modeling, offering a clear design philosophy for context engineering.
- **Incremental delta updates as a key innovation**: This design completely eliminates context collapse and supports parallel merging—a simple yet highly effective engineering solution.
- **Unsupervised self-improvement capability**: Effective context can be built from execution feedback alone, paving the way for truly self-improving agents.
- **Reusable three-role pattern**: The Generator–Reflector–Curator paradigm is transferable to other LLM systems that need to learn from experience.

## Limitations & Future Work
- As the number of bullets grows, the context may exceed the context window, necessitating more intelligent retrieval or compression strategies.
- De-duplication relies on the quality of semantic embeddings; similar but non-identical bullets may accumulate over time.
- The current design requires the Generator, Reflector, and Curator to use the same model, limiting the flexibility to exploit models of different sizes for cost optimization.
- The effect of sequential dependency in online mode—where earlier samples influence the context seen by later ones—and whether this introduces systematic bias has not been thoroughly analyzed.

## Related Work & Insights
- **vs. Dynamic Cheatsheet**: ACE builds upon Dynamic Cheatsheet but addresses its context collapse problem by introducing the Reflector role and delta update mechanism.
- **vs. GEPA**: GEPA is a prompt optimizer (targeting concise prompts), whereas ACE is a context engineering framework (targeting comprehensive playbooks). The two embody fundamentally different philosophies, and ACE significantly outperforms GEPA on both agent and financial tasks.
- **vs. TextGrad**: TextGrad optimizes prompts via gradient-style textual feedback, while ACE accumulates strategies as structured bullets, avoiding the information loss inherent in rewriting.

## Supplementary Discussion

### Why Is Context Engineering More Important Than Prompt Engineering?
Prompt Engineering is static—once a system prompt is written, it remains fixed. Context Engineering is dynamic—the context evolves continuously based on the agent's runtime experience, better suiting real-world agents operating in complex environments. The playbook delta update mechanism is a concrete realization of this philosophy.

## Rating
- Novelty: ⭐⭐⭐⭐ The "evolving playbook" concept and delta update design represent genuine innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two benchmark categories, multiple baselines, comprehensive ablations, and leaderboard comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, persuasive concepts, and smooth narrative.
- Value: ⭐⭐⭐⭐⭐ An important contribution to the context engineering direction with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] InfiAgent: Self-Evolving Pyramid Agent Framework for Infinite Scenarios](infiagent_self-evolving_pyramid_agent_framework_for_infinite_scenarios.md)
- [\[ICLR 2026\] Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents](your_agent_may_misevolve_emergent_risks_in_self-evolving_llm_agents.md)
- [\[ICLR 2026\] CoMind: Towards Community-Driven Agents for Machine Learning Engineering](comind_towards_community-driven_agents_for_machine_learning_engineering.md)
- [\[ICLR 2026\] ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models](toolweaver_weaving_collaborative_semantics_for_scalable_tool_use_in_large_langua.md)
- [\[ACL 2026\] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](../../ACL2026/llm_agent/agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)

</div>

<!-- RELATED:END -->
