---
title: >-
  [Paper Note] Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies
description: >-
  [ICLR 2026][Signal & Communication][Multi-agent systems] This paper proposes Multi-Agent System Search (MASS), a framework that automatically discovers high-performance multi-agent system (MAS) designs through a three-stage interleaved strategy of prompt and topology optimization: local prompt optimization → topology search → global prompt optimization.
tags:
  - ICLR 2026
  - "Signal & Communication"
  - Multi-agent systems
  - prompt optimization
  - topology search
  - LLM agent
  - automated design
date: 2026-05-08
content_hash: 5899ca5f80264b99
---

# Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies

**Conference**: ICLR 2026
**arXiv**: [2502.02533](https://arxiv.org/abs/2502.02533)
**Code**: None
**Area**: Signal Communication
**Keywords**: Multi-agent systems, prompt optimization, topology search, LLM agent, automated design

## TL;DR

This paper proposes Multi-Agent System Search (MASS), a framework that automatically discovers high-performance multi-agent system (MAS) designs through a three-stage interleaved strategy of prompt and topology optimization: local prompt optimization → topology search → global prompt optimization.

## Background & Motivation

1. **Background**: LLM-based multi-agent systems (MAS), leveraging interaction and collaboration among multiple agents, outperform single-agent systems on complex tasks such as code generation, reasoning, and question answering.

2. **Limitations of Prior Work**: Designing effective MAS requires simultaneously considering per-agent prompt design and inter-agent topology orchestration, whose combination yields an enormous search space. Existing automated methods (e.g., ADAS, AFlow) either optimize topology while ignoring prompts, or employ poorly scoped search spaces.

3. **Key Challenge**: Prompts and topology are two critical factors in MAS design, yet their interaction is complex—prompt sensitivity amplifies across cascaded agents, and not all topologies positively affect performance. The combinatorial complexity of joint optimization is prohibitively high.

4. **Goal**: To systematically analyze the impact of various factors in the MAS design space and propose an efficient automated optimization framework.

5. **Key Insight**: Empirical analysis first reveals that prompt optimization is more token-efficient than simply scaling the number of agents, and that beneficial topologies constitute only a small fraction of the search space. Based on these findings, the search space is pruned and prompt/topology optimization is interleaved.

6. **Core Idea**: An interleaved optimization strategy proceeding from local to global and from prompts to topology can efficiently overcome the combinatorial complexity of MAS design.

## Method

### Overall Architecture

MASS is a three-stage optimization framework. **Stage 1** (1PO) performs local prompt optimization independently for each topology building block. **Stage 2** (2TO) conducts topology optimization within the pruned search space using the prompts optimized in Stage 1. **Stage 3** (3PO) performs global joint prompt optimization over the best topology. The search space encompasses five categories of building blocks: Aggregate, Reflect, Debate, Summarize, and Tool-use.

### Key Designs

**1. Influence-Based Search Space Pruning**

- **Function**: Restricts the topology search space to a subset with positive influence, reducing search complexity.
- **Mechanism**: Computes the incremental influence of each building block as $I_{a_i} = \mathcal{E}(a_i^*) / \mathcal{E}(a_0^*)$, converts it into a selection probability via Softmax as $p_a = \text{Softmax}(I_a, t)$, and prunes the search space via rejection sampling: a dimension is rejected if $u > p_{a_i}$ where $u \sim \text{Uniform}(0,1)$.
- **Design Motivation**: Experiments demonstrate that not all topologies exert a positive influence (e.g., only debate yields gains on HotpotQA); searching the full space introduces detrimental building blocks that degrade performance.

**2. Interleaved Three-Stage Optimization Strategy**

- **Function**: Decouples the combinatorial complexity of joint MAS optimization while ensuring co-optimization of prompts and topology.
- **Mechanism**: Stage 1 first warms up the single-agent prompt $a_0^* \leftarrow \mathcal{O}_\mathcal{D}(a_0)$, then optimizes each building block's prompt under a minimal configuration $a_i^* \leftarrow \mathcal{O}_\mathcal{D}(a_i | a_0^*)$. Stage 2 randomly samples topology configurations from the pruned space and evaluates them. Stage 3 performs global joint prompt optimization over the best topology.
- **Design Motivation**: Directly applying automatic prompt optimization (APO) to MAS is infeasible due to inter-agent dependencies and sparse rewards. The local-then-global strategy decomposes the complexity into manageable subproblems.

**3. Plug-and-Play Prompt Optimizer Integration**

- **Function**: Maintains compatibility with arbitrary prompt optimizers.
- **Mechanism**: Uses MIPRO as the default optimizer, supporting joint optimization of instructions and few-shot demonstrations. Bootstraps 3 demonstrations, 10 instruction candidates, and runs 10 optimization rounds.
- **Design Motivation**: The choice of prompt optimizer should not constrain the framework's applicability; a plug-and-play design ensures flexibility.

### Loss & Training

The optimization objective is task-specific validation metrics (e.g., accuracy on MATH, F1 on DROP). The MASS framework itself is a gradient-free search process: Stages 1 and 3 employ a prompt optimizer (MIPRO), while Stage 2 uses rejection sampling. Ten topology candidates are searched, each evaluated three times with the average taken.

## Key Experimental Results

### Main Results

Performance comparison across 8 benchmark tasks on Gemini 1.5 Pro:

| Method | MATH | DROP | HotpotQA | MuSiQue | MBPP | HumanEval | LCB | Avg. |
|--------|------|------|----------|---------|------|-----------|-----|------|
| CoT | 71.67 | 70.55 | 57.43 | 37.81 | 68.33 | 86.67 | 66.33 | 65.28 |
| Self-Consistency | 77.33 | 74.06 | 58.60 | 41.81 | 69.50 | 86.00 | 70.33 | 68.18 |
| Multi-Agent Debate | 78.67 | 71.78 | 64.87 | 46.00 | 68.67 | 86.67 | 73.67 | 70.26 |
| ADAS | 80.00 | 72.96 | 65.88 | 41.95 | 73.00 | 87.67 | 65.17 | 69.72 |
| **MASS** | **84.67** | **90.52** | **69.91** | **51.40** | **86.50** | **91.67** | **82.33** | **78.79** |

On Gemini 1.5 Flash, MASS achieves an average score of 74.30%, representing a 13.43 percentage point improvement over CoT (60.87%).

### Ablation Study

| Configuration | Avg. Performance | Notes |
|---------------|-----------------|-------|
| CoT (baseline) | 65.28% | Single-agent zero-shot reasoning |
| Stage 1 (1PO) | ~71% | Local prompt optimization, +6% over single-agent APO |
| Stage 1+2 (1PO+2TO) | ~74% | Additional +3% from topology optimization |
| Stage 1+2+3 (full MASS) | 78.79% | Further ~2% gain from global prompt optimization |
| Topology search w/o pruning | Decreased | Introduces detrimental building blocks |
| Topology search w/o Stage 1 | Decreased | Unoptimized agents lead to search in low-quality space |

### Key Findings

- Prompt optimization is far more token-efficient than simply increasing the number of agents: an optimized single agent with Self-Consistency outperforms a 9-agent SC system with default prompts.
- Not all topologies positively influence MAS performance—beneficial topologies constitute only a small fraction of the search space.
- MASS enables full parallelization of Stage 1 and Stage 2 optimization, whereas ADAS and AFlow are iterative algorithms that require waiting for prior steps to complete.
- Three MAS design principles emerge: (1) optimize individual agents before composition; (2) compose topologies with demonstrated influence; (3) model inter-agent dependencies through global optimization.

## Highlights & Insights

- **Analysis-Driven Design**: Rather than hastily proposing a method, the paper first conducts extensive experiments to analyze the impact of various factors in the MAS design space, yielding convincing conclusions.
- **Elegant Analogy to NAS**: The insight from neural architecture search—that search space design matters more than the search algorithm—is aptly applied to MAS design.
- **Underestimated Importance of Prompt Optimization**: The paper exposes a critical factor largely overlooked by most MAS works.
- **Parallelizable Optimization**: In practical deployment, this substantially reduces optimization time.

## Limitations & Future Work

- Building blocks in the search space must still be predefined, precluding the discovery of entirely novel agent interaction patterns.
- Topology construction rules follow a fixed sequence, which may limit more flexible agent composition strategies.
- Optimization costs remain high (requiring numerous API calls), making the approach potentially unsuitable for cost-sensitive scenarios.
- Cross-task transfer warrants further exploration—whether discovered MAS design principles can be directly applied to new tasks remains an open question.

## Related Work & Insights

- DSPy and MIPRO provide the infrastructure for prompt optimization, upon which MASS constructs MAS-level optimization.
- ADAS generates new topologies via a meta-agent but neglects prompt optimization; AFlow searches via MCTS but without search space pruning.
- Insight: When designing complex systems, analyzing the influence of individual components and pruning the search space is more efficient than brute-force search over the full space.

## Rating

- Novelty: ⭐⭐⭐⭐ The interleaved optimization idea is novel; the analysis-driven methodology is worth emulating.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Eight tasks, four LLMs, multiple baselines, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ In-depth analysis, rigorous logic, and clear figures.
- Value: ⭐⭐⭐⭐ Provides a systematic framework and design principles for automated MAS design.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)
- [\[ACL 2026\] Solver-Independent Automated Problem Formulation via LLMs for High-Cost Simulation-Driven Design](../../ACL2026/signal_comm/solver-independent_automated_problem_formulation_via_llms_for_high-cost_simulati.md)
- [\[CVPR 2026\] FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection](../../CVPR2026/signal_comm/faar_efficient_frequency-aware_multi-task_fine-tuning_via_automatic_rank_selecti.md)
- [\[AAAI 2026\] Text-Guided Channel Perturbation and Pretrained Knowledge Integration for Unified Multi-Modality Image Fusion](../../AAAI2026/signal_comm/text-guided_channel_perturbation_and_pretrained_knowledge_integration_for_unifie.md)
- [\[NeurIPS 2025\] The Last Vote: A Multi-Stakeholder Framework for Language Model Governance](../../NeurIPS2025/signal_comm/the_last_vote_a_multi-stakeholder_framework_for_language_model_governance.md)

<!-- RELATED:END -->
