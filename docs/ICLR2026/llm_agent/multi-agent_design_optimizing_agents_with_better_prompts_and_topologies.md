---
title: >-
  [Paper Note] Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies
description: >-
  [ICLR 2026][LLM Agent][multi-agent system] This paper provides a systematic analysis of the respective contributions of prompt design and topology design in multi-agent systems (MAS), finding that prompt optimization is the single most critical factor—a single agent with optimized prompts can outperform complex multi-agent topologies. The paper proposes Mass, a three-stage framework (block-level prompt → topology → workflow-level prompt) that achieves state-of-the-art performance across 8 benchmarks.
tags:
  - ICLR 2026
  - LLM Agent
  - multi-agent system
  - prompt optimization
  - topology search
  - automated MAS design
  - workflow optimization
date: 2026-05-08
content_hash: b61c061f6e7a329f
---

# Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies

**Conference**: ICLR 2026
**arXiv**: [2502.02533](https://arxiv.org/abs/2502.02533)
**Code**: To be confirmed
**Area**: Agent
**Keywords**: multi-agent system, prompt optimization, topology search, automated MAS design, workflow optimization

## TL;DR
This paper provides a systematic analysis of the respective contributions of prompt design and topology design in multi-agent systems (MAS), finding that prompt optimization is the single most critical factor—a single agent with optimized prompts can outperform complex multi-agent topologies. The paper proposes Mass, a three-stage framework (block-level prompt → topology → workflow-level prompt) that achieves state-of-the-art performance across 8 benchmarks.

## Background & Motivation
**State of the Field**: Multi-agent systems (MAS) organize multiple LLM agents through topologies such as Debate, Reflect, and Aggregate. Recent work has explored automated MAS design methods, including ADAS and AFlow.

**Limitations of Prior Work**: It remains unclear whether MAS performance gains stem from multi-agent topologies or from better prompts. Many complex topologies actually degrade performance, yet the underlying reasons are not well understood.

**Root Cause**: The benefit of increasing agent count and topology complexity is uncertain—sometimes helpful, sometimes harmful.

**Paper Goals**: ① Quantify the relative contributions of prompts vs. topologies; ② Design a unified automated framework that jointly optimizes both.

**Starting Point**: Controlled variable analysis—first optimizing prompts in isolation, then layering topology search on top.

**Core Idea**: Prompt optimization >> topology selection; however, joint optimization of both > either alone.

## Method

### Overall Architecture
Mass performs three-stage alternating optimization: ① Block-level prompt optimization (independently optimizing the instruction and exemplar for each agent module) → ② Workflow topology optimization (pruning the search space based on the incremental influence of each module) → ③ Workflow-level prompt optimization (globally and jointly optimizing prompts over the selected optimal topology).

### Key Designs

1. **Block-level Prompt Optimization (Warm-up Stage)**:

    - Independently performs joint instruction + exemplar optimization for each agent module
    - Iteratively refines prompts using validation-set feedback
    - Serves as "pre-training" for topology search, ensuring prompt quality for each module
    - Design Motivation: Experiments show that prompt-optimized single agents already outperform complex topologies such as SC, Reflect, and Debate

2. **Workflow Topology Optimization**:

    - Computes the incremental influence $I_{a_i}$ of each module
    - Prunes the search space via softmax-based probability sampling
    - Evaluates candidate topologies using prompts optimized in Stage 1
    - Finding: Not all topologies contribute positively—e.g., on HotpotQA, only debate yields a ~3% gain

3. **Workflow-level Prompt Optimization**:

    - After the optimal topology is determined, jointly optimizes prompts for all agents globally
    - Accounts for inter-agent interaction effects (i.e., how information flow within the topology influences prompt design)
    - Performs fine-grained adjustment to adapt prompts to the final topology

## Key Experimental Results

### Main Results (8 Benchmarks)

| Method | MATH | HotpotQA | MMLU | Average |
|--------|------|----------|------|---------|
| SC (Self-Consistency) | Baseline | Baseline | Baseline | Baseline |
| Reflect | Slightly higher | Slightly lower | Slightly higher | Mixed |
| Debate | Slightly higher | +3% | Slightly lower | Mixed |
| ADAS | High | High | High | Strong baseline |
| **Mass** | **Highest** | **Highest** | **Highest** | **SOTA** |

### Ablation Study

| Comparison | Conclusion |
|------------|------------|
| Single agent + prompt optimization vs. multi-agent (no prompt optimization) | **Single agent is superior** |
| Mass (3-stage) vs. prompt-only vs. topology-only | Mass is **significantly best** |
| Transfer from Gemini 1.5 Pro → Claude 3.5 Sonnet | **Findings transfer across models** |

### Key Findings
- **A prompt-optimized single agent already outperforms SC/Reflect/Debate**—challenging the intuition that "more agents is always better"
- Mass achieves SOTA on all 8 benchmarks, significantly outperforming automated baselines such as ADAS and AFlow
- Not all topologies have a positive impact—in approximately 50% of cases, additional topology components actually hurt performance
- Findings transfer across model families (Gemini → Claude → Mistral)

## Highlights & Insights
- **"Prompt > Topology"** is the central finding—offering an important corrective for the MAS community: optimize prompts before pursuing complex topologies
- The **three-stage alternating optimization** design is well-motivated—warm-up first, then search, then fine-tune, avoiding cold-start issues
- **Incremental influence pruning** effectively reduces the topology search space
- The approach is generalizable to any MAS framework—Mass's methodology is not tied to specific topology types

## Limitations & Future Work
- The search space relies on predefined building blocks (Aggregate/Reflect/Debate), limiting the discovery of arbitrary structures
- The optimization process requires validation-set feedback, and computational cost grows with the number of agents
- Topology construction rules follow a fixed order, constraining the discovery of unconventional topologies
- Dynamic topology selection at inference time (e.g., selecting topology based on input difficulty) is not considered

## Related Work & Insights
- **vs. ADAS**: ADAS searches over agent architectures; Mass jointly optimizes prompts and topology
- **vs. AFlow**: AFlow searches workflows via code generation; Mass optimizes via validation-set feedback
- **vs. DSPy**: DSPy optimizes prompt pipelines; Mass simultaneously optimizes topology
- Insight: The performance bottleneck in MAS may lie not in topology complexity but in the prompt quality of individual agents

## Supplementary Discussion

### Why Do Complex Topologies Often Fail?
Multi-agent collaboration introduces additional "communication overhead"—each agent's output may contain noise or irrelevant information, which accumulates and ultimately disrupts the final decision. Debate topology is effective on HotpotQA (where multi-perspective discussion aids fact verification) but degrades performance on mathematical tasks (where reasoning is deterministic and debate introduces unwanted variance). This suggests that topology selection should be task-dependent rather than following a "one topology fits all" approach.

### Necessity of Three-Stage Optimization

Experiments demonstrate that prompt-only optimization or topology-only search are both inferior to the full three-stage joint optimization.

The key reason is the interaction effect between prompts and topology: the optimal prompt may differ across topologies, and the optimal topology likewise depends on prompt quality. This establishes MAS design as a joint optimization problem that cannot be decomposed into independent subproblems.

## Rating
- Novelty: ⭐⭐⭐⭐ The "Prompt > Topology" finding is valuable; the three-stage design is well-reasoned
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 benchmarks, cross-model validation, comprehensive ablations
- Writing Quality: ⭐⭐⭐⭐ Controlled variable analysis is clearly presented
- Value: ⭐⭐⭐⭐ Offers direct practical guidance for MAS design

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Automated Multi-Agent Workflows for RTL Design](../../NeurIPS2025/llm_agent/automated_multi-agent_workflows_for_rtl_design.md)
- [\[NeurIPS 2025\] Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?](../../NeurIPS2025/llm_agent/debate_or_vote_which_yields_better_decisions_in_multi-agent_large_language_model.md)
- [\[ICLR 2026\] M²-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining](m2-miner_multi-agent_enhanced_mcts_for_mobile_gui_agent_data_mining.md)
- [\[ICLR 2026\] M2-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining](m2-miner_multi-agent_enhanced_mcts_for_mobile_gui_agent.md)
- [\[ICLR 2026\] HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatre](hamlet_a_hierarchical_and_adaptive_multi-agent_framework_for_live_embodied_theat.md)

<!-- RELATED:END -->
