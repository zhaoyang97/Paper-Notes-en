---
title: >-
  [Paper Note] MATA: A Trainable Hierarchical Automaton System for Multi-Agent Visual Reasoning
description: >-
  [ICLR 2026][Interpretability][Multi-Agent Systems] This paper proposes MATA (Multi-Agent hierarchical Trainable Automaton), which formulates multi-agent visual reasoning as a hierarchical finite-state automaton. The top-…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Multi-Agent Systems"
  - "Hierarchical Finite-State Automaton"
  - "Visual Reasoning"
  - "Trainable State Controller"
  - "Collaboration and Competition"
date: 2026-05-08
content_hash: 85030ba2d1b80a4e
---

# MATA: A Trainable Hierarchical Automaton System for Multi-Agent Visual Reasoning

**Conference**: ICLR 2026
**arXiv**: [2601.19204](https://arxiv.org/abs/2601.19204)  
**Code**: [GitHub](https://github.com/ControlNet/MATA)  
**Area**: Interpretability
**Keywords**: Multi-Agent Systems, Hierarchical Finite-State Automaton, Visual Reasoning, Trainable State Controller, Collaboration and Competition

## TL;DR

This paper proposes MATA (Multi-Agent hierarchical Trainable Automaton), which formulates multi-agent visual reasoning as a hierarchical finite-state automaton. The top-level state transitions are learned by a trainable hyper agent (an LLM-based state controller), while each agent internally employs a rule-based sub-automaton. Collaboration and competition are realized through shared memory. MATA achieves state-of-the-art performance on multiple visual reasoning benchmarks.

## Background & Motivation

Visual reasoning requires models to interpret relationships among entities in visual scenes. Existing approaches suffer from the following limitations:

**End-to-end VLMs**: The implicit reasoning process is difficult to audit and prone to hallucinations on complex queries involving spatial relations or counting.

**Compositional methods** (e.g., ViperGPT, HYDRA): While improving interpretability, most rely on single-agent designs or manually engineered pipelines.

**Multi-agent methods**: Agents are assigned disjoint roles with hard-coded pipeline connections, unable to handle error propagation or support competition among functionally overlapping agents.

**Rigidity of rule-based transitions**: Manually written transition functions become increasingly intractable as the number of states grows.

The core problem is: **how can a system learn when to invoke which agent?** The authors formulate this decision problem as learning the transition function of a finite-state automaton.

## Method

### Overall Architecture

MATA is a hierarchical Mealy machine $\mathcal{M}_\theta = (S, S_0, \Sigma, \Lambda, \delta_\theta, \Gamma)$ with two levels:
- **Top level (Hyper Automaton)**: States correspond to individual agents; the transition function $\delta_\theta$ is learned by a trainable LLM controller.
- **Bottom level (Sub-Automaton)**: Each agent internally operates as a rule-based finite-state machine, ensuring reliable micro-control.

### Key Designs

#### 1. State Definition

The state set $S = S_{\text{agent}} \cup S_{\text{life}}$, where:

**Agent states** (three agents representing distinct reasoning paths):
- **Oneshot Reasoner**: Generates and executes a program in a single pass; suitable for directly solvable queries.
- **Stepwise Reasoner**: Generates Python programs incrementally for multi-step reasoning; suitable for complex queries.
- **Specialized Agent**: Fast perception experts (e.g., object detection, simple QA).

**Lifecycle states**: Initial (entry point), Final (terminates and outputs), Failure (unrecoverable error).

The three agents are designed to be **both collaborative and competitive**: collaboration manifests in downstream agents reading intermediate results written to shared memory by upstream agents; competition manifests in functionally overlapping agents substituting for failed ones.

#### 2. Shared Memory

All agents read from and write to a structured shared memory $m_t$, accumulating intermediate variables, perception results, program history, and verification feedback. Memory is **append-only**, ensuring the full reasoning trace is auditable. At each step, the hyper agent observes $m_t$ and selects the next state $s_{t+1} = \delta_\theta(s_t, m_t)$.

#### 3. Trainable Hyper Agent

The transition function $\delta_\theta$ is implemented by an SFT fine-tuned LLM. A text prompt $x_t$ is constructed from shared memory, and the LLM maps it to a distribution over available states to select the next state.

#### 4. Transition Trajectory Data Generation (MATA-SFT-90K)

**Step 1: Construct transition trajectory trees.** For each (image, query) pair, at every decision point the system branches into all available agent states, executes the corresponding sub-automaton, and saves memory checkpoints.

**Step 2: Bottom-up scoring.** Leaf nodes are scored by task metrics (Accuracy for VQA, IoU for VG); non-leaf nodes propagate the maximum child value upward:
$$V(s) = \begin{cases} \text{metric}(\hat{y}_s, y), & s \in \text{Leaves} \\ \max_{s' \in \text{Child}(s)} V(s'), & \text{otherwise} \end{cases}$$

**Step 3: Generate SFT data.** Each decision-point text prompt is paired with the state label of the optimal child node to form a training sample. A total of 90,854 samples are collected.

#### 5. Failure Handling Mechanism

When an agent reports an unrecoverable error, the failed agent is **temporarily removed** from the candidate state set, allowing the hyper agent to select an alternative agent and avoid infinite retries.

### Loss & Training

A standard SFT loss is used to train Qwen3 4B as the LLM state controller. Optimization uses AdamW with cosine decay and 5% warmup, batch size 64, trained for 8 epochs. The maximum number of steps at inference is $T=15$.

Three SFT configurations are evaluated: in-domain (trained on the target dataset's training split), domain-transfer (trained on non-target datasets), and general (trained jointly on all data).

## Key Experimental Results

### Main Results

**GQA (Compositional Visual Question Answering)**:

| Type | Method | Accuracy |
|------|--------|----------|
| End-to-end | InternVL2.5 (8B) | 61.5 |
| End-to-end | InternVL3.5 (8B) | 63.8 |
| Compositional | HYDRA | 52.8 |
| Compositional | **MATA (General)** | **64.9** |

**OK-VQA (Requires External Knowledge)**:

| Type | Method | Accuracy |
|------|--------|----------|
| End-to-end | InternVL3.5 (8B) | 75.7 |
| Compositional | DWIM | 62.8 |
| Compositional | **MATA (Domain-Specific)** | **76.5** |

**Referring Expression Comprehension (RefCOCO series)**:

| Method | RefCOCO | RefCOCO+ | RefCOCOg | Ref-Adv |
|--------|---------|----------|----------|---------|
| Florence2-L | 95.1 | 92.5 | 90.9 | 71.8 |
| NAVER | 96.2 | 92.8 | 91.6 | 75.4 |
| **MATA (General)** | **96.3** | **93.8** | **90.7** | **77.3** |

### Ablation Study

**Hyper Agent Component Ablation**:

| Hierarchical Automaton | Transition Strategy | SFT | GQA | OK-VQA | RefCOCO | Inference Time |
|----------------------|-------------------|-----|-----|--------|---------|----------------|
| ✗ | Exhaustive Ensemble | ✗ | 57.7 | 71.5 | 87.7 | 34.58s |
| ✓ | Random | ✗ | 57.1 | 71.1 | 85.3 | 6.91s |
| ✓ | LLM | ✗ | 58.5 | 75.1 | 95.8 | 8.07s |
| ✓ | LLM | ✓ | **64.9** | **76.5** | **96.3** | **8.01s** |

**Generalization Analysis**: The performance gap between cross-domain transfer and in-domain settings is less than 1%, indicating that the learned transition strategy is largely task-agnostic.

### Key Findings

1. **Compositional methods comprehensively surpass comparable-scale end-to-end VLMs for the first time**: MATA outperforms InternVL3.5 on GQA and OK-VQA.
2. **SFT yields substantial performance gains**: With only 90K samples, GQA accuracy improves from 58.5% to 64.9% (+6.4%).
3. **Small models are sufficient for scheduling**: A 0.6B model after SFT achieves in-domain performance approaching the 4B model.
4. **Collaboration + Competition > Pure Collaboration**: The three-agent design allows competition on the same task, with one agent substituting for another upon failure.
5. **Learned transitions >> Rule-based transitions**: MATA surpasses the hand-crafted NAVER by 1.9% on Ref-Adv.

## Highlights & Insights

- **Formal elegance**: Modeling multi-agent scheduling as Mealy machine transition function learning preserves interpretability while enabling flexibility.
- **Hierarchical decomposition**: Cross-agent transitions are learned while intra-agent steps are rule-based, cleanly separating what to learn from what to formalize.
- **Data generation pipeline**: Transition trajectory trees + bottom-up scoring + SFT data generation constitute a generalizable framework for multi-agent policy learning.
- **System 1 + System 2**: The Specialized/Oneshot/Stepwise agent design echoes the fast-and-slow thinking dichotomy in cognitive science.

## Limitations & Future Work

1. Scalability of trajectory tree search: Exhaustive search is feasible with 3 agents, but search cost grows exponentially as the number of agents increases.
2. Inference latency: An average of 8s/query remains high for real-time applications.
3. Dependence on foundation models: Performance is bounded by the capabilities of the underlying VLMs and detectors.
4. Simplicity of failure recovery: Only removing the failed agent is currently supported; more sophisticated recovery strategies may yield further improvements.
5. Limited training data sources: Training relies solely on the training splits of 5 datasets.

## Related Work & Insights

MATA follows the development trajectory of ViperGPT → HYDRA → NAVER, and is the first to introduce a learnable multi-agent transition policy. Compared to LLM-based multi-agent methods such as MetaGPT, MATA is more formally rigorous and supports competitive mechanisms. The transition trajectory tree generation approach is conceptually analogous to Monte Carlo Tree Search, but focuses specifically on agent selection.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The hierarchical automaton combined with learnable transition functions is a novel and formally complete framework design.
- **Technical Quality**: ⭐⭐⭐⭐⭐ — The Mealy machine formalization, trajectory tree data generation, and SFT training pipeline are tightly integrated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multi-benchmark comparisons, detailed ablations, generalization analysis, and model scale analysis are all provided.
- **Practicality**: ⭐⭐⭐⭐ — The framework is general-purpose, but inference cost is relatively high.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Formalization is clear and exposition is rigorous.
- **Overall**: ⭐⭐⭐⭐⭐ (9.0/10)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Auditing Cascading Risks in Multi-Agent Systems via Semantic–Geometric Co-evolution](auditing_cascading_risks_in_multi-agent_systems_via_semanti-geometric_co-evolut.md)
- [\[AAAI 2026\] ToC: Tree-of-Claims Search with Multi-Agent Language Models](../../AAAI2026/interpretability/toc_tree-of-claims_search_with_multi-agent_language_models.md)
- [\[AAAI 2026\] iMAD: Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference](../../AAAI2026/interpretability/imad_intelligent_multi-agent_debate_for_efficient_and_accura.md)
- [\[ICLR 2026\] Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data](behavior_learning_bl_learning_hierarchical_optimization_structures_from_data.md)
- [\[NeurIPS 2025\] AgentiQL: An Agent-Inspired Multi-Expert Framework for Text-to-SQL Generation](../../NeurIPS2025/interpretability/agentiql_an_agent-inspired_multi-expert_framework_for_text-to-sql_generation.md)

</div>

<!-- RELATED:END -->
