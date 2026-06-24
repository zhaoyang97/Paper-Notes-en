---
title: >-
  [Paper Note] DeepSolution: Boosting Complex Engineering Solution Design via Tree-based Exploration and Bi-point Thinking
description: >-
  [ACL 2025][Model Compression][RAG] This paper proposes SolutionBench, a new benchmark, and SolutionRAG, a new framework, for complex engineering solution design. By leveraging tree-based exploration and bi-point thinking (alternating design and review) within a RAG framework, it progressively generates reliable engineering solutions satisfying multiple constraints, achieving state-of-the-art (SOTA) results across 8 engineering domains.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "RAG"
  - "tree search"
  - "engineering solution design"
  - "bi-point thinking"
  - "benchmark"
date: 2026-05-08
content_hash: b6f73560d67c598a
---

# DeepSolution: Boosting Complex Engineering Solution Design via Tree-based Exploration and Bi-point Thinking

**Conference**: ACL 2025  
**arXiv**: [2502.20730](https://arxiv.org/abs/2502.20730)  
**Code**: [GitHub](https://github.com/Li-Z-Q/DeepSolution)  
**arXiv**: [2502.20730](https://arxiv.org/abs/2502.20730)  
**Code**: [https://github.com/Li-Z-Q/DeepSolution](https://github.com/Li-Z-Q/DeepSolution)  
**Area**: Model Compression  
**Keywords**: RAG, tree search, engineering solution design, bi-point thinking, benchmark  
**Authors**: Zhuoqun Li, Haiyang Yu, Xuanang Chen, Hongyu Lin, Yaojie Lu, Fei Huang, Xianpei Han, Yongbin Li, Le Sun (ISCAS & Tongyi Lab)  

## TL;DR

This paper proposes SolutionBench, a new benchmark, and SolutionRAG, a new framework, for complex engineering solution design. By leveraging tree-based exploration and bi-point thinking (alternating design and review) within a RAG framework, it progressively generates reliable engineering solutions satisfying multiple constraints, achieving state-of-the-art (SOTA) results across 8 engineering domains.

## Background & Motivation

- **Task Definition**: Complex engineering solution design requires generating complete and feasible solutions for engineering demands characterized by multiple real-world constraints (e.g., designing a safe and efficient hospital construction plan in an area with 3000mm annual rainfall, expansive soil, and high-frequency earthquakes).
- **Limitations of Prior Work**: Previous RAG research focused primarily on Multi-hop QA and Long-form QA, where answers are entity fragments or concatenated paragraphs. In contrast, engineering solution design involves a flexible improvement process and complete solutions that satisfy all constraints, showing fundamental differences.
- **Key Challenge**: (1) The improvement path from sub-optimal to reliable solutions is highly flexible and lacks fixed reasoning patterns; (2) Demands contain multiple real-world constraints, making it extremely difficult to satisfy all of them in a single generation.

## Method

### 1. SolutionBench Benchmark Construction

Construction process: Collect technical reports from authoritative engineering journals $\rightarrow$ extract structured content using GPT-4o with hand-crafted templates $\rightarrow$ manual verification and deduplication $\rightarrow$ combine into a dataset and knowledge base spanning 8 engineering domains.

Each data item contains 5 fields:
- **Requirement**: Complex engineering requirements from real-world scenarios.
- **Solution**: Standard solutions designed by industry experts.
- **Analytical Knowledge**: Professional knowledge used in analyzing the requirements.
- **Technical Knowledge**: Technical knowledge utilized to resolve the requirements.
- **Explanation**: Detailed description of the expert's solution design process.

Covers 8 domains: Environmental, Mining, Transportation, Aerospace, Telecom, Construction, Water Conservancy, and Agriculture, totaling approximately 950 data items and 6,000 pieces of knowledge.

### 2. SolutionRAG System

The core idea is to perform tree search reasoning over a **Bi-point Thinking Tree**:

**(a) Bi-point Thinking Tree Structure**
- **Solution Node**: Stores solutions designed for the requirements (low reliability at shallower depth, higher reliability at deeper depth).
- **Comment Node**: Stores review comments pointing out the limitations of a specific solution.
- The two types of nodes alternate: Solution Node $\rightarrow$ Comment Node $\rightarrow$ Better Solution Node $\rightarrow$ ...

**(b) Node Expansion — Design & Review**
- **Design**: Given requirement $q$, upper-level comment $c$, and historical solution $s$, the LLM first samples $H$ improvement proposals $\rightarrow$ retrieves relevant knowledge for each proposal from the knowledge base $\rightarrow$ integrates this info to generate a better solution.
- **Review**: Given requirement $q$ and current solution $s$, the LLM similarly generates $H$ review directions $\rightarrow$ retrieves knowledge $\rightarrow$ generates review comments.

**(c) Node Evaluation and Pruning**
- Use LLM logits to score solution nodes and comment nodes.
    - Solution score: Concatenate Solution + Comment + suffix "According to the comment, above solution is reliable", and take the average logits as the reliability score.
    - Comment score: Concatenate Old Solution + Comment + New Solution + suffix "Comparing the new solution and old solution, the comment is helpful", and take the average logits as the helpfulness score.
- Only the $W$ highest-scoring nodes are kept at each layer to balance efficiency and performance.

**Hyperparameter Settings**: Maximum tree depth $L=5$, number of child nodes per node $H=2$, number of retained nodes $W=1$, base model Qwen2.5-7B-Instruct, retrieval model NV-Embed-v2, and top-$R=10$ retrieved documents.

## Key Experimental Results

### Table 1: Data Statistics of SolutionBench

| Engineering Domain | Data Count | Knowledge Count |
|:---------|:---------|:---------|
| Environmental | 119 | 554 |
| Mining | 117 | 543 |
| Transportation | 124 | 870 |
| Aerospace | 115 | 802 |
| Telecom | 116 | 840 |
| Construction | 118 | 858 |
| Water Conservancy | 119 | 802 |
| Agriculture | 122 | 868 |

### Table 2: Main Results (Analytical Score / Technical Score)

| Method | Environmental | Mining | Transportation | Aerospace | Telecom | Construction | Water Conservancy | Agriculture |
|:-----|:-----|:-----|:-----|:---------|:-----|:-----|:-----|:-----|
| o1-2024-12-17 | 60.5/48.3 | 51.9/37.5 | 57.3/44.7 | 57.8/47.6 | 63.5/52.3 | 61.2/52.0 | 59.9/50.4 | 62.9/52.2 |
| Naive-RAG | 64.8/62.2 | 57.2/40.1 | 62.7/54.9 | 67.7/65.4 | 67.4/66.8 | 66.2/63.3 | 66.0/57.5 | 65.7/63.0 |
| Self-RAG | 64.2/63.6 | 56.1/41.6 | 62.9/56.5 | 68.8/69.9 | 67.6/66.9 | 66.7/65.9 | 64.8/58.6 | 65.1/61.1 |
| **SolutionRAG** | **66.4/67.9** | **59.7/50.5** | **64.1/58.5** | **69.9/72.7** | **68.8/69.0** | **67.9/68.0** | **66.0/60.7** | **66.9/65.2** |

- SolutionRAG achieves SOTA across all 8 domains.
- On Mining TS, it secures a $+10.4$ gain over Naive-RAG and a $+8.9$ gain over Self-RAG.

### Table 3: Ablation Study (Overall AS/TS)

| Configuration | Overall AS | Overall TS |
|:-----|:-----------|:-----------|
| SolutionRAG (Full) | 66.2 | 64.1 |
| w/o Tree Structure (degrades to single chain) | 62.7 | 61.7 |
| w/o Bi-point Thinking (solutions only, no reviews) | 62.9 | 61.5 |

Tree search and bi-point thinking contribute comparably, both showing significantly positive effects.

## Highlights & Insights

- Novelly defines the "complex engineering solution design" task, filling a research gap for RAG in engineering.
- Bi-point thinking (alternating design and review) acts as a structured upgrade to self-refine, explicitly ensuring solution completeness through review nodes.
- The tree search + logit-based pruning mechanism is simple and efficient, outperforming larger models like o1 with only a 7B model.
- Constructs a high-quality benchmark, SolutionBench, spanning 8 engineering domains with expert annotations.

## Limitations & Future Work

- Relies solely on existing LLM capabilities without incorporating reinforcement learning, leaving the solution quality capped by the base model.
- Due to GPU resource constraints, the hyperparameter space (such as tree width $H$ and depth $L$) is not fully explored.
- Evaluation relies on GPT-4o scoring, which may introduce rating bias.
- The paper's area label is marked as model_compression, but the actual direction is RAG/engineering design, presenting a taxonomy bias.

## Related Work & Insights

| Evaluation Dimension | Existing Methods | SolutionRAG |
|:---------|:---------|:------------|
| Task Type | Multi-hop QA / Long-form QA | Complex engineering solution design (multi-constraint + complete solution) |
| Inference Structure | Single-chain iteration (Self-RAG/RQ-RAG) | Bi-point thinking tree (alternating solution-review) |
| Constraint Satisfaction | No explicit guarantee mechanism | Review nodes explicitly detect missing constraints |
| MCTS-based RAG | Lacks mechanism to guarantee engineering constraints | Guarantees solution reliability through bi-point thinking |

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to define the engineering solution design task and construct a dedicated benchmark; the bi-point thinking tree is an interesting architectural innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive testing across 8 domains, ablation studies, tree depth analysis, and pruning effectiveness validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, intuitive figures, and well-defined tasks.
- **Value**: ⭐⭐⭐ — The benchmark and system hold practical value for engineering, but the GPT-4o evaluation cost is high.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Pre-training Distillation for Large Language Models: A Design Space Exploration](pre-training_distillation_for_large_language_models_a_design_space_exploration.md)
- [\[ACL 2025\] TeamLoRA: Boosting Low-Rank Adaptation with Expert Collaboration and Competition](teamlora_boosting_low-rank_adaptation_with_expert_collaboration_and_competition.md)
- [\[ACL 2025\] Data Laundering: Artificially Boosting Benchmark Results through Knowledge Distillation](data_laundering_artificially_boosting_benchmark_results_through_knowledge_distil.md)
- [\[ACL 2025\] CAMI: A Counselor Agent Supporting Motivational Interviewing through State Inference and Topic Exploration](cami_a_counselor_agent_supporting_motivational_interviewing_through_state_infere.md)
- [\[NeurIPS 2025\] Ensemble++: Scalable Exploration via Ensemble](../../NeurIPS2025/model_compression/scalable_exploration_via_ensemble.md)

</div>

<!-- RELATED:END -->
