---
title: >-
  [Paper Note] HoWToBench: Holistic Evaluation for LLM's Capability in Human-level Writing using Tree of Writing
description: >-
  [ACL 2026][LLM Evaluation][Tree-of-Writing] This paper identifies "negotiation inconsistency" in LLM-as-a-judge for long-form open-ended writing—where sub-score aggregation is unstable and unexplainable. It proposes Tree…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Tree-of-Writing"
  - "Writing Evaluation"
  - "Negotiation Bias"
  - "Length Paradox"
  - "Robustness"
date: 2026-05-08
content_hash: b4c69c71fe652608
---

# HoWToBench: Holistic Evaluation for LLM's Capability in Human-level Writing using Tree of Writing

**Conference**: ACL 2026  
**arXiv**: [2604.19071](https://arxiv.org/abs/2604.19071)  
**Code**: https://github.com/ZhuoerFeng/ACL2026-Tree-of-Writing (Available)  
**Area**: LLM Evaluation / Writing / LLM-as-a-judge  
**Keywords**: Tree-of-Writing, Writing Evaluation, Negotiation Bias, Length Paradox, Robustness

## TL;DR
This paper identifies "negotiation inconsistency" in LLM-as-a-judge for long-form open-ended writing—where sub-score aggregation is unstable and unexplainable. It proposes Tree-of-Writing (ToW), which explicitly models writing evaluation as a tree-like pipeline with three main nodes (Content / Format / Impression), leaf nodes, and explicit LLM-negotiator weights. On HoWToBench (1,302 Chinese entries across 12 genres), it increases system-level Pearson correlation from 0.85-0.89 to **0.93** and maintains robustness against common text perturbations.

## Background & Motivation

**Background**: LLM writing evaluation currently relies on either reference-based overlap metrics like BLEU/ROUGE (which lack discriminative power for long-form generation), the LLM-as-a-judge paradigm where GPT-4 provides a total score (suffering from verbosity/position bias), or Auto-Plan where the LLM dynamically determines sub-dimensions and weights (resulting in inconsistent scores across runs).

**Limitations of Prior Work**: The authors define the uncertainty of on-the-spot LLM aggregation as **Negotiation Inconsistency**. In Auto-Plan with self-consistency $N=5$, the fluctuation of sub-score proportions is $\delta = 0.273, \sigma = 0.059$, indicating the weighting logic drifts for the same article. Furthermore, long-form writing evaluation is inherently multi-dimensional (language, logic, plot, format, paragraphing), rendering simple averaging unreasonable.

**Key Challenge**: Human experts evaluate writing through explicit hierarchical decisions—scoring based on sub-criteria and then weighting by genre. However, LLM-as-a-judge implicitly embeds this process into a single generation, causing weight and scoring decisions to interfere, making them irreproducible and unexplainable.

**Goal**: (1) Propose a writing evaluation framework that reduces negotiation bias, is explainable, and is robust to perturbations; (2) Establish a large-scale Chinese benchmark covering professional human writing across multiple genres.

**Key Insight**: Decompose the human scoring process into two independent LLM calls: "sub-scoring" and "instruction-based weight planning." Sub-scores are provided by expert agents (rubric + reference), while weights are determined separately by a negotiator $J_W$ based on the instruction, followed by weighted aggregation using Depth First Search (DFS).

**Core Idea**: Use a tree structure and explicit weights to decompose "evaluation as a single generation" into "evaluation as a reproducible graph traversal," transforming an unexplainable black box into an auditable workflow.

## Method

### Overall Architecture
Two parallel components: (1) **ToW Framework**: The evaluation tree root $R$ connects to three main nodes $V_C$ (Content), $V_F$ (Format), and $V_I$ (Impression). $V_C$ and $V_F$ have specific leaf nodes $L_i$ (opening-ending, language-rhetoric, plots-structure, paragraphing, formatting...). Each leaf node is scored 1-10 using rubrics and references (Format involves hybrid regex + LLM). Leaf weights are provided by $J_W$ after reviewing each instruction, subject to $\sum w = 1, w \in (-1, 1)$, while main node weights are averaged. (2) **HoWToBench Dataset**: 1,302 professional human-authored texts from five high-quality websites, reverse-constructed into instruction + grounding info, covering 12 genres across 3 task types (Completion / Guide / Open).

### Key Designs

1.  **Tree-of-Writing (Main Nodes + Leaf Nodes)**:
    - **Function**: Decomposes holistic writing scoring into a structured, traversable, and auditable graph.
    - **Mechanism**: $\text{Score}(R) = \sum_{j \in \{C,F,I\}} w_{E_j} \text{Score}(V_j)$, where $\text{Score}(V_C) = \sum_{L_i \in \text{Child}(V_C)} w_{E_{V_C L_i}} \text{Score}(L_i)$. Leaf nodes for $V_C$ (opening-ending / language-rhetoric / argumentative-logics / emotion) are scored by LLMs using rubrics and references. $V_F$ leaf nodes (plots-structure / paragraphing / formatting) use mixed regex (for markdown headers/list levels) and LLM scoring.
    - **Design Motivation**: Rubrics and references anchor the LLM, preventing it from creating arbitrary standards. Regex handles hard rules for format, while LLM handles soft rules. The tree structure forces independent scoring for each dimension before aggregation, decoupling "scoring" from "aggregation."

2.  **Explicit Weight Allocation via LLM-Negotiator $J_W$**:
    - **Function**: Converts the determination of dimension importance into a separate, reproducible LLM call.
    - **Mechanism**: For each instruction $\mathcal{I}^i$, $J_W$ outputs leaf weights $(w_{E_{V_X L_1}}, \cdots, w_{E_{V_X L_n}})^i$ constrained by $\sum w = 1, w \in (-1, 1)$. Weight distributions vary significantly by genre (e.g., "logics" is high and variable in argumentative writing, بينما "opening-ending" remains stable at ~10% across genres).
    - **Design Motivation**: Making weights explicit allows for self-consistency comparisons. In tests, ToW's weight variance ($\delta = 0.080, \sigma = 0.017$) was significantly lower than Auto-Plan's (0.273 / 0.059). Weight visualization also enhances explainability.

3.  **HoWToBench Dataset Construction (Reverse Construction)**:
    - **Function**: Low-cost generation of instruction + grounding triplets strictly aligned with high-quality human references.
    - **Mechanism**: Crawl original texts $\mathcal{R}$ → Classify by genre using GPT-4o (98.6% accuracy) → Filter with Claude-3.5 using rubrics (retaining scores ≥4) → For Completion tasks, manually remove segments to create $\mathcal{G}$ and use templates for $\mathcal{I}$; for Guide / Open tasks, use Gemini-2.0-Flash for back-construction to generate $(S, T, \mathcal{G})$ triplets.
    - **Design Motivation**: (a) Reverse construction ensures references are human-authored. (b) Progressive difficulty (Completion / Guide / Open) distinguishes model capabilities under varying information density. (c) Selection of the best reference from four candidates (Human + 3 LLMs) by experts for 137/1302 cases ensures high-quality upper bounds.

### Loss & Training
The ToW framework is training-free; all LLM calls are implemented through prompt engineering using GPT-4o as the primary judge. HoWToBench was quality-audited by three humanities experts (96.85% pass rate). The MetaEditor meta-evaluation set (221 instructions × 9 LLMs) was scored twice by 36 annotators with writing backgrounds (Cohen's κ=0.71, Pearson=0.87).

## Key Experimental Results

### Main Results

| Method (Overall) | Cost ($) | Pearson ρ | Kendall τ | Spearman σ |
|---|---|---|---|---|
| BLEU-1 | - | 0.75 | 0.56 | 0.72 |
| ROUGE-L | - | 0.46 | 0.06 | 0.17 |
| Elaborated Rubric - best | 1.31 | 0.89 | 0.67 | 0.87 |
| Elaborated Rubric + SC (n=10) | 13.17 | 0.89 | 0.61 | 0.82 |
| Auto-Plan - best | 0.89 | 0.88 | 0.67 | 0.83 |
| Auto-Plan + SC (n=10) | 8.93 | 0.88 | 0.61 | 0.82 |
| Average scoring (ToW w/o plan) | 7.02 | 0.89 | 0.61 | 0.82 |
| **ToW (Ours)** | 7.34 | **0.93** | **0.83** | **0.93** |

| LLM | All | Completion | Guide | Open |
|---|---|---|---|---|
| DeepSeek-R1 | **6.10** | 6.10 | 6.15 | 6.06 |
| o3-mini | 5.86 | 6.16 | 5.80 | 5.69 |
| GPT-4o-1120 | 5.81 | **6.60** | 5.61 | 5.36 |
| Claude-3.5-Sonnet | 5.58 | 5.55 | 5.76 | 5.43 |
| Gemini-2.0-Flash | 5.43 | 5.43 | 5.53 | 5.33 |
| DeepSeek-V3 | 5.42 | 5.44 | 5.52 | 5.31 |
| Llama-3.3-70B-Instruct | 4.59 | 4.36 | 4.89 | 4.47 |

### Ablation Study

| Configuration (Guide / Open) | ρ Guide | τ Guide | ρ Open | τ Open |
|---|---|---|---|---|
| ToW (full) | **0.85** | **0.76** | **0.89** | **0.78** |
| w/o Content | 0.81 | 0.76 | 0.84 | 0.78 |
| w/o Format | 0.80 | 0.65 | 0.89 | 0.72 |
| w Content only | 0.79 | 0.65 | 0.89 | 0.78 |
| w Format only | 0.71 | 0.71 | 0.67 | 0.50 |
| w Impression only | 0.81 | 0.70 | 0.90 | 0.72 |

| Robustness Perturbations (Initial Score) | ToW | Auto-Plan | BLEU | BLEU-rt |
|---|---|---|---|---|
| Initial | 5.41 | 6.82 | 24.66 | 37.43 |
| Drop Paragraph | -0.36 | -0.06 | -7.27 | -2.07 |
| Repeat Paragraph | -0.49 | -0.30 | **+4.23** | -0.35 |
| Change to Comment | -0.30 | **+0.08** | +0.97 | -2.37 |
| Change to Poem | -0.62 | **+0.82** | -8.50 | +1.91 |

### Key Findings
- **ToW achieves SOTA across correlation metrics and is robust to perturbations**: Under the "Change to Poem" perturbation, Auto-Plan scores increased by +0.82 (deceived), and BLEU increased by +4.23 under "Repeat Paragraph"—both instances of reward hacking. ToW correctly identified score decreases across all six perturbations.
- **Quantitative evidence of Negotiation Bias**: Auto-Plan + Self-Consistency showed sub-score proportion drift ($\delta=0.273$), whereas ToW's weight drift was $\delta=0.080$, proving that explicit aggregation is more stable.
- **Negative correlation between length and quality**: In Guide tasks, input length vs. overall score correlation was -0.44, counter-intuitively suggesting that providing more grounding info does not necessarily lead to better LLM writing, contradicting the simple narrative of verbosity bias.
- **GPT-4o performed best in Completion (6.60) but dropped in Open tasks (5.36)**: This suggests strong "imitative" but weaker "creative" capabilities. DeepSeek-R1 was the most stable across all tasks, highlighting the potential of reasoning models in open-ended writing.
- **Format nodes alone are insufficient but critical**: While "Format only" yielded low correlation ($\rho=0.67$), removing it significantly harmed performance, indicating that format is a necessary but not sufficient dimension for human-like evaluation.

## Highlights & Insights
- **ToW transforms "evaluation" into "graph traversal"**: This explicit pipeline concept is transferable to any complex subjective evaluation scenario (code review, medical diagnosis, design aesthetics), with the negotiator-based weight decision being a key component.
- **The Reverse Construction of HoWToBench is highly valuable**: 1,302 professional human references across 12 genres and 3 task types, with a 96.85% expert pass rate, makes it one of the most solid ground truths for Chinese long-form writing evaluation.
- **The "Length Paradox" is a significant counter-intuitive finding**: While LLMs are generally perceived to prefer long outputs, this study proves that for high-quality evaluation, "longer $\neq$ better," correcting the simplified narrative of verbosity bias in literature.

## Limitations & Future Work
- **Covered only category-level genres**: Specific capabilities within sub-genres (e.g., suspense vs. romance under fiction) were not tested; multi-lingual expansion is also an open question.
- **Single-round generation**: Real-world writing processes such as self-iteration, human-AI collaboration, and multi-turn feedback were not covered.
- **Tree scalability**: The authors did not systematically verify if adding new leaf nodes would compromise existing correlations; the current 3+4 level hierarchy might not cover more complex tasks like academic peer review.
- **Cost**: ToW costs are comparable to Rubric self-consistency ($7.34 vs $6.53). While performance isn't just a result of compute, $0.5-$1 per evaluation is still expensive for industrial deployment.

## Related Work & Insights
- **vs. WritingBench (Wu et al. 2025)**: They utilize Auto-Plan for on-the-fly rubric generation. This paper proves that such implicit aggregation fails to converge under self-consistency, requiring an explicit negotiator $J_W$ for stability.
- **vs. AlignBench (Liu et al. 2024)**: That benchmark uses 4-dimensional rubrics for 75 tasks. This work scales to 8+ leaf nodes, 12 genres, and 1,302 tasks, utilizing "reverse construction + expert audit" to improve reference quality.
- **vs. HelloBench (Que et al. 2024)**: They proposed a taxonomy for long-text capability. This paper adopts a similar hierarchical evaluation concept while providing a complete, reproducible framework and benchmark.

## Rating
- Novelty: ⭐⭐⭐⭐ The "evaluation as graph traversal" paradigm and "negotiation inconsistency" concept are clear original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 1,302 tasks × 12 genres × 10 LLMs with multiple baselines, 36 expert annotators, and comprehensive robustness/ablation/length analyses.
- Writing Quality: ⭐⭐⭐⭐ Transparent progression from motivation to framework, data, and validation; visualizations (trees, weight distributions) are well-explained.
- Value: ⭐⭐⭐⭐⭐ HoWToBench is a high-quality open resource, and the "explicit negotiator" idea in the ToW framework is highly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Reward Modeling for Scientific Writing Evaluation](reward_modeling_for_scientific_writing_evaluation.md)
- [\[ACL 2026\] Fin-Bias: Comprehensive Evaluation for LLM Decision-Making under human bias in Finance Domain](fin-bias_comprehensive_evaluation_for_llm_decision-making_under_human_bias_in_fi.md)
- [\[ACL 2026\] HumanLLM: Benchmarking and Improving LLM Anthropomorphism via Human Cognitive Patterns](humanllm_benchmarking_and_improving_llm_anthropomorphism_via_human_cognitive_pat.md)
- [\[ACL 2026\] AgentEval: DAG-Structured Step-Level Evaluation for Agentic Workflows with Error Propagation Tracking](agenteval_dag-structured_step-level_evaluation_for_agentic_workflows_with_error_.md)
- [\[ACL 2026\] MultiFileTest: A Multi-File-Level LLM Unit Test Generation Benchmark and Impact of Error Fixing Mechanisms](multifiletest_a_multi-file-level_llm_unit_test_generation_benchmark_and_impact_o.md)

</div>

<!-- RELATED:END -->
