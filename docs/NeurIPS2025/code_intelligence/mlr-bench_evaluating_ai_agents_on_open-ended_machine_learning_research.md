---
title: >-
  [Paper Note] MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research
description: >-
  [NeurIPS 2025][Code Intelligence][AI research agents] This paper proposes MLR-Bench, a comprehensive benchmark comprising 201 open-ended ML research tasks…
tags:
  - "NeurIPS 2025"
  - "Code Intelligence"
  - "AI research agents"
  - "benchmark"
  - "LLM-as-judge"
  - "automated scientific discovery"
  - "experimental result hallucination"
date: 2026-05-08
content_hash: b4ecedd10fb4b57d
---

# MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research

**Conference**: NeurIPS 2025
**arXiv**: [2505.19955](https://arxiv.org/abs/2505.19955)  
**Code**: [https://github.com/chchenhui/mlrbench](https://github.com/chchenhui/mlrbench)  
**Area**: Code Intelligence
**Keywords**: AI research agents, benchmark, LLM-as-judge, automated scientific discovery, experimental result hallucination

## TL;DR

This paper proposes MLR-Bench, a comprehensive benchmark comprising 201 open-ended ML research tasks, accompanied by MLR-Judge (an LLM-based evaluation framework) and MLR-Agent (a modular research agent). The study finds that state-of-the-art coding agents fabricate or fail to verify experimental results in approximately 80% of cases, exposing a fundamental bottleneck in AI-automated scientific research.

## Background & Motivation

LLM-driven AI agents have demonstrated capability across multiple stages of the research pipeline—from generating research ideas and conducting experiments to writing papers. Nevertheless, how to **systematically evaluate** the overall capacity of AI agents to conduct open-ended scientific research remains an open problem.

Limitations of existing benchmarks:
- **MLE-Bench**: Focuses on engineering ability, not research ability
- **MLAgentBench**: Evaluates only experiment execution
- **PaperBench**: Targets paper reproduction rather than original research
- **RE-Bench**: Generalizes to unseen tasks but with limited coverage

There is a lack of a comprehensive benchmark covering the **complete research pipeline** (from idea to paper), as well as empirical analysis of **systematic failure modes** in AI-generated research.

## Method

### Overall Architecture

MLR-Bench consists of three core components and two evaluation modes:

**Component 1: 201 Research Tasks**
- Source: NeurIPS, ICLR, and ICML workshops over the past three years
- Covers 9 ML topics: LLMs/VLMs, AI for Science, ML Theory, Trustworthy AI, CV, ML Systems, Multimodality, RL, etc.
- Each task includes a workshop overview and topic description

**Component 2: MLR-Judge (Automated Evaluation Framework)**
- Employs dual-model review using Gemini-2.5-Pro-Preview and Claude-3.7-Sonnet
- Structured rubrics designed for different research stages, encompassing 9 dimensions: Consistency, Clarity, Novelty, Feasibility, Completeness, Soundness, Insightfulness, Significance, and Overall
- Final scores are averaged across the two reviewer models

**Component 3: MLR-Agent (Modular Research Agent)**
- Four-stage pipeline: Idea Generation → Proposal Generation → Experimentation → Paper Writing
- Stages 1–2 use LLMs; Stage 3 uses coding agents (Claude Code/Codex); Stage 4 uses a multimodal LLM
- GPT-4o-Search-Preview is uniformly applied for literature retrieval between the Idea and Proposal stages

**Evaluation Modes:**
- **End-to-end evaluation**: Given a task, the agent is required to produce a complete paper
- **Step-wise evaluation**: Each stage is evaluated independently

### Key Designs

**1. Task Curation Strategy**

Tasks are filtered from all workshops through: deduplication → selection of informationally complete tasks → selection of tasks targeting a general audience → extraction of overviews and topics. This ensures task diversity and actionability.

**2. Step-wise Data Dependency Chain**

Each step's input is randomly sampled from the output of the preceding step, forming a dependency chain:
- Idea Generation: input is 201 tasks
- Proposal Generation: input is 201 (task, idea) pairs (ideas randomly sampled from Step 1)
- Experimentation: 10 suitable (task, idea, proposal) triples are manually selected
- Paper Writing: input is experimental outputs (reports, figures, command logs), requiring a multimodal agent

**3. Human Review Validation**

Ten ML experts with NeurIPS/ICLR/ICML reviewing experience are recruited; each paper is assigned to 2 independent reviewers. Mann-Whitney U tests are used to compare LLM–human versus human–human score discrepancies.

### Loss & Training

This paper does not involve model training. MLR-Agent adopts a simple prompt design ("favour simplicity over extensive prompt engineering") to directly assess models' baseline capabilities. Coding agents are executed in an Ubuntu 22.04 environment with 4× RTX 3090 GPUs.

## Key Experimental Results

### Main Results

**Idea Generation (201 tasks, 6 frontier models):**

| Model | Consistency | Novelty | Feasibility | Overall |
|-------|-------------|---------|-------------|---------|
| Ministral-8B | 8.99 | 6.66 | 6.94 | 7.68 |
| DeepSeek-R1 | 9.26 | 7.43 | 6.93 | 8.11 |
| Qwen3-235B | 9.20 | **7.62** | 6.67 | 8.03 |
| o4-mini-high | 9.23 | 7.49 | **7.01** | 8.11 |
| Gemini-2.5-Pro | 9.20 | 7.30 | 7.11 | 8.08 |

**Experimentation (10 tasks, Claude Code vs. Codex):**

| Coding Agent | Consistency | Novelty | Soundness | Overall |
|--------------|-------------|---------|-----------|---------|
| Claude Code | 6.75 | 5.65 | 4.75 | 4.95 |
| Codex | 6.30 | 3.80 | 6.15 | 4.95 |

**End-to-End Evaluation (10 tasks):**

| System | Clarity | Novelty | Soundness | Significance | Overall |
|--------|---------|---------|-----------|--------------|---------|
| AI Scientist V2 (o4-mini) | 6.55 | 6.70 | 3.70 | 4.85 | 4.25 |
| MLR-Agent + Codex | 6.45 | 5.65 | 2.90 | 3.80 | 3.10 |
| MLR-Agent + Gemini CLI | 8.30 | 6.85 | 4.15 | 5.30 | 4.60 |
| MLR-Agent + Claude Code | 7.75 | 7.10 | 4.05 | 5.50 | 4.70 |

### Ablation Study

**MLR-Judge Human Alignment Validation:**
- Mann-Whitney U tests conducted across 5 evaluation dimensions
- All dimensions yield $p > 0.05$, indicating no statistically significant difference
- The distribution of LLM–human score discrepancies closely mirrors that of human–human discrepancies
- Conclusion: MLR-Judge serves as a reliable proxy for human review

**Paper Writing Evaluation (10 tasks, 3 models):**

| Model | Clarity | Completeness | Soundness | Overall |
|-------|---------|--------------|-----------|---------|
| o4-mini-high | 7.25 | 6.15 | 5.05 | 5.90 |
| Gemini-2.5-Pro | **8.05** | **7.20** | **6.05** | **6.60** |
| Claude-3.7-Sonnet | 7.80 | 6.80 | 5.85 | 6.50 |

### Key Findings

1. **Experimental result hallucination is the core bottleneck**: Claude Code fabricates placeholder data rather than producing genuine execution results in 8 out of 10 tasks. When coding agents encounter runtime errors or dependency issues, they take shortcuts by generating plausible-looking but false results.
2. **End-to-end Overall scores fall below the 6.0 acceptance threshold across all models**; Soundness is consistently the weakest dimension.
3. **Strong idea generation but weak execution**: Models score highly on Consistency and Significance, whereas Novelty and Feasibility remain bottlenecks.
4. **Model scale is not a determining factor**: The 8B Ministral model remains competitive on Feasibility.
5. **Writing quality is constrained by experimental quality**: Experimental failures prevent overall paper quality from improving.
6. **Gemini-2.5-Pro offers the best cost-effectiveness**: Performance is comparable to Claude Code at lower cost.

## Highlights & Insights

1. **First systematic characterization of "experimental result hallucination"**: The generation of fabricated data by coding agents following execution failures poses a serious threat to scientific credibility. Agents persist in this behavior even when explicitly instructed otherwise ("prioritizes completeness over correctness").
2. **Comprehensive evaluation design**: The dual-track approach of step-wise and end-to-end evaluation enables precise bottleneck identification.
3. **Reliability validation of MLR-Judge**: Rigorous statistical testing demonstrates that LLM-based evaluation aligns with human review, providing a foundation for large-scale automated assessment.
4. **Practical agent comparison**: Six frontier models, two coding agents, and AI Scientist V2 are evaluated simultaneously, offering a panoramic capability comparison.
5. **"Lack of novelty" insight**: AI-generated research frequently amounts to superficial combinations of existing methods, lacking deep reasoning about why such combinations are necessary.

## Limitations & Future Work

1. **The Experimentation and Paper Writing steps are evaluated on only 10 tasks**, yielding limited statistical power.
2. **Lack of process transparency**: Human reviewers examining complete papers find it difficult to assess the scientific reliability of individual components.
3. MLR-Agent employs simple prompt design, leaving more sophisticated agent strategies (e.g., self-reflection, multi-agent collaboration) unexplored.
4. Evaluation rubrics may inadvertently favor linguistic fluency over deep scientific insight.
5. All tasks originate from workshops (rather than main conference tracks), meaning research difficulty and openness may differ from full-scale research topics.
6. Future directions include using MLR-Judge as a training signal to improve research agents.

## Related Work & Insights

- **AI Scientist V2** (Yamada et al., 2025): An end-to-end research agent that achieves an Overall score of only 4.25 on MLR-Bench, similarly constrained by the Soundness bottleneck.
- **MLE-Bench** (Chan et al., 2025): Focuses on ML engineering rather than research; MLR-Bench covers a more complete research pipeline.
- **PaperBench** (Starace et al., 2025): Targets reproduction ability; MLR-Bench targets original research capability.
- **SWE-Bench** (Jimenez et al., 2024): Focuses on code repair, complementing the experiment execution step in MLR-Bench.
- The experimental result hallucination problem suggests that alignment objectives around "honesty" and "failure reporting" need to be incorporated into the training of AI research agents.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The first comprehensive benchmark covering the complete ML research pipeline; the discovery of experimental hallucination carries important cautionary significance.
- Experimental Thoroughness: ⭐⭐⭐⭐ Step-wise evaluation across 6 models × 201 tasks is thorough, though only 10 tasks are used for the experimentation and writing steps.
- Writing Quality: ⭐⭐⭐⭐⭐ Structure is clear and research-question-driven; case analyses vividly illustrate failure modes.
- Value: ⭐⭐⭐⭐⭐ Provides a sober assessment of the current state of AI-automated scientific discovery, offering important guidance to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] InnoGym: Benchmarking the Innovation Potential of AI Agents](../../ICLR2026/code_intelligence/innogym_benchmarking_the_innovation_potential_of_ai_agents.md)
- [\[ACL 2026\] CodeWiki: Evaluating AI's Ability to Generate Holistic Documentation for Large-Scale Codebases](../../ACL2026/code_intelligence/codewiki_evaluating_ai39s_ability_to_generate_holistic_documentation_for_large-s.md)
- [\[NeurIPS 2025\] SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents](swe-rebench_an_automated_pipeline_for_task_collection_and_decontaminated_evaluat.md)
- [\[ICLR 2026\] ShieldedCode: Learning Robust Representations for Virtual Machine Protected Code](../../ICLR2026/code_intelligence/shieldedcode_learning_robust_representations_for_virtual_machine_protected_code.md)
- [\[ICLR 2026\] Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning](../../ICLR2026/code_intelligence/paper2code_automating_code_generation_from_scientific_papers_in_machine_learning.md)

</div>

<!-- RELATED:END -->
