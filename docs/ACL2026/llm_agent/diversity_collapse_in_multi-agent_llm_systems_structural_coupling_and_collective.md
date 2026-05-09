---
title: >-
  [Paper Note] Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation
description: >-
  [ACL 2026 Findings][LLM Agent][Multi-agent systems] Through evaluating over 10,000 research proposals, this paper systematically reveals the phenomenon of "diversity collapse" in multi-agent LLM systems across three levels — model intelligence, agent cognition, and system dynamics. Stronger models, authority-driven role assignments, and dense communication topologies all suppress semantic diversity, with the root cause residing in interaction structure rather than insufficient model capability.
tags:
  - ACL 2026 Findings
  - LLM Agent
  - Multi-agent systems
  - diversity collapse
  - structural coupling
  - idea generation
  - collaboration topology
date: 2026-05-08
content_hash: f0653dcf5b75e54f
---

# Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation

**Conference**: ACL 2026 Findings
**arXiv**: [2604.18005](https://arxiv.org/abs/2604.18005)
**Code**: [https://github.com/Xtra-Computing/MAS_Diversity](https://github.com/Xtra-Computing/MAS_Diversity)
**Area**: LLM Agent
**Keywords**: Multi-agent systems, diversity collapse, structural coupling, idea generation, collaboration topology

## TL;DR

Through evaluating over 10,000 research proposals, this paper systematically reveals the phenomenon of "diversity collapse" in multi-agent LLM systems across three levels — model intelligence, agent cognition, and system dynamics. Stronger models, authority-driven role assignments, and dense communication topologies all suppress semantic diversity, with the root cause residing in interaction structure rather than insufficient model capability.

## Background & Motivation

**State of the Field**: Multi-agent systems (MAS) are increasingly employed for open-ended creative generation tasks such as research hypothesis generation, strategic planning, and creative design, under the expectation that collective interaction among multiple agents broadens the exploration space. MAS frameworks typically assign different roles or perspectives to individual agents, anticipating that their collision of viewpoints will yield diverse outputs.

**Limitations of Prior Work**: (1) Whether MAS genuinely produces greater diversity than single-model generation has never been systematically validated; (2) existing MAS frameworks are typically built on homogeneous underlying models that share pre-training distributions and alignment objectives, meaning multi-agent interaction may merely amplify shared priors rather than introduce genuine diversity; (3) the conditions under which MAS backfires — not expanding the solution space but instead causing premature convergence — remain unclear.

**Root Cause**: Intuitively, more interaction should yield more diverse outcomes; however, interaction itself may be the source of diversity loss. Increased collaboration leads to greater mutual influence and trajectory synchronization, ultimately triggering diversity collapse.

**Paper Goals**: To systematically diagnose diversity problems in MAS creative generation across three bottom-up levels: model level, cognitive level, and system level.

**Starting Point**: Research proposal generation is adopted as a standardized creative task, as it is both open-ended and structurally constrained, making it suitable for quantitative evaluation. The study employs a design of 20 topics × 50 independent discussions = 1,000 proposals per configuration.

**Core Idea**: Diversity collapse is a form of collective failure driven by structural coupling — interaction structures unintentionally contract agents' exploration space, independent of model capability.

## Method

### Overall Architecture

A general multi-agent interaction framework is constructed comprising three phases: role instantiation (assigning distinct personas to agents), iterative discussion (multi-round dialogue under specific topologies), and proposal synthesis (aggregating discussions into structured research proposals). Diversity is then analyzed across three levels: model intelligence (single-model diversity across different LLMs), agent cognition (effects of role and authority structures), and system dynamics (effects of group size, discussion rounds, and topology).

### Key Designs

1. **Multi-dimensional Diversity Measurement System**:

    - Function: Comprehensively quantify the semantic diversity of generated ideas.
    - Mechanism: Four complementary metrics are employed — Vendi Score (measuring the effective number of independent semantic patterns via kernel matrix spectral entropy), structural disorder $1-\phi$ (average cosine distance between individuals and the group mean, where low values indicate echo-chamber effects), semantic dispersion PCD (mean pairwise cosine distance), and lexical uniqueness (IDF-weighted n-gram statistics). Human evaluation validates that the Vendi Score achieves 87% agreement with human judgment.
    - Design Motivation: No single metric can fully capture diversity; a comprehensive assessment requires evaluation from four complementary perspectives: effective pattern count, distributional shape, pairwise distance, and surface redundancy.

2. **Three-Level Analysis Framework**:

    - Function: Diagnose the root cause of diversity collapse in a bottom-up manner.
    - Mechanism: **Model level** — identifies a "computational efficiency paradox" whereby stronger aligned models produce higher-quality outputs but exhibit diminishing marginal diversity. **Cognitive level** — compares five collaboration structures (naive / leader-driven / horizontal / interdisciplinary / vertical) and finds that authority-driven structures suppress diversity, while horizontal collaboration led by junior researchers achieves the highest diversity (Vendi 8.08 vs. interdisciplinary 4.65). **System level** — increasing group size yields diminishing returns (Vendi/N decreases from 1.03 to 0.47), and dense communication topologies accelerate premature convergence.
    - Design Motivation: Decomposing the complex dynamics of multi-agent systems into independently analyzable levels enables precise identification of the problem's origin.

3. **Topology Intervention Experiments (NGT / Subgroups)**:

    - Function: Validate whether process-level interventions can mitigate diversity collapse.
    - Mechanism: Standard discussion, Nominal Group Technique (NGT, in which agents independently "blind-write" before discussion), and subgroup topology (partitioning the social graph into local subgroups) are compared. NGT maximizes diversity in the initial phase, while subgroups maintain the highest constructive conflict density in later stages.
    - Design Motivation: If the root cause lies in interaction structure, modifying the interaction pattern should alleviate the collapse — a hypothesis confirmed by the experimental results.

## Key Experimental Results

### Main Results

| Cognitive Structure | Vendi Score | Semantic Dispersion | Structural Disorder | Overall Quality |
|---|---|---|---|---|
| Horizontal (Junior) | **8.08** | **0.31** | **0.170** | 7.88 |
| Vertical (Mixed) | 6.93 | 0.296 | 0.161 | 8.32 |
| Leader-driven | 6.08 | 0.285 | 0.154 | 8.03 |
| Naive | 5.57 | 0.272 | 0.146 | 7.95 |
| Interdisciplinary | 4.65 | 0.25 | 0.19 | **8.50** |

### Ablation Study

| Configuration | Vendi Score | Diversity Utilization | Note |
|---|---|---|---|
| N=3 agents | ~3.1 | 1.03 | Baseline, high efficiency |
| N=5 agents | ~3.8 | 0.76 | Diminishing returns begin |
| N=7 agents | ~3.3 | 0.47 | Severe diminishing returns |
| Standard topology | Low | — | Diversity continuously declines |
| NGT topology | Initially high | — | Effective during blind-writing phase |
| Subgroup topology | High in later stage | — | Maintains constructive conflict |

### Key Findings

- **Computational Efficiency Paradox**: Stronger aligned models (e.g., GPT-4.1) yield higher per-sample quality but lower diversity; alignment functions as a form of global semantic regularization that compresses the exploration space.
- **Authority Suppresses Diversity**: Horizontal collaboration led by junior researchers achieves 73% higher diversity than interdisciplinary expert groups (Vendi 8.08 vs. 4.65), while the quality gap is only 0.6 points (on a 10-point scale), indicating that authority induces a "sycophancy trap."
- **Ringelmann Effect in System Dynamics**: The marginal diversity gain from adding more agents decreases sharply, analogous to "social loafing" observed in human groups.
- **"Intra-Consensus Expansion" Pattern**: Diversity can increase locally within a single session as discussion deepens, yet diversity contracts across sessions due to structural convergence.

## Highlights & Insights

- **Theoretical Framework of "Structural Coupling"**: A unified explanation is proposed — diversity collapse stems not from insufficient model capability but from the interaction structure itself, which inherently contracts the exploration space. This insight carries important implications for all MAS designers.
- **Asymmetric Quality–Diversity Trade-off**: Interdisciplinary teams achieve the highest quality but the lowest diversity, demonstrating that optimizing for quality and optimizing for diversity are distinct objectives requiring explicit trade-off.
- **Experimental Scale and Rigor**: A comprehensive cross-factorial study involving 10,000+ proposals, 20 topics, and diverse topologies, cognitive structures, and models — validated through human annotation — provides a highly robust empirical foundation.
- **Subgroup Topology as a Diversity-Preserving Strategy**: Creating "local pockets of divergence" resists premature consensus and can be directly applied to real-world MAS design.

## Limitations & Future Work

- The study uses research proposal generation as the sole task; whether findings generalize to other open-ended tasks such as code generation and creative writing remains to be validated.
- All agents share the same underlying LLM; the effects of heterogeneous model ensembles are not fully explored.
- Evaluation relies on embedding-space semantic metrics, which may miss certain forms of conceptual innovation.
- The paper is lengthy (56 pages); core findings could be presented more concisely.
- No systematic solutions are proposed; the contribution is primarily diagnostic.

## Related Work & Insights

- **vs. Du et al. (2024) multi-agent debate**: The debate framework assumes that interaction improves reasoning; this paper demonstrates that in creative tasks, interaction can be counterproductive.
- **vs. Wang et al. (2025a) echo-chamber effect**: This paper extends the echo-chamber effect from social media to LLM multi-agent systems and provides quantitative analysis.
- **vs. Moon et al. (2025)**: Both address diversity in MAS, but this paper's three-level analysis is more systematic and the experimental scale is substantially larger.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic revelation of diversity collapse in MAS creative generation, with the introduction of the structural coupling theory.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10,000+ proposals, 20 topics, multi-dimensional cross-factorial analysis, human validation — exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Analysis is thorough and visualizations are excellent, but the paper is excessively long.
- Value: ⭐⭐⭐⭐⭐ Offers important guidance for MAS design; the finding that "more collaboration does not equal more diversity" carries broad impact.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[AAAI 2026\] Parallelism Meets Adaptiveness: Scalable Documents Understanding in Multi-Agent LLM Systems](../../AAAI2026/llm_agent/parallelism_meets_adaptiveness_scalable_documents_understanding_in_multi-agent_l.md)
- [\[ACL 2026\] SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](silo-bench_a_scalable_environment_for_evaluating_distributed_coordination_in_mul.md)
- [\[CVPR 2026\] SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation](../../CVPR2026/llm_agent/sceneassistant_a_visual_feedback_agent_for_openvoc.md)
- [\[AAAI 2026\] FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation](../../AAAI2026/llm_agent/finrpt_dataset_evaluation_system_and_llm-based_multi-agent_framework_for_equity_.md)

<!-- RELATED:END -->
