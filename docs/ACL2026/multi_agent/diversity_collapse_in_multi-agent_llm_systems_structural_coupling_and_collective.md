---
title: >-
  [Paper Note] Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation
description: >-
  [ACL 2026 Findings][Multi-Agent][Multi-agent systems] By evaluating over 10,000 research proposals, this paper systematically reveals the "diversity collapse" phenomenon in multi-agent LLM systems across three levels: mo…
tags:
  - "ACL 2026 Findings"
  - "Multi-Agent"
  - "Multi-agent systems"
  - "diversity collapse"
  - "structural coupling"
  - "idea generation"
  - "collaboration topology"
date: 2026-05-08
content_hash: 57fd6fc36bac2f94
---

# Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.18005](https://arxiv.org/abs/2604.18005)  
**Code**: [https://github.com/Xtra-Computing/MAS_Diversity](https://github.com/Xtra-Computing/MAS_Diversity)  
**Area**: LLM Agent  
**Keywords**: Multi-agent systems, diversity collapse, structural coupling, idea generation, collaboration topology

## TL;DR

By evaluating over 10,000 research proposals, this paper systematically reveals the "diversity collapse" phenomenon in multi-agent LLM systems across three levels: model intelligence, agent cognition, and system dynamics. It finds that stronger models, authority-driven role assignments, and dense communication topologies inhibit semantic diversity, with the root cause being the interaction structure rather than insufficient model capabilities.

## Background & Motivation

**Background**: Multi-agent systems (MAS) are increasingly utilized for open-ended idea generation (e.g., scientific hypothesis generation, strategic planning, creative design). The underlying expectation is that the collective interaction of multiple agents will broaden the exploration space. MAS frameworks typically assign different roles/perspectives to agents, expecting diverse ideas to emerge through collision.

**Limitations of Prior Work**: (1) Whether MAS truly generates more diversity than single models remains systematically unverified; (2) existing MAS frameworks are often based on homogeneous underlying models (sharing pre-training distributions and alignment objectives), suggesting that multi-agent interactions might simply amplify shared priors rather than introducing true diversity; (3) under what conditions does MAS "backfire"—leading to premature convergence instead of expanding the solution space?

**Key Challenge**: Intuitively, more interaction should produce more diverse results, but in practice, interaction itself may be the source of diversity loss. Increased collaboration leads to more mutual influence and trajectory synchronization, eventually triggering diversity collapse.

**Goal**: Systematically diagnose diversity issues in MAS idea generation from three bottom-up levels: model level, cognitive level, and system level.

**Key Insight**: Uses "scientific research proposal generation" as a standardized task for idea generation because it possesses both openness and structural constraints, making it suitable for quantitative evaluation. The study designed 20 topics × 50 independent discussions = 1,000 proposals/configurations.

**Core Idea**: Diversity collapse is a collective failure driven by "structural coupling"—the interaction structure unintentionally contracts the agents' exploration space, rather than a lack of model capability.

## Method

### Overall Architecture

A general multi-agent interaction framework is constructed, consisting of three stages: role instantiation (assigning different personas to agents), iterative discussion (multi-round dialogue under specific topologies), and proposal synthesis (summarizing discussions into structured research proposals). Diversity is then analyzed across three levels: model intelligence (single-model diversity of different LLMs), agent cognition (influence of different roles/authority structures), and system dynamics (influence of group size/rounds/topology).

### Key Designs

1. **Multi-dimensional Diversity Measurement System**:

    - **Function**: Comprehensively quantifies the semantic diversity of ideas.
    - **Mechanism**: Utilizes four complementary metrics: Vendi Score (measures the number of effectively independent semantic patterns based on kernel matrix spectral entropy), structural disorder $1-\phi$ (average cosine distance between individuals and the group mean; low values indicate echo chamber effects), semantic dispersion PCD (mean pairwise cosine distance), and lexical uniqueness (IDF-weighted n-gram statistics). Human evaluation verified that the Vendi Score aligns with human judgment at an 87% consistency rate.
    - **Design Motivation**: A single metric cannot fully capture diversity; comprehensive evaluation is required across effective pattern counts, distribution shapes, pairwise distances, and surface redundancy.

2. **Three-level Analysis Framework**:

    - **Function**: Diagnoses the root causes of diversity collapse bottom-up.
    - **Mechanism**: **Model Level**—discovers the "computational efficiency paradox": stronger aligned models yield higher quality but diminishing marginal diversity. **Cognitive Level**—compares five collaboration structures (Naive/Leader-driven/Horizontal/Interdisciplinary/Vertical), finding that authority-driven structures suppress diversity, while horizontal collaboration led by junior researchers shows the highest diversity (Vendi 8.08 vs. Interdisciplinary 4.65). **System Level**—increasing group size brings diminishing returns (Vendi/N drops from 1.03 to 0.47), and dense communication topologies accelerate premature convergence.
    - **Design Motivation**: Decomposing complex multi-agent dynamics into independently analyzable levels facilitates precise identification of problem sources.

3. **Topology Intervention Experiments (NGT / Subgroups)**:

    - **Function**: Verifies whether process interventions can mitigate diversity collapse.
    - **Mechanism**: Compares standard discussion, Nominal Group Technique (NGT, independent "blind writing" before discussion), and subgroup topologies (dividing the social graph into local subgroups). NGT maximizes diversity in the initial stage, while subgroups maintain the highest density of constructive conflict in later stages.
    - **Design Motivation**: If the root cause lies in the interaction structure, changing the interaction method should mitigate collapse—as confirmed by experimental results.

## Key Experimental Results

### Main Results

| Cognitive Structure | Vendi Score | Semantic Dispersion | Structural Disorder | Overall Quality |
|----------|------------|-----------|-----------|---------|
| Horizontal (Junior) | **8.08** | **0.31** | **0.170** | 7.88 |
| Vertical (Mixed) | 6.93 | 0.296 | 0.161 | 8.32 |
| Leader-driven | 6.08 | 0.285 | 0.154 | 8.03 |
| Naive | 5.57 | 0.272 | 0.146 | 7.95 |
| Interdisciplinary | 4.65 | 0.25 | 0.19 | **8.50** |

### Ablation Study

| Configuration | Vendi Score | Diversity Utilization | Description |
|------|-------------|------------|------|
| N=3 agents | ~3.1 | 1.03 | Baseline, high efficiency |
| N=5 agents | ~3.8 | 0.76 | Diminishing returns begin |
| N=7 agents | ~3.3 | 0.47 | Severe diminishing returns |
| Standard Topology | Low | - | Continuous decline in diversity |
| NGT Topology | High (Initial) | - | Effective during blind writing |
| Subgroup Topology | High (Late) | - | Maintains constructive conflict |

### Key Findings

- **Computational Efficiency Paradox**: Stronger aligned models (e.g., GPT-5.1) exhibit higher single-sample quality but lower diversity; alignment acts as a global semantic regularization that compresses the exploration space.
- **Authority Suppresses Diversity**: Horizontal collaboration led by junior researchers is 73% more diverse than interdisciplinary expert groups (Vendi 8.08 vs. 4.65), while the quality gap is only 0.6 points (on a 10-point scale), indicating that authority leads to a "sycophancy trap."
- **Ringelmann Effect in System Dynamics**: The marginal diversity gain from increasing the number of agents drops sharply, similar to "social loafing" in human groups.
- **"Expansion within Consensus" Mode**: Diversity can increase locally within a single session (deepening of discussion), but cross-session diversity contracts (structural convergence).

## Highlights & Insights

- **"Structural Coupling" Theoretical Framework**: Proposes a unified explanation—diversity collapse is not caused by weak models, but because the interaction structure itself contracts the exploration space. This insight serves as a warning to all MAS designers.
- **Asymmetric Quality-Diversity Relationship**: Interdisciplinary teams exhibit the highest quality but lowest diversity, indicating that optimizing for quality and optimizing for diversity are distinct goals requiring explicit trade-offs.
- **Experimental Scale and Rigor**: A solid empirical foundation including 10,000+ proposals, 20 topics, and comprehensive cross-experiments on topologies/cognitive structures/models, all validated by human assessment.
- **Subgroup Topology as a Diversity Preservation Strategy**: Resists premature consensus by creating "local pockets of disagreement," which can be directly applied to real-world MAS design.

## Limitations & Future Work

- The task is limited to "scientific research proposal generation"; whether conclusions generalize to other open-ended tasks like code generation or creative writing remains to be verified.
- All agents share the same underlying LLM; the effects of heterogeneous model ensembles are not fully explored.
- Evaluation depends on semantic metrics in embedding spaces, which may miss certain types of conceptual innovation.
- The paper is lengthy (56 pages); core findings could be presented more concisely.
- A systematic solution was not proposed; the work is primarily diagnostic.

## Related Work & Insights

- **vs. Du et al. (2024) Multi-agent Debate**: While the debate framework assumes interaction improves reasoning, Ours proves that interaction can be counterproductive in creative tasks.
- **vs. Wang et al. (2025a) Echo Chamber Effect**: Ours extends the echo chamber effect from social media to LLM multi-agent systems and provides quantitative analysis.
- **vs. Moon et al. (2025)**: Also focuses on diversity issues in MAS, but the three-level analysis in Ours is more systematic and the experimental scale is larger.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to systematically reveal diversity collapse in MAS idea generation and propose the "structural coupling" theory.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10,000+ proposals, 20 topics, multi-dimensional cross-analysis, and human validation make it extremely thorough.
- Writing Quality: ⭐⭐⭐⭐ Deep analysis and excellent visualization, though the length is quite extensive.
- Value: ⭐⭐⭐⭐⭐ Significant guidance for MAS design; the conclusion that "more collaboration does not equal more diversity" has broad impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems](seeing_the_whole_elephant_a_benchmark_for_failure_attribution_in_llm-based_multi.md)
- [\[ICML 2026\] EngiAgent: Fully Connected Coordination of LLM Agents for Solving Open-ended Engineering Problems with Feasible Solutions](../../ICML2026/multi_agent/engiagent_fully_connected_coordination_of_llm_agents_for_solving_open-ended_engi.md)
- [\[ACL 2026\] Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](memory-augmented_llm-based_multi-agent_system_for_automated_feature_generation_o.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems](cia_inferring_the_communication_topology_from_llm-based_multi-agent_systems.md)

</div>

<!-- RELATED:END -->
