---
title: >-
  [Paper Note] SEED-SET: Scalable Evolving Experimental Design for System-level Ethical Testing
description: >-
   This paper proposes SEED-SET, a framework that formulates ethical evaluation of autonomous systems as a hierarchical Bayesian experimental design problem, jointly integrating objective metrics and subjective value judgments to efficiently generate test cases with high ethical alignment under limited evaluation budgets.
tags:

date: 2026-05-08
content_hash: eb5a16dd6330ce2d
---

# SEED-SET: Scalable Evolving Experimental Design for System-level Ethical Testing

## Basic Information

- **Conference**: ICLR 2026
- **arXiv**: [2603.01630](https://arxiv.org/abs/2603.01630)
- **Code**: [Project Page](https://anjaliparashar.github.io/seed-site/)
- **Area**: AI Safety / Autonomous System Evaluation
- **Keywords**: Ethical Testing, Bayesian Experimental Design, Gaussian Process, LLM Evaluator, Autonomous Systems

## TL;DR

This paper proposes SEED-SET, a framework that formulates ethical evaluation of autonomous systems as a hierarchical Bayesian experimental design problem, jointly integrating objective metrics and subjective value judgments to efficiently generate test cases with high ethical alignment under limited evaluation budgets.

## Background & Motivation

### State of the Field
Autonomous systems (e.g., drones, power grid distribution) are increasingly deployed in high-stakes domains, making ethical alignment evaluation critically important. Ethical assessment faces three major challenges:

**Measurement difficulty**: Ethical behaviors (fairness, social acceptability) lack ground-truth labels;

**Subjectivity dependence**: Value alignment varies across stakeholders and evolves over time, requiring continuous revision of static benchmarks;

**Evaluation cost**: Real-system evaluation is budget-constrained, making large-scale human feedback collection infeasible.

### Limitations of Prior Work
- Rule-based ethical benchmarks rely on predefined criteria and lack specificity;
- RL/RLHF-based methods assume abundant simulation or expert annotations, requiring large sample sizes;
- Preference-based methods and large-scale human studies focus on only a single dimension.

### Mechanism
The framework jointly models **objective metrics** (e.g., fire damage, grid cost) and **subjective preferences** (stakeholder ethical judgments), efficiently generating test scenarios via hierarchical Gaussian processes and Bayesian experimental design.

## Method

### Overall Architecture

SEED-SET (Scalable Evolving Experimental Design for System-level Ethical Testing) comprises three components:

1. **Hierarchical Variational Gaussian Process (HVGP)** as the surrogate model
2. **Joint acquisition strategy** for adaptive test case generation
3. **LLM agent** as a substitute for human preference evaluation

### 1. Problem Formulation

Given a black-box autonomous system $\mathcal{S}_\pi$ and scenario space $\mathcal{X}$, the ethical compliance function is decomposed into:

- **Objective layer**: $f_{\text{obj}}: \mathcal{X} \to \mathcal{Y}$, mapping scenario parameters to measurable metrics (cost, resilience, etc.)
- **Subjective layer**: $f_{\text{subj}}: \mathcal{Y} \to \mathbb{R}$, producing an ethical utility score from objective metrics

### 2. Hierarchical Variational Gaussian Process (HVGP)

Ethical evaluation is modeled as a two-level VGP hierarchy:

**Objective GP**: Learns a surrogate model $g: x \to y$ predicting objective metrics for a given scenario
$$
p(f(x)|\mathcal{D}) = \mathcal{N}(\mu(x), k(x, x'))
$$

**Subjective GP**: Learns a preference model $h: y \to z$ mapping objective metrics to subjective ethical scores

Since subjective evaluations lack ground-truth labels, **pairwise preference elicitation** is adopted: oracle $\mathcal{T}: (y, y') \to \{1, 2\}$ compares the ethical quality of two scenarios.

The hierarchical structure offers **two key advantages**:
- **Interpretability**: Ethical preferences are anchored to observable system behaviors
- **Data efficiency**: Exploiting the dependence of subjective on objective layers reduces the number of required evaluations

### 3. Joint Acquisition Strategy

The core innovation—an acquisition function that simultaneously balances objective exploration and subjective exploitation:

$$
V(x) = \underbrace{I(g_x; y|\mathcal{D})}_{\text{Objective information gain}} + \mathbb{E}_{q_\phi(y|x)}\left[\underbrace{I(h_y; z|\mathcal{D})}_{\text{Subjective information gain}} + \underbrace{\mathbb{E}_{q_\psi(h_y)}[h_y]}_{\text{Preference exploitation}}\right]
$$

Roles of the three terms:
- **First term**: Reduces uncertainty in the objective metric space (scenario exploration)
- **Second term**: Improves estimation of the subjective utility function (preference learning)
- **Third term**: Directs search toward regions of high ethical utility (preference exploitation)

### 4. LLM Proxy Evaluator

GPT-4o is used as a stakeholder proxy for pairwise preference evaluation. The prompt includes:

1. **Task description**: Domain-specific context
2. **Objective metrics**: Measurable outcomes for both scenarios
3. **Subjective criteria**: Ethical preferences encoded in natural language

## Key Experimental Results

### Case 1: Power Grid Resource Allocation (IEEE 5/30-Bus)

| Method | 5-Bus Preference Score (↑) | 30-Bus Preference Score (↑) |
|--------|---------------------------|----------------------------|
| Random | Low | Low |
| Single GP | Moderate | Fails |
| VS-AL-1 | Fails | Fails |
| VS-AL-2 | Fails | Fails |
| **HVGP (SEED-SET)** | **Highest** | **Highest** |

### Case 2: Firefighting Rescue (Drone Navigation)

| Method | Preference Score (↑) | Coverage (↑) |
|--------|---------------------|-------------|
| Random | Low | Low |
| Single GP | Moderate | Moderate |
| HVGP (MI1+MI2 exploration only) | Moderate–High | Moderate–High |
| HVGP (Pref exploitation only) | High | Moderate |
| **HVGP (Full acquisition)** | **Highest** | **Highest** |

### Ablation Study: Acquisition Strategy Components

| Acquisition Strategy | Optimal Test Case Rate (↑) | Spatial Coverage (↑) |
|---------------------|--------------------------|---------------------|
| Random sampling | 1× | 1× |
| MI1+MI2 only | 1.4× | 1.1× |
| Pref only | 1.6× | 0.9× |
| **Full $V(x)$** | **2×** | **1.25×** |

### Key Findings

1. **SEED-SET generates twice as many optimal test cases as baselines**, with a 1.25× improvement in search space coverage;
2. **Significant advantage in high-dimensional scenarios**: Single GP fails entirely on the 30-Bus case (40-dimensional design space), while HVGP remains effective;
3. **Hierarchical modeling is essential**: Decomposing $f$ into $f_{\text{obj}} + f_{\text{subj}}$ yields more accurate estimates than directly modeling $f(x) \to z$;
4. **All three acquisition terms are necessary**: Removing any single term degrades performance;
5. **LLM proxy is reliable**: TrueSkill ratings confirm that GPT-4o preference judgments align with trends from handcrafted preference functions;
6. **Adaptable to different stakeholders**: Switching subjective criteria in the prompt enables rapid adaptation to different ethical standards.

## Highlights & Insights

- First ethical testing framework for autonomous systems that jointly considers objective metrics and subjective value judgments
- The hierarchical HVGP design anchors subjective preferences to observable behaviors, enhancing interpretability
- The joint acquisition strategy elegantly balances exploration and exploitation, with each of the three terms serving a distinct purpose
- Using an LLM as a proxy evaluator reduces dependence on human experts
- The framework is domain-agnostic, applicable to power grids, firefighting, transportation, and other scenarios

## Limitations & Future Work

- Assumes stakeholders report preferences truthfully (Assumption A2); strategic misreporting is not addressed
- Assumes the set of objective metrics is fully known and fixed (Assumption A3); dynamic metric expansion is not considered
- The LLM proxy may inherit GPT-4o's biases; preference consistency across different LLMs requires further validation
- Scalability of VGP in extremely high-dimensional settings remains constrained by the number of inducing points
- Handcrafted preference scoring functions rely on domain expertise

## Related Work & Insights

- **AI ethics frameworks**: NIST AI RMF 1.0 (2023), IEEE standards
- **Bayesian experimental design**: Rainforth et al. (2024), Chaloner & Verdinelli (1995)
- **Preference learning**: RLHF (Christiano et al., 2017), pairwise comparison GP (Chu & Ghahramani, 2005)
- **Active learning**: Preference elicitation (Keswani et al., 2024)
- **LLM evaluators**: Huang et al. (2025) use LLMs for preference evaluation

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First application of hierarchical Bayesian experimental design to ethical testing
- Technical depth: ⭐⭐⭐⭐ — HVGP + joint acquisition + LLM evaluator as an integrated system
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three case studies + multi-dimensional ablation + stakeholder analysis
- Value: ⭐⭐⭐⭐ — Domain-agnostic framework, though real-world deployment requires validation with actual stakeholders

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Time-Evolving Dynamical System for Learning Latent Representations of Mouse Visual Cortex](../../NeurIPS2025/interpretability/time-evolving_dynamical_system_for_learning_latent_representations_of_mouse_visu.md)
- [\[ICLR 2026\] MATA: A Trainable Hierarchical Automaton System for Multi-Agent Visual Reasoning](mata_a_trainable_hierarchical_automaton_system_for_multi-agent_visual_reasoning.md)
- [\[ACL 2026\] StructKV: Preserving the Structural Skeleton for Scalable Long-Context Inference](../../ACL2026/interpretability/structkv_preserving_the_structural_skeleton_for_scalable_long-context_inference.md)
- [\[ACL 2026\] Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures](../../ACL2026/interpretability/towards_intrinsic_interpretability_of_large_language_modelsa_survey_of_design_pr.md)
- [\[CVPR 2026\] TDATR: Improving End-to-End Table Recognition via Table Detail-Aware Learning and Cell-Level Visual Alignment](../../CVPR2026/interpretability/tdatr_improving_end-to-end_table_recognition_via_table_detail-aware_learning_and.md)

<!-- RELATED:END -->
