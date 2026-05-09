---
title: >-
  [Paper Note] Principles2Plan: LLM-Guided System for Operationalising Ethical Principles into Plans
description: >-
  [AAAI 2026][LLM Safety][Ethical Planning] This paper presents Principles2Plan, an interactive prototype system that enables collaborative human–LLM operationalisation of high-level ethical principles (e.g., beneficence, privacy) into context-sensitive ethical rules, which are then embedded into a PDDL planner to generate ethically compliant action plans.
tags:
  - AAAI 2026
  - LLM Safety
  - Ethical Planning
  - LLM
  - Automated Planning
  - PDDL
  - Human-AI Collaboration
date: 2026-05-08
content_hash: 1209a2afc5440da6
---

# Principles2Plan: LLM-Guided System for Operationalising Ethical Principles into Plans

**Conference**: AAAI 2026
**arXiv**: [2512.08536](https://arxiv.org/abs/2512.08536)
**Code**: None
**Area**: AI Safety
**Keywords**: Ethical Planning, LLM, Automated Planning, PDDL, Human-AI Collaboration

## TL;DR

This paper presents Principles2Plan, an interactive prototype system that enables collaborative human–LLM operationalisation of high-level ethical principles (e.g., beneficence, privacy) into context-sensitive ethical rules, which are then embedded into a PDDL planner to generate ethically compliant action plans.

## Background & Motivation

As robots and autonomous systems are increasingly deployed in human living environments, ensuring that their behaviors both achieve intended goals and respect ethical principles has become a critical challenge. High-level ethical principles (e.g., beneficence, privacy) are inherently **abstract and highly context-dependent**. For example:

- **Autonomous driving scenario**: When a passenger requires urgent medical attention, taking an unauthorized shortcut may be justifiable (beneficence takes precedence over traffic regulations).
- **Leisure travel scenario**: Under the same beneficence principle, adhering to standard traffic rules is more appropriate.

This characteristic—whereby **the same principle leads to different behaviors under different contexts**—makes fully automated ethical planning extremely difficult.

Limitations of existing Computational Machine Ethics (CME) approaches:

**Top-down methods**: Rules are pre-specified; transparent but lacking adaptability, with significant manual encoding overhead.

**Bottom-up methods**: Ethical behavior is inferred from data; flexible but lacking interpretability.

**Hybrid methods**: Still require substantial manual encoding of ethical rules.

**Core Motivation**: The emergence of LLMs offers the possibility of reducing the manual effort required to encode ethical rules. The key question is whether LLMs' comprehension and generation capabilities can be leveraged—under human oversight—to automatically produce context-sensitive ethical rules.

## Method

### Overall Architecture

Principles2Plan adopts a **four-step interactive workflow** that guides users from input to the generation of ethical plans:

```
Input Page → Ethical Rules Editor → Code Editor → Output Plan Page
```

The system targets users including: domain experts in ethically sensitive fields, AI ethics researchers, and practitioners interested in the intersection of ethics, LLMs, and planning.

### Key Designs

#### 1. **Input Page**

Users provide the following key information to initiate ethical planning:

- **PDDL Files**: Upload `problem.pddl` and `domain.pddl` to define the planning problem and domain.
- **Initial State & Assumptions**: Contextual background information about the problem or domain.
- **High-Level Ethical Principles**: Such as beneficence and privacy.
- **Model Selection**: Users may choose which LLM to employ.

The system provides example problems from three ethically sensitive domains—autonomous driving, elderly care, and firefighting rescue—which can be loaded directly to populate the input fields.

The LLM generates context-sensitive ethical rules in real time based on all provided inputs. Each rule includes **ethical features** representing positive or negative ethical characteristics (e.g., "dishonesty" is a negative feature).

#### 2. **Ethical Rules Editor**

Since LLM-generated rules may be inconsistent or incomplete, the system provides a human-in-the-loop review mechanism:

- **Add, Delete, Modify**: Users may add missing rules, remove inappropriate ones, and revise existing rules.
- **LLM Explanations**: The system provides the LLM's reasoning for each rule, explaining why it was generated given the current problem and principles.
- **Priority Assignment**: Users assign importance levels from 1 to 5 to each ethical feature.
- **Positive/Negative Feature Highlighting**: The system highlights positive and negative ethical features to facilitate rapid identification.

This step embodies the **human-in-the-loop** design philosophy, ensuring that the final rules benefit from the LLM's generative capacity while remaining subject to expert review and correction.

#### 3. **Code Editor**

User-confirmed ethical rules are automatically translated by the LLM into **PDDL-Ethical code** (a PDDL extension that supports ethical constructs). Users may:

- Review syntax-highlighted code.
- Cross-check the code against the ethical rules from the previous step.
- Verify consistency between rules and generated code.

The code is subsequently converted via a **transpiler** (based on the method of jedwabny et al. 2022) into standard PDDL with action costs, which is then submitted to a domain-independent classical planner (Fast Downward).

#### 4. **Output Plan Page**

The system presents two plans side by side:

- **Plan with Ethical Rules**: The plan generated under ethical constraints.
- **Original Plan**: The plan generated from the original PDDL files without ethical constraints.

This **comparative display** enables users to intuitively assess the impact of ethical rules on planning outcomes.

### Loss & Training

This paper is a system demonstration paper and does not involve model training. The core technical stack consists of:

- **LLM**: DeepSeek-R1-Distill-Llama-70B as the backend.
- **Planner**: Fast Downward (domain-independent classical planner).
- **PDDL-Ethical**: An ethical extension of the planning domain-specific language.
- **Evaluation Metrics**: Sentence-BERT similarity (0.82); code generation success rate (82.2%).

## Key Experimental Results

### Main Results

As a system demonstration paper, no traditional comparative experiments are conducted. Core performance metrics are drawn from evaluations of the underlying methodology:

| Metric | Value | Description |
|---|---|---|
| Sentence-BERT Similarity | 0.82 | Semantic similarity between LLM-generated rules and reference rules |
| Code Generation Success Rate | 82.2% | Correctness rate of generated PDDL-Ethical code |
| Supported Domains | 3 | Autonomous driving, elderly care, firefighting rescue |

### Ablation Study

| System Component | Function | Necessity |
|---|---|---|
| LLM Rule Generation | Automatically generates context-sensitive ethical rules | Core (reduces manual encoding) |
| Human Review & Editing | Corrects inconsistencies and errors in LLM output | Core (ensures quality) |
| Priority Assignment | Differentiates the importance of ethical features | Critical (influences plan selection) |
| Code Cross-Checking | Ensures consistency from rules to code | Important (reduces transpilation errors) |

### Key Findings

1. **LLMs show promise but are imperfect for ethical rule generation**: A semantic similarity of 0.82 indicates that LLMs can capture most ethical intent, but human review remains necessary.
2. **Context sensitivity is the core challenge**: The same ethical principle requires substantially different rules under different contexts.
3. **Code generation success rate has room for improvement**: An 82.2% success rate implies that nearly one in five cases requires manual code repair.

## Highlights & Insights

- **First complete system supporting principle-to-plan operationalisation**: The system realises a full pipeline from abstract ethical principles to executable planning constraints via LLM, bridging the gap between abstraction and implementation.
- **Transparency by design**: Each rule is accompanied by the LLM's reasoning explanation; the mapping from rules to code is auditable; and the final plan can be compared against an ethics-unconstrained baseline.
- **Practical usability**: Example problems from three domains are provided, enabling users to experience the system immediately.
- **Pioneering direction**: This represents the first realisation of human–LLM collaborative systematic ethical rule generation within the planning community.

## Limitations & Future Work

1. **Requires PDDL expertise**: Current users must understand both PDDL and PDDL-Ethical, which limits accessibility for non-technical users.
2. **Limited LLM generation quality**: An 82.2% code success rate is insufficient, and the current version may require considerable manual intervention.
3. **Lack of large-scale user studies**: The system has not been evaluated with respect to real-user experience and acceptance.
4. **Insufficient handling of ethical conflicts**: When multiple ethical principles conflict (e.g., beneficence vs. privacy), the system relies solely on simple priority ordering, lacking deeper conflict resolution mechanisms.
5. **Restricted to classical planning**: Uncertain planning and multi-agent scenarios are not supported.

## Related Work & Insights

- **The CME tripartite taxonomy (top-down / bottom-up / hybrid)** provides a valuable classification framework for ethical AI research.
- **PDDL-Ethical**, as an ethical extension of classical planning, represents an elegant technical approach to encoding ethical preferences via action costs.
- **The intersection of LLMs and planning** is rapidly evolving, from directly using LLMs to generate plans to using LLMs to assist the planning process (e.g., constructing models, translating constraints).
- Future work could incorporate iterative dialogue, enabling the LLM to refine ethical rules based on feedback from planning outcomes.

## Rating

- Novelty: ⭐⭐⭐⭐ (First complete human–AI collaborative system for operationalising ethical principles into automated planning)
- Experimental Thoroughness: ⭐⭐⭐ (System demonstration paper; lacks comprehensive quantitative evaluation and user studies)
- Writing Quality: ⭐⭐⭐⭐ (System workflow is clearly described with well-motivated design choices)
- Value: ⭐⭐⭐⭐ (Pioneering work providing a practical tool for ethical automated planning)

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization](psm_prompt_sensitivity_minimization_via_llm-guided_black-box_optimization.md)
- [\[AAAI 2026\] SproutBench: A Benchmark for Safe and Ethical Large Language Models for Youth](sproutbench_a_benchmark_for_safe_and_ethical_large_language_models_for_youth.md)
- [\[AAAI 2026\] Uncovering Bias Paths with LLM-guided Causal Discovery: An Active Learning and Dynamic Scoring Approach](uncovering_bias_paths_with_llm-guided_causal_discovery_an_active_learning_and_dy.md)
- [\[AAAI 2026\] Perturb Your Data: Paraphrase-Guided Training Data Watermarking](perturb_your_data_paraphrase-guided_training_data_watermarking.md)
- [\[AAAI 2026\] Democratizing LLM Efficiency: From Hyperscale Optimizations to Universal Deployability](democratizing_llm_efficiency_from_hyperscale_optimizations_to_universal_deployab.md)

<!-- RELATED:END -->
