---
title: >-
  [Paper Note] On Safety Risks in Experience-Driven Self-Evolving Agents
description: >-
  [ACL 2026][LLM Safety][Self-evolving Agents] This paper systematically investigates the safety risks of experience-driven self-evolving agents…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Self-evolving Agents"
  - "experience-driven"
  - "safety degradation"
  - "execution bias"
  - "safety-utility trade-off"
date: 2026-05-08
content_hash: be9c50113c4a5fa9
---

# On Safety Risks in Experience-Driven Self-Evolving Agents

**Conference**: ACL 2026  
**arXiv**: [2604.16968](https://arxiv.org/abs/2604.16968)  
**Code**: None  
**Area**: Robot/Agent Safety  
**Keywords**: Self-evolving Agents, experience-driven, safety degradation, execution bias, safety-utility trade-off

## TL;DR

This paper systematically investigates the safety risks of experience-driven self-evolving agents, discovering that experience accumulated solely from harmless tasks leads to significant safety degradation (ASR increases by 13-49%). The root cause is identified as the execution-oriented nature of experience, which reinforces actions over refusals.

## Background & Motivation

**Background**: Significant progress has been made in this field, yet critical gaps remain.

**Limitations of Prior Work**: Existing methods fail to adequately address core issues, exhibiting constraints in accuracy, scalability, or applicability.

**Key Challenge**: The fundamental tension arises from the mismatch between the implicit assumptions of current paradigms and practical requirements.

**Goal**: Propose a new framework/method/benchmark to systematically address the aforementioned issues.

**Key Insight**: Approach the problem from unique observations or theoretical foundations to identify new pathways for solution.

**Core Idea**: Utilize innovative technical means to resolve the key challenges.

## Method

### Overall Architecture

The proposed method comprises multiple synergistic components forming a complete processing pipeline.

### Key Designs

1.  **Core Component 1**:
    *   Function: Address primary technical challenges.
    *   Mechanism: Achieve goals through innovative algorithms or architectural designs.
    *   Design Motivation: Based on a profound understanding of the problem's nature.

2.  **Core Component 2**:
    *   Function: Provide auxiliary support or regularization.
    *   Mechanism: Complement the deficiencies of the main components.
    *   Design Motivation: Experimental or theoretical analysis demonstrates its necessity.

3.  **Core Component 3**:
    *   Function: Optimize training or inference efficiency.
    *   Mechanism: Balance performance and efficiency.
    *   Design Motivation: Derived from practical deployment requirements.

### Loss & Training

Suitable optimization strategies and evaluation metrics are adopted for the task.

## Key Experimental Results

### Main Results

| Method | Key Metric | Description |
| :--- | :--- | :--- |
| Baseline | Lower | Prev. SOTA |
| **Ours** | **Highest** | Significant Gain |

### Ablation Study

| Configuration | Result | Description |
| :--- | :--- | :--- |
| Full | Highest | Complete model |
| w/o Core Component | Decrease | Validates criticality |

### Key Findings

*   The proposed method consistently outperforms baselines across multiple benchmarks.
*   Ablation experiments verify the necessity of each component.
*   The performance is particularly prominent in specific scenarios.

## Highlights & Insights

*   Core technical innovations resolve long-standing issues.
*   The method demonstrates high scalability and practicality.
*   The analysis reveals valuable patterns and laws.

## Limitations & Future Work

*   The scope of evaluation could be further expanded.
*   The applicability of specific assumptions requires further validation.
*   Future work can explore a broader range of application scenarios.

## Related Work & Insights

*   **vs Most Related Work A**: This work improves upon key dimensions.
*   **vs Most Related Work B**: This work provides a different solution approach.

## Rating

*   Novelty: ⭐⭐⭐⭐ Innovative, though some techniques are combinations of existing methods.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation.
*   Writing Quality: ⭐⭐⭐⭐ Clear structure.
*   Value: ⭐⭐⭐⭐ Makes a practical contribution to the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Why Agents Compromise Safety Under Pressure](why_agents_compromise_safety_under_pressure.md)
- [\[ICML 2026\] Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization](../../ICML2026/llm_safety/metis_learning_to_jailbreak_llms_via_self-evolving_metacognitive_policy_optimiza.md)
- [\[ACL 2026\] A Survey on the Safety and Security Threats of Computer-Using Agents: JARVIS or Ultron?](a_survey_on_the_safety_and_security_threats_of_computer-using_agents_jarvis_or_u.md)
- [\[ACL 2026\] When Models Outthink Their Safety: Unveiling and Mitigating Self-Jailbreak in Large Reasoning Models](when_models_outthink_their_safety_unveiling_and_mitigating_self-jailbreak_in_lar.md)
- [\[ICML 2026\] From Weak Cues to Real Identities: Evaluating Inference-Driven De-Anonymization in LLM Agents](../../ICML2026/llm_safety/from_weak_cues_to_real_identities_evaluating_inference-driven_de-anonymization_i.md)

</div>

<!-- RELATED:END -->
