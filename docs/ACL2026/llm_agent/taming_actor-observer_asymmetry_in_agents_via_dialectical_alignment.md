---
title: >-
  [Paper Note] Taming Actor-Observer Asymmetry in Agents via Dialectical Alignment
description: >-
  [ACL 2026][LLM Agent][Actor-Observer Asymmetry] The study identifies that LLM Agents exhibit human-like "Actor-Observer Asymmetry" (AOA) cognitive bias during role-playing—tending to attribute failures to external factor…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Actor-Observer Asymmetry"
  - "Attribution Bias"
  - "Dialectical Alignment"
  - "Multi-Agent Collaboration"
  - "Self-Reflection"
date: 2026-05-08
content_hash: 4c4be4f068220630
---

# Taming Actor-Observer Asymmetry in Agents via Dialectical Alignment

**Conference**: ACL 2026  
**arXiv**: [2604.19548](https://arxiv.org/abs/2604.19548)  
**Code**: [https://unikcc.github.io/ReTAS/](https://unikcc.github.io/ReTAS/)  
**Area**: LLM Reasoning  
**Keywords**: Actor-Observer Asymmetry, Attribution Bias, Dialectical Alignment, Multi-Agent Collaboration, Self-Reflection

## TL;DR

The study identifies that LLM Agents exhibit human-like "Actor-Observer Asymmetry" (AOA) cognitive bias during role-playing—tending to attribute failures to external factors when acting as an "actor" and to internal errors when acting as an "observer." The proposed ReTAS addresses this via dialectical reasoning (Thesis-Antithesis-Synthesis) and GRPO alignment to eliminate this bias.

## Background & Motivation

**Background**: Multi-agent LLM frameworks assign specialized capabilities (e.g., executor, reviewer) through role-playing, utilizing self-reflection and mutual auditing to enhance reliability. However, role assignment serves not only as a functional specification but also as a cognitive prior that shapes reasoning.

**Limitations of Prior Work**: When an Agent acts as an "Actor" during self-reflection and faces failure, it tends to attribute the cause to external factors (e.g., server issues). Conversely, as an "Observer" auditing others, it tends to attribute failure to internal errors (e.g., logic errors). This contradictory attribution prevents consensus between Agents and undermines collaborative reliability.

**Key Challenge**: While role-playing is a foundational design for multi-agent systems, the resulting cognitive bias is a significant side effect. Simply instructing Agents to "remain objective" is ineffective due to role inertia leading to defensive justification, while forcing opposing perspectives can lead to over-correction and unwarranted self-blame.

**Goal**: To quantify AOA bias in LLMs and design a structured reasoning method to eliminate this perspective-dependent attribution inconsistency.

**Key Insight**: Drawing from Fichtean dialectics (Thesis $\rightarrow$ Antithesis $\rightarrow$ Synthesis), robust attribution requires first expressing a stance, then encountering negation, and finally synthesizing these into a unified truth.

**Core Idea**: The ReTAS model is trained to decompose reflection into three explicit stages: Thesis (role-consistent explanation), Antithesis (simulated opposing perspective to expose blind spots), and Synthesis (reconciling conflicting views to reach a perspective-invariant conclusion), aligned using GRPO with attribution rewards.

## Method

### Overall Architecture

The framework consists of three steps: (1) Constructing the Ambiguous Failure Benchmark (AFB) to quantify AOA bias; (2) Dialectical Synthesis—generating Thesis-Antithesis-Synthesis reasoning trajectories for each attribution scenario; (3) Dialectical Alignment—training the ReTAS model using GRPO with attribution consistency rewards.

### Key Designs

1.  **Ambiguous Failure Benchmark (AFB)**:
    - **Function**: Precisely quantifies the degree of AOA bias in LLMs.
    - **Mechanism**: Constructs 200 inherently ambiguous failure scenarios where a single failure signal reasonably supports contradictory root causes (e.g., a timeout could result from infrastructure latency or aggressive configuration). Using paired counterfactual probes, the model is queried for the same scenario under actor and observer system prompts, forcing a binary choice (internal/external), and the attribution flip rate is calculated. Results show most models exhibit AOA flips in over 20% of scenarios.
    - **Design Motivation**: Scenarios without deterministic root causes are required; if a right or wrong answer were clear, attribution differences would reflect capability gaps rather than bias.

2.  **Dialectical Reasoning (Thesis-Antithesis-Synthesis)**:
    - **Function**: Eliminates perspective dependency through structured reasoning.
    - **Mechanism**: **Thesis**—generates a role-consistent explanation expressing specific expertise; **Antithesis**—simulates an opposing perspective to expose blind spots and counter-evidence in the current attribution; **Synthesis**—reconciles conflicting views to derive a perspective-invariant conclusion based on objective evidence rather than role priors. These three stages serve as an explicit structured framework for CoT.
    - **Design Motivation**: Naive instructions to "remain objective" fail due to role inertia; a mandatory structure is needed to ensure multiple perspectives are considered.

3.  **GRPO Dialectical Alignment Training**:
    - **Function**: Internalizes dialectical reasoning capabilities into model parameters.
    - **Mechanism**: Training with attribution rewards—penalizing reasoning trajectories that yield inconsistent attributions between Actor and Observer perspectives, while rewarding trajectories that converge on the true root cause. Based on the GRPO framework, the model learns to generate perspective-invariant dialectical reasoning chains.
    - **Design Motivation**: Prompting alone is insufficient for stable dialectical reasoning; RL alignment internalizes this capability.

### Loss & Training

Optimization uses the GRPO framework with attribution consistency as the reward signal. Evaluations are conducted on the AFB benchmark and various downstream tasks. Models tested include the GPT-5 series, DeepSeek-V3.2, and Qwen3-4B.

## Key Experimental Results

### Main Results

| Model | Human-Agent AOA Flip Rate | Agent-Agent AOA Flip Rate |
| :--- | :--- | :--- |
| GPT-5.1 | 6% | 26% |
| GPT-5 | 23% | 33% |
| DeepSeek-V3.2 | 15% | 39% |
| Qwen3-4B | 33% | - |
| QwQ-32B | 21% | - |

### Ablation Study

| Configuration | Attribution Consistency | Task Performance | Description |
| :--- | :--- | :--- | :--- |
| Standard Role-Playing | Low | Baseline | Presence of AOA bias |
| + "Remain Objective" Instruction | Slight Improvement | No Change | Offset by role inertia |
| + Dialectical Prompting (No Training) | Moderate Improvement | Improved | Structured but unstable |
| ReTAS (Dialectical Alignment) | **Significant Improvement** | **Significant Improvement** | Internalized dialectical reasoning |

### Key Findings

- AOA bias is pervasive across all tested models, with Agent-Agent scenarios (39% flip rate for DeepSeek) being more severe than Human-Agent scenarios.
- ReTAS effectively reduces attribution inconsistency while significantly improving the failure resolution rate in ambiguous scenarios.
- The three-stage structure of dialectical reasoning is more effective than simple "multi-perspective thinking."
- The degree of bias is proportional to model capability—stronger models generally exhibit higher consistency (GPT-5.1 has the lowest flip rate at 6%).

## Highlights & Insights

- **Introduction of social psychology theory to AI Agent analysis**: AOA is systematically verified as a human-like cognitive bias in LLMs, which has significant implications for the reliability design of multi-agent systems.
- **Creative use of dialectics as a debiasing tool**: The Thesis-Antithesis-Synthesis structure is naturally suited for reconciling conflicting perspectives and is more operational than "remain objective" instructions.
- **Ingenious AFB benchmark design**: Deliberate construction of ambiguous scenarios without deterministic root causes ensures that systematic differences in attribution are identified as cognitive bias rather than differences in capability.

## Limitations & Future Work

- The AFB benchmark scale is relatively small (200 scenarios) and may not cover all bias patterns.
- Generalization of dialectical alignment training—whether it transfers to domains not covered by AFB.
- The Synthesis stage may still be dominated by one perspective, failing to eliminate bias completely.
- Performance of AOA bias in non-failure scenarios (e.g., success attribution) remains unexplored.

## Related Work & Insights

- **vs Reflexion/Self-Reflection Methods**: Self-reflection conducted within a role framework is affected by AOA bias, which often reinforces incorrect attributions. ReTAS breaks role inertia through an explicit Antithesis stage.
- **vs Multi-Agent Debate Methods**: While debate allows different Agents to hold opposing views, it lacks a structured synthesis mechanism. ReTAS's Synthesis stage provides a clear framework for conflict reconciliation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The identification of AOA in LLMs is an original contribution; the dialectical alignment solution is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple models + specialized benchmark + ablation analysis, though the benchmark scale could be increased.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition; natural integration of social psychology theory with AI methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Assistive Agents Need Accessibility Alignment](../../ICML2026/llm_agent/position_assistive_agents_need_accessibility_alignment.md)
- [\[AAAI 2026\] MoralReason: Generalizable Moral Decision Alignment For LLM Agents Using Reasoning-Level Reinforcement Learning](../../AAAI2026/llm_agent/moralreason_generalizable_moral_decision_alignment_for_llm_agents_using_reasonin.md)
- [\[ACL 2026\] ProPer Agents: Proactivity Driven Personalized Agents for Advancing Knowledge Gap Navigation](proper_agents_proactivity_driven_personalized_agents_for_advancing_knowledge_gap.md)
- [\[ACL 2026\] Verified Critical Step Optimization for LLM Agents](verified_critical_step_optimization_for_llm_agents.md)
- [\[ACL 2026\] Exploring Reasoning Reward Model for Agents](exploring_reasoning_reward_model_for_agents.md)

</div>

<!-- RELATED:END -->
