---
title: >-
  [Paper Note] MAPS: Multi-Agent Personality Shaping for Collaborative Reasoning
description: >-
  [AAAI 2026][LLM Evaluation][Big Five Personality Theory] This paper proposes MAPS, a five-agent collaborative reasoning framework that assigns distinct "personalities" to four functional agents based on the Big Five pers…
tags:
  - "AAAI 2026"
  - "LLM Evaluation"
  - "Big Five Personality Theory"
  - "Multi-Agent Collaboration"
  - "Socratic Critique"
  - "Multimodal Scientific Reasoning"
  - "Personality Shaping"
date: 2026-05-08
content_hash: 3dd3d1160c3635cd
---

# MAPS: Multi-Agent Personality Shaping for Collaborative Reasoning

**Conference**: AAAI 2026
**arXiv**: [2503.16905](https://arxiv.org/abs/2503.16905)  
**Code**: [https://github.com/exoskeletonzj/MAPS](https://github.com/exoskeletonzj/MAPS)  
**Area**: LLM Evaluation
**Keywords**: Big Five Personality Theory, Multi-Agent Collaboration, Socratic Critique, Multimodal Scientific Reasoning, Personality Shaping

## TL;DR
This paper proposes MAPS, a five-agent collaborative reasoning framework that assigns distinct "personalities" to four functional agents based on the Big Five personality theory — Interpreter (Openness), Aligner (Agreeableness), Scholar (Conscientiousness), and Solver (Extraversion) — to achieve heterogeneous collaboration, complemented by a Critic Agent (Neuroticism → Socratic reflection) for iterative refinement. MAPS surpasses the GPT-4o baseline by 15.84% on MathVista/OlympiadBench/EMMA and, for the first time, exceeds human expert performance by 3.58%.

## Background & Motivation

**Background**: Complex scientific reasoning (mathematics, physics, chemistry) demands multi-step inference, cross-modal understanding, and domain knowledge integration. Existing approaches predominantly rely on single-agent or simple dual-agent collaboration (e.g., debating), both of which suffer from two fundamental problems.

**Limitations of Prior Work**: (a) **Behavioral homogeneity**: Multiple agents sharing identical prompts produce repetitive reasoning patterns, lacking the diversity needed for multi-perspective exploration; (b) **Absence of reflection**: Agent interactions are linear and feedback-free, making it impossible to correct errors once an early step goes wrong, leading to cascading failures.

**Key Challenge**: There is an inherent tension between requiring agents to maintain distinctive reasoning styles (diversity) while enabling effective collaboration (coherence), alongside the need for reflective self-correction mechanisms.

**Key Insight**: The work draws inspiration from the Big Five personality theory in psychology — the effectiveness of human team collaboration largely stems from the complementarity of members' personalities. This theory is mapped onto agent design, where each agent's "personality" determines its reasoning focus.

**Core Idea**: The Big Five personality theory is used to shape the reasoning styles of four functional agents (achieving heterogeneity), while a Socratic Critic agent performs reflective correction (achieving iterative refinement).

## Method

### Overall Architecture
Five agents operate in a four-step sequential reasoning pipeline followed by a critique-feedback loop:
- **Interpreter** (Openness) → interprets diagrams and extracts visual semantics
- **Aligner** (Agreeableness) → aligns visual and textual information
- **Scholar** (Conscientiousness) → retrieves and supplements domain knowledge
- **Solver** (Extraversion) → integrates information and produces the final answer
- **Critic** (Neuroticism) → applies Socratic questioning to assess the quality of each step

### Key Designs

1. **Big Five Personality → Functional Role Mapping**:

    - Function: Maps abstract personality dimensions to concrete reasoning functions.
    - Mechanism: Each agent $\mathcal{A}_k$ carries a personality embedding $\mathbf{p}_k \in \mathbb{R}^m$ that modulates its attention and reasoning preferences. The overall reasoning is a function composition: $a_i = \mathcal{A}_4 \circ \mathcal{A}_3 \circ \mathcal{A}_2 \circ \mathcal{A}_1(\mathbf{x}; \mathbf{p}_1, \ldots, \mathbf{p}_4)$
    - Design Motivation: Personality traits determine attentional focus — Openness suits divergent exploration (diagram interpretation), Conscientiousness suits rigorous verification (knowledge retrieval), Extraversion suits goal-directed reasoning (problem solving), and Agreeableness suits integrative reconciliation (information alignment).

2. **Four-Step Sequential Reasoning**:

    - **Interpreter**: $p_i = \psi_{\text{lang}}(\phi_{\text{vis}}(d_i) + W_1 \mathbf{p}_1)$, converts diagrams into structured textual descriptions.
    - **Aligner**: $l_i = \text{CrossFuse}(p_i, c_i, q_i; \mathbf{p}_2)$, fuses diagram, context, and question information via multi-head attention.
    - **Scholar**: $s_i = \text{KnowAug}(l_i, \mathcal{K}(l_i); \mathbf{p}_3)$, retrieves domain knowledge (physical laws, mathematical theorems, etc.) to augment reasoning.
    - **Solver**: $a_i = \text{Deduct}(p_i, l_i, s_i; \mathbf{p}_4)$, performs logical deduction over all accumulated information to yield the final answer.

3. **Critic Socratic Reflection**:

    - Function: Evaluates the quality of the four-step reasoning, identifies the weakest step, and triggers correction.
    - Mechanism: Given the reasoning trace $\mathcal{T} = \{p_i, l_i, s_i, a_i\}$, the Critic computes per-step confidence scores $\mathbf{f}_i = \mathcal{M}_{\text{crit}}(p_i, l_i, s_i, a_i) \in [0,1]^4$. If $f_i^{(k^*)} < \tau$ (where $k^* = \arg\min_k f_i^{(k)}$), step $k^*$ is re-executed. The Critic scores on a 0–5 scale and employs Socratic questioning (guiding reflection rather than directly providing answers).
    - Design Motivation: Proposition 1 proves that each Critic-triggered update guarantees a decrease in variational free energy: $F^{(t+1)} \leq F^{(t)}$, forming a convergent iterative optimization.

### Theoretical Guarantees
- **Proposition 1 (Monotonic Free Energy Descent)**: Each Critic-guided update satisfies $F^{(t+1)} \leq F^{(t)}$, ensuring the iterative process does not degrade.
- **Proposition 2 (Collaborative Information Bottleneck)**: The four-step reasoning process is equivalent to the constrained optimization $\min \sum_k I(\mathbf{x}; \mathcal{S}_k)$ s.t. $I(\mathcal{S}_k; a_i) \geq \epsilon$ — minimizing redundant information while preserving task-relevant information.

## Key Experimental Results

### Main Results

| Method | MathVista | OlympiadBench | EMMA | Overall Avg. |
|--------|-----------|--------------|------|-------------|
| Random | 24.30 | 0.87 | 21.00 | 16.06 |
| Human Expert | 55.90 | 37.80 | 75.17 | 52.73 |
| GPT-4o | 63.10 | 21.47 | 33.67 | 39.41 |
| GPT-4o + CoT | 63.80 | 22.27 | 35.33 | 40.47 |
| Qwen2.5-VL-72B + CoT | 74.80 | 9.59 | 37.00 | 40.46 |
| **MAPS (GPT-4o)** | **79.80** | **31.14** | **58.00** | **56.31** |

### Ablation Study (OlympiadBench)

| Configuration | Avg. Accuracy | Change |
|---------------|--------------|--------|
| MAPS (Full) | 31.14% | — |
| w/o Interpreter | 15.05% | **−16.09%** |
| w/o Scholar | 19.65% | −11.49% |
| w/o Aligner | 20.28% | −10.86% |
| w/o Critic | 24.09% | −7.05% |

### Key Findings
- **First to surpass human experts**: MAPS achieves an overall average of 56.31% vs. 52.73% for human experts, exceeding humans on both OlympiadBench and EMMA.
- **Remarkable performance gains**: MAPS improves over the GPT-4o baseline by 15.84% (absolute) and outperforms the previous best model (Qwen2.5-VL-72B + CoT) by 15.85%.
- **Interpreter is the most critical component**: Its removal causes a 16.09% drop, as diagrams carry crucial information in scientific reasoning and visual understanding is foundational.
- **Critic contributes the least yet remains indispensable**: Its removal incurs only a 7.05% drop, because the four-step reasoning pipeline is already strong on MathVista (where backtracking is rarely needed); however, the Solver receives the most Critic feedback on high-difficulty tasks such as EMMA and OlympiadBench.
- **Cross-model generalization**: Applying MAPS to Qwen2.5-VL-72B yields a 12.4% improvement on physics tasks, and a 4.2% improvement when applied to Gemini.
- **DiagramQG generalization**: MAPS achieves up to 19.51% improvement and an overall gain of 7.71%, validating cross-dataset adaptability.

## Highlights & Insights
- **Mapping the Big Five personality theory to agent design** represents a genuinely innovative interdisciplinary contribution — applying psychological personality theory to address behavioral homogeneity in AI systems, yielding a theoretically grounded heterogeneous collaboration scheme.
- **Novel application of information bottleneck theory** (Proposition 2): Multi-agent collaborative reasoning is formulated as a collaborative information bottleneck optimization, where each agent is responsible for compressing input while retaining task-relevant information, and the Critic monitors for constraint violations — providing an information-theoretic analytical perspective on multi-agent systems.
- **Complementary relationship with the co-authored MARS paper (APO direction)**: MARS focuses on prompt optimization, while MAPS focuses on reasoning collaboration; both share a five-agent + Socratic dialogue + POMDP/variational inference framework and are mutually complementary.

## Limitations & Future Work
- The four-step reasoning follows a fixed sequential order (Interpreter → Aligner → Scholar → Solver), with no support for dynamic step-skipping or looping.
- Personality embeddings are implemented via prompting rather than as truly learnable vectors; "personality shaping" is closer to role-playing prompt engineering than genuine parametric modeling.
- Evaluation is limited to scientific reasoning (mathematics, physics, chemistry), with no assessment on broader reasoning tasks (e.g., commonsense reasoning, legal reasoning).
- The Critic's 0–5 scoring rubric lacks automatic calibration and may require threshold adjustment across different tasks.
- The five-agent pipeline incurs substantial inference costs (6+ LLM calls per question, including Critic feedback).

## Related Work & Insights
- **vs. Single-Agent CoT**: CoT performs reasoning within a single model, whereas MAPS employs personality-differentiated agents for specialized reasoning and reflection, achieving a 16% improvement on MathVista.
- **vs. Simple Multi-Agent Debate**: Debate methods suffer from agent homogeneity; MAPS achieves heterogeneous collaboration through personality shaping, avoiding redundant repetition.
- **vs. MARS (concurrent work from the same group)**: MARS focuses on prompt optimization (how to craft better prompts), while MAPS focuses on reasoning collaboration (how to organize multi-agent problem solving); the two are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ The mapping from Big Five personality theory to agent design is highly original, and the information bottleneck analysis adds theoretical depth; however, the framework bears considerable similarity to MARS.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 3 benchmarks + 10 sub-tasks + ablation studies + cross-model generalization + computational efficiency analysis + case studies — exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The visualization of the personality-to-agent mapping is intuitive and the theoretical proofs are complete; some formulations, however, appear over-formalized.
- Value: ⭐⭐⭐⭐ The first result to surpass human experts on multimodal scientific reasoning is highly impactful, and the heterogeneous agent collaboration paradigm has broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Which LLM Multi-Agent Protocol to Choose?](../../ICLR2026/llm_evaluation/which_llm_multi-agent_protocol_to_choose.md)
- [\[ACL 2026\] Towards Self-Improving Error Diagnosis in Multi-Agent Systems](../../ACL2026/llm_evaluation/towards_self-improving_error_diagnosis_in_multi-agent_systems.md)
- [\[ICLR 2026\] Human-LLM Collaborative Feature Engineering for Tabular Learning](../../ICLR2026/llm_evaluation/human-llm_collaborative_feature_engineering_for_tabular_data.md)
- [\[NeurIPS 2025\] Belief-Calibrated Multi-Agent Consensus Seeking for Complex NLP Tasks](../../NeurIPS2025/llm_evaluation/belief-calibrated_multi-agent_consensus_seeking_for_complex_nlp_tasks.md)
- [\[AAAI 2026\] Where Norms and References Collide: Evaluating LLMs on Normative Reasoning](where_norms_and_references_collide_evaluating_llms_on_normative_reasoning.md)

</div>

<!-- RELATED:END -->
