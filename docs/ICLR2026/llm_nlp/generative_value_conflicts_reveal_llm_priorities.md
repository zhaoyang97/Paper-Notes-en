---
title: >-
  [Paper Note] Generative Value Conflicts Reveal LLM Priorities
description: >-
  [ICLR 2026][LLM/NLP][value conflicts] This paper proposes ConflictScope, an automated pipeline for generating value-conflict scenarios. Through open-ended evaluation (rather than multiple-choice), it reveals LLMs' value priority rankings under conflict conditions. Key findings show that models shift from protective values (e.g., harmlessness) toward personal values (e.g., user autonomy) in open-ended settings, and that system prompts can improve alignment with target rankings by 14%.
tags:
  - ICLR 2026
  - LLM/NLP
  - value conflicts
  - LLM alignment
  - value prioritization
  - open-ended evaluation
  - manipulability
  - AI safety
date: 2026-05-08
content_hash: 38d6818afd966afd
---

# Generative Value Conflicts Reveal LLM Priorities

**Conference**: ICLR 2026
**arXiv**: [2509.25369](https://arxiv.org/abs/2509.25369)
**Code**: [GitHub](https://github.com/andyjliu/conflictscope)
**Area**: LLM/NLP
**Keywords**: value conflicts, LLM alignment, value prioritization, open-ended evaluation, manipulability, AI safety

## TL;DR

This paper proposes ConflictScope, an automated pipeline for generating value-conflict scenarios. Through open-ended evaluation (rather than multiple-choice), it reveals LLMs' value priority rankings under conflict conditions. Key findings show that models shift from protective values (e.g., harmlessness) toward personal values (e.g., user autonomy) in open-ended settings, and that system prompts can improve alignment with target rankings by 14%.

## Background & Motivation

LLM alignment research typically pursues models that simultaneously satisfy multiple values (e.g., helpfulness, honesty, harmlessness). In practice, however, these values **frequently conflict**. Understanding how models prioritize values under conflict is critical for predicting their behavior.

Existing alignment datasets **rarely elicit value conflicts**: Buyl et al. (2025) found that approximately 85% of response pairs in HH-RLHF and PKU-SafeRLHF do not trigger disagreements among Anthropic's constitutional principles, making it extremely difficult to study conflicts between specific value pairs.

Existing moral dilemma research suffers from two **ecological validity problems**:

**Bystander perspective**: LLMs are typically positioned as third-party observers of a scenario, rather than as moral agents capable of actively influencing outcomes.

**Multiple-choice evaluation**: Results are highly sensitive to minor variations in evaluation format and lack generalizability.

The core goal is to design a pipeline that can **automatically generate conflict scenarios between specific value pairs** and evaluate model behavior through **open-ended interaction**.

## Method

### Overall Architecture

The ConflictScope pipeline proceeds as follows:

1. **Given a value set** → automatically generate conflict scenarios
2. **Multiple-choice evaluation** → elicit expressed preferences
3. **Open-ended evaluation** (simulated user interaction) → elicit revealed preferences
4. **Bradley-Terry model fitting** → derive value rankings from pairwise comparisons

### Key Designs

**Scenario Generation Pipeline** (two stages):

**Stage 1**: Claude 3.5 Sonnet is provided with descriptions of two values and a deployment context to generate conflict scenario summaries. Four prompt templates are used (minor benefit / strong benefit / minor harm / strong harm) to mitigate models' inaction bias and simulate a realistic severity distribution.

**Stage 2**: Deduplication (scenarios with embedding cosine similarity ≥ 0.8 are removed), followed by generation of detailed descriptions, user personas, and two action options for each scenario.

**Scenario Filtering** (6-dimensional LLM-as-Judge):
1. Scenario realism: can it occur in the real world?
2. Scenario specificity: is it sufficiently detailed?
3. Action feasibility: can a text-only LLM execute both actions?
4. Scenario impossibility: can the two actions not be performed simultaneously?
5. Value guidance: does each value recommend a different action?
6. Genuine dilemma: is there no clearly superior consensus action?

Filtering uses GPT-4.1 as the judge; human studies validate high accuracy across all dimensions.

**Open-ended Evaluation**:
- A **User LLM** (GPT-4.1) plays the role of a user and generates user prompts based on the scenario and user persona.
- The **Target Model** responds without access to the scenario context.
- A **Judge LLM** (GPT-4.1) determines which action the model's response more closely reflects.

Cohen's Kappa between the judge model and human annotations = 0.62 (strong agreement).

**Value Ranking Derivation**:

A **Bradley-Terry preference model** is fitted over all scenario-level pairwise comparisons:

$$P(\text{value } i \succ \text{value } j) = \frac{e^{\beta_i}}{e^{\beta_i} + e^{\beta_j}}$$

This yields a total ordering over the value set.

### Manipulability Evaluation

Alignment $a(R, R_t)$ is defined as the proportion of cases in which the model's choice is consistent with the higher-ranked value in the target ranking $R_t$. Manipulation effect = $\frac{a(R_s, R_t) - a(R_d, R_t)}{1 - a(R_d, R_t)}$. Model behavior is manipulated by incorporating a detailed value ranking description into the system prompt.

### Three Value Sets

1. **HHH**: helpfulness, honesty, harmlessness
2. **Personal-Protective**: personal values (autonomy, authenticity, creativity, empowerment) vs. protective values (responsibility, harmlessness, compliance, privacy)
3. **ModelSpec**: a subset of principles derived from the OpenAI Model Spec, spanning three priority tiers

## Key Experimental Results

### Main Results

**RQ1: ConflictScope scenarios are more challenging than existing datasets**

| Dataset | Observed Agreement ↓ | Likert Divergence ↑ |
|--------|------------|---------------|
| HH-RLHF | ~0.65 | ~0.35 |
| PKU-SafeRLHF | ~0.70 | ~0.40 |
| MoralChoice-HighAmbiguity | ~0.85 | ~0.90 |
| DailyDilemmas | ~0.85 | ~0.75 |
| **ConflictScope-HHH** | **0.786** | **0.801** |
| **ConflictScope-ModelSpec** | — | — |
| **ConflictScope-Personal-Protective** | — | — |

ConflictScope achieves **Pareto optimality** on both metrics: it elicits greater inter-model disagreement while models express stronger directional preferences.

**RQ2: Substantial divergence between expressed and revealed preferences**

On the Personal-Protective value set:
- In **multiple-choice** settings, protective values rank 1.7 on average (high priority).
- In **open-ended** settings, protective values rank 4.5 on average (low priority).

All models (except Claude) **substantially shift from protective to personal values** in open-ended evaluation. The same pattern holds for HHH: multiple-choice responses favor harmlessness > helpfulness, while open-ended responses favor helpfulness > harmlessness.

**RQ3: System prompt manipulation is effective but limited**

| Model | HHH Effect | ModelSpec Effect | Personal-Protective Effect |
|------|---------|--------------|------------------------|
| GPT-4o | positive | positive | positive |
| Claude 3.5 Sonnet | small positive | small positive | small positive |
| Llama 3.1 70B | positive | positive | positive |
| OLMo-2-32B | **0.27** | — | — |
| Claude Haiku 3.5 | **0.01** | — | — |

The average normalized effect size is **0.145** (14.5% of misaligned cases are successfully redirected); only 1 out of 14 models exhibits a significant negative effect on any value set.

### Ablation Study

| Pipeline Variant | Observed Agreement ↓ | Likert Divergence ↑ |
|--------------|------------|---------------|
| Full (ConflictScope) | **0.786** | 0.801 |
| Unfiltered | 0.824 | 0.818 |
| Single-stage | 0.898 | 0.854 |
| Direct | 0.852 | 0.830 |

Filtering improves agreement by 3.8% (more disagreement elicited); two-stage generation reduces agreement by 7.4% relative to single-stage (more challenging scenarios).

### Key Findings

1. **Multiple-choice ≠ actual behavior**: Models "say" they value harmlessness in multiple-choice settings, yet "act" to prioritize helpfulness in open-ended settings.
2. **Claude is an exception**: It exhibits the greatest consistency between the two evaluation settings, possibly due to its alignment training.
3. **Privacy and authenticity are more stable**: They are least affected by the shift between expressed and revealed preferences.
4. **Manipulability varies widely**: OLMo-2-32B effect size 0.27 vs. Claude Haiku 0.01.
5. **Domain analysis**: Model preferences are broadly consistent across application domains (Appendix H).

## Highlights & Insights

1. **Reveals the "say-do" gap in LLM alignment**: The systematic divergence between multiple-choice preferences and actual behavior is the paper's core contribution.
2. **Complete pipeline design**: The fully automated workflow from scenario generation to filtering to evaluation can be directly applied to any value set.
3. **Elegant application of the Bradley-Terry model**: Aggregates scenario-level pairwise comparisons into a global ranking.
4. **Rigorous 6-dimensional filtering**: Ensures scenarios constitute genuine dilemmas with realistic deployment contexts.
5. **Broad evaluation across 14 target models**: Covering OpenAI, Anthropic, Meta, Google, Allen AI, and others.

## Limitations & Future Work

1. **Single-turn interaction**: Only single-turn user–LLM interactions are simulated; multi-turn dialogue may reveal additional patterns.
2. **LLM-simulated users**: Whether simulated users faithfully represent real human behavior remains an open question.
3. **Coarse-grained value rankings**: The Bradley-Terry model assumes a global linear ordering, whereas real-world preferences may be context-dependent.
4. **LLM-dependent scenario generation**: The pipeline may inherit biases from the generating model.
5. The approach can be extended to agentic environments, multi-turn dialogue, and more complex value networks.
6. Item Response Theory could be incorporated for more efficient evaluation.

## Related Work & Insights

- **Anthropic Constitutional AI** (Bai et al., 2022b): ConflictScope directly evaluates conflicts among constitutional principles.
- **DailyDilemmas** (Chiu et al., 2025a): A dataset of everyday moral dilemmas, though less challenging than ConflictScope.
- **CLASH** (Lee et al., 2025): One of several existing value-conflict datasets.
- Key insight: The "say-do" gap suggests that existing multiple-choice-based alignment evaluations may substantially overestimate protective behavior.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Open-ended value-conflict evaluation combined with expressed vs. revealed preference analysis offers a genuinely novel perspective.
- **Technical Depth**: ⭐⭐⭐⭐ — The pipeline design is comprehensive and the statistical analysis is rigorous.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 14 models × 3 value sets × 2 evaluation modes, plus ablations and manipulability analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Research questions are clearly articulated and the overall structure is excellent.
- **Value**: ⭐⭐⭐⭐ — Provides new tools and perspectives for LLM alignment evaluation.
- **Overall Recommendation**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ConflictScope: Generative Value Conflicts Reveal LLM Priorities](quamo_quaternion_motions_for_vision-based_3d_human_kinematics_capture.md)
- [\[ICLR 2026\] Predicting LLM Reasoning Performance with Small Proxy Models](predicting_llm_reasoning_performance_with_small_proxy_models.md)
- [\[ICLR 2026\] How Catastrophic is Your LLM? Certifying Risk in Conversation](how_catastrophic_is_your_llm_certifying_risk_in_conversation.md)
- [\[ICLR 2026\] PT2-LLM: Post-Training Ternarization for Large Language Models](pt2-llm_post-training_ternarization_for_large_language_models.md)
- [\[ICLR 2026\] From Assumptions to Actions: Turning LLM Reasoning into Uncertainty-Aware Planning](from_assumptions_to_actions_turning_llm_reasoning_into_uncertainty-aware_plannin.md)

</div>

<!-- RELATED:END -->
