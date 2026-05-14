---
title: >-
  [Paper Note] Where Norms and References Collide: Evaluating LLMs on Normative Reasoning
description: >-
  [AAAI 2026][LLM Evaluation][Social norm reasoning] This paper proposes SNIC, a diagnostic testbed comprising 9,000 instances across 51 scenarios…
tags:
  - "AAAI 2026"
  - "LLM Evaluation"
  - "Social norm reasoning"
  - "reference resolution"
  - "embodied agents"
  - "diagnostic testbed"
  - "implicit knowledge"
date: 2026-05-08
content_hash: 7057a0ba5d4a4b14
---

# Where Norms and References Collide: Evaluating LLMs on Normative Reasoning

**Conference**: AAAI 2026
**arXiv**: [2602.02975](https://arxiv.org/abs/2602.02975)
**Code**: [https://github.com/TheSittingCat/SNIC](https://github.com/TheSittingCat/SNIC)
**Area**: LLM Evaluation
**Keywords**: Social norm reasoning, reference resolution, embodied agents, diagnostic testbed, implicit knowledge

## TL;DR
This paper proposes SNIC, a diagnostic testbed comprising 9,000 instances across 51 scenarios, designed to evaluate whether LLMs can leverage implicit social norms to resolve ambiguous reference expressions (e.g., "hand me the cup" when multiple cups are present). Results show that LLMs achieve an average accuracy of only 44% given scene descriptions alone; adding Prolog-based formal logic yields negligible improvement (44.2%), whereas explicitly providing a list of norms dramatically raises accuracy to 70.5% (GPT-4.1 reaches 99.6%). This demonstrates that LLMs lack implicit physical norm knowledge yet can effectively exploit explicit norms.

## Background & Motivation

**Background**: Embodied AI systems (e.g., service robots) must interpret ambiguous instructions in the physical world by drawing on social norms. For instance, "hand me the cup" — when one clean cup and two dirty cups are present — implicitly invokes the norm "prefer the clean one," without any explicit statement.

**Limitations of Prior Work**:
- Whether LLMs can infer such implicit norms in the manner humans do remains entirely unknown.
- Existing NLU research focuses on linguistic disambiguation rather than social-norm-driven disambiguation.
- No systematic benchmark exists for evaluating normative reasoning in LLMs.

**Key Challenge**: Reference resolution in the physical world frequently depends on social norms that are never explicitly stated (e.g., cleanliness over convenience, safety over speed). It remains unclear whether LLMs encode such physically grounded social norms.

**Goal**: Design a diagnostic testbed to evaluate LLMs' capacity for implicit normative reasoning.

**Key Insight**: 51 human-validated norm scenarios → rule-based augmentation to 9,000 instances (systematically varying object types and attributes while preserving normative structure) → evaluation under three conditions (description only / +Prolog / +explicit norms).

**Core Idea**: LLMs do not internalize implicit social norms but can effectively use explicit ones — embodied AI therefore requires an external normative knowledge base.

## Method

### Overall Architecture
The SNIC benchmark is constructed from 51 seed scenarios (manually created and validated by five annotators, $\kappa = 0.22$–$0.66$), expanded via rule-based programmatic augmentation (systematically varying object types and attributes while preserving normative structure) to yield 9,000 instances. Evaluation is conducted under three conditions: (1) scene description only; (2) + Prolog first-order logic; (3) + explicit norm list. Eleven models are evaluated (Granite / Phi / Llama / Qwen / GPT-4o).

### Key Designs

1. **Normative Scenario Design**:

    - **Function**: Construct reference resolution scenarios that require social norm reasoning.
    - **Mechanism**: Each scenario contains a request ("hand me X") and multiple candidate objects; the correct selection depends on an implicit norm (e.g., "in a shared kitchen, prefer the clean one"; "for an elderly person, prefer the lighter one").
    - **Design Motivation**: Physically grounded norms (rather than cultural or social norms) are selected, as they are relatively universal and most critical for embodied AI.

2. **Programmatic Augmentation**:

    - **Function**: Scale from 51 seeds to 9,000 instances covering diverse object/attribute combinations.
    - **Mechanism**: Systematically substitute object types (cup → bowl → plate) and attributes (clean/dirty → new/old) while keeping the normative structure invariant.
    - Seed quality is paramount — augmentation introduces no new norms, only additional instantiations of existing ones.

3. **Three-Condition Controlled Experiment**:

    - **Function**: Disentangle whether LLMs lack normative knowledge versus the ability to apply norms.
    - Condition 1 (description only): tests implicit knowledge.
    - Condition 2 (+Prolog): tests whether formal reasoning aids performance.
    - Condition 3 (+explicit norms): tests norm application ability.

### Loss & Training
Evaluation only — zero-shot, no training.

## Key Experimental Results

### Main Results

| Condition | Avg. Accuracy | GPT-4.1 | Phi-3 14B |
|-----------|--------------|---------|-----------|
| Description only | 44.08% | 56.98% | ~45% |
| + Prolog | 44.22% | 59.62% | ~45% |
| **+ Explicit norms** | **70.51%** | **99.6%** | **77.12%** |

### Ablation Study

| Finding | Specific Data |
|---------|--------------|
| Prolog provides almost no benefit | 44.08% → 44.22% (+0.14%) |
| Explicit norms yield large gains | 44.08% → 70.51% (+26.4%) |
| GPT-4.1 near-perfect with norms | 99.6% |
| Norm conflict handling | Cleanliness preference > safety preference (bias toward specific norm priority) |
| Poor consistency | Minor condition changes under the same norm lead to different predictions |

### Key Findings
- **LLMs fundamentally lack implicit social norm knowledge**: average accuracy of 44% with scene description alone, near chance level.
- **Prolog formal logic is ineffective**: only +0.14% — models do not understand formalized norm specifications.
- **Explicit norms yield substantial improvement** (+26%): LLMs can *apply* norms but cannot *infer* them.
- **Norm priority bias**: models favor cleanliness-related norms while neglecting safety-related ones, indicating a lack of deep understanding of norm hierarchies.
- **Consistency issues**: identical norms produce different predictions under slightly varied conditions.

## Highlights & Insights
- The finding that LLMs "lack implicit knowledge but excel at using explicit knowledge" directly informs embodied AI design: robots require an external normative database.
- The negative result that Prolog is ineffective is valuable — it demonstrates that formal specification is not a viable solution.
- Analysis of norm conflicts reveals that LLMs engage in only shallow normative processing.

## Limitations & Future Work
- Purely text-based evaluation, with no visual or multimodal input (required for real embodied scenarios).
- Only off-the-shelf models are tested; norm fine-tuning and RLHF remain unexplored.
- Human annotation agreement is limited ($\kappa \leq 0.66$).
- The 9,000 augmented instances have not been independently validated.

## Related Work & Insights
- **vs. SocialBias (Howard et al.)**: that work probes social bias; this paper probes physical norms — a distinct dimension.
- **vs. Theory of Mind benchmarks**: ToM benchmarks test mental state attribution; this paper tests social norm reasoning.
- Findings offer guidance for norm injection strategies in embodied AI.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic evaluation of LLMs' social norm reasoning; diagnostic design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ 11 models, 9,000 instances, three-condition controlled comparison.
- Writing Quality: ⭐⭐⭐⭐ Scenario design is intuitive and persuasive.
- Value: ⭐⭐⭐⭐ Important contribution to embodied AI and research on the knowledge boundaries of LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Are They Lovers or Friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues](../../ACL2026/llm_evaluation/are_they_lovers_or_friends_evaluating_llms39_social_reasoning_in_english_and_kor.md)
- [\[NeurIPS 2025\] On Evaluating LLM Alignment by Evaluating LLMs as Judges](../../NeurIPS2025/llm_evaluation/on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)
- [\[ICLR 2026\] Truthfulness Despite Weak Supervision: Evaluating and Training LLMs Using Peer Prediction](../../ICLR2026/llm_evaluation/truthfulness_despite_weak_supervision_evaluating_and_training_llms_using_peer_pr.md)
- [\[ICLR 2026\] DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science](../../ICLR2026/llm_evaluation/dare-bench_evaluating_modeling_and_instruction_fidelity_of_llms_in_data_science.md)
- [\[ACL 2026\] RoleConflictBench: A Benchmark of Role Conflict Scenarios for Evaluating LLMs' Contextual Sensitivity](../../ACL2026/llm_evaluation/roleconflictbench_a_benchmark_of_role_conflict_scenarios_for_evaluating_llms39_c.md)

</div>

<!-- RELATED:END -->
