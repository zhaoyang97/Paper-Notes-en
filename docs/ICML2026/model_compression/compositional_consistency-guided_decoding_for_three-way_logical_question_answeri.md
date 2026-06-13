---
title: >-
  [Paper Note] Compositional Consistency-Guided Decoding for Three-Way Logical Question Answering
description: >-
  [ICML 2026][Model Compression][Logical Reasoning] By leveraging the deterministic negation mapping between a hypothesis $H$ and its negation $\neg H$ in three-way logical QA…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Logical Reasoning"
  - "Consistency-Guided Decoding"
  - "Three-Way QA"
  - "Test-Time Reasoning"
  - "Negation Mapping"
date: 2026-05-08
content_hash: 8f905256374fc294
---

# Compositional Consistency-Guided Decoding for Three-Way Logical Question Answering

**Conference**: ICML 2026  
**arXiv**: [2604.06196](https://arxiv.org/abs/2604.06196)  
**Code**: None  
**Area**: LLM Reasoning  
**Keywords**: Logical Reasoning, Consistency-Guided Decoding, Three-Way QA, Test-Time Reasoning, Negation Mapping

## TL;DR

By leveraging the deterministic negation mapping between a hypothesis $H$ and its negation $\neg H$ in three-way logical QA, this work composes multiple LLM calls at test time and disambiguates through consistency constraints. This approach reduces epistemic abstentions (Unknowns) and improves reasoning accuracy without the need for training.

## Background & Motivation

**Background**: Three-way logical QA (True / False / Unknown) requires a model to determine whether a set of premises $S$ entails hypothesis $H$, entails $\neg H$, or if neither can be inferred. LLMs typically complete three-way classification via a single structured prompt, processing each query independently.

**Limitations of Prior Work**: LLMs frequently output "Unknown" in single-turn calls—many of which are not due to insufficient premises but rather the model's uncertainty or conservative behavior, termed "epistemic abstention." For example, Claude Sonnet 4.5 has an Unknown rate of 75.5% under strict zero-shot prompting, yet 72.6% of those gold labels are actually True or False. Such false abstentions significantly degrade accuracy and coverage.

**Key Challenge**: Three-way logical QA contains a strong structural constraint—the negation mapping $\mathsf{NegMap}$: True ↔ False are swapped, while Unknown remains unchanged. That is, $y(\neg H) = \mathsf{NegMap}(y(H))$. However, standard prompting treats $H$ and $\neg H$ as two unrelated queries, completely wasting this built-in compositional consistency relationship.

**Goal**: Design a training-free, solver-free test-time decoding layer that utilizes negation consistency constraints to propagate information across multiple LLM calls, thereby reducing epistemic Unknowns and improving overall reasoning quality.

**Key Insight**: Since queries for $H$ and $\neg H$ are two "noisy observations" of the same underlying logical state, a definite judgment from one side can derive the label for the other via the negation map. When both sides are uncertain, the deadlock can be broken by falling back to simpler binary entailment probing.

**Core Idea**: Upgrade single-turn prompting to multi-view compositional reasoning using negation consistency constraints—first query $H$ and $\neg H$, accept if consistent, and progressively repair or probe if inconsistent or abstained. All decisions are ultimately projected onto a consistent assignment that satisfies the negation mapping.

## Method

### Overall Architecture

CGD-PD (Consistency-Guided Decoding with Proof-Driven Disambiguation) is a test-time reasoning layer wrapped around an arbitrary LLM. The input consists of premises $S$ and hypothesis $H$, and the output is one of True / False / Unknown. The workflow involves a minimum of 2 and a maximum of 6 model calls, with core logic divided into four stages: ① Bidirectional three-way query → ② Targeted Unknown repair → ③ Binary entailment probing → ④ Conflict resolution. Each stage is triggered only if the previous one fails to reach consistency, ensuring the final decision always satisfies the negation consistency hard constraint.

### Key Designs

1.  **Bidirectional Query + Negation Consistency Projection**:
    - **Function**: Obtains three-way labels from both $H$ and $\neg H$ perspectives and checks consistency using the negation map.
    - **Mechanism**: Calls $y_H = \mathsf{Classify}(S, H)$ and $y_{\neg H} = \mathsf{Classify}(S, \neg H)$ separately. If $y_{\neg H} = \mathsf{NegMap}(y_H)$ and at least one side provides a definite label (not double Unknown), $y_H$ is returned. Negation is implemented via a normalized wrapper (e.g., "NOT: $H$") with its semantics explicitly defined in the prompt. This "dual observation + hard constraint" combination upgrades two independent noisy classification problems into a single constrained joint reasoning problem.
    - **Design Motivation**: LLMs often output inconsistent labels or over-abstain due to phrasing sensitivity; bidirectional queries provide redundant signals, while the negation map provides a basis for disambiguation.

2.  **Targeted Unknown Repair + One-Sided Projection**:
    - **Function**: Performs directional re-evaluation for sides that output Unknown to reduce epistemic abstention.
    - **Mechanism**: When one side is Unknown, a specialized $\mathsf{FixUnknown}(S, H)$ prompt is called, requiring the model to treat Unknown as a last resort—only to be kept if premises are explicitly missing, with an explanation of what is absent. After repair, if one side is definite and the other remains Unknown, the definite label is projected to $H$ via the negation map. If both sides remain Unknown, binary entailment probing is triggered: $S \models H$ and $S \models \neg H$ are queried as Yes/No questions. Only complementary patterns $(Yes, No)$ or $(No, Yes)$ are accepted as True or False; other cases (including double Yes conflicts) remain Unknown.
    - **Design Motivation**: Forcing disambiguation directly leads to over-parsing (misjudging true Unknowns as definite); the hierarchical design balances the reduction of false abstentions with the protection of true uncertainty.

3.  **Conflict Resolver**:
    - **Function**: Handles rare cases where both $y_H$ and $y_{\neg H}$ are definite labels but violate the negation mapping.
    - **Mechanism**: When both sides provide definite but contradictory labels (e.g., both are True), a specialized adjudication prompt chooses between the two consistent assignments $y_H$ and $\mathsf{NegMap}(y_{\neg H})$. This resolver is triggered infrequently.
    - **Design Motivation**: Guarantees that the output always satisfies the negation consistency hard constraint, preventing logically self-contradictory predictions.

## Key Experimental Results

### Main Results

Evaluated on the First-Order Logic fields of the FOLIO dataset validation set (204 samples), using strict zero-shot structured prompts with temperature set to 0.

| Model | Method | Accuracy (%) | Unknown Rate (%) | Epistemic Unknown Rate (%) | Avg. Calls |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-5.2 | Single | 63.7 | 57.4 | 41.5 | 1.00 |
| GPT-5.2 | CGD-PD | 68.1 | 53.9 | 36.3 | 4.36 |
| Claude Sonnet 4.5 | Single | 42.2 | 75.5 | 72.6 | 1.00 |
| Claude Sonnet 4.5 | CGD-PD | 49.0 | 58.8 | 53.3 | 4.91 |

Paired bootstrap 95% CI: GPT-5.2 accuracy gain +4.4pp (CI: +1.5 ~ +7.4), Claude accuracy gain +6.8pp (CI: +3.4 ~ +10.3).

### Coverage and Definite Label Reliability

| Model | Method | Coverage (%) | Correctness (%) | Gold Unknown Retention (%) | Gold U→T | Gold U→F |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-5.2 | Single | 42.6 | 79.3 | 88.4 | 3 | 5 |
| GPT-5.2 | CGD-PD | 46.1 | 83.0 | 88.4 | 4 | 4 |
| Claude Sonnet 4.5 | Single | 24.5 | 60.0 | 81.2 | 9 | 4 |
| Claude Sonnet 4.5 | CGD-PD | 41.2 | 61.9 | 69.6 | 13 | 8 |

### Key Findings
- CGD-PD does not simply replace Unknowns with low-quality definite labels—for GPT-5.2, correctness improved from 79.3% to 83.0% alongside a coverage increase.
- Gold Unknown retention remained stable at 88.4% for GPT-5.2, indicating no over-parsing of truly uncertain samples; however, Claude's retention dropped from 81.2% to 69.6%, showing some over-parsing issues.
- CGD-PD altered 15/204 predictions for GPT-5.2 and 34/204 for Claude, primarily converting Unknowns into correct definite labels.
- The full six-call sequence was triggered for 54% of samples in GPT-5.2 and 61% in Claude, reflecting the prevalence of Unknown outputs.

## Highlights & Insights
- **Turning Task-Inherent Compositional Structure into Decoding Constraints**: Negation mapping is an intrinsic mathematical property of three-way logical QA. The core insight of CGD-PD is that such known relationships should not be wasted in independent queries but explicitly utilized during inference. This idea can be generalized to any task with known input transformation-output constraint relationships.
- **Hierarchical Disambiguation is Superior to Global Forcing**: The progressive design—trying consistency first, then targeted repair, then reduced-dimension probing—balances reducing false abstention and protecting true uncertainty better than direct forced disambiguation.

## Limitations & Future Work
- Only validated on FOL formula inputs in FOLIO; natural language inputs introduce negation scope ambiguities that significantly increase difficulty.
- Only two API models and one strict zero-shot prompt family were used; no comparison was made with test-time baselines like self-consistency under the same call budget.
- Over-parsing of Gold Unknowns on Claude (retention drop from 81.2% to 69.6%) is a primary weakness, requiring more refined selective mechanisms.
- Lack of branch-level diagnostic logs (e.g., repairer change rates, resolver coverage) makes it difficult to precisely pinpoint the contribution of each component.

## Related Work & Insights
- **vs. Self-Consistency (Wang et al., 2023)**: Self-Consistency improves reasoning via aggregation of multiple samples from the same prompt; CGD-PD utilizes compositional constraints between logically coupled different prompts. They are complementary.
- **vs. CheckList / Metamorphic Testing (Ribeiro et al., 2020; Cho et al., 2025)**: Metamorphic testing uses transformation-relationship pairs to *evaluate* failure modes; CGD-PD further utilizes them to *guide* test-time decisions.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of turning known logical constraints into test-time decoding rules is concise and inspiring.
- Experimental Thoroughness: ⭐⭐⭐ Limited scale with only 204 validation samples from one dataset, two models, and one prompt family.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition, natural step-by-step derivation of the method, and thorough diagnostic analysis.
- Value: ⭐⭐⭐⭐ The proposed principle of "leveraging task-intrinsic compositional structure to constrain decoding" is widely transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference](../../ACL2026/model_compression/calibrated_speculative_decoding_frequency-guided_candidate_selection_for_efficie.md)
- [\[ICML 2026\] SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding](speed-bench_a_unified_and_diverse_benchmark_for_speculative_decoding.md)
- [\[ICML 2026\] Critique-Guided Distillation for Robust Reasoning via Refinement](critique-guided_distillation_for_robust_reasoning_via_refinement.md)
- [\[ICML 2026\] LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding](lk_losses_direct_acceptance_rate_optimization_for_speculative_decoding.md)
- [\[ICML 2026\] Active Tabular Augmentation via Policy-Guided Diffusion Inpainting](active_tabular_augmentation_via_policy-guided_diffusion_inpainting.md)

</div>

<!-- RELATED:END -->
