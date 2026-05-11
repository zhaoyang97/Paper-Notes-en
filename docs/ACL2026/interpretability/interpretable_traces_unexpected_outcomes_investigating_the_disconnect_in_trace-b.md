---
title: >-
  [Paper Note] Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation
description: >-
  [ACL 2026][Interpretability][CoT reasoning traces] By constructing a verifiable intermediate reasoning trace dataset via rule-based question decomposition…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "CoT reasoning traces"
  - "knowledge distillation"
  - "semantic correctness"
  - "trace faithfulness"
date: 2026-05-08
content_hash: 1b4c28ce9027199c
---

# Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation

**Conference**: ACL 2026
**arXiv**: [2505.13792](https://arxiv.org/abs/2505.13792)
**Code**: Available (GitHub)
**Area**: Interpretability / Knowledge Distillation
**Keywords**: CoT reasoning traces, knowledge distillation, semantic correctness, interpretability, trace faithfulness

## TL;DR
By constructing a verifiable intermediate reasoning trace dataset via rule-based question decomposition, this paper reveals that the semantic correctness of CoT reasoning traces correlates unreliably with final answer accuracy (correct traces lead to correct answers only 28% of the time), and that the most interpretable traces are not the most performance-enhancing ones—verbose R1 traces achieve the best performance yet are rated the least interpretable by users.

## Background & Motivation

**Background**: Reasoning-oriented LLMs (e.g., DeepSeek R1) generate CoT reasoning traces to improve performance. These traces serve not only as inference-time guidance but also as supervision signals in knowledge distillation (KD) for training smaller models.

**Limitations of Prior Work**: A prevalent yet unexamined implicit assumption is that CoT traces are both semantically correct at inference time and interpretable to end users. However, SFT training objectives require only that the final answer be correct, not that the reasoning trace be semantically valid or interpretable. The verbose and unstructured nature of reasoning traces makes it extremely difficult to verify their validity and interpretability.

**Key Challenge**: Reasoning traces are simultaneously assigned two roles—(1) as training/inference signals for LLMs to improve performance, and (2) as interpretability tools for explaining the reasoning process to users—yet these two objectives may be fundamentally at odds.

**Goal**: To independently evaluate (1) whether the semantic correctness of CoT traces correlates with task performance, and (2) whether the interpretability of CoT traces correlates with task performance.

**Key Insight**: A rule-based question decomposition approach (comprising classification steps and information retrieval steps) is used to construct SFT datasets with verifiable intermediate reasoning traces, enabling correctness and answer accuracy to be assessed independently.

**Core Idea**: Through a verifiable experimental design, this paper demonstrates that researchers should decouple "model supervision objectives" from "user-facing reasoning trace design"—the two should not be conflated.

## Method

### Overall Architecture
On open-book QA benchmarks (CoTemp QA, MS MARCO, Facebook bAbI), rule-based question decomposition is used to generate verifiable correct/incorrect intermediate reasoning traces, which are used to construct different SFT datasets for training small models. A human interpretability study with 100 participants is conducted in parallel.

### Key Designs

1. **Rule-Based Question Decomposition and Trace Construction**:

    - Function: Generate structured intermediate reasoning traces whose correctness can be independently verified.
    - Mechanism: QA questions are decomposed into two steps—(1) a classification step that determines the question type (e.g., temporal relation type), and (2) an information retrieval step that identifies the textual facts required to answer the question. Input–Trace–Output triplets are thereby constructed, with each step of the trace independently verifiable. SFT w/ Correct Traces uses correct classification and correct facts; SFT w/ Incorrect Traces uses incorrect classification and incorrect facts while preserving the correct final answer.
    - Design Motivation: LLM-generated traces are noisy and cannot be deterministically verified; rule-based decomposition ensures binary, non-probabilistic evaluation.

2. **Interpretability Comparison Across Trace Types**:

    - Function: Evaluate the interpretability–performance trade-off across different trace types.
    - Mechanism: SFT is conducted with four types of traces—(1) rule-based correct decomposition traces, (2) verbose DeepSeek R1 reasoning traces, (3) GPT-4o-mini-generated summaries of R1 traces, and (4) GPT-4o-mini-generated post-hoc explanations of R1 traces. Performance is evaluated on the same tasks alongside a human interpretability study.
    - Design Motivation: If interpretability and performance can be jointly optimized, interpretable traces should also yield strong performance; if they conflict, decoupling is necessary.

3. **100-Participant Human Interpretability Study**:

    - Function: Quantify end-user perceptions of interpretability across different trace types.
    - Mechanism: 100 participants are recruited via Prolific (25 per condition) and evaluate four trace types using standardized Likert scales across three dimensions—predictability, understandability, and faithfulness—while cognitive load is also measured.
    - Design Motivation: Model performance is measured by automatic metrics, but interpretability must be judged subjectively by human users.

### Loss & Training
SFT is conducted using Llama-3.2-1B-Instruct and Qwen3-1.7B; interpretability experiments additionally employ Qwen3-8B and Llama-3.1-8B.

## Key Experimental Results

### Main Results
Results on the CoTemp QA dataset:

| Model + Setting | Final Answer Accuracy | Classification Step Accuracy | IR Step Accuracy |
|---|---|---|---|
| Qwen3-1.7B SFT-Vanilla | 60.33% | — | — |
| Qwen3-1.7B SFT-Correct Traces | 52.88% | 47.06% | 78.99% |
| Qwen3-1.7B SFT-Incorrect Traces | **63.88%** | 20.36% | 56.92% |
| Llama SFT-Vanilla | 44.65% | — | — |
| Llama SFT-Correct Traces | 39.55% | 39.09% | 79.40% |
| Llama SFT-Incorrect Traces | **45.58%** | 18.80% | 73.62% |

### Ablation Study

| Trace Type | Interpretability Score (1–5) | Cognitive Load (1–5) | Model Performance |
|---|---|---|---|
| R1 Traces | 3.39 (lowest) | 4.59 (highest) | **Best** |
| R1 Summary | Moderate | Moderate | Moderate |
| Post-hoc Explanation | Moderate–High | Moderate–Low | Moderate |
| Decomposition Traces | **Highest** | **Lowest** | Lowest |

### Key Findings
- Correct reasoning traces lead to correct final answers only 28% of the time—semantic correctness correlates unreliably with answer accuracy.
- Models trained with incorrect traces outperform those trained with correct traces (63.88% vs. 52.88%), suggesting that reasoning traces do not function as semantic guidance for LLMs.
- R1 traces achieve the best performance but the lowest interpretability (3.39/5) and highest cognitive load (4.59/5), revealing a fundamental trade-off.
- The most interpretable decomposition traces yield the worst performance, demonstrating a conflict between interpretability and performance objectives.

## Highlights & Insights
- The finding that "semantically correct reasoning traces do not necessarily improve performance" poses a fundamental challenge to current CoT distillation practices—traces may function more as "token density regulators" than as "reasoning pathway guides."
- The recommendation to "decouple model supervision objectives from user-facing interpretability" carries significant practical implications—systems should generate two distinct sets of reasoning traces for these respective purposes.
- Rule-based question decomposition enables independent verification of trace correctness, and this experimental design methodology has broader applicability.

## Limitations & Future Work
- Validation is limited to the QA domain; conclusions may differ for mathematical reasoning, code generation, and other settings.
- Rule-based decomposition is applicable only to question types that can be structurally decomposed, limiting generalizability.
- The human study involves only 100 participants (25 per condition), resulting in limited statistical power.
- Future work should explore mechanistic explanations for why incorrect reasoning traces can also improve performance.

## Related Work & Insights
- **vs. Magister et al. (CoT distillation)**: Their work assumes that CoT traces provide valuable reasoning signals; the present paper challenges this assumption.
- **vs. Barez et al. (reasoning trace interpretability)**: They argue that reasoning traces are not interpretable to users; the present paper further quantifies the interpretability–performance trade-off.
- **vs. Kambhampati et al. (R1 trace analysis)**: They identify R1 traces as verbose and unstructured; the present paper provides systematic experimental evidence in support of this observation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Challenges core assumptions of CoT distillation with surprising and important findings.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, four trace types, and a human study, though limited in scale.
- Writing Quality: ⭐⭐⭐⭐ Argumentation is logically clear, though some result tables could be presented more intuitively.
- Value: ⭐⭐⭐⭐⭐ Provides important directional guidance for CoT distillation and interpretability research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] A Closer Look at Knowledge Distillation in Spiking Neural Network Training](../../AAAI2026/interpretability/a_closer_look_at_knowledge_distillation_in_spiking_neural_ne.md)
- [\[ACL 2026\] Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models](experiments_or_outcomes_probing_scientific_feasibility_in_large_language_models.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[ACL 2026\] Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation](understanding_new-knowledge-induced_factual_hallucinations_in_llms_analysis_and_.md)
- [\[ICLR 2026\] Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences](../../ICLR2026/interpretability/narrow_finetuning_leaves_clearly_readable_traces_in_activation_differences.md)

</div>

<!-- RELATED:END -->
