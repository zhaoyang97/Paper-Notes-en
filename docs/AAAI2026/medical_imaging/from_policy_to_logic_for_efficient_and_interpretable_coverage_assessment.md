---
title: >-
  [Paper Note] From Policy to Logic for Efficient and Interpretable Coverage Assessment
description: >-
  [AAAI 2026][Medical Imaging][Neuro-symbolic reasoning] This paper proposes a neuro-symbolic approach that combines a coverage-aware retriever with symbolic rule inference based on PyKnow, assisting human reviewers in efficiently and interpretably assessing whether medical CPT codes are covered by insurance policies. The approach reduces inference cost by 44% while improving F1 by 4.5%.
tags:
  - AAAI 2026
  - Medical Imaging
  - Neuro-symbolic reasoning
  - coverage policy
  - rule engine
  - retrieval augmentation
  - explainable AI
date: 2026-05-08
content_hash: d5508f7a63405738
---

# From Policy to Logic for Efficient and Interpretable Coverage Assessment

**Conference**: AAAI 2026
**arXiv**: [2601.01266](https://arxiv.org/abs/2601.01266)
**Code**: None
**Area**: Medical Imaging
**Keywords**: Neuro-symbolic reasoning, coverage policy, rule engine, retrieval augmentation, explainable AI

## TL;DR

This paper proposes a neuro-symbolic approach that combines a coverage-aware retriever with symbolic rule inference based on PyKnow, assisting human reviewers in efficiently and interpretably assessing whether medical CPT codes are covered by insurance policies. The approach reduces inference cost by 44% while improving F1 by 4.5%.

## Background & Motivation

In the medical insurance domain, determining whether a given medical procedure (identified by a CPT code) is covered by a specific insurance policy is a critically important yet highly complex task. Coverage policy documents (CoC/SPD) typically span hundreds of pages of intricate legal and policy language, requiring professional reviewers to manually compare CPT codes against policy clauses—a process that is labor-intensive and error-prone.

Large language models (LLMs) have demonstrated strong capabilities in legal analysis and policy interpretation, yet they suffer from three key issues: (1) **hallucination and inconsistency**—LLMs may generate plausible but inaccurate reasoning; (2) **high cost**—repeatedly invoking LLM inference over tens of thousands of CPT codes is prohibitively expensive; (3) **lack of traceability**—direct prompting approaches make it difficult for reviewers to trace decisions back to specific policy provisions.

Chain-of-thought (CoT) prompting, while capable of guiding multi-step reasoning, similarly suffers from insufficient interpretability, reasoning inconsistency, and excessive computational cost at scale. Traditional expert systems, on the other hand, ensure consistency and interpretability through rule-based inference but rely on manually encoded domain knowledge.

The paper's starting point is: **combining LLMs' natural language understanding with the deterministic inference of a symbolic rule engine**—using LLMs to generate structured attributes and rules once, and then delegating subsequent inference to a symbolic engine (PyKnow), thereby substantially reducing inference cost while maintaining interpretability.

## Method

### Overall Architecture

The system operates in two stages: (1) **Policy Text Retrieval**—a fine-tuned coverage-aware retriever extracts coverage-relevant clauses from policy documents for each CPT code; (2) **Symbolic Reasoning**—an LLM generates attributes and rules for CPT codes, which are then matched by the PyKnow inference engine to produce auditable reasoning traces.

### Key Designs

1. **Coverage-Aware Retriever**:

    - Function: Precisely retrieves clauses from policy document subsections that determine CPT code coverage status, rather than relying solely on topical similarity.
    - Mechanism: A cross-encoder using Longformer (allenai/longformer-base-4096) as backbone, modeling retrieval as a contrastive multiple-choice ranking problem. Training loss is cross-entropy: $\mathcal{L} = -\log p(i=\text{positive} \mid q, S)$
    - Design Motivation: Standard semantic search can be misled by topical similarity (e.g., an insulin pump CPT code may match diabetes self-management education rather than the relevant durable medical equipment coverage clause). The cross-encoder captures fine-grained query-passage interactions, identifying decisive phrases such as "requires prior authorization," "not covered," and "limited to."
    - Training Data: Approximately 20 certified coding experts annotated over 1.84 million (CPT, subsection, relevance) pairs across 172 coverage documents, with 1.61 million pairs used for training.
    - Architecture Rationale: Each insurance plan contains only approximately 60 candidate subsections, making exhaustive cross-encoder scoring fully feasible on modern GPUs.

2. **Attribute Generation**:

    - Function: Extracts yes/no attributes describing the characteristics of each CPT code (e.g., is_implant, is_pregnancy).
    - Mechanism: CPT codes associated with the same subsection are grouped, and an LLM prompt generates attributes shared among the group and coverage clauses, along with default values (True/False).
    - Design Motivation: Attributes serve as the bridge between natural language policy and symbolic rules. **Attributes are generated only once per CPT code** and can be reused across different insurance plans—a key mechanism for cost control.

3. **Rule Creation**:

    - Function: Generates PyKnow-format symbolic rules for each policy subsection.
    - Mechanism: For each subsection, associated CPT codes and their attributes are collected, and a structured prompt guides the LLM to generate PyKnow rule code. Rules take the form: "if is_pregnancy==True and is_maternity==True, then apply pregnancy_maternity_services rule."
    - Design Motivation: Transforms unstructured policy language into executable if-then rules, ensuring fully traceable reasoning. Rules are generated only once per plan document.

4. **Inference with PyKnow**:

    - Function: Given a CPT code and its attributes, the PyKnow engine matches triggered rules and outputs the reasoning path.
    - Mechanism: The PyKnow engine checks whether each rule's preconditions are satisfied by the current CPT code's attributes; triggered rules and relevant attributes are then presented to the human reviewer.
    - Design Motivation: The inference stage requires **no LLM calls whatsoever**, making **inference cost negligible** (\$2.5 per 1,000 CPT codes). Each inference result is fully deterministic, with no hallucination risk.

### Loss & Training

The retriever is trained with a contrastive cross-entropy loss (equivalent to InfoNCE), using the AdamW optimizer (lr=2e-5, weight decay=0.01), bf16 mixed precision, and gradient checkpointing for memory efficiency. Training runs for 2.5 epochs, taking approximately 48 hours on 8×H100 GPUs.

## Key Experimental Results

### Main Results

Performance comparison on 7 anonymized insurance plan documents (814 CPT codes per plan, 5,698 test samples total):

| Method | Context | Accuracy | F1 | Inference Cost/1K CPTs | Inference Cost/11K CPTs |
|--------|---------|----------|----|------------------------|-------------------------|
| GPT-4o-mini (fine-tuned retrieval) | Retrieved text | 0.94 | 0.96 | \$440 | \$4,840 |
| GPT-4.1 (fine-tuned retrieval) | Retrieved text | 0.92 | 0.95 | \$880 | \$9,680 |
| O3 (fine-tuned retrieval) | Retrieved text | 0.94 | 0.96 | \$880 | \$9,680 |
| GPT-4.1 (full document) | Full document | 0.82 | 0.89 | \$3,520 | \$38,720 |
| Rule-based (zero-shot retrieval) | Retrieved text | 0.85 | 0.91 | \$2 | \$22 |
| Rule-based (fine-tuned retrieval) | Retrieved text | **0.87** | **0.93** | **\$2** | **\$22** |

### Ablation Study

| Configuration | Accuracy | F1 | Notes |
|--------------|----------|----|-------|
| Zero-shot retrieval + rules | 0.85 | 0.91 | Zero-shot retriever baseline |
| Fine-tuned retrieval + rules | 0.87 | 0.93 | Fine-tuning improves accuracy by 2.69%, F1 by 1.72% |
| GPT-4.1 + full document | 0.82 | 0.89 | Full-document input performs worse, highlighting the importance of precise retrieval |
| GPT-4.1 + retrieved text | 0.92 | 0.95 | LLM + high-quality retrieval achieves strongest performance but at high cost |

**Per-plan breakdown:**

| Insurance Plan | GPT-4.1 F1 | Zero-shot Rules F1 | Fine-tuned Rules F1 |
|---------------|-----------|-------------------|---------------------|
| Plan #1 | 0.93 | 0.93 | 0.90 |
| Plan #2 | 0.87 | 0.91 | 0.90 |
| Plan #3 | 0.86 | 0.88 | **0.94** |
| Plan #4 | 0.93 | 0.90 | 0.93 |
| Plan #5 | 0.90 | 0.94 | **0.96** |
| Plan #6 | 0.86 | 0.90 | 0.93 |
| Plan #7 | 0.93 | 0.93 | 0.94 |
| Average | 0.89 | 0.91 | **0.93** |

### Key Findings
- The fine-tuned rule-based system achieves an average F1 of 0.93, **surpassing direct GPT-4.1 full-document inference** (0.89) while reducing inference cost by over 99.9%.
- Providing precisely retrieved text rather than full documents to LLMs **simultaneously improves performance and reduces cost**—full-document input introduces noise.
- The primary sources of rule failure are: 73.5% attributable to relevant attributes not being incorporated during rule generation (LLM "forgetting" later attributes), and 26.5% to incomplete rule sets.
- The one-time setup cost (retriever training \$2,680 + attribute/rule generation) reaches break-even after processing approximately 850 CPT codes.

## Highlights & Insights
- The layered design is elegant: LLMs perform one-time knowledge extraction (attributes + rules), while the symbolic engine handles repeatable deterministic inference, achieving an optimal balance between cost and interpretability.
- The design philosophy of the coverage-aware retriever is insightful—retrieval targets "coverage-decisiveness" rather than "semantic similarity," a distinction that is critical in legal and policy texts.
- The 1.84 million annotated pairs reflect the investment required in real industrial settings; annotation by 20 certified subject matter experts ensures data quality.
- The system is explicitly positioned as a tool to assist human reviewers rather than replace them, with final decision authority retained by humans—a product positioning appropriate for medical and legal domains.

## Limitations & Future Work
- Attribute generation suffers from an "forgetting" problem (source of 73.5% of errors): when attribute lists are long, LLMs tend to neglect later attributes, consistent with positional bias in long contexts.
- Rule sets may be incomplete (source of 26.5% of errors); some samples cannot be handled due to missing rules.
- Evaluation is limited to CPT codes and does not extend to HCPCS or other medical coding systems.
- Internal data and anonymization limit reproducibility.
- Additional comparisons with models specialized for legal reasoning (e.g., LawLLM) or hybrid approaches are absent.
- The maintenance and update mechanism for rules is not discussed—how incremental rule updates are handled when policy documents change remains an open question.

## Related Work & Insights
- LOGIC-LM (Pan et al., 2023) and SymbCoT (Xu et al., 2024) explore LLM-to-formal-logic translation but are not optimized for coverage assessment scenarios.
- Kant et al. (2025) evaluate LLMs' ability to generate structured rules but rely on manually designed patterns and auxiliary functions; this paper eliminates such dependencies.
- Cummins et al. (2025) encode insurance policy logic in Prolog but depend on fully manual coding; this paper automates that process using LLMs.
- The retriever design offers reference value for any RAG system requiring "decisiveness matching" rather than "semantic similarity" matching.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Training-Free Policy Violation Detection via Activation-Space Whitening in LLMs](training-free_policy_violation_detection_via_activation-space_whitening_in_llms.md)
- [\[AAAI 2026\] A Principle-Driven Adaptive Policy for Group Cognitive Stimulation Dialogue for Elderly with Cognitive Impairment](a_principle-driven_adaptive_policy_for_group_cognitive_stimu.md)
- [\[AAAI 2026\] DeepGB-TB: A Risk-Balanced Cross-Attention Gradient-Boosted Convolutional Network for Rapid, Interpretable Tuberculosis Screening](deepgb-tb_a_risk-balanced_cross-attention_gradient-boosted_convolutional_network.md)
- [\[ACL 2026\] Stable On-Policy Distillation through Adaptive Target Reformulation](../../ACL2026/medical_imaging/stable_on-policy_distillation_through_adaptive_target_reformulation.md)
- [\[AAAI 2026\] Efficient Chromosome Parallelization for Precision Medicine Genomic Workflows](efficient_chromosome_parallelization_for_precision_medicine_genomic_workflows.md)

</div>

<!-- RELATED:END -->
