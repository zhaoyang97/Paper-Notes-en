---
title: >-
  [Paper Note] CGBench: Benchmarking Language Model Scientific Reasoning for Clinical Genetics Research
description: >-
  [NeurIPS 2025][Medical LLM][clinical genetics] This paper introduces CGBench, a clinical genetics benchmark grounded in ClinGen expert annotations, designed to evaluate the scientific literature reasoning capabilities of LLMs from both variant and gene curation perspectives. The benchmark encompasses three tasks—evidence scoring, evidence verification, and experimental evidence extraction—and finds that reasoning models perform best on fine-grained tasks but underperform non-…
tags:
  - "NeurIPS 2025"
  - "Medical LLM"
  - "clinical genetics"
  - "language models"
  - "scientific reasoning"
  - "benchmark"
  - "evidence evaluation"
date: 2026-05-08
content_hash: 8a07a36eac19537d
---

# CGBench: Benchmarking Language Model Scientific Reasoning for Clinical Genetics Research

**Conference**: NeurIPS 2025
**arXiv**: [2510.11985](https://arxiv.org/abs/2510.11985)  
**Code**: [GitHub](https://github.com/owencqueen/cgbench)  
**Area**: Medical Imaging
**Keywords**: clinical genetics, language models, scientific reasoning, benchmark, evidence evaluation

## TL;DR
This paper introduces CGBench, a clinical genetics benchmark grounded in ClinGen expert annotations, designed to evaluate the scientific literature reasoning capabilities of LLMs from both variant and gene curation perspectives. The benchmark encompasses three tasks—evidence scoring, evidence verification, and experimental evidence extraction—and finds that reasoning models perform best on fine-grained tasks but underperform non-reasoning models on high-level judgments.

## Background & Motivation

**Background**: Determining gene–disease associations (gene curation) and variant pathogenicity (variant curation) are core tasks in precision medicine within clinical genetics, traditionally requiring expert manual review of extensive scientific literature and assignment of evidence codes following ACMG/AMP guidelines.

**Limitations of Prior Work**: Manual curation is labor-intensive, time-consuming, and prone to inconsistency. Existing LLM scientific benchmarks predominantly focus on narrow tasks such as multiple-choice questions and claim verification, failing to reflect the demands of real-world scientific literature synthesis and reasoning.

**Key Challenge**: Although LLM literature comprehension capabilities continue to improve, their performance on complex evidence synthesis tasks requiring adherence to precise, domain-specific guidelines remains unclear—particularly when handling highly customized instructions such as VCEP (Variant Curation Expert Panel) specifications.

**Goal**: To design a benchmark that reflects authentic scientific workflows and systematically evaluates LLM capabilities in literature-driven evidence extraction, strength assessment, and evidence classification.

**Key Insight**: Expert-annotated curation records from the ClinGen ERepo are directly leveraged to construct evaluation tasks, ensuring that ground truth derives from the highest-quality human expert review.

**Core Idea**: Three tasks of progressively increasing difficulty are extracted from ClinGen's VCI and GCI, with an LM-as-Judge methodology employed to assess explanation quality.

## Method

### Overall Architecture
CGBench comprises three main tasks: (1) VCI Evidence Scoring (E-Score): given a paper and variant information, predict the evidence code; (2) VCI Evidence Verification (E-Ver): given a paper and a specific evidence code, determine whether its criteria are satisfied; (3) GCI Experimental Evidence Extraction (E-Extract): extract structured experimental evidence entries from papers. All data originate from the ClinGen Evidence Repository.

### Key Designs

1. **VCI Evidence Scoring (E-Score)**:

    - Function: Given a variant query $q_i^v = (d_i, v_i, m_i)$ (disease, variant, mode of inheritance) and the full paper text $T_j$, select the correct evidence code from the VCEP-defined code set $\mathcal{Y}_{vcep}$.
    - Mechanism: $\mathbf{ES}(q_i^v, T_j, \mathcal{Y}_{vcep} | f_{LM}) = \hat{y}_k$, where evidence codes form a three-level hierarchy—primary code (pathogenic/benign), secondary code (strength), and tertiary code (evidence type)—with increasing difficulty at deeper levels.
    - Evaluation Metrics: Precision@5 and Recall@5.
    - Dataset Scale: 205 evaluation samples, 120 papers, 40 diseases, 191 variants.

2. **VCI Evidence Verification (E-Ver)**:

    - Function: Given a specific evidence code, determine whether the paper satisfies the criteria for that code (binary classification).
    - Formulation: $\mathbf{EV}(q_i^v, T_j, y_k | f_{LM}) = \hat{v}$, $\hat{v} \in \{\text{"met"}, \text{"not met"}\}$.
    - Dataset Scale: 242 evaluation samples (167 "not met," 119 "met"), 28 evidence codes.

3. **GCI Experimental Evidence Extraction (E-Extract)**:

    - Function: Extract structured experimental evidence tuples $(a_i, h_i, s_i, r_i)$ from papers—evidence category, explanation, score, and rationale for score modification.
    - Formulation: $\mathbf{EE}(q_i^g, T_j, \mathcal{C}_{sop} | f_{LM}) = (a_i, h_i, s_i, r_i)$.
    - Evaluation Dimensions: category matching (Precision/Recall), structured output compliance rate, normalized MAE, and ΔStrength (score change direction accuracy).
    - Dataset Scale: 336 evaluation samples, 860 diseases, 1,291 genes.

4. **LM-as-Judge Explanation Evaluation**:

    - Function: An LLM judge is used to assess the consistency between model-generated explanations and ClinGen expert explanations.
    - Mechanism: Three prompting strategies are designed (task-agnostic, task-aware, evidence-aware); following calibration against a human-annotated subset, the task-aware approach is selected (F1 = 0.744).
    - Design Motivation: Relying solely on classification accuracy fails to detect LLM hallucinations—a model may correctly classify while generating explanations that deviate from expert reasoning.

### Loss & Training
Eight LLMs are evaluated: GPT-4o, GPT-4o-mini, Claude Sonnet 3.7, Qwen2.5 72B, Llama 4, DeepSeek R1, o3-mini, and o4-mini. Chain-of-thought prompting, role-playing, and full-paper context are employed throughout. Pass@5 sampling is applied for E-Score and E-Extract tasks.

## Key Experimental Results

### Main Results (E-Score Tertiary-Level Accuracy)

| Model | Primary P@5/R@5 | Secondary P@5/R@5 | Tertiary P@5/R@5 |
|------|-------------|-------------|---------------|
| GPT-4o | 0.861/0.878 | 0.517/0.568 | 0.383/0.427 |
| o4-mini | 0.743/0.859 | 0.494/0.600 | **0.420/0.495** |
| DeepSeek R1 | 0.780/0.898 | 0.485/0.629 | 0.418/0.517 |
| Llama 4 | 0.837/0.873 | 0.471/0.532 | 0.361/0.424 |
| GPT-4o-mini | 0.841/0.849 | 0.463/0.527 | 0.278/0.341 |
| Qwen2.5 72B | 0.807/0.863 | 0.481/0.559 | 0.270/0.322 |

Reasoning models (o4-mini, DeepSeek R1) lead at the tertiary level, but exhibit lower primary-code precision, suggesting that overthinking introduces errors on simpler judgments.

### GCI Evidence Extraction

| Model | Category Precision | Category Recall | Structure Compliance | Norm. MAE ↓ | ΔStrength ↑ |
|------|--------------|------------|-----------|-------------|------------|
| GPT-4o | **0.493** | 0.787 | 98.81% | 0.196 | 0.342 |
| o4-mini | 0.425 | **0.835** | 96.73% | **0.186** | **0.445** |
| DeepSeek R1 | 0.456 | 0.734 | 61.61% | 0.228 | 0.346 |
| Llama 4 | 0.363 | 0.787 | 99.40% | 0.393 | 0.129 |

### Key Findings
- Reasoning models outperform non-reasoning models on fine-grained evidence classification (tertiary codes) but underperform GPT-4o on high-level judgments (primary codes) and evidence verification.
- All models achieve F1 < 0.634 on E-Ver and exhibit a systematic tendency to over-predict "met" (predicted positive rate ~0.66 vs. actual 0.43), revealing a pronounced weakness in LLM evidence sufficiency assessment.
- DeepSeek R1 fails structured output compliance in approximately 40% of GCI extraction cases, highlighting structured generation as a key weakness of reasoning models.
- In-context learning (ICL) prompting is effective for E-Score (tertiary-code gains are substantial at 30-shot), but its effect on E-Ver is inconsistent.
- LM Judge evaluation reveals that even when classifications are correct, GPT-4o zero-shot explanations agree with expert explanations only 48.6% of the time; this rises to 70.4% with 30-shot prompting.

## Highlights & Insights
- **Alignment with Real Scientific Workflows**: CGBench is directly grounded in the ClinGen expert curation process, making it the LLM benchmark most closely aligned with real-world scientific literature synthesis reasoning—standing in sharp contrast to the majority of multiple-choice-based scientific QA benchmarks.
- **The Double-Edged Nature of Reasoning Models**: Reasoning models (o4-mini, R1) excel on tasks requiring detailed analysis, yet underperform non-reasoning models on tasks requiring holistic judgment, revealing the cost of overthinking.
- **Correct Classification ≠ Correct Understanding**: LM Judge evaluation surfaces an important insight—models may correctly classify through superficial pattern matching while generating explanations that are inconsistent with expert reasoning or outright hallucinated.

## Limitations & Future Work
- E-Score contains only 205 samples across 33 VCEPs, averaging approximately 7 samples per VCEP, limiting statistical power.
- The LM Judge itself may harbor biases; despite human calibration, it remains an imperfect proxy.
- All models are evaluated in a frozen setting; the upper bound of fine-tuning or RAG-augmented approaches is unexplored.
- Multimodal evidence (e.g., genomic maps, protein structures) is not covered.
- Future work could explore CGBench as an agent benchmark, enabling models to actively search and filter literature rather than passively consuming provided text.

## Related Work & Insights
- **vs. SciFact/PubMedQA**: These benchmarks focus on simple claim verification or question answering; CGBench requires models to perform multi-level evidence synthesis and strength assessment following a specific guideline framework, representing an order-of-magnitude increase in complexity.
- **vs. LitGen**: LitGen also involves literature mining but does not account for the complexity of VCEP specifications—different variants carry different evidence adjudication standards.
- **vs. Agentic Scientific Workflows**: CGBench's findings (LLM performance on evidence judgment approaching random levels) suggest that standalone LLM calls are insufficient and that agent frameworks incorporating tool use and multi-turn reflection are necessary.

## Rating
- Novelty: ⭐⭐⭐⭐ First LLM benchmark targeting real clinical genetics curation workflows, with task designs closely aligned with practical needs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Eight models, multiple prompting strategies, LM Judge evaluation, and human calibration; dataset scale is somewhat limited.
- Writing Quality: ⭐⭐⭐⭐ Background knowledge is thoroughly introduced and task formalization is clear, though domain prerequisites are high.
- Value: ⭐⭐⭐⭐ Provides rigorous evaluation standards for LLM applications in scientific literature reasoning, offering meaningful guidance for the AI4Science community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] CureAgent: A Training-Free Executor-Analyst Framework for Clinical Reasoning](cureagent_a_training-free_executor-analyst_framework_for_clinical_reasoning.md)
- [\[NeurIPS 2025\] HealthSLM-Bench: Benchmarking Small Language Models for Mobile and Wearable Healthcare Monitoring](healthslm-bench_benchmarking_small_language_models_for_mobile_and_wearable_healt.md)
- [\[ACL 2026\] CURA: Clinical Uncertainty Risk Alignment for Language Model-Based Risk Prediction](../../ACL2026/medical_nlp/cura_clinical_uncertainty_risk_alignment_for_language_model-based_risk_predictio.md)
- [\[ICLR 2026\] GALAX: Graph-Augmented Language Model for Explainable Reinforcement-Guided Subgraph Reasoning in Precision Medicine](../../ICLR2026/medical_nlp/galax_graph-augmented_language_model_for_explainable_reinforcement-guided_subgra.md)
- [\[NeurIPS 2025\] Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models](position_thematic_analysis_of_unstructured_clinical_transcripts_with_large_langu.md)

</div>

<!-- RELATED:END -->
