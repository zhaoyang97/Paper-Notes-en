---
title: >-
  [Paper Note] ProMedical: Hierarchical Fine-Grained Criteria Modeling for Medical LLM Alignment via Explicit Injection
description: >-
  [ACL 2026][Medical NLP][Medical LLM alignment] ProMedical utilizes hierarchical fine-grained clinical rubrics co-constructed with physicians to unify preference data, reward models…
tags:
  - "ACL 2026"
  - "Medical NLP"
  - "Medical LLM alignment"
  - "fine-grained rubric"
  - "safety veto"
  - "reward model"
  - "GRPO"
date: 2026-05-08
content_hash: a7d553f80240875b
---

# ProMedical: Hierarchical Fine-Grained Criteria Modeling for Medical LLM Alignment via Explicit Injection

**Conference**: ACL 2026  
**arXiv**: [2604.08326](https://arxiv.org/abs/2604.08326)  
**Code**: The paper claims to release data, reward models, and benchmarks; no specific URL is provided in the cache.  
**Area**: Medical NLP  
**Keywords**: Medical LLM alignment, fine-grained rubric, safety veto, reward model, GRPO

## TL;DR
ProMedical utilizes hierarchical fine-grained clinical rubrics co-constructed with physicians to unify preference data, reward models, and benchmarks. Through explicit criteria injection, a multi-dimensional reward model is trained, enabling Qwen3-8B to achieve a 22.3% improvement in overall accuracy and a 21.7% increase in safety compliance during medical alignment.

## Background & Motivation

**Background**: Medical LLMs are now capable of answering questions regarding symptoms, diagnosis, treatment, and health management. Closed-source models have approached clinical expert levels on several medical benchmarks. However, evaluation standards in medical scenarios are becoming increasingly granular: they must not only be factually correct but also avoid hallucinations, identify risks, adhere to clinical boundaries, and demonstrate empathy and clear reasoning.

**Limitations of Prior Work**: Mainstream alignment data still relies on coarse-grained preference pairs or overall scores. Models only know which response is better without understanding whether it is due to safety, factual accuracy, completeness, tone, or clinical workflow. For high-risk medical errors, such binary signals can easily lead the model to mistake "fluent and helpful" for "safe and professional."

**Key Challenge**: There is a discrepancy between the fine-grained clinical standards required at the evaluation end and the coarse-grained preference signals provided at the training end. This inconsistency between training objectives and actual clinical evaluation makes it difficult for models to internalize complex medical protocols.

**Goal**: To build a unified framework where instruction-specific clinical rubrics are not merely post-hoc evaluation tools but are integrated into preference construction, reward modeling, and the RL alignment process.

**Key Insight**: The authors categorize medical response quality into three orthogonal dimensions: Proficiency, Excellence, and Safety. Safety is designed as a strict veto constraint to prevent models from offsetting safety violations with high-utility answers.

**Core Idea**: Explicitly inject fine-grained criteria for each medical instruction into the reward model. This allows the reward model to judge preferences "under specific rubric conditions" rather than outputting a black-box scalar that aggregates all factors.

## Method

### Overall Architecture
ProMedical consists of three layers: The first is ProMedical-Rubrics, which maps each medical instruction to clinical criteria. The second includes ProMedical-Preference-50k and ProMedical-Bench, used for training and evaluation respectively. The third is Explicit Criteria Injection, which trains a Rubric-Aware Reward Model used to guide the GRPO alignment of Qwen3-8B. Its core innovation is not proposing a new medical QA model, but rather reshaping the supervisory signals for medical alignment.

### Key Designs

1.  **Three-Component Clinical Rubric and Safety Veto**:
    - **Function**: Decomposes medical response quality into interpretable and constrainable dimensions.
    - **Mechanism**: Proficiency $S_1$ measures fundamental clinical accuracy and completeness; Excellence $S_2$ rewards attributes beyond the passing line, such as empathy and logical clarity; Safety $S_3$ detects severe hallucinations, harmful advice, or out-of-bounds behavior. Final preferences are determined not by simple summation but by first comparing safety violations, then proficiency, and finally excellence (lexicographical comparison).
    - **Design Motivation**: Medical scenarios should not allow "very helpful but seriously unsafe" responses to win. Lexicographical comparison makes safety a hard constraint.

2.  **Human-in-the-Loop (HITL) Rubric Data Construction**:
    - **Function**: Balances scalable generation with professional physician verification.
    - **Mechanism**: ProMedical-Preference-50k undergoes source filtering, semantic deduplication, difficulty screening, and expert-guided classification before multiple strong models generate candidate responses. Rubric construction uses Gemini-3-Pro-thinking combined with static expert system instructions and dynamic few-shot examples. Physicians review 500 entries per round and re-inject corrected gold standards into the example pool.
    - **Design Motivation**: Fully manual rubric writing is too costly, while fully automatic generation is prone to medical hallucinations. Iterative HITL ensures generation quality converges; the authors report a 96.40% pass rate in strict expert evaluations.

3.  **Reward Model with Explicit Criteria Injection**:
    - **Function**: Enables the reward model to learn how to "compare two responses under a specific criterion."
    - **Mechanism**: While traditional reward models learn $P(y_w \succ y_l|x)$, this work modifies it to $P(y_w \succ y_l|x,c)$, where $c$ is a specific rubric criterion. A response pair is expanded into multiple criterion-conditioned training instances, each labeled with preferences for that dimension.
    - **Design Motivation**: Scalar rewards tend to conflate safety, professionalism, and expression quality. Criterion-conditioned training explicitly unbundles supervisory signals, which are later aggregated hierarchically (safety veto, proficiency, excellence).

### Loss & Training
The reward model uses a Bradley-Terry style pairwise loss, taking instruction, candidate responses, and criteria as input to optimize the criterion-conditioned reward margin. During the policy alignment phase, ProMedical-RM serves as a proxy oracle to calculate hierarchical rewards for the GRPO sampled outputs of Qwen3-8B. The safety violation penalty coefficient is set high enough to override any positive utility, ensuring safety issues are not neutralized by other dimensions.

## Key Experimental Results

### Main Results
ProMedical-Bench includes 795 held-out samples expanded into 5,505 criterion-level pairs: 3,625 for Proficiency, 1,650 for Excellence, and 230 for Safety. Double-blind physician adjudication yielded a weighted Cohen's Kappa of 0.88.

| Model | Pointwise Proficiency | Pointwise Safety | Pairwise Safety | Overall Accuracy |
|------|-----------------------|------------------|-----------------|------------------|
| GPT-5 | 91.50 | 76.45 | 77.39 | 76.42 |
| Gemini-3-Pro | 89.80 | 64.10 | 65.65 | 64.80 |
| DeepSeek-R1 | 89.50 | 78.80 | 80.00 | 78.55 |
| Qwen3-8B | 50.15 | 62.79 | 65.64 | 64.30 |
| PairRM-LLaMA3-8B | 76.50 | 58.80 | 60.43 | 58.95 |
| medical_o1_verifier_3B | 75.20 | 51.90 | 53.04 | 51.10 |
| ProMedical-RM-8B (Llama) | 90.15 | 87.20 | 86.10 | 85.40 |
| ProMedical-RM-8B (Qwen3) | 90.85 | 88.50 | 87.39 | 86.55 |

### Ablation Study

| Model | Safety Precision | Safety Recall | Safety F1 | Notes |
|------|------------------|---------------|-----------|------|
| GPT-5 | 79.24 | 73.85 | 76.45 | Strong closed-source models still miss some safety vetoes |
| DeepSeek-R1 | 81.50 | 76.28 | 78.80 | Strong open-source reasoning model, but lower than ProMedical-RM |
| PairRM-LLaMA3-8B | 62.45 | 59.80 | 61.10 | Tends to confuse safety with textual fluency |
| medical_o1_verifier_3B | 55.30 | 50.80 | 52.95 | Significant lack of recall |
| ProMedical-RM (Llama) | 89.40 | 85.10 | 87.20 | Fine-grained supervision brings stable improvements |
| ProMedical-RM (Qwen3) | 91.50 | 86.80 | 89.09 | Best Safety Veto detection |

### External Generalization and Policy Alignment

| Method | Q | Q+Criteria | Q+Sub | Conclusion |
|------|---|------------|-------|------|
| Ultra-Medical | 80.53 | - | - | Standard preference optimization baseline |
| RaR | 79.03 | 80.10 | 81.32 | Rubric-related baseline |
| InfiMed-ORBIT | 80.85 | 81.07 | 81.63 | Fine-grained preference baseline |
| ProMedical | 81.94 | 82.32 | 83.60 | Superior across all granularities |
| ProMedical-RAG | 81.60 | 83.20 | 84.28 | Q+Sub is optimal with external medical knowledge enhancement |

### Key Findings
- ProMedical-RM-8B (Qwen3) achieved an Overall Accuracy of 86.55%, surpassing GPT-5 (76.42%) and DeepSeek-R1 (78.55%), indicating that specialized rubric-aware reward models can outperform strong general-purpose models on fine-grained clinical standards.
- The Llama backbone version reached 85.40%, only 1.2 points lower than the Qwen3 version, proving the gain primarily stems from explicit criteria injection rather than the backbone capacity.
- Meditron-70B's Overall Accuracy was only 53.40%, suggesting that parameter scale and medical pre-training do not automatically lead to safety constraint adherence.
- Safety Veto F1 improved from 76.45 (GPT-5) to 89.09 (ProMedical-RM Qwen3), with gains concentrated in high-risk medical boundary identification.

## Highlights & Insights
- The most critical contribution is shifting clinical rubrics from the evaluation end to the training end. Medical alignment is not just about having "more preference data"; preference labels must have explicit clinical justifications.
- Treating safety as a veto rather than a soft penalty is crucial. Many general alignment methods allow dimensions to offset each other, but in medical contexts, a single severe hallucination is enough to invalidate a response.
- The double-blind physician adjudication and 0.88 Kappa of ProMedical-Bench enhance the credibility of the benchmark and the significance of the reward model's improvements.
- The concept of criteria-conditioned reward models can be transferred to other high-risk domains like law, finance, and education: first decompose standards into explicit criteria, then train the model to evaluate according to those standards.

## Limitations & Future Work
- The framework depends on expert consensus. In medical issues with controversy, inconsistent guidelines, or significant regional differences, the rubric itself may be difficult to define.
- Current work only handles text modality, failing to cover images, lab results, vital signs, and structured medical records common in real clinical workflows.
- The HITL pipeline remains costly; while more scalable than purely manual efforts, each new specialty or regional standard may require recalibration.
- While the reward model guides generation, the final outputs may still produce medical hallucinations; real-world deployment necessitates human physician oversight.
- Benchmark and data construction rely on strong models for candidate generation and initial rubric drafts, requiring continuous monitoring of model bias on data distribution.

## Related Work & Insights
- **vs UltraMedical**: UltraMedical provides large-scale medical preference data; ProMedical goes further by injecting fine-grained rubrics for each instruction and distinguishing between safety, proficiency, and excellence.
- **vs HealthBench**: HealthBench emphasizes physician-written evaluation rubrics; this paper applies similar concepts to training reward models and GRPO alignment.
- **vs General Reward Models**: Models like PairRM can learn general preferences but fail to reliably handle medical safety vetoes. ProMedical-RM's advantage comes from criterion-conditioned supervision.

## Rating
- Novelty: ⭐⭐⭐⭐ Explicitly injecting instruction-specific rubrics into reward models is a solid design for high-risk alignment.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage of datasets, benchmarks, reward models, safety metrics, and external generalization.
- Writing Quality: ⭐⭐⭐⭐ Clear methodological pipeline and dense tabular information; some formula layouts are slightly complex.
- Value: ⭐⭐⭐⭐⭐ Directly valuable for medical LLM alignment and interpretable reward modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Calibrated? Not for Everyone: How Sexual Orientation and Religious Markers Distort LLM Accuracy and Confidence in Medical QA](calibrated_not_for_everyone_how_sexual_orientation_and_religious_markers_distort.md)
- [\[ACL 2026\] BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels](biohicl_hierarchical_multi-label_contrastive_learning_for_biomedical_retrieval_w.md)
- [\[ACL 2026\] HypEHR: Hyperbolic Modeling of Electronic Health Records for Efficient Question Answering](hypehr_hyperbolic_modeling_of_electronic_health_records_for_efficient_question_a.md)
- [\[ACL 2026\] Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction](efficient_and_effective_internal_memory_retrieval_for_llm-based_healthcare_predi.md)
- [\[ICML 2026\] ClinTutor-R1: Advancing Scalable and Robust One-to-Many Alignment in Clinical Socratic Education](../../ICML2026/medical_nlp/clintutor-r1_advancing_scalable_and_robust_one-to-many_alignment_in_clinical_soc.md)

</div>

<!-- RELATED:END -->
