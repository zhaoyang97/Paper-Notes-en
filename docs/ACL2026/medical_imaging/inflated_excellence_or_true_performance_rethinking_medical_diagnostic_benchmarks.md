---
title: >-
  [Paper Note] Inflated Excellence or True Performance? Rethinking Medical Diagnostic Benchmarks with Dynamic Evaluation
description: >-
  [ACL 2026][Medical Imaging][Medical Diagnostic Benchmarks] This paper proposes DyReMe, a dynamic medical diagnostic evaluation framework. Its DyGen module generates novel diagnostic cases containing clinically grounded d…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "Medical Diagnostic Benchmarks"
  - "Dynamic Evaluation"
  - "Data Contamination"
  - "Diagnostic Distractors"
  - "LLM Trustworthiness"
date: 2026-05-08
content_hash: c8c83e82fa54436c
---

# Inflated Excellence or True Performance? Rethinking Medical Diagnostic Benchmarks with Dynamic Evaluation

**Conference**: ACL 2026
**arXiv**: [2510.09275](https://arxiv.org/abs/2510.09275)  
**Code**: [Official Repository](https://arxiv.org/abs/2510.09275)  
**Area**: Medical Imaging
**Keywords**: Medical Diagnostic Benchmarks, Dynamic Evaluation, Data Contamination, Diagnostic Distractors, LLM Trustworthiness

## TL;DR

This paper proposes DyReMe, a dynamic medical diagnostic evaluation framework. Its DyGen module generates novel diagnostic cases containing clinically grounded distractors—including differential diagnoses and misdiagnosis factors—while the EvalMed module assesses LLMs across four dimensions: accuracy, veracity, helpfulness, and consistency. The results reveal that existing static benchmarks systematically overestimate LLM diagnostic capability; GPT-5 suffers an 8.25% accuracy drop on DyReMe, and all 12 evaluated LLMs exhibit significant trustworthiness deficiencies.

## Background & Motivation

**Background**: LLMs have demonstrated considerable promise in medical diagnostic assistance, with capabilities spanning clinical case analysis, pattern recognition, and decision support. Static benchmarks derived from medical licensing examinations—such as MedBench and C-Eval—have been widely adopted to evaluate these capabilities, with test items remaining fixed across models and time points.

**Limitations of Prior Work**: Static benchmarks suffer from two fundamental problems. First, data contamination leads to inflated performance estimates: because many benchmarks are publicly available and static, LLMs may have encountered test items during training, meaning high scores may reflect memorization rather than generalizable reasoning. Second, these benchmarks are misaligned with real-world clinical scenarios: exam-style benchmarks rely on standardized, well-formed case descriptions and accuracy-only evaluation protocols, whereas real patient queries are often incomplete, expressed in colloquial language, and confounded by self-diagnosis and other factors that can mislead clinical decision-making.

**Key Challenge**: Existing dynamic evaluation approaches—such as paraphrasing or adding noise to existing questions—reduce data contamination but apply only surface-level transformations that preserve the underlying clinical setup. They therefore fail to address the real-world alignment problem and continue to focus exclusively on accuracy.

**Goal**: (1) Generate novel diagnostic cases containing clinically grounded distractors that combine differential diagnoses with misdiagnosis factors; (2) Establish a multi-dimensional trustworthiness evaluation framework that goes beyond accuracy.

**Key Insight**: Four empirically documented misdiagnosis factors in clinical practice—anchoring bias, posterior probability error, attention distraction, and symptom overestimation—are operationalized as four categories of diagnostic traps: self-diagnosis, interfering medical history, external noise, and symptom misattribution. These are injected into benchmark cases to simulate clinical complexity.

**Core Idea**: Dynamic benchmark = differential diagnosis distractors + misdiagnosis factor traps + patient expression style diversification + four-dimensional trustworthiness evaluation.

## Method

### Overall Architecture

DyReMe consists of two components. DyGen (the dynamic generation module) generates challenging and diverse diagnostic questions, while EvalMed (the evaluation module) assesses LLM diagnostic performance across four dimensions. The DyGen pipeline proceeds as follows: original question → retrieval of differential diagnoses → injection of diagnostic traps → adaptation to patient expression style → iterative validator–refiner loop for quality assurance.

### Key Designs

1. **DyGen Dynamic Generation Module**:

    - **Function**: Generates novel diagnostic questions containing clinical distractors and diversified expression styles.
    - **Mechanism**: A three-stage generation pipeline: (a) *Differential diagnosis retrieval*—RAG is used to retrieve similar diagnoses $d_{dis}$ for the original diagnosis $d_{org}$ (e.g., retrieving "adrenal adenoma" for "pheochromocytoma"); (b) *Misdiagnosis factor injection*—one trap type is sampled uniformly from the four diagnostic trap categories $\mathcal{S}$ and combined with the differential diagnosis to construct a misleading question $q_{trap} = \mathcal{T}_{trap}(q_{org}, s, d_{dis})$; (c) *Expression style adaptation*—an indirect persona adaptation mechanism $q_{per} = \mathcal{T}_{persona}(q_{trap}, b)$ first extracts the expressive characteristics of a persona (knowledge level, clarity, communication style) and then rewrites the question using these features, thereby avoiding the causal confounding that would arise from directly applying persona identities. A validator–refiner iterative loop then ensures quality.
    - **Design Motivation**: Differential diagnosis distractors introduce diagnostic ambiguity; misdiagnosis factor traps simulate genuine clinical pitfalls; expression style adaptation captures the heterogeneity of patient communication. Together, these features render the benchmark substantially closer to real-world diagnostic scenarios.

2. **EvalMed Four-Dimensional Evaluation Module**:

    - **Function**: Evaluates LLM diagnostic trustworthiness across four clinically relevant dimensions beyond accuracy.
    - **Mechanism**: (a) *Accuracy*—Top-1/3/5 diagnostic hit rates; (b) *Veracity*—health misinformation is injected into questions to test whether LLMs can identify and correct false claims (e.g., "hypertension affects bone density"), measured as $\text{Ver}(M) = \frac{1}{|\mathcal{Q}|}\sum_{q} \mathbb{I}_r(q, \hat{a})$; (c) *Helpfulness*—three criteria based on standards from real medical consultation platforms (diagnostic rationale, treatment recommendations, lifestyle advice) are evaluated for coverage using a RAG-constructed scoring knowledge base; (d) *Consistency*—the normalized information entropy of diagnostic distributions across different variants of the same case is computed as $\text{Cons}(M) = \frac{1}{|\mathcal{P}|}\sum_{p_i}(1 - E_{p_i}/\log m)$.
    - **Design Motivation**: Exclusive focus on accuracy obscures critical deficiencies in real clinical deployment: failure to correct health misinformation risks spreading false beliefs, superficial responses reduce diagnostic utility, and inconsistent conclusions across different phrasings of the same case erodes patient trust.

3. **Validator–Refiner Iterative Loop**:

    - **Function**: Ensures the clinical validity and challenge level of generated questions.
    - **Mechanism**: The validator $\mathcal{V}$ assesses candidate questions along four dimensions—challengingness, logical consistency, symptom accuracy, and trap validity. Questions that pass all criteria are accepted; otherwise, they are returned to the refiner $\mathcal{R}$, which revises them according to the feedback. This process iterates until all constraints are satisfied.
    - **Design Motivation**: Questions generated directly by LLMs may exhibit logical inconsistencies or poorly designed traps; iterative refinement ensures final quality.

### Loss & Training

DyReMe does not involve model training. DyGen uses GPT-4.1 as the generator (generation temperature 0.7, validation temperature 0). Evaluation expands 800 cases from DxBench to 3,200 questions. RAG employs the Volcengine Search API and Douyin Baike. To mitigate potential self-recognition effects, GPT-4.1 is excluded from evaluation.

## Key Experimental Results

### Main Results

**Diagnostic Accuracy on Static vs. Dynamic Benchmarks (Top-1/3/5 Average)**

| Model | Static Avg. | DyVal2 (Δ) | DyReMe (Δ) |
|-------|------------|-----------|-----------|
| GPT-5 | 73.76 | 70.73 (-4.11%) | **67.67 (-8.25%)** |
| DeepSeek-V3 | 72.92 | 69.50 (-4.69%) | **65.26 (-10.51%)** |
| GPT-4o | 72.53 | 69.67 (-3.94%) | **64.74 (-10.75%)** |
| MedGemma-27B | 70.56 | 67.70 (-4.06%) | **62.97 (-10.76%)** |
| Qwen3-32B | 73.62 | 68.28 (-1.98%) | **63.85 (-8.34%)** |
| Qwen2.5-7B | 67.85 | 65.25 (-3.82%) | **57.86 (-14.71%)** |

**Cross-lingual Validation (English DDXPlus)**

| Model | DDXPlus | DyReMe | p-value |
|-------|---------|--------|---------|
| GPT-4o | 85.10 | 77.18 | <0.05 |
| Qwen2.5-32B | 72.58 | 65.24 | <0.05 |

### Ablation Study

**DyGen Component Ablation (Challengingness and Diversity)**

| Configuration | Challengingness | Expression Diversity | Diagnostic Diversity |
|--------------|----------------|---------------------|---------------------|
| DyReMe (full) | Highest | Highest | Highest |
| w/o diagnostic distractors | Significant drop | Unchanged | Significant drop |
| w/o patient expression style | Drop | Significant drop | Unchanged |

### Key Findings

- DyReMe induces larger performance drops across all LLMs; even GPT-5—which outperforms the generator GPT-4.1—declines by 8.25%, demonstrating that the benchmark remains challenging for frontier models.
- The medical-domain model WiNGPT2-9B achieves the lowest score (31.8), suggesting that current medical adaptation may capture medical facts but fails to handle real-world distractors and diverse patient expressions.
- Reasoning-oriented models (o1/o1-mini) perform only moderately (37.0/36.7), as their training emphasizes identifying single correct answers rather than handling misinformation or providing actionable guidance.
- Across all models, 20–40% of health misinformation items go uncorrected, posing a risk of misinformation propagation; consistency scores are uniformly low, indicating fragility to variations in input context.
- DyReMe demonstrates substantially superior scalability compared to existing dynamic methods: Self-BLEU decreases more slowly as $k$ increases, and the number of unique diagnoses grows continuously.

## Highlights & Insights

- The systematic operationalization of cognitive-psychological misdiagnosis factors (e.g., anchoring bias) into four categories of diagnostic traps establishes a meaningful bridge between cognitive science and NLP at the level of evaluation design.
- The indirect persona adaptation mechanism is particularly elegant: extracting expressive features rather than directly applying persona identities avoids the introduction of confounds such as "miner → pneumoconiosis."
- The four-dimensional evaluation framework has direct practical value for real-world medical AI deployment: it shifts the question from "is the answer correct?" to also ask "does the model correct misinformation?", "is the advice actionable?", and "are responses stable across paraphrases?"

## Limitations & Future Work

- The primary experiments are conducted on Chinese-language datasets, with only a single cross-lingual validation on English data; generalization to multilingual settings requires further investigation.
- The framework addresses only text-based diagnostic scenarios and does not incorporate multimodal inputs such as medical imaging or laboratory results.
- End-to-end clinical workflows—including longitudinal patient histories and multidisciplinary decision-making—are not covered, and clinical trials are needed to validate findings.
- Self-bias is mitigated but not fully eliminated; using different LLMs as generators and evaluators may introduce distinct systematic biases.

## Related Work & Insights

- **vs. DyVal2**: DyVal2 performs dynamic evaluation via noise injection and paraphrasing, but transformations remain at the surface level. DyReMe introduces deep clinical distractors, yielding performance drops approximately twice as large as those produced by DyVal2.
- **vs. Self-Evolving**: When perturbations are weak, some models (e.g., GPT-4o-mini) may score higher on dynamic benchmarks than on static ones. DyReMe ensures consistently elevated challenge levels.
- **vs. MedBench/C-Eval**: Static benchmarks are inherently susceptible to data contamination. DyReMe fundamentally addresses this problem by dynamically generating entirely new cases.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Systematically translates clinical misdiagnosis factors into dynamic evaluation design; the four-dimensional evaluation framework represents a meaningful contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 12 LLMs, multiple static and dynamic baselines, ablation studies, scalability analysis, cross-lingual validation, and human agreement studies.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated and the methodology is systematically described, though some notation definitions are scattered across sections.
- **Value**: ⭐⭐⭐⭐⭐ Exposes fundamental flaws in the evaluation of medical LLMs and points toward a more realistic paradigm for medical AI assessment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning](dr_assistant_enhancing_clinical_diagnostic_inquiry_via_structured_diagnostic_rea.md)
- [\[ACL 2026\] Can Continual Pre-training Bridge the Performance Gap between General-purpose and Specialized Language Models in the Medical Domain?](can_continual_pre-training_bridge_the_performance_gap_between_general-purpose_an.md)
- [\[AAAI 2026\] Multivariate Gaussian Representation Learning for Medical Action Evaluation](../../AAAI2026/medical_imaging/multivariate_gaussian_representation_learning_for_medical_action_evaluation.md)
- [\[AAAI 2026\] MedEyes: Learning Dynamic Visual Focus for Medical Progressive Diagnosis](../../AAAI2026/medical_imaging/medeyes_learning_dynamic_visual_focus_for_medical_progressive_diagnosis.md)
- [\[ICLR 2026\] Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series](../../ICLR2026/medical_imaging/decentralized_attention_fails_centralized_signals_rethinking_transformers_for_me.md)

</div>

<!-- RELATED:END -->
