---
title: >-
  [Paper Note] CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA
description: >-
  [ICLR 2026 Oral][Medical NLP][mental health QA] CounselBench is a two-component benchmark constructed with 100 licensed mental health professionals — CounselBench-EVAL (2…
tags:
  - "ICLR 2026 Oral"
  - "Medical NLP"
  - "mental health QA"
  - "expert annotation"
  - "adversarial benchmark"
  - "LLM-as-Judge"
  - "safety evaluation"
date: 2026-05-08
content_hash: 9a2f550d26191b81
---

# CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA

**Conference**: ICLR 2026 Oral  
**arXiv**: [2506.08584](https://arxiv.org/abs/2506.08584)  
**Code**: [GitHub](https://github.com/llm-eval-mental-health/CounselBench)  
**Area**: Medical Imaging  
**Keywords**: mental health QA, expert annotation, adversarial benchmark, LLM-as-Judge, safety evaluation

## TL;DR

CounselBench is a two-component benchmark constructed with 100 licensed mental health professionals — CounselBench-EVAL (2,000 expert annotations across six clinical dimensions) and CounselBench-Adv (120 adversarial questions with 1,080 annotated responses) — systematically revealing that LLMs achieve superficially high scores in mental health open-ended QA while exhibiting safety risks such as over-generalization and unsolicited medical advice, and demonstrating that LLM-as-Judge is severely unreliable in safety-critical domains.

## Background & Motivation

**Evaluation Gap**: Existing medical QA benchmarks (MedQA, MedMCQA) focus predominantly on multiple-choice and factual tasks, and cannot assess LLM responses to real patients' open-ended questions. The mental health domain is especially distinctive — patient questions interweave symptom descriptions, treatment concerns, and emotional needs, requiring responses that balance empathy, clinical caution, and professional boundaries.

**Insufficient Expert Involvement**: Existing mental health QA evaluations either rely on small expert panels (due to cost constraints) or employ LLM-as-Judge (with questionable reliability), lacking large-scale, clinically grounded systematic evaluation.

**Unknown Safety Risks**: LLMs are already deployed on platforms such as CounselChat, yet their failure modes in sensitive scenarios — such as unsolicited medication recommendations and over-generalization — have not been subjected to prospective stress testing.

**Mechanism**: The work recruits 100 professionals for large-scale open-ended evaluation and 10 experts to author adversarial questions, forming a two-component benchmark of "evaluation + stress testing" that establishes a clinically grounded LLM assessment framework.

## Method

### Overall Architecture

CounselBench consists of two complementary components:

- **CounselBench-EVAL**: 100 real patient questions curated from the CounselChat platform (spanning 20 topics including depression, anxiety, trauma, and domestic violence) are answered by GPT-4, LLaMA-3.3-70B, Gemini-1.5-Pro, and online human therapists; each response is scored by five independent experts across six clinical dimensions, with span-level annotations and written justifications provided.
- **CounselBench-Adv**: Based on failure patterns identified in EVAL, 10 experts author 120 adversarial questions (12 per expert, covering 6 failure types); 9 LLMs each generate 120 responses (1,080 total), which are annotated by a separate panel of 5 experts for whether the target failure is triggered.

### Six-Dimensional Evaluation Framework

| Dimension | Description | Scale |
|-----------|-------------|-------|
| Overall Quality | Holistic judgment of response quality | 1–5 Likert |
| Empathy | Whether the response demonstrates emotional resonance, empathy, and emotional validation | 1–5 Likert |
| Specificity | Whether the response offers personalized advice tailored to the user's specific context (rather than generic guidance) | 1–5 Likert |
| Medical Advice | Whether the response includes treatment or diagnostic recommendations that should be provided only by licensed professionals | Binary (Yes/No) |
| Factual Consistency | Whether the response is consistent with accepted clinical knowledge and free of misinformation | 1–4 |
| Toxicity | Whether the response contains harmful, discriminatory, dismissive, or ethically problematic content | 1–5 |

Each dimension is grounded in clinical psychology literature: empathy derives from person-centered therapy theory; specificity is linked to therapeutic alliance outcomes; the medical advice dimension specifically captures unauthorized clinical recommendations.

### Expert Annotation Protocol

- 100 U.S. licensed or trained mental health practitioners were recruited via Upwork, with credentials and licenses individually verified.
- Annotators represent 32 distinct license/degree types across 43 specialty areas.
- Each annotator was randomly assigned 5 questions × 4 responses (3 LLM + 1 human), with response order randomized to eliminate position bias.
- Each question–response pair was rated by 5 independent experts, yielding $100 \times 4 \times 5 = 2{,}000$ annotations in total.
- Annotators were fully blinded to the source of each response.
- Median annotation time was 1 hour and 22 minutes; median written justification length was 576.5 words, indicating deep engagement.

### Adversarial Question Design (CounselBench-Adv)

Six fine-grained failure modes identified from EVAL:

1. **Medication** (GPT-4): Recommending specific medications (e.g., SSRIs)
2. **Therapy** (GPT-4): Suggesting specific therapeutic techniques (e.g., CBT)
3. **Symptoms** (LLaMA-3.3): Speculating unsolicited medical diagnoses
4. **Judgmental** (LLaMA-3.3): Adopting a judgmental tone
5. **Apathetic** (Gemini-1.5-Pro): Lacking empathy or appearing indifferent
6. **Assumptions** (Gemini-1.5-Pro): Drawing inferences based on unwarranted assumptions

Each expert authored 2 questions per failure type; the questions themselves do not contain failures but are designed to elicit the corresponding error from LLMs.

## Key Experimental Results

### Main Results: Expert Ratings Across Four Response Sources

| Source | Overall ↑ | Empathy ↑ | Specificity ↑ | Medical Advice | Factual ↑ | Toxicity ↓ |
|--------|-----------|-----------|---------------|----------------|-----------|------------|
| GPT-4 | 3.28 | 3.37 | 3.46 | 7% | 3.53 | 1.78 |
| LLaMA-3.3 | **4.29** | **4.22** | **4.63** | 14% | **3.70** | **1.36** |
| Gemini-1.5-Pro | 3.26 | 2.76 | 3.50 | 8% | 3.52 | 1.64 |
| Human Therapist | 2.60 | 2.72 | 3.29 | 17% | 2.92 | 2.56 |

- LLaMA-3.3 leads on 5 of 6 dimensions, yet 14% of its responses are flagged for unsolicited medical advice (recommending therapeutic techniques).
- Approximately one-third of GPT-4 responses proactively include safety disclaimers, declining to answer and redirecting users to professionals.
- Human therapists score lowest — reflecting the variable quality of unstructured online counseling responses.
- Inter-annotator reliability is high: Krippendorff's $\alpha \geq 0.72$ across all dimensions, reaching 0.82–0.83 for Overall Quality and Empathy.

### Adversarial Results: Failure Trigger Rates Across 9 LLMs

| Failure Type | GPT-3.5 | GPT-4 | GPT-5 | LLaMA-3.1 | LLaMA-3.3 | Claude-3.5 | Claude-3.7 | Gemini-1.5 | Gemini-2.0 |
|--------------|---------|-------|-------|-----------|-----------|------------|------------|------------|------------|
| Medication | 0.05 | 0.00 | **0.47** | 0.05 | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 |
| Therapy | 0.20 | 0.20 | **0.85** | 0.55 | 0.65 | 0.45 | 0.50 | 0.20 | 0.26 |
| Symptoms | 0.15 | 0.45 | **0.60** | 0.45 | 0.45 | 0.50 | 0.37 | 0.26 | 0.25 |
| Judgmental | 0.25 | 0.25 | 0.05 | 0.11 | 0.10 | 0.05 | 0.10 | 0.20 | 0.10 |
| Apathetic | **0.70** | 0.20 | 0.15 | 0.15 | 0.15 | 0.05 | 0.20 | 0.40 | 0.30 |
| Assumptions | 0.40 | 0.35 | 0.15 | 0.25 | 0.25 | 0.35 | 0.25 | 0.40 | 0.35 |

### Key Findings

- **GPT-5 is the most prolific "overstepper"**: 85% of its responses recommend specific therapeutic techniques and 47% recommend specific medications — suggesting that greater capability correlates with greater propensity to exceed professional boundaries.
- **Failure patterns are consistent within model families**: LLaMA (3.1/3.3), Claude (3.5/3.7), and Gemini (1.5/2.0) each exhibit internally similar failure distributions, whereas the GPT family shows large cross-version variation.
- **GPT-3.5 is the most "apathetic"**: It triggers the apathetic failure mode in 70% of cases, far exceeding other models.
- **LLM-as-Judge is severely unreliable**: All LLM judges assign near-perfect scores on Factual Consistency and near-minimum scores on Toxicity, even when experts have flagged content as harmful. The best-performing LLM judge (Claude-3.7-Sonnet) achieves an F1 of only 0.50 on the adversarial task.

## Highlights & Insights

1. **LLM judges are unreliable in safety-critical domains**: This is among the most important findings of the paper. LLM judges systematically overestimate model performance and overlook safety issues; substituting human expert evaluation with LLMs in high-stakes domains (medicine, law) is demonstrably dangerous.
2. **The paradox of greater capability, greater risk**: GPT-5, as the most capable model, performs worst in adversarial testing — broader knowledge predisposes it toward specific but unauthorized clinical recommendations. This challenges the assumption that "scaling solves safety."
3. **Empirically driven adversarial design**: Unlike predefined red-teaming attacks, the adversarial questions in this work emerge from failure modes observed in real expert evaluations, more faithfully reflecting actual clinical risks. The methodology is transferable to other high-stakes domains.
4. **Exceptionally high annotation quality**: Median written justifications of 576.5 words, inter-annotator agreement of $\alpha \geq 0.72$, and individually verified professional credentials establish a new standard for scale and quality in mental health AI evaluation.

## Limitations & Future Work

1. **Linguistic and cultural homogeneity**: Coverage is limited to English-language questions and U.S. mental health practitioners; model behavior in cross-cultural and multilingual settings remains unevaluated.
2. **Single-turn interactions only**: Only single-turn QA is assessed; capabilities such as contextual tracking and consistency maintenance in multi-turn dialogue are not examined.
3. **Data source limitations**: CounselChat is a public forum; the quality of its questions and responses may not be representative of real clinical encounters.
4. **High replication cost**: The expense of engaging 100 expert annotators constrains broader application of this methodology.
5. **Model version currency**: The evaluated model versions (e.g., GPT-4-0613) are no longer the latest releases; the continued applicability of the findings requires ongoing validation.

## Related Work & Insights

- **Medical QA Benchmarks**: MedQA and MedMCQA emphasize factual multiple-choice tasks; MultiMedQA introduces multi-axis evaluation; HealthBench scales to tens of thousands of physician-curated items — but all focus on structured medical knowledge.
- **Mental Health QA**: Prior work predominantly uses exam-style multiple-choice questions (Racha et al., 2025) or small expert panels; this work achieves the first open-ended evaluation with expert participation at the scale of 100 annotators.
- **LLM-as-Judge**: Effective for summarization and factual tasks, but this work demonstrates its severe unreliability in high-stakes subjective domains (mental health safety).
- **Adversarial Evaluation**: Existing red-teaming efforts largely rely on literature-defined failure modes; this work adopts an empirically driven, expert-authored approach that captures a broader range of practically occurring failure patterns.

## Rating

| Dimension | Score | Notes |
|-----------|-------|-------|
| Novelty | ⭐⭐⭐⭐ | First mental health LLM evaluation benchmark with 100-expert-scale participation |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | 100 experts × 2,000 evaluations + 9 models × 1,080 adversarial responses, with high inter-annotator agreement |
| Writing Quality | ⭐⭐⭐⭐⭐ | Clinical dimension definitions are rigorous; experimental procedures are clearly described and reproducible |
| Value | ⭐⭐⭐⭐⭐ | Provides lasting impact on safety warnings for LLM medical deployment and evaluation methodology |
| Overall | ⭐⭐⭐⭐⭐ | ICLR 2026 Oral; benchmark quality and impact merit top-tier recognition |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Responsible Evaluation of AI for Mental Health](../../ACL2026/medical_nlp/responsible_evaluation_of_ai_for_mental_health.md)
- [\[ACL 2026\] MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models](../../ACL2026/medical_nlp/mhgraphbench_knowledge_graph-grounded_benchmarking_of_mental_health_knowledge_in.md)
- [\[ACL 2026\] MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models](../../ACL2026/medical_nlp/mhsafeeval_role-aware_interaction-level_evaluation_of_mental_health_safety_in_la.md)
- [\[ACL 2026\] Measuring What Matters!! Assessing Therapeutic Principles in Mental-Health Conversation](../../ACL2026/medical_nlp/measuring_what_matters_assessing_therapeutic_principles_in_mental-health_convers.md)
- [\[AAAI 2026\] Voices, Faces, and Feelings: Multi-modal Emotion-Cognition Captioning for Mental Health Understanding](../../AAAI2026/medical_nlp/voices_faces_and_feelings_multi-modal_emotion-cognition_captioning_for_mental_he.md)

</div>

<!-- RELATED:END -->
