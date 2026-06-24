---
title: >-
  [Paper Note] CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of Large Language Models in Mental Health Question Answering
description: >-
  [ICLR2026][Medical LLM][Mental Health QA] The authors collaborated with 100 licensed mental health professionals to construct CounselBench, a dual-component benchmark for **open-ended mental health QA**. It includes 2,000 expert evaluations with dimension-level scoring and span annotations (CounselBench-Eval), and 120 clinician-authored adversarial prompts designed to induce specific failure modes (CounselBench-Adv). The study reveals that LLMs currently exhibit "high scores…
tags:
  - "ICLR2026"
  - "Medical LLM"
  - "Mental Health QA"
  - "Expert Evaluation"
  - "Adversarial Benchmarking"
  - "LLM-as-Judge"
  - "Open-ended Generation"
date: 2026-05-08
content_hash: bb600dba0e69768c
---

# CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of Large Language Models in Mental Health Question Answering

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=8MBYRZHVWT](https://openreview.net/forum?id=8MBYRZHVWT)  
**Code**: https://github.com/llm-eval-mental-health/CounselBench  
**Area**: NLP Understanding / Medical QA / LLM Evaluation  
**Keywords**: Mental Health QA, Expert Evaluation, Adversarial Benchmarking, LLM-as-Judge, Open-ended Generation

## TL;DR
The authors collaborated with 100 licensed mental health professionals to construct CounselBench, a dual-component benchmark for **open-ended mental health QA**. It includes 2,000 expert evaluations with dimension-level scoring and span annotations (CounselBench-Eval), and 120 clinician-authored adversarial prompts designed to induce specific failure modes (CounselBench-Adv). The study reveals that LLMs currently exhibit "high scores alongside persistent safety hazards" in counseling scenarios and demonstrates that LLM-as-Judge is unreliable in this high-risk domain.

## Background & Motivation
**Background**: Most medical QA benchmarks (e.g., MedQA, MedMCQA) consist of multiple-choice or factual tasks, measuring factual recall. However, real patients ask open-ended questions—lacking single correct answers, containing ambiguous descriptions, and mixing symptoms with emotional needs.

**Limitations of Prior Work**: Mental health QA is uniquely subjective and context-dependent. Effective responses must balance empathy, actionable advice, and professional boundaries. Furthermore, digital mental health services (e.g., CounselChat, peer forums, EHR messaging) are often **single-turn asynchronous** interactions. A single instance of boundary-crossing medical advice or an insensitive tone can cause immediate harm. Existing evaluations either use multiple-choice proxies or small expert groups/LLM-as-Judge, where the former lacks ecological validity and the latter misses critical clinical failures.

**Key Challenge**: Reliably evaluating LLM counseling responses requires **large-scale, deep clinical expert involvement** in the evaluation protocol—however, expert annotation is expensive and difficult to scale, creating a dilemma for previous benchmarks.

**Goal**: (1) Define clinically grounded evaluation dimensions for open-ended mental health QA; (2) Collect large-scale expert ratings and rationales for real-world responses; (3) Proactively construct adversarial prompts to elicit systematic failure modes rather than relying on post-hoc discovery.

**Key Insight**: Since real-world platforms are single-turn and CounselChat contains public responses from licensed therapists, "Model vs. Human" responses can be blindly evaluated on the same patient questions. Furthermore, clinicians can reverse-engineer prompts based on observed failures.

**Core Idea**: Establish a practitioner-anchored evaluation framework by replacing multiple-choice proxies and LLM self-evaluation with a "100 clinical experts × multi-dimensional blind review + clinician-led adversarial construction" approach.

## Method

### Overall Architecture
CounselBench is a **two-component benchmark + an evaluation protocol**. The pipeline consists of four steps: ① Selecting 100 patient questions from CounselChat (covering 20 themes), each paired with responses from three LLMs (GPT-4, LLaMA-3.3, Gemini-1.5-Pro) and one human therapist; ② Designing a six-dimensional clinical rubric to recruit 100 licensed/trainee professionals for blind review, resulting in 2,000 evaluations with span-level annotations and written rationales (CounselBench-Eval); ③ Using nine LLMs as judges to re-evaluate the same responses using the same rubric to analyze human-AI alignment; ④ Extracting fine-grained failure modes from Eval and tasking 10 clinicians to author 120 adversarial prompts, collecting 1,080 model responses for expert verification (CounselBench-Adv).

### Key Designs

**1. Six-Dimensional Clinical Rubric: Decomposing "Good Responses" into Quality and Safety**  
The authors decomposed response quality into six clinically-grounded dimensions: **Overall Quality**, **Empathy**, **Specificity** (context-sensitivity), **Factual Consistency**, **Medical Advice** (boundary-crossing clinical recommendations), and **Toxicity**. The first three map to the "therapeutic alliance," while the latter three address safety risks. Scores utilize 5-point Likert scales, with Medical Advice using a binary (Yes/No) label paired with span extraction and rationale, allowing for granular analysis of unauthorized prescriptions or therapy recommendations.

**2. Large-Scale Blind Evaluation Protocol: 100 Experts and 5-Fold Independent Labeling**  
To ensure clinical validity at scale, 100 mental health practitioners (covering 32 license types/degrees) were recruited. Each annotator reviewed 5 questions, each with 4 responses (1 human + 3 LLMs, randomized). Responses for the same question were rated by the same group to support fair comparison, while each QA pair received 5 independent expert ratings to ensure inter-rater agreement. All ratings were double-blind. This resulted in $100 \times 4 \times 5 = 2000$ annotations with a median rationale length of 576.5 words. Inter-rater agreement reached Krippendorff's $\alpha \geq 0.7$ (mostly ~0.82).

**3. LLM-as-Judge Benchmarking: Testing Model Self-Evaluation**  
The authors tested whether LLMs could replace human judgment by having 9 models re-evaluate the same QA pairs. Findings include: ① LLM judges consistently assigned **inflated scores**, particularly for Factual Consistency; ② LLM judges were insensitive to Toxicity, failing to identify potential harm flagged by experts; ③ LLM preferences diverged significantly from experts (except GPT-5); ④ Models failed at sentence-level span extraction for medical advice or factual errors.

**4. CounselBench-Adv: Reverse-Engineering Adversarial Prompts from Observed Failures**  
The authors refined failures into six specific modes: GPT-4 often provides **specific medication** and **therapy techniques**; LLaMA-3.3 tends to **speculate on medical symptoms** and adopt a **judgmental** tone; Gemini-1.5-Pro exhibits **apathy** and relies on **groundless assumptions**. 10 clinicians then authored 120 prompts designed to induce these specific errors. 9 LLMs generated 1,080 responses, which were then verified by a different set of 5 experts.

## Key Experimental Results

### Main Results: Expert Six-Dimensional Scoring (CounselBench-Eval)

| Source | Overall↑(1-5) | Empathy↑ | Specificity↑ | Medical Advice(%Yes) | Factual↑(1-4) | Toxicity↓ |
|----------|------|------|------|------|------|------|
| GPT-4 | 3.28 | 3.37 | 3.46 | 0.07 | 3.53 | 1.78 |
| **LLaMA-3.3** | **4.29** | **4.22** | **4.63** | 0.14 | **3.70** | **1.36** |
| Gemini-1.5-Pro | 3.26 | 2.76 | 3.50 | 0.08 | 3.52 | 1.64 |
| Human Therapist | 2.60 | 2.72 | 3.29 | 0.17 | 2.92 | 2.56 |

LLaMA-3.3 led in five out of six dimensions. However, **14%** of its responses were flagged for providing unauthorized medical advice. Humans scored lower in quality dimensions, likely due to the varied and concise nature of forum-based responses.

### Ablation Study: Failure Mode Trigger Rates (CounselBench-Adv)

| Failure Mode | GPT-3.5 | GPT-4 | GPT-5 | Llama-3.1 | Llama-3.3 | Claude-3.5 | Claude-3.7 | Gemini-1.5 | Gemini-2.0 |
|----------|------|------|------|------|------|------|------|------|------|
| 1. Medication | 0.05 | 0 | **0.47** | 0.05 | 0.10 | 0 | 0 | 0 | 0 |
| 2. Therapy Tech | 0.20 | 0.20 | **0.85** | 0.55 | 0.65 | 0.45 | 0.50 | 0.20 | 0.26 |
| 3. Speculate Symptoms | 0.15 | 0.45 | 0.60 | 0.45 | 0.45 | 0.50 | 0.37 | 0.26 | 0.25 |
| 4. Judgmental | 0.25 | 0.25 | 0.05 | 0.11 | 0.10 | 0.05 | 0.10 | 0.20 | 0.10 |
| 5. Apathy | **0.70** | 0.20 | 0.15 | 0.15 | 0.15 | 0.05 | 0.20 | 0.40 | 0.30 |
| 6. Assumptions | 0.40 | 0.35 | 0.15 | 0.25 | 0.25 | 0.35 | 0.25 | 0.40 | 0.35 |

Adversarial prompts successfully induced failures: GPT-5 reached a 0.85 rate for unauthorized therapy advice and a surprisingly high 0.47 for medication recommendations.

### Key Findings
- **Model Family Failure Profiles**: Models within the same family (LLaMA/Gemini/Claude) exhibit similar failure distributions, while GPT profiles are unique.
- **Unreliable LLM Judges**: Even with in-context examples, the best detector (Claude-3.7) achieved an F1 of only **0.50** for failure mode detection.
- **High Scores ≠ Safety**: Models can achieve high quality scores while consistently providing non-constructive feedback, over-generalization, and boundary-crossing medical advice.

## Highlights & Insights
- **Empirical Adversarial Paradigm**: CounselBench-Adv focuses on "empirically-grounded" failures extracted from expert annotations rather than literature-driven categories, ensuring prompts mirror real-world model vulnerabilities.
- **Quality/Safety Dual Axis**: Decoupling quality and safety measurement enables the data to be used for both alignment tuning and the development of safety detectors.
- **Caution on LLM Judges**: The failure of LLMs to self-audit in high-risk subjective domains serves as a warning against replacing human evaluation without rigorous verification.

## Limitations & Future Work
- **Single-Turn Limitation**: The benchmark does not cover multi-turn properties like context tracking or consistency.
- **Data Source**: Using CounselChat high-voted answers may not reflect the full capabilities of clinical face-to-face sessions.
- **Model Volatility**: Rapid updates to models (e.g., GPT-5, Gemini-2.0) mean the specific failure rates are a moving target.
- **Privacy Constraints**: Scarcity of open clinical data limits the benchmark's expansion to wider, more sensitive clinical contexts.

## Related Work & Insights
- **vs. MultiMedQA / HealthBench**: While those focus on structured medical knowledge, CounselBench focuses on the empathy and boundary management critical to mental health.
- **vs. MCQ Benchmarks**: This work addresses the ambiguity of free-text generation instead of relying on objective answer keys.
- **vs. Traditional Red-Teaming**: Instead of pre-defined failure modes, CounselBench utilizes prospective clinician-authored prompts based on observed model behavior.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA](counselbench_llm_mental_health_qa.md)
- [\[ICLR 2026\] MedAraBench: Large-scale Arabic Medical Question Answering Dataset and Benchmark](medarabench_large-scale_arabic_medical_question_answering_dataset_and_benchmark.md)
- [\[ICLR 2026\] Cancer-Myth: Evaluating Large Language Models on Patient Questions with False Presuppositions](cancer-myth_evaluating_large_language_models_on_patient_questions_with_false_pre.md)
- [\[ACL 2026\] MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models](../../ACL2026/medical_nlp/mhgraphbench_knowledge_graph-grounded_benchmarking_of_mental_health_knowledge_in.md)
- [\[ACL 2026\] MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models](../../ACL2026/medical_nlp/mhsafeeval_role-aware_interaction-level_evaluation_of_mental_health_safety_in_la.md)

</div>

<!-- RELATED:END -->
