---
title: >-
  [Paper Note] Moving Beyond Medical Exams: A Clinician-Annotated Fairness Dataset of Real-World Tasks and Ambiguity in Mental Healthcare
description: >-
  [ICLR 2026][LLM Safety][mental healthcare] This paper introduces MENTAT—an evaluation dataset designed and annotated by 9 U.S. psychiatrists…
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "mental healthcare"
  - "fairness benchmark"
  - "clinical decision-making"
  - "demographic bias"
  - "expert annotation"
date: 2026-05-08
content_hash: 87a642ff170346b5
---

# Moving Beyond Medical Exams: A Clinician-Annotated Fairness Dataset of Real-World Tasks and Ambiguity in Mental Healthcare

**Conference**: ICLR 2026
**arXiv**: [2502.16051](https://arxiv.org/abs/2502.16051)  
**Code**: [GitHub](https://github.com/maxlampe/mentat) (MIT License)  
**Area**: Medical AI Evaluation / Psychiatry / Fairness
**Keywords**: mental healthcare, fairness benchmark, clinical decision-making, demographic bias, expert annotation

## TL;DR

This paper introduces MENTAT—an evaluation dataset designed and annotated by 9 U.S. psychiatrists, comprising 203 base questions (each with 5 answer choices) expanded via demographic variable substitution, covering 5 clinical practice domains: diagnosis, treatment, triage, monitoring, and documentation. By systematically substituting patient age, race, and gender, the benchmark evaluates decision-making bias across 22 language models, revealing significant and unpredictable accuracy disparities along demographic dimensions.

## Background & Motivation

**Background**: Medical AI evaluation has predominantly relied on licensure exam questions (MedQA, MMLU-Med, etc.), emphasizing factual knowledge recall. In psychiatry, however, diagnosis and management depend heavily on subjective judgment and interpersonal interaction, and standardized exam performance correlates only weakly with real-world clinical competence.

**Limitations of Prior Work**:

1. Exam-style questions focus on knowledge recall and fail to assess genuine clinical decision-making—tasks such as triage decisions, medication dose adjustments, and documentation that psychiatrists face daily are far more complex than multiple-choice questions.

2. Existing benchmarks lack deliberate design for ambiguity and uncertainty—many psychiatric decisions have no single correct answer (e.g., involuntary hospitalization judgments, emphasis choices in clinical summaries).

3. Fairness evaluation in medical AI is insufficient—the effect of patient demographic information (race, gender, age) on model decisions has not been systematically studied, yet may introduce systemic bias at scale.

4. Most existing datasets are generated with LM assistance (e.g., MedS-bench via web scraping and LM synthesis), introducing known quality and contamination concerns.

**Key Challenge**: There is a need for a psychiatric AI evaluation dataset that is entirely designed by human experts, captures genuine clinical ambiguity, and enables systematic assessment of demographic bias.

## Method

### Overall Architecture

Five psychiatrists design 203 base questions (5 options each) → demographic-irrelevant information is removed and replaced with variables (age, race, gender) → questions are expanded into multiple evaluation datasets ($\mathcal{D}_0$=183 base, $\mathcal{D}_G$=549 by gender, $\mathcal{D}_A$=915 by age, $\mathcal{D}_N$=1098 by race) → 8 expert annotators provide preference labels for triage and documentation questions → a hierarchical Bradley-Terry model generates preference probability labels.

### Key Designs

1. **Five-Domain Clinical Task Design**

    - **Diagnosis** (50 questions): Make DSM-5-TR diagnoses based on symptom information.
    - **Treatment** (47 questions): Formulate treatment plans, including specific medication dosages (typically absent from exam questions).
    - **Triage** (28 questions): Assess urgency and decide whether to escalate care—multiple reasonable answers exist.
    - **Monitoring** (49 questions): Evaluate treatment efficacy and illness severity.
    - **Documentation** (29 questions): Electronic health record documentation—multiple reasonable answers exist (e.g., how to summarize, how to code for billing).
    - Diagnosis, treatment, and monitoring have unique correct answers; triage and documentation are designed as ambiguous questions with multiple reasonable options and expert preference annotations.

2. **Hierarchical Bradley-Terry Preference Model**

    - 657 annotations are collected (average 11.5 per question) for 57 ambiguous triage/documentation questions; 8 experts independently score using a 0–100 scale.
    - Scores are converted to pairwise comparisons and a hierarchical Bradley-Terry model is fitted: $P(i \succ j | a) = \frac{1}{1 + \exp[-(\gamma_a + \alpha_a(\beta_i - \beta_j))]}$
    - Annotator-specific offset $\gamma_a$ and slope $\alpha_a$ are introduced to capture each expert's leniency or strictness.
    - Final preference probabilities per answer are derived via softmax over $\beta_{ik}$.
    - **Design Motivation**: Krippendorff's $\alpha$ ranges from 0 to 0.8, confirming genuine expert disagreement—precisely the clinical ambiguity the dataset aims to capture.

### Loss & Training

MENTAT is an evaluation-only dataset and is not used for training. Core evaluation design:

- Multiple-choice evaluation: sampling at temperature $T=0$, accuracy computed per category.
- Bias evaluation: accuracy differences compared across the same question under different demographic variables (3 gender identities × 6 racial groups × 3 age ranges).
- Free-text evaluation: three inconsistency metrics compare open-ended responses against expert annotations.
- 90%/10% split: 183 questions for evaluation + 20 questions for few-shot prompting.

## Key Experimental Results

### Main Results

Average accuracy of 22 models on $\mathcal{D}_0$:

| Task Category | All Models Avg. | OpenAI + Anthropic Avg. |
|---|---|---|
| Diagnosis | 0.77±0.03 | 0.91±0.04 |
| Treatment | 0.74±0.02 | 0.92±0.03 |
| Monitoring | 0.65±0.02 | 0.79±0.04 |
| Triage | 0.51±0.03 | 0.48±0.03 |
| Documentation | 0.44±0.03 | 0.46±0.02 |

### Ablation Study

Demographic sensitivity (average accuracy, diagnosis/monitoring categories, all models):

| Dimension | Condition | Diagnosis Accuracy | Monitoring Accuracy |
|---|---|---|---|
| Gender | Female | 0.85 | 0.71 |
| Gender | Male | 0.84 | **0.81** |
| Gender | Non-binary | 0.81 | 0.74 |
| Race | African American | **0.89** | 0.70 |
| Race | White | 0.84 | 0.75 |
| Race | Hispanic | 0.87 | 0.63 |
| Age | 18–33 | **0.90** | 0.71 |
| Age | 49–65 | 0.76 | 0.77 |

### Key Findings

- **Structured vs. ambiguous tasks**: Diagnosis/treatment accuracy reaches 0.74–0.91, while triage/documentation accuracy is approximately 0.5—models show markedly degraded performance on tasks with multiple reasonable answers.
- **Significant demographic bias**: Questions coded with male patients yield 8–10% higher accuracy than female-coded questions on monitoring, triage, and documentation; African American patients score 5% higher than White patients on diagnosis; Hispanic patients show the lowest monitoring accuracy (0.63).
- **Fine-tuning ineffective**: MMedS-Llama-3-8B, fine-tuned on MedS-bench, does not outperform its Llama3.1-8b base model on MENTAT—fine-tuning on LM-synthesized data does not improve real-world clinical decision-making.
- **Multiple-choice vs. free-text inconsistency**: Models with high multiple-choice accuracy may deviate substantially from expert options in open-ended responses.
- **Open-source models catching up**: Qwen3, Gemma3, and MedGemma even surpass closed-source models on triage and documentation categories.

## Highlights & Insights

- The dataset is entirely designed and annotated by human experts with no LM involvement, avoiding known quality issues associated with LM-synthesized data.
- The deliberate ambiguity design for triage/documentation, combined with hierarchical Bradley-Terry preference annotation, captures the intrinsic uncertainty of psychiatric decision-making.
- The systematic demographic variable substitution design enables controlled, large-scale bias analysis with far greater generalizability than case-by-case approaches.
- The benchmark adopts a clear evaluation-first positioning: prioritizing quality over scale.

## Limitations & Future Work

- The dataset is relatively small (203 base questions); although expanded via variable substitution, question diversity remains limited.
- The benchmark is confined to the U.S. psychiatric system (DSM-5-TR, U.S. billing codes, etc.) and may not generalize to other national healthcare systems.
- Multiple-choice and free-text evaluation still cannot fully capture the dynamic nature of real clinical interactions (e.g., patient interviews, multi-turn dialogue).
- Annotator bias may be present (although the team is diverse and Jensen-Shannon distance analysis did not reveal significant gender differences, sample sizes are limited).
- The benchmark currently assesses performance at the level of human competence rather than superhuman capability.

## Related Work & Insights

- **vs. MedQA/MMLU**: Exam-based benchmarks assess knowledge recall; MENTAT assesses clinical decision-making—the two are complementary.
- **vs. MedS-bench**: MedS-bench is large-scale but relies on LM-synthesized data; MENTAT is smaller but entirely human-designed.
- **vs. AIME/HumanEval/BIG-Bench Hard**: Shares the "small but high-quality" evaluation design philosophy.
- **Implications for psychiatric AI**: Current LMs achieve approximately 50% accuracy on ambiguous decision tasks, indicating a substantial gap before practical deployment; the presence of demographic bias makes discussions of superhuman performance premature.

## Rating

- Novelty: ⭐⭐⭐⭐ First fully expert-designed psychiatric decision-making and fairness evaluation dataset.
- Experimental Thoroughness: ⭐⭐⭐⭐ 22 models × 5 task categories × 3 demographic dimensions × free-text evaluation.
- Writing Quality: ⭐⭐⭐⭐ Dataset design and annotation pipeline are described in thorough detail.
- Value: ⭐⭐⭐⭐ Fills a critical gap in psychiatric AI evaluation; fairness analysis carries significant societal relevance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Beyond Procedure: Substantive Fairness in Conformal Prediction](../../ICML2026/llm_safety/beyond_procedure_substantive_fairness_in_conformal_prediction.md)
- [\[ACL 2026\] AgentCoMa: A Compositional Benchmark Mixing Commonsense and Mathematical Reasoning in Real-World Scenarios](../../ACL2026/llm_safety/agentcoma_a_compositional_benchmark_mixing_commonsense_and_mathematical_reasonin.md)
- [\[ICLR 2026\] Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark](measuring_physical-world_privacy_awareness_of_large_language_models_an_evaluatio.md)
- [\[NeurIPS 2025\] SWE-SQL: Illuminating LLM Pathways to Solve User SQL Issues in Real-World Applications](../../NeurIPS2025/llm_safety/swe-sql_illuminating_llm_pathways_to_solve_user_sql_issues_in_real-world_applica.md)
- [\[ACL 2026\] Detoxification for LLM from Dataset Itself](../../ACL2026/llm_safety/detoxification_for_llm_from_dataset_itself.md)

</div>

<!-- RELATED:END -->
