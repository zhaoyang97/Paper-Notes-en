---
title: >-
  [Paper Note] AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset
description: >-
  [ACL 2025][Medical LLM][medical QA] This work constructs AfriMed-QA (15,275 questions across 32 specialties in 16 countries), the first large-scale pan-African medical QA benchmark, systematically evaluates 30 LLMs, and reveals significant regional performance gaps and the counter-intuitive phenomenon where domain-specific biomedical models underperform general-purpose models in African healthcare contexts.
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "medical QA"
  - "LLM evaluation"
  - "African healthcare"
  - "benchmark"
  - "pan-African"
date: 2026-05-08
content_hash: 7287f4fb65469b07
---

# AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset

**Conference**: ACL 2025  
**arXiv**: [2411.15640](https://arxiv.org/abs/2411.15640)  
**Code**: [HuggingFace Dataset](https://huggingface.co/datasets/intronhealth/afrimedqa_v2)  
**Area**: Medical NLP  
**Keywords**: medical QA, LLM evaluation, African healthcare, benchmark, pan-African  

## TL;DR

This work constructs AfriMed-QA (15,275 questions across 32 specialties in 16 countries), the first large-scale pan-African medical QA benchmark, systematically evaluates 30 LLMs, and reveals significant regional performance gaps and the counter-intuitive phenomenon where domain-specific biomedical models underperform general-purpose models in African healthcare contexts.

## Background & Motivation

**Background**: LLMs in the medical field (e.g., Med-PaLM, GPT-4) have achieved near-human or superior performance on standardized exam benchmarks like the USMLE, stimulating global interest in AI-assisted healthcare, particularly in low- and middle-income countries (LMICs) facing severe clinician shortages.

**Limitations of Prior Work**: Current mainstream medical QA benchmarks (MedQA, PubMedQA, MedMCQA, MMLU) originate almost entirely from Western medical systems (e.g., the US USMLE). Training and evaluation datasets severely lack representation from the Global South, including Africa, failing to reflect region-specific disease spectrums, cultural factors, and clinical resource constraints.

**Key Challenge**: High performance of LLMs on Western benchmarks does not guarantee equivalent efficacy in African healthcare contexts. African-specific disease patterns (e.g., trypanosomiasis transmission, HPV vaccine timing variations), differences in clinical presentations (e.g., description of skin lesion pigmentation), and availability of drugs and equipment degrade model generalization capability, yet no suitable dataset exists to quantify this gap.

**Goal**: To build a large-scale, homegrown pan-African medical question-answering benchmark covering multiple countries, specialties, and question formats, and to comprehensively evaluate 30 LLMs to quantify their performance and limitations in African healthcare scenarios.

**Key Insight**: Leveraging a large-scale crowdsourcing platform to collect 15,275 medical QA questions (MCQ + SAQ + CQ) from 60+ medical schools across 16 African countries, and systematically assessing them using both quantitative metrics and blind evaluations by clinicians.

**Core Idea**: Constructing the first multi-specialty QA benchmark using pan-African native medical data to reveal regional biases in LLM clinical capabilities within African contexts.

## Method

### Overall Architecture

The pipeline of construction and evaluation for AfriMed-QA consists of three stages:

1. **Data Collection and Quality Control**: Repurposing the crowdsourcing platform of Intron Health to collect three types of medical QA data from 621 contributors (55.56% female) across 16 African countries, cross-verified by clinical expert teams (with a minimum passing threshold of $\ge 80\%$ to qualify).
2. **Quantitative Evaluation**: Conducting zero-shot evaluations on 30 LLMs, matching answer accuracy for MCQs, and using BERTScore, ROUGE-Lsum, and QuestEval for SAQs and CQs.
3. **Qualitative Evaluation**: Having 379 evaluators (58 clinicians + 321 non-clinicians) perform double-blind manual evaluations on 3,000 randomly sampled LLM responses across multiple dimensions, including correctness, harm, omission, hallucination, and local relevance.

### Key Designs

1. **Three-Tier Question Structure**
    - **Function**: To cover medical QA scenarios with varying difficulty and evaluation needs.
    - **Mechanism**: Expert MCQ (3,910 questions with 2-5 options, correct answers, and explanations) evaluates clinical knowledge accuracy; SAQ (359 questions with 1-3 paragraph open-ended answers) assesses comprehensive formulation; CQ (10,000 consumer queries) evaluates patient-facing communication quality.
    - **Design Motivation**: MCQ alone cannot fully evaluate the utility of LLMs in healthcare. SAQs test deep reasoning, whereas CQs reflect real-world patient inquiries.

2. **Multi-Dimensional Human Evaluation Framework**
    - **Function**: To address the low discriminative power of automatic metrics (especially BERTScore) and provide clinically credible quality assessments.
    - **Mechanism**: Based on the TEHAI framework and the evaluation axes of Med-PaLM, a double-blind human rating system with a 5-point Likert scale is designed. Clinicians evaluate correctness, harm, hallucination, omission, and local customization, while non-clinical personnel rate relevance, helpfulness, and localization.
    - **Design Motivation**: Automatic metrics fail to reliably differentiate the quality of open-ended medical answers (e.g., BERTScore ranges narrowly between 0.86 and 0.89 across all models), demanding human assessment for clinical viability.

3. **Geographical Representation Guarantee Mechanism**
    - **Function**: To ensure the dataset reflects the medical diversity of the African continent without being dominated by a few countries.
    - **Mechanism**: Capping submissions at 300 questions per contributor, prioritizing recruitment of clinicians in Sub-Saharan African countries based on population size, and selecting expert reviewers among professors from medical schools across 5 countries.
    - **Design Motivation**: To prevent single-country dominance and ensure the validity of cross-regional generalization assessments.

### Loss & Training

As a benchmark paper, this work does not involve model training. The evaluation strategies are as follows:

- **MCQ Evaluation**: Extraction of single-letter answers (A/B/C/D/E) from LLM outputs to compute matching accuracy, comparing modes with and without explanation generation.
- **Open-Ended Evaluation**: BERTScore for semantic similarity, QuestEval for factual consistency (which displays the largest dynamic range: 0.19-0.51), and ROUGE-Lsum for structural and n-gram overlap (range: 0.009-0.276).
- **Prompt Design**: Evaluated using a Base prompt (direct answer) and an Instruction-tuning prompt (role-playing as an African doctor) under zero-shot and few-shot settings.

## Key Experimental Results

### Main Results

Accuracy of 30 LLMs on AfriMed-QA Expert MCQ (with MedQA comparisons):

| Model | AfriMed-QA MCQ | MedQA | Gap | SAQ BERTScore | Type |
|------|---------------|-------|------|--------------|------|
| GPT-4o | 0.793 | 0.881 | -8.86 | 0.883 | Closed-source General |
| Claude-3.5 Sonnet | 0.777 | 0.833 | -5.57 | 0.857 | Closed-source General |
| Llama3-405B | 0.763 | 0.807 | -4.41 | - | Open-source General |
| GPT-4 | 0.757 | 0.799 | -4.21 | 0.873 | Closed-source General |
| Claude-3 Opus | 0.746 | 0.780 | -3.45 | 0.870 | Closed-source General |
| Gemini Ultra | 0.739 | 0.788 | -4.89 | 0.872 | Closed-source General |
| Meta Llama3 70B | 0.738 | 0.781 | -4.29 | 0.795 | Open-source General |
| GPT-4o mini | 0.718 | 0.740 | -2.24 | 0.881 | Closed-source General |
| OpenBioLLM 70B | 0.666 | 0.586 | +7.99 | 0.829 | Open-source Biomedical |
| Gemma-2B | 0.173 | 0.328 | -15.55 | 0.856 | Open-source General |

### Ablation Study

MCQ accuracy by country (averaged across 12 representative models):

| Country | Average Accuracy | Expert MCQ Count | Specialties |
|------|----------|-----------|--------|
| Kenya | 0.71 | 562 | 24 |
| Malawi | 0.70 | 347 | 27 |
| Ghana | 0.68 | 1,495 | 24 |
| South Africa | 0.57 | 54 | 1 (Pediatrics only) |
| Nigeria | 0.48 | 1,452 | 23 |

Accuracy comparison of Biomedical vs. General-purpose models (comparable parameter sizes):

| Biomedical Model | Accuracy | General Model | Accuracy | Gap |
|------------|--------|---------|--------|------|
| OpenBioLLM 8B | 0.450 | Meta Llama3.1 8B | 0.619 | -16.9 |
| OpenBioLLM 70B | 0.666 | Meta Llama3 70B | 0.738 | -7.2 |
| BioMistral 7B | 0.440 | Mistral 7B v03 | 0.508 | -6.8 |
| PMC-Llama 7B | 0.463 | Phi3 Mini 4k | 0.604 | -14.1 |

### Key Findings

1. **Significant Performance Drop in African Scenarios**: All models show lower performance on AfriMed-QA compared to the USMLE-based benchmark, with GPT-4o dropping by 8.86 percentage points and the smallest model, Gemma-2B, dropping by 15.55 percentage points.
2. **Counter-intuitive Inferiority of Biomedical Models**: Models fine-tuned on the medical domain perform worse than general-purpose models of comparable scale, likely due to overfitting on Western medical datasets.
3. **Blind Evaluators Prefer LLM Responses**: LLM responses consistently outperformed clinician-written answers in terms of relevance and helpfulness during consumer evaluations.
4. **Small Models Carry Highest Risks of Hallucination and Harm**: Llama-3-8B exhibits a hallucination rate of 9.59% and an omission rate of 21.64%, significantly higher than larger models.
5. **Uneven Performance Across Specialties**: LLMs excel in internal medicine subspecialties like rheumatology and nephrology, but perform poorly in specialties critical to Africa, such as pediatrics, infectious diseases, and obstetrics & gynecology.

## Highlights & Insights

- **Addressing a Crucial Gap**: This is the first large-scale pan-African medical QA benchmark, featuring 15,275 questions across 16 countries and 32 specialties, providing vital infrastructure for evaluating medical LLMs in the Global South.
- **Unprecedented Evaluation Scale**: Encompassing 30 evaluated models under multiple question types combined with 37,435 human ratings, this stands as the most comprehensive non-Western clinical evaluation of LLMs to date.
- **Instructive Counter-intuitive Findings**: The discovery that biomedical LLMs underperform general-purpose ones highlights that domain-specific fine-tuning can exacerbate biases inherited from training data representation.
- **Discrepancy Between Automatic and Human Metrics**: Extreme proximity in BERTScore (0.86-0.89) versus wide variance in QuestEval (0.19-0.51) reiterates that human judgment remains indispensable for medical AI assessments.
- **Real-world Implications**: Highlighting that LLMs perform worse on specialties most in-demand within Africa offers direct guidance for deploying healthcare AI in LMICs.

## Limitations & Future Work

- **Geographical Skew**: Over 60% of expert MCQs come from West Africa (Ghana: 1,495 + Nigeria: 1,452), with South Africa contributing only 54 questions, limited strictly to pediatrics.
- **Language Monolingualism**: The benchmark is entirely in English, neglecting other major African languages such as French or Swahili.
- **Lack of Multimodality**: While medical QA heavily relies on imaging and audio, this dataset currently contains text-only queries.
- **CQs Are Not Born From Real Patients**: Consumer queries are generated using prompt guidelines rather than being harvested directly from clinical patient encounters.
- **Under-explored Prompting Strategies**: Advanced reasoning techniques such as CoT, few-shot variations, and self-consistency were not systematically explored.
- **Format Discrepancies Affecting Fairness**: Output parsing issues arose (most notably for Claude Opus) due to formatting inconsistency (failing 162 times), which might underestimate true model accuracies.

## Related Work & Insights

- **Medical QA Benchmarks**: MedQA (Jin et al., 2021), MedMCQA (Pal et al., 2022), and PubMedQA (Jin et al., 2019) are all derived from Western medical examinations. AfriMed-QA fills the gap by providing an African-centric paradigm.
- **Medical LLM Evaluation**: Benchmarks such as Med-PaLM (Singhal et al., 2022) and EquityMedQA (Pfohl et al., 2024) evaluate equity but remain primarily predominated by Western environments.
- **TEHAI Framework**: This work expands the TEHAI (Reddy, 2023) evaluation framework for healthcare LLMs to incorporate geographic and specialty dimensions.
- **Insights**: Evaluating the clinical capabilities of LLMs requires more than single-region benchmarks; domain fine-tuning can introduce new biases, and deploying small models edge-side presents severe clinical safety issues.

## Rating

- Novelty: ⭐⭐⭐⭐ The first large-scale pan-African medical QA benchmark. It addresses a significant geographical gap, and the scale and diversity of the data collection are impressive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Highly comprehensive, evaluating 30 models with 37,435 human ratings alongside multi-dimensional analyses (specialty, country, task formulation, and explanations).
- Writing Quality: ⭐⭐⭐ Rich in details but somewhat redundant in structure; some portions of the Discussion sections exhibit repetition, and formatting of tables in the appendix is slightly inconsistent.
- Value: ⭐⭐⭐⭐⭐ Extremely consequential for medical AI in the Global South. The exposed regional disparities warrant attention across the research community. The dataset is fully open-sourced.

---
title: >-
  [Paper Note] AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset
description: >-
  [Medical Image][Medical QA] Constructs the first large-scale pan-African medical question-answering benchmark AfriMed-QA (15,275 questions across 32 specialties in 16 countries) to systematically evaluate 30 LLMs in African healthcare contexts, revealing significant regional and specialty differences.
tags:
  - Medical Image
  - Medical QA
  - LLM Evaluation
  - African Healthcare
  - Benchmark
  - Multilingual
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MedAraBench: Large-scale Arabic Medical Question Answering Dataset and Benchmark](../../ICLR2026/medical_nlp/medarabench_large-scale_arabic_medical_question_answering_dataset_and_benchmark.md)
- [\[ACL 2025\] ArgHiTZ at ArchEHR-QA 2025: A Two-Step Divide and Conquer Approach to Patient Question Answering for Top Factuality](arghitz_at_archehr-qa_2025_a_two-step_divide_and_conquer_approach_to_patient_que.md)
- [\[ACL 2025\] MedBioRAG: Semantic Search and Retrieval-Augmented Generation with Large Language Models for Medical and Biological QA](medbiorag_semantic_search_and_retrieval-augmented_generation_with_large_language.md)
- [\[NeurIPS 2025\] Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series](../../NeurIPS2025/medical_nlp/time-imm_a_dataset_and_benchmark_for_irregular_multimodal_multivariate_time_seri.md)
- [\[ACL 2025\] Towards Omni-RAG: Comprehensive Retrieval-Augmented Generation for Large Language Models in Medical Applications](omni_rag_medical.md)

</div>

<!-- RELATED:END -->
