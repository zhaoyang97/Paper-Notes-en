---
title: >-
  [Paper Note] MedAraBench: Large-scale Arabic Medical Question Answering Dataset and Benchmark
description: >-
  [ICLR2026][Medical LLM][Arabic Medical QA] The authors manually digitized and cleaned paper-based exam questions from medical schools in the Arabic region into 24,883 medical Multiple-Choice Questions (MCQs) with professional department and difficulty annotations. After constructing the large-scale Arabic medical QA benchmark, MedAraBench, and performing double quality checks via expert review and LLM-as-a-judge, 16 open-source and closed-source LLMs were evaluated in a zero-…
tags:
  - "ICLR2026"
  - "Medical LLM"
  - "Arabic Medical QA"
  - "Multiple-Choice Benchmark"
  - "Expert Quality Assessment"
  - "LLM-as-a-judge"
  - "QLoRA Fine-tuning"
date: 2026-05-08
content_hash: b682bf8238bb2d12
---

# MedAraBench: Large-scale Arabic Medical Question Answering Dataset and Benchmark

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=1BXojAgNrg](https://openreview.net/forum?id=1BXojAgNrg)  
**Code**: To be confirmed (authors stated the dataset and evaluation scripts will be released with the paper)  
**Area**: Medical LLM / NLP Evaluation  
**Keywords**: Arabic Medical QA, Multiple-Choice Benchmark, Expert Quality Assessment, LLM-as-a-judge, QLoRA Fine-tuning

## TL;DR
The authors manually digitized and cleaned paper-based exam questions from medical schools in the Arabic region into 24,883 medical Multiple-Choice Questions (MCQs) with professional department and difficulty annotations. After constructing the large-scale Arabic medical QA benchmark, MedAraBench, and performing double quality checks via expert review and LLM-as-a-judge, 16 open-source and closed-source LLMs were evaluated in a zero-shot setting. Results show that even the strongest model, GPT-o3, achieves an accuracy of only 0.765, exposing significant weaknesses in current models' Arabic medical reasoning.

## Background & Motivation
**Background**: Large Language Models (LLMs) have shown impressive performance on English general and medical benchmarks. Medical QA evaluation suites (MedQA, MedMCQA, MMLU-USMLE, etc.) are highly mature, primarily for English or Chinese.

**Limitations of Prior Work**: Arabic is a major language spoken by over 400 million people, yet it suffers from a severe lack of resources in medical NLP. Existing Arabic medical data either consists of open QA scraped from online consultation platforms (e.g., AraMed, lacking standardized difficulty or department mapping), machine-translated English benchmarks (e.g., translated MMLU, which lacks localized clinical context), or small-scale datasets (e.g., MedArabiQ, which has only ~700 questions and lacks expert review, department coverage, and difficulty levels). Consequently, there is no large-scale, expert-validated Arabic medical benchmark covering multiple departments and difficulty levels to fairly measure real-world clinical reasoning.

**Key Challenge**: The natural source for high-quality Arabic medical questions is medical school exams. However, these exist as scanned images on student platforms without structured digital formats. Achieving scale requires significant manual digitization and cleaning costs. Conversely, this "unstructured" nature serves as a natural barrier against data contamination.

**Goal**: Transform scattered paper-based medical exam questions into a standardized, reproducible evaluation resource equipped with quality evidence and baseline results. Specifically: (i) construct a large-scale Arabic medical MCQ dataset with department/difficulty labels and clear train/test splits; (ii) prove data quality through expert review and LLM-as-a-judge; (iii) provide zero-shot baselines for 16 SOTA models under a unified protocol.

**Core Idea**: Use "manual digitization of medical school paper exams + strict filtering + double quality control" to create a benchmark of 24k questions across 19 departments and 5 difficulty levels, quantifying the performance gap in Arabic medical reasoning.

## Method

### Overall Architecture
MedAraBench is a benchmark construction pipeline of "Data → Quality Control → Evaluation" rather than a model architecture. The input is a collection of scanned paper exams, and the output is a standardized benchmark with train/test splits, multi-dimensional annotations, expert quality evidence, and baseline scores for 16 models. The pipeline consists of four steps: **Data Collection and Preprocessing** (digitization, filtering, annotation, stratified splitting), **Quality Assessment** (double-blind expert review and LLM-as-a-judge cross-validation), **Zero-shot Benchmark Evaluation** (16 SOTA models under a unified protocol), and **Few-shot + QLoRA Fine-tuning** to verify data utility.

### Key Designs

**1. Constructing the Dataset via Manual Digitization of Paper Exams: Leveraging Unstructured Sources to Combat Contamination**  
The data source consists of scanned paper exams hosted on student platforms in the Arabic region. Since these do not contain personal or patient information, anonymization was unnecessary. Professional typists were hired to input scans question-by-question into a unified MCQ format. This addresses the trade-off between scale and quality: medical school exams are high-quality carriers of clinical knowledge vetted by educators. Manual digitization, while costly, ensures these questions have never been publicly available in a structured digital format, significantly reducing the risk of data contamination during model pre-training.

**2. Strict Five-Person Manual Filtering + Multi-dimensional Annotation: Ensuring Rich and Usable Information**  
NLP researchers manually inspected the raw data and identified issues such as missing/deformed answers, incomplete options, poor formatting, or non-MCQ content. Five researchers filtered the data based on strict standards, reducing the initial 34,333 questions to 24,883. Each question includes three types of annotations: (i) Number of options (4-way or 5-way); (ii) Difficulty level (corresponding to medical years Y1–Y5); (iii) Medical department (19 categories such as Anatomy, Pediatrics, Surgery, etc., inherited from the scan archives). The authors deliberately **avoided** terminology standardization, arguing that real-world clinical QA is inherently unstandardized, and maintaining this better reflects clinical reality.

**3. Stratified Random Split: Ensuring Uniform Distribution Across 19 Departments**  
A stratified random split (80% training, 20% testing) was applied based on departments. This ensures that every department is proportionately represented in both sets (e.g., if a department has 100 questions, 80 go to training and 20 to testing), preventing evaluation bias. The final split resulted in 19,894 training and 4,989 test questions.

**4. Expert Review + LLM-as-a-judge: Proving Data Quality via Statistical Samples and Cross-validation**  
Two quality control paths were designed. **Expert Review** used four binary metrics (high/low): Medical Accuracy, Clinical Relevance, Question Difficulty, and Question Quality (clarity, homogeneity of options, single best answer, and absence of cues). The sample size was determined using Cochran’s formula. For an infinite population at a 95% confidence level, $\pm5\%$ error, and $p=0.5$:

$$n_0 = \frac{z^2 \, p(1-p)}{e^2}$$

With $z=1.96, p=0.5, e=0.05$, $n_0 = 384$. Adjusting for the finite population $N$:

$$n = \frac{n_0}{1 + \frac{n_0 - 1}{N}}$$

Resulting in $n=378$. Two certified physicians with 20+ years of experience in Anesthesia and Internal Medicine performed double-blind reviews. **LLM-as-a-judge** supplemented this by using top SOTA models (gpt-o3, gemini-2.0-flash, etc.) to score the entire test set and calculate Pearson correlation coefficients against expert scores on the 378 samples.

### Loss & Training
For the evaluation protocol, all 16 models used a temperature of 0. Results were parsed using pattern matching for letters (A–D). Fine-tuning validation used QLoRA on Llama-3.1-8B-instruct with 4-bit quantization, targeting q/k/v/o projections in the attention layers for 800 steps. Few-shot evaluation provided 3 expert-vetted example questions from the training set.

## Key Experimental Results

### Main Results
Zero-shot results for 16 models (Table 4, Overall Accuracy):

| Category | Model | Overall Accuracy |
| :--- | :--- | :--- |
| Closed-source · General | GPT-o3 | 0.765 |
| Closed-source · General | GPT-5 | 0.764 |
| Closed-source · General | Claude-Sonnet-4 | 0.694 |
| Closed-source · General | GPT-4.1 | 0.673 |
| Closed-source · General | Gemini-2.0-Flash | 0.654 |
| Open-source · General | DeepSeek-chat-v3 | 0.620 |
| Open-source · General | Qwen-plus | 0.618 |
| Open-source · General | Llama-3.3-70B-instruct | 0.547 |
| Open-source · Arabic | Fanar-C-1-8.7B | 0.498 |
| Open-source · Arabic | Allam-7B-instruct | 0.447 |
| Open-source · Medical | MedGemma-4B-it | 0.390 |
| Open-source · Medical | BiMedix-Bi-27B | 0.390 |
| Open-source · Arabic | c4ai-command-r7b-arabic | 0.381 |
| Open-source · Medical | Med42-8B | 0.318 |
| Open-source · Medical | Apollo-7B | 0.238 |
| Open-source · General | Llama-3.1-8B-instruct | 0.170 |

The gap is clear: closed-source reasoning models lead significantly, but even GPT-o3 (0.765) is far from expert level. Specialized Arabic/medical models often fall below 0.5.

### Key Findings
- **Reasoning Models Lead**: GPT-o3/GPT-5 outperform others significantly, highlighting the importance of reasoning in medical NLP.
- **QLoRA Beats In-context Learning**: Few-shot led to minor gains (+12.4%), while QLoRA nearly doubled performance (+88.2%), proving the value of the MedAraBench training set.
- **LLM Judges are Unreliable**: Correlation between LLM-as-a-judge and experts is weak-to-moderate, with difficulty showing almost no correlation ($<0.039$).
- **MedAraBench is Harder than MedArabiQ**: Models generally score higher on the older MedArabiQ, suggesting MedAraBench is more challenging.

## Highlights & Insights
- **Unstructured Sources as a Barrier**: Digitizing paper exams is framed as an advantage for preventing contamination rather than just an engineering burden.
- **Deliberate Non-standardization**: Preserving terminology inconsistencies mirrors real-world clinical environments.
- **Statistical Rigor**: Using Cochran's formula for sampling makes the quality control process defensible.
- **Accuracy vs. Understanding**: The authors note that models might solve questions via statistical co-occurrence (disease-treatment) rather than true clinical reasoning.

## Limitations & Future Work
- **Constraint to MCQs**: Cannot evaluate generative clinical reasoning.
- **Contamination Risks**: Despite the source being non-digital, 100% absence from pre-training data cannot be guaranteed.
- **MSA Assumption**: The benchmark assumes Modern Standard Arabic, though dialects are common in clinical practice.
- **Expert Inter-rater Reliability**: Low Kappa scores in some dimensions (e.g., Question Quality) reflect the subjectivity of clinical judgment.

## Related Work & Insights
- **vs. MedQA / MMLU**: These are English-centric. MedAraBench brings this paradigm to Arabic with added difficulty mapping.
- **vs. Translated MMLU**: Machine translation lacks localized clinical nuances; MedAraBench uses native Arabic medical exams.
- **vs. AraMed**: AraMed is larger but lacks expert labels or department/difficulty structures.
- **vs. MedArabiQ**: MedArabiQ is much smaller (~700 questions). MedAraBench is its direct large-scale upgrade with superior vetting.

## Rating
- Novelty: ⭐⭐⭐⭐ (Fills a gap in large-scale Arabic medical evaluation with a clever anti-contamination approach).
- Experimental Thoroughness: ⭐⭐⭐⭐ (Extensive model baseline + dual quality control).
- Writing Quality: ⭐⭐⭐⭐ (Rigorous methodology and deep reflection on model learning).
- Value: ⭐⭐⭐⭐⭐ (Essential for the development of multi-lingual medical LLMs).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of Large Language Models in Mental Health Question Answering](counselbench_a_large-scale_expert_evaluation_and_adversarial_benchmarking_of_lar.md)
- [\[ACL 2025\] AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset](../../ACL2025/medical_nlp/afrimed_qa_pan_african.md)
- [\[ICLR 2026\] CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA](counselbench_llm_mental_health_qa.md)
- [\[AAAI 2026\] Expert-Guided Prompting and Retrieval-Augmented Generation for Emergency Medical Service Question Answering](../../AAAI2026/medical_nlp/expert-guided_prompting_and_retrieval-augmented_generation_for_emergency_medical.md)
- [\[ACL 2026\] Query Pipeline Optimization for Cancer Patient Question Answering Systems](../../ACL2026/medical_nlp/query_pipeline_optimization_for_cancer_patient_question_answering_systems.md)

</div>

<!-- RELATED:END -->
