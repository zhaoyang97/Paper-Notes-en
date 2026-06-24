---
title: >-
  [Paper Note] Automated Structured Radiology Report Generation
description: >-
  [ACL 2025][Medical LLM][Radiology report generation] This work proposes a new task, Structured Radiology Report Generation (SRRG), which leverages LLMs to restructure free-text reports into standardized formats. It also introduces SRR-BERT, a 55-label disease classification model, and F1-SRR-BERT, an evaluation metric, addressing the challenges of report generation and evaluation caused by highly diverse reporting styles.
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "Radiology report generation"
  - "structured reporting"
  - "disease classification"
  - "chest X-ray"
  - "evaluation metrics"
date: 2026-05-08
content_hash: 78dd5888834a20f2
---

# Automated Structured Radiology Report Generation

**Conference**: ACL 2025  
**arXiv**: [2505.24223](https://arxiv.org/abs/2505.24223)  
**Code**: [huggingface.co/StanfordAIMI](https://huggingface.co/StanfordAIMI)  
**Area**: Medical NLP  
**Keywords**: Radiology report generation, structured reporting, disease classification, chest X-ray, evaluation metrics

## TL;DR

This work proposes a new task, Structured Radiology Report Generation (SRRG), which leverages LLMs to restructure free-text reports into standardized formats. It also introduces SRR-BERT, a 55-label disease classification model, and F1-SRR-BERT, an evaluation metric, addressing the challenges of report generation and evaluation caused by highly diverse reporting styles.

## Background & Motivation

Automated chest X-ray (CXR) report generation is an important medical NLG task that can reduce the workload of radiologists. Currently, the two primary datasets, MIMIC-CXR and CheXpert Plus, consist of free-text reports. These reports are highly variable in style and lack structure, which introduces two major challenges:

**Generation difficulty**: The diversity of free-text reports makes it difficult for models to generate consistent, clinically meaningful reports.

**Evaluation difficulty**: Existing evaluation metrics (NLG metrics like BLEU and ROUGE, and clinical metrics like F1-RadGraph) struggle to accurately capture subtle differences in radiological interpretation, given that a single finding can be described in multiple ways.

Meanwhile, there has been a continuous clinical demand for more consistent and structured radiology reporting. This practical necessity and technical challenge motivated the authors to propose the SRRG task—restructuring free-text reports into a standardized format alongside more precise evaluation methods.

## Method

### Overall Architecture

The SRRG framework comprises three core contributions: (1) defining structured reporting desiderata and utilizing LLMs to construct a large-scale structured report dataset; (2) training SRR-BERT, a fine-grained disease classification model; and (3) proposing the F1-SRR-BERT evaluation metric. Together, these components form a comprehensive pipeline from data and model to evaluation.

### Key Designs

1. **Structured report desiderata**: Strict standards for report formats are defined:

    - A report consists of six sections: Exam Type, History, Technique, Comparison, Findings, and Impression.
    - The Findings section is organized by predefined anatomical headings: Lungs and Airways, Pleura, Cardiovascular, Hila and Mediastinum, Tubes/Catheters/Support Devices, Musculoskeletal and Chest Wall, Abdominal, and Other.
    - The Impression section lists key findings numbered in descending order of clinical importance.
    - Historical comparisons and personally identifiable information (dates, names, institutions, etc.) are strictly excluded, retaining only patient gender and age.

2. **Dataset Construction**: 

    - GPT-4 Turbo is utilized to restructure free-text reports from MIMIC-CXR and CheXpert Plus into a structured format.
    - SRRG-Findings contains 184,542 samples (181,874 in the training set).
    - SRRG-Impression contains 409,927 samples (405,972 in the training set).
    - Five board-certified radiologists manually reviewed and validated 464 reports.
    - The mapping for the two datasets is: X-ray $\rightarrow$ Findings and X-ray $\rightarrow$ Impression respectively.

3. **SRR-BERT Disease Classification Model (55 labels)**:

    - The 14 labels of CheXbert are expanded to 55 disease labels to cover more fine-grained findings in the lungs, pleura, heart, mediastinum, musculoskeletal system, and abdomen.
    - Each finding maps to zero, one, or multiple disease labels.
    - Each disease is assigned one of three statuses: Present, Absent, or Uncertain.
    - Data annotation employs a three-model voting mechanism: GPT-4 Turbo, GPT-4 Turbo 1106 Preview, and GPT-4o independently annotate the data, and the consensus of at least two models is taken.
    - Fine-tuned on CXR-BERT, annotating a total of 1,506,158 valid sentences.

4. **F1-SRR-BERT Evaluation Metric**: 

    - SRR-BERT is used to predict diseases for both generated and reference reports to calculate the F1 score.
    - It provides two granularities: the leaves level (finest granularity with 55 labels) and the upper level (coarser classification with 25 categories).
    - It supports two modes: aligned (evaluation matched sequentially) and unaligned (evaluation treats findings as a set).
    - The aligned mode evaluates whether the model orders findings by clinical importance.

### Loss & Training

SRR-BERT uses CXR-BERT as the pretrained backbone and undergoes weakly supervised fine-tuning on the StructUtterances dataset. The annotated data contains 1,506,158 sentences and 1,782,983 labels. Training is conducted across four configurations: leaves, upper, leaves with statuses, and upper with statuses, with separate models trained for each.

## Key Experimental Results

### Main Results

**Disease Classification Performance:**

| Model Configuration | Micro F1 | Macro F1 | Weighted F1 |
|---------|----------|----------|-------------|
| SRR-BERT (Leaves) | 0.84 | 0.55 | 0.82 |
| SRR-BERT (Upper) | 0.84 | 0.65 | 0.83 |
| SRR-BERT (Leaves+Statuses) | 0.80 | 0.28 | 0.77 |
| SRR-BERT (Upper+Statuses) | 0.80 | 0.38 | 0.78 |

**Comparison with CheXbert (Mapped to 14 classes):**

| Input Type | CheXbert F1 | SRR-BERT F1 | Note |
|---------|-------------|-------------|------|
| Structured sentence (Leaves mapping) | 0.65 | 0.84 | SRR-BERT +19% |
| Structured sentence (Upper mapping) | 0.50 | 0.86 | SRR-BERT +36% |
| Full report (Upper mapping) | 0.56 | 0.70 | SRR-BERT still superior |

**Report Generation Model Benchmark (SRRG-Impression unaligned, Test):**

| Model | BLEU | ROUGE-L | F1-RadGraph | F1-SRR-BERT |
|------|------|---------|-------------|-------------|
| CheXpert-Plus | 14.84 | 28.01 | 22.14 | 46.48 |
| MAIRA-2 | 8.12 | 27.82 | 20.37 | 50.36 |
| CheXagent | 6.95 | 27.18 | 19.70 | 50.63 |
| RaDialog | 3.32 | 21.59 | 12.32 | 39.22 |

### Ablation Study

| Configuration | Key Metric | Note |
|------|---------|------|
| Unaligned evaluation | BLEU 14.84 | Lax evaluation ignoring sequence order |
| Aligned evaluation | BLEU 3.78 | Significant drop of ~11 points when considering order |
| Findings task | BLEU ~3.5 | More challenging than the Impression task |
| Category prediction | F1 ~77% | Relatively accurate prediction of anatomical partitions |

### Key Findings

- Findings generation is more challenging than Impression generation: traditional metric scores are significantly lower.
- Aligned evaluation is stricter than unaligned evaluation: CheXpert-Plus's BLEU score on SRRG-Impression drops from 14.84 to 3.78.
- SRR-BERT significantly outperforms CheXbert across all comparative settings, validating the effectiveness of 55-label fine-grained classification.
- SRR-BERT maintains robust performance even when non-structured full reports are used as input.
- The category prediction accuracy of all models is around 75-78%, illustrating that correct classification of anatomical structures is achievable.
- CheXagent performs prominently in Recall, while CheXpert-Plus leads in traditional metrics.

## Highlights & Insights

- **Novel task definition**: Defining the conversion of unstructured reports $\rightarrow$ structured reports as a new task is both clinically aligned and convenient for automated evaluation.
- **Exquisite evaluation metric design**: F1-SRR-BERT integrates a hierarchical disease classification system with both aligned and unaligned evaluation modes, amending the shortcomings of traditional NLG metrics in the medical domain.
- **Large dataset scale**: Constructed based on MIMIC-CXR and CheXpert Plus, totaling nearly 600,000 structured reports.
- **Thorough clinical validation**: Five board-certified radiologists participated in the review, enhancing the clinical credibility of the results.
- **Comprehensive 55-label coverage**: Expanding from 14 to 55 labels greatly enhances the granularity of disease classification.

## Limitations & Future Work

- Structured rewriting depends on GPT-4, potentially introducing LLM-specific hallucinations or information loss.
- There are partially ambiguous areas in the label space (e.g., the F1 score for the "Air space opacity" category is only 0.62).
- Macro F1 scores are relatively low (only 0.55 for leaves), indicating that the classification of rare labels still needs improvement.
- The focus is solely on chest X-rays, without extension to other imaging modalities (e.g., CT, MRI).
- The training of end-to-end structured report generation models remains unexplored.
- The clinical utility of structured reporting requires larger-scale prospective clinical validation.

## Related Work & Insights

- CheXbert is a classic 14-label disease classification model, which this work extends to 55 labels.
- F1-RadGraph (Delbrouck et al., 2022) evaluates report quality based on knowledge graphs, while the proposed F1-SRR-BERT provides complementary, fine-grained evaluation.
- GREEN (Ostmeier et al., 2024) and RadFact (Bannur et al., 2024) focus on clinical factualness evaluation.
- MAIRA-2 (Bannur et al., 2024) is currently one of the leading report generation models.
- The concept of structured reporting can be generalized to other medical report generation tasks (e.g., pathology, ultrasound).

## Rating

- Novelty: ⭐⭐⭐⭐ Structured report generation is a valuable new task definition; both SRR-BERT and F1-SRR-BERT are well-designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive multi-model benchmarking, detailed classification comparisons, and human review validation, though lacking end-to-end training experiments.
- Writing Quality: ⭐⭐⭐⭐ The paper features a complete structure and detailed dataset statistics, although some tables are quite dense.
- Value: ⭐⭐⭐⭐ Offers a standardized framework and superior evaluation tools for radiology report generation, yielding practical clinical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Radar: Enhancing Radiology Report Generation with Supplementary Knowledge Injection](radar_radiology_report_gen.md)
- [\[ACL 2025\] Online Iterative Self-Alignment for Radiology Report Generation](oisa_radiology_report_gen.md)
- [\[ACL 2025\] The Impact of Auxiliary Patient Data on Automated Chest X-Ray Report Generation and How to Incorporate It](auxiliary_patient_data_xray.md)
- [\[ACL 2025\] CSTRL: Context-Driven Sequential Transfer Learning for Abstractive Radiology Report Summarization](cstrl_context-driven_sequential_transfer_learning_for_abstractive_radiology_repo.md)
- [\[ACL 2026\] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation](../../ACL2026/medical_nlp/march_multi-agent_radiology_clinical_hierarchy_for_ct_report_generation.md)

</div>

<!-- RELATED:END -->
