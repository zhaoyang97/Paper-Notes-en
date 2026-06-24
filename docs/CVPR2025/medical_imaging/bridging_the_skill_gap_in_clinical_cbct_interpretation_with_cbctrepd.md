---
title: >-
  [Paper Note] Bridging the Skill Gap in Clinical CBCT Interpretation with CBCTRepD
description: >-
  [CVPR 2025][Medical Imaging][CBCT report generation] Proposes CBCTRepD, the first bilingual report generation system for oral and maxillofacial CBCT. By constructing a dataset of 7,408 high-quality CBCT-report pairs and establishing a multi-level clinical evaluation framework, it consistently improves report quality across radiologists of different experience levels, especially in reducing missed lesions and standardizing report structures.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "CBCT report generation"
  - "Radiology-AI collaboration"
  - "Oral and maxillofacial"
  - "Medical report generation"
  - "Multi-level evaluation"
date: 2026-05-08
content_hash: bfaae822eac123d0
---

# Bridging the Skill Gap in Clinical CBCT Interpretation with CBCTRepD

**Conference**: CVPR 2025  
**arXiv**: [2603.10933](https://arxiv.org/abs/2603.10933)  
**Code**: None  
**Area**: Medical Imaging  
**Keywords**: CBCT report generation, Radiology-AI collaboration, Oral and maxillofacial, Medical report generation, Multi-level evaluation

## TL;DR

Proposes CBCTRepD, the first bilingual report generation system for oral and maxillofacial CBCT. By constructing a dataset of 7,408 high-quality CBCT-report pairs and establishing a multi-level clinical evaluation framework, it consistently improves report quality across radiologists of different experience levels, especially in reducing missed lesions and standardizing report structures.

## Background & Motivation

**Background**: Generative AI has made significant progress in generating reports for 2D medical images such as chest X-rays. Mainstream methods like R2Gen and CheXpert have achieved promising automated metrics in structured report generation. However, cone-beam computed tomography (CBCT), a key imaging modality in dentistry, remains almost unexplored in AI-based report generation.

**Limitations of Prior Work**: CBCT report generation faces three major challenges: (1) **Data Scarcity**: Lack of large-scale, high-quality CBCT-report paired datasets, as existing dental imaging datasets mostly focus on single tasks (e.g., caries detection, tooth segmentation) rather than comprehensive radiological reports; (2) **Complexity of 3D Interpretation**: CBCT is 3D volumetric data requiring simultaneous attention to concurrent pathologies across multiple anatomical regions (teeth, jawbones, joints, maxillary sinuses, etc.), which is far more complex than interpreting 2D images; (3) **Evaluation Gap**: Existing automated assessment metrics (e.g., BLEU, ROUGE) fail to fully reflect the clinical utility of reports, and evaluation frameworks designed for radiologist-AI collaborative scenarios are virtually non-existent.

**Key Challenge**: CBCT interpretation is highly dependent on radiologist experience; junior radiologists easily miss co-occurring lesions and atypical findings, while senior radiologist resources are severely limited. How AI can genuinely bridge this "skill gap" between different experience levels, rather than merely pursuing automated metric improvements, is the core issue of focus.

**Goal**: To construct a practical CBCT report generation system that not only automatically generates high-quality draft reports but, more importantly, consistently improves report quality across all experience levels in human-AI collaboration.

**Key Insight**: The authors approach the problem from the real-world clinical workflow of "radiologist-AI co-authoring." Rather than treating AI as a replacement, it is viewed as a collaborative partner where AI first generates draft reports, which are then refined and finalized by clinicians of varying experience levels.

**Core Idea**: Through large-scale dataset building and a multi-level clinical evaluation framework, this study demonstrates that AI report generation under the "assistance" (rather than "replacement") paradigm can consistently enhance report quality across different experience levels.

## Method

### Overall Architecture

The overall pipeline of CBCTRepD comprises three core phases: data preparation, model training, and collaborative evaluation. The input consists of 3D volumetric oral and maxillofacial CBCT data, and the output is a structured, bilingual (Chinese and English) radiological report. The system designs a complete link from data acquisition to clinical deployment: first constructing a large-scale labeled dataset, then training a report generation model based on this dataset, and finally validating the practical clinical value of AI under diverse collaboration scenarios using a multi-level evaluation framework.

### Key Designs

1. **Large-Scale CBCT-Report Paired Dataset**:

    - Function: To provide high-quality training data for the report generation model.
    - Mechanism: Approximately 7,408 CBCT studies were collected from multiple medical institutions, covering 55 oral disease entities and various acquisition devices/parameter settings. Each report was written by experienced radiologists using standardized templates, including systematic anatomical checkups and structured descriptions of findings. The data covers a broad spectrum of pathologies from common (dental caries, apical periodontitis) to rare (jaw cysts, TMJ abnormalities) conditions.
    - Design Motivation: Existing dental imaging datasets are small-scale and task-specific, which cannot support full report generation. A large-scale, highly diverse dataset is the foundation of the system's generalization capability.

2. **Bilingual Report Generation Model**:

    - Function: To automatically generate structured radiological reports from CBCT volumetric data.
    - Mechanism: A visual encoder is adopted to extract 3D CBCT features, combined with a language model to generate structured reports covering multiple anatomical regions. The model supports bilingual outputs in Chinese and English, with report formats adhering to clinically standardized templates organized by anatomical regions (dentition, periapical, alveolar bone, jawbone, TMJ, maxillary sinus, etc.) to describe the findings.
    - Design Motivation: Structured reports offer better clinical utility than free text, and bilingual support meets the healthcare requirements of different language environments.

3. **Multi-Level Clinical Evaluation Framework**:

    - Function: To comprehensively assess the clinical utility of AI-generated reports.
    - Mechanism: Three evaluation layers are designed: (1) Automated metric evaluation (BLEU, ROUGE, clinical entity F1, etc.) to assess text generation quality; (2) Radiologist-centered evaluation, where senior radiologists double-blindly score the overall completeness, accuracy, and formatting of AI-generated reports; (3) Clinician-centered evaluation, assessing the information completeness and operational utility of reports in real clinical decision-making. In human-AI collaborative scenarios, report quality differences are evaluated for junior, intermediate, and senior radiologists before and after utilizing AI drafts.
    - Design Motivation: Traditional automated metrics fail to reflect clinical utility; only multi-perspective evaluation can truly validate the system's value within actual clinical workflows.

### Loss & Training

Model training follows a standard sequence-to-sequence training paradigm, primarily utilizing cross-entropy loss. During training, report-level teacher forcing and structured template constraints are applied to ensure the formatting standard of the generated reports. Specific training hyperparameters are not fully disclosed in the paper.

## Key Experimental Results

### Main Results

| Evaluation Dimension | Metric | CBCTRepD | Baseline Methods / Benchmark | Description |
|----------|------|----------|--------------|------|
| Automated Metrics | BLEU-4 | Best | Other report generation models | Text generation quality |
| Automated Metrics | Clinical Entity F1 | Best | Other report generation models | Lesion detection accuracy |
| Radiologist Evaluation | Writing Quality Score | Close to intermediate radiologists | Junior/Intermediate/Senior radiologists | AI drafts can reach intermediate level |
| Radiologist Evaluation | Degree of Standardization | Close to intermediate radiologists | Junior/Intermediate/Senior radiologists | Standardization of report structures |

### Collaborative Experiment (Core Highlight)

| Radiologist Level | Without AI Assistance | With AI Assistance | Performance Gain |
|----------|----------|----------|----------|
| Junior Radiologist | Junior Level | Close to Intermediate Level | Significant improvement in completeness and standardization |
| Intermediate Radiologist | Intermediate Level | Close to Senior Level | Reduced omissions, improved comprehensiveness |
| Senior Radiologist | Senior Level | Senior+ (Reduced omissions) | Reduces omission errors, including clinically critical missed diagnoses |

### Key Findings

- **Consistent Improvement Across Levels**: AI assistance is valuable not only for junior radiologists but also for intermediate and senior clinicians. Missed diagnoses by senior experts (especially co-occurring lesions across different anatomical regions) are significantly reduced with AI prompting.
- **Omission Errors Are Key**: The most common mistakes made by radiologists of all levels are omissions rather than false positives. AI's systematic scanning ability effectively covers human attentional blind spots.
- **Report Standardization**: The structured templates of AI drafts act as an implicit normalizing guide, unifying reporting styles among different clinicians.

## Highlights & Insights

- **Innovative Evaluation Paradigm**: Designing "radiologist-AI collaboration" as the core evaluation scenario (rather than just comparing standalone AI output vs. human reports) aligns better with actual clinical AI deployment. This evaluation strategy can be extended to assessing other medical AI systems.
- **Bridging the Skill Gap**: Experiments prove that AI assistance can raise the baseline performance "upward" across all levels of clinicians. This has immense practical value for scenarios with unbalanced healthcare resources, such as primary care hospitals.
- **Large-Scale Oral CBCT Dataset**: The paired dataset of 7,408 cases covering 55 diseases is a major contribution in itself, filling a significant data gap in dental and oral imaging AI.

## Limitations & Future Work

- Focuses solely on the single modality of oral and maxillofacial CBCT; it remains unclear whether the method can generalize to CT report generation for other parts of the body.
- The dataset sources might be concentrated within specific institutions, and regional variations in diagnostic standards and reporting styles have not been fully validated.
- Comparison with general multimodal large language models (such as directly prompt-reading CBCT with GPT-4V) is missing, making it difficult to assess the performance gap between a specialized system and general-purpose models.
- Future research can explore standardizing this collaborative evaluation framework and extending it to other medical image report generation tasks.

## Related Work & Insights

- **vs. R2Gen / CheXpert Series**: Traditional medical report generation focuses on 2D chest X-ray images, where data and methods are relatively mature. This work is the first to systematically extend report generation to 3D CBCT, presenting new challenges in data scarcity and 3D interpretation complexity.
- **vs. General Multimodal Models like GPT-4V**: While general models have powerful language comprehension abilities, they lack dental specialty knowledge and standardized reporting templates, which might lead to hallucinations or missing professional details in clinical settings.
- The "AI-assisted stratified evaluation" methodology presented in this paper is highly instructive—when evaluating any medical AI, the focus should not solely be on standalone AI performance but should emphasize human-AI collaboration outcomes.

## Rating

- Novelty: ⭐⭐⭐⭐ First to systematically address oral CBCT report generation, with an innovative collaborative evaluation framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-level evaluation, though specific numbers are hard to fully verify due to unavailability of certain files/HTML.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and evaluation design.
- Value: ⭐⭐⭐⭐⭐ High clinical utility with a collaborative focus, alongside a significant dataset contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CerebraGloss: Instruction-Tuning a Large Vision-Language Model for Fine-Grained Clinical EEG Interpretation](../../ICLR2026/medical_imaging/cerebragloss_instruction-tuning_a_large_vision-language_model_for_fine-grained_c.md)
- [\[NeurIPS 2025\] Mind the (Data) Gap: Evaluating Vision Systems in Small Data Applications](../../NeurIPS2025/medical_imaging/mind_the_data_gap_evaluating_vision_systems_in_small_data_applications.md)
- [\[NeurIPS 2025\] FairGRPO: Fair Reinforcement Learning for Equitable Clinical Reasoning](../../NeurIPS2025/medical_imaging/fairgrpo_fair_reinforcement_learning_for_equitable_clinical_reasoning.md)
- [\[NeurIPS 2025\] MTBBench: A Multimodal Sequential Clinical Decision-Making Benchmark in Oncology](../../NeurIPS2025/medical_imaging/mtbbench_a_multimodal_sequential_clinical_decision-making_benchmark_in_oncology.md)
- [\[ICML 2025\] EEG-Language Pretraining for Highly Label-Efficient Clinical Phenotyping](../../ICML2025/medical_imaging/eeg-language_pretraining_for_highly_label-efficient_clinical_phenotyping.md)

</div>

<!-- RELATED:END -->
