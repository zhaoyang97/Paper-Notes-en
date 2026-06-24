---
title: >-
  [Paper Note] Aligning AI Research with the Needs of Clinical Coding Workflows: Eight Recommendations Based on US Data Analysis and Critical Review
description: >-
  [ACL 2025][Medical LLM][clinical coding] Through an in-depth analysis of the MIMIC dataset and existing automated clinical coding research, this position paper points out that current evaluation methodologies (such as focusing only on the top-50 high-frequency codes and using inappropriate metrics) are severely disconnected from real clinical scenarios. It proposes eight specific recommendations to improve evaluation methods and research directions.
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "clinical coding"
  - "ICD coding"
  - "automated coding"
  - "evaluation methodology"
  - "MIMIC"
date: 2026-05-08
content_hash: 78c5c05bd7aca692
---

# Aligning AI Research with the Needs of Clinical Coding Workflows: Eight Recommendations Based on US Data Analysis and Critical Review

**Conference**: ACL 2025  
**arXiv**: [2412.18043](https://arxiv.org/abs/2412.18043)  
**Code**: None  
**Area**: Medical NLP  
**Keywords**: clinical coding, ICD coding, automated coding, evaluation methodology, MIMIC  

## TL;DR
Through an in-depth analysis of the MIMIC dataset and existing automated clinical coding research, this position paper points out that current evaluation methodologies (such as focusing only on the top-50 high-frequency codes and using inappropriate metrics) are severely disconnected from real clinical scenarios. It proposes eight specific recommendations to improve evaluation methods and research directions.

## Background & Motivation

**Background**: Clinical coding is the process of translating clinical notes into ICD codes, which are used for hospital billing and epidemiological research. Current AI coding research primarily approaches this as a multi-label classification problem, utilizing public datasets dominated by MIMIC (331,675 admissions).

**Limitations of Prior Work**: Existing studies suffer from a disconnect between evaluation and practical needs: (1) evaluating only the top-50 high-frequency codes, whereas thousands of codes exist in practice; (2) using AUC-ROC as the primary metric, which leads to misleadingly high scores due to highly imbalanced data; (3) ignoring the importance of coding sequence in clinical coding; and (4) lacking metrics for direct comparison with human coding accuracy.

**Key Challenge**: There is a mismatch between fully automated coding pursued by academia and the demand for computer-assisted coding in clinical practice. The instance accuracy of SOTA models (PLM-ICD) is less than 1.1%, whereas human coders achieve approximately 54%-67.5%, representing a huge gap.

**Goal**: (1) Reveal specific deficiencies of existing evaluation strategies; (2) offer improvement recommendations to align research closer to clinical needs; (3) propose a new methodology beyond fully automated coding.

**Key Insight**: Starting from the clinical coding workflow, this paper systematically analyzes the shortcomings of evaluation methods and proposes new research directions such as computer-assisted coding and code auditing.

**Core Idea**: Instead of pursuing seemingly unattainable fully automated coding, it is more practical to integrate AI into clinical coding workflows as an assistant tool, while revising existing evaluation methods to more realistically reflect system performance.

## Method

### Overall Architecture
This is not a methodology paper, but rather a position paper. The overall structure consists of: (1) describing the clinical coding workflow; (2) revealing evaluation deficiencies based on MIMIC data analysis; (3) proposing eight recommendations for improvement; and (4) presenting a new workflow-inspired methodology.

### Key Analysis

1. **Top-50 Code Coverage Analysis**:
    - **Function**: Quantify the coverage of actual data by the top-50 most common codes.
    - **Key Findings**: The top-50 codes only cover 33.92% of the code occurrences; 0% of admission records are fully covered by the top-50; even when expanded to the top-800, approximately 80% of admission records are still not fully covered.
    - **Problem**: Model rankings are inconsistent under different code sets (top-50 vs. full). For example, CNN outperforms CAML on top-50 but performs worse on full codes, indicating a lack of generalizability in top-50 evaluations.

2. **Threshold and Metric Analysis**:
    - **Function**: Analyze the issues of using a global threshold (0.5) and AUC-ROC metrics on imbalanced data.
    - **Key Findings**: PLM-ICD's macro AUC-ROC is >95% (seemingly excellent), but its MAP is <70% (poor practical precision). This is because the dominance of the negative class in imbalanced data artificially inflates the AUC-ROC.
    - **Recommendation**: Report AUC-PR (Average Precision) alongside AUC-ROC, and transition to dynamic thresholds rather than a global fixed 0.5.

3. **Comparison with Human Coding Accuracy**:
    - **Function**: Calculate the Jaccard Score (coding accuracy) of PLM-ICD and compare it with human coders.
    - **Key Data**: PLM-ICD achieves a top three-digit accuracy of 55.22%, compared to a human median of 83.2%; its instance accuracy (EMR) is <1.1%, whereas human accuracy is approximately 54%-67.5%.
    - **Recommendation**: EMR and Jaccard Score must be reported to show the real gap between AI and human coders.

4. **Importance of Code Ordering**:
    - **Function**: Highlight that existing research ignores the sequence of coding required by clinical standards.
    - **Core Argument**: Official ICD-10-CM guidelines require etiology codes to precede manifestation codes, and anesthesia codes to immediately follow surgical codes. Existing studies completely ignore sequence information.

### Proposed New Methodology

The authors propose new research directions to integrate AI into clinical coding workflows:

1. **Sequential Task**:
    - Transforms multi-label classification into step-by-step single-label prediction.
    - Predicts one code per step and obtains human feedback as input for the next step.
    - Three designs: classifier chains, single/multi-classifier iteration, and seq2seq decoding + feedback.
    - Evaluation metrics: Precision@k, number of steps to reach full coverage, and feedback convergence rate.

2. **Recall Task**:
    - Transforms the problem into multiple-choice questions, maximizing relevant options and minimizing the total number of options.
    - High-confidence codes are automatically assigned, while low-confidence ones are presented to coders for selection.
    - Optimizes Recall@k rather than traditional F1.

3. **Structural Task**:
    - Utilizes the hierarchical structure of ICD codes to perform a two-stage process: first predicting the parent code (first three digits), then predicting sub-codes.
    - Experiments demonstrate that parent code prediction is much simpler (micro F1: 29.1% vs. 10.5%).
    - Delegates difficult sub-code prediction to humans.

4. **Code Auditing**:
    - If the model's Precision@1 is 95%, it can act as an offline auditor after manual coding.
    - Flags high-confidence missing codes and prompts the coder to review.
    - Does not disrupt the normal coding workflow.

## Key Experimental Results

### Main Results: PLM-ICD Coding Accuracy

| Dataset | 3-digit Accuracy | 4-digit Accuracy | Full Code Accuracy |
|--------|------------|------------|------------|
| MIMIC-III Clean | 52.84 ±0.34 | 46.21 ±0.33 | 44.01 ±0.33 |
| MIMIC-IV ICD-9 | 55.22 ±0.19 | 49.28 ±0.19 | 46.75 ±0.18 |
| MIMIC-IV ICD-10 | 51.17 ±0.22 | 44.97 ±0.22 | 42.05 ±0.22 |

### Key Statistical Analysis

| Analysis Dimension | Finding | Explanation |
|---------|------|------|
| Top-50 Coverage | 33.92% | Only accounts for total code occurrences |
| Full Coverage (top-50) | 0% | No admission records are fully covered |
| Full Coverage (top-800) | 20.48% | Still 80% of admission records are not fully covered |
| SOTA Instance Accuracy | <1.1% | vs Human 54%-67.5% |
| PLM-ICD AUC-ROC | >95% | Seemingly excellent but misleading |
| PLM-ICD MAP | <70% | Practical precision is far worse than shown by AUC-ROC |

### Key Findings
- Document length has little impact on model performance, with minimal differences after truncation from 4,000 to 2,500 words.
- In MIMIC-IV, only about 1% of admission records contain a single unique ICD-10-CM three-digit code, while more than half contain at least 6.
- MIMIC-III covers only 50.16% of the 17,800 possible ICD-9-CM codes.
- MIMIC-IV ICD-10 covers only 18.78% of the 139,000 possible codes.

## Highlights & Insights
- **Workflow Perspective**: The authors analyze the problem from the perspective of the complete clinical coding workflow rather than a merely technical "how to improve classification accuracy" view. This approach of backward-designing research directions from actual needs is highly valuable.
- **Revealing Metric Distortion**: The misleading nature of AUC-ROC in imbalanced scenarios is a classic problem, but is systematically demonstrated here for the first time in the clinical coding domain, quantitatively showing the massive disparity between >95% AUC-ROC and <70% MAP.
- **Paradigm Shift from Automation to Human-AI Collaboration**: The proposed methodologies (sequential, recall, and structural tasks) shift the coding problem from "replacing humans" to "assisting humans," which is far more practically feasible.
- **Hierarchical Prediction**: Utilizing the natural hierarchy of ICD codes, predicting three-digit parent codes is much easier than full-code prediction (F1: 29.1% vs. 10.5%). This observation can be transferred to other hierarchical label classification tasks.

## Limitations & Future Work
- The analysis is based solely on the US MIMIC dataset. Since coding systems and workflows vary across countries/regions, the generalizability of the recommendations is limited.
- The paper does not provide empirical validation for the proposed methods; all new methodologies remain at the theoretical proposal stage.
- There is insufficient discussion on the application of LLMs in clinical coding, particularly regarding the potential of models like GPT-4 for coding assistance.
- The challenges of multilingual clinical coding are not discussed, despite many countries using localized versions of ICD.

## Related Work & Insights
- **vs PLM-ICD (Huang et al., 2022)**: PLM-ICD is the current SOTA automated coding model utilizing pre-trained language models with code-specific attention. This paper uses it as the primary subject of analysis to expose its practical clinical limitations.
- **vs CAML (Mullenbach et al., 2018)**: Pioneered the CNN + per-code attention approach. Its ranking inconsistency between top-50 and full-code evaluations supports this paper's argument that top-50 evaluation is unreliable.
- **vs Edin et al. (2023)**: Provided systematic replication and benchmarking. This work heavily cites their findings to support its analysis.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Although not a technical innovation, the workflow perspective systematically analyzing evaluation deficiencies is unique and highly valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ The data analysis is comprehensive and in-depth, but the proposed new methodology lacks empirical validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Well-structured, strongly argued, with concrete and actionable recommendations.
- **Value**: ⭐⭐⭐⭐ Holds significant guiding significance for the clinical coding research direction; the eight recommendations are highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mind the Gap: Aligning Knowledge Bases with User Needs to Enhance Mental Health Retrieval](../../NeurIPS2025/medical_nlp/mind_the_gap_aligning_knowledge_bases_with_user_needs_to_enhance_mental_health_r.md)
- [\[ACL 2025\] RedactX: An LLM-Powered Framework for Automatic Clinical Data De-Identification](redactor_an_llm-powered_framework_for_automatic_clinical_data_de-identification.md)
- [\[ACL 2026\] Language Reconstruction with Brain Predictive Coding from fMRI Data](../../ACL2026/medical_nlp/language_reconstruction_with_brain_predictive_coding_from_fmri_data.md)
- [\[ACL 2025\] A Modular Approach for Clinical SLMs Driven by Synthetic Data with Pre-Instruction Tuning, Model Merging, and Clinical-Tasks Alignment](a_modular_approach_for_clinical_slms_driven_by_synthetic_data_with_pre-instructi.md)
- [\[NeurIPS 2025\] Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models](../../NeurIPS2025/medical_nlp/position_thematic_analysis_of_unstructured_clinical_transcripts_with_large_langu.md)

</div>

<!-- RELATED:END -->
