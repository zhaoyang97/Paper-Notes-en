---
title: >-
  [Paper Note] The Impact of Auxiliary Patient Data on Automated Chest X-Ray Report Generation and How to Incorporate It
description: >-
  [ACL 2025 (Long Paper)][Medical LLM][Chest X-ray report generation] This paper investigates how to integrate Emergency Department (ED) patient data (vital signs, medications, triage information, etc.) into multimodal language models for automated chest X-ray report generation. It proposes a method to convert heterogeneous tabular data, text, and images into unified embeddings, which significantly improves the clinical accuracy of reports on the MIMIC-CXR + MIMIC-IV-ED dataset…
tags:
  - "ACL 2025 (Long Paper)"
  - "Medical LLM"
  - "Chest X-ray report generation"
  - "multimodal language models"
  - "auxiliary patient data"
  - "electronic health records"
  - "reinforcement learning"
date: 2026-05-08
content_hash: 17c4e460e3a797eb
---

# The Impact of Auxiliary Patient Data on Automated Chest X-Ray Report Generation and How to Incorporate It

**Conference**: ACL 2025 (Long Paper)  
**arXiv**: [2406.13181](https://arxiv.org/abs/2406.13181)  
**Code**: [GitHub (Anonymous)](https://anonymous.4open.science/r/anon-D83E) | [HuggingFace Model](https://huggingface.co/aehrc/cxrmate)  
**Area**: Medical NLP  
**Keywords**: Chest X-ray report generation, multimodal language models, auxiliary patient data, electronic health records, reinforcement learning  

## TL;DR
This paper investigates how to integrate Emergency Department (ED) patient data (vital signs, medications, triage information, etc.) into multimodal language models for automated chest X-ray report generation. It proposes a method to convert heterogeneous tabular data, text, and images into unified embeddings, which significantly improves the clinical accuracy of reports on the MIMIC-CXR + MIMIC-IV-ED datasets, outperforming multiple baseline models including CXRMate-RRG24.

## Background & Motivation
- Chest X-ray (CXR) report generation is an important medical AI task, but existing approaches primarily rely on CXR images and limited radiology data (such as the indication section).
- In actual clinical scenarios, radiologists refer to patients' clinical information (vital signs, medication history, chief complaints, etc.) when interpreting medical images, which can significantly improve diagnostic accuracy.
- However, existing CXR report generation models rarely leverage emergency department (ED) patient record data.
- Although the trend of integrating Electronic Health Record (EHR) systems into radiology workflows is increasingly evident, there is a lack of systematic research on how to transform heterogeneous patient data sources into embeddings usable by language models.

## Core Problem
1. Which auxiliary patient data sources can effectively improve the clinical accuracy of CXR report generation?
2. How can heterogeneous data types (numerical, categorical, textual, time-series, and images) be transformed into a unified embedding representation for multimodal language models?

## Method

### Overall Architecture
The model is based on the CXRMate-RRG24 architecture, employing UniFormer as the image encoder and Llama as the decoder. Diverse patient data from MIMIC-CXR and MIMIC-IV-ED are converted into patient data embeddings, serving as prompts input into the decoder to generate the findings and impression sections of radiology reports. Each embedding is composed of the sum of four components: patient data embedding + source embedding + position embedding + time delta embedding.

### Key Designs
1. **Time Delta Embeddings**: The time difference between the event and the examination is mapped via $D = 1/\Delta + 1$ and projected onto the decoder's hidden dimension through an FNN (with SiLU activation), encouraging the model to focus on chronologically closer events. Position embeddings are sorted by time delta, utilizing Rotary Position Embedding (RoPE) to assign higher attention weights to the most recent data.

2. **Grouped Embeddings**: Numerical and categorical columns in tables are grouped by time delta to form feature vectors (numerical values are kept as-is, and categorical values are one-hot activated), which are then transformed into embeddings via an FNN. High-cardinality columns (such as medicine names) are processed into textual embeddings using a tokenizer and token embeddings. Comparative experiments demonstrate that Grouped Embeddings outperform both separate embeddings and values-to-text methods.

3. **Report Section Embeddings**: Indication (reason for study), history, and comparison sections from radiology reports are utilized as extra inputs. The history section is investigated for CXR report generation for the first time.

4. **Three-stage Training**:
    - Stage 1: Teacher Forcing training on MIMIC-CXR using only images.
    - Stage 2: TF training using multi-source data on the joint MIMIC-CXR + MIMIC-IV-ED dataset (freezing the image encoder).
    - Stage 3: Optimization using SCST reinforcement learning with a hybrid reward of CXR-BERT + BERTScore + ARN.

5. **ARN Metric**: The Absence of Repeated N-grams (ARN) metric is proposed to measure the repetition rate of generated text, which is incorporated into the RL reward function to reduce repetitive outputs.

6. **Reward per Section**: Rewards are computed separately for the findings and impression sections ($\alpha_1=0.75, \alpha_2=0.25$) to prevent the findings section from dominating the optimization of the impression section.

## Key Experimental Results

### Data Source Ablation (Table 1, findings + impression)

| Data Source Configuration | RG | CX | CB | G |
|-----------|-----|-----|-----|-----|
| Images only | 24.54 | 30.10 | 59.25 | 35.16 |
| + triage | 24.59 | 31.33 | 62.79 | 35.78 |
| + reconciled medicines | 25.10 | 32.05 | 64.70 | 36.32 |
| + indication | 25.01 | 32.78 | 65.49 | 35.88 |
| + history | 24.88 | 31.66 | 63.91 | 35.76 |
| **effective sources (h=0)**| **25.52** | **32.49** | **65.93** | **36.26** |

### Baseline Comparison (Table 2, findings only)

| Model | RG | CX | CB | G | BS | B4|
|------|-----|-----|-----|-----|-----|-----|
| CXRMate | 26.5 | 33.9 | 71.3 | 40.3 | 30.5 | 7.5 |
| CXRMate-RRG24 | 28.9 | 31.2 | 58.2 | 40.2 | 31.0 | 6.6 |
| **Ours + RL + ARN** | **30.2** | **33.6** | **78.0** | **40.7** | **37.3** | **7.6** |

Ours significantly outperforms CXRMate-RRG24 (which uses 550,395 exams) despite our model being trained on only 76,398 exams.

### Ablation Study
- **Effective Data Sources**: The four data sources (triage, reconciled medicines, indication, history) each significantly improve performance, and their combination yields even better results.
- **Ineffective Data Sources**: The ED stays table, metadata table, and administered medicines did not yield significant improvements.
- **Prior Exams**: Utilizing 1-2 prior exams improves performance, but using 3 causes degradation, potentially due to attention dilution.
- **Combining Effective Sources + Prior Exams Causes Performance Drop**: Excess inputs lead to attention dilution.
- **Comparison of Tabular Embedding Methods**: Grouped embeddings > Values-to-text > Separate embeddings (RG values are 31.69, 30.70, and 25.28 respectively).
- **ARN Reward**: Effectively reduces repetition (with ARN increasing from 93.5 to 99.3), although other metrics show a slight decline.

## Highlights & Insights
- Systematically investigates the impact of ED patient data on CXR report generation for the first time, identifying several new effective data sources such as triage, medications, and medical history.
- Proposes a general framework to transform heterogeneous data (numerical, categorical, textual, time-series, and images) into unified embeddings.
- Outperforms the SOTA model with significantly less training data (76K exams vs. 550K), demonstrating the value of auxiliary patient data.
- Thorough case analysis: Conducts in-depth analysis on how auxiliary data influences model predictions using four categories of cases (TP, FP, TN, and FN).
- Proposes the ARN metric and section-specific reward mechanism, addressing the text repetition issue in RL training.
- Utilizes the history section of radiology reports for CXR report generation for the first time, revealing that it is as crucial as the indication section.

## Limitations & Future Work
- **Single-Center Data Bias**: Data is sourced solely from the Beth Israel Deaconess Medical Center; thus, generalizability remains to be validated.
- **Lack of Radiologist Human Evaluation**: Evaluation currently relies solely on automated metrics.
- **Attention Dilution**: Due to model architecture constraints, self-attention weights are diluted when the input is excessively large (e.g., multiple prior exams + all effective sources), which degrades performance.
- **Evidence Balancing Issue**: The model occasionally struggles to properly balance auxiliary data with image evidence, leading to false positives (misled by auxiliary data) or false negatives (failing to utilize clinical evidence).
- **Insufficient Model Interpretability**: The decision-making process of multimodal language models remains a black box.
- **Future Directions**: Employing larger LLM decoders with stronger reasoning capabilities, exploring hierarchical attention mechanisms, and scaling to multi-institutional datasets.

## Related Work & Insights
- **vs. CXRMate/CXRMate-RRG24**: This study builds upon the CXRMate-RRG24 architecture and outperforms the original model with significantly less training data after incorporating auxiliary patient data.
- **vs. Indication-only Methods** (Nguyen et al., 2023): This study shows that the history section is equally important and that combining multiple sources yields superior performance.
- **vs. Prior Exams-only Methods** (Wu et al., 2022): This study discovers that prior exams can be detrimental when combined with other multi-source data, identifying attention dilution as the key bottleneck.
- **vs. Multimodal EHR Models** (MeTra, ETHOS): While those works focus on predictive tasks (such as ICU survival rates), this work applies multimodal EHR data to a language generation task (report generation).
- **vs. CXR-LLaVA, MedXChat, RaDialog**: Ours significantly outperforms these LLM-based methods across multiple metrics.

## Insights & Implications
- **Data-Driven vs. Model-Driven**: This work strongly demonstrates that "richer data" is more critical than "larger models/more training samples"—76K exams with auxiliary data outperforming 550K exams using images alone.
- **General Paradigm for Heterogeneous Data Fusion**: The framework transforming numerical, categorical, textual, temporal, and image data into unified embeddings is highly generalizable to other multimodal medical AI tasks.
- **Attention Dilution as the Key Bottleneck for Multi-source Input Methods**: Future research could explore selective attention, gating mechanisms, or hierarchical encoding to alleviate this issue.
- **The Double-Edged Sword of Auxiliary Data**: Auxiliary data can provide supportive evidence (TP) but also confounding noise (FP); teaching models "discriminative reasoning" is a crucial open challenge.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Automated Structured Radiology Report Generation](automated_structured_radiology_report_generation.md)
- [\[ACL 2025\] CheXalign: Preference Fine-tuning in Chest X-ray Interpretation Models without Human Feedback](chexalign_preference_finetuning.md)
- [\[ACL 2025\] Follow-up Question Generation for Enhanced Patient-Provider Conversations](follow-up_question_generation_for_enhanced_patient-provider_conversations.md)
- [\[ACL 2025\] Online Iterative Self-Alignment for Radiology Report Generation](oisa_radiology_report_gen.md)
- [\[ACL 2025\] Radar: Enhancing Radiology Report Generation with Supplementary Knowledge Injection](radar_radiology_report_gen.md)

</div>

<!-- RELATED:END -->
