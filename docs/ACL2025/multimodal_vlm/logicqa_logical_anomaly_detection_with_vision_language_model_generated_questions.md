---
title: >-
  [Paper Note] LogicQA: Logical Anomaly Detection with Vision Language Model Generated Questions
description: >-
  [ACL 2025][Multimodal VLM][Logical Anomaly Detection] Proposes the LogicQA framework, which utilizes pretrained VLMs to automatically generate anomaly-related questions and detects logical anomalies through a QA voting mechanism. It achieves SOTA performance in training-free, annotation-free, and few-shot settings while providing natural language explanations for the detected anomalies.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Logical Anomaly Detection"
  - "VLM"
  - "QA-based Detection"
  - "Interpretability"
  - "Few-shot"
date: 2026-05-08
content_hash: 46dde42da0505dd6
---

# LogicQA: Logical Anomaly Detection with Vision Language Model Generated Questions

**Conference**: ACL 2025  
**arXiv**: [2503.20252](https://arxiv.org/abs/2503.20252)  
**Code**: None  
**Area**: Multimodal VLM / Anomaly Detection  
**Keywords**: Logical Anomaly Detection, VLM, QA-based Detection, Interpretability, Few-shot

## TL;DR

Proposes the LogicQA framework, which utilizes pretrained VLMs to automatically generate anomaly-related questions and detects logical anomalies through a QA voting mechanism. It achieves SOTA performance in training-free, annotation-free, and few-shot settings while providing natural language explanations for the detected anomalies.

## Background & Motivation

### Problem Definition

Anomaly detection (AD) is crucial for quality control in industrial manufacturing. Anomalies are classified into two categories:
- **Structural Anomalies**: Local defects such as deformations, contamination, etc., which can be detected via pixel-level heatmaps.
- **Logical Anomalies**: Logically anomalous but physically normal appearances that violate predefined constraints (e.g., object count, arrangement, existence), requiring reasoning and interpretation.

Detecting logical anomalies is particularly challenging because they appear visually "normal," requiring an understanding of logical rules to identify violations.

### Limitations of Prior Work

**Reconstruction-based methods** (e.g., AutoEncoder): Require training on a large amount of normal images, which is impractical for few-shot scenarios.

**Memory-bank methods** (e.g., PatchCore): Rely on the feature bank of pretrained vision models, requiring expensive fine-tuning and providing no explanations.

**LogicAD** (Jin et al., 2025): Employs VLMs but relies on category-specific guided Chain-of-Thought (CoT) prompting, requiring carefully designed prompts for each category.

**LogiCode** (Zhang et al., 2024): Uses LLMs to generate Python logic constraints but relies on detailed manual annotations.

### Core Motivation

Existing methods either lack interpretability (providing only heatmaps without explaining the underlying reasons) or require extensive data, annotations, or category-specific prompts. Industrial scenarios demand a **training-free, annotation-free, and auto-prompting** framework that can detect logical anomalies and explain them using only a few normal images.

## Method

### Overall Architecture

LogicQA is a four-stage pipeline:
1. **Describe Normal Images**: The VLM generates detailed textual descriptions for a few normal images.
2. **Summarize Normal Context**: Shared normal attributes are extracted and summarized.
3. **Generate Main Questions**: The VLM generates checklist-style key questions.
4. **Testing**: Image anomalies are determined via a QA-based voting mechanism.

### Key Designs

1. **Normal Image Description (Stage 1)**

    - Input: A single normal image + a predefined definition of normality (from the dataset).
    - The VLM generates a detailed textual description capturing key elements (location, count, appearance).
    - Process three different normal images to ensure generalization.
    - Design Motivation: Directing the VLM to focus on logically relevant features rather than background noise.

2. **Normal Context Summarization (Stage 2)**

    - Descriptions of multiple images are fed into the VLM to extract shared attributes.
    - Focuses on the most consistent and core features to prevent overfitting to specific samples.
    - Function: Establishes a robust "normality" baseline.

3. **Main Question Generation (Stage 3)**

    - Taking the summary and definition of normality as inputs, the VLM generates key questions (Main-Qs) for anomaly detection.
    - Decomposes anomaly detection into multiple focused questions rather than a single query.
    - **Key Filtering Mechanism**: Validating questions on multiple normal images and filtering out those with accuracies below 80%.
    - Avoids few-shot bias and ensures the generalizability of the question set.

4. **Testing Phase (Stage 4)**

    - Each Main-Q is expanded into 5 semantically equivalent sub-questions (Sub-Qs).
    - For each Main-Q, the final answer is determined by a majority vote of its Sub-Qs.
    - Any Main-Q answered as "No" $\to$ the image is flagged as anomalous.
    - The specific Main-Q marked as "No" provides the explanation for the logical anomaly.
    - Computes anomaly scores using the VLM's log probabilities.

### Anomaly Score Calculation

For each Main-Q, the highest log probability among the Sub-Qs consistent with the voting result is selected:

$$s_i = \max_j \{ \log p(q_{ij}(x)) \mid q_{ij}(x) = Q_i(x) \}$$

The anomaly score is calculated as:
- When classified as normal: $1 - \text{Median}(\{e^{s_i}\})$
- When classified as anomalous: $\text{Median}(\{e^{s_i}\})$

### Image Preprocessing

Special handling for the MVTec LOCO AD dataset:
- **BPM (Background Patch Masking)**: Isolates target objects and removes background interference.
- **Lang-SAM**: Integrates GroundingDINO and SAM2 to segment homogeneous objects, addressing the visual limitations of VLMs in multi-object recognition.

## Key Experimental Results

### Main Results — MVTec LOCO AD Logical Anomaly Detection

| Method | Few-shot | Interpretable | Auto-Prompt | AUROC(%) | F1-max(%) |
|------|----------|--------|-----------|----------|-----------|
| PatchCore | ✓ | ✗ | ✗ | 74.0 | - |
| GCAD | ✗ | ✗ | ✗ | 86.0 | - |
| AST | ✗ | ✗ | ✗ | 79.7 | - |
| WinCLIP | ✓ | ✗ | ✗ | 64.3 | 59.5 |
| LogicAD | ✓ | ✓ | ✗ | 86.0 | 83.7 |
| **LogicQA** | **✓** | **✓** | **✓** | **87.6** | **87.0** |

LogicQA improves AUROC by 1.6% and F1-max by 3.3%. On the splicing connectors category, it achieves an AUROC gain of 19% and an F1-max gain of 15.4%.

### Semiconductor SEM Dataset

| Method | AUROC(%) | F1-max(%) |
|------|----------|-----------|
| PatchCore | 79.2 | 77.8 |
| InternVL-2.5 8B | - | 85.2 |
| **LogicQA (GPT-4o)** | **90.3** | **92.4** |

Compared to PatchCore, AUROC is improved by 11.1%, and F1-max is improved by 14.6%.

### Ablation Study

**Performance of different VLMs (F1-max %)**:

| Category | GPT-4o | Gemini-1.5 Flash | InternVL-2.5 38B |
|------|--------|-------------------|-------------------|
| Breakfast Box | 91.6 | 83.3 | 88.2 |
| Pushpins | 97.6 | 98.9 | 93.7 |
| Screw Bag | 64.5 | 91.7 | 62.6 |
| **Average** | **87.0** | **79.7** | **77.6** |

**Human Evaluation Consistency**: Annotators and the model achieved 98% agreement on normal images and 85-86% agreement on anomalous images, demonstrating that LogicQA not only detects anomalies but also provides correct explanations for them.

### Key Findings

1. **Training-free methods can outperform fully-trained methods**: LogicQA outperforms GCAD and AST, which require large amounts of training data, using only 3 normal images.
2. **Interpretability does not sacrifice performance**: The framework achieves state-of-the-art detection performance while simultaneously providing natural language explanations.
3. **Framework Generalization**: Shows robust performance across diverse vision language models, ranging from commercial APIs (GPT-4o) to open-source models (InternVL-2.5).
4. **Industrial Applicability**: Validated on real-world semiconductor SEM data, performing well even without the Main-Q filtering mechanism.

## Highlights & Insights

- **Question Decomposition**: Reformulates anomaly detection as QA on a series of focused questions, akin to a divide-and-conquer strategy in CoT reasoning.
- **Voting Mechanism to Mitigate Hallucination**: Employs majority voting on multiple semantically equivalent questions to alleviate VLM unreliability and hallucinations.
- **High Level of Automation**: Accomplished without manual prompt design, manual annotation, or training. Adapting to new scenarios only requires changing class names and definitions.
- **Innovative Use of Log Probabilities**: Converts VLM output confidence into continuous anomaly scores, enabling the computation of AUROC from binary QA decisions.

## Limitations & Future Work

1. **Dependence on VLM Visual Capabilities**: Performance is bounded by the visual recognition accuracy of the underlying VLM, currently requiring specific pre-processing steps (BPM, Lang-SAM).
2. **Generalizability of MainQs**: Requires diverse normal images to generate a generalized question set; otherwise, it may introduce bias when few-shot samples are insufficient.
3. **Unaddressed Structural Anomalies**: The framework focuses exclusively on logical anomalies, and its performance on pixel-level structural defects remains unevaluated.
4. **Computational Cost**: Each test image necessitates multiple VLM requests (multiple Main-Qs $\times$ 5 Sub-Qs), potentially leading to high latency.
5. **Instability on Specific Categories (e.g., Screw Bag)**: GPT-4o scores 64.5%, whereas Gemini achieves 91.7%, suggesting sensitivity to the choice of the underlying VLM for certain categories.

## Related Work & Insights

- Contrast with LogicAD: LogicQA eliminates the dependency on category-specific guided CoT prompts, streamlining the deployment process.
- Comparison with LogiCode: LogiCode generates Python logical constraint codes, whereas LogicQA operates via natural language questions, offering higher intuitiveness and interpretability.
- Insight: The QA-based detection paradigm can be extended to other visual inspection tasks requiring reasoning and explanation, such as medical imaging and structural damage detection.

## Rating

| Metric | Score (1-5) | Description |
|------|-----------|------|
| Novelty | ⭐⭐⭐⭐ | The QA plus voting anomaly detection paradigm is novel. |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Evaluated on public and industrial datasets, using multiple VLMs and human evaluation. |
| Writing Quality | ⭐⭐⭐⭐ | Clear workflow with abundant figures and tables. |
| Value | ⭐⭐⭐⭐⭐ | Offers direct application prospects for industrial visual inspection. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions](teaching_vlm_ask_ambiguity.md)
- [\[ICCV 2025\] AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models](../../ICCV2025/multimodal_vlm/aigi_holmes_towards_explainable_and_generalizable_ai_generated_image_detection_via_mllm.md)
- [\[CVPR 2026\] ADSeeker: A Knowledge-Grounded Reasoning Framework for Industry Anomaly Detection and Reasoning](../../CVPR2026/multimodal_vlm/adseeker_a_knowledge-grounded_reasoning_framework_for_industry_anomaly_detection.md)
- [\[ACL 2025\] Centurio: On Drivers of Multilingual Ability of Large Vision-Language Model](centurio_multilingual_vlm.md)
- [\[ICCV 2025\] Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection](../../ICCV2025/multimodal_vlm/exploiting_vision_language_model_for_training-free_3d_point_cloud_ood_detection_.md)

</div>

<!-- RELATED:END -->
