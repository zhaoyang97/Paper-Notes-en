---
title: >-
  [Paper Note] EchoAgent: Towards Reliable Echocardiography Interpretation with "Eyes", "Hands" and "Minds"
description: >-
  [CVPR 2026][Medical Imaging][Echocardiography] This paper proposes EchoAgent, an agent system that simulates the "eyes–hands–minds" collaborative workflow of echocardiography clinicians. Through three stages—an Expertise-Driven Cognition engine (mind), a Hierarchical Collaboration Toolkit (eyes + hands), and an Orchestrated Reasoning Hub—the system achieves end-to-end reliable echocardiography interpretation, attaining state-of-the-art performance on multiple benchmarks.
tags:
  - CVPR 2026
  - Medical Imaging
  - Echocardiography
  - Agent System
  - Multimodal Large Language Model
  - Cardiac Function Assessment
  - Tool Calling
date: 2026-05-08
content_hash: 5c3156fb90f52b14
---

# EchoAgent: Towards Reliable Echocardiography Interpretation with "Eyes", "Hands" and "Minds"

**Conference**: CVPR 2026
**arXiv**: [2604.05541](https://arxiv.org/abs/2604.05541)
**Code**: N/A
**Area**: Medical Imaging
**Keywords**: Echocardiography, Agent System, Multimodal Large Language Model, Cardiac Function Assessment, Tool Calling

## TL;DR

This paper proposes EchoAgent, an agent system that simulates the "eyes–hands–minds" collaborative workflow of echocardiography clinicians. Through three stages—an Expertise-Driven Cognition engine (mind), a Hierarchical Collaboration Toolkit (eyes + hands), and an Orchestrated Reasoning Hub—the system achieves end-to-end reliable echocardiography interpretation, attaining state-of-the-art performance on multiple benchmarks.

## Background & Motivation

Echocardiography (Echo) is one of the most important non-invasive imaging modalities for cardiac function assessment, yet its clinical value depends critically on expert interpretation. Clinicians must simultaneously coordinate three capabilities:

**"Eyes" (Visual Observation)**: Recognizing diverse cardiac views such as the apical two-chamber, four-chamber, and parasternal long-axis views.

**"Hands" (Manual Operation)**: Localizing and segmenting cardiac structures and quantitatively measuring key parameters.

**"Minds" (Expert Reasoning)**: Acquiring clinical knowledge, integrating multimodal evidence, and performing reliable diagnostic reasoning.

Existing approaches follow two lines of development, each with notable limitations:

- **Task-specific deep learning models** (e.g., MemSAM, EchoONE): Proficient at isolated tasks such as segmentation, possessing "eyes + hands" but lacking "minds" and thus incapable of autonomous complete diagnostic reasoning.
- **Multimodal large language models** (e.g., GPT-5, Qwen2.5-VL): Capable of visual question answering with "eyes + minds," but lacking domain-specific Echo knowledge and quantitative "hands," leading to clinically ungrounded reasoning.

A unified end-to-end solution integrating "eyes–hands–minds" therefore remains absent. EchoAgent is designed to fill this gap.

## Method

### Overall Architecture

EchoAgent comprises three core stages that simulate the complete clinician workflow from learning → observation → operation → reasoning:

1. **Expertise-Driven Cognition Engine (EDC)**: Constructs a domain knowledge base, endowing the agent with a professional "mind."
2. **Hierarchical Collaboration Toolkit (HC)**: Equips the agent with perception and operation tools, providing "eyes" and "hands."
3. **Orchestrated Reasoning Hub (OR)**: Coordinates the above components to enable end-to-end interpretable reasoning.

### Key Designs

1. **Expertise-Driven Cognition Engine (EDC)**:

    - Domain knowledge is sourced from four authoritative repositories: the UMLS medical knowledge base and AHA/ASE/EACVI echocardiography guidelines.
    - Heterogeneous documents are decomposed into semantic knowledge primitives $P=\{p_1, p_2, \ldots, p_D\}$.
    - A medical concept encoder $f_\theta(\cdot)$ maps knowledge into a high-dimensional semantic space.
    - Indexing is organized across 14 cardiac anatomical regions (left ventricle, mitral valve, aortic valve, etc.).
    - A RAG retrieval mechanism supports anatomy-specific knowledge retrieval, fetching the top-$k$ most relevant primitives to construct a structured knowledge base $R$.

2. **Hierarchical Collaboration Toolkit (HC)**: A three-tier progressive structure.

    - **Perceptual Layer**: Employs the EchoPrime foundation model to parse video streams and automatically classify echocardiographic view types (48 views).
    - **Operational Layer**: Applies a USFM-based customized segmentation model to automatically delineate key cardiac structures (left ventricle, aorta, right ventricle, left atrium, etc.).
    - **Functional Layer**: Integrates fine-tuned versions of USFM and EchoPrime to compute key clinical parameters including ejection fraction (EF), chamber dimensions, and right atrial pressure.

3. **Orchestrated Reasoning Hub (OR Hub)**: The core reasoning engine.

    - **Knowledge Retrieval and Task Decomposition**: Given a diagnostic query $Q$, the hub retrieves the relevant knowledge base $R_{a_q}$, decomposes it into an executable step sequence $S=\{s_1,\ldots,s_n\}$, and maps each step to the optimal tool.
    - **Dynamic Reasoning Graph Construction**: Incrementally constructs a multimodal reasoning graph $G=(N,E)$, where nodes encode diagnostic concepts, evidence, and data anchors, and edges represent generation, support–contradiction, and derivation relationships.
    - **Adaptive Reasoning Workflow**: Hypothesis confidence is assessed via Bayesian posterior estimation $P(h_m|G(t)) \propto P(G(t)|h_m) \cdot P(h_m)$; when confidence falls below threshold, supplementary examinations (e.g., switching views for re-measurement) are automatically triggered until the evidence graph reaches a consistency threshold.

### Loss & Training

- The base MLLM is Qwen3-VL-Plus.
- Foundation models in the HC toolkit (EchoPrime, USFM) are fine-tuned separately on echocardiographic data.
- The knowledge base is dynamically retrieved via the RAG mechanism, requiring no end-to-end joint training.
- The CAMUS dataset is split into training/validation/test sets at a 7:1:2 ratio.
- EF computation follows Simpson's biplane method (SMOD).

## Key Experimental Results

### Main Results

**Single-structure task (EF grading, CAMUS dataset)**:

| Method | Type | Normal Acc | Mildly Reduced Acc | Considerably Reduced Acc | Mean Acc |
|------|------|-----------|-------------------|------------------------|---------|
| EchoONE | Task-specific | 74.00 | 64.00 | 80.00 | 72.67 |
| GPT-5 | General MLLM | 44.00 | 61.00 | 55.00 | 53.33 |
| GPT-5* (augmented) | E-H-M | 78.00 | 69.00 | 89.00 | 78.67 |
| **EchoAgent** | **E-H-M** | **88.00** | **80.00** | **92.00** | **80.00** |

**Multi-structure task (EchoQA, MIMIC-EchoQA dataset)**:

| Method | Pericardium | Aortic Valve | Mitral Valve | Ventricles | Atria | Vessels | Others |
|------|------------|-------------|-------------|-----------|-------|---------|--------|
| GPT-5 | 60.98 | 40.91 | 36.78 | 26.32 | 36.99 | 38.71 | 44.44 |
| GPT-5* | 69.51 | 60.61 | 59.77 | 47.89 | 63.01 | 41.94 | 66.67 |
| **EchoAgent** | **84.15** | **82.58** | **81.61** | **75.26** | **80.82** | **77.42** | **70.37** |

EchoAgent achieves accuracy exceeding 70% across all 7 anatomical structure categories, outperforming the best-performing MLLM by an average of 31.45%.

### Ablation Study

| Configuration | EF Grading Acc | EchoQA Acc | Notes |
|------|---------------|------------|------|
| Baseline (eyes+minds) | 35.00 | 43.57 | Qwen3-VL-Plus only |
| +EDC (expert mind) | 50.00 (+15.00) | 51.45 (+7.88) | Domain knowledge added |
| +HC (skilled hands) | 73.00 (+37.00) | 59.97 (+16.40) | Operational tools added |
| +EDC+HC+OR (full) | **80.00** (+45.00) | **79.42** (+35.85) | Full collaboration |

### Key Findings

1. Adding tools alone (GPT-5*) yields substantial gains (+48.67%), yet still falls short of EchoAgent, demonstrating that tools, knowledge, and orchestration are all indispensable.
2. EchoAgent achieves AUROC of 98.43%, 87.79%, and 93.88% at the three EF grading thresholds, indicating strong clinical utility.
3. General MLLMs exhibit highly uneven performance across anatomical structures (e.g., GPT-5 attains only 26.32% on Ventricles), whereas EchoAgent maintains consistently high performance.
4. Quantitative operational capability ("hands") contributes most to EF grading (+37%), while the knowledge engine is more critical for knowledge-intensive tasks.

## Highlights & Insights

1. **Successful application of the agent paradigm**: Modeling medical image analysis as an agent workflow rather than a single model is a promising direction. The "eyes–hands–minds" analogy is both intuitive and effective.
2. **Dynamic reasoning graph design**: Constructing an incrementally built multimodal reasoning graph enables traceable reasoning, representing a significant improvement over black-box LLM outputs.
3. **Adaptive mechanism**: The closed-loop design that automatically gathers supplementary evidence under low-confidence conditions mirrors the iterative confirmation process in actual clinical practice.
4. **Broad coverage**: Support for 48 view types and 14 anatomical structures approaches the clinical scope of a comprehensive echocardiographic examination.

## Limitations & Future Work

1. **Real-time performance unverified**: The paper does not discuss inference latency; multi-round tool invocations may be time-consuming and potentially incompatible with real-time clinical requirements.
2. **Dependence on upstream model quality**: The accuracy of segmentation models in the HC toolkit directly affects higher-level reasoning, introducing error propagation risks.
3. **Limited dataset scale**: CAMUS contains only 500 cases and MIMIC-EchoQA only 622, leaving generalizability to be validated at larger scale.
4. **Knowledge base update mechanism unclear**: As clinical guidelines evolve continuously, how the EDC engine incorporates new knowledge is not addressed.
5. **Insufficient comparison with other agent systems**: Methods such as MedRAX and other medical agent frameworks are not included for comparison.

## Related Work & Insights

- **MedRAX**: A conceptually similar medical agent approach, but targeting chest X-rays rather than echocardiography.
- **EchoPrime / EchoONE**: Serving as foundation models within EchoAgent's toolkit, these works demonstrate the value of domain-specific foundation models.
- **LangChain**: Provides the engineering foundation for the agent framework implementation.
- Insight: This paradigm could be extended to other complex medical imaging modalities (e.g., multi-sequence CT/MRI analysis), with the key challenge being the design of modality-specific toolkits.

## Rating

- Novelty: ⭐⭐⭐⭐ — Applying the agent paradigm to echocardiography interpretation is relatively novel, though the overall framework of agent + RAG + tool calling is not entirely unprecedented.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two datasets with comprehensive ablation and comparison, though dataset scale is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ — The "eyes–hands–minds" analogy is woven consistently throughout, with clear logic and excellent readability.
- Value: ⭐⭐⭐⭐ — Substantial potential for real-world medical AI applications, demonstrating the engineering value of the agent paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Bridging the Skill Gap in Clinical CBCT Interpretation with CBCTRepD](bridging_the_skill_gap_in_clinical_cbct_interpreta.md)
- [\[CVPR 2026\] CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation](cure_curriculum-guided_multi-task_training_for_reliable_anatomy_grounded_report_.md)
- [\[ICLR 2026\] Causal Interpretation of Neural Network Computations with Contribution Decomposition](../../ICLR2026/medical_imaging/causal_interpretation_of_neural_network_computations_with_contribution_decomposi.md)
- [\[ICCV 2025\] GDKVM: Echocardiography Video Segmentation via Spatiotemporal Key-Value Memory with Gated Delta Rule](../../ICCV2025/medical_imaging/gdkvm_echocardiography_video_segmentation_via_spatiotemporal_key-value_memory_wi.md)
- [\[CVPR 2026\] OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)

</div>

<!-- RELATED:END -->
