---
title: >-
  [Paper Note] Thinking Like a Botanist: Challenging Multimodal Language Models with Intent-Driven Chain-of-Inquiry
description: >-
  [ACL 2026][Medical Imaging][Plant Pathology VQA] This paper proposes PlantInquiryVQA, a benchmark comprising 24,950 plant images and 138,068 question–answer pairs…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "Plant Pathology VQA"
  - "Chain-of-Inquiry"
  - "Multi-step Visual Reasoning"
  - "Diagnostic Reasoning"
  - "Multimodal Evaluation"
date: 2026-05-08
content_hash: 552b7206238a9bda
---

# Thinking Like a Botanist: Challenging Multimodal Language Models with Intent-Driven Chain-of-Inquiry

**Conference**: ACL 2026
**arXiv**: [2604.20983](https://arxiv.org/abs/2604.20983)
**Code**: [github.com/syed-nazmus-sakib/PlantInquiryVQA](https://github.com/syed-nazmus-sakib/PlantInquiryVQA)
**Area**: Medical Imaging / Plant Pathology Diagnosis
**Keywords**: Plant Pathology VQA, Chain-of-Inquiry, Multi-step Visual Reasoning, Diagnostic Reasoning, Multimodal Evaluation

## TL;DR
This paper proposes PlantInquiryVQA, a benchmark comprising 24,950 plant images and 138,068 question–answer pairs, along with a Chain-of-Inquiry (CoI) framework that simulates the adaptive diagnostic inquiry strategies of expert botanists. The benchmark is used to evaluate 18 MLLMs on multi-step visual reasoning for plant pathology diagnosis. Results show that structured inquiry significantly improves diagnostic accuracy and reduces hallucinations; nonetheless, even the strongest model achieves a clinical utility score of only 0.188.

## Background & Motivation

**Background**: VQA datasets constitute a core paradigm for evaluating multimodal reasoning and have been extended to domains such as medical imaging and scientific image analysis. State-of-the-art VQA benchmarks now address multi-panel, multiple-choice, and vision-language grounding scenarios. Agricultural vision datasets (e.g., PlantVillage, PlantDoc) primarily target classification and segmentation tasks and do not support interactive question-answering reasoning.

**Limitations of Prior Work**: Existing VQA benchmarks are fundamentally *question-centric*—treating each image as an independent query input rather than as the starting point for goal-directed, adaptive inquiry. In specialized domains such as plant pathology, effective visual reasoning does not consist of answering isolated questions; rather, it emerges from a sequence of interdependent inquiries in which each question builds on prior observations along a sequential narrative trajectory. Expert botanists conduct holistic assessments through hierarchical, evidence-driven inquiry strategies progressing from species identification → disease diagnosis → prognostic prediction.

**Key Challenge**: While significant progress has been made in enabling LLMs to perform Chain-of-Thought reasoning, analogous multi-step exploration has not been sufficiently instantiated at the level of VQA dataset design. CoT is typically regarded as a prompting strategy or an implicit architectural capability, rather than an explicit structural requirement encoded in the dataset itself.

**Goal**: To construct a dataset-level Chain-of-Inquiry framework in which the question sequence itself reflects the adaptive, decision-driven workflow of domain experts.

**Key Insight**: In plant pathology, each specimen warrants unique diagnostic consideration based on its visual appearance. When symptoms are ambiguous, experts prioritize differential diagnosis and comparative visual analysis; when symptoms are severe, experts shift toward disease management and prevention strategies. The sequence and intent of inquiry are as diagnostically important as the answers themselves.

**Core Idea**: Formalize the Chain-of-Inquiry framework by modeling diagnostic trajectories as ordered question–answer sequences conditioned on visual cues and cognitive intent, with questioning strategies automatically adjusted along the diagnostic → prognostic → management axis according to disease severity.

## Method

### Overall Architecture
PlantInquiryVQA is constructed in three phases: (1) fine-grained visual cues are extracted from plant images using a VLM guided by an expert-designed schema; (2) a botanical knowledge structure is established that maps disease severity to diagnostic intent; and (3) a dynamic LLM-based generation pipeline produces dialogue trajectories by injecting reasoning modules specific to each diagnostic intent and the associated visual evidence. The dataset covers 34 crop species and is organized around 7 question categories and 12 distinct CoI trajectory types.

### Key Designs

1. **Chain-of-Inquiry Formalization**

    - **Function**: Models diagnostic reasoning as a visual-semantic trajectory conditioned on diagnostic intent.
    - **Mechanism**: Given an image $x$ and its visual cues $v_x$, a CoI is defined as an ordered $T$-turn dialogue $C(x, v_x) = \langle (q_1, a_1), \ldots, (q_T, a_T) \rangle$, where each question $q_t$ is conditioned on the visual evidence $v_x$, the prior context $H_{t-1}$, and a latent diagnostic intent $k \in \mathcal{K}$. The intent space is partitioned into three levels: diagnostic ($k_D$, identifying health status and differential diagnosis), prognostic ($k_P$, predicting disease trajectory and causal etiology), and management ($k_M$, prescribing remediation strategies and counterfactual prevention reasoning). Mild symptoms map to diagnostic intent, moderate to prognostic intent, and severe to management intent.
    - **Design Motivation**: Specimens at different severity levels require distinct questioning strategies—mild symptoms necessitate differential diagnosis to distinguish similar pathologies, whereas severe lesions call for a focus on immediate remediation and counterfactual analysis. Explicitly encoding intent into the dataset enables assessment of whether models can adaptively adjust their reasoning chains.

2. **Visual Cue Extraction and CoI Classification**

    - **Function**: Extracts structured diagnostic features from plant images and classifies them into 12 CoI trajectory types.
    - **Mechanism**: Six botanists (2 PhD-level, 4 graduate-level) were recruited to define a "Visual Parsing Schema" spanning three diagnostic dimensions: symptomatology, distribution patterns, and disease severity quantification. Qwen3-VL-4B automatically extracts visual cues (accuracy 73.6%), with GPT-4V used for cross-validation; experts performed clinical factuality checks on annotated instances and 5,000 random samples (factuality score 93.8%). Experts categorized standard diagnostic inquiries into 7 types: visual perception and grounding, diagnostic reasoning, causal reasoning, risk assessment, prognostic prediction, prescriptive reasoning, and counterfactual reasoning. The 12 CoI trajectory types span 4 health states × 3 severity levels × 2 instance diversity variants × 3 cognitive intents.
    - **Design Motivation**: Classical plant pathology literature describes the biological stages of diagnosis but lacks a standardized taxonomy for visual dialogue inquiry. This gap was addressed by having experts clinically evaluate 600 random samples and document their questioning strategies.

3. **Structured Generation Pipeline**

    - **Function**: Dynamically assembles dialogue trajectories tailored to each plant specimen.
    - **Mechanism**: The pipeline is driven by a configuration tuple $T = (c, s, k_s, V_{cues})$ (biological condition, severity, intent, visual cues). The cognitive objective $k$ modulates information density according to severity $s$. Qwen2.5-7B-Instruct is used to dynamically assemble dialogue trajectories from question templates, with reasoning modules (e.g., `temporal_evolution`, `remediation_strategy`) injected to increase complexity.
    - **Design Motivation**: Decoupling the configuration tuple enables the generation of diverse reasoning chains even for the same image (e.g., management recommendations for mild vs. severe cases), ensuring coverage of the full diagnostic difficulty spectrum from routine identification to complex multi-step clinical reasoning.

### Loss & Training
PlantInquiryVQA is an evaluation benchmark. Assessment employs standard lexical metrics (F1, BLEU-4, ROUGE-L) alongside seven domain-specific scores: disease identification ($S_{dis}$), safety ($S_{safe}$), clinical utility ($S_{clin}$), visual grounding ($S_{vg}$), visual feature extraction efficiency ($E$), popularity bias ($B$), and cross-category fairness ($F$).

## Key Experimental Results

### Main Results (Performance of 18 MLLMs on Key Metrics)

| Model | F1 | Disease ID | Clinical Utility | Safety | Visual Grounding |
|---|---|---|---|---|---|
| Gemini-3-Flash | **0.255** | **0.444** | **0.188** | **0.147** | 0.259 |
| Seed-1.6-Flash | 0.226 | 0.344 | 0.120 | 0.075 | 0.394 |
| Grok-4.1-Fast | 0.203 | 0.224 | 0.067 | 0.009 | **0.498** |
| Ministral-3B | 0.166 | 0.189 | 0.059 | 0.020 | 0.372 |

### Ablation Study (Effect of Structured Inquiry on Diagnostic Efficiency: Guided vs. Scaffolded)

| Model | Scaffolded Efficiency | Guided Efficiency | Efficiency Gain |
|---|---|---|---|
| Gemini-2.5-Flash | 2.60 | 3.67 | +41.15% |
| Qwen2.5-VL-32B | 1.60 | 2.94 | +83.75% |
| Gemma-3-27B | 1.88 | 2.38 | +26.60% |

### Key Findings
- **Substantial domain gap**: Even the strongest model, Gemini-3-Flash, achieves a clinical utility score of only 0.188 and a safety score of 0.147, far below the threshold required for autonomous deployment.
- **"Seeing" does not imply "diagnosing"**: Grok-4.1-Fast achieves the highest visual grounding score (0.498) yet the lowest disease identification score (0.224), demonstrating that accurately describing visual symptoms does not entail the ability to reach a correct diagnosis.
- **Structured inquiry reduces hallucinations**: Inquiry-guided diagnosis is significantly more accurate than direct diagnosis across all severity levels; specific questions compel models to attend to fine-grained features (e.g., lesion margins, halo presence), thereby constraining the search space.
- **CoI structure is the primary driver**: The Cascading mode (conditioning on the model's own prior responses) retains 96.3% of the efficiency and 81.7% of the diagnostic accuracy of the Guided mode, indicating that structured inquiry itself—rather than perfect memory—drives the observed improvements.

## Highlights & Insights
- **Chain-of-Inquiry as a dataset-level structural constraint**: Elevating CoT from a prompting strategy to an explicit structural requirement of the dataset represents a paradigm shift transferable to any domain requiring multi-step reasoning evaluation (e.g., medical image diagnosis, engineering fault diagnosis).
- **Intent-driven adaptive inquiry**: Automatically adjusting the questioning strategy according to disease severity (diagnostic → prognostic → management) offers a design principle—intent-visual coupling—that can inform dialogue strategy design in agentic systems.
- **Decoupled finding on visual grounding vs. diagnostic reasoning**: The results reveal that "describing symptoms" and "making a diagnosis" are separable capabilities, pointing toward concrete directions for model improvement.

## Limitations & Future Work
- Plant pathology typically requires multisensory information (tactile, environmental, etc.); single-frame images are insufficient to fully replicate expert diagnostic workflows.
- Even top-tier models produce "false safety" errors (misclassifying diseased specimens as healthy), relegating current systems to decision-support tools rather than autonomous replacements.
- The benchmark is English-only, limiting accessibility for smallholder farmers in non-English-speaking regions.
- Visual cue extraction relies primarily on Qwen3-VL-4B automation, and some extracted cues may lack sufficient precision.

## Related Work & Insights
- **vs. PlantVillage/PlantDoc**: These datasets support only classification/segmentation and not interactive reasoning. PlantInquiryVQA provides multi-step structured question answering.
- **vs. Medical VQA (PMC-VQA, VQA-RAD)**: These focus on human medicine and are single-turn; PlantInquiryVQA targets plant pathology and employs multi-step chain reasoning.
- **vs. BloomVQA**: BloomVQA organizes questions according to Bloom's taxonomy but relies on a static classification scheme; PlantInquiryVQA conditions the question sequence on visual evidence and diagnostic intent.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The paradigm shift of CoI from a prompting strategy to a dataset-level structural constraint is a genuine contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 18 models, comprehensive ablations, and multiple evaluation metrics; dataset construction is partially automated.
- **Writing Quality**: ⭐⭐⭐⭐ Framework design is clearly articulated and experimental analysis is thorough, though some sections contain excessive tables.
- **Value**: ⭐⭐⭐⭐ Provides an important benchmark for diagnostic reasoning in agricultural AI; the CoI concept has cross-domain transfer potential.
- **Overall**: ⭐⭐⭐⭐ The approach is both novel and practically grounded, revealing genuine gaps in MLLM capabilities for specialized diagnostic reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning](dr_assistant_enhancing_clinical_diagnostic_inquiry_via_structured_diagnostic_rea.md)
- [\[CVPR 2026\] Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation](../../CVPR2026/medical_imaging/multimodal_protein_language_models_for_enzyme_kinetic_parameters_from_substrate_.md)
- [\[CVPR 2026\] MedKCO: Medical Vision-Language Pretraining via Knowledge-Driven Cognitive Orchestration](../../CVPR2026/medical_imaging/medkco_medical_vision-language_pretraining_via_knowledge-driven_cognitive_orches.md)
- [\[ACL 2026\] Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation](text-attributed_knowledge_graph_enrichment_with_large_language_models_for_medica.md)
- [\[ACL 2026\] RiTeK: A Dataset for Large Language Models Complex Reasoning over Textual Knowledge Graphs in Medicine](ritek_a_dataset_for_large_language_models_complex_reasoning_over_textual_knowled.md)

</div>

<!-- RELATED:END -->
