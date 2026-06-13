---
title: >-
  [Paper Note] Thinking Like a Botanist: Challenging Multimodal Language Models with Intent-Driven Chain-of-Inquiry
description: >-
  [ACL 2026][Multimodal VLM][Plant Pathology VQA] This paper proposes the PlantInquiryVQA benchmark and the Chain-of-Inquiry (CoI) framework, containing 24,950 plant images and 138…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Plant Pathology VQA"
  - "Chain-of-Inquiry"
  - "Multi-step Visual Reasoning"
  - "Diagnostic Reasoning"
  - "Multimodal Evaluation"
date: 2026-05-08
content_hash: 7460c1c291813185
---

# Thinking Like a Botanist: Challenging Multimodal Language Models with Intent-Driven Chain-of-Inquiry

**Conference**: ACL 2026  
**arXiv**: [2604.20983](https://arxiv.org/abs/2604.20983)  
**Code**: [github.com/syed-nazmus-sakib/PlantInquiryVQA](https://github.com/syed-nazmus-sakib/PlantInquiryVQA)  
**Area**: Medical Imaging / Plant Pathology Diagnosis  
**Keywords**: Plant Pathology VQA, Chain-of-Inquiry, Multi-step Visual Reasoning, Diagnostic Reasoning, Multimodal Evaluation

## TL;DR
This paper proposes the PlantInquiryVQA benchmark and the Chain-of-Inquiry (CoI) framework, containing 24,950 plant images and 138,068 QA pairs. It simulates the adaptive diagnostic questioning strategies of botanists to evaluate the multi-step visual reasoning capabilities of 18 MLLMs in plant pathology diagnosis. Findings show that structured questioning significantly improves diagnostic accuracy and reduces hallucinations, though even the strongest model achieves a clinical utility score of only 0.188.

## Background & Motivation

**Background**: VQA datasets are a core paradigm for evaluating multimodal reasoning, having expanded into medical imaging and scientific image analysis. Advanced VQA benchmarks now focus on multi-panel, multiple-choice, and vision-language grounded QA pairs. Datasets in the agricultural vision domain (e.g., PlantVillage, PlantDoc) primarily target classification and segmentation tasks and do not support interactive QA reasoning.

**Limitations of Prior Work**: Current VQA benchmarks are fundamentally "question-centric"—treating each image as the input for an independent query rather than the starting point of a goal-oriented adaptive inquiry. In specialized fields like plant pathology, effective visual reasoning emerges not from answering isolated questions but from a series of interdependent inquiries, where each question builds on prior observations following a sequential narrative trajectory. Expert botanists conduct holistic assessments through a hierarchical, evidence-driven questioning strategy: species identification $\rightarrow$ disease diagnosis $\rightarrow$ prognosis prediction.

**Key Challenge**: While LLMs have made significant progress in achieving Chain-of-Thought reasoning, similar multi-step exploration has not been fully investigated in VQA dataset design. CoT is typically viewed as a prompting strategy or an implicit capability of model architectures rather than an explicit structural requirement of the dataset itself.

**Goal**: Construct a dataset-level Chain-of-Inquiry framework so that the question sequences themselves reflect the adaptive, decision-driven workflows of domain experts.

**Key Insight**: In plant pathology, each sample receives unique diagnostic considerations based on its visual appearance. When symptoms are ambiguous, experts prioritize differential diagnosis and comparative visual analysis; when symptoms are severe, they shift toward disease management and prevention strategies. The sequence and intent of questioning are as important as the answers themselves.

**Core Idea**: Formalize a Chain-of-Inquiry framework that models diagnostic trajectories as ordered QA sequences conditioned on visual cues and cognitive intent, automatically adjusting the questioning strategy from diagnosis $\rightarrow$ prognosis $\rightarrow$ management based on disease severity.

## Method

### Overall Architecture
The construction of PlantInquiryVQA involves three stages: (1) Using VLMs to extract fine-grained visual cues from plant images according to an expert-designed schema; (2) Building a botanical knowledge structure to map disease severity to diagnostic intent; (3) A dynamic LLM generation pipeline that generates dialogue trajectories by injecting specific reasoning modules based on diagnostic intent and visual evidence. The dataset covers 34 crop species, featuring 7 question categories and 12 unique CoI trajectories.

### Key Designs

1. **Chain-of-Inquiry Formalization**:
    - **Function**: Models diagnostic reasoning as a visual-semantic trajectory conditioned on diagnostic intent.
    - **Mechanism**: For a given image $x$ and visual cues $v_x$, CoI is defined as an ordered sequence of $T$ dialogue turns $C(x, v_x) = \langle (q_1, a_1), \ldots, (q_T, a_T) \rangle$, where each question $q_t$ is conditioned on visual evidence $v_x$, prior context $H_{t-1}$, and latent diagnostic intent $k \in \mathcal{K}$. The intent space is divided into three levels: diagnosis ($k_D$, identifying health status and differential diagnosis), prognosis ($k_P$, predicting disease trajectory and causal etiology), and management ($k_M$, prescribing strategies and counterfactual prevention reasoning). Mild symptoms $\rightarrow$ diagnostic intent; moderate $\rightarrow$ prognostic intent; severe $\rightarrow$ management intent.
    - **Design Motivation**: Samples of different severity require different questioning strategies—mild symptoms require differential diagnosis to distinguish similar pathologies, while severe damage requires a focus on immediate remediation and counterfactual analysis. Explicitly encoding intent allows the dataset to test whether models can adaptively adjust their reasoning chains.

2. **Visual Cue Extraction and CoI Categorization**:
    - **Function**: Extracts structured diagnostic features from plant images and categorizes 12 types of CoI trajectories.
    - **Mechanism**: Six botanists (2 PhD level + 4 graduate level) were recruited to define a "Visual Parsing Schema" covering three diagnostic dimensions: symptomatology, distribution patterns, and disease severity quantification. Qwen3-VL-4B was used to automatically extract visual cues (73.6% accuracy), cross-validated by GPT-4V, with experts performing clinical fact-checks on labeled instances and 5,000 random samples (factuality score of 93.8%). Experts categorized standard diagnostic inquiries into 7 classes: visual perception & grounding, diagnostic reasoning, causal reasoning, risk assessment, prognostic prediction, prescriptive reasoning, and counterfactual reasoning. The 12 CoI trajectories cover 4 health states $\times$ 3 severities $\times$ 2 instance diversities $\times$ 3 cognitive intents.
    - **Design Motivation**: Classical plant pathology literature describes biological stages of diagnosis but lacks a standard taxonomy for visual dialogue inquiries. This gap is bridged by having experts clinically evaluate 600 random samples and record their questioning strategies.

3. **Structured Generation Pipeline**:
    - **Function**: Dynamically assembles dialogue trajectories for each plant sample.
    - **Mechanism**: The pipeline is driven by a configuration tuple $T = (c, s, k_s, V_{cues})$ (biological condition, severity, intent, visual cues). Cognitive goal $k$ regulates information density based on severity $s$. Qwen2.5-7B-Instruct dynamically assembles dialogue trajectories from question templates and injects specific reasoning modules (e.g., temporal_evolution, remediation_strategy) to enhance complexity.
    - **Design Motivation**: Decoupling the configuration tuple allows for the generation of diverse reasoning chains even for the same image (e.g., suggesting management for mild vs. severe cases), ensuring coverage across the full diagnostic difficulty spectrum from routine identification to complex multi-step clinical reasoning.

### Loss & Training
PlantInquiryVQA is an evaluation benchmark. Performance is measured using standard lexical metrics (F1, BLEU-4, ROUGE-L) and seven domain-specific scores: disease identification ($S_{dis}$), safety ($S_{safe}$), clinical utility ($S_{clin}$), visual grounding ($S_{vg}$), visual feature extraction efficiency (E), popularity bias (B), and cross-class fairness (F).

## Key Experimental Results

### Main Results (Performance of 18 MLLMs, Key Metrics)

| Model | F1 | Disease Identification | Clinical Utility | Safety | Visual Grounding |
|------|-----|---------|-----------|--------|---------|
| Gemini-3-Flash | **0.255** | **0.444** | **0.188** | **0.147** | 0.259 |
| Seed-1.6-Flash | 0.226 | 0.344 | 0.120 | 0.075 | 0.394 |
| Grok-4.1-Fast | 0.203 | 0.224 | 0.067 | 0.009 | **0.498** |
| Ministral-3B | 0.166 | 0.189 | 0.059 | 0.020 | 0.372 |

### Ablation Study (Impact of Structured Questioning on Diagnostic Efficiency, Guided vs. Scaffolded)

| Model | Scaffolded Efficiency | Guided Efficiency | Gain |
|------|---------------|-----------|---------|
| Gemini-2.5-Flash | 2.60 | 3.67 | +41.15% |
| Qwen2.5-VL-32B | 1.60 | 2.94 | +83.75% |
| Gemma-3-27B | 1.88 | 2.38 | +26.60% |

### Key Findings
- **Significant Domain Gap**: Even the strongest model, Gemini-3-Flash, achieves a clinical utility of only 0.188 and safety of 0.147, falling far short of requirements for autonomous deployment.
- **"Seeing" is not "Diagnosing"**: Grok-4.1-Fast has the highest visual grounding (0.498) but the lowest disease identification (0.224), indicating that accurate description of visual symptoms does not equate to correct diagnosis.
- **Structured Questioning Reduces Hallucination**: Question-guided diagnosis is significantly more accurate than direct diagnosis across all severity levels. Specific questions force the model to focus on fine-grained features (e.g., lesion margins, presence of halos), constraining the search space.
- **CoI Structure is the Main Driver**: The Cascading mode (using the model's own prior answers) retains 96.3% of the efficiency and 81.7% of the diagnostic accuracy of the Guided mode, suggesting that the structured questioning itself (rather than perfect memory) drives improvement.

## Highlights & Insights
- **Chain-of-Inquiry as a Dataset-Level Structural Constraint**: Elevating CoT from a prompting strategy to an explicit structural requirement of the dataset is a novel contribution that can be generalized to any field requiring multi-step reasoning evaluation (e.g., medical image diagnosis, engineering troubleshooting).
- **Intent-Driven Adaptive Questioning**: Automatically adjusting questioning strategies (diagnosis $\rightarrow$ prognosis $\rightarrow$ management) based on disease severity. This intent-visual coupling design philosophy could inspire the dialogue strategy design of Agent systems.
- **Decoupling of Visual Grounding and Diagnostic Reasoning**: Revealing that "describing symptoms" and "making a diagnosis" are separable capabilities points toward specific directions for model improvement.

## Limitations & Future Work
- Plant pathology often requires multi-sensory information (tactile, environmental), and a single-frame image is insufficient to fully replicate expert diagnostic workflows.
- Even top-tier models commit "false safety" errors (misidentifying diseased samples as healthy); they currently serve only as decision-support tools rather than replacements.
- The benchmark is only in English, limiting accessibility for smallholder farmers in non-English speaking regions.
- Visual cue extraction relies primarily on Qwen3-VL-4B automation, and some cues may lack precision.

## Related Work & Insights
- **vs. PlantVillage/PlantDoc**: Those support only classification/segmentation without interactive reasoning. PlantInquiryVQA provides multi-step structured QA.
- **vs. Medical VQA (PMC-VQA, VQA-RAD)**: Those focus on human medicine and single-turn QA, whereas PlantInquiryVQA targets plant pathology and multi-step chained reasoning.
- **vs. BloomVQA**: While organized by Bloom's taxonomy, it relies on static classification. PlantInquiryVQA conditions the question sequence on visual evidence and diagnostic intent.

## Rating
- Novelty: ⭐⭐⭐⭐ The paradigm shift of CoI from prompting strategy to dataset structure is a novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ 18 models, full ablations, and diverse evaluation metrics, though dataset construction is partially automated.
- Writing Quality: ⭐⭐⭐⭐ Framework design is clear and experimental analysis is deep, though some tables are excessive.
- Value: ⭐⭐⭐⭐ Provides an important benchmark for agricultural AI diagnostic reasoning; the CoI approach has cross-domain transfer value.
- Overall: ⭐⭐⭐⭐ Offers a novel and practical perspective, revealing the true gap for MLLMs in professional diagnostic reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Enhancing Multimodal Large Language Models for Ancient Chinese Character Evolution Analysis via Glyph-Driven Fine-Tuning](enhancing_multimodal_large_language_models_for_ancient_chinese_character_evoluti.md)
- [\[CVPR 2026\] Circuit Tracing in Vision-Language Models: Understanding the Internal Mechanisms of Multimodal Thinking](../../CVPR2026/multimodal_vlm/circuit_tracing_in_vision-language_models_understanding_the_internal_mechanisms_.md)
- [\[ICCV 2025\] GRAB: A Challenging GRaph Analysis Benchmark for Large Multimodal Models](../../ICCV2025/multimodal_vlm/grab_a_challenging_graph_analysis_benchmark_for_large_multimodal_models.md)
- [\[AAAI 2026\] Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](../../AAAI2026/multimodal_vlm/plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)
- [\[CVPR 2026\] Thinking Diffusion: Penalize and Guide Visual-Grounded Reasoning in Diffusion Multimodal Language Models](../../CVPR2026/multimodal_vlm/thinking_diffusion_penalize_and_guide_visual-grounded_reasoning_in_diffusion_mul.md)

</div>

<!-- RELATED:END -->
