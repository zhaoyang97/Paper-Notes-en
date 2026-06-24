---
title: >-
  [Paper Note] SPA-VL: A Comprehensive Safety Preference Alignment Dataset for Vision Language Models
description: >-
  [CVPR 2025][Multimodal VLM][Safety Alignment] SPA-VL constructs a large-scale safety preference alignment dataset for VLMs containing 100,788 quadruplets (query, image, preferred response, dispreferred response), covering 6 domains, 13 categories, and 53 subcategories of harmful content. Based on diverse responses from 12 VLMs and a fully automated annotation pipeline, models trained with DPO/PPO achieve significant safety improvements while maintaining helpfulness.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Safety Alignment"
  - "Preference Learning"
  - "RLHF"
  - "VLM Safety"
  - "Dataset"
date: 2026-05-08
content_hash: 10f8e4edc29bcb1d
---

# SPA-VL: A Comprehensive Safety Preference Alignment Dataset for Vision Language Models

**Conference**: CVPR 2025  
**arXiv**: [2406.12030](https://arxiv.org/abs/2406.12030)  
**Code**: [https://sqrtizhang.github.io/SPA-VL/](https://sqrtizhang.github.io/SPA-VL/)  
**Area**: Multimodal VLM  
**Keywords**: Safety Alignment, Preference Learning, RLHF, VLM Safety, Dataset

## TL;DR
SPA-VL constructs a large-scale safety preference alignment dataset for VLMs containing 100,788 quadruplets (query, image, preferred response, dispreferred response), covering 6 domains, 13 categories, and 53 subcategories of harmful content. Based on diverse responses from 12 VLMs and a fully automated annotation pipeline, models trained with DPO/PPO achieve significant safety improvements while maintaining helpfulness.

## Background & Motivation
- **Background**: VLMs (such as GPT-4V, LLaVA) have made significant progress in multimodal understanding but face severe safety issues. Although LLMs have been safety-aligned, the safety alignment of visual encoders remains weak, making them vulnerable to attacks initiated through the visual modality.
- **Limitations of Prior Work**: (1) Most VLM safety work focuses on evaluation benchmarks or attack detection, lacking large-scale, high-quality datasets for training; (2) existing safety methods mostly rely on SFT or inference-time guard rails, leaving RLHF methods under-explored; (3) benign inputs can still trigger unsafe outputs, necessitating comprehensive alignment data.
- **Key Challenge**: Safety vs. Helpfulness—over-safety can lead to models refusing benign requests, requiring a balance between the two.
- **Goal**: Construct the first large-scale, fully automated safety preference alignment dataset for VLMs to make RLHF safety alignment feasible.
- **Key Insight**: Ensure data quality across three dimensions: data coverage breadth (53 subcategories), response diversity (12 models), and query complexity (3 query types).
- **Core Idea**: Large-scale + diverse + multi-objective (harmless + helpful) preference data = better safety alignment.

## Method

### Overall Architecture
The construction of SPA-VL is divided into three phases: (1) Image Collection: Relevant images are retrieved from LAION-5B based on a hierarchical harmful classification system; (2) Query Construction: Three types of queries (easy questions, hard questions, hard statements) are generated for each image, using Gemini for generation and refinement; (3) Preference Labelling: Diverse responses are collected from 12 VLMs, safety-labeled using MD-Judge, and then rank-ordered by GPT-4V based on harmlessness and helpfulness criteria. The entire pipeline is fully automated.

### Key Designs
1. **Hierarchical Harmful Content Classification System**:
    - **Function**: Ensures comprehensive coverage of harmful content in the dataset.
    - **Mechanism**: Establishes a hierarchical structure of 6 primary domains (toxicity, unfairness, information erosion, safety threats, illegal activities, fraud, etc.) $\rightarrow$ 15 secondary categories $\rightarrow$ 53 tertiary subcategories. This refers to the usage policies of OpenAI, LLaMA, Gemini, Claude, and the safety guidelines of Llama Guard.
    - **Design Motivation**: Harmful content types are diverse and multi-layered; coarse-grained categorization leads to insufficient coverage of specific types, whereas tertiary categorization ensures granularity and systematic coverage.

2. **Multi-Model and Multi-Type Q&A Generation**:
    - **Function**: Provides diverse queries and responses to enhance alignment robustness.
    - **Mechanism**: (1) Queries: For each image, Gemini generates three types of queries: easy questions (directly relevant), hard questions (refined to be more complex), and hard statements (in statement form) to cover different attack difficulties; (2) Responses: Responses are gathered from 12 different VLMs (both open-source and closed-source). For each query, responses from two models with different safety levels are randomly paired as a preference pair.
    - **Design Motivation**: Models trained on a single query type lack robustness (demonstrated by ablation studies); multi-source responses reduce model-specific biases and ensure that preferred/dispreferred responses span different safety levels.

3. **Fully Automated Preference Annotation Pipeline**:
    - **Function**: Generates high-quality preference data without human annotation.
    - **Mechanism**: First, MD-Judge is used to classify responses as safe or unsafe, then response pairs are selected from different safety-level model groups, and finally GPT-4V evaluates preferences based on both harmlessness and helpfulness (queries are run twice with swapped response orders to eliminate position bias).
    - **Design Motivation**: Human annotation is highly costly and involves exposure to harmful content. A fully automated pipeline is not only scalable but also prevents annotators from encountering harmful material.

### Loss & Training
- **DPO Training**: Freezes the visual encoder, updating the projection layer and LLM weights.
- **PPO Training**: Also freezes the visual encoder, using a reward model to guide training.
- Base Model: LLaVA-1.5 (7B)
- Data scale ablation was conducted thoroughly from 100 to 90K samples.

## Key Experimental Results

### Main Results

| Model | MM-SafetyBench Avg↓ | AdvBench vanilla↓ | AdvBench suffix↓ | HarmEval USR↓ |
|------|---------------------|-------------------|-------------------|--------------|
| LLaVA-7B (baseline) | 20.54 | 98.08 | 99.81 | 44.15 |
| LLaVA + SPA-VL-DPO | **0.60** | **0.00** | **0.00** | **0.00** |
| LLaVA + SPA-VL-PPO | 0.45 | 0.19 | 2.12 | 0.00 |
| Gemini-1.5-pro | 0.00 | 0.38 | 0.38 | 1.89 |
| mPLUG-Owl-7B | 21.88 | 100 | 100 | 52.45 |

### Ablation Study

| Data Scale | MM-SafetyBench Avg↓ | AdvBench vanilla↓ | HarmEval USR↓ | Help Score↑ |
|---------|---------------------|-------------------|--------------|-------------|
| 100 (DPO) | 19.94 | 97.89 | 43.40 | 51.00 |
| 1K | 18.75 | 91.54 | 26.04 | 58.50 |
| 10K | 2.53 | 5.77 | 0.38 | 63.00 |
| 90K | 1.49 | 0.00 | 0.75 | 70.00 |

### Key Findings
- The attack success rate of the trained model drops sharply from over 20% to near 0%, indicating exceptionally significant improvements in safety.
- Data scale is a critical factor: from 100 to 90K, both safety and helpfulness continuously improve, refuting the expectation of a trade-off.
- The diversity of response sources is crucial: training only with response pairs from safe models (Safe Group) still leaves a 65% attack success rate on AdvBench suffix, whereas using responses from all 12 models (All Group) reduces it to 6.54%.
- Mixing the 3 query types is more effective than using any single type, yielding overall superior safety.
- Both DPO and PPO methods are effective; DPO is slightly better for safety, while PPO is slightly better for helpfulness.
- Updating both the projection layer and the LLM during training yields better results than only updating the LLM.

## Highlights & Insights
- The dataset scale (100K+) and coverage (53 subcategories) are currently the most comprehensive in the field of VLM safety.
- The design of the fully automated construction process has immense practical value, bypassing the ethical issues of exposing human annotators to harmful content.
- The experimental finding that "safety and helpfulness improve simultaneously" challenges the intuitive assumption of a trade-off between the two, demonstrating that high-quality preference data can achieve both.
- Utilizing 12 different models as response sources ensures the diversity and generalization capability of the preference data.

## Limitations & Future Work
- Harmful images retrieved from LAION-5B via CLIP search may contain retrieval noise.
- Using GPT-4V as a preference annotator carries inherent bias, and a completely reliable ground truth is lacking.
- The method was validated only on LLaVA-1.5 (7B); performance on larger models and newer architectures remains unknown.
- The safety taxonomy, while comprehensive, is relatively static and requires continuous updates as socio-cultural norms evolve.
- Using jailbreak strategies with Gemini to generate hard queries raises certain safety and ethical concerns.

## Related Work & Insights
- **vs VLGuard**: VLGuard provides only 2,000 SFT training images, whereas SPA-VL delivers 100K+ RLHF preference pairs, pushing boundaries in both scale and methodological depth.
- **vs RLAIF-V**: RLAIF-V primarily addresses the hallucination problem, whereas SPA-VL focuses on safety alignment; their objectives differ, yet their DPO methodologies are complementary.
- **vs MLLM-Protector**: MLLM-Protector functions as a plug-and-play defense at inference time, whereas SPA-VL represents fundamental alignment during training; the two can be combined.
- **vs LLaVA-RLHF**: LLaVA-RLHF only trains the LLM layers, while SPA-VL demonstrates that updating both the projection layer and the LLM yields superior performance.
- **vs MM-SafetyBench**: MM-SafetyBench is an evaluation benchmark, while SPA-VL is a training dataset, making their roles complementary.

## Rating
- **Novelty**: ⭐⭐⭐ The core contribution lies in the dataset rather than method innovation, but the scale and systematic nature of the dataset successfully fill a critical gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extremely thorough ablations on data scale, model diversity, query types, and training strategies.
- **Writing Quality**: ⭐⭐⭐⭐ The dataset construction process is clearly described, with rich statistical analysis.
- **Value**: ⭐⭐⭐⭐ As the first large-scale VLM safety preference dataset, it provides essential support for future research in safety alignment.

---

> This note is based on a full reading of the paper, covering all sections of Dataset Construction, Experiments, and Analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment](task_preference_optimization_improving_multimodal_large_language_models_with_vis.md)
- [\[CVPR 2025\] Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion](florence-vl_enhancing_vision-language_models_with_generative_vision_encoder_and_.md)
- [\[CVPR 2025\] Debiasing Multimodal Large Language Models via Noise-Aware Preference Optimization](debiasing_multimodal_large_language_models_via_noise-aware_preference_optimizati.md)
- [\[CVPR 2025\] Post-pre-training for Modality Alignment in Vision-Language Foundation Models](post-pre-training_for_modality_alignment_in_vision-language_foundation_models.md)
- [\[CVPR 2025\] SmartCLIP: Modular Vision-language Alignment with Identification Guarantees](smartclip_modular_vision-language_alignment_with_identification_guarantees.md)

</div>

<!-- RELATED:END -->
