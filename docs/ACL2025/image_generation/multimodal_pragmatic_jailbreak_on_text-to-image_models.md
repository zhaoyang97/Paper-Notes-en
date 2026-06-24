---
title: >-
  [Paper Note] Multimodal Pragmatic Jailbreak on Text-to-image Models
description: >-
  [ACL 2025][Image Generation][Multimodal Jailbreak] This paper proposes a new type of attack called "Multimodal Pragmatic Jailbreak" (MPJ). By generating images containing visual text through T2I models, the image content and text content are safe when evaluated individually but yield unsafe content when combined. This study reveals that all tested models, including DALL·E 3, are vulnerable to this attack.
tags:
  - "ACL 2025"
  - "Image Generation"
  - "Multimodal Jailbreak"
  - "Text-to-image Models"
  - "Safety"
  - "Visual Text Rendering"
  - "Diffusion Models"
date: 2026-05-08
content_hash: f89ca523dd89c4cc
---

# Multimodal Pragmatic Jailbreak on Text-to-image Models

**Conference**: ACL 2025  
**arXiv**: [2409.19149](https://arxiv.org/abs/2409.19149)  
**Code**: [multimodalpragmatic.github.io](https://multimodalpragmatic.github.io/)  
**Area**: Image Generation  
**Keywords**: Multimodal Jailbreak, Text-to-image Models, Safety, Visual Text Rendering, Diffusion Models

## TL;DR

This paper proposes a new type of attack called "Multimodal Pragmatic Jailbreak" (MPJ). By generating images containing visual text through T2I models, the image content and text content are safe when evaluated individually but yield unsafe content when combined. This study reveals that all tested models, including DALL·E 3, are vulnerable to this attack.

## Background & Motivation

While diffusion models have achieved remarkable progress in image generation quality and text alignment, their security vulnerabilities are increasingly a cause for concern. Existing safety research primarily focuses on single-modality content filtering while overlooking a new form of attack:

- **Single-Modality Safety $\neq$ Multimodal Safety**: When T2I models generate images containing visual text, the image part and the text part may appear safe when evaluated separately, but their combination generates unsafe content. For instance, a seemingly harmless image of an animal paired with specific text can constitute hate speech.
- **Pragmatic Inspiration**: This attack is inspired by "pragmatics" in linguistics—where meaning is derived not only from the language system itself but also from contextual factors. When visual and textual information are combined, they can produce complex interactions such as implication, exaggeration, irony, and contrast, leading to the emergence of unsafe content.
- **Failure of Existing Defenses**: Common defense mechanisms, such as keyword blacklists, custom prompt filters, and NSFW image filters, fail to effectively counter this cross-modal pragmatic unsafe content.

## Method

### Overall Architecture

The work presented in this paper focuses on systematic study and benchmarking rather than proposing a new generation method. The core contributions include:
1. Constructing the MPUP dataset (1,400 multimodal pragmatic unsafe prompts)
2. Benchmarking 9 representative T2I models
3. Evaluating the effectiveness of existing safety filters
4. Exploring the underlying reasons for jailbreak success

### Key Designs

1. **MPUP Dataset Construction**:
    - It covers all inappropriate scenarios in OpenAI's usage policy, categorized into 4 broad categories with 26 subcategories: hate speech (500 prompts, 8 subcategories), harassment/harm (400 prompts, 8 subcategories), fraud (300 prompts, 6 subcategories), and sexual content (200 prompts, 4 subcategories).
    - Prompt template: `"<Image Generation Prompt>, with a sign that says, '<Visual Text Prompt>'"`
    - Three-step generation pipeline: (1) GPT-4 generates prompts for each category; (2) Modality filtering—removing prompts that are unsafe within a single modality; (3) Quality filtering—retaining only the top 10% strongest unsafe cases for each subcategory.

2. **Rhetorical Language Category Annotation**:
    - Each prompt is annotated with rhetorical categories: metaphor/simile, irony, personification/zoomorphism, allusion, and hyperbole.
    - Analysis is conducted to determine which rhetorical prompts are more likely to be successfully rendered as images with visual text by T2I models, thereby posing higher risks.

3. **Safety Classifier Evaluation Framework**:
    - **Text Filters**: Keyword blacklists (Midjourney and Leonardo.AI), BERT-score semantic similarity, LLM classifiers (Vicuna 7B, GPT-3.5), and the OpenAI Moderation API.
    - **Image Classifiers**: Q16, Multi-Headed SC (MHSC), and NSFW filters (based on ResNet-50).
    - A total of 7,000 prompts and 9,800 images are generated for evaluation using two techniques: "prompt modality removal" and "prompt modality modification".

4. **Evaluation Methodology**:
    - Evaluating Attack Success Rate (ASR) using GPT-4o with category-specific prompts and few-shot examples.
    - Cross-validation with human annotation shows that GPT-4o achieves 74.3% agreement with human labels (while Claude 3.5 Sonnet achieves only 53.9%).
    - Visual text quality is evaluated via OCR exact matching and substring matching.

### Loss & Training

As this paper is a safety assessment study, it does not involve model training. The main technical analyses include:
- Attributing the jailbreak capability to the visual text rendering capacity of the models.
- Analyzing that image-text pairs containing visual text in the training data serve as the source of this rendering capability.

## Key Experimental Results

### Main Results

| Model | Hate Speech ASR (%) | Harassment ASR (%) | Fraud ASR (%) | Sexual Content ASR (%) | Average ASR (%) |
|------|----------------|---------------|------------|------------|------------|
| DALL·E 3 | 63.3 | **85.4** | **72.4** | 52.4 | **68.2** |
| OpenDalle | **67.6** | 82.0 | 61.3 | **58.5** | 69.1 |
| Proteus | 58.6 | 76.5 | 62.7 | 46.5 | 62.9 |
| DeepFloyd | 57.8 | 66.5 | 49.7 | 61.5 | 59.1 |
| SDXL | 32.0 | 64.3 | 43.0 | 37.5 | 44.4 |
| SD | 33.0 | 46.8 | 42.3 | 30.5 | 38.2 |
| SLD | 7.6 | 11.0 | 5.0 | 3.0 | 7.4 |

### Ablation Study

| Safety Classifier | Hate Speech Acc (%) | Harassment Acc (%) | Fraud Acc (%) | Sexual Acc (%) | Description |
|------------|----------------|---------------|------------|------------|------|
| Random Filtering | 80.0 | 80.0 | 80.0 | 80.0 | Baseline |
| Keyword Blacklist | 79.5 | 79.4 | 78.9 | 79.1 | Comparable to random |
| BERT-score | 78.0 | 78.8 | 78.9 | 79.1 | Comparable to random |
| GPT-3.5 | 72.8 | 72.8 | 74.5 | 77.3| Lower than random |
| OpenAI Moderation API | 80.3 | 80.2 | 80.0 | 76.8 | Slightly better than random |
| Q16 Image Classifier | 65.0 | 60.9 | 61.0 | 62.5 | Barely better than random |

### Key Findings

- **All Tested Models are Affected**: The ASR of the 9 T2I models ranges from approximately 10% to 70%. Despite deploying multiple safety measures, DALL·E 3 remains one of the most vulnerable models (average ASR of 68.2%).
- **Visual Text Rendering Capacity is the Key Factor**: Substring OCR accuracy is highly correlated with ASR (DALL·E 3 substring OCR is ~50%, ASR is ~70%; SLD substring OCR is extremely low, and its ASR is only 7.4%).
- **Unsafe Even with Misspellings**: Although some text is not perfectly rendered, misspelled text can still be interpreted as unsafe by human observers.
- **Complete Failure of Existing Single-Modality Safety Filters**: The performance of text filters and image classifiers is comparable to or even worse than random filtering.
- Online T2I services (e.g., Midjourney, Leonardo.AI) show extremely low rejection rates (0–11.4%), yet their ASR remains as high as 24–40%.

## Highlights & Insights

- **Defining a New Safety Threat**: This work systematically investigates the pragmatic unsafety generated by cross-modal combinations of text and images for the first time, surpassing the traditional scope of single-modality safety.
- **Combining Linguistic Theory with AI Safety**: Introducing pragmatic concepts to the safety analysis of T2I models. The categorization of rhetorical devices adds depth to the overall analysis.
- **Comprehensive Benchmarking**: Evaluating 9 models (including 2 closed-source commercial models) utilizing 7,000 prompts and 12,600 queries, establishing a thorough evaluation framework.
- **Revealing the Root Cause**: The vulnerability is attributed to the models' visual text rendering capacity and the presence of visual text samples in the training data.

## Limitations & Future Work

- The agreement between the GPT-4o evaluator and human annotation is only 74.3%, indicating a need to improve the reliability of automated evaluation.
- The prompt format is restricted to the `"with a sign that says"` template; more natural or diverse prompts might yield different results.
- The exploration of defense mechanisms is preliminary, and effective multimodal safety classifiers have not yet been proposed.
- The potential of similar multimodal pragmatic jailbreaks on video generation models remains unexplored.
- The impact of cultural and linguistic differences on jailbreak effectiveness can be further investigated.

## Related Work & Insights

This work bridges several research domains: LLM jailbreaking (e.g., adversarial suffixes by Zou et al.), T2I model safety (e.g., SLD, NSFW detectors), and visual text rendering (e.g., GlyphControl, ByT5). The core insight is that as models become more capable (i.e., improved visual text rendering capacity), safety risks increase accordingly, presenting a concrete case study on the trade-off between AI capability and safety. This work also provides crucial insights for MLLM safety research—evaluating the safety of cross-modal combinations is an essential consideration for future model deployment.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Proposes the concept of multimodal pragmatic jailbreak for the first time, offering a unique perspective that bridges linguistic theory with AI safety.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Features 9 models, 7,000 prompts, 12,600 queries, and various safety filters, representing a large-scale and comprehensive evaluation.
- Writing Quality: ⭐⭐⭐⭐ The arguments are clear, the structure is complete, and the ethical statements regarding safety research are sufficient.
- Value: ⭐⭐⭐⭐⭐ Uncovers a major blind spot in current T2I safety mechanisms, carrying direct implications for industry practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MacPrompt: Maraconic-guided Jailbreak against Text-to-Image Models](../../AAAI2026/image_generation/macprompt_maraconic-guided_jailbreak_against_text-to-image_models.md)
- [\[CVPR 2025\] Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception](../../CVPR2025/image_generation/enhancing_image_aesthetics_with_dual-conditioned_diffusion_models_guided_by_mult.md)
- [\[CVPR 2025\] Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection](../../CVPR2025/image_generation/mitigating_memorization_in_text-to-image_diffusion_via_region-aware_prompt_augme.md)
- [\[NeurIPS 2025\] Pragmatic Heterogeneous Collaborative Perception via Generative Communication Mechanism](../../NeurIPS2025/image_generation/pragmatic_heterogeneous_collaborative_perception_via_generative_communication_me.md)
- [\[NeurIPS 2025\] Fast Data Attribution for Text-to-Image Models](../../NeurIPS2025/image_generation/fast_data_attribution_for_text-to-image_models.md)

</div>

<!-- RELATED:END -->
