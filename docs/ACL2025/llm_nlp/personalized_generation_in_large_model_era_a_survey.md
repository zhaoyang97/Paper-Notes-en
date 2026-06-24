---
title: >-
  [Paper Note] Personalized Generation In Large Model Era: A Survey
description: >-
  [ACL 2025 (Findings)][LLM (Other)][Personalized Generation] The first systematic survey on cross-modal Personalized Generation (PGen), presenting a unified user-centric perspective to integrate research from NLP, CV, and IR communities under a single framework, covering six modalities: text, image, video, audio, 3D, and cross-modality.
tags:
  - "ACL 2025 (Findings)"
  - "LLM (Other)"
  - "Personalized Generation"
  - "Survey"
  - "User Modeling"
  - "LLM"
  - "Diffusion Model"
  - "Multimodal"
date: 2026-05-08
content_hash: ca8880eb6b1d1a7e
---

# Personalized Generation In Large Model Era: A Survey

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2503.02614](https://arxiv.org/abs/2503.02614)  
**Code**: Unreleased  
**Area**: Other  
**Keywords**: Personalized Generation, Survey, User Modeling, LLM, Diffusion Model, Multimodal  

## TL;DR

The first systematic survey on cross-modal Personalized Generation (PGen), presenting a unified user-centric perspective to integrate research from NLP, CV, and IR communities under a single framework, covering six modalities: text, image, video, audio, 3D, and cross-modality.

## Background & Motivation

- **Core Observation**: In the era of large models, content generation is shifting from one-size-fits-all generation to Personalized Generation (PGen). However, research in different communities (NLP, CV, IR) remains siloed, lacking a unified perspective.
- **Limitations of Prior Surveys**: Existing surveys are either model-centric (e.g., focusing specifically on personalization of LLMs/diffusion models) or task-centric (e.g., dialogue generation, role-playing), lacking a cross-community panoramic survey.
- **Goal of Ours**: Proposes the first modality-agnostic unified framework to systematically organize research across the boundaries of NLP, CV, and IR communities.

## Method

### Overall Architecture—Unified User-Centric Perspective

PGen relies on two types of user inputs: (1) **Personalized Context**: historical data containing user preferences; (2) **Multimodal Instructions**: signals such as text prompts and voice commands that explicitly express content requirements. Generative models learn preferences from the personalized context to generate customized content according to instructions.

### Key Designs—Five Personalized Context Dimensions

| Context Type | Description | Common Tasks |
|-----------|------|---------|
| **User Profile** | Age, gender, occupation, location, etc. | Dialogue systems, e-commerce product images |
| **User Document** | Reviews, emails, social media posts | Writing assistant, personalized recommendation |
| **User Behavior** | Interactions such as search, click, purchase, etc. | Recommendation systems, information retrieval |
| **Personal Face/Body** | Facial structure, body shape, expression, movement | Portrait generation, virtual try-on |
| **Personalized Subject** | User-specific concepts such as pets, personal belongings, etc. | Subject-driven generation |

### Three Core Objectives

1. **High Quality**: Coherence, relevance, and aesthetics of generated content
2. **Instruction Alignment**: Accurately following the user's multimodal instructions
3. **Personalization**: Consistency with user preferences and personalized context

### PGen Workflow

**User Modeling Stage**:
- Representation Learning: Encoding into dense embeddings or discrete text representations
- Prompt Engineering: Designing task-specific prompts to organize user info
- RAG: Filtering irrelevant information and integrating external relevant data

**Generation Modeling Stage**:
- Step 1 - Foundation Model Selection: LLM / MLLM / Diffusion Model
- Step 2 - Guidance Mechanism: Instruction guidance (ICL, instruction tuning) + Structural guidance (adapter, cross-attention)
- Step 3 - Optimization Strategy: Tuning-free (模型融合, multi-turn interaction) / Supervised Fine-Tuning (Full or PEFT) / Preference Optimization (RLHF, DPO)

### Multi-level Taxonomy

The survey is organized hierarchically as **Modality -> Personalized Context -> Task**, covering 200+ papers:

| Modality | Representative Tasks | Representative Methods |
|------|-----------|-----------|
| **Text** | Recommendation, writing assistant, dialogue, role-playing | LLM-Rec, REST-PG, PAED, CharacterLLM |
| **Image** | Subject-driven T2I, face generation, virtual try-on | DreamBooth, PhotoMaker, IDM-VTON |
| **Video** | Subject-driven T2V, Talking Head, dance generation | AnimateDiff, EMO, AnimateAnyone |
| **3D** | Image-to-3D, 3D face/body | MVDream, DreamBooth3D, DreamWaltz |
| **Audio** | Music generation, text-to-speech | UMP, DiffAVA |
| **Cross-modal** | Personalized captions/comments, dialogue | MyVLM, Yo'LLaVA |

## Experiments

This is a survey paper and does not contain original experiments. The primary contribution lies in the systematic review and taxonomy of the literature.

### Dataset Summary

| Modality | Representative Datasets |
|------|-----------|
| Text | LaMP, LongLaMP, Amazon Reviews, MovieLens |
| Image | DreamBooth dataset, VITON-HD, DeepFashion |
| Video | TikTok Dance, HDTF (Talking Head) |
| 3D | ShapeNet, THuman2.0 |
| Audio | LibriSpeech, MusicNet |

### Evaluation Metrics Summary

| Objective | Metrics |
|------|------|
| Quality | FID, IS, CLIP Score, BLEU, Perplexity |
| Instruction Alignment | CLIP-T, BERT-Score |
| Personalization | CLIP-I, DINO Score, Face-Sim, User Study |

### Key Findings

- Personalization research in the text modality is the most mature, followed by the image modality, while video/3D/audio modalities remain in early stages.
- User behavior and user documents are the most commonly used personalized contexts in the text modality, whereas the CV field relies more on personal faces/bodies and personalized subjects.
- PEFT (especially LoRA) has emerged as the mainstream strategy for cross-modal personalization fine-tuning.

## Highlights & Insights

- Incorporates personalization generation research from the NLP, CV, and IR communities into a unified framework for the first time, filling an important gap in literature reviews.
- The proposed modality-agnostic workflow (user modeling -> generation modeling) provides a common language for researchers across different communities.
- The multi-level taxonomy is clear and extensible, making it easy to track research progress in specific subfields.
- The future work section discusses five open challenges, including scalability, preference evolution, privacy, and fairness.

## Limitations & Future Work

- As a survey paper, the depth of discussion for each subfield is limited, and the literature coverage of some emerging directions (such as 3D personalization) may not be comprehensive.
- Although the unified framework provides high-level abstractions, technological discrepancies between different modalities remain significant, which somewhat limits the framework's practical guidance.
- Performance data of alternative methods are not compared, lacking quantitative comparative analysis of different approaches.
- Due to the literature search cutoff date, some of the most recent works might have been missed.

## Related Work & Insights

- **Model-Centric Surveys**: Zhang et al. (2024) focus on LLM personalization; Zhang et al. (2024) discuss diffusion model personalization.
- **Task-Centric Surveys**: Chen et al. (2024) discuss personalized dialogue; Tseng et al. (2024) discuss role-playing.
- **Foundation Model Surveys**: Wu et al. (2024) review multimodal large language models.
- **Recommendation System Surveys**: Ayemowa et al. (2024) discuss generative recommendation.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| **Novelty** | 4 |
| Technical Depth | 3 |
| **Experimental Thoroughness** | N/A (Survey) |
| **Writing Quality** | 4 |
| **Total Score** | **3.7** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Pragmatics in the Era of Large Language Models: A Survey on Datasets, Evaluation, Opportunities and Challenges](pragmatics_survey.md)
- [\[ACL 2025\] LLMs + Persona-Plug = Personalized LLMs](llms_persona-plug_personalized_llms.md)
- [\[ACL 2025\] From Selection to Generation: A Survey of LLM-based Active Learning](from_selection_to_generation_a_survey.md)
- [\[ACL 2025\] Large Language Models in Bioinformatics: A Survey](large_language_models_in_bioinformatics_a_survey.md)
- [\[ACL 2025\] A Survey on Efficient Large Language Model Training: From Data-centric Perspectives](a_survey_on_efficient_large_language.md)

</div>

<!-- RELATED:END -->
