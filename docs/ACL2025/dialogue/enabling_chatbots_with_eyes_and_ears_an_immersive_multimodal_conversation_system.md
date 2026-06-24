---
title: >-
  [Paper Note] Enabling Chatbots with Eyes and Ears: An Immersive Multimodal Conversation System
description: >-
  [ACL 2025][Dialogue Systems] This paper proposes an immersive multimodal conversation system that endows chatbots with "eyes and ears." It constructs the M3C dataset, a multi-session multi-party dialogue dataset integrating vision and audio, and designs a dialogue model consisting of a dialogue module and a multimodal memory retrieval module, enabling dynamic, long-term conversations where multiple speakers share audiovisual experiences.
tags:
  - "ACL 2025"
  - "Dialogue Systems"
date: 2026-05-08
content_hash: c93c3da29cae7328
---

# Enabling Chatbots with Eyes and Ears: An Immersive Multimodal Conversation System

**Conference**: ACL 2025  
**arXiv**: [2506.00421](https://arxiv.org/abs/2506.00421)  
**Code**: None  
**Area**: Dialogue Systems  

## TL;DR

This paper proposes an immersive multimodal conversation system that endows chatbots with "eyes and ears." It constructs the M3C dataset, a multi-session multi-party dialogue dataset integrating vision and audio, and designs a dialogue model consisting of a dialogue module and a multimodal memory retrieval module, enabling dynamic, long-term conversations where multiple speakers share audiovisual experiences.

## Background & Motivation

1. **Existing multimodal dialogues prioritize "eyes" over "ears"**: Current research mainly focuses on image-related dialogues (visual dialogue, image instruction tuning, etc.), giving chatbots "eyes," whereas the auditory aspect ("ears") is severely lacking, with no solution simultaneously integrating vision and hearing.

2. **Static interactions limit dialogue naturalness**: In the existing paradigm, chatbots answer questions after receiving a shared image. This represents a static interaction mode of a "discussing modality" rather than a "naturally integrated modality," failing to capture the dynamic, real-time nature of real human communication.

3. **Insufficient exploration of multi-party + multi-session scenarios**: Although multimodality has been explored in multi-party and multi-session dialogues, it is constrained by specific task limitations and remains difficult to integrate seamlessly into dynamic, natural conversations.

4. **Lack of datasets with shared spatio-temporal experiences**: In existing datasets (e.g., PhotoChat, DialogCC), speakers do not experience the audiovisual inputs in the same space and time, which does not reflect real-world multi-person situations.

5. **Importance of long-term memory mechanisms**: In real conversations, people recall previous shared experiences. However, existing models lack effective multimodal memory storage and retrieval mechanisms to support coherent dialogues across multiple sessions.

6. **Challenges in autonomous multi-party interaction**: To achieve autonomous multi-agent dialogue without human intervention, the model needs to determine when it is its turn to speak, which remains an insufficiently solved key challenge in multi-party conversations.

## Method

### Dataset: M3C (Multimodal Multi-Session Multi-Party Conversation)

**Data Scale and Structure**:
- 54K dialogue episodes (34K train/8K validation/12K test), totaling 2.5M dialogue turns.
- Each episode contains 4 speakers across 3 consecutive sessions.
- In each session, the main speaker interacts with 2 different partners.
- Each session contains 2 multimodal inputs (visual or auditory).
- All speakers share the same audiovisual experience in a unified spatio-temporal environment.

**Data Construction Pipeline**:
1. **Modality Structuring**: Using COCO images (24K) and AudioCaps/Clotho audio (73K) as seeds, image tags are optimized via GPT-4o mini.
2. **Scene Preparation**: Generating speaker profiles, session partners, modal inputs, and time intervals; grouping similar modalities based on K-means clustering ($K=30$) by location tags.
3. **Dialogue and Memory Generation**: Generating dialogues session-by-session, creating memory summaries from the main speaker's perspective, and connecting related elements via memory links.
4. **Quality Filtering**: Excluding episodes that fail spatio-temporal consistency checks via machine-verified questions.

### Model Architecture

**Base Model**: Qwen2-VL-2B-Instruct, with audio understanding capabilities extended via CLAP + a linear adapter.

**Dialogue Module**:
- Responsible for three tasks: dialogue generation, memory generation, and memory linking.
- Generates responses based on dialogue history and multimodal inputs during an active session.
- Constructs memory units integrated with multimodal perception after a session ends.
- Explicitly associates new memories with semantically or perceptually related past memories via structured memory links.

**Retrieval Module**:
- Retrieves relevant memories from the multimodal memory bank based on the current dialogue context.
- Jointly embeds the entire session (dialogue + perceived modality) into a shared representation space.
- Measures memory relevance using cosine similarity: $\operatorname{sim}(c, m_i) = \cos(E_c(c), E_m(m_i))$.
- Selects the Top-1 most relevant memory to augment the dialogue.

**Training Strategy**: Two-stage fine-tuning—first fine-tuning on vision-language tasks (with audio as text captions), then integrating the linear adapter to process raw audio. The model supports model-to-model dialogues and autonomously manages turn-taking.

## Key Experimental Results

### Human and Automatic Evaluation (Table 2)

| Evaluation Dimension | Human Score | Automatic Score (o3-mini) |
|---------|---------|-----------------|
| **Dataset Quality** | | |
| Coherence & Consistency | 4.81 | 4.99 |
| Memorability | 4.63 | 4.99 |
| Modality Alignment | 4.21 | 4.26 |
| Modality Engagement | 4.36 | 4.57 |
| Dataset Overall | 4.50 | 4.70 |
| **Model Performance** | | |
| Naturalness | 4.34 | 4.68 |
| Immersiveness | 4.14 | 4.56 |
| Memorability | 4.35 | 4.46 |
| Model Overall | 4.28 | 4.57 |

### Retrieval Module Performance (Table 4)

| Model | Image R@1 | Image MRR | Audio R@1 | Audio MRR |
|------|---------|---------|---------|---------|
| Qwen2-VL-2B | 66.77 | 77.56 | - | - |
| LLaMA-3.2-11B-Vision | 72.41 | 78.90 | - | - |
| Qwen2-Audio-7B | - | - | 69.94 | 80.72 |
| **Ours** | **92.99** | **95.06** | **92.83** | **94.78** |

**Cross-Dataset Comparison (Table 3)**: When GPT-4o-mini and Claude-3.5-Sonnet evaluated the "immersive naturalness" of M3C versus other datasets, M3C achieved selection rates of 81% and 99%, respectively.

**Multi-Party Dialogue Performance (Table 5)**: Next-speaker prediction accuracy—Ours 85.2% vs Qwen2-VL baseline 10.3%.

### Ablation Study

- **Audio Captions vs. Raw Audio**: The model equipped with the audio adapter aligns directly with auditory experiences, whereas caption-based models rely excessively on prompt text content.
- **Multimodal Memory**: With the retriever, the model can reference specific details from prior sessions (e.g., "the shells collected last time"), whereas without the retriever, it generates generic responses.
- **Beyond Three Sessions**: Although trained only on three-session data, the model supports long-term dialogues across more sessions thanks to its independent memory mechanism.

## Highlights & Insights

- **First Audiovisual Multi-Party Multi-Session Dialogue Dataset**: M3C is the first open-domain dialogue dataset where all speakers synchronously experience images and audio within a shared space and time.
- **Multimodal Memory Retrieval**: Coherent cross-session dialogue is achieved through structured memory links and cross-modal retrieval, where associations are established at storage time (rather than search time).
- **Autonomous Multi-Party Dialogue**: The model autonomously determines turn-taking and when modal inputs should appear, enabling multi-agent conversation without human intervention.
- **Substantial Lead in Retrieval Performance**: Achieving over 92% R@1 in both image and audio retrieval, significantly outperforming comparison models.

## Limitations & Future Work

- The annotations for the audio dataset were not optimized to the same level as the image annotations, which may affect the immersive quality of the audio portion.
- The base model is constrained by a 2B-parameter VLM, which lacks the native vision-audio-language joint understanding capabilities of larger models.
- The dataset is generated by GPT-4o mini rather than human conversations, potentially introducing machine generation bias.
- The model scale is relatively small (2B), and its performance and generalizability on larger models have not yet been verified.

## Related Work & Insights

| Comparison Direction | Advantages of Ours |
|---------|---------|
| **PhotoChat / DialogCC / Stark** | These datasets support only the visual modality and are mostly single-party/single-session; M3C covers both images and audio, supports multi-party multi-session dialogues, and demonstrates dominating immersiveness with an 81-99% selection rate under GPT-4o-mini evaluation. |
| **Audio Dialogues / MELD** | Audio Dialogues only supports audio QA tasks; although MELD includes audio and video, it consists of short dialogues oriented toward sentiment analysis. M3C offers open-domain, long-term, multi-turn dialogues with coordinated audiovisual design and shared experience configurations. |
| **MiSC (Jang et al., 2024)** | MiSC supports multi-session multi-party dialogues but lacks modal inputs; M3C extends this to a multi-party, multi-partner per session setting with audiovisual modalities, dramatically increasing interaction complexity and realism. |

## Rating

- ⭐⭐⭐⭐ **Novelty**: The setting of audiovisual multi-party multi-session dialogue represents a clear innovation, and the design of multimodal memory retrieval is reasonable.
- ⭐⭐⭐⭐ **Experimental Thoroughness**: A multi-dimensional evaluation combining human, automatic, and cross-dataset metrics is presented, including ablation and quantitative analyses.
- ⭐⭐⭐⭐ **Value**: Provides a dataset and modeling paradigm for building more natural multimodal dialogue systems.
- ⭐⭐⭐⭐ **Writing Quality**: The structure is clear, cases are rich, and the dataset comparison tables are highly informative.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Towards Proactive Information Probing: Customer Service Chatbots Harvesting Value from Conversation](../../ACL2026/dialogue/towards_proactive_information_probing_customer_service_chatbots_harvesting_value.md)
- [\[ACL 2025\] Exploring Persona Sentiment Sensitivity in Personalized Dialogue Generation](persona_sentiment_dialogue.md)
- [\[ACL 2025\] EnSToM: Enhancing Dialogue Systems with Entropy-Scaled Steering Vectors for Topic Maintenance](enstom_enhancing_dialogue_systems_with_entropy-scaled_steering_vectors_for_topic.md)
- [\[AAAI 2026\] Chatsparent: An Interactive System for Detecting and Mitigating Cognitive Fatigue in LLMs](../../AAAI2026/dialogue/chatsparent_an_interactive_system_for_detecting_and_mitigating_cognitive_fatigue.md)
- [\[ACL 2025\] Know Your Mistakes: Towards Preventing Overreliance on Task-Oriented Conversational AI Through Accountability Modeling](know_your_mistakes_towards_preventing_overreliance_on_task-oriented_conversation.md)

</div>

<!-- RELATED:END -->
