---
title: >-
  [Paper Note] SingaKids: A Multilingual Multimodal Dialogic Tutor for Language Learning
description: >-
  [ACL 2025][Multimodal VLM][Intelligent Tutoring Systems] This paper proposes SingaKids, a multilingual multimodal dialogic language learning tutoring system tailored for primary school students. Through a picture description task, it integrates dense image captioning, multilingual dialogue, speech understanding, and child-friendly speech generation, supporting interactive learning across four languages: English, Chinese, Malay, and Tamil.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Intelligent Tutoring Systems"
  - "Multilingual Dialogue"
  - "Picture Description"
  - "Language Learning"
  - "Scaffolding"
date: 2026-05-08
content_hash: 9154f46e4a39e2df
---

# SingaKids: A Multilingual Multimodal Dialogic Tutor for Language Learning

**Conference**: ACL 2025  
**arXiv**: [2506.02412](https://arxiv.org/abs/2506.02412)  
**Code**: None  
**Area**: Multimodal/Educational AI  
**Keywords**: Intelligent Tutoring Systems, Multilingual Dialogue, Picture Description, Language Learning, Scaffolding

## TL;DR

This paper proposes SingaKids, a multilingual multimodal dialogic language learning tutoring system tailored for primary school students. Through a picture description task, it integrates dense image captioning, multilingual dialogue, speech understanding, and child-friendly speech generation, supporting interactive learning across four languages: English, Chinese, Malay, and Tamil.

## Background & Motivation

Generative AI shows immense potential for personalized learning in education. However, in language learning scenarios, applications aimed at children still face multiple challenges:

**Inconsistent Cross-lingual Performance**: Most LLMs perform exceptionally in high-resource languages like English, but their performance drops significantly in low-resource languages such as Malay and Tamil. This poses a major barrier to educational applications in Singapore's multilingual environment.

**Lack of Child-Friendly Design**: Existing systems are mostly designed for adults and lack considerations for children's cognitive load, attention span, and developmental appropriateness. Children require simplified instructions, engaging dialogic patterns, and age-appropriate scaffolding support.

**Disconnect Between Dialogic Pedagogy and Practice**: Traditional intelligent tutoring systems heavily rely on rule-based systems or massive human-annotated datasets. Although the new generation of LLM-driven conversational tutors reduces data requirements, how to effectively integrate pedagogical and learning science principles remains an open question.

SingaKids is proposed against this backdrop to establish a multilingual interactive learning environment tailored for Singaporean primary school students through a picture description task.

## Method

### Overall Architecture

The system architecture comprises three key modules, forming a complete educational dialogue pipeline:

1. **Multimodal Understanding Module**:
    - Scene Understanding: Extracting keywords, objects, and events from images.
    - Multilingual Automatic Speech Recognition (ASR): Transcribing students' spoken responses into text.
    - Speech Evaluation: Assessing students' speaking proficiency.

2. **Multilingual LLM Core Module**:
    - Multilingual Semantic Understanding: Interpreting student responses in context.
    - Language Evaluation: Assessing the linguistic accuracy and completeness of descriptions.
    - Scaffolding Guidance: Determining the appropriate level of support.
    - Pedagogical Anchoring: Establishing high-level instructional goals (e.g., vocabulary comprehension or sentence construction).

3. **Output Module**:
    - Multilingual Text-to-Speech (TTS): Converting text into natural and engaging speech.
    - Keyword Highlighting: Emphasizing important keywords or pronunciation errors.

### Key Designs

#### 1. Dense Image Captioning

- **Function**: Generating rich descriptions for each key event in the image.
- **Mechanism**: Adopting a two-stage approach: event bounding box proposal followed by caption generation.
    - The first phase utilizes person/object detection (Liu et al., 2024a), human segmentation (Kirillov et al., 2023), and depth estimation (Bhat et al., 2023) for probabilistic reasoning.
    - The second phase applies chain-of-thought prompting on InternVL2.5 to integrate global contextual understanding into individual event captions.
- **Design Motivation**: State-of-the-art multimodal LLMs (especially smaller ones) perform suboptimally on dense image content, tending to generate generic descriptions and being prone to hallucination.
- **Effect**: Achieved 75% sentence-level accuracy on the image test set.

#### 2. Optimizing Multilingual ASR

- **Function**: Improving speech recognition capabilities for Malay and Tamil, particularly for children's speech.
- **Mechanism**: Fine-tuning Whisper-large-V3 as the base model using large-scale collected local data.
    - Tamil: 2,800 hours, Malay: 1,000 hours, sourced from over 1,000 native speakers across different ages and linguistic backgrounds.
- **Design Motivation**: Preliminary analysis revealed a significant performance gap in low-resource languages and children's speech.

| Language | Test Set | WER Before FT | WER After FT |
|------|--------|-----------|-----------|
| Malay | Conversational Speech | 40.5% | 28.4% |
| Malay | Children's Speech | 20.3% | 5.1% |
| Tamil | Bloom Speech | 10.3% | 7.1% |
| Tamil | Children's Speech | 13.7% | 7.9% |

#### 3. Optimizing Dialogic LLM

**Multilingual Capability Enhancement**:

- Base Model: Qwen1.5-4B (balancing performance and efficiency).
- Two-stage optimization pipeline:
    - Stage 1: Continual pre-training on 14B tokens of a quadrilingual mixed dataset, applying balanced sampling rates to boost Malay and Tamil performance.
    - Stage 2: Enhancing multilingual instruction-following capabilities through multi-task learning and cross-lingual alignment, including a multilingual role-playing corpus.

**Scaffolding Guidance Enhancement**:

- Grounded in dialogic pedagogy theory (Alexander, 2006), where teachers foster the exchange of ideas through probing, cueing, elaborating, or reviewing.
- Synthesizing dialogue samples with GPT-4 to train the smaller model to deliver scaffolded interactions based on student responses.
- Building student persona classification (based on the Big Five framework) integrating both cognitive and non-cognitive aspects.
- Side benefit: Scaffolding training improves system robustness against inappropriate language and out-of-domain inputs.

#### 4. Optimizing Multilingual TTS

- Framework: VITS (non-autoregressive, balancing speech quality and efficiency).
- Data: Malay (22h adult + 9h child), Tamil (63h adult + 1.5h child).
- Supporting multi-speaker generation utilizing one-hot speaker embeddings.

### Loss & Training

The system employs a module-by-module optimization strategy, with components trained independently and integrated afterward:
- ASR: Fine-tuned based on Whisper-large-V3.
- LLM: Continual pre-training + Instruction tuning + Scaffolding enhancement.
- TTS: Multi-speaker VITS training.
- All experiments were conducted on Nvidia A100 40/80GB GPUs.

## Key Experimental Results

### TTS Evaluation (Main Results)

| Metric | Malay (Adult) | Malay (Child) | Tamil (Adult) | Tamil (Child) |
|------|--------------|--------------|----------------|----------------|
| MOS (Subjective) | >3.50 | >3.50 | >3.50 | >3.50 |
| CER (Speech Intelligibility) | <10% | <10% | <10% | <10% |

Evaluating with 20 native listeners, the speech intelligibility surpassed 90%.

### User Study & Scaffolding Analysis (Ablation Study)

| Scaffolding Type | High-Performing Students | Low-Performing Students |
|-----------|----------|----------|
| Feeding back | 69% | 43% |
| Explanation | 21% | 9% |
| Hints | 5% | 12% |
| Social-emotional | 17% | 31% |

An empirical study on 35 Grade 1 and 2 students (IRB-2024-218) demonstrates that the system adaptively tailors its pedagogical strategies based on student performance.

### Key Findings

1. **Adaptive Scaffolding is Effective**: High-performing students receive more feedback and explanations, guiding them toward deeper understanding, whereas low-performing students receive more hints and socio-emotional support.
2. **Significant ASR Improvements on Children's Speech**: Malay children's speech WER plummeted from 20.3% to 5.1%.
3. **Substantial Boost in Multilingual Abilities**: Through continual pre-training and cross-lingual alignment, both translation and instruction-following abilities improved.
4. **Scaffolding Training Enhances Robustness**: When facing inappropriate or out-of-domain inputs, the system successfully steers the students back to the picture description task.

## Highlights & Insights

- **Comprehensive Systems Engineering Approach**: Instead of an isolated model-level innovation, the work organically integrates four modules—ASR, LLM, TTS, and image understanding—into a fully functional educational system.
- **Blending Scaffolding Theory with AI**: Systematically incorporating dialogic pedagogy from learning sciences into LLM training, achieving adaptive teaching using personalized student personas.
- **Focus on Low-Resource Languages**: Tailoring optimization specifically to Malay and Tamil, demonstrating a strong commitment to linguistic equity.
- **Real-World Validation**: Conducting user studies with actual primary school students rather than relying solely on automated evaluation metrics.

## Limitations & Future Work

1. **Hallucination Issues**: LLMs still carry the risk of hallucinations and biases, which might lead to communication errors in educational settings.
2. **Noisy Environments**: Classroom noise and children's typical speech patterns elevate ASR errors, necessitating noise-robust speech recognition and speaker diarization.
3. **Student Disengagement**: Some students withdraw when encountering persistent challenges; the system needs better mechanisms to trigger modeling strategies.
4. **Guiding Lower-Grade Students**: The system cannot yet fully replace adult/parent guidance for younger children.
5. **Visual Complexity**: When an image contains too many objects, children are easily distracted, requiring auxiliary visual highlighting.

## Related Work & Insights

- Evolution of Intelligent Tutoring Systems (ITS): Moving from rule-based engines to generative LLM-driven dialogic tutors.
- Application prospects of multimodal LLMs in education.
- Dialogic pedagogy theory (Alexander, 2006) provides a solid theoretical foundation for AI educational system designs.
- Personalized learning pathways and adaptive feedback remain core directions for educational AI.

## Rating

- **Novelty**: ⭐⭐⭐ (System integration innovation, limited single-module innovation)
- **Experimental Thoroughness**: ⭐⭐⭐ (Quantitative evaluation carried out for each module, but user study size is relatively small)
- **Writing Quality**: ⭐⭐⭐⭐ (Well-structured, comprehensive system exposition)
- **Value**: ⭐⭐⭐⭐ (Possesses highly practical application value for real-world educational scenarios)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] COSMMIC: Comment-Sensitive Multimodal Multilingual Indian Corpus](cosmmic_commentsensitive_multimodal_multilingual_indian_corpus.md)
- [\[ACL 2025\] mOSCAR: A Large-scale Multilingual and Multimodal Document-level Corpus](moscar_a_large-scale_multilingual_and_multimodal_document-level_corpus.md)
- [\[ACL 2025\] Centurio: On Drivers of Multilingual Ability of Large Vision-Language Model](centurio_multilingual_vlm.md)
- [\[ACL 2025\] MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering](mtvqa_benchmarking_multilingual_text-centric_visual_question_answering.md)
- [\[ACL 2025\] DALR: Dual-level Alignment Learning for Multimodal Sentence Representation Learning](dalr_dual-level_alignment_learning_for_multimodal_sentence_representation_learni.md)

</div>

<!-- RELATED:END -->
