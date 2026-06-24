---
title: >-
  [Paper Note] Evaluating Multimodal Language Models as Visual Assistants for Visually Impaired Users
description: >-
  [ACL 2025][Multimodal VLM][Visual Assistance] Through user surveys, this work identifies the core needs and challenges of visually impaired individuals regarding AI visual assistants. It designs an evaluation framework covering five user-centric tasks: image captioning, multilingual VQA, optical Braille recognition, video object recognition, and video QA. By systematically evaluating 12 MLLMs, it reveals significant deficiencies of current models in cultural understanding…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Visual Assistance"
  - "Visually Impaired Users"
  - "MLLM Evaluation"
  - "Braille Recognition"
  - "Cultural Sensitivity"
  - "Multilingual VQA"
  - "Video Understanding"
date: 2026-05-08
content_hash: 8d7c582a5617a1c3
---

# Evaluating Multimodal Language Models as Visual Assistants for Visually Impaired Users

**Conference**: ACL 2025  
**arXiv**: [2503.22610](https://arxiv.org/abs/2503.22610)  
**Code**: [https://github.com/MalvinaNikandrou/visual-assistant-eval](https://github.com/MalvinaNikandrou/visual-assistant-eval)  
**Authors**: Antonia Karamolegkou, Malvina Nikandrou, Georgios Pantazopoulos, Danae Sanchez Villegas, Phillip Rust, Ruchira Dhar, Daniel Hershcovich, Anders Søgaard  
**Institutions**: University of Copenhagen, Heriot-Watt University  
**Area**: Multimodal Large Language Models / Assistive Technologies  
**Keywords**: Visual Assistance, Visually Impaired Users, MLLM Evaluation, Braille Recognition, Cultural Sensitivity, Multilingual VQA, Video Understanding

## TL;DR

Through user surveys, this work identifies the core needs and challenges of visually impaired individuals regarding AI visual assistants. It designs an evaluation framework covering five user-centric tasks: image captioning, multilingual VQA, optical Braille recognition, video object recognition, and video QA. By systematically evaluating 12 MLLMs, it reveals significant deficiencies of current models in cultural understanding, multilingual support, Braille reading, assistive device recognition, and hallucination control.

## Background & Motivation

**Background**: MLLMs (e.g., GPT-4V, Qwen2-VL) have been integrated into assistive services for the visually impaired (such as Be My Eyes and Aira). However, existing evaluation benchmarks primarily target general visual reasoning (e.g., VQA, MMLU) and lack targeted assessments for accessibility scenarios.

**Key Challenge**:
   - Images or videos captured by visually impaired users are often of poor quality (blurry, poorly framed, or badly lit).
   - Users cannot verify the correctness of model outputs themselves, making hallucination issues particularly critical.
   - Multilingual and multicultural needs are not covered by existing evaluations.
   - Specialized needs, such as Braille recognition and assistive device identification, are almost entirely overlooked.

**Motivation**: Based on survey feedback from real visually impaired users, this study designs a user-centric evaluation framework to comprehensively reveal the capabilities and limitations of MLLMs in assisting visually impaired individuals.

## Method

### Overall Architecture: User-Driven Five-Task Evaluation

This work is divided into two main parts: a **user survey** (to understand needs) and a **systematic evaluation** (to quantify performance).

### User Survey Design

- Recruit 106 participants with varying degrees of visual impairment (via the Prolific platform).
- Two-stage survey: open-ended questions (usage scenarios and experienced challenges) + Likert scale ratings.
- **Key Findings**:
    - **87%** of users already use or are willing to use AI visual assistants.
    - Most common use cases: description, transcription, translation, and identification (e.g., identifying products while shopping, understanding chemical/mathematical diagrams, choosing clothes, and interpreting facial expressions).
    - Key challenges (TF-IDF analysis): inaccuracies/hallucinations, difficulties with handwriting recognition, insufficient multilingual support, and weak spatial understanding.

### Five Core Evaluation Tasks

**Task 1: Image Captioning**
- Dataset: VizWiz-Captions (500 images) + Cultural Extension (324 images, covering 60 cultures).
- Metric: RefCLIPScore.
- Evaluated Dimensions: General description capability vs. culturally sensitive description capability.

**Task 2: Image QA**
- Dataset: VizWiz VQA validation set + custom multilingual extension (34 languages).
- Translation Pipeline: Machine translation + manual quality check.
- Metric: VQA Accuracy.

**Task 3: Optical Braille Recognition**
- A brand-new task that contributes two novel datasets:
    - Sentence-level Braille-to-text: 100k sentences for training + NTREX-128/FLORES-200 for evaluation.
    - Paragraph-level cross-script QA: Adapted from SQuAD (130k training + 11.9k evaluation).
- Braille text is rendered as images with applied quality degradations (simulating photographic flaws common to visually impaired users).
- Metrics: chrF++ (for transcription), F1/EM (for QA).

**Task 4: Video Object Recognition**
- Dataset: ORBIT (1,036 video clips, 92 object categories, including assistive device categories).
- Distinction: General objects vs. assistive devices (e.g., Braille displays, white canes).
- Metric: LAVE protocol (LLM-as-a-Judge, 1-3 scale).

**Task 5: Video QA**
- Self-built dataset: 98 videos, 882 QA pairs.
- Three question types: Descriptive (attributes), spatial understanding (spatial relationships), and adversarial (non-existent objects).
- Adversarial questions test whether the model hallucinates answers.
- Metric: LAVE protocol.

### Evaluated Models

12 mainstream MLLMs, including Qwen2-VL, InternVL2.5, LLaVA-v1.6, MiniCPM-V-2.6, PaliGemma, Phi-3.5-Vision, etc.

## Key Experimental Results

### Image Captioning

| Model | Original VizWiz | Cultural Extension |
|------|-----------|----------|
| PaliGemma | **81.0** | 55.0 |
| MiniCPM-V-2.6 | 78.0 | 74.8 |
| Qwen2-VL | 75.9 | **76.9** |
| LLaVA-v1.6 | 72.3 | 52.2 |

- 5 out of 9 models suffer a significant performance drop (20-25 points) in cultural scenarios.
- Even the best-performing models miss key cultural details in about 1/3 of the descriptions.

### Image QA

| Model | English | Multilingual |
|------|------|--------|
| PaliGemma | **75.6** | 16.9 |
| MiniCPM-V-2.6 | 72.2 | 30.7 |
| Qwen2-VL | 61.9 | **44.9** |

- Models pre-trained on VizWiz data (PaliGemma, MiniCPM) perform best in English but show the sharpest performance drops in multilingual settings.
- Qwen2-VL shows the most stable multilingual performance (small fluctuations of 35.4-49.0 across languages).
- The gap between high-, medium-, and low-resource languages is **minimal**, indicating a general lack of reliable support even for high-resource languages.

### Braille Recognition

| Model | chrF++ (Zero-shot) |
|------|---------------|
| Qwen2-VL | **73.8** |
| Phi-3-Vision | 9.9 |
| All other models | < 9.1 |

- **Only Qwen2-VL demonstrates non-trivial Braille understanding capabilities**; other models are almost entirely unable to recognize Braille.
- LoRA fine-tuning on Llama-3.2-Vision achieves 88.2 chrF++, proving that learning Braille reading is feasible and saturates with 30k samples.

### Video Object Recognition

| Model | General Objects | Assistive Devices |
|------|---------|---------|
| Qwen2-VL | **69.8%** | 39.7% |
| MiniCPM-V-2.6 | 65.1% | **44.2%** |
| LLaVA-Video | 65.7% | 41.3% |

- The recognition rate for assistive devices (20-44%) is substantially lower than that for general objects (52-70%), highlighting a clear capability gap.

### Video QA

| Model | Descriptive | Spatial | Adversarial | Average |
|------|--------|------|--------|------|
| LLaVA-Video | **78.2** | 63.4 | 7.7 | 49.8 |
| MiniCPM-V-2.6 | 68.7 | 63.3 | **17.7** | **49.9** |
| VideoChat-Flash | 72.4 | **64.1** | 9.2 | 48.6 |

- Accuracy on adversarial questions (concerning non-existent objects) is extremely low (7-18%), showing that models tend to hallucinate answers rather than admit uncertainty.
- Even when explicitly prompted that they can answer with "uncertain," improvement remains limited.

## Highlights & Insights

1. **User-Driven Evaluation Design**: Designing evaluations based on the real needs of 106 visually impaired users ensures the practical significance of the tasks.
2. **First Braille Recognition Benchmark**: Proposing two brand-new Braille recognition tasks and datasets fills a major gap in MLLM evaluation.
3. **"Hallucination is the Greatest Enemy"**: For visually impaired users who cannot independently verify the output, model hallucination is far more dangerous than in average user scenarios.
4. **Cultural Blind Spots Exposed**: Even the best model fails to include cultural details in 1/3 of its descriptions—a real barrier for visually impaired travelers who need to understand multicultural environments.
5. **Feasibility of Braille Recognition**: Although current models almost entirely fail to read Braille, fine-tuning experiments demonstrate that they can learn it with a moderate amount of data (30k entries), showing a clear development path for next-generation models.

## Limitations & Future Work

1. **Missing Navigation Assistance Tasks**: Navigation is a core need of visually impaired users but is not covered in this work.
2. **Controlled Environment Assessment**: It does not fully capture the complexity of dynamic, real-world scenes (such as real-time response and in-motion video understanding).
3. **Model Coverage**: Only 12 models were evaluated, excluding leading closed-source models like GPT-4V/4o.
4. **Multilingual Translation Quality**: Due to the use of machine translation paired with manual checking, translation quality for low-resource languages might still be insufficient.
5. **Lack of User Interaction Evaluation**: The evaluation is based on offline benchmarks and does not assess multi-turn dialog or interactive guidance scenarios.

## Related Work & Insights

- **MLLM Evaluation Benchmarks**: MMLU, MME, VQAv2, etc., focus on general capabilities. A comprehensive evaluation by Lee et al. (2024) indicates that no single model leads across all areas.
- **Assistive Applications for the Visually Impaired**: VizWiz series (Gurari et al. 2018/2020), ORBIT dataset (Massiceti et al. 2021).
- **Culturally Sensitive Evaluation**: Karamolegkou et al. (2024) identified overlooked cultural context in VizWiz.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ The Braille recognition task is entirely new, and the user-survey-driven evaluation design is highly convincing.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ The five core tasks provide comprehensive coverage, featuring in-depth horizontal comparisons across 12 models.
- **Writing Quality**: ⭐⭐⭐⭐ The structure is clear, demonstrating coherent logic from identifying requirements to formulating evaluations.
- **Value**: ⭐⭐⭐⭐⭐ This work directly guides the development direction of next-generation visual assistance technologies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search](mcts_video_captioning_eval.md)
- [\[ACL 2025\] AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)
- [\[ICCV 2025\] Visual Intention Grounding for Egocentric Assistants](../../ICCV2025/multimodal_vlm/visual_intention_grounding_for_egocentric_assistants.md)
- [\[ACL 2025\] Aligning VLM Assistants with Personalized Situated Cognition](aligning_vlm_assistants_with_personalized_situated.md)
- [\[ACL 2025\] Unsolvable Problem Detection: Evaluating Trustworthiness of Large Multimodal Models](unsolvable_problem_detection.md)

</div>

<!-- RELATED:END -->
