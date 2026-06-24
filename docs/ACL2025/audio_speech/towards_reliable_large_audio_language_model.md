---
title: >-
  [Paper Note] Towards Reliable Large Audio Language Model
description: >-
  [ACL 2025][Audio & Speech][Large Audio Language Model] This paper presents the first systematic study on the reliability of Large Audio Language Models (LALMs), proposing training-free methods (IDK/MCoT/Task Agent) and a training-based method (LoRA SFT on model-specific IDK datasets). It also designs the Reliability Gain Index (RGI) metric to evaluate improvements in reliability, revealing that "knowing when to say I don't know" is a cross-modal transferable meta-capability.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Large Audio Language Model"
  - "Reliability"
  - "IDK"
  - "Cross-modal Transfer"
  - "Abstention Ability"
date: 2026-05-08
content_hash: eed36b1b3eaa8b2f
---

# Towards Reliable Large Audio Language Model

**Conference**: ACL 2025  
**arXiv**: [2505.19294](https://arxiv.org/abs/2505.19294)  
**Code**: None  
**Area**: Audio & Speech  
**Keywords**: Large Audio Language Model, Reliability, IDK, Cross-modal Transfer, Abstention Ability

## TL;DR

This paper presents the first systematic study on the reliability of Large Audio Language Models (LALMs), proposing training-free methods (IDK/MCoT/Task Agent) and a training-based method (LoRA SFT on model-specific IDK datasets). It also designs the Reliability Gain Index (RGI) metric to evaluate improvements in reliability, revealing that "knowing when to say I don't know" is a cross-modal transferable meta-capability.

## Background & Motivation

**Background**: Large Audio Language Models (LALMs) such as Qwen2-Audio have made significant progress in understanding and reasoning over multimodal audio including speech, music, and ambient sound, enabling them to handle various tasks such as ASR, audio captioning, and emotion recognition.

**Limitations of Prior Work**: Despite their excellent performance, LALMs lack the ability to recognize their own knowledge boundaries. When faced with questions they cannot answer, they do not actively refuse to answer, but instead provide incorrect or overconfident responses. This is particularly dangerous in high-risk scenarios like medical applications and autonomous driving.

**Key Challenge**: While many reliability enhancement works exist in the text LLM domain (such as IDK datasets, Prudence Score, etc.), research on the reliability of audio language models is almost non-existent. The unique nature of audio data—where speech, music, and ambient sounds differ vastly in structure and content—makes direct transfer of text-domain methods highly challenging.

**Goal**: (1) How to systematically enhance the abstention capability of LALMs? (2) How to accurately evaluate the effectiveness of different reliability enhancement methods? (3) Can reliability awareness transfer across different audio modalities?

**Key Insight**: The authors observe that existing evaluation metrics (Accuracy, Truthfulness, Reliability) fail to differentiate the trade-offs of different methods between "conservatism" and "humility". Therefore, they propose a new metric, RGI, to measure reliability improvements from the perspective of relative gain.

**Core Idea**: Introduce the reliability enhancement paradigm of text LLMs to the audio modality, leveraging a dual-pathway approach (training-free + training-based) to improve LALMs' abstention capability, and demonstrating through the RGI metric that reliability awareness is a cross-modally transferable "meta-capability."

## Method

### Overall Architecture

The input consists of audio and a question. The model needs to determine whether it has the capacity to answer correctly: if it can, it outputs the answer; if not, it outputs "I don't know". The authors explore two major categories of methods: (1) training-free inference-time enhancements, and (2) supervised fine-tuning based on model-specific IDK datasets.

### Key Designs

1. **IDK Prompting (Training-free)**:

    - **Function**: Appends prompts after the input question to encourage the model to actively say "I don't know" when uncertain.
    - **Mechanism**: Leverages the model's instruction-following capability to activate its expression of uncertainty by appending prompt texts.
    - **Design Motivation**: Served as the simplest baseline to verify if the model possesses latent abstention capabilities at zero cost.

2. **MCoT Prompting (Training-free)**:

    - **Function**: Multimodal Chain-of-Thought reasoning, requiring the model to analyze step-by-step before answering.
    - **Mechanism**: Borrowing the Chain-of-Thought concept, it decomposes complex questions into smaller sequential steps, allowing the model to better assess its confidence during reasoning.
    - **Design Motivation**: Step-by-step reasoning exposes the model's uncertainties in intermediate steps, leading to more reliable final judgments.

3. **Task Agent (Training-free)**:

    - **Function**: A three-step reasoning agent—first identifying the audio type (speech/sound/music), then generating corresponding content (ASR/AAC/MC), and finally producing the answer based on the original audio and generated content.
    - **Mechanism**: Explicating implicit reasoning through intermediate tool usage. It performs ASR for speech to obtain text content and generates captions for ambient sound/music, allowing the model to make judgments based on richer information.
    - **Design Motivation**: Audio data varies greatly among speech, music, and ambient sound, making unified processing suboptimal. Calling different tools based on the audio type provides more targeted context.

4. **LoRA Fine-Tuning based on IDK Dataset (Training-based)**:

    - **Function**: Construction of a model-specific IDK dataset followed by supervised fine-tuning with LoRA.
    - **Mechanism**: Perform inference sampling $N$ times for each question; if the model answers correctly all $N$ times, keep the original label, otherwise mark the answer as "IDK". Use the K@N threshold to control stringency (5@5 is used in this work), then conduct 1 epoch of LoRA SFT on this dataset.
    - **Design Motivation**: Different models have different knowledge boundaries, requiring the construction of model-specific IDK datasets. Training the model to explicitly learn when to abstain is more direct and effective than prompting.
    - **Comparison with Prior Work**: First to introduce the IDK training paradigm of text LLMs into the multimodal audio domain.

### Evaluation Metric Design

One of the core contributions is proposing the Reliability Gain Index (RGI):

- **Accuracy** = $N_c / N$, proportion of correct answers
- **Truthfulness** = $1 - N_w / N$, proportion of non-incorrect answers
- **Reliability** = $\text{Rej} \times \text{Acc} + (1-\text{Rej}) \times \text{Tru}$, comprehensive reliability
- **Relative increase in conservatism** $\Delta\text{Con} = (N_c - N_{cc}) / N_c$, the proportion of previously correct answers that are now rejected (lower is better)
- **Relative increase in humility** $\Delta\text{Hum} = (N_w - N_{ww}) / N_w$, the proportion of previously incorrect answers that are now correctly rejected (higher is better)
- $\text{RGI} = \log(\Delta\text{Hum} / \Delta\text{Con})$, where a positive value indicates effectiveness, and higher values are better

### Loss & Training

- Achieved parameter-efficient fine-tuning using DeepSpeed + LoRA (via PEFT library).
- Utilized Qwen2-Audio-7B-Instruct as the base model.
- Trained on the IDK dataset of each modality for 1 epoch.
- The LoRA alpha weight is a key hyperparameter: if too small, the model fails to learn to abstain; if too large, it becomes overly conservative.

## Key Experimental Results

### Main Results: Comparison of Accuracy / Truthfulness / Reliability

Based on the performance of Qwen2-Audio-7B-Instruct on the MMAU dataset (sound/music/speech):

| Method | Training? | Sound Acc% | Sound Rel% | Music Acc% | Music Rel% | Speech Acc% | Speech Rel% | Total Rel% |
|------|-------|-----------|-----------|-----------|-----------|------------|------------|-----------|
| Baseline | ✗ | 60.96 | 60.96 | 55.09 | 55.09 | 50.75 | 50.75 | 55.60 |
| IDK Prompting | ✗ | 58.26 | 73.03 | 54.19 | 65.19 | 43.84 | 56.18 | 64.85 |
| MCoT Prompting | ✗ | 57.96 | 67.13 | 51.50 | 67.53 | 44.74 | 57.71 | 64.29 |
| Task Agent | ✗ | 58.56 | 70.68 | 53.29 | 68.22 | 46.25 | 57.93 | 65.66 |
| LoRA SFT | ✓ | 61.71 | 70.71 | 51.35 | 66.43 | 47.90 | 59.91 | 65.68 |
| Human | - | 86.31 | 86.31 | 78.22 | 78.22 | 82.17 | 82.17 | 82.23 |

### Ablation Study: RGI Metric Comparison

| Method | Sound $\Delta\text{Con}\%$ | Sound $\Delta\text{Hum}\%$ | Sound RGI | Music RGI | Speech RGI | Total RGI |
|------|------------|------------|----------|----------|-----------|----------|
| IDK Prompting | 10.81 | 20.12 | 0.27 | 0.20 | 0.02 | 0.16 |
| MCoT Prompting | 11.71 | 14.41 | 0.09 | 0.25 | 0.09 | 0.15 |
| Task Agent | 9.61 | 16.52 | 0.24 | 0.27 | 0.17 | 0.23 |
| LoRA SFT | 6.91 | 15.62 | 0.36 | 0.23 | 0.19 | 0.26 |

### Key Findings

- **All methods improve reliability**: Both Truthfulness and Reliability are improved, but Accuracy drops, showing that enhancing reliability comes at the cost of "helpfulness" (utility).
- **Training-based methods outperform inference-time methods**: LoRA SFT achieves higher RGI (0.26 vs. 0.15–0.23 for training-free methods) with less loss in Accuracy, striking a better balance between conservatism and humility.
- **Higher RGI on Sound and Music**: This indicates that the model's knowledge boundaries on these two modalities are clearer, whereas performance on Speech is relatively poorer.
- **Cross-modal transfer is effective**: Training on one modality and testing on another yields RGI > 0 across the board, validating the cross-modal transferability of reliability awareness.
- **Non-monotonic effect of LoRA alpha**: A very small alpha is sufficient to learn high RGI, whereas an excessively large alpha leads to over-conservatism ($\text{RGI} < 0$), indicating that reliability awareness is an easily acquired capability.
- **Small variation in IDK ratio**: Moving from 50.2% for 1@5 to 63.5% for 5@5 shows smaller variations compared to text LLMs, indicating that LALM has higher response stability.

## Highlights & Insights

- **Discovery of a "meta-capability"**: Reliability awareness (knowing when to say "I don't know") can transfer across sound, music, and speech. This implies that abstention capability does not depend on the content understanding of a specific modality but is a general model-level capability, offering significant inspiration for building unified multimodal reliable systems.
- **Ingenious RGI metric design**: $\text{RGI} = \log(\Delta\text{Hum}/\Delta\text{Con})$ separately measures "good abstention" (humility) and "bad abstention" (conservatism), which traditional metrics cannot distinguish. A method might achieve high Reliability simply by rejecting everything (over-conservatism), but RGI can effectively detect this issue.
- **Modality-aware design of Task Agent**: Identifying the audio type first and then calling the corresponding tool (ASR/AAC/MC) represents a pipeline concept that is transferable to other multimodal tasks, such as classifying the image type before selecting a processing strategy in vision-language models.

## Limitations & Future Work

- **Only supports simple rejection**: The model can only say "I don't know" and cannot explain the reason for abstaining or actively query the user for more information, resulting in limited interactivity.
- **Validated only on Qwen2-Audio**: Even though other models were tested in the appendix, the main experiments only used one model, so the generalizability of the conclusions remains to be verified.
- **Evaluation limited to MMAU**: This dataset is in a multiple-choice format, and the effectiveness of the reliability enhancement methods under open-ended question-answering scenarios has not been verified.
- **High cost of building the IDK dataset**: The 5@5 threshold requires sampling inference 5 times for each question, which is computationally expensive.
- **Cross-modal transfer limited to intra-audio**: Although sound, music, and speech differ significantly, they all belong to the audio modality. Whether this capability can transfer to more distant modalities such as video or images remains to be explored.

## Related Work & Insights

- **vs. Reliability in Text LLMs (Cheng et al., 2024)**: The text domain first introduced the concept of IDK datasets and Knowledge Quadrants. This paper extends them to the multimodal audio setting and finds that the variation in the IDK ratio of audio models is relatively small (50.2% -> 63.5%), suggesting that LALMs are more stable in their responses than text LLMs.
- **vs. Reliability Evaluation by Xu et al. (2024a)**: Xu proposed a weighted Reliability metric, but it cannot differentiate between "good abstention" and "over-conservatism". The RGI proposed in this paper compensates for this shortcoming by comparing the gains in humility and conservatism.
- **vs. Qwen2-Audio**: As one of the strongest open-source LALMs currently available, its reliability on MMAU remains low (Rel = 55.6%), indicating that even powerful models require additional reliability enhancement.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic study on LALM reliability, well-designed RGI metric, inspiring discovery of the "meta-capability".
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three audio modalities, four methods, cross-modal experiments, and hyperparameter analysis, but validated only on a single dataset.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-defined problems, complete mathematical derivations, and rich diagrams.
- Value: ⭐⭐⭐⭐ Opens up a new direction for research on LALM reliability, with practical value in the RGI metric and cross-modal transfer discoveries.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Benchmarking Open-ended Audio Dialogue Understanding for Large Audio-Language Models](audio_dialogue_benchmark.md)
- [\[AAAI 2026\] AHAMask: Reliable Task Specification for Large Audio Language Models without Instructions](../../AAAI2026/audio_speech/ahamask_reliable_task_specification_for_large_audio_language.md)
- [\[ACL 2025\] Investigating and Enhancing Vision-Audio Capability in Omnimodal Large Language Models](investigating_and_enhancing_vision-audio_capability_in_omnimodal_large_language_.md)
- [\[ACL 2025\] Contextual Biasing with the Knowledgeable External Language Model for End-to-End Speech Recognition](contextual_biasing_with_the_knowledgeable_external_language_model_for_end-to-end.md)
- [\[ACL 2025\] Who Can Withstand Chat-Audio Attacks? An Evaluation Benchmark for Large Audio-Language Models](who_can_withstand_chat-audio_attacks_an_evaluation_benchmark_for_large_audio-lan.md)

</div>

<!-- RELATED:END -->
