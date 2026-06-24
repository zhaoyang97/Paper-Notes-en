---
title: >-
  [Paper Note] Investigating and Enhancing Vision-Audio Capability in Omnimodal Large Language Models
description: >-
  [ACL 2025][Audio & Speech][Omnimodal Large Language Models] It is discovered that current Omnimodal Large Language Models (OLLMs) perform significantly worse on vision-audio tasks than on vision-text tasks. The primary reason is the lack of direct alignment between vision and audio modalities. Consequently, this work proposes Self-KD (Self-Knowledge Distillation) to enhance vision-audio capabilities by leveraging the OLLM's own vision-text components as a teacher.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Omnimodal Large Language Models"
  - "Vision-Audio Alignment"
  - "Self-Knowledge Distillation"
  - "Inter-modal Gap"
  - "Multimodal Fusion"
date: 2026-05-08
content_hash: fe6bce8565bc172c
---

# Investigating and Enhancing Vision-Audio Capability in Omnimodal Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2503.00059](https://arxiv.org/abs/2503.00059)  
**Code**: [https://github.com/isruihu/Self-KD](https://github.com/isruihu/Self-KD)  
**Area**: Speech  
**Keywords**: Omnimodal Large Language Models, Vision-Audio Alignment, Self-Knowledge Distillation, Inter-modal Gap, Multimodal Fusion

## TL;DR

It is discovered that current Omnimodal Large Language Models (OLLMs) perform significantly worse on vision-audio tasks than on vision-text tasks. The primary reason is the lack of direct alignment between vision and audio modalities. Consequently, this work proposes Self-KD (Self-Knowledge Distillation) to enhance vision-audio capabilities by leveraging the OLLM's own vision-text components as a teacher.

## Background & Motivation

**Background**: Omnimodal Large Language Models (OLLMs) such as GPT-4o integrate vision, language, and audio capabilities, representing the frontier of multimodal AI. Open-source OLLMs like VITA, VITA-1.5, and Megrez have demonstrated strong performance on standard vision-text tasks.

**Limitations of Prior Work**: The performance of OLLMs on vision-audio inputs is significantly weaker than on vision-text inputs. For instance, Megrez-3B drops from an average score of 68.96 under text queries to 49.72 under audio queries (a decrease of 19.24); VITA-8x7B plummets from 70.04 to 7.84 (a decrease of 62.20). When the same text question is converted to audio, the model may output completely contradictory answers.

**Key Challenge**: In the training pipeline of OLLMs, vision-text and audio-text alignments are conducted separately, but direct alignment between vision and audio is never established. The model can only implicitly learn to integrate these two modalities during the vision-audio SFT stage, which experiments show is insufficient.

**Goal**: (1) Systematically quantify and analyze the causes of the vision-audio gap in OLLMs; (2) propose an effective method to narrow this gap.

**Key Insight**: Attention weight analysis reveals that the model pays less attention to vision tokens under audio queries than under text queries. The MMAlign benchmark is constructed to verify that vision-audio alignment is weaker than vision-text alignment. Based on this, a method is proposed to distill the vision-audio components using the model's own vision-text components.

**Core Idea**: The vision-text capability of an OLLM is far stronger than its vision-audio capability, allowing the former to serve as a teacher to enhance the latter via self-knowledge distillation.

## Method

### Overall Architecture

Self-KD is a knowledge distillation training framework characterized by the teacher and the student coming from different components of the same model:

1. **Teacher Model** $p_T$: The vision-text component of the OLLM (Vision Encoder + Text Embedding Layer + LLM).
2. **Student Model** $p_S$: The vision-audio component of the OLLM (Vision Encoder + Audio Encoder + LLM).
3. **Inputs**: The teacher uses a vision-text sample $x^t$, while the student uses the corresponding vision-audio sample $x^a$ (the text question is converted to audio via TTS).
4. **Training Goal**: Force the student to mimic the teacher's behavior in processing visual information.

### Key Designs

#### 1. Problem Discovery & Analysis

- **Function**: Systematically quantify and analyze the vision-audio capability gap in OLLMs.
- **Key Findings**:
    - All tested OLLMs (VITA/VITA-1.5/Megrez) show a significant drop in performance under audio queries.
    - Models exhibit a higher "Yes" bias under audio queries.
    - Audio responses are usually relevant to the image but inaccurate—the model "sees" the image but fails to correctly integrate information.
- **Attention Analysis**: Under audio queries, the attention weights from query tokens to vision tokens are significantly lower in the middle-to-late layers compared to text queries. However, there is negligible difference in the attention weights from response tokens to vision/query tokens.

#### 2. MMAlign Benchmark

- **Function**: Construct a benchmark specifically for evaluating the quality of vision-text and vision-audio alignment.
- **Mechanism**: Based on the ARO dataset, each sample contains an image and two descriptions (one correct, one distractor), and the model must choose the correct description. The questions are asked in both text and audio formats.
- **Results**: The text query accuracy of all models is much higher than that of audio queries (e.g., VITA-1.5: 75.67 vs. 32.83), directly proving that vision-audio alignment is weaker than vision-text alignment.

#### 3. Self-KD Training Framework

- **Function**: Utilize the OLLM's own vision-text component to guide the training of its vision-audio component.
- **Mechanism**:
    - Regular SFT Loss: $L_{\text{SFT}} = \mathbb{E}[-\log p_S(y|x^a)]$
    - Self-KD Loss (KL Divergence): $L_{\text{Self-KD}} = \text{KL}(p_T \| p_S) = \mathbb{E}\left[\log \frac{p_T(y|x^t)}{p_S(y|x^a)}\right]$
    - Total Loss: $L = \alpha L_{\text{Self-KD}} + (1-\alpha) L_{\text{SFT}}$
- **Design Motivation**: Unlike conventional KD where both teacher and student share the same input, in Self-KD the teacher uses vision-text inputs while the student uses corresponding vision-audio inputs. This forces the student to learn to process audio in the same manner as text, including allocating more attention to vision tokens.

### Loss & Training

- **Total Loss**: $L = \alpha L_{\text{Self-KD}} + (1-\alpha) L_{\text{SFT}}$, where the hyperparameter $\alpha$ controls the ratio between SFT and KD.
- **Audio-Text Alignment Stage**: ASR datasets such as LibriSpeech, Common Voice, GigaSpeech, and Libriheavy are used, totaling 988k samples.
- **Vision-Audio SFT and Self-KD Stage**: 50k instruction-following samples are sampled from llava-1.5-mix-665k, and text questions are converted to audio via TTS.
- The audio encoder uses Whisper-large-v3, projected into the LLM space via a single-layer MLP.
- Base Models: InternVL2 series (1B/2B/4B/8B) and Qwen2VL series (2B/7B).

## Key Experimental Results

### Main Results

**Average performance of different models on Self-KD vs. conventional SFT** (average score across 8 vision benchmarks):

| Model | VL (Text) | SFT (Audio) | Self-KD (Audio) | KD Gain |
|------|----------|-----------|---------------|---------|
| InternVL2-1B | 49.68 | 21.16 | **33.84** | +12.68 |
| InternVL2-2B | 53.94 | 22.52 | **36.58** | +14.06 |
| InternVL2-4B | 59.38 | 32.22 | **42.30** | +10.08 |
| InternVL2-8B | 69.91 | 38.71 | **51.45** | +12.74 |
| Qwen2VL-2B | 64.77 | 46.21 | **52.58** | +6.37 |
| Qwen2VL-7B | 75.14 | 67.75 | **68.27** | +0.52 |

Self-KD significantly outperforms conventional SFT across almost all configurations, with the InternVL2 series showing improvements of 10-14 points, while the Qwen2VL series gains are smaller (likely because its vision-text alignment is inherently better).

### Ablation Study

**SFT vs. Self-KD on MMAlign Benchmark** (Relation/Attribute accuracy of InternVL2-1B):

| Model | Relation (SFT) | Relation (Self-KD) | Attribute (SFT) | Attribute (Self-KD) |
|------|---------------|-------------------|-----------------|---------------------|
| InternVL2-1B | 42.67 | **50.67** | — | — |

Self-KD also outperforms SFT in multimodal alignment quality, further validating its effectiveness.

**Attention Weight Changes**: After Self-KD, attention weights on vision tokens under audio queries increase, making the behavior closer to the text query pattern.

### Key Findings

1. **The gap between VL and VA capability is a universal phenomenon**: It exists in all tested OLLMs and is not an isolated case.
2. **Stronger VL capability leads to better VA performance after SFT**: VL performance is positively correlated with VA performance (e.g., InternVL2-8B has the strongest VL at 69.91, and its VA after SFT is also the best at 38.71).
3. **The effectiveness of Self-KD is proportional to the model's VL capability**: The stronger the teacher, the better the distillation effect.
4. **The Qwen series shows smaller improvements**: This might be because the Qwen series inherently boasts high-quality vision-text alignment, making conventional SFT sufficient.
5. **The root of the problem lies in attention allocation**: Under audio queries, the attention given by query to vision is insufficient, and Self-KD can improve this behavior.

## Highlights & Insights

1. **Novel and important problem definition**: This is the first work to systematically quantify and analyze the vision-audio capability gap in OLLMs, filling an evaluation vacancy.
2. **In-depth analysis**: It comprehensively reveals the nature of the problem through three dimensions: attention weight analysis, "Yes" bias analysis, and the MMAlign benchmark.
3. **Elegant solution**: Self-KD does not introduce external models; instead, it leverages the OLLM's own strong VL capability to enhance its weak VA capability.
4. **Cross-model generalizability**: It is effective across both InternVL2 (4 sizes) and Qwen2VL (2 sizes), verifying the universality of the method.

## Limitations & Future Work

1. TTS-synthesized audio may differ from real human speech; hence, evaluations might not fully reflect real-world scenarios.
2. The training data size is small (only 50k samples); scaling up the data could further improve performance.
3. The experiments only test English scenarios; multilingual vision-audio capabilities remain to be validated.
4. Self-KD requires simultaneous forward propagation of both teacher and student components, which increases training costs.
5. Direct vision-audio alignment methods (e.g., contrastive learning) have not been explored, which could offer a more fundamental solution.

## Related Work & Insights

- **GPT-4o (Hurst et al., 2024)**: Representing commercial OLLMs, it might suffer from similar vision-audio capability issues.
- **VITA/VITA-1.5 (Fu et al., 2024/2025)**: Open-source OLLM baselines, displaying the most severe degradation in VA performance.
- **Megrez (Li et al., 2025)**: A 3B lightweight model but with the smallest VA gap (19.24), indicating that model architecture design is also crucial.
- **Bi et al. (2024)**: Pioneering study showing that attention distribution reflects modal alignment, which provides a theoretical basis for the analysis in this work.
- **Insight**: Hidden alignment issues in multimodal models ("out of sight" modal alignments) might be more severe than expected. Self-distillation serves as a low-cost approach for cross-modal capability transfer.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The discovery of the problem is novel, and while Self-KD is simple, its entry point is unique.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Extremely comprehensive, covering 6 models, 4 scales, 8 benchmarks, and 3 analytical dimensions.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The logical progression from problem to analysis to solution is exceptionally clear.
- **Value**: ⭐⭐⭐⭐ — Reveals a critical vulnerability in OLLMs; the Self-KD method exhibits excellent generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Benchmarking Open-ended Audio Dialogue Understanding for Large Audio-Language Models](audio_dialogue_benchmark.md)
- [\[ACL 2025\] Towards Reliable Large Audio Language Model](towards_reliable_large_audio_language_model.md)
- [\[NeurIPS 2025\] AudSemThinker: Enhancing Audio-Language Models through Reasoning over Semantics of Sound](../../NeurIPS2025/audio_speech/audsemthinker_enhancing_audio-language_models_through_reasoning_over_semantics_o.md)
- [\[ACL 2025\] Who Can Withstand Chat-Audio Attacks? An Evaluation Benchmark for Large Audio-Language Models](who_can_withstand_chat-audio_attacks_an_evaluation_benchmark_for_large_audio-lan.md)
- [\[ACL 2025\] SpeechIQ: Speech-Agentic Intelligence Quotient Across Cognitive Levels in Voice Understanding by Large Language Models](speechiq_speechagentic_intelligence_quotient_across_cognitive.md)

</div>

<!-- RELATED:END -->
