---
title: >-
  [Paper Note] Lyra: An Efficient and Speech-Centric Framework for Omni-Cognition
description: >-
  [ICCV 2025][Audio & Speech][Multimodal Large Language Models] This paper proposes Lyra, a speech-centric and efficient omni-cognition MLLM framework. Through three key strategies—multimodal LoRA, a latent cross-modality regularizer, and a latent multi-modality extractor—Lyra achieves state-of-the-art performance across vision-language-speech modalities with less training data, and is the first to support speech inputs spanning several hours.
tags:
  - ICCV 2025
  - "Audio & Speech"
  - Multimodal Large Language Models
  - Speech Understanding
  - Omni-Cognition
  - Efficient Inference
  - Long-form Speech
date: 2026-05-08
content_hash: 06030ce17f8c568f
---

# Lyra: An Efficient and Speech-Centric Framework for Omni-Cognition

**Conference**: ICCV 2025
**arXiv**: [2412.09501](https://arxiv.org/abs/2412.09501)
**Code**: [github.com/dvlab-research/Lyra](https://github.com/dvlab-research/Lyra)
**Area**: Speech / Multimodal
**Keywords**: Multimodal Large Language Models, Speech Understanding, Omni-Cognition, Efficient Inference, Long-form Speech

## TL;DR

This paper proposes Lyra, a speech-centric and efficient omni-cognition MLLM framework. Through three key strategies—multimodal LoRA, a latent cross-modality regularizer, and a latent multi-modality extractor—Lyra achieves state-of-the-art performance across vision-language-speech modalities with less training data, and is the first to support speech inputs spanning several hours.

## Background & Motivation

### State of the Field

Current multimodal large language models (MLLMs) primarily focus on vision-language or speech-language dual-modality interaction. Truly omni-modal models—capable of simultaneously processing images, video, speech, and audio—remain underdeveloped. While OpenAI's GPT-4o has demonstrated the potential of omni-modal interaction, open-source omni-modal models still exhibit notably limited speech capabilities.

### Limitations of Prior Work

**Speech modality neglected**: Existing omni-modal models (VITA, AnyGPT, EMOVA, etc.) focus primarily on the speech-text relationship without deeply exploring cross-modal interactions between speech and other modalities such as vision.

**Data scale vs. training cost trade-off**: Scaling to full omni-modality requires larger datasets and significantly more computational resources.

**Long-form speech processing constrained**: Existing models are limited by speech encoders (e.g., Whisper) to at most 30 seconds to 1 minute of audio input.

**Incomplete evaluation standards**: Prior omni-modal models evaluate speech-text capability solely via LibriSpeech WER, neglecting cross-modal performance between speech and vision.

### Key Findings

Good speech-text performance does not imply good speech-vision performance. Experiments show that when trained with $\mathcal{L}_{CE}$ alone, the speech+image (S+I) setting underperforms text+image (T+I) by 8 percentage points on MM-Vet (53.1 vs. 61.1), indicating that the semantic gap between speech tokens and text tokens requires dedicated bridging.

## Method

### Overall Architecture

Lyra consists of four core components: a Latent Cross-Modality Regularizer (LCMR), multimodal LoRA, a Latent Multi-Modality Extractor (LMME), and a streaming speech-text generation module. Inputs from each modality are processed by their respective encoders and projectors before being fed into the LLM, where the multimodal LoRA and LMME modules operate collaboratively.

### Key Designs

#### 1. **Latent Cross-Modality Regularizer (LCMR)**

- **Function**: Aligns speech tokens to their corresponding transcription text tokens prior to the LLM input, bridging the semantic gap between speech and text.
- **Mechanism**: Since speech tokens and transcription text tokens differ in sequence length (speech sequences are typically longer), Dynamic Time Warping (DTW) is employed to compute the minimum alignment distance:

$$\mathbf{D}_{l,s} = \text{dist}(l,s) + \min\{\mathbf{D}_{l,s-1}, \mathbf{D}_{l-1,s}, \mathbf{D}_{l-1,s-1}\}$$

where $\text{dist}(l,s) = -\log[\text{softmax}(\mathbf{X}_{[\text{speech}],l} \mathbf{X}_{[\text{STT}],s}^\top / \tau)]$

The regularization loss is defined as $\mathcal{L}_{LCMR} = \frac{1}{L+S}\mathbf{D}_{L,S}$, and the total loss is $\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{LCMR}$.

- **Design Motivation**: Although speech and text are highly overlapping in semantics, directly using speech tokens as instructions significantly degrades cross-modal performance. DTW alignment of variable-length sequences encourages speech token representations to approach text token representations before entering the LLM.

#### 2. **Multi-Modality LoRA**

- **Function**: Employs independent LoRA adapters for different modality combinations, extending speech capabilities while preserving the original visual competence.
- **Mechanism**: The output is formulated as $\mathbf{H} = (\mathbf{B}_{[M]}\mathbf{A}_{[M]} + \mathbf{W})\mathbf{X}_{[M]}$, where $\mathbf{A}_{[M]}$ and $\mathbf{B}_{[M]}$ are low-rank adapters corresponding to modality combination $M$.
- **Design Motivation**: Under limited data quantity and quality, direct joint training across vision, speech, and language modalities degrades the model's existing visual capabilities. The parameter-efficient nature of LoRA mitigates this issue.

#### 3. **Latent Multi-Modality Extractor (LMME)**

- **Function**: Dynamically filters multimodal tokens based on their relevance to the text query, discarding redundant tokens to improve efficiency.
- **Mechanism**: The LLM is partitioned into $n$ blocks. At the end of each block, the top-k non-text tokens are selected based on their attention scores with the text query:

$$\text{topk}\left(\text{softmax}\left(\frac{\mathbf{Q}_{[\text{text}]}\mathbf{K}_{[\backslash\text{text}]}^\top}{\sqrt{d}}\right)\right)$$

Each block retains $\rho L$ tokens, with the token count decaying exponentially across blocks.
- **Design Motivation**: In multimodal long-context settings (high-resolution images, long videos, long-form speech), the majority of tokens are irrelevant to the instruction, introducing both computational overhead and noise. Dynamic filtering retains only 10%–25% of the tokens most relevant to the instruction.

### Loss & Training

Four-stage training pipeline:
1. Speech encoder pre-training (text-to-speech).
2. Joint text-image-speech training (LLM + projectors).
3. Long-form speech capability extension.
4. Speech generator training (streaming text + audio output).

Speech token compression: Empirical results confirm that compressing 30-second speech from 1,500 tokens to 300 tokens incurs negligible performance loss (TextVQA$^S$: 77.8% vs. 76.8%), reducing a 2-hour audio from 360,000 tokens to a tractable range.

## Key Experimental Results

### Main Results

| Method | Params | TextVQA | MME | MM-Vet | VideoMME | TextVQA$^S$ | DocVQA$^S$ | LibriSpeech↓ |
|--------|--------|---------|-----|--------|----------|------------|-----------|-------------|
| Mini-Gemini | 8B | 71.9 | 1989 | 53.5 | - | - | - | - |
| LLaVA-OV | 7B | 65.4 | 1998 | 57.5 | 58.2 | - | - | - |
| Intern-VL2 | 8B | 77.4 | 2211 | 60.0 | 54.0 | - | - | - |
| VITA | 66B | - | 2097 | 41.6 | 59.2 | - | - | 8.1 |
| EMOVA | 14B | 82.0 | 2205 | 55.8 | - | - | - | 4.0 |
| **Lyra-Mini** | **3B** | 78.3 | 1884 | 51.2 | 55.0 | 73.4 | 74.8 | 2.1 |
| **Lyra-Base** | **9B** | 82.6 | 2335 | **63.5** | 62.8 | **80.0** | **85.5** | **2.0** |
| **Lyra-Pro** | **74B** | **83.5** | **2485** | **71.4** | **69.9** | **81.0** | **89.4** | **1.8** |

Lyra-Base achieves approximately 9% improvement over the best omni-modal model on image-speech tasks, and approximately 2% improvement on speech-text tasks.

### Ablation Study

| Component | TextVQA (S+I) | TextVQA (T+I) | MM-Vet (S+I) | MM-Vet (T+I) | LibriSpeech (S+T) |
|-----------|--------------|--------------|-------------|-------------|------------------|
| Baseline (w/o LCMR) | - | 82.3 | - | 62.8 | - |
| $\mathcal{L}_{CE}$ only | 76.7 | 79.5 | 53.1 | 61.1 | 1.9 |
| $\mathcal{L}_{CE} + \lambda\mathcal{L}_{LCMR}$ | **77.8** | **80.1** | **58.1** | **62.6** | 2.0 |

LMME efficiency gains:

| Token Count | Prefill Time (Baseline → LMME(4,0.7)) | Memory (Baseline → LMME(4,0.7)) |
|:-----------:|:------------------------------------:|:-------------------------------:|
| $2^{13}$ | 0.65s → 0.37s | 30G → 18G |
| $2^{15}$ | 2.99s → 1.23s | 60G → 30G |
| $2^{16}$ | OOM → 3.05s | OOM → 46G |

### Key Findings

- LCMR improves both speech-image and text-image performance, indicating that speech alignment training positively transfers to overall multimodal capability.
- Evaluating the speech modality solely via LibriSpeech WER is insufficient: under $\mathcal{L}_{CE}$, the speech-text WER gap is negligible (1.9 vs. 2.0), yet the speech-vision performance gap is substantial.
- LMME ultimately retains only 10%–25% of multimodal tokens, achieving over 50% training speedup.
- Compressing speech tokens to 300 represents the optimal performance-efficiency trade-off.

## Highlights & Insights

1. **Speech-centric evaluation perspective**: This work exposes a blind spot in existing omni-modal model evaluation—strong speech-text performance does not guarantee strong speech-vision performance.
2. **DTW for cross-modal alignment**: DTW is elegantly repurposed to address the variable-length mismatch between speech and text token sequences.
3. **Unified tri-modal token compression framework**: LMME is uniformly applicable to image, video, and speech tokens, substantially reducing resource requirements in long-context scenarios.
4. **First long-form speech SFT dataset**: A dataset of 12K long-speech samples (ranging from minutes to 2 hours) fills a critical gap in training data for long-form speech understanding.

## Limitations & Future Work

1. Lyra relies on Qwen2-VL as the visual backbone, which imposes an upper bound on visual capability.
2. The computational complexity of DTW alignment is $O(L \times S)$, which may become a bottleneck for long-form speech.
3. The long-form speech SFT dataset contains only 12K samples, which remains limited in scale.
4. Speech generation quality is not quantitatively evaluated (e.g., via MOS scores).
5. Only English speech is supported; multilingual generalizability has not been validated.

## Related Work & Insights

- Qwen2-VL's dynamic-resolution visual token processing provides Lyra with a strong visual backbone.
- The 30-second limitation of Whisper-large-v3 as a speech encoder motivates the proposed long-form speech processing solution.
- FastV's visual token pruning concept is extended by LMME to the multimodal setting.
- The high-resolution image slicing strategy from LLaVA-NeXT is adapted for segmenting long-form speech inputs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The speech-centric omni-modal framework design and evaluation perspective are original; the use of DTW for cross-modal alignment is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive comparisons across vision-language, vision-speech, and speech-language settings with detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with persuasive motivation.
- **Value**: ⭐⭐⭐⭐⭐ — Fills a gap in speech-centric omni-modal MLLMs; open-sourced code and datasets represent a significant contribution to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Understanding Co-speech Gestures in-the-wild](understanding_co-speech_gestures_in-the-wild.md)
- [\[ICCV 2025\] Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations](zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)
- [\[ICCV 2025\] Learning to See Inside Opaque Liquid Containers using Speckle Vibrometry](learning_to_see_inside_opaque_liquid_containers_using_speckle_vibrometry.md)
- [\[ICCV 2025\] 2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining](25_years_in_class_a_multimodal_textbook_for_visionlanguage_p.md)
- [\[ICCV 2025\] VGGSounder: Audio-Visual Evaluations for Foundation Models](vggsounder_audio-visual_evaluations_for_foundation_models.md)

</div>

<!-- RELATED:END -->
