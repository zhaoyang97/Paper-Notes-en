---
title: >-
  [Paper Note] EMoVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions
description: >-
  [CVPR 2025][Audio & Speech][Omni-modal LLM] EMoVA is proposed as the first end-to-end omni-modal LLM that achieves visual understanding, speech recognition, and emotion-controllable speech synthesis simultaneously through a semantic-acoustic decoupled speech tokenizer, outperforming GPT-4o on vision-language benchmarks and achieving a 2.9% WER in speech recognition.
tags:
  - "CVPR 2025"
  - "Audio & Speech"
  - "Omni-modal LLM"
  - "Semantic-Acoustic Decoupling"
  - "Emotional Speech Generation"
  - "Speech Understanding"
  - "End-to-End Dialogue"
date: 2026-05-08
content_hash: e293b7a4677b727a
---

# EMoVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions

**Conference**: CVPR 2025  
**arXiv**: [2409.18042](https://arxiv.org/abs/2409.18042)  
**Code**: [https://emova-ollm.github.io/](https://emova-ollm.github.io/)  
**Area**: Audio & Speech / Multimodal LLM  
**Keywords**: Omni-modal LLM, Semantic-Acoustic Decoupling, Emotional Speech Generation, Speech Understanding, End-to-End Dialogue

## TL;DR

EMoVA is proposed as the first end-to-end omni-modal LLM that achieves visual understanding, speech recognition, and emotion-controllable speech synthesis simultaneously through a semantic-acoustic decoupled speech tokenizer, outperforming GPT-4o on vision-language benchmarks and achieving a 2.9% WER in speech recognition.

## Background & Motivation

**Background**: The field of multimodal LLMs is evolving rapidly, with vision-language models (e.g., LLaVA, Qwen-VL) and speech-language models (e.g., SpeechGPT, VITA) achieving significant progress respectively. However, end-to-end models supporting "seeing, hearing, and speaking" simultaneously remain scarce.

**Limitations of Prior Work**: Existing omni-modal approaches face two key issues: (1) Entanglement of semantic and acoustic information during speech tokenization—if speech is directly quantized into discrete tokens (e.g., via HuBERT K-means), the speaking style (emotion/intonation) interferes with the learning of semantic content, and vice versa; (2) Interference between different modalities during joint multimodal training—training vision before speech (or vice versa) leads to catastrophic forgetting.

**Key Challenge**: Speech is inherently a composite signal of "content + style". If both are quantized jointly using a unified codebook, the LLM must reason about content and predict intonation simultaneously when forecasting the next token, which makes the task overly complex.

**Goal**: Design an end-to-end omni-modal LLM capable of high-quality visual/speech input processing and emotion-controllable speech output simultaneously.

**Key Insight**: Decouple the speech signal into semantic representations (quantized into discrete tokens for the LLM) and acoustic style representations (kept continuous for controlling the emotion and pitch of synthesized speech), allowing the LLM to focus solely on semantic reasoning.

**Core Idea**: Semantic-acoustic decoupled speech tokenizer + joint multimodal training = end-to-end "seeing-hearing-speaking" omni-modal LLM.

## Method

### Overall Architecture

Four core components: (1) Visual encoder QwenViT + MLP projector to process images; (2) S2U (Speech-to-Unit) tokenizer to decouple and quantize speech into semantic tokens with a 4096 codebook (25 tokens/second); (3) Core LLM based on Qwen-2.5 (3B/7B/72B); (4) U2S (Unit-to-Speech) detokenizer to synthesize speech waveforms from semantic tokens + style embeddings based on VITS. Three-stage training: Visual alignment $\rightarrow$ Omni-modal joint alignment $\rightarrow$ Omni-modal instruction tuning.

### Key Designs

1. **Semantic-Acoustic Decoupled Speech Tokenizer (S2U)**:

    - **Function**: Separate speech into distinct semantic content and acoustic style representations.
    - **Mechanism**: The SPIRAL encoder first extracts speech embeddings, which then branch into two paths: the semantic path quantizes embeddings into discrete tokens $E_{\text{semantic}}$ of a 4096-size codebook via Finite Scalar Quantization (FSQ), while the acoustic path maintains continuous representations $E_{\text{style}}$ to control synthesized emotion/pitch. An auxiliary phoneme decoder is attached to the semantic branch to ensure sufficient linguistic information is preserved.
    - **Design Motivation**: Ablation studies show that during joint training, decoupling offers significant advantages over the entangled version (HuBERT K-means) in both ASR and vision tasks. Decoupling allows the LLM to focus strictly on "what was said" rather than "how it was said".

2. **Omni-Modal Joint Alignment Training**:

    - **Function**: Train vision-text and speech-text alignments simultaneously to prevent catastrophic forgetting among modalities.
    - **Mechanism**: Stage 2 uses 7.4M samples (image-text + speech-text) for joint training. All modalities are unified into a text-centric sequence: visual features are mapped to the text embedding space via a projector, and speech tokens are directly incorporated into the LLM vocabulary.
    - **Design Motivation**: Ablations show that Joint Training > Sequential Training (either Vision $\rightarrow$ Speech or Speech $\rightarrow$ Vision), where the latter degrades the performance of the first-trained modality due to catastrophic forgetting.

3. **Emotion-Controllable Speech Synthesis (U2S)**:

    - **Function**: Generate speech with specified emotion and intonation.
    - **Mechanism**: A VITS-based conditional VAE that takes semantic tokens + style embeddings (4 emotions $\times$ 3 intonations $\times$ 2 genders = 24 combinations) as input. The LLM first generates a text response, then predicts the emotion label and speech tokens, which are synthesized into audio by U2S.
    - **Design Motivation**: Chain-of-Modality generation: The sequential generation of Text $\rightarrow$ Emotion $\rightarrow$ Speech minimizes and controls the decision space at each step.

### Loss & Training

Stage 1: Visual projector alignment (LCS-558K). Stage 2: Autoregressive language model loss 

$$\mathcal{L} = -\sum_i \log P(x_i|x_{<i})$$

with 7.4M joint alignment data. Stage 3: 4.4M multi-task instruction tuning + emotion labels. S2U is pre-trained using contrastive loss + phoneme reconstruction loss (on 20K hours of unlabeled speech). U2S utilizes the VAE + GAN loss from VITS. Training is completed on 128 $\times$ Ascend 910B NPUs.

## Key Experimental Results

### Main Results

Vision-language benchmarks (EMOVA-72B vs. other omni-modal LLMs):

| Benchmark | EMOVA-72B | GPT-4o | Gemini-1.5-Pro |
|-----------|-----------|--------|---------------|
| MME | **2402** | 2329 | 2228 |
| MMBench | **86.4** | 83.4 | 73.9 |
| TextVQA | 81.4 | - | 73.5 |
| DocVQA | **95.9** | 92.8 | 93.1 |
| OCRBench | **843** | 736 | 754 |

Speech Recognition (LibriSpeech test-clean WER$\downarrow$):

| Model | WER |
|------|-----|
| Mini-Omni | 4.8 |
| VITA | 8.1 |
| **EMOVA-72B** | **2.9** |
| Whisper-Large | 3.0 |

### Ablation Study

| Training Strategy | MMBench | ASR WER |
|---------|---------|---------|
| Vision-only | 83.2 | - |
| Speech-only | - | 4.1 |
| Sequential (VL $\rightarrow$ Speech) | 81.5 | 4.3 |
| Sequential (Speech $\rightarrow$ VL) | 83.0 | 5.9 |
| **Joint (Decoupled)** | **83.8** | **4.1** |
| Joint (Entangled/HuBERT) | 82.1 | 6.3 |

### Key Findings
- **Joint training outperforms sequential training**: It prevents catastrophic forgetting, allowing all modalities to benefit simultaneously.
- **Decoupling $\gg$ Entanglement**: Semantic-acoustic decoupling reduces the ASR WER from 6.3% to 4.1% and improves vision metrics from 82.1% to 83.8%.
- **72B model supports better ASR**: Achieving a WER of 2.9%, which matches or exceeds Whisper-Large (3.0%), demonstrating that LLM scaling directly benefits cross-modal tasks.
- **Emotion classification accuracy $>75\%$**: Validates the effectiveness of emotion control in synthesized speech.

## Highlights & Insights

- **Fundamental Insight into Semantic-Acoustic Decoupling**: Speech "content" and "style" represent two fundamentally different kinds of information. LLMs excel at reasoning about content but are less suited for modeling fine-grained acoustic details. Decoupling allows each module to perform its specific role.
- **Outperforming GPT-4o on 11/15 benchmarks**: The 72B version surpasses GPT-4o on most vision-language benchmarks, demonstrating that open-source omni-modal LLMs are catching up with closed-source models.
- **Chain-of-Modality Generation Strategy**: Generating Text $\rightarrow$ Emotion labels $\rightarrow$ Speech tokens sequentially reduces the decision space at each step.

## Limitations & Future Work

- **Half-Duplex Limitation**: The model can only process input and output alternately, failing to listen and speak concurrently like human conversation.
- **Dependency on Text Intermediation**: Speech generation requires generating text before synthesizing speech, which increases latency.
- **Single Vision Encoder**: Only QwenViT is used, without integrating self-supervised vision models (e.g., DINOv2) or Mixture of Experts (MoE) vision models.
- **Only Supports Visual Understanding**: No support for image generation (unlike Emu3).
- **Enormous Training Resources**: Requiring 128 $\times$ Ascend 910B NPUs, posing a very high barrier to replication.

## Related Work & Insights

- **vs. VITA**: VITA also supports speech but yields a WER of 8.1% (significantly trailing EMoVA's 2.9%) and lacks support for emotion-controllable synthesis.
- **vs. Mini-Omni**: Mini-Omni offers faster real-time speech interaction but possesses weaker understanding capabilities (WER of 4.8%).
- **vs. AnyGPT**: AnyGPT uses entangled SpeechTokens without decoupling, resulting in poor joint multimodal training performance.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of semantic-acoustic decoupling and joint training strategy is effective and novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across vision, speech, and emotion, with thorough ablations and comparisons across three model scales.
- Writing Quality: ⭐⭐⭐⭐ Clear architecture, though some training details are somewhat scattered.
- Value: ⭐⭐⭐⭐⭐ The first open-source omni-modal LLM to comprehensively outperform GPT-4o on major benchmarks, marking a significant milestone.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Hear What You See: Video-to-Audio Generation with Diffusion Transformer and Semantic-Temporal Alignment-Ranked Direct Preference Optimization](../../CVPR2026/audio_speech/hear_what_you_see_video-to-audio_generation_with_diffusion_transformer_and_seman.md)
- [\[ICCV 2025\] Learning to See Inside Opaque Liquid Containers using Speckle Vibrometry](../../ICCV2025/audio_speech/learning_to_see_inside_opaque_liquid_containers_using_speckle_vibrometry.md)
- [\[ICLR 2026\] VowelPrompt: Hearing Speech Emotions from Text via Vowel-level Prosodic Augmentation](../../ICLR2026/audio_speech/vowelprompt_hearing_speech_emotions_from_text_via_vowel-level_prosodic_augmentat.md)
- [\[ACL 2025\] SpeechIQ: Speech-Agentic Intelligence Quotient Across Cognitive Levels in Voice Understanding by Large Language Models](../../ACL2025/audio_speech/speechiq_speechagentic_intelligence_quotient_across_cognitive.md)
- [\[ICML 2025\] Long-Form Speech Generation with Spoken Language Models](../../ICML2025/audio_speech/long-form_speech_generation_with_spoken_language_models.md)

</div>

<!-- RELATED:END -->
