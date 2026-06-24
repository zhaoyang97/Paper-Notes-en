---
title: >-
  [Paper Note] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering
description: >-
  [AAAI2026][Audio & Speech][Spoken Question Answering] This paper proposes CLSR, an end-to-end contrastive language-speech retriever. By first converting acoustic representations into text-like representations and then aligning them with text, it efficiently extracts question-relevant segments from long-form audio, providing RAG support for downstream LALM-based long spoken question answering.
tags:
  - "AAAI2026"
  - "Audio & Speech"
  - "Spoken Question Answering"
  - "Contrastive Learning"
  - "Retrieval-Augmented Generation"
  - "Speech-Text Alignment"
  - "CIF"
date: 2026-05-08
content_hash: 59ad60d8f5debdba
---

# End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering

**Conference**: AAAI2026  
**arXiv**: [2511.09282](https://arxiv.org/abs/2511.09282)  
**Code**: [193746/CLSR](https://github.com/193746/CLSR)  
**Area**: Audio & Speech  
**Keywords**: Spoken Question Answering, Contrastive Learning, Retrieval-Augmented Generation, Speech-Text Alignment, CIF

## TL;DR
This paper proposes CLSR, an end-to-end contrastive language-speech retriever. By first converting acoustic representations into text-like representations and then aligning them with text, it efficiently extracts question-relevant segments from long-form audio, providing RAG support for downstream LALM-based long spoken question answering.

## Background & Motivation
- Existing SQA (Spoken Question Answering) models can mostly handle short audio of no more than 1 minute, whereas speech in real-world scenarios (meetings, lectures, online discussions) often exceeds 10 minutes.
- Although Large Audio-Language Models (LALMs) have strong speech understanding capabilities, their inference speed slows down and accuracy degrades on long audio.
- RAG has shown significant effects in text-based long-context QA, naturally raising a question: can RAG be applied to speech to retrieve the most question-relevant segments from long audio?
- Existing speech retrievers (such as CLAP, SpeechDPR) show insufficient performance—CLAP excels at "audio effect-to-text" alignment rather than "speech content-to-text" alignment, while SpeechDPR is limited by text-free training and data scarcity.

## Core Problem
How to construct an end-to-end speech-text retriever that can achieve or even exceed the retrieval accuracy of pipeline methods without relying on cascaded ASR + text retrieval, while significantly reducing the inference time and error rate of long spoken question answering?

## Method

### Overall Architecture
CLSR consists of two parts:
1. **Left Part**: A non-autoregressive Attention Encoder-Decoder (AED) based on CIF (Continuous Integrate-and-Fire), which takes speech $X$ as input and outputs the token probability distribution $D$.
2. **Right Part**: A Transformer text encoder (frozen BGE-base) that receives text-like embeddings or ground-truth text embeddings, outputting sentence-level representations for contrastive learning.

### CIF Module
- The speech encoder (SAN-M structure) extracts acoustic features $H^s$.
- CIF calculates the weight of each frame $\alpha_i \in [0,1]$ via convolution, accumulating frame-by-frame until it exceeds the threshold $\beta$, thereby mapping the time steps to the number of tokens to obtain the acoustic representation $E^a$.
- This step achieves soft monotonic alignment from the "frame level" to the "token level".

### Sampler Training Optimization
- Training is conducted in two passes: the first pass directly predicts the token distribution using $E^a$ to obtain the ASR output $Y^{asr}$.
- The second pass compares $Y^{asr}$ with the ground-truth $Y^{con}$, replacing the correct embeddings into $E^a$ at erroneous token positions with a sampling ratio $\lambda$, generating the mixed feature $E^s$.
- $E^s$ is then used to re-predict the token distribution $D'$, enhancing the context modeling capability of the decoder.

### VQ Adaptor (Vector Quantization Adaptor)
- Applies argmax to the token probability distribution $D$ to obtain the index of the highest probability token $q_i$.
- Uses temperature-scaled softmax ($\gamma=0.1$) + straight-through gradient estimation to maintain gradient propagation.
- Performs matrix multiplication of the quantized one-hot matrix $Q^{st}$ and the embedding weights of the text encoder $W^{te}$ to obtain the text-like embedding $E^{Y'}$.
- Core Idea: Instead of directly aligning acoustic and text representations, the acoustic representation is "translated" into an approximate representation in the text space via VQ, and contrastive learning is then performed within the text space.

### Contrastive Learning & Loss Function
- Input the context's text-like embedding and the question's text embedding into the text encoder, extracting sentence-level representations using the CLS token.
- Trains the alignment using cosine similarity + NLL loss.
- Total Loss: $$\mathcal{L}_{total} = (1-\alpha-\beta)\mathcal{L}_{ASR} + \alpha\mathcal{L}_{MAE} + \beta\mathcal{L}_{NLL}$$, where $\alpha=\beta=\frac{1}{3}$.

### Training Strategy
- **Pre-training Stage**: Pre-trains Paraformer (ASR) on LibriSpeech 460h, and pre-trains BGE on clean text.
- **Joint Training**: Freezes BGE, jointly optimizing the ASR module and the contrastive loss.
- **Post-training**: Freezes ASR, fine-tuning BGE for a few epochs to adapt to the text-like representations.

## Key Experimental Results

### Datasets
Four datasets: Spoken-SQuAD, LibriSQA, SLUE-SQA-5 (real recordings), DRCD (Chinese)

### Main Results (Spoken-SQuAD*)

| Model | Paradigm | WER↓ | Q→C R@1 | Q→C R@10 |
|------|------|------|---------|----------|
| CLAP | E2E | - | 2.93 | 14.84 |
| Whisper+BGE | Pipeline | 19.39 | 69.93 | 90.53 |
| **CLSR** | **E2E** | **15.14** | **70.03** | **90.68** |

- CLSR significantly outperforms CLAP (R@1 improved from ~3% to ~70%) and SpeechDPR across all four datasets.
- It performs comparably to or better than the Whisper+BGE pipeline method, while achieving a lower WER (15.14 vs 19.39).
- On LibriSQA, CLSR achieves R@1=85.04%, close to the performance of text-only BGE (86.91%).

### Ablation Study Key Points
- **Removing the VQ adaptor**: R@10 plummets from ~86% to ~44%, validating the core value of text-like representations.
- **Removing the Sampler**: WER increases from 15.01 to 16.18, and retrieval recall also drops.
- **Pre-training ASR and BGE** both significantly contribute to the final performance.
- A WER of ~16.75% serves as a threshold; beyond this value, retrieval performance drops sharply.

### Practical Performance on Long-form Speech SQA
Tested on Spoken Wikipedia (average audio duration of 30 minutes):
- Without CLSR: EM=18.00, F1=23.55, time consumed 7,935s
- **With CLSR: EM=27.60, F1=35.10, time consumed 783s (10× acceleration)**

## Highlights & Insights
1. **First to introduce RAG into the SQA domain**, providing a systematic framework for long-form spoken question answering.
2. The **text-like representation bridging strategy** cleverly evades the difficulty of direct speech-text alignment, leveraging mature text contrastive learning models to achieve high-quality cross-modal retrieval.
3. **No large-scale speech-text pre-training is required**; joint training utilizing only the task data achieves comparable performance to pipeline methods.
4. The straight-through estimation design of the VQ adaptor guarantees the feasibility of end-to-end training.

## Limitations & Future Work
- It is only tested on TTS-synthesized speech and limited real-world recordings, leaving its robustness to noisy environments and multi-speaker scenarios unknown.
- Currently, long audio is fixedly segmented into 40-second chunks, lacking an adaptive semantic segmentation strategy.
- BGE is frozen during joint training, resulting in limited gains from post-training; better unfreezing strategies can be explored.
- Contrastive evaluation with newer speech foundation models (e.g., Whisper-v3, SeamlessM4T) is missing.
- The long-audio experiments only utilize 500 samples, which is a relatively small scale.

## Related Work & Insights

| Method | Characteristics | Limitations |
|------|------|------|
| CLAP | Audio-text contrastive learning | Suitable for sound effect matching, unsuitable for speech content retrieval |
| SpeechDPR | Speech retrieval without text training | Data scarcity leads to poor performance (R@20 of only 19.94) |
| Whisper+BGE | ASR cascaded text retrieval | Relies on ASR quality, error propagation, weak Chinese capability |
| **CLSR** | **VQ bridging + joint training** | **E2E reaches pipeline levels, optimizing both WER and retrieval simultaneously** |

### Inspirations & Connections
- The concept of text-like representation can be transferred to other cross-modal retrieval tasks (e.g., in video-text retrieval, translating video representations into the text space first).
- The combination of CIF + VQ can serve as a universal "speech-to-discrete token" front-end, replacing traditional discretization schemes.
- The long-audio RAG framework can be integrated with streaming ASR to achieve real-time meeting QA systems.

## Rating
- Novelty: ⭐⭐⭐⭐ (First to introduce RAG into SQA, text-like representation bridging is novel)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Four datasets + ablation + long audio validation, though long-audio scale is relatively small)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, complete equations, helpful diagrams)
- Value: ⭐⭐⭐⭐ (Provides a practical framework for long-form spoken QA, with promising 10× inference acceleration)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] VoxMind: An End-to-End Agentic Spoken Dialogue System](../../ACL2026/audio_speech/voxmind_an_end-to-end_agentic_spoken_dialogue_system.md)
- [\[ACL 2025\] Contextual Biasing with the Knowledgeable External Language Model for End-to-End Speech Recognition](../../ACL2025/audio_speech/contextual_biasing_with_the_knowledgeable_external_language_model_for_end-to-end.md)
- [\[ICML 2025\] Long-Form Speech Generation with Spoken Language Models](../../ICML2025/audio_speech/long-form_speech_generation_with_spoken_language_models.md)
- [\[ACL 2026\] VAPO: End-to-end Slide-Enhanced Speech Recognition with Omni-modal Large Language Models](../../ACL2026/audio_speech/vapo_end-to-end_slide-enhanced_speech_recognition_with_omni-modal_large_language.md)
- [\[ACL 2025\] OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation](../../ACL2025/audio_speech/omniflatten_an_end-to-end_gpt_model_for_seamless_voice_conversation.md)

</div>

<!-- RELATED:END -->
