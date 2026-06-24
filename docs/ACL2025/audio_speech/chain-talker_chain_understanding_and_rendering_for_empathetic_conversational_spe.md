---
title: >-
  [Paper Note] Chain-Talker: Chain Understanding and Rendering for Empathetic Conversational Speech Synthesis
description: >-
  [ACL 2025][Audio & Speech][Conversational Speech Synthesis] This paper proposes Chain-Talker, which achieves interpretable empathetic conversational speech synthesis through a three-stage chain modeling (emotion understanding $\rightarrow$ semantic understanding $\rightarrow$ empathetic rendering), and develops CSS-EmCap, an automatic annotation pipeline to generate emotional captions for conversational speech.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Conversational Speech Synthesis"
  - "Empathetic Speech"
  - "Chain Modeling"
  - "Emotion Captioning"
  - "Semantic Encoding"
date: 2026-05-08
content_hash: 66cdae4451bbfcc9
---

# Chain-Talker: Chain Understanding and Rendering for Empathetic Conversational Speech Synthesis

**Conference**: ACL 2025  
**arXiv**: [2505.12597](https://arxiv.org/abs/2505.12597)  
**Code**: [https://github.com/AI-S2-Lab/Chain-Talker](https://github.com/AI-S2-Lab/Chain-Talker)  
**Area**: Speech  
**Keywords**: Conversational Speech Synthesis, Empathetic Speech, Chain Modeling, Emotion Captioning, Semantic Encoding

## TL;DR

This paper proposes Chain-Talker, which achieves interpretable empathetic conversational speech synthesis through a three-stage chain modeling (emotion understanding $\rightarrow$ semantic understanding $\rightarrow$ empathetic rendering), and develops CSS-EmCap, an automatic annotation pipeline to generate emotional captions for conversational speech.

## Background & Motivation

**Background**: Conversational Speech Synthesis (CSS) aims to generate speech matched with the conversational context, based on the emotional and stylistic context of user-agent interactions. In recent years, GPT-based CSS models (such as GPT-Talker) have significantly improved naturalness and expressiveness by directly predicting speech token sequences.

**Limitations of Prior Work**: Current generative CSS models suffer from two interpretability issues: (1) speech generation lacks a deep understanding of conversational emotion, making it difficult to achieve genuine empathy; (2) general discrete speech codes contain excessive redundant information, mixing semantic and acoustic information, which limits expressiveness.

**Key Challenge**: The end-to-end approach of directly predicting speech tokens from conversational context lacks interpretability: the model neither explicitly understands emotional shifts nor efficiently disentangles semantic from acoustic information.

**Goal**: To enable CSS systems to progressively understand dialogue emotions, extract semantic information, and eventually generate empathetic speech much like humans, while also automatically generating high-quality emotional caption annotations for conversational speech.

**Key Insight**: Drawing inspiration from the human Chain-of-Thought process, CSS is decomposed into three stages: first understanding emotion, then understanding semantics, and finally rendering empathetically. Concurrently, an automatic context-aware emotional captioning pipeline is constructed using LLMs.

**Core Idea**: To mimic the three-stage chain modeling of human cognition—first perceiving emotion, then encoding semantics, and finally rendering speech—to achieve interpretable empathetic conversational speech synthesis.

## Method

### Overall Architecture

Chain-Talker consists of two main components: **EmGPT** and **Synthesizer**. EmGPT is responsible for emotion and semantic understanding based on an autoregressive GPT architecture, while the Synthesizer handles empathetic speech rendering based on an OT-CFM (Optimal Transport Conditional Flow Matching) model.

The input sequence is defined as $\mathcal{Q} = (\langle BOS \rangle, \mathcal{H}, \mathcal{C}, \langle EOS \rangle)$, where $\mathcal{H}$ denotes the dialogue history and $\mathcal{C}$ represents the current utterance to be synthesized.

### Key Designs

**Module 1: Unified Context Tokenization**

- **Function**: Codify multi-modal dialogue information into a unified sequence.
- **Mechanism**: Alternately concatenate user and agent utterances in the sequence of speaker information, audio, textual content, and emotional captions. Text is encoded into $T_n^t$ using BPE, emotional captions are encoded into $T_n^d$, speaker embeddings $T_n^p$ are extracted using a pre-trained speaker verification model, and speech is encoded into $T_n^a$ via a supervised ASR model (with VQ).
- **Design Motivation**: To enable the model to understand the context prior to predicting emotions and subsequently generating the corresponding speech, unifying multi-modal information within a singular sequence space.

**Module 2: Emotion Understanding**

- **Function**: Predict the emotional caption tokens of the current utterance based on the dialogue context.
- **Mechanism**: Utilize EmGPT to autoregressively predict the emotional caption $T_N^d$: $p(T_{N,:}^d | \Re_{1 \to N-1}, T_{N,:}^p, T_{N,:}^t; \Theta) = \prod_{j=0}^{D} p(T_{N,j}^d | T_{N,<j}^d, \Re_{1 \to N-1}, T_{N,:}^p, T_{N,:}^t; \Theta)$
- **Design Motivation**: To explicitly comprehend emotional dynamics within dialogues, providing emotional guidance for subsequent speech generation.

**Module 3: Semantic Understanding**

- **Function**: Predict purely semantic speech encodings on top of the understood emotions.
- **Mechanism**: EmGPT utilizes the predicted emotional caption $T_N^d$ along with context information to secure semantic encodings $T_N^a$: $p(T_{N,:}^a | \Re_{1 \to N-1}, T_N^p, T_N^t, T_N^d; \Theta)$
- **Design Motivation**: To employ purely semantic encodings produced by a supervised ASR model, avoiding acoustic redundancy inherent in general discrete codes.

**Module 4: Empathetic Rendering**

- **Function**: Synthesize the final empathetic speech leveraging the emotional captions and semantic encodings.
- **Mechanism**: Adopt OT-CFM as the backbone to predict Mel-spectrograms and utilize HiFi-GAN to synthesize waveforms. The OT-CFM leverages emotional captions $U_N^d$, speaker information $U_{agent}^p$, semantic encodings $T_N^a$, and masked Mel-spectrograms $U_{agent}^m$ simultaneously to predict vector fields: $\frac{d\phi_t(X)}{dt} = \nu_t(\phi_t(X), t | U_{agent}^p, U_N^d, T_N^a, U_{agent}^m)$
- **Design Motivation**: To utilize emotional captions to guide emotion and style rendering during the decoding stage, rather than directly decoding from raw speech tokens.

### Loss & Training

**Training Loss**:
- EmGPT training involves two losses: $\mathcal{L}_{caption}$ (cross-entropy loss of the emotional caption tokens) and $\mathcal{L}_{speech}$ (cross-entropy loss of the semantic encodings).
- The Synthesizer employs the OT-CFM loss: $\mathcal{L}_{OT\text{-}CFM} = \mathbb{E}_{t,X_0,X_1}[\|\omega_t(\phi_t^{OT}(X_0,X_1)|X_1) - \nu_t(\phi_t^{OT}(X_0,X_1)|\theta)\|]$

**Multi-Stage Training**:
- **First Stage**: Train on large-scale single-sentence TTS data (based on CosyVoice-300M-25Hz, with approximately 170,000 hours of speech data).
- **Second Stage**: Fine-tune on dialogue data, learning to infer emotional captions from the dialogue context and predict semantic encodings.
- The Synthesizer can be trained independently in single-sentence mode.

**CSS-EmCap Pipeline**: 
- Multi-level attribute extraction: sentence-level style factors (gender, pitch, energy, speaking rate) + dialogue-level emotion classification.
- Two-step generation: generate base captions based on context and attributes, then expand and enrich them via synonym substitution and variation in emotional intensity.

## Key Experimental Results

### Main Results

Evaluated on three datasets: NCSSD, DailyTalk, and MultiDialog (totaling approximately 384 hours).

| Method | DMOS-N ↑ | DMOS-E ↑ | ACCm ↑ | DDTW ↓ | SSIM ↑ |
|------|---------|---------|--------|--------|--------|
| Ground Truth | 4.467 | 4.571 | - | - | - |
| CCATTS | 3.423 | 3.469 | 0.462 | 67.851 | 0.765 |
| GPT-Talker | 3.962 | 3.913 | 0.562 | 44.625 | 0.814 |
| GPT-Talker_c | 4.045 | 4.102 | 0.589 | 40.374 | 0.829 |
| **Chain-Talker** | **4.147** | **4.239** | **0.612** | **38.784** | **0.862** |

CSS-EmCap annotation quality:

| Method | DMOS-C ↑ | SIM_R ↑ | SIM_G ↑ | DIS-1 ↑ | DIS-2 ↑ |
|------|---------|---------|---------|---------|---------|
| Qwen2-Audio | 4.212 | 0.431 | 0.534 | 0.086 | 0.174 |
| SECap | 4.268 | 0.475 | 0.617 | 0.081 | 0.186 |
| **CSS-EmCap** | **4.462** | **0.568** | **0.694** | **0.106** | **0.296** |

### Ablation Study

| Configuration | DMOS-N ↑ | DMOS-E ↑ | ACCm ↑ | DDTW ↓ | SSIM ↑ |
|------|---------|---------|--------|--------|--------|
| Chain-Talker | 4.147 | 4.239 | 0.612 | 38.784 | 0.862 |
| w/o context | 3.982 | 3.984 | 0.564 | 43.589 | 0.847 |
| w/o captions | 4.037 | 4.084 | 0.571 | 43.479 | 0.836 |
| w/o $\mathcal{L}^{caption}$ | 3.947 | 3.956 | 0.568 | 45.764 | 0.829 |
| w/o First-Stage | 3.756 | 3.789 | 0.517 | 52.640 | 0.793 |

### Key Findings

1. Chain-Talker outperforms the strongest baseline by 0.102 in naturalness MOS (DMOS-N) and by 0.112 in expressiveness MOS (DMOS-E).
2. Removing dialogue history (w/o context) leads to a descent of 0.255 in DMOS-E, verifying the importance of context modeling.
3. Removing the emotional caption loss (w/o $\mathcal{L}^{caption}$) causes a decline of 0.2 in DMOS-N and 0.283 in DMOS-E.
4. The absence of first-stage pre-training leads to a drastic degradation across all metrics, indicating the necessity of large-scale pre-training.
5. Optimal performance is achieved when the dialogue history length is $N$=3, peaking at around 200 epochs.

## Highlights & Insights

- The introduction of the **chain modeling paradigm** is quite elegant, decomposing the complex CSS task into an interpretable three-step cognitive chain that aligns with the human intuition of "perceiving emotion first, then understanding content, and finally expressing".
- The design of the **CSS-EmCap pipeline** is highly instructive: first extracting structured attributes, then prompting an LLM to generate descriptive natural language captions, and finally validating them to close the loop.
- Employing **supervised semantic encodings** as a replacement for general discrete codes is a critical design choice—eliminating acoustic redundancies makes semantic understanding much cleaner.
- Emotional captions (natural language) outperform conventional emotion labels (DMOS-C 4.462 vs. GT's 4.327), demonstrating that natural language is more expressive in controlling conversational styles.

## Limitations & Future Work

1. **Inference Latency**: The average response latency of 2.5 seconds fails to meet-real time interaction requirements; streaming inference is a vital future direction.
2. **Data Scale**: The dialogue dataset is restricted to 384 hours and predominantly features young speakers, failing to adequately cover styles for children and elderly individuals.
3. **Security Risks**: Zero-shot voice cloning capabilities could potentially be abused for voice spoofing.
4. The generation of emotional captions relies on the Gemini API, increasing dependence on external LLMs.

## Related Work & Insights

- **GPT-Talker** pioneered the GPT-based CSS paradigm, but lacked explicit emotional understanding.
- **CosyVoice** provides the foundational architecture for supervised semantic tokens and the OT-CFM synthesizer.
- The successful application of **Chain-of-Thought (CoT)** in conversational tasks inspired the chain modeling paradigm in CSS.
- The audio comprehension capability of **Qwen2-Audio/SECap** offers a comparative benchmark for emotional caption generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (First implementation of chain modeling in CSS with clear logic)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Three datasets + comprehensive ablation studies + visualization analysis)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear structure and rich illustrations/tables)
- **Value**: ⭐⭐⭐⭐ (Both the CSS-EmCap pipeline and the chain-based design hold solid reference value for the community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing](../../NeurIPS2025/audio_speech/thinksound_chain-of-thought_reasoning_in_multimodal_large_language_models_for_au.md)
- [\[ACL 2025\] Autoregressive Speech Synthesis without Vector Quantization](autoregressive_speech_synthesis_without_vq.md)
- [\[ICLR 2026\] PrismAudio: Decomposed Chain-of-Thought and Multi-dimensional Rewards for Video-to-Audio Generation](../../ICLR2026/audio_speech/prismaudio_decomposed_chain-of-thought_and_multi-dimensional_rewards_for_video-t.md)
- [\[ACL 2025\] Does Your Voice Assistant Remember? Analyzing Conversational Context Recall and Utilization in Voice Interaction Models](does_your_voice_assistant_remember_analyzing_conversational_context_recall_and_u.md)
- [\[ACL 2025\] SpeechIQ: Speech-Agentic Intelligence Quotient Across Cognitive Levels in Voice Understanding by Large Language Models](speechiq_speechagentic_intelligence_quotient_across_cognitive.md)

</div>

<!-- RELATED:END -->
