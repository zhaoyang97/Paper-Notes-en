---
title: >-
  [Paper Note] OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation
description: >-
  [ACL 2025][Audio & Speech][Full-duplex conversation] OmniFlatten is proposed—an end-to-end full-duplex voice conversation model based on Qwen2-0.5B. By employing a three-stage progressive post-training scheme (modality alignment $\rightarrow$ half-duplex $\rightarrow$ full-duplex conversation learning) and a unified flatten operation, it achieves low-latency natural full-duplex voice interaction without modifying the GPT architecture, reducing the turn-taking response time to…
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Full-duplex conversation"
  - "End-to-end voice model"
  - "GPT architecture"
  - "chunking"
  - "Multi-stage training"
  - "Modality alignment"
  - "turn-taking"
date: 2026-05-08
content_hash: 93aa0f4d2cbbd71b
---

# OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation

**Conference**: ACL 2025  
**arXiv**: [2410.17799](https://arxiv.org/abs/2410.17799)  
**Code**: Unreleased  
**Demo**: [https://omniflatten.github.io/](https://omniflatten.github.io/)  
**Authors**: Qinglin Zhang, Luyao Cheng, Chong Deng, Qian Chen, Wen Wang, etc.  
**Institutions**: Tongyi Lab, Alibaba Group  
**Area**: Voice Conversation / Multimodal  
**Keywords**: Full-duplex conversation, End-to-end voice model, GPT architecture, chunking, Multi-stage training, Modality alignment, turn-taking

## TL;DR

OmniFlatten is proposed—an end-to-end full-duplex voice conversation model based on Qwen2-0.5B. By employing a three-stage progressive post-training scheme (modality alignment $\rightarrow$ half-duplex $\rightarrow$ full-duplex conversation learning) and a unified flatten operation, it achieves low-latency natural full-duplex voice interaction without modifying the GPT architecture, reducing the turn-taking response time to only $193\text{ ms}$, which significantly outperforms Moshi's $553\text{ ms}$.

## Background & Motivation

**Background**:
   - **Cascaded systems** (such as Qwen-audio): These systems connect LLM conversation modules with external ASR/TTS. They rely on half-duplex interaction, which leads to high latency.
   - **End-to-end systems** (such as SpeechGPT, LLaMA-Omni, GLM-4-Voice): These directly model speech-to-speech, but most are turn-based and do not support full-duplex.
   - **Full-duplex systems**: Moshi achieves full-duplex through multi-stream parallel modeling, but requires complex acoustic delay and inner monologue designs, which are not natively supported by standard GPT models.

**Key Challenge**:
   - Full-duplex conversation requires simultaneously processing **speaking, listening, and thinking**, which involves complex human-computer interaction behaviors such as interruption, backchanneling, and overlapping speech.
   - How to achieve full-duplex capability without modifying the native GPT architecture.
   - There is a lack of large-scale training data for full-duplex conversations.

**Key Innovations**: The "flatten" operation is proposed, which flattens multi-stream data (user speech/text + assistant speech/text) into a single unified sequence. This allows standard GPT models to handle full-duplex conversations without structure modifications.

## Method

### Overall Architecture

OmniFlatten consists of the following core components:
- **Audio Tokenizer**: Employs CosyVoice's speech tokenizer (single codebook, $4096$ codes) to convert speech into a discrete token sequence.
- **Base Model**: Qwen2-0.5B (a text LLM, small in scale but highly cost-effective).
- **Audio Detokenizer**: OT-CFM + HifiGAN vocoder, which converts speech tokens back into audio.

### Three-Stage Training Scheme

#### Stage 1: Modality Alignment

- **Goal**: Enable the text LLM to learn the speech-to-text correspondence, obtaining ASR and TTS capabilities.
- **Training Data**: Approximately $100,000$ hours of audio ($30\%$ open-source + $70\%$ proprietary), including Aishell-3, LibriTTS, Wenetspeech, etc.
- **Training Format**:
    - ASR: `[ASR][SOS]speech tokens[EOS][SOT]text tokens[EOT]`
    - TTS: `[TTS][SOT]text tokens[EOT][SOS]speech tokens[EOS]`
- **Sequence Length**: $1024$ tokens

#### Stage 2: Half-duplex Dialogue Training

- **Goal**: Learn basic multi-turn conversation capabilities (where the user and assistant take turns speaking without overlap).
- **Four-stream Data**: User speech, user text, assistant text, assistant speech.
- **Flatten Operation**: Flattens the four-stream data into a single sequence based on the actual timeline of speech.
- **Curriculum Learning**: Master simple turn-based conversations first, before moving on to complex full-duplex learning.

#### Stage 3: Full-duplex Dialogue Training

Progressive training is conducted in two steps:

**Step 1 — Three-stream Training**:
- Remove the user text stream, retaining only user speech + assistant text + assistant speech.
- Introduce **chunking**: Segment speech and text sequences into fixed-size short chunks (speech chunk = $10$ tokens, text chunk = $2$ tokens).
- Alternate chunks: Input speech $\rightarrow$ Output text $\rightarrow$ Output speech.
- Fill silent regions with `silent_text_token` and `silent_speech_token`.

**Step 2 — Two-stream Training**:
- Further remove the assistant text stream, retaining only user speech $\rightarrow$ assistant speech.
- Eliminate dependency on intermediate text representations, achieving pure speech-to-speech full-duplex generation.

### Data Synthesis Pipeline

Due to the lack of real full-duplex conversation data, a comprehensive synthesis pipeline was designed:

1. **Text Dialogue Collection**: Collect approximately $390,000$ multi-turn conversation texts from Alpaca, Moss, BelleCN, and UltraChat.
2. **Speech Synthesis**: Use CosyVoice to convert text to speech (user voices sampled from LibriSpeech and 3D-Speaker, assistant voice is fixed).
3. **Interaction Simulation**: Simulate three key scenarios—normal conversation transition, user interruption, and assistant waiting.
4. **Noise Addition**: Sample background noise from the MUSAN dataset and mix it into the user channel with $15\text{--}30\text{ dB}$ SNR.
5. **Total Volume**: $2000$ hours of multi-channel voice conversation data.

### Training Details

- Maximum sequence length during the dialogue learning stage: $8192$ tokens.
- Apply loss masking on the user channel (masking loss computation for user inputs).
- AdamW optimizer, weight decay = $0.1$, maximum learning rate of $2\text{e-}5$.
- $5$ epochs, with each batch containing $100\text{M}$ tokens.

## Experiments

### ASR Performance Evaluation (Table 1)

ASR performance after the Modality Alignment stage:

| Model | Librispeech clean | Librispeech other | Wenetspeech meeting | Wenetspeech net |
|------|------------------|------------------|--------------------|-----------------| 
| Whisper-Small | 3.13 (WER) | 7.37 | 25.62 (CER) | 16.66 |
| Whisper-Large | 1.82 | 3.50 | 18.87 | 10.48 |
| VITA | 8.14 | 18.4 | 12.15 | 16.53 |
| OmniFlatten | 7.91 | 19.21 | 26.1 | 19.0 |

OmniFlatten's ASR performance is comparable to VITA. Although it is not as strong as specialized Whisper models, it demonstrates that the modality alignment stage successfully established the correspondence between speech and text.

### TTS Performance Evaluation (Table 2)

| Model | LibriTTS (WER↓) | AIShell-3 (CER↓) |
|------|-----------------|-------------------|
| Original | 2.66 | 2.52 |
| ChatTTS | 8.32 | 3.87 |
| CosyVoice | 2.89 | 3.82 |
| OmniFlatten | 4.51 | 4.46 |

The TTS quality is reasonable, falling between CosyVoice and ChatTTS.

### Full-duplex Dialogue Quality Evaluation (Table 3, LLM Score 1–10)

| Model | Parameters | En Text | En Speech | Zh Text | Zh Speech |
|------|------|---------|-----------|---------|-----------|
| Qwen2-0.5B-Instruct | 0.5B | 6.75 | - | 6.98 | - |
| Qwen2-7B-Instruct | 7B | 8.37 | - | 8.09 | - |
| LLaMA-Omni | 8B | 6.01 | 5.50 | 4.17 | 3.89 |
| Moshi | 7B | 3.92 | 3.46 | - | - |
| GLM-Voice | 9B | 6.97 | 6.40 | 7.02 | 6.69 |
| OmniFlatten Direct 3-stream | 0.5B | 2.99 | 2.59 | 4.94 | 3.95 |
| OmniFlatten 3-stream w/o Half-duplex | 0.5B | 3.89 | 3.54 | 5.25 | 4.76 |
| OmniFlatten 3-stream Full | 0.5B | **4.88** | **3.92** | **5.60** | **5.15** |
| OmniFlatten 2-stream Full | 0.5B | - | 2.19 | - | 3.06 |
| Ground Truth | - | 7.65 | - | 6.83 | - |

**Key Findings**:
- Every training stage contributes to performance improvement: Direct 3-stream ($2.99$) $\rightarrow$ adding Modality Alignment ($3.89$) $\rightarrow$ adding Half-duplex ($4.88$).
- OmniFlatten outperforms Moshi in English ($4.88$ vs $3.92$) and outperforms LLaMA-Omni in Chinese.
- The 2-stream model (speech output only) suffers a significant performance drop, proving that intermediate text representations remain crucial.

### Turn-taking Performance (Table 4)

| Model | Asst Turn-taking Acc@1/5/10/25 | Asst Response Time | User Turn-taking Acc@1/5/10/25 | User Response Time |
|------|-------------------------------|-------------|-------------------------------|-------------|
| Moshi | 2.9/18.8/38.5/55.1% | 553ms | 0.0/6.2/14.8/45.7% | 753ms |
| OmniFlatten | **20.6/53.6/66.3/71.7%** | **193ms** | **10.9/30.9/41.8/51.8%** | **287ms** |

OmniFlatten comprehensively outperforms Moshi in turn-taking:
- Assistant response time: $193\text{ ms}$ vs $553\text{ ms}$ (**$2.9$ times faster**)
- User interruption response time: $287\text{ ms}$ vs $753\text{ ms}$ (**$2.6$ times faster**)
- Acc@1: $20.6\%$ vs $2.9\%$ (**$7$ times higher**)

## Highlights & Insights

1. **Unity of the Flatten Operation**: By flattening multi-stream data into a single sequence, it unifies training methods across different modalities and tasks, avoiding Moshi's complex parallel design.
2. **Effectiveness of Progressive Training**: The ablation study clearly demonstrates the marginal contribution of each stage: Modality Alignment ($+0.9$) $\rightarrow$ Half-duplex ($+1.0$), validating the value of curriculum learning in multimodal dialogs.
3. **Full-duplex Capabilities in Tiny Models**: Achieving full-duplex conversations with only 0.5B parameters proves that full-duplex capabilities can be obtained through training strategies rather than scale alone.
4. **Extremely Low Turn-taking Latency**: The assistant response time of $193\text{ ms}$ is close to the natural response latency of human conversation (approximately $200\text{ ms}$) and is far superior to Moshi.
5. **No Modifications to standard GPT Architecture**: The method is completely compatible with standard Transformer decoders and can be directly applied to any pretrained LLM.

## Limitations & Future Work

1. **Small Base Model**: The capacity of Qwen2-0.5B limits dialogue quality (text score is only $4.88$ vs $7.65$ for Ground Truth). The paper acknowledges that scaling up the model size could significantly boost performance.
2. **Limited Synthetic Data Quality**: The full-duplex data is generated using TTS synthesis + rule-based simulations, which fails to capture the rich interactive patterns of real conversations (such as variations in backchannels and tone).
3. **Low User Turn-taking Success Rate**: Even within $25$ tokens, OmniFlatten's user interruption recognition accuracy is only $51.8\%$, which is far from practical.
4. **Severe Performance Decay of 2-stream Model**: Removing the intermediate text representation leads to a major performance drop ($5.60 \rightarrow 3.06$), indicating that pure speech-to-speech full-duplex remains an unsolved challenge.
5. **Lack of Backchannel Modeling**: The current model does not support more complex interactive behaviors (such as user-side "mhm" vocalizations or quick verification feedback from the assistant).
6. **Evaluation Relies on LLM Scoring**: Dialogue quality assessment heavily relies on Qwen-max LLM scoring; its correlation with human evaluation has not been validated.

## Related Work & Insights

- **Half-duplex Voice Conversation**: SpeechGPT (Zhang et al., 2023), Mini-Omni (Xie & Wu, 2024a), LLaMA-Omni (Fang et al., 2024), GLM-4-Voice (Zeng et al., 2024)
- **Full-duplex Voice Conversation**: Moshi (Défossez et al., 2024) multi-stream parallel, dGSLM (Nguyen et al., 2023), VITA (Fu et al., 2024), SyncLM (Veluri et al., 2024) chunk alternating
- **Speech Tokenizer**: CosyVoice (Du et al., 2024a), HifiGAN (Kong et al., 2020)
- **Multimodal LLM**: Qwen-Audio (Chu et al., 2024), SALMONN (Tang et al., 2024)
- **Full-duplex Exploration**: LSLM (Ma et al., 2024) listening-while-speaking in real-time

## Rating

⭐⭐⭐⭐ — The method is simple and elegant (no architecture changes, progressive training, and unified flattening), with excellent turn-taking latency metrics. However, the small size of the base model limits the dialog quality, and synthetic data limits real-world scenarios. Nevertheless, this remains a valuable exploratory work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Distilling an End-to-End Voice Assistant Without Instruction Training Data](distilling_an_end-to-end_voice_assistant_without_instruction_training_data.md)
- [\[ACL 2025\] Contextual Biasing with the Knowledgeable External Language Model for End-to-End Speech Recognition](contextual_biasing_with_the_knowledgeable_external_language_model_for_end-to-end.md)
- [\[ACL 2025\] DNCASR: End-to-End Training for Speaker-Attributed ASR](dncasr_end-to-end_training_for_speaker-attributed_asr.md)
- [\[NeurIPS 2025\] E2E-VGuard: Adversarial Prevention for Production LLM-based End-To-End Speech Synthesis](../../NeurIPS2025/audio_speech/e2e-vguard_adversarial_prevention_for_production_llm-based_end-to-end_speech_syn.md)
- [\[AAAI 2026\] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering](../../AAAI2026/audio_speech/end-to-end_contrastive_language-speech_pretraining_model_for_long-form_spoken_qu.md)

</div>

<!-- RELATED:END -->
