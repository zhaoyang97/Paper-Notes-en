---
title: >-
  [Paper Note] When Large Language Models Meet Speech: A Survey on Integration Approaches
description: >-
  [ACL 2025][LLM (Other)][speech-LLM integration] A systematic survey of integration approaches between speech and Large Language Models (LLMs), categorizing existing works into three major paradigms: text-based, latent-representation-based, and audio-token-based integrations. It covers application scenarios including ASR, S2TT, S2ST, and TTS, and provides comparisons of the strengths/weaknesses of each approach alongside future challenges.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "speech-LLM integration"
  - "text-based integration"
  - "latent-representation"
  - "audio tokens"
  - "speech language model"
  - "multimodal LLM"
date: 2026-05-08
content_hash: 1ef7c853fefc957a
---

# When Large Language Models Meet Speech: A Survey on Integration Approaches

**Conference**: ACL 2025  
**arXiv**: [2502.19548](https://arxiv.org/abs/2502.19548)  
**Code**: None (Survey paper)  
**Area**: LLM/NLP  
**Keywords**: speech-LLM integration, text-based integration, latent-representation, audio tokens, speech language model, multimodal LLM

## TL;DR
A systematic survey of integration approaches between speech and Large Language Models (LLMs), categorizing existing works into three major paradigms: text-based, latent-representation-based, and audio-token-based integrations. It covers application scenarios including ASR, S2TT, S2ST, and TTS, and provides comparisons of the strengths/weaknesses of each approach alongside future challenges.

## Background & Motivation

**Background**: LLMs have achieved tremendous success in textual tasks, prompting researchers to explore extending their capabilities to the speech modality. This has led to a surge of Speech-LLM integration works (e.g., AudioGPT, SALMONN, Qwen-Audio, Moshi).

**Limitations of Prior Work**: Existing surveys cover speech language models (Peng et al., 2024) and audio language models (Latif et al., 2023), but lack a systematic taxonomy dedicated specifically to **Speech-LLM integration methods**, making it difficult for researchers to quickly grasp the trade-offs of different integration paradigms.

**Key Challenge**: Speech is a continuous signal, whereas LLMs inherently process discrete tokens. Bridging this modality gap is the core challenge of all methods, and different integration strategies strike different trade-offs among accuracy, latency, and interpretability.

**Goal**: To provide a clear taxonomic framework to help researchers understand the mechanisms, application scenarios, and pros/cons of three integration paradigms (text-based, latent-representation-based, and audio-token-based).

**Key Insight**: Organizing literature based on "integration methods" rather than "application tasks", and comparing the performance of various approaches across tasks like ASR, S2TT, S2ST, and TTS under a unified framework.

**Core Idea**: The tokenization method significantly impacts LLM performance. Speech-LLM integration should not be limited to discrete tokenization; all three paradigms have their own applicable scenarios and development potential.

## Method

### Overall Architecture
Speech-LLM integration methods are categorized into three major classes: (a) text-based integration, where LLMs process text alongside ASR/TTS pipelines; (b) latent-representation-based integration, where continuous vectors from a speech encoder are fed directly into the LLM's embedding space; (c) audio-token-based integration, where speech is discretized into semantic/acoustic tokens as inputs/outputs for the LLM. Each class is further divided into specific strategies.

### Key Designs 1: Text-based Integration
- **Function**: Converts speech to text first before processing with an LLM, including three sub-approaches: cascaded integration, LLM rescoring, and generative error correction (GER).
- **Mechanism**: The cascaded approach is the most straightforward (ASR $\rightarrow$ LLM $\rightarrow$ TTS). Rescoring uses LLM language probability to re-rank the N-best hypothesis list via the equation: $Y^* = \arg\max_{Y_i} [(1-\lambda)\log p_{AM}(Y_i|X) + \lambda \log p_{LLM}(Y_i)]$. GER goes a step further, prompting the LLM to directly generate better transcription results based on the N-best hypotheses (the H2T paradigm).
- **Design Motivation**: Preserves the native text-processing capabilities of LLMs, offering simple implementation and strong interpretability. However, it suffers from inherent flaws like error propagation and loss of non-textual information (e.g., prosody, emotion). GER is able to surpass the oracle limit of the N-best hypotheses compared to rescoring, potentially producing output superior to all candidates. Lin et al. (2024) further introduced a Mixture-of-Experts (MoE) architecture to let different experts handle different types of generation errors.

### Key Designs 2: Latent-representation-based Integration
- **Function**: Extracts continuous latent representations using a speech encoder (e.g., HuBERT, Whisper encoder), which are fed directly into the LLM's embedding space via modality adaptation modules, bypassing the intermediate text step.
- **Mechanism**: The key lies in Modality Adaptation—solving the issue where the speech frame rate (50-100 fps) is significantly higher than the text token sequence length. Three main adaptation strategies are: ① convolutional downsampling (simple stacking or convolutional compression); ② CTC compression (removing blank frames or merging duplicate frames based on CTC predictions); ③ Q-Former (learnable fixed-length query vectors extracting information via cross-attention). Comparative experiments show Q-Former $>$ convolutional downsampling $>$ CTC compression.
- **Design Motivation**: Avoids information loss from intermediate text representations and achieves deeper cross-modal integration. However, this comes at the cost of losing interpretability and incurring higher training costs (requiring simultaneous handling of the encoder and LLM). Freezing parts of the modules and utilizing LoRA fine-tuning is the mainstream strategy to reduce costs. Wu et al. (2023) proposed a two-stage training scheme, training the encoder first before enabling PEFT, ensuring a more stable gradient flow.

### Key Designs 3: Audio-token-based Integration
- **Function**: Discretizes speech into token sequences to be processed uniformly with text tokens. This is sub-divided into semantic tokens (S3M + k-means clustering), acoustic tokens (neural audio codecs like EnCodec), and hybrid approaches combining both.
- **Mechanism**: Semantic tokens capture the linguistic content of speech, while acoustic tokens preserve high-fidelity audio quality. A typical paradigm is the two-stage architecture (e.g., AudioLM) that predicts semantic tokens first, followed by acoustic tokens. Moshi introduces a hierarchical autoregressive architecture containing Temporal Transformer and Depth Transformer to achieve full-duplex speech dialogue.
- **Design Motivation**: Audio tokens unify speech and text at the token level, allowing LLMs to naturally generate speech output (whereas latent-representation methods typically only support speech input, not output). However, semantic tokens lack audio quality and acoustic tokens lack rich semantics, so how to integrate both remains an open question.

## Key Experimental Results

### Main Results — ASR Task Performance Comparison (LibriSpeech test-clean / test-other WER↓)

| Model | Integration Method | test-clean | test-other |
|------|---------|-----------|-----------|
| Whisper-large-v2 | Non-LLM Baseline | 2.7 | 5.2 |
| HyPoradise | Text-based → GER | 1.8 | 3.7 |
| Seed-ASR | Latent-rep → Conv Downsampling | **1.5** | **2.8** |
| SALMONN | Latent-rep → Q-Former | 2.1 | 4.9 |
| Qwen2-Audio | Latent-rep → Other Adaptation | 1.6 | 3.6 |
| SpeechGPT-Gen | Audio-token → Semantic Token | 2.4 | — |

### S2TT Task Performance Comparison (CoVoST2 de-en / zh-en BLEU↑)

| Model | Integration Method | de→en | zh→en |
|------|---------|-------|-------|
| Whisper-large-v2 | Non-LLM Baseline | 36.3 | 18.0 |
| GenTranslate-V2 | Text-based → GER | **40.6** | **23.3** |
| LLaST | Latent-rep → Other Adaptation | 41.2 | 24.8 |
| AudioPaLM | Audio-token → Semantic + Acoustic | 43.4 | 25.5 |

### Pros & Cons of the Three Approaches (Survey Summary)

| Dimension | Text-based | Latent-rep-based | Audio-token-based |
|------|--------|---------|-----------|
| Integration Depth | Shallow | **Deepest** | Medium |
| Interpretability | **Best** | Worst | Medium |
| Speech Generation Capability | ✓ (Requires TTS) | ✗ (Typically cannot) | **✓ (Native)** |
| Real-time Processing Capability | Poor (Multi-step latency) | Good | Good |
| Implementation Difficulty | **Simplest** | Medium | Medium |

### Key Findings
- **Latent-representation-based methods achieve the best overall ASR performance**: Seed-ASR (WER 1.5/2.8) and Qwen2-Audio (WER 1.6/3.6) outperform text-based methods.
- **Text-based GER still holds unique value**: HyPoradise reduces WER from 2.7 to 1.8 without requiring an additional speech encoder.
- **Audio-token-based methods excel in generative tasks (TTS/S2ST)**: AudioPaLM achieves substantial gains over traditional cascaded systems in S2ST (de-en ASR-BLEU 37.2 vs 33.6).
- **Q-Former > Conv Downsampling > CTC Compression**: The choice of modality adaptation strategy significantly impacts the performance of latent-representation methods.
- **Direct comparisons remain difficult**: Backbone LLMs, training datasets, and protocols vary greatly across methods, lacking a unified benchmark.
- **Training strategies have a significant impact**: Pham et al. (2024) demonstrated that LoRA fine-tuning of LLMs significantly boosts performance; full fine-tuning of encoders performs best, though partial fine-tuning is more cost-effective.
- **Spirit-LM and Moshi represent the frontier of audio-token methods**: On Topic-StoryCloze, Moshi achieves 83.6 and Spirit-LM achieves 82.9, significantly outperforming the TWIST series, proving the effectiveness of fusing semantic and acoustic tokens.

## Highlights & Insights
- **Clear and practical tripartite taxonomy**: Organizing the literature around integration methods (rather than tasks) fills a gap in existing surveys, offering an efficient entry path for new researchers.
- **Insight on the "Integration Depth vs. Interpretability" trade-off**: Deeper integration yields better performance but lower controllability, which is a fundamental tension in the Speech-LLM field.
- **Fine-grained comparison of modality adaptation strategies** (Convolution/CTC/Q-Former) provides a quantitative basis for selecting specific methods.
- **Audio-token-based integration is the most promising unified paradigm**: It naturally handles both speech input and output, though its current maturity is lower than that of latent-representation methods.
- **Analysis of training strategies**: The survey systematically summarizes the trade-offs of freezing vs. fine-tuning different modules. LoRA-tuned LLM combined with a fully fine-tuned encoder represents the current best practice; two-stage training (training the encoder first before PEFT) enhances stability.
- **Systematic summary of six major challenges**: Covers loss of textual information, speech-text alignment, semantic-acoustic token integration, lack of fair comparison protocols, insufficient multilingual support, and real-time processing latency, providing a clear roadmap for future research.
- **Comprehensive background on speech representations**: Spanning from filter banks to self-supervised models (HuBERT, wav2vec 2.0) to semantic/acoustic tokens, providing readers with a complete knowledge framework of speech representation.
- **Real-time processing challenges**: The survey explicitly notes that the massive size of LLMs increases latency, whereas real-time models like Moshi and Llama-Omni demonstrate the potential of deep integration to reduce latency.

## Limitations & Future Work
- As a survey paper, it does not include original experimental verification. Quantitative comparisons are limited by the different experimental setups of the primary papers.
- Mainly covers English-centric works, with limited discussion on multilingual scenarios.
- In a rapidly evolving field, some of the latest developments may be omitted (e.g., the speech capabilities of GPT-4o are not fully discussed).
- Lack of fair comparative experiments under a unified benchmark (which the authors acknowledge as an important future direction).
- A broad standard is adopted for the definition of "LLMs" (>10B parameters), sometimes incorporating studies of smaller models, making the boundary somewhat fuzzy.
- Discussion on computational cost remains at a qualitative level, lacking numerical comparisons of actual FLOPs or latency.
- Privacy and security issues (such as audio deepfakes, speaker privacy protection, etc.) are not discussed in detail.
- The processing of speech emotion and paralinguistic information is not discussed deeply, which represents a crucial source of extra information in speech compared to text.

## Related Work & Insights
- **vs. Peng et al. (2024) Speech Language Model Survey**: Peng focuses on the modeling paradigm of the "speech language model" itself, whereas this work focuses on "how speech integrates with LLMs." The former looks at model architecture; the latter looks at integration interfaces, offering complementary views.
- **vs. Latif et al. (2023) Audio Language Model Survey**: Latif covers a broader audio modality (including music, environmental sound), while this work focuses specifically on the taxonomy of speech-LLM integration, offering deeper analysis.
- **vs. Ghosh et al. (2024a) Multimodal Language Model Survey**: Multimodal surveys cover the panoramic view of vision + speech + text, whereas this work provides a more fine-grained classification (three paradigms + sub-classes) in the speech dimension, making it more tailored for researchers in the speech domain.
- **Insights**: Researchers new to Speech-LLM can quickly position themselves based on task requirements: choose latent-representation-based integration if only speech understanding is needed; choose audio-token-based integration if speech generation is required; choose text-based integration if resources are limited or debuggability is a priority. The Temporal+Depth Transformer architecture of Moshi (Défossez et al., 2024) serves as a vital reference for real-time full-duplex conversations.

## Rating
- Novelty: ⭐⭐⭐⭐ The taxonomic framework is novel and practical, but as a survey, it lacks original methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ Aggregates a large number of quantitative results for horizontal comparisons, though restricted by non-unified experimental settings.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured, outstanding taxonomic charts, concise language, and comprehensive introduction of background knowledge.
- Value: ⭐⭐⭐⭐⭐ Fills a gap in surveys on Speech-LLM integration methods, offering high reference value for domain researchers, especially suitable for beginners to quickly grasp the full picture of the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Recent Advances in Speech Language Models: A Survey](recent_advances_in_speech_language_models_a_survey.md)
- [\[ACL 2025\] Large Language Models in Bioinformatics: A Survey](large_language_models_in_bioinformatics_a_survey.md)
- [\[ACL 2025\] Knowledge Boundary of Large Language Models: A Survey](knowledge_boundary_survey.md)
- [\[ACL 2025\] Dynamic Knowledge Integration for Evidence-Driven Counter-Argument Generation with Large Language Models](dynamic_knowledge_integration_for_evidence-driven_counter-argument_generation_wi.md)
- [\[ACL 2025\] InfiniSST: Simultaneous Translation of Unbounded Speech with Large Language Model](infinisst_simultaneous_translation_of_unbounded_speech_with_large_language_model.md)

</div>

<!-- RELATED:END -->
