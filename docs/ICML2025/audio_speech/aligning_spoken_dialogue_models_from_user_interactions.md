---
title: >-
  [Paper Note] Aligning Spoken Dialogue Models from User Interactions
description: >-
  [ICML2025][Audio & Speech][Full-duplex spoken dialogue] This work introduces the first comprehensive preference alignment framework designed for a full-duplex spoken dialogue model (Moshi). By automatically constructing content and temporal preference pairs from over 150k real user voice interactions and performing DPO-LN alignment exclusively on text tokens, this approach achieves an average QA improvement of 3.1% and a safety increase of 6.9%…
tags:
  - "ICML2025"
  - "Audio & Speech"
  - "Full-duplex spoken dialogue"
  - "preference alignment"
  - "DPO"
  - "AI feedback"
  - "temporal preference"
date: 2026-05-08
content_hash: fec620a2f5d06479
---

# Aligning Spoken Dialogue Models from User Interactions

**Conference**: ICML2025  
**arXiv**: [2506.21463](https://arxiv.org/abs/2506.21463)  
**Code**: None  
**Area**: Image Segmentation  
**Keywords**: Full-duplex spoken dialogue, preference alignment, DPO, AI feedback, temporal preference

## TL;DR
This work introduces the first comprehensive preference alignment framework designed for a full-duplex spoken dialogue model (Moshi). By automatically constructing content and temporal preference pairs from over 150k real user voice interactions and performing DPO-LN alignment exclusively on text tokens, this approach achieves an average QA improvement of 3.1% and a safety increase of 6.9%, with human evaluations confirming enhanced multi-turn dialogue quality.

## Background & Motivation

**Background**: Current preference alignment (RLHF/DPO) has achieved massive success on textual LLMs, but almost all prior work targets the text modality alone. In the speech dialogue domain, a few studies focus on aligning TTS audio quality or speech continuation, but a preference learning framework for real-time speech-to-speech dialogue models is still absent.

**Limitations of Prior Work**: Textual preference data is unsuitable for speech scenarios due to three fundamental mismatches: (1) **Style bias**—textual preferences favor long replies, listicles, or code, which are virtually unpronounceable; (2) **Lack of temporal signals**—text dialogues are split by turns, losing key temporal information such as interruptions, overlaps, and pauses; (3) **Insufficient turns**—existing preference data typically contains only 1-2 turns, whereas real voice dialogues contain a large number of potentially overlapping "turns".

**Key Challenge**: Full-duplex voice models (such as Moshi) allow both parties to speak, overlap, and interrupt at any time. This continuous interaction mode cannot be modeled using existing "turn-by-turn preference pairs". Extracting meaningful preference signals from unstructured, continuous voice dialogues presents a novel challenge.

**Goal**: (1) How to automatically construct preference datasets from large-scale raw voice dialogues? (2) How to adapt offline alignment methods like DPO to a multi-stream (text + audio $\times$ 2) full-duplex architecture? (3) How do content and temporal preferences respectively affect the model's behavior?

**Key Insight**: The authors leverage deployed real user dialogues as the data source. They employ an LLM Judge to automatically detect problematic responses and generate improved versions. The speech is then re-synthesized using TTS to construct large-scale preference pairs. This entire pipeline requires no human annotation and protects privacy by discarding the users' original audio.

**Core Idea**: Mining content and temporal preference pairs automatically from real full-duplex voice dialogues using AI feedback, and aligning the full-duplex spoken dialogue model via DPO-LN in the text token space.

## Method

### Overall Architecture
The input consists of a large volume of raw full-duplex voice dialogues between users and the base Moshi model. The pipeline consists of three steps: (1) Data construction—mining preference pairs from raw dialogues; (2) Synthesis—reconstructing user speech and improved model responses using TTS; (3) Alignment training—optimizing preferences on the multi-stream architecture using DPO-LN. The final output is the aligned Moshi-Aligned model.

### Key Designs

1. **Preference data construction pipeline**:

    - **Function**: Automatically extracting preference pairs from over 150k raw voice dialogues.
    - **Mechanism**: Whisper is first used to transcribe all dialogues, obtaining texts with timestamps. Mistral Large 2 is then employed as an LLM Judge to evaluate toxic or low-quality responses using a Likert-5 scale across multiple axes (helpfulness, safety, factual accuracy, tone, interruption, no response), after which the same LLM generates improved responses. The problematic responses act as "rejected" and the improved ones as "chosen" to form preference pairs.
    - **Design Motivation**: Human annotation of voice preferences is extremely costly and difficult to scale, whereas an AI feedback pipeline can process massive dialogue corpora automatically. Discarding the users' original audio while preserving only transcriptions and model audio addresses privacy requirements.

2. **Decoupling content preference vs. temporal preference**:

    - **Function**: Categorizing preference pairs into three types: Type-A (content differences only), Type-B (model interrupting the user), and Type-C (model remaining silent).
    - **Mechanism**: Type-A handles factual/safety/instruction-following issues detected by the LLM Judge to generate better text responses. Types B and C algorithmically detect temporal anomalies—for interruptions, the response is delayed until after the user finishes; for silence, an appropriate response is generated after the user's speech. For multi-turn dialogues with multiple problematic responses, only the first problematic response is extracted along with at most one additional sample.
    - **Design Motivation**: Content and timing constitute two fundamentally different preference dimensions in spoken dialogue. Separating them allows for an analysis of their individual contributions and helps identify the optimal mixing ratio.

3. **Multi-stream DPO-LN (text tokens only)**:

    - **Function**: Adapting standard DPO to Moshi's three-stream architecture (text + model audio + user audio).
    - **Mechanism**: Theoretically, policy probabilities should be jointly calculated over text and audio tokens as $\pi(y|x) = \pi(T^y|x,A^y,A'^y) \cdot \pi(A^y|x,T^y,A'^y)$. However, experiments revealed that joint optimization leads to training instability. Ultimately, the probability is computed exclusively on text tokens as $\pi^T(y|x) = \pi(T^y|x,A^y,A'^y)$, utilizing length-normalized DPO-LN as the training objective.
    - **Design Motivation**: The synthesized preferred responses do not guarantee better audio quality than the original responses. Audio token preference signals tend to be noisy, making text-only token alignment more stable and effective.

### Loss & Training
DPO-LN serves as the primary training objective. The learning rate is set to $5 \times 10^{-9}$ for the Temporal Transformer and $1 \times 10^{-6}$ for the Depth Transformer, with a batch size of 16, trained for one epoch on the dataset. The final data mixture includes 93,490 preference pairs, consisting of 27% purely temporal issues and 73% content-related issues. Variations such as SimPO and APO-Zero are also compared.

## Key Experimental Results

### Main Results: Comparison of Alignment Methods

| Algorithm | WebQA | LlamaQA | TriviaQA | Avg QA | ALERT | XSTest | Avg Safety |
|---|---|---|---|---|---|---|---|
| Moshi-Instruct (Baseline) | 25.8 | 60.3 | 22.1 | 36.1 | 80.0 | 61.8 | 70.9 |
| DPO | 26.3 | 58.7 | 23.5 | 36.2 | 83.2 | 67.6 | 75.4 |
| SimPO | 30.2 | 59.3 | 25.2 | 38.2 | 85.7 | 60.4 | 73.1 |
| APO-Zero | 30.0 | 61.7 | 25.4 | 39.0 | 85.6 | 70.2 | 77.9 |
| **DPO-LN** | **30.0** | **62.3** | **25.4** | **39.2** | **85.3** | **70.4** | **77.8** |

### Ablation Study: Effect of Preference Data Types

| Data Type | Count | Avg QA | Avg Safety | Replay Length |
|---|---|---|---|---|
| Type-A (Content only) | 30,045 | 36.7 | 67.7 | 26.5 |
| Type-B (Interruption) | 16,177 | 37.2 | 70.1 | 26.1 |
| Type-C (Silence) | 72,223 | 39.4 | 77.2 | 88.5 |
| B+C | 88,400 | 39.6 | 76.6 | 87.0 |
| All (Unique context) | 154,301 | 39.8 | 77.8 | 81.2 |
| **Final mix** | **93,490** | **39.2** | **77.8** | **51.4** |

### Key Findings
- **Type-C (silence) contributes the most**: Using it alone brings a +3.3% QA improvement and +6.3% safety improvement, but significantly increases the speaking rate (Replay Length surges from 20.8 to 88.5).
- **The Type-B+C combination** suppresses the excessive speaking rate while maintaining the improvements in QA and safety.
- **DPO solely on text tokens significantly outperforms joint text+audio DPO**: Joint training yields an average QA score of only ~35, which is about 4 points lower than the 39.2 scored by text-only.
- **Cross-model transferability**: Directly reusing preference data on models with different voices remains effective (safety +11.0), but the speaking rate control degrades.

## Highlights & Insights
- **First work on preference alignment for full-duplex spoken dialogue**: This study fills the massive gap between spoken dialogue AI and textual LLM alignment. The entire framework, from data construction to training and evaluation, is novel.
- **Counter-intuitive finding of "aligning only on text tokens"**: Although the model simultaneously generates both text and audio, preference learning is most effective when conducted purely within the text space. Synthesized audio does not guarantee superior acoustic quality, a finding that offers valuable guidance for future multimodal alignment research.
- **Temporal preference as a unique dimension of spoken dialogue**: Issues like interruption and silence are non-existent in textual dialogues. Decoupling content and temporal preferences and analyzing their mixture ratio reveals the unique complexity of speech-based alignment.

## Limitations & Future Work
- Re-synthesizing user voice with TTS loses speaker identity and subtle prosody cues, which may impact the realism of the preference data.
- The preference data mainly focuses on the first problematic response of the dialogue, and the alignment effect decays during long dialogues (>2 minutes).
- The framework is evaluated only on Moshi, although the pipeline itself is model-agnostic.
- When there is a prominent discrepancy in voice between the source and target models, cross-model transferability can diverge.

## Related Work & Insights
- **vs. Textual DPO/RLHF**: Text-based alignment assumes explicit turn boundaries, whereas this work handles unbounded continuous dual-stream dialogues, demanding new definitions for preference pairs and data construction.
- **vs. TTS Alignment (Zhang et al. 2024)**: TTS alignment focuses on the audio quality of single-turn generation, whereas this work focuses on content correctness and temporal rationality across multi-turn interactions.
- **vs. RLAIF**: The AI feedback pipeline in this paper shares similar concepts with RLAIF but scales it to the speech modality and introduces preference judgments in the temporal dimension.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Extends preference alignment to full-duplex spoken dialogue for the first time; both the problem definition and methodology are entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual validation using objective metrics and human evaluation, with thorough ablation, though it lacks comparisons with a broader set of baseline models.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is well-structured, with a clear motivation and rigorous experimental design.
- Value: ⭐⭐⭐⭐⭐ Highly pioneering for alignment research in speech-interactive AI, with a framework generalizable to other full-duplex models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models](../../ACL2025/audio_speech/wavrag_audio-integrated_retrieval_augmented_generation_for_spoken_dialogue_model.md)
- [\[ICML 2025\] NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction](ntpp_generative_speech_language_modeling_for_dual-channel_spoken_dialogue_via_ne.md)
- [\[ICLR 2026\] ParaS2S: Benchmarking and Aligning Spoken Language Models for Paralinguistic-Aware Speech-to-Speech Interaction](../../ICLR2026/audio_speech/paras2s_benchmarking_and_aligning_spoken_language_models_for_paralinguistic-awar.md)
- [\[ICML 2025\] Long-Form Speech Generation with Spoken Language Models](long-form_speech_generation_with_spoken_language_models.md)
- [\[ACL 2026\] VoxMind: An End-to-End Agentic Spoken Dialogue System](../../ACL2026/audio_speech/voxmind_an_end-to-end_agentic_spoken_dialogue_system.md)

</div>

<!-- RELATED:END -->
