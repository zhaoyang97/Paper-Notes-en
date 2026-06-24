---
title: >-
  [Paper Note] Does Your Voice Assistant Remember? Analyzing Conversational Context Recall and Utilization in Voice Interaction Models
description: >-
  [ACL 2025][Audio & Speech][Voice interaction models] Systematically evaluates the conversation history recall capabilities of open-source voice interaction models, introduces the ContextDialog benchmark, and reveals that these models are far weaker than text models in recalling past speech information, a gap that RAG methods also struggle to bridge effectively.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Voice interaction models"
  - "conversation memory"
  - "context recall"
  - "retrieval-augmented generation"
  - "multi-turn dialogue"
date: 2026-05-08
content_hash: 9d3e9a61b438b981
---

# Does Your Voice Assistant Remember? Analyzing Conversational Context Recall and Utilization in Voice Interaction Models

**Conference**: ACL 2025  
**arXiv**: [2502.19759](https://arxiv.org/abs/2502.19759)  
**Code**: [https://huggingface.co/datasets/ContextDialog/ContextDialog](https://huggingface.co/datasets/ContextDialog/ContextDialog)  
**Area**: Speech  
**Keywords**: Voice interaction models, conversation memory, context recall, retrieval-augmented generation, multi-turn dialogue

## TL;DR

Systematically evaluates the conversation history recall capabilities of open-source voice interaction models, introduces the ContextDialog benchmark, and reveals that these models are far weaker than text models in recalling past speech information, a gap that RAG methods also struggle to bridge effectively.

## Background & Motivation

**Background**: Voice assistants have become indispensable utilities in daily life. With the development of LLMs, the community has shifted from cascaded ASR→LLM→TTS pipelines to end-to-end approaches. Closed-source models such as GPT-4o and Gemini 2.0 demonstrate excellent performance in remembering and recalling past utterances in multi-turn dialogues.

**Limitations of Prior Work**: Although open-source voice interaction models perform exceptionally in single-turn interactions, it remains unclear whether they can effectively maintain and utilize conversational history context in multi-turn dialogues. Current benchmarks do not explicitly require models to exploit conversational history for response generation.

**Key Challenge**: Closed-source models have exhibited strong conversation memory capabilities, but whether the open-source community possesses equivalent capabilities remains largely unexplored. Crucially, existing benchmarks only evaluate semantic coherence across turns rather than demanding actual recall of specific past information.

**Goal**: (1) Can open-source voice interaction models recall past conversational content and generate relevant responses? (2) Can RAG methods compensate for the memory deficits of these models? (3) How robust are these models to retrieval errors?

**Key Insight**: Constructs ContextDialog, a specialized speech-to-speech benchmark where QA pairs require reliance on dialogue history to be answered; systematically evaluates models from both native recall capabilities and RAG-augmented dimensions.

**Core Idea**: Open-source voice interaction models possess a severe bottleneck in recalling past speech content—they are not only far weaker than text-only LLMs but are also difficult to improve via RAG due to high sensitivity to retrieval noise.

## Method

### Overall Architecture

The study investigates two experimental dimensions: (1) internal model recall capability evaluation, directly testing whether the model can answer questions based on dialogue history; (2) RAG augmentation evaluation, supplying relevant past utterances via an external retrieval module to test whether the model can effectively utilize them.

### Key Designs

**Module 1: ContextDialog Benchmark Construction**

- **Function**: Build a speech-to-speech dialogue recall evaluation benchmark.
- **Mechanism**: 
  1. **Text QA Generation**: Based on the MultiDialog corpus (~340 hours, 12 speakers, $\ge 10$ turns per dialogue), QA pairs are generated from dialogue transcriptions using GPT-4o. The target information must appear only once, not be a Yes/No question, and be based on the first or second half of either user/model speech, yielding 4 QA pairs per dialogue.
  2. **QA Verification**: A three-stage verification process using o1-mini: (1) questions should not be answerable without the supporting utterance context; (2)(3) questions must be answerable once the supporting utterance is provided.
  3. **Speech Synthesis**: Utilize Fish Speech (speaker-adaptive TTS) to generate speech QA that matches the original speaker's timbre. Five generations are performed per pair, taking the one with the lowest WER, with non-zero WERs manually checked.
- **Design Motivation**: Ensures that the questions strictly require dialogue history for answering, while maintaining speaker consistency in the speech.

**Module 2: Recall Capability Evaluation**

- **Function**: Test differences in model recall between past user utterances and its own past utterances.
- **Mechanism**: Separately evaluates the recall of past user utterances (speech-only format) and past model utterances (dual text+speech format). Evaluates oral responses ($\mathcal{S} \to \mathcal{T}, \underline{\mathcal{S}}$) and intermediate text responses ($\mathcal{S} \to \underline{\mathcal{T}}, \mathcal{S}$), measured by GPT Score (1-to-5 scale).
- **Design Motivation**: Distingush modality differences—the model has a text backup for its own utterances but only speech input for user utterances.

**Module 3: RAG Augmentation Evaluation**

- **Function**: Test whether retrieval augmentation can improve model memory.
- **Mechanism**: Transcribe past utterances using ASR (whisper-large-v3-turbo) and store them; use e5-large-v2 to extract embeddings for cosine similarity retrieval of the top-k items. Inject retrieved text into generation using the format "Based on your/my statement: ...". Also test SONAR (direct retrieve from speech) as a comparison.
- **Design Motivation**: Evaluate the actual effectiveness of mature NLP RAG techniques in voice interaction scenarios.

### Loss & Training

- No model training is involved; instead, existing open-source models (GLM-4-Voice, Lyra, Freeze-Omni, MiniCPM-o) are evaluated.
- Uses LLM-as-a-judge (gpt-4o-mini) for a 5-point scale evaluation.
- Transcribes speech responses with whisper-large-v3 before evaluation.

## Key Experimental Results

### Main Results

Dialogue recall performance (GPT Score, $\mathcal{S} \to \mathcal{T}, \underline{\mathcal{S}}$):

| Model | User | System | Overall | WER |
|------|------|--------|---------|-----|
| GLM-4-Voice | 1.94 | 2.76 | 2.35 | 8.36% |
| Lyra | 2.51 | 3.16 | 2.83 | 5.90% |
| Freeze-Omni | 1.73 | 2.28 | 2.00 | 12.36% |
| MiniCPM-o | 2.44 | 2.84 | 2.64 | 24.90% |

Corresponding text backbone LLM performance:

| Model | Overall |
|------|---------|
| glm-4-9b-chat | 4.10 |
| Qwen2-VL-7B | 3.84 |
| Qwen2-7B | 4.03 |
| Qwen2.5-7B | 4.06 |

### Ablation Study

Effect of RAG augmentation (Supporting = correct supporting utterance, Irrelevant = unrelated utterance):

| Model | No RAG | Supporting | Irrelevant |
|------|-------|------------|------------|
| GLM-4-Voice | 2.35 | **2.60** | 1.87 |
| Lyra | 2.83 | **3.44** | 1.96 |
| Freeze-Omni | 2.00 | **2.38** | 1.54 |
| MiniCPM-o | **2.64** | 2.49 | 1.63 |

RAG effects with different retrievers and top-k (Lyra as an example):

| Retriever | ASR | top-1 | top-2 | top-3 |
|--------|-----|-------|-------|-------|
| e5-large-v2 | ✓ | 2.83 | 2.68 | 2.52 |
| e5-large-v2 | ✗ | 2.94 | 2.78 | 2.68 |
| SONAR | - | 2.48 | 2.39 | 2.25 |

Lyra No RAG baseline = 2.83, RAG brought no improvement.

### Key Findings

1. **Speech models are far weaker than text models**: All speech models scored around 2.0–2.83 Overall, whereas their corresponding text LLMs reached 3.84–4.10, showing a gap of approximately 1–2 points.
2. **Modality recall gap**: All speech models recalled their own utterances (System) better than they did user utterances (User), $p<0.01$. This is because the model's own utterances are backed up by both text and speech, whereas user utterances are speech-only.
3. **Frozen LLMs exacerbate the issue**: Freeze-Omni (which freezes the LLM during training) performed the worst (2.00), demonstrating that expanding LLMs to speech significantly degrades long-context processing capabilities.
4. **RAG is largely ineffective**: In actual retrieval-augmented settings, all models performed close to or worse than the baseline. Increasing top-k introduces more noise instead.
5. **Models are highly sensitive to retrieval noise**: Providing the correct supporting utterance improves scores, but providing irrelevant utterances causes sharp drops (Lyra: 2.83 $\to$ 1.96).
6. **Attention analysis**: The attention weights historical model assigns to supporting utterances are much lower than weights assigned to its own past statements, indicating an inherent attention bias.

## Highlights & Insights

- **Fills an important gap**: Conducts the first systematic evaluation of dialogue memory capacity in open-source voice interaction models, revealing a massive gap compared to closed-source models.
- **Exquisite ContextDialog design**: Three-stage verification ensures questions strictly depend on history; TTS generation maintains speaker consistency.
- **User vs. System recall gap** exposes a fundamental limitation of current architectures—the internal text representations are the primary carrier of memory, while memory for raw audio information is extremely weak.
- The finding that **RAG fails** has significant practical implications—one cannot simply copy-paste NLP RAG techniques to voice scenarios.
- Attention map analysis visually demonstrates the models' "neglect" of user utterances.

## Limitations & Future Work

1. **Only covers a subset of open-source models**: Not all multi-turn voice interaction models were evaluated.
2. **RAG is solely based on text**: Open-source voice retrieval modules are immature, preventing pure speech-based RAG.
3. **Synthetic data**: QA pairs were generated by GPT-4o, and speech was synthesized by TTS, rather than using real human-to-human interactions.
4. **Simple question types**: Only tested direct recall, not including complex questions that require reasoning.
5. The latency impact of current methods was not evaluated.
6. Future directions: improving long-context modeling, developing robust voice RAG technologies, and designing specialized memory modules.

## Related Work & Insights

- **SLAM-Omni** improves multi-turn modeling by storing transcribed text and injecting it as a prefix.
- **Lyra** explores techniques for processing long audio histories and expanding context windows.
- **MultiDialog** provides approximately 340 hours of multi-speaker dialogue corpus as a building foundation.
- **Fish Speech**'s speaker-adaptive TTS ensures speaker consistency in synthesized speech.
- RAG techniques in NLP (Atlas, REALM) provide reference directions for speech-scenario improvements.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (First to focus on the memory capability of voice interaction models, proposing a dedicated benchmark)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Four models + RAG analysis + attention visualization + multi-angle ablations)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear structure, deep analysis, and insightful findings)
- **Value**: ⭐⭐⭐⭐⭐ (Exposes critical bottlenecks in open-source voice models; highly instructive for community development)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Distilling an End-to-End Voice Assistant Without Instruction Training Data](distilling_an_end-to-end_voice_assistant_without_instruction_training_data.md)
- [\[ACL 2025\] Finding A Voice: Exploring the Potential of African American Dialect and Voice Generation for Chatbots](aae_voice_chatbot.md)
- [\[ICLR 2026\] WearVox: An Egocentric Multichannel Voice Assistant Benchmark for Wearables](../../ICLR2026/audio_speech/wearvox_an_egocentric_multichannel_voice_assistant_benchmark_for_wearables.md)
- [\[ACL 2025\] SpeechIQ: Speech-Agentic Intelligence Quotient Across Cognitive Levels in Voice Understanding by Large Language Models](speechiq_speechagentic_intelligence_quotient_across_cognitive.md)
- [\[ICLR 2026\] UniSS: Unified Expressive Speech-to-Speech Translation with Your Voice](../../ICLR2026/audio_speech/uniss_unified_expressive_speech-to-speech_translation_with_your_voice.md)

</div>

<!-- RELATED:END -->
