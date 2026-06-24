---
title: >-
  [Paper Note] VoxRAG: A Step Toward Transcription-Free RAG Systems in Spoken Question Answering
description: >-
  [ACL 2025][Information Retrieval & RAG][Spoken Retrieval] Proposes VoxRAG, a modular speech-to-speech retrieval-augmented generation system. It utilizes CLAP audio embeddings to bypass transcription and retrieve semantically relevant audio segments directly from spoken queries. It validates the feasibility of transcription-free spoken retrieval in a podcast question-answering scenario, achieving a Recall@10 of 0.60 on somewhat relevant segments.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Spoken Retrieval"
  - "Transcription-Free RAG"
  - "CLAP Embeddings"
  - "Podcast QA"
  - "Speech-to-Speech"
date: 2026-05-08
content_hash: 73a369e6d6145391
---

# VoxRAG: A Step Toward Transcription-Free RAG Systems in Spoken Question Answering

**Conference**: ACL 2025  
**arXiv**: [2505.17326](https://arxiv.org/abs/2505.17326)  
**Code**: None  
**Area**: Spoken Question Answering / Retrieval-Augmented Generation  
**Keywords**: Spoken Retrieval, Transcription-Free RAG, CLAP Embeddings, Podcast QA, Speech-to-Speech

## TL;DR

Proposes VoxRAG, a modular speech-to-speech retrieval-augmented generation system. It utilizes CLAP audio embeddings to bypass transcription and retrieve semantically relevant audio segments directly from spoken queries. It validates the feasibility of transcription-free spoken retrieval in a podcast question-answering scenario, achieving a Recall@10 of 0.60 on somewhat relevant segments.

## Background & Motivation

The workflow of traditional question-answering RAG systems is: user text query $\rightarrow$ vector database retrieval of text documents $\rightarrow$ LLM generation of answers. The core assumption of this architecture is that all content is in textual format. However, with the explosive growth of audio media such as podcasts, audiobooks, and meeting recordings, an emerging and critical research direction is: can semantic retrieval be performed directly in the audio domain, without relying on intermediate automatic speech recognition (ASR) transcription?

The issues with relying on ASR transcription include:

**Accumulation of transcription errors**: Informal speech, overlapping speakers, and background noise (music, laughter, etc.) lead to poor ASR quality, with errors continuously amplified in downstream operations.

**Information loss**: Paralinguistic information such as intonation, emotion, and speaker identity is entirely lost during transcription.

**Computational redundancy**: Transcribing before retrieval introduces unnecessary intermediate steps and latency.

Existing works either apply text-only RAG to transcribed text (such as the TREC Podcasts Track), employ mixed-modality systems (such as combining COLA and RoBERTa), or require joint distillation of ASR and DPR (such as SpeechDPR). True "speech-native" retrieval—end-to-end retrieval directly from spoken queries to spoken documents—remains an under-explored field.

The core idea of this study is to leverage CLAP (Contrastive Language-Audio Pretraining) embeddings to map audio into a joint audio-language embedding space. This enables semantic-level retrieval even without precise lexical matching, thereby realizing a retrieval pipeline that maintains the audio format all the way from query to document.

## Method

### Overall Architecture

VoxRAG is a modular speech-to-speech RAG system divided into three stages:
1. **Podcast Indexing**: Preprocess audio $\rightarrow$ segment $\rightarrow$ embed $\rightarrow$ construct index
2. **Retrieval**: Spoken query $\rightarrow$ embed $\rightarrow$ cosine similarity search $\rightarrow$ return Top-10 segments
3. **Answer Generation**: Transcription of retrieved segments + transcription of query $\rightarrow$ GPT-4o generates answer

The key characteristic is that the retrieval stage is conducted entirely in the audio domain, independent of any text. Transcribed text is utilized only during the final answer generation stage.

### Key Designs

1. **Podcast Indexing Pipeline**:

    - **Audio Preprocessing**: Load podcast files, convert to mono, resample to 16kHz.
    - **Speaker Diarization**: Employ NeMo's ClusteringDiarizer for speaker identification and segmentation, assigning a speaker ID to each segment.
    - **Silence-Aware Segmentation**: Use Silero VAD to detect active speech intervals, merging them with speaker labels to define segment boundaries and ensuring each segment does not exceed 90 seconds.
    - **CLAP Embeddings**: The core technology. CLAP maps audio to a joint audio-language embedding space, rendering semantically similar audio and text close in the vector space. Unlike traditional wav2vec 2.0, which focuses on acoustic/phonemic details, CLAP learns to associate audio with linguistic meaning, allowing podcast segments to be processed like "semantic paragraphs".
    - **Optional Transcription**: Generate transcriptions using Faster-Whisper, utilized solely for LLM input and display without participating in retrieval.

2. **Retrieval Strategy**:

    - The spoken query goes through the same preprocessing and CLAP embedding pipeline.
    - Use FAISS to conduct $L_2$-normalized cosine similarity search, returning the Top-10 segments.
    - Retrieved segments are accompanied by context (preceding and succeeding segments) to enhance coherence.
    - Two configurations are evaluated: (i) pure cosine similarity, and (ii) cosine similarity + ms-marco-MiniLM-L6-v2 cross-encoder reranking.
    - Results show that cross-encoder reranking actually degrades performance (since the reranker is trained on text and is unsuitable for audio embeddings).

3. **Answer Generation**:

    - Annotate the transcribed text of retrieved segments with speaker and segment IDs.
    - Input the transcribed query and segments as a prompt to GPT-4o.
    - Display the answer and audio players for each segment in a Gradio interface.

### Loss & Training

VoxRAG is a training-free modular system, where all components utilize pretrained models:
- CLAP for audio embedding (pretrained contrastive learning model)
- NeMo for speaker diarization
- Silero VAD for voice activity detection
- Faster-Whisper for optional transcription
- GPT-4o for answer generation

## Key Experimental Results

### Main Results

**Retrieval Quality**:

| Configuration | Relevance Criterion | Recall@10 | nDCG@10 |
|---|---|---|---|
| Cosine Similarity | Very Relevant | 0.34 | 0.03 |
| Cosine Similarity | Somewhat Relevant | 0.60 | 0.27 |
| Cosine + Cross-Encoder | Very Relevant | 0.26 | 0.03 |
| Cosine + Cross-Encoder | Somewhat Relevant | 0.46 | 0.14 |

**Answer Quality (0-2 Scale)**:

| Dimension | Mean | SD | Difference Rel. to Relevance | Cohen's d | p-value |
|---|---|---|---|---|---|
| Relevance | 0.84 | 0.87 | — | — | — |
| Accuracy | 0.58 | 0.81 | -0.26 | 0.49 | <0.01 |
| Completeness | 0.56 | 0.81 | -0.28 | 0.52 | <0.01 |
| Precision | 0.46 | 0.81 | -0.38 | 0.67 | <0.01 |

### Ablation Study

| Dimension of Analysis | Finding | Description |
|---|---|---|
| Cosine vs. Cosine + Rerank | Pure Cosine is Better | Text rerankers are incompatible with audio embeddings. |
| Very vs. Somewhat Relevant | SR is Much Higher Than VR | The system excels at thematic alignment but is weaker at precise matching. |
| Relevance vs. Precision | Significant Gap ($d=0.67$) | Answers are "thematically relevant" but lack factual details. |
| Perfect Score Queries | 20% of Queries Achieve Full Score | The effect is excellent when embedding alignment succeeds. |
| "Shower"-related Queries | 4/10 Full Score | Possibly reflects CLAP's training bias towards specific concepts. |

**Correlation Analysis**:
- Accuracy, completeness, and precision are highly correlated ($r > 0.91$), capturing a shared dimension of "factual correctness".
- Relevance correlates less strongly with other dimensions ($r \approx 0.77$), suggesting that "thematic relevance" alone is insufficient to produce high-quality answers.

### Key Findings

- **Transcription-free spoken retrieval is feasible**: Although absolute performance is moderate, the system achieves a Recall@10 of 0.60 under the Somewhat Relevant criterion, proving the fundamental viability of pure audio retrieval.
- **Gap between thematic alignment and precise matching**: The immense disparity between VR and SR (Recall@10: 0.34 vs. 0.60) reveals that current audio embeddings fall short in fine-grained semantic matching. (VR = Very Relevant, SR = Somewhat Relevant)
- **Ineffectiveness of cross-encoder reranking**: Text-based rerankers degrade performance on audio embeddings, indicating a need for reranking strategies explicitly designed for the audio domain.
- **Perfect performance on 20% of queries**: Indicates that when embedding alignment and segment selection are successful, the system yields excellent results; the challenge lies in consistency.
- **Value of error analysis**: The audio playback function in the Gradio interface is crucial for identifying retrieval errors that are not apparent in pure text.

## Highlights & Insights

- **Courage of Proof-of-Concept**: Though absolute performance figures are modest, its value lies in opening a new research direction as a proof-of-concept for the first end-to-end speech-to-speech RAG system.
- **Advantages of Modular Architecture**: Individual components (segmentation, embedding, retrieval, generation) are independently replaceable, facilitating progressive future improvements.
- **Insights into CLAP's Limitations**: Although CLAP establishes coarse semantic alignment, it is insufficient for fine-grained factual retrieval. This points out directions for improving audio embedding models.
- **Innovation in Evaluation Methodology**: Employs an LLM-as-a-judge (RAGElo) approach to evaluate retrieval and answer quality, establishing a baseline protocol for spoken RAG evaluation.
- **Candor Regarding "Semi-transcription" Contradiction**: The authors frankly acknowledge that the system still relies on transcription during the generation stage; such academic honesty is highly commendable.

## Limitations & Future Work

- **Lack of Textual Baseline Comparison**: Fails to compare with traditional text retrieval methods such as BM25 or DPR, making it impossible to quantify the cost of bypassing transcription.
- **Generation Still Relies on Transcription**: While retrieval is purely auditory, answer generation requires Whisper transcriptions, reintroducing ASR noise.
- **Evaluation Bias**: Using GPT-4o for both generation and evaluation can introduce self-consistency bias.
- **Single-Episode Data**: Tests are limited to one episode of the "Trash Taste" podcast (2 hours), limiting data diversity and generalizability.
- **Future Directions**:
    - Fine-tune CLAP or employ specialized speech retrieval embeddings like SpeechDPR.
    - Explore speech-native generation methods (e.g., Spectron) to achieve entire transcription-free operations.
    - Introduce larger-scale evaluations spanning multiple episodes, speakers, and languages.
    - Design reranking strategies tailored specifically to the audio domain.

## Related Work & Insights

- **SpeechDPR** (Lin et al., 2024): Distills speech embeddings from an ASR+DPR system; potentially more suitable for speech retrieval than CLAP, though it requires paired training data.
- **SEAL** (Sun et al., 2025): Achieves cross-modal retrieval via independent encoders (without relying on transcriptions), aligning with the direction of VoxRAG.
- **SpeechRAG** (Min et al., 2025): Integrates speech retrieval with LLMs but uses textual queries to retrieve audio, as opposed to the spoken queries of VoxRAG.
- **DUAL** (Lin et al., 2022): Uses discrete speech units for speech-only retrieval without paired text training, representing a more profound speech-native direction.
- VoxRAG's contribution lies in exploring a "retrieval-first, speech-native" architectural direction, complementing DUAL's span prediction model.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Represents an emerging and crucial research direction—transcription-free spoken RAG.
- **Experimental Thoroughness**: ⭐⭐⭐ Weak experimental setup featuring only a single episode, lacking a textual baseline, and having a small sample size (50 queries).
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear problem statement, with an honest and thorough analysis of limitations.
- **Value**: ⭐⭐⭐⭐ Valuable as a proof-of-concept, though still far from a practical system.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] NeuSym-RAG: Hybrid Neural Symbolic Retrieval with Multiview Structuring for PDF Question Answering](neusym_rag_pdf_qa.md)
- [\[ACL 2025\] Investigating Language Preference of Multilingual RAG Systems](investigating_language_preference_of_multilingual_rag_systems.md)
- [\[ACL 2025\] GRAF: Graph Retrieval Augmented by Facts for Romanian Legal Multi-Choice Question Answering](graf_graph_retrieval_augmented_by_facts_for_romanian_legal_multi-choice_question.md)
- [\[ACL 2025\] From Ambiguity to Accuracy: The Transformative Effect of Coreference Resolution on RAG Systems](from_ambiguity_to_accuracy_the_transformative_effect_of_coreference_resolution_o.md)
- [\[ACL 2025\] ComRAG: Retrieval-Augmented Generation with Dynamic Vector Stores for Real-time Community Question Answering in Industry](comrag_retrieval-augmented_generation_with_dynamic_vector_stores_for_real-time_c.md)

</div>

<!-- RELATED:END -->
