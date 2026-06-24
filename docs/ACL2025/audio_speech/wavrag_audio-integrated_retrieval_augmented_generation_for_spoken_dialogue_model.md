---
title: >-
  [Paper Note] WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models
description: >-
  [ACL 2025][Audio & Speech][Retrieval-Augmented Generation] Proposed WavRAG, the first end-to-end, natively audio-compatible retrieval-augmented generation framework. It achieves unified retrieval over mixed audio-text knowledge bases via WavRetriever and enhances the contextual capabilities of spoken dialogue models using Chain-of-Thought (CoT) reasoning, achieving an approximate 10$\times$ speedup while maintaining performance comparable to state-of-the-art (SOTA) text RAG.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Retrieval-Augmented Generation"
  - "Spoken Dialogue"
  - "Multimodal Retrieval"
  - "End-to-End Audio"
  - "Chain-of-Thought"
date: 2026-05-08
content_hash: 5ecfceb8fe05e426
---

# WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models

**Conference**: ACL 2025  
**arXiv**: [2502.14727](https://arxiv.org/abs/2502.14727)  
**Code**: None  
**Area**: Spoken Dialogue / RAG  
**Keywords**: Retrieval-Augmented Generation, Spoken Dialogue, Multimodal Retrieval, End-to-End Audio, Chain-of-Thought

## TL;DR

Proposed WavRAG, the first end-to-end, natively audio-compatible retrieval-augmented generation framework. It achieves unified retrieval over mixed audio-text knowledge bases via WavRetriever and enhances the contextual capabilities of spoken dialogue models using Chain-of-Thought (CoT) reasoning, achieving an approximate 10$\times$ speedup while maintaining performance comparable to state-of-the-art (SOTA) text RAG.

## Background & Motivation

Retrieval-Augmented Generation (RAG) has become the mainstream paradigm for enhancing the external knowledge integration capabilities of LLMs. However, existing RAG frameworks are predominantly designed for text, exhibiting severe limitations in spoken dialogue scenarios:

1. **Issues with Cascaded ASR+RAG Pipelines**: Existing solutions first use Automatic Speech Recognition (ASR) to convert speech to text and then perform text RAG. This indirect approach loses rich acoustic information embedded in the audio (e.g., tone, ambient sound, music). Moreover, ASR introduces extra latency and transcription errors, which propagate through the downstream components.
2. **Neglect of Broad Audio Modalities**: Audio encompasses not only human speech but also environmental sounds, music, animal calls, and various other acoustic features that fall beyond the scope of traditional ASR.
3. **Text-Only Knowledge Bases**: Traditional RAG knowledge bases are purely text-based, rendering them incapable of leveraging audio-specific information.
4. **Lack of End-to-End Solutions**: Designing a fully end-to-end, audio-compatible RAG system remains a major challenge.

**Goal**: To construct an RAG framework capable of directly processing raw audio for embedding and retrieval, while integrating both audio and text into a unified knowledge representation.

## Method

### Overall Architecture

WavRAG consists of four steps: (1) a dual-modal encoder generates embeddings for audio and text queries; (2) top-$K$ documents are retrieved from a mixed audio-text knowledge base using cosine similarity; (3) CoT reasoning processes and analyzes the retrieved information; and (4) an LLM generates the final response grounded on the retrieved knowledge.

### Key Designs

1. **WavRetriever (Multimodal Retriever)**: Built upon Qwen2-Audio, it freezes the pre-trained audio encoder parameters while training the projection layer and the backbone LLM. Its core innovation lies in adapting the model into a multimodal retriever using a contrastive learning framework—pulling the embeddings of queries and positive knowledge samples closer while pushing negative samples apart. It adopts the InfoNCE loss function with a temperature parameter $\tau$ controlling distribution sharpness, alongside in-batch negative sampling. Inputs can be raw audio, text, or a mixture of both, which are encoded into a shared embedding space. The **Design Motivation** is to bypass the computational overhead and error propagation of ASR, directly extracting semantic representations from raw audio.

2. **Mixed Audio-Text Knowledge Base**: This extends traditional text-only knowledge bases to a unified knowledge base $\mathcal{K}$ containing audio, text, or a mixture of both. Each knowledge entry can be an audio description accompanied by the corresponding audio clip, a text-only document, a speech transcript coupled with the raw speech, etc. This enables the RAG system to retrieve acoustic information that is difficult to convey textually (such as specific bird chirps or music styles).

3. **CoT-Enhanced Generation**: Zero-Shot-CoT reasoning and Self-Consistency mechanisms are incorporated during the generation phase. Zero-Shot-CoT guides the model to perform structured reasoning over the retrieved multimodal knowledge using the cue "Let's think step-by-step". For Self-Consistency, the Universal Self-Consistency (USC) method is applied—generating multiple reasoning paths and having the LLM itself select the most consistent response instead of relying on a simple majority vote. The **Design Motivation** is to assist the spoken dialogue model in better managing and synthesizing the retrieved multimodal information.

### Loss & Training

* **Retriever Training**: InfoNCE contrastive learning loss $\mathcal{L} = -[\frac{\text{sim}(r_q, r_k^+)}{\tau} - \log Z]$
* Training Data: 1.5M samples covering 5 retrieval scenarios (S2T, S2S, T2S, T2T, AT2AT)
* The audio encoder of Qwen2-Audio is frozen, while the projection layer and the LLM backbone are trained.
* Speech queries are synthesized using CosyVoice2 TTS, with various voice prompts and noise augmentation added.
* The generation component is not trained, opting instead for off-the-shelf generator models such as GPT-4o or QwenAudio.

## Key Experimental Results

### Main Results — Retrieval Performance

| Task | Model | R@10 | Speed |
|------|------|------|------|
| Speech2Text (HotpotQA) | BGE+Whisper-Large | 0.8895 | 1.92s |
| Speech2Text (HotpotQA) | **WavRAG** | **0.8898** | **0.23s** (8.35$\times$ speedup) |
| Speech2Speech (SLUE) | BGE+Whisper-Large | 0.7196 | 4.63s |
| Speech2Speech (SLUE) | **WavRAG** | **0.7221** | **0.22s** (14.38$\times$ speedup) |
| Text2Speech (Spoken-SQuAD) | BGE | 0.8497 | - |
| Text2Speech (Spoken-SQuAD) | **WavRAG** | **0.9023** | 0.11s |

### Main Results — Generation Performance (GPT-4o, top-2)

| Method | HotpotQA EM | SLUE EM | Custom Dataset FS |
|------|------------|---------|-------------|
| TextRAG | 0.3457 | 0.3359 | - |
| WavRAG | 0.4186 | 0.4315 | 0.6408 |
| **WavRAG-CoT** | **0.4286** | **0.5239** | **0.6487** |

### Ablation Study

| Configuration | R@1 | R@10 | nDCG@10 | Description |
|------|-----|------|---------|------|
| Qwen2-Audio (Original) | 0.0675 | 0.1868 | 0.1212 | Without contrastive learning |
| **WavRAG** | **0.2728** | **0.6313** | **0.5381** | After contrastive learning |
| $\Delta$ Gain | +0.2053 | +0.4445 | +0.4169 | Contrastive learning is crucial |

### Key Findings

* WavRAG matches the performance of the strongest ASR+BGE baseline on speech-to-text retrieval tasks, while being 5-14 times faster.
* In mixed audio-text retrieval tasks, WavRAG significantly outperforms all baselines (CLAP, BGE, etc.), with R@10 improving from 0.08 to 0.63.
* Contrastive learning is critical for retrieval performance, yielding a gain of 0.2-0.3 in R@1 and over 0.4 in nDCG@10.
* CoT reasoning consistently enhances generation quality, particularly on SLUE, where the EM score improves from 0.4315 to 0.5239.
* Performance degrades when increasing the retrieved documents from top-2 to top-3; however, CoT reasoning effectively mitigates this issue.
* Human evaluations demonstrate that the generated output scores highly in grammar, factual accuracy, relevance, and usefulness.

## Highlights & Insights

* **Paradigm Shift**: The first end-to-end, native audio-compatible RAG framework, breaking the bottleneck of cascaded ASR pipelines.
* **Modality Unification**: Audio and text are encoded within the same embedding space, achieving true cross-modal retrieval.
* **Significant Speed Advantage**: Bypassing the ASR phase yields a 5-14$\times$ speedup, which is critical for real-time spoken dialogue systems.
* **New Capabilities for Audio Knowledge Bases**: The framework can retrieve and utilize acoustic information (such as ambient sounds and music) that cannot be described textually, expanding the boundaries of RAG.
* **CoT for Information Overload**: While performance degrades as the number of retrieved documents increases, the structured reasoning of CoT effectively mitigates this issue.

## Limitations & Future Work

* Current efforts focus solely on semantic-level RAG enrichment, leaving acoustic-level characteristics (e.g., speech prosody, emotional tone) unexplored.
* The audio encoder parameters remain fully frozen; end-to-end fine-tuning could potentially yield further performance improvements.
* The construction and maintenance costs of mixed-modality knowledge bases are not discussed.
* The generation component directly adopts off-the-shelf LLMs (GPT-4o, QwenAudio) without specific optimization for RAG scenarios.
* Although mitigated by CoT, the performance degradation that occurs when $Top-k$ increases remains fundamentally unresolved.

## Related Work & Insights

* LLM2Vec (BehnamGhader et al., 2024) inspired the approach of utilizing fine-tuned LLMs for embeddings.
* Successful experiences from E5-V and VLM2VEC in the visual domain have been transferred to the audio modality.
* The effectiveness of contrastive learning in multimodal retrieval is validated once again.
* **Insight**: Similar end-to-end unified retrieval schemes could be applied to other multimodal RAG scenarios, such as video RAG or sensor data RAG.

## Rating

| Dimension | Score (1-5) | Explanation |
|------|-----------|------|
| Innovation | 4 | First end-to-end audio RAG; the mixed knowledge base design is novel |
| Practicality | 4 | A 10$\times$ speedup is of significant value to real-time dialogue systems |
| Experimental Thoroughness | 4 | Comprehensive evaluation across multiple tasks and baselines, ablation studies, and human evaluation |
| Writing Quality | 4 | The framework description is clear and the comparisons are intuitive |
| Overall Score | 4 | An important work extending RAG to the audio modality |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Aligning Spoken Dialogue Models from User Interactions](../../ICML2025/audio_speech/aligning_spoken_dialogue_models_from_user_interactions.md)
- [\[ACL 2025\] Benchmarking Open-ended Audio Dialogue Understanding for Large Audio-Language Models](audio_dialogue_benchmark.md)
- [\[ACL 2026\] PlanRAG-Audio: Planning and Retrieval Augmented Generation for Long-form Audio Understanding](../../ACL2026/audio_speech/planrag-audio_planning_and_retrieval_augmented_generation_for_long-form_audio_un.md)
- [\[ACL 2026\] MARQUIS: A Three-Stage Pipeline for Video Retrieval-Augmented Generation](../../ACL2026/audio_speech/marquis_a_three-stage_pipeline_for_video_retrieval-augmented_generation.md)
- [\[ACL 2025\] CLaMP 3: Universal Music Information Retrieval Across Unaligned Modalities and Unseen Languages](clamp_3_universal_music_information_retrieval_across_unaligned_modalities_and_un.md)

</div>

<!-- RELATED:END -->
