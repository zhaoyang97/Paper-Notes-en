---
title: >-
  [Paper Note] Computational Narrative Understanding for Expressive Text-to-Speech
description: >-
  [ACL 2026][Audio & Speech][Audiobooks] This paper extracts character direct speech from fictional audiobooks to construct LibriQuote…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Audiobooks"
  - "Expressive Speech"
  - "Narrative Understanding"
  - "Character Dialogue"
  - "Dataset"
date: 2026-05-08
content_hash: 806a4b2a8753063e
---

# Computational Narrative Understanding for Expressive Text-to-Speech

**Conference**: ACL 2026  
**arXiv**: [2509.04072](https://arxiv.org/abs/2509.04072)  
**Code**: [GitHub](https://github.com/deezer/libriquote)  
**Area**: Speech Synthesis / Expressive TTS  
**Keywords**: Audiobooks, Expressive Speech, Narrative Understanding, Character Dialogue, Dataset

## TL;DR

This paper extracts character direct speech from fictional audiobooks to construct LibriQuote, a large-scale expressive speech dataset (5.3K hours of quotes + 12.7K hours of narration). It utilizes speech verbs and adverbs as pseudo-labels for speaking styles. Experiments demonstrate that fine-tuning flow-matching models on this data improves both expressiveness and intelligibility, establishing LibriQuote-test as a challenging benchmark for expressive TTS.

## Background & Motivation

**Background**: Recent TTS systems have achieved significant progress via large-scale multi-domain speech corpora (e.g., Emilia, ~100K hours), demonstrating naturalness and voice-following capabilities. Audiobooks (e.g., LibriSpeech, LibriHeavy) remain the most common open-source TTS data sources.

**Limitations of Prior Work**: (1) Existing audiobook datasets (LibriTTS, LibriHeavy) completely ignore narrative structure during segmentation—either discarding character quotes or mixing quotes with neutral narration in 30-second clips, leading to inconsistent prosodic distributions within segments; (2) There is a misconception that audiobooks lack expressive diversity, ignoring the rich prosodic variations inherent in character dialogues of fiction; (3) Existing expressive datasets are either small-scale (EXPRESSO is only tens of hours) or have limited labeling schemes (discrete emotion labels only).

**Key Challenge**: Audiobooks contain rich expressive speech, but current segmentation methods make it difficult for TTS models to utilize these resources. Segments mixing neutral narration and expressive quotes force models to lean towards learning simpler neutral components.

**Goal**: (1) Build a large-scale expressive speech dataset centered on character quotes; (2) Label speaking styles using speech verbs/adverbs from narrative contexts as pseudo-labels; (3) Verify the impact of this dataset on TTS expressiveness and intelligibility.

**Key Insight**: Grounded in narratology (Genette’s Narrative Discourse theory), the authors utilize quote detection and text-audio alignment to systematically extract and label character quotes from LibriVox fiction.

**Core Idea**: Character quotes in audiobooks naturally constitute large-scale, diverse expressive speech data. Narrators switch speaking styles based on context, while surrounding speech verbs/adverbs (e.g., "he whispered softly") provide natural style pseudo-labels.

## Method

### Overall Architecture

Data construction pipeline: LibriVox fiction audio → Corresponding Project Gutenberg text → BookNLP quote detection → ASR transcription (Zipformer-Transducer) → Text-audio alignment (Levenshtein alignment) → Quote audio segmentation → LLM extraction of speech verb/adverb pseudo-labels → Construction of high-expressiveness subset $\mathbf{Q}_f$.

### Key Designs

1.  **Narrative-Aware Quote Segmentation**:
    *   **Function**: Separating character direct quotes from neutral narration in audiobooks.
    *   **Mechanism**: BookNLP detects quote boundaries in the original text. Combined with ASR transcripts and text-audio alignment (two-stage: Longest Common Subsequence coarse alignment + Levenshtein fine alignment), each quote is mapped to precise audio segments. Average quote duration is 5.5s, while narration averages 11.8s.
    *   **Design Motivation**: Existing datasets segment randomly by sentence boundaries; 75% of segments contain only narration, while 25% contain 1-12 quotes. More quotes correlate with higher prosodic standard deviation (Spearman $\rho=0.218$). Isolating quotes provides clean expressive samples.

2.  **Speech Verb/Adverb Pseudo-Label Extraction**:
    *   **Function**: Extracting natural language descriptions of speaking styles from narrative context.
    *   **Mechanism**: A context window (approx. 100 words) surrounding the quote is used, replacing all quotes with special tokens to preserve narrative structure. An LLM (Phi-4) performs few-shot extraction of speech verbs (e.g., whispered, shouted) and adverbs (e.g., softly, angrily). LLM self-reported confidence scores are used to prune uncertain predictions.
    *   **Design Motivation**: Speech verbs and adverbs are critical cues used by narrators to adjust styles. Cohen's $\kappa=0.87$ indicates high labeling consistency.

3.  **High-Expressiveness Subset Construction $\mathbf{Q}_f$**:
    *   **Function**: Filtering the most expressive quotes for data-efficient expressive TTS training.
    *   **Mechanism**: Includes quotes with non-empty adverb pseudo-labels or speech verbs belonging to a manually defined list of expressive verbs. $\mathbf{Q}_f$ contains 377,776 quotes (11%), totaling 379 hours.
    *   **Design Motivation**: The full set contains many neutral verbs like "said"; the high-expressiveness subset achieves better expressive gains with less data.

### Loss & Training

SparkTTS (Autoregressive): Standard language modeling loss on semantic tokens fine-tuning the LLM backbone (Qwen2-0.5B). F5-TTS (Flow-matching): Official fine-tuning scripts. Training configurations include fine-tuning on different subsets and training from scratch.

## Key Experimental Results

### Main Results

**TTS Evaluation on LibriQuote-test**

| Model Configuration | WER ↓ | SIM-O ↑ | CtxMOS ↑ |
| :--- | :--- | :--- | :--- |
| GT (Ground Truth) | 6.5 | - | 3.55 |
| SparkTTS (Baseline) | 4.8 | 0.46 | 2.94 |
| SparkTTS FT($\mathbf{Q}_f$) | 4.6 | 0.47 | 2.97 |
| SparkTTS Scratch($\mathbf{Q}$) | 9.5 | 0.40 | 3.09 |
| SparkTTS Full($\mathbf{N} \cup \mathbf{Q}$) | 5.1 | 0.41 | 3.30 |
| F5-TTS (Baseline) | 6.9 | 0.53 | 2.95 |
| F5-TTS FT($\mathbf{Q}_f$) | **6.6** | **0.54** | **3.33** |

### Ablation Study

**Out-of-Domain Evaluation (LibriSpeech-PC / SeedTTS)**

| Configuration | LibriSpeech WER ↓ | SeedTTS WER ↓ |
| :--- | :--- | :--- |
| SparkTTS | 3.06 | 2.64 |
| FT($\mathbf{Q}_f$) | **2.10** | **2.07** |
| FT($\mathbf{Q}$) | **2.00** | **1.90** |

### Key Findings

*   F5-TTS fine-tuning improved CtxMOS from 2.95 to 3.33 (significant) while decreasing WER—flow-matching models can improve both expressiveness and intelligibility simultaneously.
*   SparkTTS fine-tuning mainly improved intelligibility (OOD WER dropped from 3.06 to 2.10) with limited expressiveness gains.
*   Training from scratch improves expressiveness (CtxMOS 3.09) but sacrifices intelligibility (WER 9.5).
*   The full dataset (Narration + Quotes) works better when trained from scratch (CtxMOS 3.30), showing complementarity.
*   67% of quotes in LibriQuote-test were predicted as non-neutral, compared to only 9% in LibriHeavy.

## Highlights & Insights

*   Solving TTS data issues from the perspective of narratology—audiobooks do not lack expressiveness; they lack the correct data segmentation method.
*   Speech verb/adverb pseudo-labels offer a natural and low-cost expressive annotation method, bypassing manual emotional labeling.
*   The high-expressiveness subset $\mathbf{Q}_f$, despite being only 379 hours, yields significant results, proving data quality > data volume.

## Limitations & Future Work

*   LibriVox readers are volunteers, resulting in inconsistent recording quality and expressive skills.
*   Only covers English fiction; not yet extended to other languages or genres.
*   Does not yet explore how to use contextual speech verbs/adverbs to control synthesis style during inference.

## Related Work & Insights

*   **vs LibriHeavy**: LibriHeavy does not distinguish between quotes and narration; the narrative-aware segmentation in Ours reveals neglected expressive signals.
*   **vs EXPRESSO**: EXPRESSO is high quality but only tens of hours with 26 predefined styles; LibriQuote provides 5.3K hours of natural expressive diversity.
*   **vs Emotion Speech Datasets**: Discrete emotion labels are too coarse; speech verbs/adverbs provide fine-grained style descriptions.

## Rating

*   Novelty: ⭐⭐⭐⭐ Narrative-aware segmentation + speech verb pseudo-labeling paradigm is novel.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model and multi-config experiments involving OOD and human evaluation.
*   Writing Quality: ⭐⭐⭐⭐ Clear motivation and detailed data construction pipeline.
*   Value: ⭐⭐⭐⭐ Dataset and methodology are of direct value to the expressive TTS community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MSU-Bench: Musical Score Understanding Benchmark](musical_score_understanding_benchmark_evaluating_large_language_models39_compreh.md)
- [\[ACL 2026\] Data-efficient Targeted Token-level Preference Optimization for LLM-based Text-to-Speech](data-efficient_targeted_token-level_preference_optimization_for_llm-based_text-t.md)
- [\[AAAI 2026\] DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling](../../AAAI2026/audio_speech/dualspeechlm_towards_unified_speech_understanding_and_generation_via_dual_speech.md)
- [\[ICLR 2026\] Latent Speech-Text Transformer](../../ICLR2026/audio_speech/latent_speech_text_transformer.md)
- [\[ICCV 2025\] Understanding Co-speech Gestures in-the-wild](../../ICCV2025/audio_speech/understanding_co-speech_gestures_in-the-wild.md)

</div>

<!-- RELATED:END -->
