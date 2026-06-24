---
title: >-
  [Paper Note] EuroSpeech: A Multilingual Speech Corpus
description: >-
  [NeurIPS2025][Audio & Speech][multilingual speech] This paper presents a scalable, open-source pipeline for automatically constructing the EuroSpeech dataset from recordings of 22 European parliaments — yielding 61K hours of high-quality speech-text aligned data across 22 languages, with 19 languages exceeding 1K hours. Fine-tuning Whisper on this data reduces average WER by 41.8%.
tags:
  - "NeurIPS2025"
  - "Audio & Speech"
  - "multilingual speech"
  - "parliamentary recordings"
  - "ASR"
  - "speech-text alignment"
  - "dataset construction"
  - "low-resource languages"
date: 2026-05-08
content_hash: 0cbaa3bf64d2a410
---

# EuroSpeech: A Multilingual Speech Corpus

**Conference**: NeurIPS2025
**arXiv**: [2510.00514](https://arxiv.org/abs/2510.00514)  
**Code**: [disco-eth/EuroSpeech](https://huggingface.co/datasets/disco-eth/EuroSpeech)  
**Area**: Speech Processing / Multilingual Datasets
**Keywords**: multilingual speech, parliamentary recordings, ASR, speech-text alignment, dataset construction, low-resource languages

## TL;DR

This paper presents a scalable, open-source pipeline for automatically constructing the EuroSpeech dataset from recordings of 22 European parliaments — yielding 61K hours of high-quality speech-text aligned data across 22 languages, with 19 languages exceeding 1K hours. Fine-tuning Whisper on this data reduces average WER by 41.8%.

## Background & Motivation

- **Core Problem**: Multilingual ASR/TTS models depend on large-scale annotated speech data, yet publicly available multilingual datasets exhibit severe imbalance in language coverage — most languages fall well below the 1K-hour threshold required for effective training.
- **Limitations of Prior Datasets**:
    - Common Voice covers 133 languages, but only 8 exceed 1K hours.
    - VoxPopuli draws from EU parliamentary sessions but covers only 16 languages with 1.8K hours total, none exceeding 1K hours.
    - Whisper's 680K-hour training set is not publicly released; MMS-Lab covers 1,107 languages but remains private.
    - FLEURS contains only 1.4K hours and targets benchmarking rather than training.
- **Opportunity**: National parliament recordings paired with official transcripts constitute a natural source of high-quality multilingual speech. However, data formats are highly heterogeneous, transcripts are not verbatim, and recordings are long and unsegmented — making scalable processing with existing pipelines impractical.
- **Motivation**: Design a source-agnostic, automated pipeline capable of handling non-verbatim transcripts and cross-format data sources to construct a large-scale, linguistically balanced multilingual speech dataset.

## Method

### 1. Overall Pipeline Architecture

The full pipeline comprises three stages:

1. **Data Source Collection and Metadata Curation**: Manual inspection of parliamentary websites → writing custom scraping scripts → generating standardized CSVs containing audio/video URLs, transcript links, and session IDs.
2. **Download Pipeline**: A dispatch architecture distributes URLs to dedicated handlers (direct download, YouTube, dynamic pages, etc.), with support for resumable transfers, parallel downloading, and PostgreSQL-based status tracking.
3. **Alignment Pipeline**: Audio segmentation → ASR transcription → two-stage dynamic alignment → CER filtering → output of aligned dataset.

### 2. Two-Stage Dynamic Alignment Algorithm (Core Contribution)

To address noise in parliamentary transcripts — including non-verbatim text, speaker annotations, and procedural language — a coarse-to-fine alignment strategy is proposed:

**Stage 1 — Coarse Search**:
- For each ASR-transcribed segment, a sliding window of length $n$ (number of words in the ASR segment) is applied.
- Search proceeds sequentially from the previous match position `last_end_idx`.
- CER is computed for each window against the ASR text; the first candidate with CER < 30% is selected. If none exists, the $k=3$ candidates with the lowest CER are retained.

**Stage 2 — Refined Search**:
- Local optimization is applied to candidates identified in the coarse search.
- The start position is varied within $\pm 15$ words, and window size is tested over the range $(L-15, L+15)$.
- The (start, size) combination minimizing CER is selected as the final match.

**Fallback Mechanism**:
- If CER exceeds threshold $\theta$ after local search, a global coarse search is re-executed from the beginning of the transcript, resolving position drift caused by prior misalignments.
- If the global search also fails, a default match is performed (refined search near `last_end_idx`, retained regardless of CER), ensuring complete dataset coverage.

### 3. Transcript Preprocessing

- Built-in parsers for PDF, DOCX, HTML, TXT, and SRT formats.
- An optional LLM-based cleaning step (defaulting to Gemini Flash 2.0) removes speaker labels, procedural annotations, and other non-speech content.
- In German-language tests, LLM-based cleaning reduced the median CER of aligned segments from 12.3% to 9.7%.

### 4. Multi-Transcript Selection Strategy

A parliament may release multiple transcripts in different formats for the same session, with no explicit correspondence to the audio. The proposed solution:
- Aligns audio against all candidate transcripts × all formats independently.
- Selects the best format by median CER.
- Selects the final transcript(s) according to a configurable criterion (lowest CER or all below a threshold).

### 5. CER-Based Tiered Quality Filtering

CER serves as the primary quality control metric. Aligned duration statistics at multiple thresholds are reported:

| Filter Level | Aligned Duration | Proportion |
|---|---|---|
| All aligned | 78.1K h | 100% |
| CER < 30% | 61.0K h | 78.2% |
| CER < 20% | 50.5K h | 65.4% |
| CER < 10% | 32.3K h | 41.0% |

CER < 20% defines the primary dataset (consistent with VoxPopuli), with audio segments of 3–20 seconds and a sampling rate of 16 kHz.

## EuroSpeech Dataset Analysis

### Language Coverage and Scale

| Language | Duration (CER<20%) | Language | Duration (CER<20%) |
|---|---|---|---|
| Croatian | 5615.8 h | Bulgarian | 2200.1 h |
| Danish | 5559.8 h | German | 2184.4 h |
| Norwegian | 3866.7 h | Serbian | 1855.7 h |
| Portuguese | 3293.5 h | Finnish | 1848.2 h |
| Italian | 2813.7 h | Latvian | 1218.8 h |
| Lithuanian | 2681.2 h | Ukrainian | 1191.1 h |
| English | 2609.3 h | Slovenian | 1156.4 h |
| Slovak | 2553.6 h | Estonian | 1014.9 h |
| Greek | 2395.4 h | Bosnian | 691.3 h |
| Swedish | 2312.8 h | Icelandic | 647.4 h |
| French | 2249.8 h | Maltese | 613.0 h |

**Key Comparison**: EuroSpeech surpasses the scale of existing public datasets for 12 languages, with 8 languages crossing the 1K-hour threshold for the first time. For 5 languages, the data volume is 10–100× that of the previous best (e.g., Lithuanian: 25h → 2681h; Slovak: 61h → 2554h; Maltese: 44h → 613h).

### Data Splits

Train/dev/test splits are defined at the level of complete parliamentary sessions, preventing segments from the same session from appearing across different splits.

## Key Experimental Results

### ASR Fine-Tuning Evaluation

Whisper v3 Turbo is fine-tuned on approximately 200 hours of the lowest-CER data per language for 6 low-resource languages, and evaluated on the out-of-domain FLEURS test set:

| Language | Baseline WER | Fine-tuned WER | Relative Gain |
|---|---|---|---|
| Maltese | 72.2% | 25.9% | **64.1%** |
| Icelandic | 20.0% | 15.0% | 25.0% |
| Lithuanian | 25.0% | 15.9% | 36.4% |
| Latvian | 19.3% | 11.1% | 42.5% |
| Slovenian | 20.5% | 13.0% | 36.7% |
| Estonian | 18.4% | 9.9% | **46.1%** |
| **Average** | **29.2%** | **15.1%** | **41.8%** |

- Maltese shows the largest improvement (64.1%), transitioning from near-unusable to practically viable.
- All 6 languages achieve substantial gains on the **out-of-domain** test set, validating the dataset's generalization value.
- Training efficiency is high: only 200 hours of data and 1.3–43 GPU hours per language are required.

### Computational Cost

| Stage | Resource Consumption |
|---|---|
| Video download | ~3,930 CPU·h |
| Transcript retrieval | ~280 CPU·h |
| Alignment processing | ~5,548 GPU·h (mixed GPU types) |
| ASR fine-tuning | 1.3–43 h/language (A6000) |

## Highlights & Insights

- **Balanced coverage as core value**: Rather than maximizing total duration, EuroSpeech's distinguishing contribution is that all 22 languages exceed 500 hours — in contrast, only 8 of Common Voice's 133 languages exceed 1K hours.
- **Elegant two-stage alignment design**: Linear scanning ensures efficiency; local refined search ensures precision; two-level fallback ensures robustness — all without requiring manual transcript preprocessing.
- **Engineering value of the pipeline**: The modular design (metadata CSV → download dispatch → alignment → filtering) enables extension to new parliaments by writing only a metadata scraping script, with full reuse of core logic.
- **LLM-assisted cleaning reduces barriers**: Gemini Flash is used to automatically remove non-speech elements from PDF transcripts, substantially reducing manual preprocessing effort.
- **Deliberate selection of hardest languages for evaluation**: The 6 languages on which Whisper performs worst are specifically chosen — demonstrating the largest absolute gains while also serving as a rigorous stress test of pipeline robustness (poorer ASR → noisier alignment → stronger fault tolerance required).

## Limitations & Future Work

- **Domain homogeneity**: Coverage is limited to formal parliamentary speech, which is planned and formal in style; models may generalize poorly to conversational or informal settings.
- **Limited dialect and variety coverage**: Parliamentary language tends toward standard/official registers, with insufficient representation of dialectal and sociolinguistic variation.
- **Dependence on ASR quality**: Alignment quality is bounded by the underlying ASR system; for low-resource languages where ASR is inherently weak, alignment may introduce systematic biases.
- **Geographic limitation**: Coverage is restricted to 22 European nations; low-resource languages in Africa, Asia, and South America are not addressed.
- **Potential misuse risk**: The data contains identifiable speech from public political figures and could be exploited for voice synthesis or deepfake generation, though the formal parliamentary register may partially limit misuse.
- **Future directions**: Extension to conversational speech, incorporation of speaker metadata, and a 24 kHz TTS-oriented variant.

## Related Work & Insights

| Dataset | Languages | Total Duration | Languages >1K h | Public |
|---|---|---|---|---|
| Common Voice | 133 | 22.1K h | 8 | ✓ |
| VoxPopuli | 16 | 1.8K h | 0 | ✓ |
| YODAS | 149 | 369.5K h | 13 | ✓ |
| Whisper Data | 91 | 680K h | 16 | ✗ |
| MMS-Lab | 1,107 | 44.7K h | 0 | ✗ |
| **EuroSpeech** | **22** | **61K h** | **19** | **✓** |

Among publicly available datasets, EuroSpeech achieves the highest number of languages exceeding 1K hours (19), trading breadth of language coverage for depth within each language.

**Further Insights**:
- **Tiered quality–quantity tradeoff**: The multi-level CER filtering design (10%/20%/30%) is broadly applicable — different downstream tasks have different noise tolerances (TTS requires high quality; ASR pretraining can leverage noisier data).
- **Systematic utilization of public data**: Parliamentary recordings represent a class of "government-released but systematically underutilized" data sources; analogous opportunities exist in other modalities (legal texts, policy documents, sign language video).
- **Pipeline-first thinking**: Rather than manually annotating small high-quality datasets, designing robust pipelines to automatically extract value from large noisy sources is the prevailing paradigm for large-scale dataset construction.

## Rating
- Novelty: ⭐⭐⭐ — pipeline is not novel per se, but the scale is significant
- Experimental Thoroughness: ⭐⭐⭐⭐ — validated across multiple languages for ASR
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐ — large-scale multilingual speech resources carry high community value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MultiMed: Multilingual Medical Speech Recognition via Attention Encoder Decoder](../../ACL2025/audio_speech/multimed_multilingual_medical_speech_recognition_via_attention_encoder_decoder.md)
- [\[ACL 2025\] GigaSpeech 2: An Evolving, Large-Scale and Multi-domain ASR Corpus for Low-Resource Languages](../../ACL2025/audio_speech/gigaspeech2_low_resource_asr.md)
- [\[ICLR 2026\] Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion](../../ICLR2026/audio_speech/scalable_multilingual_multimodal_machine_translation_with_speech-text_fusion.md)
- [\[ACL 2025\] SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models](../../ACL2025/audio_speech/speechweave_diverse_multilingual_synthetic_text_audio_data_generation_pipeline_f.md)
- [\[ACL 2026\] From Flat Language Labels to Typological Priors: Structured Language Conditioning for Multilingual Speech-to-Speech Translation](../../ACL2026/audio_speech/from_flat_language_labels_to_typological_priors_structured_language_conditioning.md)

</div>

<!-- RELATED:END -->
