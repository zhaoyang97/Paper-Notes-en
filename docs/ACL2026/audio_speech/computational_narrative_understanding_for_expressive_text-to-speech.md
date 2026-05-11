---
title: >-
  [Paper Note] Computational Narrative Understanding for Expressive Text-to-Speech
description: >-
  [ACL 2026][Audio & Speech][Audiobooks] This paper extracts character direct speech from fiction audiobooks to construct a large-scale expressive speech dataset…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Audiobooks"
  - "Expressive Speech"
  - "Narrative Understanding"
  - "Character Dialogue"
  - "Dataset"
date: 2026-05-08
content_hash: 8ed3c387ebe57103
---

# Computational Narrative Understanding for Expressive Text-to-Speech

**Conference**: ACL 2026
**arXiv**: [2509.04072](https://arxiv.org/abs/2509.04072)
**Code**: [GitHub](https://github.com/deezer/libriquote)
**Area**: Speech Synthesis / Expressive TTS
**Keywords**: Audiobooks, Expressive Speech, Narrative Understanding, Character Dialogue, Dataset

## TL;DR

This paper extracts character direct speech from fiction audiobooks to construct a large-scale expressive speech dataset, LibriQuote (5.3K hours of quotations + 12.7K hours of narration), annotating speaking style with speech verb and adverb pseudo-labels derived from narrative context. Experiments demonstrate that fine-tuning a flow-matching model simultaneously improves expressiveness and intelligibility, and that LibriQuote-test constitutes a challenging benchmark for expressive TTS.

## Background & Motivation

**Background**: Recent TTS systems have achieved substantial progress through large-scale multi-domain speech corpora (e.g., Emilia, ~100K hours), demonstrating strong naturalness and voice-following capabilities. Audiobooks (e.g., LibriSpeech, LibriHeavy) remain the most common open-source TTS data source.

**Limitations of Prior Work**: (1) Existing audiobook datasets (LibriTTS, LibriHeavy) entirely disregard narrative structure during segmentation—either discarding character quotations or mixing quoted speech with neutral narration within the same 30-second segment, resulting in segments with highly heterogeneous prosodic distributions; (2) audiobooks are sometimes considered to lack expressive diversity, an assumption that overlooks the rich prosodic variation inherent in fictional character dialogue; (3) existing expressive speech datasets are either small in scale (EXPRESSO contains only tens of hours) or limited in annotation scheme (relying solely on discrete emotion labels).

**Key Challenge**: Fictional audiobooks contain abundant expressive speech in character dialogue, yet conventional segmentation strategies prevent TTS models from exploiting this resource—segments that intermix neutral narration and expressive quotations bias models toward learning the simpler neutral portions.

**Goal**: (1) Construct a large-scale expressive speech dataset centered on character quotations; (2) annotate speaking style using speech verbs and adverbs from narrative context as pseudo-labels; (3) validate the dataset's effectiveness in improving TTS expressiveness and intelligibility.

**Key Insight**: Drawing on narratological theory (Genette's theory of narrative discourse), the paper applies quotation detection and text-audio alignment techniques to systematically extract and annotate character quotations from LibriVox fiction recordings.

**Core Idea**: Character quotations in audiobooks naturally constitute large-scale, diverse expressive speech data—narrators modulate their speaking style according to context when voicing character dialogue, while surrounding speech verbs and adverbs (e.g., "he whispered softly") provide natural style pseudo-labels.

## Method

### Overall Architecture

Data construction pipeline: LibriVox fiction audio → download corresponding Project Gutenberg text → BookNLP quotation detection → ASR transcription (Zipformer-Transducer) → text-audio alignment (Levenshtein alignment) → quotation audio segmentation → LLM-based speech verb/adverb pseudo-label extraction → construction of high-expressiveness subset $\mathbf{Q}_f$.

### Key Designs

1. **Narrative-Aware Quotation Segmentation**:

    - Function: Separates character direct speech from neutral narration in audiobooks.
    - Mechanism: BookNLP detects quotation boundaries in the original text; combined with ASR transcription and two-stage text-audio alignment (coarse alignment via longest common subsequence + fine alignment via Levenshtein distance), each quotation is mapped to a precise audio segment. Mean quotation duration is 5.5 s; mean narration duration is 11.8 s.
    - Design Motivation: Existing datasets segment by sentence boundaries, yielding segments where 75% contain only narration and 25% contain 1–12 quotations; prosodic standard deviation increases with the number of quotations per segment (Spearman $\rho = 0.218$). Separating quotations yields clean expressive speech samples.

2. **Speech Verb/Adverb Pseudo-Label Extraction**:

    - Function: Extracts natural-language descriptions of speaking style from narrative context.
    - Mechanism: A context window of approximately 100 words before and after each quotation is constructed; all quotations are replaced with special tokens to preserve only the narrative structure, and an LLM (Phi-4) extracts speech verbs (e.g., *whispered*, *shouted*) and adverbs (e.g., *softly*, *angrily*) via few-shot prompting. LLM-reported confidence scores are used to prune uncertain predictions, maximizing precision.
    - Design Motivation: Speech verbs and adverbs are key cues by which narrators modulate speaking style; Cohen's $\kappa = 0.87$ indicates high inter-annotator agreement.

3. **High-Expressiveness Subset Construction $\mathbf{Q}_f$**:

    - Function: Filters the most expressive quotations for data-efficient expressive TTS training.
    - Mechanism: Includes all quotations with non-empty adverb pseudo-labels, plus quotations whose speech verbs belong to a manually defined list of expressive verbs. The resulting $\mathbf{Q}_f$ contains 377,776 quotations (11% of the full set), totaling 379 hours.
    - Design Motivation: The full quotation set contains a large proportion of neutral verbs such as *said*; the high-expressiveness subset achieves better expressive gains with substantially less data.

### Loss & Training

SparkTTS (autoregressive): the LLM backbone (Qwen2-0.5B) is fine-tuned using a standard language modeling loss over semantic tokens. F5-TTS (flow-matching): official fine-tuning scripts are used. Training configurations include fine-tuning and training-from-scratch experiments across different data subsets.

## Key Experimental Results

### Main Results

**TTS Evaluation on LibriQuote-test**

| Model Configuration | WER ↓ | SIM-O ↑ | CtxMOS ↑ |
|---|---|---|---|
| GT (ground truth) | 6.5 | - | 3.55 |
| SparkTTS (baseline) | 4.8 | 0.46 | 2.94 |
| SparkTTS FT($\mathbf{Q}_f$) | 4.6 | 0.47 | 2.97 |
| SparkTTS Scratch($\mathbf{Q}$) | 9.5 | 0.40 | 3.09 |
| SparkTTS Full($\mathbf{N} \cup \mathbf{Q}$) | 5.1 | 0.41 | 3.30 |
| F5-TTS (baseline) | 6.9 | 0.53 | 2.95 |
| F5-TTS FT($\mathbf{Q}_f$) | **6.6** | **0.54** | **3.33** |

### Ablation Study

**Out-of-Domain Evaluation (LibriSpeech-PC / SeedTTS)**

| Configuration | LibriSpeech WER ↓ | SeedTTS WER ↓ |
|---|---|---|
| SparkTTS | 3.06 | 2.64 |
| FT($\mathbf{Q}_f$) | **2.10** | **2.07** |
| FT($\mathbf{Q}$) | **2.00** | **1.90** |

### Key Findings

- Fine-tuning F5-TTS raises CtxMOS from 2.95 to 3.33 (significant improvement) while simultaneously reducing WER, demonstrating that flow-matching models can improve both expressiveness and intelligibility.
- Fine-tuning SparkTTS primarily improves intelligibility (out-of-domain WER reduced from 3.06 to 2.10), with limited gains in expressiveness.
- Training from scratch improves expressiveness (CtxMOS 3.09) at the cost of intelligibility (WER 9.5).
- Training from scratch on the full dataset (narration + quotations) yields stronger results (CtxMOS 3.30), indicating complementarity between the two data types.
- 67% of quotations in LibriQuote-test are predicted as non-neutral emotion, compared to 91% neutral in LibriHeavy.

## Highlights & Insights

- The paper addresses a TTS data problem from a narratological perspective—fiction audiobooks do not lack expressiveness; what is lacking is the correct segmentation strategy.
- Speech verb/adverb pseudo-labels constitute an extremely natural and low-cost annotation scheme for expressiveness, requiring no manual emotion labeling.
- The high-expressiveness subset $\mathbf{Q}_f$ comprises only 379 hours yet yields substantial improvements, demonstrating that data quality outweighs data quantity.

## Limitations & Future Work

- LibriVox narrators are volunteers, resulting in variable recording quality and expressive performance.
- Coverage is limited to English fiction; other languages and genres are not explored.
- Inference-time utilization of contextual speech verbs/adverbs for controllable style synthesis remains unexplored.

## Related Work & Insights

- **vs. LibriHeavy**: LibriHeavy does not distinguish quotations from narration; the narrative-aware segmentation proposed in this paper reveals expressive signals that have previously been overlooked.
- **vs. EXPRESSO**: EXPRESSO is high quality but contains only tens of hours with 26 predefined styles; LibriQuote provides 5.3K hours of naturally diverse expressiveness.
- **vs. Emotional Speech Datasets**: Discrete emotion labels are overly coarse; speech verbs and adverbs provide finer-grained style descriptions.

## Rating

- Novelty: ⭐⭐⭐⭐ The data construction paradigm combining narrative-aware segmentation with speech verb pseudo-labels is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model, multi-configuration experiments including out-of-domain and human evaluations.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated; the data construction pipeline is described in thorough detail.
- Value: ⭐⭐⭐⭐ The dataset and methodology offer direct utility to the expressive TTS community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Latent Speech-Text Transformer](../../ICLR2026/audio_speech/latent_speech_text_transformer.md)
- [\[AAAI 2026\] DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling](../../AAAI2026/audio_speech/dualspeechlm_towards_unified_speech_understanding_and_generation_via_dual_speech.md)
- [\[ACL 2026\] MSU-Bench: Musical Score Understanding Benchmark](musical_score_understanding_benchmark_evaluating_large_language_models39_compreh.md)
- [\[ICCV 2025\] Understanding Co-speech Gestures in-the-wild](../../ICCV2025/audio_speech/understanding_co-speech_gestures_in-the-wild.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)

</div>

<!-- RELATED:END -->
