---
title: >-
  [Paper Note] Standard-to-Dialect Transfer Trends Differ across Text and Speech: A Case Study on Intent and Topic Classification in German Dialects
description: >-
  [ACL2026][Audio & Speech][Dialect Transfer] This paper systematically compares three transfer paths—text, speech, and ASR cascade—using German-Bavarian intent classification and German-Swiss German topic classification.…
tags:
  - "ACL2026"
  - "Audio & Speech"
  - "Dialect Transfer"
  - "Spoken Language Understanding (SLU)"
  - "Intent Classification"
  - "ASR Cascade"
  - "German Dialects"
date: 2026-05-08
content_hash: d85a3deae4186e2a
---

# Standard-to-Dialect Transfer Trends Differ across Text and Speech: A Case Study on Intent and Topic Classification in German Dialects

**Conference**: ACL2026  
**arXiv**: [2510.07890](https://arxiv.org/abs/2510.07890)  
**Code**: https://github.com/mainlp/dialects-text-vs-speech  
**Area**: Speech / Dialect NLP  
**Keywords**: Dialect Transfer, Spoken Language Understanding (SLU), Intent Classification, ASR Cascade, German Dialects

## TL;DR
This paper systematically compares three transfer paths—text, speech, and ASR cascade—using German-Bavarian intent classification and German-Swiss German topic classification. The study finds that the optimal solution for standard languages does not necessarily apply to dialects: while text models perform best on standard German, speech models are generally more robust for dialect inputs.

## Background & Motivation
**Background**: A common setting in dialect NLP involves using only standard language training data and a small amount of dialect evaluation data. Consequently, researchers typically train standard language text models and transfer them directly to dialect text. This paradigm has revealed significant performance drops in written dialects, particularly because non-standard spelling disrupts subword tokenization.

**Limitations of Prior Work**: Many dialects exist primarily in oral form rather than stable written forms. Studying only text input simplifies away real-world speech issues and ignores a key variable: speech models process continuous acoustic signals and do not rely on fixed vocabularies, potentially making them more robust to inconsistent spelling in dialects.

**Key Challenge**: In high-resource standard language tasks, text models often outperform speech models, and cascaded systems (ASR followed by text classification) are usually reasonable. However, for low-resource, non-standardized dialects, ASR might either "standardize" the dialect into text or produce severe errors. Direct speech models might bypass spelling issues but lack task-specific labeled data.

**Goal**: The authors aim to determine whether the trends for text-only, speech-only, and ASR cascaded input paths are consistent in the same standard-to-dialect transfer task, and whether these trends are reproducible across two content classification tasks: intent classification and topic classification.

**Key Insight**: The paper focuses on German and its closely related dialects, constructing and utilizing as much parallel data as possible across standard/dialect and text/speech modalities. This allows the same content to be compared under different input modalities. The primary task is virtual assistant intent classification, with SwissDial topic classification as an auxiliary task.

**Core Idea**: Instead of defaulting to the assumption that "the strongest input modality for the standard language is also best for dialects," this study treats text, speech, and ASR cascade as three comparable transfer paths to directly measure discrepancies in their trends between standard languages and dialects.

## Method

### Overall Architecture
The paper compares three settings: text-only (classifier trained on German text, tested on German and dialect text); speech-only (speech classifier trained on German audio, tested on German and dialect audio); and cascaded (audio transcribed via ASR, then processed by the standard German text classifier).

The main experiments are based on German training data from MASSIVE / Speech-MASSIVE and test data from xSID (German and Bavarian). The authors recorded xSID-audio to provide speech versions for the same set of German and Bavarian sentences. Auxiliary experiments use SwissDial, where German text is parallel in content to eight Swiss German dialects for topic classification.

### Key Designs
1. **Three-Path Transfer Comparison**:
	- Function: Compares text, speech, and ASR cascade within the same standard-to-dialect transfer framework.
	- Mechanism: All systems are trained or tuned using only standard German and evaluated on standard German vs. dialect inputs. This decouples "input modality differences" from "standard-to-dialect variations."
	- Design Motivation: If only text models were compared, conclusions would be dominated by non-standard spelling; if only ASR were compared, conclusions would be confounded by transcription errors. The three-path design reveals how different modalities alter transfer trends.

2. **Parallel or Comparable Dialect Evaluation Data**:
	- Function: Ensures performance differences stem from linguistic variations and modalities rather than differing label sets or content distributions.
	- Mechanism: In intent classification, MASSIVE labels are mapped to the 10 intent labels used in xSID, resulting in 2.5k/459/361 training, dev, and test instances, with 412 xSID test instances. In topic classification, SwissDial is organized into 10 topics with a 1.5k/194/396 split.
	- Design Motivation: Dialect resources are often fragmented. Inconsistent tasks and labels make cross-modal conclusions difficult to interpret. Label alignment and parallel test sets mitigate these confounding factors.

3. **Fine-grained Comparison of Model Families and ASR Behavior**:
	- Function: Examines whether conclusions are driven by specific model scales or ASR systems.
	- Mechanism: Text models include mBERT, mDeBERTa, and XLM-R (base/large); speech models include mHuBERT, XLS-R, MMS, and various Whisper sizes. Additional ASR models are tested for cascaded systems. Correlation between ASR error rates and classification performance gaps is recorded.
	- Design Motivation: A key phenomenon of ASR in dialects is "normalization"—transcriptions may lean toward standard German. This behavior can sometimes assist text classification or produce nonsensical output, requiring analysis alongside classification results.

### Loss & Training
The study primarily utilizes fine-tuning of pre-trained encoders with classification heads, reporting mean accuracy across three random seeds. Learning rates and epoch counts are selected based on the German training/dev sets. The authors intentionally avoid instruction-tuned text or audio LLMs to prevent the introduction of unknown classification-related instruction data, ensuring variations stem from input representations rather than training corpora.

## Key Experimental Results

### Main Results
The most significant result is not the highest score of a single model, but the trend: text is strongest for standard German; speech-only holds an advantage for dialects; ASR cascade depends on whether the transcription effectively normalizes the dialect.

| Comparison | Standard German Trend | Dialect Trend | Key Figures |
|------------|-----------------------|---------------|-------------|
| Overall Performance Gap | Standard usually higher than dialect | Dialects still show significant drop | Intent gap: 6.1-36.7 pp; Topic gap: 7.9-12.0 pp |
| text-only vs speech-only | Text is typically best | Speech models are generally more robust | Text-speech gap up to 23.8 pp (DE); Speech outperforms text by 14.8 pp (Bavarian) |
| speech-only vs cascaded | Gap near 0 for German | Speech-only comprehensively outperforms cascaded for Bavarian | Speech-only > cascaded by 5.6-17.9 pp (Bavarian) |
| cascaded vs text-only | Cascaded often lower than gold text | Good ASR can make cascaded better than dialect text | Best cascaded near German text-only for Swiss German |

Data and task scales are as follows:

| Data / Task | Language/Dialect | Modality | Scale | Description |
|-------------|------------------|----------|-------|-------------|
| MASSIVE / Speech-MASSIVE | German | Text & Speech | 2.5k / 459 / 361 | Train/Dev/Test for intent classification training |
| xSID / xSID-audio | German, Bavarian | Text & Speech | 412 test instances | First German-Bavarian speech intent dataset |
| SwissDial | German, 8 Swiss dialects | Text & Speech | 1.5k / 194 / 396 | Auxiliary topic classification, 10 topics |

ASR analysis indicates cascaded success depends heavily on transcription quality, especially normalization towards standard German.

| Analysis Item | Result | Meaning |
|---------------|--------|---------|
| Correlation of DE ASR error vs. cascaded-text gap | Pearson r ≈ -0.72 to -0.98 | Closer transcription to standard text leads to cascaded performance nearing text-only |
| Swiss German error rate relative to DE reference | r ≈ -0.94 to -0.99 | Text models handle dialect audio better when normalized to standard German |
| Bavarian correlations | r ≈ -0.85 to -0.92 (DE ref); r ≈ -0.66 to -0.93 (dialect ref) | Both normalization and preservation of dialect info affect classification |
| xSID manual ASR observation | 125 DE/Bavarian samples | Intent keywords are often preserved even if sentences are not fully fluent |

### Ablation Study
This paper is not a modular ablation study but provides an "analytical ablation" of modality selection through comparative settings.

| Configuration | Finding | Explanation |
|---------------|---------|-------------|
| Text-only input | Strongest for standard German; significant drop for dialect text | Standard writing matches pre-trained vocab; dialect spelling variance causes tokenization/lexical issues |
| Speech-only input | Not necessarily strongest for German, but more stable for dialects | Continuous acoustic representations bypass non-standard spelling; dialect differences are primarily phonological |
| Text classification after ASR | Good for German; high variance for dialects | ASR acts as a normalizer or generates noisy text |
| Using ASR-tuned speech encoder | Generally better than non-ASR-tuned models | ASR pre-training helps understand utterance content even if the task isn't transcription |

### Key Findings
- The common "text outperforms speech" conclusion for standard languages does not directly transfer to dialect scenarios.
- For dialect inputs, speech-only systems are more stable than cascaded ones, particularly in Bavarian intent classification.
- The upper limit of cascaded systems is defined by ASR normalization capability, but risks stem from ASR transcription errors in low-resource dialects.
- mBERT is one of the few models including Bavarian in its training corpus, leading to atypical performance on Bavarian text.

## Highlights & Insights
- The paper transforms the question "Should dialect NLP focus on text or speech?" into a measurable problem rather than an intuition-based choice.
- The value of xSID-audio is high: it enables intent classification comparisons across text and speech with identical content, speakers, and linguistic variants.
- ASR is not just a preprocessing module; it rewrites dialect input into a specific text variant, and this normalization is a core part of model behavior.
- For low-resource oral dialects, building speech evaluation sets may be more relevant for real-world applications than expanding written dialect data.

## Limitations & Future Work
- Limited language coverage (mostly German, Bavarian, Swiss German); results may not generalize to languages with greater phonological or script differences.
- Small scale of dialect audio; xSID-audio uses read speech, which differs from spontaneous speech in natural dialogue.
- Cascaded systems were primarily trained on gold text; though ASR-text training trends were similar, complex ASR adaptation remains unexplored.
- Absence of instruction-tuned audio/text LLMs means the study does not address whether end-to-end LLM speech understanding changes these trends.
- Future work could investigate dialect speech understanding in multi-speaker, multi-region, noisy environments, and real-world virtual assistant interactions.

## Related Work & Insights
- **vs. xSID written dialect work**: Previous work focused on standard-to-dialect transfer in text; this paper extends this to speech, showing the bottleneck is not just spelling.
- **vs. Speech-MASSIVE / spoken intent datasets**: These datasets cover multiple languages or standard accents; this paper emphasizes standard vs. non-standard variant differences within a single language.
- **vs. cascaded SLU**: Traditional cascaded SLU assumes better ASR leads to better downstream tasks; this paper suggests that for dialects, "more standardized" ASR output might be more beneficial for text classification than verbatim dialect preservation.
- **vs. token-free dialect models**: While token-free models bypass spelling issues in text, speech-only models bypass the writing system entirely at the input level.
- **Insight**: Dialect applications should not be selected solely based on standard language benchmarks; they must be validated on the target dialect, modality, and task.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Clear problem formulation; valuable data contribution extending dialect transfer to speech.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Comprehensive model families and ASR analysis, though language/scenario scope is limited.
- Writing Quality: ⭐⭐⭐⭐☆ Built around interpretable trends, though some original tables are dense.
- Value: ⭐⭐⭐⭐☆ Directly informs dialect NLP, spoken language understanding, and low-resource data design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] FC-TTS: Style and Timbre Control in Zero-Shot Text-to-Speech with Disentangled Speech Representations](fc-tts_style_and_timbre_control_in_zero-shot_text-to-speech_with_disentangled_sp.md)
- [\[ACL 2026\] Computational Narrative Understanding for Expressive Text-to-Speech](computational_narrative_understanding_for_expressive_text-to-speech.md)
- [\[ACL 2026\] UniSonate: A Unified Model for Speech, Music, and Sound Effect Generation with Text Instructions](unisonate_a_unified_model_for_speech_music_and_sound_effect_generation_with_text.md)
- [\[AAAI 2026\] A Mind Cannot Be Smeared Across Time](../../AAAI2026/audio_speech/a_mind_cannot_be_smeared_across_time.md)
- [\[ICLR 2026\] Latent Speech-Text Transformer](../../ICLR2026/audio_speech/latent_speech_text_transformer.md)

</div>

<!-- RELATED:END -->
