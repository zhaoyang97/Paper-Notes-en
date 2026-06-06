---
title: >-
  [Paper Note] Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion
description: >-
  [ICLR 2026][Audio & Speech][Speech-guided translation] This paper proposes a Speech-guided Machine Translation (SMT) framework that synthesizes source-language speech via TTS and jointly feeds it with text into an MLLM f…
tags:
  - "ICLR 2026"
  - "Audio & Speech"
  - "Speech-guided translation"
  - "Multimodal LLM"
  - "Self-evolution"
  - "TTS"
  - "Multilingual translation"
date: 2026-05-08
content_hash: 280163a3a8800855
---

# Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion

**Conference**: ICLR 2026
**arXiv**: [2602.21646](https://arxiv.org/abs/2602.21646)  
**Code**: [https://github.com/yxduir/LLM-SRT](https://github.com/yxduir/LLM-SRT)  
**Area**: Multimodal Translation / Speech
**Keywords**: Speech-guided translation, Multimodal LLM, Self-evolution, TTS, Multilingual translation

## TL;DR
This paper proposes a Speech-guided Machine Translation (SMT) framework that synthesizes source-language speech via TTS and jointly feeds it with text into an MLLM for translation. A self-evolution mechanism automatically selects beneficial synthetic speech samples for continual training. The approach achieves state-of-the-art performance on Multi30K, surpassing all MMT methods, and attains average SOTA across 108 translation directions on FLORES-200 with only 9B parameters.

## Background & Motivation

Multimodal machine translation has traditionally relied on images for disambiguation (e.g., "bank" is translated differently depending on the visual context). However, image-based MMT has fundamental limitations: (1) multilingual image–text paired data is extremely scarce, with existing datasets covering only a handful of languages such as English, German, and French; and (2) images provide no benefit for general-purpose translation and even introduce noise—experiments show that image-guided translation actually lowers COMET scores on general translation benchmarks.

Speech offers natural advantages: it is semantically aligned with text (a spoken utterance and its transcript share the same content), and cross-lingual speech data covers 100+ languages (e.g., FLEURS, CoVoST-2). More importantly, speech carries prosodic information—stress, intonation, and rhythm—that provides disambiguation cues unavailable in text alone. For example, the rising intonation of a question can guide the model toward the correct translational register.

The key questions are: can TTS-synthesized speech substitute for real speech given the scarcity and high acquisition cost of real speech data, and how can one automatically identify which synthetic speech samples benefit translation rather than introducing noise? These questions motivate the design of the Self-Evolution mechanism.

## Method

### Overall Architecture
Source text → CosyVoice2 speech synthesis → Whisper encoding → Q-Former + MLP alignment to the text space → GemmaX2-28-9B joint processing of speech and text → translation output. A self-evolution mechanism continuously selects positive samples by comparing translation quality with and without speech, and uses only positive samples for continual training.

### Key Designs

1. **Three-stage curriculum learning**: ASR (speech–text mapping) → S2TT (cross-lingual + cross-modal) → SMT (joint input), with progressive module unfreezing.
2. **Self-evolution mechanism**: Speech is synthesized for the source text; COMET scores $S_1$ (text-only) and $S_2$ (speech + text) are computed separately. Samples where $S_2 > S_1$ are treated as positives; only positive samples are used for continual training, iterated until convergence.
3. **TTS synthesis strategy**: Zero-shot multilingual synthesis via CosyVoice2 with random voice cloning from the training set to increase prosodic diversity. Prompt text and predicted duration are strictly aligned with real speech–text pairs to ensure semantic and prosodic consistency.

### Model Architecture

| Component | Parameters | Notes |
|-----------|-----------|-------|
| Whisper encoder | ~635M | Frozen |
| Q-Former + MLP adapter | ~80.5M | Trainable throughout |
| GemmaX2-28-9B (LoRA) | ~9.2B | r=16, alpha=32 |

### Loss & Training
Standard cross-entropy loss. Stage III fine-tunes GemmaX2-28-9B with LoRA (r=16, alpha=32). Training uses 4×A100, AdamW with lr=1e-4, linear warmup for 1K steps followed by linear decay. Self-evolution evaluation uses COMET scores with a fixed reference voice for synthesis. The entire training pipeline completes within one week.

## Key Experimental Results

### Main Results

| Dataset | Metric | SMT-9B | Prev. SOTA (Image) | Gain |
|---------|--------|--------|--------------------|------|
| Multi30K eng→deu | BLEU | 47.0 | 45.3 | +1.7 |
| Multi30K eng→fra | BLEU | 67.0 | 67.5 | −0.5 |
| FLORES-200 eng→27 avg | spBLEU | 40.5 | 39.3 | +1.2 |
| FLORES-200 108 directions | COMET | SOTA | — | Outperforms all baselines |

### Ablation Study

| Configuration | Result | Notes |
|---------------|--------|-------|
| Text only | COMET 87.0 | Baseline |
| + Real speech | COMET 87.8 | Speech is beneficial |
| + Synthetic speech | COMET 87.7 | Nearly equivalent to real speech |
| + Synthetic + Self-evolution | COMET 88.2 | Further improvement after filtering |
| Image-guided | COMET 86.5 | Introduces noise—confirms speech outperforms images |

### Key Findings
- The performance gap between synthetic and real speech is negligible (validated on CoVoST-2), confirming the viability of TTS as a substitute for real speech.
- Self-evolution converges within 1–2 rounds; the positive sample ratio is approximately 60–70%, meaning roughly one-third of synthetic speech samples are actually harmful to translation.
- Low-resource languages benefit more, as prosodic cues provide valuable auxiliary signals when training data is scarce.
- SMT-9B achieves average SOTA across 108 translation directions on FLORES-200, with only 1/67 the parameters of DeepSeek-V3.
- The progressive three-stage curriculum design (ASR→S2TT→SMT) with incremental module unfreezing is effective.
- Prosodic cues contribute most to disambiguating polysemous words—e.g., "lead" can be translated as "铅" or "引导" depending on its pronunciation.

## Highlights & Insights
- Replacing images with speech for MMT represents a pragmatic paradigm shift: speech data scales to 102+ languages, far exceeding the coverage of image–text pairs (typically only English, German, and French).
- Self-evolution elegantly resolves the question of when speech is helpful—not all speech is valuable, and approximately 30–40% of synthetic speech samples introduce noise.
- Prosodic cues are most valuable for polysemous word disambiguation, precisely the scenario where conventional text-based translation struggles most.
- The theoretical framework of the Modality-Agnostic Hypothesis is instructive: any auxiliary modality that provides semantically relevant information and can be aligned to the text space may enhance translation.
- The three-stage curriculum (ASR→S2TT→SMT) enables the model to progress from shallow alignment to deep cross-modal fusion.
- Zero-shot multilingual synthesis with random voice cloning via CosyVoice2 provides prosodic diversity, which proves more effective than using a single synthesized voice.

## Limitations & Future Work
- Inference requires an additional TTS step (CosyVoice2 synthesis), adding approximately 0.5–1s of latency.
- Benefits of speech augmentation are validated only at the 9B scale; whether larger models continue to benefit remains to be explored.
- Biases in the COMET metric may affect positive sample selection in self-evolution—inaccurate COMET scores for certain languages could introduce noise.
- The contribution of prosody across different language types (tonal vs. non-tonal languages) is insufficiently analyzed.
- TTS synthesis quality sets an upper bound on the benefit of speech augmentation; TTS quality for low-resource languages may be inadequate.
- Validation is limited to translation; the effect of speech augmentation on other cross-lingual tasks (e.g., cross-lingual summarization, cross-lingual QA) remains unknown.

## Related Work & Insights
- **vs. image-guided MMT (Soul-Mix, Bridge, etc.)**: Speech data covers far more languages than image–text pairs; SMT outperforms all baselines across 108 directions on FLORES-200.
- **vs. text-only MT (DeepSeek-V3, NLLB-54B)**: SMT-9B uses only 1/67 the parameters of DeepSeek-V3-671B yet achieves superior performance on FLORES-200.
- **vs. speech translation (S2TT)**: S2TT directly translates speech to text, whereas SMT uses speech as an auxiliary modality to enhance text-based translation.
- **Insight**: Speech modality may offer untapped value in other NLP tasks that require prosodic cues, such as sentiment analysis and sarcasm detection.

## Rating
- Novelty: ⭐⭐⭐⭐ — Replacing images with speech for MMT is a pragmatic paradigm shift; the self-evolution mechanism is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Evaluated on Multi30K, FLORES-200 (108 directions), CoVoST-2, and WMT24++.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, intuitive self-evolution diagram, and a theoretically grounded Modality-Agnostic Hypothesis.
- Value: ⭐⭐⭐⭐ — First systematic framework leveraging speech for multilingual machine translation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SPARTA: Scalable and Principled Benchmark of Tree-Structured Multi-hop QA over Text and Tables](sparta_scalable_and_principled_benchmark_of_tree-structured_multi-hop_qa_over_te.md)
- [\[ICLR 2026\] Latent Speech-Text Transformer](latent_speech_text_transformer.md)
- [\[NeurIPS 2025\] EuroSpeech: A Multilingual Speech Corpus](../../NeurIPS2025/audio_speech/eurospeech_a_multilingual_speech_corpus.md)
- [\[ICLR 2026\] VowelPrompt: Hearing Speech Emotions from Text via Vowel-level Prosodic Augmentation](vowelprompt_hearing_speech_emotions_from_text_via_vowel-level_prosodic_augmentat.md)
- [\[ICLR 2026\] TripleSumm: Adaptive Triple-Modality Fusion for Video Summarization](triplesumm_adaptive_triple-modality_fusion_for_video_summarization.md)

</div>

<!-- RELATED:END -->
