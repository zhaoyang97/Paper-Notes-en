---
title: >-
  [Paper Note] Hard to Be Heard: Phoneme-Level ASR Analysis of Phonologically Complex, Low-Resource Endangered Languages
description: >-
  [ACL 2026][Audio & Speech][ASR] This paper presents a phoneme-level ASR analysis of two extremely phonologically complex, low-resource endangered East Caucasian languages (Archi and Rutul)…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "ASR"
  - "low-resource"
  - "endangered languages"
  - "phoneme-level analysis"
  - "East Caucasian"
  - "wav2vec2"
  - "Whisper"
  - "frequency effect"
date: 2026-05-08
content_hash: 95265276804fad63
---

# Hard to Be Heard: Phoneme-Level ASR Analysis of Phonologically Complex, Low-Resource Endangered Languages

**Conference**: ACL 2026
**arXiv**: [2604.18204](https://arxiv.org/abs/2604.18204)
**Code**: [GitHub](https://github.com/mahesh-ak/north_caucasian_asr) | [Data](https://huggingface.co/datasets/mahesh27/archi_rutul_asr)
**Area**: Speech Recognition / Low-Resource Endangered Languages
**Keywords**: ASR, low-resource, endangered languages, phoneme-level analysis, East Caucasian, wav2vec2, Whisper, frequency effect

## TL;DR

This paper presents a phoneme-level ASR analysis of two extremely phonologically complex, low-resource endangered East Caucasian languages (Archi and Rutul), finding that phoneme recognition accuracy follows an S-shaped learning curve with respect to training frequency, and that many errors attributed to phonological complexity are in fact primarily caused by data scarcity.

## Background & Motivation

**Background**: ASR research is predominantly focused on high-resource languages and evaluated at the word or character level. Systematic ASR benchmarks and phoneme-level behavioral analyses are lacking for typologically extreme languages. Archi has 16 vowels and 73–81 consonant phonemes (one of the largest consonant inventories among non-click languages), while Rutul also features a large consonant inventory and distinctive articulations.

**Limitations of Prior Work**: (1) No established ASR benchmarks or standardized resources exist for Archi or Rutul; (2) existing ASR studies rarely analyze behavior at the phoneme level, particularly for phonologically complex languages; (3) original annotations heterogeneously mix IPA, romanization, and Cyrillic scripts, making them unsuitable for direct training use; (4) it remains unclear whether ASR errors stem from phonological complexity or data scarcity.

**Key Challenge**: When a language simultaneously exhibits extreme phonological complexity and extreme data scarcity, which factor should ASR failures be attributed to? If complexity is the issue, better model architectures are needed; if data is the issue, more data collection is required.

**Goal**: To compile standardized ASR resources for Archi and Kina Rutul, systematically evaluate multiple state-of-the-art models, and reveal the true sources of errors through phoneme-level analysis.

**Key Insight**: Using the phoneme as the unit of analysis, the paper establishes a quantitative functional relationship between phoneme recognition performance and training frequency.

**Core Idea**: Phoneme recognition F1 follows an S-shaped (logistic) function of log training frequency—near zero for extremely rare phonemes, rising sharply once a threshold is reached, and saturating at high frequencies—indicating that data scarcity is the primary bottleneck rather than phonological complexity.

## Method

**Overall Architecture**: Data curation and standardization (unified into IPA) → multi-model evaluation (wav2vec2 variants / Whisper / Qwen2-Audio / gpt-4o) → phoneme-level error analysis → frequency–performance relationship modeling.

**Key Designs**:

1. **Language-Specific Phoneme Vocabulary with Heuristic Average Initialization (w2v2l-custom-avg)**
    - **Function**: Defines a target-language-appropriate output vocabulary for wav2vec2 and handles composite phonemes.
    - **Mechanism**: Maps composite phonemes (e.g., labialized $k^w$, pharyngealized sounds) to single tokens rather than subsequences. Output layer parameters are initialized by averaging the pre-trained parameters of the constituent IPA symbols: $W_{*i} = \frac{1}{k}\sum W_{*i_j}^{\text{old}},\ b_i = \frac{1}{k}\sum b_{i_j}^{\text{old}}$. This even enables zero-shot evaluation.
    - **Design Motivation**: Standard tokenizers decompose composite phonemes into sequences (e.g., $k^w \to$ 'k', 'w'), losing phoneme integrity. Average initialization provides new tokens with meaningful starting representations, avoiding learning from scratch.

2. **Word-Level $n$-gram Language Model Integration (w2v2l-custom-avg-lm)**
    - **Function**: Leverages linguistic constraints to reduce word error rate.
    - **Mechanism**: Integrates a word-level 3-gram language model over CTC outputs, jointly optimizing via beam search: $\sum \log p_{\text{ctc}}(x_i) + \beta \cdot m(X) + \alpha \cdot \sum \log p_{\text{lm}}(w_i | w_{i-1}, \ldots, w_{i-n})$, implemented with KenLM.
    - **Design Motivation**: Unlike prior work using character- or phoneme-level $n$-grams, word-level LMs more effectively constrain the decoding space in extremely low-resource settings.

3. **S-Shaped Frequency–Performance Relationship Modeling**
    - **Function**: Quantifies and separates the contributions of data scarcity and phonological complexity.
    - **Mechanism**: Fits a logistic function $f(x) = \frac{L}{1 + \exp(-k(x - x_0))}$ to F1 as a function of $\log_{10}(\text{training frequency})$, where $L$ is the asymptotic F1, $k$ is the slope, and $x_0$ is the midpoint. Parameters are estimated via the Levenberg–Marquardt algorithm; $R^2$ quantifies goodness of fit; 95% confidence intervals are derived via the Delta method.
    - **Design Motivation**: If performance is largely explained by frequency (high $R^2$), complexity is not the primary cause; individual points deviating from the S-curve suggest model-specific generalization effects.

## Key Experimental Results

**Main Results (ASR error rates, lower is better)**:

| Model | Parameters | Archi WER/PER | Rutul WER/PER |
|-------|-----------|---------------|---------------|
| gpt-4o-transcribe (zero-shot) | — | 0.982 / 0.436 | 0.994 / 0.514 |
| wav2vec2-large-ipa | 0.3B | 0.559 / 0.135 | 0.795 / 0.220 |
| w2v2l-custom-avg (Ours) | 0.3B | 0.479 / 0.122 | 0.725 / **0.195** |
| w2v2l-custom-avg-lm (Ours) | 0.3B | **0.465** / 0.122 | **0.697** / 0.206 |
| w2v2l-custom-cpy1 | 0.3B | 0.462 / 0.123 | 0.738 / 0.203 |
| whisper-large-v3 | 1.5B | 0.402 / **0.107** | 0.778 / 0.251 |
| Qwen2-Audio-7B | 8.4B | 0.579 / 0.180 | 0.778 / 0.239 |
| Qwen2.5-Omni-7B | 10.8B | 0.705 / 0.199 | 0.852 / 0.257 |

**Initialization Strategy Comparison (PER)**:

| Initialization | Archi | Rutul |
|---------------|-------|-------|
| Random (custom) | 0.147 | 0.222 |
| Copy (cpy1) | 0.123 | 0.203 |
| Average (avg, Ours) | **0.122** | **0.195** |

**Key Findings**:
- **Proposed method is competitive with Whisper**: w2v2l-custom-avg (0.3B parameters) achieves PER 0.195 on Rutul, outperforming Whisper (1.5B, PER 0.251) with 5× fewer parameters.
- **gpt-4o zero-shot completely fails**: WER approaches 1.0, demonstrating that unfineted general-purpose models are unusable on extreme languages.
- **S-shaped relationship is robust**: Strong S-shaped relationships between F1 and log training frequency are observed for most model–language pairs.
- **Whisper anomaly on Archi**: Whisper partially deviates from the S-curve on Archi, suggesting that multilingual pre-training encodes phonological knowledge beyond what frequency alone predicts.
- **Weak correlation with complexity**: Pearson correlations between phoneme-class F1 and complexity measures are weak (mostly between −0.1 and −0.5), weakening further after controlling for frequency.
- **Average initialization improves even zero-shot performance**: CER decreases from 0.593 to 0.544 on Archi, indicating that initialization itself carries useful cross-lingual information.

## Highlights & Insights

- **Causal attribution breakthrough**: The S-shaped fitting elegantly decouples "phonological complexity" from "data scarcity"—if performance is explained by frequency, complexity is not the primary cause.
- **First ASR benchmark for East Caucasian languages**: Establishes a reproducible evaluation framework for two endangered languages that previously lacked any ASR resources.
- **Simplicity and effectiveness of average initialization**: By simply averaging the weights of constituent symbols, the method provides an effective warm start for composite phonemes without requiring additional data.
- **Practical low-resource strategy**: Demonstrates that a fine-tuned 0.3B-parameter model trained on 45–75 minutes of data can compete with a 1.5B-parameter model.

## Limitations & Future Work

- Datasets are extremely small (Archi: 45 minutes / 2 speakers; Rutul: 75 minutes / ~15 speakers), limiting statistical power.
- Archi data consists of read speech while Rutul data is spontaneous speech, introducing substantial condition differences.
- The sigmoid relationship is descriptive rather than theoretical; other functional forms may be equally plausible.
- Data augmentation and semi-supervised methods are not explored.
- Future work should extend to additional East Caucasian languages and other phonologically complex languages.

## Related Work & Insights

- **Taguchi et al. (2023)**: wav2vec2-large-ipa multilingual IPA pre-trained model, serving as the primary baseline in this work.
- **Yusuyin et al. (2025)**: Phoneme initialization strategy (copying base phoneme weights); this paper proposes a superior average initialization.
- **Boulianne (2022)**: Minutes of data combined with multilingual pre-training can yield useful phoneme recognizers.
- **Cognitive science frequency effects**: Logistic functions describing log-frequency–performance relationships have analogues in cognitive models.
- **Insights**: (1) The bottleneck in low-resource ASR lies in data quantity rather than linguistic complexity; (2) language-specific vocabularies combined with intelligent initialization are key to efficient fine-tuning; (3) phoneme-level evaluation is more diagnostically informative than word- or character-level evaluation.

## Rating

- **Novelty**: ★★★★☆ — First systematic ASR analysis targeting East Caucasian languages; the S-shaped finding is meaningful.
- **Experimental Thoroughness**: ★★★★☆ — Broad model coverage and rich analytical dimensions, though data size limits statistical reliability.
- **Writing Quality**: ★★★★☆ — Technically rigorous and scientifically sound.
- **Value**: ★★★★☆ — Offers direct practical guidance for endangered language speech technology and low-resource ASR.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Multimodal In-Context Learning for ASR of Low-Resource Languages](multimodal_in-context_learning_for_asr_of_low-resource_languages.md)
- [\[ACL 2026\] TellWhisper: Tell Whisper Who Speaks When](tellwhisper_tell_whisper_who_speaks_when.md)
- [\[ACL 2026\] Beyond Transcription: Unified Audio Schema for Perception-Aware AudioLLMs](beyond_transcription_unified_audio_schema_for_perception-aware_audiollms.md)
- [\[ACL 2026\] Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition](pseudo2real_task_arithmetic_for_pseudo-label_correction_in_automatic_speech_reco.md)
- [\[AAAI 2026\] PSA-MF: Personality-Sentiment Aligned Multi-Level Fusion for Multimodal Sentiment Analysis](../../AAAI2026/audio_speech/psa-mf_personality-sentiment_aligned_multi-level_fusion_for_multimodal_sentiment.md)

</div>

<!-- RELATED:END -->
