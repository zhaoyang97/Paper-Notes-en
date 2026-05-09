---
title: >-
  [Paper Note] Multimodal In-Context Learning for ASR of Low-Resource Languages
description: >-
  [ACL 2026][Audio & Speech][Multimodal in-context learning] This paper systematically investigates whether multimodal in-context learning (MICL) enables speech LLMs to handle unseen endangered languages, and proposes a MICL-based hypothesis selection system that combines the complementary strengths of acoustic models and speech LLMs, achieving substantial ASR improvements across three endangered languages.
tags:
  - ACL 2026
  - "Audio & Speech"
  - Multimodal in-context learning
  - low-resource ASR
  - speech large language model
  - cross-lingual transfer
  - hypothesis selection
date: 2026-05-08
content_hash: e43379fc5be27024
---

# Multimodal In-Context Learning for ASR of Low-Resource Languages

**Conference**: ACL 2026
**arXiv**: [2601.05707](https://arxiv.org/abs/2601.05707)
**Code**: [github](https://github.com/ZL-KA/MICL)
**Area**: Audio & Speech / Low-Resource ASR
**Keywords**: Multimodal in-context learning, low-resource ASR, speech large language model, cross-lingual transfer, hypothesis selection

## TL;DR

This paper systematically investigates whether multimodal in-context learning (MICL) enables speech LLMs to handle unseen endangered languages, and proposes a MICL-based hypothesis selection system that combines the complementary strengths of acoustic models and speech LLMs, achieving substantial ASR improvements across three endangered languages.

## Background & Motivation

**State of the Field**: Of the 7,000+ languages worldwide, current ASR systems cover only a tiny fraction, with the primary bottleneck being the scarcity of annotated data. Speech LLMs (e.g., Phi4, Qwen3-Omni) possess strong multi-task capabilities, yet their performance remains largely confined to high-resource languages seen during training.

**Limitations of Prior Work**: (1) Existing ICL research focuses predominantly on the text modality and high-resource languages; (2) the effectiveness of multimodal ICL (MICL) for speech LLMs on unseen languages has not been thoroughly studied; (3) directly applying speech LLMs to prompt-based ASR on unseen languages yields extremely poor results (WER > 100%).

**Root Cause**: Although speech LLMs possess powerful in-context learning capabilities, how to leverage these capabilities effectively under data-scarce conditions for endangered languages remains unclear.

**Paper Goals**: To validate the effectiveness of MICL for unseen languages, analyze its internal mechanisms, and construct a practical ASR system.

**Starting Point**: A systematic experimental design is adopted—comparing three modality configurations (text-only ICL, audio+text ICL, and multimodal ICL) and evaluating two speech LLMs on three endangered languages from distinct language families.

**Core Idea**: Although MICL cannot directly enable speech LLMs to produce accurate transcriptions, it can be integrated with acoustic models via hypothesis selection, leveraging MICL's language understanding capability to rerank candidate transcriptions.

## Method

### Overall Architecture

(1) **MICL Analysis**: Three prompting paradigms are designed—T-ICL (text-only), ICL (text + target audio), and MICL (audio–text pairs + target audio)—evaluated using perplexity. (2) **Cross-lingual Fine-Tuning**: Fine-tuning is conducted on 143 auxiliary languages (excluding target languages) to assess transfer effectiveness. (3) **Hypothesis Selection System**: An MMS acoustic model generates N-best candidates, and a speech LLM computes language model scores via MICL to jointly rerank and select the optimal hypothesis.

### Key Designs

1. **Three Prompting Modality Configurations**: T-ICL ($c_i = t_i$, text-only exemplars) measures the contribution of textual context; ICL ($c_i = t_i$ + target audio $a^*$) isolates the effect of the target audio; MICL ($c_i = (a_i, t_i)$ + $a^*$) tests the additional benefit of paired audio–text exemplars. Comparing the three configurations quantifies the marginal contribution of each modality.

2. **Cross-Lingual Instruction Fine-Tuning (XFT)**: MICL instruction fine-tuning is performed on 143 languages from ML-SUPERB 2.0, explicitly excluding the target languages. Only decoder parameters are updated via LoRA, and 1–10 in-context samples are randomly selected during training. The motivation is to teach the model to follow the MICL prompt format and exploit contextual information more effectively, rather than to learn the target languages directly.

3. **MICL Hypothesis Selection System**: Given a 10-best candidate list from MMS, the optimal hypothesis is selected via joint scoring:
$$\hat{h} = \arg\max_{h^{(k)}} \left[\text{Acoustic\_score}(h^{(k)}) + \text{LM\_score}_{MICL}(h^{(k)})\right]$$
where the LM score is the log-likelihood of the speech LLM conditioned on MICL. The design motivation is that acoustic models excel at basic recognition while speech LLMs excel at contextual understanding, making the two complementary.

### Loss & Training

During fine-tuning, the loss is computed only over the target transcription tokens, with in-context exemplars serving as conditional inputs. LoRA adapters are used with all other parameters frozen. Evaluation metrics are perplexity (PPL, used for configuration selection) and word error rate (WER, used for final evaluation).

## Key Experimental Results

### Main Results (Qwen3-Omni Perplexity, Pre-trained Model)

| Language | Setting | 0-shot | 1-shot | 5-shot | 10-shot | 50-shot | 100-shot |
|---|---|---|---|---|---|---|---|
| Khinalug | T-ICL | 1302 | 289 | 69 | 57 | 44 | 43 |
| Khinalug | ICL | 54 | 28 | 11 | 10 | 11 | 15 |
| Khinalug | MICL | 58 | 30 | 9 | 10 | 8 | 13 |
| Kichwa | ICL | 18 | 10 | 5 | 4 | 3 | 3 |
| Kichwa | MICL | 17 | 7 | 4 | 4 | 3 | 4 |
| Mboshi | ICL | 178 | 51 | 21 | 16 | 10 | 9 |
| Mboshi | MICL | 189 | 34 | 13 | 10 | 7 | 9 |

### Hypothesis Selection WER Results

| Model | Khinalug | Kichwa | Mboshi |
|---|---|---|---|
| Acoustic Model (MMS) | 42.1 | 17.3 | 31.4 |
| Phi4 ASR-FT | 41.5 | 17.4 | 29.9 |
| Phi4 XFT | 41.0 | 17.1 | 29.6 |
| Phi4 TFT | 40.8 | 16.6 | 28.6 |
| Qwen3-Omni | 40.7 | 17.2 | 30.0 |
| N-gram LM | 39.6 | 17.7 | 30.6 |
| Oracle | 36.5 | 12.4 | 22.1 |

### Key Findings

- MICL enables both speech LLMs to learn unseen languages, with perplexity consistently decreasing as the number of in-context examples increases.
- Qwen3-Omni continues to benefit from audio exemplars in long-context settings (100-shot), whereas Phi4 benefits primarily in short-context settings (≤3 shots).
- Attention analysis reveals that the models allocate more attention to text (65–70%) than to audio (30–35%), exhibiting a layer-dependent pattern.
- Cross-lingual fine-tuning approaches target-language fine-tuning performance on Kichwa, suggesting that linguistic diversity enhances generalization.

## Highlights & Insights

- **MICL is effective for unseen languages**: This is the first systematic demonstration that speech LLMs can learn uncovered endangered languages through multimodal in-context learning.
- **Attention mechanism analysis**: A layer-dependent modality preference pattern is identified—shallow and deep layers favor audio, while intermediate layers favor text.
- **Practical system design**: The hypothesis selection system combining an acoustic model and a speech LLM is simple yet effective, requiring no end-to-end training.

## Limitations & Future Work

- Due to computational constraints, cross-lingual fine-tuning is conducted only on Phi4.
- The number of in-context samples during fine-tuning is capped at 1–10, which may limit performance in long-context settings.
- The three endangered languages each have very limited data (2–4 hours), and the generalizability of the findings requires further validation.
- Future work could explore larger-scale cross-lingual instruction fine-tuning and evaluation on a broader set of endangered languages.

## Related Work & Insights

- This work extends text-based ICL for low-resource languages (Li & Niehues, 2025b) to the multimodal setting.
- The hypothesis selection framework is generalizable to other low-resource multimodal tasks.
- The attention analysis findings are consistent with the text-bias phenomenon observed in vision LLMs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First systematic study of MICL for endangered language ASR, with a distinctive perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage across 3 languages × 2 models × multiple ICL configurations × attention analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Experimental design is clear and systematic, with rigorous comparative logic across configurations.
- **Value**: ⭐⭐⭐⭐ Provides a novel technical pathway for ASR of endangered languages with meaningful societal impact.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Hard to Be Heard: Phoneme-Level ASR Analysis of Phonologically Complex, Low-Resource Endangered Languages](hard_to_be_heard_phoneme-level_asr_analysis_of_phonologically_complex_low-resour.md)
- [\[ACL 2026\] Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective](learning_invariant_modality_representation_for_robust_multimodal_learning_from_a.md)
- [\[ACL 2026\] Music Audio-Visual Question Answering Requires Specialized Multimodal Designs](music_audio-visual_question_answering_requires_specialized_multimodal_designs.md)
- [\[NeurIPS 2025\] A Multi-Task Benchmark for Abusive Language Detection in Low-Resource Settings](../../NeurIPS2025/audio_speech/a_multitask_benchmark_for_abusive_language_detection_in_lowr.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)

<!-- RELATED:END -->
