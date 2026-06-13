---
title: >-
  [Paper Note] Speech-Hands: A Self-Reflection Voice Agentic Approach to Speech Recognition and Audio Reasoning with Omni Perception
description: >-
  [ACL2026][Audio & Speech][Speech Recognition] Speech-Hands is proposed as a learnable speech agent framework that decides whether to trust its own perception or external ASR hypotheses by generating explicit action token…
tags:
  - "ACL2026"
  - "Audio & Speech"
  - "Speech Recognition"
  - "Audio Reasoning"
  - "Multimodal Agent"
  - "Self-Reflection"
  - "Generative Error Correction"
date: 2026-05-08
content_hash: 8cf5aaa468e60609
---

<!-- Generated automatically by src/gen_stubs.py -->
# Speech-Hands: A Self-Reflection Voice Agentic Approach to Speech Recognition and Audio Reasoning with Omni Perception

**Conference**: ACL2026 Oral  
**arXiv**: [2601.09413](https://arxiv.org/abs/2601.09413)
**Code**: [GitHub](https://YukinoWan.github.io/Speech-Hands/)
**Area**: audio_speech
**Keywords**: Speech Recognition, Audio Reasoning, Multimodal Agent, Self-Reflection, Generative Error Correction

## TL;DR

Speech-Hands is proposed as a learnable speech agent framework that decides whether to trust its own perception or external ASR hypotheses by generating explicit action tokens (`<internal>`/`<external>`/`<rewrite>`) at inference time. It achieves an average WER reduction of 12.1% across 7 benchmarks on the OpenASR leaderboard and reaches 77.37% accuracy in Audio QA.

## Background & Motivation

Omni-multimodal models (e.g., Qwen2.5-Omni) can process audio and text simultaneously. However, a critical and counter-intuitive finding is that naively fine-tuning omni-models to fuse speech recognition and external sound understanding tasks often **degrades performance**. Preliminary experiments show that using Qwen2.5-Omni for Generative Error Correction (GER) on Whisper's N-best hypotheses leads to WER deterioration (8.52%-9.05%) across 7 ASR benchmarks. Further zero-shot analysis reveals that foundation models lack intrinsic arbitration capabilities—their decisions are highly sensitive to prompt phrasing rather than the correct answer. This indicates a need for an **explicit self-reflection mechanism** that allows the model to learn when to trust itself and when to seek external help.

## Method

### Overall Architecture

Speech-Hands models speech understanding as an agentic decision process. Given an input audio $A$ and an optional query $Q$, the model first generates its own response $H_{omni}$ (internal perception) while obtaining a response $H_{ext}$ from an external model. Then, based on the full context $(A, Q, H_{omni}, H_{ext})$, the model generates an explicit action token to guide the subsequent generation strategy: `<internal>` to trust itself, `<external>` to adopt the external result, or `<rewrite>` for fusion-based rewriting.

### Key Designs

1.  **ASR Action Token Construction**: For each training sample, the WER of the internal transcription $T_{int}$, external transcription $T_{ext}$, and GER-fused transcription $T_{ger}$ are calculated. If $T_{int}$ is identical to the ground truth ($WER=0$) or has the lowest WER, it is labeled `<internal>`; if $T_{ext}$ is optimal, it is labeled `<external>`; if $T_{ger}$ is optimal, it is labeled `<rewrite>`. This fine-grained WER-based labeling provides a strong supervision signal.
2.  **Audio QA Action Token Construction**: Since QA provides discrete correct/incorrect signals, a multi-sampling stability strategy is introduced. The external model is sampled 5 times, and a majority vote determines the `<external>` or `<rewrite>` label, reducing the impact of external prediction randomness on the decision boundary.
3.  **Unified End-to-End Training**: Each training sample is formatted as "action token + target text." Action selection and subsequent generation are jointly supervised via a single cross-entropy loss, enabling the model to internalize the mapping from multimodal evidence to action selection.

### Loss & Training

-   Standard cross-entropy loss is used to jointly optimize the action token and the target sequence.
-   Fine-tuning is based on Qwen2.5-Omni for 5 epochs with a batch size of 64, a learning rate of $1e-4$ (cosine decay), and fp16 training.
-   A maximum of 20,000 training samples per dataset are used (limited by inference computation).

## Key Experimental Results

### Main Results

ASR Task (7 OpenASR datasets, WER%):

| Method | AMI | Tedlium | GigaSpeech | SPGISpeech | VoxPopuli | Libri-clean | Libri-other | Average WER↓ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Whisper-v2-large | 16.88 | 4.32 | 11.45 | 3.94 | 7.57 | 2.91 | 5.15 | 7.17 |
| Qwen2.5-Omni | 19.77 | 5.17 | 11.26 | 4.58 | 6.59 | 2.09 | 3.85 | 7.33 |
| Phi-4-MM | 11.69 | 2.90 | 9.78 | 3.13 | 5.93 | 1.68 | 3.83 | 6.14 |
| GER ⇒ Whisper | 23.44 | 6.15 | 12.15 | 3.94 | 7.53 | 2.97 | 4.89 | 8.44 |
| **Speech-Hands ⇌ parakeet** | **11.20** | **4.37** | **11.10** | **2.26** | **6.02** | **1.67** | **3.18** | **5.69** |

Audio QA Task (Accuracy%):

| Method | Bio-acoustic | Soundscape | Complex QA | Average Acc↑ |
| :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-Omni | 47.32 | 56.32 | 59.89 | 57.87 |
| AudioFlamingo 3 | 71.88 | 57.31 | 81.26 | 74.49 |
| **Speech-Hands + majority** | **81.25** | **59.4** | **85.7** | **77.37** |

### Ablation Study

| Experiment Content | Key Findings |
| :--- | :--- |
| Prompt Ablation (GER SFT) | All prompt strategies failed (WER 8.44-9.05), proving implicit fusion is infeasible. |
| Zero-shot Arbitration | Model decisions are sensitive to prompt phrasing rather than the correct answer (validated by confusion matrix). |
| Action token F1 | `<internal>` F1 > 0.8 (most datasets), `<external>` F1 0.65-0.89, `<rewrite>` F1 < 0.4 (limited by data sparsity). |
| Training Data Size | Outperforms full-training baselines with only 20k samples per dataset. |

### Key Findings

-   Cascaded GER (ASR followed by LLM correction) is **consistently inferior** to the original ASR, whereas the parallel agentic architecture of Speech-Hands is **consistently superior** to both baselines.
-   Despite the extreme sparsity of the `<rewrite>` label (<2%), the model maintains high precision when triggered, reflecting cautious but reliable rewrite detection.
-   On AMI (meeting speech, the nosiest scenario), Speech-Hands reduces the WER of Qwen2.5-Omni from 19.77% to 11.20%, a 43% Gain.

## Highlights & Insights

-   **Core Insight**: The fundamental problem of multimodal models is not insufficient perception capacity, but a lack of a mechanism to arbitrate between multiple information sources. Explicit action tokens transform implicit information fusion into an interpretable decision process.
-   **"Knowing what one doesn't know"**: The framework analogies the self-reflection ability in developmental psychology, evolving from an egocentric perspective to a stage where it can "step outside its own thinking" to evaluate the reliability of its beliefs.
-   **Natural Generalization**: The transition from ASR to Audio QA requires no architectural modifications, only an adjustment of the action token construction strategy.

## Limitations & Future Work

-   Training data for the `<rewrite>` action is extremely sparse, leading to low F1; data augmentation strategies are needed.
-   Currently, only Qwen2.5-Omni is used as the backbone; generalization to other omni-models remains to be verified.
-   Tool-calling actions (e.g., calling external APIs) have not yet been implemented and represent a future direction.
-   Multilingual ASR scenarios have not been explored.

## Related Work & Insights

-   **Generative Error Correction (GER)** (Yang et al., 2023): Pure text-based cascaded correction cannot utilize raw audio; Ours proves this is a fundamental limitation of "non-agentic" approaches.
-   **Qwen2.5-Omni / Phi-4-MM**: Current state-of-the-art omni-models still lack explicit arbitration mechanisms.
-   **Self-Reflection** (Madaan et al., 2023): Existing reflection methods intervene only after perceptual fusion; the innovation of Speech-Hands lies in reflecting on the **perceptual behavior itself**.
-   Insight: The action token approach can be extended to any multimodal task requiring arbitration between multiple information sources.

## Rating

| Dimension | Score (1-10) |
| :--- | :--- |
| Novelty | 8 |
| Experimental Thoroughness | 8 |
| Writing Quality | 7 |
| Value | 8 |
| Total Score | 7.8 |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] VAPO: End-to-end Slide-Enhanced Speech Recognition with Omni-modal Large Language Models](vapo_end-to-end_slide-enhanced_speech_recognition_with_omni-modal_large_language.md)
- [\[ACL 2026\] An Exploration of Mamba for Speech Self-Supervised Models](an_exploration_of_mamba_for_speech_self-supervised_models.md)
- [\[ACL 2026\] \[b\] = \[d\] − \[t\] + \[p\]: Self-supervised Speech Models Discover Phonological Vector Arithmetic](bd-tp_self-supervised_speech_models_discover_phonological_vector_arithmetic.md)
- [\[ACL 2026\] Beyond Transcription: Unified Audio Schema for Perception-Aware AudioLLMs](beyond_transcription_unified_audio_schema_for_perception-aware_audiollms.md)
- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](closing_the_modality_reasoning_gap_for_speech_large_language_models.md)

</div>

<!-- RELATED:END -->
