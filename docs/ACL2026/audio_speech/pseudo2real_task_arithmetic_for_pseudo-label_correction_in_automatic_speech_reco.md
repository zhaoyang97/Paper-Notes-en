---
title: >-
  [Paper Note] Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition
description: >-
  [ACL 2026][Audio & Speech][Pseudo-label correction] This paper proposes Pseudo2Real, a parameter-space correction method that computes a "correction vector" by taking the weight difference between a ground-truth model an…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Pseudo-label correction"
  - "Task Arithmetic"
  - "Parameter space correction"
  - "Accent adaptation"
  - "Whisper"
date: 2026-05-08
content_hash: fe6c97b30a676631
---

# Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition

**Conference**: ACL 2026 Findings  
**arXiv**: [2510.08047](https://arxiv.org/abs/2510.08047)  
**Code**: None  
**Area**: Speech Processing / Domain Adaptation  
**Keywords**: Pseudo-label correction, Task Arithmetic, Parameter space correction, Accent adaptation, Whisper

## TL;DR

This paper proposes Pseudo2Real, a parameter-space correction method that computes a "correction vector" by taking the weight difference between a ground-truth model and a pseudo-labeled model in the source domain. This vector is applied to a target-domain pseudo-label fine-tuned model to rectify systematic pseudo-label bias, achieving up to a 35% relative WER reduction across ten African accents in AfriSpeech-200.

## Background & Motivation

**Background**: ASR systems often face a scarcity of labeled data when encountering new domains (e.g., new accents). Pseudo-labeling (generating labels with a teacher model) is a common domain adaptation strategy, but pseudo-labels inherit systematic biases from the teacher model.

**Limitations of Prior Work**: (1) Confidence filtering and consistency checks only suppress noise and cannot correct structured bias patterns; (2) Iterative self-training (e.g., Noisy Student) requires multiple training rounds and still propagates repetitive teacher errors; (3) Weight-space methods like EMA use training trajectory averages and do not specifically target pseudo-label bias.

**Key Challenge**: When the target domain lacks ground-truth annotations, how can systematic error patterns in pseudo-labels be identified and corrected?

**Goal**: Design a reusable parameter-space correction method that rectifies pseudo-label bias without requiring target-domain labels.

**Key Insight**: Based on Linear Mode Connectivity—models fine-tuned from the same pre-trained starting point reside in a shared low-loss region, meaning weight differences can be interpreted as meaningful directions rather than noise.

**Core Idea**: The weight difference between a real-label model and a pseudo-label model in the source domain captures the direction of pseudo-label bias. Adding the scaled correction vector to the target-domain pseudo-label model corrects this bias.

## Method

### Overall Architecture

Starting from a pre-trained backbone $\theta^{\text{pre}}$: (1) Fine-tune on the source domain using real and pseudo labels to obtain $\theta_s^{\text{real}}$ and $\theta_s^{\text{pseudo}}$ respectively; (2) Compute the correction vector $\tau = \theta_s^{\text{real}} - \theta_s^{\text{pseudo}}$; (3) Fine-tune on the target domain using pseudo labels to obtain $\theta_t^{\text{pseudo}}$; (4) Apply the correction: $\theta_t^{\text{corrected}} = \theta_t^{\text{pseudo}} + \lambda\tau$.

### Key Designs

1.  **Single Correction Vector (Pseudo2Real)**:
    - **Function**: Captures and corrects systematic biases introduced by pseudo-labels in the source domain.
    - **Mechanism**: The correction vector $\tau = \theta_s^{\text{real}} - \theta_s^{\text{pseudo}}$ encodes the parameter-space direction "from pseudo-labels to real labels." Scaled addition to the target-domain model achieves cross-domain correction.
    - **Design Motivation**: Linear Mode Connectivity ensures that weight differences between models fine-tuned from the same initialization represent meaningful directions. The Task Arithmetic framework demonstrates that such directions can be composed and transferred.

2.  **Sub-group Correction Vectors (Pseudo2Real-SC)**:
    - **Function**: Performs more granular correction for differentiated pseudo-label biases across different speaker sub-groups.
    - **Mechanism**: Uses ECAPA-TDNN to extract speaker embeddings followed by k-means clustering for grouping. A correction vector $\tau_c$ is calculated for each sub-group, and the final correction is the average across all sub-groups: $\theta_t^{\text{corrected}} = \theta_t^{\text{pseudo}} + \frac{\lambda}{C}\sum_{c=1}^{C}\tau_c$.
    - **Design Motivation**: Pseudo-label quality varies by accent, pronunciation, and recording conditions. A uniform correction vector cannot capture fine-grained bias. Sub-group clustering is fully automated and requires no domain labels.

3.  **Cross-Accent Cross-Validation**:
    - **Function**: Comprehensively evaluates the cross-domain transferability of correction vectors.
    - **Mechanism**: Splits 10 accents into two folds (spanning different language families), alternating them as source and target domains.
    - **Design Motivation**: The 10 accents span three major language families (Niger-Congo, Afroasiatic, Indo-European), making cross-accent adaptation highly challenging.

### Loss & Training

Standard ASR fine-tuning loss. Whisper tiny/base/small/medium/large-v2 models are used. Only one parameter $\lambda$ needs tuning during correction.

## Key Experimental Results

### Main Results

**AfriSpeech-200 WER Comparison (Whisper tiny, average across 10 accents)**

| Method | Avg WER |
| :--- | :--- |
| Pre-trained $\theta^{\text{pre}}$ | 106.5 |
| Source Real $\theta_s^{\text{real}}$ | 88.2 |
| Pseudo-label $\theta_t^{\text{pseudo}}$ | 89.3 |
| Confidence Filtering | 88.7 |
| Error Correction (EC) | — |
| **Ours (Pseudo2Real)** | **~58** |
| **Ours (Pseudo2Real-SC)** | **~55** |

### Ablation Study

**Correction Effect Across Different Model Scales**

| Whisper Scale | Pseudo-label WER | +Pseudo2Real WER | Gain (Relative) |
| :--- | :--- | :--- | :--- |
| tiny (39M) | 89.3 | ~58 | ~35% |
| base (74M) | — | — | Consistent Improvement |
| large-v2 (1.55B) | — | — | Diminishing Gains |

### Key Findings

- Pseudo2Real achieves up to a 35% relative WER reduction on Whisper tiny.
- Sub-group clustering (SC) further improves performance, indicating that pseudo-label bias indeed varies by speaker.
- Correction vectors are most effective on smaller models, while larger models exhibit stronger inherent error correction capabilities.
- The method is consistently effective across all 10 accents, even when bridging different language families.
- The optimal value for $\lambda$ is between 0.5 and 1.0 and is relatively insensitive to selection.

## Highlights & Insights

- Extremely simple method—requires only one vector subtraction and one vector addition, with no iterative training.
- The insight that "pseudo-label bias can be parameterized" holds theoretical value—shifting label noise processing from sample space to parameter space.
- Orthogonal and combinable with existing methods such as confidence filtering and iterative self-training.

## Limitations & Future Work

- Requires concurrent availability of real and pseudo labels in the source domain, limiting applicability in entirely unlabeled scenarios.
- The correction vector assumes similar pseudo-label bias patterns between source and target domains; it may fail for significantly divergent domains.
- Evaluated only on accent adaptation; effectiveness in other domain adaptation scenarios like noisy environments or far-field speech remains unverified.
- Linear Mode Connectivity assumptions might not hold under extreme domain shifts.

## Related Work & Insights

- **vs SYN2REAL (Su et al., 2024)**: The latter corrects the acoustic gap between synthetic and real speech, while this work corrects the annotation gap between real and pseudo labels—different problem dimensions.
- **vs Noisy Student**: The latter requires multiple rounds of iterative training, whereas this work requires only a single vector operation.
- **vs EMA**: EMA smoothes training trajectories without specifically targeting pseudo-label bias; this work explicitly captures the "pseudo-to-real" direction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Novel perspective applying Task Arithmetic to pseudo-label correction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 10 accents × 5 model scales × 6 baselines + clustering ablation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear intuition and rigorous experimental design.
- **Value**: ⭐⭐⭐⭐ Simple and effective, suitable for integration with existing pseudo-labeling pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] \[b\] = \[d\] − \[t\] + \[p\]: Self-supervised Speech Models Discover Phonological Vector Arithmetic](bd-tp_self-supervised_speech_models_discover_phonological_vector_arithmetic.md)
- [\[ICLR 2026\] Pay Attention to CTC: Fast and Robust Pseudo-Labelling for Unified Speech Recognition](../../ICLR2026/audio_speech/pay_attention_to_ctc_fast_and_robust_pseudo-labelling_for_unified_speech_recogni.md)
- [\[ACL 2026\] Mind the Pause: Disfluency-Aware Objective Tuning for Multilingual Speech Correction with LLMs](mind_the_pause_disfluency-aware_objective_tuning_for_multilingual_speech_correct.md)
- [\[ACL 2026\] MCGA: A Multi-task Classical Chinese Literary Genre Audio Corpus](mcga_a_multi-task_classical_chinese_literary_genre_audio_corpus.md)
- [\[ACL 2026\] Speech-Hands: A Self-Reflection Voice Agentic Approach to Speech Recognition and Audio Reasoning with Omni Perception](speech-hands_a_self-reflection_voice_agentic_approach_to_speech_recognition_and_.md)

</div>

<!-- RELATED:END -->
