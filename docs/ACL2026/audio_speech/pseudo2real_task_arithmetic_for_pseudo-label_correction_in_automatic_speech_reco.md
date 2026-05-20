---
title: >-
  [Paper Note] Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition
description: >-
  [ACL 2026][Audio & Speech][pseudo-label correction] This paper proposes Pseudo2Real, a parameter-space correction method that computes a "correction vector" as the weight difference between a real-label model and a pseud…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "pseudo-label correction"
  - "task arithmetic"
  - "parameter-space correction"
  - "accent adaptation"
  - "Whisper"
date: 2026-05-08
content_hash: 60f16a28f23404a7
---

# Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition

**Conference**: ACL 2026
**arXiv**: [2510.08047](https://arxiv.org/abs/2510.08047)  
**Code**: N/A  
**Area**: Speech Processing / Domain Adaptation
**Keywords**: pseudo-label correction, task arithmetic, parameter-space correction, accent adaptation, Whisper

## TL;DR

This paper proposes Pseudo2Real, a parameter-space correction method that computes a "correction vector" as the weight difference between a real-label model and a pseudo-label model trained on a source domain, then applies this vector to a pseudo-label fine-tuned model on the target domain to rectify systematic pseudo-label bias. The method achieves up to 35% relative WER reduction across ten African accents in AfriSpeech-200.

## Background & Motivation

**Background**: ASR systems face scarce annotated data when encountering new domains (e.g., novel accents). Pseudo-labeling—generating labels with a teacher model—is a common domain adaptation strategy, but pseudo-labels inherit systematic biases from the teacher model.

**Limitations of Prior Work**: (1) Confidence filtering and consistency checks suppress noise but cannot correct structured bias patterns; (2) iterative self-training (e.g., Noisy Student) requires multiple training rounds and still propagates the teacher's repeated errors; (3) weight-space methods such as EMA average over training trajectories without targeting pseudo-label bias specifically.

**Key Challenge**: When no ground-truth annotations exist in the target domain, how can systematic error patterns in pseudo-labels be identified and corrected?

**Goal**: Design a reusable parameter-space correction method that corrects pseudo-label bias without requiring target-domain labels.

**Key Insight**: Linear mode connectivity—models fine-tuned from the same pre-trained initialization reside in a shared low-loss basin, so weight differences can be interpreted as meaningful directions rather than noise.

**Core Idea**: The weight difference between a real-label model and a pseudo-label model trained on the source domain captures the direction of pseudo-label bias; adding a scaled correction vector to the target-domain pseudo-label model rectifies this bias.

## Method

### Overall Architecture

Starting from a pre-trained backbone $\theta^{\text{pre}}$: (1) fine-tune on the source domain with real labels and pseudo-labels to obtain $\theta_s^{\text{real}}$ and $\theta_s^{\text{pseudo}}$, respectively; (2) compute the correction vector $\tau = \theta_s^{\text{real}} - \theta_s^{\text{pseudo}}$; (3) fine-tune on the target domain with pseudo-labels to obtain $\theta_t^{\text{pseudo}}$; (4) apply the correction $\theta_t^{\text{corrected}} = \theta_t^{\text{pseudo}} + \lambda\tau$.

### Key Designs

1. **Single Correction Vector (Pseudo2Real)**:

    - **Function**: Captures and corrects systematic bias introduced by pseudo-labels in the source domain.
    - **Mechanism**: The correction vector $\tau = \theta_s^{\text{real}} - \theta_s^{\text{pseudo}}$ encodes the parameter-space direction from pseudo-labels to real labels. Scaling it and adding it to the target-domain model achieves cross-domain correction.
    - **Design Motivation**: Linear mode connectivity guarantees that weight differences between models fine-tuned from the same initialization represent meaningful directions; the task arithmetic framework demonstrates that such directions can be composed and transferred.

2. **Sub-group Correction Vector (Pseudo2Real-SC)**:

    - **Function**: Provides finer-grained correction for differential pseudo-label bias across speaker sub-groups.
    - **Mechanism**: Speaker embeddings are extracted with ECAPA-TDNN and grouped via k-means clustering. A separate correction vector $\tau_c$ is computed per sub-group, and the final correction averages across all sub-groups: $\theta_t^{\text{corrected}} = \theta_t^{\text{pseudo}} + \frac{\lambda}{C}\sum_{c=1}^{C}\tau_c$.
    - **Design Motivation**: Pseudo-label quality varies with accent, pronunciation, and recording conditions; a single correction vector cannot capture fine-grained bias. Sub-group clustering requires no domain labels and is fully automatic.

3. **Cross-accent Cross-fold Validation**:

    - **Function**: Comprehensively evaluates the cross-domain transferability of the correction vector.
    - **Mechanism**: Ten accents are split into two folds (spanning different language families) that alternately serve as source and target domains.
    - **Design Motivation**: The ten accents span the Niger-Congo, Afro-Asiatic, and Indo-European families, making cross-accent adaptation highly challenging.

### Loss & Training

Standard ASR fine-tuning loss. Five model scales of Whisper are used: tiny/base/small/medium/large-v2. Correction requires tuning only a single scalar $\lambda$.

## Key Experimental Results

### Main Results

**AfriSpeech-200 WER Comparison (Whisper tiny, average over 10 accents)**

| Method | Avg. WER |
|--------|----------|
| Pre-trained $\theta^{\text{pre}}$ | 106.5 |
| Source real $\theta_s^{\text{real}}$ | 88.2 |
| Pseudo-label $\theta_t^{\text{pseudo}}$ | 89.3 |
| Confidence filtering | 88.7 |
| Error correction (EC) | — |
| **Pseudo2Real** | **~58** |
| **Pseudo2Real-SC** | **~55** |

### Ablation Study

**Correction Effect across Model Scales**

| Whisper Scale | Pseudo-label WER | +Pseudo2Real WER | Relative Reduction |
|---------------|-----------------|------------------|--------------------|
| tiny (39M) | 89.3 | ~58 | ~35% |
| base (74M) | — | — | consistent gain |
| large-v2 (1.55B) | — | — | diminishing gain |

### Key Findings

- Pseudo2Real achieves up to 35% relative WER reduction on Whisper tiny.
- Sub-group clustering (SC) yields further improvements, confirming that pseudo-label bias varies across speakers.
- The correction vector is most effective on smaller models; larger models possess stronger intrinsic error-correction capacity.
- Consistent gains are observed across all ten accents, even across different language families.
- The optimal $\lambda$ falls in the range 0.5–1.0 and is relatively insensitive to exact selection.

## Highlights & Insights

- The method is remarkably simple—requiring only one vector subtraction and one vector addition, with no iterative training.
- The insight that "pseudo-label bias can be parameterized" carries theoretical value: it shifts the handling of label noise from sample space to parameter space.
- The approach is orthogonal to existing methods such as confidence filtering and iterative self-training and can be combined with them.

## Limitations & Future Work

- The method requires both real labels and pseudo-labels in the source domain, limiting applicability in fully unsupervised settings.
- The correction vector assumes that pseudo-label bias patterns are similar between source and target domains, which may fail when domain gaps are large.
- Evaluation is limited to accent adaptation; generalization to other domain adaptation scenarios (e.g., noisy environments, far-field speech) remains unverified.
- The linear mode connectivity assumption may not hold under extreme domain discrepancy.

## Related Work & Insights

- **vs. SYN2REAL (Su et al., 2024)**: SYN2REAL corrects the acoustic gap between synthetic and real speech, whereas Pseudo2Real corrects the annotation gap between real labels and pseudo-labels—the two methods address different problem dimensions.
- **vs. Noisy Student**: Noisy Student requires multiple rounds of iterative training; Pseudo2Real requires only a single vector operation.
- **vs. EMA**: EMA smooths training trajectories without targeting pseudo-label bias explicitly; Pseudo2Real directly captures the "pseudo-to-real" direction in parameter space.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Applying task arithmetic to pseudo-label correction is a novel perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 10 accents × 5 model scales × 6 baselines + clustering ablation.
- **Writing Quality**: ⭐⭐⭐⭐ — Methodological intuition is clear and experimental design is rigorous.
- **Value**: ⭐⭐⭐⭐ — Simple and effective; readily combinable with existing pseudo-label methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Pay Attention to CTC: Fast and Robust Pseudo-Labelling for Unified Speech Recognition](../../ICLR2026/audio_speech/pay_attention_to_ctc_fast_and_robust_pseudo-labelling_for_unified_speech_recogni.md)
- [\[ACL 2026\] MCGA: A Multi-task Classical Chinese Literary Genre Audio Corpus](mcga_a_multi-task_classical_chinese_literary_genre_audio_corpus.md)
- [\[ACL 2026\] Do We Need Distinct Representations for Every Speech Token? Unveiling and Exploiting Redundancy in Large Speech Language Models](do_we_need_distinct_representations_for_every_speech_token_unveiling_and_exploit.md)
- [\[ACL 2026\] An Exploration of Mamba for Speech Self-Supervised Models](an_exploration_of_mamba_for_speech_self-supervised_models.md)
- [\[ACL 2026\] Computational Narrative Understanding for Expressive Text-to-Speech](computational_narrative_understanding_for_expressive_text-to-speech.md)

</div>

<!-- RELATED:END -->
