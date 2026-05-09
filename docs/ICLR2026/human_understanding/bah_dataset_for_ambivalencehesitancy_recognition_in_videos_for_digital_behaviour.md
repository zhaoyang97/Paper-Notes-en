---
title: >-
  [Paper Note] BAH Dataset for Ambivalence/Hesitancy Recognition in Videos for Digital Behaviour Analysis
description: >-
  [ICLR 2026][Human Understanding][ambivalence/hesitancy recognition] This paper introduces BAH, the first multimodal dataset for Ambivalence/Hesitancy (A/H) recognition in videos, comprising 1,118 video clips (8.26 hours total) from 224 participants across 9 Canadian provinces, annotated by behavioural science experts, with frame-level and video-level baseline experimental results provided.
tags:
  - ICLR 2026
  - Human Understanding
  - ambivalence/hesitancy recognition
  - multimodal video dataset
  - behaviour change
  - affective computing
  - domain adaptation
date: 2026-05-08
content_hash: cd8122570d996b1a
---

# BAH Dataset for Ambivalence/Hesitancy Recognition in Videos for Digital Behaviour Analysis

**Conference**: ICLR 2026
**arXiv**: [2505.19328](https://arxiv.org/abs/2505.19328)
**Code**: [github.com/sbelharbi/bah-dataset](https://github.com/sbelharbi/bah-dataset)
**Area**: Human Behaviour Understanding / Affective Computing
**Keywords**: ambivalence/hesitancy recognition, multimodal video dataset, behaviour change, affective computing, domain adaptation

## TL;DR

This paper introduces BAH, the first multimodal dataset for Ambivalence/Hesitancy (A/H) recognition in videos, comprising 1,118 video clips (8.26 hours total) from 224 participants across 9 Canadian provinces, annotated by behavioural science experts, with frame-level and video-level baseline experimental results provided.

## Background & Motivation

Ambivalence and Hesitancy (A/H) are core psychological states in behaviour change, characterized by the simultaneous experience of desire and resistance to change. In face-to-face clinical interviews, healthcare providers can identify A/H through non-verbal cues such as vocal tone, facial expressions, and body language, enabling targeted personalized interventions. However, in digital health (eHealth) intervention settings, automatic, reliable, and non-intrusive A/H recognition remains unavailable.

Existing affective computing research has primarily focused on basic emotions (e.g., the seven universal expressions), continuous emotional dimensions (valence–arousal), and pain estimation. Although compound emotion recognition has advanced, A/H—as a more subtle complex emotion involving internal conflict of attitudes and intentions—remains entirely unexplored in the machine learning community. The fundamental reason is the absence of dedicated training and evaluation datasets. The BAH dataset is proposed specifically to address this gap.

## Method

### Overall Architecture

The construction of the BAH dataset constitutes a complete end-to-end pipeline: from participant recruitment and data collection, to expert annotation protocol design, baseline model evaluation, and domain adaptation experiments. Rather than presenting an algorithmic innovation, this work is a systematic contribution centred on the dataset itself.

### Key Designs

1. **Data Collection Platform (AER)**: The paper develops the "Automatic Expression Recognition" web platform (www.aerstudy.ca), through which participants remotely record their responses using their own devices (computers with cameras and microphones). Built-in calibration tests ensure data quality, and a virtual avatar guides participants through the entire procedure. The design motivation is to enable large-scale, diverse, and low-cost data collection.

2. **Seven Elicitation Questions**: The behavioural science team carefully designed seven questions intended to elicit neutral, positive, negative, ambivalent, willing, resistant, and hesitant responses respectively. For example, Question 4 ("Tell us something you enjoy doing but wish you could stop") is specifically designed to elicit ambivalence. This design ensures that A/H emerges naturally rather than through acted performance.

3. **Multi-level Annotation Scheme**: Three behavioural science experts annotate the data following a purpose-designed codebook, including:

    - **Video-level annotation**: overall judgment of whether A/H is present
    - **Frame-level annotation**: precise temporal boundaries of A/H occurrences
    - **Cue annotation**: records of specific cues used to identify A/H (facial expressions, verbal content, audio, body language, cross-modal inconsistency)

4. **Diversity Assurance**: 224 participants aged 18–66 are recruited from 9 Canadian provinces, covering diverse genders (59.8% male, 39.3% female), ethnicities (52.2% White, 21.0% Asian, 11.6% mixed, etc.), and age groups. 65.2% are non-students, reducing recruitment bias.

### Dataset Statistics

- **Total**: 1,118 video clips, 8.26 hours in total, with 638 clips containing A/H and 1.5 hours of A/H content
- **Frame count**: 714,005 frames, of which 131,103 contain A/H (only 18.36%)
- **A/H segment characteristics**: 1,274 A/H segments in total, with mean duration of $4.25 \pm 2.47$ seconds (approximately $102.92 \pm 59.16$ frames), ranging from 0.01 to 23.8 seconds
- **Severe class imbalance**: positive-class frames account for only 18.36% at the frame level, a characteristic requiring special consideration during training and evaluation

### Evaluation Metrics

Given the severe class imbalance, three metrics are employed: positive-class F1 score, weighted F1 (WF1), and positive-class Average Precision (AP). Since WF1 is biased towards the negative class (predicting all negative yields WF1 of 0.7148), F1 and AP more faithfully reflect true recognition capability.

## Key Experimental Results

### Main Results: Frame-level Classification

| Modality Combination | F1 | WF1 | AP |
|---|---|---|---|
| Visual (ResNet152+TCN) | 0.2213 | 0.7450 | 0.2674 |
| Audio | 0.2099 | 0.7387 | 0.2520 |
| Text Transcription | 0.2486 | 0.7149 | 0.2047 |
| Visual + Audio | 0.2873 | 0.7338 | 0.2818 |
| Visual + Text | 0.3046 | 0.7424 | 0.2809 |
| Trimodal Fusion | 0.2737 | 0.7396 | 0.2416 |

### Ablation Study: Impact of Temporal Context Modelling

| Configuration | F1 | WF1 | AP | Note |
|---|---|---|---|---|
| ResNet152 without context | 0.1757 | 0.7086 | 0.2096 | Single-frame independent classification |
| ResNet152 + TCN with context | 0.2213 | 0.7450 | 0.2674 | Temporal context modelling |

### Zero-shot Inference (Video-LLaVA)

| Prompting Strategy | Frame-level F1 | Video-level F1 |
|---|---|---|
| Simple prompt | 0.0000 | 0.0000 |
| Definition only | 0.1360–0.3296 | 0.1836–0.7575 |
| Transcription + definition | 0.3604 | 0.7233 |

### Domain Adaptation (Personalisation)

| Method | F1 | WF1 | AP |
|---|---|---|---|
| Source-only | $0.1547 \pm 0.1608$ | $0.6814 \pm 0.1687$ | $0.2462 \pm 0.1665$ |
| UDA (MMD) | $0.2418 \pm 0.1513$ | $0.6494 \pm 0.1484$ | $0.2608 \pm 0.1685$ |
| UDA (Sub-Based) | $0.2674 \pm 0.1475$ | $0.6461 \pm 0.1534$ | $0.2673 \pm 0.1642$ |
| Oracle | 0.3699 | — | — |

### Key Findings

1. **A/H recognition is highly challenging**: All baseline models achieve F1 below 0.32 and AP below 0.28, indicating that A/H recognition is substantially more difficult than basic emotion recognition.
2. **Text transcription is critical**: The text modality alone (F1: 0.2486) already outperforms the visual modality (F1: 0.2213), and the Visual + Text combination achieves the best F1 (0.3046).
3. **Temporal context is beneficial**: Modelling temporal dependencies with TCN improves performance across all backbones, as A/H is not an instantaneous phenomenon.
4. **Zero-shot M-LLMs heavily depend on text**: Video-LLaVA's performance is highly sensitive to the inclusion of text transcriptions; purely visual zero-shot recognition is nearly ineffective.
5. **Personalisation shows promise**: Subject-based UDA improves F1 from 0.1547 to 0.2674, though a substantial gap remains relative to the Oracle upper bound (0.3699).

## Highlights & Insights

- **Strong originality**: This is the first dataset in the ML community dedicated to A/H recognition, filling a critical gap at the intersection of behavioural science and machine learning.
- **High annotation quality**: Behavioural science experts annotate according to a rigorous codebook, providing not only A/H occurrence labels but also detailed cues including facial, verbal, audio, body language, and cross-modal inconsistency signals.
- **Multi-task applicability**: The dataset supports diverse research directions including frame-level classification, video-level classification, personalised learning (domain adaptation), and interpretability analysis.
- **Cross-modal inconsistency** is a key cue for A/H recognition—for example, verbally saying "yes" while shaking the head "no"—providing important inspiration for future method design.
- **Data collection is conducted in naturalistic, in-the-wild conditions** (participants using their own devices), which increases task difficulty while enhancing practical applicability.

## Limitations & Future Work

1. **Some videos are annotated by a single annotator**, lacking systematic inter-annotator agreement verification.
2. **The visual modality uses only cropped, aligned faces**, overlooking body language information in full frames—despite annotators emphasising body language as an important cue.
3. **Severe class imbalance** (only 18% positive frames at the frame level): although downsampling is employed, more advanced imbalanced learning methods are not explored.
4. **Feature fusion strategies are simple** (e.g., concatenation), failing to adequately exploit cross-modal inconsistency, which is a core characteristic of A/H.
5. **Systematic evaluation of large-scale pretrained models is lacking**: zero-shot experiments cover only Video-LLaVA, without evaluating stronger multimodal large language models.

## Related Work & Insights

- Similar to C-EXPR-DB (compound emotions) but more targeted: BAH focuses specifically on the clinically relevant A/H state.
- Complementary to MESC (emotional support conversations) and IEMOCAP (acted performance): BAH uses natural responses from real participants.
- Significant implications for digital health intervention: automatic A/H recognition could substantially improve the personalisation and effectiveness of eHealth interventions.
- Cross-modal inconsistency detection may draw on methods from deception detection and sarcasm recognition.
- The "textualisation" direction for multimodal large language models (M-LLMs) warrants deeper investigation: converting visual and audio cues into textual descriptions to leverage LLM reasoning capabilities.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (first A/H dataset; pioneering contribution to the field)
- Experimental Thoroughness: ⭐⭐⭐⭐ (comprehensive baselines but limited algorithmic innovation)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, detailed appendices)
- Value: ⭐⭐⭐⭐⭐ (fills an important gap; broad application prospects)

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach](../../CVPR2026/human_understanding/team_leya_in_10th_abaw_competition_multimodal_ambivalencehesitancy_recognition_a.md)
- [\[CVPR 2026\] BROTHER: Behavioral Recognition Optimized Through Heterogeneous Ensemble Regularization for Ambivalence and Hesitancy](../../CVPR2026/human_understanding/brother_behavioral_recognition_optimized_through_heterogeneous_ensemble_regulari.md)
- [\[AAAI 2026\] Facial-R1: Aligning Reasoning and Recognition for Facial Emotion Analysis](../../AAAI2026/human_understanding/facial-r1_aligning_reasoning_and_recognition_for_facial_emotion_analysis.md)
- [\[CVPR 2026\] Editing Physiological Signals in Videos Using Latent Representations](../../CVPR2026/human_understanding/editing_physiological_signals_in_videos_using_latent_representations.md)
- [\[ICLR 2026\] NeuroGaze-Distill: Brain-informed Distillation and Depression-Inspired Geometric Priors for Robust Facial Emotion Recognition](neurogaze-distill_brain-informed_distillation_and_depression-inspired_geometric_.md)

<!-- RELATED:END -->
