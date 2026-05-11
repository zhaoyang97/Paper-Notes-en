---
title: >-
  [Paper Note] MDD: A Dataset for Text-and-Music Conditioned Duet Dance Generation
description: >-
  [ICCV 2025][Human Understanding][duet dance generation] This paper introduces the Multimodal DuetDance (MDD) dataset — the first large-scale, professional-grade duet dance dataset simultaneously integrating motion, music…
tags:
  - "ICCV 2025"
  - "Human Understanding"
  - "duet dance generation"
  - "multimodal dataset"
  - "text-conditioned motion generation"
  - "motion capture"
  - "SMPL-X"
date: 2026-05-08
content_hash: a179ca7719f15987
---

# MDD: A Dataset for Text-and-Music Conditioned Duet Dance Generation

**Conference**: ICCV 2025
**arXiv**: [2508.16911](https://arxiv.org/abs/2508.16911)
**Code**: [https://gprerit96.github.io/mdd-page](https://gprerit96.github.io/mdd-page)
**Area**: Human Motion Understanding / Dance Generation
**Keywords**: duet dance generation, multimodal dataset, text-conditioned motion generation, motion capture, SMPL-X

## TL;DR

This paper introduces the Multimodal DuetDance (MDD) dataset — the first large-scale, professional-grade duet dance dataset simultaneously integrating motion, music, and text descriptions. MDD comprises 620 minutes of motion capture data spanning 15 dance styles and over 10K fine-grained text annotations, and defines two new tasks: Text-to-Duet and Text-to-Dance Accompaniment.

## Background & Motivation

Duet dancing represents one of the most complex forms of interactive human motion, requiring precise coordination and synchronization between two performers. Compared to solo dance, duets involve intricate spatial relationships, dynamic partner interactions, and continuous adaptation to musical rhythm. Limitations of existing work:

**InterGen/Inter-X**: Provide dyadic interaction motion datasets with text annotations, but lack professional dance movements and synchronized audio.

**Duolando (DD100)**: The first duet dance dataset, but contains only 1.95 hours of data and no text annotations.

**InterDance**: 3.93 hours of duet dance data, still without text annotations.

**TM2D**: Combines text and music but suffers from data distribution mismatch.

Core Gap: **No existing dataset simultaneously integrates motion, music, and text modalities to support duet dance generation.**

## Method

### Overall Architecture

MDD is a **dataset and benchmark** contribution rather than a methodological paper. The core contributions lie in dataset construction and new task formulation.

### Key Designs

1. **Data Collection Pipeline**:

    - **Music Selection**: Royalty-free music is prioritized, with 50–60 tracks prepared per dance style.
    - **Motion Capture**: OptiTrack system with 16 infrared cameras at 120 fps and 53 reflective markers.
    - **Subjects**: 30 dancers (16 female, 14 male), all at intermediate or advanced level with at least 3 years of experience.
    - **Post-processing**: Outlier removal, Gaussian filtering, zero-pose correction, and segment-aware blending smoothing.
    - **Motion Representation**: SMPL-X parametric model ($\theta \in \mathbb{R}^{N \times 55 \times 3}$, $\beta \in \mathbb{R}^{N \times 10}$, $t \in \mathbb{R}^{N \times 3}$).

2. **Fine-Grained Text Annotation System**:

    - Annotation dimensions cover three categories: **spatial relationships** (interaction position, orientation, contact points), **body movements** (action type, body parts), and **rhythm** (energy, tempo).
    - Dancers self-annotate to ensure professional terminology accuracy.
    - GPT-4o grammar refinement followed by a second round of expert review.
    - Average annotation length of 41 words (longer than existing motion-text datasets), with a vocabulary of 1,722 unique words.
    - Over 10,187 annotations released.

3. **Two New Task Definitions**:

    - **Text-to-Duet**: Given a text description $c$ and music $m$, generate a duet dance $(\mathbf{x_l}, \mathbf{x_f})$, learning a function $F(c, m) \mapsto \mathbf{x}$.
    - **Text-to-Dance Accompaniment**: Given text $c$, music $m$, and leader motion $\mathbf{x_l}$, generate follower motion $\mathbf{x_f}$, learning a function $G(c, m, \mathbf{x_l}) \mapsto \mathbf{x_f}$.

### Loss & Training

The optimization objective for motion representation fitting is:
$$E(\theta, t, \beta) = \lambda_1 \frac{1}{N} \sum_{j \in \mathcal{J}} \lambda_p \|J_j(M(\theta, t, \beta)) - g_j\|_2^2 + \lambda_2 \|\theta\|_2^2$$

All baseline models are trained with the AdamW optimizer, batch size 64, for 3,000 epochs.

## Key Experimental Results

### Main Results — Text-to-Duet

| Method | R-Prec Top1↑ | R-Prec Top3↑ | FID↓ | MM Dist↓ | BED↑ | BAS↑ |
|--------|-------------|-------------|------|----------|------|------|
| Ground Truth | 0.231 | 0.522 | 0.065 | 0.077 | 0.327 | 0.170 |
| MDM (text-only) | 0.082 | 0.192 | 1.420 | 2.133 | 0.211 | 0.186 |
| MDM (both) | 0.061 | 0.163 | 1.739 | 2.244 | 0.194 | 0.231 |
| InterGen (text-only) | 0.113 | 0.305 | 0.405 | 1.462 | 0.422 | 0.194 |
| InterGen (both) | 0.105 | 0.302 | 0.426 | 1.532 | 0.385 | 0.185 |
| InterGen w. Jukebox | **0.138** | **0.341** | **0.410** | **1.396** | **0.454** | 0.184 |

### Main Results — Text-to-Dance Accompaniment

| Method | R-Prec Top1↑ | FID↓ | MM Dist↓ | BED↑ | BAS↑ |
|--------|-------------|------|----------|------|------|
| Ground Truth | 0.231 | 0.065 | 0.077 | 0.327 | 0.170 |
| DuoLando (text-only) | 0.047 | 1.538 | 2.811 | 0.311 | 0.195 |
| DuoLando (music-only) | 0.069 | 0.721 | 2.633 | 0.305 | 0.216 |
| DuoLando (both) | **0.078** | **0.698** | **2.113** | **0.395** | **0.224** |

### Ablation Study — Text Ablation (InterGen, Text-to-Duet)

| Text Type | R-Prec Top1↑ | FID↓ | MM Dist↓ | BED↑ |
|-----------|-------------|------|----------|------|
| No text (music only) | 0.023 | 2.014 | 2.526 | 0.364 |
| Action names | 0.061 | 0.721 | 2.211 | 0.355 |
| Raw annotations | 0.091 | 0.511 | 1.722 | 0.381 |
| GPT-4o refined text | **0.105** | **0.426** | **1.532** | **0.385** |

### Key Findings

- InterGen consistently outperforms MDM, indicating its greater suitability for interactive generation tasks.
- Jukebox music embeddings marginally outperform MFCCs, suggesting that richer music representations improve generation quality.
- Multimodal conditioning (text + music) clearly outperforms unimodal conditioning in the Dance Accompaniment task.
- GPT-4o refined text descriptions yield the best results, demonstrating that LLMs can improve annotation quality.
- In user studies, motions generated with GPT-4o refined text receive the highest scores for text alignment and overall quality.

## Highlights & Insights

- **Fills an Important Gap**: The first duet dance dataset simultaneously providing motion, music, and text modalities.
- **Scale and Quality**: 620 minutes of professional motion capture data across 15 dance styles, making it the largest existing duet dance dataset.
- **Systematic Annotation Design**: A structured annotation framework covering spatial relationships, body movements, and rhythm.
- **Well-Motivated Task Definitions**: Text-to-Duet and Text-to-Dance Accompaniment correspond to two practically relevant generation scenarios — coordinated generation and follower generation.
- **Comprehensive Dataset Comparison**: Detailed comparison against 11 related datasets clearly highlights the contributions of MDD.

## Limitations & Future Work

- The significant performance gap between baselines and ground truth indicates that the tasks are highly challenging.
- Although positioned as large-scale, the dataset may still be insufficient for data-driven methods, particularly for certain dance styles with only approximately 30 minutes of data.
- Only OptiTrack markers are used; facial expressions and finger details are absent.
- Text annotation relies on GPT-4o refinement, which may introduce model bias.
- The text evaluator is trained on MDD, potentially leading to overfitting in evaluation.
- The BAS metric may reward jittery motions and should be interpreted with caution.

## Related Work & Insights

- The dyadic interaction motion datasets InterGen and Inter-X provide methodological references for this work.
- Duolando's follower GPT with reinforcement learning architecture serves as a baseline paradigm for Dance Accompaniment.
- The multi-stage annotation pipeline (dancer annotation → LLM refinement → expert review) is worth adopting in future dataset construction efforts.
- Modeling the dynamic leader–follower relationship in duet dance is an important direction for future research.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to integrate text, music, and motion across all three modalities in a duet dance setting; unique dataset positioning.
- **Experimental Thoroughness**: ⭐⭐⭐ Baseline adaptations are reasonable but limited in number; deeper analysis is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with thorough dataset statistics and analysis.
- **Value**: ⭐⭐⭐⭐ The dataset offers significant value to the multi-person motion generation and dance AI communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GenM3: Generative Pretrained Multi-path Motion Model for Text Conditional Human Motion Generation](genm3_generative_pretrained_multi-path_motion_model_for_text_conditional_human_m.md)
- [\[ICCV 2025\] HUMOTO: A 4D Dataset of Mocap Human Object Interactions](humoto_a_4d_dataset_of_mocap_human_object_interactions.md)
- [\[CVPR 2026\] MoLingo: Motion-Language Alignment for Text-to-Human Motion Generation](../../CVPR2026/human_understanding/molingo_motion-language_alignment_for_text-to-motion_generation.md)
- [\[CVPR 2026\] HandDreamer: Zero-Shot Text to 3D Hand Model Generation](../../CVPR2026/human_understanding/handdreamer_zero_shot_text_to_3d_hand_model_generation.md)
- [\[ICCV 2025\] UDC-VIT: A Real-World Video Dataset for Under-Display Cameras](udc-vit_a_real-world_video_dataset_for_under-display_cameras.md)

</div>

<!-- RELATED:END -->
